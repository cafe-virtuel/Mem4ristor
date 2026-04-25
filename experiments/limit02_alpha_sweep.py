#!/usr/bin/env python3
"""
LIMIT-02 Power-Law Normalization Sweep (v2) - Using core.py degree_power mode.

Tests D/deg(i)^alpha for alpha in [0, 1] on BA m={3, 5, 10} to find
if an intermediate exponent bridges the dead zone.

Created: 2026-04-10 (Antigravity, v3.2.0 consolidation)
"""
import sys, os, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from mem4ristor.core import Mem4Network
from mem4ristor.metrics import calculate_cognitive_entropy

N_BA   = 100
STEPS  = 3000
SEEDS  = [42, 123, 777]
# NOTE: I_STIM = 0.0 means heretic_mask *= -1 is a no-op (I_eff = 0).
# Heretics are INACTIVE in this regime. Results reflect endogenous dynamics only.
I_STIM = 0.0
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
M_TARGETS = [2, 3, 4, 5, 6, 8, 10]


def make_ba(n, m, seed):
    rng = np.random.RandomState(seed)
    adj = np.zeros((n, n), dtype=float)
    for i in range(m + 1):
        for j in range(i + 1, m + 1):
            adj[i, j] = adj[j, i] = 1.0
    degrees = np.sum(adj, axis=1)
    for new_node in range(m + 1, n):
        probs = degrees[:new_node] / degrees[:new_node].sum()
        targets = rng.choice(new_node, size=m, replace=False, p=probs)
        for t in targets:
            adj[new_node, t] = adj[t, new_node] = 1.0
        degrees = np.sum(adj, axis=1)
    return adj


def run_experiment(adj, alpha, seed, steps=STEPS):
    net = Mem4Network(
        adjacency_matrix=adj.copy(),
        heretic_ratio=0.15,
        coupling_norm='degree_power',
        seed=seed,
    )
    net.degree_power_alpha = alpha
    net._compute_coupling_weights()

    trace_cont = []
    v_snapshots = []
    for step in range(steps):
        net.step(I_stimulus=I_STIM)
        if step % 10 == 0:
            trace_cont.append(net.calculate_entropy())
            v_snapshots.append(net.v.copy())

    tail_cont = trace_cont[int(len(trace_cont) * 0.75):]
    tail_v = v_snapshots[int(len(v_snapshots) * 0.75):]
    h_cog = np.mean([calculate_cognitive_entropy(v) for v in tail_v])
    return np.mean(tail_cont), h_cog


if __name__ == '__main__':
    print("=" * 90)
    print("LIMIT-02 POWER-LAW SWEEP: D/deg(i)^alpha")
    print(f"N={N_BA} | m: {M_TARGETS} | alpha: {ALPHAS} | Seeds={len(SEEDS)}")
    print("=" * 90)

    t0 = time.time()
    results = {}

    for m in M_TARGETS:
        print(f"\n--- BA m={m} ---")
        print(f"  {'alpha':>5}  {'H_cont(100-bin)':>16}  {'H_cog(5-bin)':>13}")
        for alpha in ALPHAS:
            h_cont_list, h_cog_list = [], []
            for seed in SEEDS:
                adj = make_ba(N_BA, m, seed)
                h_cont, h_cog = run_experiment(adj, alpha, seed)
                h_cont_list.append(h_cont)
                h_cog_list.append(h_cog)
            h_mean = np.mean(h_cont_list)
            h_std  = np.std(h_cont_list)
            h_cog_mean = np.mean(h_cog_list)
            star = " ***" if h_cog_mean > 0.3 else ""
            print(f"  alpha={alpha:.1f}  H_cont={h_mean:.4f}±{h_std:.4f}  H_cog={h_cog_mean:.4f}{star}")
            results[(m, alpha)] = (h_mean, h_std, h_cont_list, h_cog_mean)

    # Summary
    elapsed = time.time() - t0
    print(f"\n{'='*90}")
    print("SUMMARY: H_stable(m, alpha)")
    print(f"{'='*90}")

    print("\n[H_cont — 100-bin continuous entropy]")
    header = f"{'alpha':>6}"
    for m in M_TARGETS:
        header += f"  {'m='+str(m):>7}"
    print(header)
    print("-" * (8 + 9 * len(M_TARGETS)))
    for alpha in ALPHAS:
        row = f"{alpha:>6.1f}"
        for m in M_TARGETS:
            h = results[(m, alpha)][0]
            star = "*" if h > 0.5 else " "
            row += f"  {h:>6.3f}{star}"
        print(row)

    print("\n[H_cog — 5-bin cognitive entropy, KIMI thresholds ±0.4/1.2]")
    print("(Expected: H_cog ≈ 0 in endogenous regime, I_stim=0 → heretics inactive)")
    header2 = f"{'alpha':>6}"
    for m in M_TARGETS:
        header2 += f"  {'m='+str(m):>7}"
    print(header2)
    print("-" * (8 + 9 * len(M_TARGETS)))
    for alpha in ALPHAS:
        row = f"{alpha:>6.1f}"
        for m in M_TARGETS:
            h_cog = results[(m, alpha)][3]
            star = "*" if h_cog > 0.3 else " "
            row += f"  {h_cog:>6.3f}{star}"
        print(row)

    # Best alpha per topology (continuous metric)
    print(f"\nOptimal alpha per topology (H_cont):")
    for m in M_TARGETS:
        best_alpha = max(ALPHAS, key=lambda a: results[(m, a)][0])
        best_h = results[(m, best_alpha)][0]
        best_h_cog = results[(m, best_alpha)][3]
        print(f"  BA m={m}: alpha*={best_alpha:.1f} -> H_cont={best_h:.4f}  H_cog={best_h_cog:.4f}")

    print(f"\nElapsed: {elapsed:.1f}s")
