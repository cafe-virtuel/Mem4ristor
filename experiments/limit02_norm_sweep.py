#!/usr/bin/env python3
"""
LIMIT-02 Normalization Sweep — Degree-based coupling on Barabási-Albert networks.

Tests 4 coupling_norm modes on BA graphs to determine which (if any) breaks
hub strangulation. Control: lattice 10×10 with uniform norm.

Metrics: H_stable (mean entropy over last 25% of run), per seed.
"""
import sys, os, time
import numpy as np

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mem4ristor.core import Mem4Network

# ── Parameters ──────────────────────────────────────────────────────
N_BA       = 100          # nodes for BA graphs
BA_M       = 3            # BA attachment parameter
STEPS      = 3000         # simulation steps per run
TAIL_FRAC  = 0.25         # fraction of run used for H_stable
SEEDS      = [42, 123, 777, 2024, 9999]  # 5 seeds
I_STIM     = 0.0          # no external stimulus (hardest case)

NORM_MODES = ['uniform', 'degree', 'degree_linear', 'degree_log']

def make_ba_adjacency(n, m, seed):
    """Barabási-Albert preferential attachment (symmetric adjacency)."""
    rng = np.random.RandomState(seed)
    adj = np.zeros((n, n), dtype=float)
    # Start with a fully connected clique of m+1 nodes
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

def run_experiment(label, adj, norm, seed, steps=STEPS):
    """Run a single experiment, return H_stable and entropy trace."""
    if adj is None:
        # Lattice mode
        net = Mem4Network(size=10, heretic_ratio=0.15, seed=seed,
                          coupling_norm=norm)
    else:
        net = Mem4Network(size=10, heretic_ratio=0.15, seed=seed,
                          adjacency_matrix=adj.copy(), coupling_norm=norm)

    trace = []
    for step in range(steps):
        net.step(I_stimulus=I_STIM)
        if step % 10 == 0:
            trace.append(net.calculate_entropy())

    tail_start = int(len(trace) * (1 - TAIL_FRAC))
    h_stable = np.mean(trace[tail_start:])
    h_std = np.std(trace[tail_start:])
    return h_stable, h_std, trace

# ── Main ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 72)
    print("LIMIT-02 NORMALIZATION SWEEP")
    print(f"BA: N={N_BA}, m={BA_M} | Steps={STEPS} | Seeds={len(SEEDS)}")
    print(f"Norm modes: {NORM_MODES}")
    print("=" * 72)

    results = {}
    t0 = time.time()

    # ── Control: Lattice 10×10 uniform ──
    print("\n[CONTROL] Lattice 10×10, uniform norm")
    lat_h = []
    for seed in SEEDS:
        h, hstd, _ = run_experiment("lattice_uniform", None, 'uniform', seed)
        lat_h.append(h)
        print(f"  seed={seed}: H_stable={h:.4f} ± {hstd:.4f}")
    results['lattice_uniform'] = (np.mean(lat_h), np.std(lat_h))
    print(f"  → MEAN: {np.mean(lat_h):.4f} ± {np.std(lat_h):.4f}")

    # ── BA experiments: each norm mode ──
    for norm in NORM_MODES:
        label = f"BA_{norm}"
        print(f"\n[BA] norm={norm}")
        h_list = []
        for seed in SEEDS:
            adj = make_ba_adjacency(N_BA, BA_M, seed)
            h, hstd, _ = run_experiment(label, adj, norm, seed)
            h_list.append(h)

            # Degree stats for first seed only
            if seed == SEEDS[0]:
                degrees = np.sum(adj, axis=1)
                print(f"  Degree stats: min={degrees.min():.0f}, max={degrees.max():.0f}, "
                      f"mean={degrees.mean():.1f}, std={degrees.std():.1f}")

            print(f"  seed={seed}: H_stable={h:.4f} ± {hstd:.4f}")

        results[label] = (np.mean(h_list), np.std(h_list))
        print(f"  → MEAN: {np.mean(h_list):.4f} ± {np.std(h_list):.4f}")

    # ── Summary ─────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"{'Configuration':<25} {'H_stable':>10} {'± std':>10}")
    print("-" * 50)
    for key, (mean_h, std_h) in results.items():
        marker = " ★" if mean_h > 0.5 else ""
        print(f"{key:<25} {mean_h:>10.4f} {std_h:>10.4f}{marker}")

    print(f"\nElapsed: {elapsed:.1f}s")
    print("\n★ = H_stable > 0.5 (diversity preserved)")

    # ── Verdict ─────────────────────────────────────────────────────
    best_ba = max(
        [(k, v) for k, v in results.items() if k.startswith("BA_")],
        key=lambda x: x[1][0]
    )
    print(f"\nBest BA config: {best_ba[0]} → H={best_ba[1][0]:.4f}")

    lattice_h = results['lattice_uniform'][0]
    ratio = best_ba[1][0] / lattice_h if lattice_h > 0 else 0
    print(f"Lattice control: H={lattice_h:.4f}")
    print(f"Recovery ratio: {ratio:.1%} of lattice performance")

    if best_ba[1][0] > 0.3:
        print("\n✓ Degree normalization PARTIALLY resolves hub strangulation")
    elif best_ba[1][0] > 0.01:
        print("\n~ Marginal improvement — further investigation needed")
    else:
        print("\n✗ Degree normalization alone does NOT resolve hub strangulation")
