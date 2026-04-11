#!/usr/bin/env python3
"""
LIMIT-02 Multi-Topology Sweep — Validate degree_linear across diverse networks.

Tests uniform vs degree_linear coupling on:
  1. Barabási-Albert (m=1,3,5,10) — scale-free, variable density
  2. Configuration Model (γ=2.5, 3.0, 4.0) — tunable power-law exponent
  3. Holme-Kim (m=3, p_tri=0.5,0.9) — scale-free with clustering
  4. Watts-Strogatz (k=4, p=0.1,0.3) — small-world (control)
  5. Erdős-Rényi (p=0.06, 0.12) — random (control)

For each: compare uniform vs degree_linear, 5 seeds, 3000 steps.
"""
import sys, os, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from mem4ristor.core import Mem4Network

# ── Parameters ──────────────────────────────────────────────────────
N = 100
STEPS = 3000
TAIL_FRAC = 0.25
SEEDS = [42, 123, 777, 2024, 9999]

# ── Graph generators ────────────────────────────────────────────────

def make_ba(n, m, seed):
    """Barabási-Albert preferential attachment."""
    rng = np.random.RandomState(seed)
    adj = np.zeros((n, n))
    for i in range(m + 1):
        for j in range(i + 1, m + 1):
            adj[i, j] = adj[j, i] = 1.0
    degrees = np.sum(adj, axis=1)
    for new in range(m + 1, n):
        probs = degrees[:new] / degrees[:new].sum()
        targets = rng.choice(new, size=min(m, new), replace=False, p=probs)
        for t in targets:
            adj[new, t] = adj[t, new] = 1.0
        degrees = np.sum(adj, axis=1)
    return adj

def make_configuration_model(n, gamma, seed, k_min=2, k_max=None):
    """Configuration model with power-law degree sequence P(k) ~ k^(-gamma)."""
    rng = np.random.RandomState(seed)
    if k_max is None:
        k_max = int(np.sqrt(n))

    # Generate degree sequence from power-law
    ks = np.arange(k_min, k_max + 1, dtype=float)
    probs = ks ** (-gamma)
    probs /= probs.sum()
    degrees = rng.choice(ks, size=n, p=probs).astype(int)

    # Ensure even sum of degrees
    if degrees.sum() % 2 == 1:
        degrees[rng.randint(n)] += 1

    # Create edge stubs and shuffle
    stubs = []
    for i, d in enumerate(degrees):
        stubs.extend([i] * d)
    rng.shuffle(stubs)

    # Pair stubs to form edges
    adj = np.zeros((n, n))
    for idx in range(0, len(stubs) - 1, 2):
        u, v = stubs[idx], stubs[idx + 1]
        if u != v:  # no self-loops
            adj[u, v] = adj[v, u] = 1.0

    return adj

def make_holme_kim(n, m, p_tri, seed):
    """Holme-Kim: BA with triad formation (clustering)."""
    rng = np.random.RandomState(seed)
    adj = np.zeros((n, n))
    for i in range(m + 1):
        for j in range(i + 1, m + 1):
            adj[i, j] = adj[j, i] = 1.0
    degrees = np.sum(adj, axis=1)

    for new in range(m + 1, n):
        probs = degrees[:new] / degrees[:new].sum()
        # First edge: preferential attachment
        first = rng.choice(new, p=probs)
        adj[new, first] = adj[first, new] = 1.0
        added = {first}

        for _ in range(m - 1):
            if rng.rand() < p_tri and len(added) > 0:
                # Triad formation: connect to neighbor of last added
                last = list(added)[-1]
                neighbors_of_last = np.where(adj[last, :new] > 0)[0]
                candidates = [nn for nn in neighbors_of_last if nn not in added and nn != new]
                if candidates:
                    pick = rng.choice(candidates)
                    adj[new, pick] = adj[pick, new] = 1.0
                    added.add(pick)
                    continue
            # Fallback: preferential attachment
            probs_updated = degrees[:new].copy()
            for a in added:
                probs_updated[a] = 0
            if probs_updated.sum() > 0:
                probs_updated /= probs_updated.sum()
                pick = rng.choice(new, p=probs_updated)
                adj[new, pick] = adj[pick, new] = 1.0
                added.add(pick)

        degrees = np.sum(adj, axis=1)
    return adj

def make_watts_strogatz(n, k, p, seed):
    """Watts-Strogatz small-world network."""
    rng = np.random.RandomState(seed)
    adj = np.zeros((n, n))
    # Ring lattice with k/2 neighbors on each side
    for i in range(n):
        for j in range(1, k // 2 + 1):
            adj[i, (i + j) % n] = adj[(i + j) % n, i] = 1.0

    # Rewire with probability p
    for i in range(n):
        for j in range(1, k // 2 + 1):
            target = (i + j) % n
            if adj[i, target] == 1.0 and rng.rand() < p:
                adj[i, target] = adj[target, i] = 0.0
                candidates = [c for c in range(n) if c != i and adj[i, c] == 0.0]
                if candidates:
                    new_target = rng.choice(candidates)
                    adj[i, new_target] = adj[new_target, i] = 1.0
    return adj

def make_erdos_renyi(n, p, seed):
    """Erdős-Rényi random graph G(n,p)."""
    rng = np.random.RandomState(seed)
    adj = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if rng.rand() < p:
                adj[i, j] = adj[j, i] = 1.0
    return adj

def degree_stats(adj):
    """Return min, max, mean, std, and heterogeneity ratio."""
    d = np.sum(adj, axis=1)
    d = d[d > 0]
    return d.min(), d.max(), d.mean(), d.std(), d.max() / max(d.min(), 1)

def run_one(adj, norm, seed):
    """Run single experiment, return H_stable."""
    net = Mem4Network(size=10, heretic_ratio=0.15, seed=seed,
                      adjacency_matrix=adj.copy(), coupling_norm=norm)
    trace = []
    for step in range(STEPS):
        net.step(I_stimulus=0.0)
        if step % 10 == 0:
            trace.append(net.calculate_entropy())
    tail = int(len(trace) * (1 - TAIL_FRAC))
    return np.mean(trace[tail:])

# ── Topology definitions ────────────────────────────────────────────
TOPOLOGIES = [
    # (label, generator_func, kwargs_dict)
    ("BA m=1 (tree)",        make_ba,              {"m": 1}),
    ("BA m=3",               make_ba,              {"m": 3}),
    ("BA m=5",               make_ba,              {"m": 5}),
    ("BA m=10",              make_ba,              {"m": 10}),
    ("ConfigModel γ=2.5",   make_configuration_model, {"gamma": 2.5}),
    ("ConfigModel γ=3.0",   make_configuration_model, {"gamma": 3.0}),
    ("ConfigModel γ=4.0",   make_configuration_model, {"gamma": 4.0}),
    ("Holme-Kim m=3 p=0.5", make_holme_kim,       {"m": 3, "p_tri": 0.5}),
    ("Holme-Kim m=3 p=0.9", make_holme_kim,       {"m": 3, "p_tri": 0.9}),
    ("Watts-Strogatz k=4 p=0.1", make_watts_strogatz, {"k": 4, "p": 0.1}),
    ("Watts-Strogatz k=4 p=0.3", make_watts_strogatz, {"k": 4, "p": 0.3}),
    ("Erdős-Rényi p=0.06",  make_erdos_renyi,     {"p": 0.06}),
    ("Erdős-Rényi p=0.12",  make_erdos_renyi,     {"p": 0.12}),
]

# ── Main ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 80)
    print("LIMIT-02 MULTI-TOPOLOGY SWEEP")
    print(f"N={N} | Steps={STEPS} | Seeds={len(SEEDS)} | Norms: uniform vs degree_linear")
    print("=" * 80)

    results = []
    t0 = time.time()

    for label, gen_func, kwargs in TOPOLOGIES:
        print(f"\n{'─' * 80}")
        print(f"[{label}]")

        h_uniform = []
        h_dlinear = []

        for i, seed in enumerate(SEEDS):
            adj = gen_func(N, seed=seed, **kwargs)

            if i == 0:
                dmin, dmax, dmean, dstd, ratio = degree_stats(adj)
                print(f"  Degrees: min={dmin:.0f} max={dmax:.0f} "
                      f"mean={dmean:.1f} std={dstd:.1f} ratio={ratio:.1f}")

            h_u = run_one(adj, 'uniform', seed)
            h_d = run_one(adj, 'degree_linear', seed)
            h_uniform.append(h_u)
            h_dlinear.append(h_d)

            print(f"  seed={seed}: uniform={h_u:.4f}  degree_linear={h_d:.4f}")

        mu, su = np.mean(h_uniform), np.std(h_uniform)
        md, sd = np.mean(h_dlinear), np.std(h_dlinear)
        delta = md - mu

        results.append({
            'label': label,
            'h_uniform': mu, 'std_uniform': su,
            'h_dlinear': md, 'std_dlinear': sd,
            'delta': delta,
            'deg_ratio': ratio
        })

        winner = "degree_linear" if delta > 0.05 else ("uniform" if delta < -0.05 else "≈ equal")
        print(f"  → uniform: {mu:.4f}±{su:.4f}  |  degree_linear: {md:.4f}±{sd:.4f}  |  Δ={delta:+.4f}  [{winner}]")

    # ── Summary table ───────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'=' * 80}")
    print("SUMMARY TABLE")
    print(f"{'=' * 80}")
    print(f"{'Topology':<28} {'deg_ratio':>9} {'H_uniform':>10} {'H_dlinear':>10} {'Δ':>8} {'Winner':>14}")
    print("─" * 80)

    for r in results:
        delta = r['delta']
        if delta > 0.05:
            winner = "degree_linear ★"
        elif delta < -0.05:
            winner = "uniform ★"
        else:
            winner = "≈ equal"
        print(f"{r['label']:<28} {r['deg_ratio']:>9.1f} "
              f"{r['h_uniform']:>10.4f} {r['h_dlinear']:>10.4f} "
              f"{delta:>+8.4f} {winner:>14}")

    print(f"\nElapsed: {elapsed:.1f}s")

    # ── Verdicts ────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("VERDICTS")
    print(f"{'=' * 80}")

    dl_wins = [r for r in results if r['delta'] > 0.05]
    u_wins  = [r for r in results if r['delta'] < -0.05]
    ties    = [r for r in results if abs(r['delta']) <= 0.05]

    print(f"\ndegree_linear superior ({len(dl_wins)}):")
    for r in dl_wins:
        print(f"  {r['label']}: Δ={r['delta']:+.4f}")

    print(f"\nuniform superior ({len(u_wins)}):")
    for r in u_wins:
        print(f"  {r['label']}: Δ={r['delta']:+.4f}")

    print(f"\n≈ equal ({len(ties)}):")
    for r in ties:
        print(f"  {r['label']}: Δ={r['delta']:+.4f}")

    # Regression: does deg_ratio predict when degree_linear helps?
    ratios = np.array([r['deg_ratio'] for r in results])
    deltas = np.array([r['delta'] for r in results])
    if len(ratios) > 2:
        corr = np.corrcoef(ratios, deltas)[0, 1]
        print(f"\nCorrelation(deg_ratio, Δ) = {corr:.3f}")
        if corr > 0.5:
            print("→ Degree heterogeneity predicts when degree_linear helps")
        elif corr < -0.5:
            print("→ Inverse: degree_linear hurts on heterogeneous networks (?)")
        else:
            print("→ Weak correlation: other factors at play")
