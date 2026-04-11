#!/usr/bin/env python3
"""
LIMIT-02 BA Attachment Parameter Sweep — The m=5 Mystery.

Investigation: Why does BA m=5 fail with BOTH uniform and degree_linear
normalization, while m=3 (degree_linear) and m=10 (uniform) each have a winner?

Hypothesis: BA m=5 is in a phase transition zone between the two normalization
regimes. This sweep maps H_stable as a function of m for both normalizations,
plus graph metrics (clustering, spectral gap, degree ratio) to identify the
transition mechanism.

Protocol:
  - m ∈ {1, 2, 3, 4, 5, 6, 7, 8, 10, 15} on BA (N=100)
  - 5 seeds per (m, norm) pair
  - 3000 steps, H_stable = mean of last 25%
  - Graph metrics: mean degree, max degree, deg_ratio, clustering coeff,
    spectral gap (algebraic connectivity)

Reference: PROJECT_STATUS.md §3sexies

Created: 2026-04-10 (Antigravity, v3.2.0 consolidation)
"""
import sys, os, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mem4ristor.core import Mem4Network


# ── Parameters ──────────────────────────────────────────────────────
N_BA       = 100
M_VALUES   = [1, 2, 3, 4, 5, 6, 7, 8, 10, 15]
STEPS      = 3000
TAIL_FRAC  = 0.25
SEEDS      = [42, 123, 777, 2024, 9999]
I_STIM     = 0.0
NORMS      = ['uniform', 'degree_linear']


def make_ba_adjacency(n, m, seed):
    """Barabási-Albert preferential attachment (symmetric adjacency)."""
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


def graph_metrics(adj):
    """Compute graph-level metrics from adjacency matrix."""
    degrees = np.sum(adj, axis=1)
    n = adj.shape[0]

    # Degree stats
    deg_min = degrees.min()
    deg_max = degrees.max()
    deg_mean = degrees.mean()
    deg_ratio = deg_max / max(deg_min, 1)

    # Clustering coefficient (local average)
    clustering = 0.0
    for i in range(n):
        neighbors = np.where(adj[i] > 0)[0]
        k_i = len(neighbors)
        if k_i < 2:
            continue
        # Count edges between neighbors
        sub = adj[np.ix_(neighbors, neighbors)]
        triangles = np.sum(sub) / 2  # undirected
        possible = k_i * (k_i - 1) / 2
        clustering += triangles / possible
    clustering /= n

    # Spectral gap (algebraic connectivity = 2nd smallest eigenvalue of Laplacian)
    L = np.diag(degrees) - adj
    try:
        from scipy.linalg import eigh
        vals = eigh(L, eigvals_only=True)
        spectral_gap = vals[1] if len(vals) > 1 else 0.0
    except Exception:
        spectral_gap = 0.0

    # Edge connectivity proxy: average shortest path via spectral gap
    # Higher spectral gap = better connected = more path redundancy
    edge_density = np.sum(adj) / (n * (n - 1))

    return {
        'deg_min': deg_min,
        'deg_max': deg_max,
        'deg_mean': deg_mean,
        'deg_ratio': deg_ratio,
        'clustering': clustering,
        'spectral_gap': spectral_gap,
        'edge_density': edge_density,
    }


def run_single(adj, norm, seed, steps=STEPS):
    """Run a single experiment, return H_stable."""
    net = Mem4Network(
        adjacency_matrix=adj.copy(),
        heretic_ratio=0.15,
        coupling_norm=norm,
        seed=seed,
    )
    trace = []
    for step in range(steps):
        net.step(I_stimulus=I_STIM)
        if step % 10 == 0:
            trace.append(net.calculate_entropy())

    tail_start = int(len(trace) * (1 - TAIL_FRAC))
    h_stable = np.mean(trace[tail_start:])
    return h_stable


# ── Main ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 80)
    print("LIMIT-02 BA ATTACHMENT PARAMETER SWEEP — THE m=5 MYSTERY")
    print(f"N={N_BA} | m values: {M_VALUES} | Steps={STEPS} | Seeds={len(SEEDS)}")
    print(f"Norms: {NORMS}")
    print("=" * 80)

    t0 = time.time()

    # Storage
    all_results = []  # list of dicts

    for m in M_VALUES:
        print(f"\n{'─'*60}")
        print(f"BA m={m}")
        print(f"{'─'*60}")

        # Graph metrics (use first seed)
        adj_sample = make_ba_adjacency(N_BA, m, SEEDS[0])
        metrics = graph_metrics(adj_sample)
        print(f"  Graph: deg_range=[{metrics['deg_min']:.0f}, {metrics['deg_max']:.0f}], "
              f"deg_ratio={metrics['deg_ratio']:.1f}, "
              f"clustering={metrics['clustering']:.3f}, "
              f"spectral_gap={metrics['spectral_gap']:.3f}, "
              f"edge_density={metrics['edge_density']:.4f}")

        for norm in NORMS:
            h_list = []
            for seed in SEEDS:
                adj = make_ba_adjacency(N_BA, m, seed)
                h = run_single(adj, norm, seed)
                h_list.append(h)

            h_mean = np.mean(h_list)
            h_std = np.std(h_list)
            marker = " ★" if h_mean > 0.5 else ""
            print(f"  {norm:<16} H_stable={h_mean:.4f} ± {h_std:.4f}{marker}")

            all_results.append({
                'm': m,
                'norm': norm,
                'h_mean': h_mean,
                'h_std': h_std,
                'h_values': h_list,
                **metrics
            })

    # ── Summary Table ─────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'m':>3} {'deg_ratio':>10} {'clustering':>11} {'spec_gap':>10} "
          f"{'H_uniform':>10} {'H_deg_lin':>10} {'Winner':>12}")
    print("─" * 75)

    for m in M_VALUES:
        r_uni = [r for r in all_results if r['m'] == m and r['norm'] == 'uniform'][0]
        r_dl  = [r for r in all_results if r['m'] == m and r['norm'] == 'degree_linear'][0]

        h_uni = r_uni['h_mean']
        h_dl  = r_dl['h_mean']

        if h_uni > 0.5 and h_dl > 0.5:
            winner = "both"
        elif h_uni > h_dl + 0.1:
            winner = "uniform"
        elif h_dl > h_uni + 0.1:
            winner = "deg_linear"
        elif h_uni < 0.1 and h_dl < 0.1:
            winner = "NEITHER"
        else:
            winner = "marginal"

        print(f"{m:>3} {r_uni['deg_ratio']:>10.1f} {r_uni['clustering']:>11.3f} "
              f"{r_uni['spectral_gap']:>10.3f} {h_uni:>10.4f} {h_dl:>10.4f} "
              f"{winner:>12}")

    print(f"\nElapsed: {elapsed:.1f}s")

    # ── Analysis ──────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("ANALYSIS")
    print(f"{'='*80}")

    # Find crossover point
    for i in range(len(M_VALUES) - 1):
        m1, m2 = M_VALUES[i], M_VALUES[i+1]
        r_uni_1 = [r for r in all_results if r['m'] == m1 and r['norm'] == 'uniform'][0]
        r_dl_1  = [r for r in all_results if r['m'] == m1 and r['norm'] == 'degree_linear'][0]
        r_uni_2 = [r for r in all_results if r['m'] == m2 and r['norm'] == 'uniform'][0]
        r_dl_2  = [r for r in all_results if r['m'] == m2 and r['norm'] == 'degree_linear'][0]

        # Check if there's a regime change between m1 and m2
        diff_1 = r_dl_1['h_mean'] - r_uni_1['h_mean']
        diff_2 = r_dl_2['h_mean'] - r_uni_2['h_mean']
        if diff_1 * diff_2 < 0:  # sign change = crossover
            print(f"  ⚡ Regime crossover detected between m={m1} and m={m2}!")
            print(f"     m={m1}: H_dl - H_uni = {diff_1:+.4f}")
            print(f"     m={m2}: H_dl - H_uni = {diff_2:+.4f}")

    # Dead zone identification
    dead_m = [m for m in M_VALUES
              if all(r['h_mean'] < 0.3
                     for r in all_results if r['m'] == m)]
    if dead_m:
        print(f"\n  ☠ Dead zone (H < 0.3 for BOTH norms): m ∈ {dead_m}")
        for m in dead_m:
            r = [r for r in all_results if r['m'] == m][0]
            print(f"     m={m}: clustering={r['clustering']:.3f}, "
                  f"spectral_gap={r['spectral_gap']:.3f}, "
                  f"deg_ratio={r['deg_ratio']:.1f}")

    # Correlation analysis
    print(f"\n  Correlation between graph metrics and normalization winner:")
    wins_dl = [(r['clustering'], r['spectral_gap'], r['deg_ratio'], r['edge_density'])
               for m in M_VALUES
               for r in [next(r2 for r2 in all_results if r2['m'] == m and r2['norm'] == 'degree_linear')]
               if r['h_mean'] > 0.5]
    wins_uni = [(r['clustering'], r['spectral_gap'], r['deg_ratio'], r['edge_density'])
                for m in M_VALUES
                for r in [next(r2 for r2 in all_results if r2['m'] == m and r2['norm'] == 'uniform')]
                if r['h_mean'] > 0.5]

    if wins_dl:
        arr = np.array(wins_dl)
        print(f"  degree_linear wins (n={len(wins_dl)}): "
              f"clustering={arr[:,0].mean():.3f}, "
              f"spectral_gap={arr[:,1].mean():.3f}, "
              f"deg_ratio={arr[:,2].mean():.1f}, "
              f"edge_density={arr[:,3].mean():.4f}")
    if wins_uni:
        arr = np.array(wins_uni)
        print(f"  uniform wins (n={len(wins_uni)}): "
              f"clustering={arr[:,0].mean():.3f}, "
              f"spectral_gap={arr[:,1].mean():.3f}, "
              f"deg_ratio={arr[:,2].mean():.1f}, "
              f"edge_density={arr[:,3].mean():.4f}")
