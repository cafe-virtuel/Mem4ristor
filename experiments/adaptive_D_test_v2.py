"""
adaptive_D_test_v2.py — Session 008 (follow-up)
Test: D(u) = D_max * u (alternative formula)

Hypothesis (from diagnostic):
The naive D(u) = D_max * (1-u) fails because:
  - u saturates to 0.99 fast -> D_eff -> 0 -> coupling eliminated

But the sigmoid creates a NATURAL negative feedback:
  - u HIGH (0.99) -> u_filter = tanh(pi*(0.5-0.99)) ≈ -0.9999
  - Negative u_filter flips Laplacian sign -> NEGATIVE coupling
  - Negative coupling DRIVES u DOWN toward 0.5-0 range

So the REAL mechanism for self-regulation might be:
  D(u) = D_max * u (instead of D_max * (1-u))

With D(u) = D_max * u:
  - u starts at ~0.05 (cold_start)
  - D_eff = 0.15 * 0.05 = 0.0075 (small but non-zero)
  - Coupling slightly active, sigma_social rises, u rises
  - As u rises to 0.3: D_eff = 0.15 * 0.3 = 0.045
  - Coupling strengthens, u reaches 0.5, D_eff = 0.075
  - At u=0.5: sigmoid(u) = tanh(0) = 0 -> u_filter = social_leakage = 0.05
  - At u=0.6 (clamped reference): sigmoid(u) = tanh(-0.188) = -0.186 -> u_filter = -0.136

Protocols:
  A. D=0 (baseline)
  B. D=0.15 constant (reference)
  C. D=0.15 + u_clamp=0.6 (reference from Session 007)
  D. D(u) = 0.15 * (1-u) — naive adaptive (fails, confirmed)
  E. D(u) = 0.15 * u — alternative adaptive (test)
  F. D(u) = 0.30 * u — alternative with higher D_max
  G. D(u) = 0.30 * u + 0.02 — offset to prevent zero coupling
"""

import numpy as np
import networkx as nx
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from mem4ristor.topology import Mem4Network
from mem4ristor.metrics import calculate_continuous_entropy, calculate_pairwise_synchrony


def make_ba(N=100, m=5, seed=42):
    G = nx.barabasi_albert_graph(N, m, seed=seed)
    adj = nx.to_numpy_array(G)
    return adj


def run_protocol(protocol, D_const=0.15, D_max=0.15, D_offset=0.0,
                  n_seeds=5, steps=1000):
    """Run one protocol across n_seeds. Returns dict of arrays."""
    results = {
        'H_cont': [],
        'synchrony': [],
        'u_mean': [],
        'D_eff_mean': [],
    }

    for seed in range(42, 42 + n_seeds):
        adj = make_ba(N=100, m=5, seed=seed)
        net = Mem4Network(
            adjacency_matrix=adj,
            heretic_ratio=0.15,
            seed=seed,
            cold_start=True,
        )

        # Set D in config for constant protocols
        if protocol in ('B', 'C', 'D', 'E', 'F', 'G'):
            net.model.cfg['coupling']['D'] = D_const

        H_ts = []
        u_ts = []
        D_eff_ts = []
        v_ts = []

        for t in range(steps):
            # Adaptive D: override D_eff each step based on u_mean (BEFORE step)
            u_mean = net.model.u.mean()
            if protocol == 'D':
                # Naive formula (fails)
                D_adaptive = D_max * (1 - u_mean)
                net.model.D_eff = D_adaptive / np.sqrt(net.model.N)
            elif protocol == 'E':
                # Alternative: D proportional to u
                D_adaptive = D_max * u_mean
                net.model.D_eff = D_adaptive / np.sqrt(net.model.N)
            elif protocol == 'F':
                D_adaptive = D_max * u_mean
                net.model.D_eff = D_adaptive / np.sqrt(net.model.N)
            elif protocol == 'G':
                # Offset formula: ensures minimum coupling
                D_adaptive = D_max * u_mean + D_offset
                net.model.D_eff = D_adaptive / np.sqrt(net.model.N)

            net.step(I_stimulus=0.5)

            # Clamp u if protocol C
            if protocol == 'C':
                net.model.u[:] = 0.6

            H_ts.append(net.calculate_entropy())
            u_ts.append(net.model.u.mean())
            v_ts.append(net.v.copy())

            # Record D_eff (already set for adaptive protocols)
            if protocol in ('D', 'E', 'F', 'G'):
                D_eff_ts.append(net.model.D_eff * np.sqrt(net.model.N))
            else:
                D_eff_ts.append(net.model.cfg['coupling']['D'])

        # Compute metrics over last 200 steps (steady state)
        H_arr = np.array(H_ts[-200:])
        u_arr = np.array(u_ts[-200:])
        D_arr = np.array(D_eff_ts[-200:])
        v_hist = np.array(v_ts[-200:])

        synchrony = calculate_pairwise_synchrony(v_hist)

        results['H_cont'].append(np.mean(H_arr))
        results['synchrony'].append(synchrony)
        results['u_mean'].append(np.mean(u_arr))
        results['D_eff_mean'].append(np.mean(D_arr))

    return {k: np.array(v) for k, v in results.items()}


def main():
    print("=" * 72)
    print("ADAPTIVE D TEST v2 — D(u) = D_max * u alternative formula")
    print("=" * 72)

    protocols = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    labels = {
        'A': 'D=0 baseline',
        'B': 'D=0.15 constant',
        'C': 'D=0.15 + u_clamp=0.6',
        'D': 'D(u)=0.15*(1-u) [naive]',
        'E': 'D(u)=0.15*u [alternative]',
        'F': 'D(u)=0.30*u [higher D]',
        'G': 'D(u)=0.30*u+0.02 [offset]',
    }

    all_results = {}

    for p in protocols:
        print(f"\nRunning protocol {p}: {labels[p]}...", end=' ', flush=True)
        if p == 'E':
            res = run_protocol(p, D_const=0.15, D_max=0.15, n_seeds=5)
        elif p == 'F':
            res = run_protocol(p, D_const=0.15, D_max=0.30, n_seeds=5)
        elif p == 'G':
            res = run_protocol(p, D_const=0.15, D_max=0.30, D_offset=0.02, n_seeds=5)
        else:
            res = run_protocol(p, n_seeds=5)
        all_results[p] = res
        H_mean = res['H_cont'].mean()
        u_mean = res['u_mean'].mean()
        print(f"H={H_mean:.4f}, u={u_mean:.4f}")

    # Summary table
    print("\n" + "=" * 72)
    print("RESULTS SUMMARY (n=5 seeds, last 200 steps)")
    print("=" * 72)
    print(f"{'Protocol':<32} {'H_cont':>10} {'+/-':>9} {'Sync':>8} {'u_mean':>10} {'D_eff':>10}")
    print("-" * 72)

    for p in protocols:
        r = all_results[p]
        H = r['H_cont']
        sy = r['synchrony']
        u = r['u_mean']
        D = r['D_eff_mean']
        print(f"{labels[p]:<32} {H.mean():>10.4f} {H.std():>9.4f} "
              f"{sy.mean():>8.4f} {u.mean():>10.4f} {D.mean():>10.4f}")

    # Ranking
    print("\n" + "=" * 72)
    print("RANKING by H_cont (descending)")
    print("=" * 72)
    sorted_protocols = sorted(protocols, key=lambda p: all_results[p]['H_cont'].mean(), reverse=True)
    for rank, p in enumerate(sorted_protocols, 1):
        H = all_results[p]['H_cont'].mean()
        u = all_results[p]['u_mean'].mean()
        delta_vs_C = H - all_results['C']['H_cont'].mean()
        print(f"  {rank}. {labels[p]:<32} H={H:.4f}  u={u:.4f}  "
              f"[vs clamp: {delta_vs_C:+.4f}]")

    # Key question: does D(u)=D_max*u achieve higher H than naive?
    print("\n" + "=" * 72)
    print("KEY QUESTION: Does D(u) = D_max*u beat the naive formula?")
    print("=" * 72)
    H_D = all_results['D']['H_cont'].mean()  # naive
    H_E = all_results['E']['H_cont'].mean()  # alternative
    H_F = all_results['F']['H_cont'].mean()  # alternative higher
    H_G = all_results['G']['H_cont'].mean()  # alternative with offset
    u_D = all_results['D']['u_mean'].mean()
    u_E = all_results['E']['u_mean'].mean()
    u_F = all_results['F']['u_mean'].mean()
    u_G = all_results['G']['u_mean'].mean()

    print(f"  Naive D(u)=D_max*(1-u):      H={H_D:.4f}, u={u_D:.4f}")
    print(f"  Alternative D(u)=D_max*u:    H={H_E:.4f}, u={u_E:.4f}  "
          f"[delta naive: {H_E-H_D:+.4f}]")
    print(f"  Alternative D(u)=0.30*u:    H={H_F:.4f}, u={u_F:.4f}  "
          f"[delta naive: {H_F-H_D:+.4f}]")
    print(f"  Alternative + offset:         H={H_G:.4f}, u={u_G:.4f}  "
          f"[delta naive: {H_G-H_D:+.4f}]")

    print("\n[DONE]")


if __name__ == '__main__':
    main()
