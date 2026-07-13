"""
adaptive_D_test.py — Session 008
Test: D(u) adaptatif vs D constant vs u_clamp

Idee de Session 007 (genie idea):
  Instead of clamping u artificially, modulate D dynamically based on u:
  D_eff(u) = D_max * (1 - u_mean)

This creates a natural negative feedback:
  - u HIGH (doubtful) -> D_eff LOW -> less coupling -> sigma_social drops -> u falls
  - u LOW (certain)    -> D_eff HIGH -> more coupling -> sigma_social rises -> u rises

Protocol:
  BA m=5, N=100, heretic_ratio=0.15, cold_start=True, 1000 steps
  Compare 5 protocols:
    A. D=0        (baseline, no coupling)
    B. D=0.15     (constant, known positive effect with clamping)
    C. D=0.15 + u_clamp=0.6 each step (reference from Session 007)
    D. D_adaptive(u) = 0.15 * (1 - u_mean) — per-step override
    E. D_adaptive(u) = 0.30 * (1 - u_mean) — higher D_max

Metrics: H_cont (100 bins), synchrony (Pearson), u_mean (final 200 steps)
"""

import numpy as np
import networkx as nx
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from mem4ristor.dynamics import Mem4ristorV3
from mem4ristor.topology import Mem4Network
from mem4ristor.metrics import calculate_continuous_entropy, calculate_pairwise_synchrony


def make_ba(N=100, m=5, seed=42):
    G = nx.barabasi_albert_graph(N, m, seed=seed)
    adj = nx.to_numpy_array(G)
    return adj


def run_protocol(protocol, D_const=0.15, D_max=0.15, n_seeds=5, steps=1000):
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

        # Override D if needed (protocol B, D, E use D>0)
        if protocol in ('B', 'C', 'D', 'E'):
            net.model.cfg['coupling']['D'] = D_const

        H_ts = []
        u_ts = []
        D_eff_ts = []
        v_ts = []  # for synchrony

        for t in range(steps):
            # Adaptive D: override D_eff each step based on u_mean
            if protocol == 'D':
                u_mean = net.model.u.mean()
                D_adaptive = D_max * (1 - u_mean)
                net.model.D_eff = D_adaptive / np.sqrt(net.model.N)
            elif protocol == 'E':
                u_mean = net.model.u.mean()
                D_adaptive = D_max * (1 - u_mean)
                net.model.D_eff = D_adaptive / np.sqrt(net.model.N)

            net.step(I_stimulus=0.5)

            # Clamp u if protocol C
            if protocol == 'C':
                net.model.u[:] = 0.6

            H_ts.append(net.calculate_entropy())
            u_ts.append(net.model.u.mean())
            v_ts.append(net.v.copy())

            # Record D_eff
            if protocol in ('D', 'E'):
                D_eff_ts.append(net.model.D_eff * np.sqrt(net.model.N))  # back to D
            else:
                D_eff_ts.append(net.model.cfg['coupling']['D'])

        # Compute metrics over last 200 steps (steady state)
        H_arr = np.array(H_ts[-200:])
        u_arr = np.array(u_ts[-200:])
        D_arr = np.array(D_eff_ts[-200:])
        v_hist = np.array(v_ts[-200:])  # (T, N)

        # Synchrony: pairwise Pearson correlation (binning-independant)
        synchrony = calculate_pairwise_synchrony(v_hist)

        results['H_cont'].append(np.mean(H_arr))
        results['synchrony'].append(synchrony)
        results['u_mean'].append(np.mean(u_arr))
        results['D_eff_mean'].append(np.mean(D_arr))

    return {k: np.array(v) for k, v in results.items()}


def main():
    print("=" * 70)
    print("ADAPTIVE D TEST — Session 008")
    print("=" * 70)

    protocols = ['A', 'B', 'C', 'D', 'E']
    labels = {
        'A': 'D=0 baseline',
        'B': 'D=0.15 constant',
        'C': 'D=0.15 + u_clamp=0.6',
        'D': 'D_adaptive = 0.15*(1-u)',
        'E': 'D_adaptive = 0.30*(1-u)',
    }

    all_results = {}

    for p in protocols:
        print(f"\nRunning protocol {p}: {labels[p]}...")
        if p == 'E':
            res = run_protocol(p, D_const=0.15, D_max=0.30, n_seeds=5)
        elif p == 'D':
            res = run_protocol(p, D_const=0.15, D_max=0.15, n_seeds=5)
        else:
            res = run_protocol(p, n_seeds=5)
        all_results[p] = res

    # Summary table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY (n=5 seeds, last 200 steps)")
    print("=" * 70)
    print(f"{'Protocol':<35} {'H_cont':>10} {'+/-':>9} {'Sync':>8} {'u_mean':>10} {'D_eff':>10}")
    print("-" * 70)

    for p in protocols:
        r = all_results[p]
        H = r['H_cont']
        sy = r['synchrony']
        u = r['u_mean']
        D = r['D_eff_mean']
        print(f"{labels[p]:<35} {H.mean():>10.4f} {H.std():>9.4f} "
              f"{sy.mean():>8.4f} {u.mean():>10.4f} {D.mean():>10.4f}")

    # Key comparison: adaptive D vs clamping
    print("\n" + "=" * 70)
    print("KEY COMPARISON: Does D_adaptive beat u_clamp?")
    print("=" * 70)
    H_C = all_results['C']['H_cont'].mean()
    H_D = all_results['D']['H_cont'].mean()
    H_E = all_results['E']['H_cont'].mean()
    u_C = all_results['C']['u_mean'].mean()
    u_D = all_results['D']['u_mean'].mean()
    u_E = all_results['E']['u_mean'].mean()

    print(f"  Protocol C (clamp u=0.6):  H={H_C:.4f}, u_mean={u_C:.4f}")
    print(f"  Protocol D (D=0.15*(1-u)): H={H_D:.4f}, u_mean={u_D:.4f}")
    print(f"  Protocol E (D=0.30*(1-u)): H={H_E:.4f}, u_mean={u_E:.4f}")
    print()
    print(f"  Delta H (D vs clamp): {H_D - H_C:+.4f} bits")
    print(f"  Delta H (E vs clamp): {H_E - H_C:+.4f} bits")
    print(f"  u_mean achieved: clamp={u_C:.4f} vs adaptive_D={u_D:.4f} vs adaptive_E={u_E:.4f}")

    # Does adaptive D achieve stable u in productive range?
    print("\n" + "=" * 70)
    print("U STABILITY: Is u kept in productive range without clamping?")
    print("=" * 70)
    for p in protocols:
        r = all_results[p]
        u_vals = r['u_mean']
        print(f"  {labels[p]:<30}: u_mean={u_vals.mean():.4f} +/- {u_vals.std():.4f}")

    print("\n[DONE]")


if __name__ == '__main__':
    main()
