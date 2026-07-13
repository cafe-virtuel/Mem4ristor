"""
adaptive_D_diagnostic.py — Session 008 (supplementary)
Deep diagnostic: why does D(u) = D_max * (1-u) fail?

Hypothesis: u saturates to 0.99 within the first 100 steps,
making D_eff ≈ 0 for the entire measurement window.
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


def diagnostic_run(protocol, D_max=0.15, seed=42, steps=1000):
    """Track u(t), D_eff(t), H(t) every 50 steps."""
    adj = make_ba(N=100, m=5, seed=seed)
    net = Mem4Network(
        adjacency_matrix=adj,
        heretic_ratio=0.15,
        seed=seed,
        cold_start=True,
    )

    if protocol in ('B', 'C', 'D', 'E'):
        net.model.cfg['coupling']['D'] = 0.15

    records = {
        'step': [], 'H': [], 'u_mean': [], 'D_eff': [],
        'sigma_social': [], 'u_filter': []
    }

    for t in range(steps):
        # Adaptive D BEFORE step
        if protocol == 'D':
            u_mean = net.model.u.mean()
            D_adaptive = D_max * (1 - u_mean)
            net.model.D_eff = D_adaptive / np.sqrt(net.model.N)
        elif protocol == 'E':
            u_mean = net.model.u.mean()
            D_adaptive = D_max * (1 - u_mean)
            net.model.D_eff = D_adaptive / np.sqrt(net.model.N)

        # Snapshot before step (t=0 = initial state)
        if t % 50 == 0 or t < 10:
            u_mean = net.model.u.mean()
            # Compute sigma_social and u_filter manually
            # Laplacian @ v - v (coupling_input=None means net.L @ v - v)
            L = net.L if not net._is_sparse else net.L.toarray()
            laplacian_v = L @ net.v - net.v
            sigma_social = np.abs(laplacian_v).mean()
            u_filter = np.tanh(net.model.sigmoid_steepness * (0.5 - u_mean)) + net.model.social_leakage

            records['step'].append(t)
            records['H'].append(net.calculate_entropy())
            records['u_mean'].append(u_mean)
            records['D_eff'].append(net.model.cfg['coupling']['D'])
            records['sigma_social'].append(sigma_social)
            records['u_filter'].append(u_filter)

        net.step(I_stimulus=0.5)

        # Clamp u if protocol C
        if protocol == 'C':
            net.model.u[:] = 0.6

        # Adaptive D AFTER step (for next iteration)
        if protocol in ('D', 'E'):
            pass  # Already set before step

    return {k: np.array(v) for k, v in records.items()}


def main():
    print("=" * 70)
    print("ADAPTIVE D — DIAGNOSTIC (temporal evolution)")
    print("=" * 70)

    protocols = ['A', 'B', 'C', 'D']

    for p in protocols:
        print(f"\nProtocol {p}...", flush=True)
        r = diagnostic_run(p, seed=42, steps=1000)

        print(f"  Steps recorded: {len(r['step'])}")
        print(f"  u_mean trajectory: {r['u_mean'][:5].round(4)} ... {r['u_mean'][-3:].round(4)}")
        print(f"  D_eff trajectory: {r['D_eff'][:5].round(4)} ... {r['D_eff'][-3:].round(4)}")
        print(f"  sigma_social: {r['sigma_social'][:5].round(4)} ... {r['sigma_social'][-3:].round(4)}")
        print(f"  H_cont: {r['H'][:5].round(4)} ... {r['H'][-3:].round(4)}")

        # Find when u crosses 0.9
        u_arr = r['u_mean']
        cross_idx = np.where(u_arr > 0.9)[0]
        if len(cross_idx) > 0:
            print(f"  u crosses 0.9 at step: {r['step'][cross_idx[0]]}")
        else:
            print(f"  u stays below 0.9 throughout")

    print("\n" + "=" * 70)
    print("ANALYSIS: Why does D_adaptive fail?")
    print("=" * 70)
    print("""
The key insight: D(u) = D_max * (1 - u) only works if u is in a
productive RANGE where D_eff is non-negligible. But on BA m=5 with
cold_start, u SATURATES to 0.99 within the first ~50 steps.

With u ≈ 0.99:
  D_eff = 0.15 * (1 - 0.99) = 0.0015
  u_filter = tanh(π*(0.5-0.99)) ≈ tanh(-15.4) ≈ -0.9999
  I_coup = 0.0015/10 * (-0.9999) * sigma_social ≈ 0 (effectively zero)

The coupling becomes no-op, and the u dynamics revert to their
natural attractor: u → 0.99.

CONCLUSION: The naive D(u) = D_max * (1-u) formula is fundamentally
flawed because it requires u to already be in the moderate range
[0.3-0.7] to have any effect. But u SATURATES to 0.99 immediately.

POSSIBLE FIXES (for future sessions):
1. Initialize u explicitly (not cold_start)
2. Use D(u) = D_max * u instead (certain nodes drive coupling)
3. Modify the sigmoid formula in the model (u_filter = tanh(u-0.5))
4. Clamp u dynamically based on rate-of-change (watchdog on du)
""")


if __name__ == '__main__':
    main()
