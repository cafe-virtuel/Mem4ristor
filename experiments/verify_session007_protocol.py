"""
verify_session007_protocol.py — Session 008 (cross-check)
Verify: does D=0.15 + u_clamp=0.6 really give H=4.50?

The Session 007 summary says verify_D_effect.py found:
  D=0.000: H=4.0160
  D=0.150: H=4.5007
  Delta: +0.485 bits

But our run gives:
  D=0 constant: H=3.3414
  D=0 + u_clamp=0.6: H=3.9699
  Delta: +0.6285 bits

The ABSOLUTE values are different (3.34 vs 4.01 for D=0), suggesting
different conditions (seeds? steps? topology?).

Let's directly replicate the verify_D_effect.py conditions:
- BA m=5, N=100
- cold_start=True
- u_clamp=0.6 (re-clamped every step)
- I_stimulus=0.5
- 1000 steps
- Measure H on last 200 steps

This will tell us if our implementation matches Session 007's claims.
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


def run_verification(D_values=[0.0, 0.15], n_seeds=5, steps=1000,
                     u_clamp=None, I=0.5, D_adaptive_fn=None):
    """Direct replication of verify_D_effect.py conditions.

    D_adaptive_fn: if not None, call D_adaptive_fn(net, t) each step
                   to override D_eff. Takes (net, step) -> D value.
    """
    results = {}

    for D in D_values:
        H_list = []
        u_list = []
        sync_list = []

        for seed in range(42, 42 + n_seeds):
            adj = make_ba(N=100, m=5, seed=seed)
            net = Mem4Network(
                adjacency_matrix=adj,
                heretic_ratio=0.15,
                seed=seed,
                cold_start=True,
            )
            net.model.cfg['coupling']['D'] = D

            H_ts = []
            u_ts = []
            v_ts = []

            for t in range(steps):
                # Override D_eff if adaptive
                if D_adaptive_fn is not None:
                    D_val = D_adaptive_fn(net, t)
                    net.model.D_eff = D_val / np.sqrt(net.model.N)

                net.step(I_stimulus=I)

                if u_clamp is not None:
                    net.model.u[:] = u_clamp

                H_ts.append(net.calculate_entropy())
                u_ts.append(net.model.u.mean())
                v_ts.append(net.v.copy())

            # Last 200 steps
            H_arr = np.array(H_ts[-200:])
            u_arr = np.array(u_ts[-200:])
            v_hist = np.array(v_ts[-200:])

            sync = calculate_pairwise_synchrony(v_hist)

            H_list.append(np.mean(H_arr))
            u_list.append(np.mean(u_arr))
            sync_list.append(sync)

        results[D] = {
            'H': np.array(H_list),
            'u': np.array(u_list),
            'sync': np.array(sync_list)
        }

    return results


def main():
    print("=" * 72)
    print("VERIFY SESSION 007 PROTOCOL")
    print("=" * 72)

    # Replicate Session 007's conditions: u_clamp=0.6, I=0.5, D=[0, 0.15]
    print("\n[Test 1] D=[0, 0.15], u_clamp=0.6, I=0.5 (Session 007 conditions)")
    res = run_verification(D_values=[0.0, 0.15], n_seeds=5, steps=1000,
                           u_clamp=0.6, I=0.5)

    for D, data in res.items():
        H = data['H']
        u = data['u']
        sync = data['sync']
        print(f"  D={D}: H={H.mean():.4f} +/- {H.std():.4f}, "
              f"u={u.mean():.4f}, sync={sync.mean():.4f}")

    delta_H = res[0.15]['H'].mean() - res[0.0]['H'].mean()
    print(f"  Delta H (D=0.15 - D=0): {delta_H:+.4f}")
    print(f"  Session 007 claimed: +0.485 bits")

    # Test 2: Without clamping, I=0.5
    print("\n[Test 2] D=[0, 0.15], NO clamping, I=0.5")
    res2 = run_verification(D_values=[0.0, 0.15], n_seeds=5, steps=1000,
                            u_clamp=None, I=0.5)

    for D, data in res2.items():
        H = data['H']
        u = data['u']
        print(f"  D={D}: H={H.mean():.4f} +/- {H.std():.4f}, u={u.mean():.4f}")

    delta_H_2 = res2[0.15]['H'].mean() - res2[0.0]['H'].mean()
    print(f"  Delta H (D=0.15 - D=0): {delta_H_2:+.4f}")

    # Test 3: D(u) = 0.30*u + 0.02 — best protocol from adaptive_D_test_v2
    print("\n[Test 3] D(u)=0.30*u+0.02, I=0.5")
    def D_fn_G(net, t):
        u_mean = net.model.u.mean()
        return 0.30 * u_mean + 0.02

    res3 = run_verification(D_values=[0.0], n_seeds=5, steps=1000,
                            u_clamp=None, I=0.5, D_adaptive_fn=D_fn_G)
    H_G = res3[0.0]['H'].mean()
    u_G = res3[0.0]['u'].mean()
    sync_G = res3[0.0]['sync'].mean()
    print(f"  D(u)=0.30*u+0.02: H={H_G:.4f} +/- {res3[0.0]['H'].std():.4f}, "
          f"u={u_G:.4f}, sync={sync_G:.4f}")
    print(f"  Delta vs clamp: {H_G - res[0.15]['H'].mean():+.4f}")

    # Test 4: D(u) = 0.30*u + 0.02, sweep I values
    print("\n[Test 4] D(u)=0.30*u+0.02, sweep I=[0.1, 0.3, 0.5, 1.0]")
    for I in [0.1, 0.3, 0.5, 1.0]:
        res4 = run_verification(D_values=[0.0], n_seeds=5, steps=1000,
                                u_clamp=None, I=I, D_adaptive_fn=D_fn_G)
        H = res4[0.0]['H'].mean()
        u = res4[0.0]['u'].mean()
        sync = res4[0.0]['sync'].mean()
        print(f"  I={I}: H={H:.4f} +/- {res4[0.0]['H'].std():.4f}, "
              f"u={u:.4f}, sync={sync:.4f}")

    # Test 5: Compare clamping u=0.6 vs D(u)=0.30*u+0.02 at different I
    print("\n[Test 5] Compare: u_clamp=0.6 vs D_adaptive at I=0.5")
    res5 = run_verification(D_values=[0.15], n_seeds=5, steps=1000,
                            u_clamp=0.6, I=0.5)
    H_clamp = res5[0.15]['H'].mean()
    print(f"  Clamp u=0.6, D=0.15: H={H_clamp:.4f} +/- {res5[0.15]['H'].std():.4f}")
    print(f"  D_adaptive G:          H={H_G:.4f} +/- {res3[0.0]['H'].std():.4f}")
    print(f"  Difference: {H_G - H_clamp:+.4f}")

    print("\n[DONE]")


if __name__ == '__main__':
    main()
