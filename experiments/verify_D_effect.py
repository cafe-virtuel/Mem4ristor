"""
verify_D_effect.py
Audit script for AUDIT-004 claim:
"D coupling at u_clamp=0.6 gives +0.51 bits on BA m=5"

Claim from Session 005:
- BA m=5, coupling_norm='degree_linear'
- u_clamp=0.6 (manual reset each step), I=0.5
- D=0.0: H=3.96 (baseline)
- D=0.15: H=4.48 (+0.51 bits)

Expected: positive D effect when u is maintained at 0.6.
Method: manual re-clamping of u each step (u[:] = 0.6)
"""

import sys
import numpy as np
sys.path.insert(0, 'src')

from mem4ristor.core import Mem4Network
import networkx as nx

def run_D_audit(D_values, u_clamp_value, m=5, N=100, n_seeds=5, I_stim=0.5, steps=1000):
    """Verify D coupling effect with manual u re-clamping."""
    results = []
    
    for D in D_values:
        H_values = []
        u_finals = []
        
        for seed in range(42, 42 + n_seeds):
            # Create BA graph
            G = nx.barabasi_albert_graph(N, m, seed=seed)
            adj = nx.to_numpy_array(G)
            
            # Create network with specified D via config override
            net = Mem4Network(
                adjacency_matrix=adj,
                heretic_ratio=0.15,
                seed=seed,
                cold_start=True,
                coupling_norm='degree_linear'
            )
            
            # Override D in config
            net.model.cfg['coupling']['D'] = D
            
            # Initialize u at clamp value
            net.model.u[:] = u_clamp_value
            
            # Run with manual re-clamping each step
            for _ in range(steps):
                net.step(I_stimulus=I_stim)
                # Re-clamp u to maintain value (the "u_clamp" procedure)
                net.model.u[:] = u_clamp_value
            
            H = net.calculate_entropy()
            H_values.append(H)
            u_finals.append(net.model.u.mean())
        
        mean_H = np.mean(H_values)
        std_H = np.std(H_values)
        mean_u = np.mean(u_finals)
        
        results.append({
            'D': D,
            'H_mean': mean_H,
            'H_std': std_H,
            'u_final': mean_u,
            'seeds': n_seeds
        })
        
        print(f"D={D:.3f}: H={mean_H:.4f} +/- {std_H:.4f}, u_final={mean_u:.4f}")
    
    return results

if __name__ == '__main__':
    print("=" * 60)
    print("VERIFYING D COUPLING EFFECT WITH U_CLAMP=0.6")
    print("=" * 60)
    print("Claim: D=0.15 gives +0.51 bits vs D=0 on BA m=5")
    print("Config: BA m=5, N=100, degree_linear, u_clamp=0.6, I=0.5")
    print("Method: manual u re-clamping each step")
    print()
    
    results = run_D_audit(
        D_values=[0.0, 0.15],
        u_clamp_value=0.6,
        m=5,
        N=100,
        n_seeds=5,
        I_stim=0.5,
        steps=1000
    )
    
    print()
    print("RESULT SUMMARY:")
    print("-" * 40)
    baseline = results[0]['H_mean']
    for r in results:
        delta = r['H_mean'] - baseline
        print(f"D={r['D']:.3f}: H={r['H_mean']:.4f} (delta={delta:+.4f})")
    
    # Check if claim is reproduced
    if len(results) == 2:
        delta = results[1]['H_mean'] - results[0]['H_mean']
        claimed_delta = 0.51
        print()
        print(f"Claimed delta: +{claimed_delta}")
        print(f"Actual delta:  {delta:+.4f}")
        if abs(delta - claimed_delta) < 0.2:
            print("VERDICT: CLAIM REPRODUCED (+/- 0.2)")
        elif delta > 0:
            print("VERDICT: POSITIVE EFFECT CONFIRMED (magnitude differs)")
        else:
            print("VERDICT: CLAIM NOT REPRODUCED - D effect is NOT positive")