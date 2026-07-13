#!/usr/bin/env python3
"""Mini-experiment: eigenvector centrality (spectral) vs degree_linear on BA m=5 (dead zone)."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from mem4ristor.topology import Mem4Network
from mem4ristor.metrics import calculate_continuous_entropy

def run_experiment(BA_m, coupling_norm, seed, n_steps=3000, I_stim=0.0):
    """Run a single experiment and return H_stable (last 25% of steps)."""
    np.random.seed(seed)
    
    # Build BA graph
    import networkx as nx
    G = nx.barabasi_albert_graph(100, BA_m, seed=seed)
    adj = nx.to_numpy_array(G)
    
    # Run with I_stim=0.5 to match preprint dead zone sweep conditions (Section 4.4)
    net = Mem4Network(
        adjacency_matrix=adj,
        heretic_ratio=0.15,
        seed=seed,
        cold_start=True,
        coupling_norm=coupling_norm
    )
    
    # Run simulation
    H_history = []
    for step in range(n_steps):
        net.step(I_stimulus=0.0)  # I_stim=0 matches dead zone sweep protocol (preprint Section 4.4)
        if step >= int(n_steps * 0.75):
            H = calculate_continuous_entropy(net.model.v, bins=100)
            H_history.append(H)
    
    return np.mean(H_history) if H_history else 0.0

def main():
    seeds = [42, 123, 777]
    results = {}
    
    configs = [
        ('spectral', [3, 5, 8]),
        ('degree_linear', [3, 5, 8]),
        ('uniform', [3, 5, 8]),
    ]
    
    print(f"{'coupling':<15} {'m':>4} {'H_stable':>10} {'std':>8}")
    print("-" * 42)
    
    for coupling, ms in configs:
        for m in ms:
            Hs = []
            for seed in seeds:
                H = run_experiment(m, coupling, seed)
                Hs.append(H)
            mean_H = np.mean(Hs)
            std_H = np.std(Hs)
            results[(coupling, m)] = (mean_H, std_H)
            print(f"{coupling:<15} {m:>4} {mean_H:>10.4f} {std_H:>8.4f}")
    
    print("\n=== DEAD ZONE FOCUS: m=5 ===")
    for coupling in ['spectral', 'degree_linear', 'uniform']:
        mean_H, std_H = results[(coupling, 5)]
        print(f"  {coupling:<15}: H_stable = {mean_H:.4f} ± {std_H:.4f}")
    
    # Check if spectral beats degree_linear on m=5
    spec_m5 = results[('spectral', 5)][0]
    deg_m5 = results[('degree_linear', 5)][0]
    print(f"\nSpectral gain over degree_linear at m=5: {spec_m5 - deg_m5:.4f} bits")
    
    return results

if __name__ == '__main__':
    main()