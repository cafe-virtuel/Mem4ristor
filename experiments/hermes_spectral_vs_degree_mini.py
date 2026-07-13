#!/usr/bin/env python3
"""Mini-experiment: spectral vs degree_linear head-to-head on BA m=5 dead zone.
Q: Can eigenvector-centrality-based coupling rescue diversity in the dead zone
   (m=5, lambda2~3), where degree_linear also fails?
   Paper claim: "neither normalization prevents diversity collapse"
   But the prior run showed degree_linear=3.53 > spectral=3.42 at m=5 with I_stim=0.
   We need to test at I_stim=0.5 (functional regime) to see if spectral normalization
   provides an advantage. Also test at I_stim=0 to probe the dead zone directly.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from mem4ristor.topology import Mem4Network
from mem4ristor.metrics import calculate_continuous_entropy
import networkx as nx

def run_once(adj, coupling_norm, seed, n_steps=3000, I_stim=0.5):
    np.random.seed(seed)
    net = Mem4Network(
        adjacency_matrix=adj,
        heretic_ratio=0.15,
        seed=seed,
        cold_start=True,
        coupling_norm=coupling_norm
    )
    H_hist = []
    for step in range(n_steps):
        net.step(I_stimulus=I_stim)
        if step >= int(n_steps * 0.75):
            H = calculate_continuous_entropy(net.model.v, bins=100)
            H_hist.append(H)
    return np.mean(H_hist) if H_hist else 0.0

def main():
    BA_m = 5
    N = 100
    seeds = [42, 123, 777]

    # Build ONE graph per seed (fixed topology, vary coupling)
    graphs = {}
    for seed in seeds:
        G = nx.barabasi_albert_graph(N, BA_m, seed=seed)
        graphs[seed] = nx.to_numpy_array(G)

    configs = [
        ('spectral',      0.0),
        ('spectral',      0.5),
        ('degree_linear', 0.0),
        ('degree_linear', 0.5),
        ('uniform',       0.0),
        ('uniform',      0.5),
    ]

    results = {}
    print(f"{'norm':<15} {'I_stim':>7} {'H_cont_mean':>12} {'H_cont_std':>10}  raw values")
    print("-" * 65)

    for norm, I_stim in configs:
        Hs = []
        for seed in seeds:
            adj = graphs[seed]
            H = run_once(adj, norm, seed, I_stim=I_stim)
            Hs.append(H)
        results[(norm, I_stim)] = (np.mean(Hs), np.std(Hs))
        print(f"{norm:<15} {I_stim:>7.1f} {np.mean(Hs):>12.4f} {np.std(Hs):>10.4f}  {Hs}")

    print("\n=== DEAD ZONE ANALYSIS (I_stim=0, BA m=5) ===")
    for norm in ['spectral', 'degree_linear', 'uniform']:
        mu, sd = results[(norm, 0.0)]
        print(f"  {norm:<15}: H = {mu:.4f} ± {sd:.4f}")

    print("\n=== FUNCTIONAL REGIME (I_stim=0.5, BA m=5) ===")
    for norm in ['spectral', 'degree_linear', 'uniform']:
        mu, sd = results[(norm, 0.5)]
        print(f"  {norm:<15}: H = {mu:.4f} ± {sd:.4f}")

    # lambda_2 for each graph
    print("\n=== ALGEBRAIC CONNECTIVITY ===")
    for seed in seeds:
        adj = graphs[seed]
        L = nx.laplacian_matrix(nx.from_numpy_array(adj))
        eigvals = np.sort(np.linalg.eigvals(L.toarray()))
        lam2 = eigvals[1]
        deg = np.array(adj.sum(axis=1)).flatten()
        print(f"  seed={seed}: lambda_2={lam2:.4f}, mean_deg={deg.mean():.2f}, max_deg={deg.max()}")

if __name__ == '__main__':
    main()