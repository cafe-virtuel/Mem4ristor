import os
import sys
import numpy as np
import json
import time
import networkx as nx

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from mem4ristor.core import Mem4ristorV2
from mem4ristor.benchmarks.engine import KuramotoModel, VoterModel, ConsensusModel

class Mem4Benchmark:
    def __init__(self, G, seed=42):
        self.N = G.number_of_nodes()
        self.model = Mem4ristorV2(seed=seed)
        self.model._initialize_params(self.N)
        self.adj = nx.to_numpy_array(G)
        deg = np.sum(self.adj, axis=1)
        deg[deg == 0] = 1
        self.adj = self.adj / deg[:, None]
    def step(self, I_stim=0.5): self.model.step(I_stim, self.adj)
    def get_states(self): return self.model.v

def calculate_mds(v):
    v_binned = np.digitize(v, [-99, -1.5, -0.8, 0.8, 1.5, 99]) - 1
    counts = np.bincount(v_binned, minlength=5)
    probs = counts / len(v)
    p_nz = probs[probs > 0.01] # 1% threshold
    h = -np.sum(p_nz * np.log2(p_nz)) if len(p_nz) > 1 else 0.0
    n_occ = len(p_nz)
    return h * (n_occ / 5.0), h, n_occ

def run_benchmark(model_class, name, G, n_runs=5, steps=500, bias_val=0.5):
    results = []
    for r in range(n_runs):
        m = model_class(G, seed=42+r) if model_class == Mem4Benchmark else model_class(G.number_of_nodes(), seed=42+r)
        for _ in range(steps): m.step(I_stim=bias_val)
        mds, h, n_occ = calculate_mds(m.get_states())
        results.append(mds)
    return np.mean(results), np.std(results)

def main():
    N = 100
    G = nx.watts_strogatz_graph(N, 4, 0.1, seed=42)
    models = [(Mem4Benchmark, "Mem4Ristor v2.0.4.1"), (ConsensusModel, "Consensus"), (VoterModel, "Voter"), (KuramotoModel, "Kuramoto")]
    print(f"[INFO] Benchmarking MDS (Multimodal Diversity Score) | N={N}, steps=500, bias=0.5")
    print("| Model | MDS (Mean) | Std Dev | Status |")
    print("|-------|------------|---------|--------|")
    for m, name in models:
        mean, std = run_benchmark(m, name, G)
        status = "HEALTHY" if mean > 0.2 else "COLLAPSED"
        print(f"| {name} | {mean:.3f} | {std:.3f} | {status} |")

if __name__ == "__main__":
    main()
