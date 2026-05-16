import sys
import os
sys.path.insert(0, '../src')
from mem4ristor.graph_utils import make_ba
import numpy as np

def f(p):
    adj = make_ba(800, 3, 42)
    rng = np.random.RandomState(42)
    n = 800
    i_idx, j_idx = np.triu_indices(n, k=1)
    mask = (adj[i_idx, j_idx] == 0) & (rng.rand(len(i_idx)) < p)
    adj[i_idx[mask], j_idx[mask]] = 1
    adj[j_idx[mask], i_idx[mask]] = 1
    degree = adj.sum(axis=1)
    L = np.diag(degree) - adj
    return float(np.sort(np.linalg.eigvalsh(L))[1])

for p in np.linspace(0.0, 0.05, 15):
    print(f"p={p:.4f} -> lambda2={f(p):.4f}")
