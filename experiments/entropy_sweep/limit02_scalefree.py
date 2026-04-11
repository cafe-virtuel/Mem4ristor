#!/usr/bin/env python3
"""
LIMIT-02 Investigation: Scale-Free Hub Strangulation & V4 Rewiring
Date: 2026-03-21 | Investigator: Claude Opus 4.6

See experiments/entropy_sweep/README.md for context.
Run: python limit02_scalefree.py
"""
import numpy as np, time, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from mem4ristor.core import Mem4Network

def make_barabasi_albert(N, m=3, seed=42):
    rng = np.random.RandomState(seed)
    adj = np.zeros((N, N), dtype=float)
    for i in range(m+1):
        for j in range(i+1, m+1): adj[i,j] = adj[j,i] = 1
    for node in range(m+1, N):
        deg = np.sum(adj, axis=1); deg[:node] += 1e-6
        p = deg[:node] / np.sum(deg[:node])
        targets = set()
        while len(targets) < m: targets.add(rng.choice(node, p=p))
        for t in targets: adj[node,t] = adj[t,node] = 1
    return adj

def run_sf(N=100, m=3, hr=0.15, rw_thresh=0.8, rw_cool=50, enable_rw=True,
           hub_heretics=False, n_steps=3000, seed=42, I_stim=0.0):
    adj = make_barabasi_albert(N, m, seed)
    net = Mem4Network(adjacency_matrix=adj.copy(), heretic_ratio=hr, seed=seed,
                      rewire_threshold=rw_thresh if enable_rw else 999.0, rewire_cooldown=rw_cool)
    if hub_heretics:
        deg = np.sum(adj, axis=1); n_h = int(N*hr)
        net.model.heretic_mask = np.zeros(N, dtype=bool)
        net.model.heretic_mask[np.argsort(deg)[-n_h:]] = True
    ent = []
    for s in range(n_steps):
        net.step(I_stim)
        if s % 10 == 0: ent.append(net.calculate_entropy())
    ent = np.array(ent)
    return {'H_stable': np.mean(ent[int(len(ent)*0.75):]), 'H_peak': np.max(ent),
            'rewires': net.rewire_count, 'max_deg': np.max(np.sum(net.adjacency_matrix, axis=1))}

if __name__ == '__main__':
    seeds = [42, 137, 314, 256, 999]
    print("LIMIT-02: Scale-Free Hub Strangulation\n")
    for label, kw in [("SF baseline", dict(enable_rw=False)),
                       ("SF + V4 rewiring", dict(enable_rw=True)),
                       ("SF + hub heretics", dict(enable_rw=False, hub_heretics=True)),
                       ("SF + hub heretics + rw", dict(enable_rw=True, hub_heretics=True))]:
        res = [run_sf(seed=s, **kw) for s in seeds]
        h = np.mean([r['H_stable'] for r in res])
        print(f"  {label:<30s}: H_stable={h:.4f}")
    # Control
    res = []
    for s in seeds:
        net = Mem4Network(size=10, heretic_ratio=0.15, seed=s)
        ent = []
        for i in range(3000):
            net.step(0.0)
            if i % 10 == 0: ent.append(net.calculate_entropy())
        res.append(np.mean(ent[int(len(ent)*0.75):]))
    print(f"  {'Lattice 10x10 (control)':<30s}: H_stable={np.mean(res):.4f}")
