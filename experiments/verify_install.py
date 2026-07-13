#!/usr/bin/env python3
"""Quick verification script for Mem4ristor TEST_HERMES"""
import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from mem4ristor.core import Mem4Network

np.random.seed(42)

# Test 1: Basic chimera on lattice
net = Mem4Network(size=10, heretic_ratio=0.15)
for _ in range(500):
    net.step(I_stimulus=0.5)

H = net.calculate_entropy()
u_range = (net.u.min(), net.u.max())
print("Test 1 - Lattice 10x10:")
print("  H_cog = %.4f" % H)
print("  u range = [%.4f, %.4f]" % (u_range[0], u_range[1]))
print("  Heretics = %d/%d" % (net.heretic_mask.sum(), net.N))

# Test 2: Scale-free with degree_linear normalization
import networkx as nx
try:
    G = nx.barabasi_albert_graph(100, 3, seed=42)
    adj = nx.to_numpy_array(G)
    net2 = Mem4Network(adjacency_matrix=adj, heretic_ratio=0.15,
                       coupling_norm='degree_linear', seed=42)
    for _ in range(500):
        net2.step(I_stimulus=0.5)
    H2 = net2.calculate_entropy()
    print("\nTest 2 - BA scale-free (degree_linear):")
    print("  H_cog = %.4f" % H2)
    print("  u range = [%.4f, %.4f]" % (net2.u.min(), net2.u.max()))
except Exception as e:
    print("\nTest 2 - FAILED: %s" % e)
    H2 = None

print("\nVerification: COMPLETE")
if H > 0 and H2 is not None and H2 > 0:
    print("STATUS: OK - Both tests passed")
elif H > 0 and H2 is None:
    print("STATUS: PARTIAL - Lattice OK, SF failed (networkx issue)")
else:
    print("STATUS: CHECK NEEDED")
