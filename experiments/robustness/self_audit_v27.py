import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mem4ristor.core import Mem4ristorV2
import os

# MKL Determinism Fix
os.environ['NUMPY_MKL_CBWR'] = 'COMPATIBLE'

def run_self_audit():
    print("=== MEM4RISTOR V2.7 SELF-AUDIT (RED TEAM) ===")
    
    # 1. Topological Strangulation Attack
    # Can heretics survive in a Scale-Free network with a massive hub?
    strangulation_attack()
    
    # 2. Resonance Injection Attack
    # Can a frequency-matched stimulus force a global lock?
    resonance_attack()
    
    # 3. Entropy Resolution Attack
    # Is our 'diversity' real or just noise-jitter?
    entropy_resolution_attack()

def strangulation_attack():
    print("\n[ATTACK 1] Topological Strangulation (Barabási-Albert)")
    N = 100
    G = nx.barabasi_albert_graph(N, 3, seed=42)
    adj = nx.to_numpy_array(G)
    
    # Stratified heretics vs conformist hubs
    model = Mem4ristorV2(seed=42)
    model._initialize_params(N=N)
    
    # Force heretics to be the lowest degree nodes (strangulation)
    degrees = dict(G.degree())
    sorted_nodes = sorted(degrees, key=degrees.get)
    heretic_ids = sorted_nodes[:15] # 15% lowest degree
    
    model.heretic_mask = np.zeros(N, dtype=bool)
    model.heretic_mask[heretic_ids] = True
    
    # Run with high stimulus pressure on the hubs
    sol = model.solve_rk45((0, 50), I_stimulus=1.0, adj_matrix=adj)
    h_final = model.calculate_entropy()
    
    print(f"  Entropie finale (Hubs dominants) : {h_final:.4f}")
    if h_final < 0.8:
        print("  FAIL: Les hubs conformistes ont 'étouffé' la dissidence.")
    else:
        print("  PASS: La diversité survit même avec des hubs hostiles.")

def resonance_attack():
    print("\n[ATTACK 2] Resonance Injection")
    N = 50
    model = Mem4ristorV2(seed=42)
    model._initialize_params(N=N)
    adj = nx.to_numpy_array(nx.cycle_graph(N))
    
    # Oscillating stimulus I(t)
    def run_osc(freq):
        # We need to modify step to accept time-varying I
        # For simplicity, we'll step manually with Euler for this attack
        h_trace = []
        for t in range(500):
            I_t = 1.0 * np.sin(2 * np.pi * freq * t * 0.05)
            model.step(I_stimulus=I_t, coupling_input=adj @ model.v - model.v)
            if t > 400: h_trace.append(model.calculate_entropy())
        return np.mean(h_trace)

    h_low = run_osc(0.1)
    h_high = run_osc(1.0)
    print(f"  Entropie (ω=0.1) : {h_low:.4f}")
    print(f"  Entropie (ω=1.0) : {h_high:.4f}")
    
    if abs(h_low - h_high) > 0.5:
        print("  FAIL: Fragilité fréquentielle détectée.")
    else:
        print("  PASS: Robustesse temporelle validée.")

def entropy_resolution_attack():
    print("\n[ATTACK 3] Entropy Resolution (Bins check)")
    model = Mem4ristorV2(seed=42)
    model._initialize_params(N=100)
    
    # Run to stable state
    for _ in range(500): model.step(I_stimulus=0.5)
    
    v = model.v
    h_bins_2 = calculate_custom_entropy(v, bins=2)
    h_bins_10 = calculate_custom_entropy(v, bins=10)
    
    print(f"  H (2 bins) : {h_bins_2:.4f}")
    print(f"  H (10 bins) : {h_bins_10:.4f}")
    
    # If H(10) is much higher than H(2), it's just jitter
    if h_bins_10 > 2.0 * h_bins_2:
        print("  FAIL: Diversité superficielle (jitter).")
    else:
        print("  PASS: Diversité structurelle confirmée.")

def calculate_custom_entropy(v, bins):
    counts, _ = np.histogram(v, bins=bins, range=(-2.0, 2.0))
    probs = counts / np.sum(counts)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

if __name__ == "__main__":
    run_self_audit()
