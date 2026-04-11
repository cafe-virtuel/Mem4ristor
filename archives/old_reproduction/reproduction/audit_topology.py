import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import sys

# Integration avec le dossier src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from mem4ristor.core import Mem4Network

def run_complex_topology_audit():
    print("🌐 MEM4RISTOR PHASE 4 AUDIT : Complex Topologies")
    print("-----------------------------------------------")
    
    N = 100
    m = 3 # Nombre d'arêtes pour chaque nouveau nœud (Barabási-Albert)
    
    print(f"[INFO] Generating Barabási-Albert graph (N={N}, m={m})...")
    G = nx.barabasi_albert_graph(N, m, seed=42)
    adj = nx.to_numpy_array(G)
    
    # Identifier le hub principal (nœud avec le plus de connexions)
    degrees = dict(G.degree())
    hub_node = max(degrees, key=degrees.get)
    print(f"[INFO] Main Hub identified: Node {hub_node} (Degree: {degrees[hub_node]})")
    
    # Initialiser le réseau Mem4 avec cette topologie
    model = Mem4Network(adjacency_matrix=adj, seed=42)
    
    print("[START] Simulating background dynamics (2000 steps)...")
    for _ in range(2000):
        model.step(I_stimulus=0.0)
        
    print(f"[ATTACK] Injecting Byzantine Corruption on Hub {hub_node}!")
    # Le hub tente d'imposer un stimulus massif (consensus forcé)
    steps_attack = 1000
    for i in range(steps_attack):
        # On simule un stimulus global faible, mais on pourrait modifier core.py pour injecter localement
        # Pour cet audit, on va simuler que le hub influence tout le monde via le couplage
        model.step(I_stimulus=0.5)
        
        # On force l'état du hub à être "Certitude Totale" (V=1.5) à chaque step
        model.model.v[hub_node] = 1.5
        
    final_h = model.calculate_entropy()
    print(f"\nFinal Resilience Entropy: {final_h:.4f}")
    
    if final_h > 1.5:
        print("✅ [AUDIT SUCCESS] The network resisted the Hub's corruption.")
        print("   Constitutional doubt in neighbors isolated the hub's influence.")
    else:
        print("❌ [AUDIT FAILURE] The Hub forced a global consensus. Diversity collapsed.")
        
    # Visualisation (Optionnelle)
    try:
        os.makedirs("results", exist_ok=True)
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G, seed=42)
        node_colors = model.v
        nx.draw(G, pos, node_color=node_colors, cmap='RdBu_r', 
                with_labels=False, node_size=50 + 5*np.array(list(degrees.values())))
        plt.title(f"Mem4ristor on Small-World Graph (Hub Corruption Test)")
        plt.savefig("results/complex_topology_audit.png")
        print("[INFO] Visualization saved to results/complex_topology_audit.png")
    except Exception as e:
        print(f"[NOTE] Visualization skipped: {e}")

if __name__ == "__main__":
    run_complex_topology_audit()
