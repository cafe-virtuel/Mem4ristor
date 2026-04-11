import os
import sys
# Resolve Paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '../../'))
sys.path.append(os.path.join(ROOT_DIR, 'src'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm

from mem4ristor.core import Mem4Network

def run_topological_attack():
    print("🕸️ TOPOLOGICAL ATTACK: BEYOND THE GRID (v2.3)")
    print("---------------------------------------------")
    
    N = 100
    ratios = np.linspace(0.0, 0.4, 11)
    topologies = ['Grid (2D)', 'Small-World (WS)', 'Random (ER)']
    steps = 3000
    i_stim = 1.1
    
    results = []
    
    for topo in topologies:
        print(f"\nTesting Topology: {topo}")
        
        # Generation de l'adjacence
        if topo == 'Grid (2D)':
            # On utilise le modele par défaut qui fait sa grille
            adj = None 
        elif topo == 'Small-World (WS)':
            G = nx.watts_strogatz_graph(N, k=4, p=0.1)
            adj = nx.to_numpy_array(G)
            # Normalisation (moyenne des voisins)
            row_sums = adj.sum(axis=1)
            adj = adj / row_sums[:, np.newaxis]
        elif topo == 'Random (ER)':
            G = nx.erdos_renyi_graph(N, p=0.04) # p tel que k~4
            adj = nx.to_numpy_array(G)
            # Gérer les noeuds isolés
            row_sums = adj.sum(axis=1)
            row_sums[row_sums == 0] = 1
            adj = adj / row_sums[:, np.newaxis]
            
        for r in tqdm(ratios, desc=f"Sweep {topo}"):
            # On utilise une taille fictive de 10 car N=100
            model = Mem4Network(size=10, heretic_ratio=r, seed=42, adjacency_matrix=adj)
            
            for t in range(steps):
                model.step(I_stimulus=i_stim)
            
            h = model.calculate_entropy()
            results.append({'Topology': topo, 'Ratio': r, 'Entropy': h})
            
    df = pd.DataFrame(results)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    for topo in topologies:
        subset = df[df['Topology'] == topo]
        plt.plot(subset['Ratio'], subset['Entropy'], marker='o', label=topo)
        
    plt.axvline(x=0.15, color='black', linestyle='--', alpha=0.5, label='15% Threshold')
    plt.xlabel('Heretic Ratio ($\eta$)')
    plt.ylabel('Final Entropy (H)')
    plt.title('Topological Invariance of the 15% Heretic Law\n(Grid vs Complex Networks)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    results_dir = os.path.join(ROOT_DIR, "results")
    os.makedirs(os.path.join(results_dir, 'plots'), exist_ok=True)
    plt.savefig(os.path.join(results_dir, 'plots/topological_attack_comparison.png'), dpi=300)
    print(f"\n✅ Comparison plot saved to {os.path.join(results_dir, 'plots/topological_attack_comparison.png')}")
    
    # Conclusion textuelle
    avg_h_at_15 = df[np.isclose(df['Ratio'], 0.16, atol=0.05)]['Entropy'].mean()
    print(f"\nMean Entropy at ~15%: {avg_h_at_15:.4f}")
    if avg_h_at_15 > 1.0:
        print("🏆 [VERDICT] The 15% Law is TOPOLOGICALLY ROBUST.")
    else:
        print("⚠️ [CAUTION] The model shows sensitivity to network topology.")

if __name__ == "__main__":
    run_topological_attack()
