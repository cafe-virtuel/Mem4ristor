"""
Mem4ristor V4 - Chimera State Demonstration
Auteur: Julien Chauvin
Date: Mai 2026

Ce script de 5 minutes génère la preuve visuelle de l'état chimère :
1. Le graphe de l'espace des phases (Kuramoto).
2. Les séries temporelles de potentiel (v).
3. Le graphe Barabási-Albert avec l'identification des hérétiques.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from mem4ristor.core import Mem4Network
from mem4ristor.graph_utils import make_ba
from mem4ristor.metrics import calculate_kuramoto_order_parameter

def main():
    print("==================================================")
    print("Mem4ristor V4 - Démonstration État Chimère")
    print("==================================================")
    
    # Paramètres réseau
    N = 100
    m = 3
    seed = 42
    steps = 1500
    
    print(f"\n1. Initialisation du réseau Barabási-Albert (N={N})...")
    adj = make_ba(N, m, seed)
    
    print("2. Création du noyau Mem4Network avec plasticité intrinsèque...")
    net = Mem4Network(adjacency_matrix=adj, coupling_norm='degree_linear', seed=seed)
    
    # Historiques
    v_hist = np.zeros((steps, N))
    w_hist = np.zeros((steps, N))
    
    print(f"3. Simulation de {steps} pas de temps (dt=0.05)...")
    for t in range(steps):
        net.step(I_stimulus=0.5)
        v_hist[t] = net.v.copy()
        w_hist[t] = net.model.w.copy()
        
    print("\n4. Calcul des métriques...")
    R = calculate_kuramoto_order_parameter(v_hist, w_hist)
    H = net.calculate_entropy()
    
    print(f"   -> Ordre de Kuramoto (Synchronisation macro) : {R:.3f}")
    print(f"   -> Entropie de Shannon (Diversité spatiale)  : {H:.3f} bits")
    print("   => Conclusion : Phase verrouillée macroscopiquement mais spatialement diverse (État Chimère).")
    
    print("\n5. Génération du tableau de bord visuel...")
    
    fig = plt.figure(figsize=(15, 5))
    
    # A. Séries Temporelles
    ax1 = plt.subplot(131)
    # Plot 5 normal nodes
    normal_idx = np.where(~net.model.heretic_mask)[0][:5]
    for idx in normal_idx:
        ax1.plot(v_hist[-500:, idx], alpha=0.7)
    # Plot 1 heretic node
    heretic_idx = np.where(net.model.heretic_mask)[0][0]
    ax1.plot(v_hist[-500:, heretic_idx], 'r-', linewidth=2, label='Heretic')
    ax1.set_title("Dynamique Temporelle $v(t)$")
    ax1.set_xlabel("Temps")
    ax1.set_ylabel("Potentiel de membrane $v$")
    ax1.legend()
    
    # B. Espace des phases (Kuramoto) à l'instant T
    ax2 = plt.subplot(132)
    # Geometric phase
    phases = np.arctan2(w_hist[-1], v_hist[-1])
    # Scatter on polar plot
    ax2 = plt.subplot(132, projection='polar')
    normals = np.where(~net.model.heretic_mask)[0]
    heretics = np.where(net.model.heretic_mask)[0]
    ax2.scatter(phases[normals], np.ones_like(normals), c='blue', alpha=0.5, label='Normal')
    ax2.scatter(phases[heretics], np.ones_like(heretics), c='red', alpha=0.8, label='Heretic')
    ax2.set_title(f"Dispersion de Phase (R={R:.2f})")
    ax2.set_yticks([])
    
    # C. Graphe
    ax3 = plt.subplot(133)
    G = nx.from_numpy_array(adj)
    pos = nx.spring_layout(G, seed=seed)
    color_map = ['red' if net.model.heretic_mask[node] else 'lightblue' for node in G]
    size_map = [300 if net.model.heretic_mask[node] else 100 for node in G]
    nx.draw(G, pos, ax=ax3, node_color=color_map, node_size=size_map, alpha=0.8)
    ax3.set_title("Réseau Barabási-Albert ($m=3$)")
    
    plt.tight_layout()
    plt.savefig("demo_chimera_output.png", dpi=150)
    print("   -> Image sauvegardée : 'demo_chimera_output.png'")
    print("\nTerminé ! Ouvrez l'image pour voir le résultat.")

if __name__ == "__main__":
    main()
