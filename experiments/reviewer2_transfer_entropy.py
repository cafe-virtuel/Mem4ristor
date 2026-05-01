"""
Reviewer 2 Defense - Causal Illusion (Transfer Entropy)
Auteur : L'Architecte (Antigravity)
Date : 30 Avril 2026

Objectif : Prouver que l'information circule asymétriquement des Hérétiques vers les Normaux.
Le Reviewer 2 clame que l'Information Mutuelle est symétrique. Nous utilisons ici
l'Entropie de Transfert (Transfer Entropy) discrète sur les états cognitifs 
pour prouver la Causalité de Granger directionnelle.
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from mem4ristor.core import Mem4Network
from mem4ristor.graph_utils import make_ba
from mem4ristor.metrics import get_cognitive_states, calculate_transfer_entropy

def main():
    print("=== DÉFENSE REVIEWER 2 : CAUSALITÉ (TRANSFER ENTROPY) ===")
    
    N = 100
    m = 5
    seed = 42
    steps = 10000
    
    adj = make_ba(N, m, seed)
    net = Mem4Network(adjacency_matrix=adj, coupling_norm='degree_linear', seed=seed)
    
    # We need to force heretics to be active and normal nodes to be receptive
    # The default behavior forces I_stimulus=0.5
    v_hist = []
    
    # Transient
    for _ in range(500):
        net.step(I_stimulus=0.5)
        
    for _ in range(steps):
        net.step(I_stimulus=0.5)
        v_hist.append(net.v.copy())
        
    v_hist = np.array(v_hist)
    
    # Subsample to avoid Y_t == Y_{t-1} due to small dt=0.05.
    v_hist = v_hist[::10]
    steps_sub = len(v_hist)
    
    # Binarize signal to 2 states using median to maximize entropy
    cog_hist = (v_hist > np.median(v_hist, axis=0)).astype(int)
    
    # Identify heretics and normals
    heretics = np.where(net.model.heretic_mask)[0]
    normals = np.where(~net.model.heretic_mask)[0]
    
    if len(heretics) == 0 or len(normals) == 0:
        print("Erreur: Pas assez d'hérétiques ou de normaux.")
        return
        
    print(f"Réseau simulé: {len(heretics)} hérétiques, {len(normals)} normaux.")
    print("Calcul de l'Entropie de Transfert sur 200 paires croisées...")
    
    n_pairs = 200
    te_h_to_n = []
    te_n_to_h = []
    
    rng = np.random.RandomState(42)
    for _ in range(n_pairs):
        h = rng.choice(heretics)
        n = rng.choice(normals)
        
        # print(f"h mean: {np.mean(cog_hist[:, h])}, n mean: {np.mean(cog_hist[:, n])}")
        
        # TE from Heretic to Normal
        te_1 = calculate_transfer_entropy(cog_hist[:, h], cog_hist[:, n], bins=2)
        te_h_to_n.append(te_1)
        
        # TE from Normal to Heretic
        te_2 = calculate_transfer_entropy(cog_hist[:, n], cog_hist[:, h], bins=2)
        te_n_to_h.append(te_2)
        
    te_h_to_n = np.array(te_h_to_n)
    te_n_to_h = np.array(te_n_to_h)
    
    mean_hn = np.mean(te_h_to_n)
    mean_nh = np.mean(te_n_to_h)
    
    print(f"  -> TE(Hérétique -> Normal) : {mean_hn:.5f} bits")
    print(f"  -> TE(Normal -> Hérétique) : {mean_nh:.5f} bits")
    
    asymmetry = mean_hn / (mean_nh + 1e-9)
    print(f"  -> Ratio d'asymétrie (Flux) : {asymmetry:.2f}x")
    
    print("\n--- CONCLUSION ---")
    if mean_nh > mean_hn * 1.5:
        print("[ECHEC PARTIEL] Le Reviewer 2 a soulevé un point valide : L'information circule asymétriquement, mais des NORMAUX vers les HERETIQUES !")
        print("Les Hérétiques ne sont pas les 'dictateurs' du réseau, ils sont bombardés par la majorité et réagissent.")
    elif mean_hn > mean_nh * 1.5:
        print("[SUCCES] L'information circule asymétriquement des Hérétiques vers les Normaux.")
    else:
        print("[ECHEC] Pas de causalité asymétrique forte détectée.")
        
    # Plot violin
    plt.figure(figsize=(6, 5))
    plt.violinplot([te_h_to_n, te_n_to_h], showmeans=True)
    plt.xticks([1, 2], ['Hérétique -> Normal', 'Normal -> Hérétique'])
    plt.ylabel('Transfer Entropy (bits)')
    plt.title('Défense Reviewer 2 : Causalité d\'Information')
    plt.grid(True, alpha=0.3)
    plt.savefig('reviewer2_causality.png')
    print("\nGraphique généré : reviewer2_causality.png")

if __name__ == "__main__":
    main()
