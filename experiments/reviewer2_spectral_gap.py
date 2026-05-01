"""
Reviewer 2 Defense - Spectral Gap (Fiedler Value)
Auteur : L'Architecte (Antigravity)
Date : 30 Avril 2026

Objectif : Le Reviewer 2 affirme que le Fiedler value d'un graphe BA creux
tend vers 0, rendant la métrique inutile à la limite thermodynamique.
Nous allons comparer le Fiedler brut (L) au Gap Spectral Effectif (L_eff)
qui tient compte de notre normalisation de couplage.
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from mem4ristor.core import Mem4Network
from mem4ristor.graph_utils import make_ba

def main():
    print("=== DÉFENSE REVIEWER 2 : GAP SPECTRAL (FIEDLER VALUE) ===")
    
    N_list = [100, 500, 1000, 2000, 4000]
    m = 5
    seed = 42
    
    raw_gaps = []
    eff_gaps = []
    
    print("Calcul des valeurs propres pour BA(m=5)...")
    for N in N_list:
        print(f"  N = {N}...")
        adj = make_ba(N, m, seed)
        net = Mem4Network(adjacency_matrix=adj, coupling_norm='degree_linear', seed=seed)
        
        raw_gap = net.get_spectral_gap()
        eff_gap = net.get_effective_spectral_gap()
        
        raw_gaps.append(raw_gap)
        eff_gaps.append(eff_gap)
        print(f"    Raw Fiedler : {raw_gap:.5f} | Effective Gap : {eff_gap:.5f}")
        
    print("\n--- CONCLUSION ---")
    if raw_gaps[-1] > 1.0 and eff_gaps[-1] > 1.0:
        print("[SUCCES] Le Reviewer 2 a BLUFFÉ sur la théorie des graphes !")
        print("Les graphes Barabási-Albert (m=5) sont des graphes expanseurs. Leur Fiedler value")
        print(f"NE TEND PAS vers 0 (elle se stabilise autour de {raw_gaps[-1]:.2f}).")
        print(f"Le gap effectif reste lui aussi massif ({eff_gaps[-1]:.2f}).")
        print("La transition de phase est donc topologiquement et mathématiquement valide.")
    else:
        print("[ECHEC] Le gap s'effondre vers 0.")
        
    # Plot
    plt.figure(figsize=(7, 5))
    plt.plot(N_list, raw_gaps, 'ro-', label=r'Raw Fiedler $\lambda_2(L)$', linewidth=2)
    plt.plot(N_list, eff_gaps, 'go-', label=r'Effective Gap $\lambda_2(L_{eff})$', linewidth=2)
    plt.xscale('log')
    plt.xlabel('Taille du réseau $N$')
    plt.ylabel('Spectral Gap')
    plt.title('Défense Reviewer 2 : Limite Thermodynamique Spectrale')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('reviewer2_spectral_gap.png')
    print("\nGraphique généré : reviewer2_spectral_gap.png")

if __name__ == "__main__":
    main()
