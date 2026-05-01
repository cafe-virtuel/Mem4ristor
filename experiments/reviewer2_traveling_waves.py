"""
Reviewer 2 Defense - Zero-Lag Blindness & Traveling Waves
Auteur : L'Architecte (Antigravity)
Date : 30 Avril 2026

Objectif : Le Reviewer 2 affirme que notre "désynchronisation" (entropie spatiale élevée)
est simplement une onde progressive (traveling wave). Une onde progressive a une faible 
corrélation à lag=0, mais une très forte corrélation à lag=tau (déphasage).

Méthode : Calculer la Time-Lagged Cross-Correlation (TLCC) maximale pour toutes 
les paires de nœuds.
- Si max_tau(TLCC) ≈ 1 pour tous, c'est une onde progressive rigide.
- Si max_tau(TLCC) est faible, c'est une véritable désynchronisation spatio-temporelle.
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import correlate

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from mem4ristor.core import Mem4Network
from mem4ristor.graph_utils import make_ba

def max_time_lagged_cross_correlation(v1, v2):
    # Normalize signals
    v1_n = (v1 - np.mean(v1)) / (np.std(v1) + 1e-9)
    v2_n = (v2 - np.mean(v2)) / (np.std(v2) + 1e-9)
    
    # Compute cross-correlation
    corr = correlate(v1_n, v2_n, mode='full') / len(v1)
    
    # Return the maximum correlation across all lags
    return np.max(corr)

def main():
    print("=== DÉFENSE REVIEWER 2 : ZERO-LAG BLINDNESS ===")
    
    N = 100
    m = 3
    seed = 42
    steps = 2000
    
    adj = make_ba(N, m, seed)
    
    # --- 1. Dead Zone (Sans Bruit) - Référence Synchronisée ---
    print("\n1. Simulation Dead Zone (Synchronisation)...")
    net_dz = Mem4Network(adjacency_matrix=adj, coupling_norm='degree_linear', seed=seed)
    
    v_hist_dz = []
    for _ in range(steps):
        net_dz.step(I_stimulus=0.0)
        v_hist_dz.append(net_dz.v.copy())
    v_hist_dz = np.array(v_hist_dz)[steps//2:] # Keep second half
    
    # --- 2. Mem4ristor FULL (Doute + Heretics) forcé ---
    print("2. Simulation Mem4ristor FULL (Désynchronisation structurée)...")
    net_full = Mem4Network(adjacency_matrix=adj, coupling_norm='degree_linear', seed=seed)
    v_hist_full = []
    for _ in range(steps):
        net_full.step(I_stimulus=0.5)
        v_hist_full.append(net_full.v.copy())
    v_hist_full = np.array(v_hist_full)[steps//2:] # Keep second half
    
    # --- 3. Calcul TLCC Max sur N=500 paires aléatoires ---
    print("\nCalcul de la TLCC maximale (Time-Lagged Cross-Correlation)...")
    n_pairs = 500
    pairs = [(np.random.randint(0, N), np.random.randint(0, N)) for _ in range(n_pairs)]
    # Remove i=j
    pairs = [p for p in pairs if p[0] != p[1]]
    
    tlcc_dz = []
    tlcc_full = []
    
    for i, j in pairs:
        tlcc_dz.append(max_time_lagged_cross_correlation(v_hist_dz[:, i], v_hist_dz[:, j]))
        tlcc_full.append(max_time_lagged_cross_correlation(v_hist_full[:, i], v_hist_full[:, j]))
        
    tlcc_dz = np.array(tlcc_dz)
    tlcc_full = np.array(tlcc_full)
    
    print(f"  -> Dead Zone : Moyenne Max-TLCC = {np.mean(tlcc_dz):.3f} ± {np.std(tlcc_dz):.3f}")
    print(f"  -> FULL Mem4ristor : Moyenne Max-TLCC = {np.mean(tlcc_full):.3f} ± {np.std(tlcc_full):.3f}")
    
    print("\n--- CONCLUSION ---")
    if np.mean(tlcc_full) < 0.6:
        print("[SUCCES] Le Reviewer 2 a TORT : Même avec un décalage temporel libre, les nœuds ne sont pas corrélés.")
        print("Ce N'EST PAS une onde progressive. C'est une véritable diversité spatio-temporelle.")
    else:
        print("[ECHEC] Le Reviewer 2 a RAISON : Les nœuds sont fortement corrélés avec un déphasage.")
        print("C'est une onde progressive rigide.")
        
    # Plot violin
    plt.figure(figsize=(6, 5))
    plt.violinplot([tlcc_dz, tlcc_full], showmeans=True)
    plt.xticks([1, 2], ['Dead Zone', 'FULL Mem4ristor'])
    plt.ylabel('Max Time-Lagged Cross-Correlation')
    plt.title('Défense Reviewer 2 : Absence d\'Ondes Progressives')
    plt.grid(True, alpha=0.3)
    plt.savefig('reviewer2_traveling_waves.png')
    print("\nGraphique généré : reviewer2_traveling_waves.png")

if __name__ == "__main__":
    main()
