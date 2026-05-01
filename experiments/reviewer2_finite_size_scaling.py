"""
Reviewer 2 Defense - Finite Size Scaling & Thermodynamic Limit
Auteur : L'Architecte (Antigravity)
Date : 30 Avril 2026

Objectif : Vérifier l'attaque du Reviewer 2 qui affirme que la "Dead Zone" et "l'Escape"
sont des illusions de taille finie car le couplage s'effondre avec 1/sqrt(N).

Protocole :
1. Tester N = [64, 256, 1024, 4096]
2. Mesurer l'entropie H_cont sur un graphe Barabasi-Albert (m=5, sparse)
3. Constater l'effondrement (le Reviewer 2 a raison sur l'existant).
4. Activer un nouveau mode 'scale_invariant' = True et relancer.
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from mem4ristor.core import Mem4Network
from mem4ristor.graph_utils import make_ba

def run_scaling_experiment(N_list, scale_invariant=False, seed=42):
    results = []
    
    for N in N_list:
        print(f"Testing N={N} (Scale Invariant={scale_invariant})...")
        start_t = time.time()
        
        # Dead Zone (m=5) - pas de bruit
        adj_dz = make_ba(N, 5, seed)
        net_dz = Mem4Network(adjacency_matrix=adj_dz, coupling_norm='degree_linear', seed=seed)
        
        # FIX: On implémente le scale-invariant hack temporairement si demandé
        if scale_invariant:
            # On neutralise le 1/sqrt(N) dans dynamics
            net_dz.model.D_eff = net_dz.model.cfg['coupling']['D']
            # On neutralise le 1/sqrt(N) de target_mean dans topology
            degrees = np.maximum(np.array(adj_dz.sum(axis=1)).flatten(), 1.0)
            net_dz.node_weights = (1.0 / degrees) / np.mean(1.0 / degrees)
            
        for step in range(1500): # dt=0.05, T=75
            net_dz.step(I_stimulus=0.0)
            
        H_dz = net_dz.calculate_entropy(bins=100)
        
        # Escape (m=5) - avec bruit fort (eta=0.5)
        net_esc = Mem4Network(adjacency_matrix=adj_dz, coupling_norm='degree_linear', seed=seed)
        net_esc.model.cfg['noise']['sigma_v'] = 0.5
        
        if scale_invariant:
            net_esc.model.D_eff = net_esc.model.cfg['coupling']['D']
            net_esc.node_weights = (1.0 / degrees) / np.mean(1.0 / degrees)
            
        for step in range(1500):
            net_esc.step(I_stimulus=0.0)
            
        H_esc = net_esc.calculate_entropy(bins=100)
        
        elapsed = time.time() - start_t
        print(f"  -> Dead Zone H: {H_dz:.3f} | Escape H: {H_esc:.3f} | Time: {elapsed:.1f}s")
        
        results.append({
            'N': N,
            'Scale_Invariant': scale_invariant,
            'H_DeadZone': H_dz,
            'H_Escape': H_esc
        })
        
    return pd.DataFrame(results)

if __name__ == "__main__":
    N_list = [64, 256, 1024]
    
    print("=== PHASE 1 : Code actuel (La critique du Reviewer 2) ===")
    df_current = run_scaling_experiment(N_list, scale_invariant=False)
    
    print("\n=== PHASE 2 : Code corrigé (Indépendant de l'échelle) ===")
    df_fixed = run_scaling_experiment(N_list, scale_invariant=True)
    
    df_all = pd.concat([df_current, df_fixed], ignore_index=True)
    df_all.to_csv('reviewer2_finite_size_scaling.csv', index=False)
    
    # Plot
    plt.figure(figsize=(10, 6))
    
    # Current
    curr = df_all[df_all['Scale_Invariant'] == False]
    plt.plot(curr['N'], curr['H_DeadZone'], 'r--', marker='o', label='Dead Zone (Current 1/sqrt(N))')
    plt.plot(curr['N'], curr['H_Escape'], 'b--', marker='o', label='Escape (Current 1/sqrt(N))')
    
    # Fixed
    fixed = df_all[df_all['Scale_Invariant'] == True]
    plt.plot(fixed['N'], fixed['H_DeadZone'], 'r-', marker='s', label='Dead Zone (Scale-Invariant)')
    plt.plot(fixed['N'], fixed['H_Escape'], 'b-', marker='s', label='Escape (Scale-Invariant)')
    
    plt.xscale('log')
    plt.xticks(N_list, N_list)
    plt.xlabel('Network Size (N)')
    plt.ylabel('Continuous Entropy H (100 bins)')
    plt.title('Reviewer 2 Defense: Thermodynamic Limit & Finite Size Scaling')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('reviewer2_finite_size_scaling.png')
    print("\nGraph saved to reviewer2_finite_size_scaling.png")
