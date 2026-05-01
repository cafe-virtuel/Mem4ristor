"""
Reviewer 2 Defense - Kuramoto Order Parameter
Auteur : L'Architecte (Antigravity)
Date : 30 Avril 2026

Objectif : Le Reviewer 2 prétend que la diversité mesurée par l'entropie (H_cont)
n'est qu'un artefact de jitter thermique autour d'une "variété synchronisée stable",
et que le Paramètre d'Ordre de Kuramoto (R) le prouverait.

On va donc calculer R.
- Si R ≈ 1, le Reviewer 2 a raison (chaos microscopique, synchronisation macroscopique).
- Si R ≈ 0, le Reviewer 2 a tort (désynchronisation macroscopique réelle).
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from mem4ristor.core import Mem4Network
from mem4ristor.graph_utils import make_ba
from mem4ristor.metrics import calculate_kuramoto_order_parameter

def main():
    print("=== DÉFENSE REVIEWER 2 : KURAMOTO ORDER PARAMETER ===")
    
    N = 100
    m = 5
    seed = 42
    steps = 3000
    
    adj = make_ba(N, m, seed)
    
    # --- 1. Dead Zone (Sans Bruit) ---
    print("\n1. Simulation Dead Zone (Baseline)...")
    net_dz = Mem4Network(adjacency_matrix=adj, coupling_norm='degree_linear', seed=seed)
    
    v_hist_dz = []
    w_hist_dz = []
    for _ in range(steps):
        net_dz.step(I_stimulus=0.0)
        v_hist_dz.append(net_dz.v.copy())
        w_hist_dz.append(net_dz.model.w.copy())
    
    v_hist_dz = np.array(v_hist_dz)
    w_hist_dz = np.array(w_hist_dz)
    R_dz = calculate_kuramoto_order_parameter(v_hist_dz, w_hist_dz)
    H_dz = net_dz.calculate_entropy(bins=100)
    print(f"  -> Dead Zone : H_cont = {H_dz:.3f} | Kuramoto R = {R_dz:.3f}")
    
    # --- 2. Escape (Bruit Fort, pas de heretics/u) - Résonance Stochastique pure ---
    print("\n2. Simulation Escape (Reviewer 2 claim: 'just thermal noise')...")
    net_esc = Mem4Network(adjacency_matrix=adj, coupling_norm='degree_linear', seed=seed)
    net_esc.model.cfg['noise']['sigma_v'] = 0.5
    
    v_hist_esc = []
    w_hist_esc = []
    for _ in range(steps):
        net_esc.step(I_stimulus=0.0)
        v_hist_esc.append(net_esc.v.copy())
        w_hist_esc.append(net_esc.model.w.copy())
        
    v_hist_esc = np.array(v_hist_esc)
    w_hist_esc = np.array(w_hist_esc)
    R_esc = calculate_kuramoto_order_parameter(v_hist_esc, w_hist_esc)
    H_esc = net_esc.calculate_entropy(bins=100)
    print(f"  -> Bruit Thermique : H_cont = {H_esc:.3f} | Kuramoto R = {R_esc:.3f}")
    
    # --- 3. Mem4ristor FULL (Doute + Heretics) forcé ---
    print("\n3. Simulation Mem4ristor FULL (Doute + Hérétiques sous forçage)...")
    net_full = Mem4Network(adjacency_matrix=adj, coupling_norm='degree_linear', seed=seed)
    # Le vrai escape Mem4ristor nécessite un forçage pour activer les hérétiques
    v_hist_full = []
    w_hist_full = []
    for _ in range(steps):
        net_full.step(I_stimulus=0.5)
        v_hist_full.append(net_full.v.copy())
        w_hist_full.append(net_full.model.w.copy())
        
    v_hist_full = np.array(v_hist_full)
    w_hist_full = np.array(w_hist_full)
    R_full = calculate_kuramoto_order_parameter(v_hist_full, w_hist_full)
    H_full = net_full.calculate_entropy(bins=100)
    print(f"  -> FULL Mem4ristor : H_cont = {H_full:.3f} | Kuramoto R = {R_full:.3f}")
    
    print("\n--- CONCLUSION ---")
    if R_esc < 0.2 and R_full < 0.2:
        print("[SUCCES] Le Reviewer 2 a TORT : R est proche de 0. Ce n'est PAS une variété synchronisée perturbée par du bruit.")
        print("La désynchronisation macroscopique est réelle.")
    else:
        print("[ECHEC] Le Reviewer 2 a RAISON : R reste élevé. La diversité mesurée n'est qu'un jitter autour d'un consensus macroscopique.")

if __name__ == "__main__":
    main()
