"""
Reviewer 2 Defense - Initial Conditions (Symmetry Breaking)
Auteur : L'Architecte (Antigravity)
Date : 30 Avril 2026

Objectif : Le Reviewer 2 prétend que notre état chimère est un artefact 
d'une brisure de symétrie chaotique due à une condition initiale homogène v=0.
Fait : Notre code initialise par défaut de façon aléatoire (cold_start=False).
Ce script va démontrer que peu importe l'initialisation (homogène ou aléatoire), 
le réseau converge vers le MÊME attracteur macroscopique (Même Entropie H, même Kuramoto R).
L'état chimère est robuste et intrinsèque à la dynamique.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from mem4ristor.core import Mem4Network
from mem4ristor.graph_utils import make_ba
from mem4ristor.metrics import calculate_kuramoto_order_parameter

def run_experiment(cold_start, name):
    N = 100
    m = 3
    seed = 42
    steps = 3000
    
    adj = make_ba(N, m, seed)
    
    # Mode FULL (Heretics activés)
    net = Mem4Network(adjacency_matrix=adj, coupling_norm='degree_linear', seed=seed, cold_start=cold_start)
    
    v_hist = []
    w_hist = []
    
    # Transient
    for _ in range(500):
        net.step(I_stimulus=0.5)
        
    # Measurement
    for _ in range(steps):
        net.step(I_stimulus=0.5)
        v_hist.append(net.v.copy())
        w_hist.append(net.model.w.copy())
        
    v_hist = np.array(v_hist)
    w_hist = np.array(w_hist)
    
    R = calculate_kuramoto_order_parameter(v_hist, w_hist)
    H = net.calculate_entropy(bins=100)
    
    print(f"[{name}]")
    print(f"  -> Entropie spatiale (H_cont) : {H:.3f} bits")
    print(f"  -> Ordre de Kuramoto (R)      : {R:.3f}")
    return H, R

def main():
    print("=== DÉFENSE REVIEWER 2 : CONDITIONS INITIALES ===")
    
    print("\n1. Démarrage Aléatoire Bruitée (Default: cold_start=False)")
    print("   v dans [-1.5, 1.5], w dans [0, 1]")
    H_rand, R_rand = run_experiment(cold_start=False, name="Random I.C.")
    
    print("\n2. Démarrage Homogène (cold_start=True)")
    print("   v=0, w=0 pour tous les nœuds")
    H_cold, R_cold = run_experiment(cold_start=True, name="Homogeneous I.C.")
    
    print("\n--- CONCLUSION ---")
    diff_H = abs(H_rand - H_cold)
    diff_R = abs(R_rand - R_cold)
    
    print(f"Différence d'Entropie : {diff_H:.3f}")
    print(f"Différence Kuramoto R : {diff_R:.3f}")
    
    if diff_H < 0.5 and diff_R < 0.2:
        print("[SUCCES] Le Reviewer 2 a TORT : L'état chimère est atteint indépendamment des conditions initiales.")
        print("C'est un véritable attracteur structurel du réseau Mem4ristor.")
    else:
        print("[ECHEC] Le système est dépendant des conditions initiales.")

if __name__ == "__main__":
    main()
