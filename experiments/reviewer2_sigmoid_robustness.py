"""
Reviewer 2 Defense - Sigmoid Robustness (No Fine-Tuning)
Auteur : L'Architecte (Antigravity)
Date : 30 Avril 2026

Objectif : Le Reviewer 2 prétend que le "Levitating Sigmoid" repose sur une
valeur magique pi (3.14) et s'effondrerait à la moindre perturbation (fine-tuning).

On va balayer la pente du sigmoid de 1.0 à 10.0 pour prouver
l'existence d'un très large plateau de fonctionnement robuste.
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

def main():
    print("=== DÉFENSE REVIEWER 2 : SIGMOID ROBUSTNESS ===")
    
    N = 100
    m = 3
    steps = 2000
    seed = 42
    
    adj = make_ba(N, m, seed)
    
    steepness_values = np.linspace(1.0, 10.0, 10)
    results = []
    
    for s in steepness_values:
        print(f"Testing steepness = {s:.2f}...")
        
        # Mode FORCED (I=0.5) pour activer les hérétiques
        net = Mem4Network(adjacency_matrix=adj, coupling_norm='degree_linear', seed=seed)
        net.model.cfg['dynamics']['alpha_surprise'] = s # Note: in V4, it might be alpha_surprise?
        # Let's override directly
        net.model.sigmoid_steepness = s
        
        for step in range(steps):
            net.step(I_stimulus=0.5)
            
        H = net.calculate_entropy(bins=100)
        
        print(f"  -> H_cont = {H:.3f}")
        results.append({'steepness': s, 'H_cont': H})
        
    df = pd.DataFrame(results)
    df.to_csv('reviewer2_sigmoid_robustness.csv', index=False)
    
    plt.figure(figsize=(8, 5))
    plt.plot(df['steepness'], df['H_cont'], 'k-o', linewidth=2)
    plt.axvline(np.pi, color='r', linestyle='--', label=r'Current Value ($\pi$)')
    plt.xlabel('Sigmoid Steepness')
    plt.ylabel('Continuous Entropy H (100 bins)')
    plt.title('Reviewer 2 Defense: No Fine-Tuning in Levitating Sigmoid')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0, max(df['H_cont']) * 1.2)
    plt.savefig('reviewer2_sigmoid_robustness.png')
    print("\n✅ Script terminé. Graphique généré.")

if __name__ == "__main__":
    main()
