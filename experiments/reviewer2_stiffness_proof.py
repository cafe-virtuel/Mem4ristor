"""
Reviewer 2 Defense - Stiffness & Euler Stability
Auteur : L'Architecte (Antigravity)
Date : 30 Avril 2026

Objectif : Le Reviewer 2 affirme que dt=0.05 est inconditionnellement instable
pour l'oscillateur FHN couplé (stiffness), et que notre dynamique n'est qu'un
artefact d'instabilité numérique ("ping-ponging").

Preuve mathématique :
On calcule le Jacobien complet (2N x 2N) du système à chaque pas de temps.
Pour que la méthode d'Euler explicite soit stable, il faut que pour toutes les
valeurs propres lambda du Jacobien : |1 + lambda * dt| <= 1.
Si cette condition est respectée, le Reviewer 2 a mathématiquement tort.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from mem4ristor.core import Mem4Network
from mem4ristor.graph_utils import make_ba

def main():
    print("=== DÉFENSE REVIEWER 2 : EULER STABILITY PROOF ===")
    
    N = 30 # Small N for fast eigenvalue computation (O(N^3))
    m = 3
    seed = 42
    dt = 0.05
    steps = 1000
    
    adj = make_ba(N, m, seed)
    net = Mem4Network(adjacency_matrix=adj, coupling_norm='degree_linear', seed=seed)
    # Enable scale invariant coupling for mathematical correctness
    degrees = np.maximum(np.array(adj.sum(axis=1)).flatten(), 1.0)
    net.model.D_eff = net.model.cfg['coupling']['D']
    net.node_weights = (1.0 / degrees) / np.mean(1.0 / degrees)
    
    eps = net.model.cfg['dynamics']['epsilon']
    b = net.model.cfg['dynamics']['b']
    divisor = net.model.cfg['dynamics']['v_cubic_divisor']
    
    L = net.L.toarray() if hasattr(net.L, 'toarray') else net.L
    
    max_modulus_history = []
    
    print(f"Simulation de {steps} pas (N={N})... Calcul des valeurs propres O(N^3).")
    
    for step in range(steps):
        net.step(I_stimulus=0.5) # Forced to trigger wide range of v
        
        # Build full 2Nx2N Jacobian
        v = net.v
        
        # J_vv = diag(1 - 3*v^2 / divisor) - D_eff * scale_factors * L
        D_eff = net.model.D_eff
        scale_factors = net.node_weights
        
        # Compute effective laplacian operator on v
        # l_v = -(L @ v) * scale_factors * D_eff
        # So d(l_v_i) / d v_j = - L_ij * scale_factors[i] * D_eff
        
        L_scaled = L * scale_factors[:, np.newaxis] * D_eff
        J_vv = np.diag(1.0 - 3.0 * (v**2) / divisor) - L_scaled
        
        J_vw = -np.eye(N)
        J_wv = eps * np.eye(N)
        J_ww = -eps * b * np.eye(N)
        
        J_top = np.hstack((J_vv, J_vw))
        J_bot = np.hstack((J_wv, J_ww))
        J = np.vstack((J_top, J_bot))
        
        # Eigenvalues
        eigenvalues = np.linalg.eigvals(J)
        
        # Euler stability criterion: |1 + lambda * dt| <= 1
        stability_modulus = np.abs(1.0 + eigenvalues * dt)
        
        max_modulus = np.max(stability_modulus)
        max_modulus_history.append(max_modulus)
        
    max_modulus_history = np.array(max_modulus_history)
    
    print("\n--- RÉSULTATS ---")
    print(f"Max modulus observé sur toute la trajectoire : {np.max(max_modulus_history):.5f}")
    if np.max(max_modulus_history) <= 1.0 + 1e-4:
        print("[SUCCES] L'intégrateur Euler est STABLE. Le Reviewer 2 a TORT.")
    else:
        print("[ECHEC] Le Reviewer 2 a RAISON, l'intégrateur est instable localement.")
        
    plt.figure(figsize=(8, 4))
    plt.plot(max_modulus_history, 'b-', label='Max |1 + \lambda dt|')
    plt.axhline(1.0, color='r', linestyle='--', label='Limite de stabilité (=1)')
    plt.xlabel('Timestep')
    plt.ylabel('Stabilité (<= 1 requis)')
    plt.title('Preuve Mathématique de Stabilité (Défense Reviewer 2)')
    plt.legend()
    plt.savefig('reviewer2_stiffness_proof.png')
    print("Graphique sauvegardé : reviewer2_stiffness_proof.png")

if __name__ == "__main__":
    main()
