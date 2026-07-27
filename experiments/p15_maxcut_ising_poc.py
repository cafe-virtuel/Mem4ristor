"""
EXPERIMENT P15 : Max-Cut & Verre de Spins d'Ising (Version Auditée & Rectifiée).

Contexte :
Le problème du Max-Cut consiste à trouver la configuration de spins s_i in {-1, +1}
qui maximise la coupe W_cut = 0.25 * sum_{i,j} J_ij (1 - s_i s_j).

Audit Scientifique (27/07/2026) :
Julien (Le Barman) a exigé un audit de vérification des paramètres.
La version initiale utilisait une convention de signe pour le Recuit Simulé qui minimisait
l'énergie d'Ising ferromagnétique (alignement s_i s_j = +1), ce qui minimisait la coupe.
Pendant ce temps, le doute de Mem4ristor (u ~ 1.0 -> u_filter ~ -0.99) inversait automatiquement
le couplage, poussant Mem4ristor vers la coupe.

Après alignement rigoureux des conventions de signe (Metropolis sur W_cut) :
- Recuit Simulé (SA) : Coupe Moyenne = 125.50 (Explore les configurations discrètes de spins)
- Recherche Gloutonne  : Coupe Moyenne = 115.80
- Mem4ristor FULL      : Coupe Moyenne = 80.90 (Relaxation continue FHN, u_filter inversé)

Conclusion d'Honnêteté Scientifique :
Mem4ristor bascule naturellement en mode anti-synchronisation grâce au doute (u_filter < 0),
ce qui le pousse vers la coupe sans algorithme explicite. Néanmoins, pour la recherche combinatoire
discrète pure (Max-Cut), le Recuit Simulé discret optimisé reste supérieur à la relaxation continue FHN.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mem4ristor.graph_utils import make_ba
from mem4ristor.dynamics import Mem4ristorV3

def compute_cut_and_energy(s, J):
    energy = -0.5 * np.dot(s, np.dot(J, s))
    W_cut = 0.25 * np.sum(np.abs(J) - J * np.outer(s, s))
    return energy, W_cut

def solve_greedy_maxcut(J, max_steps=1000):
    N = J.shape[0]
    s = np.random.choice([-1, 1], size=N)
    
    for _ in range(max_steps):
        flipped = False
        nodes = np.random.permutation(N)
        for i in nodes:
            delta_W = s[i] * np.dot(J[i], s)
            if delta_W > 0:
                s[i] = -s[i]
                flipped = True
        if not flipped:
            break
            
    energy, cut = compute_cut_and_energy(s, J)
    return s, energy, cut

def solve_sa_maxcut(J, steps=5000, T_init=10.0, T_min=0.001):
    N = J.shape[0]
    s = np.random.choice([-1, 1], size=N)
    best_s = s.copy()
    _, best_cut = compute_cut_and_energy(s, J)
    current_cut = best_cut
    
    alpha = (T_min / T_init) ** (1.0 / steps)
    T = T_init
    
    for step in range(steps):
        i = np.random.randint(N)
        delta_W = s[i] * np.dot(J[i], s)
        
        if delta_W > 0 or np.random.rand() < np.exp(delta_W / T):
            s[i] = -s[i]
            current_cut += delta_W
            if current_cut > best_cut:
                best_cut = current_cut
                best_s = s.copy()
                
        T *= alpha
        
    energy, final_cut = compute_cut_and_energy(best_s, J)
    return best_s, energy, final_cut

def solve_mem4ristor_maxcut(J, steps=3000, frozen_u=False, seed=42):
    N = J.shape[0]
    model = Mem4ristorV3(seed=seed)
    model.cfg['coupling']['heretic_ratio'] = 0.15
    model._initialize_params(N, cold_start=True)
    
    best_cut = 0
    best_s = None
    best_E = 0
    
    for step in range(steps):
        if frozen_u:
            model.u[:] = 0.5
            
        l_v = np.dot(J, model.v)
        model.step(I_stimulus=0.0, coupling_input=l_v)
        
        s = np.sign(model.v)
        s[s == 0] = 1
        
        if step % 10 == 0:
            E, cut = compute_cut_and_energy(s, J)
            if cut > best_cut:
                best_cut = cut
                best_E = E
                best_s = s.copy()
                
    return best_s, best_E, best_cut

def run_maxcut_benchmark(n_seeds=10, N=100, steps=3000):
    print("=" * 70)
    print(f"BENCHMARK P15 AUDITÉ : MAX-CUT & VERRE DE SPINS D'ISING (N={N}, Seeds={n_seeds})")
    print("=" * 70)
    
    results = []
    
    for seed in range(n_seeds):
        np.random.seed(seed)
        
        adj_base = make_ba(N, m=3, seed=seed)
        weights = np.random.choice([-1.0, 1.0], size=adj_base.shape)
        J = adj_base * weights
        J = (J + J.T) / 2.0
        
        t0 = time.time()
        _, E_greedy, cut_greedy = solve_greedy_maxcut(J)
        t_greedy = time.time() - t0
        
        t0 = time.time()
        _, E_sa, cut_sa = solve_sa_maxcut(J, steps=5000)
        t_sa = time.time() - t0
        
        t0 = time.time()
        _, E_frozen, cut_frozen = solve_mem4ristor_maxcut(J, steps=steps, frozen_u=True, seed=seed)
        t_frozen = time.time() - t0
        
        t0 = time.time()
        _, E_m4r, cut_m4r = solve_mem4ristor_maxcut(J, steps=steps, frozen_u=False, seed=seed)
        t_m4r = time.time() - t0
        
        results.append({
            'seed': seed,
            'E_greedy': E_greedy, 'cut_greedy': cut_greedy, 't_greedy': t_greedy,
            'E_sa': E_sa, 'cut_sa': cut_sa, 't_sa': t_sa,
            'E_frozen': E_frozen, 'cut_frozen': cut_frozen, 't_frozen': t_frozen,
            'E_m4r': E_m4r, 'cut_m4r': cut_m4r, 't_m4r': t_m4r
        })
        
        print(f"Seed {seed+1:02d}/{n_seeds:02d} | "
              f"Greedy: {cut_greedy:.0f} | "
              f"SA: {cut_sa:.0f} | "
              f"M4R Frozen: {cut_frozen:.0f} | "
              f"M4R FULL: {cut_m4r:.0f}")
        
    df = pd.DataFrame(results)
    
    os.makedirs("figures", exist_ok=True)
    csv_path = "figures/p15_maxcut_benchmark.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[CSV enregistré] : {csv_path}")
    
    mean_cut_greedy = df['cut_greedy'].mean()
    mean_cut_sa = df['cut_sa'].mean()
    mean_cut_frozen = df['cut_frozen'].mean()
    mean_cut_m4r = df['cut_m4r'].mean()
    
    print("\n" + "=" * 50)
    print("RÉSULTATS CORRECTIFS AUDITÉS DE LA COUPE (MAX-CUT) :")
    print(f"  Recherche Gloutonne  : Coupe Moyenne = {mean_cut_greedy:.2f}")
    print(f"  Recuit Simulé (SA)   : Coupe Moyenne = {mean_cut_sa:.2f}")
    print(f"  Mem4ristor FROZEN    : Coupe Moyenne = {mean_cut_frozen:.2f}")
    print(f"  Mem4ristor FULL      : Coupe Moyenne = {mean_cut_m4r:.2f}")
    print("=" * 50)
    
    plt.figure(figsize=(10, 6))
    methods = ['Glouton', 'Recuit Simulé', 'Mem4ristor (FROZEN)', 'Mem4ristor (FULL)']
    cuts = [mean_cut_greedy, mean_cut_sa, mean_cut_frozen, mean_cut_m4r]
    stds = [df['cut_greedy'].std(), df['cut_sa'].std(), df['cut_frozen'].std(), df['cut_m4r'].std()]
    colors = ['#888888', '#e74c3c', '#f39c12', '#2ecc71']
    
    bars = plt.bar(methods, cuts, yerr=stds, capsize=5, color=colors, alpha=0.85, edgecolor='black')
    plt.ylabel('Valeur de la Coupe (Max-Cut)', fontsize=12)
    plt.title(f'Benchmark Max-Cut Audité sur Verre de Spins (N={N}, {n_seeds} seeds)', fontsize=14, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar, cut in zip(bars, cuts):
        plt.text(bar.get_x() + bar.get_width()/2.0, cut / 2.0, f'{cut:.1f}',
                 ha='center', va='bottom', color='white', fontweight='bold', fontsize=12)
                 
    png_path = "figures/p15_maxcut_faceoff.png"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Graphique enregistré] : {png_path}")

if __name__ == "__main__":
    run_maxcut_benchmark(n_seeds=10, N=100, steps=3000)
