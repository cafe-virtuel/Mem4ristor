"""
Campagne Lourde - Cumulant de Binder U4
Auteur : L'Architecte (Antigravity)
Date : 19 Mai 2026

Objectif : Prouver formellement l'ordre de la transition de phase (Dead Zone)
via le cumulant de Binder U4 dans le régime d'hérétiques actifs.
"""

import sys
import os
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Résolution robuste du chemin des modules src/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from mem4ristor.core import Mem4Network
from mem4ristor.graph_utils import make_ba

# --- CONFIGURATION PHYSIQUE DU PREPRINT ---
LAMBDA2_CRIT = 2.31

# Paramètres par défaut de la campagne lourde
DEFAULT_N_SIZES = [100, 200, 400]
DEFAULT_N_SEEDS = 40
DEFAULT_STEPS = 2500
DEFAULT_C_MAX = 15.0
DEFAULT_C_STEPS = 15

def _fiedler(adj):
    """Calcule la valeur de Fiedler (valeur propre de Fiedler) de la matrice d'adjacence."""
    degree = np.array(adj.sum(axis=1)).flatten()
    L = np.diag(degree) - adj
    eigs = np.sort(np.linalg.eigvalsh(L))
    return float(eigs[1]) if len(eigs) > 1 else 0.0

def make_continuous_ba(N, m_base, p_add, seed):
    """Génère un graphe BA(m_base) et ajoute des arêtes aléatoires (p_add)."""
    adj = make_ba(N, m_base, seed)
    rng = np.random.RandomState(seed + 1000)
    for i in range(N):
        for j in range(i + 1, N):
            if adj[i, j] == 0 and rng.rand() < p_add:
                adj[i, j] = adj[j, i] = 1.0
    return adj

def run_single_sim(args):
    """
    Simule une seule réalisation d'un réseau de taille N, de connectivité c, et de graine seed.
    """
    N, m_base, c, seed, steps = args
    p_add = c / N
    adj = make_continuous_ba(N, m_base, p_add, seed)
    l2 = _fiedler(adj)
    
    # Filtrage large pour capturer l'ensemble de la transition de phase
    if l2 < 0.1 or l2 > 15.0:
        return {'N': N, 'c': c, 'seed': seed, 'lambda2': l2, 'H_stable': np.nan}
        
    net = Mem4Network(adjacency_matrix=adj, coupling_norm='degree_linear', seed=seed)
    
    # Enregistrement de l'entropie spatiale continue sur les derniers 25% des pas de simulation
    H_list = []
    tail_start = int(steps * 0.75)
    for step in range(steps):
        net.step(I_stimulus=0.5)  # Régime d'hérétiques actifs
        if step >= tail_start:
            H_list.append(net.calculate_entropy(bins=100))
            
    H_stable = np.mean(H_list) if H_list else np.nan
    return {'N': N, 'c': c, 'seed': seed, 'lambda2': l2, 'H_stable': H_stable}

def main():
    parser = argparse.ArgumentParser(description="Campagne lourde pour le Cumulant de Binder U4")
    parser.add_argument('--dry-run', action='store_true', help="Lancer un test rapide à petite échelle")
    args = parser.parse_args()
    
    if args.dry_run:
        print(">>> MODE DRY-RUN ACTIF (test rapide) <<<")
        N_SIZES = [100]
        N_SEEDS = 3
        STEPS = 500
        c_vals = np.linspace(0.0, 8.0, 5)
    else:
        print(">>> MODE CAMPAGNE COMPLÈTE ACTIF <<<")
        N_SIZES = DEFAULT_N_SIZES
        N_SEEDS = DEFAULT_N_SEEDS
        STEPS = DEFAULT_STEPS
        c_vals = np.linspace(0.0, DEFAULT_C_MAX, DEFAULT_C_STEPS)
        
    m_base = 3
    
    print(f"Tailles de réseaux (N) : {N_SIZES}")
    print(f"Graines par configuration : {N_SEEDS}")
    print(f"Pas de simulation par run : {STEPS}")
    print(f"Balayage du paramètre c : {c_vals}")
    print(f"Nombre total de simulations planifiées : {len(N_SIZES) * len(c_vals) * N_SEEDS}")
    
    tasks = []
    for N in N_SIZES:
        for c in c_vals:
            for seed in range(N_SEEDS):
                tasks.append((N, m_base, c, seed, STEPS))
                
    results = []
    max_workers = os.cpu_count() or 4
    print(f"Exécution parallèle sur {max_workers} processus...")
    
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_single_sim, task): task for task in tasks}
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Simulations en cours"):
            try:
                res = future.result()
                if not np.isnan(res['H_stable']):
                    results.append(res)
            except Exception as e:
                print(f"Erreur sur une tâche : {e}")
                
    t_elapsed = time.time() - t_start
    print(f"Simulations terminées en {t_elapsed:.1f} secondes.")
    
    if not results:
        print("Erreur: Aucun résultat valide généré.")
        return
        
    df = pd.DataFrame(results)
    
    # S'assurer que le dossier figures existe
    figures_dir = os.path.join(PROJECT_ROOT, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    raw_csv_path = os.path.join(figures_dir, 'v6_binder_cumulant_raw.csv')
    df.to_csv(raw_csv_path, index=False)
    print(f"Données brutes sauvegardées dans : {raw_csv_path}")
    
    print("\nCalcul du Cumulant de Binder U4...")
    # Groupement par bins de connectivité lambda2 pour agréger les statistiques d'ensemble
    df['lambda2_bin'] = np.round(df['lambda2'] * 4) / 4.0  # Bins de taille 0.25 pour l'analyse statistique
    
    grouped = df.groupby(['N', 'lambda2_bin'])
    
    u4_results = []
    min_group_size = 2 if args.dry_run else 4
    for (N, l2_bin), group in grouped:
        H_vals = group['H_stable'].values
        if len(H_vals) < min_group_size:
            continue
            
        m2 = np.mean(H_vals**2)
        m4 = np.mean(H_vals**4)
        
        if m2 > 1e-9:
            U4 = 1.0 - (m4 / (3.0 * (m2**2)))
        else:
            U4 = np.nan
            
        u4_results.append({
            'N': N,
            'lambda2_bin': l2_bin,
            'lambda2_mean': group['lambda2'].mean(),
            'U4': U4,
            'H_mean': np.mean(H_vals),
            'H_std': np.std(H_vals),
            'count': len(H_vals)
        })
        
    if not u4_results:
        print("Erreur: Pas assez de données binned pour calculer U4.")
        return
        
    df_u4 = pd.DataFrame(u4_results).dropna()
    u4_csv_path = os.path.join(figures_dir, 'v6_binder_cumulant_U4.csv')
    df_u4.to_csv(u4_csv_path, index=False)
    print(f"Cumulants de Binder U4 sauvegardés dans : {u4_csv_path}")
    
    # --- RENDER DES GRAPHIQUES MODERNES ---
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.titlesize': 14,
        'grid.alpha': 0.25,
        'grid.linestyle': '--'
    })
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Palette de couleurs haut de gamme (bleu royal, ambre chaud, vert émeraude)
    colors = {100: '#1f77b4', 200: '#ff7f0e', 400: '#2ca02c'}
    markers = {100: 'o', 200: 's', 400: '^'}
    
    # Panel 1: U4 vs lambda2
    ax = axes[0]
    for N in N_SIZES:
        sub = df_u4[df_u4['N'] == N].sort_values('lambda2_mean')
        if len(sub) > 0:
            ax.plot(sub['lambda2_mean'], sub['U4'], marker=markers.get(N, 'o'), 
                    color=colors.get(N, '#333333'), linewidth=2, markersize=6,
                    label=f'N = {N}')
            
    ax.axvline(LAMBDA2_CRIT, color='#d62728', linestyle='--', linewidth=1.5, label=f'λ2_crit = {LAMBDA2_CRIT}')
    ax.set_xlabel('Connectivité algébrique λ2 (Valeur de Fiedler)')
    ax.set_ylabel('Cumulant de Binder U4')
    ax.set_title('Cumulant de Binder U4 vs λ2\n(Indicateur de l\'ordre de transition)')
    ax.legend(frameon=True, facecolor='white', edgecolor='none')
    ax.grid(True)
    ax.set_xlim(1.0, 7.5)
    ax.set_ylim(0.0, 0.7)
    
    # Panel 2: <H_stable> vs lambda2
    ax2 = axes[1]
    for N in N_SIZES:
        sub = df_u4[df_u4['N'] == N].sort_values('lambda2_mean')
        if len(sub) > 0:
            ax2.errorbar(sub['lambda2_mean'], sub['H_mean'], yerr=sub['H_std']/np.sqrt(sub['count']),
                         marker=markers.get(N, 'o'), color=colors.get(N, '#333333'), 
                         linewidth=2, markersize=6, capsize=3, elinewidth=1,
                         label=f'N = {N}')
            
    ax2.axvline(LAMBDA2_CRIT, color='#d62728', linestyle='--', linewidth=1.5)
    ax2.set_xlabel('Connectivité algébrique λ2 (Valeur de Fiedler)')
    ax2.set_ylabel('Entropie stable moyenne <H_stable> (bits)')
    ax2.set_title('Paramètre d\'ordre d\'entropie vs λ2\n(Effondrement vers la Dead Zone)')
    ax2.legend(frameon=True, facecolor='white', edgecolor='none')
    ax2.grid(True)
    ax2.set_xlim(1.0, 7.5)
    ax2.set_ylim(-0.1, 4.5)
    
    # Inset de la dérivée pour le plus grand N simulé (pour valider la divergence)
    largest_N = max(N_SIZES) if N_SIZES else None
    if largest_N is not None:
        sub_large = df_u4[df_u4['N'] == largest_N].sort_values('lambda2_mean')
        if len(sub_large) > 2:
            l2_vals = sub_large['lambda2_mean'].values
            H_vals = sub_large['H_mean'].values
            
            d_l2 = np.diff(l2_vals)
            d_H = np.diff(H_vals)
            
            # Éviter les divisions par zéro
            valid_diff = d_l2 > 1e-5
            l2_mid = l2_vals[:-1][valid_diff] + d_l2[valid_diff] / 2.0
            deriv = np.abs(d_H[valid_diff] / d_l2[valid_diff])
            
            # Ajouter l'inset au Panel 2
            inset_ax = ax2.inset_axes([0.5, 0.45, 0.45, 0.45])
            inset_ax.plot(l2_mid, deriv, color=colors.get(largest_N, '#333333'), linewidth=1.5, marker='.')
            inset_ax.axvline(LAMBDA2_CRIT, color='#d62728', linestyle='--', linewidth=1.0)
            inset_ax.set_title(f'|d<H>/dλ2| pour N={largest_N}', fontsize=9)
            inset_ax.grid(True, alpha=0.15)
            inset_ax.tick_params(labelsize=8)
            inset_ax.set_xlim(1.0, 7.5)
            
    plt.tight_layout()
    plot_path = os.path.join(figures_dir, 'v6_binder_cumulant.png')
    plt.savefig(plot_path, dpi=300)
    print(f"Graphique final haute résolution sauvegardé dans : {plot_path}")
    print("Campagne de simulation Binder Cumulant U4 terminée avec succès !")

if __name__ == "__main__":
    main()
