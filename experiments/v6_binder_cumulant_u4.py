"""
Campagne Lourde - Cumulant de Binder U4
Auteur : L'Architecte (Antigravity)
Date : 15 Mai 2026

Objectif : Prouver formellement l'ordre de la transition de phase (Dead Zone)
via le cumulant de Binder U4.
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from mem4ristor.core import Mem4Network
from mem4ristor.graph_utils import make_ba

# --- CONFIGURATION LOURDE ---
N_SIZES = [100, 200, 400]
N_SEEDS = 40  # 40 runs par point
STEPS = 2500
LAMBDA2_CRIT = 2.31

def _fiedler(adj):
    degree = np.array(adj.sum(axis=1)).flatten()
    L = np.diag(degree) - adj
    eigs = np.sort(np.linalg.eigvalsh(L))
    return float(eigs[1])

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
    N, m_base, p_add, seed = args
    adj = make_continuous_ba(N, m_base, p_add, seed)
    l2 = _fiedler(adj)
    
    if l2 < 0.5 or l2 > 5.0:
        return {'N': N, 'p_add': p_add, 'seed': seed, 'lambda2': l2, 'H_stable': np.nan}
        
    net = Mem4Network(adjacency_matrix=adj, coupling_norm='degree_linear', seed=seed)
    
    for _ in range(STEPS):
        net.step(I_stimulus=0.0)
        
    H = net.calculate_entropy(bins=100)
    return {'N': N, 'p_add': p_add, 'seed': seed, 'lambda2': l2, 'H_stable': H}

def main():
    print(f"=== CAMPAGNE LOURDE BINDER CUMULANT U4 ===")
    
    tasks = []
    # Pour balayer lambda2 autour de 2.31, on prend BA m=3 (lambda2 ~1.4) 
    # et on rajoute des arêtes (p_add de 0 à 0.05).
    m_base = 3
    p_vals = np.linspace(0.0, 0.05, 15)
    
    for N in N_SIZES:
        for p_add in p_vals:
            for seed in range(N_SEEDS):
                tasks.append((N, m_base, p_add, seed))
                
    results = []
    max_workers = os.cpu_count() or 4
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_single_sim, task): task for task in tasks}
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Simulating"):
            try:
                res = future.result()
                if not np.isnan(res['H_stable']):
                    results.append(res)
            except Exception as e:
                pass
                
    if not results:
        print("Erreur: Aucun résultat valide généré.")
        return
        
    df = pd.DataFrame(results)
    df.to_csv('../figures/v6_binder_cumulant_raw.csv', index=False)
    
    print("\nCalcul de U4...")
    # On groupe par bins de lambda2 car les graphes aléatoires ont une variance sur lambda2
    # Même avec le même p_add, lambda2 varie.
    df['lambda2_bin'] = np.round(df['lambda2'] * 5) / 5.0  # bins de 0.2
    
    grouped = df.groupby(['N', 'lambda2_bin'])
    
    u4_results = []
    for (N, l2_bin), group in grouped:
        H_vals = group['H_stable'].values
        if len(H_vals) < 5:
            continue
            
        m2 = np.mean(H_vals**2)
        m4 = np.mean(H_vals**4)
        
        if m2 > 1e-9:
            U4 = 1.0 - (m4 / (3.0 * (m2**2)))
        else:
            U4 = np.nan
            
        u4_results.append({
            'N': N, 'lambda2_bin': l2_bin, 'lambda2_mean': group['lambda2'].mean(),
            'U4': U4, 'H_mean': np.mean(H_vals), 'count': len(H_vals)
        })
        
    if not u4_results:
        print("Erreur: Pas assez de données pour U4.")
        return
        
    df_u4 = pd.DataFrame(u4_results).dropna()
    df_u4.to_csv('../figures/v6_binder_cumulant_U4.csv', index=False)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = {100: 'blue', 200: 'orange', 400: 'green'}
    
    ax = axes[0]
    for N in N_SIZES:
        sub = df_u4[df_u4['N'] == N].sort_values('lambda2_mean')
        if len(sub) > 0:
            ax.plot(sub['lambda2_mean'], sub['U4'], marker='o', label=f'N={N}', color=colors.get(N, 'black'))
            
    ax.axvline(LAMBDA2_CRIT, color='red', linestyle='--', label='λ2_crit (2.31)')
    ax.set_xlabel('λ2 (Fiedler value)')
    ax.set_ylabel('Binder Cumulant U4')
    ax.set_title('Binder Cumulant (Indicateur d\'ordre de transition)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    for N in N_SIZES:
        sub = df_u4[df_u4['N'] == N].sort_values('lambda2_mean')
        if len(sub) > 0:
            ax2.plot(sub['lambda2_mean'], sub['H_mean'], marker='o', label=f'N={N}', color=colors.get(N, 'black'))
    ax2.axvline(LAMBDA2_CRIT, color='red', linestyle='--')
    ax2.set_xlabel('λ2 (Fiedler value)')
    ax2.set_ylabel('<H_stable>')
    ax2.set_title('Paramètre d\'ordre vs λ2')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../figures/v6_binder_cumulant.png', dpi=150)
    print("Figure sauvegardée : ../figures/v6_binder_cumulant.png")

if __name__ == "__main__":
    main()
