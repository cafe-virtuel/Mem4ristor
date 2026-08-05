import sys
import os
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

N_HEROIC = 1600
N_SEEDS = 20
STEPS = 2500

def _fiedler(adj):
    degree = np.array(adj.sum(axis=1)).flatten()
    L = np.diag(degree) - adj
    return float(np.sort(np.linalg.eigvalsh(L))[1])

def make_continuous_ba(N, m_base, p_add, seed):
    adj = make_ba(N, m_base, seed)
    rng = np.random.RandomState(seed + 1000)
    i_idx, j_idx = np.triu_indices(N, k=1)
    mask = (adj[i_idx, j_idx] == 0) & (rng.rand(len(i_idx)) < p_add)
    adj[i_idx[mask], j_idx[mask]] = 1.0
    adj[j_idx[mask], i_idx[mask]] = 1.0
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
    print("=== ULTIMATE HEROIC RUN N=1600 ===")
    tasks = []
    p_vals = np.linspace(0.001, 0.0035, 8)
    
    for p_add in p_vals:
        for seed in range(N_SEEDS):
            tasks.append((N_HEROIC, 3, p_add, seed))
            
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
        print("Erreur: Aucun résultat.")
        return
        
    df_new = pd.DataFrame(results)
    
    # Charger l'ancien raw data et concaténer
    raw_file = '../figures/v6_binder_cumulant_raw.csv'
    if os.path.exists(raw_file):
        df_old = pd.read_csv(raw_file)
        df_old = df_old[df_old['N'] != N_HEROIC] # remove old N=1600 if any
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
        
    df.to_csv(raw_file, index=False)
    
    print("\nCalcul de U4...")
    df['lambda2_bin'] = np.round(df['lambda2'] * 5) / 5.0
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
        
    df_u4 = pd.DataFrame(u4_results).dropna()
    df_u4.to_csv('../figures/v6_binder_cumulant_U4.csv', index=False)
    print("Données N=1600 ajoutées aux CSV.")

if __name__ == "__main__":
    main()
