"""
Dry-run etendu : Campagne Binder FSS elargie (m>=5, lambda2 dans [1.4, 8+])

But : sonder le signal Binder U4 et la depression H_stable sur les topologies
denses (m=5, m=7) avant de decider si on lance la campagne lourde (option 1)
ou si on reformule la section Binder du preprint (option 2).

Difference avec v6_binder_cumulant_u4.py :
  - m_base est balaye (m=5, m=7) au lieu d'etre hard-code a 3
  - c_vals cible lambda2 dans [1.4, 8+] (c_vals plus dense aux petites valeurs)
  - Steps reduit (600) pour aller vite, on garde N=100,200 pour FSS minimal
  - Sortie : CSV raw + CSV U4 agrege + 1 plot

Ne touche PAS au script d'origine. Script ephemere de sondage.
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from mem4ristor.core import Mem4Network
from mem4ristor.graph_utils import make_ba

LAMBDA2_CRIT = 2.31

# === DRY-RUN : on reduit pour sonder vite ===
N_SIZES = [100, 200]
# Cartographie complete : m=1,2 (topologies peu denses, l2 peut descendre <2)
# + m=5,7 (topologies denses, l2 reste eleve).
M_VALUES = [1, 2, 5, 7]
N_SEEDS = 6
STEPS = 600
# c_vals cible : on veut des lambda2 entre 0.5 et 9+
# Plus de points sous 2.0 pour bien voir si un crossing U4 apparait la-bas.
C_VALS = np.concatenate([
    np.linspace(0.0, 2.0, 9),     # zone sous-critique et critique (dense en points)
    np.linspace(2.5, 9.0, 8),     # zone sur-critique
])


def _fiedler(adj):
    degree = np.array(adj.sum(axis=1)).flatten()
    L = np.diag(degree) - adj
    eigs = np.sort(np.linalg.eigvalsh(L))
    return float(eigs[1]) if len(eigs) > 1 else 0.0


def make_continuous_ba(N, m_base, p_add, seed):
    adj = make_ba(N, m_base, seed)
    rng = np.random.RandomState(seed + 1000)
    for i in range(N):
        for j in range(i + 1, N):
            if adj[i, j] == 0 and rng.rand() < p_add:
                adj[i, j] = adj[j, i] = 1.0
    return adj


def run_single_sim(args):
    N, m_base, c, seed, steps = args
    p_add = c / N
    adj = make_continuous_ba(N, m_base, p_add, seed)
    l2 = _fiedler(adj)

    # Filtre : on veut la plage [1.4, 8+]
    if l2 < 0.5 or l2 > 12.0:
        return {'N': N, 'm': m_base, 'c': c, 'seed': seed, 'lambda2': l2, 'H_stable': np.nan}

    net = Mem4Network(adjacency_matrix=adj, coupling_norm='degree_linear', seed=seed)

    H_list = []
    tail_start = int(steps * 0.75)
    for step in range(steps):
        net.step(I_stimulus=0.5)
        if step >= tail_start:
            H_list.append(net.calculate_entropy(bins=100))

    H_stable = np.mean(H_list) if H_list else np.nan
    return {'N': N, 'm': m_base, 'c': c, 'seed': seed, 'lambda2': l2, 'H_stable': H_stable}


def main():
    print(">>> DRY-RUN ETENDU Binder FSS (m>=5, lambda2 dans [1.4, 8+]) <<<")
    print(f"N_SIZES = {N_SIZES}")
    print(f"M_VALUES = {M_VALUES}")
    print(f"N_SEEDS = {N_SEEDS} par config")
    print(f"STEPS = {STEPS}")
    print(f"C_VALS ({len(C_VALS)} points) = {C_VALS}")
    n_runs = len(N_SIZES) * len(M_VALUES) * len(C_VALS) * N_SEEDS
    print(f"Total runs planifies : {n_runs}")
    print()

    tasks = []
    for N in N_SIZES:
        for m in M_VALUES:
            for c in C_VALS:
                for seed in range(N_SEEDS):
                    tasks.append((N, m, c, seed, STEPS))

    results = []
    max_workers = os.cpu_count() or 4
    print(f"Execution parallele sur {max_workers} workers...")
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_single_sim, task): task for task in tasks}
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Sims"):
            try:
                res = future.result()
                if not np.isnan(res['H_stable']):
                    results.append(res)
            except Exception as e:
                print(f"Erreur : {e}")

    t_elapsed = time.time() - t_start
    print(f"\nTermine en {t_elapsed:.1f} s. {len(results)} runs valides / {len(tasks)}.")

    if not results:
        print("Aucun resultat. Verifier que lambda2 tombe dans la plage.")
        return

    df = pd.DataFrame(results)
    figures_dir = os.path.join(PROJECT_ROOT, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    raw_path = os.path.join(figures_dir, 'v6_binder_dryrun_extended_raw.csv')
    df.to_csv(raw_path, index=False)
    print(f"Raw CSV : {raw_path}")

    # === Distribution de lambda2 par (N, m) : on verifie qu'on couvre bien [1.4, 8+] ===
    print("\n=== Distribution lambda2 par (N, m) ===")
    for N in N_SIZES:
        for m in M_VALUES:
            sub = df[(df['N'] == N) & (df['m'] == m)]
            if len(sub) > 0:
                l2 = sub['lambda2'].values
                print(f"  N={N}, m={m}: l2 min={l2.min():.2f}, max={l2.max():.2f}, "
                      f"median={np.median(l2):.2f}, n={len(l2)}")

    # === Agreagation Binder U4 par bins lambda2 ===
    df['lambda2_bin'] = np.round(df['lambda2'] * 4) / 4.0  # bins de 0.25
    grouped = df.groupby(['N', 'm', 'lambda2_bin'])

    u4_results = []
    for (N, m, l2_bin), group in grouped:
        H_vals = group['H_stable'].values
        if len(H_vals) < 3:
            continue
        m2 = np.mean(H_vals**2)
        m4 = np.mean(H_vals**4)
        if m2 > 1e-9:
            U4 = 1.0 - (m4 / (3.0 * (m2**2)))
        else:
            U4 = np.nan
        u4_results.append({
            'N': N, 'm': m, 'lambda2_bin': l2_bin,
            'lambda2_mean': group['lambda2'].mean(),
            'U4': U4,
            'H_mean': np.mean(H_vals),
            'H_std': np.std(H_vals),
            'H_var': np.var(H_vals),
            'count': len(H_vals)
        })

    if not u4_results:
        print("Pas assez de donnees pour U4.")
        return

    df_u4 = pd.DataFrame(u4_results).dropna()
    u4_path = os.path.join(figures_dir, 'v6_binder_dryrun_extended_U4.csv')
    df_u4.to_csv(u4_path, index=False)
    print(f"U4 CSV : {u4_path}")

    # === Resume des resultats cles ===
    print("\n=== RESUME (pour decision option 1 vs 2) ===")
    for N in N_SIZES:
        for m in M_VALUES:
            sub = df_u4[(df_u4['N'] == N) & (df_u4['m'] == m)].sort_values('lambda2_mean')
            if len(sub) >= 3:
                l2_vals = sub['lambda2_mean'].values
                U4_vals = sub['U4'].values
                H_vals = sub['H_mean'].values
                U4_range = U4_vals.max() - U4_vals.min()
                H_drop = H_vals[0] - H_vals[-1] if len(H_vals) > 1 else 0
                print(f"  N={N}, m={m}: U4 range=[{U4_vals.min():.3f}, {U4_vals.max():.3f}] "
                      f"(amplitude {U4_range:.3f}), H drop={H_drop:.2f} bits, "
                      f"l2 span=[{l2_vals.min():.2f}, {l2_vals.max():.2f}], n_bins={len(sub)}")
                # Crossing detection : U4 qui passe de <2/3 a >2/3 ?
                crossings = np.where(np.diff(np.sign(U4_vals - 2/3)))[0]
                if len(crossings) > 0:
                    cross_l2 = l2_vals[crossings[0]]
                    print(f"    >>> CROSSING U4=2/3 detecte a lambda2 ~ {cross_l2:.2f}")
                else:
                    print(f"    >>> PAS de crossing U4=2/3 (U4 reste plat ou monotone)")

    # === Plot : 2 lignes x 4 colonnes (U4 et H) par m ===
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    colors_N = {100: '#1f77b4', 200: '#ff7f0e'}
    markers_m = {1: 'o', 2: 's', 5: '^', 7: 'D'}

    for i, m in enumerate(M_VALUES):
        # U4
        ax = axes[0, i]
        for N in N_SIZES:
            sub = df_u4[(df_u4['N'] == N) & (df_u4['m'] == m)].sort_values('lambda2_mean')
            if len(sub) > 0:
                ax.plot(sub['lambda2_mean'], sub['U4'],
                        marker=markers_m[m], color=colors_N[N],
                        linewidth=2, markersize=6, label=f'N={N}')
        ax.axvline(LAMBDA2_CRIT, color='red', linestyle='--', linewidth=1.0, label=f'lambda2_crit=2.31')
        ax.axhline(2/3, color='gray', linestyle=':', linewidth=1.0, alpha=0.6, label='U4=2/3')
        ax.set_xlabel('lambda2 (Fiedler)')
        ax.set_ylabel('Binder U4')
        ax.set_title(f'Binder U4 a m={m}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.0, 9.5)
        ax.set_ylim(0.0, 0.75)

        # H
        ax2 = axes[1, i]
        for N in N_SIZES:
            sub = df_u4[(df_u4['N'] == N) & (df_u4['m'] == m)].sort_values('lambda2_mean')
            if len(sub) > 0:
                ax2.errorbar(sub['lambda2_mean'], sub['H_mean'],
                             yerr=sub['H_std']/np.sqrt(sub['count']),
                             marker=markers_m[m], color=colors_N[N],
                             linewidth=2, markersize=6, capsize=3,
                             label=f'N={N}')
        ax2.axvline(LAMBDA2_CRIT, color='red', linestyle='--', linewidth=1.0)
        ax2.set_xlabel('lambda2 (Fiedler)')
        ax2.set_ylabel('<H_stable> (bits)')
        ax2.set_title(f'Entropie stable a m={m}')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0.0, 9.5)

    plt.suptitle('Cartographie Binder FSS (m=1,2,5,7) - I_stim=0.5, degree_linear', fontsize=13)
    plt.tight_layout()
    plot_path = os.path.join(figures_dir, 'v6_binder_dryrun_extended.png')
    plt.savefig(plot_path, dpi=200)
    print(f"\nPlot : {plot_path}")
    print("\n=== FIN DRY-RUN ===")


if __name__ == "__main__":
    main()
