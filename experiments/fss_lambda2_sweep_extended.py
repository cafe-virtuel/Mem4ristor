"""
FSS Lambda2 Extended Sweep — Mem4ristor Campaign
Auteur: Hermes Agent Session 009+
Date: 2026-05-30

Objectif: Capturer la vraie transition de phase topologique
- BA m=1..10 (incluant dead zone m>=5)
- Lambda2 sweep dans [1.4, 8+]
- D=0.15 (sweet spot AUDIT-012)
- Metriques: H_cont (SECONDAIRE) + synchrony (PRIMAIRE)
- Computes Binder U4 pour N=100/200/400

Bloqueur arXiv: section Binder FSS，要求 m>=5 dans le sweep.
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from mem4ristor.core import Mem4Network
from mem4ristor.graph_utils import make_ba


def _fiedler(adj):
    """Calcule la valeur de Fiedler (valeur propre de Fiedler)."""
    degree = np.array(adj.sum(axis=1)).flatten()
    L = np.diag(degree) - adj
    eigs = np.sort(np.linalg.eigvalsh(L))
    return float(eigs[1]) if len(eigs) > 1 else 0.0


def _compute_synchrony(v_history):
    """Calcule la synchronie (Pearson correlation moyenne par paires)."""
    n_nodes = v_history.shape[1]
    if n_nodes < 2:
        return 0.0
    # Pearson correlation matrix
    v_centered = v_history - v_history.mean(axis=0, keepdims=True)
    v_std = v_history.std(axis=0, keepdims=True)
    v_std[v_std == 0] = 1.0  # avoid div by zero
    corr_matrix = np.dot(v_centered.T, v_centered) / (v_centered.shape[0] - 1)
    std_product = np.dot(v_std.T, v_std)
    corr_matrix = corr_matrix / (std_product + 1e-12)
    # Extract upper triangle (excluding diagonal)
    n = corr_matrix.shape[0]
    upper_tri_indices = np.triu_indices(n, k=1)
    if upper_tri_indices[0].size == 0:
        return 0.0
    return float(np.mean(corr_matrix[upper_tri_indices]))


def run_single_sim(args):
    """
    Simule une realisation unique.
    Retourne: lambda2, H_cont, synchrony, u_mean au steady-state.
    """
    N, m, D, seed, steps = args

    # Build BA graph
    adj = make_ba(N, m, seed)
    lambda2 = _fiedler(adj)

    # Filter: only keep in our lambda2 range of interest
    if lambda2 < 1.0 or lambda2 > 9.0:
        return {
            'N': N, 'm': m, 'D': D, 'seed': seed,
            'lambda2': lambda2, 'H_cont': np.nan,
            'synchrony': np.nan, 'u_mean': np.nan
        }

    # Build network with D coupling
    net = Mem4Network(
        adjacency_matrix=adj,
        coupling_norm='degree_linear',
        seed=seed
    )

    # Override D coupling — set D_eff directly (computed once at __init__)
    # Must also set cfg['coupling']['D'] for code that reads from there
    net.model.cfg['coupling']['D'] = D
    net.model.D_eff = D / float(np.sqrt(net.model.N))

    # Collect steady-state metrics (last 25% of steps)
    H_list = []
    sync_list = []
    v_history = []

    tail_start = int(steps * 0.75)

    for step in range(steps):
        net.step(I_stimulus=0.5)
        if step >= tail_start:
            H_list.append(net.calculate_entropy(bins=100))
            v_history.append(net.model.v.copy())

    if v_history:
        v_arr = np.array(v_history)
        synchrony = _compute_synchrony(v_arr)
    else:
        synchrony = np.nan

    H_cont = np.mean(H_list) if H_list else np.nan
    u_mean = float(net.model.u.mean()) if hasattr(net.model, 'u') else np.nan

    return {
        'N': N, 'm': m, 'D': D, 'seed': seed,
        'lambda2': lambda2,
        'H_cont': H_cont,
        'synchrony': synchrony,
        'u_mean': u_mean
    }


def compute_binder_u4(H_values):
    """Calcule le cumulant de Binder U4."""
    if len(H_values) < 4:
        return np.nan
    H_values = np.array(H_values)
    m2 = np.mean(H_values**2)
    m4 = np.mean(H_values**4)
    if m2 > 1e-9:
        return 1.0 - (m4 / (3.0 * (m2**2)))
    return np.nan


def main():
    parser = argparse.ArgumentParser(description="FSS Lambda2 Extended Sweep")
    parser.add_argument('--dry-run', action='store_true', help="Petit test rapide")
    parser.add_argument('--binder', action='store_true', help="Calcule U4 sur N=100/200/400 (lent)")
    args = parser.parse_args()

    # === CONFIGURATION ===
    if args.dry_run:
        N_SIZES = [100]
        M_VALUES = [3, 5, 7]
        N_SEEDS = 3
        STEPS = 500
        D_VALUES = [0.0, 0.15]
        D_ADAPTIVE = True
        print(">>> DRY RUN MODE <<<")
    elif args.binder:
        N_SIZES = [100, 200, 400]
        M_VALUES = list(range(1, 11))  # m=1..10
        N_SEEDS = 20
        STEPS = 2000
        D_VALUES = [0.0, 0.15]
        D_ADAPTIVE = True
        print(">>> BINDER FSS MODE (N=100/200/400, m=1..10) <<<")
    else:
        N_SIZES = [100]
        M_VALUES = list(range(1, 11))  # m=1..10
        N_SEEDS = 10
        STEPS = 2000
        D_VALUES = [0.0, 0.15]
        D_ADAPTIVE = True
        print(">>> STANDARD FSS MODE (N=100, m=1..10, 10 seeds) <<<")

    figures_dir = os.path.join(PROJECT_ROOT, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    print(f"N sizes: {N_SIZES}")
    print(f"m values (BA): {M_VALUES}")
    print(f"D values: {D_VALUES}")
    print(f"Steps: {STEPS}")
    print(f"Seeds per config: {N_SEEDS}")

    # === BUILD TASKS ===
    tasks = []
    for N in N_SIZES:
        for m in M_VALUES:
            for D in D_VALUES:
                for seed in range(N_SEEDS):
                    tasks.append((N, m, D, seed, STEPS))

    total_tasks = len(tasks)
    print(f"Total simulations: {total_tasks}")

    # === RUN ===
    results = []
    t_start = time.time()

    for task in tqdm(tasks, desc="FSS Sweep (seq)"):
        try:
            res = run_single_sim(task)
            if not np.isnan(res['H_cont']):
                results.append(res)
        except Exception as e:
            print(f"Error: {e}")

    t_elapsed = time.time() - t_start
    print(f"Done in {t_elapsed:.1f}s — {len(results)} valid results")

    if not results:
        print("No valid results!")
        return

    df = pd.DataFrame(results)

    # === SAVE RAW ===
    raw_csv = os.path.join(figures_dir, 'fss_lambda2_sweep_raw.csv')
    df.to_csv(raw_csv, index=False)
    print(f"Raw data: {raw_csv}")

    # === AGGREGATE BY lambda2_bin ===
    df['lambda2_bin'] = np.round(df['lambda2'] * 4) / 4.0  # 0.25 bins

    grouped = df.groupby(['N', 'm', 'D', 'lambda2_bin']).agg(
        lambda2_mean=('lambda2', 'mean'),
        H_cont_mean=('H_cont', 'mean'),
        H_cont_std=('H_cont', 'std'),
        sync_mean=('synchrony', 'mean'),
        sync_std=('synchrony', 'std'),
        u_mean_mean=('u_mean', 'mean'),
        count=('H_cont', 'count')
    ).reset_index()

    agg_csv = os.path.join(figures_dir, 'fss_lambda2_sweep_agg.csv')
    grouped.to_csv(agg_csv, index=False)
    print(f"Aggregated: {agg_csv}")

    # === BINDER U4 (if --binder) ===
    if args.binder:
        print("\nComputing Binder U4...")
        u4_results = []
        for (N, m, D), g in df.groupby(['N', 'm', 'D']):
            for l2_bin, lg in g.groupby('lambda2_bin'):
                H_vals = lg['H_cont'].dropna().values
                if len(H_vals) < 4:
                    continue
                U4 = compute_binder_u4(H_vals)
                u4_results.append({
                    'N': N, 'm': m, 'D': D,
                    'lambda2_bin': l2_bin,
                    'lambda2_mean': lg['lambda2'].mean(),
                    'U4': U4,
                    'H_mean': np.mean(H_vals),
                    'H_std': np.std(H_vals),
                    'count': len(H_vals)
                })

        df_u4 = pd.DataFrame(u4_results).dropna(subset=['U4'])
        u4_csv = os.path.join(figures_dir, 'fss_lambda2_sweep_U4.csv')
        df_u4.to_csv(u4_csv, index=False)
        print(f"U4 data: {u4_csv}")

    # === PLOTS ===
    _plot_results(df, grouped, figures_dir, N_SIZES, M_VALUES, D_VALUES, args.binder, df_u4 if args.binder else None)

    # === PRINT SUMMARY TABLE ===
    print("\n" + "="*70)
    print("SUMMARY: H_cont (SECONDAIRE) and Synchrony (PRIMAIRE)")
    print("="*70)
    print(f"{'m':>3} {'lambda2':>8} {'D':>5} {'H_cont':>8} {'Sync':>8} {'u_mean':>8} {'Phase':>12}")
    print("-"*70)

    for m in sorted(df['m'].unique()):
        for D in sorted(df['D'].unique()):
            sub = df[(df['m'] == m) & (df['D'] == D)]
            if len(sub) == 0:
                continue
            l2 = sub['lambda2'].mean()
            H = sub['H_cont'].mean()
            sync = sub['synchrony'].mean()
            u = sub['u_mean'].mean()

            # Phase classification (on synchrony PRIMARY)
            if sync > 0.3:
                phase = "SYNC (BAD)"
            elif H < 1.0:
                phase = "DEAD_ZONE"
            elif sync < 0.1 and H > 3.0:
                phase = "FUNCTIONAL"
            elif sync < 0.1:
                phase = "TRANSITIONAL"
            else:
                phase = "UNKNOWN"

            print(f"{m:3d} {l2:8.3f} {D:5.2f} {H:8.3f} {sync:8.4f} {u:8.3f} {phase:>12}")

    print("\n>>> METRIC HIERRARCHY: synchrony = PRIMARY, H_cont = SECONDARY <<<")
    print("    sync ~0 = decorrelated (GOAL). sync >0.3 = re-synchronized (BAD).")


def _plot_results(df, grouped, figures_dir, N_SIZES, M_VALUES, D_VALUES, has_binder, df_u4):
    """Genere les figures du sweep."""

    plt.rcParams.update({
        'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
        'grid.alpha': 0.25, 'grid.linestyle': '--'
    })

    # Color map for m values
    cmap = plt.cm.viridis
    m_colors = {m: cmap(i / max(len(M_VALUES) - 1, 1))
                 for i, m in enumerate(sorted(M_VALUES))}

    # === FIGURE 1: H_cont vs lambda2 (colored by m) ===
    fig1, axes1 = plt.subplots(1, 2, figsize=(15, 6))

    for D in D_VALUES:
        ax = axes1[0] if D == D_VALUES[0] else axes1[1]
        d_label = f"D={D}"
        sub = df[df['D'] == D].sort_values('lambda2')

        # Plot each m as a different line
        for m in sorted(sub['m'].unique()):
            m_sub = sub[sub['m'] == m]
            ax.scatter(m_sub['lambda2'], m_sub['H_cont'],
                      color=m_colors[m], alpha=0.7, s=20, label=f"m={m}")

        # Aggregate mean
        g_sub = grouped[grouped['D'] == D].sort_values('lambda2_mean')
        ax.plot(g_sub['lambda2_mean'], g_sub['H_cont_mean'],
               color='black', linewidth=2, linestyle='--', label='Mean')

        ax.set_xlabel('Lambda2 (Fiedler)')
        ax.set_ylabel('H_cont (bits)')
        ax.set_title(f'H_cont vs lambda2 — D={D}\n(SECONDAIRE metric)')
        ax.legend(fontsize=8, loc='upper right', ncol=2)
        ax.grid(True)

    fig1.tight_layout()
    fig1.savefig(os.path.join(figures_dir, 'fss_H_cont_vs_lambda2.png'), dpi=150)
    print(f"Figure: fss_H_cont_vs_lambda2.png")

    # === FIGURE 2: SYNCHRONY vs lambda2 (PRIMARY METRIC) ===
    fig2, axes2 = plt.subplots(1, 2, figsize=(15, 6))

    for i, D in enumerate(D_VALUES):
        ax = axes2[i]
        sub = df[df['D'] == D].sort_values('lambda2')

        for m in sorted(sub['m'].unique()):
            m_sub = sub[sub['m'] == m]
            ax.scatter(m_sub['lambda2'], m_sub['synchrony'],
                      color=m_colors[m], alpha=0.7, s=20, label=f"m={m}")

        # Aggregate
        g_sub = grouped[grouped['D'] == D].sort_values('lambda2_mean')
        ax.plot(g_sub['lambda2_mean'], g_sub['sync_mean'],
               color='black', linewidth=2, linestyle='--', label='Mean')

        ax.axhline(0.0, color='gray', linestyle=':', alpha=0.5)
        ax.axhline(0.3, color='red', linestyle='--', alpha=0.5, label='Sync=0.3 (BAD)')

        ax.set_xlabel('Lambda2 (Fiedler)')
        ax.set_ylabel('Synchrony (Pearson r)')
        ax.set_title(f'Synchrony vs lambda2 — D={D}\n(PRIMARY metric: ~0 = decorrelated)')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True)
        ax.set_ylim(-0.2, 0.8)

    fig2.tight_layout()
    fig2.savefig(os.path.join(figures_dir, 'fss_sync_vs_lambda2.png'), dpi=150)
    print(f"Figure: fss_sync_vs_lambda2.png")

    # === FIGURE 3: 2D heatmap H_cont + Sync ===
    fig3, axes3 = plt.subplots(1, len(D_VALUES), figsize=(15, 5))

    for i, D in enumerate(D_VALUES):
        ax = axes3[i] if len(D_VALUES) > 1 else axes3
        sub = grouped[grouped['D'] == D]

        # pivot table for heatmap
        pivot = sub.pivot_table(
            values='H_cont_mean',
            index='m',
            columns='lambda2_bin',
            aggfunc='mean'
        )

        im = ax.imshow(pivot.values, aspect='auto', origin='lower',
                      cmap='viridis', vmin=0, vmax=5)
        ax.set_xlabel('lambda2 bin')
        ax.set_ylabel('m (BA)')
        ax.set_title(f'H_cont heatmap D={D}')
        ax.set_xticks(range(0, len(pivot.columns), max(1, len(pivot.columns)//8)))
        ax.set_xticklabels([f"{p:.1f}" for p in pivot.columns[::max(1, len(pivot.columns)//8)]], fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        plt.colorbar(im, ax=ax, label='H_cont')

    fig3.tight_layout()
    fig3.savefig(os.path.join(figures_dir, 'fss_heatmap_H_cont.png'), dpi=150)
    print(f"Figure: fss_heatmap_H_cont.png")

    # === FIGURE 4: Sync heatmap (PRIMARY) ===
    fig4, axes4 = plt.subplots(1, len(D_VALUES), figsize=(15, 5))

    for i, D in enumerate(D_VALUES):
        ax = axes4[i] if len(D_VALUES) > 1 else axes4
        sub = grouped[grouped['D'] == D]

        pivot = sub.pivot_table(
            values='sync_mean',
            index='m',
            columns='lambda2_bin',
            aggfunc='mean'
        )

        im = ax.imshow(pivot.values, aspect='auto', origin='lower',
                      cmap='RdBu_r', vmin=-0.2, vmax=0.6)
        ax.set_xlabel('lambda2 bin')
        ax.set_ylabel('m (BA)')
        ax.set_title(f'Sync heatmap D={D}\n(PRIMARY: ~0 = good)')
        ax.set_xticks(range(0, len(pivot.columns), max(1, len(pivot.columns)//8)))
        ax.set_xticklabels([f"{p:.1f}" for p in pivot.columns[::max(1, len(pivot.columns)//8)]], fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        plt.colorbar(im, ax=ax, label='Sync')

    fig4.tight_layout()
    fig4.savefig(os.path.join(figures_dir, 'fss_heatmap_sync.png'), dpi=150)
    print(f"Figure: fss_heatmap_sync.png")

    # === FIGURE 5: Binder U4 (if computed) ===
    if has_binder and df_u4 is not None and len(df_u4) > 0:
        fig5, axes5 = plt.subplots(1, 2, figsize=(15, 6))

        colors_N = {100: '#1f77b4', 200: '#ff7f0e', 400: '#2ca02c'}
        markers_N = {100: 'o', 200: 's', 400: '^'}

        for D in D_VALUES:
            ax = axes5[0] if D == D_VALUES[0] else axes5[1]
            d_sub = df_u4[df_u4['D'] == D]

            for N in N_SIZES:
                n_sub = d_sub[d_sub['N'] == N].sort_values('lambda2_mean')
                if len(n_sub) > 0:
                    ax.plot(n_sub['lambda2_mean'], n_sub['U4'],
                           marker=markers_N.get(N, 'o'), color=colors_N.get(N, '#333'),
                           linewidth=2, markersize=6, label=f'N={N} D={D}')

            ax.axhline(0.0, color='gray', linestyle=':', alpha=0.5)
            ax.set_xlabel('Lambda2')
            ax.set_ylabel('Binder U4')
            ax.set_title(f'Binder U4 vs lambda2 — D={D}')
            ax.legend(fontsize=8)
            ax.grid(True)

        fig5.tight_layout()
        fig5.savefig(os.path.join(figures_dir, 'fss_binder_U4.png'), dpi=150)
        print(f"Figure: fss_binder_U4.png")

    print("All figures saved.")


if __name__ == "__main__":
    main()
