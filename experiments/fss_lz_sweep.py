"""
FSS Lambda2 + LZ Complexity Sweep — Mem4ristor Campaign
Auteur: Hermes Agent Session 010
Date: 2026-05-31

Objectif: Desambiguer sync=0 (degenerative) via LZ76 complexity.
  3 regimes a distinguer:
    sync ~0 + LZ moderee (0.2-0.8)  -> vraie diversite cognitive (functional)
    sync ~0 + LZ ~0                   -> noeuds geles (dead zone deguisee)
    sync ~0 + LZ ~1                   -> bruit chaotique pur

Metriques:
  - synchrony (PRIMAIRE, Pearson r)
  - LZ76 complexity (SECONDAIRE, desambiguise sync=0)
  - H_cont (TERTIAIRE, illustration)

Protocols: BA m=1..10, 10 seeds, N=100
  D=0          : no coupling (baseline)
  D=0.15       : static coupling (fixed)
  D=0.50*u     : adaptive coupling (D_eff = 0.50 * u_mean / sqrt(N))

Question: est-ce que le regime FUNCTIONAL (LZ<0.85) apparait a m<7
avec D=0.50*u? Si oui, le seuil depend du protocole.
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from mem4ristor.core import Mem4Network
from mem4ristor.graph_utils import make_ba
from mem4ristor.metrics import calculate_temporal_lz_complexity


def _fiedler(adj):
    """Valeur propre de Fiedler (lambda2 du Laplacien)."""
    degree = np.array(adj.sum(axis=1)).flatten()
    L = np.diag(degree) - adj
    eigs = np.sort(np.linalg.eigvalsh(L))
    return float(eigs[1]) if len(eigs) > 1 else 0.0


def _compute_synchrony(v_history):
    """Pearson correlation moyenne par paires (version compacte)."""
    n_nodes = v_history.shape[1]
    if n_nodes < 2:
        return 0.0
    v_centered = v_history - v_history.mean(axis=0, keepdims=True)
    v_std = v_history.std(axis=0, keepdims=True)
    v_std[v_std == 0] = 1.0
    corr_matrix = np.dot(v_centered.T, v_centered) / (v_centered.shape[0] - 1)
    std_product = np.dot(v_std.T, v_std)
    corr_matrix = corr_matrix / (std_product + 1e-12)
    n = corr_matrix.shape[0]
    upper_tri_indices = np.triu_indices(n, k=1)
    if upper_tri_indices[0].size == 0:
        return 0.0
    return float(np.mean(corr_matrix[upper_tri_indices]))


def run_single_sim(args):
    """
    Simule une realisation unique.
    Retourne: lambda2, H_cont, synchrony, LZ_complexity, u_mean.

    Protocol D values:
      0.0       -> D_eff = 0 (no coupling)
      0.15      -> D_eff = 0.15 / sqrt(N) (static coupling)
      0.50      -> D_eff updated each step = 0.50 * u_mean / sqrt(N) (adaptive)
    """
    N, m, D, seed, steps = args

    # Build BA graph
    adj = make_ba(N, m, seed)
    lambda2 = _fiedler(adj)

    # Filtre: hors de la plage lambda2 on saute
    if lambda2 < 1.0 or lambda2 > 9.0:
        return {
            'N': N, 'm': m, 'D': D, 'seed': seed,
            'lambda2': lambda2, 'H_cont': np.nan,
            'synchrony': np.nan, 'lz_full': np.nan,
            'u_mean': np.nan
        }

    # Build network
    net = Mem4Network(
        adjacency_matrix=adj,
        coupling_norm='degree_linear',
        seed=seed
    )

    # Protocol: D=0.50 -> mark as adaptive, handled inside step loop
    is_adaptive = bool(D == 0.50)
    net.model.cfg['coupling']['D'] = float(D)
    if not is_adaptive:
        net.model.D_eff = D / float(np.sqrt(net.model.N))

    # Collect steady-state metrics (last 25% of steps)
    H_list = []
    v_history = []
    tail_start = int(steps * 0.75)

    for step in range(steps):
        net.step(I_stimulus=0.5)
        # Adaptive D: update D_eff = 0.50 * u_mean / sqrt(N) each step
        if is_adaptive:
            u_mean = float(net.model.u.mean())
            net.model.D_eff = (0.50 * u_mean) / float(np.sqrt(net.model.N))
        if step >= tail_start:
            H_list.append(net.calculate_entropy(bins=100))
            v_history.append(net.model.v.copy())

    v_arr = np.array(v_history)           # shape (T_tail, N)
    synchrony = _compute_synchrony(v_arr)
    lz_complexity = calculate_temporal_lz_complexity(v_arr, n_bins=5)
    H_cont = np.mean(H_list) if H_list else np.nan
    u_mean = float(net.model.u.mean()) if hasattr(net.model, 'u') else np.nan

    return {
        'N': N, 'm': m, 'D': D, 'seed': seed,
        'lambda2': lambda2,
        'H_cont': H_cont,
        'synchrony': synchrony,
        'lz_full': lz_complexity,
        'u_mean': u_mean
    }


def classify_regime(row):
    """
    Classifie le regime 2D (sync, LZ) selon la grille:
      sync > 0.3            -> SYNCED (no goal)
      sync <= 0.3 and LZ < 0.15  -> FROZEN (dead zone deguisee)
      sync <= 0.3 and 0.15 <= LZ < 0.85 -> FUNCTIONAL (vraie diversite)
      sync <= 0.3 and LZ >= 0.85 -> CHAOTIC (bruit pur)
    """
    sync = row.get('synchrony', np.nan)
    lz = row.get('lz_full', np.nan)
    if np.isnan(sync) or np.isnan(lz):
        return 'UNKNOWN'
    if sync > 0.3:
        return 'SYNCED'
    if lz < 0.15:
        return 'FROZEN'
    if lz < 0.85:
        return 'FUNCTIONAL'
    return 'CHAOTIC'


def main():
    parser = argparse.ArgumentParser(description="FSS Lambda2 + LZ Complexity Sweep")
    parser.add_argument('--dry-run', action='store_true', help="Petit test rapide")
    parser.add_argument('--binder', action='store_true', help="Calcule U4 sur N=100/200/400 (lent)")
    parser.add_argument('--nseeds', type=int, default=10, help="Nombre de seeds (default 10)")
    args = parser.parse_args()

    # === CONFIGURATION ===
    if args.dry_run:
        N_SIZES = [100]
        M_VALUES = [3, 5, 7]
        N_SEEDS = 3
        STEPS = 500
        D_VALUES = [0.0, 0.15, 0.50]
        print(">>> DRY RUN MODE <<<")
    elif args.binder:
        N_SIZES = [100, 200, 400]
        M_VALUES = list(range(1, 11))  # m=1..10
        N_SEEDS = 20
        STEPS = 2000
        D_VALUES = [0.0, 0.15, 0.50]
        print(">>> BINDER FSS MODE (N=100/200/400, m=1..10) <<<")
    else:
        N_SIZES = [100]
        M_VALUES = list(range(1, 11))   # m=1..10
        N_SEEDS = args.nseeds
        STEPS = 2000
        D_VALUES = [0.0, 0.15, 0.50]
        print(f">>> STANDARD MODE (N=100, m=1..10, {N_SEEDS} seeds, 3 protocols) <<<")

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

    for task in tqdm(tasks, desc="FSS+LZ Sweep"):
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
    raw_csv = os.path.join(figures_dir, 'fss_lz_sweep_raw.csv')
    df.to_csv(raw_csv, index=False)
    print(f"Raw data: {raw_csv}")

    # === CLASSIFY REGIMES ===
    df['regime'] = df.apply(classify_regime, axis=1)

    # === AGGREGATE BY lambda2_bin ===
    df['lambda2_bin'] = np.round(df['lambda2'] * 4) / 4.0

    grouped = df.groupby(['N', 'm', 'D', 'lambda2_bin']).agg(
        lambda2_mean=('lambda2', 'mean'),
        H_cont_mean=('H_cont', 'mean'),
        H_cont_std=('H_cont', 'std'),
        sync_mean=('synchrony', 'mean'),
        sync_std=('synchrony', 'std'),
        lz_mean=('lz_full', 'mean'),
        lz_std=('lz_full', 'std'),
        u_mean_mean=('u_mean', 'mean'),
        count=('H_cont', 'count')
    ).reset_index()

    agg_csv = os.path.join(figures_dir, 'fss_lz_sweep_agg.csv')
    grouped.to_csv(agg_csv, index=False)
    print(f"Aggregated: {agg_csv}")

    # === REGIME COUNTS ===
    print("\n" + "=" * 60)
    print("REGIME DISTRIBUTION (sync vs LZ 2D)")
    print("=" * 60)
    regime_counts = df['regime'].value_counts()
    for r, cnt in regime_counts.items():
        pct = 100 * cnt / len(df)
        print(f"  {r:12s}: {cnt:4d} ({pct:5.1f}%)")

    # === SUMMARY TABLE ===
    print("\n" + "=" * 75)
    print("SUMMARY: synchrony (PRIMAIRE) x LZ (SECONDAIRE) x lambda2")
    print("=" * 75)
    print(f"{'m':>3} {'l2':>7} {'D':>5} {'Sync':>7} {'LZ':>7} {'H_cont':>7} {'Regime':>12}")
    print("-" * 75)

    for m in sorted(df['m'].unique()):
        for D in sorted(df['D'].unique()):
            sub = df[(df['m'] == m) & (df['D'] == D)]
            if len(sub) == 0:
                continue
            l2 = sub['lambda2'].mean()
            sync = sub['synchrony'].mean()
            lz = sub['lz_full'].mean()
            H = sub['H_cont'].mean()
            regime = sub['regime'].mode()[0] if len(sub) > 0 else 'UNKNOWN'
            print(f"{m:3d} {l2:7.3f} {D:5.2f} {sync:7.4f} {lz:7.4f} {H:7.3f} {regime:>12}")

    print("\n>>> 3-REGIME KEY <<<")
    print("  FUNCTIONAL : sync~0 + LZ 0.15-0.85  -> vraie diversite cognitive")
    print("  FROZEN     : sync~0 + LZ<0.15        -> noeuds geles (dead zone)")
    print("  CHAOTIC    : sync~0 + LZ>=0.85       -> bruit chaotique")
    print("  SYNCED     : sync>0.3                 -> consensus (no goal)")

    # === PLOTS ===
    _plot_results(df, grouped, figures_dir, D_VALUES)

    print("\nAll figures saved.")


def _plot_results(df, grouped, figures_dir, D_VALUES):
    """Genere les figures du sweep."""

    plt.rcParams.update({
        'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
        'grid.alpha': 0.25, 'grid.linestyle': '--'
    })

    # Colormap for lambda2
    l2_min, l2_max = df['lambda2'].min(), df['lambda2'].max()
    norm = plt.Normalize(vmin=l2_min, vmax=l2_max)
    cmap_l2 = plt.cm.plasma

    regime_colors = {
        'FUNCTIONAL': '#2ca02c',   # green
        'FROZEN':     '#1f77b4',   # blue
        'CHAOTIC':    '#d62728',   # red
        'SYNCED':     '#ff7f0e',   # orange
        'UNKNOWN':    '#7f7f7f',   # gray
    }
    regime_markers = {
        'FUNCTIONAL': 'o',
        'FROZEN':     's',
        'CHAOTIC':    '^',
        'SYNCED':     'D',
        'UNKNOWN':    'x',
    }

    # =====================================================================
    # FIGURE 1: 2D Scatter — synchrony vs LZ (COLOR = lambda2)
    # =====================================================================
    fig1, axes1 = plt.subplots(1, len(D_VALUES), figsize=(14, 6),
                                sharex=False, sharey=False)
    if len(D_VALUES) == 1:
        axes1 = [axes1]

    for i, D in enumerate(D_VALUES):
        ax = axes1[i]
        sub = df[df['D'] == D]

        # Plot each regime group with distinct markers
        for regime in ['FUNCTIONAL', 'FROZEN', 'CHAOTIC', 'SYNCED']:
            r_sub = sub[sub['regime'] == regime]
            if len(r_sub) == 0:
                continue
            ax.scatter(r_sub['synchrony'], r_sub['lz_full'],
                      color=regime_colors[regime],
                      marker=regime_markers[regime],
                      alpha=0.7, s=40, label=regime,
                      edgecolors='white', linewidths=0.3)

        # Threshold lines
        ax.axvline(0.3, color='gray', linestyle='--', alpha=0.6, linewidth=1)
        ax.axhline(0.15, color='gray', linestyle=':', alpha=0.6, linewidth=1)
        ax.axhline(0.85, color='gray', linestyle=':', alpha=0.6, linewidth=1)

        # Zone labels
        ax.text(0.05, 0.50, 'FROZEN\n(LZ<0.15)', fontsize=8,
                color='#1f77b4', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.6))
        ax.text(0.05, 0.55, 'FUNCTIONAL\n(0.15<LZ<0.85)', fontsize=8,
                color='#2ca02c', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.6))
        ax.text(0.05, 0.93, 'CHAOTIC\n(LZ>=0.85)', fontsize=8,
                color='#d62728', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.6))
        ax.text(0.50, 0.05, 'SYNCED\n(sync>0.3)', fontsize=8,
                color='#ff7f0e', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.6))

        ax.set_xlabel('Synchrony (Pearson r)\nPRIMARY metric')
        ax.set_ylabel('LZ76 Complexity\nSECONDARY metric')
        ax.set_title(f'D={D}  —  Sync vs LZ 2D Regime Map')
        ax.set_xlim(-0.25, 0.75)
        ax.set_ylim(-0.05, 1.08)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

    fig1.tight_layout()
    fig1.savefig(os.path.join(figures_dir, 'fss_lz_2d_scatter.png'), dpi=150)
    print(f"Figure: fss_lz_2d_scatter.png")

    # =====================================================================
    # FIGURE 2: 2D Scatter — synchrony vs LZ (COLOR = m / lambda2)
    #    Cette fois colorier par lambda2 pour voir la transition topologique
    # =====================================================================
    fig2, axes2 = plt.subplots(1, len(D_VALUES), figsize=(14, 6),
                                sharex=False, sharey=False)
    if len(D_VALUES) == 1:
        axes2 = [axes2]

    for i, D in enumerate(D_VALUES):
        ax = axes2[i]
        sub = df[df['D'] == D].sort_values('lambda2')

        sc = ax.scatter(sub['synchrony'], sub['lz_full'],
                       c=sub['lambda2'], cmap=cmap_l2,
                       alpha=0.75, s=45, edgecolors='white', linewidths=0.3)

        ax.axvline(0.3, color='gray', linestyle='--', alpha=0.6)
        ax.axhline(0.15, color='gray', linestyle=':', alpha=0.6)
        ax.axhline(0.85, color='gray', linestyle=':', alpha=0.6)

        ax.set_xlabel('Synchrony (Pearson r)')
        ax.set_ylabel('LZ76 Complexity')
        ax.set_title(f'D={D}  —  Sync vs LZ (color = lambda2)')
        ax.set_xlim(-0.25, 0.75)
        ax.set_ylim(-0.05, 1.08)
        ax.grid(True, alpha=0.3)
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Lambda2 (Fiedler)', fontsize=9)

    fig2.tight_layout()
    fig2.savefig(os.path.join(figures_dir, 'fss_lz_2d_lambda2.png'), dpi=150)
    print(f"Figure: fss_lz_2d_lambda2.png")

    # =====================================================================
    # FIGURE 3: LZ76 vs lambda2 (heatmap-style per m)
    # =====================================================================
    fig3, axes3 = plt.subplots(1, len(D_VALUES), figsize=(14, 5),
                                sharey=True)
    if len(D_VALUES) == 1:
        axes3 = [axes3]

    for i, D in enumerate(D_VALUES):
        ax = axes3[i]
        sub = grouped[grouped['D'] == D].sort_values('lambda2_mean')

        # Plot LZ vs lambda2 colored by m
        m_vals = sorted(sub['m'].unique())
        cmap_m = plt.cm.viridis
        for mi, m in enumerate(m_vals):
            m_sub = sub[sub['m'] == m]
            ax.plot(m_sub['lambda2_mean'], m_sub['lz_mean'],
                   marker='o', markersize=5,
                   color=cmap_m(mi / max(len(m_vals) - 1, 1)),
                   label=f'm={m}', alpha=0.8)

        ax.axhline(0.15, color='blue', linestyle='--', alpha=0.5, label='FROZEN threshold')
        ax.axhline(0.85, color='red', linestyle='--', alpha=0.5, label='CHAOTIC threshold')

        ax.set_xlabel('Lambda2 (Fiedler)')
        ax.set_ylabel('LZ76 Complexity')
        ax.set_title(f'D={D}  —  LZ vs lambda2 (per m)')
        ax.set_ylim(-0.05, 1.08)
        ax.legend(fontsize=7, ncol=2, loc='upper right')
        ax.grid(True, alpha=0.3)

    fig3.tight_layout()
    fig3.savefig(os.path.join(figures_dir, 'fss_lz_vs_lambda2.png'), dpi=150)
    print(f"Figure: fss_lz_vs_lambda2.png")

    # =====================================================================
    # FIGURE 4: 3-panel regime heatmap (m vs lambda2) for each D
    # =====================================================================
    fig4, axes4 = plt.subplots(1, len(D_VALUES), figsize=(14, 5))
    if len(D_VALUES) == 1:
        axes4 = [axes4]

    regime_to_num = {'FROZEN': 0, 'FUNCTIONAL': 1, 'CHAOTIC': 2, 'SYNCED': 3, 'UNKNOWN': -1}

    for i, D in enumerate(D_VALUES):
        ax = axes4[i]
        sub = grouped[grouped['D'] == D].copy()
        sub['regime_num'] = sub.apply(
            lambda r: regime_to_num.get(
                classify_regime({'synchrony': r['sync_mean'], 'lz_full': r['lz_mean']})
            ), axis=1
        )

        pivot = sub.pivot_table(
            values='regime_num',
            index='m',
            columns='lambda2_bin',
            aggfunc='mean'
        )

        cmap_regime = plt.cm.colors.ListedColormap([
            '#1f77b4',   # 0 = FROZEN (blue)
            '#2ca02c',   # 1 = FUNCTIONAL (green)
            '#d62728',   # 2 = CHAOTIC (red)
            '#ff7f0e',   # 3 = SYNCED (orange)
        ])
        bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
        norm_regime = plt.cm.colors.BoundaryNorm(bounds, cmap_regime.N)

        im = ax.imshow(pivot.values, aspect='auto', origin='lower',
                      cmap=cmap_regime, norm=norm_regime)

        ax.set_xlabel('Lambda2 bin')
        ax.set_ylabel('m (BA)')
        ax.set_title(f'Regime map D={D}\nBlue=FROZEN, Green=FUNCTIONAL, Red=CHAOTIC, Orange=SYNCED')
        ax.set_xticks(range(0, len(pivot.columns), max(1, len(pivot.columns) // 8)))
        ax.set_xticklabels([f"{p:.1f}" for p in pivot.columns[::max(1, len(pivot.columns) // 8)]], fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)

        # Add colorbar with regime labels
        cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
        cbar.ax.set_yticklabels(['FROZEN', 'FUNCTIONAL', 'CHAOTIC', 'SYNCED'])
        cbar.ax.tick_params(labelsize=8)

    fig4.tight_layout()
    fig4.savefig(os.path.join(figures_dir, 'fss_lz_regime_map.png'), dpi=150)
    print(f"Figure: fss_lz_regime_map.png")

    # =====================================================================
    # FIGURE 5: Sync vs lambda2 with LZ color (the key combined view)
    # =====================================================================
    fig5, axes5 = plt.subplots(1, len(D_VALUES), figsize=(14, 5),
                                sharey=False)
    if len(D_VALUES) == 1:
        axes5 = [axes5]

    for i, D in enumerate(D_VALUES):
        ax = axes5[i]
        sub = df[df['D'] == D].sort_values('lambda2')

        sc = ax.scatter(sub['lambda2'], sub['synchrony'],
                       c=sub['lz_full'], cmap='RdYlGn',
                       vmin=0, vmax=1,
                       alpha=0.75, s=45, edgecolors='white', linewidths=0.3)

        ax.axhline(0.3, color='red', linestyle='--', alpha=0.6, label='Sync=0.3 (SYNCED)')
        ax.axhline(0.0, color='gray', linestyle=':', alpha=0.4)

        ax.set_xlabel('Lambda2 (Fiedler)')
        ax.set_ylabel('Synchrony (Pearson r)')
        ax.set_title(f'D={D}  —  Sync vs lambda2 (color = LZ76)\nGreen=FUNCTIONAL, Red=FROZEN/CHAOTIC')
        ax.set_ylim(-0.25, 0.75)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('LZ76 Complexity', fontsize=9)

    fig5.tight_layout()
    fig5.savefig(os.path.join(figures_dir, 'fss_lz_sync_lambda2.png'), dpi=150)
    print(f"Figure: fss_lz_sync_lambda2.png")


if __name__ == "__main__":
    main()
