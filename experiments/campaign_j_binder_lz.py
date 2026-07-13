"""
Campaign J — Binder + LZ Joint Analysis
Auteur: Hermes
Date: 2026-05-31

Objectif: Verifier si la reinterpretation "chaos pas freeze" (LZ) et le resultat
Binder (U4, phase transition) sont coherents ou en tension.

Protocol: Sur les memes simulations, mesurer:
  - U4 (Binder cumulant)  — confirme la transition de phase
  - LZ (LZ76 temporal)    — distingue structure vs chaos
  - H_stable              — entropie spectrale continue

Design: identique a v6_binder_cumulant_u4.py (meme reseau, memes parametres),
  avec conservation de v_history pour calculer LZ sur le tail.

Sweep: c in [0, 15] -> lambda2 couvre [~1, ~6] (zone critique: 2.0-3.0)
N sizes: 100, 200, 400
Seeds: 40 par configuration (comme original v6)
Steps: 2500 (1875 warmup + 625 stat, tail = derniers 25%)
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
from mem4ristor.metrics import calculate_continuous_entropy, calculate_temporal_lz_complexity

# Configuration physique
LAMBDA2_CRIT = 2.31

# Parametres campagne
DEFAULT_N_SIZES = [100, 200, 400]
DEFAULT_N_SEEDS = 40
DEFAULT_STEPS = 2500
DEFAULT_C_MAX = 15.0
DEFAULT_C_STEPS = 15  # c in [0, 15] step 1.0 -> good resolution around 2.31

LZ_N_BINS = 5  # must match the published threshold (5-bin LZ for consistency with paper)


def _fiedler(adj):
    """Valeur propre de Fiedler (algebraic connectivity)."""
    degree = np.array(adj.sum(axis=1)).flatten()
    L = np.diag(degree) - adj
    eigs = np.sort(np.linalg.eigvalsh(L))
    return float(eigs[1]) if len(eigs) > 1 else 0.0


def make_continuous_ba(N, m_base, p_add, seed):
    """BA(m_base) + random edge additions scaled by p_add = c/N."""
    adj = make_ba(N, m_base, seed)
    rng = np.random.RandomState(seed + 1000)
    for i in range(N):
        for j in range(i + 1, N):
            if adj[i, j] == 0 and rng.rand() < p_add:
                adj[i, j] = adj[j, i] = 1.0
    return adj


def run_single_sim(args):
    """
    Simule une realisation du reseau Mem4.
    Mesure: H_stable, U4, LZ sur le tail (derniers 25% des pas).
    """
    N, m_base, c, seed, steps = args
    p_add = c / N
    adj = make_continuous_ba(N, m_base, p_add, seed)
    l2 = _fiedler(adj)

    # Filtrage large
    if l2 < 0.1 or l2 > 15.0:
        return {
            'N': N, 'c': c, 'seed': seed, 'lambda2': l2,
            'H_stable': np.nan, 'U4': np.nan, 'LZ': np.nan,
            'sync': np.nan, 'H_cont': np.nan
        }

    net = Mem4Network(adjacency_matrix=adj, coupling_norm='degree_linear', seed=seed)

    # Enregistrement du tail (derniers 25%)
    tail_start = int(steps * 0.75)
    H_list = []
    v_history = []  # pour LZ et sync

    for step in range(steps):
        net.step(I_stimulus=0.5)
        if step >= tail_start:
            # Snapshot de tension: net.v est l'array (N,) des tensions
            v_snap = net.v.copy()
            if v_snap is not None and len(v_snap) == N:
                H_list.append(calculate_continuous_entropy(v_snap, bins=100))
                v_history.append(v_snap.copy())

    H_stable = np.mean(H_list) if H_list else np.nan

    # Calculs sur le tail
    if len(v_history) >= 10:
        v_hist = np.array(v_history)  # shape (T_tail, N)
        T_tail = v_hist.shape[0]

        # LZ76 complexity (moyenne sur nodes, normalise)
        LZ = calculate_temporal_lz_complexity(v_hist, n_bins=LZ_N_BINS)

        # Pairwise synchrony
        sync = _pairwise_synchrony(v_hist)

        # U4 via eq: U4 = 1 - <m^4>/(3<m^2>^2) avec m = H_stable
        H_vals = np.array(H_list)
        m2 = np.mean(H_vals ** 2)
        m4 = np.mean(H_vals ** 4)
        U4 = (1.0 - m4 / (3.0 * m2**2)) if m2 > 1e-9 else np.nan

        # H_cont (derniere frame)
        H_cont = calculate_continuous_entropy(v_hist[-1], bins=100)

    else:
        LZ = np.nan
        sync = np.nan
        U4 = np.nan
        H_cont = np.nan

    return {
        'N': N, 'c': c, 'seed': seed, 'lambda2': l2,
        'H_stable': H_stable, 'U4': U4, 'LZ': LZ,
        'sync': sync, 'H_cont': H_cont
    }


def _pairwise_synchrony(v_history):
    """
    Mean pairwise Pearson correlation of v(t) traces.
    Subsampled pour O(N)complexite.
    """
    T, N = v_history.shape
    if T < 2 or N < 2:
        return 0.0

    # Subsample: max 500 pairs
    max_pairs = 500
    n_pairs = min(max_pairs, N * (N - 1) // 2)
    rng = np.random.RandomState(42)

    corrs = []
    for _ in range(n_pairs):
        i = rng.randint(N)
        j = rng.randint(N - 1)
        if j >= i:
            j += 1
        vi = v_history[:, i]
        vj = v_history[:, j]
        std_i = np.std(vi)
        std_j = np.std(vj)
        if std_i > 1e-9 and std_j > 1e-9:
            cov = np.mean((vi - np.mean(vi)) * (vj - np.mean(vj)))
            corrs.append(cov / (std_i * std_j))

    return np.mean(corrs) if corrs else 0.0


def main():
    parser = argparse.ArgumentParser(description="Campaign J: Binder + LZ Joint Analysis")
    parser.add_argument('--dry-run', action='store_true', help="Test rapide petite echelle")
    parser.add_argument('--plot-only', action='store_true', help="Plot depuis CSV existant sans relancer")
    args = parser.parse_args()

    if args.plot_only:
        print(">>> MODE PLOT ONLY <<<")
        plot_results()
        return

    if args.dry_run:
        print(">>> MODE DRY-RUN ACTIF <<<")
        N_SIZES = [100]
        N_SEEDS = 3
        STEPS = 500
        c_vals = np.linspace(0.0, 8.0, 5)
    else:
        print(">>> MODE CAMPAGNE COMPLETE ACTIF <<<")
        N_SIZES = DEFAULT_N_SIZES
        N_SEEDS = DEFAULT_N_SEEDS
        STEPS = DEFAULT_STEPS
        c_vals = np.linspace(0.0, DEFAULT_C_MAX, DEFAULT_C_STEPS)

    m_base = 3

    print(f"N sizes: {N_SIZES}")
    print(f"Seeds per config: {N_SEEDS}")
    print(f"Steps per run: {STEPS}")
    print(f"c sweep: {c_vals}")
    print(f"Total simulations: {len(N_SIZES) * len(c_vals) * N_SEEDS}")

    tasks = []
    for N in N_SIZES:
        for c in c_vals:
            for seed in range(N_SEEDS):
                tasks.append((N, m_base, c, seed, STEPS))

    results = []
    max_workers = os.cpu_count() or 4
    print(f"Parallel execution on {max_workers} processes...")

    t_start = time.time()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_single_sim, task): task for task in tasks}
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Simulations"):
            try:
                res = future.result()
                if not np.isnan(res['H_stable']):
                    results.append(res)
            except Exception as e:
                print(f"Error: {e}")

    t_elapsed = time.time() - t_start
    print(f"Simulations done in {t_elapsed:.1f}s — {len(results)} valid results")

    if not results:
        print("No valid results. Aborting.")
        return

    df = pd.DataFrame(results)
    figures_dir = os.path.join(PROJECT_ROOT, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    # Save raw data
    raw_path = os.path.join(figures_dir, 'campaign_j_raw.csv')
    df.to_csv(raw_path, index=False)
    print(f"Raw data saved: {raw_path}")

    # Compute binned statistics for U4
    df['lambda2_bin'] = np.round(df['lambda2'] * 4) / 4.0

    grouped = df.groupby(['N', 'lambda2_bin'])
    agg_results = []
    for (N, l2_bin), group in grouped:
        if len(group) < 3:
            continue
        agg_results.append({
            'N': N,
            'lambda2_bin': l2_bin,
            'lambda2_mean': group['lambda2'].mean(),
            'H_stable_mean': group['H_stable'].mean(),
            'H_stable_std': group['H_stable'].std(),
            'U4_mean': group['U4'].mean(),
            'U4_std': group['U4'].std(),
            'LZ_mean': group['LZ'].mean(),
            'LZ_std': group['LZ'].std(),
            'sync_mean': group['sync'].mean(),
            'sync_std': group['sync'].std(),
            'H_cont_mean': group['H_cont'].mean(),
            'count': len(group)
        })

    df_agg = pd.DataFrame(agg_results).dropna()
    agg_path = os.path.join(figures_dir, 'campaign_j_agg.csv')
    df_agg.to_csv(agg_path, index=False)
    print(f"Aggregated data saved: {agg_path}")

    # Generate plots
    plot_results(df, df_agg, figures_dir)

    print("\n=== CAMPAIGN J SUMMARY ===")
    print_zone_analysis(df_agg)
    print("\nCampaign J complete.")


def plot_results(df=None, df_agg=None, figures_dir=None):
    """Generate all plots from aggregated CSV."""
    if df_agg is None:
        # Load from file
        figures_dir = figures_dir or os.path.join(PROJECT_ROOT, 'figures')
        agg_path = os.path.join(figures_dir, 'campaign_j_agg.csv')
        raw_path = os.path.join(figures_dir, 'campaign_j_raw.csv')
        if not os.path.exists(agg_path):
            print(f"No aggregated CSV found at {agg_path}")
            return
        df_agg = pd.read_csv(agg_path)
        if os.path.exists(raw_path):
            df = pd.read_csv(raw_path)
        else:
            df = None

    plt.rcParams.update({
        'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
        'grid.alpha': 0.25, 'grid.linestyle': '--'
    })

    ns = sorted(df_agg['N'].unique())
    colors = {100: '#1f77b4', 200: '#ff7f0e', 400: '#2ca02c'}
    markers = {100: 'o', 200: 's', 400: '^'}

    # --- Figure 1: U4 vs lambda2 (Binder) ---
    fig1, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax = axes[0]
    for N in ns:
        sub = df_agg[df_agg['N'] == N].sort_values('lambda2_mean')
        if len(sub) > 0:
            ax.errorbar(sub['lambda2_mean'], sub['U4_mean'],
                        yerr=sub['U4_std'], marker=markers.get(N, 'o'),
                        color=colors.get(N, '#333'), linewidth=2, markersize=6,
                        label=f'N={N}', capsize=3)
    ax.axvline(LAMBDA2_CRIT, color='#d62728', linestyle='--', linewidth=1.5, label=f'λ2_crit={LAMBDA2_CRIT}')
    ax.axhline(2/3, color='gray', linestyle=':', linewidth=1, label='U4=2/3 (Ising)')
    ax.set_xlabel('Algebraic Connectivity λ2')
    ax.set_ylabel('Binder Cumulant U4')
    ax.set_title('Binder Cumulant U4 vs λ2\n(Convergence avec N?)')
    ax.legend()
    ax.grid(True)

    # --- Figure 1 right: LZ vs lambda2 ---
    ax2 = axes[1]
    for N in ns:
        sub = df_agg[df_agg['N'] == N].sort_values('lambda2_mean')
        if len(sub) > 0:
            ax2.errorbar(sub['lambda2_mean'], sub['LZ_mean'],
                         yerr=sub['LZ_std'], marker=markers.get(N, 'o'),
                         color=colors.get(N, '#333'), linewidth=2, markersize=6,
                         label=f'N={N}', capsize=3)
    ax2.axvline(LAMBDA2_CRIT, color='#d62728', linestyle='--', linewidth=1.5, label=f'λ2_crit={LAMBDA2_CRIT}')
    ax2.axhline(0.85, color='orange', linestyle=':', linewidth=1.5, label='LZ=0.85 (threshold)')
    ax2.set_xlabel('Algebraic Connectivity λ2')
    ax2.set_ylabel('LZ76 Complexity')
    ax2.set_title('LZ76 Complexity vs λ2\n(Structure vs Chaos)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    fig1.savefig(os.path.join(figures_dir, 'campaign_j_binder_and_lz.png'), dpi=150)
    print(f"Saved: campaign_j_binder_and_lz.png")

    # --- Figure 2: H_stable + sync ---
    fig2, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax3 = axes[0]
    for N in ns:
        sub = df_agg[df_agg['N'] == N].sort_values('lambda2_mean')
        if len(sub) > 0:
            ax3.errorbar(sub['lambda2_mean'], sub['H_stable_mean'],
                         yerr=sub['H_stable_std'], marker=markers.get(N, 'o'),
                         color=colors.get(N, '#333'), linewidth=2, markersize=6,
                         label=f'N={N}', capsize=3)
    ax3.axvline(LAMBDA2_CRIT, color='#d62728', linestyle='--', linewidth=1.5, label=f'λ2_crit={LAMBDA2_CRIT}')
    ax3.set_xlabel('Algebraic Connectivity λ2')
    ax3.set_ylabel('H_stable (100-bin)')
    ax3.set_title('Spectral Entropy H_stable vs λ2')
    ax3.legend()
    ax3.grid(True)

    ax4 = axes[1]
    for N in ns:
        sub = df_agg[df_agg['N'] == N].sort_values('lambda2_mean')
        if len(sub) > 0:
            ax4.errorbar(sub['lambda2_mean'], sub['sync_mean'],
                         yerr=sub['sync_std'], marker=markers.get(N, 'o'),
                         color=colors.get(N, '#333'), linewidth=2, markersize=6,
                         label=f'N={N}', capsize=3)
    ax4.axvline(LAMBDA2_CRIT, color='#d62728', linestyle='--', linewidth=1.5, label=f'λ2_crit={LAMBDA2_CRIT}')
    ax4.set_xlabel('Algebraic Connectivity λ2')
    ax4.set_ylabel('Pairwise Synchrony')
    ax4.set_title('Synchrony vs λ2 (Lower = More Independent)')
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    fig2.savefig(os.path.join(figures_dir, 'campaign_j_entropy_sync.png'), dpi=150)
    print(f"Saved: campaign_j_entropy_sync.png")

    # --- Figure 3: LZ vs U4 scatter (coherence check) ---
    if df is not None and len(df) > 0:
        fig3, ax = plt.subplots(figsize=(8, 6))
        # Color by lambda2 zone
        for zone, color, label in [
            (('lambda2', 0, 2.0), '#2ca02c', 'λ2 < 2 (sparse)'),
            (('lambda2', 2.0, 3.0), '#ff7f0e', '2 ≤ λ2 ≤ 3 (critical)'),
            (('lambda2', 3.0, 15.0), '#d62728', 'λ2 > 3 (dense)')
        ]:
            col, lo, hi = zone
            mask = (df[col] >= lo) & (df[col] < hi)
            sub = df[mask]
            if len(sub) > 0:
                ax.scatter(sub['U4'], sub['LZ'], alpha=0.4, s=20, c=color, label=label)
        ax.axhline(0.85, color='orange', linestyle=':', linewidth=1.5, label='LZ=0.85')
        ax.axvline(2/3, color='gray', linestyle=':', linewidth=1, label='U4=2/3')
        ax.set_xlabel('Binder U4')
        ax.set_ylabel('LZ76 Complexity')
        ax.set_title('U4 vs LZ: Structure or Chaos?\n(Each dot = one simulation run)')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        fig3.savefig(os.path.join(figures_dir, 'campaign_j_u4_vs_lz.png'), dpi=150)
        print(f"Saved: campaign_j_u4_vs_lz.png")


def print_zone_analysis(df_agg):
    """Print zone-by-zone summary of metrics."""
    print("\n--- Zone Analysis ---")
    zones = [
        ('Sparse', 0, 2.0),
        ('Critical', 2.0, 3.0),
        ('Dense', 3.0, 15.0)
    ]
    for name, lo, hi in zones:
        mask = (df_agg['lambda2_mean'] >= lo) & (df_agg['lambda2_mean'] < hi)
        sub = df_agg[mask]
        if len(sub) == 0:
            print(f"  {name} ({lo}-{hi}): no data")
            continue
        u4 = sub['U4_mean'].mean()
        lz = sub['LZ_mean'].mean()
        h = sub['H_stable_mean'].mean()
        syn = sub['sync_mean'].mean()
        print(f"  {name} ({lo}-{hi}): U4={u4:.4f}, LZ={lz:.4f}, H={h:.2f}, sync={syn:.3f}")

    print("\n--- Coherence Check ---")
    # For each N, find the lambda2 with minimum U4 (strongest transition signal)
    for N in sorted(df_agg['N'].unique()):
        sub = df_agg[df_agg['N'] == N]
        if len(sub) == 0:
            continue
        min_u4_row = sub.loc[sub['U4_mean'].idxmin()]
        l2_min = min_u4_row['lambda2_mean']
        u4_min = min_u4_row['U4_mean']
        # Find closest LZ at same N and lambda2 (within 0.5 bin)
        lz_candidates = sub[(sub['lambda2_mean'] - l2_min).abs() < 0.5]
        if len(lz_candidates) > 0:
            lz_at_min = lz_candidates['LZ_mean'].mean()
            chaos = lz_at_min > 0.85
            label = "CHAOS (LZ>0.85)" if chaos else "STRUCTURED (LZ<0.85)"
            print(f"  N={int(N)}: U4 minimum at λ2={l2_min:.2f} → U4={u4_min:.4f}, LZ={lz_at_min:.4f} → {label}")
        else:
            print(f"  N={int(N)}: U4 minimum at λ2={l2_min:.2f} → U4={u4_min:.4f}, LZ data unavailable")


if __name__ == "__main__":
    main()