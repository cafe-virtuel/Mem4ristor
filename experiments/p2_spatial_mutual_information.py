#!/usr/bin/env python3
"""
Piste A4 -- Information Mutuelle Spatio-Temporelle (2026-04-24)

Hypothese : H_stable (entropie marginale spatiale) ne distingue pas le
desordre aleatoire de la diversite structuree. L'Information Mutuelle (MI)
entre noeuds voisins vs noeuds distants sur le graphe revele une longueur
de correlation caracteristique du regime Mem4ristor.

Prediction :
  - FULL : MI decroit avec la distance (structure locale preservee)
  - FROZEN_U : MI haute et plate (synchronie globale, pas de structure locale)
  - NO_SIGMOID : MI basse et plate (desordre non structure)
  - NO_HERETIC : collapse global, MI ~ 0 partout

4 ablations x Lattice 10x10 et BA m=3 N=100.
Regime force (I_stim=0.5) pour activer les heretiques.

Metriques : MI(d) = MI moyenne entre paires a distance d sur le graphe.

Script  : experiments/p2_spatial_mutual_information.py
Figures : figures/p2_spatial_mutual_information.png
CSV     : figures/p2_spatial_mutual_information.csv

Reference : PROJECT_STATUS.md §P2-AUDIT Piste A4
"""
import sys, os, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from mem4ristor.core import Mem4Network
from mem4ristor.metrics import calculate_spatial_mutual_information

# -- Parametres ---------------------------------------------------------------
N_BA    = 100
M_BA    = 3
STEPS   = 3000
WARM_UP = 1000
SEEDS   = [42, 123, 777]
I_STIM  = 0.5   # force : heretiques actifs

ABLATIONS = {
    'FULL':        {'heretic_ratio': 0.15, 'freeze_u': False, 'use_sigmoid': True},
    'NO_HERETIC':  {'heretic_ratio': 0.00, 'freeze_u': False, 'use_sigmoid': True},
    'FROZEN_U':    {'heretic_ratio': 0.15, 'freeze_u': True,  'use_sigmoid': True},
    'NO_SIGMOID':  {'heretic_ratio': 0.15, 'freeze_u': False, 'use_sigmoid': False},
}

TOPOS = ['lattice', 'ba_m3']


def make_ba(n, m, seed):
    rng = np.random.RandomState(seed)
    adj = np.zeros((n, n), dtype=float)
    for i in range(m + 1):
        for j in range(i + 1, m + 1):
            adj[i, j] = adj[j, i] = 1.0
    degrees = adj.sum(axis=1)
    for new_node in range(m + 1, n):
        probs = degrees[:new_node] / degrees[:new_node].sum()
        targets = rng.choice(new_node, size=m, replace=False, p=probs)
        for t in targets:
            adj[new_node, t] = adj[t, new_node] = 1.0
        degrees = adj.sum(axis=1)
    return adj


def run_ablation(topo, ablation_cfg, seed):
    hr = ablation_cfg['heretic_ratio']
    freeze_u = ablation_cfg['freeze_u']

    if topo == 'lattice':
        net = Mem4Network(size=10, heretic_ratio=hr, seed=seed,
                          coupling_norm='degree_linear')
        adj = None   # lattice adjacency computed internally via stencil
    else:
        adj = make_ba(N_BA, M_BA, seed)
        net = Mem4Network(adjacency_matrix=adj.copy(), heretic_ratio=hr,
                          seed=seed, coupling_norm='degree_linear')

    if not ablation_cfg['use_sigmoid']:
        # Replace sigmoid with identity: u_filter = (0.5 - u) + delta
        net.model.cfg['coupling']['social_leakage'] = 0.01
        net.model.sigmoid_steepness = 0.0   # tanh(0 * x) = 0 -> linear regime

    v_history = []
    for step in range(STEPS):
        if freeze_u:
            u_backup = net.model.u.copy()
        net.step(I_stimulus=I_STIM)
        if freeze_u:
            net.model.u[:] = u_backup
        if step >= WARM_UP:
            v_history.append(net.v.copy())

    v_arr = np.array(v_history)   # (T_tail, N)

    # Get adjacency for MI computation
    if topo == 'lattice':
        # Build lattice adjacency from stencil
        N = net.N
        size = net.size
        adj_lat = np.zeros((N, N), dtype=float)
        for i in range(size):
            for j in range(size):
                node = i * size + j
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = (i + di) % size, (j + dj) % size
                    adj_lat[node, ni * size + nj] = 1.0
        adj_for_mi = adj_lat
    else:
        adj_for_mi = adj

    mi_by_dist = calculate_spatial_mutual_information(
        v_arr, adj_for_mi, n_bins=20, max_pairs_per_dist=150, max_dist=8)
    return mi_by_dist


# -- Main ---------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 80)
    print("Piste A4 -- Information Mutuelle Spatio-Temporelle")
    print(f"I_stim={I_STIM} | steps={STEPS} | warm_up={WARM_UP} | seeds={SEEDS}")
    print("=" * 80)

    t0   = time.time()
    rows = []
    all_results = {}

    for topo in TOPOS:
        print(f"\nTopologie : {topo}")
        all_results[topo] = {}
        for abl_name, abl_cfg in ABLATIONS.items():
            print(f"  {abl_name}...")
            # Aggregate MI by distance across seeds
            dist_accumulator = {}
            for seed in SEEDS:
                mi_dict = run_ablation(topo, abl_cfg, seed)
                for d, (mean_mi, _) in mi_dict.items():
                    dist_accumulator.setdefault(d, []).append(mean_mi)
            # Average across seeds
            agg = {d: (np.mean(v), np.std(v)) for d, v in dist_accumulator.items()}
            all_results[topo][abl_name] = agg
            # Print summary
            dists = sorted(agg.keys())
            mi_d1 = agg.get(1, (0, 0))[0]
            mi_d3 = agg.get(3, (0, 0))[0]
            mi_d5 = agg.get(5, (0, 0))[0]
            decay = mi_d1 - mi_d5 if mi_d1 > 0 else 0
            print(f"    d=1:{mi_d1:.4f}  d=3:{mi_d3:.4f}  d=5:{mi_d5:.4f}  decay={decay:.4f}")
            for d in dists:
                m_mi, s_mi = agg[d]
                rows.append({
                    'topo': topo, 'ablation': abl_name, 'distance': d,
                    'mi_mean': m_mi, 'mi_std': s_mi,
                })

    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed:.1f}s")

    # -- CSV ------------------------------------------------------------------
    import csv, pathlib
    fig_dir = pathlib.Path(__file__).resolve().parents[1] / 'figures'
    fig_dir.mkdir(exist_ok=True)
    csv_path = fig_dir / 'p2_spatial_mutual_information.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV : {csv_path}")

    # -- Figure ---------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        abl_colors = {
            'FULL':       'steelblue',
            'NO_HERETIC': 'gray',
            'FROZEN_U':   'darkorange',
            'NO_SIGMOID': 'crimson',
        }
        abl_ls = {
            'FULL':       '-',
            'NO_HERETIC': ':',
            'FROZEN_U':   '--',
            'NO_SIGMOID': '-.',
        }

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        for ax, topo in zip(axes, TOPOS):
            for abl_name in ABLATIONS:
                agg = all_results[topo].get(abl_name, {})
                dists = sorted(agg.keys())
                if not dists:
                    continue
                xs = dists
                ys = [agg[d][0] for d in dists]
                es = [agg[d][1] for d in dists]
                ax.errorbar(xs, ys, yerr=es, marker='o', label=abl_name,
                            color=abl_colors[abl_name],
                            linestyle=abl_ls[abl_name], capsize=3)
            ax.set_xlabel('Graph hop-distance d')
            ax.set_ylabel('Mean MI (bits)')
            ax.set_title(f'MI decay — {topo}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(left=0.5)

        fig.suptitle(
            'Piste A4 -- Spatial Mutual Information vs Graph Distance\n'
            f'I_stim={I_STIM}, coupling=degree_linear, {len(SEEDS)} seeds',
            fontsize=11
        )
        plt.tight_layout()
        png_path = fig_dir / 'p2_spatial_mutual_information.png'
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        print(f"Figure : {png_path}")
    except Exception as e:
        print(f"[matplotlib error] {e}")
