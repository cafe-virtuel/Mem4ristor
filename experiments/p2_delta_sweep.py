#!/usr/bin/env python3
"""
Piste A5 -- Sweep delta de la Levitating Sigmoid (2026-04-24)

Hypothese : Le parametre delta=0.01 dans
    w_i(u_i) = tanh(pi*(0.5 - u_i)) + delta
brise la symetrie parfaite au point de doute maximal (u=0.5).
Delta a ete introduit comme fix technique (LIMIT-01 : eviter couplage nul),
mais c'est en realite un **parametre de controle de la symetrie sociale** :
  - delta > 0 : biais vers l'attraction faible (consensus)
  - delta < 0 : biais vers la repulsion faible (divergence)
  - delta = 0 : symetrie parfaite

Il existe un delta_crit qui maximise la complexite LZ temporelle.

Sweep : delta in [-0.10, -0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05, 0.10]
        sur Lattice 10x10 et BA m=3 N=100.
Regime : I_stim=0.5 (force), coupling_norm='degree_linear'.
Metriques : H_cont, H_cog, LZ temporelle, pairwise_synchrony.

Script  : experiments/p2_delta_sweep.py
Figures : figures/p2_delta_sweep.png
CSV     : figures/p2_delta_sweep.csv

Reference : PROJECT_STATUS.md §P2-AUDIT Piste A5
"""
import sys, os, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from mem4ristor.core import Mem4Network
from mem4ristor.graph_utils import make_ba
from mem4ristor.metrics import (
    calculate_cognitive_entropy,
    calculate_continuous_entropy,
    calculate_temporal_lz_complexity,
    calculate_pairwise_synchrony,
)

# -- Parametres ---------------------------------------------------------------
DELTAS  = [-0.10, -0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05, 0.10]
SEEDS   = [42, 123, 777]
I_STIM  = 0.5
STEPS   = 3000
WARM_UP = 750
N_BA    = 100
M_BA    = 3
RECORD_EVERY = 5   # pour historique LZ




def run_one(topo, delta, seed):
    if topo == 'lattice':
        net = Mem4Network(size=10, heretic_ratio=0.15, seed=seed,
                          coupling_norm='degree_linear')
    else:
        adj = make_ba(N_BA, M_BA, seed)
        net = Mem4Network(adjacency_matrix=adj, heretic_ratio=0.15, seed=seed,
                          coupling_norm='degree_linear')

    net.model.social_leakage = delta

    v_snaps   = []
    v_history = []
    for step in range(STEPS):
        net.step(I_stimulus=I_STIM)
        if step >= WARM_UP:
            v_snaps.append(net.v.copy())
            if step % RECORD_EVERY == 0:
                v_history.append(net.v.copy())

    v_s = np.array(v_snaps)
    v_h = np.array(v_history)

    h_cont = float(np.mean([calculate_continuous_entropy(v) for v in v_s[::10]]))
    h_cog  = float(np.mean([calculate_cognitive_entropy(v)  for v in v_s[::10]]))
    lz     = calculate_temporal_lz_complexity(v_h) if len(v_h) > 1 else 0.0
    sync   = calculate_pairwise_synchrony(v_s)
    return h_cont, h_cog, lz, sync


# -- Main ---------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 80)
    print("Piste A5 -- Sweep delta (Levitating Sigmoid social_leakage)")
    print(f"delta: {DELTAS}")
    print(f"I_stim={I_STIM} | steps={STEPS} | seeds={SEEDS}")
    print("=" * 80)

    t0   = time.time()
    rows = []

    for topo in ['lattice', 'ba_m3']:
        print(f"\nTopologie : {topo}")
        print(f"  {'delta':>7}  {'H_cont':>7}  {'H_cog':>7}  {'LZ':>7}  {'sync':>7}")
        for delta in DELTAS:
            h_c_l, h_k_l, lz_l, sy_l = [], [], [], []
            for seed in SEEDS:
                h_c, h_k, lz, sy = run_one(topo, delta, seed)
                h_c_l.append(h_c); h_k_l.append(h_k)
                lz_l.append(lz);   sy_l.append(sy)
            h_c_m = np.mean(h_c_l); h_k_m = np.mean(h_k_l)
            lz_m  = np.mean(lz_l);  sy_m  = np.mean(sy_l)
            flag = " <-- MAX_LZ" if lz_m == max(lz_l) else ""
            print(f"  {delta:>7.3f}  {h_c_m:>7.4f}  {h_k_m:>7.4f}  {lz_m:>7.4f}  {sy_m:>7.4f}")
            rows.append({
                'topo': topo, 'delta': delta,
                'h_cont_mean': h_c_m, 'h_cont_std': np.std(h_c_l),
                'h_cog_mean':  h_k_m, 'h_cog_std':  np.std(h_k_l),
                'lz_mean':     lz_m,  'lz_std':      np.std(lz_l),
                'sync_mean':   sy_m,  'sync_std':     np.std(sy_l),
            })

        # Best delta per topo
        topo_rows = [r for r in rows if r['topo'] == topo]
        best = max(topo_rows, key=lambda r: r['lz_mean'])
        print(f"  --> delta_crit (max LZ) = {best['delta']:.3f}  "
              f"LZ={best['lz_mean']:.4f}  H_cog={best['h_cog_mean']:.4f}")

    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed:.1f}s")

    # -- CSV ------------------------------------------------------------------
    import csv, pathlib
    fig_dir = pathlib.Path(__file__).resolve().parents[1] / 'figures'
    fig_dir.mkdir(exist_ok=True)
    csv_path = fig_dir / 'p2_delta_sweep.csv'
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

        topos  = ['lattice', 'ba_m3']
        colors = {'lattice': 'steelblue', 'ba_m3': 'darkorange'}
        labels = {'lattice': 'Lattice 10x10', 'ba_m3': 'BA m=3 N=100'}

        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        metrics = [
            ('h_cog_mean',  'H_cog (5-bin KIMI)',  'bits'),
            ('lz_mean',     'LZ complexity',        'normalized LZ'),
            ('sync_mean',   'Pairwise Synchrony',   'correlation'),
            ('h_cont_mean', 'H_cont (100-bin)',     'bits'),
        ]

        for ax, (key, title, ylabel) in zip(axes.flat, metrics):
            for topo in topos:
                xs  = [r['delta'] for r in rows if r['topo'] == topo]
                ys  = [r[key]     for r in rows if r['topo'] == topo]
                std_key = key.replace('mean', 'std')
                es  = [r.get(std_key, 0) for r in rows if r['topo'] == topo]
                ax.errorbar(xs, ys, yerr=es, marker='o', label=labels[topo],
                            color=colors[topo], capsize=3)
            ax.axvline(0.01, color='gray', linestyle='--', alpha=0.5, label='default delta=0.01')
            ax.axvline(0.0,  color='black', linestyle=':', alpha=0.3, label='delta=0 (sym)')
            ax.set_xlabel('delta (social_leakage)')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.suptitle(
            'Piste A5 -- delta sweep (Levitating Sigmoid)\n'
            f'I_stim={I_STIM}, coupling=degree_linear, {len(SEEDS)} seeds',
            fontsize=11
        )
        plt.tight_layout()
        png_path = fig_dir / 'p2_delta_sweep.png'
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        print(f"Figure : {png_path}")
    except Exception as e:
        print(f"[matplotlib error] {e}")
