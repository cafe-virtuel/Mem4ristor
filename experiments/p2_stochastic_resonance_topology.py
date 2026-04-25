#!/usr/bin/env python3
"""
Item 10 -- Stochastic Resonance x Topology (2026-04-25)

Hypothese : Pour chaque topologie (caracterisee par lambda2 = connectivite
algebrique), il existe un sigma_noise optimal sigma* qui maximise la diversite
cognitive H_cog (resonance stochastique). sigma* depend-il systematiquement
de lambda2 ?

Approche :
  1. 7 topologies couvrant un large spectre de lambda2 (BA m=2..8, Lattice, ER).
  2. Sweep sigma in [0, 0.01, 0.03, 0.07, 0.15, 0.30, 0.50, 0.80, 1.20].
  3. Injection de bruit uniforme via sigma_v_vec = full(N, sigma).
  4. sigma* = argmax H_cog(sigma) par topologie.
  5. Scatter sigma* vs lambda2 -> relation SR-topologie.

Topologies : BA m=2/3/5/8 (N=100), Lattice 10x10, ER p=0.05/0.10 (N=100).
Normalisation : degree_linear (regime non-dead-zone).
Regime : I_stim=0.5 (heretiques actifs).

Script  : experiments/p2_stochastic_resonance_topology.py
Figures : figures/p2_stochastic_resonance_topology.png
CSV     : figures/p2_stochastic_resonance_topology.csv

Reference : PROJECT_STATUS.md Item 10
"""
import sys, os, time
import numpy as np
from scipy.linalg import eigh

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
SIGMAS  = [0.0, 0.01, 0.03, 0.07, 0.15, 0.30, 0.50, 0.80, 1.20]
SEEDS   = [42, 123, 777]
I_STIM  = 0.5
STEPS   = 3000
WARM_UP = 750
N       = 100


# -- Topology builders --------------------------------------------------------



def make_er(n, p, seed):
    rng = np.random.RandomState(seed)
    adj = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.rand() < p:
                adj[i, j] = adj[j, i] = 1.0
    # Ensure connectivity: add spanning tree if isolated nodes exist
    degrees = adj.sum(axis=1)
    for i in range(n):
        if degrees[i] == 0:
            j = rng.randint(0, n)
            while j == i:
                j = rng.randint(0, n)
            adj[i, j] = adj[j, i] = 1.0
    return adj


def make_lattice(size):
    n = size * size
    adj = np.zeros((n, n), dtype=float)
    for i in range(size):
        for j in range(size):
            node = i * size + j
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = (i + di) % size, (j + dj) % size
                adj[node, ni * size + nj] = 1.0
    return adj


def compute_lambda2(adj):
    deg = adj.sum(axis=1)
    L   = np.diag(deg) - adj
    evals = eigh(L, eigvals_only=True, subset_by_index=[1, 1])
    return float(evals[0])


# -- Topology registry --------------------------------------------------------
# Each entry: (name, adj_builder_fn, Mem4Network kwargs)
def get_topologies(seed):
    topos = []

    for m in [2, 3, 5, 8]:
        adj = make_ba(N, m, seed)
        topos.append({
            'name':   f'BA_m{m}',
            'adj':    adj,
            'lambda2': compute_lambda2(adj),
        })

    # Lattice 10x10
    adj_lat = make_lattice(10)
    topos.append({
        'name':   'Lattice_10x10',
        'adj':    adj_lat,
        'lambda2': compute_lambda2(adj_lat),
    })

    for p in [0.05, 0.10]:
        adj = make_er(N, p, seed)
        topos.append({
            'name':   f'ER_p{int(p*100):02d}',
            'adj':    adj,
            'lambda2': compute_lambda2(adj),
        })

    return topos


# -- Single run ---------------------------------------------------------------

def run_one(adj, sigma, seed):
    net = Mem4Network(adjacency_matrix=adj.copy(), heretic_ratio=0.15,
                      seed=seed, coupling_norm='degree_linear')
    n_nodes = net.N
    sigma_vec = np.full(n_nodes, sigma) if sigma > 0 else None

    v_snaps   = []
    v_history = []
    for step in range(STEPS):
        net.step(I_stimulus=I_STIM, sigma_v_vec=sigma_vec)
        if step >= WARM_UP:
            v_snaps.append(net.v.copy())
            if step % 5 == 0:
                v_history.append(net.v.copy())

    v_s = np.array(v_snaps)
    v_h = np.array(v_history)

    h_cog  = float(np.mean([calculate_cognitive_entropy(v)  for v in v_s[::10]]))
    h_cont = float(np.mean([calculate_continuous_entropy(v) for v in v_s[::10]]))
    lz     = calculate_temporal_lz_complexity(v_h) if len(v_h) > 1 else 0.0
    sync   = calculate_pairwise_synchrony(v_s)
    return h_cog, h_cont, lz, sync


# -- Main ---------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 80)
    print("Item 10 -- Stochastic Resonance x Topology")
    print(f"sigma sweep : {SIGMAS}")
    print(f"I_stim={I_STIM} | steps={STEPS} | warm_up={WARM_UP} | seeds={SEEDS}")
    print("=" * 80)

    t0   = time.time()
    rows = []

    # Build topology list once (seed=42 for structure, rerun per sim-seed)
    topo_list = get_topologies(seed=42)

    # Print lambda2 per topo
    print("\nTopologies et lambda2 (seed=42 pour la structure) :")
    for t in topo_list:
        print(f"  {t['name']:20s}  lambda2={t['lambda2']:.4f}")
    print()

    # Main sweep
    for topo in topo_list:
        name = topo['name']
        lam2 = topo['lambda2']
        print(f"\nTopologie : {name}  (lambda2={lam2:.4f})")
        print(f"  {'sigma':>7}  {'H_cog':>7}  {'H_cont':>7}  {'LZ':>7}  {'sync':>7}")

        best_hcog = -1.0
        best_sigma = SIGMAS[0]
        sigma_hcog = []

        for sigma in SIGMAS:
            hcog_l, hcont_l, lz_l, sy_l = [], [], [], []
            for seed in SEEDS:
                # Rebuild adjacency with same seed for structure reproducibility
                # but use each sim-seed for dynamics
                adj_s = get_topologies(seed=42)[topo_list.index(topo)]['adj']
                hcog, hcont, lz, sy = run_one(adj_s, sigma, seed)
                hcog_l.append(hcog); hcont_l.append(hcont)
                lz_l.append(lz);    sy_l.append(sy)

            hcog_m  = np.mean(hcog_l);  hcog_s  = np.std(hcog_l)
            hcont_m = np.mean(hcont_l); lz_m    = np.mean(lz_l)
            sy_m    = np.mean(sy_l)

            flag = ""
            if hcog_m > best_hcog:
                best_hcog  = hcog_m
                best_sigma = sigma
                flag = " <-- sigma*"

            print(f"  {sigma:>7.3f}  {hcog_m:>7.4f}  {hcont_m:>7.4f}  {lz_m:>7.4f}  {sy_m:>7.4f}{flag}")
            sigma_hcog.append(hcog_m)

            rows.append({
                'topo':         name,
                'lambda2':      lam2,
                'sigma':        sigma,
                'h_cog_mean':   hcog_m,   'h_cog_std':   hcog_s,
                'h_cont_mean':  hcont_m,  'h_cont_std':  np.std(hcont_l),
                'lz_mean':      lz_m,     'lz_std':      np.std(lz_l),
                'sync_mean':    sy_m,     'sync_std':     np.std(sy_l),
            })

        print(f"  --> sigma* = {best_sigma:.3f}  H_cog(sigma*) = {best_hcog:.4f}")

    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed:.1f}s")

    # -- CSV ------------------------------------------------------------------
    import csv, pathlib
    fig_dir = pathlib.Path(__file__).resolve().parents[1] / 'figures'
    fig_dir.mkdir(exist_ok=True)
    csv_path = fig_dir / 'p2_stochastic_resonance_topology.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV : {csv_path}")

    # -- sigma* vs lambda2 summary ------------------------------------------
    print("\nsigma* vs lambda2 (argmax H_cog par topo) :")
    print(f"  {'topo':20s}  {'lambda2':>8}  {'sigma*':>8}  {'H_cog*':>8}")
    # Collect sigma* per topo
    topo_names = list({r['topo'] for r in rows})
    sr_summary = []
    for name in sorted(topo_names, key=lambda n: next(r['lambda2'] for r in rows if r['topo']==n)):
        topo_rows = [r for r in rows if r['topo'] == name]
        best = max(topo_rows, key=lambda r: r['h_cog_mean'])
        sr_summary.append({'topo': name, 'lambda2': best['lambda2'],
                           'sigma_star': best['sigma'], 'h_cog_star': best['h_cog_mean']})
        print(f"  {name:20s}  {best['lambda2']:>8.4f}  {best['sigma']:>8.3f}  {best['h_cog_mean']:>8.4f}")

    # Pearson correlation sigma* ~ lambda2
    l2_arr = np.array([s['lambda2']   for s in sr_summary])
    ss_arr = np.array([s['sigma_star'] for s in sr_summary])
    if np.std(l2_arr) > 1e-9 and np.std(ss_arr) > 1e-9:
        r_sl = float(np.corrcoef(l2_arr, ss_arr)[0, 1])
        print(f"\nCorrelation Pearson(sigma*, lambda2) = {r_sl:+.4f}")
    else:
        r_sl = float('nan')
        print("\nCorrelation Pearson(sigma*, lambda2) = N/A (variance nulle)")

    # -- Figure ---------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(16, 10))

        # Panel A: H_cog(sigma) curves per topology (2 subplots: BA + others)
        ax1 = fig.add_subplot(2, 3, 1)
        ax2 = fig.add_subplot(2, 3, 2)
        ax3 = fig.add_subplot(2, 3, 3)
        ax4 = fig.add_subplot(2, 1, 2)

        cmap_topos = plt.cm.viridis
        all_topo_names = sorted({r['topo'] for r in rows},
                                 key=lambda n: next(r['lambda2'] for r in rows if r['topo']==n))
        colors = {n: cmap_topos(i / max(len(all_topo_names)-1, 1))
                  for i, n in enumerate(all_topo_names)}

        for name in all_topo_names:
            topo_rows = sorted([r for r in rows if r['topo'] == name], key=lambda r: r['sigma'])
            xs  = [r['sigma']      for r in topo_rows]
            ys  = [r['h_cog_mean'] for r in topo_rows]
            es  = [r['h_cog_std']  for r in topo_rows]
            lam2 = topo_rows[0]['lambda2']
            lbl  = f"{name} (l2={lam2:.3f})"
            ax1.errorbar(xs, ys, yerr=es, marker='o', label=lbl,
                         color=colors[name], capsize=3, linewidth=1.5)
            ax2.errorbar(xs, [r['lz_mean'] for r in topo_rows],
                         yerr=[r['lz_std'] for r in topo_rows],
                         marker='s', label=lbl, color=colors[name], capsize=3, linewidth=1.5)
            ax3.errorbar(xs, [r['sync_mean'] for r in topo_rows],
                         yerr=[r['sync_std'] for r in topo_rows],
                         marker='^', label=lbl, color=colors[name], capsize=3, linewidth=1.5)

        for ax, title, ylabel in [
            (ax1, 'H_cog (cognitive entropy)', 'H_cog (bits)'),
            (ax2, 'LZ complexity',             'LZ (normalized)'),
            (ax3, 'Pairwise synchrony',         'sync'),
        ]:
            ax.set_xlabel('sigma_noise')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend(fontsize=6, ncol=1)
            ax.grid(True, alpha=0.3)

        # Panel B: sigma* vs lambda2 scatter
        l2s = [s['lambda2']   for s in sr_summary]
        sss = [s['sigma_star'] for s in sr_summary]
        ax4.scatter(l2s, sss, s=120, c=[colors[s['topo']] for s in sr_summary],
                    zorder=3, edgecolors='k', linewidths=0.5)
        for s in sr_summary:
            ax4.annotate(s['topo'], (s['lambda2'], s['sigma_star']),
                         textcoords='offset points', xytext=(5, 3), fontsize=7)
        if not np.isnan(r_sl):
            # Regression line
            x_fit = np.linspace(min(l2s), max(l2s), 50)
            p     = np.polyfit(l2s, sss, 1)
            ax4.plot(x_fit, np.polyval(p, x_fit), 'r--', linewidth=1.5,
                     label=f'r={r_sl:+.3f}')
            ax4.legend(fontsize=9)
        ax4.set_xlabel('lambda2 (algebraic connectivity)')
        ax4.set_ylabel('sigma* (optimal noise)')
        ax4.set_title(f'sigma* vs lambda2  |  Pearson r={r_sl:+.3f}')
        ax4.grid(True, alpha=0.3)

        fig.suptitle(
            f'Item 10 -- Stochastic Resonance x Topology\n'
            f'I_stim={I_STIM}, coupling=degree_linear, {len(SEEDS)} seeds',
            fontsize=11
        )
        plt.tight_layout()
        png_path = fig_dir / 'p2_stochastic_resonance_topology.png'
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        print(f"Figure : {png_path}")
    except Exception as e:
        print(f"[matplotlib error] {e}")
