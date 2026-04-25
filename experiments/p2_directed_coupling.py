#!/usr/bin/env python3
"""
Piste A3 -- Couplage Asymetrique et Graphes Diriges (2026-04-24)

Hypothese : La strangulation par les hubs (dead zone BA m>=5) est
exacerbee par la symetrie du couplage : hub influence la peripherie
ET peripherie influence le hub avec la meme force. Un reseau dirige
ou les hubs "parlent mais n'ecoutent pas" (HUB_BCAST) ou "ecoutent
mais ne parlent pas" (HUB_LISTEN) modifie le mecanisme de strangulation
sans necessiter de normalisation ad-hoc.

Trois types de couplage directionnel :
  SYMM       : BA m=5 non-dirige (baseline dead zone)
  HUB_BCAST  : A[peripheral, hub] = 1  (hub -> peripherie)
               Le hub n'ecoute personne, la peripherie suit le hub.
  HUB_LISTEN : A[hub, peripheral] = 1  (peripherie -> hub)
               Le hub ecoute tout le monde, la peripherie evolue librement.

Chaque mode est teste avec coupling_norm in {'uniform', 'degree_linear'}.

Le Laplacien dirige est construit nativement par _rebuild_laplacian()
avec la matrice asymetrique : l_v[i] = sum_j A[i,j]*v[j] - D_in(i)*v[i].

Metriques : H_cont, H_cog, pairwise_synchrony.
Sweep : m=5 (dead zone), N=100, I_stim=0.0 (endogene), 3 seeds, 3000 steps.
Comparaison supplementaire m=3 (reference hors dead zone).

Script  : experiments/p2_directed_coupling.py
Figures : figures/p2_directed_coupling.png
CSV     : figures/p2_directed_coupling.csv

Reference : PROJECT_STATUS.md §P2-AUDIT Piste A3
"""
import sys, os, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from mem4ristor.core import Mem4Network
from mem4ristor.metrics import (
    calculate_cognitive_entropy,
    calculate_continuous_entropy,
    calculate_pairwise_synchrony,
)

# -- Parametres ---------------------------------------------------------------
N        = 100
STEPS    = 3000
WARM_UP  = int(STEPS * 0.25)
SEEDS    = [42, 123, 777]
I_STIM   = 0.0    # endogene (les resultats SPICE suggerent que la dead zone est la)
NORMS    = ['uniform', 'degree_linear']
M_VALS   = [3, 5]  # m=3 reference, m=5 dead zone


def make_ba_undirected(n, m, seed):
    """BA non-dirige standard."""
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


def make_ba_directed(n, m, seed, mode):
    """
    BA dirige.

    mode='HUB_BCAST' : A[new_node, hub] = 1
        => new_node ecoute hub (hub diffuse vers la peripherie).
        Laplacien : l_v[new_node] = sum de ce que hub dit - d_in*v[new_node]
        Hub : d_in ~ 0, evolue librement.

    mode='HUB_LISTEN' : A[hub, new_node] = 1
        => hub ecoute new_node (peripherie parle au hub).
        Laplacien : l_v[hub] = grand (ecoute tout) - d_in*v[hub]
        Peripherie : d_in ~ 0, evolue librement.
    """
    rng = np.random.RandomState(seed)
    adj = np.zeros((n, n), dtype=float)
    # Seed complet : bidirectionnel pour les m+1 premiers noeuds
    for i in range(m + 1):
        for j in range(i + 1, m + 1):
            adj[i, j] = adj[j, i] = 1.0
    degrees = adj.sum(axis=1)  # degre total pour la probabilite d'attachement PA
    for new_node in range(m + 1, n):
        probs = degrees[:new_node] / degrees[:new_node].sum()
        targets = rng.choice(new_node, size=m, replace=False, p=probs)
        for hub in targets:
            if mode == 'HUB_BCAST':
                # new_node ecoute hub : A[new_node, hub] = 1
                adj[new_node, hub] = 1.0
            else:  # HUB_LISTEN
                # hub ecoute new_node : A[hub, new_node] = 1
                adj[hub, new_node] = 1.0
        # Pour le PA on utilise le degre total (in+out) de la version sym
        # pour conserver la meme probabilite d'attachement preferentiel
        degrees[new_node] += m
        for hub in targets:
            degrees[hub] += 1
    return adj


def run_one(adj, norm, seed, steps=STEPS, warm_up=WARM_UP):
    net = Mem4Network(
        adjacency_matrix=adj.copy(),
        heretic_ratio=0.15,
        coupling_norm=norm,
        seed=seed,
    )
    v_snapshots = []
    v_history   = []
    for step in range(steps):
        net.step(I_stimulus=I_STIM)
        if step >= warm_up:
            if step % 10 == 0:
                v_snapshots.append(net.v.copy())
            v_history.append(net.v.copy())

    v_snaps = np.array(v_snapshots)
    v_hist  = np.array(v_history)

    h_cont = float(np.mean([calculate_continuous_entropy(v) for v in v_snaps]))
    h_cog  = float(np.mean([calculate_cognitive_entropy(v)  for v in v_snaps]))
    sync   = calculate_pairwise_synchrony(v_hist)
    return h_cont, h_cog, sync


# -- Main ---------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 88)
    print("Piste A3 -- Couplage Asymetrique et Graphes Diriges")
    print(f"N={N} | I_stim={I_STIM} | steps={STEPS} | seeds={SEEDS}")
    print(f"Modes : SYMM, HUB_BCAST, HUB_LISTEN  x  norms : {NORMS}")
    print("=" * 88)

    t0   = time.time()
    rows = []

    GRAPH_MODES = ['SYMM', 'HUB_BCAST', 'HUB_LISTEN']

    for m in M_VALS:
        label = f"BA m={m} {'(dead zone ref)' if m==5 else '(hors dead zone ref)'}"
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"  {'Mode':<12}  {'Norm':<14}  {'H_cont':>7}  {'H_cog':>7}  {'sync':>7}")
        print(f"  {'-'*58}")
        for gmode in GRAPH_MODES:
            for norm in NORMS:
                h_c_l, h_k_l, sy_l = [], [], []
                for seed in SEEDS:
                    if gmode == 'SYMM':
                        adj = make_ba_undirected(N, m, seed)
                    else:
                        adj = make_ba_directed(N, m, seed, gmode)
                    h_c, h_k, sy = run_one(adj, norm, seed)
                    h_c_l.append(h_c); h_k_l.append(h_k); sy_l.append(sy)
                h_c_m = np.mean(h_c_l); h_k_m = np.mean(h_k_l); sy_m = np.mean(sy_l)
                flag = " ***" if h_k_m > 0.3 else ""
                print(f"  {gmode:<12}  {norm:<14}  {h_c_m:>7.4f}  {h_k_m:>7.4f}  {sy_m:>7.4f}{flag}")
                rows.append({
                    'm': m, 'graph_mode': gmode, 'norm': norm,
                    'h_cont_mean': h_c_m, 'h_cont_std': np.std(h_c_l),
                    'h_cog_mean':  h_k_m, 'h_cog_std':  np.std(h_k_l),
                    'sync_mean':   sy_m,  'sync_std':    np.std(sy_l),
                })

    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed:.1f}s")

    # -- CSV ------------------------------------------------------------------
    import csv, pathlib
    fig_dir = pathlib.Path(__file__).resolve().parents[1] / 'figures'
    fig_dir.mkdir(exist_ok=True)
    csv_path = fig_dir / 'p2_directed_coupling.csv'
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

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        x = np.arange(len(GRAPH_MODES))
        width = 0.35
        norm_colors = {'uniform': 'steelblue', 'degree_linear': 'darkorange'}

        for ax, m in zip(axes, M_VALS):
            for ni, norm in enumerate(NORMS):
                vals = [
                    next(r['h_cog_mean'] for r in rows
                         if r['m']==m and r['graph_mode']==gm and r['norm']==norm)
                    for gm in GRAPH_MODES
                ]
                errs = [
                    next(r['h_cog_std'] for r in rows
                         if r['m']==m and r['graph_mode']==gm and r['norm']==norm)
                    for gm in GRAPH_MODES
                ]
                offset = (ni - 0.5) * width
                ax.bar(x + offset, vals, width, yerr=errs, label=norm,
                       color=norm_colors[norm], alpha=0.85, capsize=4)

            ax.set_xticks(x)
            ax.set_xticklabels(GRAPH_MODES, fontsize=10)
            ax.set_ylabel('H_cog (5-bin KIMI, bits)')
            dead = '(dead zone)' if m == 5 else '(hors dead zone)'
            ax.set_title(f'BA m={m} {dead}')
            ax.axhline(0.3, color='red', linestyle='--', alpha=0.5, label='seuil diversite')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

        fig.suptitle(
            'Piste A3 -- Directed Coupling\n'
            f'N={N}, I_stim={I_STIM}, heretic_ratio=0.15, {len(SEEDS)} seeds',
            fontsize=11
        )
        plt.tight_layout()
        png_path = fig_dir / 'p2_directed_coupling.png'
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        print(f"Figure : {png_path}")
    except Exception as e:
        print(f"[matplotlib error] {e}")
