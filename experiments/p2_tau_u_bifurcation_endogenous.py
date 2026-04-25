#!/usr/bin/env python3
"""
Piste D — Bifurcation tau_u en Regime ENDOGENE (I_STIM = 0.0) (2026-04-25)

Contexte : L'audit Manus (§1.5 version modifiee) affirmait que
p2_tau_u_bifurcation.py utilisait I_STIM=0.0 (ERREUR : il utilise 0.5).
Cette experience realise la vraie bifurcation endogene que Manus decrivait.

Hypothese : La bifurcation tau_u documentee en §3quatervigies (sous I_STIM=0.5)
est-elle aussi presente en regime PUREMENT ENDOGENE (I_STIM=0.0, heretiques
inactifs) ? Si oui, la dynamique adaptative de u structure le reseau sans
aucun forçage externe — claim plus fort pour Paper 2.

Methode : Sweep identique a p2_tau_u_bifurcation.py mais I_STIM = 0.0.
Memes topologies (Lattice 10x10, BA m=3 N=100), meme sweep tau_u.
Metriques : H_cont, H_cog, pairwise_synchrony, FFT dominante.

Script  : experiments/p2_tau_u_bifurcation_endogenous.py
Figures : figures/p2_tau_u_bifurcation_endogenous.png
CSV     : figures/p2_tau_u_bifurcation_endogenous.csv

Reference : PROJECT_STATUS.md §3quatervigies (I_STIM=0.5), §3untrigies §1.5
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
    calculate_pairwise_synchrony,
)

# -- Parametres ---------------------------------------------------------------
TAU_U_VALUES = [0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
SEEDS        = [42, 123, 777]
I_STIM       = 0.0    # ENDOGENE PUR — heretiques mathematiquement inactifs
STEPS        = 4000
WARM_UP      = 1000
N            = 100


def make_lattice(size):
    n   = size * size
    adj = np.zeros((n, n), dtype=float)
    for i in range(size):
        for j in range(size):
            node = i * size + j
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni2, nj2 = (i + di) % size, (j + dj) % size
                adj[node, ni2 * size + nj2] = 1.0
    return adj


def dominant_frequency(v_mean_trace, dt=0.05):
    """Frequence dominante par FFT sur la trace de v_mean(t)."""
    n   = len(v_mean_trace)
    if n < 4:
        return 0.0, 0.0
    fft = np.abs(np.fft.rfft(v_mean_trace - v_mean_trace.mean()))
    freqs = np.fft.rfftfreq(n, d=dt)
    freqs[0] = np.nan   # ignore DC
    idx = np.nanargmax(fft)
    return float(freqs[idx]), float(fft[idx] / n)


def run_one(adj, tau_u, seed):
    net = Mem4Network(adjacency_matrix=adj.copy(), heretic_ratio=0.15,
                      seed=seed, coupling_norm='degree_linear')
    net.model.cfg['doubt']['tau_u'] = tau_u

    v_snaps   = []
    v_history = []
    v_mean_tr = []

    for step in range(STEPS):
        net.step(I_stimulus=I_STIM)
        if step >= WARM_UP:
            v_snaps.append(net.v.copy())
            v_mean_tr.append(float(net.v.mean()))
            if step % 5 == 0:
                v_history.append(net.v.copy())

    v_s = np.array(v_snaps)
    v_h = np.array(v_history)

    h_cog  = float(np.mean([calculate_cognitive_entropy(v)  for v in v_s[::10]]))
    h_cont = float(np.mean([calculate_continuous_entropy(v) for v in v_s[::10]]))
    sync   = calculate_pairwise_synchrony(v_s)
    f_dom, peak_pow = dominant_frequency(np.array(v_mean_tr))

    return h_cog, h_cont, sync, f_dom, peak_pow


# -- Main ---------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 80)
    print("Piste D — Bifurcation tau_u : REGIME ENDOGENE (I_STIM = 0.0)")
    print(f"tau_u sweep : {TAU_U_VALUES}")
    print(f"I_stim={I_STIM} | steps={STEPS} | warm_up={WARM_UP} | seeds={SEEDS}")
    print("ATTENTION : heretiques INACTIFS (flip *= -1 sur vecteur nul)")
    print("=" * 80)

    t0   = time.time()
    rows = []

    topologies = {
        'Lattice_10x10': make_lattice(10),
        'BA_m3':         make_ba(N, 3, seed=42),
    }

    for topo_name, adj in topologies.items():
        print(f"\nTopologie : {topo_name}")
        print(f"  {'tau_u':>7}  {'H_cog':>7}  {'H_cont':>7}  {'sync':>7}  "
              f"{'f_dom':>7}  {'peak_pw':>8}")

        for tau_u in TAU_U_VALUES:
            hcog_l, hcont_l, sync_l, fdom_l, ppow_l = [], [], [], [], []
            for seed in SEEDS:
                hcog, hcont, sync, fdom, ppow = run_one(adj, tau_u, seed)
                hcog_l.append(hcog); hcont_l.append(hcont)
                sync_l.append(sync); fdom_l.append(fdom); ppow_l.append(ppow)

            hcog_m  = np.mean(hcog_l);  hcont_m = np.mean(hcont_l)
            sync_m  = np.mean(sync_l);  fdom_m  = np.mean(fdom_l)
            ppow_m  = np.mean(ppow_l)

            print(f"  {tau_u:>7.2f}  {hcog_m:>7.4f}  {hcont_m:>7.4f}  {sync_m:>7.4f}  "
                  f"{fdom_m:>7.4f}  {ppow_m:>8.4f}")

            rows.append({
                'topo':        topo_name,
                'tau_u':       tau_u,
                'h_cog_mean':  hcog_m,  'h_cog_std':  float(np.std(hcog_l)),
                'h_cont_mean': hcont_m, 'h_cont_std': float(np.std(hcont_l)),
                'sync_mean':   sync_m,  'sync_std':   float(np.std(sync_l)),
                'f_dom_mean':  fdom_m,  'f_dom_std':  float(np.std(fdom_l)),
                'peak_pow_mean': ppow_m,
            })

    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed:.1f}s")

    # -- CSV ------------------------------------------------------------------
    import csv, pathlib
    fig_dir = pathlib.Path(__file__).resolve().parents[1] / 'figures'
    fig_dir.mkdir(exist_ok=True)
    csv_path = fig_dir / 'p2_tau_u_bifurcation_endogenous.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV : {csv_path}")

    # -- Comparaison I_STIM=0.5 vs I_STIM=0.0 --------------------------------
    print("\n--- Comparaison avec I_STIM=0.5 (§3quatervigies) ---")
    print("BA m=3 :")
    ba_rows = [r for r in rows if r['topo'] == 'BA_m3']
    for r in ba_rows:
        regime = ("bifurcation?" if r['h_cog_mean'] > 0.1 else "dead")
        print(f"  tau_u={r['tau_u']:6.1f}  H_cog={r['h_cog_mean']:.4f}  "
              f"sync={r['sync_mean']:.4f}  f_dom={r['f_dom_mean']:.4f}  [{regime}]")

    # -- Figure ---------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 4, figsize=(18, 9))
        topo_names = list(topologies.keys())

        for row_idx, topo_name in enumerate(topo_names):
            tr = [r for r in rows if r['topo'] == topo_name]
            tau_vals = [r['tau_u'] for r in tr]

            for col_idx, (key, label, color) in enumerate([
                ('h_cog_mean',   'H_cog (5-bin)',        'steelblue'),
                ('h_cont_mean',  'H_cont (100-bin)',      'darkorange'),
                ('sync_mean',    'Pairwise Synchrony',    'forestgreen'),
                ('f_dom_mean',   'Dominant Freq (FFT)',   'crimson'),
            ]):
                ax = axes[row_idx][col_idx]
                ys  = [r[key] for r in tr]
                err = [r[key.replace('mean', 'std')] for r in tr
                       if key.replace('mean', 'std') in r]
                ax.errorbar(tau_vals, ys, yerr=err if err else None,
                            marker='o', color=color, linewidth=2, capsize=3)
                ax.set_xscale('log')
                ax.set_xlabel('tau_u')
                ax.set_ylabel(label)
                ax.set_title(f'{topo_name}\n{label}')
                ax.grid(True, alpha=0.3)
                # Mark tau_u=10 (default)
                ax.axvline(10, color='gray', linestyle='--', alpha=0.5, label='default tau_u=10')
                if col_idx == 0:
                    ax.legend(fontsize=7)

        fig.suptitle(
            f'Piste D — Bifurcation tau_u : REGIME ENDOGENE (I_STIM=0.0)\n'
            f'Comparer avec §3quatervigies (I_STIM=0.5) — coupling=degree_linear, {len(SEEDS)} seeds',
            fontsize=11
        )
        plt.tight_layout()
        png_path = fig_dir / 'p2_tau_u_bifurcation_endogenous.png'
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        print(f"Figure : {png_path}")
    except Exception as e:
        print(f"[matplotlib error] {e}")
