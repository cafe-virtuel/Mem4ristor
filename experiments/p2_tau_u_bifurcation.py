#!/usr/bin/env python3
"""
Piste A2 -- Bifurcation tau_u (Dynamique Temporelle du Doute) (2026-04-24)

Hypothese : La constante de temps du doute tau_u controle une bifurcation
entre un regime de "frustration figee" (tau_u petit, u se met a jour vite,
noeuds bloques en opposition statique) et un regime de "chimere respirante"
(tau_u grand, u evolue lentement, clusters de consensus qui se font et
se defont dynamiquement, oscillations a frequence caracteristique).

Un pic de frequence dans le spectre de Fourier moyen de v(t) devrait
emerger a une valeur critique tau_u*.

Topologies testees : Lattice 10x10, BA m=3 N=100.
Regime : I_stim=0.5 (force) pour activer les heretiques.
Sweep : tau_u in [0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100].

Metriques :
  - calculate_pairwise_synchrony (coordination inter-noeuds)
  - H_cont (100-bin, diversite spatiale)
  - H_cog (5-bin KIMI)
  - Frequence dominante du spectre de Fourier moyen de v(t)
  - Puissance spectrale au pic (indicateur d'oscillations collectives)

Script  : experiments/p2_tau_u_bifurcation.py
Figures : figures/p2_tau_u_bifurcation.png
CSV     : figures/p2_tau_u_bifurcation.csv

Reference : PROJECT_STATUS.md §P2-AUDIT Piste A2
"""
import sys, os, time
import numpy as np

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
SEEDS        = [42, 123, 777, 456, 999]
I_STIM       = 0.5     # regime force : heretiques actifs
STEPS        = 4000    # besoin de long pour capturer les oscillations lentes
WARM_UP      = 1000    # ignore le transitoire
DT           = 0.05    # dt standard

TOPOLOGIES = {
    'lattice': {'size': 10, 'adj': None},
    'ba_m3':   {'size': None, 'adj': 'generate'},
}




def dominant_frequency(v_history, dt):
    """
    Compute the dominant oscillation frequency from the mean power spectrum
    of v(t) across all nodes.

    v_history : (T, N) array
    Returns : (freq_hz, peak_power_normalized)
    """
    T, N = v_history.shape
    if T < 4:
        return 0.0, 0.0
    # Mean signal across nodes
    mean_v = v_history.mean(axis=1)   # (T,)
    # Detrend
    mean_v -= mean_v.mean()
    # FFT
    fft_vals = np.fft.rfft(mean_v)
    power    = np.abs(fft_vals) ** 2
    freqs    = np.fft.rfftfreq(T, d=dt)
    # Exclude DC (freq=0)
    power[0] = 0.0
    peak_idx   = np.argmax(power)
    peak_freq  = freqs[peak_idx]
    peak_power = power[peak_idx] / (power.sum() + 1e-12)
    return float(peak_freq), float(peak_power)


def run_one(topo_name, tau_u, seed):
    if topo_name == 'lattice':
        net = Mem4Network(size=10, heretic_ratio=0.15, seed=seed,
                          coupling_norm='degree_linear')
    else:  # ba_m3
        adj = make_ba(100, 3, seed)
        net = Mem4Network(adjacency_matrix=adj, heretic_ratio=0.15, seed=seed,
                          coupling_norm='degree_linear')

    # Override tau_u after construction
    net.model.cfg['doubt']['tau_u'] = tau_u

    v_history = []
    for step in range(STEPS):
        net.step(I_stimulus=I_STIM)
        if step >= WARM_UP:
            v_history.append(net.v.copy())

    v_arr = np.array(v_history)   # (T_tail, N)

    h_cont = float(np.mean([calculate_continuous_entropy(v) for v in v_arr[::5]]))
    h_cog  = float(np.mean([calculate_cognitive_entropy(v) for v in v_arr[::5]]))
    sync   = calculate_pairwise_synchrony(v_arr)
    freq, peak_power = dominant_frequency(v_arr, DT)

    return h_cont, h_cog, sync, freq, peak_power


# -- Main ---------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 80)
    print("Piste A2 -- Bifurcation tau_u")
    print(f"tau_u: {TAU_U_VALUES}")
    print(f"I_stim={I_STIM} | steps={STEPS} | warm_up={WARM_UP} | seeds={SEEDS}")
    print("=" * 80)

    t0   = time.time()
    rows = []

    for topo in ['lattice', 'ba_m3']:
        print(f"\nTopologie : {topo}")
        print(f"  {'tau_u':>7}  {'H_cont':>7}  {'H_cog':>7}  {'sync':>7}  {'f_dom(Hz)':>10}  {'peak_pwr':>9}")
        for tau_u in TAU_U_VALUES:
            h_c_l, h_k_l, sy_l, fr_l, pp_l = [], [], [], [], []
            for seed in SEEDS:
                h_c, h_k, sy, fr, pp = run_one(topo, tau_u, seed)
                h_c_l.append(h_c); h_k_l.append(h_k); sy_l.append(sy)
                fr_l.append(fr);   pp_l.append(pp)
            h_c_m = np.mean(h_c_l);  h_k_m = np.mean(h_k_l)
            sy_m  = np.mean(sy_l);   fr_m  = np.mean(fr_l)
            pp_m  = np.mean(pp_l)
            flag = " <-- PEAK" if pp_m > 0.10 else ""
            print(f"  {tau_u:>7.2f}  {h_c_m:>7.4f}  {h_k_m:>7.4f}  {sy_m:>7.4f}"
                  f"  {fr_m:>10.5f}  {pp_m:>9.4f}{flag}")
            rows.append({
                'topo': topo, 'tau_u': tau_u,
                'h_cont_mean': h_c_m, 'h_cont_std': np.std(h_c_l),
                'h_cog_mean':  h_k_m, 'h_cog_std':  np.std(h_k_l),
                'sync_mean':   sy_m,  'sync_std':    np.std(sy_l),
                'freq_dom':    fr_m,  'freq_std':    np.std(fr_l),
                'peak_power':  pp_m,  'peak_pwr_std': np.std(pp_l),
            })

    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed:.1f}s")

    # -- CSV ------------------------------------------------------------------
    import csv, pathlib
    fig_dir = pathlib.Path(__file__).resolve().parents[1] / 'figures'
    fig_dir.mkdir(exist_ok=True)
    csv_path = fig_dir / 'p2_tau_u_bifurcation.csv'
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
            ('h_cog_mean',   'H_cog (5-bin KIMI)',  'bits'),
            ('sync_mean',    'Pairwise Synchrony',   'correlation'),
            ('freq_dom',     'Dominant freq f_dom',  'Hz (simulation units)'),
            ('peak_power',   'Spectral peak power',  'fraction of total power'),
        ]

        for ax, (key, title, ylabel) in zip(axes.flat, metrics):
            for topo in topos:
                xs = [r['tau_u'] for r in rows if r['topo'] == topo]
                ys = [r[key]     for r in rows if r['topo'] == topo]
                ys_std = [r.get(key.replace('mean', 'std'), r.get(key+'_std', 0))
                          for r in rows if r['topo'] == topo]
                ax.errorbar(xs, ys, yerr=ys_std, marker='o', label=labels[topo],
                            color=colors[topo], capsize=3)
            ax.set_xscale('log')
            ax.set_xlabel('tau_u')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)

        fig.suptitle(
            f'Piste A2 -- tau_u Bifurcation\n'
            f'I_stim={I_STIM}, coupling=degree_linear, {len(SEEDS)} seeds',
            fontsize=11
        )
        plt.tight_layout()
        png_path = fig_dir / 'p2_tau_u_bifurcation.png'
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        print(f"Figure : {png_path}")
    except Exception as e:
        print(f"[matplotlib error] {e}")
