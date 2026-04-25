#!/usr/bin/env python3
"""
Piste C — Ablation sigma_social → bruit pur (2026-04-25)

Contexte : Manus §1.4 (audit modifié) argumente que σ_social = |laplacian_v|
pourrait n'être qu'un proxy du bruit couplé — i.e. la variable u serait
simplement pilotée par du bruit coloré spatial, sans information structurelle.

Hypothese : Si σ_social transporte de l'information topologique au-delà d'un
signal d'amplitude équivalent (bruit blanc), alors remplacer σ_social par du
bruit pur à même RMS doit dégrader significativement la dynamique du réseau
(entropie cognitive, synchronie, fréquence dominante).

4 conditions :
  FULL          — σ_social = |laplacian_v| (référence, pas d'override)
  SS_NOISE      — σ_social remplacé par bruit Gaussien à même RMS (mesuré warmup)
  SS_STATIC     — σ_social figé à sa moyenne temporelle par noeud (warmup)
  FROZEN_U      — epsilon_u = 0 (u ne bouge pas du tout, référence d'ablation totale)

Methode : BA m=3 N=100 (topologie principale Paper 2), I_STIM=0.5, 3 seeds.
Metriques : H_cont, H_cog, pairwise_synchrony, f_dom.

Script  : experiments/p2_sigma_social_ablation.py
Figures : figures/p2_sigma_social_ablation.png
CSV     : figures/p2_sigma_social_ablation.csv

Reference : PROJECT_STATUS.md §3untrigies §1.4 (Manus critique σ_social)
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
SEEDS    = [42, 123, 777]
I_STIM   = 0.5
STEPS    = 5000
WARM_UP  = 1000
N        = 100

CONDITIONS = ['FULL', 'SS_NOISE', 'SS_STATIC', 'FROZEN_U']


def dominant_frequency(v_mean_trace, dt=0.05):
    n = len(v_mean_trace)
    if n < 4:
        return 0.0, 0.0
    fft   = np.abs(np.fft.rfft(v_mean_trace - v_mean_trace.mean()))
    freqs = np.fft.rfftfreq(n, d=dt)
    freqs[0] = np.nan
    idx   = np.nanargmax(fft)
    return float(freqs[idx]), float(fft[idx] / n)


def run_one(adj, condition, seed):
    rng_ab = np.random.RandomState(seed + 9000)  # separate RNG for ablation noise

    net = Mem4Network(adjacency_matrix=adj.copy(), heretic_ratio=0.15,
                      seed=seed, coupling_norm='degree_linear')

    if condition == 'FROZEN_U':
        net.model.cfg['doubt']['epsilon_u'] = 0.0

    # ---- Phase 1 : Warm-up (always FULL, to measure sigma_social statistics) -
    ss_warmup_acc = []   # list of per-node sigma_social vectors during warmup

    for step in range(WARM_UP):
        # Run one step normally (FULL), but capture the Laplacian signal
        # We re-compute laplacian_v ourselves to measure sigma_social
        if net.adjacency_matrix is not None:
            lv = -(net.L @ net.v)
            if net.coupling_norm != 'uniform':
                from mem4ristor.topology import Mem4Network as _MN
                D_cfg = net.model.cfg['coupling']['D']
                uniform_D_eff = D_cfg / np.sqrt(net.N)
                scale_factors = (net.node_weights * D_cfg) / uniform_D_eff
                lv = lv * scale_factors
        else:
            lv = np.zeros(net.N)
        ss_warmup_acc.append(np.abs(lv))
        net.step(I_stimulus=I_STIM)

    ss_warmup = np.array(ss_warmup_acc)          # (WARM_UP, N)
    ss_rms_global = float(np.sqrt(np.mean(ss_warmup ** 2)))   # global RMS
    ss_mean_per_node = ss_warmup.mean(axis=0)     # (N,) temporal mean per node

    # ---- Phase 2 : Run post-warmup with condition active --------------------
    v_snaps   = []
    v_mean_tr = []

    for step in range(STEPS):
        # Compute sigma_social_override based on condition
        if condition == 'FULL' or condition == 'FROZEN_U':
            override = None
        elif condition == 'SS_NOISE':
            # Pure Gaussian noise at same global RMS
            override = np.abs(rng_ab.normal(0.0, ss_rms_global, net.N))
        elif condition == 'SS_STATIC':
            # Frozen at warmup temporal mean per node
            override = ss_mean_per_node.copy()
        else:
            override = None

        net.step(I_stimulus=I_STIM, sigma_social_override=override)
        v_snaps.append(net.v.copy())
        v_mean_tr.append(float(net.v.mean()))

    v_s = np.array(v_snaps)

    h_cog  = float(np.mean([calculate_cognitive_entropy(v)  for v in v_s[::10]]))
    h_cont = float(np.mean([calculate_continuous_entropy(v) for v in v_s[::10]]))
    sync   = calculate_pairwise_synchrony(v_s)
    f_dom, peak_pow = dominant_frequency(np.array(v_mean_tr))

    return h_cog, h_cont, sync, f_dom, peak_pow


# -- Main ---------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 80)
    print("Piste C — Ablation sigma_social (Manus §1.4)")
    print(f"I_stim={I_STIM} | steps={STEPS} | warm_up={WARM_UP} | seeds={SEEDS}")
    print(f"Conditions : {CONDITIONS}")
    print("=" * 80)

    t0  = time.time()
    adj = make_ba(N, 3, seed=42)
    rows = []

    for condition in CONDITIONS:
        print(f"\nCondition : {condition}")
        print(f"  {'seed':>6}  {'H_cog':>7}  {'H_cont':>7}  {'sync':>7}  "
              f"{'f_dom':>7}  {'peak_pw':>8}")

        hcog_l, hcont_l, sync_l, fdom_l, ppow_l = [], [], [], [], []
        for seed in SEEDS:
            hcog, hcont, sync, fdom, ppow = run_one(adj, condition, seed)
            hcog_l.append(hcog); hcont_l.append(hcont)
            sync_l.append(sync); fdom_l.append(fdom); ppow_l.append(ppow)
            print(f"  {seed:>6}  {hcog:>7.4f}  {hcont:>7.4f}  {sync:>7.4f}  "
                  f"{fdom:>7.4f}  {ppow:>8.4f}")

        hcog_m  = np.mean(hcog_l);  hcont_m = np.mean(hcont_l)
        sync_m  = np.mean(sync_l);  fdom_m  = np.mean(fdom_l)
        ppow_m  = np.mean(ppow_l)

        print(f"  {'MEAN':>6}  {hcog_m:>7.4f}  {hcont_m:>7.4f}  {sync_m:>7.4f}  "
              f"{fdom_m:>7.4f}  {ppow_m:>8.4f}")

        rows.append({
            'condition':     condition,
            'h_cog_mean':    hcog_m,  'h_cog_std':   float(np.std(hcog_l)),
            'h_cont_mean':   hcont_m, 'h_cont_std':  float(np.std(hcont_l)),
            'sync_mean':     sync_m,  'sync_std':     float(np.std(sync_l)),
            'f_dom_mean':    fdom_m,  'f_dom_std':    float(np.std(fdom_l)),
            'peak_pow_mean': ppow_m,
        })

    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed:.1f}s")

    # -- Interpretation --------------------------------------------------------
    print("\n--- Interprétation §1.4 Manus ---")
    full = next(r for r in rows if r['condition'] == 'FULL')
    for r in rows:
        delta_hcog = r['h_cog_mean'] - full['h_cog_mean']
        delta_sync  = r['sync_mean']  - full['sync_mean']
        verdict = "~FULL (Manus ok)" if abs(delta_hcog) < 0.02 and abs(delta_sync) < 0.02 else (
                  "DEGRADE (information structurelle)" if (delta_hcog < -0.02 or delta_sync < -0.02) else
                  "AMELIORE (bruit benefique)")
        print(f"  {r['condition']:15s}  dH_cog={delta_hcog:+.4f}  dsync={delta_sync:+.4f}  -> {verdict}")

    # -- CSV -------------------------------------------------------------------
    import csv, pathlib
    fig_dir  = pathlib.Path(__file__).resolve().parents[1] / 'figures'
    fig_dir.mkdir(exist_ok=True)
    csv_path = fig_dir / 'p2_sigma_social_ablation.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV : {csv_path}")

    # -- Figure ----------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        metrics = [
            ('h_cog_mean',  'h_cog_std',  'H_cog (5-bin)',        'steelblue'),
            ('h_cont_mean', 'h_cont_std', 'H_cont (100-bin)',      'darkorange'),
            ('sync_mean',   'sync_std',   'Pairwise Synchrony',    'forestgreen'),
            ('f_dom_mean',  'f_dom_std',  'Dominant Freq (FFT)',   'crimson'),
        ]

        fig, axes = plt.subplots(1, 4, figsize=(18, 5))
        x     = np.arange(len(CONDITIONS))
        width = 0.6

        for ax, (key, key_std, label, color) in zip(axes, metrics):
            ys   = [r[key]     for r in rows]
            errs = [r[key_std] for r in rows]
            bars = ax.bar(x, ys, width, yerr=errs, color=color, alpha=0.75,
                          capsize=4, error_kw={'linewidth': 1.5})
            ax.set_xticks(x)
            ax.set_xticklabels(CONDITIONS, rotation=15, ha='right', fontsize=9)
            ax.set_ylabel(label)
            ax.set_title(label)
            ax.grid(axis='y', alpha=0.3)
            # Highlight FULL as reference
            bars[0].set_edgecolor('black')
            bars[0].set_linewidth(2)

        fig.suptitle(
            f'Piste C — Ablation σ_social (BA m=3, I_STIM={I_STIM})\n'
            f'Noir = FULL (référence). Manus §1.4 : σ_social est-il juste du bruit ?',
            fontsize=11
        )
        plt.tight_layout()
        png_path = fig_dir / 'p2_sigma_social_ablation.png'
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        print(f"Figure : {png_path}")
    except Exception as e:
        print(f"[matplotlib error] {e}")
