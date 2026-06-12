#!/usr/bin/env python3
"""
POC Transduction GST réaliste — Vague 2, étape 2 du PHOTONIC_PATHWAY — 2026-06-12
Claude Code (Fable) / Julien Chauvin

SUITE DE : photonic_transduction_poc.py (stimulus optique idéal : OK à Λ≥10)
           photonic_coupling_poc.py (tout-optique idéal : OK à Λ≥10)

QUESTION : un matériau de transduction RÉALISTE (type GST) — qui SATURE et qui a
une INERTIE — préserve-t-il les régimes ? Quel est le τ_matériau maximal tolérable ?

CHAÎNE DE TRANSDUCTION COMPLÈTE (par nœud, par pas) :
  1. Photons : k ~ Poisson(Λ=10), P = k/Λ          (budget validé par les POC 1-2)
  2. Saturation (absorbeur saturable normalisé) :
         T(P) = P·(1+s)/(1+s·P)
     s=0 -> linéaire (contrôle) ; s=1 -> compression douce ; s=3 -> compression forte.
     Normalisation : T(1)=1 (le point de fonctionnement nominal est préservé,
     c'est la DISTORSION autour de ce point qui est testée — compression des pics,
     dilatation des creux, redressement du bruit).
  3. Inertie du matériau (passe-bas du 1er ordre) :
         I_t = I_{t-1} + (I_nom·T - I_{t-1}) / τ_mat
     τ_mat en PAS de simulation ∈ {0 (instantané), 1, 3, 10, 30, 100}.
  4. Le signe hérétique est appliqué par le réseau (détection différentielle,
     comme POC 1-2). La saturation s'applique à la magnitude optique AVANT
     l'inversion — réaliste : le matériau voit la puissance, pas le signe.

CE QUE DONNE LE RÉSULTAT :
  - s_max tolérable -> contrainte de linéarité du détecteur (spécification).
  - τ_mat_max tolérable -> BANDE PASSANTE requise du matériau RELATIVE à la
    dynamique du réseau. C'est le pont vers le temps physique : si le GST répond
    en τ_GST secondes, alors dt_physique ≥ τ_GST/τ_mat_max, et la conversion
    Λ -> watts devient possible (PHOTONIC_PATHWAY §4).

PROTOCOLE : BA m=3 (fonctionnel) + m=5 (dead zone), N=100, I_nominal=0.5,
heretic_ratio=0.15, degree_linear, WARM_UP=1000 + STEPS=3000, 10 seeds canoniques.
Grille : s ∈ {0, 1, 3} × τ_mat ∈ {0, 1, 3, 10, 30, 100}, Λ=10 fixé,
+ contrôle DETERMINISTIC (électrique idéal). 380 runs.

NOTE D'HONNÊTETÉ : T(P) est un modèle générique d'absorbeur saturable, pas une
courbe GST mesurée. Le jour où une courbe matériau réelle est disponible, elle
remplace T(P) — la chaîne et le protocole restent valables.

Sorties : figures/photonic_gst_transduction_poc.csv / _agg.csv / .png
"""
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'src'))

from mem4ristor.core import Mem4Network
from mem4ristor.graph_utils import make_ba
from mem4ristor.metrics import (
    calculate_cognitive_entropy,
    calculate_continuous_entropy,
    calculate_pairwise_synchrony,
)

SEEDS = [42, 123, 777, 17, 256, 1337, 99, 314, 2024, 888]
M_VALUES = [3, 5]
LAM = 10                      # budget de photons validé par POC 1-2
S_VALUES = [0.0, 1.0, 3.0]    # force de saturation (0 = linéaire)
TAU_VALUES = [0, 1, 3, 10, 30, 100]   # inertie matériau (pas de simulation)
I_NOMINAL = 0.5
N = 100
WARM_UP = 1000
STEPS = 3000
HERETIC = 0.15
COUPLING_NORM = 'degree_linear'


def saturate(P, s):
    """Absorbeur saturable normalisé : T(1)=1, linéaire si s=0."""
    if s == 0.0:
        return P
    return P * (1.0 + s) / (1.0 + s * P)


def run_one(adj, seed, s_sat, tau_mat, deterministic=False):
    net = Mem4Network(adjacency_matrix=adj.copy(), heretic_ratio=HERETIC,
                      seed=seed, coupling_norm=COUPLING_NORM)
    rng_phot = np.random.RandomState(seed + 77000)
    I_state = np.full(N, I_NOMINAL)   # état du matériau (sortie du passe-bas)

    def stimulus():
        nonlocal I_state
        if deterministic:
            return I_NOMINAL
        P = rng_phot.poisson(LAM, N) / LAM
        I_target = I_NOMINAL * saturate(P, s_sat)
        if tau_mat == 0:
            I_state = I_target
        else:
            I_state = I_state + (I_target - I_state) / tau_mat
        return I_state.copy() if tau_mat else I_state

    for _ in range(WARM_UP):
        net.step(I_stimulus=stimulus())
    snaps = []
    for _ in range(STEPS):
        net.step(I_stimulus=stimulus())
        snaps.append(net.v.copy())
    v_s = np.array(snaps)
    return {
        'h_cont': float(np.mean([calculate_continuous_entropy(v) for v in v_s[::10]])),
        'h_cog': float(np.mean([calculate_cognitive_entropy(v) for v in v_s[::10]])),
        'sync': float(calculate_pairwise_synchrony(v_s)),
    }


def main():
    import csv
    t0 = time.time()
    rows = []
    conds = [('DETERMINISTIC', None, None)]
    conds += [(f's{s:g}_tau{t}', s, t) for s in S_VALUES for t in TAU_VALUES]
    total = len(M_VALUES) * len(conds) * len(SEEDS)
    done = 0

    for m in M_VALUES:
        adj = make_ba(N, m, seed=42)
        for label, s_sat, tau in conds:
            for seed in SEEDS:
                r = run_one(adj, seed, s_sat if s_sat is not None else 0.0,
                            tau if tau is not None else 0,
                            deterministic=(label == 'DETERMINISTIC'))
                rows.append({'m': m, 'condition': label,
                             's_sat': '' if s_sat is None else s_sat,
                             'tau_mat': '' if tau is None else tau,
                             'seed': seed, **r})
                done += 1
            sub = [r for r in rows if r['m'] == m and r['condition'] == label]
            print(f"m={m} {label:>14s} : "
                  f"H_cont={np.mean([r['h_cont'] for r in sub]):.3f}"
                  f"±{np.std([r['h_cont'] for r in sub]):.3f}  "
                  f"H_cog={np.mean([r['h_cog'] for r in sub]):.4f}  "
                  f"sync={np.mean([r['sync'] for r in sub]):+.4f}  "
                  f"[{done}/{total}, {time.time()-t0:.0f}s]")

    fig_dir = HERE.parent / 'figures'
    raw_path = fig_dir / 'photonic_gst_transduction_poc.csv'
    with open(raw_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

    agg = []
    for m in M_VALUES:
        for label, s_sat, tau in conds:
            sub = [r for r in rows if r['m'] == m and r['condition'] == label]
            agg.append({
                'm': m, 'condition': label,
                's_sat': '' if s_sat is None else s_sat,
                'tau_mat': '' if tau is None else tau,
                'n_seeds': len(sub),
                'h_cont_mean': float(np.mean([r['h_cont'] for r in sub])),
                'h_cont_std': float(np.std([r['h_cont'] for r in sub])),
                'h_cog_mean': float(np.mean([r['h_cog'] for r in sub])),
                'sync_mean': float(np.mean([r['sync'] for r in sub])),
            })
    agg_path = fig_dir / 'photonic_gst_transduction_poc_agg.csv'
    with open(agg_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=agg[0].keys())
        w.writeheader(); w.writerows(agg)

    print("\n" + "=" * 80)
    # ASCII uniquement dans les prints (console Windows cp1252 — cf. limit02)
    print(f"VERDICT - chaine GST complete (Poisson Lambda={LAM} -> saturation s -> inertie tau)")
    print("=" * 80)
    for m in M_VALUES:
        ref = next(a for a in agg if a['m'] == m and a['condition'] == 'DETERMINISTIC')
        regime = 'FONCTIONNEL' if m == 3 else 'DEAD ZONE'
        print(f"\nBA m={m} ({regime}) - ref : H_cont={ref['h_cont_mean']:.3f} "
              f"H_cog={ref['h_cog_mean']:.4f} sync={ref['sync_mean']:+.4f}")
        for s in S_VALUES:
            print(f"  [saturation s={s:g}]")
            for tau in TAU_VALUES:
                a = next(x for x in agg if x['m'] == m and x['condition'] == f's{s:g}_tau{tau}')
                d_hcont = a['h_cont_mean'] - ref['h_cont_mean']
                d_hcog = a['h_cog_mean'] - ref['h_cog_mean']
                d_sync = a['sync_mean'] - ref['sync_mean']
                ok = abs(d_hcont) < 0.15 and abs(d_sync) < 0.05 and abs(d_hcog) < 0.1
                print(f"    tau={tau:>3d} : dH_cont={d_hcont:+.3f}  dH_cog={d_hcog:+.4f}  "
                      f"dsync={d_sync:+.4f}  -> {'OK' if ok else 'DEVIATION'}")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        x_tau = [max(t, 0.5) for t in TAU_VALUES]   # log-x, tau=0 affiché à 0.5
        for row_i, m in enumerate(M_VALUES):
            ref = next(a for a in agg if a['m'] == m and a['condition'] == 'DETERMINISTIC')
            for ax, key, label in zip(axes[row_i],
                                      ('h_cont_mean', 'h_cog_mean', 'sync_mean'),
                                      ('H_cont (bits)', 'H_cog (bits)', 'Pairwise sync')):
                for s, color in zip(S_VALUES, ('forestgreen', 'darkorange', 'crimson')):
                    ys = [next(a for a in agg if a['m'] == m and a['condition'] == f's{s:g}_tau{t}')[key]
                          for t in TAU_VALUES]
                    ax.plot(x_tau, ys, marker='o', color=color, label=f's={s:g}')
                ax.axhline(ref[key], color='gray', ls='--', alpha=0.6)
                ax.set_xscale('log')
                ax.set_title(f"BA m={m} — {label}", fontsize=9)
                ax.set_xlabel('τ_matériau (pas)')
                ax.grid(alpha=0.3)
        axes[0][0].legend(fontsize=8, title='saturation')
        fig.suptitle(f'Chaîne de transduction GST réaliste (Λ={LAM} photons/nœud/pas)\n'
                     'pointillés gris = contrôle électrique idéal', fontsize=11)
        plt.tight_layout()
        png = fig_dir / 'photonic_gst_transduction_poc.png'
        plt.savefig(png, dpi=150, bbox_inches='tight')
        print(f"\nFigure : {png}")
    except Exception as e:
        print(f"[matplotlib] {e}")

    print(f"CSV : {raw_path}\n      {agg_path}")
    print(f"Wall time : {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
