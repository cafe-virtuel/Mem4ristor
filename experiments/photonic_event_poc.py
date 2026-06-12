#!/usr/bin/env python3
"""
POC Evenement dynamique a travers la chaine GST — Vague 2, etape 2bis — 2026-06-12
Claude Code (Fable) / Julien Chauvin

SUITE DE : photonic_gst_transduction_poc.py (etape 2 : chaine GST OK en regime
STATIONNAIRE — l'inertie ne coute rien quand le stimulus est constant).

QUESTION (la vraie contrainte de bande passante) : un EVENEMENT TRANSITOIRE
survit-il a la traversee du materiau ? On reutilise le protocole du claim [13]
(event_phase_transition.py, commit 667a2a9) : forcing d'un noeud PERIPHERIQUE
(degre minimal) sur BA m=3 -> bifurcation positive dH > 0.

NOTE AUDIT-024 : le claim [13] (dH=+1.20 bits) date d'avril = ANCIEN bruit.
La condition ELEC (deterministe) de ce POC RE-MESURE l'effet avec le code
actuel — c'est la nouvelle reference, le chiffre d'avril n'est pas comparable.

CHAINE OPTIQUE (canal stimulus uniquement, par noeud, par pas) :
  1. Cible : I_target(t) = 0 partout, sauf i_event=1.5 sur le noeud peripherique
     pendant t_event=150 pas (grille du claim [13] : I>=0.8, T>=50).
  2. Photons : k ~ Poisson(LAM_BASE * I_target / I_REF), LAM_BASE=10 photons/pas
     pour I_REF=1.0 (budget valide aux etapes precedentes). I=0 -> 0 photon
     (silence optique exact).
  3. Saturation : T(P) = P(1+s)/(1+sP), s=1 (compression douce realiste).
     A i_event=1.5 le pic est REELLEMENT comprime (~1.36) — c'est voulu.
  4. Inertie : passe-bas 1er ordre, tau_mat in {0, 1, 3, 10, 30, 100, 300} pas.
     Reponse indicielle : fraction transmise = 1 - exp(-t_event/tau_mat).
     Prediction : tau=30 -> 99% ; tau=100 -> 78% ; tau=300 -> 39%.

MESURES :
  - dH = H_cont(post) - H_cont(pre), fenetres de 100 pas (protocole [13] exact).
  - i_eff_max : amplitude max reellement atteinte au noeud force (diagnostic
    mecanique direct, comparable a la prediction analytique ci-dessus).

LECTURE : tau_mat_max ou dH reste comparable a ELEC = bande passante requise
du materiau RELATIVE a la duree de l'evenement -> avec la cinetique du materiau
choisi (GST ~ns-us, WO3 ~ms), fixe ENFIN l'echelle dt_physique et la conversion
Lambda -> watts (PHOTONIC_PATHWAY section 4).

PROTOCOLE : BA m=3 (le claim [13] positif), N=100, heretic_ratio=0.15,
degree_linear, PRE=1000 / EVENT=150 / POST=2000 pas, 10 seeds canoniques.
Conditions : ELEC + GST_tau x7 = 8 x 10 = 80 runs.

Sorties : figures/photonic_event_poc.csv / _agg.csv / .png
(prints ASCII uniquement — console Windows cp1252, regle du 12/06)
"""
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'src'))

from mem4ristor.core import Mem4Network
from mem4ristor.graph_utils import make_ba
from mem4ristor.metrics import calculate_continuous_entropy

SEEDS = [42, 123, 777, 17, 256, 1337, 99, 314, 2024, 888]
N = 100
M_BA = 3
STEPS_PRE = 1000
T_EVENT = 150
STEPS_POST = 2000
WINDOW = 100                  # fenetre H (protocole [13])
I_EVENT = 1.5
I_REF = 1.0
LAM_BASE = 10                 # photons/pas pour I=1.0 (budget valide)
S_SAT = 1.0
TAU_VALUES = [0, 1, 3, 10, 30, 100, 300]
HERETIC = 0.15
COUPLING_NORM = 'degree_linear'


def saturate(P, s):
    if s == 0.0:
        return P
    return P * (1.0 + s) / (1.0 + s * P)


def run_one(adj, target_idx, seed, tau_mat, optical):
    net = Mem4Network(adjacency_matrix=adj.copy(), heretic_ratio=HERETIC,
                      seed=seed, coupling_norm=COUPLING_NORM)
    rng_phot = np.random.RandomState(seed + 77000)
    I_state = np.zeros(N)
    i_eff_max = 0.0

    def transduce(I_target):
        nonlocal I_state, i_eff_max
        if not optical:
            return I_target
        lam_vec = LAM_BASE * np.maximum(I_target, 0.0) / I_REF
        k = rng_phot.poisson(lam_vec)
        P = k / LAM_BASE                      # puissance normalisee (I_REF -> 1)
        I_raw = I_REF * saturate(P, S_SAT)
        if tau_mat == 0:
            I_state = I_raw
        else:
            I_state = I_state + (I_raw - I_state) / tau_mat
        i_eff_max = max(i_eff_max, float(I_state.max()))
        return I_state.copy()

    snap_pre, snap_post = [], []
    zero = np.zeros(N)
    for step in range(STEPS_PRE):
        net.step(I_stimulus=transduce(zero))
        if step >= STEPS_PRE - WINDOW:
            snap_pre.append(net.v.copy())
    I_vec = np.zeros(N)
    I_vec[target_idx] = I_EVENT
    for _ in range(T_EVENT):
        net.step(I_stimulus=transduce(I_vec))
    for step in range(STEPS_POST):
        net.step(I_stimulus=transduce(zero))
        if step >= STEPS_POST - WINDOW:
            snap_post.append(net.v.copy())

    h_pre = float(calculate_continuous_entropy(np.array(snap_pre).flatten()))
    h_post = float(calculate_continuous_entropy(np.array(snap_post).flatten()))
    return {'h_pre': h_pre, 'h_post': h_post, 'delta_h': h_post - h_pre,
            'i_eff_max': i_eff_max if optical else I_EVENT}


def main():
    import csv
    t0 = time.time()
    adj = make_ba(N, M_BA, seed=42)
    degrees = adj.sum(axis=1)
    target_idx = int(np.argmin(degrees))
    print(f"Noeud peripherique : #{target_idx} (degre {int(degrees[target_idx])})")
    print(f"Evenement : I={I_EVENT} pendant {T_EVENT} pas | chaine : "
          f"Poisson({LAM_BASE}/I_REF) -> saturation s={S_SAT:g} -> tau_mat")

    conds = [('ELEC', None, False)] + [(f'GST_tau{t}', t, True) for t in TAU_VALUES]
    rows = []
    for label, tau, optical in conds:
        for seed in SEEDS:
            r = run_one(adj, target_idx, seed, tau if tau is not None else 0, optical)
            rows.append({'condition': label, 'tau_mat': '' if tau is None else tau,
                         'seed': seed, **r})
        sub = [r for r in rows if r['condition'] == label]
        pred = ('' if not optical or tau in (None, 0) else
                f"  (prediction transmission {100*(1-np.exp(-T_EVENT/tau)):.0f}%)")
        print(f"{label:>10s} : dH={np.mean([r['delta_h'] for r in sub]):+.3f}"
              f"+-{np.std([r['delta_h'] for r in sub]):.3f}  "
              f"i_eff_max={np.mean([r['i_eff_max'] for r in sub]):.3f}"
              f"{pred}  [{time.time()-t0:.0f}s]")

    fig_dir = HERE.parent / 'figures'
    raw_path = fig_dir / 'photonic_event_poc.csv'
    with open(raw_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

    agg = []
    for label, tau, optical in conds:
        sub = [r for r in rows if r['condition'] == label]
        agg.append({'condition': label, 'tau_mat': '' if tau is None else tau,
                    'n_seeds': len(sub),
                    'delta_h_mean': float(np.mean([r['delta_h'] for r in sub])),
                    'delta_h_std': float(np.std([r['delta_h'] for r in sub])),
                    'i_eff_max_mean': float(np.mean([r['i_eff_max'] for r in sub])),
                    'transmission_pred': (1.0 if not optical or not tau else
                                          float(1 - np.exp(-T_EVENT / tau)))})
    agg_path = fig_dir / 'photonic_event_poc_agg.csv'
    with open(agg_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=agg[0].keys())
        w.writeheader(); w.writerows(agg)

    ref = next(a for a in agg if a['condition'] == 'ELEC')
    print()
    print("=" * 70)
    print(f"VERDICT - evenement [13] a travers la chaine GST "
          f"(ref ELEC : dH={ref['delta_h_mean']:+.3f})")
    print("=" * 70)
    for a in agg:
        if a['condition'] == 'ELEC':
            continue
        frac = (a['delta_h_mean'] / ref['delta_h_mean']
                if abs(ref['delta_h_mean']) > 1e-9 else float('nan'))
        ok = abs(a['delta_h_mean'] - ref['delta_h_mean']) < max(
            0.3, 2 * ref['delta_h_std'])
        print(f"  tau={a['tau_mat']:>4} : dH={a['delta_h_mean']:+.3f} "
              f"({100*frac:5.1f}% de l'effet)  i_eff={a['i_eff_max_mean']:.3f}  "
              f"-> {'OK' if ok else 'EFFET DEGRADE'}")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        x = [max(t, 0.5) for t in TAU_VALUES]
        ys = [next(a for a in agg if a['condition'] == f'GST_tau{t}')['delta_h_mean']
              for t in TAU_VALUES]
        es = [next(a for a in agg if a['condition'] == f'GST_tau{t}')['delta_h_std']
              for t in TAU_VALUES]
        ax1.errorbar(x, ys, yerr=es, marker='o', color='steelblue', label='chaîne GST')
        ax1.axhline(ref['delta_h_mean'], color='gray', ls='--',
                    label=f"ELEC (dH={ref['delta_h_mean']:+.2f})")
        ax1.axhline(0, color='black', lw=0.5)
        ax1.set_xscale('log'); ax1.set_xlabel('τ_matériau (pas)')
        ax1.set_ylabel('ΔH (bits)'); ax1.grid(alpha=0.3); ax1.legend(fontsize=8)
        ax1.set_title('Effet [13] transmis vs inertie matériau', fontsize=10)

        ie = [next(a for a in agg if a['condition'] == f'GST_tau{t}')['i_eff_max_mean']
              for t in TAU_VALUES]
        pred = [I_EVENT * saturate(I_EVENT / I_REF, S_SAT) / (I_EVENT / I_REF) *
                (1 - np.exp(-T_EVENT / t)) if t > 0 else None for t in TAU_VALUES]
        ax2.plot(x, ie, marker='o', color='crimson', label='i_eff_max mesuré')
        pred_x = [max(t, 0.5) for t in TAU_VALUES if t > 0]
        pred_y = [I_REF * saturate(I_EVENT / I_REF, S_SAT) * (1 - np.exp(-T_EVENT / t))
                  for t in TAU_VALUES if t > 0]
        ax2.plot(pred_x, pred_y, ls=':', color='black',
                 label='prédiction 1-exp(-T/τ) (saturée)')
        ax2.axhline(I_EVENT, color='gray', ls='--', alpha=0.5, label=f'cible {I_EVENT}')
        ax2.set_xscale('log'); ax2.set_xlabel('τ_matériau (pas)')
        ax2.set_ylabel('Amplitude effective au nœud forcé')
        ax2.grid(alpha=0.3); ax2.legend(fontsize=8)
        ax2.set_title('Réponse indicielle du matériau', fontsize=10)

        fig.suptitle(f'Événement [13] (I={I_EVENT}, {T_EVENT} pas, nœud périphérique) '
                     f'à travers la chaîne GST (Λ={LAM_BASE}, s={S_SAT:g})', fontsize=11)
        plt.tight_layout()
        png = fig_dir / 'photonic_event_poc.png'
        plt.savefig(png, dpi=150, bbox_inches='tight')
        print(f"\nFigure : {png}")
    except Exception as e:
        print(f"[matplotlib] {e}")

    print(f"CSV : {raw_path}\n      {agg_path}")
    print(f"Wall time : {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
