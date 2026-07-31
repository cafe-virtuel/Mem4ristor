#!/usr/bin/env python3
"""
B6 — LE SEUIL DE GAIN DU CAPTEUR DE DESACCORD — 2026-07-31
Claude Code (Opus 5) / Julien Chauvin.

POURQUOI CE SCRIPT EXISTE
-------------------------
La prediction falsifiable B6 (docs/FUTURE_WORK.md) est la seule affirmation du
projet qui sorte de la simulation : un laboratoire peut la tester sur de vrais
STNO couples, par spectroscopie micro-onde standard (methode Romera et al. 2018).
Elle portait une reserve non levee depuis le 09/07 :

    « au capteur brut (non calibre), l'effet est nul — la prediction suppose
      qu'un vrai circuit de detection aurait un gain suffisant, hypothese non
      verifiee. »

Cette reserve etait FLOUE (« un gain suffisant » : combien ?) et, telle qu'elle
etait redigee, elle accusait le mauvais coupable. Ce script la remplace par une
SPECIFICATION CHIFFREE.

DEUX CORRECTIONS PREALABLES, etablies par LECTURE, sans aucune simulation
------------------------------------------------------------------------
1. La phrase de FUTURE_WORK generalisait a tort. Au capteur brut, l'effet est nul
   dans UN modele sur trois (§8, celui-ci), et REEL dans les deux autres :
   §7 Kuramoto d=+2.28/+1.05, §9 macrospin LLGS d=+2.42/+1.61, aucun IC ne
   chevauchant zero. Corrige le 31/07 dans FUTURE_WORK et dans le BILAN.
2. LA NON-ISOCHRONICITE EST DISCULPEE, et la preuve dormait dans le CSV existant :
   a n_nonlin=0 — c'est-a-dire AUCUNE non-isochronicite, modele isochrone —
   l'effet est DEJA nul au capteur brut (diff +0.0076 BA / +0.0036 lattice, les
   deux IC chevauchant zero). Elle ne peut donc pas etre la cause.
   Et `u` au capteur brut ne depend pas d'elle : 0.0608 / 0.0609 / 0.0599 pour
   n_nonlin = 0 / 3 / 10. Variation < 0.0011.

LA VRAIE CAUSE, ET ELLE EST D'ECHELLE
-------------------------------------
Dans b2_stno_amplitude_phase_poc.py, la chaine est :
    sigma_social_for_u = |S| * gain_u                        (L110)
    du ∝ (K_U * sigma_social_for_u + SIGMA_BASELINE - u)     (L121)
      -> a l'equilibre : u ≈ gain*|S| + 0.05
    u_filter = tanh(pi*(0.5 - u)) + 0.01                     (L111)
      -> le couplage ne bascule en REPULSIF que si u > 0.5
Donc la bascule exige |S| > 0.45/gain. Ce modele produit |S| ≈ 0.011 :
il manque un facteur ~41. Le capteur ne mesure pas la meme grandeur a la meme
echelle que dans le modele Kuramoto (moyenne VECTORIELLE COMPLEXE de (a_j - a_i),
dont les contributions s'annulent, contre une moyenne de sinus d'ordre 1).
C'est un probleme d'UNITE DE CAPTEUR, pas de physique d'oscillateur.

QUESTION DE CE SCRIPT
---------------------
Quel gain faut-il ? C'est une specification d'ingenierie (« quel ampli ? »),
pas un parametre libre qu'on ajuste jusqu'a ce que ca marche.

CRITERES ECRITS AVANT MESURE (et leur sort, rapporte a l'execution)
-------------------------------------------------------------------
  G1  GATE DE FIDELITE — ce harnais doit reproduire les u_mean du CSV du 09/07
      a moins de 0.001 : BA gain=1 -> 0.0608, gain=10 -> 0.5034 ;
      LATTICE gain=1 -> 0.0607, gain=10 -> 0.5252. Si G1 echoue, tout le reste
      est ininterpretable — le script le dit et n'interprete rien.
  P5  le gain minimal de bascule est entre 3 et 7.            -> VERIFIEE (7)
  P6  la transition est ABRUPTE (saut de u_mean de <0.1 a >0.4 en moins d'un
      facteur 2 de gain), consequence attendue de la boucle auto-renforcante.
                                                              -> ECHOUEE, et
      c'est le resultat le plus utile : la montee est REGULIERE, sans emballement.
  P7  le seuil est du meme ordre sur les deux topologies (facteur < 2).
                                                              -> VERIFIEE (7 et 7)

CE QUE LA MESURE A APPRIS ET QUE PERSONNE N'AVAIT PREDIT
--------------------------------------------------------
L'effet est deja FRANC AVANT toute bascule : a gain=5, Cohen d = +1.35 sur BA
alors que 0/10 graines franchissent u=0.5. Le mecanisme n'a donc PAS besoin de
l'inversion de polarite du couplage — la modulation « douce » de son amplitude
suffit. La consequence pour un experimentateur est directe : la cible n'est pas
« faire basculer u », c'est « obtenir un effet detectable », et cela demande
moins de gain.

RESERVES, a ne pas perdre
-------------------------
- Ce modele n'a AUCUN BRUIT sur le capteur. Un vrai circuit d'amplification en
  ajoute, et rien ici ne dit que l'effet y survit. C'est la question suivante.
- Le canal de couplage electrique reel (Romera) n'est toujours pas modelise, et
  la geometrie testee dans le §9 verrouille en ANTIPHASE la ou la litterature
  rapporte plutot un verrouillage en phase.
- Un gain n'est pas gratuit en surface ni en consommation : ce script ne chiffre
  aucun cout, il chiffre une exigence.

SORTIES : figures/b6_sensor_gain_threshold.csv (par run)
          figures/b6_sensor_gain_threshold_summary.csv (1 ligne par cellule)
          figures/b6_sensor_gain_threshold.png
Les CSV vont dans figures/ et NON dans figures/scratch/ (gitignore) : les
donnees des trois POC STNO y dorment et ne sont pas verifiables par qui clone le
depot — dette signalee le 31/07, non traitee ici.
"""
import csv
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'src'))

# Import et NON copie : toute divergence de harnais serait invisible autrement.
from b2_stno_amplitude_phase_poc import run_one, SEEDS, N        # noqa: E402
from mem4ristor.graph_utils import make_ba, make_lattice_adj      # noqa: E402

GAINS = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
N_NONLIN_VALUES = [0.0, 10.0]   # 0 = isochrone ; 10 = non-isochronicite maximale testee le 09/07
U_SWITCH = 0.5                  # seuil de bascule de polarite (dynamics.py, inchange)

# References du CSV du 09/07 pour le gate de fidelite (n_nonlin=0).
G1_REF = {('BA_m3', 1.0): 0.0608, ('BA_m3', 10.0): 0.5034,
          ('LATTICE_10x10', 1.0): 0.0607, ('LATTICE_10x10', 10.0): 0.5252}
G1_TOL = 0.001


def main():
    t0 = time.time()
    topologies = {'BA_m3': make_ba(N, 3, seed=42),
                  'LATTICE_10x10': make_lattice_adj(10, periodic=True)}
    rows, summary = [], []

    print("=" * 94)
    print("  B6 — SEUIL DE GAIN DU CAPTEUR DE DESACCORD")
    print(f"  Bascule de polarite du couplage : u > {U_SWITCH}")
    print("=" * 94)

    for topo_name, adj in topologies.items():
        for n_nonlin in N_NONLIN_VALUES:
            frozen = []
            for seed in SEEDS:
                r = run_one(adj, seed, 'FROZEN_U', 1.0, n_nonlin)
                rows.append({'topology': topo_name, 'n_nonlin': n_nonlin, 'condition': 'FROZEN_U',
                             'gain_u': '', 'seed': seed, **r})
                frozen.append(r['R_mean'])
            frozen = np.array(frozen)

            print(f"\n--- {topo_name}  n_nonlin={n_nonlin:.0f} ---   R_FROZEN = {frozen.mean():.4f}")
            print(f"{'gain':>6} {'u_mean':>8} {'u_max':>8} {'|S|':>9} {'bascule':>9} "
                  f"{'R_FULL':>8} {'diff':>9} {'Cohen d':>9}")
            for gain in GAINS:
                rs = []
                for seed in SEEDS:
                    r = run_one(adj, seed, 'FULL', gain, n_nonlin)
                    rows.append({'topology': topo_name, 'n_nonlin': n_nonlin, 'condition': 'FULL',
                                 'gain_u': gain, 'seed': seed, **r})
                    rs.append(r)
                full = np.array([r['R_mean'] for r in rs])
                u_mean = float(np.mean([r['u_mean'] for r in rs]))
                u_max = float(np.mean([r['u_max'] for r in rs]))
                n_switch = int(sum(1 for r in rs if r['u_max'] >= U_SWITCH))
                diff = float(frozen.mean() - full.mean())
                pooled = float(np.sqrt((full.var(ddof=1) + frozen.var(ddof=1)) / 2))
                cohen_d = diff / pooled if pooled > 1e-12 else float('nan')
                s_implied = (u_mean - 0.05) / gain   # |S| deduit de l'equilibre de u

                summary.append({
                    'topology': topo_name, 'n_nonlin': n_nonlin, 'gain_u': gain,
                    'u_mean': u_mean, 'u_max_mean': u_max, 'sigma_social_implied': s_implied,
                    'n_seeds_switching': n_switch, 'n_seeds': len(SEEDS),
                    'R_FULL_mean': float(full.mean()), 'R_FROZEN_mean': float(frozen.mean()),
                    'diff_FROZEN_minus_FULL': diff, 'cohen_d': float(cohen_d),
                })
                print(f"{gain:>6.1f} {u_mean:>8.4f} {u_max:>8.4f} {s_implied:>9.5f} "
                      f"{n_switch:>6}/10 {full.mean():>8.4f} {diff:>+9.4f} {cohen_d:>+9.2f}")

    fig_dir = HERE.parent / 'figures'
    fig_dir.mkdir(exist_ok=True)
    raw_path = fig_dir / 'b6_sensor_gain_threshold.csv'
    with open(raw_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    sum_path = fig_dir / 'b6_sensor_gain_threshold_summary.csv'
    with open(sum_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)

    # ---------------- VERDICTS IMPRIMES A L'EXECUTION ----------------
    print("\n" + "=" * 94)
    print("VERDICTS (criteres ecrits avant mesure, cf. docstring)")
    print("=" * 94)

    g1_ok, g1_detail = True, []
    for (topo, gain), ref in G1_REF.items():
        got = next(s['u_mean'] for s in summary
                   if s['topology'] == topo and s['gain_u'] == gain and s['n_nonlin'] == 0.0)
        ok = abs(got - ref) <= G1_TOL
        g1_ok &= ok
        g1_detail.append(f"{topo} gain={gain:.0f} : {got:.4f} vs {ref:.4f} {'OK' if ok else 'ECART'}")
    print(f"  [G1 fidelite] {'PASSE' if g1_ok else 'ECHOUE'}")
    for d in g1_detail:
        print(f"      {d}")
    if not g1_ok:
        print("  /!\\ G1 ECHOUE : le harnais a diverge du POC du 09/07. RIEN CI-DESSOUS "
              "N'EST INTERPRETABLE. Diagnostiquer avant de lire les verdicts suivants.")
        return

    for n_nonlin in N_NONLIN_VALUES:
        seuils = {}
        for topo in topologies:
            cells = [s for s in summary if s['topology'] == topo and s['n_nonlin'] == n_nonlin]
            seuils[topo] = next((s['gain_u'] for s in sorted(cells, key=lambda x: x['gain_u'])
                                 if s['n_seeds_switching'] >= 6), None)
        print(f"  [seuil de bascule, n_nonlin={n_nonlin:.0f}] "
              + " · ".join(f"{t} : {v if v else 'hors plage'}" for t, v in seuils.items()))

    # Le fait non predit : effet franc AVANT toute bascule.
    print("\n  [effet AVANT bascule] — le mecanisme ne requiert pas l'inversion de polarite :")
    for s in summary:
        if s['n_seeds_switching'] == 0 and s['cohen_d'] >= 1.0:
            print(f"      {s['topology']:15s} n_nonlin={s['n_nonlin']:.0f} gain={s['gain_u']:.0f} : "
                  f"Cohen d={s['cohen_d']:+.2f} avec 0/10 graines au-dessus de u={U_SWITCH}")

    # La non-isochronicite change-t-elle le capteur ? (disculpation, re-mesuree ici)
    print("\n  [non-isochronicite et capteur] u_mean au gain=1, par n_nonlin :")
    for topo in topologies:
        vals = [(s['n_nonlin'], s['u_mean']) for s in summary
                if s['topology'] == topo and s['gain_u'] == 1.0]
        txt = " · ".join(f"n_nonlin={nn:.0f} -> {v:.4f}" for nn, v in sorted(vals))
        spread = max(v for _, v in vals) - min(v for _, v in vals)
        print(f"      {topo:15s} {txt}   (etendue {spread:.4f})")
    print("      => si l'etendue est negligeable, la non-isochronicite n'agit PAS sur le capteur.")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
        for ax, key, lab in ((axes[0], 'cohen_d', "Cohen's d (FROZEN_U - FULL)"),
                             (axes[1], 'u_mean', "u moyen")):
            for topo, color in zip(topologies, ('steelblue', 'crimson')):
                for nn, style in zip(N_NONLIN_VALUES, ('-', '--')):
                    ys = [s[key] for s in sorted(
                        (s for s in summary if s['topology'] == topo and s['n_nonlin'] == nn),
                        key=lambda x: x['gain_u'])]
                    ax.plot(GAINS, ys, style, marker='o', color=color, ms=4,
                            label=f"{topo} n_nl={nn:.0f}")
            ax.set_xlabel('gain du capteur de desaccord')
            ax.set_ylabel(lab)
            ax.grid(alpha=0.3)
        axes[1].axhline(U_SWITCH, color='k', ls=':', lw=1)
        axes[1].annotate('bascule de polarite', (1.1, U_SWITCH + 0.02), fontsize=8)
        axes[0].axhline(1.0, color='gray', ls=':', lw=1)
        axes[0].annotate("d = 1 (effet franc)", (1.1, 1.05), fontsize=8)
        axes[0].legend(fontsize=7)
        fig.suptitle("B6 — de quel gain de capteur un laboratoire a-t-il besoin ?", fontsize=10)
        plt.tight_layout()
        png = fig_dir / 'b6_sensor_gain_threshold.png'
        plt.savefig(png, dpi=150, bbox_inches='tight')
        print(f"\nFigure : {png}")
    except Exception as e:
        print(f"[matplotlib] {e}")

    print(f"\nCSV : {raw_path}\n      {sum_path}")
    print(f"Wall time : {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
