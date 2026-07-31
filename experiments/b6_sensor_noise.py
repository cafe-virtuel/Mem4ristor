#!/usr/bin/env python3
"""
B6 — LE BRUIT DU CAPTEUR : il AIDE, et c'est une mauvaise nouvelle — 2026-07-31
Claude Code (Opus 5) / Julien Chauvin. Suite directe de b6_sensor_gain_threshold.py.

D'OU VIENT LA QUESTION
----------------------
De Julien : « je pensais que le bruit etait meme benefique pour M4R ? ».
Verification prealable : OUI pour le bruit de DYNAMIQUE — la variabilite de
fabrication augmente l'entropie de +0.05 a +0.75 bits (PHOTONIC_PATHWAY §4quater,
12/06/2026), et les versions historiques du preprint parlaient de « turning
fabrication imperfections into diversity-enhancing features ». Mais le bruit du
CAPTEUR est autre chose : du bruit sur une INFORMATION, pas sur le mouvement.
Jamais teste avant aujourd'hui.

POURQUOI SON INTUITION POUVAIT VALOIR ICI AUSSI (mecanisme, pas espoir)
-----------------------------------------------------------------------
Le capteur mesure |S|, une VALEUR ABSOLUE. Un bruit symetrique ajoute AVANT le
module augmente systematiquement le resultat : E[|S+eps|] > |E[S]|. C'est de la
RECTIFICATION. Le bruit devait donc faire MONTER u — exactement ce qui manque
(|S| ~ 0.011, il en faut 0.45 pour basculer). Et le 29/07 (P17) avait etabli que
seul le NIVEAU atteint par u compte, pas son adaptativite.

CRITERES ECRITS AVANT MESURE, ET LEUR SORT
-------------------------------------------
  G0  fidelite : a bruit nul, cette boucle (reecrite pour injecter le bruit)
      doit reproduire POC.run_one au chiffre pres.        -> PASSE (0.0608, 0.1569)
  Q1  le bruit fait monter u par rectification.           -> VERIFIEE
  Q2  a gain=1, un bruit suffisant fait franchir u=0.5 ET donne d >= 1.
      -> A MOITIE : l'effet arrive (d=+3.22 a sigma=0.8) mais SANS bascule
         (0/10, u_max=0.41). Conforme au fait etabli le matin meme : le
         mecanisme n'a pas besoin de l'inversion de polarite.
  Q3  CONTROLE CONTRE MON ENVIE : l'effet du bruit sera equivalent a celui d'un
      u FIGE au meme niveau (ecart de Cohen d < 0.5).
      -> VERIFIEE 4 fois sur 6. Et dans les 2 exceptions, le u FIGE fait MIEUX
         (+7.98 contre +5.78 ; +3.73 contre +3.22). Le bruit n'apporte donc
         JAMAIS d'avantage sur le niveau.
  Q4  CONTROLE PLUS DUR : un capteur qui ne mesure QUE du bruit (aucune
      information locale) fera aussi bien.
      -> VERIFIEE A FORT BRUIT (ecart 0.01), REJETEE A FAIBLE BRUIT
         (+2.68 avec signal contre +0.98 sans, a sigma=0.10). Il existe donc une
         fenetre ou l'information compte encore, et elle est etroite.

CE QUI EST ETABLI
-----------------
1. Le bruit du capteur AIDE : a gain=1 (aucun amplificateur), sigma=0.8 donne
   Cohen d = +3.22, contre +0.08 avec un capteur propre — et MIEUX qu'un
   amplificateur de gain 7 sans bruit (+2.75). Le bruit REMPLACE l'ampli.
2. MAIS il apporte du NIVEAU, pas de l'INFORMATION. Deux controles convergents :
   un u fige au meme niveau fait aussi bien ou mieux (Q3) ; et a fort bruit un
   capteur AVEUGLE fait exactement aussi bien (Q4, ecart 0.01).
3. Troisieme porte vers la meme conclusion que le 29/07 : seul le niveau compte.

CONSEQUENCE POUR LE PROTOCOLE EXPERIMENTAL B6 — LA PARTIE IMPORTANTE
---------------------------------------------------------------------
Bonne nouvelle : la reserve « le bruit du capteur pourrait tuer l'effet » est
LEVEE — c'est l'inverse. Un laboratoire n'a pas besoin d'une chaine de detection
propre.
Mauvaise nouvelle, et elle touche le POUVOIR DISCRIMINANT de la prediction : si
un capteur bruite produit le meme effet qu'un capteur informe, alors l'experience
telle que formulee ne teste plus le MECANISME DU DOUTE — elle teste « un
couplage repulsif en moyenne ». Le controle actuel (couplage fige a sa valeur
INITIALE) ne suffit pas a distinguer les deux.
=> IL FAUT UN TROISIEME BRAS : couplage fixe regle AU NIVEAU MOYEN ATTEINT par
le doute. C'est exactement le controle FROZEN_U(0.95) que Julien avait fait
ajouter le 28/07 sur la niche, transpose au protocole physique. Sans lui, un
labo mesurerait un effet reel et l'attribuerait a la mauvaise cause.

SORTIES : figures/b6_sensor_noise.csv / _summary.csv / .png  (versionnees)
"""
import csv
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'src'))

import b2_stno_amplitude_phase_poc as POC                     # noqa: E402
from mem4ristor.graph_utils import make_ba                     # noqa: E402

SEEDS = POC.SEEDS
N, DT, WARM_UP, STEPS = POC.N, POC.DT, POC.WARM_UP, POC.STEPS
N_NONLIN = 0.0
GAINS = [1.0, 5.0]
SIGMAS = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8]
U_SWITCH = 0.5
G0_REF = {1.0: 0.0608, 5.0: 0.1569}      # POC du 09/07 + balayage du 31/07 au matin
G0_TOL = 0.001


def run(adj, seed, condition, gain_u, sigma_capteur=0.0,
        capteur_mode='signal', u_frozen_at=None):
    """Copie FIDELE de POC.run_one avec TROIS ajouts, et rien d'autre :
      - sigma_capteur : bruit gaussien complexe ajoute a S AVANT le module,
        la ou le bruit d'une vraie chaine de detection s'ajoute ;
      - capteur_mode='bruit_pur' : S remplace par du bruit seul (controle Q4) ;
      - u_frozen_at : u maintenu constant (controle Q3).
    Le RNG du capteur est SEPARE de celui de la dynamique : a sigma=0 la
    trajectoire est donc strictement identique au POC — c'est ce que G0 verifie.
    Reecrire cette boucle etait inevitable (run_one n'expose pas le capteur) ;
    G0 est la contrepartie obligatoire de cette reecriture."""
    rng = np.random.RandomState(seed)
    rng_capteur = np.random.RandomState(seed + 100000)
    n = adj.shape[0]
    deg = adj.sum(axis=1)
    deg_safe = np.where(deg > 0, deg, 1.0)

    omega = POC.OMEGA0 + rng.normal(0, POC.SIGMA_OMEGA, n)
    a = 0.05 * (rng.randn(n) + 1j * rng.randn(n))
    u = np.full(n, POC.SIGMA_BASELINE if u_frozen_at is None else u_frozen_at)

    R_traj, u_traj, s_traj = [], [], []
    for t in range(WARM_UP + STEPS):
        diff = a[None, :] - a[:, None]
        S = (adj * diff).sum(axis=1) / deg_safe

        S_capte = np.zeros(n, dtype=complex) if capteur_mode == 'bruit_pur' else S
        if sigma_capteur > 0:
            S_capte = S_capte + (rng_capteur.normal(0, sigma_capteur, n)
                                 + 1j * rng_capteur.normal(0, sigma_capteur, n))
        sigma_social = np.abs(S_capte)

        sigma_social_for_u = np.zeros(n) if condition == 'FROZEN_U' else sigma_social * gain_u
        u_filter = np.tanh(np.pi * (0.5 - u)) + POC.SOCIAL_LEAKAGE

        p = np.abs(a) ** 2
        growth = POC.GAMMA_PLUS - POC.GAMMA_MINUS * (1.0 + POC.Q * p)
        eta = (rng.normal(0, POC.SIGMA_NOISE, n)
               + 1j * rng.normal(0, POC.SIGMA_NOISE, n)) / np.sqrt(DT)
        da = (growth + 1j * (omega + N_NONLIN * p)) * a + POC.K_COUPLING * u_filter * S + eta

        if u_frozen_at is None:
            sigma_safe = np.clip(sigma_social_for_u, 0.0, 100.0)
            eps_adapt = POC.EPSILON_U * np.clip(
                1.0 + POC.ALPHA_SURPRISE * sigma_safe, 1.0, POC.SURPRISE_CAP)
            du = eps_adapt * (POC.K_U * sigma_social_for_u + POC.SIGMA_BASELINE - u) / POC.TAU_U
            u = np.clip(u + du * DT, 0.0, 1.0)

        a = a + da * DT
        if not np.all(np.isfinite(a)):
            raise OverflowError(f"divergence Euler (seed={seed}, t={t})")
        if t >= WARM_UP:
            R_traj.append(float(np.abs(np.mean(np.exp(1j * np.angle(a))))))
            u_traj.append(float(u.mean()))
            s_traj.append(float(sigma_social.mean()))

    return {'R_mean': float(np.mean(R_traj)), 'u_mean': float(np.mean(u_traj)),
            'u_max': float(np.max(u_traj)), 'sigma_social_capte': float(np.mean(s_traj))}


def cohen(full, frozen):
    pooled = np.sqrt((full.var(ddof=1) + frozen.var(ddof=1)) / 2)
    return float((frozen.mean() - full.mean()) / pooled) if pooled > 1e-12 else float('nan')


def main():
    t0 = time.time()
    adj = make_ba(N, 3, seed=42)
    rows, summary = [], []

    frozen_runs = [run(adj, s, 'FROZEN_U', 1.0) for s in SEEDS]
    frozen = np.array([r['R_mean'] for r in frozen_runs])
    for s, r in zip(SEEDS, frozen_runs):
        rows.append({'bras': 'FROZEN_U', 'gain_u': '', 'sigma_capteur': '', 'seed': s, **r})

    print("=" * 96)
    print(f"  B6 — BRUIT DU CAPTEUR.  BA m=3, n_nonlin={N_NONLIN:.0f}.  R_FROZEN = {frozen.mean():.4f}")
    print("=" * 96)

    g0_ok = True
    for gain, ref in G0_REF.items():
        got = float(np.mean([run(adj, s, 'FULL', gain)['u_mean'] for s in SEEDS]))
        ok = abs(got - ref) <= G0_TOL
        g0_ok &= ok
        print(f"  [G0 fidelite] gain={gain:.0f} : u_mean={got:.4f} vs reference {ref:.4f}  "
              f"{'OK' if ok else 'ECART'}")
    if not g0_ok:
        print("  /!\\ G0 ECHOUE : la boucle reecrite a diverge du POC. RIEN N'EST INTERPRETABLE.")
        return

    for gain in GAINS:
        print(f"\n--- gain = {gain:.0f} ---")
        print(f"{'sigma_capt':>11} {'|S| capte':>10} {'u_mean':>8} {'u_max':>8} "
              f"{'bascule':>8} {'R_FULL':>8} {'Cohen d':>9}")
        for sg in SIGMAS:
            rs = [run(adj, s, 'FULL', gain, sigma_capteur=sg) for s in SEEDS]
            for s, r in zip(SEEDS, rs):
                rows.append({'bras': 'FULL', 'gain_u': gain, 'sigma_capteur': sg, 'seed': s, **r})
            full = np.array([r['R_mean'] for r in rs])
            d = cohen(full, frozen)
            um = float(np.mean([r['u_mean'] for r in rs]))
            n_sw = int(sum(1 for r in rs if r['u_max'] >= U_SWITCH))

            # Controles Q3 / Q4, uniquement la ou il y a un effet a expliquer
            d_fige = d_pur = float('nan')
            if sg > 0 and d >= 1.0:
                fz = np.array([run(adj, s, 'FULL', gain, u_frozen_at=um)['R_mean'] for s in SEEDS])
                d_fige = cohen(fz, frozen)
                pur = np.array([run(adj, s, 'FULL', gain, sigma_capteur=sg,
                                    capteur_mode='bruit_pur')['R_mean'] for s in SEEDS])
                d_pur = cohen(pur, frozen)

            summary.append({
                'gain_u': gain, 'sigma_capteur': sg, 'u_mean': um,
                'u_max_mean': float(np.mean([r['u_max'] for r in rs])),
                'sigma_social_capte': float(np.mean([r['sigma_social_capte'] for r in rs])),
                'n_seeds_switching': n_sw, 'R_FULL_mean': float(full.mean()),
                'R_FROZEN_mean': float(frozen.mean()), 'cohen_d': d,
                'cohen_d_u_fige_meme_niveau': d_fige, 'cohen_d_capteur_aveugle': d_pur,
            })
            print(f"{sg:>11.2f} {summary[-1]['sigma_social_capte']:>10.4f} {um:>8.4f} "
                  f"{summary[-1]['u_max_mean']:>8.4f} {n_sw:>6}/10 {full.mean():>8.4f} {d:>+9.2f}")

    fig_dir = HERE.parent / 'figures'
    fig_dir.mkdir(exist_ok=True)
    with open(fig_dir / 'b6_sensor_noise.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(fig_dir / 'b6_sensor_noise_summary.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)

    print("\n" + "=" * 96)
    print("VERDICTS — le bruit apporte-t-il de l'INFORMATION, ou seulement du NIVEAU ?")
    print("=" * 96)
    for s in summary:
        if np.isnan(s['cohen_d_u_fige_meme_niveau']):
            continue
        e3 = abs(s['cohen_d'] - s['cohen_d_u_fige_meme_niveau'])
        e4 = abs(s['cohen_d'] - s['cohen_d_capteur_aveugle'])
        print(f"  gain={s['gain_u']:.0f} sigma={s['sigma_capteur']:.2f} : "
              f"bruit d={s['cohen_d']:+.2f} | u FIGE meme niveau d={s['cohen_d_u_fige_meme_niveau']:+.2f} "
              f"(ecart {e3:.2f} -> {'niveau seul' if e3 < 0.5 else 'DIFFERENT'}) | "
              f"capteur AVEUGLE d={s['cohen_d_capteur_aveugle']:+.2f} "
              f"(ecart {e4:.2f} -> {'info inutile' if e4 < 0.5 else 'info utile'})")
    print("\n  Lecture : quand « u fige au meme niveau » fait aussi bien ou MIEUX, le bruit")
    print("  n'apporte pas d'information — seulement du niveau. Quand « capteur aveugle »")
    print("  fait aussi bien, le dispositif fonctionne SANS RIEN MESURER du reseau.")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
        for ax, gain in zip(axes, GAINS):
            cells = sorted((s for s in summary if s['gain_u'] == gain),
                           key=lambda x: x['sigma_capteur'])
            xs = [c['sigma_capteur'] for c in cells]
            ax.plot(xs, [c['cohen_d'] for c in cells], marker='o', color='steelblue',
                    label='capteur bruite (signal + bruit)')
            ax.plot(xs, [c['cohen_d_capteur_aveugle'] for c in cells], marker='s', ls='--',
                    color='crimson', label='capteur AVEUGLE (bruit seul)')
            ax.plot(xs, [c['cohen_d_u_fige_meme_niveau'] for c in cells], marker='^', ls=':',
                    color='gray', label='u FIGE au meme niveau')
            ax.axhline(1.0, color='k', ls=':', lw=1)
            ax.set_xlabel('bruit du capteur (sigma)'); ax.set_ylabel("Cohen's d")
            ax.set_title(f'gain = {gain:.0f}'); ax.grid(alpha=0.3)
        axes[0].legend(fontsize=7)
        fig.suptitle("B6 — le bruit du capteur aide, mais il apporte du NIVEAU, pas de l'INFORMATION",
                     fontsize=10)
        plt.tight_layout()
        plt.savefig(fig_dir / 'b6_sensor_noise.png', dpi=150, bbox_inches='tight')
        print(f"\nFigure : {fig_dir / 'b6_sensor_noise.png'}")
    except Exception as e:
        print(f"[matplotlib] {e}")

    print(f"\nCSV : {fig_dir / 'b6_sensor_noise.csv'}")
    print(f"      {fig_dir / 'b6_sensor_noise_summary.csv'}")
    print(f"Wall time : {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
