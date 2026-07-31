#!/usr/bin/env python3
"""
B6 — LE TROISIEME BRAS SUR LE VOLET 2 : le retard de flip est-il specifique au doute ?
=======================================================================================
2026-07-31 (soir). Claude Code (Opus 5) / Julien Chauvin.
Suite directe de b6_third_arm.py, qui a tue le pouvoir discriminant du VOLET 1.

CE QUE LE VOLET 1 A PERDU (mesure de ce soir, b6_third_arm.py)
--------------------------------------------------------------
Sur la synchronisation STATIONNAIRE, un couplage FIXE regle au niveau moyen atteint
reproduit le mecanisme a 0.24 de Cohen d pres (P2 rejetee), et son reglage TRANSFERE
d'une topologie a l'autre et d'une dispersion de frequence a l'autre — parfois mieux
que le reglage local (P3 rejetee 4/4). Le volet 1 de B6 ne teste donc pas le doute :
il teste « un couplage anti-synchronisant en moyenne ».

POURQUOI LE VOLET 2 EST LA SEULE PORTE QUI RESTE
-------------------------------------------------
Le volet 2 (P12, 12/07, b1d_stno_deceptive_poc.py) est un effet TRANSITOIRE : sous
tache trompeuse, le couplage module par le desaccord RETARDE la recuperation apres
le leurre (flip_time moyen 5274.7 pas contre 3466.7 — le « +52 % » cite dans
FUTURE_WORK B6). Son mecanisme est explicitement temporel, decrit le 12/07 comme la
« CICATRICE DE DOUTE » : le conflit fait monter u DURABLEMENT, donc u_filter baisse,
donc le couplage se coupe et VERROUILLE la trace du leurre.
Un couplage CONSTANT ne peut pas, par construction, monter pendant le conflit et
redescendre apres. Si le retard est bien une hysteresis, le bras 3 doit ECHOUER a le
reproduire — et B6 redevient discriminante, sur le TRANSITOIRE au lieu du stationnaire.

LES CINQ BRAS
-------------
  B1   FULL              u dynamique (le mecanisme)                    [= STNO_FULL]
  B2   FIXE_INIT         u fige a 0.05 -> couplage quasi plein         [= STNO_FROZEN_U]
  B3a  FIXE_NIVEAU_U     u fige a <u> de B1                            (non realisable)
  B3b  FIXE_COUPLAGE     u_filter fige a <u_filter> de B1, reglage GLOBAL
                         (un seul couplage cable — ce qu'un labo fait vraiment)
  B3c  FIXE_COUPLAGE_LOCAL  u_filter fige a <u_filter> de B1 POUR CE T_pulse
                         (reglage le PLUS FAVORABLE possible au bras fixe)

CRITERES ET PREDICTIONS — ECRITS AVANT LA MESURE
------------------------------------------------
  G0  FIDELITE (BLOQUANT). B1 et B2 doivent reproduire le CSV du 12/07
      (figures/scratch/b1d_stno_deceptive_poc.csv) au pas pres, par T_pulse :
        FULL   : 1294.0 / 5102.6 / 6838.6 / 7863.7   (moyenne 5274.7)
        FROZEN : 1291.2 / 2611.4 / 4239.6 / 5724.8   (moyenne 3466.7)
      Tolerance 0.5 pas. Si G0 echoue, RIEN n'est interpretable.

  Q1  LE BRAS FIXE REPRODUIT-IL LE RETARD ? (prediction RISQUEE, directionnelle —
      c'est elle qui decide si B6 garde un pouvoir discriminant)
      On mesure la FRACTION DU RETARD reproduite :
          frac(X) = (flip(X) - flip(B2)) / (flip(B1) - flip(B2))
      Predit : frac(B3b) <= 0.50 — le couplage fixe recupere plus de la moitie du
      retard, donc il NE reproduit PAS la cicatrice.
        - Q1 vraie  -> LE VOLET 2 EST LE DISCRIMINANT. B6 est sauvee et doit etre
          reformulee autour du transitoire, pas de la synchronisation stationnaire.
        - Q1 fausse -> le retard n'est qu'un effet de NIVEAU de couplage, la
          « cicatrice de doute » est mal nommee, et B6 perd aussi son volet 2.
          A ecrire tel quel.

  Q2  CONTROLE : 3a et 3b disent-ils la meme chose ici aussi ?
      Predit VRAI : |flip(3a) - flip(3b)| < 10 % de flip(B2). En regime stationnaire
      l'ecart de Jensen etait de 0.017 seulement (G1 rejetee ce soir). Si les deux
      divergent ICI, c'est une information NEUVE : cela voudrait dire que la
      non-linearite de tanh compte dans le transitoire alors qu'elle ne comptait pas
      dans le stationnaire.

  Q3  CONTROLE ADJACENT OBLIGATOIRE : le reglage LOCAL (B3c, un couplage par T_pulse,
      le plus favorable qu'on puisse offrir au bras fixe) ne doit pas sauver le bras
      fixe si Q1 est vraie. Sans ce controle, un echec de B3b pourrait n'etre qu'un
      mauvais reglage global — et on attribuerait au doute ce qui ne serait qu'une
      calibration ratee. C'est exactement l'erreur que ce troisieme bras existe pour
      empecher.

  PRESOMPTION NEGATIVE, ECRITE AVANT : si le bras fixe REPRODUIT le retard, alors la
  « cicatrice de doute » du 12/07 est mal nommee — ce ne serait pas une hysteresis
  mais simplement un couplage plus faible EN MOYENNE, et il faudra le dire sans
  chercher a repecher le volet 2 sur une autre observable.

Coeur NON touche. b1d_stno_deceptive_poc.py NON modifie (il produit un resultat cite).
SORTIES : figures/b6_third_arm_transient.csv / _summary.csv  (VERSIONNEES)
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'src'))

import b1d_stno_deceptive_poc as P12                              # noqa: E402
from mem4ristor.graph_utils import make_lattice_adj                # noqa: E402

# Toutes les constantes viennent de P12 — aucun reglage propre.
N, DT, MAX_BUDGET = P12.N, P12.DT, P12.MAX_BUDGET
T_PULSE_LEVELS, SEEDS = P12.T_PULSE_LEVELS, P12.SEEDS
WARMUP_STEPS, W_READ, ISCALE = P12.WARMUP_STEPS, P12.W_READ, P12.ISCALE

G0_REF = {
    'B1_FULL': {500: 1294.0, 1500: 5102.6, 3000: 6838.6, 4500: 7863.7},
    'B2_FIXE_INIT': {500: 1291.2, 1500: 2611.4, 3000: 4239.6, 4500: 5724.8},
}
G0_TOL = 0.5
Q1_MAX_FRAC = 0.50
Q2_MAX_ECART = 0.10


def _step(a, u, gp, omega, adj, deg, eta, libre, filter_frozen_at):
    """Copie FIDELE de P12._step_one avec DEUX ajouts et rien d'autre :
      - libre=False fige u (couvre B2 et B3a) ;
      - filter_frozen_at remplace u_filter par une constante (B3b/B3c).
    Le nombre et l'ordre des tirages du RNG sont INCHANGES — c'est ce qui rend G0
    verifiable au pas pres."""
    S = (adj @ a) / deg - a
    abs_s = np.abs(S)
    sigma_for_u = abs_s * P12.GAIN_U if libre else np.zeros_like(u)
    if filter_frozen_at is None:
        u_filter = np.tanh(np.pi * (0.5 - u)) + P12.SOCIAL_LEAKAGE
    else:
        u_filter = np.full_like(u, filter_frozen_at)
    p = np.abs(a) ** 2
    growth = gp - P12.GAMMA_MINUS * (1.0 + P12.Q * p)
    da = (growth + 1j * omega) * a + P12.K_COUPLING * u_filter * S + eta
    if libre:
        sigma_safe = np.clip(sigma_for_u, 0.0, 100.0)
        eps_adapt = P12.EPSILON_U * np.clip(
            1.0 + P12.ALPHA_SURPRISE * sigma_safe, 1.0, P12.SURPRISE_CAP)
        du = eps_adapt * (P12.K_U * sigma_for_u + P12.SIGMA_BASELINE - u) / P12.TAU_U
        u = np.clip(u + du * DT, 0.0, 1.0)
    return a + da * DT, u, u_filter


def simulate(adj, deg, stim_on, stim_off, seed, t_pulse, bras,
             u_frozen_at=None, filter_frozen_at=None):
    """Paire differentielle de P12.simulate, allegee : on ne calcule que ce dont
    la decision a besoin (dmat lisse sur W_READ) plus les niveaux <u> / <u_filter>.
    sig_p, sig_s et la fenetre longue ne servaient qu'aux regles d'arret."""
    rng = np.random.default_rng(seed)
    libre = (bras == 'B1_FULL')
    omega = P12.OMEGA0 + rng.normal(0, P12.SIGMA_OMEGA, N)
    phases = rng.uniform(0.0, 2.0 * np.pi, N)
    p_star = (P12.GAMMA_PLUS - P12.GAMMA_MINUS) / (P12.GAMMA_MINUS * P12.Q)
    a = np.sqrt(p_star) * np.exp(1j * phases)
    u = np.full(N, P12.SIGMA_BASELINE if u_frozen_at is None else u_frozen_at)
    inv_sqrt_dt = 1.0 / np.sqrt(DT)

    for _ in range(WARMUP_STEPS):
        noise = rng.normal(0.0, P12.SIGMA_NOISE, size=(2, N))
        eta = (noise[0] + 1j * noise[1]) * inv_sqrt_dt
        a, u, _ = _step(a, u, P12.GAMMA_PLUS, omega, adj, deg, eta, libre, filter_frozen_at)
    if not np.all(np.isfinite(a)):
        return None

    a_pos, a_neg = a.copy(), a.copy()
    u_pos, u_neg = u.copy(), u.copy()
    dmat = np.empty((MAX_BUDGET, N))
    u_sum = uf_sum = 0.0

    for t in range(MAX_BUDGET):
        stim = stim_on if t < t_pulse else stim_off
        noise = rng.normal(0.0, P12.SIGMA_NOISE, size=(2, N))
        eta = (noise[0] + 1j * noise[1]) * inv_sqrt_dt
        a_pos, u_pos, uf_pos = _step(a_pos, u_pos, P12.GAMMA_PLUS + ISCALE * stim,
                                     omega, adj, deg, eta, libre, filter_frozen_at)
        a_neg, u_neg, uf_neg = _step(a_neg, u_neg, P12.GAMMA_PLUS - ISCALE * stim,
                                     omega, adj, deg, eta, libre, filter_frozen_at)
        if not (np.all(np.isfinite(a_pos)) and np.all(np.isfinite(a_neg))):
            return None
        dmat[t] = np.abs(a_pos) ** 2 - np.abs(a_neg) ** 2
        u_sum += 0.5 * float(u_pos.mean() + u_neg.mean())
        uf_sum += 0.5 * float(uf_pos.mean() + uf_neg.mean())

    csum = np.cumsum(dmat, axis=0)
    dsm = np.empty_like(dmat)
    for t in range(MAX_BUDGET):
        lo = max(0, t - W_READ + 1)
        dsm[t] = (csum[t] - (csum[lo - 1] if lo > 0 else 0.0)) / (t - lo + 1)
    dec = np.where(dsm.mean(axis=1) >= 0, 1, -1).astype(int)
    return dec, u_sum / MAX_BUDGET, uf_sum / MAX_BUDGET


def serie(adj, deg, bras, t_pulse, rows, **kw):
    """Les 12 graines d'une condition. Retourne (flips, <u>, <u_filter>)."""
    flips, us, ufs = [], [], []
    for seed in SEEDS:
        rng = np.random.RandomState(3000 + seed)
        stim_on, stim_off, dstar = P12.make_deceptive(rng)
        out = simulate(adj, deg, stim_on, stim_off, seed * 10 + 1, t_pulse, bras, **kw)
        if out is None:
            raise RuntimeError(f"divergence (bras={bras}, t_pulse={t_pulse}, seed={seed})")
        dec, u_m, uf_m = out
        ft = P12.flip_time(dec, dstar)
        flips.append(ft); us.append(u_m); ufs.append(uf_m)
        rows.append({'bras': bras, 't_pulse': t_pulse, 'seed': seed, 'dstar': dstar,
                     'flip_time': ft, 'flip_ok': int(ft <= MAX_BUDGET),
                     'u_mean': u_m, 'u_filter_mean': uf_m,
                     'reglage_filtre': kw.get('filter_frozen_at', ''),
                     'reglage_u': kw.get('u_frozen_at', '')})
    return np.array(flips, float), float(np.mean(us)), float(np.mean(ufs))


def main() -> int:
    t0 = time.time()
    adj = make_lattice_adj(P12.SIDE, periodic=True).astype(float)
    deg = adj.sum(axis=1)
    rows, res = [], {}

    # ------------------------------------------------- G0 : B1 et B2 (BLOQUANT)
    print("=" * 104)
    print("  G0 — FIDELITE : B1 et B2 reproduisent-ils le CSV du 12/07 ?")
    print("=" * 104)
    g0_ok = True
    niveaux = {}
    for bras in ('B1_FULL', 'B2_FIXE_INIT'):
        for tp in T_PULSE_LEVELS:
            f, um, ufm = serie(adj, deg, bras, tp, rows)
            res[(bras, tp)] = f
            if bras == 'B1_FULL':
                niveaux[tp] = (um, ufm)
            ref = G0_REF[bras][tp]
            ok = abs(f.mean() - ref) <= G0_TOL
            g0_ok &= ok
            print(f"  {bras:14s} T_pulse={tp:5d} : flip={f.mean():8.1f} vs {ref:8.1f}  "
                  f"{'OK' if ok else 'ECART ' + format(abs(f.mean() - ref), '.1f')}"
                  f"   [{time.time() - t0:.0f}s]")
    if not g0_ok:
        print("\n  /!\\ G0 ECHOUE. La boucle a diverge du POC du 12/07. RIEN N'EST INTERPRETABLE.")
        return 1
    u_glob = float(np.mean([v[0] for v in niveaux.values()]))
    uf_glob = float(np.mean([v[1] for v in niveaux.values()]))
    print(f"  -> G0 PASSE (8/8, tolerance {G0_TOL} pas).")
    print(f"  Niveaux mesures sur B1 : <u> = {u_glob:.4f} -> couplage naif "
          f"{np.tanh(np.pi * (0.5 - u_glob)) + P12.SOCIAL_LEAKAGE:+.4f} ; "
          f"<u_filter> REEL = {uf_glob:+.4f}  (ecart de Jensen "
          f"{uf_glob - (np.tanh(np.pi * (0.5 - u_glob)) + P12.SOCIAL_LEAKAGE):+.4f})")

    # ------------------------------------------- les trois bras fixes calibres
    print("\n" + "=" * 104)
    print("  LES BRAS FIXES — regles sur les niveaux atteints par B1")
    print("=" * 104)
    for bras, kw_fn in (
            ('B3a_FIXE_NIVEAU_U', lambda tp: {'u_frozen_at': u_glob}),
            ('B3b_FIXE_COUPLAGE', lambda tp: {'filter_frozen_at': uf_glob}),
            ('B3c_FIXE_COUPLAGE_LOCAL', lambda tp: {'filter_frozen_at': niveaux[tp][1]})):
        for tp in T_PULSE_LEVELS:
            f, _, _ = serie(adj, deg, bras, tp, rows, **kw_fn(tp))
            res[(bras, tp)] = f
        moy = float(np.mean([res[(bras, tp)].mean() for tp in T_PULSE_LEVELS]))
        print(f"  {bras:26s} flip moyen = {moy:8.1f}   [{time.time() - t0:.0f}s]")

    # ------------------------------------------------------------- LE TABLEAU
    bras_all = ['B1_FULL', 'B2_FIXE_INIT', 'B3a_FIXE_NIVEAU_U',
                'B3b_FIXE_COUPLAGE', 'B3c_FIXE_COUPLAGE_LOCAL']
    print("\n" + "=" * 104)
    print("  RETARD DE FLIP PAR T_PULSE (pas ; > 9000 = jamais bascule)")
    print("=" * 104)
    print(f"{'T_pulse':>8}" + "".join(f"{b.split('_', 1)[0]:>13}" for b in bras_all)
          + f"{'frac(3b)':>11}{'frac(3c)':>11}")
    summary = []
    for tp in list(T_PULSE_LEVELS) + ['GLOBAL']:
        vals = {b: (float(np.mean([res[(b, t)].mean() for t in T_PULSE_LEVELS]))
                    if tp == 'GLOBAL' else float(res[(b, tp)].mean())) for b in bras_all}
        denom = vals['B1_FULL'] - vals['B2_FIXE_INIT']
        frac = {b: ((vals[b] - vals['B2_FIXE_INIT']) / denom if abs(denom) > 1e-9 else float('nan'))
                for b in ('B3a_FIXE_NIVEAU_U', 'B3b_FIXE_COUPLAGE', 'B3c_FIXE_COUPLAGE_LOCAL')}
        print(f"{str(tp):>8}" + "".join(f"{vals[b]:>13.1f}" for b in bras_all)
              + f"{frac['B3b_FIXE_COUPLAGE']:>11.2f}{frac['B3c_FIXE_COUPLAGE_LOCAL']:>11.2f}")
        summary.append({'t_pulse': tp, **{f'flip_{b}': vals[b] for b in bras_all},
                        'retard_B1_moins_B2': denom,
                        **{f'frac_{b}': frac[b] for b in frac}})

    # ---------------------------------------------------------------- VERDICTS
    g = summary[-1]
    print("\n" + "=" * 104)
    print("  VERDICTS — confrontes a ce qui etait ecrit AVANT")
    print("=" * 104)
    q1 = g['frac_B3b_FIXE_COUPLAGE'] <= Q1_MAX_FRAC
    ecart_q2 = abs(g['flip_B3a_FIXE_NIVEAU_U'] - g['flip_B3b_FIXE_COUPLAGE'])
    q2 = ecart_q2 < Q2_MAX_ECART * g['flip_B2_FIXE_INIT']
    q3 = g['frac_B3c_FIXE_COUPLAGE_LOCAL'] <= Q1_MAX_FRAC
    print(f"  [Q1] {'VERIFIEE' if q1 else 'REJETEE '}  le bras fixe (reglage global) reproduit "
          f"{g['frac_B3b_FIXE_COUPLAGE']:.0%} du retard  (<= {Q1_MAX_FRAC:.0%} attendu)")
    print(f"  [Q2] {'VERIFIEE' if q2 else 'REJETEE '}  |flip(3a) - flip(3b)| = {ecart_q2:.1f} pas "
          f"(< {Q2_MAX_ECART:.0%} de flip(B2) = {Q2_MAX_ECART * g['flip_B2_FIXE_INIT']:.1f} attendu)")
    print(f"  [Q3] {'VERIFIEE' if q3 else 'REJETEE '}  le bras fixe RE-REGLE PAR T_PULSE reproduit "
          f"{g['frac_B3c_FIXE_COUPLAGE_LOCAL']:.0%} du retard  (<= {Q1_MAX_FRAC:.0%} attendu)")
    print()
    if q1 and q3:
        print("  => LE VOLET 2 EST LE DISCRIMINANT. Un couplage fixe, meme regle au mieux,")
        print("     ne reproduit pas la cicatrice. B6 doit etre reformulee sur le TRANSITOIRE.")
    elif not q1:
        print("  => LE VOLET 2 TOMBE AUSSI. Le retard est un effet de NIVEAU de couplage, pas")
        print("     une hysteresis : la « cicatrice de doute » du 12/07 est mal nommee.")
    else:
        print("  => RESULTAT MIXTE : le reglage global echoue mais le reglage local sauve le")
        print("     bras fixe. C'est un probleme de CALIBRATION, pas un discriminant.")

    fig_dir = HERE.parent / 'figures'
    fig_dir.mkdir(exist_ok=True)
    with open(fig_dir / 'b6_third_arm_transient.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(fig_dir / 'b6_third_arm_transient_summary.csv', 'w', newline='',
              encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)
    print(f"\nCSV : {fig_dir / 'b6_third_arm_transient.csv'}")
    print(f"      {fig_dir / 'b6_third_arm_transient_summary.csv'}")
    print(f"Wall time : {time.time() - t0:.1f}s")
    return 0


if __name__ == '__main__':
    sys.exit(main())
