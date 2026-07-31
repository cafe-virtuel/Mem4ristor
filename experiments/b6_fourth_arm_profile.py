#!/usr/bin/env python3
"""
B6 — LE QUATRIEME BRAS : une RAMPE PREENREGISTREE suffit-elle ? — 2026-07-31 (soir)
====================================================================================
Claude Code (Opus 5) / Julien Chauvin. Ferme le dernier verrou du protocole B6.

L'OBJECTION A LAQUELLE CE SCRIPT REPOND
----------------------------------------
b6_third_arm_transient.py a etabli ce soir qu'aucun couplage CONSTANT ne reproduit le
retard de flip du volet 2 : le bras fixe en reproduit -21 %, c'est-a-dire qu'il part
dans le SENS OPPOSE (il accelere la recuperation la ou le doute la retarde), et cela
meme RE-REGLE pour chaque T_pulse (Q3 verifiee).

Objection restante, et elle est evidente pour n'importe quel relecteur : « il suffit de
programmer une RAMPE ». Un couplage VARIABLE DANS LE TEMPS mais PREENREGISTRE — un
generateur de fonctions suffit, aucune variable d'etat, aucun doute. Si une rampe
reproduit le retard, tout le resultat de ce soir retombe.

LES DEUX BRAS AJOUTES
---------------------
  B4a  PROFIL_LOCAL      u_filter(t) = <u_filter(t)>_graines mesure sur B1 POUR CE
                         T_pulse, rejoue tel quel. Le cadeau MAXIMAL au bras
                         preenregistre : il suppose qu'on connait deja la duree du
                         leurre, ce qu'aucun dispositif ne sait.
  B4b  PROFIL_TRANSFERE  le MEME profil, mesure a T_pulse=1500, applique a 3000 et
                         4500. C'est ce qu'un laboratoire peut reellement cabler :
                         une rampe reglee une fois, sur un leurre dont il ne connait
                         pas l'horaire.

Le profil couvre warmup + mesure (10000 pas) : le dispositif a son couplage des
l'allumage, il ne se met pas en marche au debut de la stimulation.

CRITERES ET PREDICTIONS — ECRITS AVANT LA MESURE
------------------------------------------------
On reprend la mesure de b6_third_arm_transient :
    frac(X) = (flip(X) - flip(B2)) / (flip(B1) - flip(B2))
  frac = 1 : reproduit tout le retard ; frac = 0 : n'en reproduit rien ;
  frac < 0 : part dans le sens oppose.

  G0  FIDELITE (BLOQUANT). B1 et B2 doivent reproduire le CSV du 12/07 au pas pres
      (memes references qu'il y a une heure, deja verifiees 8/8 au dixieme de pas).
      Tolerance 0.5 pas. Si G0 echoue, RIEN n'est interpretable.

  S1  LE PROFIL LOCAL REPRODUIT-IL LE RETARD ?   Predit VRAI : frac(B4a) >= 0.50.
      Raisonnement : le profil moyen contient TOUTE l'information temporelle du
      mecanisme — u monte pendant le conflit, redescend apres. S'il ne suffit pas,
      c'est que le retard ne tient pas a la trajectoire MOYENNE de u.
      C'est la prediction qui m'ARRANGE LE MOINS : si S1 est vraie, ce qui est
      necessaire est une modulation temporelle, PAS l'adaptativite. Je l'ecris quand
      meme parce que c'est ce que je crois.

  S2  LE PROFIL TRANSFERE TIENT-IL ? (prediction RISQUEE — c'est elle qui decide)
      Predit FAUX : frac(B4b) <= 0.50, a T_pulse=3000 ET 4500.
      Raisonnement : le profil est cale sur l'horaire d'un leurre de 1500 pas ;
      applique a un leurre qui dure deux ou trois fois plus, il coupe le couplage
      trop tot et le rend trop tot.
        - S2 vraie -> LA PREDICTION B6 EST DEFINITIVEMENT DISCRIMINANTE : il faut
          que le couplage reponde au conflit QUAND IL ARRIVE, ce qu'aucune rampe
          preenregistree ne peut faire sans connaitre l'horaire a l'avance.
        - S2 fausse -> une rampe suffit. Le doute n'est pas necessaire, meme au
          volet 2, et il faut l'ecrire sans chercher une autre observable.

  S3  CONTROLE ADJACENT, comme au volet 1 : le profil transfere se juge CONTRE le
      profil local dans la MEME condition (ratio frac(B4b)/frac(B4a)), pas dans
      l'absolu. Sans ce rapport on confondrait « la condition est plus dure » avec
      « le reglage ne transfere pas » — c'est le defaut que P3 a corrige ce soir.

  PRESOMPTION NEGATIVE, ECRITE AVANT : si meme le profil LOCAL echoue (S1 fausse),
  alors ce n'est pas la trajectoire MOYENNE de u qui porte le retard, mais sa
  variabilite PAR NOEUD et PAR RUN. Cela rendrait le mecanisme encore plus difficile
  a imiter — donc encore plus discriminant — MAIS cela voudrait aussi dire que mon
  modele mental (« une hysteresis moyenne ») est FAUX. Le rapporter tel quel, sans
  encaisser le benefice sans payer le cout.

Coeur NON touche. b1d_stno_deceptive_poc.py NON modifie.
SORTIES : figures/b6_fourth_arm_profile.csv / _summary.csv  (VERSIONNEES)
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

N, DT, MAX_BUDGET = P12.N, P12.DT, P12.MAX_BUDGET
SEEDS, WARMUP_STEPS = P12.SEEDS, P12.WARMUP_STEPS
W_READ, ISCALE = P12.W_READ, P12.ISCALE
T_PULSE_LEVELS = P12.T_PULSE_LEVELS
TP_REGLAGE = 1500                      # le T_pulse sur lequel la rampe est calee
TP_CIBLES = [3000, 4500]               # les T_pulse ou on l'applique sans re-regler

G0_REF = {
    'B1_FULL': {500: 1294.0, 1500: 5102.6, 3000: 6838.6, 4500: 7863.7},
    'B2_FIXE_INIT': {500: 1291.2, 1500: 2611.4, 3000: 4239.6, 4500: 5724.8},
}
G0_TOL = 0.5
S_MAX_FRAC = 0.50


def _step(a, u, gp, omega, adj, deg, eta, libre, filtre):
    """Copie FIDELE de P12._step_one. `filtre` : None = u_filter calcule depuis u ;
    scalaire = couplage impose a ce pas (c'est par la que passe la rampe)."""
    S = (adj @ a) / deg - a
    abs_s = np.abs(S)
    sigma_for_u = abs_s * P12.GAIN_U if libre else np.zeros_like(u)
    if filtre is None:
        u_filter = np.tanh(np.pi * (0.5 - u)) + P12.SOCIAL_LEAKAGE
    else:
        u_filter = np.full_like(u, filtre)
    p = np.abs(a) ** 2
    growth = gp - P12.GAMMA_MINUS * (1.0 + P12.Q * p)
    da = (growth + 1j * omega) * a + P12.K_COUPLING * u_filter * S + eta
    if libre:
        sigma_safe = np.clip(sigma_for_u, 0.0, 100.0)
        eps_adapt = P12.EPSILON_U * np.clip(
            1.0 + P12.ALPHA_SURPRISE * sigma_safe, 1.0, P12.SURPRISE_CAP)
        du = eps_adapt * (P12.K_U * sigma_for_u + P12.SIGMA_BASELINE - u) / P12.TAU_U
        u = np.clip(u + du * DT, 0.0, 1.0)
    return a + da * DT, u, float(u_filter.mean())


def simulate(adj, deg, stim_on, stim_off, seed, t_pulse, bras, profil=None):
    """Paire differentielle de P12.simulate. `profil` : tableau de longueur
    WARMUP_STEPS + MAX_BUDGET donnant le couplage impose a chaque pas (bras 4).
    Retourne (dec, profil_mesure) — le profil mesure ne sert que pour B1."""
    rng = np.random.default_rng(seed)
    libre = (bras == 'B1_FULL')
    omega = P12.OMEGA0 + rng.normal(0, P12.SIGMA_OMEGA, N)
    phases = rng.uniform(0.0, 2.0 * np.pi, N)
    p_star = (P12.GAMMA_PLUS - P12.GAMMA_MINUS) / (P12.GAMMA_MINUS * P12.Q)
    a = np.sqrt(p_star) * np.exp(1j * phases)
    u = np.full(N, P12.SIGMA_BASELINE)
    inv_sqrt_dt = 1.0 / np.sqrt(DT)
    uf_traj = np.empty(WARMUP_STEPS + MAX_BUDGET)

    for k in range(WARMUP_STEPS):
        noise = rng.normal(0.0, P12.SIGMA_NOISE, size=(2, N))
        eta = (noise[0] + 1j * noise[1]) * inv_sqrt_dt
        a, u, uf = _step(a, u, P12.GAMMA_PLUS, omega, adj, deg, eta, libre,
                         None if profil is None else float(profil[k]))
        uf_traj[k] = uf
    if not np.all(np.isfinite(a)):
        return None

    a_pos, a_neg = a.copy(), a.copy()
    u_pos, u_neg = u.copy(), u.copy()
    dmat = np.empty((MAX_BUDGET, N))

    for t in range(MAX_BUDGET):
        stim = stim_on if t < t_pulse else stim_off
        noise = rng.normal(0.0, P12.SIGMA_NOISE, size=(2, N))
        eta = (noise[0] + 1j * noise[1]) * inv_sqrt_dt
        ff = None if profil is None else float(profil[WARMUP_STEPS + t])
        a_pos, u_pos, uf_p = _step(a_pos, u_pos, P12.GAMMA_PLUS + ISCALE * stim,
                                   omega, adj, deg, eta, libre, ff)
        a_neg, u_neg, uf_n = _step(a_neg, u_neg, P12.GAMMA_PLUS - ISCALE * stim,
                                   omega, adj, deg, eta, libre, ff)
        if not (np.all(np.isfinite(a_pos)) and np.all(np.isfinite(a_neg))):
            return None
        dmat[t] = np.abs(a_pos) ** 2 - np.abs(a_neg) ** 2
        uf_traj[WARMUP_STEPS + t] = 0.5 * (uf_p + uf_n)

    csum = np.cumsum(dmat, axis=0)
    dsm = np.empty_like(dmat)
    for t in range(MAX_BUDGET):
        lo = max(0, t - W_READ + 1)
        dsm[t] = (csum[t] - (csum[lo - 1] if lo > 0 else 0.0)) / (t - lo + 1)
    return np.where(dsm.mean(axis=1) >= 0, 1, -1).astype(int), uf_traj


def serie(adj, deg, bras, t_pulse, rows, profil=None):
    """Les 12 graines. Retourne (flips, profil moyen sur les graines)."""
    flips, profils = [], []
    for seed in SEEDS:
        rng = np.random.RandomState(3000 + seed)
        stim_on, stim_off, dstar = P12.make_deceptive(rng)
        out = simulate(adj, deg, stim_on, stim_off, seed * 10 + 1, t_pulse, bras, profil)
        if out is None:
            raise RuntimeError(f"divergence (bras={bras}, t_pulse={t_pulse}, seed={seed})")
        dec, uf_traj = out
        ft = P12.flip_time(dec, dstar)
        flips.append(ft); profils.append(uf_traj)
        rows.append({'bras': bras, 't_pulse': t_pulse, 'seed': seed, 'dstar': dstar,
                     'flip_time': ft, 'flip_ok': int(ft <= MAX_BUDGET)})
    return np.array(flips, float), np.mean(profils, axis=0)


def main() -> int:
    t0 = time.time()
    adj = make_lattice_adj(P12.SIDE, periodic=True).astype(float)
    deg = adj.sum(axis=1)
    rows, res, profils = [], {}, {}

    print("=" * 100)
    print("  G0 — FIDELITE : B1 et B2 reproduisent-ils le CSV du 12/07 ?  (et B1 livre les profils)")
    print("=" * 100)
    g0_ok = True
    for bras in ('B1_FULL', 'B2_FIXE_INIT'):
        for tp in T_PULSE_LEVELS:
            f, prof = serie(adj, deg, bras, tp, rows)
            res[(bras, tp)] = f
            if bras == 'B1_FULL':
                profils[tp] = prof
            ref = G0_REF[bras][tp]
            ok = abs(f.mean() - ref) <= G0_TOL
            g0_ok &= ok
            print(f"  {bras:14s} T_pulse={tp:5d} : flip={f.mean():8.1f} vs {ref:8.1f}  "
                  f"{'OK' if ok else 'ECART'}   [{time.time() - t0:.0f}s]")
    if not g0_ok:
        print("\n  /!\\ G0 ECHOUE. RIEN N'EST INTERPRETABLE.")
        return 1
    print(f"  -> G0 PASSE (8/8, tolerance {G0_TOL} pas).")

    p = profils[TP_REGLAGE]
    print(f"\n  Profil cale sur T_pulse={TP_REGLAGE} : couplage de {p.min():+.4f} a {p.max():+.4f} ;"
          f" avant stim {p[WARMUP_STEPS - 1]:+.4f}, a la fin {p[-1]:+.4f}")

    print("\n" + "=" * 100)
    print("  LES DEUX BRAS PREENREGISTRES")
    print("=" * 100)
    for tp in T_PULSE_LEVELS:
        f, _ = serie(adj, deg, 'B4a_PROFIL_LOCAL', tp, rows, profil=profils[tp])
        res[('B4a_PROFIL_LOCAL', tp)] = f
        print(f"  B4a profil LOCAL       T_pulse={tp:5d} : flip={f.mean():8.1f}"
              f"   [{time.time() - t0:.0f}s]")
    for tp in TP_CIBLES:
        f, _ = serie(adj, deg, 'B4b_PROFIL_TRANSFERE', tp, rows, profil=p)
        res[('B4b_PROFIL_TRANSFERE', tp)] = f
        print(f"  B4b profil TRANSFERE   T_pulse={tp:5d} : flip={f.mean():8.1f}"
              f"   [{time.time() - t0:.0f}s]")

    print("\n" + "=" * 100)
    print("  FRACTION DU RETARD REPRODUITE   (1.00 = tout ; 0 = rien ; < 0 = sens oppose)")
    print("=" * 100)
    print(f"{'T_pulse':>8}{'B1':>10}{'B2':>10}{'B4a local':>12}{'B4b transf.':>13}"
          f"{'frac(4a)':>10}{'frac(4b)':>10}")
    summary = []
    for tp in T_PULSE_LEVELS:
        b1, b2 = res[('B1_FULL', tp)].mean(), res[('B2_FIXE_INIT', tp)].mean()
        b4a = res[('B4a_PROFIL_LOCAL', tp)].mean()
        den = b1 - b2
        f4a = (b4a - b2) / den if abs(den) > 1e-9 else float('nan')
        b4b = res[('B4b_PROFIL_TRANSFERE', tp)].mean() if tp in TP_CIBLES else float('nan')
        f4b = (b4b - b2) / den if (tp in TP_CIBLES and abs(den) > 1e-9) else float('nan')
        # T_pulse=500 : le retard vaut ~2.8 pas, toute fraction y explose -> non cite.
        note = '  <- retard quasi nul, fractions NON INTERPRETABLES' if abs(den) < 50 else ''
        print(f"{tp:>8}{b1:>10.1f}{b2:>10.1f}{b4a:>12.1f}{b4b:>13.1f}{f4a:>10.2f}{f4b:>10.2f}{note}")
        summary.append({'t_pulse': tp, 'flip_B1': b1, 'flip_B2': b2, 'flip_B4a': b4a,
                        'flip_B4b': b4b, 'retard_B1_moins_B2': den,
                        'frac_B4a': f4a, 'frac_B4b': f4b})

    # GLOBAL restreint aux T_pulse ou un retard existe (>= 1500), comme le volet 2.
    utiles = [s for s in summary if s['retard_B1_moins_B2'] >= 50]
    cibles = [s for s in summary if s['t_pulse'] in TP_CIBLES]
    g_f4a = float(np.mean([s['frac_B4a'] for s in utiles]))
    g_f4b = float(np.mean([s['frac_B4b'] for s in cibles]))
    g_f4a_cibles = float(np.mean([s['frac_B4a'] for s in cibles]))
    summary.append({'t_pulse': 'GLOBAL_utiles', 'frac_B4a': g_f4a, 'frac_B4b': g_f4b})

    print("\n" + "=" * 100)
    print("  VERDICTS — confrontes a ce qui etait ecrit AVANT")
    print("=" * 100)
    s1 = g_f4a >= S_MAX_FRAC
    s2 = g_f4b <= S_MAX_FRAC
    ratio = g_f4b / g_f4a_cibles if abs(g_f4a_cibles) > 1e-9 else float('nan')
    print(f"  [S1] {'VERIFIEE' if s1 else 'REJETEE '}  le profil LOCAL reproduit {g_f4a:.0%} du "
          f"retard (>= {S_MAX_FRAC:.0%} attendu)  [T_pulse >= 1500]")
    print(f"  [S2] {'VERIFIEE' if s2 else 'REJETEE '}  le profil TRANSFERE reproduit {g_f4b:.0%} du "
          f"retard (<= {S_MAX_FRAC:.0%} attendu)  [T_pulse 3000 et 4500]")
    print(f"  [S3] contexte : sur les MEMES T_pulse, transfere/local = {ratio:.0%} "
          f"({g_f4b:.0%} contre {g_f4a_cibles:.0%})")
    print()
    if s1 and s2:
        print("  => B6 EST DEFINITIVEMENT DISCRIMINANTE. Une rampe preenregistree reproduit le")
        print("     retard SI on lui donne l'horaire du leurre, et echoue sans lui. Le couplage")
        print("     doit repondre au conflit QUAND IL ARRIVE — c'est cela que le doute apporte.")
    elif not s1:
        print("  => S1 REJETEE : la trajectoire MOYENNE de u ne porte pas le retard. Mon modele")
        print("     mental (« une hysteresis moyenne ») est FAUX. Le mecanisme tient a la")
        print("     variabilite PAR NOEUD et PAR RUN — plus dur a imiter, mais mal decrit jusqu'ici.")
    else:
        print("  => S2 REJETEE : une rampe reglee une fois suffit, meme sans connaitre l'horaire.")
        print("     Le doute n'est pas necessaire au volet 2 non plus. A ecrire tel quel.")

    fig_dir = HERE.parent / 'figures'
    fig_dir.mkdir(exist_ok=True)
    with open(fig_dir / 'b6_fourth_arm_profile.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    keys = sorted({k for s in summary for k in s})
    with open(fig_dir / 'b6_fourth_arm_profile_summary.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader(); w.writerows(summary)
    np.savetxt(fig_dir / 'b6_fourth_arm_profil_tp1500.csv', profils[TP_REGLAGE],
               delimiter=',', header='u_filter_moyen_par_pas (warmup 1000 + mesure 9000)')
    print(f"\nCSV : {fig_dir / 'b6_fourth_arm_profile.csv'}")
    print(f"      {fig_dir / 'b6_fourth_arm_profile_summary.csv'}")
    print(f"      {fig_dir / 'b6_fourth_arm_profil_tp1500.csv'}")
    print(f"Wall time : {time.time() - t0:.1f}s")
    return 0


if __name__ == '__main__':
    sys.exit(main())
