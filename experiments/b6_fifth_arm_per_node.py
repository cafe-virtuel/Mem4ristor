#!/usr/bin/env python3
"""
B6 — LE CINQUIEME BRAS : profil PAR NOEUD. Qu'est-ce qui porte le retard ? — 2026-07-31
========================================================================================
Claude Code (Opus 5) / Julien Chauvin. Derniere porte du protocole B6.

CE QUI EST ETABLI CE SOIR, ET CE QUI NE L'EST PAS
--------------------------------------------------
b6_third_arm_transient.py : aucun couplage CONSTANT ne reproduit le retard de flip du
volet 2 (-21 %, sens oppose), meme re-regle par T_pulse.
b6_fourth_arm_profile.py  : aucune RAMPE PREENREGISTREE non plus (-6 %), et pas par
defaut de calage — le profil moyen <u_filter(t)> lui etait donne EXACT, mesure sur les
memes graines, pour le meme T_pulse. Ma prediction S1 (« le profil moyen suffira »)
est REJETEE.

Consequence, et elle demolit la description du 12/07 autant que la mienne : le retard ne
tient PAS a la trajectoire MOYENNE du couplage. La « cicatrice de doute » n'est donc pas
une hysteresis moyenne. Reste a savoir de quoi elle est faite. Deux candidats, et ce
script les separe :
  (i)  l'HETEROGENEITE SPATIALE — chaque noeud a sa propre trajectoire de couplage ;
  (ii) l'ASYMETRIE ENTRE LES DEUX BRAS de la paire differentielle — le couplage de la
       copie + et celui de la copie - divergent parce qu'elles recoivent +stim et -stim.

LES DEUX BRAS AJOUTES (profils enregistres RUN PAR RUN, jamais moyennes sur les graines :
les masques de stimulation different d'une graine a l'autre, moyenner melangerait les
roles leurre / verite / neutre)
------------------------------------------------------------------------------------
  B5a  PAR_NOEUD_PAR_COPIE  u_filter_i(t) rejoue exactement tel qu'il etait dans B1,
                            separement pour la copie + et la copie -.
                            CE N'EST PAS UN RESULTAT, C'EST UN GATE D'INSTRUMENT :
                            il doit redonner B1 au pas pres.
  B5b  PAR_NOEUD_COMMUN     le MEME profil par noeud, mais MOYENNE ENTRE LES DEUX
                            COPIES et impose aux deux. Garde toute l'heterogeneite
                            spatiale et toute la dynamique temporelle ; supprime la
                            seule asymetrie entre les bras.

CRITERES ET PREDICTIONS — ECRITS AVANT LA MESURE
------------------------------------------------
    frac(X) = (flip(X) - flip(B2)) / (flip(B1) - flip(B2))

  G0  FIDELITE (BLOQUANT). B1 et B2 reproduisent le CSV du 12/07 sur T_pulse
      1500/3000/4500 (tolerance 0.5 pas). T_pulse=500 est EXCLU de toute la campagne :
      le retard y vaut 2.8 pas, toute fraction y explose (artefact deja signale).

  G1  GATE D'INSTRUMENT (BLOQUANT). frac(B5a) doit valoir 1.00 a 0.02 pres.
      Ce que ce gate protege : u n'agit QUE par u_filter, donc rejouer u_filter par
      noeud et par copie doit redonner exactement B1. S'il echoue, soit mon
      enregistrement est decale d'un pas, soit u agit par un canal que je n'ai pas vu.
      Dans les deux cas RIEN de ce qui suit n'est interpretable — et le bras 4 lui-meme
      deviendrait suspect. C'est le controle que les bras 3 et 4 n'avaient pas.

  T1  L'HETEROGENEITE SPATIALE SUFFIT-ELLE ? (prediction RISQUEE, directionnelle)
      Predit VRAI : frac(B5b) >= 0.50.
      Raisonnement : le bras 4 a echoue en ecrasant l'heterogeneite SPATIALE (un seul
      couplage pour les 100 noeuds) ; B5b la restaure entierement. Si l'espace est le
      facteur manquant, B5b doit reproduire l'essentiel du retard.
        - T1 vraie  -> le mecanisme est l'heterogeneite SPATIO-TEMPORELLE du couplage.
          Imitable EN PRINCIPE par un dispositif a couplage programmable noeud par
          noeud — difficile, pas impossible. Le protocole doit le dire, et B6 est
          discriminante contre l'electronique passive, pas contre tout.
        - T1 fausse -> le retard tient a l'ASYMETRIE ENTRE LES DEUX BRAS, c'est-a-dire
          au fait que le couplage repond au signal EFFECTIVEMENT RECU par sa copie.
          Aucun preenregistrement ne peut le faire : il faudrait connaitre le signal
          avant de l'avoir recu.

  PRESOMPTION NEGATIVE, ECRITE AVANT — et elle m'interdit la conclusion qui m'arrange :
  si T1 est fausse, NE PAS en conclure « le doute est donc indispensable ». La mesure
  se fait dans un dispositif de lecture DIFFERENTIELLE, invente le 12/07 pour annuler
  la cicatrice au premier ordre. Un effet porte par l'ecart entre les deux bras
  pourrait etre une propriete du PROTOCOLE DE LECTURE autant que du mecanisme. Il
  faudrait le verifier sur une lecture non differentielle avant d'en faire un argument
  — et cette verification n'est PAS faite ici.

Coeur NON touche. b1d_stno_deceptive_poc.py NON modifie.
SORTIES : figures/b6_fifth_arm_per_node.csv / _summary.csv  (VERSIONNEES)
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
T_PULSES = [1500, 3000, 4500]          # T_pulse=500 exclu : retard 2.8 pas, non interpretable

G0_REF = {
    'B1_FULL': {1500: 5102.6, 3000: 6838.6, 4500: 7863.7},
    'B2_FIXE_INIT': {1500: 2611.4, 3000: 4239.6, 4500: 5724.8},
}
G0_TOL = 0.5
G1_TOL = 0.02
T1_MIN_FRAC = 0.50


def _step(a, u, gp, omega, adj, deg, eta, libre, filtre):
    """Copie FIDELE de P12._step_one. `filtre` : None = u_filter calcule depuis u ;
    vecteur de taille N = couplage impose noeud par noeud a ce pas."""
    S = (adj @ a) / deg - a
    abs_s = np.abs(S)
    sigma_for_u = abs_s * P12.GAIN_U if libre else np.zeros_like(u)
    u_filter = (np.tanh(np.pi * (0.5 - u)) + P12.SOCIAL_LEAKAGE) if filtre is None else filtre
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


def simulate(adj, deg, stim_on, stim_off, seed, t_pulse, bras, prof=None, record=False):
    """Paire differentielle de P12.simulate. `prof` = (warm, pos, neg), tableaux
    (WARMUP_STEPS, N) et (MAX_BUDGET, N) de couplages imposes. record=True renvoie
    les profils observes au lieu de les imposer."""
    rng = np.random.default_rng(seed)
    libre = (bras == 'B1_FULL')
    omega = P12.OMEGA0 + rng.normal(0, P12.SIGMA_OMEGA, N)
    phases = rng.uniform(0.0, 2.0 * np.pi, N)
    p_star = (P12.GAMMA_PLUS - P12.GAMMA_MINUS) / (P12.GAMMA_MINUS * P12.Q)
    a = np.sqrt(p_star) * np.exp(1j * phases)
    u = np.full(N, P12.SIGMA_BASELINE)
    inv_sqrt_dt = 1.0 / np.sqrt(DT)
    pw, pp, pn = (None, None, None) if prof is None else prof
    rec_w = np.empty((WARMUP_STEPS, N)) if record else None
    rec_p = np.empty((MAX_BUDGET, N)) if record else None
    rec_n = np.empty((MAX_BUDGET, N)) if record else None

    for k in range(WARMUP_STEPS):
        noise = rng.normal(0.0, P12.SIGMA_NOISE, size=(2, N))
        eta = (noise[0] + 1j * noise[1]) * inv_sqrt_dt
        a, u, uf = _step(a, u, P12.GAMMA_PLUS, omega, adj, deg, eta, libre,
                         None if pw is None else pw[k])
        if record:
            rec_w[k] = uf
    if not np.all(np.isfinite(a)):
        return None

    a_pos, a_neg = a.copy(), a.copy()
    u_pos, u_neg = u.copy(), u.copy()
    dmat = np.empty((MAX_BUDGET, N))

    for t in range(MAX_BUDGET):
        stim = stim_on if t < t_pulse else stim_off
        noise = rng.normal(0.0, P12.SIGMA_NOISE, size=(2, N))
        eta = (noise[0] + 1j * noise[1]) * inv_sqrt_dt
        a_pos, u_pos, uf_p = _step(a_pos, u_pos, P12.GAMMA_PLUS + ISCALE * stim,
                                   omega, adj, deg, eta, libre,
                                   None if pp is None else pp[t])
        a_neg, u_neg, uf_n = _step(a_neg, u_neg, P12.GAMMA_PLUS - ISCALE * stim,
                                   omega, adj, deg, eta, libre,
                                   None if pn is None else pn[t])
        if not (np.all(np.isfinite(a_pos)) and np.all(np.isfinite(a_neg))):
            return None
        dmat[t] = np.abs(a_pos) ** 2 - np.abs(a_neg) ** 2
        if record:
            rec_p[t] = uf_p
            rec_n[t] = uf_n

    csum = np.cumsum(dmat, axis=0)
    dsm = np.empty_like(dmat)
    for t in range(MAX_BUDGET):
        lo = max(0, t - W_READ + 1)
        dsm[t] = (csum[t] - (csum[lo - 1] if lo > 0 else 0.0)) / (t - lo + 1)
    dec = np.where(dsm.mean(axis=1) >= 0, 1, -1).astype(int)
    return (dec, (rec_w, rec_p, rec_n)) if record else (dec, None)


def main() -> int:
    t0 = time.time()
    adj = make_lattice_adj(P12.SIDE, periodic=True).astype(float)
    deg = adj.sum(axis=1)
    rows = []
    flips = {b: {tp: [] for tp in T_PULSES}
             for b in ('B1_FULL', 'B2_FIXE_INIT', 'B5a_PAR_NOEUD_PAR_COPIE',
                       'B5b_PAR_NOEUD_COMMUN')}

    print("=" * 100)
    print("  CAMPAGNE — B1 enregistre son couplage par noeud, B5a/B5b le rejouent (graine par graine)")
    print("=" * 100)
    for tp in T_PULSES:
        for seed in SEEDS:
            rng = np.random.RandomState(3000 + seed)
            stim_on, stim_off, dstar = P12.make_deceptive(rng)
            args = (adj, deg, stim_on, stim_off, seed * 10 + 1, tp)

            out = simulate(*args, 'B1_FULL', record=True)
            if out is None:
                raise RuntimeError(f"divergence B1 (tp={tp}, seed={seed})")
            dec, (pw, pp, pn) = out
            flips['B1_FULL'][tp].append(P12.flip_time(dec, dstar))

            dec, _ = simulate(*args, 'B2_FIXE_INIT')
            flips['B2_FIXE_INIT'][tp].append(P12.flip_time(dec, dstar))

            dec, _ = simulate(*args, 'B5a_PAR_NOEUD_PAR_COPIE', prof=(pw, pp, pn))
            flips['B5a_PAR_NOEUD_PAR_COPIE'][tp].append(P12.flip_time(dec, dstar))

            moy = 0.5 * (pp + pn)
            dec, _ = simulate(*args, 'B5b_PAR_NOEUD_COMMUN', prof=(pw, moy, moy))
            flips['B5b_PAR_NOEUD_COMMUN'][tp].append(P12.flip_time(dec, dstar))

            for b in flips:
                rows.append({'bras': b, 't_pulse': tp, 'seed': seed, 'dstar': dstar,
                             'flip_time': flips[b][tp][-1]})
        print(f"  T_pulse={tp:5d} termine  [{time.time() - t0:.0f}s]")

    m = {b: {tp: float(np.mean(flips[b][tp])) for tp in T_PULSES} for b in flips}

    print("\n" + "=" * 100)
    print("  G0 — FIDELITE au CSV du 12/07")
    print("=" * 100)
    g0_ok = True
    for b in ('B1_FULL', 'B2_FIXE_INIT'):
        for tp in T_PULSES:
            ok = abs(m[b][tp] - G0_REF[b][tp]) <= G0_TOL
            g0_ok &= ok
            print(f"  {b:14s} T_pulse={tp:5d} : {m[b][tp]:8.1f} vs {G0_REF[b][tp]:8.1f}  "
                  f"{'OK' if ok else 'ECART'}")
    if not g0_ok:
        print("\n  /!\\ G0 ECHOUE. RIEN N'EST INTERPRETABLE.")
        return 1
    print("  -> G0 PASSE (6/6).")

    print("\n" + "=" * 100)
    print("  FRACTION DU RETARD REPRODUITE")
    print("=" * 100)
    print(f"{'T_pulse':>8}{'B1':>10}{'B2':>10}{'B5a':>10}{'B5b':>10}"
          f"{'frac(5a)':>11}{'frac(5b)':>11}")
    summary, f5a_all, f5b_all = [], [], []
    for tp in T_PULSES:
        den = m['B1_FULL'][tp] - m['B2_FIXE_INIT'][tp]
        f5a = (m['B5a_PAR_NOEUD_PAR_COPIE'][tp] - m['B2_FIXE_INIT'][tp]) / den
        f5b = (m['B5b_PAR_NOEUD_COMMUN'][tp] - m['B2_FIXE_INIT'][tp]) / den
        f5a_all.append(f5a); f5b_all.append(f5b)
        print(f"{tp:>8}{m['B1_FULL'][tp]:>10.1f}{m['B2_FIXE_INIT'][tp]:>10.1f}"
              f"{m['B5a_PAR_NOEUD_PAR_COPIE'][tp]:>10.1f}{m['B5b_PAR_NOEUD_COMMUN'][tp]:>10.1f}"
              f"{f5a:>11.3f}{f5b:>11.3f}")
        summary.append({'t_pulse': tp, 'retard_B1_moins_B2': den,
                        **{f'flip_{b}': m[b][tp] for b in flips},
                        'frac_B5a': f5a, 'frac_B5b': f5b})
    g5a, g5b = float(np.mean(f5a_all)), float(np.mean(f5b_all))
    summary.append({'t_pulse': 'GLOBAL', 'frac_B5a': g5a, 'frac_B5b': g5b})

    print("\n" + "=" * 100)
    print("  VERDICTS — confrontes a ce qui etait ecrit AVANT")
    print("=" * 100)
    g1 = abs(g5a - 1.0) <= G1_TOL
    t1 = g5b >= T1_MIN_FRAC
    print(f"  [G1] {'PASSE  ' if g1 else 'ECHOUE '} gate d'instrument : frac(B5a) = {g5a:.3f} "
          f"(1.000 +/- {G1_TOL} attendu)")
    if not g1:
        print("\n  /!\\ LE GATE D'INSTRUMENT ECHOUE : rejouer u_filter par noeud et par copie ne")
        print("      redonne PAS B1. Soit l'enregistrement est decale, soit u agit par un canal")
        print("      que je n'ai pas vu. T1 N'EST PAS INTERPRETABLE — et le bras 4 devient suspect.")
        t1_txt = 'NON INTERPRETABLE (G1 echoue)'
    else:
        t1_txt = 'VERIFIEE' if t1 else 'REJETEE '
    print(f"  [T1] {t1_txt}  l'heterogeneite spatiale seule reproduit {g5b:.0%} du retard "
          f"(>= {T1_MIN_FRAC:.0%} attendu)")
    print()
    if g1 and t1:
        print("  => LE MECANISME EST L'HETEROGENEITE SPATIO-TEMPORELLE du couplage. Un dispositif")
        print("     a couplage programmable noeud par noeud pourrait l'imiter : B6 discrimine")
        print("     contre l'electronique passive, PAS contre tout. A ecrire tel quel.")
    elif g1 and not t1:
        print("  => LE RETARD TIENT A L'ASYMETRIE ENTRE LES DEUX BRAS : le couplage doit repondre")
        print("     au signal EFFECTIVEMENT RECU. Aucun preenregistrement ne peut le faire.")
        print("     RESERVE OBLIGATOIRE (ecrite avant la mesure) : la lecture est DIFFERENTIELLE.")
        print("     Un effet porte par l'ecart entre bras peut tenir au PROTOCOLE DE LECTURE")
        print("     autant qu'au mecanisme. Non verifie ici — ne pas en faire un argument seul.")

    fig_dir = HERE.parent / 'figures'
    fig_dir.mkdir(exist_ok=True)
    with open(fig_dir / 'b6_fifth_arm_per_node.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    keys = sorted({k for s in summary for k in s})
    with open(fig_dir / 'b6_fifth_arm_per_node_summary.csv', 'w', newline='',
              encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader(); w.writerows(summary)
    print(f"\nCSV : {fig_dir / 'b6_fifth_arm_per_node.csv'}")
    print(f"      {fig_dir / 'b6_fifth_arm_per_node_summary.csv'}")
    print(f"Wall time : {time.time() - t0:.1f}s")
    return 0


if __name__ == '__main__':
    sys.exit(main())
