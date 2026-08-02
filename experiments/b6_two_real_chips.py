#!/usr/bin/env python3
"""
B6 — DEUX PUCES REELLES : la derniere idealisation de la lecture differentielle — 2026-08-02
=============================================================================================
Claude Code (Opus 5) / Julien Chauvin. Suite directe de b6_nondifferential_readout.py.

CE QUI RESTE A TESTER, ET POURQUOI C'EST LE DERNIER MORCEAU
------------------------------------------------------------
Le 02/08 au matin, la reserve principale du volet 2 a ete REDUITE : le BRUIT COMMUN aux
deux copies — la seule idealisation qu'aucun laboratoire ne peut realiser — ne porte
rien du tout (retard conserve a 103 % avec des bruits independants). Le volet 2 n'est
donc pas un artefact du protocole de lecture.

Mais deux idealisations subsistaient, ecrites noir sur blanc avant de conclure : dans
cette mesure, les deux copies partagent encore
  (i)  leur ETAT INITIAL (un seul warmup, copie aux deux bras) ;
  (ii) leurs FREQUENCES PROPRES omega (un seul tirage, copie aux deux bras).
Deux puces sorties d'une vraie fabrication n'ont ni l'un ni l'autre. (ii) est la plus
serieuse : c'est le DESAPPARIEMENT DE FABRICATION, et il produit un ecart de puissance
STATIQUE entre les deux reseaux, la ou le bruit ne produisait qu'un ecart fluctuant.
L'evidence a lire vaut ~0.013 ; un desappariement de frequences pourrait la noyer.

L'ECHELLE, UN SEUL CHANGEMENT A LA FOIS (chaque bras ne differe du precedent que par
un point ; toutes les lectures sont APPARIEES run par run)
----------------------------------------------------------------------------------
    lecture   etat initial   omega       bruit          ce que c'est
    L0        commun         commun      PARTAGE        la lecture du 12/07 (gate)
    L1        commun         commun      independant    acquis du 02/08 matin (gate)
    L4        INDEPENDANT    commun      independant    deux puces, fabrication parfaite
    L5        INDEPENDANT    INDEPENDANT independant    DEUX PUCES REELLES, brut
    L5c       INDEPENDANT    INDEPENDANT independant    L5 + NULLING de l'offset

L5c est L5 avec la calibration qu'un laboratoire ferait de toute facon : mesurer l'ecart
differentiel AVANT d'appliquer la stimulation, et le soustraire. C'est du nulling
d'offset standard, il n'exige aucune connaissance de la tache. La question utile n'est
donc pas « L5 brut marche-t-il ? » mais « la calibration suffit-elle a le rattraper ? ».

DEUX DEFAUTS DE MA PROPRE METRIQUE, TROUVES CE MATIN ET CORRIGES ICI
--------------------------------------------------------------------
  (a) UNE LECTURE EN PANNE GONFLE LE RETARD QU'ELLE MESURE. Un run qui ne bascule
      jamais compte flip = MAX_BUDGET+1 = 9001, soit la valeur MAXIMALE possible. La
      metrique « part du retard reproduite » est donc monotone dans le mauvais sens des
      que la precision s'effondre : ce matin L2 « conservait 72 % » du retard alors que
      ses 3 graines sur 3 a d*=+1 ne recuperaient jamais.
      -> CORRECTION : frac(X) est calculee sur le SOUS-ENSEMBLE APPARIE de X, c'est-a-dire
         les graines qui basculent dans les DEUX conditions pour X *et* pour L0. Ce
         sous-ensemble est immunise a la censure par construction. La valeur brute (toutes
         graines) est affichee a cote, et l'ecart entre les deux est lui-meme un
         diagnostic. Si le sous-ensemble tombe sous SUBSET_MIN graines, la lecture est
         declaree NON INTERPRETABLE — pas « mauvaise », non interpretable.
  (b) UNE GARDE POSEE SUR LE SEUL CONTROLE NE VOIT PAS LA CONDITION D'INTERET. Ma garde
      d'utilisabilite portait sur acc_fin en FROZEN_U seul ; L2 la passait a 0.92 et se
      cassait a 0.67 en FULL.
      -> CORRECTION : la garde porte sur acc_fin >= 0.75 dans LES DEUX conditions.

CRITERES ET PREDICTIONS — ECRITS AVANT LA MESURE, LE 2 AOUT 2026
-----------------------------------------------------------------
  G0  FIDELITE (BLOQUANT). L0 reproduit le CSV du 12/07 sur les 3 T_pulse et les 2
      conditions, a 0.5 pas pres.
  G1  REPRODUCTION DU MATIN (BLOQUANT). L1 doit rendre EXACTEMENT les memes flips que la
      campagne de ce matin : les flux de bruit sont construits pour que L0 et L1 soient
      inchanges par l'ajout des copies D et E (rng separe). Si G1 echoue, l'echelle n'est
      plus appariee et rien n'est comparable au resultat du matin.

  Q1  L'ETAT INITIAL (prediction RISQUEE, je predis VRAI) : frac(L4) >= 0.50.
      Raisonnement : apres 1000 pas de warmup, deux copies partant de phases differentes
      sont deux tirages du meme ensemble ; l'ecart qui en resulte est fluctuant, de meme
      nature que le bruit independant, qui n'a rien coute.

  Q2  LES FREQUENCES PROPRES, BRUT (prediction RISQUEE, je predis FAUX) : frac(L5) >= 0.50.
      Raisonnement : un desappariement d'omega produit un ecart de puissance STATIQUE
      entre les deux reseaux, pas fluctuant. Il ne se moyenne pas dans la fenetre de
      lecture. Il devrait deplacer le zero de la variable de decision et biaiser le signe,
      exactement comme la reference passive ce matin.

  Q3  LA CALIBRATION SUFFIT-ELLE ? (prediction RISQUEE, je predis VRAI) : frac(L5c) >= 0.50.
      C'est la question qui decide de ce qu'on ecrit a un laboratoire.
      - Q3 VRAIE  -> le protocole gagne UNE LIGNE : « annuler l'offset differentiel avant
        stimulation ». Contrainte triviale. Le volet 2 tient sur deux puces ordinaires.
      - Q3 FAUSSE -> le volet 2 exige des puces APPARIEES EN FREQUENCE, ce qui est une
        contrainte de fabrication serieuse, et elle doit etre ecrite en tete du protocole.
        Ce ne serait pas une refutation, mais un cout — et il faut le dire tel quel.

  PRESOMPTION NEGATIVE, ECRITE AVANT : si Q2 est fausse ET Q3 vraie, NE PAS ecrire « le
  desappariement ne pose pas de probleme ». Ce qui sera etabli est plus etroit : un offset
  STATIQUE se calibre. Une DERIVE differentielle (les deux puces s'ecartant lentement au
  cours du run) ne se calibre pas avec une mesure prise avant le run, et ce script ne la
  teste pas — le nulling y est un scalaire mesure au warmup.

CE QUE LA MESURE A RENDU (ajoute APRES la campagne du 2026-08-02, criteres inchanges)
--------------------------------------------------------------------------------------
  G0 / G1  PASSENT 12/12, a la decimale. L0 redonne le CSV du 12/07 et L1 redonne la
  campagne du matin, run par run : l'echelle est appariee, tout est comparable.

  L'ECHELLE COMPLETE (frac appariee, immunisee a la censure) :
      L0  paire jumelle (bruit + etat + omega communs)   1.000   <- l'ideal du 12/07
      L1  + bruits independants                          0.992   <- ne coute RIEN
      L4  + etats initiaux independants                  0.803   <- coute ~20 %
      L5  + omega independants (DEUX PUCES REELLES)      0.612   <- coute ~19 % de plus
      L5c  L5 + nulling de l'offset avant stimulation    0.625

  CE QU'UN LABORATOIRE MESURERAIT VRAIMENT, avec deux puces ordinaires : un retard de
  +34 % au lieu du +52 % publie (par T_pulse : +44 %, +41 %, +23 %). L'effet perd ~40 %
  de son amplitude ideale et reste tres largement au-dessus du mesurable.

  Q1  VERIFIEE. L'etat initial commun n'etait pas essentiel (0.803).

  Q2  REJETEE — ET MON RAISONNEMENT ETAIT FAUX, PAS SEULEMENT MON SEUIL.
      J'avais predit que le desappariement d'omega ferait tomber la lecture sous 50 %,
      par un OFFSET STATIQUE qui deplacerait le zero de la decision et biaiserait le
      signe. Mesure : frac = 0.612, et surtout acc_fin(FULL) = 0.972 — MEILLEURE que
      celle de la paire jumelle (0.944). Il n'y a aucun biais de signe. Le
      desappariement ne fausse pas la decision, il ERODE l'amplitude du signal
      differentiel. J'ai predit la mauvaise DIRECTION de defaillance, et c'est ce qui
      rend le rejet utile : le mode de panne que je redoutais (celui de la reference
      passive ce matin) ne se reproduit pas ici.

  Q3  VERIFIEE — MAIS POUR UNE RAISON QUI N'EST PAS LA MIENNE, DONC A NE PAS COMPTER
      COMME UN SUCCES. La calibration devait RATTRAPER L5. L5 n'avait pas besoin d'etre
      rattrape : 0.612 -> 0.625, soit +0.013. Sur la part du retard, le nulling
      d'offset n'apporte RIEN. Un critere qui passe pour la mauvaise raison est un
      critere en panne, et je l'ecris tel quel.
      Ce que la calibration apporte reellement est ailleurs, et c'est mesure :
      acc_fin(FULL) 0.972 -> 1.000, runs jamais aboutis 1 -> 0, sous-ensemble apparie
      9 -> 10 graines. Elle achete de la ROBUSTESSE, pas de l'amplitude. C'est une
      recommandation de protocole valable, mais pas pour le motif que j'avais ecrit.

  CE QUI EST DONC ETABLI, ET SA BORNE : la reserve « lecture differentielle » du 12/07
  est LEVEE en tant que soupcon d'artefact. Aucune des trois idealisations (bruit, etat
  initial, frequences propres) ne portait l'effet ; leur retrait cumule coute ~40 %
  d'amplitude et ZERO precision. Ce qui reste exige du dispositif est une PAIRE
  +stim / -stim — constructible — et non une paire de jumeaux.
  RESERVE QUI SUBSISTE, ecrite avant la mesure : le nulling teste ici est un SCALAIRE
  mesure au warmup. Une DERIVE differentielle lente au cours du run ne se calibre pas
  par une mesure prise avant, et ce script ne la teste pas.
  Et la borne generale, inchangee : tout ceci reste de la simulation.

Coeur NON touche. b1d_stno_deceptive_poc.py NON modifie (importe seulement).
SORTIES : figures/b6_two_real_chips.csv / _summary.csv  (VERSIONNEES)
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
T_PULSES = [1500, 3000, 4500]
CONDITIONS = ['STNO_FULL', 'STNO_FROZEN_U']
READOUTS = ['L0_JUMELLE', 'L1_BRUIT_INDEP', 'L4_ETAT_INDEP',
            'L5_OMEGA_INDEP', 'L5c_OMEGA_INDEP_CALIBRE']

G0_REF = {
    'STNO_FULL':     {1500: 5102.6, 3000: 6838.6, 4500: 7863.7},
    'STNO_FROZEN_U': {1500: 2611.4, 3000: 4239.6, 4500: 5724.8},
}
# campagne du 02/08 matin (b6_nondifferential_readout.csv), lecture L1 :
G1_REF = {
    'STNO_FULL':     {1500: 5683.5, 3000: 6740.1, 4500: 7711.4},
    'STNO_FROZEN_U': {1500: 2726.8, 3000: 4254.3, 4500: 5717.2},
}
G_TOL = 0.5
U_MIN_ACCFIN = 0.75         # desormais exigee dans LES DEUX conditions
Q_MIN_FRAC = 0.50
SUBSET_MIN = 5              # graines appariees minimales pour interpreter une frac
NOISE_OFFSET = 777_000_000  # flux de bruit de la 2e puce — IDENTIQUE au script du matin
FAB_OFFSET = 424_000_000    # flux de FABRICATION de la 2e puce (omega, phases, warmup)


def _step(a, u, gp, omega, adj, deg, eta, libre):
    """Copie FIDELE de P12._step_one (aucune modification du modele)."""
    S = (adj @ a) / deg - a
    sigma_for_u = np.abs(S) * P12.GAIN_U if libre else np.zeros_like(u)
    u_filter = np.tanh(np.pi * (0.5 - u)) + P12.SOCIAL_LEAKAGE
    p = np.abs(a) ** 2
    growth = gp - P12.GAMMA_MINUS * (1.0 + P12.Q * p)
    da = (growth + 1j * omega) * a + P12.K_COUPLING * u_filter * S + eta
    sigma_safe = np.clip(sigma_for_u, 0.0, 100.0)
    eps_adapt = P12.EPSILON_U * np.clip(
        1.0 + P12.ALPHA_SURPRISE * sigma_safe, 1.0, P12.SURPRISE_CAP)
    du = eps_adapt * (P12.K_U * sigma_for_u + P12.SIGMA_BASELINE - u) / P12.TAU_U
    return a + da * DT, np.clip(u + du * DT, 0.0, 1.0), p


def _roll_mean(vec, w):
    """Moyenne glissante causale sur une serie scalaire (equivalente a P12._roll
    suivi de .mean(axis=1) : les deux operations sont lineaires)."""
    csum = np.cumsum(vec)
    out = np.empty_like(vec)
    for t in range(len(vec)):
        lo = max(0, t - w + 1)
        out[t] = (csum[t] - (csum[lo - 1] if lo > 0 else 0.0)) / (t - lo + 1)
    return out


def _warmup(rng, libre, adj, deg, omega):
    """Un warmup complet (sans stimulation). Retourne (a, u, p_moyen_fin_de_warmup)."""
    phases = rng.uniform(0.0, 2.0 * np.pi, N)
    p_star = (P12.GAMMA_PLUS - P12.GAMMA_MINUS) / (P12.GAMMA_MINUS * P12.Q)
    a = np.sqrt(p_star) * np.exp(1j * phases)
    u = np.full(N, P12.SIGMA_BASELINE)
    inv = 1.0 / np.sqrt(DT)
    tail = []
    for k in range(WARMUP_STEPS):
        noise = rng.normal(0.0, P12.SIGMA_NOISE, size=(2, N))
        a, u, p = _step(a, u, P12.GAMMA_PLUS, omega, adj, deg,
                        (noise[0] + 1j * noise[1]) * inv, libre)
        if k >= WARMUP_STEPS - W_READ:
            tail.append(float((np.abs(a) ** 2).mean()))
    return a, u, float(np.mean(tail))


def simulate_all(adj, deg, stim_on, stim_off, seed, t_pulse, condition):
    """Cinq copies en un run. Les flux sont separes pour que L0 et L1 soient
    STRICTEMENT identiques a ceux du matin (gates G0 et G1) :
      rng  : puce 1 — omega1, phases1, warmup1, bruit de A et B   (flux d'origine P12)
      rng2 : bruit de course de C, D, E                            (identique au matin)
      rng3 : FABRICATION de la puce 2 — omega2, phases2, warmup2   (nouveau, isole)

      A : +stim, etat1, omega1, bruit1
      B : -stim, etat1, omega1, bruit1 (PARTAGE)   -> L0
      C : -stim, etat1, omega1, bruit2             -> L1
      D : -stim, etat2, omega1, bruit2             -> L4
      E : -stim, etat2, omega2, bruit2             -> L5 (et L5c apres nulling)
    """
    libre = (condition == 'STNO_FULL')
    rng = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed + NOISE_OFFSET)
    rng3 = np.random.default_rng(seed + FAB_OFFSET)

    omega1 = P12.OMEGA0 + rng.normal(0, P12.SIGMA_OMEGA, N)
    a1, u1, pw1 = _warmup(rng, libre, adj, deg, omega1)
    if not np.all(np.isfinite(a1)):
        return None
    omega2 = P12.OMEGA0 + rng3.normal(0, P12.SIGMA_OMEGA, N)   # desappariement
    # Les deux warmups de la puce 2 partent du MEME generateur : phases et bruit de
    # warmup identiques. La SEULE difference entre D et E est omega — sans quoi
    # L4 -> L5 changerait deux choses a la fois.
    a2o, u2o, _ = _warmup(np.random.default_rng(seed + FAB_OFFSET + 1),
                          libre, adj, deg, omega1)             # puce 2, omega APPARIE
    a2d, u2d, pw2 = _warmup(np.random.default_rng(seed + FAB_OFFSET + 1),
                            libre, adj, deg, omega2)           # puce 2, omega DESAPPARIE
    if not (np.all(np.isfinite(a2o)) and np.all(np.isfinite(a2d))):
        return None
    off_L5 = pw1 - pw2          # offset differentiel STATIQUE, mesurable avant stim

    aA, uA = a1.copy(), u1.copy()
    aB, uB = a1.copy(), u1.copy()
    aC, uC = a1.copy(), u1.copy()
    aD, uD = a2o.copy(), u2o.copy()
    aE, uE = a2d.copy(), u2d.copy()

    vAB = np.empty(MAX_BUDGET); vAC = np.empty(MAX_BUDGET)
    vAD = np.empty(MAX_BUDGET); vAE = np.empty(MAX_BUDGET)
    inv = 1.0 / np.sqrt(DT)

    for t in range(MAX_BUDGET):
        stim = stim_on if t < t_pulse else stim_off
        n1 = rng.normal(0.0, P12.SIGMA_NOISE, size=(2, N))
        e1 = (n1[0] + 1j * n1[1]) * inv
        n2 = rng2.normal(0.0, P12.SIGMA_NOISE, size=(2, N))
        e2 = (n2[0] + 1j * n2[1]) * inv
        gpp, gpm = P12.GAMMA_PLUS + ISCALE * stim, P12.GAMMA_PLUS - ISCALE * stim

        aA, uA, _ = _step(aA, uA, gpp, omega1, adj, deg, e1, libre)
        aB, uB, _ = _step(aB, uB, gpm, omega1, adj, deg, e1, libre)
        aC, uC, _ = _step(aC, uC, gpm, omega1, adj, deg, e2, libre)
        aD, uD, _ = _step(aD, uD, gpm, omega1, adj, deg, e2, libre)
        aE, uE, _ = _step(aE, uE, gpm, omega2, adj, deg, e2, libre)

        if not all(np.all(np.isfinite(x)) for x in (aA, aB, aC, aD, aE)):
            return None
        qA = float((np.abs(aA) ** 2).mean())
        vAB[t] = qA - float((np.abs(aB) ** 2).mean())
        vAC[t] = qA - float((np.abs(aC) ** 2).mean())
        vAD[t] = qA - float((np.abs(aD) ** 2).mean())
        vAE[t] = qA - float((np.abs(aE) ** 2).mean())

    out = {}
    for name, v, off in (('L0_JUMELLE', vAB, 0.0), ('L1_BRUIT_INDEP', vAC, 0.0),
                         ('L4_ETAT_INDEP', vAD, 0.0), ('L5_OMEGA_INDEP', vAE, 0.0),
                         ('L5c_OMEGA_INDEP_CALIBRE', vAE, off_L5)):
        d = _roll_mean(v, W_READ) - off
        out[name] = np.where(d >= 0, 1, -1).astype(int)
    return out


def main() -> int:
    t0 = time.time()
    adj = make_lattice_adj(P12.SIDE, periodic=True).astype(float)
    deg = adj.sum(axis=1)
    rows = []

    print('=' * 104)
    print('  B6 — DEUX PUCES REELLES (2026-08-02) — criteres ecrits avant la mesure')
    print('=' * 104)
    total = len(CONDITIONS) * len(T_PULSES) * len(SEEDS)
    done = 0
    for cond in CONDITIONS:
        for tp in T_PULSES:
            for seed in SEEDS:
                r = np.random.RandomState(3000 + seed)
                stim_on, stim_off, dstar = P12.make_deceptive(r)
                out = simulate_all(adj, deg, stim_on, stim_off, seed * 10 + 1, tp, cond)
                done += 1
                if out is None:
                    raise RuntimeError(f'divergence ({cond}, tp={tp}, seed={seed})')
                for ro in READOUTS:
                    dec = out[ro]
                    rows.append({'readout': ro, 'condition': cond, 't_pulse': tp,
                                 'seed': seed, 'dstar': dstar,
                                 'flip_time': P12.flip_time(dec, dstar),
                                 'acc_final': int(dec[-1] == dstar)})
            print(f'  {cond:<14} T_pulse={tp:5d}  [{done}/{total}]  {time.time() - t0:.0f}s')

    def sel(**kw):
        return [r for r in rows if all(r[k] == v for k, v in kw.items())]

    def mflip(seeds=None, **kw):
        v = [r for r in sel(**kw) if seeds is None or r['seed'] in seeds]
        return float(np.mean([r['flip_time'] for r in v])) if v else float('nan')

    def maccfin(**kw):
        return float(np.mean([r['acc_final'] for r in sel(**kw)]))

    # --------------------------- G0 et G1 -----------------------------------
    print('\n' + '=' * 104)
    print('  G0 / G1 — FIDELITE (BLOQUANTS) : L0 au CSV du 12/07, L1 a la campagne de ce matin')
    print('=' * 104)
    ok_all = True
    for tag, ro, ref in (('G0', 'L0_JUMELLE', G0_REF), ('G1', 'L1_BRUIT_INDEP', G1_REF)):
        for cond in CONDITIONS:
            for tp in T_PULSES:
                got = mflip(readout=ro, condition=cond, t_pulse=tp)
                ok = abs(got - ref[cond][tp]) <= G_TOL
                ok_all &= ok
                print(f'  [{tag}] {cond:<14} T_pulse={tp:5d} : {got:8.1f} vs '
                      f"{ref[cond][tp]:8.1f}   {'OK' if ok else 'ECART'}")
    if not ok_all:
        print('\n  /!\\ UN GATE DE FIDELITE ECHOUE. RIEN N\'EST INTERPRETABLE. Campagne annulee.')
        return 1
    print('  -> G0 et G1 PASSENT (12/12).')

    # ---------------------- garde U (LES DEUX conditions) --------------------
    print('\n' + '=' * 104)
    print('  U — GARDE D\'UTILISABILITE, corrigee : acc_fin >= 0.75 dans LES DEUX conditions')
    print('=' * 104)
    usable = {}
    for ro in READOUTS:
        aF = maccfin(readout=ro, condition='STNO_FULL')
        aZ = maccfin(readout=ro, condition='STNO_FROZEN_U')
        usable[ro] = aF >= U_MIN_ACCFIN and aZ >= U_MIN_ACCFIN
        print(f'  {ro:<26} acc_fin  FULL={aF:.3f}  FROZEN={aZ:.3f}   '
              f"{'UTILISABLE' if usable[ro] else 'INUTILISABLE'}")

    # ------------- sous-ensembles apparies (immunises a la censure) ----------
    def flips_ok(ro):
        bad = {r['seed'] for r in sel(readout=ro) if r['flip_time'] > MAX_BUDGET}
        return set(SEEDS) - bad

    base = flips_ok('L0_JUMELLE')
    print('\n' + '=' * 104)
    print('  SOUS-ENSEMBLES APPARIES — correction du defaut trouve ce matin')
    print('  (un run qui ne bascule jamais compte flip=9001, donc GONFLE le retard :')
    print('   frac est calculee sur les graines qui basculent partout, L0 inclus)')
    print('=' * 104)
    subset = {}
    for ro in READOUTS:
        s = base & flips_ok(ro)
        subset[ro] = s
        print(f'  {ro:<26} graines appariees avec L0 : {len(s):2d}  {sorted(s)}')

    # ---------------------------- resultats ---------------------------------
    print('\n' + '=' * 104)
    print('  LE RETARD, LECTURE PAR LECTURE')
    print('=' * 104)
    print(f"{'lecture':<26}{'T_pulse':>8}{'FULL':>10}{'FROZEN':>10}{'retard':>9}"
          f"{'accF':>7}{'accZ':>7}{'cens':>6}")
    summary, retard_brut, retard_app = [], {}, {}
    for ro in READOUTS:
        brut, app = [], []
        for tp in T_PULSES:
            ff = mflip(readout=ro, condition='STNO_FULL', t_pulse=tp)
            fz = mflip(readout=ro, condition='STNO_FROZEN_U', t_pulse=tp)
            aF = maccfin(readout=ro, condition='STNO_FULL', t_pulse=tp)
            aZ = maccfin(readout=ro, condition='STNO_FROZEN_U', t_pulse=tp)
            cens = sum(1 for r in sel(readout=ro, t_pulse=tp)
                       if r['flip_time'] > MAX_BUDGET)
            brut.append(ff - fz)
            if len(subset[ro]) >= SUBSET_MIN:
                app.append(mflip(subset[ro], readout=ro, condition='STNO_FULL', t_pulse=tp)
                           - mflip(subset[ro], readout=ro, condition='STNO_FROZEN_U',
                                   t_pulse=tp))
            print(f'{ro:<26}{tp:>8}{ff:>10.1f}{fz:>10.1f}{ff - fz:>9.1f}'
                  f'{aF:>7.2f}{aZ:>7.2f}{cens:>6d}')
            summary.append({'readout': ro, 't_pulse': tp, 'flip_full': ff,
                            'flip_frozen': fz, 'retard_brut': ff - fz,
                            'accfin_full': aF, 'accfin_frozen': aZ, 'n_censures': cens})
        retard_brut[ro] = float(np.mean(brut))
        retard_app[ro] = float(np.mean(app)) if app else float('nan')
        print(f"{'':26}{'MOYEN':>8}{'':>10}{'':>10}{retard_brut[ro]:>9.1f}"
              f'   (apparie : {retard_app[ro]:.1f})')

    print('\n' + '=' * 104)
    print('  PART DU RETARD CONSERVEE  —  la colonne « apparie » est celle qui decide')
    print('=' * 104)
    print(f"{'lecture':<26}{'frac brute':>12}{'frac appariee':>15}{'n':>5}  statut")
    frac = {}
    for ro in READOUTS:
        fb = retard_brut[ro] / retard_brut['L0_JUMELLE']
        fa = retard_app[ro] / retard_app['L0_JUMELLE']
        frac[ro] = fa
        interp = usable[ro] and len(subset[ro]) >= SUBSET_MIN
        print(f'{ro:<26}{fb:>12.3f}{fa:>15.3f}{len(subset[ro]):>5d}  '
              f"{'' if interp else 'NON INTERPRETABLE'}")
        summary.append({'readout': ro, 't_pulse': 'MOYEN',
                        'retard_brut': retard_brut[ro], 'retard_apparie': retard_app[ro],
                        'frac_brute': fb, 'frac_appariee': fa,
                        'n_graines_appariees': len(subset[ro]),
                        'utilisable': int(usable[ro])})

    # ------------------------------ verdicts --------------------------------
    print('\n' + '=' * 104)
    print('  VERDICTS — confrontes a ce qui etait ecrit AVANT la mesure')
    print('=' * 104)

    def verdict(nom, ro, predite):
        ok = (usable[ro] and len(subset[ro]) >= SUBSET_MIN
              and frac[ro] >= Q_MIN_FRAC)
        conforme = (ok == predite)
        print(f"  [{nom}] predite {'VRAIE ' if predite else 'FAUSSE'} -> "
              f"{'VERIFIEE' if conforme else 'REJETEE '}   "
              f'frac({ro}) = {frac[ro]:.3f}  '
              f"(utilisable={usable[ro]}, n={len(subset[ro])})")
        return ok

    q1 = verdict('Q1', 'L4_ETAT_INDEP', True)
    q2 = verdict('Q2', 'L5_OMEGA_INDEP', False)
    q3 = verdict('Q3', 'L5c_OMEGA_INDEP_CALIBRE', True)
    print()
    if q1 and q3:
        print('  => DEUX PUCES ORDINAIRES SUFFISENT, A CONDITION DE CALIBRER. Le protocole')
        print('     gagne une ligne : annuler l\'offset differentiel avant stimulation.')
        if not q2:
            print('     Sans cette calibration (L5 brut) la mesure ne tient pas : le')
            print('     desappariement de frequences deplace le zero de la decision.')
        print('     PRESOMPTION NEGATIVE, ecrite avant : ce qui est etabli est qu\'un offset')
        print('     STATIQUE se calibre. Une DERIVE differentielle au cours du run ne se')
        print('     calibre pas par une mesure prise avant, et n\'est pas testee ici.')
    elif q1 and not q3:
        print('  => LE VOLET 2 EXIGE DES PUCES APPARIEES EN FREQUENCE. La calibration')
        print('     d\'offset ne suffit pas. Ce n\'est pas une refutation, c\'est un COUT')
        print('     DE FABRICATION, et il doit etre ecrit en tete du protocole.')
    elif not q1:
        print('  => L\'ETAT INITIAL COMMUN PORTAIT DEJA UNE PART DE L\'EFFET. La lecture')
        print('     differentielle exige deux reseaux prepares ensemble — contrainte bien')
        print('     plus lourde que prevu. A ecrire sans adoucissement.')

    fig = HERE.parent / 'figures'
    fig.mkdir(exist_ok=True)
    with open(fig / 'b6_two_real_chips.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    keys = sorted({k for s in summary for k in s})
    with open(fig / 'b6_two_real_chips_summary.csv', 'w', newline='',
              encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader(); w.writerows(summary)
    print(f"\nCSV : {fig / 'b6_two_real_chips.csv'}")
    print(f"      {fig / 'b6_two_real_chips_summary.csv'}")
    print(f'Wall time : {time.time() - t0:.1f}s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
