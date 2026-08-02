#!/usr/bin/env python3
"""
B6 — LA LECTURE NON DIFFERENTIELLE : le retard du volet 2 tient-il sans le protocole ? — 2026-08-02
====================================================================================================
Claude Code (Opus 5) / Julien Chauvin. Point n°1 laisse ouvert le 31/07 au soir.

CE QUI EST EN JEU
-----------------
Le 31/07, le volet 1 de B6 est tombe (un couplage FIXE reproduit l'effet a 0.24 de
Cohen d pres, et son reglage TRANSFERE). Il ne reste au dossier B6 que le VOLET 2 :
le retard de recuperation apres leurre, +52 % en FULL contre FROZEN_U (5274.7 vs
3466.7 pas, CSV du 12/07). Quatre bras de controle ont montre qu'aucune boucle
OUVERTE ne le reproduit.

Mais la RESERVE PRINCIPALE, ecrite AVANT ces mesures dans b6_fifth_arm_per_node.py,
n'a jamais ete levee : la lecture est une PAIRE DIFFERENTIELLE. Deux copies jumelles
du meme reseau, l'une recevant +stim et l'autre -stim, et — c'est le point dur —
PARTAGEANT LA MEME REALISATION DE BRUIT. L'effet isole est precisement porte par
l'ecart entre les deux bras. Un effet porte par cet ecart peut etre une propriete du
PROTOCOLE DE LECTURE autant que du mecanisme.

Ce script decompose cette reserve en idealisations SEPAREES et mesure laquelle porte
le retard. Il ne cherche pas a sauver B6 : il cherche a savoir a quel dispositif reel
la proposition falsifiable s'applique encore, s'il en reste un.

L'ECHELLE DES LECTURES (de la plus idealisee a la plus realisable)
------------------------------------------------------------------
    lecture              copies   bruit         reference          realisable en labo ?
    L0 PAIRE_JUMELLE     2        PARTAGE       l'autre bras       NON (bruit commun exact)
    L1 PAIRE_BRUIT_INDEP 2        independant   l'autre bras       oui, deux puces
    L2 REF_NON_STIM      2        independant   une puce au repos  oui, deux puces
    L3 COPIE_UNIQUE      1        --            son propre niveau  oui, UNE puce
                                                avant stimulation

  L0 est la lecture du 12/07, reproduite ici par le NOUVEAU code (c'est le gate G0).
  L1 ne change QU'UNE chose : la copie - tire son propre bruit. C'est l'idealisation
     qu'aucun laboratoire ne peut avoir, isolee seule.
  L2 est la lecture du lancement 2 du 12/07, documentee comme contaminee par la
     « cicatrice de doute » (elle confond « u est monte » avec « evidence positive »).
     Elle est ici RE-MESUREE en termes de RETARD, ce qui n'avait jamais ete fait.
  L3 est le dispositif minimal : une seule puce, comparee a son propre niveau
     d'emission mesure juste avant la stimulation.

IDEALISATIONS QUI RESTENT, ET ELLES RENDENT LE TEST GENEREUX (a dire tel quel)
------------------------------------------------------------------------------
Dans L1/L2, la seconde copie garde le MEME etat initial apres warmup et les MEMES
frequences propres omega que la premiere. Deux puces reelles n'ont ni l'un ni l'autre.
C'est deliberement genereux : si le retard meurt deja sous cette version optimiste,
il meurt a fortiori dans un vrai dispositif. Si au contraire il survit, la reserve
n'est PAS levee pour autant — il restera a tester le desaccord d'omega.

CRITERES ET PREDICTIONS — ECRITS AVANT LA MESURE, LE 2 AOUT 2026
-----------------------------------------------------------------
    retard(X)  = moyenne sur T_pulse in {1500,3000,4500} de
                 [ flip_FULL(X,tp) - flip_FROZEN(X,tp) ]
    frac(X)    = retard(X) / retard(L0)

  G0  FIDELITE (BLOQUANT). L0 reproduit le CSV du 12/07 sur les 3 T_pulse, pour les
      DEUX conditions, a 0.5 pas pres. C'est le gate d'instrument : le nouveau code
      simule quatre copies au lieu de deux, et si le flux de bruit de la paire
      d'origine a bouge d'un tirage, RIEN de ce qui suit n'est interpretable.
      T_pulse=500 est EXCLU de toute la campagne (retard 2.8 pas, toute fraction y
      explose — artefact deja signale le 31/07).

  U   GARDE D'UTILISABILITE (pre-ecrite, par lecture). Une lecture n'est
      interpretable que si acc_fin >= 0.75 en FROZEN_U. FROZEN_U est le controle de
      loyaute de la tache : elle y est documentee a 100 % de bascule depuis le 12/07.
      Une lecture qui ne resout meme pas la tache a doute gele ne mesure rien du tout,
      et ses chiffres en FULL ne doivent pas etre lus. Cette garde peut faire tomber
      L2 et L3 sans qu'aucune conclusion sur le doute n'en sorte.

  P1  LE COEUR (prediction RISQUEE, directionnelle). Je predis VRAI : frac(L1) >= 0.50.
      Raisonnement : le mecanisme etabli le 31/07 est que le couplage repond au signal
      EFFECTIVEMENT RECU par sa copie. Cela n'exige pas que les deux copies partagent
      leur bruit — seulement qu'elles recoivent des stimuli opposes. Le bruit commun
      ne sert qu'a nettoyer la lecture, pas a produire l'asymetrie.
      - P1 VRAIE  -> le +52 % ne depend pas du bruit commun. L'idealisation restante
        (deux puces, +stim / -stim) est CONSTRUCTIBLE. Le volet 2 redevient un
        argument autonome, avec sa portee reduite et ecrite : il porte sur un
        dispositif DIFFERENTIEL, pas sur une puce isolee.
      - P1 FAUSSE -> le retard est porte par l'annulation exacte du mode commun,
        c'est-a-dire par le PROTOCOLE DE LECTURE. Le volet 2 tombe comme le volet 1,
        et il ne reste RIEN de B6 comme prediction falsifiable. A ecrire tel quel,
        sans adoucissement.

  P2  (prediction RISQUEE, je predis FAUX) : frac(L2) >= 0.50.
      Je predis que NON : contre une reference non stimulee, la cicatrice de doute
      redevient le signal dominant en FULL, et elle est INDEPENDANTE du signe de
      l'evidence. C'est deja documente qualitativement (12/07, lancement 2) ; ce qui
      est neuf ici est de le mesurer en retard.

  P3  (prediction RISQUEE, je predis VRAI) : L3 echoue la garde U.
      Raisonnement : le reseau couple n'atteint jamais l'equilibre en 9000 pas (FAIT
      PHYSIQUE 1 du 12/07) ; la derive lente de la puissance moyenne domine un signal
      d'evidence de l'ordre de 0.013. Une puce seule ne devrait rien decider.

  D1  DIAGNOSTIC, PAS UN CRITERE — et je dis pourquoi. La cicatrice de doute biaise
      la lecture dans un sens fixe, donc elle devrait creuser un ecart entre les
      graines a d*=+1 et celles a d*=-1. Ce serait la signature directe de la
      contamination. MAIS le tirage des masques donne 3 graines a d*=+1 contre 9 a
      d*=-1 : une moyenne sur 3 runs ne supporte aucun seuil. La ventilation est donc
      REPORTEE COMME DIAGNOSTIC, avec ses effectifs, et aucune conclusion ne s'y
      appuie. (Critere pose sur la mauvaise statistique = le defaut du 27/07,
      rapporte dans le contexte projet ; on ne le refait pas.)

  PRESOMPTION NEGATIVE, ECRITE AVANT : si P1 est vraie, NE PAS conclure « le volet 2
  est sauve ». Il sera montre robuste a UNE idealisation sur trois (le bruit commun) ;
  l'etat initial commun et les omega communs resteront non testes, et sont ecrits
  ci-dessus. Le gain autorise est exactement : « la lecture differentielle n'a pas
  besoin d'un bruit commun exact », rien de plus.

CE QUE LA MESURE A RENDU (ajoute APRES la campagne du 2026-08-02, criteres inchanges)
--------------------------------------------------------------------------------------
  G0  PASSE 6/6, a la decimale. Le nouveau code (quatre copies) reproduit la paire du
      12/07 exactement.

  P1  VERIFIEE. frac(L1) = 1.029 sur les 12 graines ; 1.016 sur le sous-ensemble
      jamais censure. Retirer le bruit commun ne retire RIEN du retard : 2478.9 pas
      contre 2409.7. L1 a exactement le meme profil de censure que L0 (les memes 2 runs
      sur 36, graines 7 et 11 a T_pulse=4500) — la comparaison est propre.
      => L'idealisation qu'aucun laboratoire ne peut avoir n'est PAS ce qui porte l'effet.

  P3  VERIFIEE. L3 (une seule puce) rend acc_fin = 0.583 en FROZEN_U et echoue la garde.
      Le detail dit pourquoi : ses 27 echecs en FULL sont EXACTEMENT les 9 graines a
      d*=-1, aux trois T_pulse. La puce seule ne decide pas, elle repond « +1 » toujours ;
      la derive lente de la puissance moyenne domine l'evidence, comme prevu.

  P2  REJETEE — ET LE CRITERE ETAIT MAL POSE. C'EST LA VRAIE LECON DU JOUR.
      frac(L2) = 0.720, donc au-dessus du seuil : sur le papier, la reference passive
      « conserve » 72 % du retard. C'est FAUX au sens ou on voudrait le lire.
      En FULL, L2 laisse 12 runs sur 36 sans jamais basculer, et ce sont les graines
      2, 4 et 10 aux trois T_pulse — c'est-a-dire LES TROIS SEULES GRAINES A d*=+1,
      toutes, integralement. La lecture ne mesure pas un retard : elle a un BIAIS DE
      SIGNE, et les runs qui n'aboutissent jamais comptent flip = MAX_BUDGET+1 = 9001.
      Or 9001 est la plus grande valeur possible : UNE LECTURE QUI ECHOUE GONFLE LE
      RETARD QU'ELLE EST CENSEE MESURER. Mon critere « frac >= 0.50 » est donc
      monotone dans le mauvais sens — il recompense la panne.
      Sur le sous-ensemble jamais censure, frac(L2) retombe a 0.639 (et ce
      sous-ensemble est entierement a d*=-1, donc lui non plus n'est pas lisible).
      CONCLUSION HONNETE : le chiffre de L2 n'est pas interpretable, ni pour ni contre.
      Ce qui est etabli sur L2 est plus simple et plus dur : CONTRE UNE REFERENCE
      PASSIVE, LA DECISION DEVIENT BIAISEE EN FULL — la « cicatrice de doute » du 12/07
      redevient le signal dominant, exactement comme le lancement 2 l'avait decrit
      qualitativement.

  D1  Le diagnostic que j'avais refuse d'ancrer dans un critere (effectifs 3 contre 9)
      est celui qui porte l'information : ecart d*=+1 / d*=-1 de 81 pas en L0, 113 en L1,
      3172 en L2, 7741 en L3. Il ne s'agissait pas d'un effectif trop faible pour un
      effet subtil, mais d'un effet total (3 graines sur 3, 3 T_pulse sur 3).

  DEFAUT DE MA GARDE U, a ecrire : je l'ai posee sur acc_fin en FROZEN_U, en
  raisonnant que FROZEN est le controle de loyaute de la tache. Le raisonnement tient,
  mais il rate le cas reel : L2 passe la garde a 0.917 en FROZEN et se casse a 0.67 en
  FULL — c'est-a-dire dans la condition qui nous interesse. Une garde d'utilisabilite
  doit porter sur TOUTES les conditions comparees, pas seulement sur le controle.

  PORTEE DE CE QUI EST GAGNE, ecrite avant et non elargie apres : la reserve est levee
  POUR LE BRUIT COMMUN, et pour rien d'autre. L'etat initial et les frequences propres
  omega restent COMMUNS aux deux copies dans L1 ; deux puces reelles n'ont ni l'un ni
  l'autre. Le volet 2 porte sur un dispositif DIFFERENTIEL a deux reseaux recevant
  +stim et -stim — ce qui est constructible — et NON sur une puce isolee, ou il n'y a
  plus de decision du tout.
  [SUITE, LE MEME JOUR : b6_two_real_chips.py a retire les deux idealisations restantes.
   Etat initial independant -> 0.803 du retard ; omega independants EN PLUS (soit deux
   puces reellement distinctes) -> 0.612, avec une precision qui MONTE a 0.972. Aucune
   des trois idealisations ne porte l'effet. La reserve est LEVEE ; il reste un cout
   d'amplitude de ~40 %, soit +34 % de retard mesurable au lieu de +52 %.]

Coeur NON touche. b1d_stno_deceptive_poc.py NON modifie (importe seulement).
SORTIES : figures/b6_nondifferential_readout.csv / _summary.csv  (VERSIONNEES)
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
T_PULSES = [1500, 3000, 4500]          # T_pulse=500 exclu (retard 2.8 pas)
CONDITIONS = ['STNO_FULL', 'STNO_FROZEN_U']
READOUTS = ['L0_PAIRE_JUMELLE', 'L1_PAIRE_BRUIT_INDEP', 'L2_REF_NON_STIM',
            'L3_COPIE_UNIQUE']

# --- CSV du 12/07 (figures/b1d_stno_deceptive_poc.csv — VERSIONNE le 02/08, apres
#     regeneration IDENTIQUE AU BIT PRES a l'artefact du 12/07 ; identiques aux
#     valeurs codees dans b6_fifth_arm_per_node.py G0_REF) -----------------------
G0_REF = {
    'STNO_FULL':     {1500: 5102.6, 3000: 6838.6, 4500: 7863.7},
    'STNO_FROZEN_U': {1500: 2611.4, 3000: 4239.6, 4500: 5724.8},
}
G0_TOL = 0.5
U_MIN_ACCFIN = 0.75        # garde d'utilisabilite, mesuree en FROZEN_U
P1_MIN_FRAC = 0.50
NOISE_OFFSET = 777_000_000  # flux de bruit de la SECONDE puce (independant de P12)


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


def _roll(mat, w):
    """Moyenne glissante causale, identique a P12.simulate._roll."""
    csum = np.cumsum(mat, axis=0)
    out = np.empty_like(mat)
    for t in range(mat.shape[0]):
        lo = max(0, t - w + 1)
        out[t] = (csum[t] - (csum[lo - 1] if lo > 0 else 0.0)) / (t - lo + 1)
    return out


def simulate_all(adj, deg, stim_on, stim_off, seed, t_pulse, condition):
    """Quatre copies en un seul run, pour que les quatre lectures soient APPARIEES.

      A : +stim, flux de bruit d'origine (celui de P12)
      B : -stim, MEME flux que A            -> L0 = A - B  (la paire jumelle du 12/07)
      C : -stim, flux INDEPENDANT           -> L1 = A - C
      D :  0    , flux INDEPENDANT (le meme que C, jamais utilise en meme temps)
                                            -> L2 = A - D
      L3 = A - (niveau moyen de A mesure en fin de warmup)

    Le flux de A/B consomme EXACTEMENT les memes tirages que P12.simulate : le rng
    d'origine n'est jamais sollicite par C ni D (rng2 separe). C'est ce qui rend G0
    reproductible au dixieme de pas.
    """
    rng = np.random.default_rng(seed)                 # flux d'origine, intouche
    rng2 = np.random.default_rng(seed + NOISE_OFFSET)  # seconde puce
    libre = (condition == 'STNO_FULL')
    omega = P12.OMEGA0 + rng.normal(0, P12.SIGMA_OMEGA, N)
    phases = rng.uniform(0.0, 2.0 * np.pi, N)
    p_star = (P12.GAMMA_PLUS - P12.GAMMA_MINUS) / (P12.GAMMA_MINUS * P12.Q)
    a = np.sqrt(p_star) * np.exp(1j * phases)
    u = np.full(N, P12.SIGMA_BASELINE)
    inv_sqrt_dt = 1.0 / np.sqrt(DT)

    for _ in range(WARMUP_STEPS):                     # warmup commun (comme P12)
        noise = rng.normal(0.0, P12.SIGMA_NOISE, size=(2, N))
        eta = (noise[0] + 1j * noise[1]) * inv_sqrt_dt
        a, u, _ = _step(a, u, P12.GAMMA_PLUS, omega, adj, deg, eta, libre)
    if not np.all(np.isfinite(a)):
        return None
    p_base = float((np.abs(a) ** 2).mean())           # niveau avant stimulation (L3)

    aA, aB, aC, aD = a.copy(), a.copy(), a.copy(), a.copy()
    uA, uB, uC, uD = u.copy(), u.copy(), u.copy(), u.copy()
    pA = np.empty((MAX_BUDGET, N))
    dAB = np.empty((MAX_BUDGET, N))
    dAC = np.empty((MAX_BUDGET, N))
    dAD = np.empty((MAX_BUDGET, N))

    for t in range(MAX_BUDGET):
        stim = stim_on if t < t_pulse else stim_off
        noise = rng.normal(0.0, P12.SIGMA_NOISE, size=(2, N))
        eta = (noise[0] + 1j * noise[1]) * inv_sqrt_dt
        noise2 = rng2.normal(0.0, P12.SIGMA_NOISE, size=(2, N))
        eta2 = (noise2[0] + 1j * noise2[1]) * inv_sqrt_dt
        gp_pos, gp_neg = P12.GAMMA_PLUS + ISCALE * stim, P12.GAMMA_PLUS - ISCALE * stim

        aA, uA, ppA = _step(aA, uA, gp_pos, omega, adj, deg, eta, libre)
        aB, uB, ppB = _step(aB, uB, gp_neg, omega, adj, deg, eta, libre)
        aC, uC, ppC = _step(aC, uC, gp_neg, omega, adj, deg, eta2, libre)
        aD, uD, ppD = _step(aD, uD, P12.GAMMA_PLUS, omega, adj, deg, eta2, libre)

        if not all(np.all(np.isfinite(x)) for x in (aA, aB, aC, aD)):
            return None
        qA = np.abs(aA) ** 2
        pA[t] = qA
        dAB[t] = qA - np.abs(aB) ** 2
        dAC[t] = qA - np.abs(aC) ** 2
        dAD[t] = qA - np.abs(aD) ** 2

    out = {}
    for name, mat in (('L0_PAIRE_JUMELLE', dAB), ('L1_PAIRE_BRUIT_INDEP', dAC),
                      ('L2_REF_NON_STIM', dAD)):
        d_var = _roll(mat, W_READ).mean(axis=1)
        out[name] = np.where(d_var >= 0, 1, -1).astype(int)
    d_var = _roll(pA, W_READ).mean(axis=1) - p_base
    out['L3_COPIE_UNIQUE'] = np.where(d_var >= 0, 1, -1).astype(int)
    return out


def main() -> int:
    t0 = time.time()
    adj = make_lattice_adj(P12.SIDE, periodic=True).astype(float)
    deg = adj.sum(axis=1)
    rows = []

    print('=' * 102)
    print('  B6 — LA LECTURE NON DIFFERENTIELLE (2026-08-02) — criteres ecrits avant la mesure')
    print('=' * 102)
    total = len(CONDITIONS) * len(T_PULSES) * len(SEEDS)
    done = 0
    for cond in CONDITIONS:
        for tp in T_PULSES:
            for seed in SEEDS:
                rng = np.random.RandomState(3000 + seed)
                stim_on, stim_off, dstar = P12.make_deceptive(rng)
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

    def mflip(**kw):
        v = sel(**kw)
        return float(np.mean([r['flip_time'] for r in v]))

    def maccfin(**kw):
        v = sel(**kw)
        return float(np.mean([r['acc_final'] for r in v]))

    # ------------------------------- G0 -------------------------------------
    print('\n' + '=' * 102)
    print('  G0 — FIDELITE de L0 au CSV du 12/07 (BLOQUANT)')
    print('=' * 102)
    g0_ok = True
    for cond in CONDITIONS:
        for tp in T_PULSES:
            got = mflip(readout='L0_PAIRE_JUMELLE', condition=cond, t_pulse=tp)
            ref = G0_REF[cond][tp]
            ok = abs(got - ref) <= G0_TOL
            g0_ok &= ok
            print(f'  {cond:<14} T_pulse={tp:5d} : {got:8.1f} vs {ref:8.1f}   '
                  f"{'OK' if ok else 'ECART'}")
    if not g0_ok:
        print('\n  /!\\ G0 ECHOUE : le nouveau code ne reproduit pas la paire du 12/07.')
        print('      RIEN de ce qui suit n\'est interpretable. Campagne annulee.')
        return 1
    print('  -> G0 PASSE (6/6). Le nouveau code reproduit la lecture d\'origine.')

    # --------------------------- garde U ------------------------------------
    print('\n' + '=' * 102)
    print('  U — GARDE D\'UTILISABILITE (acc_fin en FROZEN_U, seuil 0.75)')
    print('=' * 102)
    usable = {}
    for ro in READOUTS:
        acc = maccfin(readout=ro, condition='STNO_FROZEN_U')
        usable[ro] = acc >= U_MIN_ACCFIN
        print(f'  {ro:<22} acc_fin(FROZEN_U) = {acc:.3f}   '
              f"{'UTILISABLE' if usable[ro] else 'INUTILISABLE -> chiffres FULL non lus'}")

    # ----------------------- retards et fractions ---------------------------
    print('\n' + '=' * 102)
    print('  LE RETARD, LECTURE PAR LECTURE')
    print('=' * 102)
    print(f"{'lecture':<22}{'T_pulse':>8}{'FULL':>10}{'FROZEN':>10}{'retard':>9}"
          f"{'rel':>8}{'accF':>7}{'accZ':>7}{'cens':>6}")
    summary = []
    retard = {}
    for ro in READOUTS:
        per_tp = []
        for tp in T_PULSES:
            ff = mflip(readout=ro, condition='STNO_FULL', t_pulse=tp)
            fz = mflip(readout=ro, condition='STNO_FROZEN_U', t_pulse=tp)
            aF = maccfin(readout=ro, condition='STNO_FULL', t_pulse=tp)
            aZ = maccfin(readout=ro, condition='STNO_FROZEN_U', t_pulse=tp)
            cens = sum(1 for r in sel(readout=ro, t_pulse=tp)
                       if r['flip_time'] > MAX_BUDGET)
            per_tp.append(ff - fz)
            print(f'{ro:<22}{tp:>8}{ff:>10.1f}{fz:>10.1f}{ff - fz:>9.1f}'
                  f'{(ff - fz) / fz:>8.1%}{aF:>7.2f}{aZ:>7.2f}{cens:>6d}')
            summary.append({'readout': ro, 't_pulse': tp, 'flip_full': ff,
                            'flip_frozen': fz, 'retard': ff - fz,
                            'retard_relatif': (ff - fz) / fz, 'accfin_full': aF,
                            'accfin_frozen': aZ, 'n_censures': cens})
        retard[ro] = float(np.mean(per_tp))
        print(f"{'':22}{'MOYEN':>8}{'':>10}{'':>10}{retard[ro]:>9.1f}")
    for ro in READOUTS:
        summary.append({'readout': ro, 't_pulse': 'MOYEN', 'retard': retard[ro],
                        'frac_du_retard_L0': retard[ro] / retard['L0_PAIRE_JUMELLE'],
                        'utilisable': int(usable[ro])})

    # ------------------------- D1 : diagnostic dstar -------------------------
    print('\n' + '=' * 102)
    print('  D1 — DIAGNOSTIC (PAS un critere) : flip en FULL ventile par signe de d*')
    print('       Effectifs tres desequilibres (3 graines a d*=+1, 9 a d*=-1) :')
    print('       aucune conclusion ne s\'appuie sur ce tableau.')
    print('=' * 102)
    for ro in READOUTS:
        dp = [r['flip_time'] for r in sel(readout=ro, condition='STNO_FULL', dstar=1)]
        dm = [r['flip_time'] for r in sel(readout=ro, condition='STNO_FULL', dstar=-1)]
        print(f'  {ro:<22} d*=+1 : {np.mean(dp):8.1f} (n={len(dp)})   '
              f'd*=-1 : {np.mean(dm):8.1f} (n={len(dm)})   '
              f'ecart {abs(np.mean(dp) - np.mean(dm)):7.1f}')

    # ------------------------------ verdicts --------------------------------
    print('\n' + '=' * 102)
    print('  VERDICTS — confrontes a ce qui etait ecrit AVANT la mesure')
    print('=' * 102)
    f1 = retard['L1_PAIRE_BRUIT_INDEP'] / retard['L0_PAIRE_JUMELLE']
    f2 = retard['L2_REF_NON_STIM'] / retard['L0_PAIRE_JUMELLE']
    p1 = usable['L1_PAIRE_BRUIT_INDEP'] and f1 >= P1_MIN_FRAC
    p2 = usable['L2_REF_NON_STIM'] and f2 >= P1_MIN_FRAC
    p3 = not usable['L3_COPIE_UNIQUE']
    print(f"  [P1] predite VRAIE  -> {'VERIFIEE' if p1 else 'REJETEE '}   "
          f'frac(L1) = {f1:.3f}  (>= {P1_MIN_FRAC} attendu, lecture utilisable requise)')
    print(f"  [P2] predite FAUSSE -> {'REJETEE (elle etait vraie)' if p2 else 'VERIFIEE'}   "
          f'frac(L2) = {f2:.3f}')
    print(f"  [P3] predite VRAIE  -> {'VERIFIEE' if p3 else 'REJETEE '}   "
          f"L3 (une seule puce) {'echoue' if p3 else 'PASSE'} la garde d'utilisabilite")
    print()
    if p1:
        print('  => LE RETARD NE DEPEND PAS DU BRUIT COMMUN. L\'idealisation restante')
        print('     (deux puces, +stim / -stim) est CONSTRUCTIBLE. Le volet 2 redevient un')
        print('     argument autonome, a portee reduite et ecrite : il porte sur un')
        print('     dispositif DIFFERENTIEL, pas sur une puce isolee.')
        print('     PRESOMPTION NEGATIVE, ecrite avant : etat initial et omega restent')
        print('     COMMUNS aux deux copies dans ce test. La reserve n\'est levee que')
        print('     pour le bruit. Ne pas ecrire « le volet 2 est sauve ».')
    else:
        print('  => LE RETARD EST PORTE PAR L\'ANNULATION EXACTE DU MODE COMMUN, donc par le')
        print('     PROTOCOLE DE LECTURE et non par le seul mecanisme. Le volet 2 tombe comme')
        print('     le volet 1 : il ne reste RIEN de B6 comme prediction falsifiable en l\'etat.')
        print('     A ecrire tel quel, sans adoucissement.')

    fig = HERE.parent / 'figures'
    fig.mkdir(exist_ok=True)
    with open(fig / 'b6_nondifferential_readout.csv', 'w', newline='',
              encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    keys = sorted({k for s in summary for k in s})
    with open(fig / 'b6_nondifferential_readout_summary.csv', 'w', newline='',
              encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader(); w.writerows(summary)
    print(f"\nCSV : {fig / 'b6_nondifferential_readout.csv'}")
    print(f"      {fig / 'b6_nondifferential_readout_summary.csv'}")
    print(f'Wall time : {time.time() - t0:.1f}s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
