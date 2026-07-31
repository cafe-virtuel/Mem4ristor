#!/usr/bin/env python3
"""
B6 — LE TROISIEME BRAS DU PROTOCOLE : rendre la prediction discriminante — 2026-07-31 (soir)
Claude Code (Opus 5) / Julien Chauvin. Suite directe de b6_sensor_noise.py.

D'OU VIENT LA QUESTION
----------------------
Mesure du 31/07 matin (b6_sensor_noise.py, controle Q4) : a fort bruit, un capteur
AVEUGLE — qui ne mesure rien du reseau — produit le meme effet qu'un capteur informe
(Cohen d +7.91 contre +7.93, ecart 0.01). Consequence : l'experience B6 telle que
formulee ne teste plus le MECANISME DU DOUTE, elle teste « un couplage repulsif en
moyenne ». Le controle historique (bras 2, couplage fige a sa valeur INITIALE) ne
distingue pas les deux. Un laboratoire mesurerait un effet reel et l'attribuerait a la
mauvaise cause — et nous lui aurions fourni le protocole qui permet cette erreur.

LE BRAS MANQUANT EST AMBIGU, ET L'AMBIGUITE EST PHYSIQUEMENT DECISIVE
---------------------------------------------------------------------
SPINTRONIC_PATHWAY §11 ecrit : « couplage fixe regle AU NIVEAU MOYEN ATTEINT ». Mais
`u` est une VARIABLE INTERNE : un labo ne peut ni la lire ni la regler. Ce qu'il cable,
c'est le COUPLAGE, u_filter = tanh(pi*(0.5-u)) + 0.01. Or tanh est non lineaire, donc
par Jensen  <tanh(pi*(0.5-u))> != tanh(pi*(0.5-<u>)),  et `u` varie a la fois dans le
temps et par noeud. Il y a donc DEUX bras 3 possibles, et ils ne sont pas equivalents :

  3a  u fige a <u>                 -> transposition litterale de FROZEN_U(0.95) du 28/07.
                                      NON REALISABLE sur un dispositif.
  3b  u_filter fige a <u_filter>   -> ce qu'un laboratoire peut reellement cabler.
                                      C'EST CELUI QUI DOIT ALLER DANS LE PROTOCOLE.

f(u) = tanh(pi*(0.5-u)) est CONCAVE pour u < 0.5 (f'' = -2pi^2 sech^2(g) tanh(g), g>0).
Le regime nominal a <u> ~ 0.16, donc on attend <f(u)> <= f(<u>) : le couplage moyen
REEL est plus faible — donc moins synchronisant — que celui deduit du niveau moyen.

LES QUATRE BRAS MESURES ICI
---------------------------
  B1  FULL             u dynamique (le mecanisme)
  B2  FIXE_INIT        u fige a sigma_baseline=0.05 -> u_filter=+0.898 (controle historique)
  B3a FIXE_NIVEAU_U    u fige a <u> mesure sur B1
  B3b FIXE_COUPLAGE    u_filter fige a <u_filter> mesure sur B1   <- le bras du protocole

Cohen d est toujours calcule CONTRE B2, comme dans tout le corpus (d > 0 = moins
synchronise que le controle historique).

CRITERES ET PREDICTIONS — ECRITS AVANT LA MESURE
------------------------------------------------
  G0  FIDELITE (BLOQUANT). La boucle reecrite doit reproduire le CSV du 09/07 au
      chiffre pres sur 4 references (tolerance 0.001). Si G0 echoue, RIEN n'est
      interpretable et le script s'arrete.

  G1  L'ECART DE JENSEN EST-IL REEL ? (prediction RISQUEE)
      Predit : |<u_filter> - (tanh(pi*(0.5-<u>)) + 0.01)| >= 0.05, dans le sens
      <u_filter> INFERIEUR. Si l'ecart est negligeable, 3a et 3b sont la meme chose,
      la distinction ci-dessus ne sert a rien, et il faut le dire.

  P1  LE BRAS 3a REPRODUIT-IL LE MECANISME ?   Predit VRAI : |d(3a) - d(B1)| < 0.5.
      C'est la prediction PRUDENTE — point d'ancrage, pas decouverte. Base : P17 du
      29/07 (FHN, u fige a 0.997 desynchronise autant que le doute vivant) et Q3 du
      31/07 (ce meme modele, verifie 6 fois sur 6).

  P2  LE BRAS 3b — CELUI QU'UN LABO CABLERAIT — REPRODUIT-IL LE MECANISME ?
      (prediction RISQUEE et DIRECTIONNELLE)
      Predit : NON, |d(3b) - d(B1)| >= 1.0, ET dans le sens d(3b) > d(B1).
      Trois issues, toutes ecrites avant :
        - P2 vraie dans ce sens  -> mauvaise nouvelle pour le doute (un couplage fixe
          fait MIEUX), bonne nouvelle pour la precision du protocole ;
        - P2 fausse (3b ~ B1)    -> le volet 1 de B6 ne teste pas le doute, point ;
        - P2 vraie dans le sens INVERSE (d(3b) < d(B1)) -> il y a de l'information dans
          la VARIATION du couplage, et la prediction redevient discriminante telle quelle.

  P3  LE DISCRIMINANT CANDIDAT : L'AUTO-CALIBRATION. (prediction RISQUEE — c'est elle
      qui decide de ce qu'on ecrit dans le protocole publie)
      Un couplage fixe doit etre REGLE ; `u` s'y etablit SEUL. On regle donc B3b dans
      une condition, puis on CHANGE la condition sans re-regler :
        (a) dispersion de frequence sigma_omega — ce qu'un labo SUBIT et ne controle
            pas (dispersion de fabrication des STNO). Regle a 0.15, applique a 0.30
            et 0.075.
        (b) topologie — regle sur BA m=3, applique a lattice 10x10, et reciproquement.
      Predit : d(B3b TRANSFERE) <= 0.50 * d(B3b RE-REGLE SUR PLACE), meme condition,
      memes graines.
        - P3 vraie  -> la prediction B6 redevient discriminante, sur l'AUTO-CALIBRATION
          et non sur la performance. C'est ce qu'il faut ecrire au labo.
        - P3 fausse -> le couplage fixe est robuste au transfert, et il ne reste rien a
          revendiquer pour le doute dans ce dispositif. A ecrire aussi.

      DEFAUT DE MON PREMIER CRITERE, CORRIGE AVANT LA MESURE ET LAISSE VISIBLE : j'avais
      d'abord ecrit « B1 conserve >= 70 % de son Cohen d sous transfert ». Pour le
      transfert de TOPOLOGIE c'est vrai PAR CONSTRUCTION — B1 y est le meme run qu'en
      nominal (memes graines, meme sigma_omega, meme topologie), le ratio vaut 1 sans
      rien mesurer. Un critere qu'on ne peut pas perdre ne rapporte rien (lecon du
      31/07 matin). P3 ne se juge donc QUE sur le ratio transfere/re-regle. Le ratio
      sur B1 reste AFFICHE comme contexte — il dit si la nouvelle condition est plus
      dure pour tout le monde — mais il n'entre pas dans le verdict.

  CONTROLE ADJACENT OBLIGATOIRE (lecon du 29/07 : ce n'est jamais la vigilance qui
  rattrape, c'est une forme ecrite avant). En condition transferee on mesure AUSSI un
  B3b RE-REGLE sur place. Sans lui, on ne peut pas separer « la nouvelle condition est
  plus dure pour tout le monde » de « le reglage ne transfere pas ». C'est la seule
  comparaison qui porte l'argument.

  PRESOMPTION NEGATIVE, ECRITE AVANT : j'attends que le transfert en sigma_omega morde
  PLUS que le transfert de topologie, parce que <u> depend directement du desaccord
  capte. Si c'est l'inverse, ne pas repecher — le rapporter tel quel.

  R1/R2  REPLICATION D'UNE OBSERVATION *NON* PREENREGISTREE (phase 4, ajoutee APRES
      avoir vu le premier passage — et c'est precisement pourquoi elle exige des
      graines neuves). Au premier passage, d(B1) >= d(B3b) partout, ecart moyen ~0.24.
      Un motif de SIGNE, jamais annonce ; le prendre pour un resultat sans le rejouer
      serait exactement le repechage que ce depot traque.
      ATTENTION AU DENOMINATEUR, corrige AVANT de rejouer : le tableau du premier
      passage affiche SIX lignes mais ne contient que QUATRE conditions distinctes —
      les deux lignes « transfert de topologie » recalculent d(B1) et d(B3b) sur
      EXACTEMENT les memes runs que les deux lignes nominales (memes graines, meme
      sigma_omega, meme topologie ; seul le bras TRANSFERE y est neuf). Compter
      « 5 sur 5 » aurait double deux mesures. Les quatre conditions distinctes sont
      BA/0.15, LATTICE/0.15, BA/0.30, BA/0.075 — et l'une d'elles (BA/0.30) a un
      ecart EXACTEMENT NUL.
      Graines 3081-3090, JAMAIS utilisees (le corpus va jusqu'a 3080, 29/07).
        R1 : d(B1) > d(B3b) dans AU MOINS 3 des 4 conditions distinctes.
        R2 : ecart moyen d(B1) - d(B3b) >= 0.15.
      Si R1 ET R2 passent, l'avantage residuel du doute est REEL mais PETIT (~0.2 de
      Cohen d) — trois a cinq fois sous le seuil de 1.0 preenregistre pour parler de
      discrimination, donc trop petit pour qu'une manip le separe de son bruit
      experimental. C'est la formulation obligatoire, pas une option.
      Si l'un des deux echoue : c'est du bruit, et il n'y a RIEN a en dire.

SORTIES : figures/b6_third_arm.csv / _summary.csv / .png  (VERSIONNEES)
"""
import csv
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'src'))

import b2_stno_amplitude_phase_poc as POC                        # noqa: E402
from mem4ristor.graph_utils import make_ba, make_lattice_adj      # noqa: E402

SEEDS = POC.SEEDS
N, DT, WARM_UP, STEPS = POC.N, POC.DT, POC.WARM_UP, POC.STEPS
N_NONLIN = 0.0
GAIN_NOMINAL = 5.0            # regime recommande par le §10 (« effet franc : gain 5 a 7 »)
SIGMA_OMEGA_REF = POC.SIGMA_OMEGA          # 0.15 — condition de reglage
SIGMA_OMEGA_TRANSFERT = [0.30, 0.075]      # double / moitie

# G0 — references du CSV du 09/07. Il vivait dans figures/scratch/ (gitignore) quand ce
# script a ete ecrit ; il est VERSIONNE dans figures/ depuis le 31/07 au soir, et se rejoue
# a l'identique sur toutes les colonnes hors IC (verifie : 20 lignes, 0 difference).
# figures/b2_stno_amplitude_phase_poc_agg.csv,
# n_nonlin=0. Ce sont les valeurs qui appuient le tableau publie du §7/§8.
G0_REF = {
    ('BA_m3', 'FULL', 1.0): 0.6126,
    ('BA_m3', 'FIXE_INIT', 1.0): 0.6202,
    ('BA_m3', 'FULL', 10.0): 0.2583,
    ('LATTICE_10x10', 'FULL', 10.0): 0.1660,
}
G0_TOL = 0.001

JENSEN_MIN = 0.05       # G1
P1_MAX = 0.5            # P1 : ecart tolere entre d(3a) et d(B1)
P2_MIN = 1.0            # P2 : ecart requis entre d(3b) et d(B1)
P3_B1_KEEP = 0.70       # P3 : fraction du Cohen d que B1 doit conserver
P3_B3B_LOSE = 0.50      # P3 : fraction du Cohen d que B3b transfere doit perdre


def run(adj, seed, bras, gain_u=GAIN_NOMINAL, sigma_omega=SIGMA_OMEGA_REF,
        u_frozen_at=None, filter_frozen_at=None):
    """Copie FIDELE de POC.run_one avec DEUX ajouts, et rien d'autre :
      - u_frozen_at      : u maintenu constant (bras 3a) ;
      - filter_frozen_at : u_filter remplace par une constante (bras 3b) — c'est la
        seule voie par laquelle u agit, donc figer le filtre suffit a figer le couplage.
    sigma_omega est expose pour le transfert P3 ; a sa valeur par defaut la trajectoire
    est strictement celle du POC — c'est ce que G0 verifie."""
    rng = np.random.RandomState(seed)
    n = adj.shape[0]
    deg = adj.sum(axis=1)
    deg_safe = np.where(deg > 0, deg, 1.0)

    omega = POC.OMEGA0 + rng.normal(0, sigma_omega, n)
    a = 0.05 * (rng.randn(n) + 1j * rng.randn(n))

    u_init = POC.SIGMA_BASELINE if u_frozen_at is None else u_frozen_at
    u = np.full(n, u_init)
    u_libre = (u_frozen_at is None and filter_frozen_at is None and bras == 'FULL')

    R_traj, u_traj, uf_traj = [], [], []
    for t in range(WARM_UP + STEPS):
        diff = a[None, :] - a[:, None]
        S = (adj * diff).sum(axis=1) / deg_safe
        sigma_social = np.abs(S)

        sigma_social_for_u = sigma_social * gain_u if u_libre else np.zeros(n)
        if filter_frozen_at is None:
            u_filter = np.tanh(np.pi * (0.5 - u)) + POC.SOCIAL_LEAKAGE
        else:
            u_filter = np.full(n, filter_frozen_at)

        p = np.abs(a) ** 2
        growth = POC.GAMMA_PLUS - POC.GAMMA_MINUS * (1.0 + POC.Q * p)
        eta = (rng.normal(0, POC.SIGMA_NOISE, n)
               + 1j * rng.normal(0, POC.SIGMA_NOISE, n)) / np.sqrt(DT)
        da = (growth + 1j * (omega + N_NONLIN * p)) * a + POC.K_COUPLING * u_filter * S + eta

        if u_libre:
            sigma_safe = np.clip(sigma_social_for_u, 0.0, 100.0)
            eps_adapt = POC.EPSILON_U * np.clip(
                1.0 + POC.ALPHA_SURPRISE * sigma_safe, 1.0, POC.SURPRISE_CAP)
            du = eps_adapt * (POC.K_U * sigma_social_for_u + POC.SIGMA_BASELINE - u) / POC.TAU_U
            u = np.clip(u + du * DT, 0.0, 1.0)

        a = a + da * DT
        if not np.all(np.isfinite(a)):
            raise OverflowError(f"divergence Euler (bras={bras}, seed={seed}, t={t})")
        if t >= WARM_UP:
            R_traj.append(float(np.abs(np.mean(np.exp(1j * np.angle(a))))))
            u_traj.append(float(u.mean()))
            uf_traj.append(float(np.mean(u_filter)))

    return {'R_mean': float(np.mean(R_traj)), 'u_mean': float(np.mean(u_traj)),
            'u_filter_mean': float(np.mean(uf_traj))}


def cohen(bras, ref):
    """d > 0 = le bras est MOINS synchronise que le controle historique."""
    pooled = np.sqrt((bras.var(ddof=1) + ref.var(ddof=1)) / 2)
    return float((ref.mean() - bras.mean()) / pooled) if pooled > 1e-12 else float('nan')


def campagne(adj, topo, sigma_omega, gain, rows, filtre_transfere=None, etiquette=''):
    """Les 4 bras dans une condition. Retourne les Cohen d et les niveaux mesures.
    filtre_transfere : si fourni, ajoute un 5e bras B3b_TRANSFERE avec CE couplage-la
    (regle dans une AUTRE condition) — le coeur du test P3."""
    def serie(bras, **kw):
        rs = [run(adj, s, bras, gain, sigma_omega, **kw) for s in SEEDS]
        for s, r in zip(SEEDS, rs):
            rows.append({'condition': etiquette, 'topology': topo, 'sigma_omega': sigma_omega,
                         'gain_u': gain, 'bras': bras, 'seed': s, **r})
        return rs

    r_b1 = serie('FULL')
    r_b2 = serie('FIXE_INIT')
    b1 = np.array([r['R_mean'] for r in r_b1])
    b2 = np.array([r['R_mean'] for r in r_b2])

    u_niveau = float(np.mean([r['u_mean'] for r in r_b1]))
    uf_niveau = float(np.mean([r['u_filter_mean'] for r in r_b1]))

    b3a = np.array([r['R_mean'] for r in serie('FIXE_NIVEAU_U', u_frozen_at=u_niveau)])
    b3b = np.array([r['R_mean'] for r in serie('FIXE_COUPLAGE', filter_frozen_at=uf_niveau)])

    out = {'topology': topo, 'sigma_omega': sigma_omega, 'gain_u': gain,
           'etiquette': etiquette, 'u_niveau': u_niveau, 'u_filter_niveau': uf_niveau,
           'u_filter_naif': float(np.tanh(np.pi * (0.5 - u_niveau)) + POC.SOCIAL_LEAKAGE),
           'R_B1': float(b1.mean()), 'R_B2': float(b2.mean()),
           'R_B3a': float(b3a.mean()), 'R_B3b': float(b3b.mean()),
           'd_B1': cohen(b1, b2), 'd_B3a': cohen(b3a, b2), 'd_B3b': cohen(b3b, b2),
           'd_B3b_transfere': float('nan'), 'filtre_transfere': filtre_transfere}
    out['jensen'] = out['u_filter_niveau'] - out['u_filter_naif']

    if filtre_transfere is not None:
        b3t = np.array([r['R_mean'] for r in
                        serie('FIXE_COUPLAGE_TRANSFERE', filter_frozen_at=filtre_transfere)])
        out['R_B3b_transfere'] = float(b3t.mean())
        out['d_B3b_transfere'] = cohen(b3t, b2)
    return out


def main():
    global SEEDS                     # la phase 4 rejoue tout sur des graines neuves
    t0 = time.time()
    topos = {'BA_m3': make_ba(N, 3, seed=42),
             'LATTICE_10x10': make_lattice_adj(10, periodic=True)}
    rows, summary = [], []

    # ---------------------------------------------------------------- G0 (BLOQUANT)
    print("=" * 100)
    print("  G0 — FIDELITE : la boucle reecrite reproduit-elle le CSV du 09/07 ?")
    print("=" * 100)
    g0_ok = True
    for (topo, bras, gain), ref in G0_REF.items():
        got = float(np.mean([run(topos[topo], s, bras, gain)['R_mean'] for s in SEEDS]))
        ok = abs(got - ref) <= G0_TOL
        g0_ok &= ok
        print(f"  {topo:15s} {bras:10s} gain={gain:4.0f} : R={got:.4f} vs {ref:.4f}  "
              f"{'OK' if ok else 'ECART ' + format(abs(got - ref), '.4f')}")
    if not g0_ok:
        print("\n  /!\\ G0 ECHOUE. La boucle a diverge du POC. RIEN N'EST INTERPRETABLE.")
        return 1
    print(f"  -> G0 PASSE (4/4, tolerance {G0_TOL}).  [{time.time() - t0:.0f}s]")

    # -------------------------------------------- PHASE 2 : les 4 bras, regime nominal
    print("\n" + "=" * 100)
    print(f"  PHASE 2 — LES QUATRE BRAS, regime nominal du protocole (gain={GAIN_NOMINAL:.0f},"
          f" sigma_omega={SIGMA_OMEGA_REF})")
    print("=" * 100)
    print(f"{'topologie':16s} {'<u>':>7} {'<filtre>':>9} {'naif':>7} {'Jensen':>8} "
          f"{'d(B1)':>7} {'d(B3a)':>7} {'d(B3b)':>7}")
    ref_cond = {}
    for topo, adj in topos.items():
        o = campagne(adj, topo, SIGMA_OMEGA_REF, GAIN_NOMINAL, rows, etiquette='nominal')
        summary.append(o)
        ref_cond[topo] = o
        print(f"{topo:16s} {o['u_niveau']:>7.4f} {o['u_filter_niveau']:>9.4f} "
              f"{o['u_filter_naif']:>7.4f} {o['jensen']:>+8.4f} "
              f"{o['d_B1']:>+7.2f} {o['d_B3a']:>+7.2f} {o['d_B3b']:>+7.2f}  "
              f"[{time.time() - t0:.0f}s]")

    # ------------------------------------------------- PHASE 3 : transfert (P3)
    print("\n" + "=" * 100)
    print("  PHASE 3 — TRANSFERT : le reglage tient-il quand la condition change ?")
    print("=" * 100)
    print(f"{'condition':34s} {'d(B1)':>7} {'d(B3b) local':>13} {'d(B3b) TRANSFERE':>17}")

    # (a) transfert en dispersion de frequence, sur BA
    for sg in SIGMA_OMEGA_TRANSFERT:
        o = campagne(topos['BA_m3'], 'BA_m3', sg, GAIN_NOMINAL, rows,
                     filtre_transfere=ref_cond['BA_m3']['u_filter_niveau'],
                     etiquette=f'transfert_sigma_omega_{sg}')
        summary.append(o)
        print(f"{'BA, sigma_omega ' + format(sg, '.3f') + ' (regle a 0.15)':34s} "
              f"{o['d_B1']:>+7.2f} {o['d_B3b']:>+13.2f} {o['d_B3b_transfere']:>+17.2f}  "
              f"[{time.time() - t0:.0f}s]")

    # (b) transfert de topologie, croise
    for topo, autre in (('LATTICE_10x10', 'BA_m3'), ('BA_m3', 'LATTICE_10x10')):
        o = campagne(topos[topo], topo, SIGMA_OMEGA_REF, GAIN_NOMINAL, rows,
                     filtre_transfere=ref_cond[autre]['u_filter_niveau'],
                     etiquette=f'transfert_topologie_depuis_{autre}')
        summary.append(o)
        print(f"{topo + ' (regle sur ' + autre + ')':34s} "
              f"{o['d_B1']:>+7.2f} {o['d_B3b']:>+13.2f} {o['d_B3b_transfere']:>+17.2f}  "
              f"[{time.time() - t0:.0f}s]")

    # ------------- PHASE 4 : replication du motif de signe, GRAINES NEUVES (R1/R2)
    print("\n" + "=" * 100)
    print("  PHASE 4 — REPLICATION sur graines 3081-3090 (JAMAIS utilisees) : d(B1) > d(B3b) ?")
    print("=" * 100)
    seeds_canon = SEEDS
    SEEDS = list(range(3081, 3091))
    print(f"{'condition':34s} {'d(B1)':>8} {'d(B3b)':>8} {'ecart':>8}")
    ecarts = []
    for topo, sg in ((t, SIGMA_OMEGA_REF) for t in topos) :
        o = campagne(topos[topo], topo, sg, GAIN_NOMINAL, rows, etiquette='replication')
        summary.append(o)
        ecarts.append(o['d_B1'] - o['d_B3b'])
        print(f"{topo + ', sigma_omega ' + format(sg, '.3f'):34s} "
              f"{o['d_B1']:>+8.2f} {o['d_B3b']:>+8.2f} {ecarts[-1]:>+8.2f}  [{time.time()-t0:.0f}s]")
    for sg in SIGMA_OMEGA_TRANSFERT:
        o = campagne(topos['BA_m3'], 'BA_m3', sg, GAIN_NOMINAL, rows, etiquette='replication')
        summary.append(o)
        ecarts.append(o['d_B1'] - o['d_B3b'])
        print(f"{'BA_m3, sigma_omega ' + format(sg, '.3f'):34s} "
              f"{o['d_B1']:>+8.2f} {o['d_B3b']:>+8.2f} {ecarts[-1]:>+8.2f}  [{time.time()-t0:.0f}s]")
    SEEDS = seeds_canon
    n_pos = sum(1 for e in ecarts if e > 0)
    ecart_moyen = float(np.mean(ecarts))
    r1, r2 = n_pos >= 4, ecart_moyen >= 0.15

    # ------------------------------------------------------------------ VERDICTS
    print("\n" + "=" * 100)
    print("  VERDICTS — chaque critere confronte a ce qui etait ecrit AVANT")
    print("=" * 100)

    verdicts = []
    for o in summary:
        if o['etiquette'] != 'nominal':
            continue
        j = abs(o['jensen'])
        g1 = j >= JENSEN_MIN and o['jensen'] < 0
        p1 = abs(o['d_B3a'] - o['d_B1']) < P1_MAX
        ecart2 = o['d_B3b'] - o['d_B1']
        p2 = abs(ecart2) >= P2_MIN and ecart2 > 0
        verdicts += [('G1', o['topology'], g1,
                      f"ecart de Jensen {o['jensen']:+.4f} (seuil {JENSEN_MIN} en valeur absolue,"
                      f" sens attendu : negatif)"),
                     ('P1', o['topology'], p1,
                      f"|d(3a) - d(B1)| = {abs(o['d_B3a'] - o['d_B1']):.2f} (< {P1_MAX} attendu)"),
                     ('P2', o['topology'], p2,
                      f"d(3b) - d(B1) = {ecart2:+.2f} (>= +{P2_MIN} attendu)")]

    for o in summary:
        if not o['etiquette'].startswith('transfert'):
            continue
        base = ref_cond[o['topology']]['d_B1']
        keep_b1 = o['d_B1'] / base if abs(base) > 1e-9 else float('nan')
        # LE critere : le bras 3b transfere contre le MEME bras re-regle sur place.
        keep_3b = o['d_B3b_transfere'] / o['d_B3b'] if abs(o['d_B3b']) > 1e-9 else float('nan')
        p3 = keep_3b <= P3_B3B_LOSE
        trivial = ' (=100 % par construction)' if o['etiquette'].startswith('transfert_topo') else ''
        verdicts.append(('P3', o['etiquette'], p3,
                         f"B3b transfere conserve {keep_3b:.0%} du 3b re-regle sur place "
                         f"(<= {P3_B3B_LOSE:.0%} attendu)  |  contexte, hors verdict : "
                         f"B1 a {keep_b1:.0%} du nominal{trivial}"))

    verdicts.append(('R1', 'replication graines 3081-3090', r1,
                     f"d(B1) > d(B3b) dans {n_pos} des 4 conditions distinctes (>= 3 attendu)"))
    verdicts.append(('R2', 'replication graines 3081-3090', r2,
                     f"ecart moyen d(B1) - d(B3b) = {ecart_moyen:+.2f} (>= 0.15 attendu)"))

    for nom, ou, ok, detail in verdicts:
        print(f"  [{nom}] {'VERIFIEE ' if ok else 'REJETEE  '} {ou:34s} {detail}")

    if r1 and r2:
        print(f"\n  -> L'avantage residuel du doute sur le couplage fixe est REEL et REPLIQUE,"
              f"\n     mais il vaut {ecart_moyen:+.2f} de Cohen d — {1.0 / max(ecart_moyen, 1e-9):.0f} fois"
              f" sous le seuil de 1.0 preenregistre"
              f"\n     pour parler de discrimination. Trop petit pour qu'une manip le separe de son"
              f"\n     bruit experimental. NE JAMAIS le citer sans cette phrase.")

    # ------------------------------------------------------------------- SORTIES
    fig_dir = HERE.parent / 'figures'
    fig_dir.mkdir(exist_ok=True)
    with open(fig_dir / 'b6_third_arm.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    keys = sorted({k for o in summary for k in o})
    with open(fig_dir / 'b6_third_arm_summary.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader(); w.writerows(summary)
    print(f"\nCSV : {fig_dir / 'b6_third_arm.csv'}")
    print(f"      {fig_dir / 'b6_third_arm_summary.csv'}")
    print(f"Wall time : {time.time() - t0:.1f}s")
    return 0


if __name__ == '__main__':
    sys.exit(main())
