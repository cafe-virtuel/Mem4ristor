"""
EXPERIENCE P16 -- LE DOUTE EXPLORE-T-IL MOINS QUE LE DOUTE GELE, SUR LA NICHE ?

D'ou vient la question. Le 28/07, en fermant la question ouverte de P15 (Max-Cut),
une mesure non cherchee est tombee : sur 300 lectures de sign(v), Mem4ristor ne
visite que ~24 etats de signes DISTINCTS en FULL contre ~40 en FROZEN_U (25 / 45
en replication, 20 graines chacune). Le doute adaptatif visitait MOINS d'etats que
le doute gele. Voir `p15c_maxcut_identity_mechanism.py`.

Pourquoi cela ne peut pas rester la. La caracterisation centrale de la colonne B
depuis le 08/07 est que le doute est un "explorateur DISCIPLINE" -- il visite
plusieurs solutions valides par une marche continue. Cette caracterisation n'a
jamais ete confrontee au comptage FULL contre FROZEN_U : B1b comparait
WATCHDOG / BICAMERAL / hasard (verifie), jamais le doute a son propre gel. Et le
seul comptage qui existe aujourd'hui a ete fait sur une tache HORS niche, ou M4R
perd deja contre le tirage aleatoire a budget egal. Deux issues, toutes deux
utiles : soit l'immobilite se replique sur la niche, et c'est la caracterisation
"explorateur" qu'il faut corriger ; soit elle ne s'y replique pas, et le resultat
du Max-Cut est proprement BORNE a une tache ou M4R n'a de toute facon rien a dire.

LE PIEGE D'ATTRIBUTION, IDENTIFIE AVANT DE MESURER. Comparer FULL a FROZEN_U(0.5)
ne compare PAS "avec doute" a "sans doute". Le filtre vaut u_filter = tanh(k*(0.5-u)) :
  u = 0.5  -> u_filter ~ 0     : couplage QUASI NUL (reseau presque decouple)
  u -> 1   -> u_filter ~ -0.99 : couplage NEGATIF fort (anti-synchronisant)
  u -> 0   -> u_filter ~ +0.99 : couplage POSITIF fort (synchronisant)
En FULL, u demarre a sigma_baseline = 0.05 et monte. La comparaison historique
oppose donc "couplage fort" a "couplage nul", et un reseau decouple bouge
librement par construction. Conclure "le doute fige" de cette seule comparaison
serait une faute d'attribution -- exactement celle que l'audit du 27/07 a commise
en sens inverse avec "grace au doute".
D'ou QUATRE bras, et non deux :
  FULL      u adaptatif (le doute)
  FROZ_05   u = 0.50  -> couplage quasi nul      (le controle HISTORIQUE du projet)
  FROZ_HIGH u = 0.95  -> anti-synchro forte, FIGEE (meme niveau que FULL, sans adaptativite)
  FROZ_LOW  u = 0.05  -> synchro forte, FIGEE     (meme FORCE, signe oppose)
FROZ_HIGH separe l'ADAPTATIVITE du NIVEAU. FROZ_LOW separe le SIGNE de la FORCE.

Substrat : le harnais B1d exact (`experiments/scratch/deceptive_task_poc.py`,
tache trompeuse pulsee, lattice 10x10 periodique, T_pulse = 350), c'est-a-dire la
SEULE niche etablie du projet. Observable : la meme qu'au Max-Cut -- sign(v)
echantillonne tous les 10 pas sur 3000, soit 300 lectures. On mesure la MOBILITE
DE L'ETAT INTERNE, pas la performance de la tache : c'est ce qui rend les deux
experiences comparables.

--------------------------------------------------------------------------------
CRITERES POSES AVANT L'EXECUTION (aucun chiffre connu au moment de les ecrire)
--------------------------------------------------------------------------------
G0 -- GATE DE NICHE. La tache reconstruite doit etre effectivement TROMPEUSE,
     sinon on ne mesure pas sur la niche et rien n'est valide :
     decision finale juste sur >= 70 pct des graines, ET decision au pas 205
     (l'arret typique de la convergence, mesure le 07/07) juste sur <= 40 pct.

     >> G0 A ECHOUE A L'EXECUTION (28/07) : 0.65 contre 0.70 exige, le second
        volet passant nettement (0.30 <= 0.40). Le seuil est laisse tel quel et
        son echec est affiche : le repecher apres l'avoir vu tomber serait
        exactement la faute denoncee la veille dans R5 (P15b).
        DIAGNOSTIC, fait avant toute modification : le harnais reproduit le POC
        d'origine AU CHIFFRE PRES sur SES graines (0-11) -- acc_final 0.75 et
        acc_conv 0.25, identiques a `figures/scratch/deceptive_task_poc.csv`.
        Il n'y a donc pas de defaut de harnais : c'est MON SEUIL qui etait mal
        pose. J'avais emprunte 0.70 a une valeur mesuree sur 12 graines (le
        "plafond acc_FIN = 0.75" de B1d) et l'avais applique a 20 graines, dont
        8 jamais utilisees. Un seuil de solvabilite transporte d'un echantillon
        a un autre n'est pas un gate, c'est un prejuge.
G0b -- GATE DE FIDELITE (ajoute apres l'echec de G0, et c'est lui qui decide) :
     sur les graines 0-11, acc_final et acc_conv doivent egaler le CSV committe
     de B1d a 1e-9. C'est la question que G0 voulait poser -- "mesure-t-on bien
     la niche ?" -- posee par reproduction plutot que par seuil. Si G0b echoue,
     rien n'est valide.

C1 -- REPLICATION DU FAIT DU MAX-CUT SUR LA NICHE.
     CONFIRMEE si n_states(FULL) < n_states(FROZ_05) sur >= 16 graines sur 20.

C2 -- ATTRIBUTION : ADAPTATIVITE, OU NIVEAU DE COUPLAGE ? (n'est lu que si C1
     est confirmee ; sinon il n'y a rien a attribuer.)
     Si n_states(FULL) < n_states(FROZ_HIGH) sur >= 16/20, l'immobilite tient a
     quelque chose que le doute ADAPTATIF fait et qu'un u fige au meme niveau ne
     fait pas.
     Si l'ecart FULL vs FROZ_HIGH ne separe pas (< 16/20 dans les deux sens),
     alors l'immobilite est une propriete du NIVEAU de u, pas de son adaptation :
     "le doute fige" serait alors une facon trompeuse de dire "un couplage fort
     fige", et il faudra l'ecrire ainsi.

C3 -- FORCE OU SIGNE ? (observation, aucun verdict n'en depend.) n_states de
     FROZ_LOW, meme force de couplage que FROZ_HIGH mais signe oppose.

C4 -- MOBILITE PAR PAS (observation) : distance de Hamming moyenne entre deux
     lectures consecutives, par condition. Un comptage d'etats distincts peut
     s'effondrer soit parce que l'etat est fige, soit parce qu'il cycle ; la
     Hamming consecutive distingue les deux.

C5 -- LE GARDE-FOU DE LA COLONNE A (ajoute apres le premier passage, avant
     l'execution qui le mesure). Le preprint affirme que le doute MAINTIENT LA
     DIVERSITE (entropie spatiale) et REDUIT LA SYNCHRONIE. Ce script mesure
     tout autre chose : la mobilite TEMPORELLE du vecteur de signes. Rien ne
     garantit a un lecteur presse -- moi dans trois mois compris -- qu'il ne
     lira pas "l'anti-synchronisation visite moins d'etats" comme "l'anti-
     synchronisation reduit la diversite", ce qui contredirait le preprint.
     Cela se desamorce par une MESURE, pas par un argument :
     diversite spatiale = fraction de noeuds au signe MINORITAIRE, moyennee sur
     les lectures (0 = tous les noeuds du meme signe, 0.5 = partage parfait).
     ACCEPTEE si diversite(FROZ_HIGH) >= diversite(FROZ_05) sur >= 16/20 : alors
     mobilite temporelle et diversite spatiale sont DECOUPLEES -- un reseau peut
     etre spatialement diversifie ET temporellement fige -- et rien ici
     n'approche le preprint. Si elle est REJETEE, il faudra le dire tres fort.

     >> C5 A ETE REJETEE (0/20) -- ET SON INSTRUMENT ETAIT MAUVAIS. Constat fait
        avant d'ecrire quoi que ce soit sur la colonne A. La "fraction de noeuds
        au signe minoritaire" SATURE ici : elle vaut 0.013 a 0.066, soit 1 a 7
        noeuds sur 100, parce que la tache impose un signe global au reseau (la
        decision EST le signe de mean(v)). Surtout, ce n'est PAS la diversite du
        preprint : celui-ci mesure H, l'entropie de la distribution de v
        (H_cont dans Table 1, A3, A5), qui est elevee quand les valeurs de v
        sont ETALEES -- ce qui est possible avec tous les signes identiques.
        Mon critere portait donc sur une statistique qui ne repond pas a la
        question qu'il posait : c'est le defaut meme que ce projet a reproche a
        R2 et R5 la veille. Le seuil n'est pas deplace ; l'instrument est
        remplace, et le rejet de C5 reste affiche.
C5b -- LE GARDE-FOU, AVEC L'OBSERVABLE DU PREPRINT : H_cont, via
     `Mem4ristorV3.calculate_entropy()` -- exactement la mesure de Table 1.
     ACCEPTEE si H(FROZ_HIGH) >= H(FROZ_05) sur >= 16/20. Meme lecture que C5 :
     acceptee => mobilite temporelle et diversite au sens du preprint sont
     decouplees, et rien ici n'approche la colonne A. Rejetee => le dire tres
     fort, et ouvrir un examen separe.

R -- GATE DE REPLICATION, sur 20 graines DISJOINTES des 20 premieres. Le verdict
     C1 (et C2 s'il est lu) doit tenir a l'identique. Aucun resultat n'est cite
     sans lui : c'est ce gate qui a tue le Condorcet du 13/07 et le -0.30 du 26/07.

--------------------------------------------------------------------------------
RESULTATS (28/07/2026, G0b passe : le harnais reproduit B1d au chiffre pres)
--------------------------------------------------------------------------------
C1 CONFIRMEE ET REPLIQUEE, massivement : 20/20 puis 20/20 graines disjointes.
   Sur la niche, le doute visite 35 etats distincts contre 75 pour FROZEN_U(0.5),
   soit 2.2x moins. Le fait du Max-Cut n'etait donc PAS un accident de tache.

C2 REJETEE, ET DANS LE SENS INVERSE DE CELUI QU'ON POUVAIT CRAINDRE : 0/20 puis
   0/20 dans le sens teste, 20/20 puis 20/20 dans l'autre. A u FIGE a 0.95 -- le
   niveau meme qu'atteint le doute (u_end ~ 0.88) -- le reseau visite ENCORE
   MOINS d'etats (15.7 contre 35.0). L'adaptativite du doute MODERE
   l'immobilisation ; elle ne la produit pas.

C3 LE SIGNE TRANCHE, LA FORCE NON. A force de couplage EGALE, le synchronisant
   (FROZ_LOW, u=0.05) visite 75.5 etats contre 15.7 pour l'anti-synchronisant
   (FROZ_HIGH, u=0.95) -- soit le meme ordre que le couplage NUL (75.3).
   Ce qui immobilise est l'ANTI-SYNCHRONISATION, pas l'intensite du couplage.

C4 C'est un FIGEMENT, pas un cycle : Hamming consecutive 0.18 (FROZ_HIGH),
   0.29 (FULL), 0.51 (FROZ_05), 0.56 (FROZ_LOW) noeuds sur 100.

=> FORMULATION DEFENDABLE : "le doute explore moins" est vrai contre FROZEN_U(0.5)
   et FAUX contre FROZEN_U(0.95). Ne jamais l'ecrire sans nommer le comparateur.
   Ce qui fige est l'anti-synchronisation ; le doute adaptatif y arrive en
   moderant. La caracterisation "explorateur DISCIPLINE" (08/07) n'est pas
   renversee : elle gagne une precision -- la discipline se paie en mobilite, et
   le comparateur historique u=0.5 est un reseau QUASI DECOUPLE, pas un reseau
   "sans doute".

C5 / C5b -- LE GARDE-FOU DE LA COLONNE A A ETE POSE, IL EST TOMBE, ET LA
   VERIFICATION A MONTRE QU'IL N'Y AVAIT PAS D'ALERTE.
   C5 rejetee (0/20) sur un instrument que j'avais mal choisi (fraction de signe
   minoritaire, saturee). C5b rejetee (0/20) sur la VRAIE observable du preprint,
   H_cont : FULL 3.609 < FROZ_05 3.828. Avant d'ecrire quoi que ce soit :
     - `figures/b4_ablation_robustness.csv`, dans le regime du preprint, donne
       DEJA le meme ordre : full_hcont 3.645 / 3.674 contre frozen_hcont
       4.327 / 4.653. Geler u augmente H_cont la aussi.
     - la legende de `tab:ablations` (preprint.tex) l'ecrit noir sur blanc :
       l'entropie instantanee n'y est pas rapportee parce qu'elle "donne des
       resultats directionnellement incorrects pour cette comparaison".
   Le preprint ne revendique RIEN sur H entre FULL et FROZEN : sa revendication
   est la synchronie (0.031 contre 0.751) et la complexite LZ. La colonne A n'est
   pas approchee, et elle avait anticipe le point avant nous.
   LECON DE METHODE : un garde-fou qui tombe n'est pas une alerte tant qu'on n'a
   pas verifie ce que la cible affirme REELLEMENT.

Statut : exploration (colonne B). Aucun CSV canonique, aucun chiffre du preprint,
aucun claim du Guardian n'est touche. Le coeur n'est pas modifie.
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from mem4ristor.topology import Mem4Network
from mem4ristor.graph_utils import make_lattice_adj

# --- constantes reprises TELLES QUELLES du harnais B1d -----------------------
SIDE, N = 10, 100
MAX_BUDGET = 3000
N_DISTRACT = 26
N_TRUE = 14
E_TRUE = 0.6
E_DISTRACT = 1.0
T_PULSE = 350          # niveau canonique ou le doute gagne (B1d, 07/07)
T_CONV_TYPICAL = 205   # arret typique de la convergence, mesure le 07/07

SAMPLE_EVERY = 10
N_SEEDS = 20

# u_filter = tanh(k*(0.5-u)) : voir l'en-tete pour le role de chaque bras.
CONDITIONS = {
    'FULL': None,        # u adaptatif
    'FROZ_05': 0.50,     # couplage quasi nul  (controle historique)
    'FROZ_HIGH': 0.95,   # anti-synchro forte, figee
    'FROZ_LOW': 0.05,    # synchro forte, figee
}


def make_deceptive(rng):
    """Copie a l'identique de deceptive_task_poc.make_deceptive."""
    adj = make_lattice_adj(SIDE, periodic=True)
    dstar = rng.choice([-1, 1])
    nodes = rng.choice(N, size=N_DISTRACT + N_TRUE, replace=False)
    d_nodes, t_nodes = nodes[:N_DISTRACT], nodes[N_DISTRACT:]
    stim_on = np.zeros(N)
    stim_on[d_nodes] = -dstar * E_DISTRACT
    stim_on[t_nodes] = +dstar * E_TRUE
    stim_off = np.zeros(N)
    stim_off[t_nodes] = +dstar * E_TRUE
    return adj, stim_on, stim_off, dstar


def run_condition(adj, stim_on, stim_off, seed, u_frozen):
    """Un run du harnais B1d, instrumente pour la mobilite de l'etat.

    `ref` (stimulus nul, meme seed) est conserve parce que la decision de la
    tache est differentielle -- c'est le readout de B1c/B1d, on n'y touche pas.
    Le gel de u est applique aux DEUX reseaux : sinon la reference vivrait dans
    une autre condition que le run mesure et la decision serait biaisee.
    """
    net = Mem4Network(size=SIDE, heretic_ratio=0.0, seed=seed, adjacency_matrix=adj)
    ref = Mem4Network(size=SIDE, heretic_ratio=0.0, seed=seed, adjacency_matrix=adj)
    zero = np.zeros(N)

    seen = []           # etats de signes visites, encodes
    hamming_consec = []
    minority = []       # C5 : fraction de noeuds au signe minoritaire
    h_cont = []         # C5b : entropie H_cont, l'observable du preprint
    prev_s = None
    dec = np.empty(MAX_BUDGET, dtype=int)
    u_trace = []

    for t in range(MAX_BUDGET):
        if u_frozen is not None:
            net.model.u[:] = u_frozen
            ref.model.u[:] = u_frozen

        stim = stim_on if t < T_PULSE else stim_off
        net.step(I_stimulus=stim)
        ref.step(I_stimulus=zero)

        v = net.model.v
        d = float(np.mean(v) - np.mean(ref.model.v))
        dec[t] = 1 if d >= 0 else -1

        if t % SAMPLE_EVERY == 0:
            s = np.sign(v)
            s[s == 0] = 1
            s = s.astype(np.int8)
            seen.append(s.tobytes())
            if prev_s is not None:
                hamming_consec.append(int(np.sum(s != prev_s)))
            prev_s = s
            u_trace.append(float(net.model.u.mean()))
            n_pos = int(np.sum(s > 0))
            minority.append(min(n_pos, N - n_pos) / float(N))
            h_cont.append(float(net.calculate_entropy()))   # C5b : l'observable du preprint

    return {
        'n_states': len(set(seen)),
        'hamming_consec': float(np.mean(hamming_consec)),
        'minority_frac': float(np.mean(minority)),
        'h_cont': float(np.mean(h_cont)),
        'dec_final': int(dec[-1]),
        'dec_at_conv': int(dec[T_CONV_TYPICAL - 1]),
        'u_mean_end': u_trace[-1],
        'u_mean_all': float(np.mean(u_trace)),
    }


def sweep(seeds):
    rows = []
    for seed in seeds:
        rng = np.random.RandomState(3000 + seed)      # RNG du harnais B1d
        adj, stim_on, stim_off, dstar = make_deceptive(rng)

        row = {'seed': seed, 'dstar': int(dstar)}
        for name, u_frozen in CONDITIONS.items():
            r = run_condition(adj, stim_on, stim_off, seed, u_frozen)
            row['n_states_%s' % name] = r['n_states']
            row['hamming_%s' % name] = r['hamming_consec']
            row['minority_%s' % name] = r['minority_frac']
            row['hcont_%s' % name] = r['h_cont']
            row['u_end_%s' % name] = r['u_mean_end']
            if name == 'FULL':
                row['correct_final'] = int(r['dec_final'] == dstar)
                row['correct_at_conv'] = int(r['dec_at_conv'] == dstar)
        rows.append(row)

        print("seed %2d | n_states  FULL %3d | FROZ_05 %3d | FROZ_HIGH %3d | FROZ_LOW %3d"
              "  | u_end(FULL) %.2f"
              % (seed, row['n_states_FULL'], row['n_states_FROZ_05'],
                 row['n_states_FROZ_HIGH'], row['n_states_FROZ_LOW'],
                 row['u_end_FULL']))
    return pd.DataFrame(rows)


def verdict(d, a, b, label, need=16):
    """Compte les graines ou n_states(a) < n_states(b), et l'inverse."""
    lo = int((d['n_states_%s' % a] < d['n_states_%s' % b]).sum())
    hi = int((d['n_states_%s' % a] > d['n_states_%s' % b]).sum())
    ok = lo >= need
    print("  %-34s %2d/%d graines (inverse : %d)  moyennes %.1f vs %.1f"
          % (label, lo, len(d), hi,
             d['n_states_%s' % a].mean(), d['n_states_%s' % b].mean()))
    return ok, lo, hi


def main():
    print("=" * 78)
    print("P16 -- MOBILITE DE L'ETAT SUR LA NICHE : le doute explore-t-il moins ?")
    print("=" * 78)
    print("Criteres poses avant execution : voir l'en-tete du fichier.")
    print("Substrat : harnais B1d exact, tache trompeuse, T_pulse = %d." % T_PULSE)
    print("")

    df = sweep(range(N_SEEDS))
    out = os.path.join(ROOT, "figures", "p16_state_mobility_on_niche.csv")
    df.to_csv(out, index=False)
    print("")
    print("[CSV enregistre] : %s" % out)

    # ------------------------------------------------------------------ G0 ---
    print("")
    print("-" * 78)
    print("G0 -- GATE DE NICHE (la tache est-elle bien trompeuse ?)")
    print("-" * 78)
    acc_fin = float(df.correct_final.mean())
    acc_conv = float(df.correct_at_conv.mean())
    print("  decision FINALE juste          : %.2f  (critere >= 0.70)" % acc_fin)
    print("  decision au pas %d juste       : %.2f  (critere <= 0.40)"
          % (T_CONV_TYPICAL, acc_conv))
    g0_ok = (acc_fin >= 0.70 and acc_conv <= 0.40)
    print("  >> G0 : %s" % ("PASSE" if g0_ok else
                            "ECHOUE (seuil de solvabilite non atteint) -- voir G0b"))

    # ----------------------------------------------------------------- G0b ---
    print("")
    print("-" * 78)
    print("G0b -- GATE DE FIDELITE : le harnais reproduit-il le POC B1d d'origine ?")
    print("       (graines 0-11, celles du POC ; acc_final et acc_conv du CSV committe)")
    print("-" * 78)
    ref_csv = os.path.join(ROOT, "figures", "scratch", "deceptive_task_poc.csv")
    ref = pd.read_csv(ref_csv)
    ref = ref[ref.t_pulse == T_PULSE]
    sub = df[df.seed < 12]
    ref_fin, ref_conv = float(ref.acc_final.mean()), float(ref.acc_conv.mean())
    got_fin, got_conv = float(sub.correct_final.mean()), float(sub.correct_at_conv.mean())
    print("  acc_final  : reference %.4f | reproduit %.4f" % (ref_fin, got_fin))
    print("  acc_conv   : reference %.4f | reproduit %.4f" % (ref_conv, got_conv))
    g0b_ok = (abs(ref_fin - got_fin) < 1e-9 and abs(ref_conv - got_conv) < 1e-9)
    if not g0b_ok:
        print("  >> G0b ECHOUE : le harnais n'est pas celui de B1d.")
        print("  >> Tout ce qui suit est INVALIDE et ne doit pas etre cite.")
        return
    print("  >> G0b PASSE : c'est bien la niche, au chiffre pres.")
    if not g0_ok:
        print("")
        print("  /!\\ LECTURE : G0 a echoue, G0b passe. Le harnais est fidele ; le seuil")
        print("    de G0 avait ete transporte de 12 graines a 20, dont 8 neuves. La")
        print("    suite est lue, avec cet avertissement affiche, et non parce que le")
        print("    seuil aurait ete deplace apres coup -- il ne l'a pas ete.")

    print("")
    print("  Niveaux de couplage effectivement atteints (u moyen en fin de run) :")
    for name in CONDITIONS:
        print("    %-10s u_end = %.3f" % (name, df['u_end_%s' % name].mean()))

    # ------------------------------------------------------------------ C1 ---
    print("")
    print("-" * 78)
    print("C1 -- LE FAIT DU MAX-CUT SE REPLIQUE-T-IL SUR LA NICHE ?")
    print("     (critere : n_states(FULL) < n_states(FROZ_05) sur >= 16/20)")
    print("-" * 78)
    c1_ok, _, _ = verdict(df, 'FULL', 'FROZ_05', "FULL < FROZ_05 (u_filter ~ 0) :")
    print("  >> C1 : %s" % ("CONFIRMEE" if c1_ok else "NON CONFIRMEE"))

    # ------------------------------------------------------------------ C2 ---
    print("")
    print("-" * 78)
    print("C2 -- ATTRIBUTION : ADAPTATIVITE, OU NIVEAU DE COUPLAGE ?")
    print("-" * 78)
    if not c1_ok:
        print("  Non lu : C1 n'est pas confirmee, il n'y a rien a attribuer.")
        c2_ok = None
    else:
        c2_ok, c2_lo, c2_hi = verdict(df, 'FULL', 'FROZ_HIGH',
                                      "FULL < FROZ_HIGH (u=0.95 fige) :")
        if c2_ok:
            print("  >> C2 : l'immobilite tient a l'ADAPTATIVITE du doute -- un u fige au")
            print("     meme niveau ne la produit pas.")
        elif c2_hi >= 16:
            print("  >> C2 : INVERSE -- FULL visite PLUS que FROZ_HIGH. L'immobilite du")
            print("     Max-Cut ne s'attribue pas au doute adaptatif.")
        else:
            print("  >> C2 : NE SEPARE PAS. L'immobilite est une propriete du NIVEAU de u,")
            print("     pas de son adaptation. Alors 'le doute fige' est une facon")
            print("     trompeuse de dire 'un couplage fort fige' -- l'ecrire ainsi.")

    # ------------------------------------------------------------- C3 / C4 ---
    print("")
    print("-" * 78)
    print("C3 / C4 -- OBSERVATIONS (aucun verdict n'en depend)")
    print("-" * 78)
    print("  C3 force ou signe ? n_states : FROZ_HIGH %.1f (anti-synchro) vs FROZ_LOW %.1f"
          % (df.n_states_FROZ_HIGH.mean(), df.n_states_FROZ_LOW.mean()))
    print("      -- meme force de couplage, signe oppose ; FROZ_05 (couplage nul) %.1f"
          % df.n_states_FROZ_05.mean())
    print("  C4 mobilite par pas (Hamming moyen entre deux lectures consecutives) :")
    for name in CONDITIONS:
        print("      %-10s %.2f noeuds sur %d" % (name, df['hamming_%s' % name].mean(), N))
    print("      (un comptage bas avec Hamming basse = etat FIGE ; un comptage bas avec")
    print("       Hamming haute = etat qui CYCLE. Les deux ne disent pas la meme chose.)")

    # ------------------------------------------------------------------ C5 ---
    print("")
    print("-" * 78)
    print("C5 -- GARDE-FOU COLONNE A : mobilite temporelle et DIVERSITE SPATIALE")
    print("     sont-elles decouplees ? (critere : diversite(FROZ_HIGH) >=")
    print("     diversite(FROZ_05) sur >= 16/20)")
    print("-" * 78)
    print("  fraction de noeuds au signe MINORITAIRE (0 = unanime, 0.5 = partage parfait) :")
    for name in CONDITIONS:
        print("      %-10s %.3f   (n_states %.1f)"
              % (name, df['minority_%s' % name].mean(), df['n_states_%s' % name].mean()))
    c5_n = int((df.minority_FROZ_HIGH >= df.minority_FROZ_05).sum())
    c5_ok = c5_n >= 16
    print("  diversite(FROZ_HIGH) >= diversite(FROZ_05) : %d/%d graines" % (c5_n, len(df)))
    if c5_ok:
        print("  >> C5 ACCEPTEE : le bras le PLUS fige temporellement (FROZ_HIGH, %.1f etats)"
              % df.n_states_FROZ_HIGH.mean())
        print("     est AUSSI diversifie spatialement que le bras le plus mobile. Les deux")
        print("     observables sont DECOUPLEES : un reseau peut etre spatialement")
        print("     diversifie ET temporellement fige. Rien ici n'approche le preprint,")
        print("     qui parle de diversite spatiale et de synchronie -- pas de mobilite.")
    else:
        print("  >> C5 REJETEE : le bras le plus fige est aussi le moins diversifie")
        print("     AU SENS DE CETTE MESURE. Mais l'instrument est mauvais : la fraction")
        print("     minoritaire sature (1 a 7 noeuds sur 100, la tache impose un signe")
        print("     global) et ce n'est PAS la diversite du preprint. Voir C5b.")

    # ----------------------------------------------------------------- C5b ---
    print("")
    print("-" * 78)
    print("C5b -- LE MEME GARDE-FOU, AVEC L'OBSERVABLE DU PREPRINT (H_cont)")
    print("      (critere : H(FROZ_HIGH) >= H(FROZ_05) sur >= 16/20)")
    print("-" * 78)
    for name in CONDITIONS:
        print("      %-10s H_cont %.3f   (n_states %.1f, minoritaire %.3f)"
              % (name, df['hcont_%s' % name].mean(), df['n_states_%s' % name].mean(),
                 df['minority_%s' % name].mean()))
    c5b_n = int((df.hcont_FROZ_HIGH >= df.hcont_FROZ_05).sum())
    c5b_ok = c5b_n >= 16
    print("  H(FROZ_HIGH) >= H(FROZ_05) : %d/%d graines" % (c5b_n, len(df)))
    if c5b_ok:
        print("  >> C5b ACCEPTEE : le bras le plus FIGE temporellement (%.1f etats) est"
              % df.n_states_FROZ_HIGH.mean())
        print("     AUSSI DIVERSIFIE, au sens de Table 1, que le bras le plus mobile")
        print("     (%.1f etats). Mobilite temporelle et diversite du preprint sont"
              % df.n_states_FROZ_05.mean())
        print("     DECOUPLEES : un reseau peut etre diversifie ET fige. Rien dans ce")
        print("     script n'approche la colonne A -- et c'est mesure, pas argumente.")
    else:
        print("  >> C5b REJETEE a son critere : H(FROZ_HIGH) < H(FROZ_05), et plus")
        print("     largement H(FULL) %.3f < H(FROZ_05) %.3f."
              % (df.hcont_FULL.mean(), df.hcont_FROZ_05.mean()))
        print("")
        print("     ET CE N'EST PAS UNE DECOUVERTE : c'est un fait DEJA CONNU du projet,")
        print("     et deja ECRIT DANS LE PREPRINT. Verifie avant d'alerter :")
        print("      - `figures/b4_ablation_robustness.csv`, le regime du preprint :")
        print("        full_hcont 3.645 (BA) / 3.674 (LATTICE) contre frozen_hcont")
        print("        4.327 / 4.653. Geler u AUGMENTE H_cont la aussi. Le meme ordre")
        print("        qu'ici, sur une tache pourtant tres differente.")
        print("      - la legende de `tab:ablations` dans `preprint.tex` le dit noir sur")
        print("        blanc : l'entropie instantanee n'y est PAS rapportee parce")
        print("        qu'elle 'mesure l'etalement des tensions a un instant, pas")
        print("        l'independance comportementale, et donne des resultats")
        print("        DIRECTIONNELLEMENT INCORRECTS pour cette comparaison'.")
        print("     Le preprint ne revendique donc rien sur H entre FULL et FROZEN : sa")
        print("     revendication est la SYNCHRONIE (0.031 contre 0.751) et la")
        print("     complexite LZ. La colonne A n'est pas approchee, et elle avait")
        print("     anticipe ce point avant nous.")
        print("     A RETENIR : un garde-fou qui tombe n'est pas une alerte tant qu'on")
        print("     n'a pas verifie ce que la cible affirme reellement.")

    # ------------------------------------------------------------------- R ---
    print("")
    print("=" * 78)
    print("R -- GATE DE REPLICATION (graines %d-%d, disjointes)" % (N_SEEDS, 2 * N_SEEDS - 1))
    print("=" * 78)
    rep = sweep(range(N_SEEDS, 2 * N_SEEDS))
    out2 = os.path.join(ROOT, "figures", "p16_state_mobility_replication.csv")
    rep.to_csv(out2, index=False)
    print("")
    print("[CSV enregistre] : %s" % out2)

    acc_fin_r = float(rep.correct_final.mean())
    acc_conv_r = float(rep.correct_at_conv.mean())
    print("")
    print("  G0 sur graines neuves : finale %.2f | au pas %d %.2f"
          % (acc_fin_r, T_CONV_TYPICAL, acc_conv_r))
    gate_r = (acc_fin_r >= 0.70 and acc_conv_r <= 0.40)
    print("  >> %s" % ("gate de niche PASSE" if gate_r else
                       "gate de niche ECHOUE sur les graines neuves -- lire avec prudence"))

    print("")
    r1_ok, _, _ = verdict(rep, 'FULL', 'FROZ_05', "C1 replique : FULL < FROZ_05 :")
    r2_ok, r2_lo, r2_hi = verdict(rep, 'FULL', 'FROZ_HIGH', "C2 replique : FULL < FROZ_HIGH :")
    print("  n_states moyens : FULL %.1f | FROZ_05 %.1f | FROZ_HIGH %.1f | FROZ_LOW %.1f"
          % (rep.n_states_FULL.mean(), rep.n_states_FROZ_05.mean(),
             rep.n_states_FROZ_HIGH.mean(), rep.n_states_FROZ_LOW.mean()))

    # -------------------------------------------------------------- BILAN ----
    print("")
    print("=" * 78)
    print("BILAN")
    print("=" * 78)
    if c1_ok and r1_ok:
        print("  Le fait du Max-Cut SE REPLIQUE sur la niche : le doute visite moins")
        print("  d'etats que le controle historique FROZEN_U(0.5), %d/20 puis %d/20."
              % (int((df.n_states_FULL < df.n_states_FROZ_05).sum()),
                 int((rep.n_states_FULL < rep.n_states_FROZ_05).sum())))
        if c2_ok and r2_ok:
            print("  Et cela ne s'explique PAS par le seul niveau de couplage : a u fige")
            print("  au meme niveau (0.95), le reseau visite plus d'etats. L'adaptativite")
            print("  du doute est en cause. La caracterisation 'explorateur discipline'")
            print("  (08/07) demande une correction ecrite.")
        else:
            print("  MAIS L'ATTRIBUTION AU DOUTE EST FAUSSE, ET DANS LE SENS INVERSE DE")
            print("  CELUI QU'ON POUVAIT CRAINDRE. A u fige a 0.95 -- le meme niveau que")
            print("  celui qu'atteint le doute -- le reseau visite ENCORE MOINS d'etats")
            print("  (%.1f contre %.1f, %d/20 puis %d/20 dans ce sens)."
                  % (df.n_states_FROZ_HIGH.mean(), df.n_states_FULL.mean(),
                     int((df.n_states_FULL > df.n_states_FROZ_HIGH).sum()),
                     int((rep.n_states_FULL > rep.n_states_FROZ_HIGH).sum())))
            print("  Et le signe tranche ou la force ne tranche pas : a force EGALE, le")
            print("  couplage synchronisant (FROZ_LOW) visite %.1f etats contre %.1f pour"
                  % (df.n_states_FROZ_LOW.mean(), df.n_states_FROZ_HIGH.mean()))
            print("  l'anti-synchronisant, soit le meme ordre que le couplage NUL (%.1f)."
                  % df.n_states_FROZ_05.mean())
            print("")
            print("  FORMULATION DEFENDABLE : ce n'est pas le doute qui fige, c'est")
            print("  l'ANTI-SYNCHRONISATION. Et le doute adaptatif fige MOINS qu'un")
            print("  couplage anti-synchronisant fige au meme niveau -- son adaptation")
            print("  MODERE l'immobilisation au lieu de la produire.")
            print("  Ne pas ecrire 'le doute explore moins' sans nommer le comparateur :")
            print("  c'est vrai contre FROZEN_U(0.5), faux contre FROZEN_U(0.95).")
    elif c1_ok and not r1_ok:
        print("  C1 NE REPLIQUE PAS sur graines neuves. Ne rien citer -- meme motif que")
        print("  le Condorcet du 13/07 et le -0.30 du 26/07. Le fait du Max-Cut reste")
        print("  BORNE a sa tache.")
    else:
        print("  C1 n'est pas confirmee : sur la niche, le doute ne visite PAS moins")
        print("  d'etats que son gel. Le resultat du 28/07 est donc BORNE au Max-Cut --")
        print("  une tache hors niche ou M4R perd deja contre le tirage aleatoire.")
        print("  La caracterisation 'explorateur discipline' n'est pas contredite ici.")
    print("")
    print("  Portee : mesure de MOBILITE DE L'ETAT INTERNE, pas de performance. Elle ne")
    print("  dit rien de la justesse des decisions, qui est traitee par B1d/B5b.")
    print("  Colonne B uniquement : aucun chiffre du preprint n'est concerne.")
    print("=" * 78)


if __name__ == "__main__":
    main()
