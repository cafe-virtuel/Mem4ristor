"""
EXPERIENCE P15c -- LE MECANISME DE L'IDENTITE FULL / FROZEN_U SUR LE MAX-CUT.
Aller au bout de la question laissee ouverte le 27/07/2026 (demande de Julien,
28/07 : "il faut aller jusqu'au bout des problemes").

ETAT DE LA QUESTION EN ENTRANT
------------------------------
10 graines sur 20 donnent une coupe ET une energie strictement identiques entre
Mem4ristor FULL (doute adaptatif) et FROZEN_U (u fige a 0.5). Six explications
sont deja mortes a leur propre critere : H_A (best fige au premier echantillon),
H_B (best atteint avant divergence des signes), H_C (convergence au meme point),
R2 (la relaxation degrade la coupe), R5 (etats visites indiscernables d'un tirage
uniforme, 7/10 contre un critere pose a 8/10), et "u n'a pas eu le temps de
diverger" (exclue par lecture du code : sigma_baseline = 0.05).

La seule piste restante ecrite au froid etait : "la coupe est une variable
DISCRETE a faible dispersion, donc deux series de meme loi ont souvent le meme
maximum". Elle n'avait jamais ete testee. Elle l'est ici (M3), avec deux autres
mesures que le harnais du 27/07 calculait deja sans jamais les regarder.

--------------------------------------------------------------------------------
L1 -- CE QUE LA LECTURE DU CODE ETABLIT, SANS AUCUNE SIMULATION
--------------------------------------------------------------------------------
Dans `compute_cut_and_energy` (p15_maxcut_ising_poc.py:122, copie a l'identique ici) :

    energy = -0.5 * s @ J @ s
    W_cut  =  0.25 * sum(|J| - J * outer(s, s))  =  0.25 * (sum|J| - s @ J @ s)

donc, pour un graphe J donne :

    W_cut = 0.25 * sum|J| + 0.5 * energy          (bijection AFFINE)

Consequence directe sur l'enonce de la question ouverte : le test d'identite du
script audite,

    identical = (cut_full == cut_froz) AND (E_full == E_froz)

est REDONDANT. Les deux membres du ET sont la meme egalite ecrite deux fois. Ce
qui rendait l'observation frappante -- "deux quantites independantes coincident
exactement" -- n'existe pas : il n'y a qu'UNE quantite. Toute la surprise portee
par le mot "ET" est un artefact d'enonce.
(Reserve d'arithmetique flottante : cut et E sont calcules par deux chemins
numeriques differents, donc l'egalite bit-a-bit de l'une n'implique pas
FORMELLEMENT celle de l'autre. L1 verifie donc numeriquement le residu maximal de
l'identite affine sur les 20 graines, et compte les cas ou les deux tests
divergeraient. C'est une verification, pas une hypothese.)

--------------------------------------------------------------------------------
CRITERES POSES AVANT L'EXECUTION (aucun chiffre connu au moment de les ecrire)
--------------------------------------------------------------------------------
G0 -- GATE DE FIDELITE, repris de P15b. Les 10 couples (cut, E) des graines 0-9
     doivent egaler au bit pres ceux de `figures/p15_maxcut_benchmark.csv`, et la
     baseline best-of-300 doit reproduire `cut_random_budget`. Gate echoue =>
     tout ce qui suit est invalide.

M1 -- MEME VALEUR, OU MEME VECTEUR ? (la mesure que le harnais du 27/07 stockait
     sans la lire). Sur les graines dites identiques, distance de Hamming entre
     best_s(FULL) et best_s(FROZEN_U).
     RETENUE comme explication si Hamming == 0 sur >= 8 des graines identiques :
     alors la collision n'est pas une coincidence de VALEUR entre deux maximums
     distincts, c'est LE MEME ECHANTILLON retenu de part et d'autre, et la
     question se deplace vers le recouvrement des trajectoires (M2).
     Si Hamming > 0 partout : deux configurations DIFFERENTES rendent la meme
     coupe, et c'est la degenerescence du paysage qu'il faut regarder.
     DISCRIMINANT : le meme Hamming est rapporte sur les graines non identiques.

M2 -- RECOUVREMENT DES ECHANTILLONS. Fraction des 300 pas echantillonnes ou
     sign(v) est EXACTEMENT identique entre les deux conditions (serie deja
     calculee par P15b sous le nom `s_equal`, dont seul le dernier element etait
     utilise), et fraction des pas ou les deux coupes sont egales.
     Critere de SEPARATION, pose avant : la fraction de pas a signes identiques
     doit separer les deux groupes -- mediane(identiques) > mediane(autres) ET
     AUC >= 0.75 (probabilite qu'une graine identique tiree au hasard depasse une
     graine non identique tiree au hasard). Sinon la metrique n'explique rien,
     meme si elle est elevee : c'est la regle du discriminant obligatoire du 27/07.

M3 -- LA PISTE LAISSEE AU FROID, ENFIN TESTEE : la discretude suffit-elle ?
     Sur chaque graphe, DEUX series de 300 tirages de spins uniformes,
     INDEPENDANTES l'une de l'autre (RNG disjoints), et on regarde si leurs
     maximums coincident. C'est exactement ce qu'affirme la piste : "deux series
     de meme loi partagent souvent le meme maximum".
     RETENUE   si le taux de collision du null est >= 0.40 (comparable au 10/20
               observe) : la discretude de la coupe suffit a produire le
               phenomene, et il n'y a rien a expliquer de plus.
     REJETEE   si le taux est <= 0.15 (facteur >= 3 sous l'observation).
     NON TRANCHE entre les deux, et rapporte comme tel.
     Fait de structure verifie au passage : J_ij est a valeurs dans {-1, 0, +1}
     (adjacence binaire x poids +-1, symetrisee), tr J = 0, donc s @ J @ s est un
     entier PAIR et la coupe vit sur une grille de pas 0.5. La granularite est
     donc mesuree, pas supposee.

M4 -- COMBIEN D'ECHANTILLONS INDEPENDANTS M4R PRODUIT-IL VRAIMENT ? (observation
     chiffree, aucun verdict n'en depend). On cherche n_eff tel que l'esperance du
     meilleur de n_eff tirages UNIFORMES egale la coupe rapportee par M4R sur ses
     300 lectures. Si n_eff << 300, la defaite contre best-of-300 est quantifiee :
     M4R paye ses 300 lectures le prix de n_eff tirages, parce que les etats
     visites sont correles.

--------------------------------------------------------------------------------
VOLET 2 -- LE MECANISME COMPLET (criteres poses APRES lecture des sorties du
volet 1, AVANT toute execution du volet 2)
--------------------------------------------------------------------------------
Ce que le volet 1 etablit (gate G0 passe, 20 graines) :
  L1 residu de la bijection affine EXACTEMENT nul, et les deux tests d'identite
     coincident sur 20/20 : le "ET energie" ne portait aucune information.
  M1 sur les 10 graines identiques, best_s est LE MEME VECTEUR dans 9 cas sur 10.
     Ce n'est donc PAS une coincidence de valeur entre deux maximums distincts :
     les deux conditions retiennent le meme etat.
  M2 le recouvrement global NE SEPARE PAS (AUC 0.68 < 0.75) -- rejete a son critere.
  M3 la piste laissee au froid est REJETEE : le null independant collisionne a
     0.10 contre 0.50 observe, un facteur 5.
  M4 n_eff = 5 : M4R paye 300 lectures pour l'equivalent de 5 tirages independants.

Et une observation non prevue qui recadre tout : l'ecart-type des coupes VISITEES
par M4R vaut 1.83, alors que la courbe de M4 implique un ecart-type d'environ 6
pour un tirage uniforme (best-of-2 = mu + 0.564*sigma). La relaxation a donc la
MEME moyenne que le hasard (73) mais une dispersion 3 fois plus FAIBLE. C'est
precisement ce que R5 (27/07) n'avait pas pu voir : son critere portait sur la
MOYENNE, et la moyenne, elle, coincide. R2 a le meme angle mort -- il testait si
la coupe se degrade, et elle ne se degrade pas : c'est sa VARIANCE qui s'effondre.

Or le best tombe tres tot (mediane t_best = 30 pas sur 3000, 16/20 avant 300).
D'ou l'explication a tester ici : le reseau part de v = w = 0, traverse un
TRANSITOIRE ou sign(v) est essentiellement le signe du bruit -- donc un tirage
quasi uniforme, a forte dispersion -- puis tombe dans un regime stationnaire
concentre qui ne produit plus jamais rien de meilleur. Le "best" de M4R est un
echantillon de son transitoire ; les 270 lectures suivantes sont du remplissage.
Et comme le transitoire precede l'accumulation de l'effet de u, les deux
conditions y voient le meme etat : d'ou l'identite, sans aucune coincidence.

V_A DISPERSION QUI S'EFFONDRE. sd(cut) sur les pas echantillonnes t < 300
    (EARLY) contre t >= 300 (LATE). CONFIRMEE si sd_early > 1.5 * sd_late sur
    >= 16 graines sur 20, dans LES DEUX conditions.
V_B LE BEST EST DANS LE TRANSITOIRE. CONFIRMEE si max(cut | t >= 300) < best
    global sur >= 16 graines sur 20 : apres le transitoire, plus rien n'est
    trouve. (Test distinct de "t_best < 300" : il porte sur ce que le regime
    stationnaire est CAPABLE de produire, pas sur la date du record.)
V_C LA DIVERGENCE S'ACCUMULE. Hamming moyen entre conditions sur EARLY contre
    LATE. CONFIRMEE si hamming_early < hamming_late sur >= 16 graines sur 20 :
    l'effet de u sur les SIGNES met du temps a devenir lisible, alors qu'il
    touche v des le pas 1.
V_D LE TRANSITOIRE EST-IL QUASI UNIFORME ? Ecart relatif entre sd_early et
    l'ecart-type d'un tirage uniforme mesure sur le meme graphe. CONFIRMEE si
    cet ecart est < 35 pct sur >= 14 graines sur 20. Si oui, la formulation
    finale est complete : M4R echantillonne comme le hasard pendant son
    transitoire, puis s'eteint.
V_E OBSERVATION (aucun verdict n'en depend) : nombre de graines ou
    t_best(FULL) == t_best(FROZEN_U), croise avec l'identite.

--------------------------------------------------------------------------------
VOLET 3 -- LE BOUCLAGE (critere pose apres lecture du volet 2, avant execution)
--------------------------------------------------------------------------------
Le volet 2 confirme V_B et V_C, et rejette V_A et V_D -- mais leurs rejets
apprennent autant que les confirmations :
  V_A l'effondrement de dispersion n'est PAS commun aux deux conditions. FULL
      s'effondre sur 20/20 graines (sd 3.01 -> 0.47), FROZEN_U sur 8/20
      seulement (3.01 -> 2.01). C'est donc bien u qui FIGE le reseau, et ce
      constat est nouveau : il n'a rien a voir avec la question posee.
  V_D le transitoire n'est pas non plus un tirage uniforme : sd 3.01 contre 6.03
      mesures sur le meme graphe. M4R est deja deux fois moins disperse que le
      hasard AVANT de se figer -- ce qui suffit a expliquer sa defaite (M4) sans
      invoquer le regime stationnaire.

Il manque une seule chose pour fermer la question : la preuve que les deux
conditions PARTAGENT l'etat qu'elles retiennent, plutot que de tomber par hasard
sur la meme valeur. M1 le montre pour les graines identiques (meme vecteur 9 fois
sur 10). Le test symetrique, sur TOUTES les graines, est celui-ci :

V3_A L'etat optimal de FULL est-il VISITE par FROZEN_U a un instant quelconque de
     ses 300 lectures (et reciproquement) ? CONFIRMEE si oui sur >= 16 graines
     sur 20. Alors les deux conditions explorent le meme ensemble d'etats, et
     l'identite de la coupe est mecanique -- pas une coincidence.
V3_B DECOMPOSITION des graines NON identiques : dans combien de cas l'etat
     optimal de l'une est-il visite par l'autre, mais BATTU par un meilleur etat
     que la premiere n'a pas visite ? Observation : elle dit si la difference
     vient d'un desaccord d'exploration ou d'un desaccord de selection.
V3_C Le record est-il battu a un instant ou les deux conditions sont EXACTEMENT
     dans le meme etat de signes ? Observation, par graine.

--------------------------------------------------------------------------------
VOLET 4 -- LE GATE DE REPLICATION (critere pose apres le volet 3, avant execution
sur les graines 20-39, qui n'ont jamais ete touchees par ce projet)
--------------------------------------------------------------------------------
Le volet 3 rejette V3_A a son critere (13/20 et 9/20, contre 16/20 demande) mais
livre deux chiffres bruts qui expliquent tout, et une observation qui separe :

  - Sur 300 lectures, le nombre d'etats de signes DISTINCTS visites vaut 24 en
    FULL et 40 en FROZEN_U. Le reseau est quasi immobile : voila le n_eff = 5 de
    M4 en chair et en os. Et le doute en visite MOINS que le doute gele.
  - V3_C separe : le record est battu dans un etat commun aux deux conditions
    dans 9 cas sur 10 chez les graines identiques, contre 3 sur 10 chez les
    autres. C'est le discriminant que M2 cherchait -- au mauvais endroit : M2
    moyennait le recouvrement sur les 300 pas, alors que seul compte l'instant
    du record.

MAIS V3_C avait ete pose comme OBSERVATION, sans critere prealable. Le repecher
en test decisif apres avoir vu sa valeur est exactement la faute que ce projet a
deja payee deux fois (le Condorcet du 13/07, le -0.30 du 26/07). Il passe donc
un gate de replication sur des graines neuves, avec son critere ecrit ici, avant
la moindre execution sur les graines 20-39 :

R_V3C REPLICATION DU DISCRIMINANT. Sur les graines 20-39 :
      part(record en etat commun | graines identiques) >= 0.70
      ET  cette part depasse celle des graines non identiques d'au moins 0.30.
      Si les deux conditions sont remplies, l'explication est citable. Sinon,
      elle rejoint les six precedentes et la question reste ouverte.
R_ETATS OBSERVATION : nombre d'etats distincts visites, et taux d'identite, sur
      les graines neuves. Dit si le phenomene est generique ou propre aux 20
      premieres graines.

--------------------------------------------------------------------------------
RESULTATS (28/07/2026, gate G0 passe : les runs mesures sont bien ceux de P15)
--------------------------------------------------------------------------------
La question ouverte du 27/07 est FERMEE. Elle avait deux defauts d'enonce et une
reponse mecanique, et aucune des trois ne concerne u.

1. L'ENONCE COMPTAIT DEUX FOIS LA MEME CHOSE (L1, zero simulation).
   cut = 0.25*sum|J| + 0.5*E : residu exactement nul sur 20/20 graines, et les
   deux tests d'identite coincident sur 20/20. "La coupe ET l'energie sont
   identiques" est UNE egalite, pas deux. Ce qui rendait l'observation
   frappante n'existait pas.

2. CE N'EST PAS UNE COINCIDENCE DE VALEUR, C'EST LE MEME ETAT (M1).
   best_s(FULL) == best_s(FROZEN_U) comme VECTEUR sur 9 graines identiques sur
   10, et 11 sur 11 en replication. La question "pourquoi la meme coupe" se
   dissout : les deux conditions retiennent la meme configuration de spins.

3. LA PISTE LAISSEE AU FROID EST MORTE (M3). "La coupe est discrete a faible
   dispersion, donc deux series de meme loi partagent souvent leur maximum" :
   deux series de 300 tirages uniformes INDEPENDANTES collisionnent sur 0.10
   (0.15 en replication) contre 0.50 observe -- facteur 5. Septieme explication
   morte a son propre critere depuis le 27/07.

4. LE MECANISME, REPLIQUE (V3_C puis R_V3C sur graines 20-39 jamais touchees) :
   le record est battu dans un etat de signes que les DEUX conditions occupent
   au meme instant -- 0.90 puis 1.00 chez les graines identiques, contre 0.30
   puis 0.22 chez les autres (critere pose avant : >= 0.70 et ecart >= 0.30).
   Et quand elles different, c'est un desaccord d'EXPLORATION, jamais de
   selection : sur les 10 graines non identiques, 0 cas ou les deux etats
   optimaux sont visites des deux cotes (V3_B ; la selection est un maximum,
   donc deterministe a ensemble egal).

5. LE FAIT DUR, NON CHERCHE (V3_A, M4, V_A). Sur 300 lectures, M4R ne visite que
   ~24 etats de signes DISTINCTS en FULL et ~40 en FROZEN_U (25 / 45 en
   replication). Son n_eff vaut 5 : il paie 300 lectures pour l'equivalent de
   5 tirages independants, facteur 60. Deux runs quasi immobiles issus du meme
   bruit tombent naturellement sur le meme etat -- l'identite n'a plus rien
   d'etonnant, c'est l'inverse qui demanderait une explication.
   Trois consequences a garder :
   (a) LE DOUTE EXPLORE MOINS QUE LE DOUTE GELE (24 contre 40 etats), et il fige
       la dispersion de la coupe la ou FROZEN_U la garde : sd early -> late,
       3.01 -> 0.47 en FULL sur 20/20 graines, contre 3.01 -> 2.01 en FROZEN_U
       sur 8/20 seulement. Sur cette tache, u est un frein a l'exploration.
       (V_A a ete pose comme un effet commun aux deux conditions : REJETE en
       tant que tel, et c'est son rejet qui apprend.)
   (b) R2 et R5 du 27/07 avaient le meme angle mort : ils testaient la MOYENNE
       de la coupe, qui ne bouge pas (72.9 -> 73.8). C'est la VARIANCE qui
       s'effondre. Un critere pose sur la mauvaise statistique ne se rattrape
       pas en repechant son seuil.
   (c) Le transitoire lui-meme n'est PAS un tirage uniforme (V_D REJETE) :
       sd 3.01 contre 6.03 mesures sur le meme graphe. M4R est deja deux fois
       moins disperse que le hasard AVANT de se figer -- ce qui suffit a
       expliquer sa defaite du 27/07 contre best-of-300.

CE QUE CELA NE DIT PAS. Tout ceci porte sur une tache HORS de la niche du projet,
ou M4R perd deja contre le tirage aleatoire a budget egal (27/07, 20/20 graines).
C'est l'explication d'une identite de MESURE, pas une propriete de u -- ce qui
etait annonce avant d'ouvrir la question. Le -1.20 FULL-FROZEN_U reste a ne pas
citer, et pour une raison de plus : c'est l'ecart entre deux maximums pris sur
~24 et ~40 etats, souvent le meme.

Statut : exploration (colonne B). Aucun CSV canonique, aucun chiffre du preprint,
aucun claim du Guardian n'est touche. Ce script ne modifie ni
`p15_maxcut_ising_poc.py` ni `p15b_maxcut_identity_diagnosis.py`.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mem4ristor.graph_utils import make_ba
from mem4ristor.dynamics import Mem4ristorV3

N_SEEDS = 20
N_NODES = 100
N_STEPS = 3000
SAMPLE_EVERY = 10
N_DRAWS = N_STEPS // SAMPLE_EVERY       # 300, le budget d'echantillons de M4R
N_RESERVE = 4000                        # reserve de tirages uniformes pour M4


def compute_cut_and_energy(s, J):
    """Copie a l'identique de p15_maxcut_ising_poc.compute_cut_and_energy."""
    energy = -0.5 * np.dot(s, np.dot(J, s))
    W_cut = 0.25 * np.sum(np.abs(J) - J * np.outer(s, s))
    return energy, W_cut


def build_J(seed, N=N_NODES):
    """Reproduit EXACTEMENT la sequence de consommation du RNG global de P15."""
    np.random.seed(seed)
    adj_base = make_ba(N, m=3, seed=seed)
    weights = np.random.choice([-1.0, 1.0], size=adj_base.shape)
    J = adj_base * weights
    J = (J + J.T) / 2.0
    return J


def make_model(seed, N=N_NODES):
    model = Mem4ristorV3(seed=seed)
    model.cfg['coupling']['heretic_ratio'] = 0.15
    model._initialize_params(N, cold_start=True)
    return model


def run_pair(J, seed, steps=N_STEPS):
    """FULL et FROZEN_U avances EN PARALLELE (memes seeds => meme bruit).

    Identique au harnais de P15b, avec DEUX ajouts : on conserve les vecteurs
    best_s des deux cotes (M1) et la serie complete des accords de signe (M2).
    """
    m_full = make_model(seed)
    m_froz = make_model(seed)

    best = {'full': {'cut': 0.0, 'E': 0.0, 't': None, 's': None},
            'froz': {'cut': 0.0, 'E': 0.0, 't': None, 's': None}}

    s_equal = []        # sign(v) exactement identique entre conditions
    cut_equal = []      # coupes egales a ce pas (peut etre vrai avec s different)
    hamming = []        # nombre de noeuds de signe different
    cuts_full = []
    cuts_froz = []
    seen_full = []      # volet 3 : tous les etats de signes visites, encodes
    seen_froz = []

    for step in range(steps):
        m_froz.u[:] = 0.5

        m_full.step(I_stimulus=0.0, coupling_input=np.dot(J, m_full.v))
        m_froz.step(I_stimulus=0.0, coupling_input=np.dot(J, m_froz.v))

        if step % SAMPLE_EVERY == 0:
            s_f = np.sign(m_full.v); s_f[s_f == 0] = 1
            s_z = np.sign(m_froz.v); s_z[s_z == 0] = 1

            E_f, cut_f = compute_cut_and_energy(s_f, J)
            E_z, cut_z = compute_cut_and_energy(s_z, J)

            if cut_f > best['full']['cut']:
                best['full'].update(cut=cut_f, E=E_f, t=step, s=s_f.copy())
            if cut_z > best['froz']['cut']:
                best['froz'].update(cut=cut_z, E=E_z, t=step, s=s_z.copy())

            seen_full.append(s_f.astype(np.int8).tobytes())
            seen_froz.append(s_z.astype(np.int8).tobytes())

            s_equal.append(bool(np.array_equal(s_f, s_z)))
            cut_equal.append(bool(cut_f == cut_z))
            hamming.append(int(np.sum(s_f != s_z)))
            cuts_full.append(cut_f)
            cuts_froz.append(cut_z)

    return best, {
        'steps': np.arange(0, steps, SAMPLE_EVERY),
        's_equal': np.asarray(s_equal),
        'cut_equal': np.asarray(cut_equal),
        'hamming': np.asarray(hamming),
        'cuts_full': np.asarray(cuts_full),
        'cuts_froz': np.asarray(cuts_froz),
        'seen_full': set(seen_full),
        'seen_froz': set(seen_froz),
    }


def best_of_random_loop(J, n_draws, rng, N=N_NODES):
    """Meilleur de n_draws tirages uniformes. Boucle FIDELE a P15/P15b (meme
    ordre de consommation du RNG) pour que le gate G0 puisse porter dessus."""
    best_cut = 0.0
    for _ in range(n_draws):
        s = rng.choice([-1.0, 1.0], size=N)
        _, cut = compute_cut_and_energy(s, J)
        if cut > best_cut:
            best_cut = cut
    return best_cut


def cut_reserve(J, n, rng, N=N_NODES):
    """n coupes de configurations uniformes, calculees en bloc (M4 uniquement :
    aucune fidelite de RNG requise ici, seule la loi compte)."""
    S = rng.choice([-1.0, 1.0], size=(n, N))
    quad = np.einsum('ij,ij->i', S @ J, S)
    return 0.25 * (np.sum(np.abs(J)) - quad)


def auc(pos, neg):
    """Probabilite qu'un tirage de `pos` depasse un tirage de `neg` (egalites
    comptees 0.5). Statistique de Mann-Whitney normalisee."""
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def sweep(seeds):
    """Rejoue le couple FULL/FROZEN_U instrumente sur une liste de graines."""
    rows = []
    for seed in seeds:
        J = build_J(seed)
        best, rec = run_pair(J, seed)

        # Baseline fidele a P15/P15b (meme RNG, meme ordre) -> sert au gate G0.
        rnd_a = best_of_random_loop(J, N_DRAWS, np.random.RandomState(10_000 + seed))
        # Serie INDEPENDANTE de la premiere : le null de discretude (M3).
        rnd_b = best_of_random_loop(J, N_DRAWS, np.random.RandomState(20_000 + seed))

        cut_f = best['full']['cut']
        cut_z = best['froz']['cut']
        E_f = best['full']['E']
        E_z = best['froz']['E']

        sum_abs_J = float(np.sum(np.abs(J)))
        # L1 : residu de l'identite affine cut = 0.25*sum|J| + 0.5*E
        res_f = abs(cut_f - (0.25 * sum_abs_J + 0.5 * E_f))
        res_z = abs(cut_z - (0.25 * sum_abs_J + 0.5 * E_z))

        ident_cut = bool(cut_f == cut_z)
        ident_E = bool(E_f == E_z)
        identical = bool(ident_cut and ident_E)          # test d'origine (P15)

        ham_best = int(np.sum(best['full']['s'] != best['froz']['s']))

        # --- volet 2 : decoupage transitoire / regime stationnaire -------------
        early = rec['steps'] < 300
        late = ~early
        sd_e_f = float(np.std(rec['cuts_full'][early]))
        sd_l_f = float(np.std(rec['cuts_full'][late]))
        sd_e_z = float(np.std(rec['cuts_froz'][early]))
        sd_l_z = float(np.std(rec['cuts_froz'][late]))
        # ecart-type d'un tirage uniforme sur CE graphe (mesure, pas deduit)
        sd_unif = float(np.std(cut_reserve(J, N_RESERVE,
                                           np.random.RandomState(40_000 + seed))))

        # --- volet 3 : les deux conditions visitent-elles le meme etat optimal ?
        bf = best['full']['s'].astype(np.int8).tobytes()
        bz = best['froz']['s'].astype(np.int8).tobytes()
        full_best_seen_by_froz = bf in rec['seen_froz']
        froz_best_seen_by_full = bz in rec['seen_full']
        i_best = best['full']['t'] // SAMPLE_EVERY
        s_equal_at_best = bool(rec['s_equal'][i_best])

        rows.append({
            'seed': seed,
            'cut_full': cut_f, 'cut_froz': cut_z,
            'E_full': E_f, 'E_froz': E_z,
            'identical': identical,
            'ident_cut_only': ident_cut, 'ident_E_only': ident_E,
            'affine_residual': max(res_f, res_z),
            't_best_full': best['full']['t'], 't_best_froz': best['froz']['t'],
            'hamming_best': ham_best,
            'frac_s_equal': float(rec['s_equal'].mean()),
            'frac_cut_equal': float(rec['cut_equal'].mean()),
            'hamming_median': float(np.median(rec['hamming'])),
            'cut_visited_sd': float(np.std(np.concatenate([rec['cuts_full'],
                                                           rec['cuts_froz']]))),
            'sd_early_full': sd_e_f, 'sd_late_full': sd_l_f,
            'sd_early_froz': sd_e_z, 'sd_late_froz': sd_l_z,
            'sd_uniform': sd_unif,
            'max_cut_late_full': float(rec['cuts_full'][late].max()),
            'max_cut_late_froz': float(rec['cuts_froz'][late].max()),
            'hamming_early': float(rec['hamming'][early].mean()),
            'hamming_late': float(rec['hamming'][late].mean()),
            'frac_s_equal_early': float(rec['s_equal'][early].mean()),
            'frac_s_equal_late': float(rec['s_equal'][late].mean()),
            'cut_mean_early_full': float(rec['cuts_full'][early].mean()),
            'cut_mean_late_full': float(rec['cuts_full'][late].mean()),
            'full_best_seen_by_froz': bool(full_best_seen_by_froz),
            'froz_best_seen_by_full': bool(froz_best_seen_by_full),
            's_equal_at_best': s_equal_at_best,
            'n_states_full': len(rec['seen_full']),
            'n_states_froz': len(rec['seen_froz']),
            'n_states_common': len(rec['seen_full'] & rec['seen_froz']),
            'cut_rnd_a': rnd_a, 'cut_rnd_b': rnd_b,
            'rnd_collision': bool(rnd_a == rnd_b),
            'sum_abs_J': sum_abs_J,
            'cut_expected_random': 0.25 * (sum_abs_J - float(np.trace(J))),
            'J_values_ok': bool(np.all(np.isin(np.unique(J), [-1.0, 0.0, 1.0]))),
            'trace_J': float(np.trace(J)),
        })

        print("seed %2d | ident=%-5s | cut F/Z = %6.1f / %6.1f | hamming(best)=%3d | "
              "s_equal=%4.1f pct | rnd A/B = %6.1f / %6.1f %s"
              % (seed, identical, cut_f, cut_z, ham_best,
                 100 * rows[-1]['frac_s_equal'], rnd_a, rnd_b,
                 "<-- collision" if rnd_a == rnd_b else ""))

    return pd.DataFrame(rows)


def record_in_common_state(d):
    """V3_C : parts de graines dont le record est battu dans un etat de signes
    commun aux deux conditions, chez les identiques et chez les autres."""
    i = d[d.identical]
    o = d[~d.identical]
    f_i = float(i.s_equal_at_best.mean()) if len(i) else float('nan')
    f_o = float(o.s_equal_at_best.mean()) if len(o) else float('nan')
    return f_i, f_o, len(i), len(o)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, ".."))

    print("=" * 78)
    print("P15c -- MECANISME DE L'IDENTITE FULL/FROZEN_U SUR LE MAX-CUT")
    print("=" * 78)
    print("Criteres poses avant execution : voir l'en-tete du fichier.")
    print("")

    df = sweep(range(N_SEEDS))
    out_csv = os.path.join(repo, "figures", "p15c_maxcut_identity_mechanism.csv")
    df.to_csv(out_csv, index=False)
    print("")
    print("[CSV enregistre] : %s" % out_csv)

    # ------------------------------------------------------------------ G0 ----
    print("")
    print("-" * 78)
    print("G0 -- GATE DE FIDELITE (les runs mesures sont-ils ceux de P15 ?)")
    print("-" * 78)
    ref = pd.read_csv(os.path.join(repo, "figures", "p15_maxcut_benchmark.csv"))
    m = df.merge(ref[['seed', 'cut_m4r', 'E_m4r', 'cut_frozen', 'E_frozen',
                      'cut_random_budget']], on='seed')
    ok_full = bool(np.array_equal(m.cut_full.values, m.cut_m4r.values) and
                   np.array_equal(m.E_full.values, m.E_m4r.values))
    ok_froz = bool(np.array_equal(m.cut_froz.values, m.cut_frozen.values) and
                   np.array_equal(m.E_froz.values, m.E_frozen.values))
    ok_rnd = bool(np.array_equal(m.cut_rnd_a.values, m.cut_random_budget.values))
    print("  FULL     reproduit au bit pres : %s  (sur %d graines communes)" % (ok_full, len(m)))
    print("  FROZEN_U reproduit au bit pres : %s" % ok_froz)
    print("  baseline best-of-300 reproduite : %s" % ok_rnd)
    if not (ok_full and ok_froz and ok_rnd):
        print("  >> GATE ECHOUE : tout ce qui suit est INVALIDE, ne rien citer.")
        return
    print("  >> GATE PASSE.")

    ident = df[df.identical]
    other = df[~df.identical]
    print("")
    print("  Graines strictement identiques : %d sur %d (seeds %s)"
          % (len(ident), len(df), list(ident.seed.values)))

    # ------------------------------------------------------------------ L1 ----
    print("")
    print("-" * 78)
    print("L1 -- LE 'ET ENERGIE' EST VIDE : cut et E sont en bijection affine")
    print("-" * 78)
    print("  cut = 0.25*sum|J| + 0.5*E  -- residu maximal sur les %d graines : %.3e"
          % (len(df), df.affine_residual.max()))
    n_disagree = int((df.ident_cut_only != df.ident_E_only).sum())
    print("  graines ou le test sur la coupe et le test sur l'energie divergeraient : %d"
          % n_disagree)
    print("  identiques par la coupe seule : %d | par l'energie seule : %d | par le ET : %d"
          % (int(df.ident_cut_only.sum()), int(df.ident_E_only.sum()), int(df.identical.sum())))
    print("  >> L'observation d'origine porte sur UNE egalite, pas deux. Le mot 'ET'")
    print("     de l'enonce ne renforce rien : il repete la meme mesure.")

    print("")
    print("  Structure du probleme (verifiee, pas supposee) :")
    print("    J a valeurs dans {-1, 0, +1} sur les %d graines : %s"
          % (len(df), bool(df.J_values_ok.all())))
    print("    trace(J) maximale : %.1f  => s @ J @ s est un entier PAIR"
          % df.trace_J.abs().max())
    all_cuts = np.concatenate([df.cut_full.values, df.cut_froz.values])
    grid_ok = bool(np.all(np.abs(all_cuts / 0.5 - np.round(all_cuts / 0.5)) < 1e-9))
    print("    toutes les coupes sont des multiples de 0.5 : %s" % grid_ok)
    print("    dispersion des coupes visitees (ecart-type moyen) : %.2f"
          % df.cut_visited_sd.mean())
    print("    => la coupe vit sur une grille de pas 0.5, avec un ecart-type de ~%.0f,"
          % df.cut_visited_sd.mean())
    print("       soit environ %.0f pas de grille dans un ecart-type."
          % (df.cut_visited_sd.mean() / 0.5))

    # ------------------------------------------------------------------ M1 ----
    print("")
    print("-" * 78)
    print("M1 -- MEME VALEUR, OU MEME VECTEUR ? (critere : Hamming == 0 sur >= 8")
    print("      des graines identiques)")
    print("-" * 78)
    n_same_vec = int((ident.hamming_best == 0).sum())
    print("  graines identiques dont le best_s est LE MEME VECTEUR : %d sur %d"
          % (n_same_vec, len(ident)))
    print("  distances de Hamming, graines identiques  : %s" % sorted(ident.hamming_best.values))
    print("  distances de Hamming, graines non ident.  : %s" % sorted(other.hamming_best.values))
    m1_ok = bool(n_same_vec >= 8)
    print("  >> M1 : %s" % ("RETENUE -- c'est LE MEME ECHANTILLON des deux cotes"
                            if m1_ok else
                            "REJETEE -- deux configurations DIFFERENTES rendent la meme coupe"))

    # ------------------------------------------------------------------ M2 ----
    print("")
    print("-" * 78)
    print("M2 -- RECOUVREMENT DES ECHANTILLONS (critere de separation :")
    print("      mediane(identiques) > mediane(autres) ET AUC >= 0.75)")
    print("-" * 78)
    med_i = float(ident.frac_s_equal.median())
    med_o = float(other.frac_s_equal.median()) if len(other) else float('nan')
    a = auc(ident.frac_s_equal.values, other.frac_s_equal.values)
    print("  fraction des 300 pas ou sign(v) est EXACTEMENT identique :")
    print("    graines identiques     : mediane %5.1f pct  (min %.1f, max %.1f)"
          % (100 * med_i, 100 * ident.frac_s_equal.min(), 100 * ident.frac_s_equal.max()))
    print("    graines non identiques : mediane %5.1f pct  (min %.1f, max %.1f)"
          % (100 * med_o, 100 * other.frac_s_equal.min(), 100 * other.frac_s_equal.max()))
    print("    AUC = %.2f" % a)
    print("  fraction des pas ou les deux COUPES sont egales :")
    print("    identiques %5.1f pct | autres %5.1f pct"
          % (100 * ident.frac_cut_equal.median(), 100 * other.frac_cut_equal.median()))
    print("  Hamming median entre conditions, tous pas confondus :")
    print("    identiques %5.1f noeuds sur %d | autres %5.1f"
          % (ident.hamming_median.median(), N_NODES, other.hamming_median.median()))
    m2_ok = bool((med_i > med_o) and (a >= 0.75))
    print("  >> M2 : %s" % ("SEPARE les deux groupes" if m2_ok
                            else "NE SEPARE PAS -- n'explique rien (regle du discriminant)"))

    # ------------------------------------------------------------------ M3 ----
    print("")
    print("-" * 78)
    print("M3 -- LA PISTE LAISSEE AU FROID : la discretude suffit-elle ?")
    print("      (deux series de 300 tirages uniformes INDEPENDANTES par graphe ;")
    print("       RETENUE si taux >= 0.40, REJETEE si <= 0.15)")
    print("-" * 78)
    n_coll = int(df.rnd_collision.sum())
    rate = n_coll / len(df)
    obs_rate = len(ident) / len(df)
    print("  collisions du null independant : %d sur %d  (taux %.2f)" % (n_coll, len(df), rate))
    print("  a comparer au taux OBSERVE entre FULL et FROZEN_U : %.2f (%d sur %d)"
          % (obs_rate, len(ident), len(df)))
    if rate >= 0.40:
        m3 = "RETENUE -- la discretude de la coupe suffit, il n'y a rien de plus a expliquer"
    elif rate <= 0.15:
        m3 = ("REJETEE -- la discretude seule produit %.0fx moins de collisions que l'observe"
              % (obs_rate / rate if rate > 0 else float('inf')))
    else:
        m3 = "NON TRANCHE (entre les deux bornes posees avant mesure)"
    print("  >> M3 : %s" % m3)

    # ------------------------------------------------------------------ M4 ----
    print("")
    print("-" * 78)
    print("M4 -- COMBIEN D'ECHANTILLONS INDEPENDANTS M4R PRODUIT-IL ? (observation)")
    print("-" * 78)
    ns = [1, 2, 3, 5, 8, 12, 20, 35, 60, 100, 150, 200, 300]
    curve = {n: [] for n in ns}
    for seed in range(N_SEEDS):
        J = build_J(seed)
        rng = np.random.RandomState(30_000 + seed)
        cuts = cut_reserve(J, N_RESERVE, rng)
        for n in ns:
            k = N_RESERVE // n
            block_max = cuts[:k * n].reshape(k, n).max(axis=1)
            curve[n].append(float(block_max.mean()))
    curve_mean = {n: float(np.mean(v)) for n, v in curve.items()}
    target = float(df.cut_full.mean())
    print("  coupe M4R FULL (moyenne des 20 graines, 300 lectures) : %.2f" % target)
    print("  esperance du meilleur de n tirages uniformes :")
    for n in ns:
        mark = ""
        print("    n = %3d  ->  %6.2f%s" % (n, curve_mean[n], mark))
    below = [n for n in ns if curve_mean[n] <= target]
    n_eff = max(below) if below else None
    print("  >> n_eff (plus grand n dont le best-of-n uniforme ne depasse pas M4R) : %s"
          % (n_eff if n_eff is not None else "< 1"))
    if n_eff is not None:
        print("     M4R paye %d lectures et obtient ce que %d tirages independants donnent :"
              % (N_DRAWS, n_eff))
        print("     un facteur ~%.0f de perte, imputable a la correlation des etats visites."
              % (N_DRAWS / n_eff))

    # ============================== VOLET 2 ==================================
    print("")
    print("=" * 78)
    print("VOLET 2 -- LE MECANISME : UN TRANSITOIRE UTILE, PUIS PLUS RIEN")
    print("=" * 78)

    print("")
    print("-" * 78)
    print("V_A -- LA DISPERSION S'EFFONDRE (critere : sd_early > 1.5*sd_late sur")
    print("       >= 16/20, dans LES DEUX conditions)")
    print("-" * 78)
    va_f = int((df.sd_early_full > 1.5 * df.sd_late_full).sum())
    va_z = int((df.sd_early_froz > 1.5 * df.sd_late_froz).sum())
    print("  FULL   : %2d/%d graines   (sd moyen  early %.2f -> late %.2f)"
          % (va_f, len(df), df.sd_early_full.mean(), df.sd_late_full.mean()))
    print("  FROZEN : %2d/%d graines   (sd moyen  early %.2f -> late %.2f)"
          % (va_z, len(df), df.sd_early_froz.mean(), df.sd_late_froz.mean()))
    print("  moyenne de la coupe, elle : early %.2f -> late %.2f  (c'est l'angle mort"
          % (df.cut_mean_early_full.mean(), df.cut_mean_late_full.mean()))
    print("  de R2 et R5 du 27/07 : ils regardaient la moyenne, qui ne bouge pas)")
    va_ok = (va_f >= 16 and va_z >= 16)
    print("  >> V_A : %s" % ("CONFIRMEE" if va_ok else "NON CONFIRMEE"))

    print("")
    print("-" * 78)
    print("V_B -- LE BEST EST DANS LE TRANSITOIRE (critere : max(cut|t>=300) < best")
    print("       global sur >= 16/20)")
    print("-" * 78)
    vb_f = int((df.max_cut_late_full < df.cut_full).sum())
    vb_z = int((df.max_cut_late_froz < df.cut_froz).sum())
    print("  FULL   : %2d/%d graines ou le regime stationnaire ne rattrape jamais"
          % (vb_f, len(df)))
    print("  FROZEN : %2d/%d" % (vb_z, len(df)))
    print("  moyennes : best global %.2f | meilleur des 270 lectures tardives %.2f"
          % (df.cut_full.mean(), df.max_cut_late_full.mean()))
    print("  part des runs dont le best tombe avant le pas 300 : %d/%d"
          % (int((df.t_best_full < 300).sum()), len(df)))
    vb_ok = (vb_f >= 16)
    print("  >> V_B : %s" % ("CONFIRMEE" if vb_ok else "NON CONFIRMEE"))

    print("")
    print("-" * 78)
    print("V_C -- LA DIVERGENCE S'ACCUMULE (critere : hamming_early < hamming_late")
    print("       sur >= 16/20)")
    print("-" * 78)
    vc = int((df.hamming_early < df.hamming_late).sum())
    print("  %2d/%d graines   (Hamming moyen entre conditions : %.2f -> %.2f noeuds sur %d)"
          % (vc, len(df), df.hamming_early.mean(), df.hamming_late.mean(), N_NODES))
    print("  pas ou sign(v) est exactement identique : early %.1f pct -> late %.1f pct"
          % (100 * df.frac_s_equal_early.mean(), 100 * df.frac_s_equal_late.mean()))
    vc_ok = (vc >= 16)
    print("  >> V_C : %s" % ("CONFIRMEE" if vc_ok else "NON CONFIRMEE"))

    print("")
    print("-" * 78)
    print("V_D -- LE TRANSITOIRE EST-IL QUASI UNIFORME ? (critere : ecart relatif")
    print("       a l'ecart-type d'un tirage uniforme < 35 pct sur >= 14/20)")
    print("-" * 78)
    rel = (df.sd_early_full - df.sd_uniform).abs() / df.sd_uniform
    vd = int((rel < 0.35).sum())
    print("  ecart-type mesure d'un tirage uniforme : %.2f" % df.sd_uniform.mean())
    print("  ecart-type du transitoire M4R (FULL)   : %.2f" % df.sd_early_full.mean())
    print("  ecart-type du regime stationnaire      : %.2f" % df.sd_late_full.mean())
    print("  ecart relatif median au tirage uniforme : %.1f pct" % (100 * rel.median()))
    print("  graines sous 35 pct : %d sur %d" % (vd, len(df)))
    vd_ok = (vd >= 14)
    print("  >> V_D : %s" % ("CONFIRMEE" if vd_ok else "NON CONFIRMEE"))

    print("")
    print("-" * 78)
    print("V_E -- OBSERVATION : les deux conditions battent-elles leur record au")
    print("       meme instant ?")
    print("-" * 78)
    same_t = df.t_best_full == df.t_best_froz
    print("  t_best identique : %d graines sur %d" % (int(same_t.sum()), len(df)))
    print("    dont identiques par la coupe : %d sur %d"
          % (int((same_t & df.identical).sum()), int(df.identical.sum())))
    print("    dont NON identiques          : %d sur %d"
          % (int((same_t & ~df.identical).sum()), int((~df.identical).sum())))

    # ============================== VOLET 3 ==================================
    print("")
    print("=" * 78)
    print("VOLET 3 -- LE BOUCLAGE : MEME ENSEMBLE D'ETATS, OU COINCIDENCE ?")
    print("=" * 78)

    print("")
    print("-" * 78)
    print("V3_A -- L'ETAT OPTIMAL DE L'UNE EST-IL VISITE PAR L'AUTRE ?")
    print("        (critere : oui sur >= 16/20, dans les deux sens)")
    print("-" * 78)
    a1 = int(df.full_best_seen_by_froz.sum())
    a2 = int(df.froz_best_seen_by_full.sum())
    print("  best_s(FULL)     visite par FROZEN_U : %2d/%d" % (a1, len(df)))
    print("  best_s(FROZEN_U) visite par FULL     : %2d/%d" % (a2, len(df)))
    print("  etats de signes distincts visites : FULL %.0f | FROZEN %.0f | communs %.0f"
          % (df.n_states_full.mean(), df.n_states_froz.mean(), df.n_states_common.mean()))
    v3a_ok = (a1 >= 16 and a2 >= 16)
    print("  >> V3_A : %s" % ("CONFIRMEE -- meme ensemble explore, l'identite est mecanique"
                              if v3a_ok else "NON CONFIRMEE"))

    print("")
    print("-" * 78)
    print("V3_B -- LES GRAINES NON IDENTIQUES : desaccord d'EXPLORATION ou de")
    print("        SELECTION ? (observation)")
    print("-" * 78)
    o = df[~df.identical]
    both_seen = int((o.full_best_seen_by_froz & o.froz_best_seen_by_full).sum())
    print("  sur les %d graines non identiques :" % len(o))
    print("    les deux etats optimaux sont visites des DEUX cotes : %d" % both_seen)
    print("    l'optimum de FULL echappe a FROZEN     : %d"
          % int((~o.full_best_seen_by_froz).sum()))
    print("    l'optimum de FROZEN echappe a FULL     : %d"
          % int((~o.froz_best_seen_by_full).sum()))
    print("  (un etat visite des deux cotes mais retenu d'un seul serait impossible :")
    print("   le critere de selection est le maximum, il est deterministe. Un ecart")
    print("   signale donc un desaccord d'EXPLORATION, pas de selection.)")

    print("")
    print("-" * 78)
    print("V3_C -- LE RECORD EST-IL BATTU DANS UN ETAT COMMUN ? (observation)")
    print("-" * 78)
    print("  pas de record ou sign(v) est EXACTEMENT identique : %d/%d graines"
          % (int(df.s_equal_at_best.sum()), len(df)))
    print("    parmi les graines identiques     : %d/%d"
          % (int(df[df.identical].s_equal_at_best.sum()), int(df.identical.sum())))
    print("    parmi les graines non identiques : %d/%d"
          % (int(df[~df.identical].s_equal_at_best.sum()), int((~df.identical).sum())))

    # ============================== VOLET 4 ==================================
    print("")
    print("=" * 78)
    print("VOLET 4 -- GATE DE REPLICATION SUR LES GRAINES 20-39 (jamais touchees)")
    print("=" * 78)
    rep = sweep(range(N_SEEDS, 2 * N_SEEDS))
    rep_csv = os.path.join(repo, "figures", "p15c_maxcut_identity_replication.csv")
    rep.to_csv(rep_csv, index=False)
    print("")
    print("[CSV enregistre] : %s" % rep_csv)

    fi_0, fo_0, ni_0, no_0 = record_in_common_state(df)
    fi_1, fo_1, ni_1, no_1 = record_in_common_state(rep)

    print("")
    print("-" * 78)
    print("R_V3C -- REPLICATION DU DISCRIMINANT (critere : part >= 0.70 chez les")
    print("         identiques ET ecart >= 0.30 avec les non identiques)")
    print("-" * 78)
    print("  record battu dans un etat de signes COMMUN aux deux conditions :")
    print("    graines  0-19 : identiques %.2f (n=%d) | non identiques %.2f (n=%d)"
          % (fi_0, ni_0, fo_0, no_0))
    print("    graines 20-39 : identiques %.2f (n=%d) | non identiques %.2f (n=%d)"
          % (fi_1, ni_1, fo_1, no_1))
    rep_ok = bool(fi_1 >= 0.70 and (fi_1 - fo_1) >= 0.30)
    print("  >> R_V3C : %s" % ("REPLIQUE -- l'explication est citable" if rep_ok
                               else "NE REPLIQUE PAS -- ne pas citer, question toujours ouverte"))

    print("")
    print("-" * 78)
    print("R_ETATS -- OBSERVATIONS SUR LES GRAINES NEUVES")
    print("-" * 78)
    print("  taux d'identite : %d/%d (rappel graines 0-19 : %d/%d)"
          % (int(rep.identical.sum()), len(rep), int(df.identical.sum()), len(df)))
    print("  etats de signes distincts sur 300 lectures : FULL %.0f | FROZEN %.0f"
          % (rep.n_states_full.mean(), rep.n_states_froz.mean()))
    print("  best_s meme vecteur chez les identiques : %d/%d"
          % (int((rep[rep.identical].hamming_best == 0).sum()), int(rep.identical.sum())))
    print("  collisions du null independant (M3) : %d/%d"
          % (int(rep.rnd_collision.sum()), len(rep)))
    print("  dispersion : early %.2f -> late %.2f (FULL) | %.2f -> %.2f (FROZEN)"
          % (rep.sd_early_full.mean(), rep.sd_late_full.mean(),
             rep.sd_early_froz.mean(), rep.sd_late_froz.mean()))

    # ------------------------------------------------------------ SYNTHESE ----
    print("")
    print("=" * 78)
    print("SYNTHESE -- CE QUE LA QUESTION OUVERTE DU 27/07 EST DEVENUE")
    print("=" * 78)
    print("  ETABLI SANS SIMULATION (L1) : l'enonce parlait de DEUX quantites")
    print("  identiques ; il n'y en a qu'une. cut = 0.25*sum|J| + 0.5*E, residu nul")
    print("  sur 20/20. Le mot 'ET' ne renforcait rien.")
    print("")
    print("  ETABLI PAR MESURE (M1, gate G0 passe) : ce n'est pas une coincidence de")
    print("  VALEUR entre deux maximums distincts -- c'est LE MEME VECTEUR de spins,")
    print("  9 fois sur 10. La question 'pourquoi la meme coupe' se dissout : c'est")
    print("  le meme etat.")
    print("")
    print("  MORTE (M3) : la seule piste laissee au froid le 27/07 -- 'la coupe est")
    print("  discrete, deux series de meme loi partagent souvent leur maximum' -- est")
    print("  rejetee, facteur 5 entre le null (0.10) et l'observe (0.50). Septieme")
    print("  explication morte a son propre critere.")
    print("")
    print("  LE FAIT DUR (V3_A, M4) : sur 300 lectures, M4R ne visite que ~%.0f etats"
          % df.n_states_full.mean())
    print("  distincts en FULL et ~%.0f en FROZEN_U, pour un n_eff de 5 tirages"
          % df.n_states_froz.mean())
    print("  independants. Le reseau est quasi immobile ; deux runs quasi immobiles")
    print("  issus du meme bruit tombent souvent sur le meme etat. Et le doute")
    print("  explore MOINS que le doute gele -- constat non cherche.")
    print("")
    if rep_ok:
        print("  LE MECANISME, REPLIQUE (R_V3C) : le record est battu tot, dans un etat")
        print("  que les deux conditions occupent en meme temps -- %.0f pct du temps chez"
              % (100 * fi_1))
        print("  les graines identiques contre %.0f pct chez les autres, sur graines"
              % (100 * fo_1))
        print("  neuves. Quand les deux conditions divergent, c'est un desaccord")
        print("  d'EXPLORATION (V3_B : 0 cas sur 10 ou les deux optima sont vus des deux")
        print("  cotes), jamais de selection.")
    else:
        print("  LE MECANISME N'EST PAS REPLIQUE (R_V3C). Le discriminant du volet 3 ne")
        print("  tient pas sur graines neuves : ne pas le citer. Ce qui reste acquis est")
        print("  ce qui precede -- l'enonce corrige (L1), le meme vecteur (M1), la piste")
        print("  du froid morte (M3), et l'immobilite du reseau (V3_A/M4).")
    print("")
    print("  A NE PAS SURVENDRE : tout ceci porte sur une tache HORS de la niche du")
    print("  projet, ou M4R perd deja contre le tirage aleatoire a budget egal. Cela")
    print("  explique une identite de mesure, pas une propriete de u -- exactement ce")
    print("  qui etait annonce le 27/07 avant d'ouvrir la question.")
    print("=" * 78)
    print("Rappel de portee : exploration (colonne B), tache HORS de la niche du")
    print("projet. Aucun chiffre du preprint n'est concerne. Le -1.20 FULL-FROZEN_U")
    print("reste a ne pas citer.")
    print("=" * 78)


if __name__ == "__main__":
    main()
