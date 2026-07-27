"""
EXPERIENCE P15b -- POURQUOI 4 GRAINES SUR 10 SONT STRICTEMENT IDENTIQUES
ENTRE FULL ET FROZEN_U SUR LE MAX-CUT.

Question ouverte laissee le 27/07/2026 en tete de `p15_maxcut_ising_poc.py` :
sur 4 graines sur 10, la coupe ET l'energie sont strictement identiques entre
Mem4ristor FULL (doute adaptatif) et FROZEN_U (u fige a 0.5). Sur ces runs, u ne
change litteralement rien au resultat rapporte.

CE QUE LA LECTURE DU CODE ETABLIT AVANT TOUTE MESURE (donc PAS une hypothese) :
  - `_initialize_params(cold_start=True)` pose v = w = 0 pour tous les noeuds :
    l'etat initial est identique et deterministe dans les deux conditions.
  - Le bruit vient de `self.rng = RandomState(seed)`, interne au modele. A seed
    egal, les deux runs tirent la MEME sequence de bruit.
  - Au premier pas, I_stimulus = 0 et coupling_input = J @ v = 0, donc dv = eta :
    le signe de v apres le pas 1 est le signe du bruit, identique dans les deux runs.
  - `sigma_baseline = 0.05`, donc en FULL u DEMARRE a 0.05 et non a 0.5. Les deux
    conditions ont des u_filter differents des le premier pas (tanh(k*(0.5-u))) :
    l'explication "u n'a pas eu le temps de diverger de 0.5" est exclue d'emblee.
  La seule voie de divergence est donc u_filter -> I_coup, et elle est ouverte
  des le pas 1. L'identite observee demande une autre explication.

--------------------------------------------------------------------------------
CRITERES POSES AVANT L'EXECUTION (aucun chiffre connu au moment de les ecrire)
--------------------------------------------------------------------------------
G0 -- GATE DE FIDELITE. Les 20 couples (cut, E) reproduits ici doivent egaler au
     bit pres ceux de `figures/p15_maxcut_benchmark.csv`. Si le gate echoue, ce
     diagnostic ne porte pas sur les runs de P15 et TOUT le reste est invalide.

Trois explications candidates, chacune avec son critere d'acceptation :

H_A "le best est fige au premier echantillon et jamais ameliore"
     ACCEPTEE si, sur les graines identiques, n_improve == 1 dans LES DEUX
     conditions (le premier echantillon est le seul a avoir jamais gagne).

H_B "le best est atteint avant que les signes ne divergent"
     ACCEPTEE si, sur les graines identiques, t_best(FULL) == t_best(FROZEN)
     et t_best < t_div_s (premier pas ou sign(v) differe entre conditions).

H_C "les deux relaxations convergent au meme point"
     ACCEPTEE si, sur les graines identiques, sign(v) ne diverge jamais
     (t_div_s absent) ou si l'etat de signe final est identique.

DISCRIMINANT OBLIGATOIRE. Une explication n'est retenue que si la metrique qui la
porte SEPARE les 4 graines identiques des 6 autres. Si les 6 graines non
identiques presentent la meme valeur de metrique, l'explication n'explique rien
et elle est rejetee, meme si son critere ci-dessus est satisfait.

CONTRE-EPREUVE D'ATTRIBUTION (independante de la question posee)
     M4R echantillonne sign(v) tous les 10 pas sur 3000 pas, soit 300 tirages.
     On compare a "le meilleur de 300 tirages de s uniformement aleatoires", a
     budget d'echantillons EGAL. Si best-of-300-aleatoire >= cut_m4r sur >= 8
     graines sur 10, alors la coupe rapportee par M4R sur cette tache s'explique
     par l'echantillonnage et non par la dynamique du reseau.

Statut : exploration (colonne B). Ce script ne modifie aucun CSV canonique, aucun
chiffre du preprint, aucun claim du Guardian. Il ne touche pas `p15_maxcut_ising_poc.py`.

--------------------------------------------------------------------------------
VOLET 2 -- REPLICATION ET MECANISME (criteres poses le 27/07 au soir, APRES avoir
lu les sorties du volet 1 mais AVANT d'executer quoi que ce soit sur les graines 10-19,
qui n'ont jamais ete touchees par ce projet)
--------------------------------------------------------------------------------
Ce que le volet 1 a etabli, gate G0 passe :
  - H_A, H_B, H_C sont REJETEES. Aucune des trois explications candidates ne tient.
  - Sur les 4 graines identiques, t_best(FULL) == t_best(FROZEN) a chaque fois, et
    ce t_best est precoce (0, 40, 0, 20 pas sur 3000). Mais la graine 0, non
    identique, a elle aussi t_best egal (20/20) : la precocite ne SUFFIT pas.
  - Contre-epreuve : le meilleur de 300 tirages aleatoires bat M4R sur 10/10
    graines (91.50 contre 80.90). Non replique a ce stade.

L'explication candidate qui reste, et qu'on teste ici : la relaxation M4R DEGRADE
la coupe au fil du temps (le reseau s'ordonne, sign(v) se correle). Le meilleur
echantillon est donc trouve tot, quand les deux conditions sont encore proches --
d'ou des collisions frequentes. L'identite serait un symptome de "rien n'est
trouve apres les premiers pas", pas une propriete de u.

R1 REPLICATION DE LA CONTRE-EPREUVE. Sur les graines 10-19, best-of-300 aleatoire
   >= cut_full sur >= 8 graines sur 10. Si < 8, le resultat du volet 1 ne replique
   pas et ne doit pas etre cite.
R2 DEGRADATION. Sur les graines 10-19, la coupe moyenne echantillonnee sur la
   seconde moitie des pas (t >= 1500) est INFERIEURE a celle de la premiere moitie
   (t < 1500), sur >= 8 graines sur 10, en condition FULL comme en FROZEN.
R3 PRECOCITE. Rapporter la mediane de t_best sur les 20 graines (0-19). Observation
   chiffree, pas un test : aucune conclusion ne repose dessus.
R4 FREQUENCE D'IDENTITE. Rapporter le nombre de graines identiques sur 10-19.
   Observation. Si elle est du meme ordre que 4/10, le phenomene est generique ;
   sinon, il etait particulier aux 10 premieres graines.

--------------------------------------------------------------------------------
VOLET 3 -- LE MECANISME (critere pose apres les sorties du volet 2, avant execution)
--------------------------------------------------------------------------------
Le volet 2 a REJETE R2 : la coupe ne se degrade pas, elle est STATIONNAIRE autour
de 73 sur les 3000 pas, dans les deux conditions. Quatrieme explication morte.

Mais 73 est un nombre reconnaissable. Pour un vecteur de spins s tire uniformement,
E[W_cut] = 0.25 * sum |J_ij| exactement, puisque E[s_i s_j] = 0 pour i != j.
Ce chiffre se calcule sur le graphe seul, sans aucune simulation.

R5 INDISCERNABILITE. Sur les graines 10-19, l'ecart relatif entre la coupe MOYENNE
   echantillonnee par M4R et l'esperance analytique 0.25*sum|J| est inferieur a
   5 pour cent, sur >= 8 graines sur 10, en condition FULL comme en FROZEN.
   Si le critere est atteint : les configurations visitees par la relaxation M4R
   sont, du point de vue de la coupe, indiscernables d'un tirage uniforme -- la
   dynamique n'apporte aucun signal, et le "best" rapporte n'est que le maximum
   d'un echantillon bruite. Cela expliquerait a la fois la defaite contre
   best-of-300 (300 echantillons CORRELES ont un maximum plus faible que 300
   tirages independants de meme esperance) et la frequence des identites
   FULL/FROZEN (la coupe est une variable DISCRETE a faible dispersion : deux
   series tirees de la meme loi ont souvent le meme maximum entier).
   Si le critere n'est PAS atteint, cette explication tombe comme les quatre autres.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mem4ristor.graph_utils import make_ba
from mem4ristor.dynamics import Mem4ristorV3

N_SEEDS = 10
N_NODES = 100
N_STEPS = 3000
SAMPLE_EVERY = 10


def compute_cut_and_energy(s, J):
    """Identique a p15_maxcut_ising_poc.compute_cut_and_energy (copie volontaire :
    on ne veut pas dependre d'un import du script audite)."""
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


def run_pair_instrumented(J, seed, steps=N_STEPS):
    """Avance FULL et FROZEN_U pas a pas EN PARALLELE (memes seeds, donc meme
    sequence de bruit) pour pouvoir dater la divergence exactement."""
    m_full = make_model(seed)
    m_froz = make_model(seed)

    rec = {
        'cut_full': [], 'cut_froz': [], 'E_full': [], 'E_froz': [],
        'u_mean_full': [], 'sample_step': [],
        's_equal': [], 'v_maxdiff': [],
    }

    best = {'full': {'cut': 0, 'E': 0, 't': None, 'n_improve': 0, 's': None},
            'froz': {'cut': 0, 'E': 0, 't': None, 'n_improve': 0, 's': None}}

    t_div_v = None   # premier pas ou v differe numeriquement
    t_div_s = None   # premier pas echantillonne ou sign(v) differe

    for step in range(steps):
        m_froz.u[:] = 0.5

        m_full.step(I_stimulus=0.0, coupling_input=np.dot(J, m_full.v))
        m_froz.step(I_stimulus=0.0, coupling_input=np.dot(J, m_froz.v))

        if t_div_v is None and not np.array_equal(m_full.v, m_froz.v):
            t_div_v = step

        if step % SAMPLE_EVERY == 0:
            s_full = np.sign(m_full.v); s_full[s_full == 0] = 1
            s_froz = np.sign(m_froz.v); s_froz[s_froz == 0] = 1

            E_f, cut_f = compute_cut_and_energy(s_full, J)
            E_z, cut_z = compute_cut_and_energy(s_froz, J)

            if cut_f > best['full']['cut']:
                best['full'].update(cut=cut_f, E=E_f, t=step, s=s_full.copy())
                best['full']['n_improve'] += 1
            if cut_z > best['froz']['cut']:
                best['froz'].update(cut=cut_z, E=E_z, t=step, s=s_froz.copy())
                best['froz']['n_improve'] += 1

            equal = bool(np.array_equal(s_full, s_froz))
            if t_div_s is None and not equal:
                t_div_s = step

            rec['sample_step'].append(step)
            rec['cut_full'].append(cut_f); rec['cut_froz'].append(cut_z)
            rec['E_full'].append(E_f); rec['E_froz'].append(E_z)
            rec['u_mean_full'].append(float(m_full.u.mean()))
            rec['s_equal'].append(equal)
            rec['v_maxdiff'].append(float(np.max(np.abs(m_full.v - m_froz.v))))

    s_final_equal = bool(rec['s_equal'][-1])
    return best, t_div_v, t_div_s, s_final_equal, rec


def halves(rec, key, steps=N_STEPS):
    """Coupe moyenne sur la premiere et la seconde moitie des pas echantillonnes."""
    st = np.asarray(rec['sample_step'])
    vals = np.asarray(rec[key], dtype=float)
    first = vals[st < steps // 2].mean()
    second = vals[st >= steps // 2].mean()
    return first, second


def best_of_random(J, n_draws, seed, N=N_NODES):
    """Contre-epreuve : meilleur de n_draws configurations de spins uniformes.
    RNG dedie (seed decale) pour ne perturber ni le modele ni la construction de J."""
    rng = np.random.RandomState(10_000 + seed)
    best_cut = 0.0
    for _ in range(n_draws):
        s = rng.choice([-1.0, 1.0], size=N)
        _, cut = compute_cut_and_energy(s, J)
        if cut > best_cut:
            best_cut = cut
    return best_cut


def sweep(seeds):
    """Rejoue le couple FULL/FROZEN_U instrumente sur une liste de graines."""
    rows = []
    n_draws = N_STEPS // SAMPLE_EVERY

    for seed in seeds:
        J = build_J(seed)
        best, t_div_v, t_div_s, s_final_equal, rec = run_pair_instrumented(J, seed)
        rnd = best_of_random(J, n_draws, seed)

        identical = (best['full']['cut'] == best['froz']['cut'] and
                     best['full']['E'] == best['froz']['E'])

        cut_f1, cut_f2 = halves(rec, 'cut_full')
        cut_z1, cut_z2 = halves(rec, 'cut_froz')

        rows.append({
            'seed': seed,
            'cut_full_half1': cut_f1, 'cut_full_half2': cut_f2,
            'cut_froz_half1': cut_z1, 'cut_froz_half2': cut_z2,
            'cut_full': best['full']['cut'], 'E_full': best['full']['E'],
            'cut_froz': best['froz']['cut'], 'E_froz': best['froz']['E'],
            'identical': identical,
            't_best_full': best['full']['t'], 't_best_froz': best['froz']['t'],
            'n_improve_full': best['full']['n_improve'],
            'n_improve_froz': best['froz']['n_improve'],
            't_div_v': t_div_v, 't_div_s': t_div_s,
            's_final_equal': s_final_equal,
            'u_mean_full_end': rec['u_mean_full'][-1],
            'v_maxdiff_end': rec['v_maxdiff'][-1],
            'cut_best_of_random': rnd,
            # E[W_cut] pour s uniforme : 0.25 * (sum|J_ij| - trace J), puisque
            # E[s_i s_j] = 0 hors diagonale et s_i^2 = 1 sur la diagonale.
            'cut_expected_random': 0.25 * float(np.sum(np.abs(J)) - np.trace(J)),
        })

        print("seed %d | identiques=%-5s | cut FULL=%6.1f FROZ=%6.1f | "
              "t_best F/Z = %s/%s | n_improve F/Z = %d/%d | t_div_v=%s t_div_s=%s | rnd300=%6.1f"
              % (seed, identical, best['full']['cut'], best['froz']['cut'],
                 best['full']['t'], best['froz']['t'],
                 best['full']['n_improve'], best['froz']['n_improve'],
                 t_div_v, t_div_s, rnd))

    return pd.DataFrame(rows)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, ".."))
    ref_path = os.path.join(repo, "figures", "p15_maxcut_benchmark.csv")
    ref = pd.read_csv(ref_path)
    n_draws = N_STEPS // SAMPLE_EVERY

    print("=" * 78)
    print("P15b -- DIAGNOSTIC DE L'IDENTITE FULL / FROZEN_U SUR LE MAX-CUT")
    print("=" * 78)
    print("Criteres poses avant execution : voir l'en-tete de ce fichier.")
    print("")
    print("VOLET 1 -- graines 0-9 (celles de P15)")
    df = sweep(range(N_SEEDS))

    out_csv = os.path.join(repo, "figures", "p15b_maxcut_identity_diagnosis.csv")
    df.to_csv(out_csv, index=False)
    print("")
    print("[CSV enregistre] : %s" % out_csv)

    # ---------------------------------------------------------------- G0 ------
    print("")
    print("-" * 78)
    print("G0 -- GATE DE FIDELITE (reproduction des runs de P15)")
    print("-" * 78)
    merged = df.merge(ref[['seed', 'cut_m4r', 'E_m4r', 'cut_frozen', 'E_frozen']], on='seed')
    ok_full = np.array_equal(merged['cut_full'].values, merged['cut_m4r'].values) and \
              np.array_equal(merged['E_full'].values, merged['E_m4r'].values)
    ok_froz = np.array_equal(merged['cut_froz'].values, merged['cut_frozen'].values) and \
              np.array_equal(merged['E_froz'].values, merged['E_frozen'].values)
    gate_ok = bool(ok_full and ok_froz)
    print("  FULL   reproduit au bit pres : %s" % ok_full)
    print("  FROZEN reproduit au bit pres : %s" % ok_froz)
    if not gate_ok:
        print("  >> GATE ECHOUE : ce diagnostic ne porte pas sur les runs de P15.")
        print("  >> Tout ce qui suit est INVALIDE et ne doit pas etre cite.")
        return
    print("  >> GATE PASSE.")

    ident = df[df.identical]
    other = df[~df.identical]
    print("")
    print("  Graines strictement identiques : %d sur %d (seeds %s)"
          % (len(ident), len(df), list(ident.seed.values)))

    # ------------------------------------------------------- H_A / H_B / H_C --
    print("")
    print("-" * 78)
    print("VERDICTS SUR LES TROIS EXPLICATIONS (criteres poses avant execution)")
    print("-" * 78)

    # H_A
    hA = bool(((ident.n_improve_full == 1) & (ident.n_improve_froz == 1)).all()) and len(ident) > 0
    hA_sep = bool(((other.n_improve_full > 1) | (other.n_improve_froz > 1)).all()) if len(other) else False
    print("H_A best fige au premier echantillon, jamais ameliore")
    print("    critere       : %s (n_improve == 1 dans les deux conditions)" % hA)
    print("    discriminant  : %s (les autres graines ameliorent au moins une fois)" % hA_sep)
    print("    n_improve identiques  F/Z : %s / %s"
          % (sorted(ident.n_improve_full.unique()), sorted(ident.n_improve_froz.unique())))
    print("    n_improve autres      F/Z : %s / %s"
          % (sorted(other.n_improve_full.unique()) if len(other) else [],
             sorted(other.n_improve_froz.unique()) if len(other) else []))
    print("    VERDICT H_A : %s" % ("RETENUE" if (hA and hA_sep) else "REJETEE"))

    # H_B
    hB = bool(((ident.t_best_full == ident.t_best_froz) &
               (ident.t_best_full < ident.t_div_s.fillna(np.inf))).all()) and len(ident) > 0
    hB_sep = bool((other.t_best_full >= other.t_div_s.fillna(np.inf)).any() or
                  (other.t_best_froz >= other.t_div_s.fillna(np.inf)).any()) if len(other) else False
    print("")
    print("H_B best atteint avant divergence des signes")
    print("    critere       : %s (t_best egaux et < t_div_s)" % hB)
    print("    discriminant  : %s (au moins une autre graine a t_best >= t_div_s)" % hB_sep)
    print("    VERDICT H_B : %s" % ("RETENUE" if (hB and hB_sep) else "REJETEE"))

    # H_C
    hC = bool((ident.t_div_s.isna() | ident.s_final_equal).all()) and len(ident) > 0
    print("")
    print("H_C les deux relaxations convergent au meme point")
    print("    critere       : %s (signes jamais divergents, ou etat final identique)" % hC)
    print("    VERDICT H_C : %s" % ("RETENUE" if hC else "REJETEE"))

    # ------------------------------------------------ contre-epreuve ---------
    print("")
    print("-" * 78)
    print("CONTRE-EPREUVE D'ATTRIBUTION : M4R contre le meilleur de %d tirages aleatoires"
          % n_draws)
    print("-" * 78)
    win_rnd_full = int((df.cut_best_of_random >= df.cut_full).sum())
    win_rnd_froz = int((df.cut_best_of_random >= df.cut_froz).sum())
    print("  best-of-%d aleatoire >= M4R FULL   sur %d graines sur %d"
          % (n_draws, win_rnd_full, len(df)))
    print("  best-of-%d aleatoire >= M4R FROZEN sur %d graines sur %d"
          % (n_draws, win_rnd_froz, len(df)))
    print("  moyennes : aleatoire %.2f | FULL %.2f | FROZEN %.2f"
          % (df.cut_best_of_random.mean(), df.cut_full.mean(), df.cut_froz.mean()))
    if win_rnd_full >= 8:
        print("  >> CRITERE ATTEINT (>= 8/10) : la coupe rapportee par M4R sur cette tache")
        print("     s'explique par l'echantillonnage a budget egal, pas par la dynamique.")
    else:
        print("  >> Critere NON atteint (< 8/10) : l'echantillonnage seul ne rend pas compte")
        print("     de la coupe de M4R sur cette tache.")

    # ============================== VOLET 2 ==================================
    print("")
    print("=" * 78)
    print("VOLET 2 -- REPLICATION SUR LES GRAINES 10-19 (jamais touchees)")
    print("=" * 78)
    rep = sweep(range(N_SEEDS, 2 * N_SEEDS))
    out_csv2 = os.path.join(repo, "figures", "p15b_maxcut_identity_replication.csv")
    rep.to_csv(out_csv2, index=False)
    print("")
    print("[CSV enregistre] : %s" % out_csv2)

    print("")
    print("-" * 78)
    print("R1 -- REPLICATION DE LA CONTRE-EPREUVE (critere : >= 8 graines sur 10)")
    print("-" * 78)
    r_full = int((rep.cut_best_of_random >= rep.cut_full).sum())
    r_froz = int((rep.cut_best_of_random >= rep.cut_froz).sum())
    print("  best-of-%d aleatoire >= M4R FULL   sur %d graines sur %d" % (n_draws, r_full, len(rep)))
    print("  best-of-%d aleatoire >= M4R FROZEN sur %d graines sur %d" % (n_draws, r_froz, len(rep)))
    print("  moyennes : aleatoire %.2f | FULL %.2f | FROZEN %.2f"
          % (rep.cut_best_of_random.mean(), rep.cut_full.mean(), rep.cut_froz.mean()))
    print("  >> R1 : %s" % ("REPLIQUE" if r_full >= 8 else "NE REPLIQUE PAS -- ne pas citer"))

    print("")
    print("-" * 78)
    print("R2 -- DEGRADATION DE LA COUPE AU FIL DE LA RELAXATION (critere : >= 8/10")
    print("      dans LES DEUX conditions ; moitie 2 (t>=1500) < moitie 1 (t<1500))")
    print("-" * 78)
    deg_f = int((rep.cut_full_half2 < rep.cut_full_half1).sum())
    deg_z = int((rep.cut_froz_half2 < rep.cut_froz_half1).sum())
    print("  FULL   : %d graines sur %d degradent (moyennes %.2f -> %.2f)"
          % (deg_f, len(rep), rep.cut_full_half1.mean(), rep.cut_full_half2.mean()))
    print("  FROZEN : %d graines sur %d degradent (moyennes %.2f -> %.2f)"
          % (deg_z, len(rep), rep.cut_froz_half1.mean(), rep.cut_froz_half2.mean()))
    r2_ok = (deg_f >= 8 and deg_z >= 8)
    print("  >> R2 : %s" % ("CONFIRMEE" if r2_ok else "NON CONFIRMEE"))

    print("")
    print("-" * 78)
    print("R3 / R4 -- OBSERVATIONS CHIFFREES (aucune conclusion n'en depend)")
    print("-" * 78)
    allseeds = pd.concat([df, rep], ignore_index=True)
    print("  R3 mediane de t_best sur les 20 graines : FULL %.0f | FROZEN %.0f  (sur %d pas)"
          % (allseeds.t_best_full.median(), allseeds.t_best_froz.median(), N_STEPS))
    print("     part des runs dont le best tombe dans les 300 premiers pas (10 pct) :"
          " FULL %d/%d | FROZEN %d/%d"
          % (int((allseeds.t_best_full < 300).sum()), len(allseeds),
             int((allseeds.t_best_froz < 300).sum()), len(allseeds)))
    print("  R4 graines strictement identiques : %d sur %d en replication (seeds %s)"
          % (int(rep.identical.sum()), len(rep), list(rep[rep.identical].seed.values)))
    print("     rappel volet 1 : %d sur %d" % (int(df.identical.sum()), len(df)))

    # ============================== VOLET 3 ==================================
    print("")
    print("-" * 78)
    print("R5 -- INDISCERNABILITE (critere : ecart relatif < 5 pct sur >= 8/10,")
    print("      dans LES DEUX conditions ; graines 10-19)")
    print("-" * 78)
    mean_full = (rep.cut_full_half1 + rep.cut_full_half2) / 2.0
    mean_froz = (rep.cut_froz_half1 + rep.cut_froz_half2) / 2.0
    rel_f = (mean_full - rep.cut_expected_random).abs() / rep.cut_expected_random
    rel_z = (mean_froz - rep.cut_expected_random).abs() / rep.cut_expected_random
    ok_f = int((rel_f < 0.05).sum())
    ok_z = int((rel_z < 0.05).sum())
    print("  esperance analytique 0.25*(sum|J| - tr J) : moyenne %.2f" % rep.cut_expected_random.mean())
    print("  coupe moyenne visitee par M4R : FULL %.2f | FROZEN %.2f"
          % (mean_full.mean(), mean_froz.mean()))
    print("  ecart relatif median : FULL %.2f pct | FROZEN %.2f pct"
          % (100 * rel_f.median(), 100 * rel_z.median()))
    print("  graines sous 5 pct : FULL %d sur %d | FROZEN %d sur %d"
          % (ok_f, len(rep), ok_z, len(rep)))
    r5_ok = (ok_f >= 8 and ok_z >= 8)
    print("  >> R5 : %s" % ("CONFIRMEE" if r5_ok else "REJETEE"))
    if r5_ok:
        print("     Les configurations visitees par la relaxation M4R sont, au sens de la")
        print("     coupe, indiscernables d'un tirage uniforme de spins. La dynamique")
        print("     n'apporte aucun signal sur cette tache ; le 'best' rapporte est le")
        print("     maximum d'un echantillon bruite, et il est plus faible que celui de")
        print("     300 tirages independants parce que les etats visites sont correles.")
    else:
        print("     L'explication tombe : les etats visites ne sont pas assimilables a")
        print("     un tirage uniforme. Le mecanisme reste inexplique.")

    print("")
    print("=" * 78)


if __name__ == "__main__":
    main()
