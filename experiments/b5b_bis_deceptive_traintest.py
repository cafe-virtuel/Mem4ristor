#!/usr/bin/env python3
"""
B5b-BIS -- LA DETTE DE METHODE DE B5b, PAYEE : selection train/test au lieu de
l'oracle par run.

Pourquoi ce script existe (dette ouverte le 26/07/2026, docs/FUTURE_WORK.md section E4).
`b5b_deceptive_exploration.py` (08/07) regle l'ESN adverse "a son maximum par ORACLE
PAR RUN" : pour chaque probleme, il choisit (rho, leak) parmi 6 combinaisons EN
CONNAISSANT la bonne reponse. Le 26/07, cette procedure a ete demasquee ailleurs dans
le projet : appliquee a un signal SANS AUCUNE INFORMATION (bruit pur), elle rend
0.935 d'accuracy au lieu de 0.500 -- avec plusieurs combinaisons et une decision
binaire, il s'en trouve presque toujours une qui tombe juste. Elle ne regle pas
l'adversaire, elle fabrique du resultat.

Ici le biais joue EN FAVEUR DE L'ESN, donc CONTRE Mem4ristor (qui, lui, n'ajuste
rien) : la conclusion de B5b en sort conservative, pas invalide. C'est pourquoi elle
n'a jamais ete retiree -- mais elle bloquait toute citation au chiffre.

TROIS ENDROITS OU LE PIEGE EST PRESENT DANS B5b (releves en lisant le script) :
  1. `esn_best_by_oracle` : (rho, leak) choisis run par run en connaissant dstar.
  2. `esn_best_stop = np.maximum(acc[DROP], acc[CONV])` : la meilleure des deux
     regles d'arret prise GRAINE PAR GRAINE, donc encore un oracle, plus discret.
  3. `best_fixed` : le meilleur budget fixe choisi sur les MEMES donnees que celles
     qui servent a le mesurer (biais de selection, pas oracle par run).
Les trois sont corriges ici de la meme facon : tout choix se fait sur les graines
d'ENTRAINEMENT, toute mesure sur des graines de TEST DISJOINTES.

Harness identique a B5b : meme flux trompeur (`deceptive_task_poc`), meme MAX_BUDGET,
memes niveaux de leurre, meme grille ESN, meme lecture differentielle. Seule la
PROCEDURE DE SELECTION change -- c'est tout l'objet du script.

--------------------------------------------------------------------------------
CRITERES POSES AVANT EXECUTION
--------------------------------------------------------------------------------
G_CTRL -- GATE DE PROCEDURE (le controle qui a fait tomber l'oracle par run).
   Sur un flux SANS aucune information, la procedure corrigee doit rendre 0.5 :
   |acc(train/test) - 0.5| <= 0.10, et l'oracle par run doit rendre >= 0.80 sur ce
   meme flux (temoin : il faut voir le piege exister pour croire a sa correction).
   Si le gate echoue, AUCUN chiffre de ce script n'est publiable.

   >> PREMIER TEMOIN INVALIDE, CONSERVE ICI PLUTOT QUE RETIRE (27/07, soir).
   Premiere version du controle : stimulus = bruit gaussien fixe par noeud, constant
   dans le temps, tire independamment de dstar. Resultat : procedure corrigee 0.500
   (exactement conforme) mais oracle par run 0.550, LOIN du >= 0.80 attendu. Le gate
   a donc bloque -- et il avait raison de bloquer, mais pas pour la raison prevue :
   ce n'est pas la correction qui etait en cause, c'est le TEMOIN. Sur un flux aussi
   pauvre, l'ESN sature et les 6 combinaisons repondent la MEME chose ; l'oracle par
   run n'a alors rien a exploiter, et le controle ne teste rien du tout.
   Lecon a garder : un controle qui rend le bon chiffre pour la mauvaise raison est
   un controle en panne, pas un feu vert.

G_CTRL v2 -- CONTROLE PAR PERMUTATION D'ETIQUETTES (celui qui compte).
   Meme idee, instrument correct : on garde les VRAIES trajectoires de la tache
   trompeuse -- donc la vraie diversite de reponses entre combinaisons -- et on
   remelange les etiquettes dstar entre problemes. Le lien decision/verite est brise,
   donc toute precision au-dessus de 0.5 est fabriquee par la procedure.
   (a) oracle par run sous permutation >= 0.65 : le piege a un pouvoir REEL dans ce
       harnais. S'il est < 0.65, ce n'est pas un echec du script : c'est le resultat
       que la dette B5b etait PETITE ici, et il faut le rapporter comme tel.
   (b) selection train/test sous permutation dans [0.40, 0.60] : la correction tient.
   MESURE OBLIGATOIRE QUI DETERMINE (a) : la part des problemes ou les 6 combinaisons
   ne sont PAS unanimes. C'est elle qui donne a l'oracle par run son pouvoir de triche ;
   sans diversite, pas de triche possible.

V1 -- M4R (doute natif, regle PRE-SPECIFIEE, aucun reglage) contre le meilleur arret
   NAIF de l'ESN (regle ET hyperparametres choisis sur TRAIN, mesures sur TEST).
V2 -- LE TEST QUI COMPTE : M4R (doute natif) contre le meilleur BUDGET FIXE de l'ESN
   (budget ET hyperparametres choisis sur TRAIN, mesures sur TEST). C'est la
   comparaison qui, en 2026-07-08, donnait la PARITE (IC couvrant zero) et qui a fait
   qualifier la niche d'"etroite".
V3 -- CONTROLE DE SYMETRIE, pour ne pas creer le biais inverse : on accorde AUSSI a
   M4R le droit de choisir sa regle d'arret sur TRAIN, et on refait V1.

SENS DU BIAIS, ECRIT AVANT DE VOIR LE RESULTAT. Le biais corrige avantageait l'ESN.
L'issue attendue est donc que M4R fasse *au moins* aussi bien qu'en 2026-07-08. Trois
lectures possibles de V2, annoncees d'avance pour qu'aucune ne soit racontee apres
coup :
  IC > 0        -> la niche est PLUS FORTE que ce que B5b annoncait ; l'ancienne
                   parite etait un artefact de la procedure.
  IC couvre 0   -> la conclusion d'origine TIENT (elle etait conservative, elle est
                   maintenant propre) : le doute bat les arrets naifs, pas un horizon
                   fixe optimal.
  IC < 0        -> la niche est PLUS FAIBLE qu'annoncee, malgre un biais qui jouait
                   dans l'autre sens. A publier tel quel, sans amenagement.
Et la consigne du 26/07 midi, qui s'applique en priorite ici : si le resultat me fait
plaisir, aller voir d'ou il vient AVANT de le raconter.

Statut : exploration (colonne B). Ne touche ni `b5b_deceptive_exploration.py`, ni son
CSV, ni aucun chiffre du preprint, ni aucun claim du Guardian.
Cree : 2026-07-27 au soir (Claude Opus 5).
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "scratch"))
import deceptive_task_poc as dp  # noqa: E402

dp.MAX_BUDGET = 2000          # identique a B5b
CSV = ROOT / "figures" / "b5b_bis_deceptive_traintest.csv"
PNG = ROOT / "figures" / "b5b_bis_deceptive_traintest.png"

N = dp.N
SEEDS_TRAIN = list(range(20))        # servent A CHOISIR
SEEDS_TEST = list(range(20, 40))     # servent A MESURER -- disjointes
T_PULSE_LEVELS = [350, 700]          # identique a B5b
ESN_RHO = [0.9, 1.0, 1.1]            # identique a B5b
ESN_LEAK = [0.3, 1.0]                # identique a B5b
ESN_DENSITY = 0.1
ESN_GRID = [(r, l) for r in ESN_RHO for l in ESN_LEAK]
BUDGET_GRID = list(range(100, dp.MAX_BUDGET, 100))
N_BOOT = 10000
RNG_BOOT = np.random.RandomState(20260727)


# ---- ESN standard (identique a B5b) -----------------------------------------
def make_esn(seed, rho):
    rng = np.random.RandomState(7000 + seed)
    W = rng.uniform(-1.0, 1.0, (N, N)) * (rng.uniform(0, 1, (N, N)) < ESN_DENSITY)
    eig = np.max(np.abs(np.linalg.eigvals(W)))
    if eig > 1e-9:
        W *= rho / eig
    return W


def run_esn_decision(W, stim_on, stim_off, t_pulse, leak, T=None):
    """ESN pilote par le MEME stimulus par noeud, + run de reference stim=0."""
    T = dp.MAX_BUDGET if T is None else T
    x = np.zeros(N)
    xr = np.zeros(N)
    d_var = np.empty(T)
    dec = np.empty(T, dtype=int)
    change = np.empty(T)
    x_prev = x.copy()
    for t in range(T):
        stim = stim_on if t < t_pulse else stim_off
        x = (1.0 - leak) * x + leak * np.tanh(W @ x + stim)
        xr = (1.0 - leak) * xr + leak * np.tanh(W @ xr)
        d = float(x.mean() - xr.mean())
        d_var[t] = d
        dec[t] = 1 if d >= 0 else -1
        change[t] = float(np.mean(np.abs(x - x_prev)))
        x_prev = x.copy()
    return d_var, dec, change


def boot_ci_paired(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    m = np.empty(N_BOOT)
    for k in range(N_BOOT):
        m[k] = d[RNG_BOOT.randint(0, n, n)].mean()
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


# ---- G_CTRL v1 : TEMOIN INVALIDE, conserve --------------------------------
def procedure_control_degenerate(n_problems=40, T=400):
    """Flux SANS information : le stimulus est un bruit fixe par noeud, tire
    independamment de dstar. Aucune procedure ne peut depasser 0.5 en esperance.
    On mesure l'oracle par run (la methode fautive, temoin) et la selection
    train/test (la methode corrigee)."""
    rng = np.random.RandomState(1234)
    problems = []
    for i in range(n_problems):
        dstar = 1 if rng.rand() < 0.5 else -1
        stim = 0.35 * rng.randn(N)          # aucun lien avec dstar
        problems.append((dstar, stim, 7000 + i))

    # dec[-1] par (probleme, combo) -- calcule une fois
    finals = np.empty((n_problems, len(ESN_GRID)), dtype=int)
    for i, (dstar, stim, sd) in enumerate(problems):
        for j, (rho, leak) in enumerate(ESN_GRID):
            W = make_esn(sd % 1000, rho)
            _, dec, _ = run_esn_decision(W, stim, stim, T, leak, T=T)
            finals[i, j] = dec[-1]
    truth = np.array([p[0] for p in problems])

    hits = (finals == truth[:, None]).astype(float)
    oracle_per_run = float(hits.max(axis=1).mean())

    half = n_problems // 2
    j_best = int(hits[:half].mean(axis=0).argmax())
    train_test = float(hits[half:, j_best].mean())

    print("-" * 78)
    print("G_CTRL v1 -- TEMOIN INVALIDE (conserve, pas retire)")
    print("-" * 78)
    print("  Flux nul degenere : stimulus constant, aucune diversite entre combinaisons.")
    print("  oracle par run (methode FAUTIVE, temoin)   : %.3f   (attendu >= 0.80)"
          % oracle_per_run)
    print("  selection train/test (methode CORRIGEE)    : %.3f   (attendu ~0.50)"
          % train_test)
    print("  >> Ce controle NE TESTE RIEN : sur un flux aussi pauvre l'ESN sature et les")
    print("     6 combinaisons repondent la meme chose, donc l'oracle n'a rien a exploiter.")
    print("     Le bon chiffre pour la mauvaise raison. Remplace par G_CTRL v2 ci-dessous.")
    return oracle_per_run, train_test


# ---- G_CTRL v2 : controle par permutation d'etiquettes ----------------------
def procedure_control_permutation(raw, n_perm=200, seed=20260727):
    """Vraies trajectoires (donc vraie diversite entre combinaisons), etiquettes
    remelangees entre problemes. Toute precision > 0.5 est fabriquee par la procedure.

    Mesure sur les TROIS quantites que la selection de B5b regarde, parce que son
    score est `(oracle, a_drop + a_conv)` : si l'oracle est unanime entre combos,
    c'est le TIE-BREAK sur les accuracies d'arret -- donc sur des quantites
    MESUREES -- qui tranche, et c'est la que le piege peut mordre."""
    truth = np.array([p["dstar"] for p in raw])
    n = len(raw)
    half = n // 2
    rng = np.random.RandomState(seed)

    channels = {
        "dec_final (critere primaire)": np.array([[p["esn_final"][c] for c in ESN_GRID] for p in raw]),
        "dec a l'arret DROP (tie-break)": np.array([[p["esn_drop_dec"][c] for c in ESN_GRID] for p in raw]),
        "dec a l'arret CONV (tie-break)": np.array([[p["esn_conv_dec"][c] for c in ESN_GRID] for p in raw]),
    }

    print("-" * 78)
    print("G_CTRL v2 -- CONTROLE PAR PERMUTATION D'ETIQUETTES (%d permutations)" % n_perm)
    print("-" * 78)
    out = {}
    for label, mat in channels.items():
        non_unanimous = float(np.mean([len(set(row.tolist())) > 1 for row in mat]))
        orc, tt = [], []
        for _ in range(n_perm):
            perm = truth[rng.permutation(n)]
            hits = (mat == perm[:, None]).astype(float)
            orc.append(hits.max(axis=1).mean())
            j = int(hits[:half].mean(axis=0).argmax())
            tt.append(hits[half:, j].mean())
        hits_true = (mat == truth[:, None]).astype(float)
        out[label] = dict(diversity=non_unanimous, orc_perm=float(np.mean(orc)),
                          tt_perm=float(np.mean(tt)),
                          orc_true=float(hits_true.max(axis=1).mean()))
        print("  %-32s divergence entre combos : %3.0f pct" % (label, 100 * non_unanimous))
        print("  %-32s oracle-par-run permute  : %.3f   train/test permute : %.3f"
              % ("", out[label]["orc_perm"], out[label]["tt_perm"]))

    orc_max = max(v["orc_perm"] for v in out.values())
    tt_all = [v["tt_perm"] for v in out.values()]
    div_max = max(v["diversity"] for v in out.values())

    print("")
    print("  Critere (a) pouvoir du piege : max oracle-par-run permute = %.3f  (>= 0.65 ?)" % orc_max)
    print("  Critere (b) correction       : train/test permutes = %s  (tous dans [0.40, 0.60] ?)"
          % ", ".join("%.3f" % t for t in tt_all))
    a_ok = orc_max >= 0.65
    b_ok = all(0.40 <= t <= 0.60 for t in tt_all)
    if not a_ok:
        print("  >> CRITERE (a) NON ATTEINT sur les TROIS canaux (divergence max entre")
        print("     combinaisons : %.0f pct). Ce n'est pas un echec du script : c'est le" % (100 * div_max))
        print("     resultat que l'oracle par run n'avait PAS de prise dans ce harnais")
        print("     precis -- sans divergence entre combinaisons, il n'y a rien a trier.")
        print("     La dette E4 etait donc REELLE DANS LE CODE mais SANS EFFET sur les")
        print("     chiffres de B5b. A rapporter tel quel, sans en faire une victoire.")
    print("  >> (b) correction : %s" % ("OK" if b_ok else "PROBLEME -- ne rien publier"))
    return dict(ok=b_ok, power=a_ok, orc_perm=orc_max, tt_perm=max(tt_all),
                orc_true=out["dec_final (critere primaire)"]["orc_true"],
                diversity=div_max, per_channel=out)




# ---- sweep ------------------------------------------------------------------
def sweep(seeds):
    """Retourne, par graine, les accuracies de M4R et de l'ESN pour CHAQUE combo
    et CHAQUE regle -- aucune selection n'est faite ici."""
    m4r = {k: defaultdict(list) for k in ("DOUBT", "CONV", "ORACLE")}
    esn = {k: defaultdict(lambda: defaultdict(list)) for k in ("DROP", "CONV", "ORACLE")}
    m4r_fixed = defaultdict(lambda: defaultdict(list))          # [B][seed]
    esn_fixed = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # [combo][B][seed]
    raw = []   # decisions brutes : alimentent le controle par permutation, sans recalcul

    for seed in seeds:
        for t_pulse in T_PULSE_LEVELS:
            rng = np.random.RandomState(3000 + seed)
            adj, stim_on, stim_off, dstar = dp.make_deceptive(rng)

            # --- Mem4ristor : aucun hyperparametre, regles natives -----------
            sig, dec, d_var = dp.simulate(adj, stim_on, stim_off, seed * 10 + 1, t_pulse)
            cd = dp.stop_doubt(sig)
            cc = dp.stop_conv(d_var)
            m4r["DOUBT"][seed].append(int(dp.dec_at(dec, cd) == dstar))
            m4r["CONV"][seed].append(int(dp.dec_at(dec, cc) == dstar))
            m4r["ORACLE"][seed].append(int(dec[-1] == dstar))
            for B in BUDGET_GRID:
                m4r_fixed[B][seed].append(int(dp.dec_at(dec, B) == dstar))

            # --- ESN : TOUS les combos, sans selection ----------------------
            esn_final = {}
            esn_drop_dec = {}
            esn_conv_dec = {}
            for combo in ESN_GRID:
                rho, leak = combo
                W = make_esn(seed, rho)
                e_dvar, e_dec, e_change = run_esn_decision(W, stim_on, stim_off, t_pulse, leak)
                c_conv = dp.stop_conv(e_dvar)
                c_drop = dp.stop_doubt(e_change)
                esn["DROP"][combo][seed].append(int(dp.dec_at(e_dec, c_drop) == dstar))
                esn["CONV"][combo][seed].append(int(dp.dec_at(e_dec, c_conv) == dstar))
                esn["ORACLE"][combo][seed].append(int(e_dec[-1] == dstar))
                esn_final[combo] = int(e_dec[-1])
                esn_drop_dec[combo] = int(dp.dec_at(e_dec, c_drop))
                esn_conv_dec[combo] = int(dp.dec_at(e_dec, c_conv))
                for B in BUDGET_GRID:
                    esn_fixed[combo][B][seed].append(int(dp.dec_at(e_dec, B) == dstar))

            raw.append(dict(seed=seed, t_pulse=t_pulse, dstar=int(dstar),
                            esn_final=esn_final, esn_drop_dec=esn_drop_dec,
                            esn_conv_dec=esn_conv_dec))

        print("  graine %2d traitee" % seed, end="\r", flush=True)

    def per_seed(d, seeds):
        return [float(np.mean(d[s])) for s in seeds]

    return dict(m4r=m4r, esn=esn, m4r_fixed=m4r_fixed, esn_fixed=esn_fixed,
                per_seed=per_seed, seeds=list(seeds), raw=raw)


def main() -> int:
    CSV.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("=" * 78)
    print("B5b-BIS -- LA DETTE DE B5b PAYEE : selection train/test, oracle par run retire")
    print("=" * 78)
    print("harness B5b exact | %d graines TRAIN + %d graines TEST (disjointes) | budget max %d"
          % (len(SEEDS_TRAIN), len(SEEDS_TEST), dp.MAX_BUDGET))
    print("Criteres et sens du biais poses avant execution : voir l'en-tete du fichier.")
    print("")

    orc_deg, tt_deg = procedure_control_degenerate()
    print("")

    print("Sweep TRAIN (selection) ...")
    tr = sweep(SEEDS_TRAIN)
    print("")
    ctrl = procedure_control_permutation(tr["raw"])
    print("")
    if not ctrl["ok"]:
        print(">> La correction ne rend pas 0.5 sous permutation : la procedure elle-meme")
        print("   est suspecte. On s'arrete, aucun chiffre a citer.")
        return 1
    print("Sweep TEST  (mesure, graines disjointes) ...")
    te = sweep(SEEDS_TEST)
    print("")

    def mean_train(d):
        return float(np.mean([np.mean(d[s]) for s in SEEDS_TRAIN]))

    def mean_test(d):
        return float(np.mean([np.mean(d[s]) for s in SEEDS_TEST]))

    # ---- selections, toutes sur TRAIN ---------------------------------------
    sel = {}
    for rule in ("DROP", "CONV", "ORACLE"):
        sel[rule] = max(ESN_GRID, key=lambda c: mean_train(tr["esn"][rule][c]))
    # meilleure regle naive de l'ESN (DROP ou CONV), choisie sur TRAIN
    esn_rule = max(("DROP", "CONV"),
                   key=lambda r: mean_train(tr["esn"][r][sel[r]]))
    esn_rule_combo = sel[esn_rule]
    # meilleur (combo, budget) de l'ESN, choisi sur TRAIN
    esn_fx_combo, esn_fx_B = max(
        ((c, B) for c in ESN_GRID for B in BUDGET_GRID),
        key=lambda cb: mean_train(tr["esn_fixed"][cb[0]][cb[1]]))
    # meilleur budget de M4R, choisi sur TRAIN
    m4r_fx_B = max(BUDGET_GRID, key=lambda B: mean_train(tr["m4r_fixed"][B]))
    # meilleure regle de M4R, choisie sur TRAIN (controle de symetrie V3)
    m4r_rule = max(("DOUBT", "CONV"), key=lambda r: mean_train(tr["m4r"][r]))

    print("-" * 78)
    print("CHOIX FAITS SUR LES GRAINES TRAIN, PUIS FIGES")
    print("-" * 78)
    print("  ESN meilleure regle naive      : %s  (rho=%.1f, leak=%.1f)"
          % (esn_rule, esn_rule_combo[0], esn_rule_combo[1]))
    print("  ESN meilleur budget fixe       : B=%d  (rho=%.1f, leak=%.1f)"
          % (esn_fx_B, esn_fx_combo[0], esn_fx_combo[1]))
    print("  ESN combo pour l'oracle        : rho=%.1f, leak=%.1f"
          % (sel["ORACLE"][0], sel["ORACLE"][1]))
    print("  M4R meilleur budget fixe       : B=%d" % m4r_fx_B)
    print("  M4R meilleure regle (symetrie) : %s" % m4r_rule)
    print("  M4R doute natif                : aucun reglage, regle pre-specifiee")

    # ---- resultats sur TEST -------------------------------------------------
    r = {
        "M4R_DOUBT": mean_test(te["m4r"]["DOUBT"]),
        "M4R_CONV": mean_test(te["m4r"]["CONV"]),
        "M4R_ORACLE": mean_test(te["m4r"]["ORACLE"]),
        "M4R_FIXED": mean_test(te["m4r_fixed"][m4r_fx_B]),
        "ESN_RULE": mean_test(te["esn"][esn_rule][esn_rule_combo]),
        "ESN_FIXED": mean_test(te["esn_fixed"][esn_fx_combo][esn_fx_B]),
        "ESN_ORACLE": mean_test(te["esn"]["ORACLE"][sel["ORACLE"]]),
    }
    print("")
    print("-" * 78)
    print("RESULTATS SUR LES GRAINES TEST (disjointes, %d graines)" % len(SEEDS_TEST))
    print("-" * 78)
    print("  M4R doute natif (adaptatif)       : %.2f" % r["M4R_DOUBT"])
    print("  M4R convergence                   : %.2f" % r["M4R_CONV"])
    print("  M4R budget fixe (B=%4d, sur TRAIN): %.2f" % (m4r_fx_B, r["M4R_FIXED"]))
    print("  ESN %-6s (meilleure regle naive) : %.2f" % (esn_rule, r["ESN_RULE"]))
    print("  ESN budget fixe (B=%4d, sur TRAIN): %.2f" % (esn_fx_B, r["ESN_FIXED"]))
    print("  [reference] M4R oracle %.2f | ESN oracle %.2f"
          % (r["M4R_ORACLE"], r["ESN_ORACLE"]))

    ps = te["per_seed"]
    m4r_doubt_ps = ps(te["m4r"]["DOUBT"], SEEDS_TEST)

    print("")
    print("-" * 78)
    print("VERDICTS (IC bootstrap apparie sur les graines TEST)")
    print("-" * 78)

    v1 = boot_ci_paired(m4r_doubt_ps, ps(te["esn"][esn_rule][esn_rule_combo], SEEDS_TEST))
    print("V1  doute natif vs meilleur arret NAIF ESN (%s) : %+.2f IC[%+.2f, %+.2f]"
          % (esn_rule, v1[0], v1[1], v1[2]))
    print("    -> %s" % ("le doute BAT l'arret naif de l'ESN" if v1[1] > 0
                         else ("l'ESN BAT le doute" if v1[2] < 0 else "NON CONCLUANT (IC couvre 0)")))
    if r["ESN_RULE"] <= 0.10:
        print("    /!\\ A NE PAS SURVENDRE : l'adversaire est a %.2f, c'est-a-dire au PLANCHER."
              % r["ESN_RULE"])
        print("        Un adversaire qui repond systematiquement faux n'est pas une baseline,")
        print("        c'est une baseline en panne : son arret se declenche pendant le leurre.")
        print("        Le seul test qui engage quelque chose est V2.")

    v2 = boot_ci_paired(m4r_doubt_ps, ps(te["esn_fixed"][esn_fx_combo][esn_fx_B], SEEDS_TEST))
    print("")
    print("V2  doute natif vs ESN MEILLEUR BUDGET FIXE (B=%d) : %+.2f IC[%+.2f, %+.2f]"
          % (esn_fx_B, v2[0], v2[1], v2[2]))
    if v2[1] > 0:
        print("    -> LA NICHE EST PLUS FORTE que ce que B5b annoncait : le doute bat meme")
        print("       le meilleur ESN NON-ADAPTATIF, budget choisi honnetement sur TRAIN.")
        print("       L'ancienne parite etait donc en partie un artefact de l'oracle par run.")
    elif v2[2] < 0:
        print("    -> LA NICHE EST PLUS FAIBLE qu'annoncee, et ce MALGRE un biais corrige")
        print("       qui jouait en faveur de l'ESN. Rapporte tel quel.")
    else:
        print("    -> NON CONCLUANT AU SENS STRICT : l'IC touche zero. Formulation exacte, pour")
        print("       ne pas refaire l'erreur du 26/07 ('egale voire depasse', plus fort que les")
        print("       chiffres) : l'ecart ponctuel va dans le sens de l'ESN (%+.2f ; ESN %.2f"
              % (v2[0], r["ESN_FIXED"]))
        print("       contre %.2f pour le doute). Ce qui est ETABLI : le doute n'est pas MEILLEUR"
              % r["M4R_DOUBT"])
        print("       qu'un horizon fixe bien choisi. Ce qui ne l'est PAS : qu'il soit pire.")
        print("       La conclusion de B5b du 08/07 tient donc, et elle est maintenant propre :")
        print("       la valeur du doute reste conditionnee a l'horizon inconnu ou au cout")
        print("       d'attente (B1c). Noter que B=%d depasse le leurre le plus long (%d) :"
              % (esn_fx_B, max(T_PULSE_LEVELS)))
        print("       un budget fixe superieur a l'horizon MAXIMAL suffit -- c'est precisement")
        print("       ce qu'un horizon inconnu ou non borne interdit de choisir.")

    v3 = boot_ci_paired(ps(te["m4r"][m4r_rule], SEEDS_TEST),
                        ps(te["esn"][esn_rule][esn_rule_combo], SEEDS_TEST))
    print("")
    print("V3  CONTROLE DE SYMETRIE -- M4R %s (choisi sur TRAIN) vs ESN %s : %+.2f IC[%+.2f, %+.2f]"
          % (m4r_rule, esn_rule, v3[0], v3[1], v3[2]))
    print("    (V1 comparait une regle M4R pre-specifiee a un ESN selectionne ; ici les deux")
    print("     camps sont selectionnes de la meme facon.)")

    # ---- CSV ---------------------------------------------------------------
    with CSV.open("w", encoding="utf-8") as f:
        f.write("quantite,valeur,detail\n")
        f.write("ctrlv1_oracle_per_run,%.4f,TEMOIN INVALIDE flux degenere\n" % orc_deg)
        f.write("ctrlv1_train_test,%.4f,TEMOIN INVALIDE flux degenere\n" % tt_deg)
        f.write("ctrlv2_combo_diversity,%.4f,part des problemes ou les 6 combos divergent\n"
                % ctrl["diversity"])
        f.write("ctrlv2_oracle_perm,%.4f,oracle par run sous permutation d'etiquettes\n"
                % ctrl["orc_perm"])
        f.write("ctrlv2_train_test_perm,%.4f,selection train/test sous permutation\n"
                % ctrl["tt_perm"])
        f.write("ctrlv2_oracle_true,%.4f,oracle par run sur les vraies etiquettes\n"
                % ctrl["orc_true"])
        f.write("ctrlv2_power_criterion_met,%s,oracle_perm >= 0.65\n" % ctrl["power"])
        f.write("esn_rule,%s,choisie sur TRAIN\n" % esn_rule)
        f.write("esn_rule_rho,%.2f,choisi sur TRAIN\n" % esn_rule_combo[0])
        f.write("esn_rule_leak,%.2f,choisi sur TRAIN\n" % esn_rule_combo[1])
        f.write("esn_fixed_B,%d,choisi sur TRAIN\n" % esn_fx_B)
        f.write("esn_fixed_rho,%.2f,choisi sur TRAIN\n" % esn_fx_combo[0])
        f.write("esn_fixed_leak,%.2f,choisi sur TRAIN\n" % esn_fx_combo[1])
        f.write("m4r_fixed_B,%d,choisi sur TRAIN\n" % m4r_fx_B)
        f.write("m4r_rule_symmetry,%s,choisie sur TRAIN\n" % m4r_rule)
        for k, v in r.items():
            f.write("%s,%.4f,mesure sur TEST\n" % (k, v))
        for name, (mm, lo, hi) in (("V1", v1), ("V2", v2), ("V3", v3)):
            f.write("%s_delta,%.4f,IC[%.4f;%.4f] sur TEST\n" % (name, mm, lo, hi))
        f.write("n_seeds_train,%d,\n" % len(SEEDS_TRAIN))
        f.write("n_seeds_test,%d,\n" % len(SEEDS_TEST))
    print("")
    print("[csv] %s" % CSV)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 5))
        labels = ["M4R\nDOUTE\n(natif,\nadaptatif)", "M4R\nCONV",
                  "M4R\nBUDGET FIXE\n(B=%d)" % m4r_fx_B,
                  "ESN %s\n(regle choisie\nsur TRAIN)" % esn_rule,
                  "ESN\nBUDGET FIXE\n(B=%d)" % esn_fx_B, "ESN\noracle"]
        means = [r["M4R_DOUBT"], r["M4R_CONV"], r["M4R_FIXED"],
                 r["ESN_RULE"], r["ESN_FIXED"], r["ESN_ORACLE"]]
        colors = ["#d62728", "#ff9896", "#fdae6b", "#1f77b4", "#2ca02c", "#c7c7c7"]
        ax.bar(labels, means, color=colors, edgecolor="k")
        ax.axhline(0.5, ls=":", c="gray", label="hasard (0.5)")
        ax.set_ylabel("Precision de decision a l'arret (graines TEST)")
        ax.set_title("B5b-bis : la dette de methode payee -- hyperparametres, regle d'arret et\n"
                     "budget choisis sur graines TRAIN, mesures sur graines TEST disjointes")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(PNG, dpi=140)
        print("[png] %s" % PNG)
    except Exception as e:
        print("[png] skipped: %s" % e)

    print("")
    print("Wall time: %.1fs" % (time.time() - t0))
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
