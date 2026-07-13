#!/usr/bin/env python3
"""
P7 -- L'INDUCTEUR CHIMIQUE DU LABO : une inertie de desaccord change-t-elle
la niche de l'horloge de deliberation (P11) ?
=============================================================================
Cree : 2026-07-13 (Claude Sonnet 5, L'Ingenieur) -- piste du legs de Fable
(docs/PISTES_POUR_LA_SUITE_2026-07-12.md, section I, P7).

TRACE : `_SHADOW_LAB/laboratoire_absurde/experience_008_v2_chemical_inductor.py`
+ `_v3_stress_test.py` (16 mai 2026, Agent Flux). Lu a froid ce jour. Le
mecanisme central, une fois debarrasse du decor (duel majorite/minorite
haineuse, matrice d'identite -1/+1, jamais reutilise ailleurs dans M4R) :
une troisieme variable L qui est un FILTRE PASSE-BAS (premier ordre) du
signal de couplage entrant --
    dL/dt = beta * (I_coup - L)
-- qui se re-injecte ensuite dans dv. C'est une INERTIE : L integre le
desaccord recent au lieu d'y reagir instantanement. Les regles du Labo
interdisent d'y REPARER (on ne rend pas 008 "propre") -- rien n'interdit
d'EXTRAIRE ce mecanisme et de le re-tester ailleurs, proprement.

CE QUE CE SCRIPT NE FAIT PAS : reproduire le duel majorite/minorite de 008
(decor jetable, jamais partie du coeur M4R) ni toucher dynamics.py. A la
place, l'inertie est appliquee EN POST-TRAITEMENT au signal du side-car M4R
deja construit par P11 (`experiments/p11_universal_stopping_poc.py`,
`sidecar_traces` / `sig_sm`) -- meme protocole, memes taches, memes seeds,
UN SEUL ingredient nouveau : un filtre passe-bas supplementaire (constante
beta, comme 008) sur le signal deja lisse M4R_SIG avant la regle d'arret.
Le side-car n'est calcule QU'UNE FOIS par (probleme, k_net) -- M4R_U,
M4R_SIG et M4R_L (pour chaque beta) en derivent tous, aucune redondance.

QUESTION (le texte de la piste) : "une inertie de desaccord change-t-elle la
niche du doute -- elle pourrait allonger l'horloge de deliberation ?"
Traduction operationnelle : le signal M4R_SIG remonte deja quand le
desaccord reapparait (sortie de plateau) -- mais il peut aussi FAUSSEMENT
retomber pendant de breves fluctuations a l'interieur d'un plateau trompeur
(bruit du side-car), declenchant un arret premature. Une inertie devrait
LISSER ces faux replis SANS empecher la vraie sortie de plateau (plus lente,
donc toujours detectee) -- prediction testable, pas garantie.

PROTOCOLE : reutilise EXACTEMENT le harness P11 (24 problemes, solveur de
gradient, familles easy/trap deja pre-volees, side-car M4R identique). Une
seule regle NOUVELLE, M4R_L : signal = EMA(sig_sm, beta), meme regle d'arret
"retombe sous frac*pic-roulant" que M4R_SIG. Sweep beta in {0.01, 0.03, 0.1,
0.3} (008 utilisait beta=0.1 ; on balaie autour, pas d'invention d'un chiffre
unique) x memes (k_net, frac) que M4R_U/M4R_SIG. Choix du meilleur parametre
GLOBAL par la meme regle que P11 (succes moyen, cout en tie-break).

CRITERE PRE-FIXE (avant de lancer) : l'inertie est UTILE si M4R_L bat
M4R_SIG en succes GLOBAL (IC bootstrap apparie > 0) sans sacrifier les
faciles (succes easy >= 0.90). Verdict negatif possible et informatif :
si l'inertie retarde aussi la vraie sortie de plateau autant que les faux
replis, elle n'ajoute rien (parite ou perte) -- a rapporter tel quel.

Sorties : figures/p7_chemical_inductor_poc.csv + .png
Statut : exploratoire, hors preprint, coeur non touche (L est un
post-traitement du side-car, pas une modification de dynamics.py).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
import p11_universal_stopping_poc as p11  # noqa: E402

CSV_PATH = ROOT / "figures" / "p7_chemical_inductor_poc.csv"
PNG_PATH = ROOT / "figures" / "p7_chemical_inductor_poc.png"

BETAS = [0.01, 0.03, 0.1, 0.3]     # constante de l'inductance (008 utilisait 0.1)


def chemical_inductor(signal, beta):
    """dL/dt = beta*(signal - L), integre en Euler explicite (dt=1 iteration,
    coherent avec le pas discret du side-car -- meme convention que 008)."""
    L = np.empty_like(signal)
    L[0] = signal[0]
    for t in range(1, len(signal)):
        L[t] = L[t - 1] + beta * (signal[t] - L[t - 1])
    return L


def main() -> int:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    adj = p11.make_lattice_adj(p11.SIDE, periodic=True).astype(float)
    problems = ([("easy", s) for s in p11.SEEDS] + [("trap", s) for s in p11.SEEDS])

    print("[pre-vol] reutilise le pre-vol P11 (deja durci, memes seeds, aucun changement de tache).")

    # ---------------- UNE seule passe : TOL/PATIENCE + side-car (M4R_U/SIG/L) ----------------
    store = {}
    rows = []
    done = 0
    for kind, seed in problems:
        pb = p11.make_problem(kind, seed)
        xs, rs = p11.solve_traj(pb)
        entry = {}
        for tol in p11.TOLS:
            st = p11.stop_tol(rs, tol)
            entry[("TOL", tol)] = (st, int(abs(xs[st]) < p11.SUCCESS_TOL), 0)
        for K in p11.PATIENCE_KS:
            st = p11.stop_patience(rs, K)
            entry[("PATIENCE", K)] = (st, int(abs(xs[st]) < p11.SUCCESS_TOL), 0)
        for k_net in p11.KNETS:
            u_tr, sig_sm = p11.sidecar_traces(rs, seed * 10 + 1, k_net, adj)  # calcule UNE FOIS
            for frac in p11.FRACS:
                st_u = p11.stop_rolling(u_tr, frac)
                entry[("M4R_U", (k_net, frac))] = (st_u, int(abs(xs[st_u]) < p11.SUCCESS_TOL), st_u * k_net * p11.N)
                st_s = p11.stop_rolling(sig_sm, frac)
                entry[("M4R_SIG", (k_net, frac))] = (st_s, int(abs(xs[st_s]) < p11.SUCCESS_TOL), st_s * k_net * p11.N)
            for beta in BETAS:
                L = chemical_inductor(sig_sm, beta)          # derive du MEME sig_sm, pas de recalcul
                for frac in p11.FRACS:
                    st_l = p11.stop_rolling(L, frac)
                    entry[("M4R_L", (k_net, beta, frac))] = (
                        st_l, int(abs(xs[st_l]) < p11.SUCCESS_TOL), st_l * k_net * p11.N)
        store[(kind, seed)] = entry
        for (rule, param), (st, ok, extra) in entry.items():
            rows.append((kind, seed, rule, str(param).replace(",", ";"), st, ok, extra))
        done += 1
        if done % 6 == 0:
            print(f"  [{done}/{len(problems)}] {time.time()-t0:.0f}s")

    # ---------------- choix GLOBAL par regle ----------------
    keys = sorted(store.keys())
    RULES = [("TOL", p11.TOLS), ("PATIENCE", p11.PATIENCE_KS),
             ("M4R_U", [(k, f) for k in p11.KNETS for f in p11.FRACS]),
             ("M4R_SIG", [(k, f) for k in p11.KNETS for f in p11.FRACS]),
             ("M4R_L", [(k, b, f) for k in p11.KNETS for b in BETAS for f in p11.FRACS])]
    best = {}
    print("\n=== CHOIX GLOBAL (succes moyen sur les 24 problemes, tie-break cout) ===")
    for rule, params in RULES:
        scored = []
        for p in params:
            oks = [store[k][(rule, p)][1] for k in keys]
            costs = [store[k][(rule, p)][0] for k in keys]
            scored.append((np.mean(oks), -np.mean(costs), p))
        scored.sort(reverse=True)
        best[rule] = scored[0][2]
        print(f"  {rule:<9} -> param={scored[0][2]}  (succes={scored[0][0]:.3f}, "
              f"cout solveur moyen={-scored[0][1]:.0f} iters)")

    print(f"\n{'regle':<10}{'succes easy':>12}{'succes trap':>12}{'succes tous':>12}"
          f"{'cout iters':>11}{'cout reseau':>14}")
    print("-" * 70)
    agg_rows = []
    for rule, _ in RULES:
        p = best[rule]
        res = {"easy": [], "trap": []}
        costs, extras = [], []
        for k in keys:
            st, ok, extra = store[k][(rule, p)]
            res[k[0]].append(ok)
            costs.append(st)
            extras.append(extra)
        se, st_, sa = np.mean(res["easy"]), np.mean(res["trap"]), np.mean(res["easy"] + res["trap"])
        print(f"{rule:<10}{se:>12.2f}{st_:>12.2f}{sa:>12.2f}{np.mean(costs):>11.0f}{np.mean(extras):>14.0f}")
        agg_rows.append((rule, str(p).replace(",", ";"), se, st_, sa, np.mean(costs), np.mean(extras)))

    print("\n=== VERDICT P7 (pre-fixe : M4R_L bat M4R_SIG en succes global sans sacrifier easy>=0.90) ===")

    def okvec(rule):
        return [store[k][(rule, best[rule])][1] for k in keys]

    d, lo, hi = p11.boot_ci_paired(okvec("M4R_L"), okvec("M4R_SIG"))
    print(f"  M4R_L - M4R_SIG = {d:+.3f} CI[{lo:+.3f},{hi:+.3f}]")
    easy_L = np.mean([store[k][("M4R_L", best["M4R_L"])][1] for k in keys if k[0] == "easy"])
    print(f"  M4R_L succes easy = {easy_L:.2f} (seuil >=0.90)")
    if lo > 0 and easy_L >= 0.90:
        print("  -> UTILE : l'inertie du desaccord elargit la niche de l'horloge de deliberation.")
    elif hi < 0:
        print("  -> L'INERTIE NUIT : M4R_SIG (sans inertie) reste devant -- l'inertie retarde "
              "la vraie sortie de plateau autant que les faux replis.")
    else:
        print("  -> PARITE : l'inertie ne change pas la niche mesurablement -- resultat negatif "
              "honnete, le mecanisme extrait de 008 n'apporte rien ici tel quel.")
    d2, lo2, hi2 = p11.boot_ci_paired(okvec("M4R_L"), okvec("M4R_U"))
    print(f"\n  M4R_L - M4R_U   = {d2:+.3f} CI[{lo2:+.3f},{hi2:+.3f}]")

    with CSV_PATH.open("w", encoding="utf-8") as f:
        f.write("kind,seed,rule,param,stop_iter,success,network_node_steps\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print(f"\n[csv] {CSV_PATH}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
        pb_t = p11.make_problem("trap", 0)
        xs_t, rs_t = p11.solve_traj(pb_t)
        u_tr, sig_sm = p11.sidecar_traces(rs_t, 1, best["M4R_L"][0], adj)
        L = chemical_inductor(sig_sm, best["M4R_L"][1])
        ax = axes[0]
        ax.plot(sig_sm, c="#d62728", label="M4R_SIG (sans inertie)", alpha=0.7)
        ax.plot(L, c="#1f77b4", label=f"M4R_L (inertie beta={best['M4R_L'][1]})", lw=1.6)
        ax.set_xlabel("iteration solveur"); ax.set_ylabel("signal de desaccord")
        ax.set_title("L'inertie lisse-t-elle les faux replis ?")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax = axes[1]
        names = [r[0] for r in agg_rows]
        for i, (nm, key, col) in enumerate([("easy", 2, "#2ca02c"), ("trap", 3, "#d62728")]):
            ax.bar(np.arange(len(names)) + (i - 0.5) * 0.35, [r[key] for r in agg_rows],
                   width=0.35, color=col, label=nm)
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=8)
        ax.set_ylabel("taux de succes"); ax.set_title("Succes par regle")
        ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
        fig.suptitle("P7 -- l'inducteur chimique (008) comme inertie de l'horloge de deliberation (P11)",
                     fontsize=11)
        plt.tight_layout()
        plt.savefig(PNG_PATH, dpi=140)
        print(f"[png] {PNG_PATH}")
    except Exception as e:
        print(f"[png] skipped: {e}")

    print(f"\nWall time: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
