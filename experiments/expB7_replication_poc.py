#!/usr/bin/env python3
"""
EXPERIENCE B7 (volet 4 -- LE GATE DE REPLICATION)

CE QU'IL FAUT REPLIQUER. Le volet 3 a leve la reserve du plafond dans le sens
INVERSE de l'hypothese : a aucun niveau lisible le desaccord local mean(|L v|)
ne depasse la dispersion globale std(v), et sur le niveau durci
(CLUSTERED_K5, E_distract=1.5) il DECROCHE -- 0.70 contre 1.00, ecart -0.30
CI[-0.45,-0.17]. Autrement dit : lire la topologie ne serait pas seulement
inutile pour l'arret, elle couterait.

POURQUOI CE GATE EST OBLIGATOIRE ICI, PAS OPTIONNEL. Deux raisons, toutes deux
internes au projet. (1) L'affirmation repose sur UN SEUL point de la grille de
difficulte -- les deux autres niveaux lisibles sont non tranches. (2) Les recits
specifiques a la topologie ont deja casse deux fois dans ce projet : le claim
[13] a ete revise (0/9 configs se reproduisent au code actuel) et P3, le
detecteur d'anomalies, a ete refute et son signe inverse. Un effet topologique
non replique n'est pas un resultat dans cette maison.

PROTOCOLE, DECLARE AVANT MESURE. Deux groupes de graines COMPLETEMENT DISJOINTS
entre eux et de tout ce qui a servi jusqu'ici (les volets 1-3 ont consomme les
graines 0-59). Chaque groupe a son propre reglage de seuil sur ses propres
graines d'entrainement, et sa propre mesure sur ses propres graines reservees.
    groupe A : seuil sur 140-159, mesure sur 60-99
    groupe B : seuil sur 160-179, mesure sur 100-139
Deux niveaux : E_distract=1.0 (non tranche au volet 3) et 1.5 (le point qui
porte la conclusion), pour verifier le CONTRASTE et pas seulement le point.

CRITERE DE REPLICATION, POSE AVANT DE REGARDER : a E=1.5, l'ecart doit etre
negatif avec IC entierement sous zero DANS LES DEUX GROUPES. Un seul groupe sur
deux = ECHEC de replication, et la phrase "la lire coute" ne doit pas etre
citee. Le gate de solvabilite du volet 2 s'applique aussi aux graines neuves.

SORTIES : figures/expB7_replication_poc.csv + .png
Cree : 2026-07-27 (Claude Opus 5, L'Ingenieur) -- suite de expB7 volet 3.
"""
from __future__ import annotations

import csv
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
sys.path.insert(0, str(ROOT / "experiments" / "scratch"))
import deceptive_task_poc as dp  # noqa: E402
import expB7_spatial_structure_diagnostic_poc as b7d  # noqa: E402
import expB7_contiguous_sources_poc as b7c  # noqa: E402

dp.MAX_BUDGET = 2000
MAXB = dp.MAX_BUDGET
WARMUP = dp.WARMUP
E_DISTRACT_BASE = dp.E_DISTRACT

WORLD = "CLUSTERED_K5"
T_PULSE = 700
E_LEVELS = [1.0, 1.5]
STOP_SIGNALS = ["LAPLACIAN", "STD"]
DROP_GRID = [0.15, 0.30, 0.45, 0.60]
GATE_ACC_FINAL_MIN = 0.70
# graines 0-59 deja consommees par les volets 1-3 : ces deux groupes sont neufs
GROUPS = {
    "A": dict(train=list(range(140, 160)), test=list(range(60, 100))),
    "B": dict(train=list(range(160, 180)), test=list(range(100, 140))),
}

CSV_PATH = ROOT / "figures" / "expB7_replication_poc.csv"
PNG_PATH = ROOT / "figures" / "expB7_replication_poc.png"


def main() -> int:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("=" * 100)
    print("EXPERIENCE B7 volet 4 -- GATE DE REPLICATION sur graines JAMAIS TOUCHEES")
    print(f"monde {WORLD} | {len(E_LEVELS)} niveaux | 2 groupes disjoints "
          f"(A: seuil 140-159 / mesure 60-99 ; B: seuil 160-179 / mesure 100-139)")
    print("=" * 100)

    acc = {}
    gate = {}
    for e_dist in E_LEVELS:
        dp.E_DISTRACT = e_dist
        for gname, gsel in GROUPS.items():
            finals = []
            for group in ("train", "test"):
                for seed in gsel[group]:
                    adj, s_on, s_off, dstar = b7c.build_task(WORLD, seed)
                    sigs, dec, _ = b7d.simulate(adj, s_on, s_off, seed * 10 + 1, T_PULSE)
                    for sname in STOP_SIGNALS:
                        for drop in DROP_GRID:
                            c = b7d.stop_at(sigs[sname], drop)
                            acc.setdefault((e_dist, gname, sname, drop, group),
                                           []).append(int(dp.dec_at(dec, c) == dstar))
                    if group == "train":
                        finals.append(int(dec[-1] == dstar))
            gate[(e_dist, gname)] = float(np.mean(finals))
            print(f"  [E={e_dist} groupe {gname}] fait ({time.time() - t0:.0f}s)")
    dp.E_DISTRACT = E_DISTRACT_BASE

    print("\nRESULTATS -- chaque groupe regle son seuil sur SES graines d'entrainement")
    print(f"{'E_leurre':>9}{'groupe':>8}{'gate acc_fin':>14}{'LAPLACIAN':>11}"
          f"{'STD':>7}{'  LAP - STD (IC 95%)':>24}")
    print("-" * 80)
    rows = []
    res = {}
    for e_dist in E_LEVELS:
        for gname in GROUPS:
            chosen = {s: max(DROP_GRID,
                             key=lambda d: np.mean(acc[(e_dist, gname, s, d, "train")]))
                      for s in STOP_SIGNALS}
            a_lap = np.array(acc[(e_dist, gname, "LAPLACIAN",
                                  chosen["LAPLACIAN"], "test")], float)
            a_std = np.array(acc[(e_dist, gname, "STD", chosen["STD"], "test")], float)
            d, lo, hi = b7d.boot_ci(a_lap - a_std)
            gf = gate[(e_dist, gname)]
            res[(e_dist, gname)] = (a_lap.mean(), a_std.mean(), d, lo, hi, gf)
            print(f"{e_dist:>9.1f}{gname:>8}{gf:>14.2f}{a_lap.mean():>11.2f}"
                  f"{a_std.mean():>7.2f}{d:>+11.2f} [{lo:+.2f},{hi:+.2f}]"
                  f"{'' if gf >= GATE_ACC_FINAL_MIN else '  [gate ECHEC]'}")
            rows.append(dict(e_distract=e_dist, seed_group=gname,
                             gate_acc_final=gf,
                             gate_passed=int(gf >= GATE_ACC_FINAL_MIN),
                             drop_laplacian=chosen["LAPLACIAN"], drop_std=chosen["STD"],
                             acc_laplacian=float(a_lap.mean()),
                             acc_std=float(a_std.mean()),
                             delta=d, ci_lo=lo, ci_hi=hi,
                             n_seeds_test=len(GROUPS[gname]["test"])))

    print("\n--- GATE DE REPLICATION (critere pose avant mesure) ---")
    hard = [(1.5, g) for g in GROUPS]
    confirmed = [k for k in hard if res[k][4] < 0 and res[k][5] >= GATE_ACC_FINAL_MIN]
    print(f"  a E=1.5, groupes ou l'ecart est negatif avec IC entierement sous zero"
          f" ET gate franchi : {len(confirmed)}/2")
    for k in hard:
        lap, std, d, lo, hi, gf = res[k]
        state = ("CONFIRME" if (hi < 0 and gf >= GATE_ACC_FINAL_MIN)
                 else "gate echec" if gf < GATE_ACC_FINAL_MIN else "non tranche")
        print(f"    groupe {k[1]} : LAP {lap:.2f} vs STD {std:.2f} -> "
              f"{d:+.2f} [{lo:+.2f},{hi:+.2f}]  {state}")
    print("  contraste avec le niveau non durci (E=1.0) :")
    for g in GROUPS:
        lap, std, d, lo, hi, gf = res[(1.0, g)]
        print(f"    groupe {g} : LAP {lap:.2f} vs STD {std:.2f} -> "
              f"{d:+.2f} [{lo:+.2f},{hi:+.2f}]")

    if len(confirmed) == 2:
        print("\n  -> REPLIQUE. Sur deux groupes de graines jamais touchees et")
        print("     independants l'un de l'autre, lire le desaccord local coute quand")
        print("     la tache durcit. La phrase est citable, avec sa portee : sur CETTE")
        print("     tache, a CE niveau de difficulte, pour CE role (le signal d'arret).")
    elif len(confirmed) == 1:
        print("\n  -> ECHEC DE REPLICATION : un groupe sur deux. L'effet du volet 3")
        print("     n'est pas assez solide pour etre cite. A traiter comme du bruit")
        print("     jusqu'a preuve du contraire -- exactement ce que le gate sert a")
        print("     detecter (cf. Condorcet, 13/07, mort a la replication).")
    else:
        print("\n  -> NON REPLIQUE. Le -0.30 du volet 3 ne survit pas a des graines")
        print("     neuves : a ecrire tel quel, et la conclusion du volet 3 se limite")
        print("     alors a 'le desaccord local ne depasse pas', sans le 'il coute'.")

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[csv] {CSV_PATH}")
    make_figure(res)
    print(f"[png] {PNG_PATH}")
    print(f"\nWall time: {time.time() - t0:.1f}s")
    return 0


def make_figure(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    xs = np.arange(len(E_LEVELS))
    cols = {"A": "#d62728", "B": "#1f77b4"}

    ax = axes[0]
    for g in GROUPS:
        ax.plot(xs, [res[(e, g)][0] for e in E_LEVELS], "s-", c=cols[g],
                label=f"mean|Lv| -- group {g}")
        ax.plot(xs, [res[(e, g)][1] for e in E_LEVELS], "o--", c=cols[g], alpha=0.55,
                label=f"std(v) -- group {g}")
    ax.axhline(0.5, ls=":", c="gray")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"E={e}" for e in E_LEVELS])
    ax.set_ylabel("accuracy at stop (40 held-out seeds)")
    ax.set_ylim(0, 1.08)
    ax.set_title("Replication on untouched seeds", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for i, g in enumerate(GROUPS):
        vals = [res[(e, g)][2] for e in E_LEVELS]
        los = [res[(e, g)][2] - res[(e, g)][3] for e in E_LEVELS]
        his = [res[(e, g)][4] - res[(e, g)][2] for e in E_LEVELS]
        ax.errorbar(xs + (i - 0.5) * 0.1, vals, yerr=[los, his], fmt="o", capsize=4,
                    c=cols[g], label=f"group {g}")
    ax.axhline(0, c="k", lw=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"E={e}" for e in E_LEVELS])
    ax.set_ylabel("accuracy( mean|Lv| ) - accuracy( std(v) )")
    ax.set_title("Does reading the topology cost, on fresh seeds?", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("Experiment B7 (part 4) -- replication gate on two disjoint, "
                 "never-used seed groups", fontsize=11)
    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=140)


if __name__ == "__main__":
    raise SystemExit(main())
