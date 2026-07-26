#!/usr/bin/env python3
"""
EXPERIENCE B7 (volet 3 -- LA CONTRE-EPREUVE DU PLAFOND)

RESERVE A LEVER (posee dans le commit du volet 2). Le volet 2 a refute la
prediction "avec une vraie structure spatiale, le desaccord local mean(|L v|)
bat la dispersion globale std(v)" -- interaction +0.02 CI[-0.08,+0.12]. MAIS
std(v) y marque 1.00 dans les DEUX mondes : un PLAFOND. Un test ou l'adversaire
est au maximum ne peut pas reveler une superiorite du signal natif s'il en avait
une. La conclusion du volet 2 etait donc directionnelle, pas concluante, et cela
etait ecrit tel quel.

CE VOLET DURCIT LA TACHE jusqu'a decoller std(v) du plafond, et refait le
face-a-face. Deux leviers, tous deux deja presents dans la tache d'origine :
la DUREE du leurre (T_pulse, grille de B1d) et sa FORCE (E_DISTRACT).

AUCUNE SELECTION DE NIVEAU -- c'est le point de methode de ce volet. Le volet 2
choisissait un niveau de structure sur un gate declare ; ici, choisir le niveau
de difficulte sur la performance de std(v) reviendrait a retenir le regime ou
l'adversaire va mal, ce qui avantagerait le signal natif par regression vers la
moyenne. On mesure donc TOUS les niveaux et on rapporte TOUT. Le seul filtre,
declare avant mesure, est le gate de solvabilite (la verite doit encore pouvoir
gagner en budget illimite) -- un niveau qui casse la tache ne se lit pas.

CE QUE CHAQUE ISSUE SIGNIFIERAIT :
  - a aucun niveau hors plafond LAPLACIAN ne depasse STD -> la reserve est
    levee, le signal d'arret n'a PAS besoin de la topologie, 5e retrecissement
    de la revendication, tenable.
  - LAPLACIAN depasse a un niveau plus dur -> le volet 2 concluait sur un
    artefact de plafond, la topologie sert quand la tache est assez dure, et
    c'est la condition qu'on cherchait depuis le debut.

SORTIES : figures/expB7_ceiling_control_poc.csv + .png
Cree : 2026-07-26 (Claude Opus 5, L'Ingenieur) -- suite de expB7 volet 2.
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

SEEDS_TRAIN = list(range(20))
SEEDS_TEST = list(range(20, 60))
# (T_pulse, E_distract) -- T_pulse=700/E=1.0 est le regime des volets 1 et 2
LEVELS = [(700, 1.0), (1200, 1.0), (700, 1.5), (1200, 1.5)]
WORLDS = ["RANDOM", "CLUSTERED_K5"]
STOP_SIGNALS = ["LAPLACIAN", "STD"]
DROP_GRID = [0.15, 0.30, 0.45, 0.60]
GATE_ACC_FINAL_MIN = 0.70

CSV_PATH = ROOT / "figures" / "expB7_ceiling_control_poc.csv"
PNG_PATH = ROOT / "figures" / "expB7_ceiling_control_poc.png"


def main() -> int:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("=" * 100)
    print("EXPERIENCE B7 volet 3 -- CONTRE-EPREUVE DU PLAFOND")
    print(f"{len(LEVELS)} niveaux de difficulte x {len(WORLDS)} mondes | "
          f"{len(SEEDS_TRAIN)} TRAIN + {len(SEEDS_TEST)} TEST | AUCUNE selection de niveau")
    print("=" * 100)

    acc_grid = {}
    gate = {}
    for t_pulse, e_dist in LEVELS:
        for world in WORLDS:
            dp.E_DISTRACT = e_dist          # patch: lu par make_deceptive*
            finals, always = [], []
            for group, seeds in (("train", SEEDS_TRAIN), ("test", SEEDS_TEST)):
                for seed in seeds:
                    adj, s_on, s_off, dstar = b7c.build_task(world, seed)
                    sigs, dec, _ = b7d.simulate(adj, s_on, s_off, seed * 10 + 1, t_pulse)
                    for sname in STOP_SIGNALS:
                        for drop in DROP_GRID:
                            c = b7d.stop_at(sigs[sname], drop)
                            acc_grid.setdefault(
                                (t_pulse, e_dist, world, sname, drop, group), []
                            ).append(int(dp.dec_at(dec, c) == dstar))
                    if group == "train":    # le gate se juge sur TRAIN
                        finals.append(int(dec[-1] == dstar))
                        always.append(float(np.mean(dec[WARMUP:t_pulse] == dstar)))
            gate[(t_pulse, e_dist, world)] = (float(np.mean(finals)), float(np.mean(always)))
            print(f"  [T={t_pulse} E={e_dist} {world}] fait ({time.time() - t0:.0f}s)")
    dp.E_DISTRACT = E_DISTRACT_BASE

    print("\nRESULTATS -- tous les niveaux, aucun ecarte apres coup")
    print(f"{'T_pulse':>8}{'E_leurre':>10}{'monde':>15}{'gate acc_fin':>14}"
          f"{'LAPLACIAN':>11}{'STD':>7}{'  LAP - STD (IC 95%)':>24}{'':>6}")
    print("-" * 100)
    rows = []
    results = {}
    for t_pulse, e_dist in LEVELS:
        for world in WORLDS:
            chosen = {s: max(DROP_GRID,
                             key=lambda d: np.mean(
                                 acc_grid[(t_pulse, e_dist, world, s, d, "train")]))
                      for s in STOP_SIGNALS}
            a_lap = np.array(acc_grid[(t_pulse, e_dist, world, "LAPLACIAN",
                                       chosen["LAPLACIAN"], "test")], float)
            a_std = np.array(acc_grid[(t_pulse, e_dist, world, "STD",
                                       chosen["STD"], "test")], float)
            d, lo, hi = b7d.boot_ci(a_lap - a_std)
            gf, ga = gate[(t_pulse, e_dist, world)]
            ok = gf >= GATE_ACC_FINAL_MIN
            ceiling = a_std.mean() >= 0.995
            flag = ("" if ok else " [gate ECHEC, non lu]") + (" [plafond]" if ceiling else "")
            print(f"{t_pulse:>8}{e_dist:>10.1f}{world:>15}{gf:>14.2f}"
                  f"{a_lap.mean():>11.2f}{a_std.mean():>7.2f}"
                  f"{d:>+11.2f} [{lo:+.2f},{hi:+.2f}]{flag}")
            results[(t_pulse, e_dist, world)] = (a_lap.mean(), a_std.mean(), d, lo, hi,
                                                 ok, ceiling)
            rows.append(dict(t_pulse=t_pulse, e_distract=e_dist, world=world,
                             gate_acc_final=gf, gate_always_correct=ga,
                             gate_passed=int(ok), std_at_ceiling=int(ceiling),
                             drop_laplacian=chosen["LAPLACIAN"], drop_std=chosen["STD"],
                             acc_laplacian=float(a_lap.mean()), acc_std=float(a_std.mean()),
                             delta=d, ci_lo=lo, ci_hi=hi, n_seeds_test=len(SEEDS_TEST)))

    readable = [k for k, v in results.items() if v[5]]
    wins = [k for k in readable if results[k][3] > 0]      # IC entierement > 0
    losses = [k for k in readable if results[k][4] < 0]    # IC entierement < 0
    print("\n--- LECTURE ---")
    print(f"  niveaux lisibles (gate franchi) : {len(readable)}/{len(results)}")
    print("  NOTE DE METHODE 1 -- le plafond n'empeche de detecter qu'une SUPERIORITE")
    print("  du signal natif ; une inferioriete reste parfaitement mesurable meme quand")
    print("  std(v) est a 1.00. Le critere de lecture est donc le SIGNE de l'ecart sur")
    print("  les niveaux lisibles, pas la seule sortie du plafond.")
    print("  NOTE DE METHODE 2 -- allonger le leurre (T_pulse) ne durcit PAS la tache")
    print("  du point de vue du signal d'arret : les arrets tombent vers 270-310 pas,")
    print("  bien avant la fin du leurre, donc la trajectoire vue est identique. Les")
    print("  lignes T=1200 reproduisent T=700 a l'identique -- attendu, pas un bug.")
    print("  Seule la FORCE du leurre (E_distract) est un levier reel ici.")
    for k in readable:
        lap, std, d, lo, hi, _, ceil = results[k]
        tag = "SUPERIEUR" if lo > 0 else ("INFERIEUR" if hi < 0 else "non tranche")
        print(f"    T={k[0]} E={k[1]} {k[2]:<13} LAP {lap:.2f} vs STD {std:.2f} "
              f"-> {d:+.2f} [{lo:+.2f},{hi:+.2f}] {tag}"
              f"{' (std au plafond)' if ceil else ''}")
    if wins:
        print("  -> LE VOLET 2 CONCLUAIT SUR UN ARTEFACT : a au moins un niveau lisible,")
        print("     le desaccord local DEPASSE significativement la dispersion globale.")
        print("     C'est la condition cherchee -- a repliquer sur graines disjointes")
        print("     avant d'y croire (les recits topologiques ont deja casse 2 fois).")
    elif losses:
        print("  -> RESERVE LEVEE, ET DANS LE SENS INVERSE DE L'HYPOTHESE. Le desaccord")
        print("     local ne depasse a AUCUN niveau lisible, et quand la tache durcit il")
        print("     se DEGRADE nettement pendant que la dispersion globale tient. Le")
        print("     signal d'arret n'a pas seulement le droit d'ignorer la topologie :")
        print("     la lire coute. 5e retrecissement de la revendication, tenable.")
    else:
        print("  -> aucun ecart tranche sur les niveaux lisibles : la reserve reste")
        print("     ouverte, la grille de difficulte ne separe pas les deux signaux.")

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[csv] {CSV_PATH}")
    make_figure(results)
    print(f"[png] {PNG_PATH}")
    print(f"\nWall time: {time.time() - t0:.1f}s")
    return 0


def make_figure(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    labels = [f"T={t}\nE={e}" for t, e in LEVELS]
    xs = np.arange(len(LEVELS))

    ax = axes[0]
    for i, world in enumerate(WORLDS):
        ax.plot(xs, [results[(t, e, world)][1] for t, e in LEVELS], "o--",
                c=["#7f7f7f", "#2ca02c"][i], label=f"std(v) -- {world}")
        ax.plot(xs, [results[(t, e, world)][0] for t, e in LEVELS], "s-",
                c=["#7f7f7f", "#2ca02c"][i], alpha=0.6,
                label=f"mean|Lv| -- {world}")
    ax.axhline(1.0, ls=":", c="red", label="ceiling")
    ax.axhline(0.5, ls=":", c="gray")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("accuracy at stop (40 held-out seeds)")
    ax.set_ylim(0, 1.08)
    ax.set_title("Does hardening the task leave the ceiling?", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for i, world in enumerate(WORLDS):
        vals = [results[(t, e, world)][2] for t, e in LEVELS]
        los = [results[(t, e, world)][2] - results[(t, e, world)][3] for t, e in LEVELS]
        his = [results[(t, e, world)][4] - results[(t, e, world)][2] for t, e in LEVELS]
        ax.errorbar(xs + (i - 0.5) * 0.12, vals, yerr=[los, his], fmt="o",
                    capsize=4, c=["#7f7f7f", "#2ca02c"][i], label=world)
    ax.axhline(0, c="k", lw=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("accuracy( mean|Lv| ) - accuracy( std(v) )")
    ax.set_title("The gap, at every difficulty level (none discarded)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("Experiment B7 (part 3) -- the ceiling control: is the part-2 "
                 "refutation an artefact of a saturated baseline?", fontsize=11)
    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=140)


if __name__ == "__main__":
    raise SystemExit(main())
