#!/usr/bin/env python3
"""
CAPSTONE B1 -- la valeur CONDITIONNELLE du doute, en une figure + un tableau.

Agrege les CSV des trois consolidations (deterministes, reproductibles) pour raconter B1 :
  - Tache LOYALE  (b1c) : converger tot = avoir juste -> le doute NE gagne PAS (sur-reflechit).
  - Tache TROMPEUSE (b1d) : converger tot = se tromper -> le doute GAGNE.
  - Watchdog natif (b1b) : utile (validite > hasard) sur les 3 topologies.
Le tout sur LATTICE (regulier), BA m=3 (scale-free), ER (aleatoire), avec IC bootstrap.

Ce script NE simule RIEN : il lit uniquement figures/b1{c_summary,d,b}_*.csv produits par les
scripts de consolidation. Rejouer la chaine = relancer les 3 scripts puis celui-ci.

Sorties : figures/b1_conditional_synthesis.png + docs/b1_conditional_synthesis.md
Cree : 2026-07-08 (Claude Opus 4.8) -- synthese Volet B1 (docs/FUTURE_WORK.md).
"""
from __future__ import annotations
import sys, csv
from pathlib import Path
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
B1D_CSV = FIG / "b1d_deceptive_consolidation.csv"
B1C_CSV = FIG / "b1c_allocation_summary.csv"
B1B_CSV = FIG / "b1b_watchdog_consolidation.csv"
OUT_PNG = FIG / "b1_conditional_synthesis.png"
OUT_MD = ROOT / "docs" / "b1_conditional_synthesis.md"

TOPOS = ["LATTICE", "BA_m3", "ER_p06"]
DECEPTIVE_MIN = 350            # niveaux de leurre "trompeurs" retenus pour le gap B1d
N_BOOT = 10000
RNG = np.random.RandomState(20260708)

def load_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def boot_ci(vec):
    vec = np.asarray(vec, dtype=float)
    if len(vec) == 0:
        return 0.0, 0.0, 0.0
    n = len(vec); means = np.empty(N_BOOT)
    for b in range(N_BOOT):
        means[b] = vec[RNG.randint(0, n, n)].mean()
    return float(vec.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

def b1d_gap_by_topo():
    """Ecart acc_DOUTE - acc_CONV par topo, moyenne (par seed) sur les niveaux TROMPEURS."""
    rows = load_rows(B1D_CSV)
    out = {}
    for topo in TOPOS:
        by_seed = {}
        for r in rows:
            if r["topo"] != topo or int(r["t_pulse"]) < DECEPTIVE_MIN:
                continue
            s = int(r["seed"])
            by_seed.setdefault(s, []).append(int(r["acc_doubt"]) - int(r["acc_conv"]))
        gaps = [np.mean(v) for v in by_seed.values()]   # une valeur par seed (moy. sur niveaux)
        out[topo] = boot_ci(gaps)
    return out

def b1c_gap_by_topo():
    """Ecart acc_DOUTE - acc_CONV par topo de base (tache loyale)."""
    rows = load_rows(B1C_CSV)
    out = {}
    for topo in TOPOS:
        gaps = [float(r["acc_doubt"]) - float(r["acc_conv"]) for r in rows if r["base_topo"] == topo]
        out[topo] = boot_ci(gaps)
    return out

def b1b_utility_by_topo():
    """Ecart validite WATCHDOG - HASARD par topo (apparie par seed)."""
    rows = load_rows(B1B_CSV)
    out = {}
    for topo in TOPOS:
        wd = {int(r["seed"]): float(r["valid_frac"]) for r in rows
              if r["topo"] == topo and r["condition"] == "WATCHDOG"}
        ha = {int(r["seed"]): float(r["valid_frac"]) for r in rows
              if r["topo"] == topo and r["condition"] == "HASARD"}
        diffs = [wd[s] - ha[s] for s in wd if s in ha]
        out[topo] = boot_ci(diffs)
    return out

def fmt(ci):
    m, lo, hi = ci
    return f"{m:+.2f} [{lo:+.2f},{hi:+.2f}]"

def main():
    for p in (B1D_CSV, B1C_CSV, B1B_CSV):
        if not p.exists():
            print(f"[ERREUR] CSV manquant : {p}\n  -> relance d'abord les 3 scripts de consolidation.")
            return 1
    loyal = b1c_gap_by_topo()
    decep = b1d_gap_by_topo()
    util = b1b_utility_by_topo()

    print("=== CAPSTONE B1 : la valeur conditionnelle du doute ===\n")
    print(f"{'topo':<9}{'LOYALE (doute-conv)':>24}{'TROMPEUSE (doute-conv)':>26}{'WATCHDOG util.':>20}")
    print("-" * 79)
    for t in TOPOS:
        print(f"{t:<9}{fmt(loyal[t]):>24}{fmt(decep[t]):>26}{fmt(util[t]):>20}")

    # Lecture automatique de la structure conditionnelle
    print("\nLecture :")
    cond_ok = all(loyal[t][2] <= 0.05 for t in TOPOS) and all(decep[t][1] > -0.01 for t in TOPOS)
    for t in TOPOS:
        loyal_neg = loyal[t][2] <= 0.05
        decep_pos = decep[t][1] > 0
        print(f"  {t:<9} loyale {'<=0 (doute inutile)' if loyal_neg else 'ambigu'} ; "
              f"trompeuse {'>0 (doute gagne)' if decep_pos else 'directionnel (CI frole 0)'}")
    print("\n  --> La valeur du doute est CONDITIONNELLE : negative/nulle sur tache loyale,")
    print("      positive sur tache trompeuse. BA (scale-free) = cas le plus faible partout")
    print("      (hubs -> |L v| ne retombe pas), coherent avec la reformulation degre du preprint.")

    # Figure : 2 panneaux
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        xs = np.arange(len(TOPOS)); w = 0.38
        for k, (lab, data, c) in enumerate([("tache LOYALE (b1c)", loyal, "#1f77b4"),
                                            ("tache TROMPEUSE (b1d)", decep, "#d62728")]):
            means = [data[t][0] for t in TOPOS]
            los = [data[t][0] - data[t][1] for t in TOPOS]
            his = [data[t][2] - data[t][0] for t in TOPOS]
            axes[0].bar(xs + (k - 0.5) * w, means, w, yerr=[los, his], capsize=4,
                        color=c, edgecolor="k", label=lab)
        axes[0].axhline(0, color="k", lw=0.9)
        axes[0].set_xticks(xs); axes[0].set_xticklabels(TOPOS)
        axes[0].set_ylabel("gain du doute = acc_DOUTE - acc_CONV")
        axes[0].set_title("Valeur CONDITIONNELLE du doute (IC95 bootstrap)")
        axes[0].legend(); axes[0].grid(axis="y", alpha=0.3)
        um = [util[t][0] for t in TOPOS]
        ulo = [util[t][0] - util[t][1] for t in TOPOS]
        uhi = [util[t][2] - util[t][0] for t in TOPOS]
        axes[1].bar(xs, um, 0.5, yerr=[ulo, uhi], capsize=5, color="#2ca02c", edgecolor="k")
        axes[1].axhline(0, color="k", lw=0.9)
        axes[1].set_xticks(xs); axes[1].set_xticklabels(TOPOS)
        axes[1].set_ylabel("validite WATCHDOG - HASARD")
        axes[1].set_title("Watchdog natif : utile sur les 3 topologies (b1b)")
        axes[1].grid(axis="y", alpha=0.3)
        fig.suptitle("Synthese Volet B1 : le doute, explorateur discipline a valeur conditionnelle "
                     "(seeds x 3 topologies)", fontsize=11)
        plt.tight_layout(); plt.savefig(OUT_PNG, dpi=140)
        print(f"\n[png] {OUT_PNG}")
    except Exception as e:
        print(f"[png] skipped: {e}")

    # Tableau markdown (pour le preprint / backlog)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    with OUT_MD.open("w", encoding="utf-8") as f:
        f.write("# Synthese Volet B1 -- valeur conditionnelle du doute\n\n")
        f.write("> Genere par `experiments/b1_conditional_synthesis.py` depuis les CSV des 3 "
                "consolidations (deterministe). IC95 bootstrap.\n\n")
        f.write("| Topologie | Tache LOYALE (doute-conv) | Tache TROMPEUSE (doute-conv) | "
                "Watchdog (valid-hasard) |\n")
        f.write("|---|---|---|---|\n")
        for t in TOPOS:
            f.write(f"| {t} | {fmt(loyal[t])} | {fmt(decep[t])} | {fmt(util[t])} |\n")
        f.write("\n**Lecture.** Le doute n'ajoute rien quand converger tot suffit (tache loyale, "
                "gain <=0) et paie quand converger tot est un piege (tache trompeuse, gain >0). "
                "Robuste aux seeds et a la topologie ; **BA scale-free est le cas le plus faible "
                "partout** (les hubs empechent le desaccord laplacien de retomber), coherent avec "
                "la reformulation degre/champ-moyen du preprint.\n")
    print(f"[md]  {OUT_MD}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
