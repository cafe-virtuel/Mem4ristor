#!/usr/bin/env python3
"""
A3 — Regression de regime sur de VRAIES simulations, etiquetee par H_cont.

CONTEXTE (backlog docs/FUTURE_WORK.md, item A3).
  Le script historique `p2_edge_betweenness_analysis.py` NE SIMULE PAS : il lit
  un dict REGIME code en dur, labellise par TYPE de topologie (12 decisions
  dupliquees x3, pas 36 observations). La "complete separation" en lambda2~2.31
  du preprint en decoulait -> quasi-tautologique.

  Le mandat lambda2 du 1er juillet (bouclage_regime_vs_predicteurs.py) a deja
  simule par (topo, seed) et montre k_harm 2/70 vs lambda2 15/70 -- MAIS en
  etiquetant le regime avec H_cog (5 bins), l'artefact de binning que A5 veut
  bannir des resultats primaires.

  Ce script refait le bouclage en etiquetant le regime par H_cont (100 bins,
  continu), pour repondre a la reserve A3/A5/C1 : "la frontiere se deplace-t-elle
  quand on passe a une metrique continue ?". Il calcule H_cog EN PARALLELE pour
  verifier qu'on reproduit bien 2/70 vs 15/70 (non-regression du resultat).

PROTOCOLE (fidele au bouclage pour comparabilite) :
  - N=200, 5 seeds, 14 topologies : BA m in {2..6}, ER p in {.03,.05,.08},
    random-regular k in {4,6,8}, ring k in {4,6,8}.
  - Regime endogene (I_stim=0), cold_start, sigma_v=0.05.
  - best-of-two-norms : pour chaque metrique, on garde le MAX sur
    {uniform, degree_linear} (le regime "dead" = meme la meilleure normalisation
    est morte, selon la metrique consideree).
  - H mesure par moyenne sur la queue (derniers TAIL_FRAC des pas) pour lisser
    le bruit du snapshot -- identique pour H_cont et H_cog (comparaison loyale).
  - On sort aussi H_cog du snapshot final (colonne hcog_snap) pour reproduire
    exactement le protocole du bouclage historique.

SORTIES :
  experiments/figures/a3_regime_regression_hcont.csv  (70 lignes, 1 par instance)
  experiments/figures/a3_regime_regression_hcont.png  (2 panneaux : vs k_harm / vs lambda2)

Cree : 2026-07-08 (Claude Opus 4.8, L'Ingenieur), backlog A3.
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
import networkx as nx
from scipy.linalg import eigh
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from mem4ristor.core import Mem4Network
from mem4ristor.metrics import (
    calculate_cognitive_entropy,
    calculate_continuous_entropy,
)

# --- parametres (identiques au bouclage sauf mention) -----------------------
N = 200
STEPS = 3000
TAIL_FRAC = 0.25            # fraction finale des pas utilisee pour moyenner H
SEEDS = [42, 123, 777, 17, 256]
I_STIM = 0.0
HERETIC_RATIO = 0.15
SIGMA_V = 0.05
NORMS = ["uniform", "degree_linear"]

# Livrables ecrits dans figures/ a la RACINE du repo (la ou le preprint les
# charge via ../../../figures/), comme p2_edge_betweenness et les autres figures.
FIG_DIR = ROOT / "figures"
CSV_PATH = FIG_DIR / "a3_regime_regression_hcont.csv"
PNG_PATH = FIG_DIR / "a3_regime_regression_hcont.png"

FAM_COLORS = {"BA": "#1f77b4", "ER": "#2ca02c", "rr": "#ff7f0e", "ring": "#d62728"}


def fiedler(adj: np.ndarray) -> float:
    d = adj.sum(1)
    return float(np.sort(eigh(np.diag(d) - adj, eigvals_only=True))[1])


def run_metrics(adj: np.ndarray, seed: int) -> dict:
    """Simule les deux normalisations, renvoie le meilleur H_cont et H_cog
    (chacun maximise sur les normes), plus le H_cog du snapshot final."""
    tail = int(STEPS * TAIL_FRAC)
    best = {"hcont": -1.0, "hcog": -1.0, "hcog_snap": -1.0}
    for norm in NORMS:
        net = Mem4Network(size=int(np.sqrt(adj.shape[0])),
                          heretic_ratio=HERETIC_RATIO, seed=seed,
                          adjacency_matrix=adj.copy(), coupling_norm=norm,
                          cold_start=True)
        net.model.cfg["noise"]["sigma_v"] = SIGMA_V
        hcont_tail, hcog_tail = [], []
        for t in range(STEPS):
            net.step(I_stimulus=I_STIM)
            if t >= STEPS - tail:
                v = net.v
                hcont_tail.append(calculate_continuous_entropy(v))
                hcog_tail.append(calculate_cognitive_entropy(v))
        hcont = float(np.mean(hcont_tail))
        hcog = float(np.mean(hcog_tail))
        hcog_snap = float(calculate_cognitive_entropy(net.v))
        # chaque metrique garde son propre meilleur choix de normalisation
        best["hcont"] = max(best["hcont"], hcont)
        best["hcog"] = max(best["hcog"], hcog)
        best["hcog_snap"] = max(best["hcog_snap"], hcog_snap)
    return best


def graphs():
    specs = []
    for m in [2, 3, 4, 5, 6]:
        specs.append((f"BA m={m}", "BA", lambda s, m=m: nx.barabasi_albert_graph(N, m, seed=s)))
    for p in [0.03, 0.05, 0.08]:
        specs.append((f"ER p={p}", "ER", lambda s, p=p: nx.erdos_renyi_graph(N, p, seed=s)))
    for k in [4, 6, 8]:
        specs.append((f"rr k={k}", "rr", lambda s, k=k: nx.random_regular_graph(k, N, seed=s)))
    for k in [4, 6, 8]:
        specs.append((f"ring k={k}", "ring", lambda s, k=k: nx.watts_strogatz_graph(N, k, 0.0, seed=s)))
    return specs


def sep_quality(rows, key, dead_key, invert=False):
    """Meilleur seuil (minimise le nb d'instances mal classees) et nb d'erreurs.
    dead si x > thr (ou x < thr si invert)."""
    xs = sorted(set(r[key] for r in rows))
    cands = xs + [(a + b) / 2 for a, b in zip(xs, xs[1:])]
    best_err, best_thr = len(rows) + 1, None
    for thr in cands:
        err = 0
        for r in rows:
            pred = (r[key] > thr) if not invert else (r[key] < thr)
            if int(pred) != r[dead_key]:
                err += 1
        if err < best_err:
            best_err, best_thr = err, thr
    return best_err, best_thr


def corr_block(rows, ykey):
    """Correlations Pearson + Spearman de y vs chaque predicteur."""
    y = np.array([r[ykey] for r in rows])
    out = {}
    for key in ["lambda2", "k_mean", "k_harm"]:
        x = np.array([r[key] for r in rows])
        pr, pp = stats.pearsonr(x, y)
        sr, sp = stats.spearmanr(x, y)
        out[key] = (pr, pp, sr, sp)
    return out


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = []

    print("=" * 104)
    print(f"A3 : regime (H_cont mesure) vs predicteurs -- N={N}, endogene, {len(SEEDS)} seeds")
    print("=" * 104)
    print(f"{'topo':<11}{'fam':>5}{'lambda2':>9}{'k_mean':>8}{'k_harm':>8}"
          f"{'H_cont':>9}{'H_cog':>8}{'hcog_snap':>10}")
    print("-" * 104)

    for name, fam, gen in graphs():
        agg = {"lambda2": [], "k_mean": [], "k_harm": [], "hcont": [], "hcog": [], "hcog_snap": []}
        for s in SEEDS:
            G = gen(s)
            adj = nx.to_numpy_array(G, dtype=float)
            deg = adj.sum(1)
            l2 = fiedler(adj)
            k_mean = float(deg.mean())
            inv_deg = float(np.mean(1.0 / np.where(deg < 1, 1, deg)))
            k_harm = 1.0 / inv_deg
            met = run_metrics(adj, s)
            row = {"name": name, "fam": fam, "seed": s,
                   "lambda2": l2, "k_mean": k_mean, "k_harm": k_harm,
                   "inv_deg": inv_deg,
                   "hcont": met["hcont"], "hcog": met["hcog"],
                   "hcog_snap": met["hcog_snap"]}
            rows.append(row)
            for k in agg:
                agg[k].append(row[k])
        print(f"{name:<11}{fam:>5}{np.mean(agg['lambda2']):>9.3f}"
              f"{np.mean(agg['k_mean']):>8.2f}{np.mean(agg['k_harm']):>8.2f}"
              f"{np.mean(agg['hcont']):>9.3f}{np.mean(agg['hcog']):>8.3f}"
              f"{np.mean(agg['hcog_snap']):>10.3f}")

    # --- CSV ---------------------------------------------------------------
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "name", "fam", "seed", "lambda2", "k_mean", "k_harm", "inv_deg",
            "hcont", "hcog", "hcog_snap"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n[csv] {CSV_PATH}")

    # --- Regression continue (le regime = H_cont, pas de seuil arbitraire) --
    print("\n" + "=" * 104)
    print("REGRESSION CONTINUE : le regime (H_cont) vs chaque predicteur (70 instances)")
    print("-" * 104)
    print(f"{'predicteur':<12}{'Pearson r':>12}{'p':>11}{'Spearman rho':>15}{'p':>11}")
    for ykey, label in [("hcont", "H_cont (continu, 100 bins)"),
                         ("hcog", "H_cog (5 bins, artefact -- controle)")]:
        print(f"\n  regime = {label}")
        cb = corr_block(rows, ykey)
        for key in ["lambda2", "k_mean", "k_harm"]:
            pr, pp, sr, sp = cb[key]
            print(f"  {key:<12}{pr:>12.3f}{pp:>11.2e}{sr:>15.3f}{sp:>11.2e}")

    # --- Classification dead/live : seuils determines par les donnees -------
    # H_cog : seuil canonique 0.10 (comme le bouclage). H_cont : pas de zero
    # (plancher de bruit) -> seuil = point milieu du plus grand gap dans les
    # valeurs triees (transparent, determine par les donnees, pas impose).
    def dead_by_threshold(rows, key, thr, below_is_dead):
        for r in rows:
            r[key + "_dead"] = int((r[key] < thr) if below_is_dead else (r[key] > thr))

    # seuil H_cont : plus grand gap
    hc_sorted = sorted(r["hcont"] for r in rows)
    gaps = [(b - a, (a + b) / 2) for a, b in zip(hc_sorted, hc_sorted[1:])]
    hcont_gap, hcont_thr = max(gaps, key=lambda g: g[0])
    dead_by_threshold(rows, "hcont", hcont_thr, below_is_dead=True)
    dead_by_threshold(rows, "hcog", 0.10, below_is_dead=True)

    n_dead_cont = sum(r["hcont_dead"] for r in rows)
    n_dead_cog = sum(r["hcog_dead"] for r in rows)
    print("\n" + "=" * 104)
    print("CLASSIFICATION dead/live -- qualite de separation par predicteur "
          "(nb mal classes / 70, plus bas = mieux)")
    print("-" * 104)
    print(f"  seuil H_cont = {hcont_thr:.3f} (milieu du plus grand gap = {hcont_gap:.3f}) "
          f"-> {n_dead_cont}/70 dead")
    print(f"  seuil H_cog  = 0.100 (canonique bouclage) -> {n_dead_cog}/70 dead")
    for dead_key, dlabel in [("hcont_dead", "regime par H_cont"),
                             ("hcog_dead", "regime par H_cog (controle)")]:
        print(f"\n  {dlabel} :")
        for key, inv in [("lambda2", False), ("k_mean", False), ("k_harm", False)]:
            err, thr = sep_quality(rows, key, dead_key, invert=inv)
            sense = "<" if inv else ">"
            print(f"    {key:<10}: {err:2d}/70 erreurs  (seuil dead {sense} {thr:.4f})")

    # --- Figure : H_cont vs k_harm (collapse) / vs lambda2 (scatter) --------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    panels = [("k_harm", "Harmonic degree  $k_{harm}=1/\\langle 1/\\deg\\rangle$", axes[0]),
              ("lambda2", "Algebraic connectivity  $\\lambda_2$", axes[1])]
    for xkey, xlabel, ax in panels:
        for fam, color in FAM_COLORS.items():
            xs = [r[xkey] for r in rows if r["fam"] == fam]
            ys = [r["hcont"] for r in rows if r["fam"] == fam]
            ax.scatter(xs, ys, color=color, s=55, edgecolors="k",
                       linewidths=0.5, alpha=0.85, zorder=5)
        ax.axhline(hcont_thr, color="grey", ls="--", lw=1,
                   label=f"dead threshold H_cont={hcont_thr:.2f}")
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("$H_{cont}$ (bits, 100-bin)", fontsize=11)
        ax.grid(alpha=0.3)
    axes[0].set_title("Regime collapses on a single curve vs harmonic degree", fontsize=10)
    axes[1].set_title("Regime scatters vs algebraic connectivity", fontsize=10)
    legend_el = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                        markeredgecolor="k", markersize=8, label=f)
                 for f, c in FAM_COLORS.items()]
    axes[0].legend(handles=legend_el, fontsize=9, loc="upper right", title="Family")
    fig.suptitle("A3 -- Regime (measured H_cont) vs predictors: harmonic degree unifies "
                 "families, lambda2 does not\n"
                 f"(N={N}, {len(SEEDS)} seeds, endogenous, best-of-two-norms)",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=140)
    print(f"\n[png] {PNG_PATH}")
    print(f"Wall time: {time.time() - t0:.1f}s | {len(rows)} instances")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
