#!/usr/bin/env python3
"""
Piste A — Bimodality analysis of ENDOGENOUS FULL synchrony (P1.5bis follow-up).

Motivation: per-seed inspection of §3novedecies revealed that ENDOGENOUS FULL
synchrony is not unimodal:
  6 seeds at sync ≈ 0.2–0.4  (mode 'coordinated')
  3 seeds at sync ≈ 0.03     (mode 'desynchronised')
The mean ± sem report masks this structure.

Hypothesis: the graph topology (λ₂ of the Laplacian, max degree, clustering,
diameter) predicts which attractor the system settles into.

Method:
  - Regenerate the 10 BA(m=3, N=100) graphs from the same seeds used in
    ablation_coordination.py.
  - Compute 4 topology metrics per graph: λ₂ (algebraic connectivity),
    max_degree, avg_clustering, diameter.
  - Load ENDOGENOUS FULL sync values from figures/ablation_coordination.csv.
  - Regress sync against each metric (Pearson + Spearman).
  - Produce a 4-panel scatter figure.

Output:
  figures/coordination_bimodality.png
  figures/coordination_bimodality.csv

Created: 2026-04-21 (P1.5bis piste A).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

# Windows consoles default to cp1252 and choke on λ/ρ/ₐ glyphs.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.linalg import eigh

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

N_NODES = 100
BA_M = 3
SEEDS = list(range(10))

CSV_IN = ROOT / "figures" / "ablation_coordination.csv"
FIG_OUT = ROOT / "figures" / "coordination_bimodality.png"
CSV_OUT = ROOT / "figures" / "coordination_bimodality.csv"


def make_ba_adjacency(n: int, m: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    adj = np.zeros((n, n), dtype=float)
    for i in range(m + 1):
        for j in range(i + 1, m + 1):
            adj[i, j] = adj[j, i] = 1.0
    degrees = np.sum(adj, axis=1)
    for new_node in range(m + 1, n):
        probs = degrees[:new_node] / degrees[:new_node].sum()
        targets = rng.choice(new_node, size=m, replace=False, p=probs)
        for t in targets:
            adj[new_node, t] = adj[t, new_node] = 1.0
        degrees = np.sum(adj, axis=1)
    return adj


def fiedler_value(adj: np.ndarray) -> float:
    deg = np.diag(adj.sum(axis=1))
    L = deg - adj
    w = eigh(L, eigvals_only=True, subset_by_index=[1, 1])
    return float(w[0])


def avg_clustering(adj: np.ndarray) -> float:
    """Mean local clustering coefficient."""
    n = adj.shape[0]
    deg = adj.sum(axis=1)
    coeffs = []
    for i in range(n):
        neigh = np.where(adj[i] > 0)[0]
        k = len(neigh)
        if k < 2:
            coeffs.append(0.0)
            continue
        sub = adj[np.ix_(neigh, neigh)]
        triangles = sub.sum() / 2.0
        coeffs.append(2.0 * triangles / (k * (k - 1)))
    return float(np.mean(coeffs))


def diameter(adj: np.ndarray) -> int:
    """BFS-based unweighted diameter."""
    n = adj.shape[0]
    neighbours = [np.where(adj[i] > 0)[0] for i in range(n)]
    best = 0
    for src in range(n):
        dist = np.full(n, -1, dtype=int)
        dist[src] = 0
        queue = [src]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in neighbours[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        best = max(best, int(dist.max()))
    return best


def main() -> int:
    # ── Load existing FULL synchrony values ───────────────────────────────
    with CSV_IN.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    sync_by_seed: dict[int, float] = {}
    lz_by_seed: dict[int, float] = {}
    for r in rows:
        if r["regime"] == "ENDOGENOUS" and r["ablation"] == "FULL":
            sync_by_seed[int(r["seed"])] = float(r["synchrony"])
            lz_by_seed[int(r["seed"])] = float(r["lz_tail"])

    # ── Compute topology metrics per seed ────────────────────────────────
    records: list[dict] = []
    for s in SEEDS:
        adj = make_ba_adjacency(N_NODES, BA_M, s)
        l2 = fiedler_value(adj)
        maxd = int(adj.sum(axis=1).max())
        cl = avg_clustering(adj)
        dia = diameter(adj)
        rec = {
            "seed":     s,
            "lambda2":  l2,
            "max_deg":  maxd,
            "avg_cl":   cl,
            "diameter": dia,
            "sync":     sync_by_seed[s],
            "lz_tail":  lz_by_seed[s],
        }
        records.append(rec)
        print(f"seed={s}  λ₂={l2:.4f}  max_deg={maxd:3d}  avg_cl={cl:.3f}  "
              f"diam={dia}  sync={rec['sync']:+.3f}  LZ={rec['lz_tail']:.3f}")

    # ── Correlation analysis ──────────────────────────────────────────────
    METRICS = [
        ("lambda2",  "λ₂ (algebraic connectivity)"),
        ("max_deg",  "Max degree"),
        ("avg_cl",   "Avg clustering coefficient"),
        ("diameter", "Graph diameter"),
    ]
    sync_vals = np.array([r["sync"] for r in records])
    print("\n=== Regression of ENDOGENOUS FULL sync against topology ===")
    print(f"{'metric':<25}  {'Pearson r':>10}  {'p':>9}  {'Spearman ρ':>11}  {'p':>9}")
    corr_results = {}
    for key, label in METRICS:
        vals = np.array([r[key] for r in records])
        pearson_r, pearson_p = stats.pearsonr(vals, sync_vals)
        spear_r, spear_p = stats.spearmanr(vals, sync_vals)
        corr_results[key] = (pearson_r, pearson_p, spear_r, spear_p)
        print(f"{label:<25}  {pearson_r:+10.3f}  {pearson_p:9.2e}  "
              f"{spear_r:+11.3f}  {spear_p:9.2e}")

    # Also test bimodality: Hartigan's dip test proxy via Silverman heuristic
    # (count modes via histogram inspection for small N)
    n_high = sum(1 for v in sync_vals if v > 0.15)
    n_low = sum(1 for v in sync_vals if v < 0.1)
    print(f"\nBimodality descriptors: {n_high} seeds with sync>0.15, "
          f"{n_low} seeds with sync<0.10 (out of {len(sync_vals)})")

    # ── CSV ───────────────────────────────────────────────────────────────
    with CSV_OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)
    print(f"[csv] {CSV_OUT}")

    # ── Figure: 4 scatter panels ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    colours = ["tab:red" if v > 0.15 else "tab:blue" for v in sync_vals]
    for ax, (key, label) in zip(axes, METRICS):
        vals = np.array([r[key] for r in records])
        ax.scatter(vals, sync_vals, c=colours, s=70, edgecolor="k", zorder=3)
        for r in records:
            ax.annotate(str(r["seed"]), (r[key], r["sync"]),
                        fontsize=8, xytext=(4, 4), textcoords="offset points")
        pr, pp, sr, sp = corr_results[key]
        ax.set_xlabel(label)
        ax.set_title(f"Pearson r={pr:+.2f} (p={pp:.2e})\n"
                     f"Spearman ρ={sr:+.2f} (p={sp:.2e})",
                     fontsize=9)
        ax.axhline(0.15, ls=":", color="grey", lw=0.8, label="bimodality threshold")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("ENDOGENOUS FULL synchrony")

    fig.suptitle(
        "Piste A — Does graph topology predict which attractor "
        "ENDOGENOUS FULL settles into?\n"
        f"(BA m={BA_M}, N={N_NODES}, {len(SEEDS)} seeds; "
        "red = 'coordinated' mode, blue = 'desynchronised' mode)",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(FIG_OUT, dpi=140)
    print(f"[png] {FIG_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
