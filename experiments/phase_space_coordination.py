#!/usr/bin/env python3
"""
Piste B — 2D phase diagram of coordination (synchrony × LZ complexity).

Motivation: §3novedecies showed that 4 ablations × 2 regimes produce 8 distinct
(synchrony, LZ) clusters that naturally occupy 4 quadrants of the plane.
This figure visualises the full coordination phase space in a single shot.

Quadrant interpretation:
  (sync ≈ 0, LZ low)  : independent structured walkers     → COGNITIVE DIVERSITY
  (sync > 0, LZ low)  : coordinated structured trajectories → STRUCTURED CONSENSUS
  (sync > 0, LZ high) : synchronised chaotic walks          → COHERENT CHAOS
  (sync ≈ 0, LZ high) : independent random walks            → PURE DISORDER

This figure makes a single scientific claim: only FULL_ENDOGENOUS and
FULL_FORCED sit in the low-LZ half of the plane. Every ablation moves the
system into either coherent chaos (FROZEN_U) or random-walker territory.

Output:
  figures/coordination_phase_space.png
  figures/coordination_phase_centroids.csv

Created: 2026-04-21 (P1.5bis piste B).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Windows cp1252 console fix for λ/ρ glyphs
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
CSV_IN = ROOT / "figures" / "ablation_coordination.csv"
FIG_OUT = ROOT / "figures" / "coordination_phase_space.png"
CSV_OUT = ROOT / "figures" / "coordination_phase_centroids.csv"

ABLATION_STYLE = {
    "FULL":       {"colour": "#2ca02c", "marker": "o"},
    "NO_HERETIC": {"colour": "#d62728", "marker": "s"},
    "NO_SIGMOID": {"colour": "#ff7f0e", "marker": "^"},
    "FROZEN_U":   {"colour": "#9467bd", "marker": "D"},
}
REGIME_ALPHA = {"ENDOGENOUS": 0.85, "FORCED": 0.85}
REGIME_EDGE = {"ENDOGENOUS": "black", "FORCED": "white"}


def main() -> int:
    with CSV_IN.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Group by (regime, ablation)
    groups: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for r in rows:
        key = (r["regime"], r["ablation"])
        groups.setdefault(key, []).append(
            (float(r["synchrony"]), float(r["lz_tail"]))
        )

    # ── Centroids CSV ─────────────────────────────────────────────────────
    centroids: list[dict] = []
    for (regime, ablation), pts in groups.items():
        arr = np.array(pts)
        centroids.append({
            "regime":   regime,
            "ablation": ablation,
            "n":        len(arr),
            "sync_mean": arr[:, 0].mean(),
            "sync_std":  arr[:, 0].std(ddof=1),
            "lz_mean":   arr[:, 1].mean(),
            "lz_std":    arr[:, 1].std(ddof=1),
        })
    with CSV_OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(centroids[0].keys()))
        w.writeheader()
        w.writerows(centroids)
    print(f"[csv] {CSV_OUT}")
    for c in centroids:
        print(f"  {c['regime']:<10} {c['ablation']:<11}  "
              f"sync={c['sync_mean']:+.3f}±{c['sync_std']:.3f}  "
              f"LZ={c['lz_mean']:.3f}±{c['lz_std']:.3f}")

    # ── Figure ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 8))

    # Quadrant background shading (quadrants defined by median of all points)
    all_sync = np.array([s for pts in groups.values() for s, _ in pts])
    all_lz = np.array([lz for pts in groups.values() for _, lz in pts])
    sync_split = 0.15   # visually motivated (cf. bimodality analysis piste A)
    lz_split = 1.6      # separates structured (<1.6) from chaotic (>1.6)

    ax.axvline(sync_split, ls="--", color="grey", alpha=0.5, lw=1)
    ax.axhline(lz_split, ls="--", color="grey", alpha=0.5, lw=1)

    # Quadrant labels
    quadrants = [
        (0.02, 1.85, "COHERENT CHAOS\n(synchronised random walks)",
         "#fdd0a2"),
        (0.02, 1.10, "COGNITIVE DIVERSITY\n(independent structured walkers)",
         "#c7e9c0"),
        (0.40, 1.85, "COORDINATED CHAOS\n(sync chaotic oscillation)",
         "#fcbba1"),
        (0.40, 1.10, "STRUCTURED CONSENSUS\n(coordinated structured)",
         "#bcbddc"),
    ]
    for x, y, txt, bg in quadrants:
        ax.text(x, y, txt, ha="left", va="center", fontsize=9,
                color="#333", alpha=0.85,
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor=bg, alpha=0.45, edgecolor="none"))

    # Plot all seed-level points
    for (regime, ablation), pts in groups.items():
        arr = np.array(pts)
        style = ABLATION_STYLE[ablation]
        alpha = REGIME_ALPHA[regime]
        edge = REGIME_EDGE[regime]
        ax.scatter(
            arr[:, 0], arr[:, 1],
            c=style["colour"],
            marker=style["marker"],
            s=80, alpha=alpha,
            edgecolor=edge, linewidth=1.2,
            zorder=3,
        )
        # Centroid crosshair
        sx, sy = arr[:, 0].mean(), arr[:, 1].mean()
        ax.scatter(
            sx, sy,
            c=style["colour"], marker=style["marker"],
            s=260, alpha=1.0,
            edgecolor="black", linewidth=2.0,
            zorder=4,
        )
        # Label on centroid
        tag = f"{ablation}\n({regime[:3]})"
        ax.annotate(
            tag, (sx, sy),
            xytext=(8, 8), textcoords="offset points",
            fontsize=7.5, fontweight="bold",
            color=style["colour"],
        )

    # Legend: ablation colour + regime shape
    from matplotlib.lines import Line2D
    handles = []
    for k, s in ABLATION_STYLE.items():
        handles.append(Line2D([0], [0], marker=s["marker"],
                              color="w", markerfacecolor=s["colour"],
                              markeredgecolor="k", markersize=9, label=k))
    handles.append(Line2D([0], [0], marker="o", color="w",
                          markerfacecolor="grey",
                          markeredgecolor="black", markersize=9,
                          label="ENDOGENOUS (dark edge)"))
    handles.append(Line2D([0], [0], marker="o", color="w",
                          markerfacecolor="grey",
                          markeredgecolor="white", markersize=9,
                          label="FORCED (white edge)"))
    ax.legend(handles=handles, loc="lower right", fontsize=8,
              framealpha=0.9)

    ax.set_xlabel("Pairwise synchrony (mean Pearson r)", fontsize=11)
    ax.set_ylabel("Normalised LZ76 complexity (stable tail)", fontsize=11)
    ax.set_title(
        "Coordination phase space — only FULL occupies the structured "
        "(low-LZ) half-plane\n"
        "(BA m=3, N=100, 10 seeds per cell, large markers = centroids)",
        fontsize=11,
    )
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.05, 0.95)
    ax.set_ylim(1.0, 2.15)
    plt.tight_layout()
    plt.savefig(FIG_OUT, dpi=140)
    print(f"[png] {FIG_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
