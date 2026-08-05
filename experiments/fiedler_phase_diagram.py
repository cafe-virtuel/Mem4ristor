#!/usr/bin/env python3
"""
lambda2 vs H_stable phase diagram — visualizing the topological transition.

For every (topology, normalization) pair already studied in
limit02_topology_sweep.py, compute:
  - lambda2  : algebraic connectivity (Fiedler value) of the unweighted Laplacian
  - H_stable : entropy averaged over the last 25% of a 3000-step run

Then scatter-plot H_stable as a function of lambda2. The expected pattern is:
  - very low lambda2  → information cannot diffuse → consensus or freeze (low H)
  - moderate lambda2  → diffusion + frustration cohabit → high H (the sweet spot)
  - very high lambda2 → too connected, hubs dominate → collapse to consensus (low H)
This is the "phase transition" plot that summarizes the negative result of
the Paper 1 dead zone.

Created: 2026-04-19 (Claude Opus 4.7, Paper 1 figure track)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from mem4ristor.core import Mem4Network  # noqa: E402

# Reuse generators from existing sweep
from limit02_topology_sweep import (  # noqa: E402
    make_ba,
    make_configuration_model,
    make_holme_kim,
    make_watts_strogatz,
    make_erdos_renyi,
    degree_stats,
)

# ── Parameters ──────────────────────────────────────────────────────
N = 100
STEPS = 3000
TAIL_FRAC = 0.25
SEEDS = [42, 123, 777]   # 3 seeds (smaller than topology_sweep for speed)
NORMS = ["uniform", "degree_linear"]

FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)


def fiedler_value(adj: np.ndarray) -> float:
    """Algebraic connectivity lambda2 of the unweighted graph Laplacian."""
    from scipy.linalg import eigh
    deg = adj.sum(axis=1)
    L = np.diag(deg) - adj
    vals = eigh(L, eigvals_only=True)
    return float(vals[1]) if vals.size > 1 else 0.0


def h_stable(adj: np.ndarray, norm: str, seed: int) -> float:
    net = Mem4Network(
        size=10, heretic_ratio=0.15, seed=seed,
        adjacency_matrix=adj.copy(), coupling_norm=norm,
    )
    trace = []
    for step in range(STEPS):
        net.step(I_stimulus=0.0)
        if step % 10 == 0:
            trace.append(net.calculate_entropy())
    tail = int(len(trace) * (1 - TAIL_FRAC))
    return float(np.mean(trace[tail:]))


# (label, generator, kwargs, family)
TOPOLOGIES = [
    ("BA m=1",                make_ba,                  {"m": 1},   "BA"),
    ("BA m=2",                make_ba,                  {"m": 2},   "BA"),
    ("BA m=3",                make_ba,                  {"m": 3},   "BA"),
    ("BA m=5",                make_ba,                  {"m": 5},   "BA"),
    ("BA m=8",                make_ba,                  {"m": 8},   "BA"),
    ("BA m=10",               make_ba,                  {"m": 10},  "BA"),
    ("CM gamma=2.5",          make_configuration_model, {"gamma": 2.5}, "CM"),
    ("CM gamma=3.0",          make_configuration_model, {"gamma": 3.0}, "CM"),
    ("CM gamma=4.0",          make_configuration_model, {"gamma": 4.0}, "CM"),
    ("HK m=3 p=0.5",          make_holme_kim,           {"m": 3, "p_tri": 0.5}, "HK"),
    ("HK m=3 p=0.9",          make_holme_kim,           {"m": 3, "p_tri": 0.9}, "HK"),
    ("WS k=4 p=0.1",          make_watts_strogatz,      {"k": 4, "p": 0.1}, "WS"),
    ("WS k=4 p=0.3",          make_watts_strogatz,      {"k": 4, "p": 0.3}, "WS"),
    ("ER p=0.06",             make_erdos_renyi,         {"p": 0.06}, "ER"),
    ("ER p=0.12",             make_erdos_renyi,         {"p": 0.12}, "ER"),
]

FAMILY_COLORS = {"BA": "C0", "CM": "C1", "HK": "C2", "WS": "C3", "ER": "C4"}
NORM_MARKERS = {"uniform": "o", "degree_linear": "s"}


def main() -> int:
    print("=" * 78)
    print("lambda2 vs H_stable phase diagram")
    print(f"N={N}, steps={STEPS}, seeds={len(SEEDS)}, "
          f"{len(TOPOLOGIES)} topologies × {len(NORMS)} norms = "
          f"{len(TOPOLOGIES) * len(NORMS)} configurations")
    print("=" * 78)

    rows = []
    t0 = time.time()
    for label, gen, kwargs, family in TOPOLOGIES:
        per_seed_l2 = []
        per_seed_h = {n: [] for n in NORMS}
        for seed in SEEDS:
            adj = gen(N, seed=seed, **kwargs)
            l2 = fiedler_value(adj)
            per_seed_l2.append(l2)
            for n in NORMS:
                per_seed_h[n].append(h_stable(adj, n, seed))
        l2_mean = float(np.mean(per_seed_l2))
        l2_std = float(np.std(per_seed_l2))
        for n in NORMS:
            h_mean = float(np.mean(per_seed_h[n]))
            h_std = float(np.std(per_seed_h[n]))
            rows.append({
                "label": label, "family": family, "norm": n,
                "lambda2_mean": l2_mean, "lambda2_std": l2_std,
                "H_mean": h_mean, "H_std": h_std,
            })
            print(f"  {label:<22} norm={n:<14} lambda2={l2_mean:6.3f}+-{l2_std:5.3f}  "
                  f"H={h_mean:6.3f}+-{h_std:5.3f}")

    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed:.1f}s")

    # ── Plot ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))

    # Shaded "sweet spot" lambda2 region: empirically [0.05, 5.0] from the data we know
    ax.axvspan(0.05, 5.0, color="gray", alpha=0.08,
               label="empirical sweet spot")

    plotted_legend = set()
    for r in rows:
        color = FAMILY_COLORS[r["family"]]
        marker = NORM_MARKERS[r["norm"]]
        key = (r["family"], r["norm"])
        if key not in plotted_legend:
            label = f"{r['family']} ({r['norm']})"
            plotted_legend.add(key)
        else:
            label = None
        ax.errorbar(
            r["lambda2_mean"], r["H_mean"],
            xerr=r["lambda2_std"], yerr=r["H_std"],
            fmt=marker, color=color, alpha=0.85, markersize=9,
            markeredgecolor="black", markeredgewidth=0.5,
            ecolor=color, elinewidth=1, capsize=3,
            label=label,
        )

    # Annotate a few critical points
    for r in rows:
        if r["norm"] == "degree_linear" and r["H_mean"] > 0.5:
            ax.annotate(
                r["label"], (r["lambda2_mean"], r["H_mean"]),
                xytext=(6, 6), textcoords="offset points",
                fontsize=7, alpha=0.7,
            )

    ax.set_xscale("symlog", linthresh=0.05)
    ax.set_xlabel("Algebraic connectivity lambda2 (Fiedler value)", fontsize=11)
    ax.set_ylabel("H_stable (Shannon entropy, last 25%)", fontsize=11)
    ax.set_title(
        "Topological phase diagram for Mem4ristor v3.2.0\n"
        "(N=100, 3000 steps, 3 seeds per point)",
        fontsize=12,
    )
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3, which="both")
    ax.axhline(0.5, color="black", lw=0.5, ls=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.85)

    fig.tight_layout()
    out = FIGURES / "fiedler_phase_diagram.png"
    fig.savefig(out, dpi=150)
    print(f"\nFigure saved: {out}")

    # Save raw data alongside the figure for reproducibility
    csv_out = FIGURES / "fiedler_phase_diagram.csv"
    with open(csv_out, "w", encoding="utf-8") as f:
        f.write("label,family,norm,lambda2_mean,lambda2_std,H_mean,H_std\n")
        for r in rows:
            f.write(f"{r['label']},{r['family']},{r['norm']},"
                    f"{r['lambda2_mean']:.6f},{r['lambda2_std']:.6f},"
                    f"{r['H_mean']:.6f},{r['H_std']:.6f}\n")
    print(f"Data saved:  {csv_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
