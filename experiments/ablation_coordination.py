#!/usr/bin/env python3
"""
Ablation study with trajectory-based coordination metrics (P1.5bis).

Motivation (PROJECT_STATUS §3octdecies):
  H_100 and H_cog5 are spatial snapshot metrics. They confound "random
  disorder" with "structured cognitive diversity": in the FORCED regime,
  ablating u or heretics *increases* spatial entropy, yet destroys the
  coordinated structure that Mem4ristor is designed to produce.

  Two complementary trajectory metrics resolve this ambiguity:

  1. pairwise_synchrony  — mean Pearson correlation of v(t) across node
     pairs.  High = nodes co-evolve (coordinated); near 0 = independent.

  2. temporal_lz_complexity — normalised LZ76 complexity of each node's
     discretised state sequence, averaged across nodes.  Low = structured /
     predictable trajectories; near 1 = random walk.

  The hypothesis: FULL should show *higher synchrony* and *lower LZ
  complexity* than ablations, even in regimes where its spatial entropy
  appears lower.

Setup (mirrors ablation_minimality.py, §3octdecies):
  - BA m=3, N=100, degree_linear, HERETIC_RATIO=0.15
  - 2 protocols: ENDOGENOUS (I_stim=0) and FORCED (I_stim=0.5)
  - 4 ablations: FULL, NO_HERETIC, NO_SIGMOID, FROZEN_U
  - 10 seeds per cell
  - Full v(t) trace stored at TRACE_STRIDE=10 → (300, 100) array per run

Outputs:
  figures/ablation_coordination.png   (2 rows × 2 cols metric panels)
  figures/ablation_coordination.csv   (raw results, 80 rows)

Created: 2026-04-21 (Claude Sonnet 4.6, P1.5bis).
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mem4ristor.topology import Mem4Network  # noqa: E402
from mem4ristor.metrics import (  # noqa: E402
    calculate_pairwise_synchrony,
    calculate_temporal_lz_complexity,
)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

N_NODES = 100
BA_M = 3
STEPS = 3000
TRACE_STRIDE = 10          # record every 10 steps → T = 300 snapshots
TAIL_FRAC = 0.25           # use last 25% for stable metrics
SEEDS = list(range(10))
HERETIC_RATIO = 0.15

STIMULI = [("ENDOGENOUS", 0.0), ("FORCED", 0.5)]
ABLATIONS = [
    ("FULL",       "Full Mem4ristor"),
    ("NO_HERETIC", "Heretic flip disabled"),
    ("NO_SIGMOID", "Levitating Sigmoid → 1.0"),
    ("FROZEN_U",   "u frozen at σ_baseline"),
]

FIG_PATH = ROOT / "figures" / "ablation_coordination.png"
CSV_PATH = ROOT / "figures" / "ablation_coordination.csv"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers (identical to ablation_minimality.py)
# ──────────────────────────────────────────────────────────────────────────────

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


def apply_ablation(net: Mem4Network, ablation: str) -> None:
    """Monkey-patch the underlying Mem4ristorV3 model for a given ablation."""
    model = net.model
    if ablation == "FULL":
        return
    if ablation == "NO_HERETIC":
        model.heretic_mask = np.zeros(model.N, dtype=bool)
    elif ablation == "NO_SIGMOID":
        model.sigmoid_steepness = 0.0
        model.social_leakage = 1.0
    elif ablation == "FROZEN_U":
        sigma_baseline = model.cfg['doubt'].get('sigma_baseline', 0.05)
        model.cfg['doubt']['epsilon_u'] = 0.0
        model.cfg['doubt']['tau_u'] = 1e12
        model.u = np.full(model.N, sigma_baseline)
    else:
        raise ValueError(f"Unknown ablation: {ablation!r}")


# ──────────────────────────────────────────────────────────────────────────────
# Core runner
# ──────────────────────────────────────────────────────────────────────────────

def run_one(ablation: str, seed: int, stimulus: float) -> dict[str, float]:
    """
    Run one cell and return a dict with synchrony and LZ complexity,
    computed on the full recorded trace.
    """
    adj = make_ba_adjacency(N_NODES, BA_M, seed)
    net = Mem4Network(
        adjacency_matrix=adj,
        heretic_ratio=HERETIC_RATIO,
        coupling_norm='degree_linear',
        seed=seed,
    )
    apply_ablation(net, ablation)

    snapshots: list[np.ndarray] = []
    for step in range(STEPS):
        net.step(I_stimulus=stimulus)
        if step % TRACE_STRIDE == 0:
            snapshots.append(net.model.v.copy())

    v_history = np.array(snapshots)  # (T, N)

    # Tail window for "stable" metrics
    cut = int(len(snapshots) * (1.0 - TAIL_FRAC))
    v_tail = v_history[cut:]

    return {
        "synchrony":   calculate_pairwise_synchrony(v_tail),
        "lz_full":     calculate_temporal_lz_complexity(v_history),
        "lz_tail":     calculate_temporal_lz_complexity(v_tail),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    METRICS = ["synchrony", "lz_full", "lz_tail"]

    # results[regime][metric][ablation] → list[float] across seeds
    results: dict = {
        regime: {m: {k: [] for k, _ in ABLATIONS} for m in METRICS}
        for regime, _ in STIMULI
    }
    rows: list[dict] = []

    for regime, I_stim in STIMULI:
        print(f"\n=== Regime: {regime} (I_stim={I_stim}) ===")
        for ablation, label in ABLATIONS:
            for seed in SEEDS:
                t_run = time.time()
                r = run_one(ablation, seed, I_stim)
                dt = time.time() - t_run
                print(
                    f"[{regime}] [{ablation:<11}] seed={seed:<2}  "
                    f"sync={r['synchrony']:+.3f}  "
                    f"lz_full={r['lz_full']:.3f}  "
                    f"lz_tail={r['lz_tail']:.3f}  ({dt:.1f}s)"
                )
                for m in METRICS:
                    results[regime][m][ablation].append(r[m])
                rows.append({
                    "regime":    regime,
                    "I_stim":    I_stim,
                    "ablation":  ablation,
                    "label":     label,
                    "seed":      seed,
                    "synchrony": r["synchrony"],
                    "lz_full":   r["lz_full"],
                    "lz_tail":   r["lz_tail"],
                })

    # ── CSV ───────────────────────────────────────────────────────────────
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["regime", "I_stim", "ablation", "label", "seed",
                        "synchrony", "lz_full", "lz_tail"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[csv] {CSV_PATH}")

    # ── Summary statistics & Welch tests ─────────────────────────────────
    METRIC_LABELS = {
        "synchrony": "Pairwise Synchrony (Pearson r)",
        "lz_full":   "LZ76 Complexity — full trace",
        "lz_tail":   "LZ76 Complexity — stable tail",
    }
    for regime, _ in STIMULI:
        for m, mname in METRIC_LABELS.items():
            full = np.array(results[regime][m]["FULL"])
            lines = [f"\n--- {regime} / {mname} ---",
                     f"{'ablation':<11}  mean ± sem     d vs FULL   p (Welch)",
                     "-" * 56]
            for ablation, _ in ABLATIONS:
                arr = np.array(results[regime][m][ablation])
                mean = arr.mean()
                sem = arr.std(ddof=1) / np.sqrt(len(arr))
                if ablation == "FULL":
                    lines.append(
                        f"{ablation:<11}  {mean:+.4f} ± {sem:.4f}  ---         ---"
                    )
                else:
                    s_pool = np.sqrt(
                        ((len(full) - 1) * full.var(ddof=1)
                         + (len(arr) - 1) * arr.var(ddof=1))
                        / (len(full) + len(arr) - 2)
                    )
                    d = (full.mean() - arr.mean()) / max(s_pool, 1e-12)
                    _, p = stats.ttest_ind(full, arr, equal_var=False)
                    lines.append(
                        f"{ablation:<11}  {mean:+.4f} ± {sem:.4f}  "
                        f"{d:+.2f}       {p:.2e}"
                    )
            print("\n".join(lines))

    # ── Figure ────────────────────────────────────────────────────────────
    # 2 rows (regimes) × 2 cols (synchrony, lz_full)
    PLOT_METRICS = [
        ("synchrony", "Pairwise Synchrony\n(higher = more coordinated)", None),
        ("lz_full",   "LZ76 Complexity — full trace\n(lower = more structured)", None),
    ]
    colors = ["#2ca02c", "#d62728", "#ff7f0e", "#9467bd"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    abl_keys = [k for k, _ in ABLATIONS]

    for row, (regime, I_stim) in enumerate(STIMULI):
        for col, (m, mname, yrange) in enumerate(PLOT_METRICS):
            ax = axes[row, col]
            xs = np.arange(len(ABLATIONS))
            means = [np.mean(results[regime][m][k]) for k in abl_keys]
            sems = [
                np.std(results[regime][m][k], ddof=1) / np.sqrt(len(SEEDS))
                for k in abl_keys
            ]
            bars = ax.bar(xs, means, yerr=sems, capsize=6, color=colors,
                          edgecolor="k", linewidth=0.8, alpha=0.85)
            rng_jit = np.random.RandomState(42)
            for i, k in enumerate(abl_keys):
                jitter = rng_jit.uniform(-0.1, 0.1, len(SEEDS))
                ax.scatter(
                    np.full(len(SEEDS), i) + jitter,
                    results[regime][m][k],
                    color="k", s=12, alpha=0.6, zorder=5,
                )
            ax.set_xticks(xs)
            ax.set_xticklabels(abl_keys, rotation=10)
            ax.set_title(
                f"{regime} (I_stim={I_stim})\n{mname}", fontsize=9
            )
            ax.axhline(means[0], ls="--", color="#2ca02c", alpha=0.5, lw=1,
                       label=f"FULL={means[0]:+.3f}")
            ax.grid(axis="y", alpha=0.3)
            ax.legend(fontsize=8)
            for b, mean, sem in zip(bars, means, sems):
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    mean + (sem + 0.01 if mean >= 0 else -(sem + 0.01)),
                    f"{mean:+.3f}", ha="center", va="bottom", fontsize=7,
                )

    axes[0, 0].set_ylabel("Pearson r")
    axes[1, 0].set_ylabel("Pearson r")
    axes[0, 1].set_ylabel("Normalised LZ76")
    axes[1, 1].set_ylabel("Normalised LZ76")

    fig.suptitle(
        "Trajectory-based coordination metrics across ablations\n"
        f"(BA m={BA_M}, N={N_NODES}, degree_linear, {STEPS} steps, n={len(SEEDS)} seeds)",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=140)
    print(f"[png] {FIG_PATH}")
    print(f"\nTotal wall time: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
