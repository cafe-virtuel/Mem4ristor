#!/usr/bin/env python3
"""
Ablation / Minimality Study for Mem4ristor v3.2.0 (KIMI critique #4).

Goal: demonstrate that the three core ingredients of the Mem4ristor kernel
are each individually necessary to sustain cognitive diversity on a
Barabási-Albert m=3 graph --- i.e. none of them can be removed without a
statistically significant entropy loss.

Ablations:
  FULL      : baseline Mem4ristor V3 (heretics + Levitating Sigmoid + u dynamics)
  NO_HERETIC: heretic_ratio = 0    (polarity flip disabled)
  NO_SIGMOID: u_filter      = 1.0  (Levitating Sigmoid replaced by a constant
                                    attractive kernel, as in classical diffusive
                                    FHN coupling)
  FROZEN_U  : u_i freezed at sigma_baseline  (metacognitive dynamics disabled)

Protocol:
  - BA m=3, N=100, per-seed graph instance
  - degree_linear coupling normalization (the reference setting for BA,
    cf. PROJECT_STATUS §3quinquies / §3quinquiesbis)
  - 3000 steps, I_stimulus = 0.0 (zero forcing, pure endogenous regime)
  - H_stable = mean of last 25% of H trace (sampled every 10 steps)
  - 10 seeds per ablation (N_SEEDS=10)
  - Metric: 100-bin continuous spectral entropy (current default)

Expected results:
  FULL        H_stable ~ 0.83 ± 0.07 (cf. regression test)
  NO_HERETIC  H_stable -> low (consensus collapse, Fragility Law)
  NO_SIGMOID  H_stable -> intermediate (classical attractive coupling is
              known to destroy polarity diversity in scale-free hubs)
  FROZEN_U    H_stable -> intermediate/low (u-driven polarity switch lost)

Outputs:
  figures/ablation_minimality.png   (bar + seed scatter with CI)
  figures/ablation_minimality.csv   (raw 40 rows)

Created: 2026-04-20 (Claude Opus 4.7, KIMI response track --- part C).
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


N_NODES = 100
BA_M = 3
STEPS = 3000
TRACE_STRIDE = 10
TAIL_FRAC = 0.25
SEEDS = list(range(10))
HERETIC_RATIO = 0.15

# We test two regimes side-by-side:
#   ENDOGENOUS: I_stim = 0.0 (the BA m=3 reference point, PROJECT_STATUS §3quinquies)
#   FORCED   : I_stim = 0.5 (the default lattice protocol, test_scientific_regression)
# This matters: the heretic polarity flip is `I_eff[mask] *= -1.0`, which is
# a no-op when I_stim = 0. So a full ablation picture requires both regimes.
STIMULI = [("ENDOGENOUS", 0.0), ("FORCED", 0.5)]

ABLATIONS = [
    ("FULL",       "Full Mem4ristor"),
    ("NO_HERETIC", "Heretic flip disabled"),
    ("NO_SIGMOID", "Levitating Sigmoid -> 1.0"),
    ("FROZEN_U",   "u frozen at sigma_baseline"),
]

FIG_PATH = ROOT / "figures" / "ablation_minimality.png"
CSV_PATH = ROOT / "figures" / "ablation_minimality.csv"


def make_ba_adjacency(n: int, m: int, seed: int) -> np.ndarray:
    """BA graph generator (matches spice_mismatch_50seeds.py / limit02_norm_sweep)."""
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
        # Zero the heretic mask. The kernel line
        #   I_eff[self.heretic_mask] *= -1.0
        # becomes a no-op: coupling stays classically attractive.
        model.heretic_mask = np.zeros(model.N, dtype=bool)
        return

    if ablation == "NO_SIGMOID":
        # Force the Levitating Sigmoid filter to a constant attractive +1.
        # We achieve this WITHOUT replacing step() by setting
        #   sigmoid_steepness = 0    -> tanh(0) = 0
        #   social_leakage    = 1.0  -> u_filter = 0 + 1 = 1
        # This reproduces classical diffusive FHN coupling exactly
        # (I_coup = D_eff * 1.0 * laplacian_v).
        model.sigmoid_steepness = 0.0
        model.social_leakage = 1.0
        return

    if ablation == "FROZEN_U":
        # Disable doubt dynamics: u stays at its init value.
        # Strategy: set tau_u very large AND epsilon_u = 0, so du ~ 0.
        # Init u at sigma_baseline (the natural attractor of eq. du/dt=0
        # when sigma_social = 0, modulo clipping).
        sigma_baseline = model.cfg['doubt'].get('sigma_baseline', 0.05)
        model.cfg['doubt']['epsilon_u'] = 0.0
        model.cfg['doubt']['tau_u'] = 1e12
        model.u = np.full(model.N, sigma_baseline)
        return

    raise ValueError(f"Unknown ablation: {ablation!r}")


def run_one(ablation: str, seed: int, stimulus: float) -> tuple[float, float]:
    """Return (H_100, H_cog5) stable averages."""
    adj = make_ba_adjacency(N_NODES, BA_M, seed)
    net = Mem4Network(
        adjacency_matrix=adj,
        heretic_ratio=HERETIC_RATIO,   # 0.15 full; we zero the mask manually for NO_HERETIC
        coupling_norm='degree_linear',
        seed=seed,
    )
    apply_ablation(net, ablation)

    trace_100, trace_cog = [], []
    for step in range(STEPS):
        net.step(I_stimulus=stimulus)
        if step % TRACE_STRIDE == 0:
            trace_100.append(net.calculate_entropy())
            trace_cog.append(net.calculate_entropy(use_cognitive_bins=True))

    cut = int(len(trace_100) * (1.0 - TAIL_FRAC))
    return float(np.mean(trace_100[cut:])), float(np.mean(trace_cog[cut:]))


def main() -> int:
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    # results[regime][metric][ablation] -> list of H per seed
    results: dict[str, dict[str, dict[str, list[float]]]] = {
        regime: {"H100": {k: [] for k, _ in ABLATIONS},
                 "Hcog": {k: [] for k, _ in ABLATIONS}}
        for regime, _ in STIMULI
    }

    rows = []
    for regime, I_stim in STIMULI:
        print(f"\n=== Regime: {regime} (I_stim = {I_stim}) ===")
        for ablation, label in ABLATIONS:
            for seed in SEEDS:
                t_run = time.time()
                h100, hcog = run_one(ablation, seed, I_stim)
                dt = time.time() - t_run
                print(f"[{regime}] [{ablation:<11}] seed={seed:<2}  "
                      f"H100={h100:.3f}  Hcog={hcog:.3f}  ({dt:.1f}s)")
                results[regime]["H100"][ablation].append(h100)
                results[regime]["Hcog"][ablation].append(hcog)
                rows.append({
                    "regime": regime,
                    "I_stim": I_stim,
                    "ablation": ablation,
                    "label": label,
                    "seed": seed,
                    "H_100": h100,
                    "H_cog5": hcog,
                })

    # CSV
    with CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["regime", "I_stim", "ablation", "label", "seed",
                           "H_100", "H_cog5"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[csv] wrote {CSV_PATH}")

    # Per-regime summary stats + tests vs FULL (both metrics)
    for regime, _ in STIMULI:
        for metric, mname in [("H100", "H_100 (100-bin continuous)"),
                              ("Hcog", "H_cog5 (5-bin cognitive)")]:
            full = np.asarray(results[regime][metric]["FULL"])
            summary: list[str] = []
            summary.append(f"\n--- {regime} / {mname} ---")
            summary.append(f"{'ablation':<11}  mean ± sem    d vs FULL   p (Welch)")
            summary.append("-" * 55)
            for ablation, _ in ABLATIONS:
                arr = np.asarray(results[regime][metric][ablation])
                mean, sem = arr.mean(), arr.std(ddof=1) / np.sqrt(len(arr))
                if ablation == "FULL":
                    summary.append(
                        f"{ablation:<11}  {mean:.4f} ± {sem:.4f}   ---         ---")
                else:
                    s_pool = np.sqrt(
                        ((len(full) - 1) * full.var(ddof=1)
                         + (len(arr) - 1) * arr.var(ddof=1))
                        / (len(full) + len(arr) - 2))
                    d = (full.mean() - arr.mean()) / max(s_pool, 1e-12)
                    _, p = stats.ttest_ind(full, arr, equal_var=False)
                    summary.append(
                        f"{ablation:<11}  {mean:.4f} ± {sem:.4f}   "
                        f"{d:+.2f}       {p:.2e}")
            print("\n".join(summary))

    # Figure: 2 rows (regimes) x 2 cols (metrics) = 4 panels
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    colors = ["#2ca02c", "#d62728", "#ff7f0e", "#9467bd"]

    for row, (regime, I_stim) in enumerate(STIMULI):
        for col, (metric, mname, ymax_hint) in enumerate([
            ("H100", "H_100 (100-bin continuous)", None),
            ("Hcog", "H_cog5 (5-bin cognitive, ceiling=log2(5))", 2.4),
        ]):
            ax = axes[row, col]
            xs = np.arange(len(ABLATIONS))
            means = [np.mean(results[regime][metric][k]) for k, _ in ABLATIONS]
            sems = [np.std(results[regime][metric][k], ddof=1)
                    / np.sqrt(len(results[regime][metric][k]))
                    for k, _ in ABLATIONS]
            labels = [k for k, _ in ABLATIONS]

            bars = ax.bar(xs, means, yerr=sems, capsize=6, color=colors,
                          edgecolor="k", linewidth=0.8, alpha=0.85)

            for i, (k, _) in enumerate(ABLATIONS):
                jitter = np.random.uniform(-0.1, 0.1, len(results[regime][metric][k]))
                ax.scatter(np.full(len(results[regime][metric][k]), i) + jitter,
                           results[regime][metric][k], color="k", s=12,
                           alpha=0.6, zorder=5)

            ax.set_xticks(xs)
            ax.set_xticklabels(labels, rotation=10)
            ax.set_title(f"{regime} (I_stim = {I_stim}) — {mname}", fontsize=10)
            ax.axhline(means[0], ls="--", color="#2ca02c", alpha=0.6, lw=1,
                       label=f"FULL = {means[0]:.3f}")
            ax.grid(axis="y", alpha=0.3)
            ax.legend(loc="lower left", fontsize=8)
            if ymax_hint is not None:
                ax.set_ylim(0, ymax_hint)

            for b, m, s in zip(bars, means, sems):
                ax.text(b.get_x() + b.get_width() / 2, m + s + 0.02,
                        f"{m:.3f}", ha="center", va="bottom", fontsize=8)

    axes[0, 0].set_ylabel("H (bits)")
    axes[1, 0].set_ylabel("H (bits)")
    fig.suptitle(
        "Minimality ablation under two protocols & two metrics\n"
        f"(BA m=3, N=100, degree_linear, {STEPS} steps, n={len(SEEDS)} seeds)",
        fontsize=12)
    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=140)
    print(f"[png] wrote {FIG_PATH}")
    print(f"\nTotal wall time: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
