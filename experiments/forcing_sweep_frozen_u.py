#!/usr/bin/env python3
"""
Piste E — Forcing sweep FULL vs FROZEN_U (PROJECT_STATUS §3novedecies-bis piste E).

Motivation:
  §3novedecies showed a striking inversion: FROZEN_U goes from sync=0.006
  (ENDOGENOUS, I_stim=0) to sync=0.751 (FORCED, I_stim=0.5). The forcing
  stimulus *creates* massive synchronisation in absence of the u regulator.

  Hypothesis: u acts as an anti-synchronisation filter.
  - FULL: sync(I_stim) stays flat — u absorbs the common forcing signal and
    prevents it from locking nodes together.
  - FROZEN_U: sync(I_stim) rises monotonically — nodes all integrate the same
    stimulus without a regulator, so they lock.

Protocol (§3novedecies-bis piste E):
  - I_stim ∈ {0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 1.00}
  - FULL vs FROZEN_U only (4-ablation study not needed here)
  - BA m=3, N=100, degree_linear, heretic_ratio=0.15
  - 3000 steps, TRACE_STRIDE=10, 7 seeds per cell (small run)
  - Metrics: pairwise_synchrony + lz_full on stable tail (last 25%)

Expected result (publishable if confirmed):
  - FULL sync ≈ constant low, LZ_full ≈ constant low
  - FROZEN_U sync rises monotonically with I_stim
  → elegant demonstration of u as a stimulus-buffering anti-sync regulator.

Outputs:
  figures/forcing_sweep_frozen_u.png   (2 panels: sync(I) + LZ(I))
  figures/forcing_sweep_frozen_u.csv   (raw data)

Created: 2026-04-24 (Claude Sonnet 4.6, Piste E).
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

from mem4ristor.topology import Mem4Network          # noqa: E402
from mem4ristor.metrics import (                     # noqa: E402
    calculate_pairwise_synchrony,
    calculate_temporal_lz_complexity,
)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

N_NODES = 100
BA_M = 3
STEPS = 3000
TRACE_STRIDE = 10
TAIL_FRAC = 0.25
SEEDS = list(range(7))
HERETIC_RATIO = 0.15

I_STIMS = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 1.00]

MODELS = ["FULL", "FROZEN_U"]
COLORS = {"FULL": "#2ca02c", "FROZEN_U": "#9467bd"}

FIG_PATH = ROOT / "figures" / "forcing_sweep_frozen_u.png"
CSV_PATH = ROOT / "figures" / "forcing_sweep_frozen_u.csv"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
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


def apply_frozen_u(net: Mem4Network) -> None:
    model = net.model
    sigma_baseline = model.cfg['doubt'].get('sigma_baseline', 0.05)
    model.cfg['doubt']['epsilon_u'] = 0.0
    model.cfg['doubt']['tau_u'] = 1e12
    model.u = np.full(model.N, sigma_baseline)


def run_one(model_name: str, I_stim: float, seed: int) -> dict[str, float]:
    adj = make_ba_adjacency(N_NODES, BA_M, seed)
    net = Mem4Network(
        adjacency_matrix=adj,
        heretic_ratio=HERETIC_RATIO,
        coupling_norm='degree_linear',
        seed=seed,
    )
    if model_name == "FROZEN_U":
        apply_frozen_u(net)

    snapshots: list[np.ndarray] = []
    for step in range(STEPS):
        net.step(I_stimulus=I_stim)
        if step % TRACE_STRIDE == 0:
            snapshots.append(net.model.v.copy())

    v_history = np.array(snapshots)
    cut = int(len(snapshots) * (1.0 - TAIL_FRAC))
    v_tail = v_history[cut:]

    return {
        "synchrony": calculate_pairwise_synchrony(v_tail),
        "lz_full":   calculate_temporal_lz_complexity(v_history),
        "lz_tail":   calculate_temporal_lz_complexity(v_tail),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # results[model][I_stim] → list of metric dicts
    results: dict[str, dict[float, dict[str, list[float]]]] = {
        m: {I: {"synchrony": [], "lz_full": [], "lz_tail": []}
            for I in I_STIMS}
        for m in MODELS
    }
    all_rows: list[dict] = []

    for model_name in MODELS:
        print(f"\n=== Model: {model_name} ===")
        for I_stim in I_STIMS:
            for seed in SEEDS:
                t_run = time.time()
                r = run_one(model_name, I_stim, seed)
                dt = time.time() - t_run
                print(
                    f"[{model_name:<8}] I={I_stim:.2f}  seed={seed}  "
                    f"sync={r['synchrony']:+.3f}  "
                    f"lz_full={r['lz_full']:.3f}  "
                    f"lz_tail={r['lz_tail']:.3f}  ({dt:.1f}s)"
                )
                for k in ("synchrony", "lz_full", "lz_tail"):
                    results[model_name][I_stim][k].append(r[k])
                all_rows.append({
                    "model": model_name, "I_stim": I_stim, "seed": seed, **r
                })

    # ── CSV ───────────────────────────────────────────────────────────────
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["model", "I_stim", "seed",
                           "synchrony", "lz_full", "lz_tail"]
        )
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n[csv] {CSV_PATH}")

    # ── Summary table ─────────────────────────────────────────────────────
    for model_name in MODELS:
        print(f"\n--- {model_name} ---")
        print(f"{'I_stim':>7}  {'sync mean±sem':>18}  {'lz_full mean±sem':>20}")
        print("-" * 55)
        for I_stim in I_STIMS:
            s_arr = np.array(results[model_name][I_stim]["synchrony"])
            l_arr = np.array(results[model_name][I_stim]["lz_full"])
            n = len(SEEDS)
            print(
                f"{I_stim:7.2f}  "
                f"{s_arr.mean():+.3f}±{s_arr.std(ddof=1)/np.sqrt(n):.3f}  "
                f"{l_arr.mean():.3f}±{l_arr.std(ddof=1)/np.sqrt(n):.3f}"
            )

    # ── Pearson monotonicity tests ────────────────────────────────────────
    print("\n--- Monotonicity (Pearson I_stim vs metric) ---")
    I_arr = np.array(I_STIMS)
    for model_name in MODELS:
        sync_means = np.array([
            np.mean(results[model_name][I]["synchrony"]) for I in I_STIMS
        ])
        lz_means = np.array([
            np.mean(results[model_name][I]["lz_full"]) for I in I_STIMS
        ])
        r_s, p_s = stats.pearsonr(I_arr, sync_means)
        r_l, p_l = stats.pearsonr(I_arr, lz_means)
        print(f"{model_name}: sync r={r_s:+.3f} p={p_s:.2e}  |  lz_full r={r_l:+.3f} p={p_l:.2e}")

    # ── Welch test at I_stim=0.50: FULL vs FROZEN_U sync ─────────────────
    I_key = 0.50
    s_full  = np.array(results["FULL"][I_key]["synchrony"])
    s_froz  = np.array(results["FROZEN_U"][I_key]["synchrony"])
    s_pool = np.sqrt(
        ((len(s_full)-1)*s_full.var(ddof=1) + (len(s_froz)-1)*s_froz.var(ddof=1))
        / (len(s_full) + len(s_froz) - 2)
    )
    d_05 = (s_froz.mean() - s_full.mean()) / max(s_pool, 1e-12)
    _, p_05 = stats.ttest_ind(s_full, s_froz, equal_var=False)
    print(f"\nWelch FULL vs FROZEN_U at I_stim={I_key}: "
          f"FULL={s_full.mean():+.3f}, FROZEN_U={s_froz.mean():+.3f}, "
          f"Cohen's d={d_05:.2f}, p={p_05:.2e}")

    # ── Figure ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    metrics_to_plot = [
        ("synchrony", "Pairwise Synchrony (Pearson r)\n(higher = more locked together)"),
        ("lz_full",   "LZ76 Complexity — full trace\n(lower = more structured)"),
    ]

    for col, (metric, ylabel) in enumerate(metrics_to_plot):
        ax = axes[col]
        for model_name in MODELS:
            means = np.array([
                np.mean(results[model_name][I][metric]) for I in I_STIMS
            ])
            sems = np.array([
                np.std(results[model_name][I][metric], ddof=1) / np.sqrt(len(SEEDS))
                for I in I_STIMS
            ])
            ax.errorbar(
                I_arr, means, yerr=sems,
                fmt='o-' if model_name == "FULL" else 's--',
                color=COLORS[model_name], capsize=4,
                linewidth=2, markersize=7, label=model_name,
            )
            # individual seeds
            rng = np.random.RandomState(42)
            for I_stim in I_STIMS:
                vals = results[model_name][I_stim][metric]
                jitter = rng.uniform(-0.008, 0.008, len(vals))
                ax.scatter(
                    np.full(len(vals), I_stim) + jitter, vals,
                    color=COLORS[model_name], s=14, alpha=0.35, zorder=5,
                )
        ax.set_xlabel("I_stimulus", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(
            f"{'Synchrony' if col==0 else 'LZ complexity'} vs forcing strength\n"
            f"(BA m={BA_M}, N={N_NODES}, heretic_ratio={HERETIC_RATIO}, "
            f"n={len(SEEDS)} seeds)",
            fontsize=9,
        )
        ax.grid(alpha=0.3)
        ax.legend(fontsize=10)

    # Annotate sync panel with key transition
    axes[0].annotate(
        f"FULL vs FROZEN_U\nat I=0.5:\nd={d_05:.1f}, p={p_05:.1e}",
        xy=(0.50, s_froz.mean()), xytext=(0.55, s_froz.mean() - 0.15),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="grey"),
    )

    fig.suptitle(
        "Piste E — Does u act as an anti-synchronisation filter?\n"
        "Hypothesis: FULL sync stays flat; FROZEN_U sync rises with I_stim",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=140)
    print(f"[png] {FIG_PATH}")
    print(f"\nTotal wall time: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
