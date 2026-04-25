#!/usr/bin/env python3
"""
Piste C — Heretic ratio sweep under forcing (P1.5bis follow-up).

Motivation: §3novedecies showed that NO_HERETIC under FORCED increases LZ
complexity (1.37 → 1.61, d=2.40). Hypothesis: the heretic flip acts as a
*temporal regulariser* that structures individual node trajectories. If
true, the effect should strengthen monotonically with heretic_ratio.

Method:
  - Sweep η ∈ {0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50}
  - BA m=3, N=100, degree_linear, I_stim=0.5, 3000 steps, TRACE_STRIDE=10
  - 5 seeds per η
  - Measure (synchrony, LZ_full, LZ_tail) from full v(t) trace

Expected:
  - Monotone decrease of LZ_tail with η (regulariser hypothesis)
  - If non-monotone → other mechanism (resonance, frustration threshold)

Output:
  figures/heretic_sweep_coordination.png   (2 panels: sync(η) + LZ(η))
  figures/heretic_sweep_coordination.csv   (raw 35 rows)

Created: 2026-04-21 (P1.5bis piste C).
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mem4ristor.topology import Mem4Network  # noqa: E402
from mem4ristor.metrics import (  # noqa: E402
    calculate_pairwise_synchrony,
    calculate_temporal_lz_complexity,
)

N_NODES = 100
BA_M = 3
STEPS = 3000
TRACE_STRIDE = 10
TAIL_FRAC = 0.25
SEEDS = list(range(5))
I_STIM = 0.5
ETAS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]

FIG_OUT = ROOT / "figures" / "heretic_sweep_coordination.png"
CSV_OUT = ROOT / "figures" / "heretic_sweep_coordination.csv"


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


def run_one(eta: float, seed: int) -> dict[str, float]:
    adj = make_ba_adjacency(N_NODES, BA_M, seed)
    net = Mem4Network(
        adjacency_matrix=adj,
        heretic_ratio=eta,
        coupling_norm="degree_linear",
        seed=seed,
    )
    # eta=0 → network auto-sets 0 heretics; no special handling needed

    snapshots: list[np.ndarray] = []
    for step in range(STEPS):
        net.step(I_stimulus=I_STIM)
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


def main() -> int:
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    results: dict[float, dict[str, list[float]]] = {
        eta: {"synchrony": [], "lz_full": [], "lz_tail": []}
        for eta in ETAS
    }
    rows: list[dict] = []

    print(f"=== Heretic ratio sweep under I_stim={I_STIM} ===")
    for eta in ETAS:
        for s in SEEDS:
            t = time.time()
            r = run_one(eta, s)
            dt = time.time() - t
            print(f"η={eta:.2f}  seed={s}  "
                  f"sync={r['synchrony']:+.3f}  "
                  f"lz_full={r['lz_full']:.3f}  "
                  f"lz_tail={r['lz_tail']:.3f}  ({dt:.1f}s)")
            for k in ["synchrony", "lz_full", "lz_tail"]:
                results[eta][k].append(r[k])
            rows.append({"eta": eta, "seed": s, **r})

    # ── CSV ───────────────────────────────────────────────────────────────
    with CSV_OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["eta", "seed", "synchrony",
                                          "lz_full", "lz_tail"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n[csv] {CSV_OUT}")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'η':>6} | {'sync mean±sem':>15} | {'LZ_full mean±sem':>17} | "
          f"{'LZ_tail mean±sem':>17}")
    print("-" * 68)
    summary = []
    for eta in ETAS:
        s_arr = np.array(results[eta]["synchrony"])
        lf_arr = np.array(results[eta]["lz_full"])
        lt_arr = np.array(results[eta]["lz_tail"])
        row = {
            "eta": eta,
            "sync_mean":    s_arr.mean(),
            "sync_sem":     s_arr.std(ddof=1) / np.sqrt(len(s_arr)),
            "lz_full_mean": lf_arr.mean(),
            "lz_full_sem":  lf_arr.std(ddof=1) / np.sqrt(len(lf_arr)),
            "lz_tail_mean": lt_arr.mean(),
            "lz_tail_sem":  lt_arr.std(ddof=1) / np.sqrt(len(lt_arr)),
        }
        summary.append(row)
        print(f"{eta:6.2f} | "
              f"{row['sync_mean']:+.3f} ± {row['sync_sem']:.3f}  | "
              f"{row['lz_full_mean']:.3f} ± {row['lz_full_sem']:.3f}     | "
              f"{row['lz_tail_mean']:.3f} ± {row['lz_tail_sem']:.3f}")

    # Monotonicity check on LZ_tail
    lz_tail_means = [row["lz_tail_mean"] for row in summary]
    diffs = np.diff(lz_tail_means)
    monotone_down = np.all(diffs <= 0.02)  # allow small noise
    monotone_up = np.all(diffs >= -0.02)
    verdict = "MONOTONE DECREASING" if monotone_down else \
              ("MONOTONE INCREASING" if monotone_up else "NON-MONOTONE")
    print(f"\nLZ_tail trend: {verdict}")
    print(f"  Δ from η=0 to η=0.50: {lz_tail_means[-1] - lz_tail_means[0]:+.3f}")

    # ── Figure: 2 panels ─────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    etas_arr = np.array(ETAS)
    sync_means = np.array([r["sync_mean"] for r in summary])
    sync_sems = np.array([r["sync_sem"] for r in summary])
    lz_tail_means_arr = np.array(lz_tail_means)
    lz_tail_sems = np.array([r["lz_tail_sem"] for r in summary])
    lz_full_means = np.array([r["lz_full_mean"] for r in summary])
    lz_full_sems = np.array([r["lz_full_sem"] for r in summary])

    ax1.errorbar(etas_arr, sync_means, yerr=sync_sems,
                 marker="o", color="#1f77b4", capsize=4, lw=1.8)
    # Per-seed scatter
    for row in rows:
        ax1.scatter(row["eta"], row["synchrony"], s=16, color="k",
                    alpha=0.4, zorder=3)
    ax1.axvline(0.15, ls="--", color="grey", alpha=0.6,
                label="preprint default η=0.15")
    ax1.axhline(0, color="k", lw=0.5)
    ax1.set_xlabel("Heretic ratio η")
    ax1.set_ylabel("Pairwise synchrony (Pearson r)")
    ax1.set_title("Synchrony vs heretic ratio (FORCED, I_stim=0.5)")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=9)

    ax2.errorbar(etas_arr, lz_tail_means_arr, yerr=lz_tail_sems,
                 marker="s", color="#d62728", capsize=4, lw=1.8,
                 label="LZ tail (stable)")
    ax2.errorbar(etas_arr, lz_full_means, yerr=lz_full_sems,
                 marker="^", color="#ff7f0e", capsize=4, lw=1.4, ls=":",
                 label="LZ full (with transient)")
    for row in rows:
        ax2.scatter(row["eta"], row["lz_tail"], s=16, color="k",
                    alpha=0.4, zorder=3)
    ax2.axvline(0.15, ls="--", color="grey", alpha=0.6,
                label="preprint default η=0.15")
    ax2.set_xlabel("Heretic ratio η")
    ax2.set_ylabel("Normalised LZ76 complexity")
    ax2.set_title("Trajectory structure vs heretic ratio")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=9)

    fig.suptitle(
        "Piste C — Does heretic flip act as a temporal regulariser?\n"
        f"(BA m={BA_M}, N={N_NODES}, degree_linear, FORCED {STEPS} steps, "
        f"n={len(SEEDS)} seeds)  —  LZ_tail trend: {verdict}",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(FIG_OUT, dpi=140)
    print(f"[png] {FIG_OUT}")
    print(f"\nTotal wall time: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
