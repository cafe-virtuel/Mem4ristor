#!/usr/bin/env python3
"""
FSS Lambda2 Sweep V2 — Decoupled Design for Clean Topological Signal

Problem with V1: D(u) = 0.50 * u drives u to saturation (~0.999)
for ALL lambda2 > 0.6, making D_eff constant across all m values.
The sweep measures u saturation, NOT lambda2 effect.

V2 Design:
  - D static = 0.0 (baseline, no coupling) — isolates topology effect
  - D static = 0.15 (config default) — standard coupling
  - D static = 0.50 (max coupling) — strong coupling

This eliminates the u feedback loop and gives a clean measurement
of how topology (via lambda2) affects H_cont and synchrony.

Expected:
  - lambda2 low -> decorrelated (high H, sync ~ 0)
  - lambda2 high -> synchronized dead zone (low H, sync -> 1)
  - Critical threshold ~ 2.31
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from mem4ristor.core import Mem4Network
from mem4ristor.metrics import calculate_continuous_entropy
from mem4ristor.graph_utils import make_ba

# ── Parameters ────────────────────────────────────────────────────────────────
N = 100
STEPS = 3000
TAIL_FRAC = 0.25
SEEDS = [42, 123, 777, 17, 256, 1337, 99, 314, 2024, 888]
M_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 10]
HERETIC_RATIO = 0.15
I_STIM = 0.5
D_VALUES = [0.0, 0.15, 0.50]  # Static D values (no adaptive)
COUPLING_NORM = "degree_linear"


def fiedler_value(adj: np.ndarray) -> float:
    """Algebraic connectivity (second smallest Laplacian eigenvalue)."""
    deg = adj.sum(axis=1)
    L = np.diag(deg) - adj
    vals = eigh(L, eigvals_only=True)
    return float(vals[1]) if len(vals) > 1 else 0.0


def synchrony_pearson(v_history: np.ndarray) -> float:
    """Pearson correlation of node voltage traces."""
    n_nodes = v_history.shape[1]
    if n_nodes < 2:
        return 0.0
    v_centered = v_history - v_history.mean(axis=0)
    norms = np.linalg.norm(v_centered, axis=0)
    valid = norms > 1e-12
    if valid.sum() < 2:
        return 0.0
    v_norm = v_centered[:, valid] / norms[valid]
    corr = np.corrcoef(v_norm.T)
    n = corr.shape[0]
    upper_tri = corr[np.triu_indices(n, k=1)]
    return float(np.mean(upper_tri))


def run_one(adj: np.ndarray, D: float, seed: int) -> dict:
    """Run Mem4Network with STATIC D (no adaptive), return metrics."""
    net = Mem4Network(
        size=10,
        heretic_ratio=HERETIC_RATIO,
        seed=seed,
        adjacency_matrix=adj.copy(),
        coupling_norm=COUPLING_NORM,
        cold_start=True,
    )
    # Static D (no adaptation)
    net.model.cfg["coupling"]["D"] = D
    net.model.D_eff = D  # Override adaptive if any

    h_trace = []
    v_history = []

    for step in range(STEPS):
        net.step(I_stimulus=I_STIM)
        h = calculate_continuous_entropy(net.v, bins=100)
        h_trace.append(h)
        v_history.append(net.v.copy())

    # Slice tail 25%
    tail_start = int(len(h_trace) * (1 - TAIL_FRAC))
    h_tail = h_trace[tail_start:]
    v_tail = np.array(v_history[tail_start:])

    h_mean = float(np.mean(h_tail))
    h_std = float(np.std(h_tail))
    sync = synchrony_pearson(v_tail)
    u_mean_final = float(net.model.u.mean())

    return {
        "h_mean": h_mean,
        "h_std": h_std,
        "synchrony": sync,
        "u_mean": u_mean_final,
    }


def main() -> int:
    print("=" * 80)
    print("FSS Lambda2 Sweep V2 — Decoupled Static D")
    print(f"N={N} | steps={STEPS} | I_stim={I_STIM} | cold_start=True")
    print(f"M values: {M_VALUES}")
    print(f"D values: {D_VALUES} (STATIC — no adaptive)")
    print(f"Seeds: {len(SEEDS)}")
    print(f"coupling_norm={COUPLING_NORM} | heretic_ratio={HERETIC_RATIO}")
    print("=" * 80)

    records = []
    t0 = time.time()

    for D in D_VALUES:
        print(f"\n{'='*80}\n--- D={D} (STATIC) ---\n{'='*80}")
        for m in M_VALUES:
            print(f"\n  m={m}:")
            l2_values = []
            h_values = []
            sync_values = []

            for seed in SEEDS:
                adj = make_ba(N, m, seed)
                l2 = fiedler_value(adj)
                metrics = run_one(adj, D, seed)

                rec = {
                    "D": D, "m": m, "seed": seed, "lambda2": l2,
                    "h_mean": metrics["h_mean"], "h_std": metrics["h_std"],
                    "synchrony": metrics["synchrony"], "u_mean": metrics["u_mean"],
                }
                records.append(rec)
                l2_values.append(l2)
                h_values.append(metrics["h_mean"])
                sync_values.append(metrics["synchrony"])

            l2_m = np.mean(l2_values)
            h_m = np.mean(h_values)
            sync_m = np.mean(sync_values)
            print(f"    lambda2={l2_m:.4f}  H={h_m:.4f}  sync={sync_m:+.4f}")

    elapsed = time.time() - t0

    # ── Aggregate per D × m ─────────────────────────────────────────────────────
    print("\n\n" + "=" * 80)
    print("FULL RESULTS TABLE (sorted by lambda2)")
    print("=" * 80)

    # Sort by lambda2
    records_sorted = sorted(records, key=lambda r: r["lambda2"])

    # Print all D values side by side
    for m in M_VALUES:
        print(f"\n--- m={m} ---")
        for D in D_VALUES:
            sub = [r for r in records if r["m"] == m and r["D"] == D]
            l2_m = np.mean([r["lambda2"] for r in sub])
            h_m = np.mean([r["h_mean"] for r in sub])
            sync_m = np.mean([r["synchrony"] for r in sub])
            u_m = np.mean([r["u_mean"] for r in sub])
            print(f"  D={D:.2f}: lambda2={l2_m:.4f}  H={h_m:.4f}  sync={sync_m:+.4f}  u={u_m:.4f}")

    # ── Phase classification by lambda2 ─────────────────────────────────────────
    print("\n\n" + "=" * 80)
    print("PHASE CLASSIFICATION (D=0.15 as reference)")
    print("=" * 80)
    print(f"\n{'m':>3} {'D':>4} {'lambda2':>8} {'H_mean':>8} {'H_std':>6} {'sync':>7} {'u_mean':>7}  Phase")
    print("-" * 60)

    # Use D=0.15 for phase classification
    for m in M_VALUES:
        for D in D_VALUES:
            sub = [r for r in records if r["m"] == m and r["D"] == D]
            l2_m = np.mean([r["lambda2"] for r in sub])
            h_m = np.mean([r["h_mean"] for r in sub])
            h_std_m = np.mean([r["h_std"] for r in sub])
            sync_m = np.mean([r["synchrony"] for r in sub])
            u_m = np.mean([r["u_mean"] for r in sub])
            if sync_m < 0.3 and h_m > 3.0:
                phase = "FUNCTIONAL"
            elif sync_m > 0.7 or (h_m < 2.5 and sync_m > 0.5):
                phase = "DEAD_ZONE"
            elif sync_m > 0.3 or h_m < 3.0:
                phase = "TRANSITIONAL"
            else:
                phase = "UNKNOWN"
            print(f"{m:3d} {D:4.2f} {l2_m:8.4f} {h_m:8.4f} {h_std_m:6.4f} {sync_m:7.4f} {u_m:7.4f}  {phase}")

    # ── Save CSV ────────────────────────────────────────────────────────────────
    figures_dir = ROOT / "figures"
    figures_dir.mkdir(exist_ok=True)
    csv_path = figures_dir / "fss_lambda2_sweep_v2.csv"

    import csv
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["D", "m", "seed", "lambda2", "h_mean", "h_std", "synchrony", "u_mean"])
        writer.writeheader()
        writer.writerows(records)

    print(f"\nSaved: {csv_path}")
    print(f"Wall time: {elapsed:.1f}s | {len(records)} runs")

    return 0


if __name__ == "__main__":
    sys.exit(main())