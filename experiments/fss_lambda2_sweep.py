#!/usr/bin/env python3
"""
FSS Lambda2 Sweep — Full Spectral Sweep of Topological Phase Transition

Quest: Map H_cont (entropy) and synchrony as a function of lambda2
(algebraic connectivity / Fiedler value) across BA topologies m=1..10.

Hypothesis: lambda2_crit ~ 2.31 separates functional (H>2, sync<0.5)
from dead zone (H<2, sync>0.5) regimes.

Method:
  - N=100, seeds=[42,123,777,17,256,1337,99,314,2024,888] (10 seeds)
  - m in [1, 2, 3, 4, 5, 6, 7, 8, 10]
  - 3000 steps, I_stim=0.5, cold_start=True, heretic_ratio=0.15
  - coupling_norm='degree_linear' (validated in Session 008)
  - D(u) = 0.50 * u (adaptive, Session 008 claim [16])

Metrics:
  - lambda2: Fiedler value of adjacency matrix (scipy eigh)
  - H_cont: 100-bin continuous entropy, last 25% of steps
  - synchrony: Pearson correlation of v traces, last 25% of steps

Output: figures/fss_lambda2_sweep.csv
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
D_MAX = 0.50  # D(u) = D_max * u (claim [16])
COUPLING_NORM = "degree_linear"


def fiedler_value(adj: np.ndarray) -> float:
    """Algebraic connectivity (second smallest Laplacian eigenvalue)."""
    deg = adj.sum(axis=1)
    L = np.diag(deg) - adj
    vals = eigh(L, eigvals_only=True)
    return float(vals[1]) if len(vals) > 1 else 0.0


def synchrony_pearson(v_history: np.ndarray) -> float:
    """
    Pearson correlation of node voltage traces.
    ~0 = decorrelated (goal), +1 = consensus, -1 = anti-sync.
    """
    n_nodes = v_history.shape[1]
    if n_nodes < 2:
        return 0.0
    # Mean-center each node's trace
    v_centered = v_history - v_history.mean(axis=0)
    # Pairwise Pearson: mean over all node pairs
    norms = np.linalg.norm(v_centered, axis=0)
    # Avoid division by zero
    valid = norms > 1e-12
    if valid.sum() < 2:
        return 0.0
    v_norm = v_centered[:, valid] / norms[valid]
    # Correlation matrix, take upper triangle mean (exclude diagonal)
    corr = np.corrcoef(v_norm.T)
    n = corr.shape[0]
    upper_tri = corr[np.triu_indices(n, k=1)]
    return float(np.mean(upper_tri))


def run_one(adj: np.ndarray, m: int, seed: int) -> dict:
    """Run Mem4Network with D(u) = D_max * u adaptive, return metrics."""
    # Build network
    net = Mem4Network(
        size=10,  # sqrt(N) for lattice-like indexing
        heretic_ratio=HERETIC_RATIO,
        seed=seed,
        adjacency_matrix=adj.copy(),
        coupling_norm=COUPLING_NORM,
        cold_start=True,
    )

    # Set D_max (used in adaptive formula)
    net.model.cfg["coupling"]["D"] = D_MAX

    h_trace = []
    v_history = []

    for step in range(STEPS):
        # Adaptive D(u) = D_max * u_mean / sqrt(N)
        u_mean = float(net.model.u.mean())
        net.model.D_eff = (D_MAX * u_mean) / np.sqrt(net.model.N)

        net.step(I_stimulus=I_STIM)

        h = calculate_continuous_entropy(net.v, bins=100)
        h_trace.append(h)
        v_history.append(net.v.copy())

    # Slice tail 25%
    tail_start = int(len(h_trace) * (1 - TAIL_FRAC))
    h_tail = h_trace[tail_start:]
    v_tail = np.array(v_history[tail_start:])  # shape (T_tail, N)

    # Metrics
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
    print("FSS Lambda2 Sweep — Full Spectral Sweep")
    print(f"N={N} | steps={STEPS} | I_stim={I_STIM} | cold_start=True | D(u)={D_MAX}*u")
    print(f"M values: {M_VALUES}")
    print(f"Seeds: {SEEDS}")
    print(f"coupling_norm={COUPLING_NORM} | heretic_ratio={HERETIC_RATIO}")
    print("=" * 80)

    records = []
    t0 = time.time()

    for m in M_VALUES:
        print(f"\n--- m={m} ---")
        for seed in SEEDS:
            # Build graph
            adj = make_ba(N, m, seed)
            l2 = fiedler_value(adj)

            # Run simulation
            metrics = run_one(adj, m, seed)

            rec = {
                "m": m,
                "seed": seed,
                "lambda2": l2,
                "h_mean": metrics["h_mean"],
                "h_std": metrics["h_std"],
                "synchrony": metrics["synchrony"],
                "u_mean": metrics["u_mean"],
            }
            records.append(rec)

            print(f"  seed={seed:4d}  lambda2={l2:.4f}  H={metrics['h_mean']:.4f}  "
                  f"sync={metrics['synchrony']:+.3f}  u={metrics['u_mean']:.4f}")

    elapsed = time.time() - t0

    # ── Aggregate per m ────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("AGGREGATE per m (mean over seeds)")
    print("=" * 80)

    # Sort records by lambda2 for cleaner table
    records_sorted = sorted(records, key=lambda r: r["lambda2"])

    print(f"\n{'m':>3} {'lambda2':>8} {'H_mean':>8} {'H_std':>6} {'sync':>7} {'u_mean':>7}")
    print("-" * 45)

    agg_records = []
    for m in M_VALUES:
        sub = [r for r in records if r["m"] == m]
        l2_m = np.mean([r["lambda2"] for r in sub])
        h_m = np.mean([r["h_mean"] for r in sub])
        h_s = np.mean([r["h_std"] for r in sub])
        sy_m = np.mean([r["synchrony"] for r in sub])
        u_m = np.mean([r["u_mean"] for r in sub])
        print(f"{m:3d} {l2_m:8.4f} {h_m:8.4f} {h_s:6.4f} {sy_m:7.4f} {u_m:7.4f}")
        agg_records.append({
            "m": m, "lambda2": l2_m, "h_mean": h_m, "h_std": h_s,
            "synchrony": sy_m, "u_mean": u_m,
        })

    # ── Save CSV ────────────────────────────────────────────────────────────────
    figures_dir = ROOT / "figures"
    figures_dir.mkdir(exist_ok=True)
    csv_path = figures_dir / "fss_lambda2_sweep.csv"

    import csv
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["m", "seed", "lambda2", "h_mean", "h_std", "synchrony", "u_mean"])
        writer.writeheader()
        writer.writerows(records)

    print(f"\nSaved: {csv_path}")
    print(f"Wall time: {elapsed:.1f}s | {len(records)} runs")

    # ── Phase classification ────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("PHASE CLASSIFICATION")
    print("=" * 80)
    for rec in sorted(agg_records, key=lambda r: r["lambda2"]):
        l2 = rec["lambda2"]
        h = rec["h_mean"]
        sync = rec["synchrony"]
        if sync < 0.5 and h > 2.0:
            phase = "FUNCTIONAL"
        elif sync > 0.5 and h < 2.0:
            phase = "DEAD_ZONE"
        else:
            phase = "TRANSITIONAL"
        print(f"m={rec['m']:2d} lambda2={l2:.4f} H={h:.4f} sync={sync:+.4f}  [{phase}]")

    return 0


if __name__ == "__main__":
    sys.exit(main())