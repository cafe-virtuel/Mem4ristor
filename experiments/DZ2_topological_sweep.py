#!/usr/bin/env python3
"""
DZ2 Topological Sweep — Mem4ristor Campaign J Supplement
Sweeps BA m=1..12 (lambda2 ~ 0.5 .. 14) on N=100 for D in {0.0, 0.15, 0.50}.
Metrics: H_cont (secondary), synchrony Pearson (PRIMARY), U4, LZ76.
Synchrony = mean Pearson correlation of node voltage traces (decoupled from binning).

Key questions:
  1. Does the lambda2 sweep profile shift with D?
  2. Does D rescue functional diversity at high lambda2 (dead zone)?
  3. What is the effective regime boundary for each D?
"""

from __future__ import annotations
import sys, os, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mem4ristor.core import Mem4Network
from mem4ristor.metrics import calculate_continuous_entropy

# ── Parameters ────────────────────────────────────────────────────────────────
N = 100
STEPS = 2000
TAIL_FRAC = 0.25
SEEDS = [42, 123, 777, 17, 256]
I_STIM = 0.5
D_VALUES = [0.0, 0.15, 0.50]
COUPLING_NORM = "degree_linear"
HERETIC = 0.15
M_VALUES = list(range(1, 9))  # BA m=1..8  (8 cells per D, 24 total runs per D)
OUTPUT_RAW = PROJECT_ROOT / "figures" / "dz2_topological_sweep.csv"
OUTPUT_AGG = PROJECT_ROOT / "figures" / "dz2_topological_sweep_agg.csv"
FIGURE = PROJECT_ROOT / "figures" / "dz2_topological_sweep.png"


def synchrony_pearson(v_history: np.ndarray) -> float:
    """Pearson correlation of node voltage traces (mean over node pairs)."""
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


def build_ba_lambda2(m, N, seed):
    """Build BA graph and return Fiedler value (lambda2)."""
    import networkx as nx
    np.random.seed(seed)
    G = nx.barabasi_albert_graph(N, m, seed=seed)
    A = nx.to_numpy_array(G)
    L = np.diag(A.sum(axis=1)) - A
    eigvals = np.sort(np.linalg.eigvalsh(L))
    return float(eigvals[1])


def run_one(N, m, seed, D, coupling_norm, I_stim, steps, tail_frac):
    """Single simulation run. Returns a dict of metrics."""
    l2 = build_ba_lambda2(m, N, seed)

    import networkx as nx
    np.random.seed(seed)
    G = nx.barabasi_albert_graph(N, m, seed=seed)
    adj = nx.to_numpy_array(G)

    net = Mem4Network(adjacency_matrix=adj, heretic_ratio=HERETIC, seed=seed,
                      coupling_norm=coupling_norm, cold_start=True)
    net.model.cfg["coupling"]["D"] = D
    net.model.D_eff = D  # Override adaptive if any

    h_trace = []
    v_history = []

    for step in range(steps):
        net.step(I_stimulus=I_stim)
        h = calculate_continuous_entropy(net.v, bins=100)
        h_trace.append(h)
        v_history.append(net.v.copy())

    tail_start = int(len(h_trace) * (1 - tail_frac))
    h_tail = h_trace[tail_start:]
    v_tail = np.array(v_history[tail_start:])

    h_mean = float(np.mean(h_tail))
    h_std = float(np.std(h_tail))
    sync = synchrony_pearson(v_tail)
    u_mean = float(net.model.u.mean())
    u_std = float(net.model.u.std())

    return {
        "N": N, "m": m, "seed": seed, "D": D,
        "lambda2": l2,
        "H_cont": h_mean,
        "H_cont_std": h_std,
        "sync": sync,
        "u_mean": u_mean,
        "u_std": u_std,
    }


def run_batch(N, m, D, seeds, coupling_norm, I_stim, steps, tail_frac, max_workers=4):
    """Run all seeds for one (m, D) cell using ThreadPoolExecutor."""
    params = [(N, m, s, D, coupling_norm, I_stim, steps, tail_frac) for s in seeds]
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(run_one, *p): p for p in params}
        for fut in as_completed(futs):
            try:
                r = fut.result(timeout=180)
                results.append(r)
            except Exception as e:
                p = futs[fut]
                results.append({"N": p[0], "m": p[1], "seed": p[2], "D": p[3],
                                "lambda2": np.nan, "H_cont": np.nan, "H_cont_std": np.nan,
                                "sync": 0.0, "u_mean": np.nan, "u_std": np.nan})
    return results


def main():
    print(f"DZ2 Topological Sweep — N={N}, M_VALUES={M_VALUES}, D={D_VALUES}, {len(SEEDS)} seeds")
    t0 = time.time()
    rows = []
    for D in D_VALUES:
        for m in M_VALUES:
            print(f"  D={D}, m={m} ...", end=" ", flush=True)
            batch = run_batch(N, m, D, SEEDS, COUPLING_NORM, I_STIM, STEPS, TAIL_FRAC)
            rows.extend(batch)
            elapsed_cell = time.time() - t0
            print(f"done ({len(batch)} runs, {elapsed_cell:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_RAW, index=False)
    print(f"Raw CSV: {OUTPUT_RAW}  ({len(df)} rows)")

    # ── Aggregate ─────────────────────────────────────────────────────────
    agg = df.groupby(["N", "m", "D"]).agg(
        lambda2_mean=("lambda2", "mean"),
        lambda2_std=("lambda2", "std"),
        H_cont_mean=("H_cont", "mean"),
        H_cont_std=("H_cont", "std"),
        sync_mean=("sync", "mean"),
        sync_std=("sync", "std"),
        u_mean_mean=("u_mean", "mean"),
        u_std_mean=("u_std", "mean"),
        count=("seed", "count"),
    ).reset_index()
    agg.to_csv(OUTPUT_AGG, index=False)
    print(f"Agg CSV: {OUTPUT_AGG}")

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.patch.set_facecolor("#1e1e2e")
    for ax in axes:
        ax.set_facecolor("#252535")
        ax.grid(True, alpha=0.3)

    colors = {0.0: "#4fc3f7", 0.15: "#ff8a65", 0.50: "#b5ff40"}
    markers = {0.0: "o", 0.15: "s", 0.50: "^"}

    for D in D_VALUES:
        sub = agg[agg["D"] == D].sort_values("lambda2_mean")
        lbl = f"D={D}"
        axes[0].errorbar(sub["lambda2_mean"], sub["H_cont_mean"], yerr=sub["H_cont_std"],
                          label=lbl, color=colors[D], marker=markers[D], ms=6, lw=2, alpha=0.85,
                          capsize=3)
        axes[1].errorbar(sub["lambda2_mean"], sub["sync_mean"], yerr=sub["sync_std"],
                          label=lbl, color=colors[D], marker=markers[D], ms=6, lw=2, alpha=0.85,
                          capsize=3)

    # Shade regime boundary zone
    for ax in axes:
        ax.axvspan(2.13, 2.50, alpha=0.12, color="#ffcc00", label="Regime boundary")
        ax.axvline(2.31, color="#ffcc00", lw=1.5, ls="--", alpha=0.7)

    axes[0].set_ylabel("H_cont (bits)", fontsize=12)
    axes[0].set_title("DZ2 Topological Sweep: H_cont and Synchrony vs lambda2", fontsize=13, color="white")
    axes[0].legend(fontsize=9)
    axes[0].set_ylim(bottom=0)

    axes[1].set_ylabel("Synchrony (Pearson R)", fontsize=12)
    axes[1].set_xlabel("Lambda2 (algebraic connectivity)", fontsize=12)
    axes[1].legend(fontsize=9)
    axes[1].set_ylim(-0.05, 1.05)

    plt.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURE, dpi=150, bbox_inches="tight")
    print(f"Figure: {FIGURE}")

    total = time.time() - t0
    print(f"\nTotal wall time: {total:.1f}s  ({total/60:.1f} min)")

    # Print summary table
    print("\nH_cont by lambda2 x D:")
    pivot = agg.pivot_table(index="lambda2_mean", columns="D", values="H_cont_mean")
    print(pivot.round(3).to_string())
    print("\nsync by lambda2 x D:")
    pivot_s = agg.pivot_table(index="lambda2_mean", columns="D", values="sync_mean")
    print(pivot_s.round(3).to_string())


if __name__ == "__main__":
    main()
