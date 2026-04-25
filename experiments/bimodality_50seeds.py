#!/usr/bin/env python3
"""
Piste F — Bimodalité ENDOGENOUS FULL sur 50 seeds + Hartigan dip test
(PROJECT_STATUS §3novedecies-bis piste F).

Motivation:
  §3vigies (Piste A) found that avg_clustering predicts synchrony mode
  (Pearson r=−0.64, p=0.045) on only 10 seeds — too small for a robust
  conclusion. The bimodal structure (6 seeds coordinated, 3 desynchronised)
  may be an artefact of small sample.

Protocol:
  - 50 independent BA(m=3, N=100) graphs (different graph_seed)
  - ENDOGENOUS (I_stim=0), FULL model, degree_linear, 3000 steps
  - Measure pairwise_synchrony on stable tail (last 25%)
  - 4 topology metrics per graph: λ₂, max_degree, avg_clustering, diameter
  - Hartigan dip test (embedded, MC p-value) on the sync distribution
  - Bimodality Coefficient (BC) as a secondary criterion (BC > 5/9 → bimodal)
  - Regression of sync vs each topology metric (Pearson + Spearman, n=50)

Output:
  figures/bimodality_50seeds.png   (sync histogram + KDE + 4 regression panels)
  figures/bimodality_50seeds.csv   (raw data, 50 rows)

Created: 2026-04-24 (Claude Sonnet 4.6, Piste F).
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

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

from mem4ristor.topology import Mem4Network          # noqa: E402
from mem4ristor.metrics import calculate_pairwise_synchrony  # noqa: E402

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

N_NODES = 100
BA_M = 3
STEPS = 3000
TRACE_STRIDE = 10
TAIL_FRAC = 0.25
N_SEEDS = 50
I_STIM = 0.0          # ENDOGENOUS

FIG_PATH = ROOT / "figures" / "bimodality_50seeds.png"
CSV_PATH = ROOT / "figures" / "bimodality_50seeds.csv"


# ──────────────────────────────────────────────────────────────────────────────
# Hartigan dip test (embedded pure Python, no external package)
# Based on Hartigan & Hartigan (1985), JASA.
# ──────────────────────────────────────────────────────────────────────────────

def _gcm(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Greatest convex minorant of (x, y) via cumulative min on slopes."""
    n = len(x)
    hull = [0]
    for i in range(1, n):
        while len(hull) >= 2:
            j, k = hull[-2], hull[-1]
            if (y[i] - y[k]) * (x[k] - x[j]) <= (y[k] - y[j]) * (x[i] - x[k]):
                hull.pop()
            else:
                break
        hull.append(i)
    # Linearly interpolate GCM at all x points
    gcm_y = np.interp(x, x[hull], y[hull])
    return gcm_y


def _lcm(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least concave majorant via negation trick."""
    return -_gcm(x, -y)


def dip_statistic(data: np.ndarray) -> float:
    """
    Hartigan dip statistic: maximum distance between ECDF and the
    best-fit unimodal CDF (computed as the pointwise minimum of the GCM
    below the mode and LCM above).

    Returns D: values near 0 → unimodal, larger → bimodal.
    """
    x = np.sort(data.astype(float))
    n = len(x)
    # ECDF at sorted points
    ecdf = np.arange(1, n + 1) / n

    # Compute GCM (convex side) and LCM (concave side) of the ECDF
    gcm_y = _gcm(x, ecdf)
    lcm_y = _lcm(x, ecdf)

    # Dip = max deviation from unimodal (taken as min|ECDF - GCM|, |ECDF - LCM|)
    d_gcm = np.abs(ecdf - gcm_y)
    d_lcm = np.abs(ecdf - lcm_y)
    return float(max(d_gcm.max(), d_lcm.max())) / 2.0


def dip_pvalue(data: np.ndarray, n_boot: int = 2000, rng_seed: int = 0) -> tuple[float, float]:
    """
    Dip statistic + MC p-value under H0 (uniform distribution).
    Returns (dip, p_value).
    """
    d_obs = dip_statistic(data)
    rng = np.random.RandomState(rng_seed)
    n = len(data)
    null_dips = np.array([
        dip_statistic(rng.uniform(size=n))
        for _ in range(n_boot)
    ])
    p = float((null_dips >= d_obs).mean())
    return d_obs, p


def bimodality_coefficient(data: np.ndarray) -> float:
    """
    Sarle's Bimodality Coefficient (BC).
    BC > 5/9 ≈ 0.555 suggests bimodality.
    """
    n = len(data)
    m3 = stats.skew(data)
    m4 = stats.kurtosis(data)          # Fisher kurtosis (excess)
    numerator = m3 ** 2 + 1
    denom = m4 + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    return float(numerator / denom) if denom != 0 else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Graph helpers
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


def topology_metrics(adj: np.ndarray) -> dict[str, float]:
    n = adj.shape[0]
    degrees = adj.sum(axis=1)
    max_deg = float(degrees.max())

    # Algebraic connectivity λ₂
    deg_vec = degrees.copy()
    L = np.diag(deg_vec) - adj
    evals = eigh(L, eigvals_only=True, subset_by_index=[1, 1])
    lambda2 = float(evals[0])

    # Average clustering coefficient (simple)
    clustering = []
    for i in range(n):
        nbrs = np.where(adj[i] > 0)[0]
        k = len(nbrs)
        if k < 2:
            clustering.append(0.0)
            continue
        pairs = 0
        edges = 0
        for a in range(k):
            for b in range(a + 1, k):
                pairs += 1
                if adj[nbrs[a], nbrs[b]] > 0:
                    edges += 1
        clustering.append(edges / pairs)
    avg_clust = float(np.mean(clustering))

    # Diameter via BFS
    def bfs_eccentricity(start: int) -> int:
        dist = {start: 0}
        queue = [start]
        while queue:
            v = queue.pop(0)
            for w in np.where(adj[v] > 0)[0]:
                if w not in dist:
                    dist[w] = dist[v] + 1
                    queue.append(w)
        return max(dist.values()) if len(dist) == n else 9999

    # Approximate diameter (sample 20 nodes for speed)
    rng_d = np.random.RandomState(42)
    sample = rng_d.choice(n, size=min(20, n), replace=False)
    diameter = float(max(bfs_eccentricity(s) for s in sample))

    return {
        "lambda2":      lambda2,
        "max_degree":   max_deg,
        "avg_clustering": avg_clust,
        "diameter":     diameter,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Core runner
# ──────────────────────────────────────────────────────────────────────────────

def run_one(seed: int) -> dict:
    adj = make_ba_adjacency(N_NODES, BA_M, seed)
    net = Mem4Network(
        adjacency_matrix=adj,
        heretic_ratio=0.15,
        coupling_norm='degree_linear',
        seed=seed,
    )
    snapshots: list[np.ndarray] = []
    for step in range(STEPS):
        net.step(I_stimulus=I_STIM)
        if step % TRACE_STRIDE == 0:
            snapshots.append(net.model.v.copy())

    v_history = np.array(snapshots)
    cut = int(len(snapshots) * (1.0 - TAIL_FRAC))
    v_tail = v_history[cut:]

    sync = calculate_pairwise_synchrony(v_tail)
    topo = topology_metrics(adj)
    return {"seed": seed, "synchrony": sync, **topo}


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    rows: list[dict] = []
    print(f"=== ENDOGENOUS FULL bimodality — {N_SEEDS} seeds ===")
    for seed in range(N_SEEDS):
        t_run = time.time()
        r = run_one(seed)
        dt = time.time() - t_run
        print(
            f"seed={seed:>2}  sync={r['synchrony']:+.3f}  "
            f"l2={r['lambda2']:.3f}  clust={r['avg_clustering']:.3f}  "
            f"max_deg={r['max_degree']:.0f}  diam={r['diameter']:.0f}  ({dt:.1f}s)"
        )
        rows.append(r)

    # ── CSV ───────────────────────────────────────────────────────────────
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["seed", "synchrony", "lambda2",
                           "max_degree", "avg_clustering", "diameter"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[csv] {CSV_PATH}")

    sync_arr = np.array([r["synchrony"] for r in rows])
    topo_keys = ["lambda2", "max_degree", "avg_clustering", "diameter"]
    topo_labels = ["λ₂ (Algebraic Connectivity)",
                   "Max Degree", "Avg Clustering", "Diameter"]

    # ── Bimodality tests ──────────────────────────────────────────────────
    dip, p_dip = dip_pvalue(sync_arr, n_boot=2000)
    bc = bimodality_coefficient(sync_arr)
    print(f"\n=== Bimodality Tests (n={N_SEEDS}) ===")
    print(f"  Synchrony: mean={sync_arr.mean():.3f} ± {sync_arr.std():.3f}")
    print(f"  Hartigan dip D={dip:.4f}, p={p_dip:.3f} (H0: unimodal; p<0.05 → bimodal)")
    print(f"  Bimodality Coefficient BC={bc:.3f} (>0.555 → bimodal)")
    bimodal_verdict = (
        "BIMODAL (both criteria)"
        if (p_dip < 0.05 and bc > 0.555)
        else (
            "BIMODAL (dip only)" if p_dip < 0.05
            else ("BIMODAL (BC only)" if bc > 0.555
                  else "UNIMODAL (neither criterion)")
        )
    )
    print(f"  Verdict: {bimodal_verdict}")

    # ── Regression ────────────────────────────────────────────────────────
    print(f"\n=== Regression: sync vs topology (n={N_SEEDS}) ===")
    print(f"{'Metric':<25}  {'Pearson r':>10}  {'p':>8}  {'Spearman rho':>13}  {'p':>8}")
    print("-" * 75)
    reg_results = {}
    for key in topo_keys:
        x = np.array([r[key] for r in rows])
        pr, pp = stats.pearsonr(x, sync_arr)
        sr, sp = stats.spearmanr(x, sync_arr)
        reg_results[key] = (pr, pp, sr, sp)
        sig_p = "*" if pp < 0.05 else ("†" if pp < 0.10 else "")
        sig_s = "*" if sp < 0.05 else ("†" if sp < 0.10 else "")
        print(f"{key:<25}  {pr:>+9.3f}  {pp:>8.3e}{sig_p}  {sr:>+12.3f}  {sp:>8.3e}{sig_s}")
    print("  (* p<0.05, † p<0.10)")

    # ── Figure ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

    # Top-left: histogram + KDE
    ax_hist = fig.add_subplot(gs[0, 0])
    ax_hist.hist(sync_arr, bins=20, density=True, alpha=0.6,
                 color="#1f77b4", edgecolor="k", linewidth=0.5)
    kde_x = np.linspace(sync_arr.min() - 0.05, sync_arr.max() + 0.05, 300)
    kde = stats.gaussian_kde(sync_arr, bw_method=0.25)
    ax_hist.plot(kde_x, kde(kde_x), "r-", linewidth=2, label="KDE")
    ax_hist.axvline(sync_arr.mean(), ls="--", color="grey",
                    label=f"mean={sync_arr.mean():.3f}")
    ax_hist.set_xlabel("Pairwise Synchrony")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title(
        f"Distribution of ENDOGENOUS FULL synchrony\n"
        f"(n={N_SEEDS} seeds, BA m={BA_M}, N={N_NODES})\n"
        f"Dip p={p_dip:.3f}  BC={bc:.3f}  → {bimodal_verdict}",
        fontsize=8,
    )
    ax_hist.legend(fontsize=8)

    # Regression panels (4 topology metrics)
    positions = [gs[0, 1], gs[0, 2], gs[1, 0], gs[1, 1]]
    for pos, key, label in zip(positions, topo_keys, topo_labels):
        ax = fig.add_subplot(pos)
        x = np.array([r[key] for r in rows])
        pr, pp, sr, sp = reg_results[key]
        ax.scatter(x, sync_arr, s=20, alpha=0.6, color="#1f77b4")
        # Regression line
        m, b = np.polyfit(x, sync_arr, 1)
        xfit = np.linspace(x.min(), x.max(), 100)
        ax.plot(xfit, m * xfit + b, "r-", linewidth=1.5)
        sig = "**" if pp < 0.01 else ("*" if pp < 0.05 else ("†" if pp < 0.10 else "ns"))
        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel("Synchrony", fontsize=9)
        ax.set_title(
            f"r={pr:+.3f} ({sig}), ρ={sr:+.3f}\n"
            f"p_Pearson={pp:.2e}  p_Spearman={sp:.2e}",
            fontsize=8,
        )
        ax.grid(alpha=0.3)

    # Bottom-right: ECDF to visualize bimodality
    ax_ecdf = fig.add_subplot(gs[1, 2])
    x_sorted = np.sort(sync_arr)
    ecdf_y = np.arange(1, N_SEEDS + 1) / N_SEEDS
    ax_ecdf.step(x_sorted, ecdf_y, color="#1f77b4", linewidth=2, label="ECDF")
    ax_ecdf.set_xlabel("Pairwise Synchrony")
    ax_ecdf.set_ylabel("Cumulative Probability")
    ax_ecdf.set_title("ECDF — inflection reveals bimodality", fontsize=9)
    ax_ecdf.grid(alpha=0.3)
    ax_ecdf.legend(fontsize=8)

    fig.suptitle(
        f"Piste F — Bimodality of ENDOGENOUS FULL synchrony (n={N_SEEDS} seeds)\n"
        f"Hartigan D={dip:.4f} p={p_dip:.3f}  |  BC={bc:.3f}  |  {bimodal_verdict}",
        fontsize=11,
    )
    plt.savefig(FIG_PATH, dpi=140)
    print(f"\n[png] {FIG_PATH}")
    print(f"Total wall time: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
