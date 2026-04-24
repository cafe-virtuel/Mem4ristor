#!/usr/bin/env python3
"""
P2-9 — Edge betweenness + diameter vs lambda2 (PROJECT_STATUS §10 P2 item 9).

Motivation:
  The dead zone (BA m>=5) is correlated with lambda2 > 2-3. But is lambda2 the
  *cause* of failure, or merely a proxy for multipath redundancy?

  Hypothesis: the true driver is path redundancy — the existence of multiple
  independent routes between any two nodes. Lambda2 captures algebraic
  connectivity but is not the same as path redundancy.

  Edge betweenness centrality (EBC) is the fraction of shortest paths that
  cross each edge. High average EBC = few bottleneck edges = low redundancy.
  Low average EBC = many parallel paths = high redundancy.

  Diameter measures the longest shortest path — a proxy for network
  compactness.

Protocol:
  - 13 topologies covering the full sweep (same as limit02_topology_sweep.py)
  - For each: compute lambda2, avg_edge_betweenness, diameter, avg_path_length
  - Scatter: lambda2 vs EBC (should anti-correlate if EBC tracks redundancy)
  - Scatter: lambda2 vs diameter (should anti-correlate — denser = shorter paths)
  - Load H_stable data from figures/fiedler_phase_diagram.csv
  - Regress EBC and diameter against H_stable to compare predictive power

Output:
  figures/p2_edge_betweenness.png   (4 panels)
  figures/p2_edge_betweenness.csv   (topology metrics table)

Created: 2026-04-24 (Claude Sonnet 4.6, P2-9).
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
import networkx as nx
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

FIG_PATH = ROOT / "figures" / "p2_edge_betweenness.png"
CSV_PATH = ROOT / "figures" / "p2_edge_betweenness.csv"
FIEDLER_CSV = ROOT / "figures" / "fiedler_phase_diagram.csv"

N = 100   # fixed size for all topologies
SEEDS = [0, 1, 2]   # 3 seeds to average graph randomness


# ──────────────────────────────────────────────────────────────────────────────
# Graph generators (same topologies as limit02_topology_sweep.py)
# ──────────────────────────────────────────────────────────────────────────────

def make_ba_nx(n: int, m: int, seed: int) -> nx.Graph:
    return nx.barabasi_albert_graph(n, m, seed=seed)


def make_ws_nx(n: int, k: int, p: float, seed: int) -> nx.Graph:
    return nx.watts_strogatz_graph(n, k, p, seed=seed)


def make_er_nx(n: int, p: float, seed: int) -> nx.Graph:
    return nx.erdos_renyi_graph(n, p, seed=seed)


TOPOLOGY_SPECS = [
    # (name, generator_fn)
    ("BA m=1",   lambda s: make_ba_nx(N, 1, s)),
    ("BA m=2",   lambda s: make_ba_nx(N, 2, s)),
    ("BA m=3",   lambda s: make_ba_nx(N, 3, s)),
    ("BA m=4",   lambda s: make_ba_nx(N, 4, s)),
    ("BA m=5",   lambda s: make_ba_nx(N, 5, s)),
    ("BA m=8",   lambda s: make_ba_nx(N, 8, s)),
    ("BA m=10",  lambda s: make_ba_nx(N, 10, s)),
    ("WS p=0.1", lambda s: make_ws_nx(N, 4, 0.1, s)),
    ("WS p=0.3", lambda s: make_ws_nx(N, 4, 0.3, s)),
    ("ER p=0.05",lambda s: make_er_nx(N, 0.05, s)),
    ("ER p=0.12",lambda s: make_er_nx(N, 0.12, s)),
    ("Lattice",  lambda s: nx.grid_2d_graph(10, 10)),
]

# Known regime from §3sexies (limit02_topology_sweep)
REGIME = {
    "BA m=1":    "uniform_wins",
    "BA m=2":    "degree_linear_wins",
    "BA m=3":    "degree_linear_wins",
    "BA m=4":    "degree_linear_marginal",
    "BA m=5":    "dead_zone",
    "BA m=8":    "dead_zone",
    "BA m=10":   "dead_zone",
    "WS p=0.1":  "degree_linear_wins",
    "WS p=0.3":  "degree_linear_wins",
    "ER p=0.05": "degree_linear_wins",
    "ER p=0.12": "dead_zone",
    "Lattice":   "uniform_wins",
}

REGIME_COLORS = {
    "uniform_wins":          "#1f77b4",
    "degree_linear_wins":    "#2ca02c",
    "degree_linear_marginal":"#ff7f0e",
    "dead_zone":             "#d62728",
}


# ──────────────────────────────────────────────────────────────────────────────
# Metric computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(G: nx.Graph) -> dict[str, float]:
    if not nx.is_connected(G):
        # Use largest connected component
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

    n = G.number_of_nodes()

    # Lambda2 via networkx
    L = nx.normalized_laplacian_matrix(G).toarray()
    evals = np.sort(np.linalg.eigvalsh(L))
    lambda2_norm = float(evals[1]) if len(evals) > 1 else 0.0

    # Unnormalized lambda2 (Fiedler value)
    L_unnorm = nx.laplacian_matrix(G).toarray().astype(float)
    evals_u = np.sort(np.linalg.eigvalsh(L_unnorm))
    lambda2 = float(evals_u[1]) if len(evals_u) > 1 else 0.0

    # Average edge betweenness centrality
    ebc = nx.edge_betweenness_centrality(G)
    avg_ebc = float(np.mean(list(ebc.values()))) if ebc else 0.0

    # Diameter and average path length (exact for connected)
    try:
        diameter = float(nx.diameter(G))
        avg_path = float(nx.average_shortest_path_length(G))
    except Exception:
        diameter = float("nan")
        avg_path = float("nan")

    # Average degree
    avg_deg = float(np.mean([d for _, d in G.degree()]))

    # Average clustering
    avg_clust = float(nx.average_clustering(G))

    return {
        "lambda2":      lambda2,
        "lambda2_norm": lambda2_norm,
        "avg_ebc":      avg_ebc,
        "diameter":     diameter,
        "avg_path":     avg_path,
        "avg_degree":   avg_deg,
        "avg_clustering": avg_clust,
        "n_nodes":      n,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    rows: list[dict] = []
    summary: list[dict] = []

    print(f"{'Topology':<14} {'lambda2':>8} {'avg_EBC':>10} {'diameter':>9} {'avg_path':>9} {'regime'}")
    print("-" * 75)

    for topo_name, gen_fn in TOPOLOGY_SPECS:
        metrics_list: list[dict] = []
        for seed in SEEDS:
            G = gen_fn(seed)
            m = compute_metrics(G)
            m["topology"] = topo_name
            m["seed"] = seed
            m["regime"] = REGIME.get(topo_name, "unknown")
            metrics_list.append(m)
            rows.append(m)

        # Mean over seeds
        keys = ["lambda2", "lambda2_norm", "avg_ebc", "diameter", "avg_path",
                "avg_degree", "avg_clustering"]
        mean_m = {k: float(np.mean([r[k] for r in metrics_list
                                    if not np.isnan(r[k])])) for k in keys}
        mean_m["topology"] = topo_name
        mean_m["regime"] = REGIME.get(topo_name, "unknown")
        summary.append(mean_m)
        print(
            f"{topo_name:<14} "
            f"{mean_m['lambda2']:>8.3f} "
            f"{mean_m['avg_ebc']:>10.5f} "
            f"{mean_m['diameter']:>9.1f} "
            f"{mean_m['avg_path']:>9.3f} "
            f"{mean_m['regime']}"
        )

    # ── CSV ───────────────────────────────────────────────────────────────
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "topology", "seed", "regime", "lambda2", "lambda2_norm",
            "avg_ebc", "diameter", "avg_path", "avg_degree",
            "avg_clustering", "n_nodes"
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[csv] {CSV_PATH}")

    # ── Load H_stable from fiedler_phase_diagram.csv if available ─────────
    h_by_topo: dict[str, float] = {}
    if FIEDLER_CSV.exists():
        with FIEDLER_CSV.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Group by topology name (averaging over seeds and norms)
                name = row.get("topology", "")
                h = float(row.get("H_stable", 0.0))
                if name not in h_by_topo:
                    h_by_topo[name] = []
                h_by_topo[name].append(h)
        h_by_topo = {k: float(np.max(v)) for k, v in h_by_topo.items()}
        print(f"\n[info] Loaded H_stable for {len(h_by_topo)} topologies from fiedler CSV")

    # ── Correlations ──────────────────────────────────────────────────────
    print("\n=== Correlations: metric vs lambda2 ===")
    l2_arr = np.array([s["lambda2"] for s in summary])
    for key in ["avg_ebc", "diameter", "avg_path", "avg_clustering"]:
        x = np.array([s[key] for s in summary
                      if not np.isnan(s[key])])
        l2_valid = np.array([s["lambda2"] for s in summary
                             if not np.isnan(s[key])])
        if len(x) >= 4:
            r, p = stats.pearsonr(l2_valid, x)
            print(f"  lambda2 vs {key:<18}: r={r:+.3f}, p={p:.3e}")

    # Dead zone indicator: 1 = dead zone, 0 = not
    dead = np.array([1 if s["regime"] == "dead_zone" else 0
                     for s in summary], dtype=float)
    print("\n=== Predictors of dead zone (point-biserial correlation) ===")
    for key in ["lambda2", "avg_ebc", "diameter", "avg_path", "avg_clustering"]:
        x = np.array([s[key] for s in summary])
        valid = ~np.isnan(x)
        if valid.sum() >= 4:
            r, p = stats.pointbiserialr(dead[valid], x[valid])
            print(f"  {key:<20}: r={r:+.3f}, p={p:.3e}")

    # ── Figure ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    plot_pairs = [
        (axes[0, 0], "lambda2",      "avg_ebc",
         "lambda2 (Fiedler)", "Avg Edge Betweenness Centrality",
         "lambda2 vs EBC\n(high EBC = few paths = low redundancy)"),
        (axes[0, 1], "lambda2",      "diameter",
         "lambda2 (Fiedler)", "Diameter (longest shortest path)",
         "lambda2 vs Diameter"),
        (axes[1, 0], "lambda2",      "avg_path",
         "lambda2 (Fiedler)", "Avg Shortest Path Length",
         "lambda2 vs Avg Path Length"),
        (axes[1, 1], "avg_ebc",      "diameter",
         "Avg EBC", "Diameter",
         "EBC vs Diameter (both redundancy proxies?)"),
    ]

    for ax, xkey, ykey, xlabel, ylabel, title in plot_pairs:
        for s in summary:
            if np.isnan(s.get(ykey, float("nan"))):
                continue
            color = REGIME_COLORS.get(s["regime"], "grey")
            ax.scatter(s[xkey], s[ykey], color=color, s=80,
                       edgecolors="k", linewidths=0.6, zorder=5)
            ax.annotate(s["topology"], (s[xkey], s[ykey]),
                        fontsize=6, ha="left", va="bottom",
                        xytext=(3, 3), textcoords="offset points")

        # Regression line (drop NaN)
        xs = np.array([s[xkey] for s in summary
                       if not np.isnan(s.get(ykey, float("nan")))])
        ys = np.array([s[ykey] for s in summary
                       if not np.isnan(s.get(ykey, float("nan")))])
        if len(xs) >= 3:
            m_fit, b_fit = np.polyfit(xs, ys, 1)
            xfit = np.linspace(xs.min(), xs.max(), 100)
            ax.plot(xfit, m_fit * xfit + b_fit, "k--", lw=1, alpha=0.5)
            r_fit, p_fit = stats.pearsonr(xs, ys)
            ax.set_title(f"{title}\nr={r_fit:+.3f}, p={p_fit:.2e}", fontsize=9)
        else:
            ax.set_title(title, fontsize=9)

        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=c, edgecolor="k", label=r.replace("_", " "))
        for r, c in REGIME_COLORS.items()
    ]
    axes[0, 0].legend(handles=legend_elements, fontsize=8, loc="upper right")

    fig.suptitle(
        "P2-9 — Is lambda2 a proxy for multipath redundancy?\n"
        f"(N={N}, {len(SEEDS)} seeds averaged per topology)",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=140)
    print(f"\n[png] {FIG_PATH}")
    print(f"Total wall time: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
