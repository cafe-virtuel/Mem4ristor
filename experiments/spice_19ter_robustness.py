#!/usr/bin/env python3
"""
SPICE mismatch robustness — P4.19ter

Three goals:
  (a) Multi-graph: repeat the best cell (eta=0.50, sigma_C=0.50) on 5 distinct
      BA m=5 N=64 seeds to confirm that H_max ~1.6 is not seed-dependent.
  (b) Dichotomy: binary-search sigma_c(eta) — the critical mismatch level where
      H crosses 0.5 (half-escape). Run at eta=0.30 (the resonance regime).
  (c) ER replication: same sweep (eta, sigma_C) on ER p=0.12 N=64 (the other
      dead zone from §3sexies) to test topology-agnosticism.

Output:
  figures/p4_19ter_multigraph.png    — box plot of H over 5 graph seeds
  figures/p4_19ter_dichotomy.png     — sigma_c(eta) curve on BA m=5
  figures/p4_19ter_er_replication.png — heatmap for ER p=0.12
  figures/p4_19ter_results.csv       — all raw (topology, eta, sigma, seed, H)

Created: 2026-04-19 (Antigravity, P4.19ter)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

RESULTS = ROOT / "experiments" / "spice" / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

from spice_dead_zone_test import (  # noqa: E402
    N, M_BA, T_END, DT, SEED,
    compute_scale_factors, parse_wrdata, h_stable,
)
from spice_noise_resonance import generate_netlist, run_ngspice  # noqa: E402

# ─── helpers ──────────────────────────────────────────────────────────────────

def make_ba(n: int, m: int, seed: int) -> np.ndarray:
    G = nx.barabasi_albert_graph(n, m, seed=seed)
    return nx.to_numpy_array(G)


def make_er(n: int, p: float, seed: int) -> np.ndarray:
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    # ER can produce isolated nodes; add a small random edge to isolates
    for node in list(nx.isolates(G)):
        partner = (node + 1) % n
        G.add_edge(node, partner)
    return nx.to_numpy_array(G)


def one_run(adj, norm, eta, sigma, seed, tag_prefix):
    """Single ngspice run. Returns H_stable."""
    init_rng = np.random.RandomState(SEED)
    init_v = init_rng.uniform(-1.0, 1.0, N)
    scale = compute_scale_factors(adj, norm)

    if sigma == 0.0:
        c_vals = np.ones(N)
    else:
        mc_rng = np.random.RandomState(SEED + 2000 + seed * 31 + int(sigma * 200))
        c_vals = np.clip(mc_rng.normal(1.0, sigma, N), 0.1, 5.0)

    tag = f"{tag_prefix}_eta{eta:g}_sig{sigma:g}_s{seed}"
    path = generate_netlist(adj, scale, init_v, T_END, DT, tag,
                            noise_amp=eta, c_values=c_vals)
    run_ngspice(path)
    _, v_sp = parse_wrdata(RESULTS / f"{tag}.dat")
    return h_stable(v_sp)


# ─── (a) Multi-graph robustness (best cell: eta=0.50, sigma=0.50) ────────────

def part_a(rows_csv):
    print("\n" + "=" * 80)
    print("(a) Multi-graph: 5 BA m=5 N=64 seeds — eta=0.50, sigma=0.50")
    print("=" * 80)
    NORM = "degree_linear"
    ETA, SIGMA = 0.50, 0.50
    GRAPH_SEEDS = [0, 7, 13, 42, 99]

    H_by_graph = []
    for gs in GRAPH_SEEDS:
        adj = make_ba(N, M_BA, gs)
        H = one_run(adj, NORM, ETA, SIGMA, seed=0,
                    tag_prefix=f"19ter_multigraph_BAm5_gs{gs}")
        H_by_graph.append(H)
        rows_csv.append(("BA_m5", gs, ETA, SIGMA, 0, H))
        print(f"  graph_seed={gs}  H={H:.3f}")

    print(f"\n  mean={np.mean(H_by_graph):.3f}  std={np.std(H_by_graph):.3f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(H_by_graph, positions=[0], widths=0.4)
    ax.scatter(np.zeros(len(H_by_graph)), H_by_graph, zorder=3, color="steelblue")
    for i, (gs, h) in enumerate(zip(GRAPH_SEEDS, H_by_graph)):
        ax.annotate(f"seed {gs}", (0, h), xytext=(0.18, h),
                    fontsize=8, va="center")
    ax.set_xticks([0])
    ax.set_xticklabels(["BA m=5, N=64\neta=0.50, sigma_C=0.50"])
    ax.set_ylabel("H_stable")
    ax.set_title("(a) Robustness across 5 graph seeds\n"
                 "Dead zone escape — degree_linear norm")
    ax.axhline(0, color="red", lw=0.8, linestyle="--", label="H=0 (dead)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = FIGURES / "p4_19ter_multigraph.png"
    fig.savefig(out, dpi=140)
    print(f"  -> {out}")
    return H_by_graph


# ─── (b) Dichotomy: sigma_c(eta) ──────────────────────────────────────────────

def part_b(rows_csv):
    print("\n" + "=" * 80)
    print("(b) Dichotomy: sigma_c(eta) — where H crosses 0.5 — BA m=5 seed=0")
    print("=" * 80)
    NORM = "degree_linear"
    H_THRESHOLD = 0.50
    ETAS = [0.10, 0.30, 0.50]
    BISECT_ROUNDS = 6   # precision 0.50 / 2^6 ≈ 0.008
    BISECT_SEEDS = 2    # average over 2 MC seeds for stability

    adj = make_ba(N, M_BA, seed=SEED)
    sigma_c = {}

    for eta in ETAS:
        lo, hi = 0.0, 0.50
        print(f"\n  eta={eta:g}: bisecting sigma in [{lo}, {hi}]")
        for _ in range(BISECT_ROUNDS):
            mid = (lo + hi) / 2
            Hs = [one_run(adj, NORM, eta, mid, seed=s,
                          tag_prefix=f"19ter_bisect_eta{eta:g}")
                  for s in range(BISECT_SEEDS)]
            H_mid = np.mean(Hs)
            rows_csv += [("BA_m5_bisect", SEED, eta, mid, s, H)
                         for s, H in enumerate(Hs)]
            print(f"    sigma={mid:.4f}  H={H_mid:.3f}  "
                  f"({'>' if H_mid >= H_THRESHOLD else '<'})")
            if H_mid >= H_THRESHOLD:
                hi = mid
            else:
                lo = mid
        sigma_c[eta] = (lo + hi) / 2
        print(f"  sigma_c(eta={eta:g}) = {sigma_c[eta]:.4f}")

    print(f"\n  sigma_c table: {sigma_c}")

    fig, ax = plt.subplots(figsize=(6, 4))
    etas_plot = list(sigma_c.keys())
    sc_plot = [sigma_c[e] for e in etas_plot]
    ax.plot(etas_plot, sc_plot, "o-", color="tomato", lw=2, ms=8)
    ax.set_xlabel("Noise amplitude η")
    ax.set_ylabel("Critical mismatch σ_c  (H ≥ 0.50)")
    ax.set_title("(b) Phase boundary: escape requires noise ↔ mismatch trade-off\n"
                 "BA m=5, N=64, degree_linear")
    ax.grid(alpha=0.3)
    ax.annotate("Lower σ_c → easier escape\nwith larger noise",
                xy=(etas_plot[-1], sc_plot[-1]),
                xytext=(etas_plot[-1] - 0.15, sc_plot[-1] + 0.05),
                fontsize=8, arrowprops=dict(arrowstyle="->"))
    fig.tight_layout()
    out = FIGURES / "p4_19ter_dichotomy.png"
    fig.savefig(out, dpi=140)
    print(f"  -> {out}")
    return sigma_c


# ─── (c) ER replication ───────────────────────────────────────────────────────

def part_c(rows_csv):
    print("\n" + "=" * 80)
    print("(c) ER replication: p=0.12 N=64 — same sweep as §3undecies")
    print("=" * 80)
    NORM = "degree_linear"
    ETAS = [0.10, 0.30, 0.50]
    SIGMAS = [0.0, 0.05, 0.10, 0.20, 0.50]
    N_SEEDS = 3

    adj = make_er(N, p=0.12, seed=SEED)

    H_mean = np.zeros((len(ETAS), len(SIGMAS)))
    H_std = np.zeros_like(H_mean)
    total = len(ETAS) * len(SIGMAS) * N_SEEDS
    run_idx = 0

    for i, eta in enumerate(ETAS):
        for j, sigma in enumerate(SIGMAS):
            Hs = []
            for s in range(N_SEEDS):
                run_idx += 1
                H = one_run(adj, NORM, eta, sigma, seed=s,
                            tag_prefix=f"19ter_er012")
                Hs.append(H)
                rows_csv.append(("ER_p012", SEED, eta, sigma, s, H))
                print(f"  [{run_idx:>3}/{total}] eta={eta:g} sigma={sigma:g} "
                      f"s={s}  H={H:.3f}")
            H_mean[i, j] = np.mean(Hs)
            H_std[i, j] = np.std(Hs)

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(H_mean, aspect="auto", origin="lower", cmap="viridis",
                   vmin=0, vmax=max(0.30, H_mean.max() * 1.05))
    ax.set_xticks(range(len(SIGMAS)))
    ax.set_xticklabels([f"{s:g}" for s in SIGMAS])
    ax.set_yticks(range(len(ETAS)))
    ax.set_yticklabels([f"{e:g}" for e in ETAS])
    ax.set_xlabel("Capacitor mismatch σ_C", fontsize=11)
    ax.set_ylabel("Noise amplitude η", fontsize=11)
    ax.set_title(f"(c) ER p=0.12, N={N}, degree_linear — "
                 f"topology-agnostic escape?\n(mean over {N_SEEDS} seeds)",
                 fontsize=10)
    for i2, _ in enumerate(ETAS):
        for j2, _ in enumerate(SIGMAS):
            txt = f"{H_mean[i2,j2]:.2f}\n±{H_std[i2,j2]:.2f}"
            col = "white" if H_mean[i2, j2] < H_mean.max() * 0.6 else "black"
            ax.text(j2, i2, txt, ha="center", va="center",
                    color=col, fontsize=8)
    fig.colorbar(im, ax=ax, label="H_stable")
    fig.tight_layout()
    out = FIGURES / "p4_19ter_er_replication.png"
    fig.savefig(out, dpi=140)
    print(f"  -> {out}")
    return H_mean


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    rows_csv = [("topology", "graph_seed", "eta", "sigma_C", "mc_seed", "H_stable")]
    t0 = time.time()

    H_multi = part_a(rows_csv)
    sigma_c = part_b(rows_csv)
    H_er    = part_c(rows_csv)

    # CSV dump
    csv_path = FIGURES / "p4_19ter_results.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        for row in rows_csv:
            f.write(",".join(str(x) for x in row) + "\n")
    print(f"\nCSV: {csv_path}")

    print("\n" + "=" * 80)
    print("SUMMARY — P4.19ter")
    print(f"  (a) BA multi-graph  : mean H = {np.mean(H_multi):.3f}  "
          f"std = {np.std(H_multi):.3f}")
    print(f"  (b) sigma_c(eta)    : {sigma_c}")
    print(f"  (c) ER p=0.12 H_max : {H_er.max():.3f}  "
          f"({'topology-agnostic OK' if H_er.max() > 0.40 else 'ER resists escape'})")
    print(f"  elapsed: {time.time()-t0:.1f}s")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
