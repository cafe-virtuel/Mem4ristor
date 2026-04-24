#!/usr/bin/env python3
"""
Piste D — Multi-topology universality of coordination metrics (PROJECT_STATUS §3novedecies-bis D).

Motivation:
  All results in §3novedecies/vigies are on BA m=3, N=100. Is the signature
  "FULL = walkers independants structures" (low sync, low LZ) universal, or
  specific to BA m=3?

  If FULL produces sync≈0 + LZ_low *everywhere*, we have a universal property
  of the Mem4ristor kernel. If topology-specific, the claim is weaker.

Protocol (§3novedecies-bis piste D):
  - 4 topologies: Lattice 10×10, BA m=3, BA m=5 (dead zone), ER p≈0.06 (sparse),
    Watts-Strogatz k=4 p=0.1
  - 4 ablations: FULL, NO_HERETIC, NO_SIGMOID, FROZEN_U
  - 2 protocols: ENDOGENOUS (I=0), FORCED (I=0.5)
  - 10 seeds per cell
  - Metrics: pairwise_synchrony + lz_full

  Total runs: 4 topologies × 4 ablations × 2 protocols × 10 seeds = 320 runs
  Estimated time: ~30-45 min

Outputs:
  figures/ablation_coordination_topo_sweep.png   (4 topology subplots × 2 metrics)
  figures/ablation_coordination_topo_sweep.csv   (raw data, 320 rows)

Created: 2026-04-24 (Claude Sonnet 4.6, Piste D).
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
STEPS = 3000
TRACE_STRIDE = 10
TAIL_FRAC = 0.25
SEEDS = list(range(10))

STIMULI = [("ENDOGENOUS", 0.0), ("FORCED", 0.5)]
ABLATIONS = [
    ("FULL",       "Full"),
    ("NO_HERETIC", "No Heretic"),
    ("NO_SIGMOID", "No Sigmoid"),
    ("FROZEN_U",   "Frozen u"),
]

TOPOLOGIES = [
    ("Lattice_10x10", None),   # will be built specially
    ("BA_m3",         {"type": "ba", "m": 3}),
    ("BA_m5",         {"type": "ba", "m": 5}),
    ("WS_k4_p01",     {"type": "ws", "k": 4, "p": 0.1}),
]

FIG_PATH = ROOT / "figures" / "ablation_coordination_topo_sweep.png"
CSV_PATH = ROOT / "figures" / "ablation_coordination_topo_sweep.csv"


# ──────────────────────────────────────────────────────────────────────────────
# Graph generators
# ──────────────────────────────────────────────────────────────────────────────

def make_lattice(n_side: int) -> np.ndarray:
    n = n_side * n_side
    adj = np.zeros((n, n), dtype=float)
    for i in range(n):
        r, c = divmod(i, n_side)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            rr, cc = r + dr, c + dc
            if 0 <= rr < n_side and 0 <= cc < n_side:
                j = rr * n_side + cc
                adj[i, j] = adj[j, i] = 1.0
    return adj


def make_ba(n: int, m: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    adj = np.zeros((n, n), dtype=float)
    for i in range(m + 1):
        for j in range(i + 1, m + 1):
            adj[i, j] = adj[j, i] = 1.0
    degrees = adj.sum(axis=1)
    for new_node in range(m + 1, n):
        probs = degrees[:new_node] / degrees[:new_node].sum()
        targets = rng.choice(new_node, size=m, replace=False, p=probs)
        for t in targets:
            adj[new_node, t] = adj[t, new_node] = 1.0
        degrees = adj.sum(axis=1)
    return adj


def make_ws(n: int, k: int, p: float, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    adj = np.zeros((n, n), dtype=float)
    # Ring lattice
    for i in range(n):
        for d in range(1, k // 2 + 1):
            j = (i + d) % n
            adj[i, j] = adj[j, i] = 1.0
    # Rewiring
    for i in range(n):
        for d in range(1, k // 2 + 1):
            j = (i + d) % n
            if rng.rand() < p:
                candidates = [x for x in range(n) if x != i and adj[i, x] == 0]
                if candidates:
                    new_j = rng.choice(candidates)
                    adj[i, j] = adj[j, i] = 0.0
                    adj[i, new_j] = adj[new_j, i] = 1.0
    return adj


def get_adjacency(topo_name: str, topo_cfg, seed: int) -> np.ndarray:
    if topo_name == "Lattice_10x10":
        return make_lattice(10)  # lattice is deterministic, seed unused
    cfg = topo_cfg
    if cfg["type"] == "ba":
        return make_ba(N_NODES, cfg["m"], seed)
    if cfg["type"] == "ws":
        return make_ws(N_NODES, cfg["k"], cfg["p"], seed)
    raise ValueError(f"Unknown topology: {topo_name}")


def coupling_norm_for(topo_name: str) -> str:
    """
    Use degree_linear for BA m=3 (known to work),
    uniform for everything else (including BA m=5 dead zone and lattice).
    """
    if topo_name == "BA_m3":
        return "degree_linear"
    return "uniform"


# ──────────────────────────────────────────────────────────────────────────────
# Ablation setup
# ──────────────────────────────────────────────────────────────────────────────

def apply_ablation(net: Mem4Network, ablation: str) -> None:
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

def run_one(topo_name: str, topo_cfg, ablation: str, seed: int,
            stimulus: float) -> dict[str, float]:
    adj = get_adjacency(topo_name, topo_cfg, seed)
    norm = coupling_norm_for(topo_name)
    net = Mem4Network(
        adjacency_matrix=adj,
        heretic_ratio=0.15,
        coupling_norm=norm,
        seed=seed,
    )
    apply_ablation(net, ablation)

    snapshots: list[np.ndarray] = []
    for step in range(STEPS):
        net.step(I_stimulus=stimulus)
        if step % TRACE_STRIDE == 0:
            snapshots.append(net.model.v.copy())

    v_history = np.array(snapshots)
    cut = int(len(snapshots) * (1.0 - TAIL_FRAC))
    v_tail = v_history[cut:]

    return {
        "synchrony": calculate_pairwise_synchrony(v_tail),
        "lz_full":   calculate_temporal_lz_complexity(v_history),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    METRICS = ["synchrony", "lz_full"]

    # results[topo][regime][metric][ablation] → list[float]
    results: dict = {
        t: {
            regime: {m: {k: [] for k, _ in ABLATIONS} for m in METRICS}
            for regime, _ in STIMULI
        }
        for t, _ in TOPOLOGIES
    }
    all_rows: list[dict] = []

    total_runs = len(TOPOLOGIES) * len(ABLATIONS) * len(STIMULI) * len(SEEDS)
    run_count = 0

    for topo_name, topo_cfg in TOPOLOGIES:
        for regime, I_stim in STIMULI:
            for ablation, label in ABLATIONS:
                for seed in SEEDS:
                    t_run = time.time()
                    r = run_one(topo_name, topo_cfg, ablation, seed, I_stim)
                    dt = time.time() - t_run
                    run_count += 1
                    print(
                        f"[{run_count:>3}/{total_runs}] "
                        f"{topo_name:<14} {regime:<11} {ablation:<11} "
                        f"seed={seed}  "
                        f"sync={r['synchrony']:+.3f}  "
                        f"lz={r['lz_full']:.3f}  ({dt:.1f}s)"
                    )
                    for m in METRICS:
                        results[topo_name][regime][m][ablation].append(r[m])
                    all_rows.append({
                        "topology": topo_name, "regime": regime,
                        "I_stim": I_stim, "ablation": ablation,
                        "seed": seed, **r,
                    })

    # ── CSV ───────────────────────────────────────────────────────────────
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["topology", "regime", "I_stim", "ablation",
                           "seed", "synchrony", "lz_full"]
        )
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n[csv] {CSV_PATH}")

    # ── Summary ───────────────────────────────────────────────────────────
    abl_keys = [k for k, _ in ABLATIONS]
    for topo_name, _ in TOPOLOGIES:
        for regime, _ in STIMULI:
            print(f"\n--- {topo_name} / {regime} ---")
            for m in METRICS:
                vals = {k: np.array(results[topo_name][regime][m][k])
                        for k in abl_keys}
                line = f"  {m:<12}"
                for k in abl_keys:
                    line += f"  {k[:8]}: {vals[k].mean():+.3f}±{vals[k].std(ddof=1)/np.sqrt(len(SEEDS)):.3f}"
                print(line)

    # ── Figure ────────────────────────────────────────────────────────────
    n_topos = len(TOPOLOGIES)
    n_regimes = len(STIMULI)
    fig, axes = plt.subplots(
        n_topos, n_regimes * 2,
        figsize=(16, 4 * n_topos),
        squeeze=False,
    )
    colors = ["#2ca02c", "#d62728", "#ff7f0e", "#9467bd"]
    xs = np.arange(len(ABLATIONS))

    for row, (topo_name, _) in enumerate(TOPOLOGIES):
        for col_base, (regime, I_stim) in enumerate(STIMULI):
            for m_idx, (metric, mname) in enumerate([
                ("synchrony", "Synchrony"),
                ("lz_full",   "LZ Complexity"),
            ]):
                col = col_base * 2 + m_idx
                ax = axes[row, col]
                means = [np.mean(results[topo_name][regime][metric][k])
                         for k in abl_keys]
                sems  = [np.std(results[topo_name][regime][metric][k], ddof=1)
                         / np.sqrt(len(SEEDS))
                         for k in abl_keys]
                bars = ax.bar(xs, means, yerr=sems, capsize=5,
                              color=colors, edgecolor="k",
                              linewidth=0.7, alpha=0.85)
                rng_jit = np.random.RandomState(42)
                for i, k in enumerate(abl_keys):
                    vals = results[topo_name][regime][metric][k]
                    jit = rng_jit.uniform(-0.12, 0.12, len(vals))
                    ax.scatter(np.full(len(vals), i) + jit, vals,
                               color="k", s=10, alpha=0.4, zorder=5)
                ax.set_xticks(xs)
                ax.set_xticklabels(["FULL", "NO_H", "NO_S", "FRZ_U"],
                                   fontsize=7, rotation=15)
                ax.set_title(
                    f"{topo_name} / {regime}\n{mname}",
                    fontsize=8,
                )
                ax.grid(axis="y", alpha=0.3)
                # Reference line at FULL value
                ax.axhline(means[0], ls="--", color="#2ca02c",
                           alpha=0.5, lw=1)

    fig.suptitle(
        "Piste D — Multi-topology universality of coordination metrics\n"
        f"(4 topologies × 4 ablations × 2 protocols, n={len(SEEDS)} seeds)",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=120)
    print(f"[png] {FIG_PATH}")
    print(f"\nTotal wall time: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
