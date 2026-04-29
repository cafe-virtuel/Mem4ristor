#!/usr/bin/env python3
"""
[11] LZ Complexity Per Node — FULL vs FROZEN_U
================================================
Objectif : mesurer la complexité de Lempel-Ziv (LZ76 normalisée) par nœud
individuel pour les configurations FULL et FROZEN_U, afin de comprendre la
*distribution* de complexité intra-réseau plutôt que sa moyenne.

Questions clés :
  1. Les hubs (degré élevé) ont-ils une complexité différente des nœuds
     périphériques ? (corrélation degré ↔ LZ)
  2. La distribution per-node est-elle plus étroite/large dans FULL vs FROZEN_U ?
  3. Le finding "FULL = walkers structurés" (LZ_full < FROZEN_U) tient-il
     au niveau de chaque nœud individuel, ou est-ce un artefact de la moyenne ?

Protocole :
  - 2 configurations : FULL, FROZEN_U
  - 2 régimes : ENDOGENOUS (I_stim=0), FORCED (I_stim=0.3)
  - 2 topologies : BA m=3 (fonctionnel), BA m=5 (dead zone)
  - 5 seeds, N=100 nœuds, 3000 steps (stride=10 → 300 snapshots)
  - Métrique : LZ76 normalisé par nœud (non moyennée)

Outputs :
  figures/lz_per_node.csv          — (N_configs × seeds × N_nodes) lignes
  figures/lz_per_node_summary.csv  — statistiques par config × régime × topo
  figures/lz_per_node.png          — violin plots + scatter degré vs LZ

Created: 2026-04-29 (Claude Sonnet 4.6, item [11])
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mem4ristor.topology import Mem4Network          # noqa: E402
from mem4ristor.metrics import _lz76_phrases         # noqa: E402
from mem4ristor.graph_utils import make_ba            # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

N_NODES      = 100
STEPS        = 3000
TRACE_STRIDE = 10          # → T = 300 snapshots
N_SEEDS      = 5
LZ_BINS      = 5           # discretisation bins pour LZ

TOPOLOGIES = [
    ("ba_m3_functional", 3),
    ("ba_m5_dead_zone",  5),
]

CONFIGS = [
    ("FULL",     0.0),
    ("FULL",     0.3),
    ("FROZEN_U", 0.0),
    ("FROZEN_U", 0.3),
]

FIG_DIR  = ROOT / "figures"
FIG_PATH = FIG_DIR / "lz_per_node.png"
CSV_PATH = FIG_DIR / "lz_per_node.csv"
SUM_PATH = FIG_DIR / "lz_per_node_summary.csv"

# ─────────────────────────────────────────────────────────────────────────────
# LZ per node (ne pas moyenner sur N)
# ─────────────────────────────────────────────────────────────────────────────

def lz_per_node(v_history: np.ndarray, n_bins: int = LZ_BINS) -> np.ndarray:
    """
    Compute normalised LZ76 complexity for EACH node independently.

    v_history : (T, N) — T timesteps, N nodes.
    Returns   : (N,) array of LZ complexity values.
    """
    T, N = v_history.shape
    if T < 2:
        return np.zeros(N)

    v_min, v_max = v_history.min(), v_history.max()
    if v_max == v_min:
        return np.zeros(N)

    bin_idx = np.floor(
        (v_history - v_min) / (v_max - v_min) * n_bins
    ).astype(int).clip(0, n_bins - 1)   # (T, N)

    log2_T = np.log2(T)
    lz_vals = np.empty(N)
    for j in range(N):
        seq = "".join(str(x) for x in bin_idx[:, j])
        c = _lz76_phrases(seq)
        lz_vals[j] = c * log2_T / T
    return lz_vals


def node_degrees(adj: np.ndarray) -> np.ndarray:
    """Return degree of each node."""
    return adj.sum(axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Ablation helpers
# ─────────────────────────────────────────────────────────────────────────────

def apply_frozen_u(net: Mem4Network) -> None:
    model = net.model
    sigma_baseline = model.cfg['doubt'].get('sigma_baseline', 0.05)
    model.cfg['doubt']['epsilon_u'] = 0.0
    model.cfg['doubt']['tau_u']     = 1e12
    model.u = np.full(model.N, sigma_baseline)


# ─────────────────────────────────────────────────────────────────────────────
# Run one cell
# ─────────────────────────────────────────────────────────────────────────────

def run_one(config: str, i_stim: float, topo_name: str, m: int,
            seed: int) -> dict:
    adj = make_ba(N_NODES, m, seed)
    net = Mem4Network(adjacency_matrix=adj, heretic_ratio=0.15,
                      coupling_norm='degree_linear', seed=seed)
    if config == "FROZEN_U":
        apply_frozen_u(net)

    snapshots: list[np.ndarray] = []
    for step in range(STEPS):
        net.step(I_stimulus=i_stim)
        if step % TRACE_STRIDE == 0:
            snapshots.append(net.model.v.copy())

    v_history = np.array(snapshots)          # (T, N)
    tail_cut  = int(len(snapshots) * 0.75)   # last 25%
    v_tail    = v_history[tail_cut:]

    lz_full = lz_per_node(v_history)         # (N,)
    lz_tail = lz_per_node(v_tail)            # (N,)
    degrees  = node_degrees(adj)             # (N,)

    return {
        "config":    config,
        "topo":      topo_name,
        "m":         m,
        "i_stim":    i_stim,
        "seed":      seed,
        "lz_full":   lz_full,
        "lz_tail":   lz_tail,
        "degrees":   degrees,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    total = len(TOPOLOGIES) * len(CONFIGS) * N_SEEDS
    print(f"[11] LZ per node : {len(TOPOLOGIES)} topos x "
          f"{len(CONFIGS)} configs x {N_SEEDS} seeds = {total} runs")
    print(f"N={N_NODES}, steps={STEPS}, stride={TRACE_STRIDE} -> "
          f"T={STEPS//TRACE_STRIDE} snapshots\n")

    all_rows: list[dict] = []    # per-node rows (for CSV)
    agg: dict = {}               # aggregated for summary + plots

    run_idx = 0
    for topo_name, m in TOPOLOGIES:
        for config, i_stim in CONFIGS:
            key = (topo_name, config, i_stim)
            lz_full_all, lz_tail_all, deg_all = [], [], []

            for seed in range(N_SEEDS):
                run_idx += 1
                r = run_one(config, i_stim, topo_name, m, seed)

                lz_full_all.append(r["lz_full"])
                lz_tail_all.append(r["lz_tail"])
                deg_all.append(r["degrees"])

                # per-node rows
                for j in range(N_NODES):
                    all_rows.append({
                        "config":   config,
                        "topo":     topo_name,
                        "m":        m,
                        "i_stim":   i_stim,
                        "seed":     seed,
                        "node":     j,
                        "degree":   int(r["degrees"][j]),
                        "lz_full":  float(r["lz_full"][j]),
                        "lz_tail":  float(r["lz_tail"][j]),
                    })

                elapsed = time.time() - t0
                pct     = run_idx / total
                eta     = (elapsed / pct) * (1 - pct) if pct > 0 else 0
                lz_mean = float(r["lz_full"].mean())
                print(f"[{run_idx:3d}/{total}] {topo_name:20s} "
                      f"{config:8s} I={i_stim:.1f} seed={seed} "
                      f"| LZ_mean={lz_mean:.4f} | ETA {eta:.0f}s", flush=True)

            agg[key] = {
                "lz_full": np.concatenate(lz_full_all),   # (N_SEEDS*N,)
                "lz_tail": np.concatenate(lz_tail_all),
                "degrees": np.concatenate(deg_all),
            }

    # ── CSV raw ──────────────────────────────────────────────────────────────
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\n[csv] {CSV_PATH}  ({len(all_rows)} lignes)")

    # -- Summary statistics ----------------------------------------------------
    print("\n" + "="*80)
    print("SYNTHESE -- LZ_full per node (mean +/- std, N x seeds pooled)")
    print("="*80)
    print(f"{'topo+config+I':40s}  mean    std     Welch p vs FULL")
    print("-"*80)

    summary_rows = []
    for topo_name, m in TOPOLOGIES:
        ref_key  = (topo_name, "FULL", 0.3)
        ref_data = agg.get(ref_key, {}).get("lz_full", np.array([0.0]))
        for config, i_stim in CONFIGS:
            key  = (topo_name, config, i_stim)
            data = agg[key]["lz_full"]
            mean = data.mean()
            std  = data.std(ddof=1)
            _, p = stats.ttest_ind(ref_data, data, equal_var=False)
            label = f"{topo_name} / {config} / I={i_stim:.1f}"
            flag  = " <- ref" if key == ref_key else ""
            print(f"{label:40s}  {mean:.4f}  {std:.4f}  p={p:.2e}{flag}")
            summary_rows.append({
                "topo": topo_name, "config": config, "i_stim": i_stim,
                "lz_full_mean": mean, "lz_full_std": std,
                "lz_tail_mean": agg[key]["lz_tail"].mean(),
                "lz_tail_std":  agg[key]["lz_tail"].std(ddof=1),
                "p_vs_full_forced": p,
            })

    with SUM_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"[csv] {SUM_PATH}")

    # -- Degree <-> LZ correlation --------------------------------------------
    print("\n" + "="*80)
    print("CORRELATION DEGRE <-> LZ_full (Pearson r)")
    print("="*80)
    for topo_name, m in TOPOLOGIES:
        for config, i_stim in CONFIGS:
            key  = (topo_name, config, i_stim)
            degs = agg[key]["degrees"]
            lzs  = agg[key]["lz_full"]
            r, p = stats.pearsonr(degs, lzs)
            print(f"  {topo_name:20s} {config:8s} I={i_stim:.1f} -> "
                  f"r={r:+.3f}  p={p:.2e}")

    # -- Figure ----------------------------------------------------------------
    # 2 rows (topologies) x 2 cols (violin LZ_full | scatter degree vs LZ)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    colors = {"FULL": "#2ca02c", "FROZEN_U": "#d62728"}
    markers_i = {0.0: "o", 0.3: "s"}

    for row, (topo_name, m) in enumerate(TOPOLOGIES):
        # Left col: violin of LZ_full distribution
        ax = axes[row, 0]
        violin_data, violin_labels, violin_colors = [], [], []
        for config, i_stim in CONFIGS:
            key  = (topo_name, config, i_stim)
            data = agg[key]["lz_full"]
            violin_data.append(data)
            violin_labels.append(f"{config}\nI={i_stim:.1f}")
            violin_colors.append(colors[config])

        parts = ax.violinplot(violin_data, showmedians=True, showextrema=True)
        for pc, col in zip(parts["bodies"], violin_colors):
            pc.set_facecolor(col)
            pc.set_alpha(0.6)
        ax.set_xticks(range(1, len(violin_data) + 1))
        ax.set_xticklabels(violin_labels, fontsize=8)
        ax.set_title(f"{topo_name}\nLZ_full distribution per node", fontsize=9)
        ax.set_ylabel("LZ76 complexity")
        ax.grid(axis="y", alpha=0.3)

        # Right col: scatter degree vs LZ (FULL forced vs FROZEN forced)
        ax = axes[row, 1]
        for config in ["FULL", "FROZEN_U"]:
            key  = (topo_name, config, 0.3)
            degs = agg[key]["degrees"]
            lzs  = agg[key]["lz_full"]
            r, _ = stats.pearsonr(degs, lzs)
            ax.scatter(degs, lzs, c=colors[config], alpha=0.3, s=8,
                       label=f"{config} (r={r:+.2f})")
            # trend line
            z = np.polyfit(degs, lzs, 1)
            x_line = np.linspace(degs.min(), degs.max(), 100)
            ax.plot(x_line, np.polyval(z, x_line), c=colors[config],
                    linewidth=1.5, linestyle="--")
        ax.set_xlabel("Node degree")
        ax.set_ylabel("LZ76 complexity")
        ax.set_title(f"{topo_name}\nDegree vs LZ_full (I=0.3)", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(
        "[11] LZ76 Complexity Per Node — FULL vs FROZEN_U\n"
        f"(N={N_NODES}, {N_SEEDS} seeds, {STEPS} steps, "
        f"stride={TRACE_STRIDE} → T={STEPS//TRACE_STRIDE})",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=140)
    print(f"\n[png] {FIG_PATH}")
    print(f"\nTotal wall time: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
