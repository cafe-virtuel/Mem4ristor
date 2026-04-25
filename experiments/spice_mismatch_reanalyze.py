#!/usr/bin/env python3
"""
Re-analyze cached P4.19 sweep with the new Phase 5 entropy metrics.

Responds to KIMI review's #1 critique: "the 5-bin ceiling (log2(5)=2.32)
artificially compresses the entropy measure; use a continuous estimate
or differential entropy instead."

This script reloads the 45 cached ngspice .dat files from
experiments/spice/results/sweep_*.dat and recomputes H_stable using:
  - calculate_continuous_entropy  (100 bins, primary — KIMI-preferred)
  - calculate_cognitive_entropy   (5 bins, KIMI-corrected +-0.4/1.2 thresholds)
  - legacy cognitive_entropy      (5 bins, old +-0.8/1.5 thresholds)

Outputs two new publishable figures:
  figures/spice_mismatch_sweep_continuous.png   (100-bin heatmap + curves)
  figures/spice_mismatch_sweep_metric_compare.png (3 metrics side-by-side)
  figures/spice_mismatch_sweep_continuous.csv

Validates that the complete-escape signature at (eta=0.5, sigma_C=0.5)
survives under the more defensible continuous metric.

Created: 2026-04-19 (Claude Opus 4.7, Phase 5 regeneration)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from mem4ristor.metrics import (  # noqa: E402
    calculate_continuous_entropy,
    calculate_cognitive_entropy,
)
from spice_dead_zone_test import (  # noqa: E402
    N, M_BA, TAIL_FRAC, parse_wrdata, cognitive_entropy as legacy_cog_entropy,
)

RESULTS = ROOT / "experiments" / "spice" / "results"
FIGURES = ROOT / "figures"

ETAS = [0.10, 0.30, 0.50]
SIGMAS = [0.0, 0.05, 0.10, 0.20, 0.50]
N_SEEDS = 3
NORM = "degree_linear"


def h_from_series(v_hist: np.ndarray, fn) -> float:
    n_steps = v_hist.shape[0]
    tail_start = int(n_steps * (1 - TAIL_FRAC))
    return float(np.mean([fn(v_hist[k]) for k in range(tail_start, n_steps)]))


def collect():
    h_cont = np.zeros((len(ETAS), len(SIGMAS), N_SEEDS))
    h_cog5 = np.zeros_like(h_cont)
    h_legacy = np.zeros_like(h_cont)
    missing = []
    for i, eta in enumerate(ETAS):
        for j, sigma in enumerate(SIGMAS):
            for k in range(N_SEEDS):
                tag = f"sweep_BA_m{M_BA}_N{N}_{NORM}_eta{eta:g}_sig{sigma:g}_s{k}"
                dat = RESULTS / f"{tag}.dat"
                if not dat.exists():
                    missing.append(dat.name)
                    continue
                _, v_sp = parse_wrdata(dat)
                h_cont[i, j, k] = h_from_series(v_sp, calculate_continuous_entropy)
                h_cog5[i, j, k] = h_from_series(v_sp, calculate_cognitive_entropy)
                h_legacy[i, j, k] = h_from_series(v_sp, legacy_cog_entropy)
    return h_cont, h_cog5, h_legacy, missing


def plot_heatmap_panels(H_mean: np.ndarray, H_std: np.ndarray, title: str,
                        cbar_label: str, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5),
                             gridspec_kw={"width_ratios": [1.2, 1.0]})

    ax = axes[0]
    vmax = max(0.10, H_mean.max() * 1.05)
    im = ax.imshow(H_mean, aspect="auto", origin="lower", cmap="viridis",
                   vmin=0, vmax=vmax)
    ax.set_xticks(range(len(SIGMAS)))
    ax.set_xticklabels([f"{s:g}" for s in SIGMAS])
    ax.set_yticks(range(len(ETAS)))
    ax.set_yticklabels([f"{e:g}" for e in ETAS])
    ax.set_xlabel("Capacitor mismatch sigma_C", fontsize=11)
    ax.set_ylabel("Noise amplitude eta", fontsize=11)
    ax.set_title(title, fontsize=11)
    for i in range(len(ETAS)):
        for j in range(len(SIGMAS)):
            txt = f"{H_mean[i,j]:.2f}\n+-{H_std[i,j]:.2f}"
            color = "white" if H_mean[i, j] < vmax * 0.6 else "black"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=9)
    fig.colorbar(im, ax=ax, label=cbar_label)

    ax2 = axes[1]
    for i, e in enumerate(ETAS):
        ax2.errorbar(SIGMAS, H_mean[i], yerr=H_std[i],
                     marker="o", capsize=3, lw=1.5, label=f"eta = {e:g}")
    ax2.set_xlabel("Capacitor mismatch sigma_C", fontsize=11)
    ax2.set_ylabel(cbar_label, fontsize=11)
    ax2.set_title("Escape curve from dead zone", fontsize=11)
    ax2.grid(alpha=0.3)
    ax2.legend(loc="best", fontsize=10)
    ax2.axhline(0, color="black", lw=0.5)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_metric_comparison(H_cont, H_cog5, H_legacy, out_path: Path):
    """3 side-by-side heatmaps under the same colormap scale (per-metric)."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    datasets = [
        (H_cont, "100-bin continuous (KIMI-preferred)", "H (bits)"),
        (H_cog5, "5-bin KIMI-corrected\nthresholds +-0.4, +-1.2", "H (bits, log2 5 = 2.32 max)"),
        (H_legacy, "5-bin legacy\nthresholds +-0.8, +-1.5", "H (bits, log2 5 = 2.32 max)"),
    ]
    for ax, (H, title, cbarl) in zip(axes, datasets):
        mean = H.mean(axis=2)
        vmax = max(0.10, mean.max() * 1.05)
        im = ax.imshow(mean, aspect="auto", origin="lower", cmap="viridis",
                       vmin=0, vmax=vmax)
        ax.set_xticks(range(len(SIGMAS)))
        ax.set_xticklabels([f"{s:g}" for s in SIGMAS])
        ax.set_yticks(range(len(ETAS)))
        ax.set_yticklabels([f"{e:g}" for e in ETAS])
        ax.set_xlabel("sigma_C")
        ax.set_ylabel("eta")
        ax.set_title(title, fontsize=10)
        for i in range(len(ETAS)):
            for j in range(len(SIGMAS)):
                color = "white" if mean[i, j] < vmax * 0.6 else "black"
                ax.text(j, i, f"{mean[i,j]:.2f}", ha="center", va="center",
                        color=color, fontsize=8)
        fig.colorbar(im, ax=ax, label=cbarl, shrink=0.85)

    fig.suptitle(
        "P4.19 entropy metric comparison  (BA m=5, N=64, degree_linear)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def write_csv(H_cont, H_cog5, H_legacy, path: Path):
    header = "eta,sigma_C,seed,H_continuous_100bin,H_cognitive_kimi,H_cognitive_legacy\n"
    lines = [header]
    for i, e in enumerate(ETAS):
        for j, s in enumerate(SIGMAS):
            for k in range(N_SEEDS):
                lines.append(
                    f"{e},{s},{k},"
                    f"{H_cont[i,j,k]:.6f},"
                    f"{H_cog5[i,j,k]:.6f},"
                    f"{H_legacy[i,j,k]:.6f}\n"
                )
    path.write_text("".join(lines), encoding="utf-8")


def main() -> int:
    print("=" * 84)
    print("Re-analyzing cached P4.19 sweep with Phase 5 metrics")
    print(f"  source : {RESULTS}/sweep_BA_m{M_BA}_N{N}_{NORM}_*.dat")
    print(f"  output : figures/spice_mismatch_sweep_{{continuous,metric_compare}}.(png|csv)")
    print("=" * 84)

    H_cont, H_cog5, H_legacy, missing = collect()
    if missing:
        print(f"WARNING: {len(missing)} missing .dat files (re-run the sweep):")
        for m in missing[:10]:
            print(f"  - {m}")
        if len(missing) > 10:
            print(f"  ... +{len(missing)-10} more")

    # Tables
    print("\nMean H_stable by metric (rows: eta in {0.10, 0.30, 0.50}; cols: sigma_C in {0.0, 0.05, 0.10, 0.20, 0.50})")
    for name, H in [("100-bin continuous", H_cont),
                    ("5-bin KIMI (+-0.4, +-1.2)", H_cog5),
                    ("5-bin legacy (+-0.8, +-1.5)", H_legacy)]:
        mean = H.mean(axis=2)
        print(f"\n  {name}:")
        header = "           " + " ".join(f"sig={s:>4g}" for s in SIGMAS)
        print(header)
        for i, e in enumerate(ETAS):
            row = "  ".join(f"{mean[i,j]:>6.3f}" for j in range(len(SIGMAS)))
            print(f"  eta={e:>4g}  {row}")

    # Write CSV
    csv_path = FIGURES / "spice_mismatch_sweep_continuous.csv"
    write_csv(H_cont, H_cog5, H_legacy, csv_path)
    print(f"\n  CSV: {csv_path}")

    # Primary figure (continuous entropy)
    H_mean = H_cont.mean(axis=2)
    H_std = H_cont.std(axis=2)
    out1 = FIGURES / "spice_mismatch_sweep_continuous.png"
    plot_heatmap_panels(
        H_mean, H_std,
        title=f"H_stable on dead zone (BA m={M_BA}, N={N}, {NORM})\n"
              f"100-bin continuous entropy (KIMI-preferred, no artificial ceiling)",
        cbar_label="H_continuous (bits)",
        out_path=out1,
    )
    print(f"  Figure (primary): {out1}")

    # Comparison figure
    out2 = FIGURES / "spice_mismatch_sweep_metric_compare.png"
    plot_metric_comparison(H_cont, H_cog5, H_legacy, out2)
    print(f"  Figure (compare): {out2}")

    # --- Verdict ---
    print()
    print("=" * 84)
    Hc_max = H_mean.max()
    iH = np.unravel_index(H_mean.argmax(), H_mean.shape)
    print(f"Continuous H_max = {Hc_max:.3f} bits "
          f"at eta={ETAS[iH[0]]:g}, sigma_C={SIGMAS[iH[1]]:g}")

    # Compare escape signatures across metrics
    for name, H in [("continuous", H_cont), ("5-bin KIMI", H_cog5),
                    ("5-bin legacy", H_legacy)]:
        m = H.mean(axis=2)
        print(f"  [{name:>12}] H(0.1,0)={m[0,0]:.3f}  H(0.3,0.5)={m[1,4]:.3f}  "
              f"H(0.5,0.5)={m[2,4]:.3f}")

    # Under the continuous metric, does the (eta=0.5, sigma=0.5) peak survive?
    rank_legacy = np.unravel_index(H_legacy.mean(axis=2).argmax(), H_mean.shape)
    rank_cont = iH
    survives = rank_cont == rank_legacy
    print(f"\nEscape peak location identical across metrics: {survives}")
    if survives:
        print("=> Phase 5 validation: the 'complete escape' signature is robust to the metric change.")
    else:
        print("=> The escape peak shifts under the continuous metric — investigate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
