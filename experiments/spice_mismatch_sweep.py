#!/usr/bin/env python3
"""
SPICE mismatch sweep — characterize the escape curve from the dead zone.

Follows up on §3decies (noise + mismatch synergy partial escape). This
script produces the publishable figure for Paper B: a 2D heatmap of
H_stable as a function of (noise amplitude eta, capacitor mismatch sigma)
on the canonical dead zone (BA m=5 N=64, degree_linear normalization).

Design:
  eta ∈ {0.10, 0.30, 0.50}             (3 noise levels — sub/at/over the
                                         partial-escape threshold)
  sigma_C ∈ {0, 0.05, 0.10, 0.20, 0.50} (CMOS-realistic to spin-glass-strong)
  seeds ∈ {0, 1, 2}                     (3 Monte Carlo trials per cell)

Total: 3 x 5 x 3 = 45 ngspice runs (~5-10 minutes at ~7s/run).

Output:
  figures/spice_mismatch_sweep.png      heatmap with error bars per cell
  figures/spice_mismatch_sweep.csv      raw (eta, sigma, seed, H) data

Created: 2026-04-19 (Claude Opus 4.7, hardware track P4.19bis)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

RESULTS = ROOT / "experiments" / "spice" / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

from limit02_topology_sweep import make_ba  # noqa: E402
from spice_dead_zone_test import (  # noqa: E402
    N, M_BA, T_END, DT, SEED, compute_scale_factors,
    parse_wrdata, h_stable,
)
from spice_noise_resonance import generate_netlist, run_ngspice  # noqa: E402

# Sweep grid
ETAS = [0.10, 0.30, 0.50]
SIGMAS = [0.0, 0.05, 0.10, 0.20, 0.50]
N_SEEDS = 3
NORM = "degree_linear"  # best from §3decies

# Mismatch clip range (avoid zero/negative caps for large sigma)
C_CLIP = (0.1, 5.0)


def main() -> int:
    print("=" * 84)
    print(f"SPICE mismatch sweep — escape curve from dead zone")
    print(f"BA m={M_BA} N={N}, norm={NORM}")
    print(f"  eta in {ETAS}, sigma_C in {SIGMAS}, seeds={N_SEEDS}")
    print(f"  total runs: {len(ETAS) * len(SIGMAS) * N_SEEDS}")
    print("=" * 84)

    adj = make_ba(N, m=M_BA, seed=SEED)
    rng_init = np.random.RandomState(SEED)
    init_v = rng_init.uniform(-1.0, 1.0, N)
    scale = compute_scale_factors(adj, NORM)

    # results[(eta, sigma)] = list of H over seeds
    results = {}
    rows_csv = [("eta", "sigma_C", "seed", "H_stable")]
    t0 = time.time()
    total_runs = len(ETAS) * len(SIGMAS) * N_SEEDS
    run_idx = 0

    for eta in ETAS:
        for sigma in SIGMAS:
            Hs = []
            for seed in range(N_SEEDS):
                run_idx += 1
                if sigma == 0.0:
                    c_vals = np.ones(N)
                else:
                    mc_rng = np.random.RandomState(SEED + 1000 + seed * 17 + int(sigma * 100))
                    c_vals = np.clip(mc_rng.normal(1.0, sigma, N), *C_CLIP)
                tag = f"sweep_BA_m{M_BA}_N{N}_{NORM}_eta{eta:g}_sig{sigma:g}_s{seed}"
                path = generate_netlist(adj, scale, init_v, T_END, DT, tag,
                                        noise_amp=eta, c_values=c_vals)
                t_run = run_ngspice(path)
                _, v_sp = parse_wrdata(RESULTS / f"{tag}.dat")
                H = h_stable(v_sp)
                Hs.append(H)
                rows_csv.append((eta, sigma, seed, H))
                print(f"  [{run_idx:>3}/{total_runs}] eta={eta:g} sig={sigma:g} s={seed}  "
                      f"H={H:.3f}  ({t_run:.1f}s)")
            results[(eta, sigma)] = Hs

    elapsed = time.time() - t0
    print(f"\n  total elapsed: {elapsed:.1f}s")

    # --- CSV ---
    csv_path = FIGURES / "spice_mismatch_sweep.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        for row in rows_csv:
            f.write(",".join(str(x) for x in row) + "\n")
    print(f"  CSV: {csv_path}")

    # --- Heatmap (mean H) + per-cell std annotations ---
    H_mean = np.array([[np.mean(results[(e, s)]) for s in SIGMAS] for e in ETAS])
    H_std = np.array([[np.std(results[(e, s)]) for s in SIGMAS] for e in ETAS])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5),
                             gridspec_kw={"width_ratios": [1.2, 1.0]})

    ax = axes[0]
    im = ax.imshow(H_mean, aspect="auto", origin="lower", cmap="viridis",
                   vmin=0, vmax=max(0.30, H_mean.max() * 1.05))
    ax.set_xticks(range(len(SIGMAS)))
    ax.set_xticklabels([f"{s:g}" for s in SIGMAS])
    ax.set_yticks(range(len(ETAS)))
    ax.set_yticklabels([f"{e:g}" for e in ETAS])
    ax.set_xlabel("Capacitor mismatch sigma_C", fontsize=11)
    ax.set_ylabel("Noise amplitude eta", fontsize=11)
    ax.set_title(
        f"H_stable on dead zone (BA m={M_BA}, N={N}, {NORM})\n"
        f"mean over {N_SEEDS} seeds, annotation = mean +- std",
        fontsize=11,
    )
    for i, _e in enumerate(ETAS):
        for j, _s in enumerate(SIGMAS):
            txt = f"{H_mean[i,j]:.2f}\n+-{H_std[i,j]:.2f}"
            color = "white" if H_mean[i, j] < H_mean.max() * 0.6 else "black"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=9)
    fig.colorbar(im, ax=ax, label="H_stable (cognitive bins)")

    # Right panel: H vs sigma_C, one curve per eta
    ax2 = axes[1]
    for i, e in enumerate(ETAS):
        ax2.errorbar(SIGMAS, H_mean[i], yerr=H_std[i],
                     marker="o", capsize=3, lw=1.5, label=f"eta = {e:g}")
    ax2.set_xlabel("Capacitor mismatch sigma_C", fontsize=11)
    ax2.set_ylabel("H_stable", fontsize=11)
    ax2.set_title("Escape curve from dead zone\n(higher = more diversity)", fontsize=11)
    ax2.grid(alpha=0.3)
    ax2.legend(loc="best", fontsize=10)
    ax2.axhline(0, color="black", lw=0.5)

    fig.suptitle(
        "SPICE: mismatch + noise can escape the BA m=5 dead zone (Paper B)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out = FIGURES / "spice_mismatch_sweep.png"
    fig.savefig(out, dpi=140)
    print(f"  Figure: {out}")

    # --- Verdict ---
    print()
    print("=" * 84)
    H_max = H_mean.max()
    iH = np.unravel_index(H_mean.argmax(), H_mean.shape)
    print(f"H_max = {H_max:.3f} at eta={ETAS[iH[0]]:g}, sigma_C={SIGMAS[iH[1]]:g}")

    # Does H grow monotonically with sigma at fixed eta?
    monotonic_in_sigma = all(
        all(H_mean[i, j] <= H_mean[i, j + 1] + 0.02 for j in range(len(SIGMAS) - 1))
        for i in range(len(ETAS))
    )
    print(f"H monotonically increases with sigma_C (within +-0.02): {monotonic_in_sigma}")

    if H_max > 0.50:
        print("=> Strong escape: noise + large mismatch break the dead zone substantially.")
    elif H_max > 0.20:
        print("=> Partial escape: hardware imperfection helps but consensus still strong.")
    else:
        print("=> Weak escape only — dead zone resists even strong disorder.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
