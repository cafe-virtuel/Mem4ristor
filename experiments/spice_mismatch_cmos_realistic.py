#!/usr/bin/env python3
"""
CMOS-realistic mismatch sweep — P4.19 rebutted against KIMI critique #2.

KIMI noted that sigma_C = 0.50 in P4.19bis is physically implausible for
modern CMOS capacitor mismatch (~3-10% typical, up to 15% for aggressive
analog process corners). This script re-samples the escape curve inside
the physical range sigma_C in [0, 0.15] with finer granularity to test
whether partial escape still exists under realistic device variation.

Grid:
  eta     in {0.10, 0.30, 0.50}                   (3 noise levels)
  sigma_C in {0.0, 0.02, 0.05, 0.08, 0.10, 0.15}  (6 CMOS-realistic values)
  seeds   in {0, 1, 2}                             (3 Monte Carlo trials)

Total: 3 x 6 x 3 = 54 ngspice runs (~6 min at ~7s/run).

Metric: 100-bin continuous entropy (Phase 5 KIMI-preferred).

Output:
  figures/spice_mismatch_cmos.png      heatmap + escape curves
  figures/spice_mismatch_cmos.csv      raw (eta, sigma, seed, H_100bin, H_5bin)

Created: 2026-04-19 (Claude Opus 4.7, KIMI response track)
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
    N, M_BA, T_END, DT, SEED, TAIL_FRAC,
    compute_scale_factors, parse_wrdata,
)
from spice_noise_resonance import generate_netlist, run_ngspice  # noqa: E402
from mem4ristor.metrics import (  # noqa: E402
    calculate_continuous_entropy,
    calculate_cognitive_entropy,
)

# Sweep grid (CMOS-realistic)
ETAS = [0.10, 0.30, 0.50]
SIGMAS = [0.0, 0.02, 0.05, 0.08, 0.10, 0.15]
N_SEEDS = 3
NORM = "degree_linear"
C_CLIP = (0.1, 5.0)


def h_from_series(v_hist: np.ndarray, fn) -> float:
    n_steps = v_hist.shape[0]
    tail_start = int(n_steps * (1 - TAIL_FRAC))
    return float(np.mean([fn(v_hist[k]) for k in range(tail_start, n_steps)]))


def main() -> int:
    print("=" * 84)
    print("CMOS-realistic mismatch sweep (KIMI critique #2)")
    print(f"BA m={M_BA} N={N}, norm={NORM}")
    print(f"  eta in {ETAS}, sigma_C in {SIGMAS}, seeds={N_SEEDS}")
    print(f"  total runs: {len(ETAS) * len(SIGMAS) * N_SEEDS}")
    print(f"  metric: 100-bin continuous entropy (primary) + 5-bin KIMI")
    print("=" * 84)

    adj = make_ba(N, m=M_BA, seed=SEED)
    rng_init = np.random.RandomState(SEED)
    init_v = rng_init.uniform(-1.0, 1.0, N)
    scale = compute_scale_factors(adj, NORM)

    H_cont = np.zeros((len(ETAS), len(SIGMAS), N_SEEDS))
    H_cog = np.zeros_like(H_cont)
    rows_csv = [("eta", "sigma_C", "seed", "H_continuous", "H_cognitive_kimi")]
    t0 = time.time()
    total_runs = len(ETAS) * len(SIGMAS) * N_SEEDS
    run_idx = 0

    for i, eta in enumerate(ETAS):
        for j, sigma in enumerate(SIGMAS):
            for k in range(N_SEEDS):
                run_idx += 1
                if sigma == 0.0:
                    c_vals = np.ones(N)
                else:
                    mc_rng = np.random.RandomState(
                        SEED + 2000 + k * 17 + int(sigma * 1000))
                    c_vals = np.clip(
                        mc_rng.normal(1.0, sigma, N), *C_CLIP)
                tag = (f"cmos_BA_m{M_BA}_N{N}_{NORM}_"
                       f"eta{eta:g}_sig{sigma:g}_s{k}")
                dat_path = RESULTS / f"{tag}.dat"
                if dat_path.exists():
                    t_run = 0.0  # cached
                else:
                    path = generate_netlist(adj, scale, init_v, T_END, DT, tag,
                                            noise_amp=eta, c_values=c_vals)
                    t_run = run_ngspice(path)
                _, v_sp = parse_wrdata(dat_path)
                Hc = h_from_series(v_sp, calculate_continuous_entropy)
                Hk = h_from_series(v_sp, calculate_cognitive_entropy)
                H_cont[i, j, k] = Hc
                H_cog[i, j, k] = Hk
                rows_csv.append((eta, sigma, k, Hc, Hk))
                cache_note = " [cached]" if t_run == 0.0 else ""
                print(f"  [{run_idx:>3}/{total_runs}] eta={eta:g} sig={sigma:g} s={k}  "
                      f"Hc={Hc:.3f} Hk={Hk:.3f}  ({t_run:.1f}s){cache_note}")

    elapsed = time.time() - t0
    print(f"\n  total elapsed: {elapsed:.1f}s")

    # --- CSV ---
    csv_path = FIGURES / "spice_mismatch_cmos.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        for row in rows_csv:
            f.write(",".join(str(x) for x in row) + "\n")
    print(f"  CSV: {csv_path}")

    # --- Figure: heatmap (100-bin) + escape curves (both metrics) ---
    Hc_mean = H_cont.mean(axis=2)
    Hc_std = H_cont.std(axis=2)
    Hk_mean = H_cog.mean(axis=2)
    Hk_std = H_cog.std(axis=2)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5),
                             gridspec_kw={"width_ratios": [1.2, 1.0]})

    ax = axes[0]
    vmax = max(0.5, Hc_mean.max() * 1.05)
    im = ax.imshow(Hc_mean, aspect="auto", origin="lower", cmap="viridis",
                   vmin=0, vmax=vmax)
    ax.set_xticks(range(len(SIGMAS)))
    ax.set_xticklabels([f"{s:g}" for s in SIGMAS])
    ax.set_yticks(range(len(ETAS)))
    ax.set_yticklabels([f"{e:g}" for e in ETAS])
    ax.set_xlabel("Capacitor mismatch sigma_C (CMOS-realistic)", fontsize=11)
    ax.set_ylabel("Noise amplitude eta", fontsize=11)
    ax.set_title(
        f"H_continuous on dead zone (BA m={M_BA}, N={N}, {NORM})\n"
        f"sigma_C ceiling = 0.15 (aggressive analog CMOS)",
        fontsize=11,
    )
    for i in range(len(ETAS)):
        for j in range(len(SIGMAS)):
            txt = f"{Hc_mean[i,j]:.2f}\n+-{Hc_std[i,j]:.2f}"
            color = "white" if Hc_mean[i, j] < vmax * 0.6 else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    color=color, fontsize=9)
    fig.colorbar(im, ax=ax, label="H_continuous (bits)")

    ax2 = axes[1]
    for i, e in enumerate(ETAS):
        ax2.errorbar(SIGMAS, Hc_mean[i], yerr=Hc_std[i],
                     marker="o", capsize=3, lw=1.5, label=f"eta={e:g} (100-bin)")
    for i, e in enumerate(ETAS):
        ax2.errorbar(SIGMAS, Hk_mean[i], yerr=Hk_std[i],
                     marker="s", ls=":", capsize=2, lw=1.0, alpha=0.6,
                     label=f"eta={e:g} (5-bin KIMI)")
    ax2.set_xlabel("sigma_C (CMOS-realistic)", fontsize=11)
    ax2.set_ylabel("H_stable", fontsize=11)
    ax2.set_title("Escape curve in CMOS-realistic range", fontsize=11)
    ax2.grid(alpha=0.3)
    ax2.legend(loc="best", fontsize=8, ncol=2)
    ax2.axhline(0, color="black", lw=0.5)

    fig.suptitle(
        "SPICE P4.19 CMOS-realistic mismatch sweep (KIMI rebuttal #2)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out = FIGURES / "spice_mismatch_cmos.png"
    fig.savefig(out, dpi=140)
    print(f"  Figure: {out}")

    # --- Verdict ---
    print()
    print("=" * 84)
    # Dead zone reference: (eta=0.1, sigma=0)
    H_dead_100 = Hc_mean[0, 0]
    H_dead_5 = Hk_mean[0, 0]
    # Best CMOS-realistic cell
    iH = np.unravel_index(Hc_mean.argmax(), Hc_mean.shape)
    print(f"H_100bin max within CMOS range: {Hc_mean[iH]:.3f} at "
          f"eta={ETAS[iH[0]]:g}, sigma_C={SIGMAS[iH[1]]:g}")
    print(f"H_5bin  max within CMOS range: {Hk_mean[iH]:.3f}")
    print(f"Dead zone (eta=0.1, sig=0): Hc={H_dead_100:.3f}  Hk={H_dead_5:.3f}")
    print()

    # Three regime diagnostic at sigma=0.15 (max CMOS-realistic)
    j_max = len(SIGMAS) - 1  # sigma=0.15
    H_low = Hc_mean[0, j_max]   # eta=0.10
    H_mid = Hc_mean[1, j_max]   # eta=0.30
    H_hi = Hc_mean[2, j_max]    # eta=0.50
    print(f"Escape at sigma_C=0.15 (CMOS ceiling):")
    print(f"  eta=0.10 (sub-threshold): Hc={H_low:.3f}")
    print(f"  eta=0.30 (stochastic res): Hc={H_mid:.3f}")
    print(f"  eta=0.50 (noise-dominated): Hc={H_hi:.3f}")
    print()

    escape_at_cmos = Hc_mean[1:, -1].max()  # best at max CMOS sigma, eta>=0.3
    if escape_at_cmos > 3.0:
        print(f"=> STRONG escape even at CMOS-realistic sigma_C<=0.15 "
              f"(H_100bin={escape_at_cmos:.2f} bits >> 2.32 ceiling).")
        print(f"   Paper B argument survives KIMI physicality objection.")
    elif escape_at_cmos > 1.5:
        print(f"=> Partial escape at CMOS-realistic sigma_C<=0.15 "
              f"(H_100bin={escape_at_cmos:.2f} bits).")
        print(f"   Publishable but Paper B needs weaker narrative.")
    else:
        print(f"=> Weak escape at CMOS-realistic sigma_C<=0.15 "
              f"(H_100bin={escape_at_cmos:.2f}).")
        print(f"   Paper B 'complete escape' claim doesn't hold in physical range.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
