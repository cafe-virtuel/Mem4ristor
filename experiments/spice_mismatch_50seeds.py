#!/usr/bin/env python3
"""
50-seed Monte Carlo validation at 3 critical points (KIMI critique #3).

Tests statistical robustness of the escape claims by running N_SEEDS=50
independent ngspice trials at three carefully chosen (eta, sigma_C) cells:

  A. Dead zone baseline        : (eta=0.10, sigma_C=0.00)
  B. Noise-only escape         : (eta=0.50, sigma_C=0.00)
  C. Noise + CMOS mismatch     : (eta=0.50, sigma_C=0.10)

Each trial gets a fresh ngspice run, so trnoise() realizations differ
seed-to-seed (ngspice uses a nondeterministic internal seed). Sigma
mismatch masks vary via numpy seed.

Total: 3 x 50 = 150 runs (~17 min at ~7s/run).

Primary outputs:
  - mean H, 95% CI (+/- 1.96*std/sqrt(N)) for each point
  - Welch t-test for H_A vs H_B (does noise alone escape?)
  - Welch t-test for H_B vs H_C (does CMOS mismatch add beyond noise?)
  - figures/spice_50seeds_validation.png (violin + mean +/- CI)
  - figures/spice_50seeds_validation.csv (raw 150 rows)

Created: 2026-04-19 (Claude Opus 4.7, KIMI response track)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

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

N_SEEDS = 50
NORM = "degree_linear"
C_CLIP = (0.1, 5.0)

POINTS = [
    ("A_dead_zone",    0.10, 0.00),
    ("B_noise_only",   0.50, 0.00),
    ("C_noise_cmos",   0.50, 0.10),
]


def h_from_series(v_hist: np.ndarray, fn) -> float:
    n_steps = v_hist.shape[0]
    tail_start = int(n_steps * (1 - TAIL_FRAC))
    return float(np.mean([fn(v_hist[k]) for k in range(tail_start, n_steps)]))


def ci95(values: np.ndarray) -> float:
    return 1.96 * values.std(ddof=1) / np.sqrt(len(values))


def main() -> int:
    print("=" * 84)
    print(f"50-seed Monte Carlo validation (KIMI critique #3)")
    print(f"BA m={M_BA} N={N}, norm={NORM}, N_SEEDS={N_SEEDS}")
    print("  A: dead zone baseline     (eta=0.10, sigma_C=0.00)")
    print("  B: noise-only escape      (eta=0.50, sigma_C=0.00)")
    print("  C: noise + CMOS mismatch  (eta=0.50, sigma_C=0.10)")
    print(f"  total runs: {len(POINTS) * N_SEEDS}")
    print("=" * 84)

    adj = make_ba(N, m=M_BA, seed=SEED)
    rng_init = np.random.RandomState(SEED)
    init_v = rng_init.uniform(-1.0, 1.0, N)
    scale = compute_scale_factors(adj, NORM)

    H_cont = {name: np.zeros(N_SEEDS) for name, _, _ in POINTS}
    H_cog = {name: np.zeros(N_SEEDS) for name, _, _ in POINTS}
    rows_csv = [("point", "eta", "sigma_C", "seed", "H_continuous", "H_cognitive_kimi")]
    t0 = time.time()
    total_runs = len(POINTS) * N_SEEDS
    run_idx = 0

    for name, eta, sigma in POINTS:
        for k in range(N_SEEDS):
            run_idx += 1
            if sigma == 0.0:
                c_vals = np.ones(N)
            else:
                mc_rng = np.random.RandomState(SEED + 3000 + k * 31 + int(sigma * 1000))
                c_vals = np.clip(mc_rng.normal(1.0, sigma, N), *C_CLIP)
            tag = (f"mc50_{name}_BA_m{M_BA}_N{N}_{NORM}_"
                   f"eta{eta:g}_sig{sigma:g}_s{k}")
            dat_path = RESULTS / f"{tag}.dat"
            if dat_path.exists():
                t_run = 0.0
            else:
                path = generate_netlist(adj, scale, init_v, T_END, DT, tag,
                                        noise_amp=eta, c_values=c_vals)
                t_run = run_ngspice(path)
            _, v_sp = parse_wrdata(dat_path)
            Hc = h_from_series(v_sp, calculate_continuous_entropy)
            Hk = h_from_series(v_sp, calculate_cognitive_entropy)
            H_cont[name][k] = Hc
            H_cog[name][k] = Hk
            rows_csv.append((name, eta, sigma, k, Hc, Hk))
            if run_idx % 10 == 0 or t_run > 0:
                cache_note = " [cached]" if t_run == 0.0 else ""
                print(f"  [{run_idx:>3}/{total_runs}] {name} s={k}  "
                      f"Hc={Hc:.3f} Hk={Hk:.3f}  ({t_run:.1f}s){cache_note}")

    elapsed = time.time() - t0
    print(f"\n  total elapsed: {elapsed:.1f}s")

    # --- CSV ---
    csv_path = FIGURES / "spice_50seeds_validation.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        for row in rows_csv:
            f.write(",".join(str(x) for x in row) + "\n")
    print(f"  CSV: {csv_path}")

    # --- Statistical summary ---
    print()
    print("=" * 84)
    print("STATISTICAL SUMMARY (100-bin continuous entropy)")
    print("=" * 84)
    print(f"{'Point':<20} {'N':>3} {'mean':>7} {'std':>7} "
          f"{'95% CI':>12} {'min':>6} {'max':>6}")
    for name, eta, sigma in POINTS:
        vals = H_cont[name]
        ci = ci95(vals)
        print(f"{name:<20} {len(vals):>3} {vals.mean():>7.3f} {vals.std(ddof=1):>7.3f} "
              f"+-{ci:>6.3f}     {vals.min():>6.3f} {vals.max():>6.3f}")

    print()
    print("WELCH T-TESTS")
    A = H_cont["A_dead_zone"]
    B = H_cont["B_noise_only"]
    C = H_cont["C_noise_cmos"]

    t_AB, p_AB = stats.ttest_ind(A, B, equal_var=False)
    cohen_AB = (B.mean() - A.mean()) / np.sqrt((A.var(ddof=1) + B.var(ddof=1)) / 2)
    print(f"  A vs B (does noise alone escape?)")
    print(f"    t={t_AB:.3f}  p={p_AB:.3e}  Cohen's d={cohen_AB:.2f}")
    verdict_AB = ("YES" if p_AB < 0.001 and abs(cohen_AB) > 0.8
                  else "weak" if p_AB < 0.05 else "NO")
    print(f"    => noise-only escape is {verdict_AB}")

    t_BC, p_BC = stats.ttest_ind(B, C, equal_var=False)
    cohen_BC = (C.mean() - B.mean()) / np.sqrt((B.var(ddof=1) + C.var(ddof=1)) / 2)
    print(f"  B vs C (does CMOS mismatch add beyond noise?)")
    print(f"    t={t_BC:.3f}  p={p_BC:.3e}  Cohen's d={cohen_BC:.2f}")
    verdict_BC = ("YES" if p_BC < 0.001 and abs(cohen_BC) > 0.5
                  else "weak" if p_BC < 0.05 else "NO")
    print(f"    => CMOS mismatch adds {verdict_BC} beyond noise alone")

    # --- Figure ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: violin
    ax = axes[0]
    data = [H_cont[name] for name, _, _ in POINTS]
    labels = [f"A\nη=0.1\nσ=0" , f"B\nη=0.5\nσ=0", f"C\nη=0.5\nσ=0.10"]
    parts = ax.violinplot(data, showmeans=True, showextrema=True, showmedians=False)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(["tab:red", "tab:orange", "tab:green"][i])
        pc.set_alpha(0.5)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(labels)
    ax.set_ylabel("H_continuous (bits)", fontsize=11)
    ax.set_title(f"Monte Carlo N={N_SEEDS} seeds per point", fontsize=11)
    ax.grid(alpha=0.3, axis="y")
    for i, name in enumerate(n for n, _, _ in POINTS):
        vals = H_cont[name]
        ci = ci95(vals)
        ax.errorbar(i + 1, vals.mean(), yerr=ci, fmt="ko", capsize=5, lw=2)

    # Right: p-values + Cohen's d annotated
    ax2 = axes[1]
    ax2.axis("off")
    txt = f"""STATISTICAL TESTS

  N = {N_SEEDS} seeds per cell
  metric: 100-bin continuous entropy

  Mean +- 95% CI:
    A (dead zone, eta=0.10, sigma=0):   {A.mean():.3f} +- {ci95(A):.3f} bits
    B (noise-only, eta=0.50, sigma=0):  {B.mean():.3f} +- {ci95(B):.3f} bits
    C (noise+CMOS, eta=0.5, sigma=0.1): {C.mean():.3f} +- {ci95(C):.3f} bits

  Welch t-tests:

    A vs B (noise-only escape):
      t = {t_AB:.2f}, p = {p_AB:.2e}
      Cohen's d = {cohen_AB:.2f}  (>|0.8| = huge)
      => escape is {verdict_AB}

    B vs C (CMOS adds beyond noise):
      t = {t_BC:.2f}, p = {p_BC:.2e}
      Cohen's d = {cohen_BC:.2f}  (|0.5| = medium)
      => mismatch effect is {verdict_BC}
"""
    ax2.text(0, 1, txt, fontfamily="monospace", fontsize=9,
             va="top", ha="left", transform=ax2.transAxes)

    fig.suptitle(
        f"P4.19 statistical validation: {N_SEEDS}-seed Monte Carlo (KIMI #3)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out = FIGURES / "spice_50seeds_validation.png"
    fig.savefig(out, dpi=140)
    print(f"\n  Figure: {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
