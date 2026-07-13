#!/usr/bin/env python3
"""
Head-to-head: spectral vs degree_linear on BA m=5 (dead zone configuration).

Parameters:
  - BA m=5, N=100
  - 10 seeds: [42, 123, 777, 17, 256, 1337, 99, 314, 2024, 888]
  - 3000 steps, I_stim=0.0, cold_start=True, heretic_ratio=0.15
  - H_cont: 100-bin continuous entropy, recorded from last 25% of steps
  - Also reports algebraic connectivity lambda_2 per seed
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from mem4ristor.core import Mem4Network
from mem4ristor.metrics import calculate_continuous_entropy
from limit02_topology_sweep import make_ba

N = 100
STEPS = 3000
TAIL_FRAC = 0.25
SEEDS = [42, 123, 777, 17, 256, 1337, 99, 314, 2024, 888]
NORMS = ["spectral", "degree_linear"]
M = 5
HERETIC_RATIO = 0.15
I_STIM = 0.0


def fiedler_value(adj: np.ndarray) -> float:
    from scipy.linalg import eigh
    deg = adj.sum(axis=1)
    L = np.diag(deg) - adj
    vals = eigh(L, eigvals_only=True)
    return float(vals[1]) if len(vals) > 1 else 0.0


def run_one(adj: np.ndarray, norm: str, seed: int) -> list[float]:
    """Run Mem4Network and return H_cont values from the tail 25% of steps."""
    net = Mem4Network(
        size=10,
        heretic_ratio=HERETIC_RATIO,
        seed=seed,
        adjacency_matrix=adj.copy(),
        coupling_norm=norm,
        cold_start=True,
    )
    h_trace = []
    for step in range(STEPS):
        net.step(I_stimulus=I_STIM)
        # record every step (cheap; we'll slice the tail)
        h_trace.append(calculate_continuous_entropy(net.v, bins=100))

    tail_start = int(len(h_trace) * (1 - TAIL_FRAC))
    return h_trace[tail_start:]


def main() -> int:
    print("=" * 80)
    print(f"BA m={M} | N={N} | steps={STEPS} | I_stim={I_STIM} | "
          f"cold_start=True | heretic_ratio={HERETIC_RATIO}")
    print(f"Seeds: {SEEDS}")
    print("=" * 80)

    results: dict[str, list[float]] = {n: [] for n in NORMS}
    lambda2_per_seed: dict[int, float] = {}

    t0 = time.time()
    for seed in SEEDS:
        adj = make_ba(N, M, seed)
        l2 = fiedler_value(adj)
        lambda2_per_seed[seed] = l2

        for norm in NORMS:
            tail = run_one(adj, norm, seed)
            h_mean_tail = float(np.mean(tail))
            results[norm].append(h_mean_tail)
            print(f"  seed={seed:<5} norm={norm:<15} λ₂={l2:.4f}  "
                  f"H_cont_tail_mean={h_mean_tail:.4f}  "
                  f"(tail n={len(tail)})")

    elapsed = time.time() - t0

    # ── Summary ────────────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    for norm in NORMS:
        vals = np.array(results[norm])
        print(f"\n{norm}:")
        print(f"  H_cont per seed : {[f'{v:.4f}' for v in vals]}")
        print(f"  mean ± std       : {vals.mean():.4f} ± {vals.std():.4f}")
        print(f"  min / max        : {vals.min():.4f} / {vals.max():.4f}")

    print()
    print("Algebraic connectivity λ₂ per seed:")
    for seed in SEEDS:
        print(f"  seed={seed:<5}  λ₂={lambda2_per_seed[seed]:.4f}")
    l2_vals = np.array(list(lambda2_per_seed.values()))
    print(f"  λ₂ mean ± std: {l2_vals.mean():.4f} ± {l2_vals.std():.4f}")

    print()
    print(f"Elapsed: {elapsed:.1f}s")

    # ── Raw H_cont values (all tail samples) ──────────────────────────────
    print()
    print("RAW H_cont per-seed (mean over tail-25%):")
    print(f"{'seed':<8}  {'spectral':>12}  {'degree_linear':>14}  {'Δ (spec-deg)':>14}  {'λ₂':>10}")
    print("-" * 65)
    for i, seed in enumerate(SEEDS):
        sp = results["spectral"][i]
        dl = results["degree_linear"][i]
        l2 = lambda2_per_seed[seed]
        print(f"{seed:<8}  {sp:>12.4f}  {dl:>14.4f}  {sp-dl:>+14.4f}  {l2:>10.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
