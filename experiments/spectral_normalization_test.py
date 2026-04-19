#!/usr/bin/env python3
"""
Test spectral normalization on the topological dead zone.

Hypothesis: Eigenvector centrality (1 / c_i) penalizes nodes whose
*global* influence on the network is largest, while degree_linear (1/deg)
only sees local adjacency. On dense / high-lambda2 graphs (BA m>=5,
ER p=0.12) every node has high degree, so degree-based norms cannot
discriminate. Spectral centrality is degenerate-safe and *should* spread
the burden across the actual influence hierarchy.

Compares uniform / degree_linear / spectral on the configurations that
collapse with both existing modes (the "dead zone" diagonal of the phase
diagram).

Created: 2026-04-19 (Claude Opus 4.7, P2 spectral track)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from mem4ristor.core import Mem4Network  # noqa: E402
from limit02_topology_sweep import (  # noqa: E402
    make_ba,
    make_erdos_renyi,
)

N = 100
STEPS = 3000
TAIL_FRAC = 0.25
SEEDS = [42, 123, 777]
NORMS = ["uniform", "degree_linear", "spectral"]


def fiedler_value(adj: np.ndarray) -> float:
    from scipy.linalg import eigh
    deg = adj.sum(axis=1)
    L = np.diag(deg) - adj
    vals = eigh(L, eigvals_only=True)
    return float(vals[1]) if vals.size > 1 else 0.0


def h_stable(adj: np.ndarray, norm: str, seed: int) -> float:
    net = Mem4Network(
        size=10, heretic_ratio=0.15, seed=seed,
        adjacency_matrix=adj.copy(), coupling_norm=norm,
    )
    trace = []
    for step in range(STEPS):
        net.step(I_stimulus=0.0)
        if step % 10 == 0:
            trace.append(net.calculate_entropy())
    tail = int(len(trace) * (1 - TAIL_FRAC))
    return float(np.mean(trace[tail:]))


# Dead-zone configurations from the phase diagram (lambda2 > 3 or near it)
DEAD_ZONE = [
    ("BA m=5",        make_ba,          {"m": 5}),
    ("BA m=8",        make_ba,          {"m": 8}),
    ("BA m=10",       make_ba,          {"m": 10}),
    ("ER p=0.12",     make_erdos_renyi, {"p": 0.12}),
    # Two "easy" controls that we already know:
    ("BA m=3 (ctrl)", make_ba,          {"m": 3}),
    ("BA m=1 (ctrl)", make_ba,          {"m": 1}),
]


def main() -> int:
    print("=" * 90)
    print("SPECTRAL NORMALIZATION on the dead zone")
    print(f"N={N}, steps={STEPS}, seeds={len(SEEDS)}")
    print("=" * 90)
    print(f"{'config':<18} {'lambda2':>9}   "
          f"{'uniform':>10} {'deg_lin':>10} {'spectral':>10}   {'best':>14}")
    print("-" * 90)

    rows = []
    t0 = time.time()
    for label, gen, kwargs in DEAD_ZONE:
        l2_seeds, h_by_norm = [], {n: [] for n in NORMS}
        for seed in SEEDS:
            adj = gen(N, seed=seed, **kwargs)
            l2_seeds.append(fiedler_value(adj))
            for n in NORMS:
                h_by_norm[n].append(h_stable(adj, n, seed))
        l2 = float(np.mean(l2_seeds))
        means = {n: float(np.mean(h_by_norm[n])) for n in NORMS}
        best_norm = max(NORMS, key=lambda n: means[n])
        best_tag = f"{best_norm} ({means[best_norm]:.3f})"
        rows.append((label, l2, means))
        print(f"{label:<18} {l2:>9.3f}   "
              f"{means['uniform']:>10.3f} {means['degree_linear']:>10.3f} "
              f"{means['spectral']:>10.3f}   {best_tag:>14}")

    elapsed = time.time() - t0
    print("-" * 90)
    print(f"Elapsed: {elapsed:.1f}s")

    # Verdict
    spectral_wins = sum(
        1 for _, _, m in rows if m["spectral"] >= max(m["uniform"], m["degree_linear"]) + 0.05
    )
    spectral_breaks_dead_zone = sum(
        1 for _, _, m in rows
        if m["spectral"] > 0.3 and max(m["uniform"], m["degree_linear"]) < 0.1
    )
    print()
    print(f"Spectral strictly wins on {spectral_wins}/{len(rows)} configurations.")
    print(f"Spectral breaks the dead zone on {spectral_breaks_dead_zone} configurations.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
