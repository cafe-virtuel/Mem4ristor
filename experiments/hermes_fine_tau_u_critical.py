#!/usr/bin/env python3
"""
Hermes Research — Fine τ_u scan at the critical boundary (BA m=4, λ₂≈2.21)
=============================================================================
Preprint notes: τ_u=10 sits "precisely at the bifurcation edge" (Section 6.4).
BA m=4 has λ₂=2.21 — just below λ₂_crit=2.31 — making it a transitional topology.
Question: Is m=4 sensitive to τ_u? Can small τ_u shifts move it toward functional
or dead-zone regime? Fine scan τ_u ∈ {2, 4, 6, 8, 10, 12, 15, 20, 30}.

Plan:
  - BA m=4 (N=100), λ₂≈2.21, degree_linear coupling
  - τ_u sweep: 9 values, 5 seeds, 3000 steps, last 25%, I_stim=0.5
  - Metrics: H_stable (100-bin), pairwise synchrony
  - Also run m=3 (functional, λ₂=1.41) and m=5 (dead zone, λ₂=2.99) for contrast

Result + interpretation written to D:/ANTIGRAVITY/.brain/hermes_research_latest.md
"""

import sys, os, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mem4ristor.core import Mem4Network
from mem4ristor.graph_utils import make_ba
from mem4ristor.metrics import calculate_continuous_entropy

# ── Config ──────────────────────────────────────────────────────────────────
STEPS   = 3000
WARMUP  = int(STEPS * 0.75)
SEEDS   = [42, 123, 777, 17, 256]
TAU_US  = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0]
M_VALUES = [3, 4, 5]  # functional / marginal / dead zone
N_NODES = 100
I_STIM  = 0.5

print("=" * 70)
print("  Fine τ_u scan at critical boundary — BA m=3/4/5")
print(f"  Steps={STEPS}, last 25%, n={len(SEEDS)} seeds, I_stim={I_STIM}")
print("=" * 70)

results = {}

for m in M_VALUES:
    label = {3: "FUNCTIONAL", 4: "MARGINAL", 5: "DEAD ZONE"}[m]
    λ2_val = {3: 1.41, 4: 2.21, 5: 2.99}[m]
    print(f"\n>>> BA m={m} [{label}] λ₂≈{λ2_val}")
    print(f"  {'τ_u':>6}  {'H_mean':>8}  {'H_std':>7}  {'sync':>8}")
    print(f"  {'-'*40}")
    results[m] = {}

    for tau_u in TAU_US:
        H_runs, sync_runs = [], []

        for seed in SEEDS:
            adj = make_ba(N_NODES, m, seed)
            net = Mem4Network(
                adjacency_matrix=adj,
                heretic_ratio=0.15,
                coupling_norm='degree_linear',
                seed=seed,
            )
            net.model.cfg['doubt']['tau_u'] = tau_u

            v_hist = []
            for step in range(STEPS):
                net.step(I_stimulus=I_STIM)
                if step >= WARMUP:
                    v_hist.append(net.model.v.copy())

            H_vals = [calculate_continuous_entropy(v) for v in v_hist]
            H_runs.append(np.mean(H_vals))

            V = np.array(v_hist)
            if V.shape[0] > 1 and V.shape[1] > 1:
                corr = np.corrcoef(V.T)
                triu = np.triu_indices(N_NODES, k=1)
                sync_runs.append(np.mean(corr[triu]))
            else:
                sync_runs.append(0.0)

        H_m = np.mean(H_runs)
        H_s = np.std(H_runs)
        sy_m = np.mean(sync_runs)
        results[m][tau_u] = (H_m, H_s, sy_m)
        print(f"  {tau_u:>6.1f}  {H_m:>8.3f}  ±{H_s:<6.3f}  {sy_m:>8.3f}")

# ── Summary ─────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("SUMMARY:")
print(f"  BA m=3 [FUNCTIONAL, λ₂=1.41]:  baseline τ_u=10 -> H={results[3][10.0][0]:.3f}")
print(f"  BA m=5 [DEAD ZONE, λ₂=2.99]:   baseline τ_u=10 -> H={results[5][10.0][0]:.3f}")
print(f"\n  BA m=4 [MARGINAL, λ₂=2.21]:")
for tau_u in TAU_US:
    H = results[4][tau_u][0]
    print(f"    τ_u={tau_u:5.1f}  H={H:.3f}")
best_tau = max(results[4].items(), key=lambda x: x[1][0])
print(f"\n  Best for m=4: τ_u={best_tau[0]} -> H={best_tau[1][0]:.3f}")
print(f"  Default τ_u=10 -> H={results[4][10.0][0]:.3f}")
print(f"{'='*70}")