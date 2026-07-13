"""
Hermes Research — τ_u sensitivity probe near the spectral dead zone
=====================================================================
Question: How does the doubt time constant τ_u affect diversity preservation
           on BA networks at the boundary of the dead zone (m=3 vs m=5)?

Plan:
  - Sweep τ_u ∈ {1, 5, 10, 20, 50, 100} on BA m=3 (functional) and BA m=5 (dead zone)
  - Measure H_stable (100-bin continuous entropy) and pairwise synchrony
  - 3 seeds, 3000 steps, last 25%, I_stim=0.5, degree_linear coupling

Interpretation target: Does faster doubt (smaller τ_u) rescue diversity on m=5?
// @DOUBT: tau_u sweep is coarse (6 values); fine scan near bifurcation edge needed.
"""

import sys, os, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from mem4ristor.core import Mem4Network
from mem4ristor.metrics import calculate_continuous_entropy

# ── helpers ──────────────────────────────────────────────────────────────────

def make_ba_adjacency(n: int, m: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    adj = np.zeros((n, n), dtype=float)
    for i in range(m + 1):
        for j in range(i + 1, m + 1):
            adj[i, j] = adj[j, i] = 1.0
    for i in range(m + 1, n):
        targets = set()
        while len(targets) < m:
            probs = [(adj[i, j] if j < i else adj[j, i]) + 1.0 for j in range(i)]
            total = sum(probs)
            r = rng.rand() * total
            cumsum, s = 0.0, 0
            for j, p in enumerate(probs):
                cumsum += p
                if cumsum >= r:
                    s = j
                    break
            targets.add(s)
        for j in targets:
            adj[i, j] = adj[j, i] = 1.0
    return adj

# ── experiment config ─────────────────────────────────────────────────────────

STEPS   = 3000
WARMUP  = int(STEPS * 0.75)
SEEDS   = [42, 123, 777]
TAU_US  = [1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
M_VALUES = [3, 5]   # m=3 (functional) vs m=5 (dead zone)
N_NODES = 100

print(f"{'='*65}")
print(f"  τ_u PROBE — Dead zone boundary (BA m=3 vs m=5)")
print(f"  {STEPS} steps, last 25%, n={len(SEEDS)} seeds")
print(f"{'='*65}")

results = {}

for m in M_VALUES:
    print(f"\n>>> BA m={m}  {'[DEAD ZONE]' if m >= 5 else '[FUNCTIONAL]'}")
    print(f"  {'τ_u':>8}  {'H_mean':>10}  {'H_std':>7}  {'sync_mean':>10}  {'sync_std':>8}")
    print(f"  {'-'*55}")
    results[m] = {}

    for tau_u in TAU_US:
        H_runs, sync_runs = [], []

        for seed in SEEDS:
            adj = make_ba_adjacency(N_NODES, m, seed)
            net = Mem4Network(
                adjacency_matrix=adj,
                heretic_ratio=0.15,
                coupling_norm='degree_linear',
                seed=seed,
            )
            # Override tau_u
            net.model.cfg['doubt']['tau_u'] = tau_u

            v_hist = []
            for step in range(STEPS):
                net.step(I_stimulus=0.5)
                if step >= WARMUP:
                    v_hist.append(net.model.v.copy())

            # H_stable: mean of per-step continuous entropies
            H_vals = [calculate_continuous_entropy(v) for v in v_hist]
            H_runs.append(np.mean(H_vals))

            # Pairwise synchrony
            V = np.array(v_hist)  # (T, N)
            if V.shape[0] > 1 and V.shape[1] > 1:
                corr_matrix = np.corrcoef(V.T)
                triu_idx = np.triu_indices(N_NODES, k=1)
                correlations = corr_matrix[triu_idx]
                sync_runs.append(np.mean(correlations))
            else:
                sync_runs.append(0.0)

        H_mean = np.mean(H_runs)
        H_std  = np.std(H_runs)
        sync_m = np.mean(sync_runs)
        sync_s = np.std(sync_runs)
        results[m][tau_u] = (H_mean, H_std, sync_m, sync_s)
        print(f"  {tau_u:>8.1f}  {H_mean:>10.3f}  ±{H_std:<6.3f}  {sync_m:>10.3f}  ±{sync_s:<7.3f}")

print(f"\n{'='*65}")
print("KEY FINDINGS:")
baseline_tau = 10.0
for m in M_VALUES:
    baseline_H = results[m].get(baseline_tau, (None, None, None, None))[0]
    best = max(results[m].items(), key=lambda x: x[1][0])
    delta = (best[1][0] - baseline_H) if baseline_H else 0
    print(f"  BA m={m}: τ_u={baseline_tau} -> H={baseline_H:.3f}  |  best τ_u={best[0]} -> H={best[1][0]:.3f}  (Δ={delta:+.3f})")
print(f"{'='*65}")
