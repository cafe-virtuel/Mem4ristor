"""
V5 FINAL — D(u)=0.50*u + alpha_meta=-4.0 : Best Combination

Campaign summary (Session 011):
  - D(u)=0.50*u discovered: adaptive coupling lowers FUNCTIONAL threshold from m>=7 to m>=6
  - Metacognition (alpha_meta) amplifies D(u) across ALL topologies
  - Optimal alpha_meta = -4.0 (freezes w, epsilon_i -> epsilon_min=0.01)
  - Compartmentalization HURTS dead-zone topologies (m>=7)
  - Best config: D(u)=0.50*u + alpha_meta=-4.0 (no compartments)

Results (10 seeds, N=100, 3000 steps):

BA m=3 (FUNCTIONAL):
  V4 baseline:                H=3.63, Sync=0.000
  D(u)+alpha=-4.0 (best):   H=5.13, Sync=0.006   (+1.50 vs V4)

BA m=5 (CRITICAL):
  V4 baseline:                H=3.21, Sync=-0.002
  D(u)+alpha=-4.0 (best):   H=5.11, Sync=0.011   (+1.91 vs V4)

BA m=7 (DEAD-ZONE):
  V4 baseline:                H=2.87, Sync=-0.002
  D(u)+alpha=-4.0 (best):   H=4.34, Sync=-0.005   (+1.46 vs V4)

BA m=10 (DEAD-ZONE):
  V4 baseline:                H=2.48, Sync=-0.004
  D(u)+alpha=-4.0 (best):   H=3.37, Sync=-0.007   (+0.89 vs V4)

Key insight: alpha_meta=-4.0 effectively disables w plasticity
(epsilon_i = epsilon * (1 + -4.0 * (0.5 - u)) -> approaches epsilon_min=0.01).
Freezing w while D(u) adapts is the sweet spot. Compartments add
no value and hurt dead-zone topologies.

Reproduction: python experiments/p2_v5_final_best.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import numpy as np
import csv
from mem4ristor.core import Mem4Network
from mem4ristor.graph_utils import make_ba
from mem4ristor.metrics import calculate_continuous_entropy, calculate_temporal_lz_complexity

N = 100
STEPS = 3000
WARMUP = int(STEPS * 0.75)
I_STIM = 0.5
SEEDS = [42, 123, 777, 1234, 5678, 9999, 314, 2718, 8765, 4321]

def compute_synchrony(v_history):
    n = v_history.shape[1]
    if n < 2:
        return 0.0
    v_c = v_history - v_history.mean(axis=0, keepdims=True)
    v_s = v_history.std(axis=0, keepdims=True)
    v_s[v_s == 0] = 1.0
    corr = np.dot(v_c.T, v_c) / (v_c.shape[0] - 1)
    std_p = np.dot(v_s.T, v_s)
    corr = corr / (std_p + 1e-12)
    idx = np.triu_indices(n, k=1)
    return float(np.mean(corr[idx])) if idx[0].size > 0 else 0.0


def run_one(m, seed, D_val, alpha_meta, comp_en=False, K=3, gamma=0.10):
    adj = make_ba(N, m, seed)
    net = Mem4Network(adjacency_matrix=adj, coupling_norm='degree_linear', seed=seed)
    net.model.cfg['coupling']['D'] = float(D_val)
    if D_val != 0.50:
        net.model.D_eff = D_val / float(np.sqrt(N))
    net.model.cfg['metacognitive'] = {
        'enabled': True,
        'alpha_meta': float(alpha_meta),
        'epsilon_min': 0.01
    }
    net.model.cfg['compartments'] = {'enabled': comp_en, 'K': K, 'gamma': gamma, 'mode': 'full'}
    H_list, v_hist = [], []
    for step in range(STEPS):
        net.step(I_stimulus=I_STIM)
        if D_val == 0.50:
            net.model.D_eff = (0.50 * net.model.u.mean()) / float(np.sqrt(N))
        if step >= WARMUP:
            H_list.append(calculate_continuous_entropy(net.model.v))
            v_hist.append(net.model.v.copy())
    v_arr = np.array(v_hist)
    return {
        'H_cont': np.mean(H_list),
        'H_std': np.std(H_list),
        'synchrony': compute_synchrony(v_arr),
        'lz_complexity': calculate_temporal_lz_complexity(v_arr, n_bins=5),
        'u_mean': float(net.model.u.mean()),
    }


CONFIGS = [
    ("V4 static D=0.15",    0.15, None,  False, 0),
    ("D(u) alpha=-0.5",     0.50, -0.5, False, 3),
    ("D(u) alpha=-4.0",     0.50, -4.0, False, 3),
    ("D(u) alpha=-4.0 K=3", 0.50, -4.0, True,  3),
]
M_VALUES = [3, 5, 7, 10]

print("=" * 78)
print("  V5 FINAL: D(u)=0.50*u + alpha_meta Sweep  |  10 seeds  |  N=100")
print("=" * 78)

results = []
for m in M_VALUES:
    regime = "FUNCTIONAL" if m <= 4 else ("CRITICAL" if m <= 6 else "DEAD-ZONE")
    print(f"\n  BA m={m} ({regime}):")
    print(f"  {'Config':<20} {'H_cont':>10} {'H_std':>8} {'Sync':>8} {'LZ':>8} {'u':>8}")
    print(f"  {'-' * 66}")
    for label, D_val, alpha, comp_en, K_use in CONFIGS:
        if D_val == 0.15:
            # V4 baseline
            H_runs, S_runs, L_runs = [], [], []
            for seed in SEEDS:
                adj = make_ba(N, m, seed)
                net = Mem4Network(adjacency_matrix=adj, coupling_norm='degree_linear', seed=seed)
                net.model.cfg['coupling']['D'] = 0.15
                net.model.D_eff = 0.15 / float(np.sqrt(N))
                H_list, v_hist = [], []
                for step in range(STEPS):
                    net.step(I_stimulus=I_STIM)
                    if step >= WARMUP:
                        H_list.append(calculate_continuous_entropy(net.model.v))
                        v_hist.append(net.model.v.copy())
                v_arr = np.array(v_hist)
                H_runs.append(np.mean(H_list))
                S_runs.append(compute_synchrony(v_arr))
                L_runs.append(calculate_temporal_lz_complexity(v_arr, n_bins=5))
            r = {'m': m, 'label': label, 'H_cont': np.mean(H_runs),
                 'H_std': np.std(H_runs), 'synchrony': np.mean(S_runs),
                 'lz_complexity': np.mean(L_runs), 'u_mean': 0.0}
        else:
            H_runs, S_runs, L_runs = [], [], []
            for seed in SEEDS:
                r_run = run_one(m, seed, D_val, alpha, comp_en, K=K_use)
                H_runs.append(r_run['H_cont'])
                S_runs.append(r_run['synchrony'])
                L_runs.append(r_run['lz_complexity'])
            r = {'m': m, 'label': label, 'H_cont': np.mean(H_runs),
                 'H_std': np.std(H_runs), 'synchrony': np.mean(S_runs),
                 'lz_complexity': np.mean(L_runs),
                 'u_mean': r_run['u_mean']}
        results.append(r)
        flag = "  SYNC!" if abs(r['synchrony']) > 0.05 else ""
        print(f"  {label:<20} {r['H_cont']:>10.3f} {r['H_std']:>8.3f} "
              f"{r['synchrony']:>8.4f} {r['lz_complexity']:>8.4f}{flag}")

# Save CSV
out_dir = os.path.join(os.path.dirname(__file__), '../figures')
os.makedirs(out_dir, exist_ok=True)
csv_path = os.path.join(out_dir, 'p2_v5_final_best.csv')
with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['m', 'label', 'H_cont', 'H_std', 'synchrony', 'lz_complexity', 'u_mean'])
    w.writeheader()
    w.writerows(results)
print(f"\n  CSV -> {csv_path}")
print(f"\n  Reproduction: python experiments/p2_v5_final_best.py")