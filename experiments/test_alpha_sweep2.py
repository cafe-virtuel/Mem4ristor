
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
import numpy as np
from mem4ristor.core import Mem4Network
from mem4ristor.graph_utils import make_ba
from mem4ristor.metrics import calculate_continuous_entropy, calculate_temporal_lz_complexity

N=100; STEPS=3000; WARMUP=int(STEPS*0.75); I_STIM=0.5
SEEDS_10 = [42, 123, 777, 1234, 5678, 9999, 314, 2718, 8765, 4321]

def compute_synchrony(v_history):
    n = v_history.shape[1]
    if n < 2: return 0.0
    v_c = v_history - v_history.mean(axis=0, keepdims=True)
    v_s = v_history.std(axis=0, keepdims=True)
    v_s[v_s==0] = 1.0
    corr = np.dot(v_c.T, v_c)/(v_c.shape[0]-1)
    std_p = np.dot(v_s.T, v_s)
    corr = corr/(std_p+1e-12)
    idx = np.triu_indices(n, k=1)
    return float(np.mean(corr[idx])) if idx[0].size > 0 else 0.0

def run_one(m, seed, D_val, alpha_meta, comp_en=False):
    adj = make_ba(N, m, seed)
    net = Mem4Network(adjacency_matrix=adj, coupling_norm='degree_linear', seed=seed)
    net.model.cfg['coupling']['D'] = float(D_val)
    if D_val != 0.50:
        net.model.D_eff = D_val / np.sqrt(N)
    net.model.cfg['metacognitive'] = {
        'enabled': True,
        'alpha_meta': float(alpha_meta),
        'epsilon_min': 0.01
    }
    net.model.cfg['compartments'] = {'enabled': comp_en, 'K': 3, 'gamma': 0.10, 'mode': 'full'}
    H_list, v_hist = [], []
    for step in range(STEPS):
        net.step(I_stimulus=I_STIM)
        if D_val == 0.50:
            net.model.D_eff = (0.50 * net.model.u.mean()) / np.sqrt(N)
        if step >= WARMUP:
            H_list.append(calculate_continuous_entropy(net.model.v))
            v_hist.append(net.model.v.copy())
    v_arr = np.array(v_hist)
    return (np.mean(H_list), np.std(H_list),
            compute_synchrony(v_arr),
            calculate_temporal_lz_complexity(v_arr, n_bins=5),
            float(net.model.u.mean()))

ALPHA_VALUES = [-1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5]
M_VALUES = [3, 5, 7]

print("="*78, flush=True)
print(" ALPHA_META SWEEP extended: alpha=-1.5..+0.5, D(u)=0.50*u, 10 seeds", flush=True)
print("="*78, flush=True)

all_data = {}
for m in M_VALUES:
    all_data[m] = {}
    regime = "FUNCTIONAL" if m <= 4 else ("CRITICAL" if m <= 6 else "DEAD-ZONE")
    print(f"\n  BA m={m} ({regime}):", flush=True)
    print(f"  {'alpha_meta':>12} {'H_cont':>10} {'Sync':>8} {'LZ':>8} {'u_mean':>8}", flush=True)
    print(f"  {'-'*55}", flush=True)

    for a in ALPHA_VALUES:
        H_runs, S_runs, L_runs = [], [], []
        for seed in SEEDS_10:
            H, H_s, S, L, u = run_one(m, seed, 0.50, a)
            H_runs.append(H); S_runs.append(S); L_runs.append(L)
        H_m = np.mean(H_runs); H_s = np.std(H_runs)
        S_m = np.mean(S_runs); L_m = np.mean(L_runs)
        print(f"  {f'alpha={a:+.2f}':>12} {H_m:>10.3f} {S_m:>8.4f} {L_m:>8.4f}", flush=True)
        all_data[m][a] = (H_m, H_s, S_m, L_m)

# V4 baselines
print(f"\n  V4 static D=0.15 baselines:", flush=True)
for m in M_VALUES:
    H_runs, S_runs = [], []
    for seed in SEEDS_10:
        adj = make_ba(N, m, seed)
        net = Mem4Network(adjacency_matrix=adj, coupling_norm='degree_linear', seed=seed)
        net.model.cfg['coupling']['D'] = 0.15
        net.model.D_eff = 0.15 / np.sqrt(N)
        H_list, v_hist = [], []
        for step in range(STEPS):
            net.step(I_stimulus=I_STIM)
            if step >= WARMUP:
                H_list.append(calculate_continuous_entropy(net.model.v))
                v_hist.append(net.model.v.copy())
        v_arr = np.array(v_hist)
        H_runs.append(np.mean(H_list))
        S_runs.append(compute_synchrony(v_arr))
    print(f"    BA m={m}: H={np.mean(H_runs):.3f}, Sync={np.mean(S_runs):.4f}", flush=True)
    all_data[m]['v4'] = (np.mean(H_runs), 0, np.mean(S_runs), 0)

print(f"\n{'='*78}", flush=True)
print("  VERDICT", flush=True)
print(f"{'='*78}", flush=True)
for m in M_VALUES:
    v4_H = all_data[m]['v4'][0]
    best = max(all_data[m].items(), key=lambda x: x[1][0] if isinstance(x[0], float) else -999)
    print(f"\n  BA m={m}: V4={v4_H:.3f}, Best=alpha={best[0]} H={best[1][0]:.3f} (delta {best[1][0]-v4_H:+.3f})", flush=True)
    print(f"    Best H improvement over D(u) only (alpha=0): {best[1][0]-all_data[m][0.0][0]:+.3f} bits", flush=True)

print(f"\n  SYNC SAFETY CHECK (threshold: |Sync| < 0.05):", flush=True)
for m in M_VALUES:
    unsafe = [a for a, d in all_data[m].items() if isinstance(a, float) and abs(d[2]) >= 0.05]
    if unsafe:
        print(f"    BA m={m}: UNSAFE at alpha={unsafe} (Sync >= 0.05)", flush=True)
    else:
        print(f"    BA m={m}: ALL SAFE (max Sync={max(abs(d[2]) for _,d in all_data[m].items() if isinstance(_,float)):.4f})", flush=True)

print(f"\n  KEY INSIGHT: Is alpha=-1.5 already saturated or still climbing?", flush=True)
for m in M_VALUES:
    a_m15 = all_data[m][-1.5][0]
    a_m10 = all_data[m][-1.0][0]
    a_m05 = all_data[m][-0.5][0]
    print(f"  BA m={m}: alpha=-1.5: {a_m15:.3f}, -1.0: {a_m10:.3f}, -0.5: {a_m05:.3f}", flush=True)
    trend = "climbing" if a_m15 > a_m10 else ("peaked" if a_m10 > a_m05 else "descending")
    print(f"    Trend: {trend}", flush=True)
