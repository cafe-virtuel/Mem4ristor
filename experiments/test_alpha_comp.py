
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

def run_one(m, seed, D_val, alpha_meta, comp_en, gamma=0.10, K=3):
    adj = make_ba(N, m, seed)
    net = Mem4Network(adjacency_matrix=adj, coupling_norm='degree_linear', seed=seed)
    net.model.cfg['coupling']['D'] = float(D_val)
    if D_val != 0.50:
        net.model.D_eff = D_val / np.sqrt(N)
    net.model.cfg['metacognitive'] = {'enabled': True, 'alpha_meta': float(alpha_meta), 'epsilon_min': 0.01}
    net.model.cfg['compartments'] = {'enabled': comp_en, 'K': K, 'gamma': gamma, 'mode': 'full'}
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

M_VALUES = [3, 5, 7, 10]

print("="*80, flush=True)
print(" ALPHA=-4.0 + COMPARTMENTS: Does K=3 improve or interfere?", flush=True)
print(" 10 seeds | 3000 steps | N=100 | D(u)=0.50*u | alpha=-4.0", flush=True)
print("="*80, flush=True)

# Configurations
CONFIGS = [
    ("V4 baseline (D=0.15)",     0.15, None, False),
    ("D(u)+a=-4.0 (no comp)",    0.50, -4.0, False),
    ("D(u)+a=-4.0 + K=2 full",  0.50, -4.0, True),
    ("D(u)+a=-4.0 + K=3 full",  0.50, -4.0, True),
    ("D(u)+a=-4.0 + K=5 full",  0.50, -4.0, True),
    ("D(u)+a=-4.0 + K=10 full", 0.50, -4.0, True),
]

all_data = {}
for m in M_VALUES:
    all_data[m] = {}
    regime = "FUNCTIONAL" if m<=4 else ("CRITICAL" if m<=6 else "DEAD-ZONE")
    print(f"\n  BA m={m} ({regime}):", flush=True)
    print(f"  {'Config':<24} {'H_cont':>10} {'H_std':>8} {'Sync':>8} {'LZ':>8} {'u':>8}", flush=True)
    print(f"  {'-'*70}", flush=True)

    for label, D_val, alpha, comp_en in CONFIGS:
        K_use = 2 if 'K=2' in label else (3 if 'K=3' in label else (5 if 'K=5' in label else (10 if 'K=10' in label else 3)))

        if D_val == 0.15 and alpha is None:
            H_runs, S_runs, L_runs = [], [], []
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
                L_runs.append(calculate_temporal_lz_complexity(v_arr, n_bins=5))
            K_use = 0
            comp_en = False
        else:
            H_runs, S_runs, L_runs, u_runs = [], [], [], []
            for seed in SEEDS_10:
                H, H_s, S, L, u = run_one(m, seed, D_val, alpha, comp_en, K=K_use)
                H_runs.append(H); S_runs.append(S); L_runs.append(L); u_runs.append(u)
        H_m = np.mean(H_runs); H_s = np.std(H_runs)
        S_m = np.mean(S_runs); L_m = np.mean(L_runs)
        flag = " <<<< SYNC" if abs(S_m) > 0.05 else ""
        print(f"  {label:<24} {H_m:>10.3f} {H_s:>8.3f} {S_m:>8.4f} {L_m:>8.4f}{flag}", flush=True)
        all_data[m][label] = (H_m, H_s, S_m, L_m)

# Verdict
print(f"\n{'='*80}", flush=True)
print("  VERDICT: alpha=-4.0 + Compartments", flush=True)
print(f"{'='*80}", flush=True)

for m in M_VALUES:
    v4   = all_data[m]["V4 baseline (D=0.15)"][0]
    du4  = all_data[m]["D(u)+a=-4.0 (no comp)"][0]
    k2   = all_data[m].get("D(u)+a=-4.0 + K=2 full", (0,))[0]
    k3   = all_data[m].get("D(u)+a=-4.0 + K=3 full", (0,))[0]
    k5   = all_data[m].get("D(u)+a=-4.0 + K=5 full", (0,))[0]
    k10  = all_data[m].get("D(u)+a=-4.0 + K=10 full", (0,))[0]

    print(f"\n  BA m={m}:", flush=True)
    print(f"    V4 baseline:         H={v4:.3f}", flush=True)
    print(f"    D(u)+a=-4.0 no comp: H={du4:.3f} ({du4-v4:+.3f})", flush=True)

    comps = [("K=2", k2), ("K=3", k3), ("K=5", k5), ("K=10", k10)]
    for kname, kval in comps:
        delta_c = kval - du4
        delta_v = kval - v4
        print(f"    D(u)+a=-4.0 + {kname}: H={kval:.3f} ({delta_c:+.3f} comp, {delta_v:+.3f} vs V4)", flush=True)

    best_comp = max(comps, key=lambda x: x[1])
    best_no   = ("no comp", du4)
    best = max([best_no] + comps, key=lambda x: x[1])
    print(f"    --> BEST: {best[0]:<18} H={best[1]:.3f}", flush=True)
