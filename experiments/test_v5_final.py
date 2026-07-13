import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
import numpy as np
from mem4ristor.core import Mem4Network
from mem4ristor.graph_utils import make_ba
from mem4ristor.metrics import calculate_continuous_entropy, calculate_temporal_lz_complexity

N=120; STEPS=3000; WARMUP=int(STEPS*0.75); I_STIM=0.5
SEEDS = [42, 123, 777, 1234, 5678, 9999, 314, 2718, 8765, 4321, 11, 22]
M_VALUES = [3, 5, 7, 10]

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

print("="*70, flush=True)
print(" V5 FINAL  |  12 seeds  |  N=120  |  STEPS=3000", flush=True)
print("="*70, flush=True)

results = []

for m in M_VALUES:
    regime = "FUNCTIONAL" if m<=4 else ("CRITICAL" if m<=6 else "DEAD-ZONE")
    print(f"\n  BA m={m} ({regime}):", flush=True)
    print(f"  {'Config':<22} {'H_cont':>10} {'H_std':>8} {'Sync':>8} {'LZ':>8} {'u_mean':>8}", flush=True)
    print(f"  {'-'*70}", flush=True)

    # Config 0: V4 static D=0.15
    H_runs, S_runs, L_runs, U_runs = [], [], [], []
    for si, seed in enumerate(SEEDS):
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
        H_runs.append(np.mean(H_list)); S_runs.append(compute_synchrony(v_arr))
        L_runs.append(calculate_temporal_lz_complexity(v_arr, n_bins=5))
        U_runs.append(float(net.model.u.mean()))
        if (si+1) % 4 == 0: print(f"    V4 seed {si+1}/12 done", flush=True)
    r = {'m':m,'label':'V4 D=0.15 static','H_cont':np.mean(H_runs),'H_std':np.std(H_runs),
         'synchrony':np.mean(S_runs),'lz_complexity':np.mean(L_runs),'u_mean':np.mean(U_runs)}
    results.append(r)
    print(f"  {'V4 D=0.15 static':<22} {r['H_cont']:>10.3f} {r['H_std']:>8.3f} {r['synchrony']:>8.4f} {r['lz_complexity']:>8.4f} {r['u_mean']:>8.4f}", flush=True)

    # Configs: D(u) alpha=-0.5, -4.0, -8.0
    for alpha_val in [-0.5, -4.0, -8.0]:
        H_runs, S_runs, L_runs, U_runs = [], [], [], []
        for si, seed in enumerate(SEEDS):
            adj = make_ba(N, m, seed)
            net = Mem4Network(adjacency_matrix=adj, coupling_norm='degree_linear', seed=seed)
            net.model.cfg['coupling']['D'] = 0.50
            net.model.cfg['metacognitive'] = {'enabled': True, 'alpha_meta': alpha_val, 'epsilon_min': 0.01}
            net.model.D_eff = 0.50 / float(np.sqrt(N))
            H_list, v_hist = [], []
            for step in range(STEPS):
                net.step(I_stimulus=I_STIM)
                if step >= WARMUP:
                    net.model.D_eff = (0.50 * net.model.u.mean()) / float(np.sqrt(N))
                    H_list.append(calculate_continuous_entropy(net.model.v))
                    v_hist.append(net.model.v.copy())
            v_arr = np.array(v_hist)
            H_runs.append(np.mean(H_list)); S_runs.append(compute_synchrony(v_arr))
            L_runs.append(calculate_temporal_lz_complexity(v_arr, n_bins=5))
            U_runs.append(float(net.model.u.mean()))
            if (si+1) % 4 == 0: print(f"    a={alpha_val} seed {si+1}/12 done", flush=True)
        label = f'D(u) alpha={alpha_val}'
        r = {'m':m,'label':label,'H_cont':np.mean(H_runs),'H_std':np.std(H_runs),
             'synchrony':np.mean(S_runs),'lz_complexity':np.mean(L_runs),'u_mean':np.mean(U_runs)}
        results.append(r)
        flag = "  SYNC!" if abs(r['synchrony']) > 0.05 else ""
        print(f"  {label:<22} {r['H_cont']:>10.3f} {r['H_std']:>8.3f} {r['synchrony']:>8.4f} {r['lz_complexity']:>8.4f} {r['u_mean']:>8.4f}{flag}", flush=True)

# Summary
print(f"\n{'='*70}", flush=True)
print("  GAIN vs V4 D=0.15", flush=True)
print(f"{'='*70}", flush=True)
print(f"  {'Config':<22} | {'m=3':>8} {'m=5':>8} {'m=7':>8} {'m=10':>8}", flush=True)
print(f"  {'-'*62}", flush=True)
for alpha_val in [-0.5, -4.0, -8.0]:
    label = f'D(u) alpha={alpha_val}'
    vals = []
    for m in M_VALUES:
        v4 = next(r['H_cont'] for r in results if r['m']==m and r['label']=='V4 D=0.15 static')
        cfg = next(r['H_cont'] for r in results if r['m']==m and r['label']==label)
        vals.append(f"{cfg-v4:>+8.3f}")
    print(f"  {label:<22} | {'  '.join(vals)}", flush=True)

# Sync audit
print(f"\n{'='*70}", flush=True)
print("  SYNCHRONY AUDIT", flush=True)
print(f"{'='*70}", flush=True)
for m in M_VALUES:
    for label in ['V4 D=0.15 static','D(u) alpha=-0.5','D(u) alpha=-4.0','D(u) alpha=-8.0']:
        r2 = next(x for x in results if x['m']==m and x['label']==label)
        flag = " SYNC!" if abs(r2['synchrony']) > 0.05 else ""
        print(f"  m={m:<3} {label:<22} sync={r2['synchrony']:>8.4f}{flag}", flush=True)

# CSV
out_dir = '../figures'
os.makedirs(out_dir, exist_ok=True)
csv_path = os.path.join(out_dir, 'p2_v5_final_12seeds.csv')
with open(csv_path, 'w', newline='') as f:
    import csv
    fields = ['m','label','H_cont','H_std','synchrony','lz_complexity','u_mean']
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader(); w.writerows(results)
print(f"\n  CSV -> {csv_path}", flush=True)
