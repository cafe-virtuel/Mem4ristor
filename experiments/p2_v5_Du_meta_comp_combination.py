"""
V5 — Combination: Metacognition + Compartmentalization + D(u)=0.50*u adaptive

Tests the 3-way combination on:
  - BA m=3 (functional topology)
  - BA m=5 (dead zone / critical topology)

Goal: Does D(u)=0.50*u + Metacog + Compart produce better results than
      any subset? Does it escape the dead zone?

Configurations (all on same BA graph):
  1. V4 pur          : D=0.15 static, no V5 features
  2. D(u)=0.50*u     : adaptive D, no V5 features
  3. Meta + D(u)     : adaptive D + metacognition
  4. Comp + D(u)     : adaptive D + compartmentalization
  5. Meta+Comp+D(u)  : adaptive D + both V5 features

Metrics (hierarchical):
  1. synchrony (PRIMARY, binning-independent)
  2. H_cont (SECONDARY, 100-bin entropy)
  3. LZ76 complexity (TERTIARY, distinguishes sync=0 degenerate cases)
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import numpy as np
import csv
from mem4ristor.core import Mem4Network
from mem4ristor.graph_utils import make_ba
from mem4ristor.metrics import calculate_continuous_entropy, calculate_temporal_lz_complexity

# === FIXED PARAMS ===
N = 100
STEPS = 3000
WARMUP = int(STEPS * 0.75)
SEEDS = [42, 123, 777, 1234, 5678, 9999, 314, 2718, 8765, 4321]
I_STIM = 0.5

# === CONFIGURATIONS ===
# Each tuple: (label, D_value, meta_cfg, comp_cfg)
# D_value: 0.15 = static, 0.50 = adaptive D(u)=0.50*u
CONFIGS = [
    ("V4 static D=0.15",    0.15, {"enabled": False},                                              {"enabled": False, "K": 3, "gamma": 0.10, "mode": "full"}),
    ("D(u)=0.50*u only",    0.50, {"enabled": False},                                              {"enabled": False, "K": 3, "gamma": 0.10, "mode": "full"}),
    ("Meta + D(u)",          0.50, {"enabled": True,  "alpha_meta": -0.5, "epsilon_min": 0.01},   {"enabled": False, "K": 3, "gamma": 0.10, "mode": "full"}),
    ("Comp + D(u)",          0.50, {"enabled": False},                                              {"enabled": True,  "K": 3, "gamma": 0.10, "mode": "full"}),
    ("All 3 combined",       0.50, {"enabled": True,  "alpha_meta": -0.5, "epsilon_min": 0.01},   {"enabled": True,  "K": 3, "gamma": 0.10, "mode": "full"}),
]

# Also test on BA m=5 (dead zone / critical)
M_VALUES = [3, 5]

def _compute_synchrony(v_history):
    """Pearson mean pairwise correlation."""
    n_nodes = v_history.shape[1]
    if n_nodes < 2:
        return 0.0
    v_centered = v_history - v_history.mean(axis=0, keepdims=True)
    v_std = v_history.std(axis=0, keepdims=True)
    v_std[v_std == 0] = 1.0
    corr_matrix = np.dot(v_centered.T, v_centered) / (v_centered.shape[0] - 1)
    std_product = np.dot(v_std.T, v_std)
    corr_matrix = corr_matrix / (std_product + 1e-12)
    n = corr_matrix.shape[0]
    upper_tri_indices = np.triu_indices(n, k=1)
    if upper_tri_indices[0].size == 0:
        return 0.0
    return float(np.mean(corr_matrix[upper_tri_indices]))


def run_config(m, seed, D_value, meta_cfg, comp_cfg, is_adaptive):
    """Run one config on one seed. Returns metrics dict."""
    adj = make_ba(N, m, seed)
    net = Mem4Network(adjacency_matrix=adj, coupling_norm='degree_linear', seed=seed)

    # D setup
    net.model.cfg['coupling']['D'] = float(D_value)
    if not is_adaptive:
        net.model.D_eff = D_value / float(np.sqrt(net.model.N))

    # V5 features
    net.model.cfg['metacognitive'] = meta_cfg
    net.model.cfg['compartments'] = comp_cfg

    H_list = []
    v_history = []
    u_buffer = []

    for step in range(STEPS):
        net.step(I_stimulus=I_STIM)
        # Adaptive D: D_eff = 0.50 * u_mean / sqrt(N)
        if is_adaptive:
            u_mean = float(net.model.u.mean())
            net.model.D_eff = (0.50 * u_mean) / float(np.sqrt(net.model.N))
        if step >= WARMUP:
            H_list.append(calculate_continuous_entropy(net.model.v))
            v_history.append(net.model.v.copy())
            u_buffer.append(net.model.u.mean())

    v_arr = np.array(v_history)
    synchrony = _compute_synchrony(v_arr)
    lz_complexity = calculate_temporal_lz_complexity(v_arr, n_bins=5)
    H_cont = np.mean(H_list) if H_list else np.nan
    H_std = np.std(H_list) if H_list else np.nan
    u_mean = np.mean(u_buffer) if u_buffer else np.nan

    return {
        'H_cont': H_cont,
        'H_std': H_std,
        'synchrony': synchrony,
        'lz_complexity': lz_complexity,
        'u_mean': u_mean,
    }


def main():
    print("=" * 76)
    print("  V5 COMBINATION: Metacognition + Compartments + D(u)=0.50*u ADAPTIVE")
    print(f"  N={N} | {STEPS} steps | warmup={WARMUP} | {len(SEEDS)} seeds | I_stim={I_STIM}")
    print("=" * 76)

    all_results = []

    for m in M_VALUES:
        print(f"\n{'=' * 76}")
        print(f"  BA m={m} ({'FUNCTIONAL' if m <= 4 else 'DEAD ZONE / CRITICAL'})")
        print(f"{'=' * 76}")
        print(f"\n  {'Configuration':<22} {'H_cont':>10} {'Sync':>8} {'LZ':>8} {'u_mean':>8}")
        print(f"  {'-' * 60}")

        for label, D_value, meta_cfg, comp_cfg in CONFIGS:
            is_adaptive = bool(D_value == 0.50)
            H_runs, Sync_runs, LZ_runs, u_runs = [], [], [], []

            for seed in SEEDS:
                try:
                    r = run_config(m, seed, D_value, meta_cfg, comp_cfg, is_adaptive)
                    H_runs.append(r['H_cont'])
                    Sync_runs.append(r['synchrony'])
                    LZ_runs.append(r['lz_complexity'])
                    u_runs.append(r['u_mean'])
                except Exception as e:
                    print(f"  ERROR seed={seed} config={label}: {e}")
                    continue

            H_mean = np.mean(H_runs)
            H_std = np.std(H_runs)
            Sync_mean = np.mean(Sync_runs)
            LZ_mean = np.mean(LZ_runs)
            u_mean = np.mean(u_runs)

            print(f"  {label:<22} {H_mean:>6.3f}±{H_std:.3f}  {Sync_mean:>6.4f}  {LZ_mean:>6.4f}  {u_mean:>6.4f}")

            all_results.append({
                'm': m,
                'label': label,
                'H_cont': H_mean,
                'H_std': H_std,
                'synchrony': Sync_mean,
                'lz_complexity': LZ_mean,
                'u_mean': u_mean,
                'is_adaptive': is_adaptive,
                'meta': meta_cfg.get('enabled', False),
                'comp': comp_cfg.get('enabled', False),
            })

    # === VERDICT ===
    print(f"\n{'=' * 76}")
    print("  VERDICT")
    print(f"{'=' * 76}\n")

    for m in M_VALUES:
        m_results = [r for r in all_results if r['m'] == m]
        print(f"  BA m={m}:")
        v4 = next(r for r in m_results if 'V4 static' in r['label'])
        best_H = max(m_results, key=lambda r: r['H_cont'])
        best_Sync = min(m_results, key=lambda r: abs(r['synchrony']))  # closest to 0
        print(f"    V4 static:  H={v4['H_cont']:.3f}, Sync={v4['synchrony']:.4f}")
        print(f"    Best H:     {best_H['label']:<22} H={best_H['H_cont']:.3f} (delta {best_H['H_cont']-v4['H_cont']:+.3f})")
        print(f"    Best Sync:  {best_Sync['label']:<22} Sync={best_Sync['synchrony']:.4f}")

        # Synergy check for 3-way combination
        du_only   = next(r for r in m_results if 'D(u)=0.50*u only' in r['label'])
        meta_d    = next(r for r in m_results if 'Meta + D(u)' in r['label'])
        comp_d    = next(r for r in m_results if 'Comp + D(u)' in r['label'])
        all3      = next(r for r in m_results if 'All 3 combined' in r['label'])

        delta_meta = meta_d['H_cont'] - du_only['H_cont']
        delta_comp = comp_d['H_cont'] - du_only['H_cont']
        delta_all3 = all3['H_cont'] - du_only['H_cont']
        expected_add = delta_meta + delta_comp
        synergy = delta_all3 - expected_add

        print(f"\n    Synergy analysis (vs D(u) only baseline):")
        print(f"      Meta + D(u)    : {delta_meta:+.3f} bits")
        print(f"      Comp + D(u)    : {delta_comp:+.3f} bits")
        print(f"      All 3 combined : {delta_all3:+.3f} bits")
        print(f"      Expected add.  : {expected_add:+.3f} bits")
        print(f"      Synergy        : {synergy:+.3f} bits ({'SYNERGIE' if synergy > 0.05 else 'ADDITIF' if synergy > -0.05 else 'INTERFERENCE'})")

        # Synchrony effect
        sync_meta = meta_d['synchrony'] - du_only['synchrony']
        sync_comp = comp_d['synchrony'] - du_only['synchrony']
        sync_all3 = all3['synchrony'] - du_only['synchrony']
        print(f"\n    Sync effect (delta vs D(u) only):")
        print(f"      Meta + D(u)    : {sync_meta:+.4f}")
        print(f"      Comp + D(u)    : {sync_comp:+.4f}")
        print(f"      All 3 combined : {sync_all3:+.4f}")
        print()

    # === SAVE CSV ===
    out_dir = os.path.join(os.path.dirname(__file__), '../figures')
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'p2_v5_Du_meta_comp_combination.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['m', 'label', 'H_cont', 'H_std', 'synchrony', 'lz_complexity', 'u_mean', 'is_adaptive', 'meta', 'comp'])
        w.writeheader()
        w.writerows(all_results)
    print(f"  CSV -> {csv_path}")
    print(f"\n{'=' * 76}")
    print("  Pour reproduire : python experiments/p2_v5_Du_meta_comp_combination.py")
    print(f"{'=' * 76}\n")