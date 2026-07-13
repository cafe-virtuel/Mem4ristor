#!/usr/bin/env python3
"""
P5(b) -- Physical hardware balance and directed graph dynamics comparison.
Calculates execution time and energy consumption of simulating Mem4ristor
networks on CPU vs. physical hardware emulation.
"""
import pathlib
import sys
import time
import csv
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'src'))

from mem4ristor.topology import Mem4Network
from mem4ristor.graph_utils import make_ba, make_directed
from mem4ristor.metrics import (
    calculate_cognitive_entropy,
    calculate_continuous_entropy,
    calculate_pairwise_synchrony,
)

# Constants
N = 100
M_VALUES = [3, 5]
GRAPH_SEEDS = [1, 2, 3]
DYN_SEEDS = [42, 123, 777]
RULES = ['UNDIRECTED', 'RANDOM', 'HUBS_LISTEN', 'HUBS_BROADCAST']
WARM_UP = 1000
STEPS = 2000
TOTAL_STEPS = WARM_UP + STEPS
HERETIC = 0.15
COUPLING_NORM = 'degree_linear'

# CPU Parameters
CPU_TDP_W = 65.0  # Typical desktop CPU power

# Physical Hardware parameters
T_NODE_MODEL = 22.25  # Model own period of isolated node FHN

# 1. Spintronics (STNO vortex)
T_NODE_STNO = 1e-9  # 1 ns
DT_PHYS_STNO = 0.05 * (T_NODE_STNO / T_NODE_MODEL)  # ~2.25 ps
POWER_STNO_NODE = 3e-3  # 3 mW per node

# 2. Photonics (GST)
T_NODE_GST = 100e-9  # 100 ns
DT_PHYS_GST = 0.05 * (T_NODE_GST / T_NODE_MODEL)  # ~225 ps
E_PHOTON_1550 = 1.28e-19  # J
PHOTONS_PER_STEP = 10  # Signal budget per step

# 3. NbO2 Neuristor
T_NODE_NBO2 = 1e-6  # 1 us
DT_PHYS_NBO2 = 0.05 * (T_NODE_NBO2 / T_NODE_MODEL)  # ~2.25 ns
E_STEP_NBO2 = 225e-15  # 225 fJ per step per node

def run_simulation(adj, seed):
    t_start = time.perf_counter()
    net = Mem4Network(adjacency_matrix=adj.copy(), heretic_ratio=HERETIC,
                      seed=seed, coupling_norm=COUPLING_NORM)
    for _ in range(WARM_UP):
        net.step(I_stimulus=0.5)
    snaps = []
    for _ in range(STEPS):
        net.step(I_stimulus=0.5)
        snaps.append(net.v.copy())
    t_end = time.perf_counter()
    
    v_s = np.array(snaps)
    cpu_time = t_end - t_start
    
    return cpu_time, {
        'h_cont': float(np.mean([calculate_continuous_entropy(v) for v in v_s[::10]])),
        'h_cog': float(np.mean([calculate_cognitive_entropy(v) for v in v_s[::10]])),
        'sync': float(calculate_pairwise_synchrony(v_s)),
    }

def main():
    t0 = time.time()
    rows = []
    
    print("=== Physical Hardware Balance & Directed Graph Simulation ===")
    
    for m in M_VALUES:
        for rule in RULES:
            print(f"Running rule {rule} for m={m}...")
            for gseed in GRAPH_SEEDS:
                adj_undirected = make_ba(N, m, seed=gseed)
                if rule == 'UNDIRECTED':
                    adj = adj_undirected
                else:
                    rng_dir = np.random.RandomState(9000 + gseed)
                    adj = make_directed(adj_undirected, rule, rng_dir)
                
                for dseed in DYN_SEEDS:
                    cpu_time, metrics = run_simulation(adj, dseed)
                    
                    # Compute budgets
                    # CPU
                    energy_cpu = cpu_time * CPU_TDP_W
                    
                    # Spintronics (STNO vortex)
                    time_stno = TOTAL_STEPS * DT_PHYS_STNO
                    energy_stno = N * TOTAL_STEPS * DT_PHYS_STNO * POWER_STNO_NODE
                    
                    # Photonics (GST)
                    time_gst = TOTAL_STEPS * DT_PHYS_GST
                    energy_gst = N * TOTAL_STEPS * PHOTONS_PER_STEP * E_PHOTON_1550
                    
                    # NbO2 Neuristor
                    time_nbo2 = TOTAL_STEPS * DT_PHYS_NBO2
                    energy_nbo2 = N * TOTAL_STEPS * E_STEP_NBO2
                    
                    rows.append({
                        'm': m,
                        'rule': rule,
                        'graph_seed': gseed,
                        'dyn_seed': dseed,
                        'cpu_time_s': cpu_time,
                        'cpu_energy_J': energy_cpu,
                        'stno_time_s': time_stno,
                        'stno_energy_J': energy_stno,
                        'gst_time_s': time_gst,
                        'gst_energy_J': energy_gst,
                        'nbo2_time_s': time_nbo2,
                        'nbo2_energy_J': energy_nbo2,
                        **metrics
                    })

    fig_dir = HERE.parent / 'figures'
    fig_dir.mkdir(exist_ok=True)
    
    # Save CSV
    csv_path = fig_dir / 'p5b_physical_balance_poc.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"Results written to: {csv_path}")

    # Compute Averages
    print("\n=== SUMMARY TABLE (Averages across seeds) ===")
    print(f"{'Rule (m=3)':<18} | {'CPU Time (s)':<12} | {'CPU Energy (J)':<14} | {'STNO Energy (pJ)':<16} | {'GST Energy (fJ)':<16} | {'NbO2 Energy (pJ)':<16}")
    print("-" * 105)
    
    agg = {}
    for r in rows:
        key = (r['m'], r['rule'])
        if key not in agg:
            agg[key] = []
        agg[key].append(r)
        
    for m in M_VALUES:
        for rule in RULES:
            sub = agg[(m, rule)]
            mean_cpu_t = np.mean([x['cpu_time_s'] for x in sub])
            mean_cpu_e = np.mean([x['cpu_energy_J'] for x in sub])
            mean_stno_e = np.mean([x['stno_energy_J'] for x in sub]) * 1e12  # to pJ
            mean_gst_e = np.mean([x['gst_energy_J'] for x in sub]) * 1e15    # to fJ
            mean_nbo2_e = np.mean([x['nbo2_energy_J'] for x in sub]) * 1e12  # to pJ
            
            print(f"{rule + f' (m={m})':<18} | {mean_cpu_t:12.4f} | {mean_cpu_e:14.4f} | {mean_stno_e:16.4f} | {mean_gst_e:16.4f} | {mean_nbo2_e:16.4f}")

    # Generate plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # We will plot the energy comparison for m=3, rule=HUBS_LISTEN as a showcase
        sub = agg[(3, 'HUBS_LISTEN')]
        mean_cpu_e = np.mean([x['cpu_energy_J'] for x in sub])
        mean_stno_e = np.mean([x['stno_energy_J'] for x in sub])
        mean_gst_e = np.mean([x['gst_energy_J'] for x in sub])
        mean_nbo2_e = np.mean([x['nbo2_energy_J'] for x in sub])
        
        mean_cpu_t = np.mean([x['cpu_time_s'] for x in sub])
        mean_stno_t = np.mean([x['stno_time_s'] for x in sub])
        mean_gst_t = np.mean([x['gst_time_s'] for x in sub])
        mean_nbo2_t = np.mean([x['nbo2_time_s'] for x in sub])
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Energy Plot
        energies = [mean_cpu_e, mean_stno_e, mean_nbo2_e, mean_gst_e]
        labels = ['CPU (simulated)', 'STNO (vortex)', 'NbO2 Neuristor', 'GST (photonic)']
        colors = ['#7f8c8d', '#e74c3c', '#f39c12', '#2ecc71']
        
        axes[0].bar(labels, energies, color=colors)
        axes[0].set_yscale('log')
        axes[0].set_ylabel('Energy (Joules, Log Scale)')
        axes[0].set_title('Energy Budget per Simulation (3000 steps, N=100)')
        for i, val in enumerate(energies):
            axes[0].text(i, val * 1.5, f"{val:.2e} J", ha='center', fontsize=9, fontweight='bold')
            
        # Time Plot
        times = [mean_cpu_t, mean_nbo2_t, mean_gst_t, mean_stno_t]
        t_labels = ['CPU (simulated)', 'NbO2 Neuristor', 'GST (photonic)', 'STNO (vortex)']
        t_colors = ['#7f8c8d', '#f39c12', '#2ecc71', '#e74c3c']
        
        axes[1].bar(t_labels, times, color=t_colors)
        axes[1].set_yscale('log')
        axes[1].set_ylabel('Execution Time (seconds, Log Scale)')
        axes[1].set_title('Execution Time Comparison (Log Scale)')
        for i, val in enumerate(times):
            axes[1].text(i, val * 1.5, f"{val:.2e} s", ha='center', fontsize=9, fontweight='bold')
            
        fig.suptitle('Mem4ristor Hardware Balance: Simulated CPU vs Physical Hardware (m=3, HUBS_LISTEN)', fontsize=13, fontweight='bold')
        plt.tight_layout()
        
        png_path = fig_dir / 'p5b_physical_balance_poc.png'
        plt.savefig(png_path, dpi=150)
        print(f"Plot saved to: {png_path}")
        
    except Exception as e:
        print(f"Error drawing plot: {e}")
        
    print(f"Wall time: {time.time()-t0:.1f}s")

if __name__ == '__main__':
    main()
