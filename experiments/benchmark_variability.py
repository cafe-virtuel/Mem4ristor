import numpy as np
import sys
import os

# Add src to path robustly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from mem4ristor.core import Mem4Network

def run_monte_carlo(n_runs=20, n_steps=500):
    print(f"Running {n_runs} Monte Carlo simulations (Steps={n_steps})...")
    print("-" * 60)
    print(f"{'Run':<5} | {'Initial H':<10} | {'Final H':<10} | {'SNR':<10} | {'Status':<10}")
    print("-" * 60)
    
    entropies_final = []
    snrs = []
    
    for i in range(n_runs):
        # Initialize typical network
        net = Mem4Network(size=10, heretic_ratio=0.15, seed=None) # None seed for random
        
        # Capture Initial Entropy
        h_init = net.calculate_entropy()
        
        # Run standard simulation
        # Using a slight stimulus to provoke dynamics
        for _ in range(n_steps):
            net.step(I_stimulus=0.5)
            
        h_final = net.calculate_entropy()
        entropies_final.append(h_final)
        
        # Estimate SNR (Signal strength of coupling vs Noise)
        # Signal ~ D_eff * |1-2u| * L * v
        # Noise = sigma_v (0.05 default)
        
        noise_floor = net.model.cfg['noise']['sigma_v']
        if noise_floor > 0:
            # Approximate signal magnitude
            start_u = net.model.u.mean() # should be around 0.5 initially then drift
            # If u is near 0.5, signal is weak. If u splits, signal is strong.
            coupling_strength = net.model.D_eff * np.abs(1.0 - 2.0 * start_u) 
            # This is a rough proxy, true SNR is complex dynamically
            snrs.append(coupling_strength / noise_floor)
        else:
            snrs.append(float('inf'))
            
        print(f"{i+1:<5} | {h_init:<10.4f} | {h_final:<10.4f} | {snrs[-1]:<10.4f} | {'OK'}")

    print("-" * 60)
    print("STATISTICAL SUMMARY")
    print("-" * 60)
    print(f"Final Entropy : {np.mean(entropies_final):.4f} ± {np.std(entropies_final):.4f}")
    if all(s != float('inf') for s in snrs):
        print(f"Hybrid SNR    : {np.mean(snrs):.4f} ± {np.std(snrs):.4f}")
    print("-" * 60)

if __name__ == "__main__":
    run_monte_carlo()
