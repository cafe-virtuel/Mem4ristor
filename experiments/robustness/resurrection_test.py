import os
import sys
# Resolve Paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '../../'))
sys.path.append(os.path.join(ROOT_DIR, 'src'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from mem4ristor.core import Mem4Network

def run_resurrection_test():
    print("🌅 RESURRECTION TEST: BREAKING TOTAL CONSENSUS (v2.3)")
    print("----------------------------------------------------")
    
    size = 10
    model = Mem4Network(size=size, heretic_ratio=0.15, seed=42)
    steps = 5000
    i_stim = 1.1
    
    # FORCER LE CONSENSUS TOTAL (H = 0)
    # Tous les neurones sont dans le même état v=1.0, w=0.0
    model.model.v[:] = 1.0
    model.model.w[:] = 0.0
    model.model.u[:] = 0.05 # Doute initial bas
    
    print(f"Initial Entropy H: {model.calculate_entropy():.4f} (Total Consensus Force)")
    
    h_history = []
    v_mean_history = []
    v_std_history = []
    
    for i in tqdm(range(steps), desc="Step Progress"):
        model.step(I_stimulus=i_stim)
        h_history.append(model.calculate_entropy())
        v_mean_history.append(np.mean(model.model.v))
        v_std_history.append(np.std(model.model.v))
        
    # Plotting
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Entropy (H)', color='tab:red')
    ax1.plot(h_history, color='tab:red', linewidth=2, label='Entropy (Diversity)')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.set_ylim(0, 2.5)
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Standard Deviation of v (Polarization)', color='tab:blue')
    ax2.plot(v_std_history, color='tab:blue', alpha=0.5, label='std(v)')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3, label='Diversity Threshold')
    
    plt.title('Resurrection Test: Spontaneous Consensus Breaking (v2.3)\nInitial Condition: $v_i = 1.0, H = 0$')
    fig.tight_layout()
    
    results_dir = os.path.join(ROOT_DIR, "results")
    os.makedirs(os.path.join(results_dir, 'plots'), exist_ok=True)
    plt.savefig(os.path.join(results_dir, 'plots/resurrection_test_v23.png'), dpi=300)
    print(f"\n✅ Resurrection plot saved to {os.path.join(results_dir, 'plots/resurrection_test_v23.png')}")
    
    # Analyse
    rising_step = next((i for i, h in enumerate(h_history) if h > 0.5), None)
    final_h = h_history[-1]
    
    print(f"\nFinal Entropy: {final_h:.4f}")
    if rising_step:
        print(f"🏆 Symmetry broken at step: {rising_step}")
        if final_h > 1.5:
            print("🚀 [VERDICT] The system SUCCESSFULLY RESTORED DIVERSITY from total consensus.")
        else:
            print("⚠️ [CAUTION] Symmetry broken but limited diversity recovery.")
    else:
        print("❌ [FAILURE] The system remained trapped in consensus.")

if __name__ == "__main__":
    run_resurrection_test()
