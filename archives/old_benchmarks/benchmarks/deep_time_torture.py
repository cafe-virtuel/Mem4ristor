import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mem4ristor.core import Mem4Network
import os
from tqdm import tqdm

def run_deep_time_torture():
    print("💀 STARTING DEEP TIME TORTURE TEST (1M STEPS)")
    print("---------------------------------------------")
    
    size = 10
    model = Mem4Network(size=size, seed=42)
    steps = 1000000
    i_stim = 1.1
    
    u_history = []
    h_history = []
    
    for i in tqdm(range(steps), desc="Torture Progress"):
        model.step(I_stimulus=i_stim)
        
        # Log tous les 1000 steps pour ne pas saturer la RAM
        if i % 1000 == 0:
            u_history.append(np.mean(model.model.u))
            h_history.append(model.calculate_entropy())
            
    # Plotting
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    ax1.set_xlabel('Steps (k)')
    ax1.set_ylabel('Mean Doubt (u)', color='tab:blue')
    ax1.plot(range(0, steps, 1000), u_history, color='tab:blue', alpha=0.7, label='Mean u')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylim(0, 1.1)
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Entropy (H)', color='tab:red')
    ax2.plot(range(0, steps, 1000), h_history, color='tab:red', alpha=0.7, label='Entropy H')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.set_ylim(0, 2.5)
    
    plt.title('Deep Time Stability: 1 Million Steps Stress-Test (v2.3)')
    fig.tight_layout()
    
    os.makedirs('results/plots', exist_ok=True)
    plt.savefig('results/plots/deep_time_stability_1M.png', dpi=300)
    print("\n✅ Deep time plot saved to results/plots/deep_time_stability_1M.png")
    
    # Final check
    final_h = h_history[-1]
    final_u = u_history[-1]
    
    print(f"Final Entropy: {final_h:.4f}")
    print(f"Final Mean Doubt: {final_u:.4f}")
    
    if final_h > 1.2 and 0.01 < final_u < 0.99:
        print("🏆 [SUCCESS] The model survived the Deep Time torture without collapse or saturation.")
    else:
        print("❌ [FAILURE] The model failed the stability test.")

if __name__ == "__main__":
    run_deep_time_torture()
