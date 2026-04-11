import numpy as np
import pandas as pd
import time
import os
import sys
import matplotlib.pyplot as plt

# Integration avec le dossier src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from mem4ristor.core import Mem4Network

def run_large_scale_exploration():
    print("🌌 MEM4RISTOR LARGE SCALE EXPLORATION (200x200)")
    print("-----------------------------------------------")
    print("[INFO] Target: 40,000 units (Bubbles of Doubt search)")
    
    size = 200
    steps = 1000
    heretic_ratio = 0.15 # Valeur optimale identifiée en Phase 2
    i_stim = 1.1 # Stimulus fort
    
    start_time = time.time()
    model = Mem4Network(size=size, heretic_ratio=heretic_ratio, seed=42)
    
    # Historique pour le rapport
    entropy_history = []
    
    print(f"[START] Beginning simulation for {steps} steps...")
    
    for i in range(steps):
        model.step(I_stimulus=i_stim)
        
        if i % 100 == 0:
            h = model.calculate_entropy()
            entropy_history.append(h)
            elapsed = time.time() - start_time
            print(f"Step {i:4d} | Entropy: {h:.4f} | Elapsed: {elapsed:.2f}s")
            
    total_time = time.time() - start_time
    print(f"\n[FINISH] Simulation completed in {total_time:.2f} seconds.")
    print(f"[SPEED] ~{steps/total_time:.2f} iterations/sec for 40,000 units.")
    
    # Save results
    os.makedirs("results", exist_ok=True)
    v_final = model.model.v.reshape((size, size))
    np.save("results/v_200x200_final.npy", v_final)
    
    # Generate visualization
    plt.figure(figsize=(10, 10))
    plt.imshow(v_final, cmap='RdBu_r', vmin=-2.0, vmax=2.0)
    plt.title(f"Mem4ristor 200x200 | I_stim={i_stim} | Ratio={heretic_ratio}")
    plt.colorbar(label='Potentiel Cognitif (V)')
    plt.axis('off')
    plt.savefig("results/bubbles_of_doubt_200x200.png", dpi=300)
    print("[SUCCESS] High-resolution map saved to results/bubbles_of_doubt_200x200.png")
    
    # Conclusion sur la percolation
    final_h = model.calculate_entropy()
    if final_h > 1.5:
        print("✅ [DISCOVERY] Diversity PERCOLATION confirmed at large scale.")
        print("   The 'Bubbles of Doubt' are spatially stable and resist global synchronization.")
    else:
        print("❌ [REPORT] Diversity collapsed at large scale. Threshold analysis needed.")

if __name__ == "__main__":
    run_large_scale_exploration()
