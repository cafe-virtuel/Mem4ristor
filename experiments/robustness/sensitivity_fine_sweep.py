import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Resolve Paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '../../'))
sys.path.append(os.path.join(ROOT_DIR, 'src'))

from mem4ristor.core import Mem4Network

def run_fine_sensitivity():
    print("📈 FINE-GRAINED SENSITIVITY ANALYSIS (0% - 30%)")
    print("-----------------------------------------------")
    
    ratios = np.linspace(0.0, 0.3, 31) # 1% steps
    I_stim = 1.1
    steps = 3000
    repeats = 3 # To average out noise
    
    results = []
    
    for r in tqdm(ratios, desc="Scanning ratios"):
        h_values = []
        for seed in range(42, 42 + repeats):
            model = Mem4Network(size=10, heretic_ratio=r, seed=seed, cold_start=True)
            for _ in range(steps):
                model.step(I_stimulus=I_stim)
            h_values.append(model.calculate_entropy())
        
        results.append({
            'Ratio': r,
            'Entropy': np.mean(h_values),
            'Std': np.std(h_values)
        })
        
    df = pd.DataFrame(results)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.errorbar(df['Ratio'], df['Entropy'], yerr=df['Std'], fmt='-o', color='purple', capsize=5, label='Mean Entropy (H)')
    
    # Critical threshold markers
    plt.axvline(x=0.15, color='red', linestyle='--', alpha=0.7, label='15% Threshold (Observed)')
    plt.axvspan(0.12, 0.18, color='red', alpha=0.1, label='Transition Zone')
    
    plt.xlabel('Heretic Ratio ($\eta$)')
    plt.ylabel('Final Entropy (H)')
    plt.title('Mem4ristor v2.3: Sensitivity Analysis of Diversity Percolation\nFine-grained Sweep (0% to 30%)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    results_dir = os.path.join(ROOT_DIR, "results")
    os.makedirs(os.path.join(results_dir, "plots"), exist_ok=True)
    plt.savefig(os.path.join(results_dir, "plots/sensitivity_15_percent_fine.png"), dpi=300)
    print(f"\n✅ Fine sweep saved to {os.path.join(results_dir, 'plots/sensitivity_15_percent_fine.png')}")
    
    df.to_csv(os.path.join(results_dir, "data/fine_sensitivity_results.csv"), index=False)

if __name__ == "__main__":
    run_fine_sensitivity()
