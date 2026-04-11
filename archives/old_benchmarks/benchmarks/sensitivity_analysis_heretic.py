import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mem4ristor.core import Mem4Network
import os
from tqdm import tqdm

def run_sensitivity_analysis():
    print("🔬 SENSITIVITY ANALYSIS: HERETIC RATIO PERCOLATION (v2.3)")
    print("-------------------------------------------------------")
    
    ratios = np.linspace(0.0, 0.4, 21) # 2% steps
    seeds = [42, 123, 777, 888, 999] # Multiple runs for error bars
    steps = 3000
    i_stim = 1.1
    
    all_results = []
    
    for r in tqdm(ratios, desc="Ratio Sweep"):
        r_entropies = []
        for s in seeds:
            model = Mem4Network(size=10, heretic_ratio=r, seed=s)
            # Paramètres standards
            for t in range(steps):
                model.step(I_stimulus=i_stim)
            
            r_entropies.append(model.calculate_entropy())
            
        all_results.append({
            'ratio': r,
            'entropy_mean': np.mean(r_entropies),
            'entropy_std': np.std(r_entropies)
        })
        
    df = pd.DataFrame(all_results)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.errorbar(df['ratio'], df['entropy_mean'], yerr=df['entropy_std'], 
                 fmt='-o', color='tab:red', ecolor='gray', capsize=5, label='Mean Entropy (H)')
    
    # Annotations topologiques
    plt.axvline(x=0.15, color='black', linestyle='--', alpha=0.5)
    plt.text(0.155, 0.5, 'Percolation Threshold ($\sim$15%)', rotation=90, verticalalignment='center')
    
    plt.fill_between(df['ratio'], 0, df['entropy_mean'], where=(df['ratio'] < 0.12), color='red', alpha=0.1, label='Collapse Zone')
    plt.fill_between(df['ratio'], 0, df['entropy_mean'], where=(df['ratio'] >= 0.12), color='green', alpha=0.1, label='Diversity Zone')
    
    plt.xlabel('Heretic Ratio ($\eta$)')
    plt.ylabel('Steady-State Entropy (H)')
    plt.title('Emergence of Diversity: Sensitivity Analysis of Heretic Ratio\n(Standard Lattice, $I_{stim}=1.1$)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs('results/plots', exist_ok=True)
    plt.savefig('results/plots/sensitivity_heretic_percolation.png', dpi=300)
    print("\n✅ Sensitivity plot saved to results/plots/sensitivity_heretic_percolation.png")
    
    # Save CSV
    os.makedirs('results/data', exist_ok=True)
    df.to_csv('results/data/sensitivity_analysis_results.csv', index=False)

if __name__ == "__main__":
    run_sensitivity_analysis()
