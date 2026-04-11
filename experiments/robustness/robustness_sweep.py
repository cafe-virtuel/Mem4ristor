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

def generate_phase_diagram():
    print("🚀 GENERATING MEM4RISTOR PHASE DIAGRAM (v2.3)")
    print("---------------------------------------------")
    
    # Paramètres du sweep
    d_range = np.linspace(0.0, 0.6, 15)
    r_range = np.linspace(0.0, 0.4, 15)
    i_stim = 1.1 # Pression constante
    steps = 2000
    
    results = []
    
    # Grid search
    for d in tqdm(d_range, desc="D Sweep"):
        for r in r_range:
            model = Mem4Network(size=10, heretic_ratio=r, seed=42)
            model.model.cfg['coupling']['D'] = d
            
            # Monitoring de l'entropie
            h_list = []
            for t in range(steps):
                model.step(I_stimulus=i_stim)
                if t > steps // 2: # On attend la stabilisation
                    h_list.append(model.calculate_entropy())
            
            avg_h = np.mean(h_list)
            results.append({'D': d, 'Ratio': r, 'Entropy': avg_h})
            
    df = pd.DataFrame(results)
    
    # Pivot pour heatmap
    pivot = df.pivot(index='Ratio', columns='D', values='Entropy')
    
    # Plotting
    plt.figure(figsize=(10, 8))
    plt.imshow(pivot, origin='lower', extent=[0.0, 0.6, 0.0, 0.4], aspect='auto', cmap='viridis')
    plt.colorbar(label='Sustained Entropy (H)')
    plt.xlabel('Coupling Strength (D)')
    plt.ylabel('Heretic Ratio')
    plt.title(f'Phase Diagram: Attractor Diversity Stabilization (v2.3)\n$I_{{stim}} = {i_stim}$')
    
    # Add contour for "Diversity Zone" (H > 1.0)
    plt.contour(pivot, levels=[1.0], colors='white', extent=[0.0, 0.6, 0.0, 0.4])
    plt.text(0.3, 0.2, "DIVERSITY ZONE", color='white', fontweight='bold', ha='center')
    
    results_dir = os.path.join(ROOT_DIR, "results")
    os.makedirs(os.path.join(results_dir, "plots"), exist_ok=True)
    plt.savefig(os.path.join(results_dir, 'plots/phase_diagram_v23.png'), dpi=300)
    print(f"\n✅ Phase diagram saved to {os.path.join(results_dir, 'plots/phase_diagram_v23.png')}")
    
    # Save CSV
    os.makedirs(os.path.join(results_dir, 'data'), exist_ok=True)
    df.to_csv(os.path.join(results_dir, 'data/robustness_sweep_results.csv'), index=False)

if __name__ == "__main__":
    generate_phase_diagram()
