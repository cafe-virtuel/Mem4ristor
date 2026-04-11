import numpy as np
import pandas as pd
import os
import sys
import time
from itertools import product
import matplotlib.pyplot as plt

# Integration avec le dossier src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from mem4ristor.core import Mem4Network

def calculate_gini(v):
    x = np.abs(v)
    if np.sum(x) == 0: return 0
    n = len(x)
    x = np.sort(x)
    index = np.arange(1, n + 1)
    return (np.sum((2 * index - n - 1) * x)) / (n * np.sum(x))

def run_mass_audit_extreme():
    print("🌪️ MEM4RISTOR MASS AUDIT : Extreme Stress-Test")
    print("---------------------------------------------")
    
    # Espace Paramétrique Élargi (Extreme Ranges)
    param_grid = {
        'i_stim': [0.0, 1.1, 2.5, 5.0],        # De nul à dictatorial
        'coupling': [0.05, 0.2, 0.8],         # De fluide à asphyxiant
        'heretic_ratio': [0.0, 0.05, 0.15, 0.3], # De 0% (Mort) à 30% (Elite)
        'tau_u': [0.1, 1.0, 10.0]             # De réactif à amnésique
    }
    
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    num_tests = len(combinations)
    print(f"[INFO] Launching {num_tests} extreme scenarios...")
    
    results = []
    size = 10 # Réduit pour la vitesse du sweep massif
    steps = 1000
    
    start_time = time.time()
    
    for i, params in enumerate(combinations):
        if i % 10 == 0:
            print(f"Progress: [{i}/{num_tests}] ({(i/num_tests)*100:.1f}%)", end="\r")
        
        # Initialisation du modèle
        model = Mem4Network(size=size, heretic_ratio=params['heretic_ratio'], seed=42)
        model.model.cfg['coupling']['D'] = params['coupling']
        model.model.cfg['doubt']['tau_u'] = params['tau_u']
        
        # Simulation
        for _ in range(steps):
            model.step(I_stimulus=params['i_stim'])
            
        # Collecte
        v = model.v
        entropy = model.calculate_entropy()
        gini = calculate_gini(v)
        mean_u = np.mean(model.model.u)
        
        # Statut du système
        status = "ALIVE (Justice)" if entropy > 1.2 else "DEAD (Consensus)"
        if params['heretic_ratio'] == 0: status = "EXTINCT (No Diversity)"
            
        results.append({
            **params,
            'final_entropy': entropy,
            'final_gini': gini,
            'mean_doubt': mean_u,
            'justice_score': entropy * (1 - gini) * mean_u,
            'system_status': status
        })
        
    end_time = time.time()
    df = pd.DataFrame(results)
    
    output_file = "results/data/mass_audit_extreme_results.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"\n\n✅ [SUCCESS] Audit completed. Results saved to {output_file}")
    
    # --- Analyse de Rupture ---
    print("\n🔍 BREAKING POINT ANALYSIS:")
    
    # 1. Impact de l'absence totale d'hérétiques
    extinct = df[df['heretic_ratio'] == 0]['final_entropy'].mean()
    normal = df[df['heretic_ratio'] == 0.15]['final_entropy'].mean()
    print(f"  - Mean Entropy (H=0%): {extinct:.4f}")
    print(f"  - Mean Entropy (H=15%): {normal:.4f}")
    print(f"  - Fragility Factor: {((normal-extinct)/normal)*100:.1f}% loss without minority.")
    
    # 2. Point de rupture Stimulus
    # On cherche le moment où l'entropie s'effondre malgré les hérétiques
    breaking_stim = df[df['final_entropy'] < 0.5]['i_stim'].min()
    print(f"  - Critical Stimulus (Death Threshold): {breaking_stim}")

    # Visualisation de la 'Justice Zone' (Heatmap Stimulus vs Hérétiques)
    try:
        pivot_data = df[df['coupling'] == 0.2].pivot_table(
            index='heretic_ratio', columns='i_stim', values='justice_score'
        )
        plt.figure(figsize=(10, 6))
        plt.imshow(pivot_data, origin='lower', aspect='auto', cmap='hot', 
                   extent=[0, 5, 0, 0.3])
        plt.colorbar(label='Justice Score (H * (1-Gini) * u)')
        plt.title("Mem4ristor Cognitive Justice Map : From Life to Death")
        plt.xlabel("Social Pressure (I_stim)")
        plt.ylabel("Structural Diversity (Heretic Ratio)")
        plot_path = "results/plots/justice_breaking_map.png"
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path)
        print(f"[INFO] Heatmap saved to {plot_path}")
    except Exception as e:
        print(f"[NOTE] Viz failed: {e}")

if __name__ == "__main__":
    run_mass_audit_extreme()
