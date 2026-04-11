import numpy as np
import pandas as pd
import os
import sys
import time
from itertools import product

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

def run_meta_audit():
    print("🤖 MEM4RISTOR META-AUDIT : Automated Parameter Sweep")
    print("--------------------------------------------------")
    
    # Espace Paramétrique (Grid Search)
    # Réduit pour le test, peut être étendu
    param_grid = {
        'i_stim': [0.5, 1.1, 1.8],         # Faible, Moyen, Fort
        'coupling': [0.1, 0.3],           # Faible, Fort
        'heretic_ratio': [0.0, 0.15, 0.3], # Pas d'élites, Standard, Fort
        'tau_u': [1.0, 5.0]                # Instantané, Mémoire
    }
    
    # Générer toutes les combinaisons
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    num_tests = len(combinations)
    print(f"[INFO] Starting {num_tests} test scenarios...")
    
    results = []
    size = 15
    steps = 1500
    
    start_time = time.time()
    
    for i, params in enumerate(combinations):
        print(f"[{i+1}/{num_tests}] Testing Stim={params['i_stim']} | D={params['coupling']} | H={params['heretic_ratio']} | Tau={params['tau_u']}...", end="\r")
        
        # Initialisation du modèle
        model = Mem4Network(size=size, heretic_ratio=params['heretic_ratio'], seed=42)
        model.model.cfg['coupling']['D'] = params['coupling']
        model.model.cfg['doubt']['tau_u'] = params['tau_u']
        
        # Simulation
        for _ in range(steps):
            model.step(I_stimulus=params['i_stim'])
            
        # Collecte des métriques finales
        v = model.v
        entropy = model.calculate_entropy()
        gini = calculate_gini(v)
        mean_u = np.mean(model.model.u)
        
        results.append({
            **params,
            'final_entropy': entropy,
            'final_gini': gini,
            'mean_doubt': mean_u,
            'justice_score': entropy * (1 - gini) * mean_u # Score agrégé (Antigravity Justice Index)
        })
        
    end_time = time.time()
    
    # Sauvegarde
    df = pd.DataFrame(results)
    output_file = "results/data/meta_audit_results.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    
    print(f"\n\n✅ [SUCCESS] Meta-Audit completed in {end_time - start_time:.2f}s")
    print(f"[INFO] Results saved to {output_file}")
    
    # Analyse rapide
    best_justice = df.loc[df['justice_score'].idxmax()]
    print("\n👑 WINNING CONFIGURATION (Max Justice Score):")
    print(best_justice)

if __name__ == "__main__":
    run_meta_audit()
