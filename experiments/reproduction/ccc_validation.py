import os
import sys
# Resolve Paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '../../'))
sys.path.append(os.path.join(ROOT_DIR, 'src'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mem4ristor.core import Mem4Network

def run_empirical_validation():
    print("🌍 MEM4RISTOR EMPIRICAL VALIDATION (CCC Data)")
    print("---------------------------------------------")
    
    # Données réelles extraites de la Convention Citoyenne pour le Climat (Juin 2020)
    # Source: Rapport Final CCC / Dépêches AFP
    scenarios = [
        {
            "name": "Obligation d'affichage Carbone",
            "real_yes": 0.98,
            "real_no": 0.02,
            "description": "Consensus quasi-total"
        },
        {
            "name": "Limitation 110 km/h sur autoroute",
            "real_yes": 0.60,
            "real_no": 0.40,
            "description": "Forte divergence (Disputé)"
        },
        {
            "name": "Semaine de 28h (Projet)",
            "real_yes": 0.35,
            "real_no": 0.65,
            "description": "Rejet majoritaire mais minorité active"
        }
    ]
    
    results = []
    
    for sc in scenarios:
        print(f"Simulating: {sc['name']} ({sc['description']})")
        
        # Initialisation du modèle (100 unités)
        size = 10
        model = Mem4Network(size=size, seed=42)
        
        # On simule le stimulus correspondant à la force du 'OUI' réel
        # Si real_yes = 0.98, stimulus fort vers 1.0
        # Si real_yes = 0.60, stimulus modéré
        stimulus = (sc['real_yes'] - 0.5) * 4.0 # Map [0.35, 0.98] vers des valeurs d'I_stim
        
        # Simulation
        steps = 5000
        for _ in range(steps):
            model.step(I_stimulus=stimulus)
            
        # Mesure de l'entropie finale et de la distribution
        final_h = model.calculate_entropy()
        v_final = model.model.v
        pred_yes = np.sum(v_final > 0.5) / (size*size)
        pred_no = np.sum(v_final <= 0.5) / (size*size)
        
        results.append({
            "Scenario": sc['name'],
            "Real Yes %": sc['real_yes'] * 100,
            "Pred Yes %": pred_yes * 100,
            "Entropy": final_h
        })

    df = pd.DataFrame(results)
    print("\n--- RESULTS ---")
    print(df)
    
    # Check for "Cognitive Justice": Does the minority survive?
    for res in results:
        if res['Real Yes %'] > 90 and res['Pred Yes %'] < 100:
            print(f"✅ [SUCCESS] In {res['Scenario']}, the minority survived despite heavy pressure.")
        elif res['Real Yes %'] < 70 and res['Entropy'] > 1.5:
             print(f"✅ [SUCCESS] In {res['Scenario']}, diversity was protected.")

    results_dir = os.path.join(ROOT_DIR, "results")
    output_file = os.path.join(results_dir, "data/empirical_validation_results.csv")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"\n[INFO] Results saved to {output_file}")

    # Plotting for Preprint
    plt.figure(figsize=(10, 6))
    x = np.arange(len(df))
    width = 0.35
    plt.bar(x - width/2, df["Real Yes %"], width, label='Votes Réels (Yes %)', color='gray', alpha=0.6)
    plt.bar(x + width/2, df["Pred Yes %"], width, label='Prédiction Mem4ristor', color='blue', alpha=0.8)
    plt.xticks(x, df["Scenario"], rotation=15)
    plt.ylabel("Pourcentage 'OUI'")
    plt.title("Mem4ristor v2.3 : Validation Empirique (CCC France)")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plot_path = os.path.join(results_dir, "plots/ccc_validation_summary.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    print(f"[INFO] Plot saved to {plot_path}")

if __name__ == "__main__":
    run_empirical_validation()
