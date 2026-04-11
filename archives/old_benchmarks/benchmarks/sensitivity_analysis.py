import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mem4ristor.core import Mem4Network
import os

def run_sensitivity_analysis():
    print("📊 MEM4RISTOR SENSITIVITY ANALYSIS (Justice Mapping)")
    print("--------------------------------------------------")
    
    # Paramètres à tester
    heretic_ratios = np.linspace(0.0, 0.4, 9)  # 0% à 40%
    coupling_strengths = np.linspace(0.1, 1.0, 10) # D_coupling
    
    results = []
    
    steps = 2000
    size = 10
    
    for ratio in heretic_ratios:
        for coupling in coupling_strengths:
            print(f"Testing Ratio: {ratio:.2f} | Coupling: {coupling:.2f}...", end="\r")
            
            # Initialisation du modèle avec les paramètres spécifiques
            # Note: ratio d'hérétiques est géré à l'initialisation du modèle interne
            model = Mem4Network(size=size, seed=42)
            model.model.heretic_ratio = ratio
            # On force la régénération du masque si nécessaire (implémentation simplifiée ici)
            model.model.heretic_mask = np.random.rand(size*size) < ratio
            
            # Injection de stimulus
            for _ in range(steps):
                model.step(I_stimulus=1.1)
            
            final_h = model.calculate_entropy()
            results.append({
                "heretic_ratio": ratio,
                "coupling_strength": coupling,
                "entropy": final_h
            })

    df = pd.DataFrame(results)
    pivot_df = df.pivot(index="heretic_ratio", columns="coupling_strength", values="entropy")
    
    # Save results
    output_path = "results/data/sensitivity_results.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print("\n[SUCCESS] Sensitivity data saved to benchmarks/sensitivity_results.csv")
    
    # Visualisation (Optionnel si matplotlib est disponible)
    try:
        plt.figure(figsize=(10, 8))
        plt.imshow(pivot_df, extent=[0.1, 1.0, 0, 0.4], origin='lower', aspect='auto', cmap='viridis')
        plt.colorbar(label='Final Entropy (bits)')
        plt.xlabel('Coupling Strength')
        plt.ylabel('Heretic Ratio')
        plt.title('Mem4ristor Stability Map (Justice Zone)')
        plot_path = "results/plots/sensitivity_map.png"
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path)
        print(f"[INFO] Sensitivity map saved to {plot_path}")
    except Exception as e:
        print(f"[NOTE] Visualization skipped: {e}")

if __name__ == "__main__":
    run_sensitivity_analysis()
