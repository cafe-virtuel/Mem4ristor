import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Integration avec le dossier src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from mem4ristor.core import Mem4Network

def run_mediation_audit():
    print("⚖️ MEM4RISTOR PHASE 6 AUDIT : Multi-Source Mediation")
    print("--------------------------------------------------")
    
    size = 20
    model = Mem4Network(size=size, seed=42)
    
    # Création du champ de stimulus bipolaire
    # Col 0-4 : Stimulus +1.2 (Opinion A)
    # Col 15-19 : Stimulus -1.2 (Opinion B)
    # Centre : Pas de stimulus direct (Zone de médiation)
    stim_grid = np.zeros((size, size))
    stim_grid[:, :5] = 1.2
    stim_grid[:, 15:] = -1.2
    stim_flat = stim_grid.flatten()
    
    print("[START] Simulating bipolar pressure (2000 steps)...")
    for _ in range(2000):
        model.step(I_stimulus=stim_flat)
        
    final_v = model.v.reshape((size, size))
    final_u = model.model.u.reshape((size, size))
    
    # Analyse de la zone de médiation (colonnes centrales 8-12)
    mediation_u = np.mean(final_u[:, 8:12])
    print(f"\n[INFO] Mean Doubt in Conflict Zone: {mediation_u:.4f}")
    
    if mediation_u > 0.4:
        print("✅ [AUDIT SUCCESS] High Doubt detected in the mediation zone.")
        print("   The system correctly identified the conflict and refused a binary choice.")
    else:
        print("❌ [AUDIT FAILURE] The system collapsed into a binary wall without active doubt.")

    # Visualisation
    try:
        os.makedirs("results", exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        im1 = ax1.imshow(final_v, cmap='RdBu_r', vmin=-2, vmax=2)
        ax1.set_title("Potentiel Cognitif (V)\n(Pôles Opposés)")
        plt.colorbar(im1, ax=ax1)
        
        im2 = ax2.imshow(final_u, cmap='viridis', vmin=0, vmax=1)
        ax2.set_title("Niveau de Doute (u)\n(Frontière de Médiation)")
        plt.colorbar(im2, ax=ax2)
        
        plt.tight_layout()
        plt.savefig("results/multi_source_mediation.png")
        print("[INFO] Results visualization saved to results/multi_source_mediation.png")
    except Exception as e:
        print(f"[NOTE] Visualization skipped: {e}")

if __name__ == "__main__":
    run_mediation_audit()
