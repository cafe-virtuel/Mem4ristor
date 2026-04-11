import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# Integration avec le dossier src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from mem4ristor.core import Mem4Network

def run_byzantine_audit():
    print("🛡️ MEM4RISTOR BYZANTINE RESILIENCE AUDIT")
    print("---------------------------------------")
    
    size = 10
    steps = 4000
    model = Mem4Network(size=size, seed=42)
    
    # Identifier les indices des "Hérétiques" et des "Normaux"
    heretic_indices = np.where(model.model.heretic_mask)[0]
    normal_indices = np.where(~model.model.heretic_mask)[0]
    
    # L'Attaquant Byzantin est l'unité 0 (un neurone normal)
    byzantine_id = normal_indices[0]
    print(f"[ATTACK] Unit {byzantine_id} (Normal) is now corrupted.")
    print("[ATTACK] Injecting forced consensus signal (V_forced = 2.0)")

    entropy_history = []
    v_byzantine_history = []
    v_neighbors_history = []
    
    # Trouver les voisins directs de l'attaquant pour l'analyse
    # En 10x10, l'unité indexée byzantine_id a des voisins calculables
    bx, by = byzantine_id // size, byzantine_id % size
    neighbors = []
    for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
        nx, ny = bx + dx, by + dy
        if 0 <= nx < size and 0 <= ny < size:
            neighbors.append(nx * size + ny)

    for i in range(steps):
        # 1. Calculer le step normalement
        model.step(I_stimulus=1.1)
        
        # 2. Injection Byzantinement forcée : Le neurone corrompu écrase son état
        # Il tente de forcer tout le monde à être d'accord avec lui (V=+2.0)
        model.model.v[byzantine_id] = 2.0
        
        if i % 100 == 0:
            h = model.calculate_entropy()
            entropy_history.append(h)
            v_byzantine_history.append(model.model.v[byzantine_id])
            v_neighbors_history.append(np.mean(model.model.v[neighbors]))
            
            if i % 1000 == 0:
                print(f"Step {i:4d} | Entropy: {h:.4f} | Neighbors Avg V: {v_neighbors_history[-1]:.4f}")

    print("---------------------------------------")
    final_h = model.calculate_entropy()
    print(f"Final Resilience Entropy: {final_h:.4f}")
    
    if final_h > 1.2:
        print("✅ [AUDIT SUCCESS] The network resisted the forced consensus.")
        print("   The constitutional doubt successfully isolated the byzantine influence.")
    else:
        print("❌ [AUDIT FAILURE] The network collapsed into consensus.")
        print("   Byzantine corruption spread to the entire system.")
        sys.exit(1)

if __name__ == "__main__":
    run_byzantine_audit()
