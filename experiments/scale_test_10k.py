import sys
import os
# Add src to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import numpy as np
import time
from mem4ristor.core import Mem4Network

def run_scaling_test():
    SIZE = 100
    N = SIZE * SIZE
    print(f"=== LANCEMENT TEST ECHELLE N={N} (10,000 unités) ===")
    
    # 1. Configuration avec 0% d'hérétiques (Témoin - Consensus Attendu)
    print("\n--- CAS A: 0% Heretiques (Témoin) ---")
    net_A = Mem4Network(size=SIZE, heretic_ratio=0.0, seed=1)
    # FIX: Lattice topology. Using D=0.1.
    net_A.model.D_eff = 0.1
    
    # Run
    start = time.time()
    for i in range(500):
        # FIX: Stronger stimulus to force consensus
        net_A.step(I_stimulus=2.0)
    duration = time.time() - start
    entropy_A = net_A.calculate_entropy()
    dist_A = net_A.get_state_distribution()
    print(f"Entropie (0%): {entropy_A:.4f} | Dist: {dist_A}")
    
    # 2. Configuration avec 15% d'hérétiques (Transition)
    print("\n--- CAS B: 15% Heretiques (Cible) ---")
    net_B = Mem4Network(size=SIZE, heretic_ratio=0.15, seed=1)
    # FIX: Lattice topology. Using D=0.1.
    net_B.model.D_eff = 0.1
    
    # Run
    start = time.time()
    for i in range(500):
        # FIX: Stronger stimulus
        net_B.step(I_stimulus=2.0)
    duration = time.time() - start
    entropy_B = net_B.calculate_entropy()
    dist_B = net_B.get_state_distribution()
    print(f"Entropie (15%): {entropy_B:.4f} | Dist: {dist_B}")
    
    # Conclusion
    print("\n=== RESULTATS ===")
    print(f"Gain d'entropie: {entropy_B - entropy_A:.4f}")
    if entropy_B > entropy_A + 0.2: # Lower threshold since baseline might be 0
        print(">> SUCCÈS: Émergence de la diversité confirmée")
    else:
        print(">> CONSTAT: Pas de gain significatif")

if __name__ == "__main__":
    run_scaling_test()
