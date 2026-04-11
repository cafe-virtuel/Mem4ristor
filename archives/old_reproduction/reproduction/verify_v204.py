import numpy as np
import os
import sys

# MEM4RISTOR v2.0.4 | INDUSTRIAL VERIFICATION SUITE
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.append(os.path.dirname(__file__))

from reference_impl import Mem4ristorV2

def run_verification():
    print("[INFO] MEMRISTOR v2.0.4 | INDUSTRIAL VERIFICATION PROTOCOL")

    print("-------------------------------------------------------")
    
    success = True
    
    # --- PHASE 1: COLD START (Resurrection) ---
    print("\n[PHASE 1] ACTIVE RESTORATION (Cold Start)")
    model_cold = Mem4ristorV2()
    model_cold.v[:] = 0.0
    model_cold.w[:] = 0.0
    model_cold.u[:] = 0.05
    
    h_idx = np.where(model_cold.heretic_mask)[0][0]
    n_idx = np.where(~model_cold.heretic_mask)[0][0]
    
    h_cold_list = []
    for i in range(2500): 
        model_cold.step(1.1)
        if i >= 2000:
            h_cold_list.append(model_cold.calculate_entropy())
        if i % 500 == 0:
            print(f"Step {i:4d} | v[n]: {model_cold.v[n_idx]:.4f} | v[h]: {model_cold.v[h_idx]:.4f} | H: {model_cold.calculate_entropy():.4f}")

    h_cold = np.mean(h_cold_list)


    print(f"Post-Resurrection Entropy H: {h_cold:.4f}")
    if h_cold > 0.1:
        print("[PASS] Symmetry successfully broken from zero-state.")

    else:
        print("[FAIL] Resurrection failed.")

        success = False
        
    # --- PHASE 2: DEEP TIME STABILITY (Steady State) ---
    print("\n[PHASE 2] DEEP-TIME STABILITY (10,000 steps)")
    model_deep = Mem4ristorV2()
    h_history = []
    for i in range(10000):
        model_deep.step(1.1)
        if i >= 5000 and i % 100 == 0:
            h_history.append(model_deep.calculate_entropy())
            
    avg_h = np.mean(h_history)
    print(f"Steady State Entropy (Avg): {avg_h:.4f}")
    if avg_h > 1.2:
        print("[PASS] Stable multi-modal state achieved.")

    else:
        print("[FAIL] Insufficient steady-state diversity.")

        success = False
        
    # --- PHASE 3: CAUSAL ISOLATION (Ablation) ---
    print("\n[PHASE 3] CAUSAL ISOLATION (Ablation Study)")
    model_abl = Mem4ristorV2()
    model_abl.v[:] = 0.0 # Force collapse risk
    model_abl.u[:] = 0.05
    model_abl.heretic_mask[:] = False # Remove mechanism
    
    for _ in range(5000): model_abl.step(1.1)
    h_abl = model_abl.calculate_entropy()
    print(f"Ablated State Entropy H: {h_abl:.4f}")
    if h_abl < 0.20:

        print("[PASS] Mechanism isolated. System collapses without heretics.")

    else:
        print("[FAIL] Spurious diversity detected!")

        success = False
        
    print("\n-------------------------------------------------------")
    if success:
        print("[VERDICT] MEM4RISTOR v2.0.4 OFFICIALLY VERIFIED")

    else:
        print("[FAIL] VERDICT: VERIFICATION FAILED")

        sys.exit(1)

if __name__ == "__main__":
    run_verification()
