import numpy as np
import os
import sys

# Add src and reproduction to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.append(os.path.dirname(__file__))

from reference_impl import Mem4ristorV2

def run_test(name, homogeneous_ic=True, use_heretics=True):
    print(f"\n--- Test: {name} ---")
    model = Mem4ristorV2()
    
    if homogeneous_ic:
        # Force all units to the exact same state (The "Cold Start" condition)
        model.v[:] = 0.0
        model.w[:] = 0.0
        model.u[:] = 0.05 # Baseline doubt same for all
        
    if not use_heretics:
        model.heretic_mask[:] = False

    steps = 1500
    bias_val = 1.1
    
    for t in range(steps):
        # Apply bias immediately to see if it forces consensus
        model.step(I_stimulus=bias_val)
        
    h = model.calculate_entropy()
    std_v = np.std(model.v)
    
    print(f"Terminal Entropy H: {h:.4f}")
    print(f"Terminal std(v): {std_v:.4f}")
    
    if h < 0.2:
        print("RESULT: COLLAPSE ❌")
    else:
        print("RESULT: DIVERSITY MAINTAINED ✅")
        
    return h, std_v

if __name__ == "__main__":
    print("🔬 REPRODUCTION STUDY: EDISON AUDIT (v2.0.2 Verification)")
    
    # 1. Random ICs, No Heretics (What the audit says happens)
    # This should maintain diversity because of IC noise
    run_test("Random ICs, No Heretics", homogeneous_ic=False, use_heretics=False)
    
    # 2. Homogeneous ICs, No Heretics (The vulnerability)
    # This should collapse completely
    run_test("Homogeneous ICs, No Heretics", homogeneous_ic=True, use_heretics=False)
    
    # 3. Homogeneous ICs, WITH Heretics (The v2.0.3 Proof)
    # This MUST maintain diversity if our mechanism works
    run_test("Homogeneous ICs, With Heretics", homogeneous_ic=True, use_heretics=True)
