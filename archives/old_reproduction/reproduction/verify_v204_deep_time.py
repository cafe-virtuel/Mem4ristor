import numpy as np
import os
import sys

# Mem4ristor v2.0.4 - Deep Time Universal Verification Protocol
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.append(os.path.dirname(__file__))

from reference_impl import Mem4ristorV2

def run_deep_time_validation(steps=50000):
    print(f"[INFO] MEMRISTOR v2.0.4 | DEEP-TIME VALIDATION ({steps} steps)")

    print("-------------------------------------------------------")
    
    model = Mem4ristorV2()
    
    # Cold Start initialization (H=0)
    model.v[:] = 0.0
    model.w[:] = 0.0
    model.u[:] = 0.05
    
    h_history = []
    points_of_erasure = 0
    
    print("Starting simulation under strong bias (I_stim = 1.1)...")
    
    for i in range(steps):
        model.step(1.1)
        
        if i % 100 == 0:
            h = model.calculate_entropy()
            h_history.append(h)
            
            # Check for collapse (Edison's criteria) - ignore warm-up phase (first 5000 steps)
            if h < 0.05 and i > 5000:
                points_of_erasure += 1

                if points_of_erasure % 10 == 0: # Throttle logs
                    print(f"[WARN] Step {i}: Trace entropy dropped below threshold (H={h:.4f})")


        
        if i % 10000 == 0 and i > 0:
            avg_h = np.mean(h_history[-100:])
            print(f"Checkpoint {i:5d} | Moving Avg Entropy H: {avg_h:.4f}")

    final_h = np.mean(h_history[-10:])
    print("\n-------------------------------------------------------")
    print(f"FINAL REPORT v2.0.4:")
    print(f"Final Entropy H (Steady State): {final_h:.4f}")
    print(f"Total Points of Erasure Detected: {points_of_erasure}")
    
    success = True
    if points_of_erasure > 0:
        print("[FAIL] Stability vulnerability detected. Points of erasure still exist.")

        success = False
    elif final_h < 0.1:
        print("[FAIL] Global entropy level insufficient for cognitive health.")

        success = False
    else:
        print("[VERDICT] MEM4RISTOR v2.0.4 CERTIFIED (Eternal Stability)")

    
    return success

if __name__ == "__main__":
    run_deep_time_validation()
