import numpy as np
import os
import sys

# Standard Edison Stress Test: Long-term Stability and Epistemic Isolation
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.append(os.path.dirname(__file__))

from reference_impl import Mem4ristorV2

def stress_test_isolation(steps=20000):
    print(f"🚩 EDISON STRESS TEST: LONG-TERM ISOLATION (Steps={steps})")
    model = Mem4ristorV2()
    
    # Cold Start
    model.v[:] = 0.0
    model.w[:] = 0.0
    model.u[:] = 0.05
    
    h_idx = np.where(model.heretic_mask)[0][0]
    
    u_history = []
    h_history = []
    
    for i in range(steps):
        model.step(1.1)
        u_history.append(model.u[h_idx])
        if i % 100 == 0:
            h = model.calculate_entropy()
            h_history.append(h)
            if h < 0.0001 and len(h_history) > 1 and h_history[-2] > 0.0001:
                print(f"🚩 CRITICAL: Point of Erasure located at Step {i}!")

            
    print(f"Final u[heretic]: {model.u[h_idx]:.4f}")
    print(f"Final Entropy H: {model.calculate_entropy():.4f}")
    
    # CRITIQUE 1: Epistemic Isolation
    if model.u[h_idx] > 0.95:
        print("🔍 AUDIT FINDING: Epistemic Isolation detected. Heretics effectively 'unplug' from the network.")
    
    # CRITIQUE 2: Diversity Decay
    recent_h = h_history[-5:]
    if np.mean(recent_h) < 0.1:
        print("🔍 AUDIT FINDING: Long-term diversity decay. Restoration is transient.")
    else:
        print("🔍 AUDIT FINDING: Diversity remains stable (albeit low).")

if __name__ == "__main__":
    stress_test_isolation()
