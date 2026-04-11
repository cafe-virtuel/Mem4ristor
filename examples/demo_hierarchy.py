import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from mem4ristor.hierarchy import HierarchicalChimera

def demo_deep_chimera():
    print("\n🏗️ DEMO: Deep Chimera (Hierarchical Architecture)")
    print("="*60)
    
    # 1. Initialize Hierarchy (V1 -> V4 -> PFC)
    brain = HierarchicalChimera(seed=42)
    print("1. Deep Chimera online (V1=64, V4=49, PFC=36).")
    
    # 2. Feedforward Propagation Scenario
    # We inject noise into V1 and watch PFC react
    steps = 100
    pfc_history = []
    v1_history = []
    martial_law_events = 0
    
    print("2. Starting Simulation (100 steps)...")
    
    for t in range(steps):
        # Sensory Input: Random noise / Stress
        stimulus = np.random.randn(64) * 0.5
        
        # Step
        status = brain.step(stimulus)
        
        # Log
        pfc_history.append(status['PFC_mean'])
        v1_history.append(status['V1_mean'])
        
        if status['PFC_status']['martial_law']:
            martial_law_events += 1
            if martial_law_events == 1:
                print(f"   ⚠️ MARTIAL LAW DECLARED at step {t}!")
                
    # 3. Analyze Dynamics
    v1_std = np.std(v1_history)
    pfc_std = np.std(pfc_history)
    
    print(f"\n3. Dynamics Analysis:")
    print(f"   V1 Volatility (Std): {v1_std:.4f} (Should be High)")
    print(f"   PFC Stability (Std): {pfc_std:.4f} (Should be Low/Controlled)")
    
    ratio = v1_std / (pfc_std + 1e-9)
    print(f"   Damping Ratio: {ratio:.2f}x")
    
    # 4. Verdict
    print("\n4. Verdict:")
    if ratio > 1.1:
        print(f"✅ SUCCESS: Hierarchy acts as a filter.")
        print(f"   PFC is {ratio:.1f}x more stable than V1.")
        if martial_law_events > 0:
            print(f"   Top-Down Control successful ({martial_law_events} inhibitor events).")
    else:
        print(f"❌ FAILURE: Layers are synchronized or unstable.")
        print(f"   Ratio: {ratio:.2f}")

if __name__ == "__main__":
    demo_deep_chimera()
