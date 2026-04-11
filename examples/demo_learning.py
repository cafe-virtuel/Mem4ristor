import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from mem4ristor.mem4ristor_v3 import Mem4ristorV3
from mem4ristor.symbiosis import CreativeProjector
from mem4ristor.cortex import LearnableCortex

def demo_memory_consolidation():
    print("\n🧠 DEMO: Memory Consolidation (The Learnable Mind)")
    print("="*60)
    
    # 1. Initialize Hybrid System
    N_neurons = 50
    input_dim = 10 # Projection dimension (semantic space)
    
    print(f"1. Initializing Hybrid System (Mem4ristor N={N_neurons} + Cortex D={input_dim})...")
    
    # Limbic System (Mem4ristor)
    limbic = Mem4ristorV3(config={'dynamics': {'dt': 0.1}}, seed=42)
    # The Dreamer (Projector)
    dreamer = CreativeProjector(limbic, num_classes=input_dim, seed=42)
    # The Cortex (Long Term Memory)
    cortex = LearnableCortex(input_dim=input_dim, hidden_dim=20, output_dim=input_dim, seed=42)
    
    # 2. Simulate Trauma
    print("\n2. Simulating Trauma (Attack on Unit #5)...")
    # We artificially scar the Mem4ristor to simulate a "bad experience"
    # Unit 5 becomes resistant (High w)
    limbic.w[5] = 50.0  # Massive scar
    limbic.w[6] = 40.0  # Neighbor also affected
    
    # 3. Generate Nightmare (Dream Cycle)
    print("\n3. Entering REM Sleep (Dream Cycle)...")
    dream_log = dreamer.dream_cycle(steps=100)
    print(f"   Dream Log generated: {dream_log.shape}")
    
    # 4. Measure Cortex BEFORE Learning
    # Pick a pattern from the nightmare and see if Cortex recognizes it
    nightmare_pattern = dream_log[50] # Arbitrary middle of correct dream
    mse_before = cortex.get_mse_on_pattern(nightmare_pattern)
    print(f"   Cortex Familiarity (MSE) BEFORE sleep: {mse_before:.4f} (High error = Unknown)")
    
    # 5. Consolidation (Sleep & Learn)
    print("\n4. Consolidating Dreams into Cortex Weights...")
    final_loss = cortex.sleep_and_learn(dream_log, learning_rate=0.05, epochs=10)
    print(f"   Consolidation finished. Final training loss: {final_loss:.4f}")
    
    # 6. Measure Cortex AFTER Learning
    mse_after = cortex.get_mse_on_pattern(nightmare_pattern)
    print(f"   Cortex Familiarity (MSE) AFTER sleep:  {mse_after:.4f} (Low error = Remembered)")
    
    # 7. Verification
    improvement = mse_before - mse_after
    print("\n5. Verdict:")
    if improvement > 0.01:
        print(f"✅ SUCCESS: The Cortex successfully consolidated the nightmare.")
        print(f"   Improvement: {improvement:.4f}")
    else:
        print(f"❌ FAILURE: The Cortex did not learn significantly.")

if __name__ == "__main__":
    demo_memory_consolidation()
