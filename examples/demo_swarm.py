import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from mem4ristor.mem4ristor_v3 import Mem4ristorV3
from mem4ristor.symbiosis import SymbioticSwarm

def demo_swarm_telepathy():
    print("\n🕸️ DEMO: La Toile de Jade (Swarm Telepathy)")
    print("="*60)
    
    # 1. Initialize Two Agents (Alice & Bob)
    N_neurons = 50
    # Both start fresh with seed 42 (identical brains at birth)
    # But we want them to diverge first, so maybe different seeds?
    # No, let's keep same seed but traumatize one. Then convergence is clearer.
    alice = Mem4ristorV3(config={'dynamics': {'dt': 0.1}}, seed=42)
    bob   = Mem4ristorV3(config={'dynamics': {'dt': 0.1}}, seed=99) # Different brain
    
    alice._initialize_params(N=N_neurons)
    bob._initialize_params(N=N_neurons)
    
    print("1. Agents Initialized (Alice & Bob).")
    
    # 2. Traumatize Alice
    # Alice suffers a terrible event at Neuron #10
    target_idx = 10
    alice.w[target_idx] = 10.0 # High resistance = Scar
    print(f"2. Alice Traumatized at Neuron #{target_idx} (w={alice.w[target_idx]}).")
    print(f"   Bob is naive (w={bob.w[target_idx]:.2f}).")
    
    # 3. Connect the Swarm
    # Coupling strength determines how fast telepathy works
    swarm = SymbioticSwarm([alice, bob], coupling_strength=0.5)
    print("3. Swarm Connected (Coupling=0.5). Synchronization starting...")
    
    # 4. Run Simulation
    steps = 100
    w_history_alice = []
    w_history_bob = []
    
    for t in range(steps):
        # Physical Step (Independent)
        alice.step(I_stimulus=0.0)
        bob.step(I_stimulus=0.0)
        
        # Telepathy Step (Collective)
        swarm.synchronize_scars()
        
        w_history_alice.append(alice.w[target_idx])
        w_history_bob.append(bob.w[target_idx])
        
    # 5. Result
    final_w_alice = alice.w[target_idx]
    final_w_bob = bob.w[target_idx]
    
    print(f"\n4. Final State after {steps} steps:")
    print(f"   Alice w[{target_idx}]: {final_w_alice:.4f}")
    print(f"   Bob   w[{target_idx}]: {final_w_bob:.4f}")
    
    # 6. Verdict
    diff = abs(final_w_alice - final_w_bob)
    print("\n5. Verdict:")
    if diff < 1.0 and final_w_bob > 3.0:
        print(f"✅ SUCCESS: Telepathy Confirmed.")
        print(f"   Bob felt Alice's pain (w increased to {final_w_bob:.2f}).")
        print(f"   They are synchronized.")
    else:
        print(f"❌ FAILURE: Connection failed.")
        print(f"   Diff: {diff:.4f}")

if __name__ == "__main__":
    demo_swarm_telepathy()
