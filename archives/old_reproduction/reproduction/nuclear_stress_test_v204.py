import numpy as np
import os
import yaml
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from mem4ristor.core import Mem4ristorV2

def run_nuclear_stress():
    print("☢️  LAUNCHING NUCLEAR STRESS TEST (HfO2 Physical Mapping) ☢️")
    print("-----------------------------------------------------------")
    
    # Load Physical Config
    cfg_path = os.path.join(os.path.dirname(__file__), "CONFIG_PHYSICAL.yaml")
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    model = Mem4ristorV2(config=cfg)
    model._initialize_params(100)
    
    # 1. BASELINE (Diversity search)
    print("[PHASE 1] Searching for diverse attractors...")
    for _ in range(1000):
        model.step(I_stimulus=0.4)
    h_start = model.calculate_entropy()
    print(f"  Baseline Entropy: {h_start:.4f}")

    # 2. THE ATTACK (Massive Consensus Pressure)
    # We apply a strong SET voltage to push everyone to the same state
    print("[PHASE 2] Initiating 'Consensus Attack' (I=1.2V)...")
    for _ in range(2000):
        model.step(I_stimulus=1.2)
    
    h_during = model.calculate_entropy()
    states = model.get_states()
    counts = np.bincount(states, minlength=6)[1:]
    
    # 3. ANALYSIS
    print(f"  Entropy under Pressure: {h_during:.4f}")
    print(f"  State Distribution: {counts}")
    
    # Resistance Metric: How many HERETICS refused the consensus?
    heretic_states = states[model.heretic_mask]
    dominant_state = np.argmax(np.bincount(states))
    dissident_count = np.sum(heretic_states != dominant_state)
    
    print("-----------------------------------------------------------")
    print(f"📊 DISSIDENCE SCORE: {dissident_count}/{np.sum(model.heretic_mask)} Heretics Resisted.")
    
    if dissident_count > 0:
        print("✅ SUCCESS: The system is PHYSICALLY RESILIENT to consensus collapse.")
    else:
        print("❌ FAILURE: Consensus collapse detected.")

if __name__ == "__main__":
    run_nuclear_stress()
