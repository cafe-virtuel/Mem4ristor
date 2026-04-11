import os
import sys
import numpy as np
import json

# Ensure src is in path for local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from mem4ristor import Mem4Network, Mem4ristorV2

def run_comparative_sim(size=10, steps=300):
    logs = []
    logs.append(f"🔬 Starting Grok's Hardware Sim: Collective Intelligence vs. Blind Consensus ({size}x{size})")
    
    # 1. Setup Populations
    # Network A: Classical (No doubt, No heretics)
    net_a = Mem4Network(size=size, heretic_ratio=0.0)
    for row in net_a.grid:
        for unit in row:
            unit.k_u = 0.0  # Disable social doubt
            unit.sigma_baseline = 0.0 # No internal doubt
            unit.u = 0.0 # Force zero doubt
            
    # Network B: Mem4ristor v2 (The Architect's choice)
    net_b = Mem4Network(size=size, heretic_ratio=0.15)
    
    # 2. Simulation Loop
    history_a = []
    history_b = []
    
    for i in range(steps):
        if i < 80:
            stim = 0.4 # Baseline signal
        elif i < 180:
            stim = 1.1 # Subtle bias (The Seducing Error)
        else:
            stim = 0.0 # Return to neutrality
            
        net_a.step(I_stimulus=stim)
        net_b.step(I_stimulus=stim)
        
        # Track Diversity (Number of unique states)
        dist_a = net_a.get_state_distribution()
        dist_b = net_b.get_state_distribution()
        
        div_a = len(dist_a)
        div_b = len(dist_b)
        
        history_a.append(div_a) # Using diversity as the key metric now
        history_b.append(div_b)
        
        if i % 30 == 0:
            logs.append(f"Step {i:03d} | Div A: {div_a} | Div B: {div_b} | Doubt B: {net_b.get_mean_doubt():.3f} | States B: {dist_b}")

    # 3. Report Results
    logs.append("\n📊 --- FINAL REPORT (Diversity Metric) ---")
    mean_div_a = np.mean(history_a[80:180])
    mean_div_b = np.mean(history_b[80:180])
    logs.append(f"Blind Consensus (A) mean diversity: {mean_div_a:.2f}")
    logs.append(f"Doubt-Protected (B) mean diversity: {mean_div_b:.2f}")
    
    if mean_div_b > mean_div_a:
        improvement = (mean_div_b - mean_div_a) / mean_div_a
        logs.append(f"✅ Result: The Doubt architecture maintained {improvement*100:.1f}% more cognitive diversity.")
        logs.append("🧠 Logic: Doubt prevents the network from collapsing into a single state (monofixation).")
    else:
        logs.append("⚠️ Stimulus too weak to induce global certitude. Increase stim in script.")
    
    print("\n🚀 --- SIMULATION RESULTS ---", flush=True)
    for log in logs:
        print(log, flush=True)
        
    # Export to JSON for foolproof reading
    results = {
        "mean_div_a": mean_div_a,
        "mean_div_b": mean_div_b,
        "improvement": (mean_div_b - mean_div_a) / mean_div_a if mean_div_a > 0 else 0,
        "history_a": history_a,
        "history_b": history_b
    }
    with open("projects/mem4ristor-v2/tests/sim_data.json", "w") as f:
        json.dump(results, f, indent=2)
        
    return history_a, history_b
    
    return history_a, history_b

if __name__ == "__main__":
    run_comparative_sim()
