import numpy as np
import os
import matplotlib.pyplot as plt
import sys
# Add current dir to path
sys.path.append(os.path.dirname(__file__))

from reference_impl import Mem4ristorV2

def run_benchmark(name, heretic_ratio=0.15, use_doubt=True):
    print(f"--- Running Benchmark: {name} ---")
    model = Mem4ristorV2()
    
    # Apply ablation overrides
    model.heretic_mask = np.random.rand(model.N) < heretic_ratio
    
    # Benchmark settings (Aligned with Grok Hardware Sim, scaled for dt=0.01)
    total_steps = model.cfg['benchmark']['steps']
    bias_start = model.cfg['benchmark']['bias_step_start']
    bias_end = model.cfg['benchmark']['bias_step_end']
    
    history = {
        'entropy': [],
        'diversity': [],
        'mean_u': []
    }
    
    for t in range(total_steps):
        # Stimulus schedule (scaled by 10 relative to Grok sim)
        if t < bias_start:
            stim = 0.4 # Baseline
        elif t < bias_end:
            stim = 1.1 # Bias phase
        else:
            stim = 0.0 # Relaxation
        
        # Override doubt if requested
        if not use_doubt:
            model.u[:] = 0
            
        model.step(stim)
        
        # Override doubt again if it's an ablation (doubt off)
        if not use_doubt:
            model.u[:] = 0

        # Metrics
        entropy = model.calculate_entropy()
        states = model.get_states()
        diversity = len(np.unique(states))
        
        history['entropy'].append(entropy)
        history['diversity'].append(diversity)
        history['mean_u'].append(np.mean(model.u))

    # Calculate average diversity during bias phase (aligned with CONFIG_DEFAULT)
    mean_div_bias = np.mean(history['diversity'][bias_start:bias_end])

    
    print(f"Results for {name}:")
    print(f"  Mean Diversity (Bias Phase): {mean_div_bias:.2f}")
    print(f"  Final Entropy: {history['entropy'][-1]:.4f}")
    
    return history, mean_div_bias

def plot_results(all_histories):
    plt.figure(figsize=(12, 8))
    
    for name, hist in all_histories.items():
        plt.plot(hist['entropy'], label=f"{name} (Entropy)")
    
    plt.axvline(x=500, color='r', linestyle='--', label='Bias Phase Start')
    plt.title("Cognitive Resilience Ablation Study (Mem4ristor v2.0)")
    plt.xlabel("Time Steps")
    plt.ylabel("Shannon Entropy (H)")
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join(os.path.dirname(__file__), "results/reproduction_results.png")
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    results = {}
    
    # 1. Full Model (v2.0)
    results['Full Model (v2.0)'], _ = run_benchmark("Full Model (v2.0)")
    
    # 2. No Doubt
    results['No Doubt'], _ = run_benchmark("No Doubt", use_doubt=False)
    
    # 3. No Heretics
    results['No Heretics'], _ = run_benchmark("No Heretics", heretic_ratio=0.0)
    
    # 4. None (Classical)
    results['Classical (u=0, h=0)'], _ = run_benchmark("Classical", heretic_ratio=0.0, use_doubt=False)
    
    # Final Summary for Table 3
    print("\n" + "="*40)
    print("FINAL ABLATION TABLE DATA:")
    print("="*40)
    # Note: These values are generated in real-time and should correspond to the paper's claims.
    
    # Plotting (requires matplotlib)
    try:
        plot_results(results)
    except Exception as e:
        print(f"Skipping plot due to: {e}")
