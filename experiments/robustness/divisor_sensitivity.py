import numpy as np
import matplotlib.pyplot as plt
from mem4ristor.core import Mem4ristorV2
import os

def run_stability_check():
    divisors = np.linspace(2.0, 10.0, 20)
    max_vals = []
    exploded = []
    
    for div in divisors:
        cfg = {
            'dynamics': {'a': 0.7, 'b': 0.8, 'epsilon': 0.08, 'alpha': 0.15, 'v_cubic_divisor': div, 'dt': 0.05},
            'coupling': {'D': 0.15, 'heretic_ratio': 0.15},
            'doubt': {'epsilon_u': 0.02, 'k_u': 1.0, 'sigma_baseline': 0.05, 'u_clamp': [0.0, 1.0], 'tau_u': 1.0},
            'noise': {'sigma_v': 0.05}
        }
        
        model = Mem4ristorV2(config=cfg, seed=42)
        model._initialize_params(N=100)
        
        v_trace = []
        is_stable = True
        for _ in range(1000):
            model.step(I_stimulus=1.0)
            v_max = np.max(np.abs(model.v))
            if v_max > 50: # Explosion threshold
                is_stable = False
                break
            v_trace.append(v_max)
        
        exploded.append(not is_stable)
        max_vals.append(np.mean(v_trace[-10:]) if is_stable else 50)

    plt.figure(figsize=(10, 6))
    plt.plot(divisors, max_vals, 'o-', label="Mean Stable Potential")
    plt.axvspan(3.0, 7.0, color='green', alpha=0.2, label="Safe Zone")
    plt.axvline(5.0, color='red', linestyle='--', label="Chosen Divisor (5.0)")
    plt.xlabel("Cubic Divisor (D in v³/D)")
    plt.ylabel("Mean Amplitude |v|")
    plt.title("System Stability vs Cubic Divisor")
    plt.legend()
    plt.grid(True)
    
    os.makedirs("results/plots", exist_ok=True)
    plt.savefig("results/plots/divisor_sensitivity.png")
    print(f"Plot saved to results/plots/divisor_sensitivity.png")

if __name__ == "__main__":
    run_stability_check()
