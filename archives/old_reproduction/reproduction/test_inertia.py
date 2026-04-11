import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Integration avec le dossier src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from mem4ristor.core import Mem4Network

def run_cognitive_inertia_test():
    print("🧠 MEM4RISTOR PHASE 5 TEST : Cognitive Inertia")
    print("----------------------------------------------")
    
    size = 10
    model = Mem4Network(size=size, seed=42)
    
    # On augmente tau_u pour voir l'effet de mémoire
    model.model.cfg['doubt']['tau_u'] = 5.0 
    
    history_u = []
    
    print("[1] Baseline (500 steps)...")
    for _ in range(500):
        model.step(I_stimulus=0.0)
        history_u.append(np.mean(model.model.u))
        
    print("[2] Intense Pressure (Trauma) (1000 steps)...")
    for _ in range(1000):
        model.step(I_stimulus=2.0)
        history_u.append(np.mean(model.model.u))
        
    print("[3] Post-Trauma Silence (Recovery) (1000 steps)...")
    for _ in range(1000):
        model.step(I_stimulus=0.0)
        history_u.append(np.mean(model.model.u))
        
    # Analyse de la cicatrice
    u_initial = history_u[499]
    u_peak = max(history_u)
    u_recovery = history_u[-1]
    
    print(f"\n[INFO] Initial Doubt: {u_initial:.4f}")
    print(f"[INFO] Peak Doubt (Trauma): {u_peak:.4f}")
    print(f"[INFO] Recovery Doubt: {u_recovery:.4f}")
    
    scare_effect = (u_recovery - u_initial) / (u_peak - u_initial)
    print(f"[RESULT] Persistence (Scar Effect): {scare_effect*100:.2f}%")
    
    if scare_effect > 0.2:
        print("✅ [SUCCESS] Cognitive Inertia confirmed. The system 'remembers' the trauma.")
    else:
        print("❌ [FAILURE] Doubt collapsed too fast. Inertia is insufficient.")

    # Visualisation
    try:
        os.makedirs("results", exist_ok=True)
        plt.figure(figsize=(10, 5))
        plt.plot(history_u, label='Mean Constitutional Doubt (u)', color='purple')
        plt.axvline(x=500, color='r', linestyle='--', label='Trauma Start')
        plt.axvline(x=1500, color='g', linestyle='--', label='Recovery Start')
        plt.title(f"Mem4ristor Cognitive Inertia (tau_u = {model.model.cfg['doubt']['tau_u']})")
        plt.xlabel("Simulation Steps")
        plt.ylabel("Doubt Level (u)")
        plt.legend()
        plt.grid(True)
        plt.savefig("results/cognitive_inertia_test.png")
        print("[INFO] Plot saved to results/cognitive_inertia_test.png")
    except Exception as e:
        print(f"[NOTE] Visualization skipped: {e}")

if __name__ == "__main__":
    run_cognitive_inertia_test()
