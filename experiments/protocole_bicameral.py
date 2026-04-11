import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from mem4ristor.core import Mem4ristorV3, Mem4Network

def sigmoid(x, k=10, center=0.5):
    return 1 / (1 + np.exp(-k * (x - center)))

class BicameralSystem:
    def __init__(self, size=10):
        self.size = size
        self.N = size * size
        
        # CHAMBRE A : LE SAGE (v2.6 Standard)
        # Low noise, conservative
        self.sage = Mem4ristorV2(seed=42)
        self.sage.cfg['noise']['sigma_v'] = 0.02
        self.sage._initialize_params(self.N)
        
        # CHAMBRE B : LE FOU (v3.0 Generative)
        # High noise, high plasticity
        self.fool = Mem4ristorV3(seed=999) # Different seed
        self.fool.cfg['noise']['sigma_v'] = 0.08 # Chaos
        self.fool.cfg['dynamics']['lambda_learn'] = 0.1 # High learning rate
        self.fool._initialize_params(self.N)
        
        # GATING PARAMS
        self.theta_urgency = 0.6  # Threshold for opening the gate
        self.gate_gain = 0.5      # Strength of coupling when open
        
        # Metrics history
        self.history = {
            'entropy_A': [],
            'entropy_B': [],
            'u_mean_A': [],
            'gate_openness': []
        }

    def step(self, t):
        # 1. Update Fool (Autonomous Chaos)
        # The Fool runs wildly, maybe stimulating itself
        self.fool.step(I_stimulus=np.sin(t/10.0)) # Internal wandering
        
        # 2. Update Sage (With Gated Input from Fool)
        
        # Calculate Sage's Confusion (Mean Doubt)
        u_mean_A = np.mean(self.sage.u)
        
        # Calculate Gate Openness (Sigmoid on Doubt)
        # If u_mean_A > theta, Gate opens.
        gate_openness = sigmoid(u_mean_A, k=15, center=self.theta_urgency)
        
        # Coupling Signal: B -> A
        # The Sage receives the Fool's state as an "Intuition" field
        # We couple v_B into I_ext of A.
        # But we only want to inject DIGESTED info. Let's inject raw v difference.
        signal_B_to_A = (self.fool.v - self.sage.v)
        
        input_from_B = self.gate_gain * gate_openness * signal_B_to_A
        
        # Sage Step (Stimulus = 0 + Input from Fool)
        # Note: We assume the Sage is facing a Blank Slate problem (Stim=0) 
        # but gets "ideas" from B.
        self.sage.step(I_stimulus=input_from_B)
        
        # Log
        self.history['u_mean_A'].append(u_mean_A)
        self.history['gate_openness'].append(gate_openness)
        self.history['entropy_A'].append(self.sage.calculate_entropy())
        self.history['entropy_B'].append(self.fool.calculate_entropy())

    def run(self, steps=1000):
        print(f"=== BICAMERAL RUN (N={self.N}) ===")
        print("Scénario : Le Sage (A) tente de rester calme, le Fou (B) délire.")
        print("Si le Sage doute (u > 0.6), il écoute le Fou.")
        
        for t in range(steps):
            # Inject artificial doubt into A at t=300 to simulate a "Crisis"
            if 300 <= t < 350:
                 self.sage.u += 0.02 # Force doubt rising
                 
            self.step(t)
            
            if t % 100 == 0:
                print(f"Step {t}: Gate={self.history['gate_openness'][-1]:.2f}, H_A={self.history['entropy_A'][-1]:.2f}")

        # Plot
        self.plot_results()

    def plot_results(self):
        steps = range(len(self.history['entropy_A']))
        
        fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        
        # Ax1: Entropy
        axes[0].plot(steps, self.history['entropy_A'], label='Sage (A) Entropy', color='blue')
        axes[0].plot(steps, self.history['entropy_B'], label='Fool (B) Entropy', color='orange', alpha=0.6)
        axes[0].set_ylabel('Entropy (H)')
        axes[0].legend()
        axes[0].set_title('Cognitive Diversity: Sage vs Fool')
        
        # Ax2: Doubt (Trigger)
        axes[1].plot(steps, self.history['u_mean_A'], label='Sage Mean Doubt (u)', color='green')
        axes[1].axhline(y=self.theta_urgency, color='red', linestyle='--', label='Urgency Threshold')
        axes[1].set_ylabel('Doubt (u)')
        axes[1].legend()
        axes[1].set_title('Sage Confusion Level')
        
        # Ax3: Valve
        axes[2].fill_between(steps, self.history['gate_openness'], color='purple', alpha=0.3)
        axes[2].plot(steps, self.history['gate_openness'], color='purple', label='Gate Openness')
        axes[2].set_ylabel('Interaction (0-1)')
        axes[2].set_title('Uncertainty Valve (B -> A Flow)')
        
        plt.tight_layout()
        plt.savefig('bicameral_trace.png')
        print("Graphique sauvegardé : bicameral_trace.png")

if __name__ == "__main__":
    sim = BicameralSystem()
    sim.run()
