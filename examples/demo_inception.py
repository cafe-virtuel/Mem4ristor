import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from mem4ristor.mem4ristor_v3 import Mem4ristorV3
from mem4ristor.sensory import SensoryFrontend
from mem4ristor.inception import DreamVisualizer

def generate_circle(size=64):
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    mask = X**2 + Y**2 <= 0.5
    img = np.zeros((size, size))
    img[mask] = 1.0
    return img

def demo_inception():
    print("\n🌌 DEMO: Project Inception (Dream Visualization)")
    print("="*60)
    
    # 1. Initialize System
    N_neurons = 100
    frontend = SensoryFrontend(output_dim=N_neurons, input_shape=(64, 64), seed=42)
    backend = Mem4ristorV3(config={'dynamics': {'dt': 0.1}}, seed=42)
    backend._initialize_params(N=N_neurons)
    
    decoder = DreamVisualizer(frontend)
    print(f"1. System initialized (Mem4ristor N={N_neurons}). Decoding Matrix created.")
    
    # 2. Imprint a Memory (The Circle)
    img_circle = generate_circle()
    stimulus_circle = frontend.perceive(img_circle)
    
    print("2. Imprinting 'Circle' memory (50 steps)...")
    for _ in range(50):
        backend.step(I_stimulus=stimulus_circle)
        
    # Check current "Perception"
    decoded_perception = decoder.decode(backend.v)
    
    # 3. Enter Dream Mode (Sensory Deprivation)
    print("3. Entering Dream Mode (No Stimulus)...")
    dream_frames = []
    
    # Reset internal state partially? No, let the "afterimage" persist and evolve.
    # We want to see the "phantom circle" fade or mutate.
    
    for _ in range(20):
        # Zero input
        backend.step(I_stimulus=0.0)
        
        # Decode Dream
        frame = decoder.decode(backend.v)
        dream_frames.append(frame)
        
    # 4. Visualization
    # We will save a plot showing Original, Perception, and Dream Evolution
    
    fig, axes = plt.subplots(1, 6, figsize=(15, 3))
    
    # Original
    axes[0].imshow(img_circle, cmap='gray')
    axes[0].set_title("Reality (Input)")
    axes[0].axis('off')
    
    # Perception (While looking)
    axes[1].imshow(decoded_perception.reshape(int(np.sqrt(decoded_perception.size)), -1), cmap='plasma')
    axes[1].set_title("Perception")
    axes[1].axis('off')
    
    # Dreams
    dream_indices = [0, 5, 10, 19]
    for i, idx in enumerate(dream_indices):
        d_img = dream_frames[idx]
        size = int(np.sqrt(d_img.size))
        axes[i+2].imshow(d_img.reshape(size, size), cmap='inferno')
        axes[i+2].set_title(f"Dream T+{idx}")
        axes[i+2].axis('off')
        
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), 'inception_result.png')
    plt.savefig(output_path)
    print(f"\n✅ Result saved to: {output_path}")

if __name__ == "__main__":
    demo_inception()
