import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from mem4ristor.mem4ristor_v3 import Mem4ristorV3
from mem4ristor.sensory import SensoryFrontend

def demo_visual_perception():
    print("\n👁️ DEMO: Visual Perception (The Eye of the Machine)")
    print("="*60)
    
    # 1. Initialize System
    N_neurons = 50
    frontend = SensoryFrontend(output_dim=N_neurons, input_shape=(64, 64))
    backend = Mem4ristorV3(config={'dynamics': {'dt': 0.1}}, seed=42)
    backend._initialize_params(N=N_neurons) # Explicitly set N
    
    print(f"1. System initialized (Mem4ristor N={N_neurons}).")
    
    # 2. Generate Test Images
    img_circle = frontend.generate_test_pattern("circle")
    img_square = frontend.generate_test_pattern("square")
    
    # 3. Perceive (Transduction)
    stimulus_circle = frontend.perceive(img_circle)
    stimulus_square = frontend.perceive(img_square)
    
    print(f"2. Images processed. Stimulus Range: [{stimulus_circle.min():.2f}, {stimulus_circle.max():.2f}]")
    
    # 4. Reaction (Mem4ristor Response)
    print("\n3. Injecting visual stimuli into Mem4ristor...")
    
    # Response to Circle
    backend.v[:] = 0 # Reset
    backend.step(I_stimulus=stimulus_circle)
    response_circle = backend.v.copy()
    
    # Response to Square
    backend.v[:] = 0 # Reset
    backend.step(I_stimulus=stimulus_square)
    response_square = backend.v.copy()
    
    # 5. Measure Difference (Cosines)
    dot_product = np.dot(response_circle, response_square)
    norm_c = np.linalg.norm(response_circle)
    norm_s = np.linalg.norm(response_square)
    cosine_sim = dot_product / (norm_c * norm_s)
    
    print(f"   Circle Response (Mean V): {np.mean(response_circle):.4f}")
    print(f"   Square Response (Mean V): {np.mean(response_square):.4f}")
    print(f"   Cosine Similarity: {cosine_sim:.4f}")
    
    # 6. Verdict
    print("\n4. Verdict:")
    if cosine_sim < 0.95:
        print(f"✅ SUCCESS: The Mem4ristor discriminates between Shapes.")
        print(f"   Similarity is low ({cosine_sim:.2f}), meaning different neural states.")
    else:
        print(f"❌ FAILURE: The system is blind (Responses are identical).")

if __name__ == "__main__":
    demo_visual_perception()
