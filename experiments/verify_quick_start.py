import sys
import os
import numpy as np

# Ensure proper path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from mem4ristor.core import Mem4Network

def verify_quick_start():
    print("Running Quick Start Verification...")
    
    try:
        # Initialize a network (N=100, 15% Heretics)
        net = Mem4Network(size=10, heretic_ratio=0.15, seed=42)

        # Run simulation for 100 steps (shortened for quick check)
        for step in range(100):
            net.step(I_stimulus=0.5)

        # Calculate final entropy (measure of diversity)
        entropy = net.calculate_entropy()
        print(f"Final System Entropy: {entropy:.4f}")
        
        if entropy > 0.0:
            print("✅ Quick Start Verification Passed!")
            return True
        else:
            print("❌ Entropy is zero. verification failed.")
            return False
            
    except Exception as e:
        print(f"❌ Verification crashed: {e}")
        return False

if __name__ == "__main__":
    success = verify_quick_start()
    sys.exit(0 if success else 1)
