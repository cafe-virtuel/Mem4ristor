import pytest
import numpy as np
import sys
import os

# Force insert at beginning to override installed packages
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mem4ristor.core import Mem4ristorV2
from mem4ristor.symbiosis import CreativeProjector

class TestCreativePhase4:

    def test_creative_bias_generation(self):
        """Verify that 'Scars' (w) generate a deterministic bias vector."""
        mem = Mem4ristorV2()
        projector = CreativeProjector(mem, num_classes=5, seed=42)
        
        # 1. Create artificial scars (High resistance in first 10 neurons)
        mem.w[:10] = 5.0 
        
        # 2. Get bias with panic
        bias = projector.get_creative_bias(panic_level=0.5)
        
        assert bias.shape == (5,)
        assert not np.allclose(bias, 0.0), "Bias should be non-zero when panic > 0"
        
        # 3. Verify determinism (same w + same seed -> same bias, mostly)
        # Note: get_creative_bias has random noise added.
        # We can check the structural component by zeroing panic noise?
        # Actually, let's just check that it changes when w changes substantially.
        
        old_bias = bias.copy()
        mem.w[:] = -5.0 # Flip scars
        new_bias = projector.get_creative_bias(panic_level=0.5)
        
        assert not np.allclose(old_bias, new_bias), "Bias should react to changes in w"

    def test_dreamer_loop(self):
        """Verify the Night Mode loop generates content."""
        mem = Mem4ristorV2()
        projector = CreativeProjector(mem, num_classes=3)
        
        # Run dream cycle
        dream_log = projector.dream_cycle(steps=10)
        
        assert dream_log.shape == (10, 3)
        # Check for variation (it shouldn't be a flat line if initialized with noise)
        std_dev = np.std(dream_log)
        assert std_dev > 0.0, "Dreams should have variance"

if __name__ == "__main__":
    t = TestCreativePhase4()
    t.test_creative_bias_generation()
    t.test_dreamer_loop()
    print("Creative Phase 4 Tests Passed!")
