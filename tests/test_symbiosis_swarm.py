import pytest
import numpy as np
import sys
import os

# Force insert at beginning to override installed packages
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mem4ristor.core import Mem4ristorV2
from mem4ristor.symbiosis import SymbioticSwarm

class TestSwarmPhase4:

    def test_swarm_synchronization(self):
        """Verify that two chips share immunity (w) via diffusion."""
        # Agent A: The Veteran (Has scars)
        agent_a = Mem4ristorV2()
        agent_a.w[:] = 1.0 # High resistance
        
        # Agent B: The Rookie (Clean slate)
        agent_b = Mem4ristorV2()
        agent_b.w[:] = 0.0 # Low resistance
        
        swarm = SymbioticSwarm([agent_a, agent_b], coupling_strength=0.5)
        
        # Run synchronization step
        # dt=0.05 (default). Strength=0.5.
        # dw_B = 0.5 * (Mean - w_B) * dt
        # Mean = 0.5. w_B starts at 0.
        # dw_B = 0.5 * 0.5 * 0.05 = 0.0125
        # w_B should increase.
        
        initial_w_b = agent_b.w.copy()
        swarm.synchronize_scars()
        
        # Check B increased
        assert np.all(agent_b.w > initial_w_b), "Rookie should inherit immunity from Veteran"
        
        # Check A decreased (diffusion works both ways unless directed)
        # Mean is 0.5. A is 1.0. A should go down.
        assert np.all(agent_a.w < 1.0), "Veteran should share load (decrease)"

if __name__ == "__main__":
    t = TestSwarmPhase4()
    t.test_swarm_synchronization()
    print("Swarm Phase 4 Tests Passed!")
