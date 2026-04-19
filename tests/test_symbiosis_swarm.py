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
        """Verify asymmetric MAX-FIELD diffusion of immunity (w) across the swarm.

        synchronize_scars() implements an "Accumulator of Wisdom/Fear" semantic:
        the swarm field is the per-index MAX of all agents' w (not the mean), and
        each agent only learns *upward* (diff > 0). Veterans keep their scars,
        rookies inherit them. See symbiosis.py:128-148 for the design rationale.
        """
        agent_a = Mem4ristorV2()
        agent_a.w[:] = 1.0  # Veteran with scars

        agent_b = Mem4ristorV2()
        agent_b.w[:] = 0.0  # Rookie, clean slate

        swarm = SymbioticSwarm([agent_a, agent_b], coupling_strength=0.5)

        initial_w_a = agent_a.w.copy()
        initial_w_b = agent_b.w.copy()
        swarm.synchronize_scars()

        # Rookie inherits immunity (upward diffusion)
        assert np.all(agent_b.w > initial_w_b), "Rookie should inherit immunity from Veteran"

        # Veteran is preserved (asymmetric — no downward leak)
        assert np.allclose(agent_a.w, initial_w_a), "Veteran scars must be preserved (max-field semantic)"

        # Expected analytic value for B: gamma * (max - w_B) * dt = 0.5 * 1.0 * 0.05 = 0.025
        expected_dw_b = 0.5 * 1.0 * agent_b.dt
        assert np.allclose(agent_b.w - initial_w_b, expected_dw_b, atol=1e-9), \
            f"Rookie update should be {expected_dw_b}, got {agent_b.w[0] - initial_w_b[0]}"

if __name__ == "__main__":
    t = TestSwarmPhase4()
    t.test_swarm_synchronization()
    print("Swarm Phase 4 Tests Passed!")
