"""
Mem4ristor V3 Core Facade.

This file serves as the main entry point to maintain backward compatibility
with previous scripts while routing logic to the new modular architecture
(`dynamics.py`, `topology.py`, `metrics.py`).

KIMI Refactoring fixes have been fully applied:
- O(1) rewiring speedup
- Strict tau_u and delta synchronization with the scientific prepint
- Continuous Differential Entropy
"""

from mem4ristor.dynamics import Mem4ristorV3, Mem4ristorV2
from mem4ristor.topology import Mem4Network
from mem4ristor.metrics import get_cognitive_states, calculate_cognitive_entropy, calculate_continuous_entropy

__all__ = [
    "Mem4ristorV3", 
    "Mem4ristorV2", 
    "Mem4Network",
    "get_cognitive_states",
    "calculate_cognitive_entropy",
    "calculate_continuous_entropy"
]
