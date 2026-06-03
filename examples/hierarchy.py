import numpy as np
from typing import Dict, Any, List
from .core import Mem4ristorV3

# Import King from experimental/ (not a production module)
import os, sys
_experimental_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'experimental')
sys.path.insert(0, _experimental_dir)
from mem4ristor_king import Mem4ristorKing

class HierarchicalChimera:
    """
    Phase 5: The Deep Chimera (Hierarchical Architecture).
    
    WARNING: EXPERIMENTAL - Depends on Mem4ristorKing (experimental/).
    
    Orchestrates multiple Mem4ristor modules in a V1 -> V4 -> PFC stack.
    
    Structure:
    1. V1 (Primary Visual Cortex):
       - Fast dynamics (High dt/epsilon).
       - High resolution (N=64 or more).
       - Driven by external stimuli.
       
    2. V4 (Associative Cortex):
       - Medium dynamics.
       - Aggregates V1 state.
       - Forms "Concepts".
       
    3. PFC (Prefrontal Cortex / King):
       - Slow dynamics (Low dt/epsilon).
       - Low resolution (N=36).
       - Driven by V4 state + Internal Frustration.
       - Can trigger 'Martial Law' (Top-down inhibition).
    """
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        
        # --- Layer 1: V1 (Reflexes) ---
        # Fast reactor
        v1_config = {
            'dynamics': {'dt': 0.2, 'epsilon': 0.2}, # Fast
            'coupling': {'D': 0.1}
        }
        self.V1 = Mem4ristorV3(config=v1_config, seed=seed)
        self.V1._initialize_params(N=64)
        
        # --- Layer 2: V4 (Association) ---
        # Feature Integrator
        v4_config = {
            'dynamics': {'dt': 0.1, 'epsilon': 0.08}, # Normal
            'coupling': {'D': 0.15}
        }
        self.V4 = Mem4ristorV3(config=v4_config, seed=seed+1)
        self.V4._initialize_params(N=49) # 7x7 grid
        
        # --- Layer 3: PFC (Executive) ---
        # The Philosopher King
        pfc_config = {
            'dynamics': {'dt': 0.05, 'epsilon': 0.005}, # Extremely Slow / Sage
            'coupling': {'D': 0.2, 'heretic_ratio': 0.2} # Highly political
        }
        self.PFC = Mem4ristorKing(config=pfc_config, seed=seed+2)
        self.PFC._initialize_params(N=36) # 6x6 grid
        
        # Connectivity Weights (Fixed random projections for now)
        # V1 (64) -> V4 (49)
        self.W_1to4 = self.rng.normal(0, 0.1, (64, 49))
        
        # V4 (49) -> PFC (36)
        self.W_4toPFC = self.rng.normal(0, 0.1, (49, 36))
        
        # Top-Down Inhibition (PFC -> V1)
        # If PFC is angry, it suppress noise in V1
        self.W_PFCto1 = -np.abs(self.rng.normal(0, 0.1, (36, 64))) # Negative weights (Inhibitory)

    def step(self, stimulus_v1: Any) -> Dict[str, Any]:
        """
        Forward Pass through the Hierarchy.
        
        Args:
            stimulus_v1: Input for V1 (e.g., from SensoryFrontend)
            
        Returns:
            Dict containing state of all layers and PFC decision.
        """
        # 1. Processing V1 (Bottom-Up)
        self.V1.step(I_stimulus=stimulus_v1)
        s1 = self.V1.v # State V1
        
        # 2. Transmission V1 -> V4
        # Input to V4 is the projected state of V1
        # We squash it with tanh to keep it in range [-1, 1]
        input_v4 = np.tanh(s1 @ self.W_1to4) * 1.0 # Unity Gain
        
        self.V4.step(I_stimulus=input_v4)
        s4 = self.V4.v # State V4
        
        # 3. Transmission V4 -> PFC
        input_pfc = np.tanh(s4 @ self.W_4toPFC) * 1.0
        
        # PFC runs with Governance (Check frustration, boredom, etc.)
        pfc_status = self.PFC.step(I_stimulus=input_pfc, target_vector=None)
        
        # 4. Top-Down Feedback (The "Shut Up" Signal)
        if pfc_status['martial_law']:
            # PFC inhibits V1 to stop the noise
            inhibition = np.abs(self.PFC.v @ self.W_PFCto1) * 0.5
            self.V1.v *= (1.0 - np.tanh(inhibition)) # Dampening factor
            
        return {
            'V1_mean': np.mean(self.V1.v),
            'V4_mean': np.mean(self.V4.v),
            'PFC_mean': np.mean(self.PFC.v),
            'PFC_status': pfc_status,
            'V1_entropy': self.V1.calculate_entropy(),
            'PFC_entropy': self.PFC.calculate_entropy()
        }
