import numpy as np
from typing import List, Optional, Tuple
from .core import Mem4ristorV2

class CreativeProjector:
    """
    Phase 4: 'L'Apnée de l'Innovation' (Creative Mutation).
    
    Transforms the 'scars' of the Mem4ristor (high resistance w) into 
    creative biases for the digital Cortex.
    
    Mechanism:
    - Innate Projection Map: A fixed random matrix mapping physical neurons (NxN) to semantic classes (K).
    - Creative Bias: When panic is high, 'w' is projected onto the class space to push AWAY from known trauma.
    """
    def __init__(self, mem4ristor_instance: Mem4ristorV2, num_classes: int = 10, seed: int = 42):
        self.mem = mem4ristor_instance
        self.num_classes = num_classes
        self.rng = np.random.RandomState(seed)
        
        # 1. Innate Projection Map (Fixed at birth)
        # Maps each of the N physical neurons to a vector in the K-dimensional class space.
        # Shape: (N, num_classes)
        self.projection_map = self.rng.normal(0, 1, (self.mem.N, self.num_classes))
        # Normalize to unit vectors
        norms = np.linalg.norm(self.projection_map, axis=1, keepdims=True)
        self.projection_map /= (norms + 1e-9)

    def get_creative_bias(self, panic_level: float = 0.0) -> np.ndarray:
        """
        Calculate the 'Creative Rebound' vector.
        
        Equation: B_creative = - eta * sum( tanh(w_i) * P_i )
        
        - w_i: Resistance of neuron i (How much it fought/knows this pattern).
        - P_i: Projection vector of neuron i.
        - eta: Scaling factor (Panic).
        
        Returns:
            bias_vector (ndarray): Shape (num_classes,). Add this to Cortex logits.
        """
        if panic_level < 0.01:
            return np.zeros(self.num_classes)
            
        # 1. Extract Scars (Resistance w) representing "Trauma/Experience"
        # We use tanh(w) to squash anomalies and focus on the "saturation" of resistance.
        w_saturated = np.tanh(self.mem.w)
        
        # 2. Project Scars into Semantic Space
        # "Where does it hurt?" -> "Which classes are we resistant to?"
        # expertise_vector[k] = sum_i (w_i * P_ik)
        expertise_vector = w_saturated @ self.projection_map
        
        # 3. Creative Rebound
        # We want to go AWAY from the trauma.
        # If we have high resistance to "Class 0", we push negative bias to Class 0.
        # This forces the Softmax to explore other classes (neighbors).
        creative_force = -1.0 * panic_level * 5.0 * expertise_vector
        
        # 4. Chaos Injection (The Spark)
        # A small random noise scaled by panic to break local minima
        chaos = self.rng.normal(0, panic_level * 0.5, self.num_classes)
        
        return creative_force + chaos

    def dream_cycle(self, steps: int = 100) -> np.ndarray:
        """
        Phase 4: 'The Dreamer' (Night Mode).
        Generates a sequence of Hallucinations based purely on internal w structure.
        
        Returns:
            dream_log (ndarray): Sequence of projected class vectors (steps, num_classes).
        """
        dream_log = []
        
        # Save current state to restore after dream? 
        # No, dreaming changes the state (v), but we shouldn't drift too far?
        # Let's assume we start from current state.
        
        for _ in range(steps):
            # 1. Run with ZERO external stimulus (Sensory Deprivation)
            # The dynamics are driven ONLY by w (Memory) and internal noise.
            self.mem.step(I_stimulus=0.0)
            
            # 2. Capture the "Dream State" (Volatile potential v)
            v_state = self.mem.v
            
            # 3. Interpret the Dream
            # Project v onto the semantic map
            # This asks: "What does this internal electrical storm look like to the Cortex?"
            dream_content = v_state @ self.projection_map
            dream_log.append(dream_content)
            
        return np.array(dream_log)

class SymbioticSwarm:
    """
    Phase 4: 'La Toile de Jade' (Swarm Telepathy).
    
    Connects multiple Mem4ristor chips via analog diffusive coupling of their 'w' (Immunity).
    """
    def __init__(self, agents: List[Mem4ristorV2], coupling_strength: float = 0.1):
        self.agents = agents
        self.coupling_strength = coupling_strength
        # Verification: All agents must have same size
        N0 = agents[0].N
        for a in agents:
            if a.N != N0:
                raise ValueError("All swarm agents must have the same size N.")

    def synchronize_scars(self):
        """
        Telepathy Step: Diffusive coupling of 'w'.
        
        dw_i/dt += gamma * (w_mean - w_i)
        
        This aligns the 'Immunity Landscapes' of all chips.
        If Chip A has a scar (high w) at index 5, Chip B will start feeling a pull
        to increase w at index 5, effectively 'learning' the trauma of A.
        """
        if len(self.agents) < 2:
            return

        # 1. Calculate Field (The Collective Unconscious)
        # Stack w vectors: (Num_Agents, N)
        w_stack = np.stack([a.w for a in self.agents])
        
        # MEAN FIELD (Consensus) -> Dilutes trauma
        # w_target = np.mean(w_stack, axis=0) 
        
        # MAX FIELD (Empathy/Alert) -> Propagates trauma
        # If ANY agent has a scar, the swarm learns it.
        # "Better safe than sorry" evolutionary strategy.
        w_target = np.max(w_stack, axis=0)
        
        # 2. Apply Pull
        for i, agent in enumerate(self.agents):
            # Difference from target
            diff = w_target - agent.w
            
            # Apply update (Euler step)
            # Only pull UPWARDS (Learning), let natural decay handle the downward drift
            # This makes the swarm an "Accumulator of Wisdom/Fear"
            
            # Filter: only learn if target > current (asymmetric coupling)
            learn_mask = (diff > 0).astype(float)
            
            agent.w += self.coupling_strength * diff * learn_mask * agent.dt
