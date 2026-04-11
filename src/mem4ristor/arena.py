import numpy as np
from typing import Tuple, Dict, Any, List
from .core import Mem4ristorV3

class GladiatorMem4ristor(Mem4ristorV3):
    """
    Subclass for fighters in the Arena.
    Adds 'Pain Learning'.
    """
    def punish(self, pain_signal: float, role: str = 'Predator'):
        """
        Apply 'Voltage Shock'.
        Strategy depends on Role.
        """
        if pain_signal > 0.01:
            pain = min(pain_signal, 1.0)
            
            if role == 'Predator':
                # PREDATOR STRATEGY: ASSIMILATION
                # Faster assimilation (0.2)
                self.w *= (1.0 - 0.2 * pain)
                # Clip minimal resistance
                self.w = np.clip(self.w, 0.01, 10.0)
                
            elif role == 'Prey':
                # PREY STRATEGY: EVASION
                # Slower mutation (0.1) to give Predator a chance
                mutation = np.random.normal(0, pain * 0.1, size=self.N) 
                self.w += mutation
                # Clip
                self.w = np.clip(self.w, 0.1, 10.0)

class Arena:
    """
    The Colosseum. Manage the fight between Predator and Prey.
    """
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        
        # Initialize Gladiators
        self.predator = GladiatorMem4ristor(config={'dynamics': {'dt': 0.1}}, seed=seed)
        self.predator._initialize_params(N=50)
        
        # Prey needs to be erratic
        self.prey = GladiatorMem4ristor(config={'dynamics': {'dt': 0.1, 'epsilon': 0.2}}, seed=seed+1)
        self.prey._initialize_params(N=50)
        
        self.history = {
            'predator_wins': 0,
            'prey_wins': 0,
            'prediction_error': []
        }

    def fight_round(self) -> Dict[str, Any]:
        """
        One round of combat.
        """
        # 1. Prey Move
        # Prey follows a predictable SINE WAVE (Training Wheels)
        if not hasattr(self, 'time_step'): self.time_step = 0
        self.time_step += 0.1
        
        phases = np.linspace(0, 2*np.pi, 50)
        signal = np.sin(self.time_step + phases)
        
        self.prey.step(I_stimulus=signal)
        v_prey = self.prey.v.copy()
        
        # 2. Predator Predicts
        # Predator tries to MATCH current state
        self.predator.step(I_stimulus=v_prey)
        v_pred = self.predator.v.copy()
        
        # 3. Judgment
        error = np.mean((v_prey - v_pred)**2)
        threshold = 2.0 
        
        predator_win = False
        
        if error < threshold:
            # Predator Caught Prey!
            self.history['predator_wins'] += 1
            predator_win = True
            
            # Punish Prey (Evasion needed)
            self.prey.punish(pain_signal=1.0, role='Prey') 
            
        else:
            # Prey Escaped!
            self.history['prey_wins'] += 1
            
            # Punish Predator (Assimilation needed)
            self.predator.punish(pain_signal=error, role='Predator')
            
        self.history['prediction_error'].append(error)
        
        return {
            'winner': 'Predator' if predator_win else 'Prey',
            'error': error,
            'prey_pain': 1.0 if predator_win else 0.0,
            'predator_pain': error if not predator_win else 0.0
        }
