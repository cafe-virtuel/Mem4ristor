import numpy as np
from typing import Tuple
from .sensory import SensoryFrontend

class DreamVisualizer:
    """
    Project Inception: The Dream Decoder.
    
    Reconstructs visual hallucinations from the abstract neural state ($v$) of the Mem4ristor.
    It inverts the sensory projection process using the Moore-Penrose pseudo-inverse.
    
    Equation:
    Image_dream = v @ P_pseudo_inverse
    """
    def __init__(self, sensory_frontend: SensoryFrontend):
        self.frontend = sensory_frontend
        
        # 1. Compute Pseudo-Inverse of the Projection Matrix
        # P shape: (Feature_Dim, N_neurons)
        # P_pinv shape: (N_neurons, Feature_Dim)
        self.P_pinv = np.linalg.pinv(self.frontend.projection)
        
    def decode(self, neural_state: np.ndarray) -> np.ndarray:
        """
        Decodes a neural state vector ($v$) into a 2D image.
        
        Strategy: Back-project neurons to feature space via pseudo-inverse,
        then reshape to a 2D grid for visualization. This won't produce
        pixel-accurate reconstructions (inverting conv/pooling is ill-posed),
        but gives a "Deep Dream" style representation of the internal state.
        
        Args:
            neural_state (np.ndarray): The v vector from Mem4ristor, shape (N,).
            
        Returns:
            np.ndarray: 2D image normalized to [0, 1], or 1D feature vector
                        if reshape is not possible.
        """
        # 1. Back-project: Neurons (N) -> Features (F)
        features = neural_state @ self.P_pinv
        
        # 2. Reshape to a square 2D image for visualization
        size = int(np.sqrt(features.shape[0]))
        
        if size * size == features.shape[0]:
            # Perfect square — reshape directly
            image = features.reshape((size, size))
        else:
            # Not a perfect square (e.g. num_filters > 1) — pad to nearest square
            side = int(np.ceil(np.sqrt(features.shape[0])))
            padding = side * side - features.shape[0]
            padded = np.pad(features, (0, padding), mode='constant')
            image = padded.reshape((side, side))
            
        # 3. Normalize for Visualization [0, 1]
        val_range = np.max(image) - np.min(image)
        if val_range > 1e-6:
            image = (image - np.min(image)) / val_range
             
        return image
