import os
import sys
import numpy as np

# Inject src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from mem4ristor.core import Mem4ristorV2 as CanonicalM4

class Mem4ristorV2(CanonicalM4):
    """
    Reference Implementation Wrapper (v2.0.4.1).
    Now inherits from the canonical src/mem4ristor/core.py.
    """
    def __init__(self, config_path=None):
        super().__init__(seed=42)
        if config_path:
            import yaml
            with open(config_path, 'r') as f:
                self.cfg = yaml.safe_load(f)
        
        # Ensure we use the 10x10 lattice by default for reference scripts
        self.L_side = 10
        self.N = 100
        self._initialize_params(self.N)
        self.adj = self._build_lattice(10)

    def _build_lattice(self, size):
        adj = np.zeros((self.N, self.N))
        for i in range(size):
            for j in range(size):
                idx = i * size + j
                neighbors = []
                if i > 0: neighbors.append((i-1) * size + j)
                if i < size - 1: neighbors.append((i+1) * size + j)
                if j > 0: neighbors.append(i * size + (j-1))
                if j < size - 1: neighbors.append(i * size + (j+1))
                for n in neighbors:
                    adj[idx, n] = 1.0 / len(neighbors)
        return adj

    def step(self, I_stimulus=0):
        # Compatibility with old reference_impl API
        super().step(I_stimulus, self.adj)

if __name__ == "__main__":
    model = Mem4ristorV2()
    print("Reference Implementation (Canonized) ready.")
    model.step()
    print(f"Entropy: {model.calculate_entropy():.4f}")
