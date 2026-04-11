import numpy as np

class BenchmarkModel:
    def __init__(self, N, seed=42):
        self.N = N
        self.rng = np.random.RandomState(seed)
        self.v = self.rng.uniform(-1, 1, N)
        
    def get_states(self):
        # Threshold mapping to align with Mem4ristor bins
        states = np.zeros(self.N, dtype=int)
        states[self.v < -0.5] = 1
        states[(self.v >= -0.5) & (self.v <= 0.5)] = 3
        states[self.v > 0.5] = 5
        return states

class KuramotoModel(BenchmarkModel):
    def __init__(self, N, K=1.0, seed=42):
        super().__init__(N, seed=seed)

        self.K = K
        self.theta = self.rng.uniform(0, 2*np.pi, N)
        self.omega = self.rng.normal(0, 0.1, N)
        self.dt = 0.1
        
    def step(self, I_stim=0):
        # I_stim acts as a common drive
        coupling = (self.K / self.N) * np.sum(np.sin(self.theta[:, None] - self.theta), axis=0)
        self.theta += (self.omega + coupling + I_stim) * self.dt
        self.v = np.cos(self.theta)

class VoterModel(BenchmarkModel):
    def __init__(self, N, seed=42):
        super().__init__(N, seed=seed)

        self.v = self.rng.choice([-1.0, 1.0], N)
        
    def step(self, I_stim=0):
        # Stochastic flip towards neighbor
        for i in range(self.N):
            target = self.rng.randint(0, self.N)
            # Stimulus bias
            p_flip = 0.5 + 0.1 * I_stim
            if self.rng.rand() < p_flip:
                self.v[i] = self.v[target]

class ConsensusModel(BenchmarkModel):
    def __init__(self, N, D=0.1, seed=42):
        super().__init__(N, seed=seed)

        self.D = D
        self.dt = 0.1
        
    def step(self, I_stim=0):
        # Linear Averaging
        avg_v = np.mean(self.v)
        self.v += (self.D * (avg_v - self.v) + 0.05 * I_stim) * self.dt
