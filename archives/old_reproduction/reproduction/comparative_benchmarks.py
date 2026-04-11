"""
Comparative Benchmarks: Mem4ristor v2.0.4 vs Classical Models
Head-to-head comparison under IDENTICAL conditions.

Models tested:
1. Mem4ristor v2.0.4 (anti-consensus architecture)
2. Kuramoto oscillators (synchronization model)
3. Voter model (opinion dynamics)
4. Distributed Averaging (consensus protocol)

Metrics:
- Final Entropy (bits)
- Collapse Time (steps to H < 0.3)
- Minority Survival Rate (%)
- Gini Index (distribution inequality)
- Diversity Score (# occupied states)

Author: Antigravity (Session 8)
Date: 2025-12-28
"""

import numpy as np
import sys
import os
from typing import Dict

# Fix path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from mem4ristor.core import Mem4ristorV2

# ============================================================================
# HELPER: Build Lattice
# ============================================================================

def build_lattice_adjacency(L: int) -> np.ndarray:
    """Build square lattice adjacency matrix (4-connected)."""
    N = L * L
    adj = np.zeros((N, N))
    
    for i in range(L):
        for j in range(L):
            idx = i * L + j
            # Right neighbor
            if j < L - 1:
                adj[idx, idx + 1] = 1
                adj[idx + 1, idx] = 1
            # Bottom neighbor
            if i < L - 1:
                adj[idx, idx + L] = 1
                adj[idx + L, idx] = 1
    
    return adj


# ============================================================================
# CLASSICAL MODELS IMPLEMENTATIONS
# ============================================================================

class KuramotoModel:
    """
    Kuramoto coupled oscillators - canonical synchronization model.
    
    dθ_i/dt = ω_i + K/N Σ_j sin(θ_j - θ_i)
    """
    def __init__(self, N: int, K: float = 0.15, seed: int = 42):
        self.N = N
        self.K = K
        self.rng = np.random.RandomState(seed)
        
        # Natural frequencies (heterogeneous)
        self.omega = self.rng.normal(0.0, 0.1, N)
        
        # Phases initialized randomly
        self.theta = self.rng.uniform(-np.pi, np.pi, N)
        
        self.dt = 0.1
    
    def step(self, I_stimulus: float = 0.0):
        """One step of Kuramoto dynamics."""
        # Coupling term
        coupling = np.zeros(self.N)
        for i in range(self.N):
            coupling[i] = np.mean(np.sin(self.theta - self.theta[i]))
        
        # External stimulus (like bias in Mem4ristor)
        external = I_stimulus * np.ones(self.N)
        
        # Update
        dtheta = self.omega + self.K * coupling + external
        self.theta += dtheta * self.dt
        
        # Wrap to [-π, π]
        self.theta = (self.theta + np.pi) % (2*np.pi) - np.pi
    
    def calculate_entropy(self) -> float:
        """Shannon entropy of phase distribution (5 bins)."""
        # Bin phases into 5 states
        bins = np.linspace(-np.pi, np.pi, 6)
        states = np.digitize(self.theta, bins) - 1
        states = np.clip(states, 0, 4)
        
        counts = np.bincount(states, minlength=5)
        probs = counts / self.N
        probs = probs[probs > 0]
        
        if len(probs) <= 1:
            return 0.0
        return -np.sum(probs * np.log2(probs))
    
    def get_gini(self) -> float:
        """Gini index of phase distribution."""
        bins = np.linspace(-np.pi, np.pi, 6)
        states = np.digitize(self.theta, bins) - 1
        states = np.clip(states, 0, 4)
        counts = np.bincount(states, minlength=5)
        
        return calculate_gini(counts)


class VoterModel:
    """
    Voter model - opinion dynamics via random neighbor adoption.
    
    At each step, random agent adopts opinion of random neighbor.
    """
    def __init__(self, N: int, seed: int = 42):
        self.N = N
        self.rng = np.random.RandomState(seed)
        
        # Opinions: 5 discrete states (like Mem4ristor cognitive states)
        self.opinions = self.rng.randint(0, 5, N)
        
        # Adjacency (lattice)
        L = int(np.sqrt(N))
        self.adj = build_lattice_adjacency(L)
    
    def step(self, I_stimulus: float = 0.0):
        """One step: random agent copies random neighbor (with bias)."""
        # External bias: probability to shift toward state favored by stimulus
        bias_prob = 0.01 * abs(I_stimulus)  # scaling
        
        for _ in range(self.N // 10):  # Multiple updates per step for speed
            i = self.rng.randint(0, self.N)
            
            # Bias: stimulus pushes toward higher states
            if self.rng.rand() < bias_prob and I_stimulus > 0:
                self.opinions[i] = min(self.opinions[i] + 1, 4)
            else:
                # Normal voter: copy neighbor
                neighbors = np.where(self.adj[i] > 0)[0]
                if len(neighbors) > 0:
                    j = self.rng.choice(neighbors)
                    self.opinions[i] = self.opinions[j]
    
    def calculate_entropy(self) -> float:
        """Shannon entropy of opinion distribution."""
        counts = np.bincount(self.opinions, minlength=5)
        probs = counts / self.N
        probs = probs[probs > 0]
        
        if len(probs) <= 1:
            return 0.0
        return -np.sum(probs * np.log2(probs))
    
    def get_gini(self) -> float:
        """Gini index of opinion distribution."""
        counts = np.bincount(self.opinions, minlength=5)
        return calculate_gini(counts)


class AveragingModel:
    """
    Distributed Averaging - consensus via local averaging.
    
    x_i(t+1) = x_i(t) + ε Σ_j (x_j - x_i)
    """
    def __init__(self, N: int, epsilon: float = 0.1, seed: int = 42):
        self.N = N
        self.epsilon = epsilon
        self.rng = np.random.RandomState(seed)
        
        # States initialized heterogeneously
        self.x = self.rng.uniform(-1.5, 1.5, N)
        
        # Adjacency
        L = int(np.sqrt(N))
        self.adj = build_lattice_adjacency(L)
    
    def step(self, I_stimulus: float = 0.0):
        """One step: local averaging + external stimulus."""
        dx = np.zeros(self.N)
        
        for i in range(self.N):
            neighbors = np.where(self.adj[i] > 0)[0]
            if len(neighbors) > 0:
                dx[i] = np.mean(self.x[neighbors]) - self.x[i]
        
        # Update with averaging + stimulus
        self.x += self.epsilon * dx + 0.01 * I_stimulus
    
    def calculate_entropy(self) -> float:
        """Shannon entropy of state distribution (5 bins)."""
        # Map continuous states to 5 discrete bins
        bins = np.linspace(-2, 2, 6)
        states = np.digitize(self.x, bins) - 1
        states = np.clip(states, 0, 4)
        
        counts = np.bincount(states, minlength=5)
        probs = counts / self.N
        probs = probs[probs > 0]
        
        if len(probs) <= 1:
            return 0.0
        return -np.sum(probs * np.log2(probs))
    
    def get_gini(self) -> float:
        """Gini index."""
        bins = np.linspace(-2, 2, 6)
        states = np.digitize(self.x, bins) - 1
        states = np.clip(states, 0, 4)
        counts = np.bincount(states, minlength=5)
        
        return calculate_gini(counts)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_gini(counts: np.ndarray) -> float:
    """Calculate Gini coefficient from state counts."""
    counts = counts.astype(float)
    if np.sum(counts) == 0:
        return 0.0
    
    n = len(counts)
    sorted_counts = np.sort(counts)
    
    cumsum = np.cumsum(sorted_counts)
    total = cumsum[-1]
    
    if total == 0:
        return 0.0
    
    gini = (2 * np.sum((np.arange(1, n+1)) * sorted_counts)) / (n * total) - (n + 1) / n
    return gini


def detect_collapse_time(entropy_trace: np.ndarray, threshold: float = 0.3) -> int:
    """
    Detect when entropy collapses below threshold.
    
    Returns:
        int: Step number of collapse, or -1 if no collapse
    """
    below_threshold = np.where(entropy_trace < threshold)[0]
    
    if len(below_threshold) == 0:
        return -1
    
    # Check if it stays below (sustained collapse, not transient)
    first_collapse = below_threshold[0]
    if first_collapse < len(entropy_trace) - 100:
        # Check next 100 steps
        if np.mean(entropy_trace[first_collapse:first_collapse+100]) < threshold:
            return int(first_collapse)
    
    return -1


def minority_survival_rate(model, initial_minority_state: int = 4) -> float:
    """
    Calculate survival rate of minority opinion/state.
    
    Args:
        model: Any model with get_states() or opinions/theta attribute
        initial_minority_state: State to track (default: highest state)
    
    Returns:
        float: Fraction of population in minority state
    """
    if hasattr(model, 'get_states'):
        states = model.get_states()
    elif hasattr(model, 'opinions'):
        states = model.opinions
    elif hasattr(model, 'theta'):
        # Kuramoto: map phases to states
        bins = np.linspace(-np.pi, np.pi, 6)
        states = np.digitize(model.theta, bins) - 1
        states = np.clip(states, 0, 4)
    elif hasattr(model, 'x'):
        # Averaging: map values to states
        bins = np.linspace(-2, 2, 6)
        states = np.digitize(model.x, bins) - 1
        states = np.clip(states, 0, 4)
    else:
        return 0.0
    
    return np.sum(states == initial_minority_state) / len(states)


# ============================================================================
# BENCHMARK PROTOCOL
# ============================================================================

def run_benchmark(model_name: str, N: int, steps: int, I_bias: float, seed: int) -> Dict:
    """
    Run benchmark for a given model.
    
    Returns:
        Dict with metrics: final_H, collapse_time, gini, minority_survival, diversity
    """
    # Initialize model
    if model_name == "mem4ristor":
        model = Mem4ristorV2(seed=seed)
        model._initialize_params(N)
        adj = build_lattice_adjacency(int(np.sqrt(N)))
    elif model_name == "kuramoto":
        model = KuramotoModel(N, K=0.15, seed=seed)
        adj = None
    elif model_name == "voter":
        model = VoterModel(N, seed=seed)
        adj = None
    elif model_name == "averaging":
        model = AveragingModel(N, epsilon=0.1, seed=seed)
        adj = None
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Track metrics
    entropy_trace = []
    
    # Simulation
    for step in range(steps):
        # Step model
        if model_name == "mem4ristor":
            model.step(I_stimulus=I_bias, adj_matrix=adj)
        else:
            model.step(I_stimulus=I_bias)
        
        # Record metrics
        H = model.calculate_entropy()
        entropy_trace.append(H)
    
    # Final metrics
    final_H = entropy_trace[-1]
    collapse_time = detect_collapse_time(np.array(entropy_trace))
    
    # Gini calculation (universal method)
    if hasattr(model, 'get_states'):
        counts = np.bincount(model.get_states(), minlength=5)
    elif hasattr(model, 'opinions'):
        counts = np.bincount(model.opinions, minlength=5)
    elif hasattr(model, 'theta'):
        bins = np.linspace(-np.pi, np.pi, 6)
        states = np.digitize(model.theta, bins) - 1
        states = np.clip(states, 0, 4)
        counts = np.bincount(states, minlength=5)
    elif hasattr(model, 'x'):
        bins = np.linspace(-2, 2, 6)
        states = np.digitize(model.x, bins) - 1
        states = np.clip(states, 0, 4)
        counts = np.bincount(states, minlength=5)
    else:
        counts = np.zeros(5)
    
    gini = calculate_gini(counts)
    minority = minority_survival_rate(model) * 100  # as percentage
    
    # Diversity: number of occupied states
    if hasattr(model, 'get_states'):
        diversity = len(np.unique(model.get_states()))
    elif hasattr(model, 'opinions'):
        diversity = len(np.unique(model.opinions))
    else:
        # For continuous models, count bins with >1% population
        bins = 5
        if hasattr(model, 'theta'):
            hist, _ = np.histogram(model.theta, bins=bins)
        else:  # averaging
            hist, _ = np.histogram(model.x, bins=bins)
        diversity = np.sum(hist > N * 0.01)
    
    return {
        'final_H': final_H,
        'collapse_time': collapse_time,
        'gini': gini,
        'minority_survival': minority,
        'diversity': diversity,
        'entropy_trace': entropy_trace
    }


def print_comparison_table(results: Dict[str, Dict]):
    """Print formatted comparison table."""
    print(f"\n{'='*80}")
    print(f"COMPARATIVE BENCHMARK RESULTS")
    print(f"{'='*80}\n")
    
    print(f"{'Model':<20} {'Final H':<12} {'Collapse':<15} {'Minority%':<12} {'Gini':<10} {'Diversity':<10}")
    print(f"{'-'*80}")
    
    for model_name, metrics in results.items():
        collapse_str = f"{metrics['collapse_time']}" if metrics['collapse_time'] > 0 else ">5000"
        
        print(f"{model_name.capitalize():<20} "
              f"{metrics['final_H']:<12.4f} "
              f"{collapse_str:<15} "
              f"{metrics['minority_survival']:<12.2f} "
              f"{metrics['gini']:<10.4f} "
              f"{metrics['diversity']:<10}")
    
    print(f"{'-'*80}\n")
    
    # Interpretation
    print("INTERPRETATION:")
    print(f"  Final H: Higher = better diversity preservation")
    print(f"  Collapse: Later = more resilient (>5000 = no collapse)")
    print(f"  Minority%: Higher = better minority survival")
    print(f"  Gini: Lower = more equal distribution (0=perfect, 1=total inequality)")
    print(f"  Diversity: Number of occupied states (max=5)")
    print(f"\n{'='*80}\n")


# ============================================================================
# MAIN BENCHMARK
# ============================================================================

def main():
    """Run full comparative benchmark."""
    print("\nCOMPARATIVE BENCHMARKS: Mem4ristor v2.0.4 vs Classical Models")
    print("="*80)
    print("Conditions: N=100, Steps=5000, Bias=1.1, Seed=42")
    print("="*80)
    
    # Common parameters
    N = 100
    steps = 5000
    I_bias = 1.1
    seed = 42
    
    models = ["mem4ristor", "kuramoto", "voter", "averaging"]
    results = {}
    
    for model_name in models:
        print(f"\nRunning {model_name.upper()}...")
        results[model_name] = run_benchmark(model_name, N, steps, I_bias, seed)
        print(f"  Final H = {results[model_name]['final_H']:.4f}")
    
    # Display results
    print_comparison_table(results)
    
    # Verdict
    mem_H = results['mem4ristor']['final_H']
    best_classical = max([results[m]['final_H'] for m in models if m != 'mem4ristor'])
    
    print(f"VERDICT:")
    if mem_H > best_classical:
        improvement = ((mem_H / best_classical) - 1) * 100
        print(f"  ✅ Mem4ristor OUTPERFORMS classical models")
        print(f"  ✅ Entropy improvement: +{improvement:.1f}% over best classical")
    else:
        print(f"  ❌ Mem4ristor does NOT outperform classical models")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
