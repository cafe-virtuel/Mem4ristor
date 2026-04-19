import numpy as np

def calculate_continuous_entropy(v_array: np.ndarray, bins: int = 100, v_min: float = -3.0, v_max: float = 3.0) -> float:
    """
    Shannon entropy estimation using a fine uniform grid.
    Replaces the previous 5-bin artificial ceiling with a continuous approximation.
    """
    counts, _ = np.histogram(v_array, bins=bins, range=(v_min, v_max))
    total = np.sum(counts)
    if total == 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    return -float(np.sum(probs * np.log2(probs)))

def calculate_cognitive_entropy(v_array: np.ndarray) -> float:
    """
    KIMI-Corrected 5-bin discrete physiological entropy.
    Uses the exact published thresholds from the preprint: ±0.4, ±1.2
    """
    bin_edges = [-np.inf, -1.2, -0.4, 0.4, 1.2, np.inf]
    counts, _ = np.histogram(v_array, bins=bin_edges)
    total = np.sum(counts)
    if total == 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    return -float(np.sum(probs * np.log2(probs)))

def get_cognitive_states(v_array: np.ndarray) -> np.ndarray:
    """
    KIMI-Corrected physiologic state extraction based on threshold ±0.4, ±1.2.
    """
    states = np.zeros_like(v_array, dtype=int)
    states[v_array < -1.2] = 1
    states[(v_array >= -1.2) & (v_array < -0.4)] = 2
    states[(v_array >= -0.4) & (v_array <= 0.4)] = 3
    states[(v_array > 0.4) & (v_array <= 1.2)] = 4
    states[v_array > 1.2] = 5
    return states
