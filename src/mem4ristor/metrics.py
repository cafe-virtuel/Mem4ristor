import numpy as np


# ---------------------------------------------------------------------------
# Snapshot-based metrics (spatial diversity at a single timestep)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Trajectory-based metrics (temporal structure & inter-node coordination)
# These require v_history of shape (T, N) — T snapshots, N nodes.
# ---------------------------------------------------------------------------

def _lz76_phrases(s: str) -> int:
    """
    LZ76 greedy parsing: count distinct phrases.
    O(n²) worst case but fine for typical trace lengths (T~300).
    """
    if not s:
        return 0
    dictionary: set[str] = set()
    phrase = ""
    count = 0
    for ch in s:
        candidate = phrase + ch
        if candidate in dictionary:
            phrase = candidate
        else:
            dictionary.add(candidate)
            count += 1
            phrase = ""
    if phrase:
        count += 1
    return count


def calculate_temporal_lz_complexity(
    v_history: np.ndarray,
    n_bins: int = 5,
) -> float:
    """
    Normalized LZ76 complexity of temporal cognitive-state sequences, averaged
    over nodes.  Low value → structured/predictable trajectories; high value
    → random (approaching the complexity of a random sequence).

    v_history: shape (T, N) — T timesteps, N nodes.
    Returns a float in (0, 1].  A fully random binary sequence converges to 1.
    """
    T, N = v_history.shape
    if T < 2:
        return 0.0

    # Discretise each node's v(t) trajectory into n_bins uniform bins
    v_min, v_max = v_history.min(), v_history.max()
    if v_max == v_min:
        return 0.0
    bin_idx = np.floor(
        (v_history - v_min) / (v_max - v_min) * n_bins
    ).astype(int).clip(0, n_bins - 1)  # shape (T, N)

    total = 0.0
    log2_T = np.log2(T)
    for j in range(N):
        seq = "".join(str(x) for x in bin_idx[:, j])
        c = _lz76_phrases(seq)
        # Normalise so that a random sequence ≈ 1
        total += c * log2_T / T
    return total / N


def calculate_pairwise_synchrony(v_history: np.ndarray) -> float:
    """
    Mean pairwise Pearson correlation of v(t) traces across all node pairs.

    v_history: shape (T, N).
    Returns a float in [-1, 1].  Near +1 = globally synchronised; near 0 =
    independent nodes; near -1 = anti-synchronised.

    For large N use a random subsample of pairs to stay O(N) instead of O(N²).
    """
    T, N = v_history.shape
    if T < 2 or N < 2:
        return 0.0

    # Subsample pairs for large N (avoid O(N²) cost)
    MAX_PAIRS = 2000
    rng = np.random.RandomState(0)
    if N * (N - 1) // 2 <= MAX_PAIRS:
        i_idx, j_idx = np.triu_indices(N, k=1)
    else:
        pairs = set()
        while len(pairs) < MAX_PAIRS:
            a, b = rng.randint(0, N, size=2)
            if a != b:
                pairs.add((min(a, b), max(a, b)))
        i_idx, j_idx = zip(*pairs)
        i_idx, j_idx = np.array(i_idx), np.array(j_idx)

    # z-score each node's trace
    mu = v_history.mean(axis=0)       # (N,)
    std = v_history.std(axis=0)       # (N,)
    std = np.where(std < 1e-12, 1.0, std)
    z = (v_history - mu) / std        # (T, N)

    # Pearson r for each pair = dot product of z-scores / T
    corrs = np.einsum('ti,tj->ij', z, z)[i_idx, j_idx] / T
    return float(corrs.mean())
