import numpy as np
from typing import Dict, Tuple


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


# ---------------------------------------------------------------------------
# Spatial Mutual Information (spatio-temporal structure)
# Requires v_history (T, N) + adjacency matrix.
# ---------------------------------------------------------------------------

def _pairwise_mi(vi: np.ndarray, vj: np.ndarray, n_bins: int = 20) -> float:
    """
    Mutual Information between two scalar time series via joint histogram.
    Returns MI in bits.
    """
    v_min = min(vi.min(), vj.min()) - 1e-9
    v_max = max(vi.max(), vj.max()) + 1e-9
    joint, _, _ = np.histogram2d(vi, vj, bins=n_bins,
                                  range=[[v_min, v_max], [v_min, v_max]])
    total = joint.sum()
    if total == 0:
        return 0.0
    joint = joint / total
    pi = joint.sum(axis=1, keepdims=True)   # marginal i
    pj = joint.sum(axis=0, keepdims=True)   # marginal j
    denom = pi * pj
    mask = (joint > 0) & (denom > 0)
    mi = float(np.sum(joint[mask] * np.log2(joint[mask] / denom[mask])))
    return max(mi, 0.0)


def calculate_spatial_mutual_information(
    v_history: np.ndarray,
    adjacency_matrix: np.ndarray,
    n_bins: int = 20,
    max_pairs_per_dist: int = 200,
    max_dist: int = 10,
    seed: int = 0,
) -> Dict[int, Tuple[float, float]]:
    """
    Compute mean MI between node pairs grouped by graph hop-distance.

    Returns a dict  {distance: (mean_MI, std_MI)}  for distances 1..max_dist
    that have at least one pair.

    v_history      : (T, N) — T timesteps, N nodes.
    adjacency_matrix: (N, N) — used to compute shortest-path distances.
    n_bins         : histogram bins for MI estimation.
    max_pairs_per_dist : cap on pairs sampled per distance bucket (speed).
    max_dist       : maximum graph distance to compute.
    """
    from scipy.sparse.csgraph import shortest_path
    from scipy.sparse import csr_matrix

    T, N = v_history.shape
    if T < 10 or N < 2:
        return {}

    rng = np.random.RandomState(seed)

    # Shortest-path distances (unweighted)
    A_bin = (adjacency_matrix > 0).astype(float)
    dist_matrix = shortest_path(csr_matrix(A_bin), method='D',
                                 directed=False, unweighted=True)

    # z-score node traces for MI stability
    mu  = v_history.mean(axis=0)
    std = v_history.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    vz  = (v_history - mu) / std   # (T, N)

    results: Dict[int, list] = {}
    for d in range(1, max_dist + 1):
        ii, jj = np.where((dist_matrix == d) & (np.arange(N)[:, None] < np.arange(N)[None, :]))
        if len(ii) == 0:
            continue
        if len(ii) > max_pairs_per_dist:
            idx = rng.choice(len(ii), max_pairs_per_dist, replace=False)
            ii, jj = ii[idx], jj[idx]
        mis = [_pairwise_mi(vz[:, i], vz[:, j], n_bins) for i, j in zip(ii, jj)]
        results[d] = mis

    return {d: (float(np.mean(v)), float(np.std(v))) for d, v in results.items()}
