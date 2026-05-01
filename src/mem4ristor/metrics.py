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
    # FIX (Reviewer 2 Vague 2): Use 1st and 99th percentiles instead of absolute min/max 
    # to prevent a single thermal outlier from stretching the bins and destroying the metric.
    v_min = np.percentile(v_history, 1)
    v_max = np.percentile(v_history, 99)
    
    if v_max <= v_min:
        return 0.0
        
    v_clipped = np.clip(v_history, v_min, v_max)
    bin_idx = np.floor(
        (v_clipped - v_min) / (v_max - v_min) * n_bins
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


def calculate_kuramoto_order_parameter(v_history: np.ndarray, w_history: np.ndarray = None) -> float:
    """
    Continuous Kuramoto order parameter via Geometric Phase (Reviewer 2 Vague 2 Fix).
    Avoids the Hilbert transform artifact on broadband/spiky signals by using the 
    actual (v, w) phase space of the oscillator.

    v_history: shape (T, N)
    w_history: shape (T, N) - Required for robust geometric phase. If None, falls back to Hilbert.
    Returns: Mean synchronization R (0 to 1).
    """
    T, N = v_history.shape
    if T < 10 or N < 2:
        return 0.0
    
    if w_history is not None:
        # Geometric Phase in (v, w) plane
        v_c = v_history - np.mean(v_history, axis=0)
        w_c = w_history - np.mean(w_history, axis=0)
        # Normalize to avoid elliptical distortion in the atan2
        v_c = v_c / (np.std(v_c, axis=0) + 1e-9)
        w_c = w_c / (np.std(w_c, axis=0) + 1e-9)
        phase = np.arctan2(w_c, v_c)
        margin = 0 # No edge artifacts with geometric phase
    else:
        # Fallback to Hilbert (deprecated for FHN broadband spikes)
        from scipy.signal import hilbert
        v_centered = v_history - np.mean(v_history, axis=0)
        analytic_signal = hilbert(v_centered, axis=0)
        phase = np.angle(analytic_signal)
        margin = max(1, int(T * 0.1))
    
    # R(t) = | 1/N * sum(e^{i * phase_j(t)}) |
    order_param_t = np.abs(np.mean(np.exp(1j * phase), axis=1))
    
    if margin > 0:
        return float(np.mean(order_param_t[margin:-margin]))
    return float(np.mean(order_param_t))


def calculate_transfer_entropy(source_history: np.ndarray, target_history: np.ndarray, bins: int = 6) -> float:
    """
    Computes Discrete Transfer Entropy TE_{X -> Y} from source to target.
    X = source_history, Y = target_history.
    TE_{X -> Y} = H(Y_{t}, Y_{t-1}) + H(Y_{t-1}, X_{t-1}) - H(Y_{t-1}) - H(Y_{t}, Y_{t-1}, X_{t-1})
    
    Both histories must be 1D numpy arrays of the same length, discretized into integers [0, bins-1].
    (e.g., using cognitive_states).
    """
    if len(source_history) != len(target_history) or len(source_history) < 2:
        return 0.0
        
    X = source_history[:-1]
    Y_prev = target_history[:-1]
    Y_curr = target_history[1:]
    
    # Compute joint histograms
    # H(Y_{t-1})
    p_Y_prev, _ = np.histogram(Y_prev, bins=bins, range=(-0.5, bins-0.5), density=True)
    p_Y_prev = p_Y_prev[p_Y_prev > 0]
    H_Y_prev = -np.sum(p_Y_prev * np.log2(p_Y_prev))
    
    # H(Y_t, Y_{t-1})
    p_Y_curr_prev, _, _ = np.histogram2d(Y_curr, Y_prev, bins=bins, range=[[-0.5, bins-0.5], [-0.5, bins-0.5]], density=True)
    p_Y_curr_prev = p_Y_curr_prev[p_Y_curr_prev > 0]
    H_Y_curr_prev = -np.sum(p_Y_curr_prev * np.log2(p_Y_curr_prev))
    
    # H(Y_{t-1}, X_{t-1})
    p_Y_prev_X_prev, _, _ = np.histogram2d(Y_prev, X, bins=bins, range=[[-0.5, bins-0.5], [-0.5, bins-0.5]], density=True)
    p_Y_prev_X_prev = p_Y_prev_X_prev[p_Y_prev_X_prev > 0]
    H_Y_prev_X_prev = -np.sum(p_Y_prev_X_prev * np.log2(p_Y_prev_X_prev))
    
    # H(Y_t, Y_{t-1}, X_{t-1})
    # For 3D histogram, we use numpy.histogramdd
    data_3d = np.vstack((Y_curr, Y_prev, X)).T
    p_3d, _ = np.histogramdd(data_3d, bins=bins, range=[[-0.5, bins-0.5]]*3, density=True)
    p_3d = p_3d[p_3d > 0]
    H_3d = -np.sum(p_3d * np.log2(p_3d))
    
    TE = H_Y_curr_prev + H_Y_prev_X_prev - H_Y_prev - H_3d
    print(f"H_Yp={H_Y_prev:.3f}, H_Ycp={H_Y_curr_prev:.3f}, H_YpXp={H_Y_prev_X_prev:.3f}, H_3d={H_3d:.3f}, TE={TE:.5f}")
    return max(0.0, float(TE))



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
