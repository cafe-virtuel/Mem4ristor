import os
# MKL Determinism Fix (v2.9.1)
os.environ['NUMPY_MKL_CBWR'] = 'COMPATIBLE'
import numpy as np
import yaml
from typing import Dict, List, Optional
from scipy.integrate import solve_ivp
from scipy import sparse as sp_sparse
from scipy.sparse.linalg import eigsh as sparse_eigsh


class Mem4ristorV3:
    """
    Canonical Implementation of Mem4ristor v3.0.0 (with v4 adaptive extensions).

    Implements extended FitzHugh-Nagumo dynamics with constitutional doubt (u),
    structural heretics, Levitating Sigmoid coupling, inhibition plasticity,
    and adaptive meta-doubt for diversity preservation in neuromorphic-inspired
    computational models.

    Core Equations:
        dv/dt = v - v³/5 - w + I_ext - α·tanh(v) + η(t)
        dw/dt = ε(v + a - bw) + dw_plasticity
        du/dt = ε_u_eff(i) · (k_u·σ_social + σ_baseline - u) / τ_u

    Adaptive Meta-Doubt (v4 extension):
        ε_u_eff(i) = ε_u × (1 + α_surprise × σ_social(i))
        When social coupling contradicts a unit's internal state, the doubt
        speed increases proportionally. This provides social-surprise-driven
        meta-plasticity: stable environments → slow doubt adaptation, volatile
        environments → fast doubt adaptation. The gain is clamped to prevent
        runaway acceleration (max 5× base speed).

    Coupling Kernel (Levitating Sigmoid):
        f(u) = tanh(π(0.5 - u)) + δ
        Eliminates the dead zone at u=0.5 where linear (1-2u) made coupling
        noise-dominated. The leakage term δ ensures non-zero coupling everywhere.

    Inhibition Plasticity:
        When a unit is dissident (u > 0.5) and under social pressure,
        its recovery variable w accumulates structural memory of dissidence.
        dw_learn = λ · σ_social · I(u>0.5) · (1 - (w/w_sat)²) - w/τ_plasticity

    Key Features:
        - Smooth repulsive coupling via Levitating Sigmoid
        - Heretic units: 15% with inverted stimulus polarity (empirical threshold)
        - 1/√N scaling for consistent dynamics across network sizes
        - Structural memory of dissidence via inhibition plasticity
        - Adaptive ε_u driven by social surprise (meta-doubt, v4)

    Migration Note (v2.9.3 → v3.0.0):
        This class replaces Mem4ristorV2. The linear kernel (1-2u) is replaced
        by tanh(π(0.5-u)) + δ. All security guards from V2 are preserved.
        See CHANGELOG_V3.md for complete migration details.

    Attributes:
        N (int): Number of units in network
        v (ndarray): Cognitive potential states (N,)
        w (ndarray): Recovery/inhibition states (N,)
        u (ndarray): Constitutional doubt levels (N,), clamped [0,1]
        heretic_mask (ndarray): Boolean mask identifying heretic units
        D_eff (float): Effective coupling strength = D/√N
        cfg (dict): Configuration parameters from YAML

    Example:
        >>> model = Mem4ristorV3(seed=42)
        >>> model._initialize_params(N=100)
        >>> for _ in range(1000):
        ...     model.step(I_stimulus=0.0)
        >>> entropy = model.calculate_entropy()
        >>> print(f"Final entropy: {entropy:.4f}")

    References:
        See docs/preprint.tex for complete mathematical specification.
    """
    def __init__(self, config: Optional[Dict] = None, seed: int = 42):
        # Default Configuration
        default_cfg = {
            'dynamics': {
                'a': 0.7, 'b': 0.8, 'epsilon': 0.08, 'alpha': 0.15,
                'v_cubic_divisor': 5.0, 'dt': 0.05,
                'lambda_learn': 0.05, 'tau_plasticity': 1000, 'w_saturation': 2.0
            },
            'coupling': {'D': 0.15, 'heretic_ratio': 0.15, 'uniform_placement': True},
            'doubt': {'epsilon_u': 0.02, 'k_u': 1.0, 'sigma_baseline': 0.05, 'u_clamp': [0.0, 1.0], 'tau_u': 1.0,
                      'alpha_surprise': 2.0, 'surprise_cap': 5.0},
            'noise': {'sigma_v': 0.05, 'use_rtn': False, 'rtn_amplitude': 0.1, 'rtn_p_flip': 0.01},
            'hysteresis': {'enabled': True, 'theta_low': 0.35, 'theta_high': 0.65,
                           'fatigue_rate': 0.0, 'base_hysteresis': 0.15}
        }

        # 1. Config Hardening: Deep Merge
        if config is None:
            try:
                cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
                with open(cfg_path, 'r') as f:
                    file_cfg = yaml.safe_load(f)
                    self.cfg = self._deep_merge(default_cfg, file_cfg)
            except (FileNotFoundError, yaml.YAMLError):
                self.cfg = default_cfg
        else:
            self.cfg = self._deep_merge(default_cfg, config)

        self.rng = np.random.RandomState(seed)
        self.dt = self.cfg['dynamics']['dt']

        # 2. Parameter Validation
        self._validate_config()

        # 3. V3 Plasticity Parameters
        self.lambda_learn = self.cfg['dynamics'].get('lambda_learn', 0.05)
        self.tau_plasticity = self.cfg['dynamics'].get('tau_plasticity', 1000)
        self.w_saturation = self.cfg['dynamics'].get('w_saturation', 2.0)

        # 4. Levitating Sigmoid Parameters
        self.sigmoid_steepness = np.pi
        self.social_leakage = 0.05  # δ: ensures non-zero coupling everywhere

        # Dimensions are defined at model level or implicitly by state
        self.N = 100  # Default
        self._initialize_params()

    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """Recursive merge of dictionaries to prevent missing keys."""
        result = base.copy()
        for key, value in update.items():
            if key in result:
                # GUARD: Type Confusion (Round 2 Fix)
                if isinstance(result[key], dict):
                    if not isinstance(value, dict):
                        raise TypeError(f"Config key '{key}' expects dict, got {type(value).__name__}")
                    result[key] = self._deep_merge(result[key], value)
                else:
                    result[key] = value
            else:
                result[key] = value
        return result

    def _validate_config(self):
        """Ensure critical parameters are safe."""
        if self.cfg['dynamics']['v_cubic_divisor'] <= 1e-9:
            raise ValueError("Configuration Error: 'v_cubic_divisor' must be > 0 to prevent division by zero.")
        if self.cfg['doubt']['tau_u'] <= 1e-9:
            raise ValueError("Configuration Error: 'tau_u' must be > 0.")
        if self.dt <= 0:
            raise ValueError("Configuration Error: 'dt' must be positive.")

        D = self.cfg['coupling'].get('D', 0.15)
        if not np.isfinite(D):
            raise ValueError("Configuration Error: 'D' must be finite.")

        if self.cfg['noise'].get('use_rtn', False):
            p_flip = self.cfg['noise'].get('rtn_p_flip', 0.01)
            if not (0.0 <= p_flip <= 1.0):
                raise ValueError(f"Configuration Error: 'rtn_p_flip' must be in [0, 1], got {p_flip}")

        heretic_ratio = self.cfg['coupling'].get('heretic_ratio', 0.15)
        if not (0.0 <= heretic_ratio <= 1.0):
            raise ValueError(f"Configuration Error: 'heretic_ratio' must be in [0, 1], got {heretic_ratio}")

    def _initialize_params(self, N=100, cold_start=False):
        # Size Validation (DoS Fix)
        if N <= 0:
            raise ValueError(f"Network size N must be positive, got {N}.")
        if N > 10_000_000:
            raise ValueError(f"Network size N={N} exceeds maximum allowed (10M) to prevent DoS.")

        self.N = N
        if cold_start:
            self.v = np.zeros(self.N)
            self.w = np.zeros(self.N)
        else:
            self.v = self.rng.uniform(-1.5, 1.5, self.N)
            self.w = self.rng.uniform(0.0, 1.0, self.N)

        self.u = np.full(self.N, self.cfg['doubt']['sigma_baseline'])

        # Anti-Clustering Placement
        heretic_ratio = self.cfg['coupling'].get('heretic_ratio', 0.15)

        if heretic_ratio <= 0:
            self.heretic_mask = np.zeros(self.N, dtype=bool)
        elif self.cfg['coupling'].get('uniform_placement', True):
            step = int(1.0 / heretic_ratio)
            if step < 1:
                step = 1
            heretic_ids = []
            for i in range(0, self.N, step):
                if len(heretic_ids) < int(self.N * heretic_ratio):
                    block_end = min(i + step, self.N)
                    heretic_ids.append(self.rng.randint(i, block_end))
            self.heretic_mask = np.zeros(self.N, dtype=bool)
            self.heretic_mask[heretic_ids] = True
        else:
            self.heretic_mask = self.rng.rand(self.N) < heretic_ratio

        self.D_eff = self.cfg['coupling']['D'] / np.sqrt(self.N)

        # V5: Hysteresis state arrays
        self.mode_state = np.zeros(self.N, dtype=bool)   # False=Sage, True=Fou
        self.time_in_state = np.zeros(self.N, dtype=float)

    def _update_hysteresis(self):
        """
        V5 Hysteresis: Update mode_state with dead-zone latching and watchdog fatigue.

        Dead Zone: [theta_low, theta_high]
          - Sage(False) -> Fou(True): only when u >= theta_high
          - Fou(True) -> Sage(False): only when u < theta_low
          - Inside dead zone: state LATCHES (no change)

        Watchdog Fatigue (V5.1): if a node stays in the same state too long,
        the effective thresholds converge toward 0.5, shrinking the dead zone.
        This prevents permanent latching and mimics memristor fatigue.
        """
        hyst = self.cfg['hysteresis']
        if not hyst.get('enabled', False):
            self.mode_state = self.u > 0.5
            return

        theta_low = hyst['theta_low']
        theta_high = hyst['theta_high']
        fatigue_rate = hyst.get('fatigue_rate', 0.0)

        # Compute effective thresholds with watchdog fatigue
        fatigue_shift = fatigue_rate * self.time_in_state
        eff_theta_low = np.minimum(theta_low + fatigue_shift, 0.5)
        eff_theta_high = np.maximum(theta_high - fatigue_shift, 0.5)

        # Transition conditions (hysteresis latching)
        switch_to_fou = (~self.mode_state) & (self.u >= eff_theta_high)
        switch_to_sage = self.mode_state & (self.u < eff_theta_low)

        # Apply transitions
        switched = switch_to_fou | switch_to_sage
        self.mode_state[switch_to_fou] = True
        self.mode_state[switch_to_sage] = False

        # Timer management: reset on switch, increment otherwise
        self.time_in_state[switched] = 0.0
        self.time_in_state[~switched] += self.dt

    def step(self, I_stimulus: float = 0.0, coupling_input: Optional[np.ndarray] = None) -> None:
        """
        Advance system by one time step using Euler integration.

        Implements Levitating Sigmoid coupling and inhibition plasticity.

        Args:
            I_stimulus: External stimulus magnitude (default: 0.0)
            coupling_input: Adjacency matrix (N, N) or pre-calculated Laplacian vector (N,).
                           If (N,N), computes: (adj @ v - v).
                           If (N,), treats as the direct Laplacian result.

        Returns:
            None (updates internal state in-place)
        """
        # GUARD: Deterministic Input (Round 3 Fix - Chaos Injection)
        if hasattr(I_stimulus, '__float__') and not isinstance(I_stimulus, (int, float, np.number, np.ndarray)):
            raise TypeError("Stimulus must be a numeric constant (int/float/array) to ensure reproducibility.")

        # GUARD 0: NaN/Inf detection on state variables
        if np.any(~np.isfinite(self.v)):
            self.v = np.nan_to_num(self.v, nan=0.0, posinf=0.0, neginf=0.0)
        if np.any(~np.isfinite(self.w)):
            self.w = np.nan_to_num(self.w, nan=0.0, posinf=0.0, neginf=0.0)
        if np.any(~np.isfinite(self.u)):
            self.u = np.nan_to_num(self.u, nan=0.5, posinf=0.5, neginf=0.5)

        # GUARD: Coupling Input Sanitization
        if coupling_input is not None:
            try:
                coupling_input = np.array(coupling_input, dtype=float)
            except (ValueError, TypeError, AttributeError):
                raise ValueError(f"Invalid coupling input: {coupling_input}. Must be numeric.")

        if coupling_input is None:
            laplacian_v = np.zeros(self.N)
        elif coupling_input.ndim == 2:
            laplacian_v = coupling_input @ self.v - self.v
        else:
            laplacian_v = coupling_input

        # GUARD 1: NaN/Inf detection in coupling
        if np.any(np.isnan(laplacian_v)) or np.any(np.isinf(laplacian_v)):
            laplacian_v = np.nan_to_num(laplacian_v, nan=0.0, posinf=1.0, neginf=-1.0)

        sigma_social = np.abs(laplacian_v)

        # Noise Modeling
        eta = self.rng.normal(0, self.cfg['noise'].get('sigma_v', 0.05), self.N)

        # RTN (Random Telegraph Noise) - Physical Hardware Signature
        if self.cfg['noise'].get('use_rtn', False):
            rtn_amp = self.cfg['noise'].get('rtn_amplitude', 0.1)
            p_flip = self.cfg['noise'].get('rtn_p_flip', 0.01)
            rtn_jumps = (self.rng.rand(self.N) < p_flip).astype(float) * rtn_amp * self.rng.choice([-1, 1], size=self.N)
            eta += rtn_jumps

        # V3 CORE: Levitating Sigmoid Coupling
        # f(u) = tanh(π(0.5 - u)) + δ
        # u=0 → tanh(π/2) ≈ +0.92 + δ (attraction)
        # u=0.5 → tanh(0) = 0 + δ (weak attraction from leakage)
        # u=1 → tanh(-π/2) ≈ -0.92 + δ (repulsion)
        u_centered = 0.5 - self.u
        u_filter = np.tanh(self.sigmoid_steepness * u_centered) + self.social_leakage

        I_coup = self.D_eff * u_filter * laplacian_v

        # Input Sanitization
        try:
            stim_arr = np.array(I_stimulus, dtype=float)
            if stim_arr.ndim == 0:
                I_eff = np.full(self.N, float(stim_arr))
            else:
                I_eff = stim_arr.flatten()
                if I_eff.size != self.N:
                    raise ValueError(f"Stimulus vector size {I_eff.size} must match network size {self.N}")
        except (ValueError, TypeError, AttributeError):
            raise ValueError(f"Invalid stimulus input: {I_stimulus}. Must be numeric.")

        if np.any(np.isnan(I_eff)):
            I_eff = np.nan_to_num(I_eff, nan=0.0)

        # GUARD 2: Clamp stimulus to prevent overflow
        I_eff = np.clip(I_eff, -100.0, 100.0)

        # Apply heretic inversion
        I_eff[self.heretic_mask] *= -1.0
        I_ext = I_eff + I_coup

        # V3 MOTOR: Inhibition Plasticity
        # Units that are actively doubting accumulate structural memory
        # V5: Use hysteresis mode_state if enabled, else hard threshold
        if self.cfg.get('hysteresis', {}).get('enabled', False):
            self._update_hysteresis()
            innovation_mask = self.mode_state.astype(float)
        else:
            innovation_mask = (self.u > 0.5).astype(float)
        plasticity_drive = self.lambda_learn * sigma_social * innovation_mask

        w_ratio = self.w / self.w_saturation
        saturation_factor = np.clip(1.0 - (w_ratio**2), 0.0, 1.0)
        plasticity_decay = self.w / self.tau_plasticity

        dw_learning = (plasticity_drive * saturation_factor) - plasticity_decay

        # FHN Dynamics
        dv = (self.v - (self.v**3) / self.cfg['dynamics']['v_cubic_divisor'] - self.w + I_ext -
              self.cfg['dynamics']['alpha'] * np.tanh(self.v) + eta)
        dw = self.cfg['dynamics']['epsilon'] * (self.v + self.cfg['dynamics']['a'] - self.cfg['dynamics']['b'] * self.w)

        # V4: Adaptive Meta-Doubt (Social Surprise)
        # ε_u_eff(i) = ε_u × (1 + α_surprise × σ_social(i))
        # When neighbors contradict, doubt accelerates. Capped to prevent runaway.
        alpha_s = self.cfg['doubt'].get('alpha_surprise', 2.0)
        surprise_cap = self.cfg['doubt'].get('surprise_cap', 5.0)
        # Pre-clamp sigma_social to prevent overflow in alpha_s * sigma_social
        sigma_social_safe = np.clip(sigma_social, 0.0, 100.0)
        epsilon_u_adaptive = self.cfg['doubt']['epsilon_u'] * np.clip(
            1.0 + alpha_s * sigma_social_safe, 1.0, surprise_cap
        )
        du = (epsilon_u_adaptive * (self.cfg['doubt']['k_u'] * sigma_social +
              self.cfg['doubt']['sigma_baseline'] - self.u)) / self.cfg['doubt']['tau_u']

        self.v += dv * self.dt
        self.w += (dw + dw_learning) * self.dt
        self.u += du * self.dt

        # GUARD 3: Clamp to prevent runaway divergence
        self.v = np.clip(self.v, -100.0, 100.0)
        self.w = np.clip(self.w, -100.0, 100.0)
        self.u = np.clip(self.u, self.cfg['doubt']['u_clamp'][0], self.cfg['doubt']['u_clamp'][1])

    def solve_rk45(self, t_span, I_stimulus=0.0, adj_matrix=None):
        """
        High-precision integration using RK45 for long-term stability.

        Uses the Levitating Sigmoid kernel and plasticity in continuous-time.
        """
        # GUARD: Solver Shape Mismatch
        if adj_matrix is not None and adj_matrix.shape != (self.N, self.N):
            raise ValueError(f"Shape mismatch: adj_matrix must be ({self.N}, {self.N}), got {adj_matrix.shape}")

        # GUARD: Time Span Validation
        if t_span[1] <= t_span[0]:
            raise ValueError(f"Invalid time span {t_span}: End time must be greater than start time.")
        if t_span[0] < 0 or t_span[1] < 0:
            raise ValueError(f"Invalid time span {t_span}: Time must be non-negative.")

        # GUARD: Singularity Prevention (Stiffness)
        duration = t_span[1] - t_span[0]
        max_step = min(0.1, duration / 10.0) if duration > 0 else 0.1

        def combined_dynamics(t, y):
            N = self.N
            v = y[:N]
            w = y[N:2*N]
            u = y[2*N:]

            if adj_matrix is None:
                laplacian_v = np.zeros(N)
            else:
                laplacian_v = adj_matrix @ v - v

            sigma_social = np.abs(laplacian_v)

            # V3: Levitating Sigmoid kernel
            u_centered = 0.5 - u
            u_filter = np.tanh(self.sigmoid_steepness * u_centered) + self.social_leakage

            I_eff = np.full(N, float(I_stimulus))
            I_eff[self.heretic_mask] *= -1.0
            I_ext = I_eff + self.D_eff * u_filter * laplacian_v

            eta = self.rng.normal(0, self.cfg['noise'].get('sigma_v', 0.05), N)

            dv = v - (v**3) / self.cfg['dynamics']['v_cubic_divisor'] - w + I_ext - \
                 self.cfg['dynamics']['alpha'] * np.tanh(v) + eta
            dw_fhn = self.cfg['dynamics']['epsilon'] * (v + self.cfg['dynamics']['a'] - self.cfg['dynamics']['b'] * w)

            # V3: Plasticity in continuous-time
            innovation_mask = (u > 0.5).astype(float)
            plasticity_drive = self.lambda_learn * sigma_social * innovation_mask
            w_ratio = w / self.w_saturation
            saturation_factor = np.clip(1.0 - (w_ratio**2), 0.0, 1.0)
            plasticity_decay = w / self.tau_plasticity
            dw_learn = (plasticity_drive * saturation_factor) - plasticity_decay

            dw = dw_fhn + dw_learn

            # V4: Adaptive Meta-Doubt in continuous-time
            alpha_s = self.cfg['doubt'].get('alpha_surprise', 2.0)
            surprise_cap = self.cfg['doubt'].get('surprise_cap', 5.0)
            epsilon_u_adaptive = self.cfg['doubt']['epsilon_u'] * np.clip(
                1.0 + alpha_s * sigma_social, 1.0, surprise_cap
            )
            du = (epsilon_u_adaptive * (self.cfg['doubt']['k_u'] * sigma_social +
                  self.cfg['doubt']['sigma_baseline'] - u)) / self.cfg['doubt']['tau_u']

            return np.concatenate([dv, dw, du])

        y0 = np.concatenate([self.v, self.w, self.u])
        sol = solve_ivp(combined_dynamics, t_span, y0, method='RK45', rtol=1e-6, max_step=max_step)

        # GUARD: Stiff System Detection
        if sol.nfev > 50000:
            import warnings
            warnings.warn(f"High computational cost ({sol.nfev} steps). System might be stiff. Consider checking parameters.")

        y_final = sol.y[:, -1]
        self.v = y_final[:self.N]
        self.w = y_final[self.N:2*self.N]
        self.u = np.clip(y_final[2*self.N:], self.cfg['doubt']['u_clamp'][0], self.cfg['doubt']['u_clamp'][1])
        return sol

    def get_states(self) -> np.ndarray:
        states = np.zeros(self.N, dtype=int)
        v = self.v
        states[v < -1.5] = 1
        states[(v >= -1.5) & (v < -0.8)] = 2
        states[(v >= -0.8) & (v <= 0.8)] = 3
        states[(v > 0.8) & (v <= 1.5)] = 4
        states[v > 1.5] = 5
        return states

    def calculate_entropy(self, bins=5, use_cognitive_bins=True) -> float:
        """
        Shannon entropy of the activation state distribution.
        Used as the primary metric for cognitive diversity.

        Args:
            bins: Number of bins for histogram (used if use_cognitive_bins=False)
            use_cognitive_bins: If True, uses state thresholds aligned with get_states()
                               for scientifically consistent entropy measurement.
        """
        # GUARD: Entropy Calculation Hardening
        try:
            if not use_cognitive_bins:
                bins = int(bins)
                if bins <= 0:
                    raise ValueError
                if bins > 1_000_000:
                    bins = 1_000_000
        except (ValueError, TypeError):
            bins = 5

        v = self.v
        if use_cognitive_bins:
            bin_edges = [-np.inf, -1.5, -0.8, 0.8, 1.5, np.inf]
            counts, _ = np.histogram(v, bins=bin_edges)
        else:
            counts, _ = np.histogram(v, bins=bins, range=(-5.0, 5.0))
        total = np.sum(counts)
        if total == 0:
            return 0.0
        probs = counts / total
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))


# Backward compatibility alias
Mem4ristorV2 = Mem4ristorV3


class Mem4Network:
    """
    High-level API for Mem4ristorV3 with formal Laplacian operators
    and doubt-driven topological rewiring (v4 extension).

    V4 Rewiring Mechanism:
        When a unit's doubt u_i exceeds rewire_threshold, it disconnects from
        its most consensual neighbor (smallest |v_i - v_j|) and reconnects to
        a random non-neighbor. This resolves hub strangulation in scale-free
        networks by allowing high-doubt units to seek diverse information sources.

        The rewiring preserves the total number of edges (degree-neutral) and
        maintains graph symmetry (undirected). A rewire_cooldown prevents
        excessive topological churn.
    """
    def __init__(self, size: int = 10, heretic_ratio: float = 0.15, seed: int = 42,
                 adjacency_matrix: Optional[np.ndarray] = None, cold_start: bool = False,
                 boundary: str = 'periodic',
                 rewire_threshold: float = 0.8, rewire_cooldown: int = 50,
                 coupling_norm: str = 'uniform',
                 auto_sparse_threshold: int = 1000):
        """
        Args:
            size: Grid side length (total units = size * size)
            heretic_ratio: Fraction of heretic units [0, 1]
            seed: Random seed for reproducibility
            adjacency_matrix: Optional explicit adjacency matrix (overrides stencil)
            cold_start: If True, initialize all units to identical state
            boundary: Boundary condition for stencil Laplacian:
                      'periodic' (toroidal wrap, default) or 'neumann' (zero-flux)
            rewire_threshold: Doubt level above which a unit can rewire (default: 0.8)
            rewire_cooldown: Minimum steps between rewires for a given unit (default: 50)
        """
        self.rng = np.random.RandomState(seed)
        self.boundary = boundary

        # V4: Rewiring parameters
        self.rewire_threshold = rewire_threshold
        self.rewire_cooldown = rewire_cooldown
        self.rewire_count = 0  # Total rewires performed (diagnostic)
        self.coupling_norm = coupling_norm
        self._weights_dirty = False

        # Sparse auto-detection and conversion
        if adjacency_matrix is not None and sp_sparse.issparse(adjacency_matrix):
            self.adjacency_matrix = adjacency_matrix.tocsr()
            self._is_sparse = True
        elif adjacency_matrix is not None and adjacency_matrix.shape[0] > auto_sparse_threshold:
            self.adjacency_matrix = sp_sparse.csr_matrix(adjacency_matrix)
            self._is_sparse = True
        else:
            self.adjacency_matrix = adjacency_matrix
            self._is_sparse = False

        if adjacency_matrix is not None:
            # GUARD: Matrix Sanitization
            if self._is_sparse:
                if np.any(np.isnan(self.adjacency_matrix.data)) or np.any(np.isinf(self.adjacency_matrix.data)):
                    raise ValueError("Adjacency matrix contains NaN or Inf.")
            else:
                if np.any(np.isnan(adjacency_matrix)) or np.any(np.isinf(adjacency_matrix)):
                    raise ValueError("Adjacency matrix contains NaN or Inf.")

            self.N = self.adjacency_matrix.shape[0]
            self.size = int(np.sqrt(self.N))
            self.use_stencil = False
            self._rebuild_laplacian()
            # V4: Per-unit cooldown timers
            self._rewire_timers = np.zeros(self.N, dtype=int)
        else:
            self.size = size
            self.N = size * size
            self.use_stencil = True
            self.L = None
            self._rewire_timers = None

        self.model = Mem4ristorV3(seed=seed)
        self.model.cfg['coupling']['heretic_ratio'] = heretic_ratio
        self.model._initialize_params(self.N, cold_start=cold_start)
        self._compute_coupling_weights()

    def _rebuild_laplacian(self):
        """Recompute Laplacian from current adjacency matrix (full rebuild)."""
        if self._is_sparse:
            degrees = np.array(self.adjacency_matrix.sum(axis=1)).flatten()
            D_sp = sp_sparse.diags(degrees, format='csr')
            self.L = D_sp - self.adjacency_matrix
        else:
            deg = np.array(np.sum(self.adjacency_matrix, axis=1)).flatten()
            D = np.diag(deg)
            self.L = D - self.adjacency_matrix
        self._weights_dirty = True

    def _update_laplacian_incremental(self, i, j, k):
        """
        Incremental Laplacian update after a single rewire: disconnect i↔j, connect i↔k.

        Instead of recomputing the full O(N²) Laplacian, we update only the 3 affected
        rows/columns (i, j, k). The Laplacian L = D - A, so:
          - Remove edge (i,j): L[i,i]-=1, L[j,j]-=1, L[i,j]+=1, L[j,i]+=1
          - Add edge (i,k):    L[i,i]+=1, L[k,k]+=1, L[i,k]-=1, L[k,i]-=1
        Net effect on degree: deg(i) unchanged, deg(j)-=1, deg(k)+=1.
        Complexity: O(1) per rewire instead of O(N²).
        """
        if self._is_sparse:
            # For sparse matrices, full rebuild is simpler and still fast
            self._rebuild_laplacian()
            return
        # Dense: O(1) incremental update
        self.L[i, i] -= 1
        self.L[j, j] -= 1
        self.L[i, j] += 1
        self.L[j, i] += 1
        self.L[i, i] += 1
        self.L[k, k] += 1
        self.L[i, k] -= 1
        self.L[k, i] -= 1


    def _compute_coupling_weights(self):
        """Compute per-node coupling weights based on coupling_norm mode.

        Modes:
          - 'uniform': Standard D/sqrt(N) for all nodes (default, backward-compatible)
          - 'degree': D_eff(i) = D / sqrt(deg(i))
          - 'degree_linear': D_eff(i) = D / deg(i) (best for BA m<=3)
          - 'degree_log': D_eff(i) = D / log(1+deg(i))
          - 'degree_power': D_eff(i) = D / deg(i)^alpha (configurable via degree_power_alpha)

        All modes normalized so mean(node_weights) = 1/sqrt(N).
        """
        if self.adjacency_matrix is not None and self.coupling_norm != 'uniform':
            if self._is_sparse:
                degrees = np.maximum(np.array(self.adjacency_matrix.sum(axis=1)).flatten(), 1.0)
            else:
                degrees = np.maximum(np.sum(self.adjacency_matrix, axis=1), 1.0)

            if self.coupling_norm == 'degree':
                raw_weights = 1.0 / np.sqrt(degrees)
            elif self.coupling_norm == 'degree_linear':
                raw_weights = 1.0 / degrees
            elif self.coupling_norm == 'degree_log':
                raw_weights = 1.0 / np.log1p(degrees)
            elif self.coupling_norm == 'degree_power':
                alpha = getattr(self, 'degree_power_alpha', 0.5)
                raw_weights = 1.0 / np.power(degrees, alpha)
            else:
                raw_weights = np.ones(self.N)

            target_mean = 1.0 / np.sqrt(self.N)
            self.node_weights = raw_weights * target_mean / np.mean(raw_weights)
        else:
            self.node_weights = np.ones(self.N) / np.sqrt(self.N)

        self._weights_dirty = False

    def _doubt_driven_rewire(self):
        """
        V4: Topological rewiring driven by constitutional doubt.

        For each unit i where u_i > rewire_threshold and cooldown has expired:
        1. Find its most consensual neighbor j (smallest |v_i - v_j|)
        2. Disconnect i ↔ j
        3. Reconnect i ↔ k where k is a random non-neighbor
        4. Reset cooldown timer for unit i

        This breaks hub strangulation by allowing high-doubt units to seek
        diverse information sources instead of being trapped in consensus echo chambers.

        Only operates on explicit adjacency matrices (not stencil grids).
        """
        if self.use_stencil or self.adjacency_matrix is None:
            return

        v = self.model.v
        u = self.model.u
        adj = self.adjacency_matrix
        rewired = False

        # Decrement cooldown timers
        self._rewire_timers = np.maximum(self._rewire_timers - 1, 0)

        # Find eligible units: high doubt AND cooldown expired
        eligible = (u > self.rewire_threshold) & (self._rewire_timers == 0)
        eligible_ids = np.where(eligible)[0]

        for i in eligible_ids:
            # Find neighbors of i (sparse-aware)
            if self._is_sparse:
                neighbors = adj.getrow(i).indices.copy()
            else:
                neighbors = np.where(adj[i] > 0)[0]
            if len(neighbors) <= 1:
                continue  # Don't rewire if only 1 neighbor (would disconnect)

            # Find the MOST CONSENSUAL neighbor (smallest |v_i - v_j|)
            # but exclude neighbors whose degree is ≤ 1 (removing them would disconnect)
            v_diffs = np.abs(v[i] - v[neighbors])
            sorted_idx = np.argsort(v_diffs)
            j = None
            for idx in sorted_idx:
                candidate = neighbors[idx]
                if (adj.getrow(candidate).nnz if self._is_sparse else np.sum(adj[candidate] > 0)) > 1:  # candidate keeps ≥1 edge after removal
                    j = candidate
                    break
            if j is None:
                continue  # All neighbors have degree 1 — can't safely rewire

            # Find non-neighbors (excluding self and current neighbors)
            if self._is_sparse:
                neighbor_set = set(adj.getrow(i).indices)
                neighbor_set.add(i)
                non_neighbors = np.array([nn for nn in range(self.N) if nn not in neighbor_set])
            else:
                non_neighbors = np.where(adj[i] == 0)[0]
                non_neighbors = non_neighbors[non_neighbors != i]
            if len(non_neighbors) == 0:
                continue  # Fully connected, can't rewire

            # Pick a random non-neighbor
            k = self.rng.choice(non_neighbors)

            # Perform rewire: disconnect i↔j, connect i↔k (symmetric)
            if self._is_sparse:
                adj_lil = adj.tolil()
                adj_lil[i, j] = 0; adj_lil[j, i] = 0
                adj_lil[i, k] = 1; adj_lil[k, i] = 1
                self.adjacency_matrix = adj_lil.tocsr()
                adj = self.adjacency_matrix
            else:
                adj[i, j] = 0; adj[j, i] = 0
                adj[i, k] = 1; adj[k, i] = 1

            # Incremental Laplacian update: O(1) instead of O(N²)
            self._update_laplacian_incremental(i, j, k)

            # Reset cooldown
            self._rewire_timers[i] = self.rewire_cooldown
            self.rewire_count += 1
            self._weights_dirty = True

    def get_spectral_gap(self) -> float:
        """
        Calculate the algebraic connectivity (Fiedler value).
        Measures how fast information spreads through the network.
        """
        if self.use_stencil:
            return 0.0

        # GUARD: Topological Consistency
        if self._is_sparse:
            diff = self.L - self.L.T
            if diff.nnz > 0 and np.max(np.abs(diff.data)) > 1e-10:
                raise ValueError
        elif not np.allclose(self.L, self.L.T, atol=1e-10):
            raise ValueError("Spectral gap requires symmetric Laplacian (undirected graph).")

        if self._is_sparse:
            try:
                vals = sparse_eigsh(self.L.astype(float), k=2, which='SM', return_eigenvectors=False)
                return np.sort(vals)[1]
            except Exception:
                return 0.0
        else:
            from scipy.linalg import eigh
            vals = eigh(self.L, eigvals_only=True)
            return vals[1] if len(vals) > 1 else 0.0

    def _calculate_laplacian_stencil(self, v):
        """
        Standard Discrete 2D Laplacian using 5-point stencil.
        L[i,j] = v[i+1,j] + v[i-1,j] + v[i,j+1] + v[i,j-1] - 4*v[i,j]

        Boundary conditions:
          - 'periodic': Toroidal wrap (default). All units have exactly 4 neighbors.
          - 'neumann': Zero-flux (dv/dn = 0). Border neighbors are reflected copies.
        """
        s = self.size
        v_grid = v.reshape((s, s))

        if self.boundary == 'periodic':
            output = (np.roll(v_grid, 1, axis=0) + np.roll(v_grid, -1, axis=0) +
                      np.roll(v_grid, 1, axis=1) + np.roll(v_grid, -1, axis=1) -
                      4 * v_grid)
        elif self.boundary == 'neumann':
            padded = np.pad(v_grid, 1, mode='edge')
            output = (padded[0:-2, 1:-1] + padded[2:, 1:-1] +
                      padded[1:-1, 0:-2] + padded[1:-1, 2:] -
                      4 * v_grid)
        else:
            raise ValueError(f"Unknown boundary condition '{self.boundary}'. Use 'periodic' or 'neumann'.")

        return output.flatten()

    def step(self, I_stimulus: float = 0.0):
        """Perform one simulation step with optional doubt-driven rewiring."""
        # V4: Topological adaptation before dynamics
        self._doubt_driven_rewire()

        # Recompute weights if topology changed (rewiring)
        if self._weights_dirty:
            self._rebuild_laplacian()
            self._compute_coupling_weights()

        if self.use_stencil:
            l_v = self._calculate_laplacian_stencil(self.v)
        else:
            l_v = -(self.L @ self.v)

        # Apply per-node coupling normalization (LIMIT-02 fix)
        if self.coupling_norm != 'uniform':
            D = self.model.cfg['coupling']['D']
            uniform_D_eff = D / np.sqrt(self.N)
            scale_factors = (self.node_weights * D) / uniform_D_eff
            l_v = l_v * scale_factors

        self.model.step(I_stimulus, l_v)

    @property
    def v(self): return self.model.v

    def calculate_entropy(self, **kwargs): return self.model.calculate_entropy(**kwargs)

    def get_state_distribution(self):
        states = self.model.get_states()
        counts = np.bincount(states, minlength=6)[1:]
        return {f"bin_{i}": int(c) for i, c in enumerate(counts)}
