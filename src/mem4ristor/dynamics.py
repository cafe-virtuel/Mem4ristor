import os
import numpy as np
import yaml
from typing import Dict, Optional
from scipy.integrate import solve_ivp
from mem4ristor.metrics import calculate_cognitive_entropy, calculate_continuous_entropy, get_cognitive_states

# MKL Determinism Fix (v2.9.1)
os.environ['NUMPY_MKL_CBWR'] = 'COMPATIBLE'

class Mem4ristorV3:
    """
    Canonical Implementation of Mem4ristor v3.0.0 (with v4 adaptive extensions).
    Refactored to separate Dynamics from Topology and Metrics.
    """
    def __init__(self, config: Optional[Dict] = None, seed: int = 42):
        default_cfg = {
            'dynamics': {
                'a': 0.7, 'b': 0.8, 'epsilon': 0.08, 'alpha': 0.15,
                'v_cubic_divisor': 5.0, 'dt': 0.05,
                'lambda_intrinsic': 0.05, 'tau_plasticity': 1000, 'w_saturation': 2.0
            },
            'coupling': {'D': 0.15, 'heretic_ratio': 0.15, 'uniform_placement': True,
                         'dynamic_heretics': {'enabled': False, 'u_threshold': 0.8, 'steps_required': 100}},
            'doubt': {'epsilon_u': 0.02, 'k_u': 1.0, 'sigma_baseline': 0.05, 'u_clamp': [0.0, 1.0], 'tau_u': 10.0,
                      'alpha_surprise': 2.0, 'surprise_cap': 5.0},
            'noise': {'sigma_v': 0.05, 'use_rtn': False, 'rtn_amplitude': 0.1, 'rtn_p_flip': 0.01},
            'hysteresis': {'enabled': True, 'theta_low': 0.35, 'theta_high': 0.65,
                           'fatigue_rate': 0.0, 'base_hysteresis': 0.15}
        }

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

        self._validate_config()

        self.lambda_intrinsic = self.cfg['dynamics'].get('lambda_intrinsic', 0.05)
        self.tau_plasticity = self.cfg['dynamics'].get('tau_plasticity', 1000)
        self.w_saturation = self.cfg['dynamics'].get('w_saturation', 2.0)

        self.sigmoid_steepness = np.pi
        # KIMI FIX: social_leakage matched to paper
        self.social_leakage = 0.01

        self.N = 100
        self._initialize_params()

    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        result = base.copy()
        for key, value in update.items():
            if key in result:
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
        if self.cfg['dynamics']['v_cubic_divisor'] <= 1e-9:
            raise ValueError("Configuration Error: 'v_cubic_divisor' must be > 0.")
        if self.cfg['doubt']['tau_u'] <= 1e-9:
            raise ValueError("Configuration Error: 'tau_u' must be > 0.")
        if self.dt <= 0:
            raise ValueError("Configuration Error: 'dt' must be positive.")
        D = self.cfg['coupling'].get('D', 0.15)
        if not np.isfinite(D):
            raise ValueError("Configuration Error: 'D' must be finite.")
        hr = self.cfg['coupling'].get('heretic_ratio', 0.15)
        if not (0.0 <= hr <= 1.0):
            raise ValueError(f"Configuration Error: 'heretic_ratio' must be in [0, 1], got {hr}")
        # Guard restored from core_backup_pre_v5.py (Manus audit 2026-04-19)
        if self.cfg['noise'].get('use_rtn', False):
            p_flip = self.cfg['noise'].get('rtn_p_flip', 0.01)
            if not (0.0 <= p_flip <= 1.0):
                raise ValueError(f"Configuration Error: 'rtn_p_flip' must be in [0, 1], got {p_flip}")

    def _initialize_params(self, N=100, cold_start=False):
        if N <= 0 or N > 10_000_000:
            raise ValueError(f"Network size N={N} invalid.")
        self.N = N
        if cold_start:
            self.v = np.zeros(self.N)
            self.w = np.zeros(self.N)
        else:
            self.v = self.rng.uniform(-1.5, 1.5, self.N)
            self.w = self.rng.uniform(0.0, 1.0, self.N)

        self.u = np.full(self.N, self.cfg['doubt']['sigma_baseline'])

        hr = self.cfg['coupling'].get('heretic_ratio', 0.15)
        if hr <= 0:
            self.heretic_mask = np.zeros(self.N, dtype=bool)
        elif self.cfg['coupling'].get('uniform_placement', True):
            step = max(int(1.0 / hr), 1)
            heretic_ids = []
            for i in range(0, self.N, step):
                if len(heretic_ids) < int(self.N * hr):
                    block_end = min(i + step, self.N)
                    heretic_ids.append(self.rng.randint(i, block_end))
            self.heretic_mask = np.zeros(self.N, dtype=bool)
            self.heretic_mask[heretic_ids] = True
        else:
            self.heretic_mask = self.rng.rand(self.N) < hr

        self.D_eff = self.cfg['coupling']['D'] / np.sqrt(self.N)
        self.mode_state = np.zeros(self.N, dtype=bool)
        self.time_in_state = np.zeros(self.N, dtype=float)

        # V4: dynamic heretics — counts consecutive steps where u_i >= u_threshold
        self.heretic_counter = np.zeros(self.N, dtype=int)
        self.dynamic_heretic_count = 0

    def _update_hysteresis(self):
        hyst = self.cfg['hysteresis']
        if not hyst.get('enabled', False):
            self.mode_state = self.u > 0.5
            return
        fatigue_shift = hyst.get('fatigue_rate', 0.0) * self.time_in_state
        eff_theta_low = np.minimum(hyst['theta_low'] + fatigue_shift, 0.5)
        eff_theta_high = np.maximum(hyst['theta_high'] - fatigue_shift, 0.5)
        switch_to_fou = (~self.mode_state) & (self.u >= eff_theta_high)
        switch_to_sage = self.mode_state & (self.u < eff_theta_low)
        switched = switch_to_fou | switch_to_sage
        self.mode_state[switch_to_fou] = True
        self.mode_state[switch_to_sage] = False
        self.time_in_state[switched] = 0.0
        self.time_in_state[~switched] += self.dt

    def step(self, I_stimulus: float = 0.0, coupling_input: Optional[np.ndarray] = None,
             sigma_v_vec: Optional[np.ndarray] = None,
             sigma_social_override: Optional[np.ndarray] = None) -> None:
        # sigma_social_override: if provided, replaces |laplacian_v| in the u dynamics
        # (du equation only). Coupling and plasticity still use the real laplacian_v.
        # Used by ablation experiments (p2_sigma_social_ablation.py).
        # GUARD: Deterministic Input (restored from core_backup_pre_v5.py)
        if hasattr(I_stimulus, '__float__') and not isinstance(I_stimulus, (int, float, np.number, np.ndarray)):
            raise TypeError("Stimulus must be a numeric constant.")
        if isinstance(I_stimulus, (dict, set, list, tuple, str)):
            # Reject container types explicitly (list/tuple could match np.number in future)
            if not isinstance(I_stimulus, (list, tuple)):
                raise TypeError(f"Stimulus type {type(I_stimulus).__name__} not supported.")

        # @DOUBT — Correction silencieuse : si v/w/u contiennent NaN/Inf, on remet à zéro sans erreur.
        # Si tu vois des résultats incohérents, appelle health_check() pour détecter ce cas.
        if np.any(~np.isfinite(self.v)): self.v = np.nan_to_num(self.v, nan=0.0, posinf=0.0, neginf=0.0)
        if np.any(~np.isfinite(self.w)): self.w = np.nan_to_num(self.w, nan=0.0, posinf=0.0, neginf=0.0)
        if np.any(~np.isfinite(self.u)): self.u = np.nan_to_num(self.u, nan=0.5, posinf=0.5, neginf=0.5)

        # GUARD: Coupling Input Sanitization (restored from core_backup_pre_v5.py)
        if coupling_input is not None:
            try:
                coupling_input = np.array(coupling_input, dtype=float)
            except (ValueError, TypeError, AttributeError):
                raise ValueError(f"Invalid coupling input: {coupling_input!r}. Must be numeric.")

        if coupling_input is None:
            laplacian_v = np.zeros(self.N)
        elif coupling_input.ndim == 2:
            laplacian_v = coupling_input @ self.v - self.v
        elif coupling_input.ndim == 1:
            laplacian_v = coupling_input
        else:
            # 0-d scalar coupling input — treat as uniform scalar on zero Laplacian
            laplacian_v = np.full(self.N, float(coupling_input))

        if np.any(~np.isfinite(laplacian_v)):
            laplacian_v = np.nan_to_num(laplacian_v, nan=0.0, posinf=1.0, neginf=-1.0)

        sigma_social = np.abs(laplacian_v)
        # Override sigma_social for u dynamics only (ablation experiment hook)
        sigma_social_for_u = sigma_social_override if sigma_social_override is not None else sigma_social
        # Euler-Maruyama scaling : le bruit thermique obéit au calcul d'Itô et s'accumule avec sqrt(dt).
        # On divise par sqrt(dt) ici car dv sera multiplié par dt à la fin (eta * dt / sqrt(dt) = eta * sqrt(dt)).
        if sigma_v_vec is not None:
            eta = self.rng.normal(0, 1, self.N) * sigma_v_vec / np.sqrt(self.dt)
        else:
            eta = self.rng.normal(0, self.cfg['noise'].get('sigma_v', 0.05), self.N) / np.sqrt(self.dt)

        if self.cfg['noise'].get('use_rtn', False):
            rtn_amp = self.cfg['noise'].get('rtn_amplitude', 0.1)
            p_flip = self.cfg['noise'].get('rtn_p_flip', 0.01)
            eta += (self.rng.rand(self.N) < p_flip).astype(float) * rtn_amp * self.rng.choice([-1, 1], size=self.N)

        u_filter = np.tanh(self.sigmoid_steepness * (0.5 - self.u)) + self.social_leakage
        I_coup = self.D_eff * u_filter * laplacian_v

        # V5: Couplage non-local par similarite de doute
        # poids_ij = exp(-(u_i - u_j)^2 / sigma_u^2), normalises par ligne
        # I_virtual_i = D_meta * (moyenne_ponderee_v_voisins_similaires - v_i)
        nlc = self.cfg.get('nonlocal_coupling', {})
        if nlc.get('enabled', False):
            D_meta  = float(nlc.get('D_meta',  0.05))
            sigma_u = float(nlc.get('sigma_u', 0.10))
            u_diff2 = (self.u[:, None] - self.u[None, :]) ** 2   # (N, N)
            W = np.exp(-u_diff2 / (sigma_u ** 2))
            np.fill_diagonal(W, 0.0)
            row_sums = W.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums < 1e-10, 1.0, row_sums)
            W_norm = W / row_sums                                  # chaque ligne somme a 1
            I_virtual = D_meta * (W_norm @ self.v - self.v)
        else:
            I_virtual = 0.0

        # V5: Compartimentalisation Dynamique (Sous-Personnalites)
        # Chaque noeud est assigne a un groupe selon son rang de doute u.
        # I_comp tire chaque noeud vers la moyenne de son groupe (attraction intra),
        # et optionnellement le repousse de la moyenne des autres groupes (repulsion inter).
        comp = self.cfg.get('compartments', {})
        if comp.get('enabled', False):
            K     = int(comp.get('K', 2))
            gamma = float(comp.get('gamma', 0.10))
            mode  = comp.get('mode', 'attraction')

            # Rang de chaque noeud selon u (0 = plus certain, N-1 = plus douteux)
            u_ranks = np.argsort(np.argsort(self.u))          # rang ordinal stable
            labels  = np.minimum((u_ranks * K) // self.N, K - 1)  # label dans [0..K-1]

            I_comp = np.zeros(self.N)
            for k in range(K):
                mask_k = (labels == k)
                n_k = mask_k.sum()
                if n_k < 2:
                    continue
                v_mean_k = self.v[mask_k].mean()
                # Attraction intra-groupe : tire v_i vers la moyenne du groupe
                I_comp[mask_k] += gamma * (v_mean_k - self.v[mask_k])
                if mode == 'full':
                    mask_other = ~mask_k
                    if mask_other.sum() > 0:
                        v_mean_other = self.v[mask_other].mean()
                        # Repulsion inter-groupe : eloigne v_i de la moyenne des autres
                        I_comp[mask_k] -= gamma * (v_mean_other - self.v[mask_k])
        else:
            I_comp = 0.0

        # GUARD: Stimulus Sanitization + Size Validation (restored from core_backup_pre_v5.py)
        try:
            stim_arr = np.array(I_stimulus, dtype=float)
        except (ValueError, TypeError, AttributeError):
            raise ValueError(f"Invalid stimulus input: {I_stimulus!r}. Must be numeric.")
        if stim_arr.ndim == 0:
            I_eff = np.full(self.N, float(stim_arr))
        else:
            I_eff = stim_arr.flatten()
            if I_eff.size != self.N:
                raise ValueError(
                    f"Stimulus vector size {I_eff.size} must match network size {self.N}")
        if np.any(~np.isfinite(I_eff)):
            I_eff = np.nan_to_num(I_eff, nan=0.0, posinf=100.0, neginf=-100.0)
        I_eff = np.clip(I_eff, -100.0, 100.0)
        I_eff[self.heretic_mask] *= -1.0
        I_ext = I_eff + I_coup + I_virtual + I_comp

        if self.cfg.get('hysteresis', {}).get('enabled', False):
            self._update_hysteresis()
            innovation_mask = self.mode_state.astype(float)
        else:
            innovation_mask = (self.u > 0.5).astype(float)

        plasticity_drive = self.lambda_intrinsic * sigma_social * innovation_mask
        w_ratio = self.w / self.w_saturation
        saturation_factor = np.clip(1.0 - (w_ratio**2), 0.0, 1.0)
        dw_learning = (plasticity_drive * saturation_factor) - (self.w / self.tau_plasticity)

        dv = (self.v - (self.v**3) / self.cfg['dynamics']['v_cubic_divisor'] - self.w + I_ext -
              self.cfg['dynamics']['alpha'] * np.tanh(self.v) + eta)

        # V5: Plasticite metacognitive — epsilon per-noeud module par u
        # Noeud certain (u~0) -> epsilon eleve -> impulsif
        # Noeud douteux (u~1) -> epsilon reduit -> prudent
        meta = self.cfg.get('metacognitive', {})
        if meta.get('enabled', False):
            alpha_meta = meta.get('alpha_meta', 0.5)
            eps_min    = meta.get('epsilon_min', 0.01)
            epsilon_i  = self.cfg['dynamics']['epsilon'] * (1.0 + alpha_meta * (0.5 - self.u))
            epsilon_i  = np.maximum(epsilon_i, eps_min)
        else:
            epsilon_i  = self.cfg['dynamics']['epsilon']

        dw = epsilon_i * (self.v + self.cfg['dynamics']['a'] - self.cfg['dynamics']['b'] * self.w)

        sigma_social_safe = np.clip(sigma_social_for_u, 0.0, 100.0)
        alpha_s = self.cfg['doubt'].get('alpha_surprise', 2.0)
        epsilon_u_adaptive = self.cfg['doubt']['epsilon_u'] * np.clip(
            1.0 + alpha_s * sigma_social_safe, 1.0, self.cfg['doubt'].get('surprise_cap', 5.0)
        )
        du = (epsilon_u_adaptive * (self.cfg['doubt']['k_u'] * sigma_social_for_u +
              self.cfg['doubt']['sigma_baseline'] - self.u)) / self.cfg['doubt']['tau_u']

        self.v += dv * self.dt
        self.w += (dw + dw_learning) * self.dt
        self.u += du * self.dt

        # Alarme de divergence : au lieu de cacher l'explosion, on l'annonce.
        if np.any(np.abs(self.v) > 100.0) or np.any(np.abs(self.w) > 100.0):
            raise OverflowError("Boom ! La simulation a déraillé et a dépassé ±100. Le filet de sécurité est retiré.")

        self.u = np.clip(self.u, *self.cfg['doubt']['u_clamp'])

        # V5: Autorégulation Topologique (ART) — Kirchhoff passif
        # Un noeud rigide (u faible) bloque le courant -> tension aux voisins
        # -> retroaction qui augmente u des voisins. Emerge du cablage, pas d'algo.
        # Requires self._adj_matrix set by Mem4Network.step() before calling model.step().
        art = self.cfg.get('topological_regulation', {})
        if art.get('enabled', False):
            u_min_art = float(art.get('u_min', 0.05))
            adj = getattr(self, '_adj_matrix', None)

            if adj is not None:
                rigid_thr = float(art.get('rigid_threshold', 0.7))
                mode_art  = art.get('mode', 'soft')
                rigidity  = 1.0 - self.u  # 1 = rigide (u faible), 0 = souple

                # Convertir sparse -> dense une seule fois si necessaire
                if hasattr(adj, 'toarray'):
                    adj_dense = adj.toarray().astype(float)
                else:
                    adj_dense = np.asarray(adj, dtype=float)

                degree = adj_dense.sum(axis=1)
                degree = np.where(degree < 1.0, 1.0, degree)  # evite div/0

                if mode_art == 'soft':
                    # Gemini : pression = rigidit?? MOYENNE des voisins
                    mean_rig = (adj_dense @ rigidity) / degree
                    pressure = mean_rig > rigid_thr
                    factor   = float(art.get('alpha_art_soft', 0.15))
                    self.u[pressure] = np.minimum(
                        self.u[pressure] * (1.0 + factor), 1.0
                    )
                elif mode_art == 'hard':
                    # Grok : proportion de voisins rigides (retroaction non-lineaire)
                    n_rigid = adj_dense @ (rigidity > rigid_thr).astype(float)
                    ratio   = n_rigid / degree
                    pressure = ratio > 0
                    factor   = float(art.get('alpha_art_hard', 0.25))
                    self.u[pressure] = np.minimum(
                        self.u[pressure] * (1.0 + factor * ratio[pressure]), 1.0
                    )

            # Plancher fixe — toujours actif quand ART enabled (meme sans matrice adj)
            self.u = np.maximum(self.u, u_min_art)

        # V4: dynamic heretics — bascule irréversible quand u_i >= u_threshold pendant steps_required steps
        dyn = self.cfg['coupling'].get('dynamic_heretics', {})
        if dyn.get('enabled', False):
            threshold = dyn.get('u_threshold', 0.8)
            steps_req = dyn.get('steps_required', 100)
            above = self.u >= threshold
            self.heretic_counter[above] += 1
            self.heretic_counter[~above] = 0
            newly_heretic = (self.heretic_counter >= steps_req) & (~self.heretic_mask)
            if np.any(newly_heretic):
                self.heretic_mask |= newly_heretic
                self.dynamic_heretic_count += int(np.sum(newly_heretic))

    def solve_rk45(self, t_span, I_stimulus=0.0, adj_matrix=None):
        """
        Adaptive RK45 Integrator.
        
        WARNING: Only numerically valid for Deterministic systems (sigma_v = 0).
        For Stochastic Differential Equations (SDEs), use the 'step()' method (Euler-Maruyama).
        """
        if self.cfg['noise'].get('sigma_v', 0.0) > 0:
            import warnings
            warnings.warn("RK45 detected with sigma_v > 0. Standard adaptive integrators "
                          "are mathematically inconsistent with stochastic noise. "
                          "Results will be non-reproducible. Use sigma_v=0 or use .step().", 
                          RuntimeWarning)
                          
        duration = t_span[1] - t_span[0]
        max_step = min(0.1, duration / 10.0) if duration > 0 else 0.1

        def combined_dynamics(t, y):
            v, w, u = y[:self.N], y[self.N:2*self.N], y[2*self.N:]
            laplacian_v = np.zeros(self.N) if adj_matrix is None else adj_matrix @ v - v
            sigma_social = np.abs(laplacian_v)
            u_filter = np.tanh(self.sigmoid_steepness * (0.5 - u)) + self.social_leakage
            
            I_eff = np.full(self.N, float(I_stimulus))
            I_eff[self.heretic_mask] *= -1.0
            I_ext = I_eff + self.D_eff * u_filter * laplacian_v
            
            eta = self.rng.normal(0, self.cfg['noise'].get('sigma_v', 0.05), self.N)
            dv = v - (v**3)/self.cfg['dynamics']['v_cubic_divisor'] - w + I_ext - self.cfg['dynamics']['alpha']*np.tanh(v) + eta
            dw_fhn = self.cfg['dynamics']['epsilon'] * (v + self.cfg['dynamics']['a'] - self.cfg['dynamics']['b'] * w)
            
            p_drive = self.lambda_intrinsic * sigma_social * (u > 0.5).astype(float)
            w_ratio = w / self.w_saturation
            sat_fact = np.clip(1.0 - (w_ratio**2), 0.0, 1.0)
            dw_learn = (p_drive * sat_fact) - (w / self.tau_plasticity)
            dw = dw_fhn + dw_learn
            
            e_adapt = self.cfg['doubt']['epsilon_u'] * np.clip(1.0 + self.cfg['doubt'].get('alpha_surprise', 2.0)*sigma_social, 1.0, 5.0)
            du = (e_adapt * (self.cfg['doubt']['k_u']*sigma_social + self.cfg['doubt']['sigma_baseline'] - u)) / self.cfg['doubt']['tau_u']
            return np.concatenate([dv, dw, du])

        y0 = np.concatenate([self.v, self.w, self.u])
        sol = solve_ivp(combined_dynamics, t_span, y0, method='RK45', rtol=1e-6, max_step=max_step)
        y_final = sol.y[:, -1]
        self.v = y_final[:self.N]
        self.w = y_final[self.N:2*self.N]
        self.u = np.clip(y_final[2*self.N:], *self.cfg['doubt']['u_clamp'])
        return sol

    # API bindings to metrics.py to avoid breaking existing scripts
    def get_states(self) -> np.ndarray:
        return get_cognitive_states(self.v)

    def calculate_entropy(self, bins=None, use_cognitive_bins=False) -> float:
        """
        By default uses the new Continuous Entropy.
        If use_cognitive_bins=True is forced, it will fall back to exact paper ±0.4/1.2 boundaries.
        """
        if use_cognitive_bins:
            return calculate_cognitive_entropy(self.v)
        return calculate_continuous_entropy(self.v, bins=bins or 100)

Mem4ristorV2 = Mem4ristorV3
