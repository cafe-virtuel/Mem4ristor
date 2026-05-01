import numpy as np
import scipy.sparse as sp_sparse
from typing import Optional
from mem4ristor.dynamics import Mem4ristorV3

class Mem4Network:
    """
    High-level API for Mem4ristorV3 with formal Laplacian operators
    and doubt-driven topological rewiring (v4 extension).
    """
    def __init__(self, size: int = 10, heretic_ratio: float = 0.15, seed: int = 42,
                 adjacency_matrix: Optional[np.ndarray] = None, cold_start: bool = False,
                 boundary: str = 'periodic',
                 rewire_threshold: float = 0.8, rewire_cooldown: int = 50,
                 coupling_norm: str = 'uniform',
                 auto_sparse_threshold: int = 1000):
        self.rng = np.random.RandomState(seed)
        self.boundary = boundary
        self.rewire_threshold = rewire_threshold
        self.rewire_cooldown = rewire_cooldown
        self.rewire_count = 0
        self.coupling_norm = coupling_norm
        self._weights_dirty = False

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
        # OBSOLETE: Replaced by _update_laplacian_incremental_swap
        pass

    def _update_laplacian_incremental_swap(self, i, j, k, l):
        if self._is_sparse:
            self._rebuild_laplacian()
            return
        # L = D - A. Degrees are perfectly preserved, so D is unchanged!
        # Break (i,j) and (k,l): A goes 1->0, so L goes -1->0 (+1)
        self.L[i, j] += 1
        self.L[j, i] += 1
        self.L[k, l] += 1
        self.L[l, k] += 1
        # Make (i,k) and (j,l): A goes 0->1, so L goes 0->-1 (-1)
        self.L[i, k] -= 1
        self.L[k, i] -= 1
        self.L[j, l] -= 1
        self.L[l, j] -= 1

    def _compute_coupling_weights(self):
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
            elif self.coupling_norm == 'spectral':
                raw_weights = 1.0 / self._eigenvector_centrality()
            else:
                raw_weights = np.ones(self.N)

            target_mean = 1.0 / np.sqrt(self.N)
            self.node_weights = raw_weights * target_mean / np.mean(raw_weights)
        else:
            self.node_weights = np.ones(self.N) / np.sqrt(self.N)
        self._weights_dirty = False

    def _eigenvector_centrality(self, max_iter: int = 200, tol: float = 1e-8) -> np.ndarray:
        A = self.adjacency_matrix
        c = np.ones(self.N) / np.sqrt(self.N)
        for _ in range(max_iter):
            c_new = A @ c
            norm = np.linalg.norm(c_new)
            if norm < 1e-15:
                deg = np.array(A.sum(axis=1)).flatten() if self._is_sparse else A.sum(axis=1)
                c_new = np.maximum(deg, 1.0)
                norm = np.linalg.norm(c_new)
            c_new = c_new / norm
            if np.linalg.norm(c_new - c) < tol:
                c = c_new
                break
            c = c_new
        if np.mean(c) < 0:
            c = -c
        c = c * (self.N / max(c.sum(), 1e-12))
        return np.maximum(c, 1e-6)

    def _doubt_driven_rewire(self):
        if self.use_stencil or self.adjacency_matrix is None:
            return

        v = self.model.v
        u = self.model.u
        adj = self.adjacency_matrix

        self._rewire_timers = np.maximum(self._rewire_timers - 1, 0)
        eligible = (u > self.rewire_threshold) & (self._rewire_timers == 0)
        eligible_ids = np.where(eligible)[0]
        
        if len(eligible_ids) == 0:
            return

        # Optimization O(N): Convert to LIL ONCE before looping
        adj_lil = None
        if self._is_sparse:
            adj_lil = adj.tolil()

        for i in eligible_ids:
            if self._is_sparse:
                neighbors = adj_lil.rows[i].copy()
            else:
                neighbors = np.where(adj[i] > 0)[0]
            
            if len(neighbors) <= 1:
                continue

            v_diffs = np.abs(v[i] - v[neighbors])
            sorted_idx = np.argsort(v_diffs)
            j = None
            for idx in sorted_idx:
                candidate = neighbors[idx]
                candidate_deg = len(adj_lil.rows[candidate]) if self._is_sparse else np.sum(adj[candidate] > 0)
                if candidate_deg > 1:
                    j = candidate
                    break
            
            if j is None:
                continue

            # FIX: Degree-Preserving Edge Swap (Reviewer 2 Vague 2)
            # Find a random edge (k, l) to swap with (i, j)
            if self._is_sparse:
                rows, cols = adj_lil.nonzero()
            else:
                rows, cols = np.where(adj > 0)
                
            # Filter valid edges (avoid self-loops and existing connections)
            valid_mask = (rows != i) & (rows != j) & (cols != i) & (cols != j)
            valid_rows = rows[valid_mask]
            valid_cols = cols[valid_mask]
            
            if len(valid_rows) == 0:
                continue
                
            # Pick a random edge
            edge_idx = self.rng.randint(len(valid_rows))
            k = valid_rows[edge_idx]
            l = valid_cols[edge_idx]
            
            # Check if new edges (i, k) or (j, l) already exist
            if self._is_sparse:
                if k in adj_lil.rows[i] or l in adj_lil.rows[j]:
                    continue
            else:
                if adj[i, k] > 0 or adj[j, l] > 0:
                    continue
            
            # Execute Swap
            if self._is_sparse:
                adj_lil[i, j] = 0; adj_lil[j, i] = 0
                adj_lil[k, l] = 0; adj_lil[l, k] = 0
                adj_lil[i, k] = 1; adj_lil[k, i] = 1
                adj_lil[j, l] = 1; adj_lil[l, j] = 1
            else:
                adj[i, j] = 0; adj[j, i] = 0
                adj[k, l] = 0; adj[l, k] = 0
                adj[i, k] = 1; adj[k, i] = 1
                adj[j, l] = 1; adj[l, j] = 1

            self._update_laplacian_incremental_swap(i, j, k, l)
            self._rewire_timers[i] = self.rewire_cooldown
            self.rewire_count += 1
            self._weights_dirty = True

        # Convert back ONCE after looping
        if self._is_sparse and adj_lil is not None:
            self.adjacency_matrix = adj_lil.tocsr()

    def get_spectral_gap(self) -> float:
        if self.use_stencil:
            return 0.0
        if self._is_sparse:
            try:
                from scipy.sparse.linalg import eigsh as sparse_eigsh
                vals = sparse_eigsh(self.L.astype(float), k=2, which='SM', return_eigenvectors=False)
                return np.sort(vals)[1]
            except Exception:
                return 0.0
        else:
            from scipy.linalg import eigh
            vals = eigh(self.L, eigvals_only=True)
            return vals[1] if len(vals) > 1 else 0.0

    def get_effective_spectral_gap(self) -> float:
        """
        Calculates the Fiedler value (2nd smallest eigenvalue) of the 
        EFFECTIVE Laplacian L_eff = Diag(scale_factors) * L.
        This accounts for the coupling normalization (e.g. 'degree_linear').
        """
        if self.use_stencil or self.L is None:
            return 0.0
        
        # Calculate scale_factors just like in step()
        D_global = self.model.cfg['coupling']['D']
        uniform_D_eff = D_global / np.sqrt(self.N)
        scale_factors = (self.node_weights * D_global) / uniform_D_eff
        
        if self._is_sparse:
            # L_eff = diags(scale_factors) @ L
            L_eff = sp_sparse.diags(scale_factors) @ self.L
            try:
                from scipy.sparse.linalg import eigs as sparse_eigs
                # Use eigs instead of eigsh because L_eff might not be perfectly symmetric
                vals = sparse_eigs(L_eff.astype(float), k=2, which='SM', return_eigenvectors=False)
                return np.sort(np.real(vals))[1]
            except Exception:
                return 0.0
        else:
            L_eff = np.diag(scale_factors) @ self.L
            from scipy.linalg import eig
            vals = eig(L_eff, right=False)
            return np.sort(np.real(vals))[1]

    def _calculate_laplacian_stencil(self, v):
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

    def step(self, I_stimulus: float = 0.0, sigma_v_vec=None, sigma_social_override=None):
        self._doubt_driven_rewire()
        if self._weights_dirty:
            self._rebuild_laplacian()
            self._compute_coupling_weights()
        if self.use_stencil:
            l_v = self._calculate_laplacian_stencil(self.v)
        else:
            l_v = -(self.L @ self.v)
        if self.coupling_norm != 'uniform':
            D = self.model.cfg['coupling']['D']
            uniform_D_eff = D / np.sqrt(self.N)
            scale_factors = (self.node_weights * D) / uniform_D_eff
            l_v = l_v * scale_factors
        self.model.step(I_stimulus, l_v, sigma_v_vec=sigma_v_vec,
                        sigma_social_override=sigma_social_override)

    @property
    def v(self): return self.model.v

    def calculate_entropy(self, **kwargs): return self.model.calculate_entropy(**kwargs)

    def get_state_distribution(self):
        from mem4ristor.metrics import get_cognitive_states
        states = get_cognitive_states(self.model.v)
        counts = np.bincount(states, minlength=6)[1:]
        return {f"bin_{i}": int(c) for i, c in enumerate(counts)}

    def health_check(self) -> dict:
        """
        Point de contrôle de santé du réseau.

        Appelle cette méthode à tout moment pour détecter les défaillances silencieuses :
        zone morte, explosion de variables, doute saturé/gelé, matrice corrompue.

        Returns
        -------
        dict avec :
            'status'       : 'ok' | 'warning' | 'critical'
            'entropy_H'    : float — entropie actuelle
            'u_mean'       : float — doute moyen
            'v_max_abs'    : float — valeur absolue maximale de v
            'rewire_count' : int   — nombre de rewirings depuis le début
            'N'            : int   — taille du réseau
            'issues'       : list of (level, message) — liste des problèmes détectés

        Exemple d'usage
        ---------------
            rapport = reseau.health_check()
            if rapport['status'] != 'ok':
                print(rapport['issues'])
        """
        import warnings
        issues = []

        # --- 1. Intégrité des variables d'état ---
        v_nan = not np.all(np.isfinite(self.model.v))
        w_nan = not np.all(np.isfinite(self.model.w))
        u_nan = not np.all(np.isfinite(self.model.u))
        if v_nan:
            issues.append(('critical', 'v contient NaN/Inf — état du réseau corrompu'))
        if w_nan:
            issues.append(('critical', 'w contient NaN/Inf — variable de récupération corrompue'))
        if u_nan:
            issues.append(('critical', 'u contient NaN/Inf — variable de doute corrompue'))

        # --- 2. Détection d'explosion silencieuse (clipping) ---
        v_max_abs = float(np.max(np.abs(self.model.v))) if np.all(np.isfinite(self.model.v)) else float('inf')
        n_clipped = int(np.sum(np.abs(self.model.v) > 50))
        if n_clipped > 0:
            issues.append((
                'warning',
                f'{n_clipped}/{self.N} nœuds ont |v| > 50 — explosion possible, clipping silencieux actif'
            ))

        # --- 3. Zone morte (effondrement de l'entropie) ---
        try:
            H = float(self.model.calculate_entropy())
        except Exception:
            H = 0.0
            issues.append(('critical', 'Impossible de calculer l\'entropie — état critique'))

        if H < 0.1:
            issues.append((
                'critical',
                f'ZONE MORTE : entropie H={H:.4f} < 0.1 — le réseau a convergé vers le consensus'
            ))
        elif H < 0.5:
            issues.append((
                'warning',
                f'Entropie faible : H={H:.4f} — approche de la zone morte (seuil critique : 0.1)'
            ))

        # --- 4. Doute saturé ou gelé ---
        u_clamp = self.model.cfg['doubt']['u_clamp']
        u_mean = float(np.mean(self.model.u))
        n_at_max = int(np.sum(self.model.u >= u_clamp[1] - 1e-6))
        n_at_min = int(np.sum(self.model.u <= u_clamp[0] + 1e-6))
        if n_at_max > self.N * 0.5:
            issues.append((
                'warning',
                f'Doute saturé : {n_at_max}/{self.N} nœuds à u_max={u_clamp[1]} — tous les nœuds doutent au maximum'
            ))
        if n_at_min > self.N * 0.5:
            issues.append((
                'warning',
                f'Doute gelé : {n_at_min}/{self.N} nœuds à u_min={u_clamp[0]} — aucun doute actif dans le réseau'
            ))

        # --- 5. Intégrité de la matrice d'adjacence ---
        if self.adjacency_matrix is not None and not self.use_stencil:
            if self._is_sparse:
                has_negative = np.any(self.adjacency_matrix.data < 0)
                has_nan = not np.all(np.isfinite(self.adjacency_matrix.data))
            else:
                has_negative = np.any(self.adjacency_matrix < 0)
                has_nan = not np.all(np.isfinite(self.adjacency_matrix))
            if has_negative:
                issues.append(('critical', 'Matrice d\'adjacence contient des valeurs négatives — Laplacien invalide'))
            if has_nan:
                issues.append(('critical', 'Matrice d\'adjacence contient NaN/Inf — topologie corrompue'))

        # --- 6. Tempête de rewiring ---
        if self.rewire_count > self.N * 10:
            issues.append((
                'warning',
                f'Rewiring excessif : {self.rewire_count} rewirings pour {self.N} nœuds — topologie instable'
            ))

        # --- Statut global ---
        if any(level == 'critical' for level, _ in issues):
            status = 'critical'
        elif issues:
            status = 'warning'
        else:
            status = 'ok'

        # Émettre un warning Python visible dans les logs si problème détecté
        if status != 'ok':
            msg = f"[Mem4Network health_check] status={status.upper()} — " + \
                  " | ".join(msg for _, msg in issues)
            warnings.warn(msg, UserWarning, stacklevel=2)

        # --- V4: dynamic heretics status ---
        dyn_cfg = self.model.cfg['coupling'].get('dynamic_heretics', {})
        dyn_enabled = dyn_cfg.get('enabled', False)
        dynamic_heretic_count = self.model.dynamic_heretic_count if dyn_enabled else None
        total_heretics = int(np.sum(self.model.heretic_mask))

        return {
            'status':                status,
            'entropy_H':             H,
            'u_mean':                u_mean,
            'v_max_abs':             v_max_abs,
            'rewire_count':          self.rewire_count,
            'N':                     self.N,
            'issues':                issues,
            'total_heretics':        total_heretics,
            'dynamic_heretic_count': dynamic_heretic_count,
        }
