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
        if self._is_sparse:
            self._rebuild_laplacian()
            return
        self.L[i, i] -= 1
        self.L[j, j] -= 1
        self.L[i, j] += 1
        self.L[j, i] += 1
        self.L[i, i] += 1
        self.L[k, k] += 1
        self.L[i, k] -= 1
        self.L[k, i] -= 1

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

            if self._is_sparse:
                neighbor_set = set(adj_lil.rows[i])
                neighbor_set.add(i)
                non_neighbors = np.array([nn for nn in range(self.N) if nn not in neighbor_set])
            else:
                non_neighbors = np.where(adj[i] == 0)[0]
                non_neighbors = non_neighbors[non_neighbors != i]
            
            if len(non_neighbors) == 0:
                continue

            k = self.rng.choice(non_neighbors)

            if self._is_sparse:
                adj_lil[i, j] = 0; adj_lil[j, i] = 0
                adj_lil[i, k] = 1; adj_lil[k, i] = 1
            else:
                adj[i, j] = 0; adj[j, i] = 0
                adj[i, k] = 1; adj[k, i] = 1

            self._update_laplacian_incremental(i, j, k)
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

    def step(self, I_stimulus: float = 0.0, sigma_v_vec=None):
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
        self.model.step(I_stimulus, l_v, sigma_v_vec=sigma_v_vec)

    @property
    def v(self): return self.model.v

    def calculate_entropy(self, **kwargs): return self.model.calculate_entropy(**kwargs)

    def get_state_distribution(self):
        from mem4ristor.metrics import get_cognitive_states
        states = get_cognitive_states(self.model.v)
        counts = np.bincount(states, minlength=6)[1:]
        return {f"bin_{i}": int(c) for i, c in enumerate(counts)}
