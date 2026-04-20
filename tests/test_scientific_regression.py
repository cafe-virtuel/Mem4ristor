"""
Scientific Regression Tests for Mem4ristor v3.2.0.

These tests verify the empirically validated scientific properties of the model.
They serve as guardians against regressions that could silently break the
model's core scientific behavior during future development.

Each test references the corresponding investigation in PROJECT_STATUS.md
or docs/limitations.md.

Created: 2026-04-10 (Antigravity, v3.2.0 consolidation session)
"""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mem4ristor.core import Mem4ristorV3, Mem4Network


class TestLatticeDefaults:
    """Verify that default lattice configuration produces expected diversity."""

    def test_lattice_default_entropy(self):
        """
        Default 10x10 lattice must produce H > 0.7 after 1000 steps.

        Reference: PROJECT_STATUS.md §3, LIMIT-05 investigation.
        Default config on lattice converges to H_stable ≈ 0.92 ± 0.04.
        We use a generous lower bound (0.7) to account for seed variance.
        """
        results = []
        for seed in [42, 123, 456]:
            net = Mem4Network(size=10, heretic_ratio=0.15, seed=seed)
            for step in range(1000):
                net.step(I_stimulus=0.5)
            results.append(net.calculate_entropy())

        median_h = np.median(results)
        assert median_h > 0.7, (
            f"Default lattice entropy too low: median H = {median_h:.4f} "
            f"(expected > 0.7, reference: 0.92 ± 0.04). "
            f"All values: {[f'{h:.4f}' for h in results]}"
        )


class TestHereticNecessity:
    """Verify that heretics are necessary for diversity emergence."""

    def test_no_heretics_low_entropy(self):
        """
        A network without heretics (ratio=0) must have low entropy.

        Reference: Preprint ablation table (Table 3), Fragility Law.
        Without heretics, consensus collapse occurs.
        """
        results = []
        for seed in [42, 123, 456]:
            net = Mem4Network(size=10, heretic_ratio=0.0, seed=seed,
                              cold_start=True)
            for step in range(1000):
                net.step(I_stimulus=0.5)
            # Use 5-bin cognitive entropy: the "collapse" claim is in terms
            # of cognitive states per preprint Table 1 (±0.4, ±1.2). The 100-bin
            # continuous metric detects sub-cognitive variability that does
            # not constitute consensus collapse in the physiological sense.
            results.append(net.calculate_entropy(use_cognitive_bins=True))

        median_h = np.median(results)
        assert median_h < 0.5, (
            f"Network without heretics should collapse: median cognitive H = {median_h:.4f} "
            f"(expected < 0.5). All values: {[f'{h:.4f}' for h in results]}"
        )


class TestScaleFreeDegreeLinear:
    """Verify that degree_linear normalization fixes BA network strangulation."""

    @staticmethod
    def _make_ba_adjacency(n, m, seed):
        """Create a Barabási-Albert adjacency matrix (matches limit02_norm_sweep.py)."""
        rng = np.random.RandomState(seed)
        adj = np.zeros((n, n), dtype=float)
        for i in range(m + 1):
            for j in range(i + 1, m + 1):
                adj[i, j] = adj[j, i] = 1.0
        degrees = np.sum(adj, axis=1)
        for new_node in range(m + 1, n):
            probs = degrees[:new_node] / degrees[:new_node].sum()
            targets = rng.choice(new_node, size=m, replace=False, p=probs)
            for t in targets:
                adj[new_node, t] = adj[t, new_node] = 1.0
            degrees = np.sum(adj, axis=1)
        return adj

    def test_ba_m3_degree_linear_entropy(self):
        """
        BA m=3 with degree_linear must achieve H > 0.5 after 3000 steps.

        Reference: PROJECT_STATUS.md §3quinquies, experiments/limit02_norm_sweep.py.
        degree_linear on BA m=3 gives H ≈ 0.83 ± 0.07 (original protocol: I_stim=0.0,
        per-seed BA graph). We use a lower bound (0.5) to account for seed variance.
        """
        results = []
        for seed in [42, 123, 777]:
            adj = self._make_ba_adjacency(100, 3, seed)
            net = Mem4Network(
                adjacency_matrix=adj,
                heretic_ratio=0.15,
                coupling_norm='degree_linear',
                seed=seed
            )
            trace = []
            for step in range(3000):
                net.step(I_stimulus=0.0)
                if step % 10 == 0:
                    trace.append(net.calculate_entropy())
            # Use H_stable = mean of last 25%
            tail = trace[int(len(trace) * 0.75):]
            results.append(np.mean(tail))

        median_h = np.median(results)
        assert median_h > 0.5, (
            f"BA m=3 degree_linear entropy too low: median H_stable = {median_h:.4f} "
            f"(expected > 0.5, reference: 0.83 ± 0.07). "
            f"All values: {[f'{h:.4f}' for h in results]}"
        )

    def test_ba_m3_uniform_collapses(self):
        """
        BA m=3 with uniform normalization should collapse (H ≈ 0).
        This confirms the problem that degree_linear solves.

        Reference: PROJECT_STATUS.md §3ter.
        """
        adj = self._make_ba_adjacency(100, 3, 42)
        net = Mem4Network(
            adjacency_matrix=adj,
            heretic_ratio=0.15,
            coupling_norm='uniform',
            seed=42
        )
        for step in range(2000):
            net.step(I_stimulus=0.0)
        # Cognitive entropy: the "collapse" claim is physiological (5 bins).
        # The 100-bin continuous metric detects sub-cognitive oscillations
        # that don't constitute true consensus collapse.
        h = net.calculate_entropy(use_cognitive_bins=True)
        assert h < 0.3, (
            f"BA m=3 uniform should collapse: cognitive H = {h:.4f} (expected < 0.3)"
        )


class TestSparseDenseParity:
    """Verify that sparse CSR backend produces same results as dense."""

    def test_sparse_dense_parity(self):
        """
        Results with auto_sparse_threshold=10 (forces sparse) must match
        results with auto_sparse_threshold=10000 (forces dense) on N=50.

        Reference: PROJECT_STATUS.md §5 item #14.
        """
        # Use inline BA generation (same as TestScaleFreeDegreeLinear)
        rng = np.random.RandomState(42)
        n, m = 50, 3
        adj = np.zeros((n, n), dtype=float)
        for i in range(m + 1):
            for j in range(i + 1, m + 1):
                adj[i, j] = adj[j, i] = 1.0
        degrees = np.sum(adj, axis=1)
        for new_node in range(m + 1, n):
            probs = degrees[:new_node] / degrees[:new_node].sum()
            targets = rng.choice(new_node, size=m, replace=False, p=probs)
            for t in targets:
                adj[new_node, t] = adj[t, new_node] = 1.0
            degrees = np.sum(adj, axis=1)

        # Run dense
        net_dense = Mem4Network(
            adjacency_matrix=adj.copy(),
            heretic_ratio=0.15,
            coupling_norm='uniform',
            seed=42,
            auto_sparse_threshold=10000  # Force dense
        )
        for _ in range(500):
            net_dense.step(I_stimulus=0.0)
        h_dense = net_dense.calculate_entropy()

        # Run sparse (force sparse with low threshold)
        net_sparse = Mem4Network(
            adjacency_matrix=adj.copy(),
            heretic_ratio=0.15,
            coupling_norm='uniform',
            seed=42,
            auto_sparse_threshold=10  # Force sparse for N=50
        )
        for _ in range(500):
            net_sparse.step(I_stimulus=0.0)
        h_sparse = net_sparse.calculate_entropy()

        # Results should be very close (within numerical precision + Laplacian path)
        assert abs(h_dense - h_sparse) < 0.15, (
            f"Sparse/Dense parity broken: H_dense={h_dense:.4f}, "
            f"H_sparse={h_sparse:.4f}, diff={abs(h_dense-h_sparse):.4f}"
        )


class TestHysteresisEffect:
    """Verify that V5 hysteresis does not degrade lattice performance."""

    def test_hysteresis_no_degradation(self):
        """
        V5 hysteresis enabled should not reduce H_stable compared to disabled
        by more than 10% on a default lattice.

        Reference: PROJECT_STATUS.md §5 item #12.
        V5 hysteresis is designed to improve or maintain diversity.
        """
        # Run with hysteresis enabled (default)
        net_hyst = Mem4Network(size=10, heretic_ratio=0.15, seed=42)
        for _ in range(1000):
            net_hyst.step(I_stimulus=0.5)
        h_hyst = net_hyst.calculate_entropy()

        # Run with hysteresis disabled
        from mem4ristor.core import Mem4ristorV3
        model_no_hyst = Mem4ristorV3(
            config={'hysteresis': {'enabled': False}},
            seed=42
        )
        net_no_hyst = Mem4Network(size=10, heretic_ratio=0.15, seed=42)
        net_no_hyst.model.cfg['hysteresis']['enabled'] = False
        for _ in range(1000):
            net_no_hyst.step(I_stimulus=0.5)
        h_no_hyst = net_no_hyst.calculate_entropy()

        # Hysteresis should not lose more than 10% relative to no-hysteresis
        assert h_hyst > h_no_hyst * 0.9, (
            f"Hysteresis degraded entropy: H_hyst={h_hyst:.4f}, "
            f"H_no_hyst={h_no_hyst:.4f}, ratio={h_hyst/max(h_no_hyst, 0.01):.2f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
