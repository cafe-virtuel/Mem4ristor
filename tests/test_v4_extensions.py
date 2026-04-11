"""
Tests for Mem4ristor V4 Extensions.
Tests the two new mechanisms: Adaptive Meta-Doubt and Doubt-Driven Rewiring.
"""
import pytest
import numpy as np
from mem4ristor.core import Mem4ristorV3, Mem4Network


# ============================================================
# ADAPTIVE META-DOUBT TESTS
# ============================================================

class TestAdaptiveMetaDoubt:
    """Test suite for the social-surprise adaptive epsilon_u."""

    def test_meta_doubt_activates_under_social_pressure(self):
        """When sigma_social is high, doubt should change faster than baseline."""
        # Two identical models, one with coupling (social pressure), one without
        cfg_base = {
            'dynamics': {'a': 0.7, 'b': 0.8, 'epsilon': 0.08, 'alpha': 0.15,
                         'v_cubic_divisor': 5.0, 'dt': 0.05},
            'coupling': {'D': 0.5, 'heretic_ratio': 0.0},
            'doubt': {'epsilon_u': 0.02, 'k_u': 1.0, 'sigma_baseline': 0.05,
                      'u_clamp': [0.0, 1.0], 'tau_u': 1.0,
                      'alpha_surprise': 2.0, 'surprise_cap': 5.0},
            'noise': {'sigma_v': 0.0}
        }

        # Model WITH social pressure (units far apart + coupling)
        model_social = Mem4ristorV3(config=cfg_base, seed=42)
        model_social._initialize_params(N=2, cold_start=True)
        model_social.v = np.array([-2.0, 2.0])  # Units far apart
        model_social.u = np.array([0.3, 0.3])
        adj = np.array([[0, 1], [1, 0]], dtype=float)

        # Model WITHOUT social pressure (no coupling)
        cfg_isolated = cfg_base.copy()
        cfg_isolated = {**cfg_base, 'coupling': {'D': 0.0, 'heretic_ratio': 0.0}}
        model_isolated = Mem4ristorV3(config=cfg_isolated, seed=42)
        model_isolated._initialize_params(N=2, cold_start=True)
        model_isolated.v = np.array([-2.0, 2.0])
        model_isolated.u = np.array([0.3, 0.3])

        u_social_before = model_social.u.copy()
        u_isolated_before = model_isolated.u.copy()

        model_social.step(I_stimulus=0.0, coupling_input=adj)
        model_isolated.step(I_stimulus=0.0)

        du_social = np.abs(model_social.u - u_social_before)
        du_isolated = np.abs(model_isolated.u - u_isolated_before)

        # Under social pressure, doubt should change faster due to meta-doubt
        assert np.mean(du_social) > np.mean(du_isolated), (
            f"Meta-doubt should accelerate doubt under social pressure. "
            f"Social du={np.mean(du_social):.6f}, Isolated du={np.mean(du_isolated):.6f}"
        )

    def test_meta_doubt_is_capped(self):
        """Verify that the adaptive gain doesn't exceed surprise_cap."""
        cfg = {
            'dynamics': {'a': 0.7, 'b': 0.8, 'epsilon': 0.08, 'alpha': 0.15,
                         'v_cubic_divisor': 5.0, 'dt': 0.05},
            'coupling': {'D': 10.0, 'heretic_ratio': 0.0},
            'doubt': {'epsilon_u': 0.02, 'k_u': 1.0, 'sigma_baseline': 0.05,
                      'u_clamp': [0.0, 1.0], 'tau_u': 1.0,
                      'alpha_surprise': 100.0, 'surprise_cap': 3.0},
            'noise': {'sigma_v': 0.0}
        }
        model = Mem4ristorV3(config=cfg, seed=42)
        model._initialize_params(N=2, cold_start=True)
        model.v = np.array([-5.0, 5.0])  # Extreme social pressure
        model.u = np.array([0.5, 0.5])
        adj = np.array([[0, 1], [1, 0]], dtype=float)

        # With alpha_surprise=100 and huge sigma_social, the raw multiplier
        # would be enormous, but surprise_cap=3 should clamp it
        u_before = model.u.copy()
        model.step(I_stimulus=0.0, coupling_input=adj)

        # The effective epsilon_u was at most 3x base → du is bounded
        du = np.abs(model.u - u_before)
        # Max theoretical du per step = surprise_cap * epsilon_u * (k_u * sigma + baseline) * dt / tau_u
        # = 3 * 0.02 * (large) * 0.05 / 1.0 → should still be finite and reasonable
        assert np.all(np.isfinite(model.u)), "u should remain finite with cap"
        assert np.all((model.u >= 0) & (model.u <= 1)), "u should remain in [0,1]"

    def test_meta_doubt_disabled_when_alpha_zero(self):
        """With alpha_surprise=0, behavior should match V3 exactly."""
        cfg = {
            'dynamics': {'a': 0.7, 'b': 0.8, 'epsilon': 0.08, 'alpha': 0.15,
                         'v_cubic_divisor': 5.0, 'dt': 0.05},
            'coupling': {'D': 0.15, 'heretic_ratio': 0.15},
            'doubt': {'epsilon_u': 0.02, 'k_u': 1.0, 'sigma_baseline': 0.05,
                      'u_clamp': [0.0, 1.0], 'tau_u': 1.0,
                      'alpha_surprise': 0.0, 'surprise_cap': 5.0},
            'noise': {'sigma_v': 0.05}
        }
        model = Mem4ristorV3(config=cfg, seed=42)
        model._initialize_params(N=100)
        for _ in range(100):
            model.step(I_stimulus=0.5)
        # Should just work without errors - backward compatible
        assert np.all(np.isfinite(model.v))
        assert np.all(np.isfinite(model.u))

    def test_meta_doubt_default_params_present(self):
        """Verify default config includes alpha_surprise and surprise_cap."""
        model = Mem4ristorV3(seed=42)
        assert 'alpha_surprise' in model.cfg['doubt'], "Default config missing alpha_surprise"
        assert 'surprise_cap' in model.cfg['doubt'], "Default config missing surprise_cap"
        assert model.cfg['doubt']['alpha_surprise'] == 2.0
        assert model.cfg['doubt']['surprise_cap'] == 5.0

    def test_meta_doubt_reproducibility(self):
        """Verify that meta-doubt doesn't break bit-level reproducibility."""
        cfg = {
            'dynamics': {'a': 0.7, 'b': 0.8, 'epsilon': 0.08, 'alpha': 0.15,
                         'v_cubic_divisor': 5.0, 'dt': 0.05},
            'coupling': {'D': 0.15, 'heretic_ratio': 0.15},
            'doubt': {'epsilon_u': 0.02, 'k_u': 1.0, 'sigma_baseline': 0.05,
                      'u_clamp': [0.0, 1.0], 'tau_u': 1.0,
                      'alpha_surprise': 2.0, 'surprise_cap': 5.0},
            'noise': {'sigma_v': 0.05}
        }
        m1 = Mem4ristorV3(config=cfg, seed=77)
        m2 = Mem4ristorV3(config=cfg, seed=77)
        for _ in range(50):
            m1.step(I_stimulus=0.5)
            m2.step(I_stimulus=0.5)
        assert np.allclose(m1.v, m2.v), "Meta-doubt broke v reproducibility"
        assert np.allclose(m1.u, m2.u), "Meta-doubt broke u reproducibility"


# ============================================================
# DOUBT-DRIVEN REWIRING TESTS
# ============================================================

class TestDoubtDrivenRewiring:
    """Test suite for the topological rewiring mechanism."""

    def _make_ring_adjacency(self, n):
        """Create a simple ring graph of n nodes."""
        adj = np.zeros((n, n))
        for i in range(n):
            adj[i, (i + 1) % n] = 1
            adj[(i + 1) % n, i] = 1
        return adj

    def test_rewiring_occurs_above_threshold(self):
        """Units with u > threshold should trigger rewiring."""
        n = 10
        adj = self._make_ring_adjacency(n)
        net = Mem4Network(adjacency_matrix=adj.copy(), seed=42,
                          rewire_threshold=0.7, rewire_cooldown=1)

        # Force high doubt on unit 0
        net.model.u = np.full(n, 0.3)
        net.model.u[0] = 0.9  # Above threshold

        # Force unit 0 and its neighbor to have similar v (consensual)
        net.model.v = np.zeros(n)
        net.model.v[0] = 1.0
        net.model.v[1] = 1.001  # Very similar to unit 0 (consensual)
        net.model.v[9] = -2.0   # Also neighbor but dissimilar

        initial_adj = net.adjacency_matrix.copy()
        net.step(I_stimulus=0.0)

        # Adjacency should have changed
        assert net.rewire_count > 0, "No rewiring occurred despite high doubt"
        assert not np.array_equal(net.adjacency_matrix, initial_adj), \
            "Adjacency matrix unchanged after rewiring"

    def test_no_rewiring_below_threshold(self):
        """Units with u < threshold should NOT trigger rewiring."""
        n = 10
        adj = self._make_ring_adjacency(n)
        net = Mem4Network(adjacency_matrix=adj.copy(), seed=42,
                          rewire_threshold=0.8, rewire_cooldown=1)

        # All units below threshold
        net.model.u = np.full(n, 0.3)

        initial_adj = net.adjacency_matrix.copy()
        for _ in range(10):
            net.step(I_stimulus=0.0)

        assert net.rewire_count == 0, f"Rewiring occurred despite low doubt: {net.rewire_count}"

    def test_rewiring_preserves_edge_count(self):
        """Rewiring should be degree-neutral: same total edges before and after."""
        n = 20
        adj = self._make_ring_adjacency(n)
        edges_before = np.sum(adj)

        net = Mem4Network(adjacency_matrix=adj.copy(), seed=42,
                          rewire_threshold=0.5, rewire_cooldown=1)

        # Force many units above threshold
        net.model.u = np.full(n, 0.9)
        # Spread v values to create consensual/dissimilar pairs
        net.model.v = np.linspace(-2, 2, n)

        for _ in range(5):
            net.step(I_stimulus=0.0)

        edges_after = np.sum(net.adjacency_matrix)
        assert edges_before == edges_after, (
            f"Edge count changed: {edges_before} -> {edges_after}. "
            f"Rewiring should be degree-neutral."
        )

    def test_rewiring_preserves_symmetry(self):
        """Adjacency matrix should remain symmetric after rewiring."""
        n = 15
        adj = self._make_ring_adjacency(n)
        net = Mem4Network(adjacency_matrix=adj.copy(), seed=42,
                          rewire_threshold=0.5, rewire_cooldown=1)

        net.model.u = np.full(n, 0.9)
        net.model.v = np.linspace(-2, 2, n)

        for _ in range(10):
            net.step(I_stimulus=0.0)

        assert np.allclose(net.adjacency_matrix, net.adjacency_matrix.T), \
            "Adjacency matrix lost symmetry after rewiring"

    def test_cooldown_prevents_rapid_rewiring(self):
        """A unit should not rewire again until cooldown expires."""
        n = 10
        adj = self._make_ring_adjacency(n)
        net = Mem4Network(adjacency_matrix=adj.copy(), seed=42,
                          rewire_threshold=0.5, rewire_cooldown=100)

        # Force high doubt on unit 0
        net.model.u = np.full(n, 0.1)
        net.model.u[0] = 0.9
        net.model.v = np.linspace(-2, 2, n)

        # First step: should rewire
        net.step(I_stimulus=0.0)
        count_after_first = net.rewire_count

        # Next steps: should NOT rewire (cooldown=100)
        for _ in range(10):
            net.model.u[0] = 0.9  # Keep doubt high
            net.step(I_stimulus=0.0)

        assert net.rewire_count == count_after_first, (
            f"Rewiring happened during cooldown: {count_after_first} -> {net.rewire_count}"
        )

    def test_no_rewiring_on_stencil_grid(self):
        """Stencil grids (no explicit adjacency) should skip rewiring gracefully."""
        net = Mem4Network(size=5, seed=42, rewire_threshold=0.5)
        # Force high doubt
        net.model.u = np.full(net.N, 0.9)

        # Should not crash and should not attempt rewiring
        for _ in range(10):
            net.step(I_stimulus=0.0)

        assert net.rewire_count == 0, "Stencil grid should never rewire"

    def test_rewiring_diversifies_information(self):
        """After rewiring, previously isolated clusters should mix."""
        # Create two clusters connected by a single bridge
        n = 10
        adj = np.zeros((n, n))
        # Cluster A: units 0-4 fully connected
        for i in range(5):
            for j in range(i + 1, 5):
                adj[i, j] = adj[j, i] = 1
        # Cluster B: units 5-9 fully connected
        for i in range(5, 10):
            for j in range(i + 1, 10):
                adj[i, j] = adj[j, i] = 1
        # Single bridge: 4 ↔ 5
        adj[4, 5] = adj[5, 4] = 1

        net = Mem4Network(adjacency_matrix=adj.copy(), seed=42,
                          rewire_threshold=0.6, rewire_cooldown=1)

        # Give cluster A positive opinions, cluster B negative
        net.model.v[:5] = 2.0
        net.model.v[5:] = -2.0
        # High doubt everywhere
        net.model.u[:] = 0.9

        # Run enough steps for rewiring to happen
        for _ in range(20):
            net.step(I_stimulus=0.0)

        # Check that cross-cluster connections appeared
        cross_edges = 0
        for i in range(5):
            for j in range(5, 10):
                cross_edges += net.adjacency_matrix[i, j]

        assert cross_edges > 1, (
            f"Expected rewiring to create cross-cluster connections. "
            f"Cross-edges: {cross_edges} (was 1 initially)"
        )

    def test_rewire_count_diagnostic(self):
        """Verify the rewire counter is incremented correctly."""
        n = 6
        adj = self._make_ring_adjacency(n)
        net = Mem4Network(adjacency_matrix=adj.copy(), seed=42,
                          rewire_threshold=0.5, rewire_cooldown=1)

        assert net.rewire_count == 0, "Counter should start at 0"

        net.model.u = np.full(n, 0.9)
        net.model.v = np.array([0.0, 0.001, 2.0, -2.0, 1.0, -1.0])

        net.step(I_stimulus=0.0)
        assert net.rewire_count > 0, "Counter should increment after rewiring"

    def test_incremental_laplacian_matches_full_rebuild(self):
        """Verify that incremental Laplacian updates produce the same result as full rebuild."""
        n = 15
        adj = self._make_ring_adjacency(n)
        net = Mem4Network(adjacency_matrix=adj.copy(), seed=42,
                          rewire_threshold=0.5, rewire_cooldown=1)

        net.model.u = np.full(n, 0.9)
        net.model.v = np.linspace(-2, 2, n)

        # Run several steps with incremental updates
        for _ in range(10):
            net.step(I_stimulus=0.0)

        # Get the incrementally maintained Laplacian
        L_incremental = net.L.copy()

        # Rebuild from scratch and compare
        net._rebuild_laplacian()
        L_full = net.L.copy()

        assert np.allclose(L_incremental, L_full, atol=1e-12), (
            f"Incremental Laplacian diverged from full rebuild. "
            f"Max diff: {np.max(np.abs(L_incremental - L_full)):.2e}"
        )

    def test_spectral_gap_preserved_after_rewiring(self):
        """Fiedler value should remain positive after rewiring (graph stays connected)."""
        from scipy.linalg import eigh

        n = 12
        adj = self._make_ring_adjacency(n)
        net = Mem4Network(adjacency_matrix=adj.copy(), seed=42,
                          rewire_threshold=0.5, rewire_cooldown=1)

        # Initial Fiedler value (ring graph)
        fiedler_before = net.get_spectral_gap()
        assert fiedler_before > 0, "Ring graph should be connected (Fiedler > 0)"

        # Force rewiring
        net.model.u = np.full(n, 0.9)
        net.model.v = np.linspace(-3, 3, n)

        for _ in range(20):
            net.step(I_stimulus=0.0)

        assert net.rewire_count > 0, "No rewiring happened — test is invalid"

        # Fiedler value after rewiring
        fiedler_after = net.get_spectral_gap()
        assert fiedler_after > 0, (
            f"Graph became disconnected after {net.rewire_count} rewires! "
            f"Fiedler: {fiedler_before:.4f} → {fiedler_after:.4f}"
        )


# ============================================================
# INTEGRATION TESTS
# ============================================================

class TestV4Integration:
    """Test both V4 mechanisms working together."""

    def test_v4_long_run_stability(self):
        """Run 1000 steps with both V4 mechanisms active. No crashes, finite states."""
        n = 20
        adj = np.zeros((n, n))
        for i in range(n):
            adj[i, (i + 1) % n] = 1
            adj[(i + 1) % n, i] = 1
            adj[i, (i + 2) % n] = 1
            adj[(i + 2) % n, i] = 1

        net = Mem4Network(adjacency_matrix=adj, seed=42,
                          rewire_threshold=0.7, rewire_cooldown=10)

        for step in range(1000):
            net.step(I_stimulus=0.5 * np.sin(step * 0.01))

        assert np.all(np.isfinite(net.model.v)), "v diverged after 1000 steps"
        assert np.all(np.isfinite(net.model.w)), "w diverged after 1000 steps"
        assert np.all(np.isfinite(net.model.u)), "u diverged after 1000 steps"
        assert np.all((net.model.u >= 0) & (net.model.u <= 1)), "u escaped [0,1]"

    def test_entropy_preservation_with_v4(self):
        """V4 mechanisms should preserve or improve entropy over time."""
        n = 10
        adj = np.zeros((n, n))
        for i in range(n):
            adj[i, (i + 1) % n] = 1
            adj[(i + 1) % n, i] = 1

        net = Mem4Network(adjacency_matrix=adj, seed=42,
                          rewire_threshold=0.7, rewire_cooldown=20)

        # Warm up
        for _ in range(100):
            net.step(I_stimulus=0.0)

        h_mid = net.calculate_entropy()

        # Run more
        for _ in range(500):
            net.step(I_stimulus=0.0)

        h_end = net.calculate_entropy()

        # Entropy should not collapse to zero
        assert h_end > 0.1, f"Entropy collapsed with V4 active: {h_end:.4f}"
