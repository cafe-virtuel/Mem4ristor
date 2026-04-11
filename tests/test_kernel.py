"""
Kernel Tests for Mem4ristor V3.
Tests the core mechanisms: heretic inversion, sigmoid coupling, plasticity, entropy.
"""
import pytest
import numpy as np
from mem4ristor.core import Mem4ristorV3


def test_heretic_inversion():
    """Verify that ALL heretic units invert the stimulus polarity."""
    cfg = {
        'dynamics': {'a': 0.7, 'b': 0.8, 'epsilon': 0.08, 'alpha': 0.15, 'v_cubic_divisor': 5.0, 'dt': 0.05},
        'coupling': {'D': 0.0, 'heretic_ratio': 1.0},
        'doubt': {'epsilon_u': 0.0, 'k_u': 1.0, 'sigma_baseline': 0.05, 'u_clamp': [0.0, 1.0], 'tau_u': 1.0},
        'noise': {'sigma_v': 0.0}
    }
    model = Mem4ristorV3(config=cfg, seed=42)
    model._initialize_params(N=10, cold_start=True)
    assert np.all(model.heretic_mask), "Not all units are heretics with ratio=1.0"

    v_pre = model.v.copy()
    model.step(I_stimulus=1.0)
    dv = model.v - v_pre
    # With all heretics, I_eff = -1.0 for all units
    # EVERY unit should move negative (not just the mean)
    negative_count = np.sum(dv < 0)
    assert negative_count == 10, f"Expected all 10 heretics to move negative, got {negative_count}"


def test_levitating_sigmoid_kernel():
    """Verify V3 sigmoid coupling at key doubt values."""
    model = Mem4ristorV3(seed=42)
    model._initialize_params(N=3)

    # Test the sigmoid filter values directly
    # u=0 -> tanh(π*0.5) + 0.05 ≈ 0.92 + 0.05 = 0.97 (strong attraction)
    model.u = np.array([0.0, 0.5, 1.0])
    u_centered = 0.5 - model.u
    u_filter = np.tanh(model.sigmoid_steepness * u_centered) + model.social_leakage

    assert u_filter[0] > 0.9, f"u=0 should give strong attraction, got {u_filter[0]:.4f}"
    assert abs(u_filter[1] - 0.05) < 0.01, f"u=0.5 should give ~δ (leakage), got {u_filter[1]:.4f}"
    assert u_filter[2] < -0.85, f"u=1 should give strong repulsion, got {u_filter[2]:.4f}"


def test_repulsive_flip():
    """Verify that coupling becomes repulsive when doubt u > 0.5."""
    cfg = {
        'dynamics': {'a': 0.7, 'b': 0.8, 'epsilon': 0.08, 'alpha': 0.15, 'v_cubic_divisor': 5.0, 'dt': 0.05},
        'coupling': {'D': 1.0, 'heretic_ratio': 0.0},
        'doubt': {'epsilon_u': 0.0, 'k_u': 1.0, 'sigma_baseline': 0.8, 'u_clamp': [0.0, 1.0], 'tau_u': 1.0},
        'noise': {'sigma_v': 0.0}
    }
    model = Mem4ristorV3(config=cfg, seed=42)
    model._initialize_params(N=2)
    model.v = np.array([-1.0, 1.0])
    model.u = np.array([0.8, 0.8])
    adj = np.array([[0, 1], [1, 0]])
    model.step(I_stimulus=0.0, coupling_input=adj)
    assert model.v[0] < -1.0, f"Unit 0 should be repelled further negative. v={model.v[0]}"
    assert model.v[1] > 1.0, f"Unit 1 should be repelled further positive. v={model.v[1]}"


def test_spatial_clustering():
    """Verify anti-clustering: heretics are uniformly distributed, not clumped."""
    model = Mem4ristorV3(seed=42)
    model._initialize_params(N=100)
    heretic_ids = np.where(model.heretic_mask)[0]
    diffs = np.diff(heretic_ids)

    # With 15% ratio and stratified placement, block size = int(1/0.15) = 6
    # Adjacent heretics (diff==1) should be rare: at most 1-2 from block boundaries
    cluster_count = np.sum(diffs == 1)
    expected_max = int(0.15 * 100 * 0.15)  # ~2 adjacent pairs by chance
    assert cluster_count <= expected_max + 1, (
        f"Heretics too clustered: {cluster_count} adjacent pairs "
        f"(max expected ~{expected_max} for block_size=6)"
    )


def test_rk45_stability():
    """Verify RK45 preserves diversity over integration window."""
    model = Mem4ristorV3(seed=42)
    model._initialize_params(N=100)
    adj = np.zeros((100, 100))
    for i in range(100):
        for j in [i-1, i+1, i-10, i+10]:
            if 0 <= j < 100:
                adj[i, j] = 1

    h_before = model.calculate_entropy()
    model.solve_rk45((0, 1.0), I_stimulus=0.5, adj_matrix=adj)
    h_after = model.calculate_entropy()

    # Entropy should not collapse: it should remain at least 50% of initial
    assert h_after > 0.5 * h_before, (
        f"Entropy collapsed during RK45: {h_before:.4f} -> {h_after:.4f}"
    )
    assert h_after > 0.3, f"Absolute entropy too low after RK45: {h_after:.4f}"


def test_reproducibility():
    """Verify bit-level reproducibility with same seed."""
    cfg = {
        'dynamics': {'a': 0.7, 'b': 0.8, 'epsilon': 0.08, 'alpha': 0.15, 'v_cubic_divisor': 5.0, 'dt': 0.05},
        'coupling': {'D': 0.15, 'heretic_ratio': 0.15},
        'doubt': {'epsilon_u': 0.02, 'k_u': 1.0, 'sigma_baseline': 0.05, 'u_clamp': [0.0, 1.0], 'tau_u': 1.0},
        'noise': {'sigma_v': 0.05}
    }
    model1 = Mem4ristorV3(config=cfg, seed=123)
    model2 = Mem4ristorV3(config=cfg, seed=123)
    for _ in range(50):  # 50 steps, not just 10
        model1.step(I_stimulus=0.5)
        model2.step(I_stimulus=0.5)
    assert np.allclose(model1.v, model2.v), f"v diverged: max diff = {np.max(np.abs(model1.v - model2.v))}"
    assert np.allclose(model1.w, model2.w), f"w diverged: max diff = {np.max(np.abs(model1.w - model2.w))}"
    assert np.allclose(model1.u, model2.u), f"u diverged: max diff = {np.max(np.abs(model1.u - model2.u))}"


def test_plasticity_activation():
    """Verify that inhibition plasticity activates only when u > 0.5.

    Strategy: Two units with IDENTICAL v and w but DIFFERENT u.
    Both receive the same coupling (same neighbor structure).
    The FHN contribution to dw is identical for both.
    The ONLY difference in dw comes from the plasticity term:
        dw_learn = lambda * sigma_social * I(u>0.5) * (1 - (w/w_sat)^2) - w/tau
    With w=0 initially, decay is 0, saturation is 1, so:
        dw_learn = lambda * sigma_social * I(u>0.5)
    Unit with u=0.1: I(u>0.5)=0 → no plasticity
    Unit with u=0.8: I(u>0.5)=1 → plasticity active
    """
    cfg = {
        'dynamics': {'a': 0.7, 'b': 0.8, 'epsilon': 0.08, 'alpha': 0.15, 'v_cubic_divisor': 5.0, 'dt': 0.05,
                     'lambda_learn': 1.0, 'tau_plasticity': 1e9, 'w_saturation': 10.0},
        'coupling': {'D': 1.0, 'heretic_ratio': 0.0},
        'doubt': {'epsilon_u': 0.0, 'k_u': 1.0, 'sigma_baseline': 0.05, 'u_clamp': [0.0, 1.0], 'tau_u': 1.0,
                  'alpha_surprise': 0.0, 'surprise_cap': 5.0},
        'noise': {'sigma_v': 0.0}
    }
    model = Mem4ristorV3(config=cfg, seed=42)
    model._initialize_params(N=4, cold_start=True)

    # Two pairs: (0,1) and (2,3) — identical v, w, but different u
    model.v = np.array([-1.0, 1.0, -1.0, 1.0])
    model.w = np.array([0.0, 0.0, 0.0, 0.0])
    model.u = np.array([0.1, 0.1, 0.8, 0.8])  # Pair 0,1: low doubt. Pair 2,3: high doubt.

    # Coupling: 0↔1 and 2↔3 (two isolated pairs)
    adj = np.array([[0, 1, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 1, 0]], dtype=float)

    w_before = model.w.copy()
    model.step(I_stimulus=0.0, coupling_input=adj)
    dw = model.w - w_before

    # FHN dw contribution is identical for units 0 and 2 (same v, w, a, b, epsilon)
    # But unit 2 has plasticity active (u=0.8 > 0.5), unit 0 does not (u=0.1 < 0.5)
    # sigma_social is |laplacian_v| = |v_neighbor - v_self| = 2.0 for both pairs
    # dw_learn for unit 2 = lambda * 2.0 * 1.0 * 1.0 * dt = 1.0 * 2.0 * 0.05 = 0.1
    # dw_learn for unit 0 = 0 (innovation_mask = 0)

    # Unit 2 should have MORE w change than unit 0
    assert abs(dw[2]) > abs(dw[0]) + 0.01, (
        f"Plasticity not detected: dw[2]={dw[2]:.6f} should exceed dw[0]={dw[0]:.6f} "
        f"by at least 0.01 (expected ~0.1 plasticity contribution)"
    )
    # Same for unit 3 vs unit 1
    assert abs(dw[3]) > abs(dw[1]) + 0.01, (
        f"Plasticity not detected: dw[3]={dw[3]:.6f} should exceed dw[1]={dw[1]:.6f}"
    )


def test_backward_compatibility_alias():
    """Verify that Mem4ristorV2 alias works for backward compatibility."""
    from mem4ristor.core import Mem4ristorV2
    model = Mem4ristorV2(seed=42)
    assert isinstance(model, Mem4ristorV3), "Mem4ristorV2 should be an alias for Mem4ristorV3"
    model.step()
    assert np.all(np.isfinite(model.v))
