"""
Robustness and Edge Case Tests for Mem4ristor V3.
Verifies system stability under extreme conditions and invalid inputs.

V3.0 Hardening: All assertions are strict - no NaN persistence accepted.
"""
import numpy as np
import pytest
from mem4ristor.core import Mem4ristorV3, Mem4Network


# =============================================================================
# Test Case 1: Memory Corruption (NaN Injection)
# =============================================================================
def test_nan_injection_zero_propagation():
    """NaN injected into v must be cleaned to ZERO propagation after step."""
    model = Mem4ristorV3(seed=42)
    model._initialize_params(N=100)

    model.v[50] = np.nan
    model.step(I_stimulus=0.5)

    nan_count = np.isnan(model.v).sum()
    assert nan_count == 0, f"NaN PROPAGATION: {nan_count}/100 units still contain NaN after step."


def test_nan_in_coupling_matrix():
    """NaN in coupling matrix must be cleaned or rejected."""
    model = Mem4ristorV3(seed=42)
    model._initialize_params(N=10)

    adj = np.ones((10, 10)) * 0.1
    adj[5, 5] = np.nan

    model.step(I_stimulus=0.0, coupling_input=adj)
    assert not np.any(np.isnan(model.v)), "NaN in adjacency matrix contaminated v"
    assert not np.any(np.isinf(model.v)), "NaN in adjacency matrix caused Inf in v"


# =============================================================================
# Test Case 2: Edge Cases (Small N)
# =============================================================================
def test_single_unit_network():
    """N=1 must remain stable and bounded over 100 steps."""
    model = Mem4ristorV3(seed=42)
    model._initialize_params(N=1)

    for _ in range(100):
        model.step(I_stimulus=0.5)

    assert np.isfinite(model.v[0]), f"N=1 diverged: v={model.v[0]}"
    assert np.isfinite(model.u[0]), f"N=1 diverged: u={model.u[0]}"
    assert np.abs(model.v[0]) < 100, f"N=1 unbounded: v={model.v[0]}"


def test_two_unit_network_symmetry():
    """N=2 with symmetric IC and coupling should stay close (noise-bounded)."""
    cfg = {
        'dynamics': {'a': 0.7, 'b': 0.8, 'epsilon': 0.08, 'alpha': 0.15, 'v_cubic_divisor': 5.0, 'dt': 0.05},
        'coupling': {'D': 0.15, 'heretic_ratio': 0.0},
        'doubt': {'epsilon_u': 0.02, 'k_u': 1.0, 'sigma_baseline': 0.05, 'u_clamp': [0.0, 1.0], 'tau_u': 1.0},
        'noise': {'sigma_v': 0.02}
    }
    model = Mem4ristorV3(config=cfg, seed=42)
    model._initialize_params(N=2)

    model.v = np.array([0.5, 0.5])
    model.w = np.array([0.5, 0.5])
    model.u = np.array([0.25, 0.25])

    adj = np.array([[0, 1], [1, 0]], dtype=float)
    model.rng = np.random.RandomState(42)

    for _ in range(100):
        model.step(I_stimulus=0.0, coupling_input=adj)

    diff = np.abs(model.v[0] - model.v[1])
    # With sigma_v=0.02 and 100 steps, divergence should be bounded
    # Tolerance: 0.3 is ~15x the noise level, generous but not absurd
    assert diff < 0.3, f"Symmetry broken beyond noise tolerance: |v0-v1|={diff:.4f}"


# =============================================================================
# Test Case 3: Invalid Parameters
# =============================================================================
def test_negative_coupling():
    """D < 0 should produce finite results (inverts coupling direction)."""
    cfg = {
        'dynamics': {'a': 0.7, 'b': 0.8, 'epsilon': 0.08, 'alpha': 0.15, 'v_cubic_divisor': 5.0, 'dt': 0.05},
        'coupling': {'D': -0.5, 'heretic_ratio': 0.15},
        'doubt': {'epsilon_u': 0.02, 'k_u': 1.0, 'sigma_baseline': 0.05, 'u_clamp': [0.0, 1.0], 'tau_u': 1.0},
        'noise': {'sigma_v': 0.02}
    }
    model = Mem4ristorV3(config=cfg, seed=42)
    model._initialize_params(N=100)

    for _ in range(100):
        model.step()

    assert np.all(np.isfinite(model.v)), "Negative D caused divergence"
    assert np.all(np.abs(model.v) < 100), "Negative D caused unbounded growth"


def test_negative_noise():
    """sigma_v < 0 must raise ValueError from numpy."""
    cfg = {
        'dynamics': {'a': 0.7, 'b': 0.8, 'epsilon': 0.08, 'alpha': 0.15, 'v_cubic_divisor': 5.0, 'dt': 0.05},
        'coupling': {'D': 0.5, 'heretic_ratio': 0.15},
        'doubt': {'epsilon_u': 0.02, 'k_u': 1.0, 'sigma_baseline': 0.05, 'u_clamp': [0.0, 1.0], 'tau_u': 1.0},
        'noise': {'sigma_v': -0.1}
    }
    model = Mem4ristorV3(config=cfg, seed=42)
    model._initialize_params(N=10)

    with pytest.raises(ValueError):
        model.step()


def test_zero_dt():
    """dt=0 must raise ValueError from config validation."""
    cfg = {
        'dynamics': {'a': 0.7, 'b': 0.8, 'epsilon': 0.08, 'alpha': 0.15, 'v_cubic_divisor': 5.0, 'dt': 0.0},
        'coupling': {'D': 0.5, 'heretic_ratio': 0.15},
        'doubt': {'epsilon_u': 0.02, 'k_u': 1.0, 'sigma_baseline': 0.05, 'u_clamp': [0.0, 1.0], 'tau_u': 1.0},
        'noise': {'sigma_v': 0.02}
    }
    with pytest.raises(ValueError, match="must be positive"):
        Mem4ristorV3(config=cfg, seed=42)


# =============================================================================
# Test Case 4: Critical Boundaries
# =============================================================================
def test_critical_doubt_boundary():
    """u=0.5: sigmoid kernel gives ~δ (leakage only), system must remain stable."""
    model = Mem4ristorV3(seed=42)
    model._initialize_params(N=100)

    model.u[:] = 0.5
    adj = np.ones((100, 100)) / 100

    v_before = model.v.copy()
    model.step(I_stimulus=0.0, coupling_input=adj)

    assert np.all(np.isfinite(model.v)), "u=0.5 caused instability"
    # V3: At u=0.5, sigmoid gives δ=0.05, not exactly zero
    # So there IS residual coupling, but very weak


def test_doubt_at_clamp_boundaries():
    """u must stay within [0, 1] even when initialized outside."""
    model = Mem4ristorV3(seed=42)
    model._initialize_params(N=10)

    model.u[0:5] = 0.0
    model.u[5:10] = 1.0

    for _ in range(100):
        model.step()

    assert np.all(model.u >= 0.0), f"u < 0 detected: min={model.u.min()}"
    assert np.all(model.u <= 1.0), f"u > 1 detected: max={model.u.max()}"


# =============================================================================
# Test Case 5: Invalid Ratios
# =============================================================================
def test_heretic_ratio_above_one():
    """heretic_ratio > 1.0 must be rejected."""
    cfg = {
        'dynamics': {'a': 0.7, 'b': 0.8, 'epsilon': 0.08, 'alpha': 0.15, 'v_cubic_divisor': 5.0, 'dt': 0.05},
        'coupling': {'D': 0.5, 'heretic_ratio': 1.5},
        'doubt': {'epsilon_u': 0.02, 'k_u': 1.0, 'sigma_baseline': 0.05, 'u_clamp': [0.0, 1.0], 'tau_u': 1.0},
        'noise': {'sigma_v': 0.02}
    }
    with pytest.raises(ValueError, match="heretic_ratio"):
        Mem4ristorV3(config=cfg, seed=42)


def test_heretic_ratio_negative():
    """heretic_ratio < 0 must be rejected."""
    cfg = {
        'dynamics': {'a': 0.7, 'b': 0.8, 'epsilon': 0.08, 'alpha': 0.15, 'v_cubic_divisor': 5.0, 'dt': 0.05},
        'coupling': {'D': 0.5, 'heretic_ratio': -0.1},
        'doubt': {'epsilon_u': 0.02, 'k_u': 1.0, 'sigma_baseline': 0.05, 'u_clamp': [0.0, 1.0], 'tau_u': 1.0},
        'noise': {'sigma_v': 0.02}
    }
    with pytest.raises(ValueError, match="heretic_ratio"):
        Mem4ristorV3(config=cfg, seed=42)


# =============================================================================
# Test Case 6: Overflow/Underflow
# =============================================================================
def test_extreme_stimulus_overflow():
    """Extreme stimulus (1e10) must be clamped, not cause overflow."""
    model = Mem4ristorV3(seed=42)
    model._initialize_params(N=10)

    for _ in range(10):
        model.step(I_stimulus=1e10)

    assert np.all(np.isfinite(model.v)), "Overflow: v contains inf/nan"
    assert np.all(np.abs(model.v) < 100), "Stimulus not properly clamped"


def test_extreme_initial_conditions():
    """v initialized to extreme values should be clamped and recover."""
    model = Mem4ristorV3(seed=42)
    model._initialize_params(N=10)
    model.v = np.full(10, 1e10)

    for _ in range(100):
        model.step()

    assert np.all(np.isfinite(model.v)), "Failed to recover from extreme IC"
    assert np.all(np.abs(model.v) <= 100), f"System didn't converge: max(v)={np.max(np.abs(model.v))}"


# =============================================================================
# Test Case 7: Type Checking
# =============================================================================
def test_wrong_coupling_shape():
    """Coupling matrix with wrong dimensions must raise error."""
    model = Mem4ristorV3(seed=42)
    model._initialize_params(N=10)
    wrong_adj = np.ones((5, 5))

    with pytest.raises((ValueError, IndexError)):
        model.step(I_stimulus=0.0, coupling_input=wrong_adj)


def test_stimulus_wrong_size():
    """Stimulus vector with wrong size must raise ValueError."""
    model = Mem4ristorV3(seed=42)
    model._initialize_params(N=10)
    wrong_stim = np.ones(5)

    with pytest.raises(ValueError):
        model.step(I_stimulus=wrong_stim)
