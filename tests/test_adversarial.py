"""
Adversarial Tests for Mem4ristor V3.
Tests known limitations and stress conditions.

V3.0 Note: xfail tests document known theoretical limitations.
The Levitating Sigmoid eliminates the dead zone at u=0.5, but
SNR collapse at high noise and Euler instability at high dt remain.
"""
import numpy as np
import pytest
from mem4ristor.core import Mem4ristorV3


@pytest.mark.xfail(reason="Known theoretical limitation: SNR collapse at high noise regimes.")
def test_snr_significance_breakdown():
    """Verify signal integrity against noise floor.

    At high noise (sigma_v=0.15), the coupling signal D_eff * |f(u)| * |laplacian|
    can be dominated by noise. V3's sigmoid helps (no dead zone) but doesn't
    eliminate the fundamental SNR issue at extreme noise levels.
    """
    cfg = {
        'dynamics': {'a': 0.7, 'b': 0.8, 'epsilon': 0.08, 'alpha': 0.15, 'v_cubic_divisor': 5.0, 'dt': 0.05},
        'coupling': {'D': 0.5, 'heretic_ratio': 0.0},
        'doubt': {'epsilon_u': 0.0, 'k_u': 1.0, 'sigma_baseline': 0.8, 'u_clamp': [0.0, 1.0], 'tau_u': 1.0},
        'noise': {'sigma_v': 0.15}
    }
    model = Mem4ristorV3(config=cfg, seed=42)
    model._initialize_params(N=100)

    # Actual SNR calculation: coupling signal vs noise floor
    # V3 sigmoid at u=0.8: tanh(π(0.5-0.8)) + 0.05 = tanh(-0.94) + 0.05 ≈ -0.74 + 0.05 = -0.69
    sigmoid_at_u08 = np.tanh(np.pi * (0.5 - 0.8)) + 0.05
    signal_strength = model.D_eff * abs(sigmoid_at_u08)
    noise_floor = cfg['noise']['sigma_v']

    snr = signal_strength / noise_floor
    assert snr > 3.0, f"SNR too low ({snr:.2f}). Coupling signal buried in noise."


@pytest.mark.xfail(reason="Known numerical instability at dt > 0.1. Recommended dt <= 0.05.")
def test_euler_drift_torture():
    """Verify Euler integration stability at high time steps."""
    cfg = {
        'dynamics': {'a': 0.7, 'b': 0.8, 'epsilon': 0.08, 'alpha': 0.15, 'v_cubic_divisor': 5.0, 'dt': 0.5},
        'coupling': {'D': 0.5, 'heretic_ratio': 0.5},
        'doubt': {'epsilon_u': 0.1, 'k_u': 2.0, 'sigma_baseline': 0.0, 'u_clamp': [0.0, 1.0], 'tau_u': 10.0},
        'noise': {'sigma_v': 0.01}
    }
    model = Mem4ristorV3(config=cfg, seed=123)
    model._initialize_params(N=625)

    h_initial = model.calculate_entropy()

    for _ in range(200):
        model.step()

    h_final = model.calculate_entropy()
    assert h_final > 0.5 * h_initial, f"Entropy collapse at high dt: H={h_final:.2f} (initial={h_initial:.2f})"


def test_scale_invariance_n1200():
    """Verify system operates correctly at N=1200 without crash."""
    model = Mem4ristorV3(seed=42)
    model._initialize_params(N=1200)

    h_before = model.calculate_entropy()

    for _ in range(10):
        model.step()

    h_after = model.calculate_entropy()

    # Must not crash AND must maintain some diversity
    assert np.all(np.isfinite(model.v)), "Non-finite values at N=1200"
    assert h_after > 0.0, f"Total entropy collapse at N=1200: H={h_after}"
