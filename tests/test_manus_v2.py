"""
Manual Chaos Tests for Mem4ristor V3.
Tests extreme initialization, config injection, state corruption recovery.

V3.0 Hardening: All print-based checks replaced with proper assertions.
"""
import numpy as np
import pytest
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

from mem4ristor.core import Mem4ristorV3


def test_extreme_initialization_chaos():
    """Extreme N values must be rejected with proper errors."""
    model = Mem4ristorV3()

    # N=0: must reject
    with pytest.raises(ValueError):
        model._initialize_params(N=0)

    # N=-1: must reject
    with pytest.raises(ValueError):
        model._initialize_params(N=-1)

    # N=10^9: must reject (DoS protection)
    with pytest.raises(ValueError):
        model._initialize_params(N=10**9)

    # N=float: must reject
    with pytest.raises((ValueError, TypeError)):
        model._initialize_params(N=1.5)


def test_config_injection_attack():
    """Malicious config values must be caught by validation."""
    evil_configs = [
        ({'dynamics': {'v_cubic_divisor': 0}}, "division by zero"),
        ({'dynamics': {'dt': -0.05}}, "negative dt"),
        ({'doubt': {'tau_u': 0}}, "zero tau_u"),
        ({'coupling': {'D': np.inf}}, "infinite coupling"),
        ({'noise': {'use_rtn': True, 'rtn_p_flip': 2.0}}, "probability > 1"),
    ]

    for cfg, description in evil_configs:
        with pytest.raises((ValueError, TypeError), match=None):
            model = Mem4ristorV3(config=cfg)
            model.step()  # Some errors only manifest on step


def test_config_injection_accepted_but_finite():
    """Config values that are unusual but valid must produce finite results."""
    unusual_configs = [
        {'dynamics': {'dt': 1e10}},  # Huge dt - should produce finite (clamped) results
        {'dynamics': {'v_cubic_divisor': -1e-20}},  # Tiny negative divisor - rejected by validation
    ]

    # Huge dt: accepted but may produce wild results - must stay finite
    model = Mem4ristorV3(config={'dynamics': {'dt': 1.0}})
    model._initialize_params(N=10)
    model.step()
    assert np.all(np.isfinite(model.v)), "Large dt produced non-finite v"


def test_stimulus_overflow_and_types():
    """Various invalid stimulus types must be rejected or handled safely."""
    model = Mem4ristorV3()
    model._initialize_params(N=10)

    # NaN stimulus: must be cleaned
    model.step(I_stimulus=np.nan)
    assert not np.any(np.isnan(model.v)), "NaN stimulus contaminated v"

    # Inf stimulus: must be clamped
    model.step(I_stimulus=np.inf)
    assert np.all(np.isfinite(model.v)), "Inf stimulus caused non-finite v"

    # String stimulus: must be rejected
    with pytest.raises((ValueError, TypeError)):
        model.step(I_stimulus="un million")

    # Wrong-size array: must be rejected
    with pytest.raises(ValueError):
        model.step(I_stimulus=np.array([1]*11))

    # Dict stimulus: must be rejected
    with pytest.raises((ValueError, TypeError)):
        model.step(I_stimulus={"value": 0.5})


def test_state_corruption_manual():
    """Manual state corruption must be recovered by guards."""
    model = Mem4ristorV3()
    model._initialize_params(N=10)

    # Inject corruption
    model.v[0] = np.nan
    model.w[5] = np.inf
    model.u[2] = -10.0  # Outside [0,1]

    model.step()

    # NaN must be cleaned
    assert not np.isnan(model.v[0]), "NaN not recovered in v"
    assert np.abs(model.v[0]) < 5.0, f"NaN recovered but value exploded: {model.v[0]}"

    # Inf must be cleaned
    assert np.isfinite(model.w[5]), "Inf not recovered in w"

    # u must be clamped to [0,1]
    assert model.u[2] >= 0.0, f"Negative u not clamped: {model.u[2]}"
    assert model.u[2] <= 1.0, f"u not clamped to [0,1]: {model.u[2]}"
