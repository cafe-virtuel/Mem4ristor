"""
Input Fuzzing Tests for Mem4ristor V3.
Throws random garbage at the model to verify crash resistance.

V3.0 Hardening: 200 iterations, input tracking, proper assertions only.
"""
import numpy as np
import pytest
import random

from mem4ristor.core import Mem4ristorV3


def get_random_garbage():
    """Generates random garbage data to throw at the model."""
    garbage_types = [
        "string",
        np.nan,
        np.inf,
        -np.inf,
        None,
        1j,  # Complex number
        {"a": 1},
        [1, "2", 3],
        object(),
        np.array([np.nan, np.inf]),
        np.full((10, 10), 9999999999.9),
        1e308,  # Max float
        -1e308
    ]
    return random.choice(garbage_types)


def test_vicious_fuzzing_inputs():
    """200 iterations of input fuzzing on step(). No crash, no NaN/Inf in state."""
    model = Mem4ristorV3(seed=123)
    model._initialize_params(N=50)

    accepted_count = 0
    rejected_count = 0

    for i in range(200):
        garbage = get_random_garbage()
        try:
            if random.choice([True, False]):
                model.step(I_stimulus=garbage)
            else:
                model.step(coupling_input=garbage)

            accepted_count += 1
            # If accepted, state MUST be clean
            assert not np.any(np.isnan(model.v)), f"NaN in v after accepting input {type(garbage).__name__}"
            assert not np.any(np.isinf(model.v)), f"Inf in v after accepting input {type(garbage).__name__}"
            assert not np.any(np.isnan(model.w)), f"NaN in w after accepting input {type(garbage).__name__}"
            assert not np.any(np.isnan(model.u)), f"NaN in u after accepting input {type(garbage).__name__}"

        except (ValueError, TypeError):
            rejected_count += 1
        except Exception as e:
            pytest.fail(f"CRASH at iteration {i}: {type(e).__name__} with input {str(garbage)[:50]}: {e}")

    # At least some inputs should be rejected (strings, dicts, objects)
    assert rejected_count > 0, "No inputs were rejected - sanitization may be too permissive"


def test_vicious_fuzzing_config():
    """200 iterations of config mutation. Must either reject or survive."""
    accepted_count = 0
    rejected_count = 0

    for i in range(200):
        bad_config = {
            'dynamics': {'dt': random.choice([0.05, 0.0, -0.1, "0.1", None])},
            'coupling': {'heretic_ratio': random.choice([0.15, 2.0, -0.5, "high"])},
            'doubt': {'tau_u': random.choice([1.0, 0.0, -1.0])}
        }

        try:
            new_model = Mem4ristorV3(config=bad_config)
            new_model.step()
            accepted_count += 1

            # If accepted, state must be clean
            assert np.all(np.isfinite(new_model.v)), f"Non-finite v with config {bad_config}"

        except (ValueError, TypeError, KeyError):
            rejected_count += 1
        except Exception as e:
            pytest.fail(f"CONFIG CRASH at iteration {i}: {type(e).__name__} with config {bad_config}: {e}")

    assert rejected_count > 0, "No configs were rejected - validation may be too permissive"
