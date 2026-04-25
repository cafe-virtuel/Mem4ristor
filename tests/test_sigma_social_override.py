"""Tests for the sigma_social_override hook in dynamics.py / topology.py.

Added 2026-04-25 (v3.2.0). Audits the ablation hook used by
experiments/p2_sigma_social_ablation.py.
"""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from mem4ristor.topology import Mem4Network
from mem4ristor.dynamics import Mem4ristorV3
from mem4ristor.graph_utils import make_ba


# --------------------------------------------------------------------------
#  Behavioural tests
# --------------------------------------------------------------------------

def test_override_changes_u_only_first_step():
    """First-step v should be identical with/without override (coupling uses
    real laplacian); u should differ when override is far from real value."""
    adj = make_ba(20, 3, seed=42)
    a = Mem4Network(adjacency_matrix=adj.copy(), heretic_ratio=0.0, seed=42)
    b = Mem4Network(adjacency_matrix=adj.copy(), heretic_ratio=0.0, seed=42)
    np.testing.assert_allclose(a.v, b.v)
    np.testing.assert_allclose(a.model.u, b.model.u)

    a.step(I_stimulus=0.5)
    b.step(I_stimulus=0.5, sigma_social_override=np.full(a.N, 99.0))

    # With same noise seed, eta is regenerated identically, coupling is the
    # same — v should match exactly on the first step.
    np.testing.assert_allclose(a.v, b.v, rtol=1e-6, atol=1e-10)
    # u must diverge because override drives the doubt equation.
    assert not np.allclose(a.model.u, b.model.u, atol=1e-6)


def test_override_none_matches_default():
    """sigma_social_override=None must reproduce default behaviour exactly."""
    adj = make_ba(20, 3, seed=42)
    a = Mem4Network(adjacency_matrix=adj.copy(), heretic_ratio=0.0, seed=42)
    b = Mem4Network(adjacency_matrix=adj.copy(), heretic_ratio=0.0, seed=42)
    for _ in range(20):
        a.step(I_stimulus=0.3)
        b.step(I_stimulus=0.3, sigma_social_override=None)
    np.testing.assert_allclose(a.v, b.v)
    np.testing.assert_allclose(a.model.u, b.model.u)


def test_override_equals_real_sigma_matches_default():
    """Passing the actual |laplacian_v| as override must reproduce the
    default trajectory bit-for-bit."""
    adj = make_ba(20, 3, seed=42)
    a = Mem4Network(adjacency_matrix=adj.copy(), heretic_ratio=0.0, seed=42)
    b = Mem4Network(adjacency_matrix=adj.copy(), heretic_ratio=0.0, seed=42)

    for _ in range(20):
        # Compute the override that is identical to what topology.py:step
        # internally uses (after coupling-norm scaling).
        lv = -(b.L @ b.v)
        if b.coupling_norm != "uniform":
            D = b.model.cfg["coupling"]["D"]
            uniform_D_eff = D / np.sqrt(b.N)
            scale_factors = (b.node_weights * D) / uniform_D_eff
            lv = lv * scale_factors
        a.step(I_stimulus=0.3)
        b.step(I_stimulus=0.3, sigma_social_override=np.abs(lv))

    np.testing.assert_allclose(a.v, b.v, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(a.model.u, b.model.u, rtol=1e-10, atol=1e-12)


def test_override_clamped_to_nonneg_range():
    """The override is fed through np.clip(.., 0, 100) before being used in
    the adaptive epsilon_u term — extreme negative values must not break u."""
    adj = make_ba(15, 3, seed=42)
    n = Mem4Network(adjacency_matrix=adj.copy(), heretic_ratio=0.0, seed=42)
    bad = -np.full(n.N, 5.0)  # negative override
    n.step(I_stimulus=0.0, sigma_social_override=bad)
    assert np.all(np.isfinite(n.model.u))
    assert np.all(n.model.u >= 0.0)
    assert np.all(n.model.u <= 1.0)


def test_override_does_not_affect_plasticity():
    """Plasticity (dw_learning) uses the real sigma_social, not the override.
    Two networks identical except for the override should evolve w
    identically on the first step where w[0]==0."""
    adj = make_ba(15, 3, seed=42)
    a = Mem4Network(adjacency_matrix=adj.copy(), heretic_ratio=0.0, seed=42)
    b = Mem4Network(adjacency_matrix=adj.copy(), heretic_ratio=0.0, seed=42)

    # Drive both to a state where some u_i > 0.5 so plasticity is active
    for _ in range(50):
        a.step(I_stimulus=0.5)
        b.step(I_stimulus=0.5)
    np.testing.assert_allclose(a.model.w, b.model.w, atol=1e-10)

    # Now diverge the override on a single step
    a.step(I_stimulus=0.5)
    b.step(I_stimulus=0.5, sigma_social_override=np.zeros(a.N))
    # Plasticity (and thus w) should still match — uses the *real* sigma_social.
    np.testing.assert_allclose(a.model.w, b.model.w, rtol=1e-6, atol=1e-10)


def test_override_propagates_through_topology_step():
    """Mem4Network.step must forward sigma_social_override to model.step()."""
    import inspect
    src = inspect.getsource(Mem4Network.step)
    assert "sigma_social_override" in src
    assert "self.model.step" in src
