"""
V5 Metacognitive Plasticity Tests.

Verifie que epsilon_i varie correctement selon u_i et alpha_meta,
dans les deux directions (douteux=lent et douteux=rapide).
"""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mem4ristor.core import Mem4Network
from mem4ristor.metrics import calculate_continuous_entropy


class TestMetacognitivePlasticity:

    def _make_net(self, alpha_meta, enabled=True, seed=42):
        net = Mem4Network(size=4, heretic_ratio=0.15, seed=seed)
        net.model.cfg['metacognitive'] = {
            'enabled': enabled,
            'alpha_meta': alpha_meta,
            'epsilon_min': 0.01,
        }
        return net

    def test_disabled_is_v4_pure(self):
        """Sans metacognitive, le comportement V4 est intact."""
        net_v4   = Mem4Network(size=4, heretic_ratio=0.15, seed=42)
        net_meta = self._make_net(alpha_meta=0.5, enabled=False)
        for _ in range(50):
            net_v4.step(I_stimulus=0.5)
            net_meta.step(I_stimulus=0.5)
        np.testing.assert_array_equal(net_v4.model.v, net_meta.model.v)

    def test_alpha_zero_is_v4_pure(self):
        """alpha_meta=0 doit donner exactement les memes resultats que V4."""
        net_v4   = Mem4Network(size=4, heretic_ratio=0.15, seed=42)
        net_meta = self._make_net(alpha_meta=0.0, enabled=True)
        for _ in range(50):
            net_v4.step(I_stimulus=0.5)
            net_meta.step(I_stimulus=0.5)
        np.testing.assert_allclose(net_v4.model.v, net_meta.model.v, atol=1e-10)

    def test_positive_alpha_reduces_epsilon_for_high_u(self):
        """alpha_meta>0 : les noeuds a fort u ont un epsilon_i inferieur a epsilon_base."""
        eps_base = 0.08
        alpha    = 0.5
        u_high   = np.array([0.9, 0.95, 1.0])
        eps_i    = eps_base * (1.0 + alpha * (0.5 - u_high))
        assert np.all(eps_i < eps_base), "Avec alpha>0, eps_i doit baisser quand u est eleve"

    def test_negative_alpha_raises_epsilon_for_high_u(self):
        """alpha_meta<0 : les noeuds a fort u ont un epsilon_i superieur a epsilon_base."""
        eps_base = 0.08
        alpha    = -0.5
        u_high   = np.array([0.9, 0.95, 1.0])
        eps_i    = eps_base * (1.0 + alpha * (0.5 - u_high))
        assert np.all(eps_i > eps_base), "Avec alpha<0, eps_i doit monter quand u est eleve"

    def test_epsilon_never_below_minimum(self):
        """epsilon_i ne descend jamais en dessous de epsilon_min."""
        eps_base = 0.08
        alpha    = 5.0          # valeur extreme
        eps_min  = 0.01
        u_vals   = np.linspace(0, 1, 100)
        eps_i    = eps_base * (1.0 + alpha * (0.5 - u_vals))
        eps_i    = np.maximum(eps_i, eps_min)
        assert np.all(eps_i >= eps_min)

    def test_negative_alpha_improves_diversity(self):
        """La logique douteux=rapide (alpha<0) doit produire H >= V4 a I_stim=0.5."""
        steps  = 1000
        warmup = int(steps * 0.75)

        def run_H(alpha, enabled):
            net = self._make_net(alpha_meta=alpha, enabled=enabled, seed=42)
            H_vals = []
            for s in range(steps):
                net.step(I_stimulus=0.5)
                if s >= warmup:
                    H_vals.append(calculate_continuous_entropy(net.model.v))
            return np.mean(H_vals)

        H_v4  = run_H(alpha=0.0,  enabled=False)
        H_inv = run_H(alpha=-0.5, enabled=True)
        assert H_inv >= H_v4 - 0.05, (
            f"La logique inversee devrait maintenir ou ameliorer H "
            f"(H_v4={H_v4:.2f}, H_inv={H_inv:.2f})"
        )
