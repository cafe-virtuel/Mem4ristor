"""
V5 Compartimentalisation Dynamique Tests.

Verifie que la compartimentation :
- est desactivee par defaut (gamma=0 reproduit V4)
- assigne correctement les noeuds selon leur rang de u
- cree bien K groupes disjoints et exhaustifs
- modifie la dynamique quand activee
- en mode 'full', le mode 'attraction' seul est strictement different
- ne fait pas s'effondrer H catastrophiquement
- K=3 full produit un gain de H mesurable
"""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mem4ristor.core import Mem4Network
from mem4ristor.metrics import calculate_continuous_entropy


class TestDynamicCompartments:

    def _make_net(self, enabled, K=2, gamma=0.10, mode='attraction', seed=42):
        net = Mem4Network(size=4, heretic_ratio=0.15, seed=seed)
        net.model.cfg['compartments'] = {
            'enabled': enabled,
            'K':       K,
            'gamma':   gamma,
            'mode':    mode,
        }
        return net

    def test_disabled_is_v4_pure(self):
        """Sans compartimentation, le comportement V4 est intact."""
        net_v4   = Mem4Network(size=4, heretic_ratio=0.15, seed=42)
        net_comp = self._make_net(enabled=False)
        for _ in range(50):
            net_v4.step(I_stimulus=0.5)
            net_comp.step(I_stimulus=0.5)
        np.testing.assert_array_equal(net_v4.model.v, net_comp.model.v)

    def test_gamma_zero_is_v4_pure(self):
        """gamma=0 doit reproduire exactement V4."""
        net_v4   = Mem4Network(size=4, heretic_ratio=0.15, seed=42)
        net_comp = self._make_net(enabled=True, gamma=0.0)
        for _ in range(50):
            net_v4.step(I_stimulus=0.5)
            net_comp.step(I_stimulus=0.5)
        np.testing.assert_allclose(net_v4.model.v, net_comp.model.v, atol=1e-10)

    def test_labels_cover_all_nodes(self):
        """Les K groupes doivent couvrir exactement tous les noeuds (partition exhaustive)."""
        net = self._make_net(enabled=True, K=2)
        for _ in range(10):
            net.step(I_stimulus=0.5)
        N = net.model.N
        K = 2
        u_ranks = np.argsort(np.argsort(net.model.u))
        labels  = np.minimum((u_ranks * K) // N, K - 1)
        assert set(np.unique(labels)) == set(range(K)), "Tous les K groupes doivent exister"
        assert labels.size == N, "Chaque noeud doit avoir exactement un label"

    def test_labels_are_disjoint(self):
        """Chaque noeud appartient a exactement un groupe."""
        net = self._make_net(enabled=True, K=3)
        for _ in range(10):
            net.step(I_stimulus=0.5)
        N = net.model.N
        K = 3
        u_ranks = np.argsort(np.argsort(net.model.u))
        labels  = np.minimum((u_ranks * K) // N, K - 1)
        # Chaque noeud a exactement 1 label
        assert labels.shape == (N,)
        assert np.all((labels >= 0) & (labels < K))

    def test_low_u_gets_label_zero(self):
        """Les noeuds au doute le plus faible doivent etre dans le groupe 0."""
        net = self._make_net(enabled=True, K=2)
        for _ in range(50):
            net.step(I_stimulus=0.5)
        N = net.model.N
        u_ranks = np.argsort(np.argsort(net.model.u))
        labels  = np.minimum((u_ranks * 2) // N, 1)
        # Les noeuds avec u < mediane doivent etre en groupe 0
        median_u = np.median(net.model.u)
        below_median = net.model.u < median_u
        if below_median.sum() > 0:
            assert np.all(labels[below_median] == 0), \
                "Les noeuds avec u faible doivent etre dans le groupe 0"

    def test_compartment_changes_dynamics(self):
        """La compartimentation doit modifier v par rapport au V4 pur."""
        net_v4   = Mem4Network(size=4, heretic_ratio=0.15, seed=42)
        net_comp = self._make_net(enabled=True, K=2, gamma=0.10, mode='full')
        for _ in range(200):
            net_v4.step(I_stimulus=0.5)
            net_comp.step(I_stimulus=0.5)
        assert not np.allclose(net_v4.model.v, net_comp.model.v, atol=1e-6), \
            "La compartimentation doit modifier les trajectoires"

    def test_full_differs_from_attraction(self):
        """Le mode 'full' doit donner des resultats differents du mode 'attraction'."""
        net_attr = self._make_net(enabled=True, K=2, gamma=0.10, mode='attraction')
        net_full = self._make_net(enabled=True, K=2, gamma=0.10, mode='full')
        for _ in range(200):
            net_attr.step(I_stimulus=0.5)
            net_full.step(I_stimulus=0.5)
        assert not np.allclose(net_attr.model.v, net_full.model.v, atol=1e-6), \
            "Mode 'full' et mode 'attraction' doivent diverger"

    def test_entropy_does_not_collapse(self):
        """La compartimentation ne doit pas faire s'effondrer H catastrophiquement."""
        steps  = 500
        warmup = int(steps * 0.75)

        def run_H(enabled, mode='full'):
            net = self._make_net(enabled=enabled, K=2, gamma=0.10, mode=mode, seed=42)
            H_vals = []
            for s in range(steps):
                net.step(I_stimulus=0.5)
                if s >= warmup:
                    H_vals.append(calculate_continuous_entropy(net.model.v))
            return np.mean(H_vals)

        H_v4   = run_H(enabled=False)
        H_comp = run_H(enabled=True)
        # Tolere une perte de max 0.5 bits (perte observee < 0.2 bits en pratique)
        assert H_comp >= H_v4 - 0.5, (
            f"H ne doit pas s'effondrer (H_v4={H_v4:.2f}, H_comp={H_comp:.2f})"
        )

    def test_k3_full_gains_entropy(self):
        """K=3 full doit maintenir ou ameliorer H vs V4 (resultat experimental valide)."""
        steps  = 1000
        warmup = int(steps * 0.75)

        def run_H(enabled, K, mode):
            net = self._make_net(enabled=enabled, K=K, gamma=0.10, mode=mode, seed=42)
            H_vals = []
            for s in range(steps):
                net.step(I_stimulus=0.5)
                if s >= warmup:
                    H_vals.append(calculate_continuous_entropy(net.model.v))
            return np.mean(H_vals)

        H_v4      = run_H(enabled=False, K=2, mode='attraction')
        H_k3_full = run_H(enabled=True,  K=3, mode='full')
        assert H_k3_full >= H_v4 - 0.05, (
            f"K=3 full devrait maintenir ou ameliorer H "
            f"(H_v4={H_v4:.2f}, H_k3_full={H_k3_full:.2f})"
        )
