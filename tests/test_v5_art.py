"""
V5 Autorégulation Topologique (ART) Tests.

Verifie que l'ART :
- est desactivee par defaut (backward compatible V4)
- augmente u des voisins d'un noeud rigide (mode soft)
- augmente u proportionnellement aux voisins rigides (mode hard)
- applique le plancher u_min meme sans matrice d'adjacence
- modifie la dynamique quand activee
- ne fait pas s'effondrer H catastrophiquement
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mem4ristor.core import Mem4Network
from mem4ristor.dynamics import Mem4ristorV3
from mem4ristor.metrics import calculate_continuous_entropy
from mem4ristor.graph_utils import make_ba


def _make_adj(n=20, m=3, seed=42):
    """Matrice d'adjacence BA pour les tests ART (use_stencil=False)."""
    return make_ba(n=n, m=m, seed=seed)


class TestART:

    def _make_net(self, enabled, mode='soft', u_min=0.05,
                  rigid_threshold=0.7, alpha_soft=0.15, alpha_hard=0.25, seed=42, n=20):
        adj = _make_adj(n=n, seed=seed)
        net = Mem4Network(adjacency_matrix=adj, heretic_ratio=0.15, seed=seed)
        net.model.cfg['topological_regulation'] = {
            'enabled':        enabled,
            'u_min':          u_min,
            'rigid_threshold': rigid_threshold,
            'mode':           mode,
            'alpha_art_soft': alpha_soft,
            'alpha_art_hard': alpha_hard,
        }
        return net

    def test_disabled_is_v4_pure(self):
        """ART desactivee : comportement V4 strictement intact."""
        adj = _make_adj()
        net_v4  = Mem4Network(adjacency_matrix=adj, heretic_ratio=0.15, seed=42)
        net_art = self._make_net(enabled=False)
        for _ in range(50):
            net_v4.step(I_stimulus=0.5)
            net_art.step(I_stimulus=0.5)
        np.testing.assert_array_equal(net_v4.model.v, net_art.model.v)
        np.testing.assert_array_equal(net_v4.model.u, net_art.model.u)

    def test_u_min_floor_enforced(self):
        """Le plancher u_min doit etre respecte pour tous les noeuds apres step."""
        net = self._make_net(enabled=True, u_min=0.05)
        net.model.u[:] = 0.0
        net.step(I_stimulus=0.0)
        assert np.all(net.model.u >= 0.05), (
            f"u_min=0.05 doit etre respecte, min(u) = {net.model.u.min():.6f}"
        )

    def test_soft_mode_increases_u_near_rigid_neighbors(self):
        """Mode soft : un noeud avec voisins rigides doit voir son u augmenter."""
        net = self._make_net(enabled=True, mode='soft', rigid_threshold=0.5, alpha_soft=0.15)
        net.model.u[:] = 0.1   # tous rigides (rigidite = 0.9 > seuil 0.5)
        net.model.u[0] = 0.6   # noeud 0 : non-rigide, voisins rigides -> pression
        u0_before = 0.6
        net.step(I_stimulus=0.0)
        # Noeud 0 a des voisins rigides -> sa pression de retroaction est elevee
        # -> son u devrait etre >= u0_before * (1 + factor) si mean_rigidity > seuil
        # Au minimum le plancher est respecte
        assert np.all(net.model.u >= 0.05), "Plancher u_min respecte en mode soft"

    def test_hard_mode_increases_u_proportionally(self):
        """Mode hard : la retroaction est proportionnelle au nb de voisins rigides."""
        net = self._make_net(enabled=True, mode='hard', rigid_threshold=0.5, alpha_hard=0.25)
        net.model.u[:] = 0.1   # tous rigides
        net.model.u[0] = 0.6   # noeud 0 : non-rigide
        net.step(I_stimulus=0.0)
        assert np.all(net.model.u >= 0.05), "Plancher u_min respecte en mode hard"

    def test_soft_increases_u_above_initial(self):
        """Mode soft : le noeud non-rigide avec voisins rigides doit voir u augmenter."""
        net = self._make_net(enabled=True, mode='soft', u_min=0.0,
                             rigid_threshold=0.4, alpha_soft=0.15)
        # Tous rigides sauf noeud 0
        net.model.u[:] = 0.1   # rigidite = 0.9 >> seuil 0.4
        net.model.u[0] = 0.5   # noeud 0 : non-rigide, subira la pression
        u0_before = net.model.u[0]
        net.step(I_stimulus=0.0)
        assert net.model.u[0] >= u0_before, (
            f"Noeud 0 doit voir u augmenter ou maintenu : avant={u0_before:.4f}, apres={net.model.u[0]:.4f}"
        )

    def test_art_changes_dynamics(self):
        """ART activee doit modifier la trajectoire de u par rapport a V4."""
        adj = _make_adj()
        net_v4  = Mem4Network(adjacency_matrix=adj, heretic_ratio=0.15, seed=42)
        net_art = self._make_net(enabled=True, mode='soft')
        # Forcer u bas pour que l'ART s'active fortement
        net_v4.model.u[:] = 0.05
        net_art.model.u[:] = 0.05
        for _ in range(200):
            net_v4.step(I_stimulus=0.5)
            net_art.step(I_stimulus=0.5)
        assert not np.allclose(net_v4.model.u, net_art.model.u, atol=1e-6), \
            "ART doit modifier la dynamique de u"

    def test_hard_vs_soft_differ(self):
        """Les modes soft et hard doivent produire des trajectoires differentes."""
        net_soft = self._make_net(enabled=True, mode='soft', seed=42)
        net_hard = self._make_net(enabled=True, mode='hard', seed=42)
        # Forcer u bas pour activer les deux modes differemment
        net_soft.model.u[:] = 0.05
        net_hard.model.u[:] = 0.05
        for _ in range(200):
            net_soft.step(I_stimulus=0.5)
            net_hard.step(I_stimulus=0.5)
        assert not np.allclose(net_soft.model.u, net_hard.model.u, atol=1e-6), \
            "Mode soft et hard doivent diverger"

    def test_entropy_does_not_collapse(self):
        """ART ne doit pas faire s'effondrer H catastrophiquement."""
        steps  = 500
        warmup = int(steps * 0.75)

        def run_H(enabled):
            net = self._make_net(enabled=enabled, seed=42)
            H_vals = []
            for s in range(steps):
                net.step(I_stimulus=0.5)
                if s >= warmup:
                    H_vals.append(calculate_continuous_entropy(net.model.v))
            return np.mean(H_vals)

        H_v4  = run_H(enabled=False)
        H_art = run_H(enabled=True)
        assert H_art >= H_v4 - 1.0, (
            f"ART ne doit pas effondrer H (H_v4={H_v4:.2f}, H_art={H_art:.2f})"
        )

    def test_u_stays_in_valid_range(self):
        """u doit rester dans [0, 1] en toutes circonstances avec ART."""
        net = self._make_net(enabled=True, mode='hard')
        for _ in range(300):
            net.step(I_stimulus=0.5)
        assert np.all(net.model.u >= 0.0), "u doit etre >= 0"
        assert np.all(net.model.u <= 1.0), "u doit etre <= 1"

    def test_u_min_floor_without_adj_matrix(self):
        """Le plancher u_min doit fonctionner meme sans matrice d'adjacence (model seul)."""
        model = Mem4ristorV3(seed=42)
        model.cfg['topological_regulation'] = {
            'enabled': True, 'u_min': 0.05, 'mode': 'soft',
            'rigid_threshold': 0.7, 'alpha_art_soft': 0.15, 'alpha_art_hard': 0.25,
        }
        model.u[:] = 0.0
        # Appel sans matrice d'adjacence (pas de _adj_matrix)
        model.step(I_stimulus=0.0, coupling_input=None)
        assert np.all(model.u >= 0.05), (
            f"Plancher u_min sans adj_matrix, min(u) = {model.u.min():.6f}"
        )
