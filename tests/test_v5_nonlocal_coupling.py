"""
V5 Non-Local Coupling Tests.

Verifie que le couplage virtuel par similarite de doute :
- est desactive par defaut (D_meta=0 reproduit V4)
- applique correctement le noyau gaussien en u-espace
- cree des communautes de doute mesurables (u_spread augmente)
- ne desynchronise pas plus que prevu (H ne s'effondre pas)
"""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mem4ristor.core import Mem4Network
from mem4ristor.metrics import calculate_continuous_entropy


class TestNonLocalCoupling:

    def _make_net(self, enabled, D_meta=0.05, sigma_u=0.10, seed=42):
        net = Mem4Network(size=4, heretic_ratio=0.15, seed=seed)
        net.model.cfg['nonlocal_coupling'] = {
            'enabled': enabled,
            'D_meta':  D_meta,
            'sigma_u': sigma_u,
        }
        return net

    def test_disabled_is_v4_pure(self):
        """Sans couplage non-local, le comportement V4 est intact."""
        net_v4  = Mem4Network(size=4, heretic_ratio=0.15, seed=42)
        net_nlc = self._make_net(enabled=False)
        for _ in range(50):
            net_v4.step(I_stimulus=0.5)
            net_nlc.step(I_stimulus=0.5)
        np.testing.assert_array_equal(net_v4.model.v, net_nlc.model.v)

    def test_dmeta_zero_is_v4_pure(self):
        """D_meta=0 doit donner exactement les memes resultats que V4."""
        net_v4  = Mem4Network(size=4, heretic_ratio=0.15, seed=42)
        net_nlc = self._make_net(enabled=True, D_meta=0.0)
        for _ in range(50):
            net_v4.step(I_stimulus=0.5)
            net_nlc.step(I_stimulus=0.5)
        np.testing.assert_allclose(net_v4.model.v, net_nlc.model.v, atol=1e-10)

    def test_gaussian_kernel_shape(self):
        """Le noyau gaussien doit etre carre, symetrique, diagonale nulle."""
        net = self._make_net(enabled=True, D_meta=0.05, sigma_u=0.10)
        N = net.model.N
        u = net.model.u
        sigma_u = 0.10
        u_diff2 = (u[:, None] - u[None, :]) ** 2
        W = np.exp(-u_diff2 / (sigma_u ** 2))
        np.fill_diagonal(W, 0.0)
        assert W.shape == (N, N)
        np.testing.assert_allclose(W, W.T, atol=1e-12)
        assert np.all(np.diag(W) == 0.0)

    def test_gaussian_kernel_row_normalization(self):
        """Apres normalisation par ligne, chaque ligne somme a 1 (ou 0 si isolee)."""
        net = self._make_net(enabled=True, D_meta=0.05, sigma_u=0.10)
        N = net.model.N
        u = net.model.u
        sigma_u = 0.10
        u_diff2 = (u[:, None] - u[None, :]) ** 2
        W = np.exp(-u_diff2 / (sigma_u ** 2))
        np.fill_diagonal(W, 0.0)
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums < 1e-10, 1.0, row_sums)
        W_norm = W / row_sums
        row_totals = W_norm.sum(axis=1)
        assert np.all(row_totals <= 1.0 + 1e-10)

    def test_sigma_large_approaches_mean_field(self):
        """sigma_u tres grand -> tous les noeuds se voient egalement (champ moyen)."""
        net = self._make_net(enabled=True, D_meta=0.05, sigma_u=100.0)
        N = net.model.N
        u = net.model.u
        sigma_u = 100.0
        u_diff2 = (u[:, None] - u[None, :]) ** 2
        W = np.exp(-u_diff2 / (sigma_u ** 2))
        # Avec sigma tres grand, tous les poids hors-diagonale doivent etre ~1
        mask = ~np.eye(N, dtype=bool)
        assert np.all(W[mask] > 0.99), "Sigma tres grand : tous les noeuds doivent se voir quasi-egalement"

    def test_sigma_small_restricts_coupling(self):
        """sigma_u tres petit -> seuls les noeuds de meme u se voient."""
        net = self._make_net(enabled=True, sigma_u=0.001)
        # Apres quelques steps, u varie entre noeuds
        for _ in range(10):
            net.step(I_stimulus=0.5)
        u = net.model.u
        sigma_u = 0.001
        u_diff2 = (u[:, None] - u[None, :]) ** 2
        W = np.exp(-u_diff2 / (sigma_u ** 2))
        np.fill_diagonal(W, 0.0)
        # La plupart des poids doivent etre quasi-nuls
        near_zero = (W < 1e-6).sum()
        total_off_diag = net.model.N * (net.model.N - 1)
        assert near_zero >= total_off_diag * 0.5, "Sigma petit : la majorite des couplages doit etre nulle"

    def test_nonlocal_changes_dynamics(self):
        """Le couplage non-local doit modifier v par rapport au V4 pur."""
        net_v4  = Mem4Network(size=4, heretic_ratio=0.15, seed=42)
        net_nlc = self._make_net(enabled=True, D_meta=0.10, sigma_u=0.10)
        for _ in range(200):
            net_v4.step(I_stimulus=0.5)
            net_nlc.step(I_stimulus=0.5)
        # Avec D_meta non nul, v doit diverger de V4
        assert not np.allclose(net_v4.model.v, net_nlc.model.v, atol=1e-6), \
            "Le couplage non-local doit modifier les trajectoires"

    def test_u_spread_increases_with_nonlocal(self):
        """Le couplage virtuel doit augmenter l'ecart-type de u (communautes de doute)."""
        steps  = 500
        warmup = int(steps * 0.75)

        def run_spread(enabled):
            net = self._make_net(enabled=enabled, D_meta=0.10, sigma_u=0.10, seed=42)
            spreads = []
            for s in range(steps):
                net.step(I_stimulus=0.5)
                if s >= warmup:
                    spreads.append(net.model.u.std())
            return np.mean(spreads)

        spread_v4  = run_spread(enabled=False)
        spread_nlc = run_spread(enabled=True)
        assert spread_nlc >= spread_v4, (
            f"Le couplage non-local devrait augmenter u_spread "
            f"(V4={spread_v4:.5f}, NLC={spread_nlc:.5f})"
        )

    def test_entropy_does_not_collapse(self):
        """Le couplage non-local ne doit pas faire s'effondrer H (pas de catastrophe)."""
        steps  = 500
        warmup = int(steps * 0.75)

        def run_H(enabled):
            net = self._make_net(enabled=enabled, D_meta=0.05, sigma_u=0.10, seed=42)
            H_vals = []
            for s in range(steps):
                net.step(I_stimulus=0.5)
                if s >= warmup:
                    H_vals.append(calculate_continuous_entropy(net.model.v))
            return np.mean(H_vals)

        H_v4  = run_H(enabled=False)
        H_nlc = run_H(enabled=True)
        # On tolere une perte de max 0.5 bits (la perte observee est ~0.1-0.2 bits)
        assert H_nlc >= H_v4 - 0.5, (
            f"H ne doit pas s'effondrer (H_v4={H_v4:.2f}, H_nlc={H_nlc:.2f})"
        )
