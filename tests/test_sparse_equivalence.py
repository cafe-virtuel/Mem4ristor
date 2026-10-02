"""
Equivalence des optimisations de performance du 02/10/2026.

Deux calculs ont ete accelerés sans changer la physique :
- ART : la matrice d'adjacence creuse n'est plus convertie en dense ;
- calculate_pairwise_synchrony : seules les paires echantillonnees sont calculees.

Ces tests comparent le nouveau code a l'ANCIENNE formule, recopiee ici telle
quelle. Si un jour ils divergent, l'optimisation a change le resultat.
"""

import sys
import os

import numpy as np
import pytest
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mem4ristor.dynamics import Mem4ristorV3
from mem4ristor.metrics import calculate_pairwise_synchrony


def _random_adj(n, mean_degree=4, seed=0):
    rng = np.random.RandomState(seed)
    a = (rng.rand(n, n) < mean_degree / n).astype(float)
    a = np.triu(a, 1)
    return a + a.T


def _art_reference(u, adj_dense, art):
    """Ancien calcul ART (avant le 02/10/2026), chemin dense."""
    u = u.copy()
    rigid_thr = float(art.get('rigid_threshold', 0.7))
    rigidity = 1.0 - u
    degree = adj_dense.sum(axis=1)
    degree = np.where(degree < 1.0, 1.0, degree)
    if art['mode'] == 'soft':
        mean_rig = (adj_dense @ rigidity) / degree
        pressure = mean_rig > rigid_thr
        factor = float(art.get('alpha_art_soft', 0.15))
        u[pressure] = np.minimum(u[pressure] * (1.0 + factor), 1.0)
    else:
        n_rigid = adj_dense @ (rigidity > rigid_thr).astype(float)
        ratio = n_rigid / degree
        pressure = ratio > 0
        factor = float(art.get('alpha_art_hard', 0.25))
        u[pressure] = np.minimum(u[pressure] * (1.0 + factor * ratio[pressure]), 1.0)
    return np.maximum(u, float(art.get('u_min', 0.05)))


def _run_steps(adj, mode, n, steps=30):
    m = Mem4ristorV3(seed=0)
    m._initialize_params(n)
    m.cfg['topological_regulation'] = {
        'enabled': True, 'mode': mode, 'rigid_threshold': 0.5, 'u_min': 0.05,
    }
    m._adj_matrix = adj
    for _ in range(steps):
        m.step(I_stimulus=0.0)
    return m


@pytest.mark.parametrize('mode', ['soft', 'hard'])
def test_art_sparse_identical_to_dense(mode):
    """Creux et dense donnent exactement la meme trajectoire de u."""
    n = 300
    a = _random_adj(n)
    m_sparse = _run_steps(sp.csr_matrix(a), mode, n)
    m_dense = _run_steps(a, mode, n)
    np.testing.assert_array_equal(m_sparse.u, m_dense.u)
    np.testing.assert_array_equal(m_sparse.v, m_dense.v)


@pytest.mark.parametrize('mode', ['soft', 'hard'])
def test_art_matches_previous_formula(mode):
    """Une etape ART creuse reproduit l'ancienne formule dense."""
    n = 200
    a = _random_adj(n, seed=1)
    art = {'enabled': True, 'mode': mode, 'rigid_threshold': 0.5, 'u_min': 0.05}
    rng = np.random.RandomState(3)
    u0 = rng.uniform(0.05, 0.95, size=n)

    m = Mem4ristorV3(seed=0)
    m._initialize_params(n)
    m.cfg['topological_regulation'] = dict(art)
    m._adj_matrix = sp.csr_matrix(a)
    # Isoler l'ART : on compare u juste avant et juste apres le bloc ART
    # en reproduisant l'etape sans ART sur une copie.
    m_ref = Mem4ristorV3(seed=0)
    m_ref._initialize_params(n)
    m_ref.cfg['topological_regulation'] = {'enabled': False}
    for model in (m, m_ref):
        model.u = u0.copy()
        model.step(I_stimulus=0.0)

    expected = _art_reference(m_ref.u, a, art)
    np.testing.assert_allclose(m.u, expected, rtol=0, atol=1e-15)


@pytest.mark.parametrize('n', [10, 80, 400])
def test_pairwise_synchrony_matches_einsum(n):
    """Le calcul par paires reproduit l'ancien einsum N x N."""
    rng = np.random.RandomState(n)
    v = rng.randn(500, n) + 0.3 * np.sin(np.linspace(0, 20, 500))[:, None]

    # Ancienne formule, recopiee depuis metrics.py (avant le 02/10/2026)
    T, N = v.shape
    rs = np.random.RandomState(0)
    if N * (N - 1) // 2 <= 2000:
        i_idx, j_idx = np.triu_indices(N, k=1)
    else:
        pairs = set()
        while len(pairs) < 2000:
            a, b = rs.randint(0, N, size=2)
            if a != b:
                pairs.add((min(a, b), max(a, b)))
        i_idx, j_idx = map(np.array, zip(*pairs))
    std = v.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    z = (v - v.mean(axis=0)) / std
    expected = float((np.einsum('ti,tj->ij', z, z)[i_idx, j_idx] / T).mean())

    assert calculate_pairwise_synchrony(v) == pytest.approx(expected, abs=1e-12)
