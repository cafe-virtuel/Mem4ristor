"""Garde-fou P5a (12/07/2026) : get_spectral_gap refuse un Laplacien non symetrique.

Mine documentee par le Mur de Planck (attaque 1, PLANCK_WALL_REPORT.md) :
eigh/eigsh assument la symetrie ; sur un graphe DIRIGE le gap retourne etait
faux (809% d'erreur demontree) et silencieux. Le garde-fou echoue bruyamment.
Aucun effet sur les graphes actuels du projet (tous non-diriges).
"""
import numpy as np
import pytest
from mem4ristor.topology import Mem4Network
from mem4ristor.graph_utils import make_lattice_adj


def test_symmetric_graph_unchanged():
    """Sur un graphe non-dirige, le gap est calcule comme avant (valeur de
    reference = 2e plus petite valeur propre du Laplacien, calcul independant)."""
    adj = make_lattice_adj(10, periodic=True)
    net = Mem4Network(size=10, heretic_ratio=0.0, seed=3, adjacency_matrix=adj)
    gap = net.get_spectral_gap()
    deg = np.asarray(adj).sum(axis=1)
    L_ref = np.diag(deg) - np.asarray(adj, dtype=float)
    vals = np.sort(np.linalg.eigvalsh(L_ref))
    assert np.isclose(gap, vals[1], atol=1e-8)


def test_directed_graph_raises():
    """Sur un graphe dirige (adjacency non symetrique), l'appel doit refuser
    de retourner une valeur silencieusement fausse."""
    adj = np.asarray(make_lattice_adj(10, periodic=True), dtype=float).copy()
    # Retirer un sens d'une arete existante -> graphe dirige
    i, j = np.argwhere(adj > 0)[0]
    adj[i, j] = 0.0
    assert not np.allclose(adj, adj.T)
    net = Mem4Network(size=10, heretic_ratio=0.0, seed=3, adjacency_matrix=adj)
    with pytest.raises(ValueError, match="not symmetric"):
        net.get_spectral_gap()
