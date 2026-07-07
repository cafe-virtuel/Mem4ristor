"""Tests du watchdog de consolidation (cycle bicameral natif, opt-in).

Le mecanisme resout le verrouillage en mode FOU (u sature > seuils SAGE bornes a 0.5).
Desactive par defaut => aucun effet sur les resultats existants.
"""
import numpy as np
from mem4ristor.topology import Mem4Network

STIM = np.zeros(100)
STIM[0] = 0.5
STIM[99] = -0.5


def test_watchdog_off_by_default():
    net = Mem4Network(size=10, heretic_ratio=0.0, seed=42)
    assert net.model.cfg['consolidation_watchdog']['enabled'] is False
    u = []
    for _ in range(400):
        net.step(I_stimulus=STIM)
        u.append(net.model.u.mean())
    # sans watchdog, u est stable : il ne cycle pas entre haut et bas
    assert np.array(u[200:]).std() < 0.1


def test_watchdog_cycles_when_enabled():
    net = Mem4Network(size=10, heretic_ratio=0.0, seed=42)
    net.model.cfg['consolidation_watchdog'] = {
        'enabled': True, 't_explore': 80, 't_consolidate': 80,
        'u_sage': 0.05, 'u_fou': 0.9,
    }
    u = []
    for _ in range(400):
        net.step(I_stimulus=STIM)
        u.append(net.model.u.mean())
    u = np.array(u[100:])
    assert u.std() > 0.15    # cycle d'amplitude reelle
    assert u.min() < 0.10    # atteint la chambre SAGE (consolidation)
    assert u.max() > 0.50    # atteint la chambre FOU (repulsion / exploration)


def test_watchdog_disabled_is_bit_identical_to_no_key():
    """OFF doit reproduire EXACTEMENT l'ancien comportement (cle absente)."""
    net1 = Mem4Network(size=10, heretic_ratio=0.0, seed=7)   # cle presente, enabled=False
    net2 = Mem4Network(size=10, heretic_ratio=0.0, seed=7)
    net2.model.cfg.pop('consolidation_watchdog', None)       # cle absente (ancien code)
    stim = np.full(100, 0.3)
    for _ in range(200):
        net1.step(I_stimulus=stim)
        net2.step(I_stimulus=stim)
    assert np.allclose(net1.model.v, net2.model.v, atol=1e-12)
    assert np.allclose(net1.model.u, net2.model.u, atol=1e-12)
