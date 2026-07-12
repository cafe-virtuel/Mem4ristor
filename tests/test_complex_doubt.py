"""Tests du doute complexe P10 (u_c dans le coeur, opt-in, 12/07/2026).

|u_c| = intensite (reste self.u, lu par tout le coeur) ; arg(u_c) = direction.
Desactive par defaut => bit-a-bit identique a l'ancien coeur (test 2).

Note de construction : chaque reseau recoit SA PROPRE matrice d'adjacence --
le rewiring pilote par le doute modifie la matrice IN-PLACE, donc un objet
partage entre deux reseaux les contaminerait mutuellement.
"""
import numpy as np
from mem4ristor.topology import Mem4Network
from mem4ristor.graph_utils import make_lattice_adj

STIM = np.zeros(100)
STIM[:20] = 0.6
STIM[80:] = -0.6


def fresh_net(seed):
    return Mem4Network(size=10, heretic_ratio=0.0, seed=seed,
                       adjacency_matrix=make_lattice_adj(10, periodic=True))


def test_off_by_default():
    net = fresh_net(11)
    assert net.model.cfg['complex_doubt']['enabled'] is False


def test_off_bit_identical_to_no_key():
    """OFF doit reproduire EXACTEMENT l'ancien comportement (cle absente)."""
    net1 = fresh_net(7)
    net2 = fresh_net(7)
    net2.model.cfg.pop('complex_doubt', None)          # cle absente (ancien code)
    for _ in range(200):
        net1.step(I_stimulus=STIM)
        net2.step(I_stimulus=STIM)
    assert np.allclose(net1.model.v, net2.model.v, atol=1e-12)
    assert np.allclose(net1.model.w, net2.model.w, atol=1e-12)
    assert np.allclose(net1.model.u, net2.model.u, atol=1e-12)


def test_on_sane_and_consistent():
    """ON : |u| dans [0,1], fini, et self.u == |self.u_c| apres chaque pas."""
    net = fresh_net(3)
    net.model.cfg['complex_doubt']['enabled'] = True
    for _ in range(300):
        net.step(I_stimulus=STIM)
    m = net.model
    assert np.all(np.isfinite(m.u)) and np.all(np.isfinite(np.abs(m.u_c)))
    assert np.all(m.u >= 0.0) and np.all(m.u <= 1.0)
    assert np.allclose(m.u, np.abs(m.u_c), atol=1e-12)


def test_on_differs_from_off():
    """ON doit faire quelque chose : la cible signee diverge du module des
    que le desaccord local change de signe quelque part."""
    net_on = fresh_net(5)
    net_on.model.cfg['complex_doubt']['enabled'] = True
    net_off = fresh_net(5)
    for _ in range(300):
        net_on.step(I_stimulus=STIM)
        net_off.step(I_stimulus=STIM)
    assert not np.allclose(net_on.model.u, net_off.model.u, atol=1e-6)


def test_interference_destructive():
    """L'essence de P10, testee UNITAIREMENT sur le canal d'interference :
    des doutes FORCES en damier (chaque noeud a ses 4 voisins en direction
    opposee -> champ social s_u = -u_c -> interference destructive maximale)
    doivent perdre du module nettement plus vite que des doutes ALIGNES
    (s_u = +u_c -> terme d'interference nul). Le rappel local vers la cible
    agit identiquement des deux cotes ; la difference isole gamma_int."""
    xs, ys = np.meshgrid(np.arange(10), np.arange(10), indexing="ij")
    checker = ((-1.0) ** (xs + ys)).flatten()

    def run(aligned):
        net = fresh_net(9)
        m = net.model
        m.cfg['complex_doubt']['enabled'] = True
        m.cfg['complex_doubt']['gamma_int'] = 0.5
        m.u = np.full(100, 0.5)
        m.u_c = (0.5 * (np.ones(100) if aligned else checker)).astype(complex)
        for _ in range(50):
            net.step(I_stimulus=0.0)
        return float(np.abs(m.u_c).mean())

    mag_aligned = run(aligned=True)
    mag_checker = run(aligned=False)
    assert mag_checker < mag_aligned - 0.05, (
        f"interference attendue : module moyen damier ({mag_checker:.3f}) "
        f"devrait etre nettement sous le module aligne ({mag_aligned:.3f})")
