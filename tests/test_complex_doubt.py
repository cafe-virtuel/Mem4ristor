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


def test_omega_scalar_equals_uniform_array():
    """P10 (13/07) : omega_u accepte desormais un array de taille N en plus du
    scalaire. Le chemin scalaire doit rester bit-a-bit identique -- un array
    UNIFORME (meme valeur partout) doit reproduire EXACTEMENT le resultat du
    scalaire equivalent."""
    def run(omega_value):
        net = fresh_net(4)
        m = net.model
        m.cfg['complex_doubt']['enabled'] = True
        m.cfg['complex_doubt']['omega_u'] = omega_value
        for _ in range(300):
            net.step(I_stimulus=STIM)
        return m.v.copy(), m.w.copy(), m.u.copy(), m.u_c.copy()

    v_s, w_s, u_s, uc_s = run(0.03)
    v_a, w_a, u_a, uc_a = run(np.full(100, 0.03))
    assert np.allclose(v_s, v_a, atol=1e-12)
    assert np.allclose(w_s, w_a, atol=1e-12)
    assert np.allclose(u_s, u_a, atol=1e-12)
    assert np.allclose(uc_s, uc_a, atol=1e-12)


def _circular_mean(angles):
    return np.angle(np.mean(np.exp(1j * angles)))


def _circular_gap(phase, mask):
    a = _circular_mean(phase[mask])
    b = _circular_mean(phase[~mask])
    return abs(np.angle(np.exp(1j * (a - b))))  # difference d'angle repliee dans [-pi, pi]


def test_omega_per_group_diverges():
    """Rotation PAR GROUPE (13/07, extension du coeur, accord explicite de
    Julien) : deux groupes a des omega_u distincts doivent developper des
    phases differentes -- l'array n'est pas juste tolere, il fait quelque
    chose de nouveau par rapport a un omega_u uniforme. Stimulus NEUTRE
    (symetrique, pas de structure spatiale) pour isoler l'effet de omega_u ;
    ecart mesure en moyenne CIRCULAIRE (les angles s'enroulent autour de
    +/-pi, une moyenne arithmetique naive fausse la mesure). Bruit stochastique
    (sigma_v, defaut 0.05) coupe : sur 100 noeuds/50 par groupe le bruit i.i.d.
    fait deriver gap_uniform de facon non monotone (verifie manuellement), ce
    test isole le mecanisme DETERMINISTE omega_u, pas une propriete statistique
    -- cf. B1/B4 pour les comparaisons multi-seeds avec bruit."""
    def run(omega_array):
        net = fresh_net(6)
        m = net.model
        m.cfg['noise']['sigma_v'] = 0.0
        m.cfg['complex_doubt']['enabled'] = True
        m.cfg['complex_doubt']['omega_u'] = omega_array
        m.u = np.full(100, 0.5)
        m.u_c = np.full(100, 0.5 + 0.3j)
        neutral_stim = np.full(100, 0.2)
        for _ in range(400):
            net.step(I_stimulus=neutral_stim)
        return np.angle(m.u_c)

    group = np.zeros(100, dtype=bool)
    group[50:] = True  # groupe B = seconde moitie du lattice 10x10

    phase_uniform = run(np.zeros(100))
    phase_grouped = run(np.where(group, 0.02, 0.0))

    gap_uniform = _circular_gap(phase_uniform, group)
    gap_grouped = _circular_gap(phase_grouped, group)
    assert gap_grouped > gap_uniform + 0.15, (
        f"ecart de phase inter-groupe attendu sous omega_u distinct : "
        f"grouped={gap_grouped:.3f} devrait depasser uniform={gap_uniform:.3f}")
