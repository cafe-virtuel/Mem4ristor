"""Invariant u_clamp (27/07/2026, accord explicite de Julien).

Contexte : le commit cf3e059 a ajoute un re-clippage de u en TOUTE FIN de step(), apres
l'ART et apres le watchdog. Mesure du 27/07 (9 configurations, worktree sur cf3e059^) :
ce re-clippage est bit-a-bit neutre partout ou la borne haute de u_clamp vaut 1.0 --
y compris ART soft, ART hard, watchdog, complex_doubt, heretiques dynamiques.

Il ne mord que si u_clamp[1] < 1.0, parce que l'ART borne en dur a 1.0 (np.minimum(..., 1.0))
et que le watchdog ecrit u_fou=0.9 sans consulter u_clamp.

Ces tests gravent DEUX choses, pour qu'elles cessent d'etre un accident de l'ordre des lignes :
  (a) u_clamp a la priorite : u reste dans ses bornes en fin de pas, ART et watchdog compris ;
  (c) une configuration ou ce choix ecraserait SILENCIEUSEMENT une intention du watchdog
      est refusee a la construction plutot que subie.

Limite connue et assumee : _validate_config() est appele par __init__. Un script qui mute
model.cfg APRES construction (plusieurs POCs le font) contourne la garde -- l'invariant (a)
reste vrai, seul l'avertissement (c) est manque.
"""
import numpy as np
import pytest

from mem4ristor.topology import Mem4Network
from mem4ristor.dynamics import Mem4ristorV3
from mem4ristor.graph_utils import make_lattice_adj


def _run(cfg_extra, steps=120, size=8, seed=1234):
    """Fait tourner un reseau avec une matrice d'adjacence EXPLICITE.

    Sans adjacence explicite, Mem4Network utilise le stencil et ne pose jamais
    _adj_matrix : le chemin ART soft/hard n'est alors jamais atteint (piege rencontre
    en construisant la mesure du 27/07).
    """
    net = Mem4Network(size=size, seed=seed, adjacency_matrix=make_lattice_adj(size, periodic=True))
    net.model.cfg = net.model._deep_merge(net.model.cfg, cfg_extra)
    for _ in range(steps):
        net.step(I_stimulus=0.5)
    return net.model.u


ART_HARD = {'topological_regulation': {'enabled': True, 'mode': 'hard', 'u_min': 0.05,
                                       'rigid_threshold': 0.7, 'alpha_art_hard': 0.25}}
ART_SOFT = {'topological_regulation': {'enabled': True, 'mode': 'soft', 'u_min': 0.05,
                                       'rigid_threshold': 0.7, 'alpha_art_soft': 0.15}}
WATCHDOG = {'consolidation_watchdog': {'enabled': True, 't_explore': 30,
                                       't_consolidate': 40, 'u_sage': 0.05, 'u_fou': 0.9}}


@pytest.mark.parametrize("label,extra", [
    ("art_hard", ART_HARD),
    ("art_soft", ART_SOFT),
    ("watchdog", WATCHDOG),
])
def test_u_stays_within_clamp_default(label, extra):
    """(a) Avec u_clamp par defaut, u reste dans [0, 1] en fin de pas."""
    u = _run(extra)
    assert np.all(u >= 0.0), f"{label}: u est passe sous u_clamp[0]"
    assert np.all(u <= 1.0), f"{label}: u a depasse u_clamp[1]"


@pytest.mark.parametrize("label,extra", [
    ("art_hard", ART_HARD),
    ("art_soft", ART_SOFT),
])
def test_u_stays_within_narrow_clamp(label, extra):
    """(a) Avec une borne haute resserree, l'ART ne peut pas faire sortir u de u_clamp.

    C'est le seul cas ou le re-clippage final de cf3e059 mord reellement : l'ART borne
    a 1.0 en dur, donc sans lui u monterait au-dessus de 0.5.
    """
    cfg = dict(extra)
    cfg['doubt'] = {'u_clamp': [0.0, 0.5]}
    u = _run(cfg)
    assert np.all(u <= 0.5 + 1e-12), f"{label}: u a depasse une borne haute resserree"
    assert np.all(u >= 0.0), f"{label}: u est passe sous u_clamp[0]"


def test_validate_rejects_watchdog_kick_above_clamp():
    """(c) u_fou hors de u_clamp est refuse : le KICK serait ecrase en silence."""
    with pytest.raises(ValueError, match="u_fou"):
        Mem4ristorV3(config={'doubt': {'u_clamp': [0.0, 0.5]},
                             'consolidation_watchdog': {'enabled': True, 'u_fou': 0.9,
                                                        'u_sage': 0.05}})


def test_validate_rejects_watchdog_sage_below_clamp():
    """(c) u_sage hors de u_clamp est refuse : la phase de consolidation serait ecrasee."""
    with pytest.raises(ValueError, match="u_sage"):
        Mem4ristorV3(config={'doubt': {'u_clamp': [0.2, 1.0]},
                             'consolidation_watchdog': {'enabled': True, 'u_fou': 0.9,
                                                        'u_sage': 0.05}})


def test_validate_accepts_coherent_watchdog():
    """(c) La configuration coherente (celle du 07/07) passe sans bruit."""
    m = Mem4ristorV3(config={'doubt': {'u_clamp': [0.0, 1.0]},
                             'consolidation_watchdog': {'enabled': True, 'u_fou': 0.9,
                                                        'u_sage': 0.05}})
    assert m.cfg['consolidation_watchdog']['enabled'] is True


def test_validate_silent_when_watchdog_disabled():
    """(c) La garde ne se declenche PAS quand le watchdog est eteint (defaut du depot).

    Sinon toute configuration a u_clamp resserre deviendrait impossible, alors que le
    watchdog ne tourne pas -- il n'y a alors aucune intention a ecraser.
    """
    m = Mem4ristorV3(config={'doubt': {'u_clamp': [0.0, 0.5]},
                             'consolidation_watchdog': {'enabled': False, 'u_fou': 0.9}})
    assert m.cfg['doubt']['u_clamp'] == [0.0, 0.5]
