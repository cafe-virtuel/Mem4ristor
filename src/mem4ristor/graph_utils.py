"""
graph_utils.py — Générateurs de topologie canoniques pour Mem4ristor.

Source unique de vérité pour make_ba(), make_er(), make_lattice_adj().
Toutes les expériences experiments/p2_*.py doivent importer depuis ce module
plutôt que de réimplémenter localement (Faille D — Audit Manus 2026-04-25).

Usage :
    from mem4ristor.graph_utils import make_ba, make_er, make_lattice_adj
"""
import numpy as np


def make_ba(n: int, m: int, seed: int) -> np.ndarray:
    """
    Génère une matrice d'adjacence Barabási-Albert par attachement préférentiel.

    Paramètres
    ----------
    n    : nombre de nœuds
    m    : nombre de liens ajoutés par nouveau nœud
    seed : graine RNG (déterministe)

    Retourne
    --------
    adj : np.ndarray (n, n) float, symétrique, valeurs 0/1

    Lève
    ----
    ValueError si (n, m) est hors du domaine du générateur.

    Note (2026-08-05, audit externe constat 5.15) : ces gardes ne changent RIEN pour
    les appels valides. Sans elles, `m >= n` produisait un IndexError dans la clique
    initiale et `m > new_node` un « ValueError: a must be greater than 0 » de NumPy,
    tous deux à des dizaines de lignes du vrai problème — l'erreur désignait le
    symptôme, pas l'appel fautif.
    """
    if n < 2:
        raise ValueError(f"make_ba : n doit valoir au moins 2 (reçu n={n}).")
    if m < 1:
        raise ValueError(f"make_ba : m doit valoir au moins 1 (reçu m={m}).")
    if n < m + 1:
        raise ValueError(
            f"make_ba : n doit être >= m+1 pour que la clique initiale tienne "
            f"(reçu n={n}, m={m} → il faut n >= {m + 1})."
        )

    rng = np.random.RandomState(seed)
    adj = np.zeros((n, n), dtype=float)

    # Clique initiale des (m+1) premiers nœuds
    for i in range(m + 1):
        for j in range(i + 1, m + 1):
            adj[i, j] = adj[j, i] = 1.0

    degrees = adj.sum(axis=1)

    for new_node in range(m + 1, n):
        probs = degrees[:new_node] / degrees[:new_node].sum()
        targets = rng.choice(new_node, size=m, replace=False, p=probs)
        for t in targets:
            adj[new_node, t] = adj[t, new_node] = 1.0
        degrees = adj.sum(axis=1)

    return adj


def make_er(n: int, p: float, seed: int, ensure_connected: bool = True) -> np.ndarray:
    """
    Génère une matrice d'adjacence Erdős-Rényi G(n, p).

    Paramètres
    ----------
    n                : nombre de nœuds
    p                : probabilité d'arête
    seed             : graine RNG
    ensure_connected : si True, ajoute une arête aléatoire pour les nœuds isolés

    Retourne
    --------
    adj : np.ndarray (n, n) float, symétrique, valeurs 0/1

    Lève
    ----
    ValueError si n < 2 ou si p n'est pas une probabilité.

    Note (2026-08-05) : `p` hors [0, 1] ne levait aucune erreur — p < 0 rendait un
    graphe vide et p > 1 un graphe complet, tous deux silencieusement. Un balayage de
    densité mal borné produisait donc des points parfaitement plausibles.
    """
    if n < 2:
        raise ValueError(f"make_er : n doit valoir au moins 2 (reçu n={n}).")
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"make_er : p doit être une probabilité dans [0, 1] (reçu p={p}).")

    rng = np.random.RandomState(seed)
    adj = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            if rng.rand() < p:
                adj[i, j] = adj[j, i] = 1.0

    if ensure_connected:
        degrees = adj.sum(axis=1)
        for i in range(n):
            if degrees[i] == 0:
                j = rng.randint(0, n)
                while j == i:
                    j = rng.randint(0, n)
                adj[i, j] = adj[j, i] = 1.0

    return adj


def make_lattice_adj(size: int, periodic: bool = True) -> np.ndarray:
    """
    Génère la matrice d'adjacence d'un réseau lattice 2D (size × size).

    Paramètres
    ----------
    size     : côté du réseau (N = size²)
    periodic : si True, conditions aux limites périodiques (tore)

    Retourne
    --------
    adj : np.ndarray (size², size²) float, symétrique, valeurs 0/1
    """
    n = size * size
    adj = np.zeros((n, n), dtype=float)

    for i in range(size):
        for j in range(size):
            node = i * size + j
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if periodic:
                    ni %= size
                    nj %= size
                elif not (0 <= ni < size and 0 <= nj < size):
                    continue
                adj[node, ni * size + nj] = 1.0

    return adj


def make_directed(adj_undirected: np.ndarray, rule: str, rng: np.random.RandomState) -> np.ndarray:
    """
    Convertit une matrice d'adjacence non-dirigée en une matrice d'adjacence dirigée
    en orientant chaque arête selon une règle spécifique.

    Paramètres
    ----------
    adj_undirected : np.ndarray
        La matrice d'adjacence non-dirigée.
    rule : str
        La règle d'orientation : 'RANDOM', 'HUBS_LISTEN' ou 'HUBS_BROADCAST'.
    rng : np.random.RandomState
        Générateur de nombres aléatoires pour résoudre les cas d'égalité ou pour RANDOM.

    Retourne
    --------
    adj_directed : np.ndarray (n, n)
        La matrice d'adjacence dirigée.
    """
    n = adj_undirected.shape[0]
    total_deg = adj_undirected.sum(axis=1)
    A = np.zeros((n, n), dtype=float)
    ii, jj = np.where(np.triu(adj_undirected, k=1) > 0)
    for i, j in zip(ii, jj):
        if rule == 'RANDOM':
            listener = i if rng.rand() < 0.5 else j
        elif rule == 'HUBS_LISTEN':
            if total_deg[i] == total_deg[j]:
                listener = i if rng.rand() < 0.5 else j
            else:
                listener = i if total_deg[i] > total_deg[j] else j
        elif rule == 'HUBS_BROADCAST':
            if total_deg[i] == total_deg[j]:
                listener = i if rng.rand() < 0.5 else j
            else:
                listener = i if total_deg[i] < total_deg[j] else j
        else:
            raise ValueError(f"Unknown rule: {rule}")
        A[listener, (j if listener == i else i)] = 1.0
    return A
