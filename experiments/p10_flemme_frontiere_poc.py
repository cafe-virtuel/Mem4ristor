#!/usr/bin/env python3
"""
P10 -- "LA FLEMME" : le cout de gamma_int est-il concentre a la FRONTIERE ?
=============================================================================
Cree : 2026-07-13 (Claude Sonnet 5, L'Ingenieur). Suite directe de
`p10_gamma_int_crosstalk_poc.py` (gamma_int=0 annule le crosstalk mesure au
niveau du GROUPE). Intuition de Julien, textuelle : "ce gamma_int c'est la
flemme en quelque sorte" -- au lieu de calculer honnetement sa propre cible
locale (k_u * laplacian_v, le desaccord REEL du noeud avec SES voisins v),
un noeud a gamma_int>0 se cale PARTIELLEMENT sur la moyenne complexe de ses
voisins u_c, quelle que soit la pertinence de ce qu'ils portent.

PREDICTION FALSIFIABLE (posee avant de lancer) : si c'est bien de la
"flemme" au sens propre -- copier le voisin plutot que de calculer soi-meme
-- le cout doit etre concentre aux noeuds A FRONTIERE avec le groupe B
(voisins directs qui portent un bit DIFFERENT, la moyenne les "flemme" vers
un melange faux), et NUL ou faible aux noeuds INTERIEURS (tous les 4
voisins du lattice sont dans le meme groupe ou idle -- la moyenne des
voisins EST deja informative pour le meme bit, "copier" ne coute rien).
Si le cout est UNIFORME (meme degradation interieur/frontiere), la lecture
"flemme localisee a la frontiere" est refutee -- gamma_int couterait pour
une autre raison (ex. un effet global de desaccord de phase, pas un
probleme de voisinage).

Protocole : reprend exactement p10_gamma_int_crosstalk_poc.py (2 groupes de
30 noeuds, lattice 10x10 periodique 4-connecte, D=1200, omega_B=0, 12 seeds
x 4 signes = 48 problemes) a gamma_int in {0.15 (defaut, crosstalk mesure),
0.0 (crosstalk annule)}. Nouveau : lecture PAR NOEUD (pas juste le vote
groupe) + classement de chaque noeud stimule par le nombre de voisins
lattice appartenant a L'AUTRE groupe (0 = interieur, >=1 = frontiere),
calcule sur l'adjacence REELLE de chaque probleme (l'appartenance aux
groupes change par seed).

Statut : exploratoire, hors preprint, aucune modification de dynamics.py.
Guardian doit rester 14/14. Sorties : figures/p10_flemme_frontiere_poc.csv + .png
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from mem4ristor.topology import Mem4Network  # noqa: E402
from mem4ristor.graph_utils import make_lattice_adj  # noqa: E402

FIG = ROOT / "figures"
SIDE, N = 10, 100
B_E = 0.8
GROUP_SIZE = 30
B_PULSE = 200
DELAY = 1200
SEEDS_PAIR = list(range(12))
SIGN_COMBOS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
GAMMA_VALUES = [0.15, 0.0]

ADJ = make_lattice_adj(SIDE, periodic=True)  # fixe : seule l'appartenance aux groupes varie par seed


def build_groups(seed):
    rng = np.random.RandomState(70000 + seed)
    both = rng.choice(N, size=2 * GROUP_SIZE, replace=False)
    mask_a = np.zeros(N, dtype=bool)
    mask_b = np.zeros(N, dtype=bool)
    mask_a[both[:GROUP_SIZE]] = True
    mask_b[both[GROUP_SIZE:]] = True
    idle = ~(mask_a | mask_b)
    return mask_a, mask_b, idle


def n_other_group_neighbors(node, other_mask):
    """Nombre de voisins lattice directs de `node` appartenant a other_mask."""
    return int(ADJ[node][other_mask].sum())


def run_pair(seed, b_a, b_b, gamma_int, delay=DELAY):
    mask_a, mask_b, idle = build_groups(seed)
    net = Mem4Network(size=SIDE, heretic_ratio=0.0, seed=seed * 10 + 1,
                       adjacency_matrix=make_lattice_adj(SIDE, periodic=True))
    m = net.model
    m.cfg['complex_doubt']['enabled'] = True
    m.cfg['complex_doubt']['gamma_int'] = gamma_int
    stim_vec = np.zeros(N)
    stim_vec[mask_a] = b_a * B_E
    stim_vec[mask_b] = b_b * B_E
    zero = np.zeros(N)
    for t in range(B_PULSE + delay):
        net.step(I_stimulus=stim_vec if t < B_PULSE else zero)
    return m.u_c.copy(), mask_a, mask_b, idle


def per_node_records(u_c, mask, other_mask, idle, b_true):
    """Pour chaque noeud du groupe `mask` : (frontiere?, correct?)."""
    idle_ref = float(np.real(u_c[idle]).mean())
    nodes = np.where(mask)[0]
    records = []
    for node in nodes:
        n_other = n_other_group_neighbors(node, other_mask)
        vote = 1 if (idle_ref - float(np.real(u_c[node]))) >= 0 else -1
        records.append((n_other > 0, int(vote == b_true)))
    return records


def sweep():
    print("=== \"LA FLEMME\" : le cout de gamma_int est-il concentre a la frontiere ? ===")
    print(f"(D={DELAY}, omega_B=0, {len(SEEDS_PAIR)} seeds x {len(SIGN_COMBOS)} signes)\n")
    all_rows = []
    for gamma_int in GAMMA_VALUES:
        t0 = time.time()
        interior_correct, boundary_correct = [], []
        n_boundary_nodes, n_interior_nodes = 0, 0
        for seed in SEEDS_PAIR:
            for b_a, b_b in SIGN_COMBOS:
                u_c, mask_a, mask_b, idle = run_pair(seed, b_a, b_b, gamma_int)
                for is_boundary, correct in per_node_records(u_c, mask_a, mask_b, idle, b_a):
                    (boundary_correct if is_boundary else interior_correct).append(correct)
                for is_boundary, correct in per_node_records(u_c, mask_b, mask_a, idle, b_b):
                    (boundary_correct if is_boundary else interior_correct).append(correct)
        n_int, n_bnd = len(interior_correct), len(boundary_correct)
        acc_int = float(np.mean(interior_correct)) if n_int else float("nan")
        acc_bnd = float(np.mean(boundary_correct)) if n_bnd else float("nan")
        se_int = float(np.std(interior_correct, ddof=1) / np.sqrt(n_int)) if n_int > 1 else float("nan")
        se_bnd = float(np.std(boundary_correct, ddof=1) / np.sqrt(n_bnd)) if n_bnd > 1 else float("nan")
        all_rows.append((gamma_int, acc_int, se_int, n_int, acc_bnd, se_bnd, n_bnd))
        print(f"gamma_int={gamma_int:<5} INTERIEUR (0 voisin autre-groupe, n={n_int:4d}) : "
              f"{acc_int:.3f} +/- {se_int:.3f}   "
              f"FRONTIERE (>=1 voisin autre-groupe, n={n_bnd:4d}) : {acc_bnd:.3f} +/- {se_bnd:.3f}   "
              f"[{time.time()-t0:.0f}s]")

    print("\n=== VERDICT ===")
    r_default = next(r for r in all_rows if r[0] == 0.15)
    r_zero = next(r for r in all_rows if r[0] == 0.0)
    gap_default = r_default[1] - r_default[4]  # interieur - frontiere, au defaut
    gap_zero = r_zero[1] - r_zero[4]
    print(f"Ecart interieur-frontiere au defaut (0.15) : {gap_default:+.3f}   "
          f"a gamma_int=0 : {gap_zero:+.3f}")
    if gap_default > 0.05 and gap_zero < gap_default - 0.05:
        print("  -> PREDICTION CONFIRMEE : le cout de gamma_int est concentre a la "
              "FRONTIERE (l'ecart interieur-frontiere existe au defaut et RETRECIT "
              "quand gamma_int->0). \"La flemme\" est localisee : copier un voisin "
              "en desaccord coute, copier un voisin d'accord ne coute rien.")
    elif gap_default <= 0.05:
        print("  -> Pas d'ecart interieur/frontiere net meme au defaut -- le cout de "
              "gamma_int n'est PAS localise au voisinage direct, la lecture \"flemme "
              "de proximite\" est REFUTEE telle quelle. Le mecanisme du crosstalk est "
              "plus diffus/global (propagation multi-pas au reseau entier) qu'un "
              "simple effet de premier voisin.")
    else:
        print("  -> Ecart interieur/frontiere present mais NE disparait PAS a "
              "gamma_int=0 -- \"la flemme\" existe mais n'explique pas a elle seule "
              "la disparition du crosstalk (autre mecanisme co-present).")

    FIG.mkdir(parents=True, exist_ok=True)
    with (FIG / "p10_flemme_frontiere_poc.csv").open("w", encoding="utf-8") as f:
        f.write("gamma_int,acc_interior,se_interior,n_interior,acc_boundary,se_boundary,n_boundary\n")
        for r in all_rows:
            f.write(",".join(f"{x:.6f}" if isinstance(x, float) else str(x) for x in r) + "\n")
    return all_rows


def main() -> int:
    t0 = time.time()
    rows = sweep()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.5, 4.6))
        labels = [f"gamma_int={r[0]}" for r in rows]
        x = np.arange(len(rows))
        w = 0.35
        ax.bar(x - w / 2, [r[1] for r in rows], w, yerr=[r[2] for r in rows],
               label="interieur (0 voisin autre-groupe)", color="#2ca02c", capsize=4)
        ax.bar(x + w / 2, [r[4] for r in rows], w, yerr=[r[5] for r in rows],
               label="frontiere (>=1 voisin autre-groupe)", color="#d62728", capsize=4)
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel("accuracy par noeud"); ax.set_ylim(0, 1.05)
        ax.set_title("\"La flemme\" : cout de gamma_int, interieur vs frontiere")
        ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(FIG / "p10_flemme_frontiere_poc.png", dpi=140)
        print(f"\n[png] {FIG / 'p10_flemme_frontiere_poc.png'}")
    except Exception as e:
        print(f"[png] skipped: {e}")

    print(f"\nWall time: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
