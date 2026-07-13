#!/usr/bin/env python3
"""
P11 -- LE MAILLON MANQUANT : le raffinement iteratif fonctionne-t-il, ou
M4R reste-t-il englue sur sa premiere impression ?
=============================================================================
Cree : 2026-07-13 (Claude Sonnet 5, L'Ingenieur). Dernier maillon de la
chaine decrite par Julien (direction -> solveur -> M4R veille -> retour du
calcul -> raffinement -> direction plus precise -> solveur), jamais teste.
« ah oui c'est vrai je l'ai oublie celui la. »

POURQUOI SE MEFIER (pas une simple formalite) : P12 (12/07,
`b1d_stno_deceptive_poc.py`) a deja trouve que M4R a une CICATRICE -- un
conflit anterieur retarde la sortie de tromperie sur le substrat STNO
(+52% de temps de flip FULL vs FROZEN). Si cette cicatrice existe aussi
ici, le "raffinement" du plan de Julien pourrait etre plus lent ou moins
fiable qu'espere : une premiere lecture FAUSSE pourrait laisser une trace
qui resiste a la correction, au lieu de se corriger proprement.

TEST LE PLUS DUR (pas le plus favorable) : simule le cas ou la PREMIERE
lecture de M4R s'est trompee (ca arrive, cf. l'accuracy mesuree aujourd'hui
n'est jamais 100% en lecture faible), PUIS une preuve corrective arrive
(le "retour du solveur", plus forte et clarifiante, dans la vraie
direction). M4R corrige-t-il, et a quel prix compare a un reseau FRAIS qui
n'a jamais eu de premiere impression fausse a desapprendre ?

Conditions (memes T2, meme stimulus correctif fort -- seule la PRE-EXPOSITION
diverge) :
  - PRIME_FAUX : T1 pas de stimulus FAIBLE dans le MAUVAIS sens, PUIS T2 pas
    de stimulus FORT dans le BON sens (meme reseau, evolution continue).
  - FRAIS      : T1 pas SANS stimulus (neutre), PUIS les MEMES T2 pas de
    stimulus FORT dans le BON sens -- aucune fausse impression a corriger.
  - Cout de la cicatrice = accuracy(FRAIS) - accuracy(PRIME_FAUX), a T2
    fixe. Si > 0, corriger coute reellement quelque chose (la cicatrice
    existe ici aussi). Si ~0, le raffinement est GRATUIT -- bonne nouvelle
    pour l'architecture de Julien.

Statut : exploratoire, hors preprint, aucune modification de dynamics.py.
Guardian doit rester 14/14. Sorties : figures/p11_refinement_scar_poc.csv + .png
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
GROUP_SIZE = 30
B_E_WEAK = 0.3
B_E_STRONG = 0.8
SEEDS = list(range(30))
T2_VALUES = [50, 150, 300, 600]  # duree de la phase corrective : la cicatrice se referme-t-elle avec le temps ?
T1 = 200


def build_group(seed):
    rng = np.random.RandomState(91000 + seed)
    mask_nodes = rng.choice(N, size=GROUP_SIZE, replace=False)
    mask = np.zeros(N, dtype=bool)
    mask[mask_nodes] = True
    idle = ~mask
    return mask, idle


def decode(u_c, mask, idle):
    diff = u_c[idle].mean() - u_c[mask].mean()
    return 1 if float(np.real(diff)) >= 0 else -1


def run_primed(seed, b_true, t2):
    """T1 pas de FAUSSE direction (faible), PUIS t2 pas de vraie direction (forte)."""
    mask, idle = build_group(seed)
    net = Mem4Network(size=SIDE, heretic_ratio=0.0, seed=seed * 10 + 1,
                       adjacency_matrix=make_lattice_adj(SIDE, periodic=True))
    m = net.model
    m.cfg['complex_doubt']['enabled'] = True
    stim_wrong = np.zeros(N)
    stim_wrong[mask] = -b_true * B_E_WEAK
    stim_correct = np.zeros(N)
    stim_correct[mask] = b_true * B_E_STRONG
    for _ in range(T1):
        net.step(I_stimulus=stim_wrong)
    for _ in range(t2):
        net.step(I_stimulus=stim_correct)
    return decode(m.u_c, mask, idle)


def run_fresh(seed, b_true, t2):
    """T1 pas SANS stimulus (neutre), PUIS t2 pas de vraie direction (forte)."""
    mask, idle = build_group(seed)
    net = Mem4Network(size=SIDE, heretic_ratio=0.0, seed=seed * 10 + 1,
                       adjacency_matrix=make_lattice_adj(SIDE, periodic=True))
    m = net.model
    m.cfg['complex_doubt']['enabled'] = True
    zero = np.zeros(N)
    stim_correct = np.zeros(N)
    stim_correct[mask] = b_true * B_E_STRONG
    for _ in range(T1):
        net.step(I_stimulus=zero)
    for _ in range(t2):
        net.step(I_stimulus=stim_correct)
    return decode(m.u_c, mask, idle)


def sweep():
    print("=== LE RAFFINEMENT CORRIGE-T-IL, OU M4R RESTE-T-IL ENGLUE ? ===")
    print(f"(T1={T1} pas de mauvaise direction, puis T2 pas de correction forte)\n")
    rows = []
    for t2 in T2_VALUES:
        t0 = time.time()
        acc_primed, acc_fresh = [], []
        for seed in SEEDS:
            b_true = 1 if (seed % 2 == 0) else -1
            g_primed = run_primed(seed, b_true, t2)
            g_fresh = run_fresh(seed, b_true, t2)
            acc_primed.append(int(g_primed == b_true))
            acc_fresh.append(int(g_fresh == b_true))
        ap, af = float(np.mean(acc_primed)), float(np.mean(acc_fresh))
        cost = af - ap
        rows.append((t2, ap, af, cost))
        print(f"T2={t2:<5} accuracy PRIME_FAUX={ap:.3f}  accuracy FRAIS={af:.3f}  "
              f"cout_cicatrice={cost:+.3f}  [{time.time()-t0:.0f}s]")

    print("\n=== VERDICT ===")
    max_cost = max(r[3] for r in rows)
    last_cost = rows[-1][3]
    if max_cost < 0.05:
        print("  -> AUCUN cout de cicatrice mesurable a aucun T2 -- le raffinement est "
              "GRATUIT ici, M4R corrige aussi facilement qu'un reseau frais. Bonne "
              "nouvelle pour l'architecture de Julien -- mais cf. reserve : la fausse "
              "impression initiale est FAIBLE (B_E_WEAK=0.3), pas testee a impression forte.")
    elif last_cost < 0.05 and max_cost >= 0.05:
        print(f"  -> Cout de cicatrice REEL a court T2 (max {max_cost:+.3f}) mais qui SE "
              f"REFERME avec plus de temps correctif (T2={T2_VALUES[-1]} : {last_cost:+.3f}) -- "
              "le raffinement fonctionne, mais demande du temps pour desapprendre "
              "l'impression fausse, pas instantane.")
    else:
        print(f"  -> Cout de cicatrice PERSISTANT meme au T2 le plus long ({last_cost:+.3f}) -- "
              "la cicatrice de P12 (STNO) se retrouve ICI AUSSI sur FHN+lattice : le "
              "raffinement de Julien a un vrai cout, pas gratuit, meme avec du temps.")

    with (FIG / "p11_refinement_scar_poc.csv").open("w", encoding="utf-8") as f:
        f.write("t2,acc_primed,acc_fresh,cout_cicatrice\n")
        for r in rows:
            f.write(",".join(f"{x:.6f}" if isinstance(x, float) else str(x) for x in r) + "\n")
    return rows


def main() -> int:
    t0 = time.time()
    FIG.mkdir(parents=True, exist_ok=True)
    rows = sweep()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.8))
        t2s = [r[0] for r in rows]
        ax.plot(t2s, [r[1] for r in rows], "o-", c="#d62728", label="PRIME_FAUX (impression fausse a corriger)")
        ax.plot(t2s, [r[2] for r in rows], "s-", c="#2ca02c", label="FRAIS (aucune impression prealable)")
        ax.set_xlabel("T2 (pas de correction)"); ax.set_ylabel("accuracy finale")
        ax.set_ylim(0, 1.05)
        ax.set_title("Le raffinement : M4R corrige-t-il une premiere impression fausse ?")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIG / "p11_refinement_scar_poc.png", dpi=140)
        print(f"\n[png] {FIG / 'p11_refinement_scar_poc.png'}")
    except Exception as e:
        print(f"[png] skipped: {e}")

    print(f"\nWall time: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
