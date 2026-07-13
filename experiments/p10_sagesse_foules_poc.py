#!/usr/bin/env python3
"""
P10 -- LA FLEMME PAIE-T-ELLE PARFOIS ? Le test le plus favorable a gamma_int.
=============================================================================
Cree : 2026-07-13 (Claude Sonnet 5, L'Ingenieur). Suite de
`p10_flemme_frontiere_poc.py`. Verdict de Julien apres 3 tests consecutifs
tous defavorables a gamma_int (memoire, crosstalk, diffusion) : « la flemme
ne rapporte rien et detruit tout -- comme une personne qui aurait la flemme
d'aller bosser : pas de travail, pas d'argent, pas d'argent, vie difficile ».

AVANT de valider cette conclusion sans reserve, un point de methode : TOUS
les tests d'aujourd'hui opposaient des voisins en DESACCORD (groupe A vs
groupe B, bits opposes) -- bien sur que "copier" un voisin qui pense le
contraire coute. Ce n'est pas encore le test le plus dur pour l'hypothese
"gamma_int n'a AUCUNE valeur" : le cas favorable jamais teste est celui de
la SAGESSE DES FOULES -- des voisins qui portent le MEME bit mais avec une
CONFIANCE INDIVIDUELLE INEGALE (certains noeuds recoivent un signal fort et
clair, d'autres un signal faible/ambigu). Un noeud "en difficulte" qui capte
un peu de la moyenne de ses voisins CONFIANTS ET D'ACCORD ne devrait rien
perdre a le faire -- c'est le cas ou gamma_int a sa meilleure chance de
rapporter quelque chose.

PREDICTION (posee avant de lancer) : au sein d'un groupe qui PARTAGE le
meme bit vrai, avec un sous-ensemble de noeuds a stimulus FAIBLE (scale
0.15) et un sous-ensemble a stimulus FORT (scale 1.0), interleaves sur le
lattice (memes 30 noeuds du groupe, assignation forte/faible aleatoire) :
gamma_int>0 devrait AMELIORER la lecture des noeuds FAIBLES (ils captent la
confiance de leurs voisins forts, memes qu'eux) sans degrader les FORTS.
Si meme CE test ne montre aucun gain, la conclusion de Julien est
confirmee sans reserve : gamma_int, tel qu'implemente, n'a de valeur
mesurable sur AUCUN protocole teste a ce jour.

Protocole : UN SEUL groupe A (30 noeuds, meme construction que
p10_vote_vs_interference_poc.py), bit b_a partage par TOUS. 15 noeuds
"forts" (stim=b_a*B_E), 15 "faibles" (stim=b_a*B_E*WEAK_SCALE),
assignation aleatoire par seed. D=1200 (reference), sweep gamma_int in
{0, 0.05, 0.15 (defaut), 0.3, 0.5}. Lecture par-noeud (vote) + par
sous-groupe (forts seuls, faibles seuls, interference).

Statut : exploratoire, hors preprint, aucune modification de dynamics.py.
Guardian doit rester 14/14. Sorties : figures/p10_sagesse_foules_poc.csv + .png
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
WEAK_SCALE = 0.15
GROUP_SIZE = 30
B_PULSE = 200
DELAY = 1200
SEEDS = list(range(20))
GAMMA_VALUES = [0.0, 0.05, 0.15, 0.3, 0.5]


def build_group_mixed(seed):
    rng = np.random.RandomState(80000 + seed)
    mask_nodes = rng.choice(N, size=GROUP_SIZE, replace=False)
    mask = np.zeros(N, dtype=bool)
    mask[mask_nodes] = True
    idle = ~mask
    strong_sub = rng.choice(mask_nodes, size=GROUP_SIZE // 2, replace=False)
    strong = np.zeros(N, dtype=bool)
    strong[strong_sub] = True
    weak = mask & (~strong)
    return mask, idle, strong, weak


def run_problem(seed, b_a, gamma_int, delay=DELAY):
    mask, idle, strong, weak = build_group_mixed(seed)
    net = Mem4Network(size=SIDE, heretic_ratio=0.0, seed=seed * 10 + 1,
                       adjacency_matrix=make_lattice_adj(SIDE, periodic=True))
    m = net.model
    m.cfg['complex_doubt']['enabled'] = True
    m.cfg['complex_doubt']['gamma_int'] = gamma_int
    stim_vec = np.zeros(N)
    stim_vec[strong] = b_a * B_E
    stim_vec[weak] = b_a * B_E * WEAK_SCALE
    zero = np.zeros(N)
    for t in range(B_PULSE + delay):
        net.step(I_stimulus=stim_vec if t < B_PULSE else zero)
    return m.u_c.copy(), mask, idle, strong, weak


def decode_vote(u_c, sub_mask, idle):
    idle_ref = float(np.real(u_c[idle]).mean())
    votes = np.sign(idle_ref - np.real(u_c[sub_mask]))
    votes[votes == 0] = 1
    tally = votes.sum()
    return 1 if tally >= 0 else -1


def decode_interference(u_c, sub_mask, idle):
    diff = u_c[idle].mean() - u_c[sub_mask].mean()
    r = float(np.real(diff))
    return 1 if r >= 0 else -1


def per_node_vote_correct(u_c, node, idle_ref, b_true):
    vote = 1 if (idle_ref - float(np.real(u_c[node]))) >= 0 else -1
    return int(vote == b_true)


def sweep():
    print("=== SAGESSE DES FOULES : gamma_int aide-t-il les noeuds FAIBLES quand "
          "les voisins sont D'ACCORD ? ===")
    print(f"(D={DELAY}, {len(SEEDS)} seeds x 2 signes = {2*len(SEEDS)} problemes par gamma_int)\n")
    rows = []
    for gamma_int in GAMMA_VALUES:
        t0 = time.time()
        weak_node_correct, strong_node_correct = [], []
        weak_group_vote, strong_group_vote = [], []
        weak_group_int, strong_group_int = [], []
        for seed in SEEDS:
            for b_a in (1, -1):
                u_c, mask, idle, strong, weak = run_problem(seed, b_a, gamma_int)
                idle_ref = float(np.real(u_c[idle]).mean())
                for node in np.where(weak)[0]:
                    weak_node_correct.append(per_node_vote_correct(u_c, node, idle_ref, b_a))
                for node in np.where(strong)[0]:
                    strong_node_correct.append(per_node_vote_correct(u_c, node, idle_ref, b_a))
                weak_group_vote.append(int(decode_vote(u_c, weak, idle) == b_a))
                strong_group_vote.append(int(decode_vote(u_c, strong, idle) == b_a))
                weak_group_int.append(int(decode_interference(u_c, weak, idle) == b_a))
                strong_group_int.append(int(decode_interference(u_c, strong, idle) == b_a))
        row = dict(
            gamma_int=gamma_int,
            weak_node=float(np.mean(weak_node_correct)),
            strong_node=float(np.mean(strong_node_correct)),
            weak_group_vote=float(np.mean(weak_group_vote)),
            strong_group_vote=float(np.mean(strong_group_vote)),
            weak_group_int=float(np.mean(weak_group_int)),
            strong_group_int=float(np.mean(strong_group_int)),
        )
        rows.append(row)
        print(f"gamma_int={gamma_int:<5} noeud faible={row['weak_node']:.3f} "
              f"noeud fort={row['strong_node']:.3f}  |  "
              f"groupe_faible(vote/int)={row['weak_group_vote']:.3f}/{row['weak_group_int']:.3f}  "
              f"groupe_fort(vote/int)={row['strong_group_vote']:.3f}/{row['strong_group_int']:.3f}  "
              f"[{time.time()-t0:.0f}s]")

    print("\n=== VERDICT ===")
    r0 = next(r for r in rows if r['gamma_int'] == 0.0)
    best_weak = max(rows, key=lambda r: r['weak_node'])
    gain_weak = best_weak['weak_node'] - r0['weak_node']
    print(f"Noeud FAIBLE : gamma_int=0 -> {r0['weak_node']:.3f}  |  meilleur "
          f"gamma_int={best_weak['gamma_int']} -> {best_weak['weak_node']:.3f} "
          f"(gain {gain_weak:+.3f})")
    if gain_weak > 0.05 and best_weak['gamma_int'] > 0.0:
        strong_at_best = next(r for r in rows if r['gamma_int'] == best_weak['gamma_int'])
        cost_strong = strong_at_best['strong_node'] - r0['strong_node']
        print(f"  -> GAIN REEL pour les noeuds faibles a gamma_int={best_weak['gamma_int']} "
              f"({gain_weak:+.3f}). Cout pour les noeuds forts au meme point : "
              f"{cost_strong:+.3f}.")
        if cost_strong > -0.03:
            print("     Le fort ne paie presque rien pour que le faible gagne -- "
                  "PREMIER cas ou gamma_int>0 rapporte net. La sagesse des foules "
                  "existe, conditionnee a l'accord entre voisins.")
        else:
            print("     Mais le fort paie aussi -- gain net a verifier (pas un "
                  "dejeuner gratuit).")
    else:
        print("  -> AUCUN gain net pour les noeuds faibles, sur AUCUN gamma_int "
              "teste. Meme dans le cas le plus favorable a gamma_int (voisins "
              "d'ACCORD, signal individuel faible), copier le voisin n'aide pas. "
              "La conclusion de Julien tient sans reserve : sur les 4 tests du "
              "13/07 (memoire, crosstalk, diffusion, sagesse des foules), "
              "gamma_int>0 n'a JAMAIS rapporte net.")

    FIG.mkdir(parents=True, exist_ok=True)
    with (FIG / "p10_sagesse_foules_poc.csv").open("w", encoding="utf-8") as f:
        f.write("gamma_int,weak_node,strong_node,weak_group_vote,strong_group_vote,"
                "weak_group_int,strong_group_int\n")
        for r in rows:
            f.write(",".join(f"{r[k]:.6f}" if isinstance(r[k], float) else str(r[k])
                              for k in ["gamma_int", "weak_node", "strong_node",
                                        "weak_group_vote", "strong_group_vote",
                                        "weak_group_int", "strong_group_int"]) + "\n")
    return rows


def main() -> int:
    t0 = time.time()
    rows = sweep()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.8))
        gs = [r['gamma_int'] for r in rows]
        ax.plot(gs, [r['weak_node'] for r in rows], "o-", color="#d62728", label="noeud faible (signal individuel)")
        ax.plot(gs, [r['strong_node'] for r in rows], "s-", color="#1f77b4", label="noeud fort (signal individuel)")
        ax.plot(gs, [r['weak_group_int'] for r in rows], "o--", color="#d62728", alpha=0.5, label="groupe faible (interference)")
        ax.set_xlabel("gamma_int"); ax.set_ylabel("accuracy"); ax.set_ylim(0, 1.05)
        ax.set_title("Sagesse des foules : gamma_int aide-t-il les noeuds faibles ?")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIG / "p10_sagesse_foules_poc.png", dpi=140)
        print(f"\n[png] {FIG / 'p10_sagesse_foules_poc.png'}")
    except Exception as e:
        print(f"[png] skipped: {e}")

    print(f"\nWall time: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
