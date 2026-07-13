#!/usr/bin/env python3
"""
P10 -- GAMMA_INT REPARE-T-IL LE CROSSTALK ENTRE DEUX GROUPES ?
=============================================================================
Cree : 2026-07-13 (Claude Sonnet 5, L'Ingenieur). Suite directe de
`p10_group_rotation_poc.py` (crosstalk reel, non repare par la separation de
frequence : A seul=1.000, A+B meme canal=0.812, A+B canal separe encore pire)
et de `p10_next_steps_poc.py` Partie 1 (CE MEME JOUR, plus tot) qui avait
trouve que **gamma_int=0 bat le defaut 0.15** sur la memoire a UN SEUL pulse
(0.88 vs 0.88 en memoire mais MEILLEUR ratio d'anti-sync, 0.579 vs 0.633).
Question jamais posee : gamma_int gouverne le canal d'INTERFERENCE SOCIALE
(moyenne complexe des u_c voisins) -- c'est structurellement le canal par
lequel un groupe B actif pourrait perturber un groupe A voisin sur le
lattice. Si gamma_int=0 coupe ce canal, le crosstalk mesure dans
p10_group_rotation_poc.py devrait-il disparaitre ?

Protocole : identique a p10_group_rotation_poc.py (2 groupes de 30 noeuds,
lattice 10x10, heretic_ratio=0.0, D=1200, B_PULSE=200), omega_B=0 fixe (la
separation de frequence est deja refutee comme reparation -- on isole
gamma_int seul). Sweep gamma_int in {0, 0.05, 0.15 (defaut), 0.3, 0.5},
readout INTERFERENCE (etabli comme la methode robuste par
p10_vote_vs_interference_poc.py) ET VOTE pour verifier que la conclusion
tient des deux cotes. Reference : solo A (gamma_int par defaut, deja mesure
= 1.000, D=1200, 40 problemes) -- ici recalcule au meme gamma_int que le
sweep pour un controle propre (gamma_int agit aussi DANS le groupe A seul,
via la cohesion entre ses 30 noeuds stimules).

Statut : exploratoire, hors preprint, aucune modification de dynamics.py.
Guardian doit rester 14/14. Sorties : figures/p10_gamma_int_crosstalk_poc.csv + .png
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
SEEDS_SOLO = list(range(20))
SIGN_COMBOS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
GAMMA_VALUES = [0.0, 0.05, 0.15, 0.3, 0.5]


def build_groups(seed):
    rng = np.random.RandomState(70000 + seed)
    both = rng.choice(N, size=2 * GROUP_SIZE, replace=False)
    mask_a = np.zeros(N, dtype=bool)
    mask_b = np.zeros(N, dtype=bool)
    mask_a[both[:GROUP_SIZE]] = True
    mask_b[both[GROUP_SIZE:]] = True
    idle = ~(mask_a | mask_b)
    return mask_a, mask_b, idle


def build_group_solo(seed):
    rng = np.random.RandomState(80000 + seed)
    mask_nodes = rng.choice(N, size=GROUP_SIZE, replace=False)
    mask = np.zeros(N, dtype=bool)
    mask[mask_nodes] = True
    idle = ~mask
    return mask, idle


def run_solo(seed, b_a, gamma_int, delay=DELAY):
    mask, idle = build_group_solo(seed)
    net = Mem4Network(size=SIDE, heretic_ratio=0.0, seed=seed * 10 + 1,
                       adjacency_matrix=make_lattice_adj(SIDE, periodic=True))
    m = net.model
    m.cfg['complex_doubt']['enabled'] = True
    m.cfg['complex_doubt']['gamma_int'] = gamma_int
    stim_vec = np.zeros(N)
    stim_vec[mask] = b_a * B_E
    zero = np.zeros(N)
    for t in range(B_PULSE + delay):
        net.step(I_stimulus=stim_vec if t < B_PULSE else zero)
    return m.u_c.copy(), mask, idle


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


def decode_interference(u_c, mask, idle):
    diff = u_c[idle].mean() - u_c[mask].mean()
    r = float(np.real(diff))
    return 1 if r >= 0 else -1


def decode_vote(u_c, mask, idle):
    idle_ref = float(np.real(u_c[idle]).mean())
    node_signs = np.sign(idle_ref - np.real(u_c[mask]))
    node_signs[node_signs == 0] = 1
    tally = node_signs.sum()
    return 1 if tally >= 0 else -1


def solo_at_gamma(gamma_int):
    correct_int, correct_vote = [], []
    for seed in SEEDS_SOLO:
        for b_a in (1, -1):
            u_c, mask, idle = run_solo(seed, b_a, gamma_int)
            correct_int.append(int(decode_interference(u_c, mask, idle) == b_a))
            correct_vote.append(int(decode_vote(u_c, mask, idle) == b_a))
    return float(np.mean(correct_int)), float(np.mean(correct_vote))


def pair_at_gamma(gamma_int):
    acc_a_int, acc_a_vote = [], []
    acc_par_int, acc_par_vote = [], []
    for seed in SEEDS_PAIR:
        for b_a, b_b in SIGN_COMBOS:
            u_c, mask_a, mask_b, idle = run_pair(seed, b_a, b_b, gamma_int)
            parity_true = b_a * b_b

            a_int = decode_interference(u_c, mask_a, idle)
            b_int = decode_interference(u_c, mask_b, idle)
            acc_a_int.append(int(a_int == b_a))
            acc_par_int.append(int(a_int * b_int == parity_true))

            a_vote = decode_vote(u_c, mask_a, idle)
            b_vote = decode_vote(u_c, mask_b, idle)
            acc_a_vote.append(int(a_vote == b_a))
            acc_par_vote.append(int(a_vote * b_vote == parity_true))

    return (float(np.mean(acc_a_int)), float(np.mean(acc_a_vote)),
            float(np.mean(acc_par_int)), float(np.mean(acc_par_vote)))


def main() -> int:
    t0 = time.time()
    print("=== GAMMA_INT REPARE-T-IL LE CROSSTALK ? (omega_B=0, D=1200) ===\n")
    rows = []
    for g in GAMMA_VALUES:
        tg = time.time()
        solo_int, solo_vote = solo_at_gamma(g)
        a_int, a_vote, par_int, par_vote = pair_at_gamma(g)
        rows.append((g, solo_int, solo_vote, a_int, a_vote, par_int, par_vote))
        print(f"gamma_int={g:<5} solo(int/vote)={solo_int:.3f}/{solo_vote:.3f}  "
              f"A_avec_B(int/vote)={a_int:.3f}/{a_vote:.3f}  "
              f"parite(int/vote)={par_int:.3f}/{par_vote:.3f}  [{time.time()-tg:.0f}s]")

    print("\n=== VERDICT ===")
    default_row = next(r for r in rows if r[0] == 0.15)
    zero_row = next(r for r in rows if r[0] == 0.0)
    print(f"Crosstalk sur A (interference) : defaut(0.15)={default_row[3]:.3f} vs "
          f"solo(0.15)={default_row[1]:.3f} (delta={default_row[3]-default_row[1]:+.3f})  |  "
          f"gamma=0={zero_row[3]:.3f} vs solo(gamma=0)={zero_row[1]:.3f} "
          f"(delta={zero_row[3]-zero_row[1]:+.3f})")
    if abs(zero_row[3] - zero_row[1]) < abs(default_row[3] - default_row[1]) - 0.05:
        print("  -> gamma_int=0 REDUIT le crosstalk sur A par rapport au defaut -- "
              "couper le canal d'interference sociale protege le decode individuel.")
    elif abs(zero_row[3] - zero_row[1]) > abs(default_row[3] - default_row[1]) + 0.05:
        print("  -> gamma_int=0 AGGRAVE le crosstalk sur A -- contre-intuitif, "
              "le crosstalk ne passe PAS principalement par l'interference sociale.")
    else:
        print("  -> gamma_int n'a pas d'effet net sur le crosstalk mesure sur A -- "
              "le mecanisme du crosstalk n'est pas (ou pas seulement) l'interference "
              "sociale gamma_int (probablement le couplage spatial direct de la "
              "dynamique v/w, hors du canal complex_doubt).")

    best_par_int = max(rows, key=lambda r: r[5])
    print(f"\nMeilleure parite (interference) : gamma_int={best_par_int[0]} -> "
          f"{best_par_int[5]:.3f} (reference gamma_int=0.15/omega_B=0 : "
          f"{default_row[5]:.3f}, cf. p10_group_rotation_poc.py: 0.812)")
    if best_par_int[5] > default_row[5] + 0.05:
        print("  -> Un gamma_int different du defaut AMELIORE la parite -- "
              "marche suivante utile : re-accorder gamma_int pour ce protocole.")
    else:
        print("  -> Aucun gamma_int testé n'ameliore nettement la parite au-dela "
              "du defaut -- le plafond de p10_group_rotation_poc.py (0.812) "
              "tient au-dela de ce levier.")

    FIG.mkdir(parents=True, exist_ok=True)
    with (FIG / "p10_gamma_int_crosstalk_poc.csv").open("w", encoding="utf-8") as f:
        f.write("gamma_int,solo_int,solo_vote,A_with_B_int,A_with_B_vote,parity_int,parity_vote\n")
        for r in rows:
            f.write(",".join(f"{x:.6f}" if isinstance(x, float) else str(x) for x in r) + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
        gs = [r[0] for r in rows]

        ax = axes[0]
        ax.plot(gs, [r[1] for r in rows], "o--", color="#2ca02c", label="solo A (interference)")
        ax.plot(gs, [r[3] for r in rows], "o-", color="#d62728", label="A + B actif (interference)")
        ax.plot(gs, [r[2] for r in rows], "s--", color="#2ca02c", alpha=0.5, label="solo A (vote)")
        ax.plot(gs, [r[4] for r in rows], "s-", color="#d62728", alpha=0.5, label="A + B actif (vote)")
        ax.set_xlabel("gamma_int"); ax.set_ylabel("accuracy decode A"); ax.set_ylim(0, 1.05)
        ax.set_title("Crosstalk sur A vs gamma_int"); ax.legend(fontsize=6.5); ax.grid(alpha=0.3)

        ax = axes[1]
        ax.plot(gs, [r[5] for r in rows], "o-", color="#9467bd", label="parite (interference)")
        ax.plot(gs, [r[6] for r in rows], "s-", color="#1f77b4", label="parite (vote)")
        ax.axvline(0.15, ls=":", c="k", alpha=0.4, label="defaut V1")
        ax.set_xlabel("gamma_int"); ax.set_ylabel("accuracy parite"); ax.set_ylim(0, 1.05)
        ax.set_title("Parite b_A*b_B vs gamma_int"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

        fig.suptitle(f"P10 -- gamma_int repare-t-il le crosstalk ? (D={DELAY}, omega_B=0)", fontsize=11)
        plt.tight_layout()
        plt.savefig(FIG / "p10_gamma_int_crosstalk_poc.png", dpi=140)
        print(f"\n[png] {FIG / 'p10_gamma_int_crosstalk_poc.png'}")
    except Exception as e:
        print(f"[png] skipped: {e}")

    print(f"\nWall time: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
