#!/usr/bin/env python3
"""
P10 -- VOTE vs INTERFERENCE : le vrai mecanisme de la genese, enfin isole.
=============================================================================
Cree : 2026-07-13 (Claude Sonnet 5, L'Ingenieur). Suite directe de
`p10_group_rotation_poc.py` : ce script concluait que "parite separee" et
"parite globale" y etaient IDENTIQUES par construction, parce que les deux
methodes agregeaient d'abord chaque groupe a UN SEUL angle moyen avant de
comparer -- ce n'est PAS ce que testait `genesis_five_states_poc.py` (11/07).

CE QUE FAIT REELLEMENT LA GENESE (relu dans le code, pas de memoire) :
  R2 (global) = signe(cos(SOMME des phases dominantes de chaque unite))
              = lecture du PRODUIT DES PHASEURS -- la definition mathematique
                de la parite elle-meme.
  Le "vote majoritaire" utilise comme reference externe dans la genese est
  la majorite des N=5 BITS INDIVIDUELS -- structurellement plafonnee a
  68.75% de correlation avec la parite a N=5 (fait mathematique, K impair
  >=3, rien a voir avec le bruit ou la dynamique).

Notre reseau physique n'a pas 5 UNITES portant chacune un bit -- il a 30
NOEUDS REDONDANTS portant CHACUN une estimation bruitee du MEME bit (la
question de la genese porte sur le codage de population, pas sur le nombre
de bits). C'est une question DIFFERENTE mais tout aussi legitime, et surtout
directement testable : sur un pool de N estimateurs bruites d'un meme signal,
MOYENNER PUIS SEUILLER UNE FOIS (interference) bat-il SEUILLER CHACUN PUIS
VOTER (vote) ? C'est le meme principe (combiner avant de seuiller preserve
de l'info que seuiller-puis-combiner perd), teste sur le bon objet cette
fois : les 30 noeuds INDIVIDUELS d'un groupe, pas le groupe deja agrege.

PARTIE 1 (le test le plus propre) : decoder UN SEUL bit b_A, groupe A SEUL
(protocole V1 exact, omega=0, aucune complication de groupe B). VOTE = signe
de CHAQUE noeud (vs reference idle), majorite des 30 signes. INTERFERENCE =
moyenne complexe des 30 noeuds (vs idle), UN SEUL signe final -- deja ce que
mesurait p10_complex_doubt_poc.py sans le nommer "interference".

PARTIE 2 : re-tester la parite a 2 groupes (omega_B=0, la condition ou
p10_group_rotation_poc.py trouvait separee==globale par construction) avec
VOTE (30+30 signes individuels, majorite par groupe, PUIS produit des deux
majorites) vs INTERFERENCE (deja mesure : 0.812 a D=1200, omega_B=0).

Statut : exploratoire, hors preprint, aucune modification de dynamics.py
(coeur deja etendu le 13/07, omega_u par-noeud). Guardian doit rester 14/14.
Sorties : figures/p10_vote_vs_interference_poc.csv + .png
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
SEEDS_SOLO = list(range(20))
SEEDS_PAIR = list(range(12))
SIGN_COMBOS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]


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


def run_solo(seed, b_a, delay=DELAY):
    mask, idle = build_group_solo(seed)
    net = Mem4Network(size=SIDE, heretic_ratio=0.0, seed=seed * 10 + 1,
                       adjacency_matrix=make_lattice_adj(SIDE, periodic=True))
    m = net.model
    m.cfg['complex_doubt']['enabled'] = True
    stim_vec = np.zeros(N)
    stim_vec[mask] = b_a * B_E
    zero = np.zeros(N)
    for t in range(B_PULSE + delay):
        net.step(I_stimulus=stim_vec if t < B_PULSE else zero)
    return m.u_c.copy(), mask, idle


def run_pair(seed, b_a, b_b, delay=DELAY):
    mask_a, mask_b, idle = build_groups(seed)
    net = Mem4Network(size=SIDE, heretic_ratio=0.0, seed=seed * 10 + 1,
                       adjacency_matrix=make_lattice_adj(SIDE, periodic=True))
    m = net.model
    m.cfg['complex_doubt']['enabled'] = True
    stim_vec = np.zeros(N)
    stim_vec[mask_a] = b_a * B_E
    stim_vec[mask_b] = b_b * B_E
    zero = np.zeros(N)
    for t in range(B_PULSE + delay):
        net.step(I_stimulus=stim_vec if t < B_PULSE else zero)
    return m.u_c.copy(), mask_a, mask_b, idle


def decode_interference(u_c, mask, idle):
    """Moyenne COMPLEXE du groupe et de idle, PUIS un seul seuillage.
    Convention de signe V1 (p10_complex_doubt_poc.py) : idle MOINS mask
    (ruc[~mask].mean() - ruc[mask].mean() == b)."""
    diff = u_c[idle].mean() - u_c[mask].mean()
    r = float(np.real(diff))
    return 1 if r >= 0 else -1


def decode_vote(u_c, mask, idle):
    """Seuillage de CHAQUE noeud du groupe (vs la meme reference idle),
    PUIS majorite des signes individuels. Meme convention (idle - noeud)."""
    idle_ref = float(np.real(u_c[idle]).mean())
    node_signs = np.sign(idle_ref - np.real(u_c[mask]))
    node_signs[node_signs == 0] = 1
    tally = node_signs.sum()
    return 1 if tally >= 0 else -1


def part1_solo():
    print("=== PARTIE 1 : un seul groupe, un seul bit -- VOTE vs INTERFERENCE ===")
    t0 = time.time()
    acc_vote, acc_int = [], []
    for seed in SEEDS_SOLO:
        for b_a in (1, -1):
            u_c, mask, idle = run_solo(seed, b_a)
            acc_vote.append(int(decode_vote(u_c, mask, idle) == b_a))
            acc_int.append(int(decode_interference(u_c, mask, idle) == b_a))
    m_vote, m_int = float(np.mean(acc_vote)), float(np.mean(acc_int))
    n = len(acc_vote)
    se_vote = float(np.std(acc_vote, ddof=1) / np.sqrt(n))
    se_int = float(np.std(acc_int, ddof=1) / np.sqrt(n))
    print(f"  n={n} problemes (D={DELAY})")
    print(f"  VOTE          : {m_vote:.3f} +/- {se_vote:.3f}")
    print(f"  INTERFERENCE  : {m_int:.3f} +/- {se_int:.3f}")
    print(f"  delta (interference - vote) = {m_int - m_vote:+.3f}  [{time.time()-t0:.0f}s]")
    return m_vote, m_int, se_vote, se_int, n


def part2_pair():
    print("\n=== PARTIE 2 : parite a 2 groupes (omega_B=0) -- VOTE vs INTERFERENCE ===")
    t0 = time.time()
    acc_vote, acc_int = [], []
    for seed in SEEDS_PAIR:
        for b_a, b_b in SIGN_COMBOS:
            u_c, mask_a, mask_b, idle = run_pair(seed, b_a, b_b)
            parity_true = b_a * b_b

            vote_a = decode_vote(u_c, mask_a, idle)
            vote_b = decode_vote(u_c, mask_b, idle)
            acc_vote.append(int(vote_a * vote_b == parity_true))

            int_a = decode_interference(u_c, mask_a, idle)
            int_b = decode_interference(u_c, mask_b, idle)
            acc_int.append(int(int_a * int_b == parity_true))
    m_vote, m_int = float(np.mean(acc_vote)), float(np.mean(acc_int))
    n = len(acc_vote)
    se_vote = float(np.std(acc_vote, ddof=1) / np.sqrt(n))
    se_int = float(np.std(acc_int, ddof=1) / np.sqrt(n))
    print(f"  n={n} problemes (D={DELAY}, omega_B=0)")
    print(f"  VOTE (parite)          : {m_vote:.3f} +/- {se_vote:.3f}")
    print(f"  INTERFERENCE (parite)  : {m_int:.3f} +/- {se_int:.3f}  (reference p10_group_rotation_poc.py: 0.812)")
    print(f"  delta (interference - vote) = {m_int - m_vote:+.3f}  [{time.time()-t0:.0f}s]")
    return m_vote, m_int, se_vote, se_int, n


def main() -> int:
    t0 = time.time()
    r1 = part1_solo()
    r2 = part2_pair()

    print("\n=== VERDICT ===")
    d1 = r1[1] - r1[0]
    d2 = r2[1] - r2[0]
    thresh = 2 * max(r1[2], r1[3])  # ~2 SE de marge pour la partie 1
    if d1 > thresh:
        print(f"(1) Solo : INTERFERENCE bat VOTE ({d1:+.3f}, au-dela de ~2 SE) -- "
              "le mecanisme de la genese (combiner avant de seuiller) se transfere "
              "au codage de population du reseau physique.")
    elif d1 < -thresh:
        print(f"(1) Solo : VOTE bat INTERFERENCE ({d1:+.3f}) -- l'inverse de la "
              "genese sur ce protocole.")
    else:
        print(f"(1) Solo : aucune difference nette ({d1:+.3f}, sous ~2 SE) -- "
              "les deux methodes sont equivalentes ici.")

    thresh2 = 2 * max(r2[2], r2[3])
    if d2 > thresh2:
        print(f"(2) Parite : INTERFERENCE bat VOTE ({d2:+.3f}) -- l'avantage "
              "survit a la composition de deux groupes.")
    elif d2 < -thresh2:
        print(f"(2) Parite : VOTE bat INTERFERENCE ({d2:+.3f}).")
    else:
        print(f"(2) Parite : aucune difference nette ({d2:+.3f}).")

    FIG.mkdir(parents=True, exist_ok=True)
    with (FIG / "p10_vote_vs_interference_poc.csv").open("w", encoding="utf-8") as f:
        f.write("part,n,acc_vote,se_vote,acc_interference,se_interference\n")
        f.write(f"solo,{r1[4]},{r1[0]:.6f},{r1[2]:.6f},{r1[1]:.6f},{r1[3]:.6f}\n")
        f.write(f"pair,{r2[4]},{r2[0]:.6f},{r2[2]:.6f},{r2[1]:.6f},{r2[3]:.6f}\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4.6))
        labels = ["Solo\n(1 bit)", "Parite\n(2 groupes)"]
        vote_vals = [r1[0], r2[0]]
        vote_err = [r1[2], r2[2]]
        int_vals = [r1[1], r2[1]]
        int_err = [r1[3], r2[3]]
        x = np.arange(2)
        w = 0.35
        ax.bar(x - w / 2, vote_vals, w, yerr=vote_err, label="VOTE", color="#1f77b4", capsize=4)
        ax.bar(x + w / 2, int_vals, w, yerr=int_err, label="INTERFERENCE", color="#d62728", capsize=4)
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel("accuracy"); ax.set_ylim(0, 1.05)
        ax.set_title(f"Vote vs Interference (D={DELAY})")
        ax.legend(); ax.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(FIG / "p10_vote_vs_interference_poc.png", dpi=140)
        print(f"\n[png] {FIG / 'p10_vote_vs_interference_poc.png'}")
    except Exception as e:
        print(f"[png] skipped: {e}")

    print(f"\nWall time: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
