#!/usr/bin/env python3
"""
P10 -- REPLICATION du test de consensus Condorcet (gamma_int a-t-il une
vraie niche ?), avant de generaliser a d'autres topologies.
=============================================================================
Cree : 2026-07-13 (Claude Sonnet 5, L'Ingenieur). Suite de
`p10_vrai_consensus_poc.py` (meme jour), qui a trouve le PREMIER gain net
et explique de gamma_int sur les 9 tests de la journee : a p_correct=0.6-0.7
(erreurs individuelles independantes, ni menteurs ni signal nul), gamma_int
modere (0.15-0.3) ameliore le consensus de groupe de +0.125 a +0.175 pts.
Reserve notee a l'epoque : n=40/point, gains pres de 2 SE.

Avant de generaliser (topologie BA/ER, piste #2 demandee par Julien), la
pratique etablie du projet (le gate de replication de la genese, 11/07) :
reproduire d'abord avec PLUS de seeds, sur une plage de graines DISJOINTE
du premier test (seeds 70000+100..149, vs 70000+0..19 dans le test
original) -- une vraie replication independante, pas un re-run identique.

Protocole : IDENTIQUE a p10_vrai_consensus_poc.py (lattice 10x10, groupe de
30 noeuds, tirage Bernoulli individuel p_correct, D=1200), mais SEEDS
etendu a 40 (80 problemes/point au lieu de 40) et restreint aux 3 points
qui comptent (gamma_int in {0.0, 0.15, 0.3} x p_correct in {0.6, 0.7} --
laisse tomber 0.05/0.5/0.8, deja caracterises comme transitoire/plafond/
nul par le premier test, pour concentrer le budget de seeds sur la
replication du signal, pas sur un nouveau balayage).

Statut : exploratoire, hors preprint, aucune modification de dynamics.py.
Guardian doit rester 14/14. Sorties : figures/p10_condorcet_replication_poc.csv + .png
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
SEEDS = list(range(100, 140))  # 40 seeds, plage DISJOINTE du test original (0..19)
GAMMA_VALUES = [0.0, 0.15, 0.3]
P_CORRECT_VALUES = [0.6, 0.7]
N_BOOT = 10000
RNG_BOOT = np.random.RandomState(20260713)


def build_group_condorcet(seed, p_correct):
    rng = np.random.RandomState(85000 + seed)
    mask_nodes = rng.choice(N, size=GROUP_SIZE, replace=False)
    mask = np.zeros(N, dtype=bool)
    mask[mask_nodes] = True
    idle = ~mask
    individual_sign = np.where(rng.random(GROUP_SIZE) < p_correct, 1, -1)
    return mask, idle, individual_sign, mask_nodes


def run_condorcet(seed, b_a, gamma_int, p_correct):
    mask, idle, individual_sign, mask_nodes = build_group_condorcet(seed, p_correct)
    net = Mem4Network(size=SIDE, heretic_ratio=0.0, seed=seed * 10 + 1,
                       adjacency_matrix=make_lattice_adj(SIDE, periodic=True))
    m = net.model
    m.cfg['complex_doubt']['enabled'] = True
    m.cfg['complex_doubt']['gamma_int'] = gamma_int
    stim_vec = np.zeros(N)
    stim_vec[mask_nodes] = b_a * B_E * individual_sign
    zero = np.zeros(N)
    for t in range(B_PULSE + DELAY):
        net.step(I_stimulus=stim_vec if t < B_PULSE else zero)
    return m.u_c.copy(), mask, idle


def decode_vote(u_c, sub_mask, idle):
    idle_ref = float(np.real(u_c[idle]).mean())
    votes = np.sign(idle_ref - np.real(u_c[sub_mask]))
    votes[votes == 0] = 1
    return 1 if votes.sum() >= 0 else -1


def decode_interference(u_c, sub_mask, idle):
    diff = u_c[idle].mean() - u_c[sub_mask].mean()
    return 1 if float(np.real(diff)) >= 0 else -1


def boot_ci_paired(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    m = np.empty(N_BOOT)
    for k in range(N_BOOT):
        m[k] = d[RNG_BOOT.randint(0, n, n)].mean()
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def sweep():
    print("=== REPLICATION Condorcet (40 seeds, plage 100-139, disjointe du test original) ===\n")
    raw = {}  # (p_correct, gamma_int) -> (vote_list, int_list)
    for p_correct in P_CORRECT_VALUES:
        for gamma_int in GAMMA_VALUES:
            t0 = time.time()
            acc_vote, acc_int = [], []
            for seed in SEEDS:
                for b_a in (1, -1):
                    u_c, mask, idle = run_condorcet(seed, b_a, gamma_int, p_correct)
                    acc_vote.append(int(decode_vote(u_c, mask, idle) == b_a))
                    acc_int.append(int(decode_interference(u_c, mask, idle) == b_a))
            raw[(p_correct, gamma_int)] = (acc_vote, acc_int)
            print(f"p_correct={p_correct} gamma_int={gamma_int:<5} "
                  f"vote={np.mean(acc_vote):.3f}  interference={np.mean(acc_int):.3f}  "
                  f"n={len(acc_vote)}  [{time.time()-t0:.0f}s]")

    print("\n=== VERDICT DE REPLICATION (IC bootstrap apparie vs gamma_int=0) ===")
    rows = []
    for p_correct in P_CORRECT_VALUES:
        vote0, int0 = raw[(p_correct, 0.0)]
        for gamma_int in GAMMA_VALUES:
            if gamma_int == 0.0:
                continue
            vote_g, int_g = raw[(p_correct, gamma_int)]
            d_vote, lo_v, hi_v = boot_ci_paired(vote_g, vote0)
            d_int, lo_i, hi_i = boot_ci_paired(int_g, int0)
            conf_vote = "CONFIRME" if lo_v > 0 else ("REFUTE (pire)" if hi_v < 0 else "non confirme (IC couvre 0)")
            conf_int = "CONFIRME" if lo_i > 0 else ("REFUTE (pire)" if hi_i < 0 else "non confirme (IC couvre 0)")
            print(f"p_correct={p_correct} gamma_int={gamma_int} vs 0.0 :")
            print(f"    vote        : delta={d_vote:+.3f} CI[{lo_v:+.3f},{hi_v:+.3f}] -> {conf_vote}")
            print(f"    interference: delta={d_int:+.3f} CI[{lo_i:+.3f},{hi_i:+.3f}] -> {conf_int}")
            rows.append((p_correct, gamma_int, np.mean(vote_g), np.mean(vote0), d_vote, lo_v, hi_v, conf_vote,
                         np.mean(int_g), np.mean(int0), d_int, lo_i, hi_i, conf_int))

    with (FIG / "p10_condorcet_replication_poc.csv").open("w", encoding="utf-8") as f:
        f.write("p_correct,gamma_int,acc_vote,acc_vote_ref,d_vote,lo_vote,hi_vote,verdict_vote,"
                "acc_int,acc_int_ref,d_int,lo_int,hi_int,verdict_int\n")
        for r in rows:
            f.write(",".join(f"{x:.6f}" if isinstance(x, float) else str(x) for x in r) + "\n")
    return raw, rows


def main() -> int:
    t0 = time.time()
    FIG.mkdir(parents=True, exist_ok=True)
    raw, rows = sweep()

    n_confirmed = sum(1 for r in rows if "CONFIRME" == r[7] or "CONFIRME" == r[13])
    print(f"\n=== BILAN GLOBAL DE LA REPLICATION ===")
    print(f"{n_confirmed}/{len(rows)*2} comparaisons (vote+interference) confirmees a n=80/point "
          f"(vs n=40 dans le test original).")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
        for ax, key, title in [(axes[0], 0, "vote"), (axes[1], 1, "interference")]:
            for p_correct in P_CORRECT_VALUES:
                ys = [np.mean(raw[(p_correct, g)][key]) for g in GAMMA_VALUES]
                ax.plot(GAMMA_VALUES, ys, "o-", label=f"p_correct={p_correct}")
            ax.set_xlabel("gamma_int"); ax.set_ylabel(f"accuracy ({title})")
            ax.set_ylim(0, 1.05); ax.set_title(f"Replication (n=80/point) -- {title}")
            ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.suptitle("P10 -- replication du test Condorcet (seeds 100-139, disjoint)", fontsize=11)
        plt.tight_layout()
        plt.savefig(FIG / "p10_condorcet_replication_poc.png", dpi=140)
        print(f"\n[png] {FIG / 'p10_condorcet_replication_poc.png'}")
    except Exception as e:
        print(f"[png] skipped: {e}")

    print(f"\nWall time: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
