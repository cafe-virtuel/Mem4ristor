#!/usr/bin/env python3
"""
P10 -- LE VRAI TEST DE CONSENSUS (Condorcet) : gamma_int a-t-il ENFIN sa
chance ?
=============================================================================
Cree : 2026-07-13 (Claude Sonnet 5, L'Ingenieur). Suite de
`p10_gamma_int_batterie_large_poc.py`, dont la Partie 2 ("vitesse de
consensus") etait mal posee : elle mesurait la dispersion de u_c sous un
stimulus HOMOGENE (tous les noeuds recoivent EXACTEMENT le meme signal),
ce qui n'est pas un scenario de consensus au sens classique -- il n'y a
jamais eu de DESACCORD INITIAL entre observateurs a reconcilier, juste un
bruit dynamique qui s'est avere croitre plutot que decroitre.

LE VRAI TEST (theoreme du jury de Condorcet, le fondement mathematique de
"la sagesse des foules") : un groupe de N observateurs INDEPENDANTS, chacun
correct avec probabilite p>0.5 (erreur de MESURE individuelle, pas
malveillance -- different des "menteurs" deja testes, qui etaient un petit
sous-groupe delibere et toujours faux). Si les erreurs sont independantes,
le VOTE MAJORITAIRE converge vers la bonne reponse quand N croit. Question
posee ici : est-ce que gamma_int (qui fait converger les u_c vers une
moyenne locale) AIDE cette convergence -- en particulier, est-ce que les
noeuds initialement TROMPES (le sous-groupe minoritaire "faux" par tirage)
se font CORRIGER par leurs voisins majoritairement corrects ?

Protocole : groupe de 30 noeuds, bit vrai b_a partage. Chaque noeud recoit
INDEPENDAMMENT (tirage Bernoulli par noeud, PAS par seed global) le bon
signe avec probabilite P_CORRECT (0.7 par defaut -- individuellement
majoritairement fiable mais 30% d'erreur individuelle), le signe inverse
sinon. Sweep gamma_int in {0, 0.05, 0.15, 0.3, 0.5} x P_CORRECT in
{0.6, 0.7, 0.8}. Lecture : (a) GROUPE entier (vote/interference) vs b_a --
le consensus final recupere-t-il la bonne reponse malgre les erreurs
individuelles ? (b) le SOUS-GROUPE initialement FAUX (par tirage) est-il
"corrige" (son vote interne bascule-t-il vers b_a) plus souvent a
gamma_int eleve ?

Statut : exploratoire, hors preprint, aucune modification de dynamics.py.
Guardian doit rester 14/14. Sorties : figures/p10_vrai_consensus_poc.csv + .png
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
SEEDS = list(range(20))
GAMMA_VALUES = [0.0, 0.05, 0.15, 0.3, 0.5]
P_CORRECT_VALUES = [0.6, 0.7, 0.8]


def build_group_condorcet(seed, p_correct):
    """Chaque noeud du groupe tire INDEPENDAMMENT son propre signe (Bernoulli
    p_correct pour le bon signe). Retourne mask, idle, et le masque des
    noeuds "initialement faux" (tirage errone) pour la lecture de correction."""
    rng = np.random.RandomState(85000 + seed)
    mask_nodes = rng.choice(N, size=GROUP_SIZE, replace=False)
    mask = np.zeros(N, dtype=bool)
    mask[mask_nodes] = True
    idle = ~mask
    individual_sign = np.where(rng.random(GROUP_SIZE) < p_correct, 1, -1)
    wrong_sub = mask_nodes[individual_sign == -1]
    wrong = np.zeros(N, dtype=bool)
    wrong[wrong_sub] = True
    return mask, idle, wrong, individual_sign, mask_nodes


def run_condorcet(seed, b_a, gamma_int, p_correct):
    mask, idle, wrong, individual_sign, mask_nodes = build_group_condorcet(seed, p_correct)
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
    return m.u_c.copy(), mask, idle, wrong


def decode_vote(u_c, sub_mask, idle):
    idle_ref = float(np.real(u_c[idle]).mean())
    votes = np.sign(idle_ref - np.real(u_c[sub_mask]))
    votes[votes == 0] = 1
    return 1 if votes.sum() >= 0 else -1


def decode_interference(u_c, sub_mask, idle):
    diff = u_c[idle].mean() - u_c[sub_mask].mean()
    return 1 if float(np.real(diff)) >= 0 else -1


def sweep():
    print("=== LE VRAI TEST DE CONSENSUS (Condorcet) : erreurs individuelles "
          "independantes, pas des menteurs deliberes ===\n")
    rows = []
    for p_correct in P_CORRECT_VALUES:
        for gamma_int in GAMMA_VALUES:
            t0 = time.time()
            acc_group_vote, acc_group_int = [], []
            wrong_corrected = []
            for seed in SEEDS:
                for b_a in (1, -1):
                    u_c, mask, idle, wrong = run_condorcet(seed, b_a, gamma_int, p_correct)
                    acc_group_vote.append(int(decode_vote(u_c, mask, idle) == b_a))
                    acc_group_int.append(int(decode_interference(u_c, mask, idle) == b_a))
                    if np.any(wrong):
                        wrong_corrected.append(int(decode_vote(u_c, wrong, idle) == b_a))
            row = dict(
                p_correct=p_correct, gamma_int=gamma_int,
                acc_group_vote=float(np.mean(acc_group_vote)),
                acc_group_int=float(np.mean(acc_group_int)),
                wrong_corrected=float(np.mean(wrong_corrected)) if wrong_corrected else float("nan"),
                n_wrong_samples=len(wrong_corrected),
            )
            rows.append(row)
            print(f"p_correct={p_correct} gamma_int={gamma_int:<5} "
                  f"groupe(vote/int)={row['acc_group_vote']:.3f}/{row['acc_group_int']:.3f}  "
                  f"minorite_fausse_corrigee={row['wrong_corrected']:.3f} (n={row['n_wrong_samples']})  "
                  f"[{time.time()-t0:.0f}s]")

    print("\n=== VERDICT ===")
    for p_correct in P_CORRECT_VALUES:
        sub = [r for r in rows if r['p_correct'] == p_correct]
        ref = next(r for r in sub if r['gamma_int'] == 0.0)
        best_int = max(sub, key=lambda r: r['acc_group_int'])
        best_wrong = max(sub, key=lambda r: r['wrong_corrected'])
        print(f"p_correct={p_correct} : groupe(interference) gamma_int=0 -> {ref['acc_group_int']:.3f} | "
              f"meilleur gamma_int={best_int['gamma_int']} -> {best_int['acc_group_int']:.3f} "
              f"(gain {best_int['acc_group_int']-ref['acc_group_int']:+.3f})")
        print(f"                 minorite corrigee gamma_int=0 -> {ref['wrong_corrected']:.3f} | "
              f"meilleur gamma_int={best_wrong['gamma_int']} -> {best_wrong['wrong_corrected']:.3f} "
              f"(gain {best_wrong['wrong_corrected']-ref['wrong_corrected']:+.3f})")

    FIG.mkdir(parents=True, exist_ok=True)
    with (FIG / "p10_vrai_consensus_poc.csv").open("w", encoding="utf-8") as f:
        f.write("p_correct,gamma_int,acc_group_vote,acc_group_int,wrong_corrected,n_wrong_samples\n")
        for r in rows:
            f.write(",".join(f"{r[k]:.6f}" if isinstance(r[k], float) else str(r[k])
                              for k in ["p_correct", "gamma_int", "acc_group_vote",
                                        "acc_group_int", "wrong_corrected", "n_wrong_samples"]) + "\n")
    return rows


def main() -> int:
    t0 = time.time()
    rows = sweep()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

        ax = axes[0]
        for p_correct in P_CORRECT_VALUES:
            sub = [r for r in rows if r['p_correct'] == p_correct]
            ax.plot([r['gamma_int'] for r in sub], [r['acc_group_int'] for r in sub],
                     "o-", label=f"p_correct={p_correct}")
        ax.set_xlabel("gamma_int"); ax.set_ylabel("accuracy groupe (interference)")
        ax.set_ylim(0, 1.05); ax.set_title("Consensus de groupe vs erreurs individuelles")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

        ax = axes[1]
        for p_correct in P_CORRECT_VALUES:
            sub = [r for r in rows if r['p_correct'] == p_correct]
            ax.plot([r['gamma_int'] for r in sub], [r['wrong_corrected'] for r in sub],
                     "o-", label=f"p_correct={p_correct}")
        ax.axhline(0.5, ls=":", c="gray")
        ax.set_xlabel("gamma_int"); ax.set_ylabel("accuracy minorite corrigee")
        ax.set_ylim(0, 1.05); ax.set_title("La minorite initialement fausse se corrige-t-elle ?")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

        fig.suptitle("P10 -- le vrai test de consensus (Condorcet)", fontsize=12)
        plt.tight_layout()
        plt.savefig(FIG / "p10_vrai_consensus_poc.png", dpi=140)
        print(f"\n[png] {FIG / 'p10_vrai_consensus_poc.png'}")
    except Exception as e:
        print(f"[png] skipped: {e}")

    print(f"\nWall time: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
