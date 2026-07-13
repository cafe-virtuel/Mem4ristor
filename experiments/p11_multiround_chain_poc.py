#!/usr/bin/env python3
"""
P11 -- LE CHAINAGE A PLUSIEURS TOURS : direction -> solveur -> veille ->
nouvelle direction -> solveur -> ... , jamais teste jusqu'ici (un seul
cycle mesure partout dans le fil P11 du 12-13/07).
=============================================================================
Cree : 2026-07-13 (Claude Sonnet 5, L'Ingenieur). Julien : "j'aimerais bien
aborder le chainage multi tours". Derniere piece non testee de
l'architecture decrite des le debut du fil.

DESIGN. Un probleme a K=4 pieges trompeurs SUCCESSIFS le long d'un meme axe
de convergence : le solveur doit franchir 4 plateaux l'un apres l'autre. A
CHAQUE tour, M4R lit une direction (meme lecture, meme warm start, meme
verification rapide + bascule que p11_coupled_pipeline_poc.py -- reutilise
tel quel via pc.solve_coupled), sur un probleme local independant (meme
derivation h_min/x_p/w_flat que pw.make_problem, seed different par tour).

LA QUESTION AU CENTRE : entre deux tours, le reseau M4R garde-t-il son etat
(u_c persiste, la memoire directionnelle de P10 pourrait aider) ou repart-il
a zero (comme partout ailleurs dans le fil aujourd'hui) ? Deux conditions
comparees, MASK IDENTIQUE dans les deux cas (meme groupe physique de 30
noeuds sert de capteur a chaque tour, seule la persistance de l'etat u_c
differe -- isole exactement la variable d'interet, pas confondue avec un
autre facteur) :
  - FRESH      : reseau reconstruit a chaque tour (bruit renouvele, aucune
                 memoire inter-tours).
  - PERSISTENT : UN SEUL reseau construit au tour 1, jamais reinitialise,
                 restimule a chaque tour par le nouveau signe b_k.
Deux hypotheses en concurrence directe pour la premiere fois : la memoire
directionnelle de u_c (P10, tient ~12 tau_u) pourrait AIDER en PERSISTENT
(le reseau "sait deja" ou chercher) ; la cicatrice (trouvee 3x -- P12/STNO,
P11/raffinement) pourrait au contraire NUIRE (une impression precedente
resiste au changement de signe du tour suivant).

CRITERE PRE-FIXE (avant de lancer) :
  1. PERSISTENT bat-il, egale-t-il, ou perd-il contre FRESH en accuracy
     moyenne sur les K tours (IC bootstrap apparie) ?
  2. L'accuracy derive-t-elle avec l'INDEX du tour (tour 1 vs tour 4) pour
     l'une ou l'autre condition -- signature de memoire (drift positif) ou
     de cicatrice (drift negatif) ?
  3. Le bilan materiel CUMULE (K tours, cout des K lectures + cout solveur
     total, temps reel) reste-t-il stable tour apres tour, ou se degrade-t-il ?

Statut : exploratoire, hors preprint, aucune modification de dynamics.py.
Guardian doit rester 14/14. Sorties : figures/p11_multiround_chain_poc.csv + .png
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
sys.path.insert(0, str(ROOT / "experiments"))
from mem4ristor.topology import Mem4Network       # noqa: E402
from mem4ristor.graph_utils import make_lattice_adj  # noqa: E402
import p11_warm_start_poc as pw       # noqa: E402
import p11_coupled_pipeline_poc as pc  # noqa: E402 -- met pw.T_READ=30, pw.B_E=0.3

FIG = ROOT / "figures"
K = 4
N_CHAINS = 40
CHECK_TOL = pc.CHECK_TOL
N_CHECK = pc.N_CHECK


def round_b(chain_seed, k):
    rng = np.random.RandomState(910_000 + chain_seed * 100 + k)
    return 1 if rng.random() < 0.5 else -1


def m4r_read_fresh(chain_seed, mask, idle, b, k):
    net = Mem4Network(size=pw.SIDE, heretic_ratio=0.0, seed=chain_seed * 10 + 1 + k,
                       adjacency_matrix=make_lattice_adj(pw.SIDE, periodic=True))
    net.model.cfg['complex_doubt']['enabled'] = True
    stim = np.zeros(pw.N)
    stim[mask] = b * pw.B_E
    for _ in range(pw.T_READ):
        net.step(I_stimulus=stim)
    diff = net.model.u_c[idle].mean() - net.model.u_c[mask].mean()
    return 1 if float(np.real(diff)) >= 0 else -1


class PersistentReader:
    """Un seul reseau, jamais reinitialise entre les tours."""

    def __init__(self, chain_seed, mask):
        self.mask = mask
        self.net = Mem4Network(size=pw.SIDE, heretic_ratio=0.0, seed=chain_seed * 10 + 1,
                                adjacency_matrix=make_lattice_adj(pw.SIDE, periodic=True))
        self.net.model.cfg['complex_doubt']['enabled'] = True

    def read(self, b, idle):
        stim = np.zeros(pw.N)
        stim[self.mask] = b * pw.B_E
        for _ in range(pw.T_READ):
            self.net.step(I_stimulus=stim)
        diff = self.net.model.u_c[idle].mean() - self.net.model.u_c[self.mask].mean()
        return 1 if float(np.real(diff)) >= 0 else -1


def run_chain(chain_seed, condition):
    mask, idle = pw.build_group(chain_seed)  # MEME groupe physique dans les 2 conditions
    persistent = PersistentReader(chain_seed, mask) if condition == "persistent" else None

    blind_total, warm_total, coupled_total = 0, 0, 0
    per_round_correct = []
    for k in range(K):
        seed_k = chain_seed * 100 + k
        b_k = round_b(chain_seed, k)
        pb = pw.make_problem(seed_k, b_k)

        blind_total += pw.solve(pb, x0=0.0)

        if condition == "fresh":
            b_guess = m4r_read_fresh(chain_seed, mask, idle, b_k, k)
        else:
            b_guess = persistent.read(b_k, idle)
        per_round_correct.append(int(b_guess == b_k))

        warm_total += pw.solve(pb, x0=b_guess * pw.X_WARM)
        coupled_total += pc.solve_coupled(pb, b_guess)

    return blind_total, warm_total, coupled_total, per_round_correct


def boot_ci_paired(a, b, n_boot=10000, seed=20260713):
    rng = np.random.RandomState(seed)
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    m = np.empty(n_boot)
    for i in range(n_boot):
        m[i] = d[rng.randint(0, n, n)].mean()
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"=== P11 -- CHAINAGE A {K} TOURS : FRESH vs PERSISTENT vs BLIND ===\n")

    chain_seeds = list(range(N_CHAINS))
    results = {}
    timing = {}
    for cond in ["fresh", "persistent"]:
        tlab = time.time()
        blind_arr, warm_arr, coupled_arr = [], [], []
        correct_by_round = [[] for _ in range(K)]
        t_read0 = time.perf_counter()
        for cs in chain_seeds:
            bt, wt, ct, corr = run_chain(cs, cond)
            blind_arr.append(bt); warm_arr.append(wt); coupled_arr.append(ct)
            for k in range(K):
                correct_by_round[k].append(corr[k])
        t_chain = time.perf_counter() - t_read0
        correct_mat = np.array(correct_by_round)  # shape (K, N_CHAINS)
        results[cond] = {
            "blind": np.array(blind_arr), "warm": np.array(warm_arr),
            "coupled": np.array(coupled_arr),
            "acc_by_round": [np.mean(c) for c in correct_by_round],
            "acc_overall": np.mean([x for c in correct_by_round for x in c]),
            "acc_per_chain": correct_mat.mean(axis=0),  # (N_CHAINS,) -- pour bootstrap apparie
            "correct_mat": correct_mat,
        }
        timing[cond] = t_chain
        print(f"[{cond:<11}] {N_CHAINS} chaines x {K} tours, {time.time()-tlab:.1f}s")

    print(f"\n{'condition':<12}{'accuracy globale':>18}" +
          "".join(f"{'tour '+str(k+1):>10}" for k in range(K)))
    print("-" * (12 + 18 + 10 * K))
    for cond in ["fresh", "persistent"]:
        r = results[cond]
        line = f"{cond:<12}{r['acc_overall']:>18.3f}"
        line += "".join(f"{a:>10.3f}" for a in r["acc_by_round"])
        print(line)

    print(f"\n{'condition':<12}{'BLIND (iters)':>16}{'WARM (iters)':>16}{'COUPLED (iters)':>18}"
          f"{'chain wall time (s)':>22}")
    print("-" * 84)
    for cond in ["fresh", "persistent"]:
        r = results[cond]
        print(f"{cond:<12}{r['blind'].mean():>16.0f}{r['warm'].mean():>16.0f}"
              f"{r['coupled'].mean():>18.0f}{timing[cond]:>22.2f}")

    print("\n=== VERDICT (criteres pre-fixes, IC bootstrap apparie sur les CHAINES) ===")
    d1, lo1, hi1 = boot_ci_paired(results["persistent"]["acc_per_chain"],
                                   results["fresh"]["acc_per_chain"])
    tag1 = ("PERSISTENT bat FRESH" if lo1 > 0 else
            ("FRESH bat PERSISTENT" if hi1 < 0 else "parite (IC couvre 0)"))
    print(f"  1. Accuracy globale : FRESH={results['fresh']['acc_overall']:.3f}  "
          f"PERSISTENT={results['persistent']['acc_overall']:.3f}  "
          f"delta={d1:+.3f} CI[{lo1:+.3f},{hi1:+.3f}] -> {tag1}")

    print("  2. Derive par index de tour (tour 1 -> tour 4), IC bootstrap sur les chaines :")
    for cond in ["fresh", "persistent"]:
        mat = results[cond]["correct_mat"]  # (K, N_CHAINS)
        d2, lo2, hi2 = boot_ci_paired(mat[-1], mat[0])  # tour K vs tour 1, apparie par chaine
        tag2 = ("memoire (drift positif confirme)" if lo2 > 0 else
                ("cicatrice (drift negatif confirme)" if hi2 < 0 else
                 "IC couvre 0 -- PAS de drift confirme, bruit d'echantillonnage probable"))
        print(f"     {cond:<11} tour1={mat[0].mean():.3f} -> tour{K}={mat[-1].mean():.3f}  "
              f"drift={d2:+.3f} CI[{lo2:+.3f},{hi2:+.3f}]  -> {tag2}")

    print("  3. Bilan materiel cumule (temps reel de la campagne de lecture, "
          f"{N_CHAINS} chaines x {K} tours) :")
    for cond in ["fresh", "persistent"]:
        per_chain = timing[cond] / N_CHAINS
        print(f"     {cond:<11} temps lecture total/chaine = {per_chain*1000:.2f} ms "
              f"({per_chain/K*1000:.2f} ms/tour)")
    gain_persist = 100 * (timing["fresh"] - timing["persistent"]) / timing["fresh"]
    print(f"     -> PERSISTENT {'economise' if gain_persist > 0 else 'coute'} "
          f"{abs(gain_persist):.0f}% de temps de lecture vs FRESH "
          f"(construction de reseau evitee sur {K-1}/{K} tours).")

    with (FIG / "p11_multiround_chain_poc.csv").open("w", encoding="utf-8") as f:
        f.write("condition,chain_seed,blind_iters,warm_iters,coupled_iters\n")
        for cond in ["fresh", "persistent"]:
            r = results[cond]
            for i, cs in enumerate(chain_seeds):
                f.write(f"{cond},{cs},{r['blind'][i]},{r['warm'][i]},{r['coupled'][i]}\n")
    with (FIG / "p11_multiround_chain_poc_by_round.csv").open("w", encoding="utf-8") as f:
        f.write("condition,round,accuracy\n")
        for cond in ["fresh", "persistent"]:
            for k, a in enumerate(results[cond]["acc_by_round"]):
                f.write(f"{cond},{k+1},{a:.6f}\n")
    print(f"\n[csv] {FIG / 'p11_multiround_chain_poc.csv'}")
    print(f"[csv] {FIG / 'p11_multiround_chain_poc_by_round.csv'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
        ax = axes[0]
        rounds = np.arange(1, K + 1)
        ax.plot(rounds, results["fresh"]["acc_by_round"], "o-", label="FRESH", color="#1f77b4")
        ax.plot(rounds, results["persistent"]["acc_by_round"], "o-", label="PERSISTENT", color="#d62728")
        ax.set_xlabel("tour")
        ax.set_ylabel("accuracy de lecture M4R")
        ax.set_title("Accuracy par tour -- memoire ou cicatrice ?")
        ax.set_xticks(rounds)
        ax.legend()
        ax.grid(alpha=0.3)
        ax = axes[1]
        labels = ["BLIND", "WARM", "COUPLED"]
        x = np.arange(len(labels))
        w = 0.35
        fresh_means = [results["fresh"][k.lower()].mean() for k in labels]
        pers_means = [results["persistent"][k.lower()].mean() for k in labels]
        ax.bar(x - w / 2, fresh_means, w, label="FRESH", color="#1f77b4")
        ax.bar(x + w / 2, pers_means, w, label="PERSISTENT", color="#d62728")
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel(f"iterations solveur cumulees ({K} tours)")
        ax.set_title("Cout solveur cumule sur la chaine")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        fig.suptitle(f"P11 -- chainage a {K} tours : FRESH vs PERSISTENT", fontsize=11)
        plt.tight_layout()
        plt.savefig(FIG / "p11_multiround_chain_poc.png", dpi=140)
        print(f"[png] {FIG / 'p11_multiround_chain_poc.png'}")
    except Exception as e:
        print(f"[png] skipped: {e}")

    print(f"\nWall time: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
