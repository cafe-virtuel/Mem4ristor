#!/usr/bin/env python3
"""
P11 -- RESERVE 2 : le seuil D*~=300 (matvec DENSE) tient-il avec un cout
CREUX/structure, plus representatif d'un vrai solveur PDE/systeme lineaire ?
=============================================================================
Cree : 2026-07-13 (Claude Sonnet 5, L'Ingenieur). Reserve posee explicitement
a la fin de p11_realistic_solver_scaling_poc.py (meme jour) : le matvec dense
D x D est le cout par-iteration le PLUS CHER plausible -- un vrai solveur a
structure creuse (bande, tridiagonale -- le cas typique d'une discretisation
1D par differences finies, Laplacien de graphe, etc.) coute normalement
O(D) et non O(D^2), ce qui repousserait D* plus loin que 300.

PROTOCOLE. Repete EXACTEMENT le sweep de p11_realistic_solver_scaling_poc.py
(meme grille de D, meme cout scalaire x[0], meme cout lecture M4R, memes
comptes d'iterations BLIND/WARM/COUPLED), en remplacant SEULEMENT le cout
par-iteration : matvec DENSE (D x D, O(D^2)) -> matvec CREUX tridiagonal
(bande=3, O(D), scipy.sparse.csr) -- l'operateur le plus courant en pratique
(Laplacien 1D discretise). Meme critere pre-fixe : trouver D* (breakeven),
comparer au D* dense (~300) et au seuil analytique (inchange, ~10.6 us/iter,
independant du type de matrice -- seul le TEMPS pour l'atteindre depend de
la structure).

Statut : exploratoire, hors preprint, aucune modification de dynamics.py.
Guardian doit rester 14/14. Sorties : figures/p11_sparse_cost_reserve_poc.csv + .png
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
from scipy import sparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
import p11_warm_start_poc as pw       # noqa: E402
import p11_coupled_pipeline_poc as pc  # noqa: E402

FIG = ROOT / "figures"
SEEDS = pc.SEEDS
N_REPEATS = 3
# grille etendue -- le creux etant O(D) au lieu de O(D^2), D* attendu bien
# plus grand, il faut sonder plus loin que le sweep dense (qui s'arretait a 10000)
D_SWEEP = [1, 10, 100, 1000, 1e4, 3e4, 1e5, 3e5, 1e6]
D_SWEEP = [int(d) for d in D_SWEEP]
MATVEC_REPEATS_PER_D = 200


def cost_per_sparse_matvec(D, seed=0, n_repeats=MATVEC_REPEATS_PER_D):
    """Cout REEL (mesure) d'un matvec CREUX tridiagonal D x D (bande=3,
    O(D)) -- represente un operateur de type Laplacien 1D discretise, le
    cas structure le plus courant en pratique (PDE, systemes lineaires)."""
    rng = np.random.RandomState(seed)
    main = rng.standard_normal(D) * 0.1 + 2.0
    off = rng.standard_normal(D - 1) * 0.05 - 1.0
    A = sparse.diags([off, main, off], offsets=[-1, 0, 1], format="csr")
    y = rng.standard_normal(D)
    for _ in range(5):
        y = y - 0.01 * (A @ y)
    t0 = time.perf_counter()
    for _ in range(n_repeats):
        y = y - 0.01 * (A @ y)
    return (time.perf_counter() - t0) / n_repeats


def measure_m4r_read_cost(seeds, n_repeats):
    b0 = 1
    pw.m4r_read(seeds[0], b0)
    per_rep = []
    for _ in range(n_repeats):
        times = np.empty(len(seeds))
        for i, seed in enumerate(seeds):
            b = 1 if (seed % 2 == 0) else -1
            t0 = time.perf_counter()
            pw.m4r_read(seed, b)
            times[i] = time.perf_counter() - t0
        per_rep.append(float(times.mean()))
    return float(np.median(per_rep))


def measure_scalar_iter_cost(seeds, n_repeats):
    b0 = 1
    pb0 = pw.make_problem(seeds[0], b0)
    pw.solve(pb0, x0=0.0)
    per_rep = []
    for _ in range(n_repeats):
        total_t, total_i = 0.0, 0
        for seed in seeds:
            b = 1 if (seed % 2 == 0) else -1
            pb = pw.make_problem(seed, b)
            t0 = time.perf_counter()
            it = pw.solve(pb, x0=0.0)
            total_t += time.perf_counter() - t0
            total_i += it
        per_rep.append(total_t / total_i)
    return float(np.median(per_rep))


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("=== P11 RESERVE 2 : matvec CREUX (bande=3) au lieu de DENSE -- D* se deplace-t-il ? ===\n")

    blind, warm, coupled, correct = pc.run_all(SEEDS)
    b_mean, w_mean, c_mean = blind.mean(), warm.mean(), coupled.mean()
    print(f"[pipeline] BLIND={b_mean:.0f}  WARM={w_mean:.0f}  COUPLED={c_mean:.0f}  "
          f"accuracy={correct.mean():.3f}\n")

    cost_read = measure_m4r_read_cost(SEEDS, N_REPEATS)
    cost_scalar = measure_scalar_iter_cost(SEEDS, N_REPEATS)
    breakeven_coupled = cost_read / (b_mean - c_mean)
    print(f"[mesure] cout/lecture M4R = {cost_read*1000:.3f} ms | "
          f"cout/iter scalaire = {cost_scalar*1e6:.3f} us")
    print(f"[seuil]  cout/iteration critique (inchange, independant de la structure) "
          f"= {breakeven_coupled*1e6:.2f} us\n")

    print(f"{'D':>9}{'cout matvec creux (us)':>24}{'cout/iter total (us)':>22}"
          f"{'BLIND (ms)':>12}{'COUPLED (ms)':>14}{'gain reel':>12}")
    print("-" * 95)
    rows = []
    D_star = None
    for D in D_SWEEP:
        c_mv = cost_per_sparse_matvec(D)
        c_iter_total = cost_scalar + c_mv
        blind_time = b_mean * c_iter_total
        coupled_time = cost_read + c_mean * c_iter_total
        gain = blind_time - coupled_time
        net_positive = gain > 0
        if net_positive and D_star is None:
            D_star = D
        print(f"{D:>9}{c_mv*1e6:>24.3f}{c_iter_total*1e6:>22.3f}"
              f"{blind_time*1000:>12.4f}{coupled_time*1000:>14.4f}"
              f"{'GAIN' if net_positive else 'perte':>12}")
        rows.append((D, c_mv, c_iter_total, blind_time, coupled_time, net_positive))

    print("\n=== VERDICT (reserve 2) ===")
    if D_star is not None:
        print(f"  D* (creux/bande) ~= {D_star}, contre D* (dense) ~= 300 mesure au POC precedent.")
        ratio = D_star / 300
        print(f"  -> le seuil se deplace d'un facteur ~{ratio:.0f}x avec une structure creuse "
              f"realiste (bande=3, O(D)) au lieu d'une matrice dense (O(D^2)).")
        if D_star > 1_000_000:
            print("  -> Ce niveau de D depasse la taille de la plupart des problemes reels : "
                  "avec une structure BANDE MINIMALE (bande=3), le bilan materiel de M4R ne "
                  "devient favorable qu'a une echelle extreme. La conclusion favorable du "
                  "sweep DENSE etait optimiste -- elle dependait fortement de la structure "
                  "de couplage supposee, pas seulement de la taille D.")
        else:
            print(f"  -> reste dans une plage de tailles de probleme plausible pour de "
                  f"nombreux solveurs PDE/lineaires reels.")
    else:
        print(f"  AUCUNE valeur testee (jusqu'a D={D_SWEEP[-1]}) ne rend le pipeline "
              f"net-positif avec un cout creux -- le bilan materiel de M4R ne tient QUE "
              f"contre des solveurs a cout dense par iteration (matrices pleines), pas "
              f"contre la grande majorite des solveurs structures reels (PDE, elements finis).")

    with (FIG / "p11_sparse_cost_reserve_poc.csv").open("w", encoding="utf-8") as f:
        f.write("D,cost_matvec_sparse_s,cost_iter_total_s,blind_time_s,coupled_time_s,net_positive\n")
        for D, cm, ct, bt, cp, np_ in rows:
            f.write(f"{D},{cm:.9f},{ct:.9f},{bt:.9f},{cp:.9f},{int(np_)}\n")
    print(f"\n[csv] {FIG / 'p11_sparse_cost_reserve_poc.csv'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        Ds = [r[0] for r in rows]
        blind_ms = [r[3] * 1000 for r in rows]
        coupled_ms = [r[4] * 1000 for r in rows]
        ax.loglog(Ds, blind_ms, "o-", label="BLIND", color="#1f77b4")
        ax.loglog(Ds, coupled_ms, "o-", label="COUPLED (M4R + check)", color="#2ca02c")
        if D_star is not None:
            ax.axvline(D_star, ls="--", c="gray", label=f"D* creux ~= {D_star}")
        ax.axvline(300, ls=":", c="orange", label="D* dense (POC precedent) = 300")
        ax.set_xlabel("D (taille de l'etat couple, structure CREUSE bande=3)")
        ax.set_ylabel("temps CPU reel (ms)")
        ax.set_title("Reserve 2 : le seuil D* avec une structure de couplage realiste (creuse)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")
        plt.tight_layout()
        plt.savefig(FIG / "p11_sparse_cost_reserve_poc.png", dpi=140)
        print(f"[png] {FIG / 'p11_sparse_cost_reserve_poc.png'}")
    except Exception as e:
        print(f"[png] skipped: {e}")

    print(f"\nWall time: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
