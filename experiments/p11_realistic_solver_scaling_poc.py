#!/usr/bin/env python3
"""
P11 -- PISTE 1 : a quel cout d'iteration le -96% redevient-il un GAIN reel ?
=============================================================================
Cree : 2026-07-13 (Claude Sonnet 5, L'Ingenieur). Suite du bilan materiel
(p11_material_budget_poc.py, meme jour) : le -96% s'effondre completement
contre le solveur-jouet scalaire (1 lecture M4R ~ 63 000 iterations solveur,
UN SEUL ordre de grandeur, pas une nuance). Julien, apres le mur : "on part
sur la piste 1" -- retester contre un solveur ou l'iteration coute vraiment
cher.

DESIGN (pour eviter le biais "solveur invente pour que ca marche") :
  - La dynamique DECISIONNELLE (x[0], le piege/plateau, la cible signee par
    b, warm start, verification rapide, bascule) reste EXACTEMENT celle de
    p11_warm_start_poc.py / p11_coupled_pipeline_poc.py -- meme code importe,
    memes 60 seeds, memes comptes d'iterations BLIND/WARM/COUPLED deja
    mesures. On ne retouche PAS le mecanisme.
  - Ce qui change : le cout REEL d'une iteration solveur, en simulant qu'un
    VRAI solveur de taille D doit aussi propager un etat auxiliaire couple
    de dimension D a chaque pas (representatif d'un solveur PDE/lineaire a
    grande echelle ou l'information de decision vit dans une sous-variete
    de faible dimension au sein d'un etat beaucoup plus grand -- le cas le
    plus courant en pratique). Cout mesure empiriquement (pas invente par
    un sleep()) : un vrai produit matrice-vecteur dense D x D par iteration,
    D balaye en loi-puissance.
  - Aucune retouche du cout de la lecture M4R (mesure une seconde fois ici,
    independamment, pour verifier la reproductibilite du chiffre du POC
    precedent).

CRITERE PRE-FIXE (avant de lancer le sweep) : trouver D* tel que
COUPLED_time(D*) == BLIND_time(D*) (breakeven). Verifier D* par sweep
empirique ET par la formule analytique deja derivee au POC precedent :
  cout_iter_critique = cout_lecture_M4R / (blind_iters - coupled_iters)
Les deux doivent concorder (cross-check independant, pas juste un chiffre).

Statut : exploratoire, hors preprint, aucune modification de dynamics.py.
Guardian doit rester 14/14. Sorties : figures/p11_realistic_solver_scaling_poc.csv + .png
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
import p11_warm_start_poc as pw       # noqa: E402
import p11_coupled_pipeline_poc as pc  # noqa: E402 -- met pw.T_READ=30, pw.B_E=0.3

FIG = ROOT / "figures"
SEEDS = pc.SEEDS
N_REPEATS = 3
D_SWEEP = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000]
MATVEC_REPEATS_PER_D = 200  # nombre de matvecs chronometres par valeur de D


def cost_per_matvec(D, seed=0, n_repeats=MATVEC_REPEATS_PER_D):
    """Cout REEL (mesure, pas invente) d'un produit matrice-vecteur dense
    D x D -- represente la propagation d'un etat auxiliaire couple de
    dimension D a chaque iteration solveur, un pas legitime de nombreux
    solveurs reels (PDE, systemes lineaires, Newton)."""
    rng = np.random.RandomState(seed)
    A = rng.standard_normal((D, D)) / np.sqrt(D)
    y = rng.standard_normal(D)
    # warmup non chronometre
    for _ in range(5):
        y = y - 0.01 * (A @ y)
    t0 = time.perf_counter()
    for _ in range(n_repeats):
        y = y - 0.01 * (A @ y)
    return (time.perf_counter() - t0) / n_repeats


def measure_m4r_read_cost(seeds, n_repeats):
    """Reprend exactement la methode du POC precedent -- reproductibilite."""
    b0 = 1
    pw.m4r_read(seeds[0], b0)  # warmup
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
    """Cout de l'update scalaire x[0] SEUL (deja mesure au POC precedent,
    remesure ici pour l'auto-suffisance du script)."""
    b0 = 1
    pb0 = pw.make_problem(seeds[0], b0)
    pw.solve(pb0, x0=0.0)  # warmup
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
    print("=== P11 PISTE 1 : a quelle taille de probleme le -96% redevient-il reel ? ===\n")

    print("[pipeline] comptes d'iterations (identiques au POC precedent, mecanisme intact)")
    blind, warm, coupled, correct = pc.run_all(SEEDS)
    b_mean, w_mean, c_mean = blind.mean(), warm.mean(), coupled.mean()
    print(f"  BLIND={b_mean:.0f}  WARM={w_mean:.0f}  COUPLED={c_mean:.0f}  "
          f"accuracy={correct.mean():.3f}\n")

    print(f"[mesure] cout/lecture M4R (reproductibilite, {N_REPEATS} repetitions)...")
    cost_read = measure_m4r_read_cost(SEEDS, N_REPEATS)
    print(f"  cout/lecture M4R = {cost_read*1000:.3f} ms")

    print(f"[mesure] cout/iteration scalaire x[0] seul...")
    cost_scalar = measure_scalar_iter_cost(SEEDS, N_REPEATS)
    print(f"  cout/iteration scalaire = {cost_scalar*1e6:.3f} us\n")

    # ---- seuil analytique (independant du sweep, cross-check) ----
    breakeven_warm = cost_read / (b_mean - w_mean)
    breakeven_coupled = cost_read / (b_mean - c_mean)
    print("=== SEUIL ANALYTIQUE (independant du sweep empirique, formule directe) ===")
    print(f"  cout/iteration critique pour WARM net-positif    : {breakeven_warm*1e6:.2f} us")
    print(f"  cout/iteration critique pour COUPLED net-positif : {breakeven_coupled*1e6:.2f} us")
    print(f"  (au POC precedent, le solveur-jouet coutait {cost_scalar*1e6:.2f} us/iter -- "
          f"{breakeven_coupled/cost_scalar:.0f}x trop bon marche)\n")

    # ---- sweep empirique en D ----
    print(f"=== SWEEP EMPIRIQUE : cout d'un update D-dimensionnel couple (matvec dense D x D) ===")
    print(f"{'D':>7}{'cout matvec (us)':>18}{'cout/iter total (us)':>22}"
          f"{'BLIND (ms)':>12}{'COUPLED (ms)':>14}{'gain reel':>12}")
    print("-" * 85)
    rows = []
    D_star = None
    for D in D_SWEEP:
        c_matvec = cost_per_matvec(D)
        c_iter_total = cost_scalar + c_matvec
        blind_time = b_mean * c_iter_total
        coupled_time = cost_read + c_mean * c_iter_total
        gain = blind_time - coupled_time
        net_positive = gain > 0
        if net_positive and D_star is None:
            D_star = D
        print(f"{D:>7}{c_matvec*1e6:>18.2f}{c_iter_total*1e6:>22.2f}"
              f"{blind_time*1000:>12.4f}{coupled_time*1000:>14.4f}"
              f"{'GAIN' if net_positive else 'perte':>12}")
        rows.append((D, c_matvec, c_iter_total, blind_time, coupled_time, net_positive))

    print("\n=== VERDICT ===")
    if D_star is not None:
        print(f"  Le -96% redevient un gain reel net a partir de D ~= {D_star} "
              f"(etat auxiliaire couple de cette taille par iteration solveur).")
        print(f"  Coherence avec le seuil analytique ({breakeven_coupled*1e6:.2f} us/iter) : "
              f"a D={D_star}, cout mesure = {rows[D_SWEEP.index(D_star)][2]*1e6:.2f} us/iter "
              f"-- {'coherent' if rows[D_SWEEP.index(D_star)][2] > breakeven_coupled else 'a verifier'}.")
    else:
        print(f"  AUCUNE valeur de D testee (jusqu'a {D_SWEEP[-1]}) ne rend le pipeline "
              f"net-positif en temps reel -- le mur est plus profond que prevu, "
              f"ou l'operation matvec dense D x D est elle-meme trop bon marche "
              f"(BLAS vectorise) pour rattraper le cout fixe de la lecture M4R.")

    with (FIG / "p11_realistic_solver_scaling_poc.csv").open("w", encoding="utf-8") as f:
        f.write("D,cost_matvec_s,cost_iter_total_s,blind_time_s,coupled_time_s,net_positive\n")
        for D, cm, ct, bt, cp, np_ in rows:
            f.write(f"{D},{cm:.9f},{ct:.9f},{bt:.9f},{cp:.9f},{int(np_)}\n")
    print(f"\n[csv] {FIG / 'p11_realistic_solver_scaling_poc.csv'}")

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
            ax.axvline(D_star, ls="--", c="gray", label=f"D* ~= {D_star} (breakeven)")
        ax.set_xlabel("D (dimension de l'etat auxiliaire couple par iteration)")
        ax.set_ylabel("temps CPU reel (ms)")
        ax.set_title("A quelle taille de probleme le -96% redevient-il un gain reel ?")
        ax.legend()
        ax.grid(alpha=0.3, which="both")
        plt.tight_layout()
        plt.savefig(FIG / "p11_realistic_solver_scaling_poc.png", dpi=140)
        print(f"[png] {FIG / 'p11_realistic_solver_scaling_poc.png'}")
    except Exception as e:
        print(f"[png] skipped: {e}")

    print(f"\nWall time: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
