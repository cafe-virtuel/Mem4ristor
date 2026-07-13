#!/usr/bin/env python3
"""
P11 -- RESERVE 1 : le mecanisme (direction/veille) tient-il en D REELLEMENT
couple, pas juste un cout ajoute a cote (padding decoratif) ?
=============================================================================
Cree : 2026-07-13 (Claude Sonnet 5, L'Ingenieur). Reserve posee explicitement
a la fin de p11_realistic_solver_scaling_poc.py : ce POC-la isolait UNIQUEMENT
le cout (un matvec D x D dont le resultat n'etait jamais reinjecte dans la
dynamique de decision x[0]). Il ne prouvait PAS que le mecanisme (accuracy de
lecture, economie relative d'iterations, bascule de verification) survit
quand le piege lui-meme vit dans un etat D-dimensionnel GENUINEMENT couple.

DESIGN. Generalisation reelle, pas cosmetique :
  - Etat x en R^D (D=10000, le seuil trouve en structure creuse -- reserve 2).
  - Matrice de couplage A CREUSE tridiagonale (bande=3, meme famille que la
    reserve 2), CENTREE SUR 1 (diagonale ~1, hors-diagonale ~coupling_scale)
    -- x[0] (la coordonnee "informative", celle que M4R devine et que le
    plateau piege) n'est PAS isolee : ses voisins x[1] la contaminent
    directement via A, et reciproquement -- un vrai couplage, pas un
    sous-systeme parallele decouple.
  - grad(x) = A @ (x - target) * h(x[0]) -- meme fonction plateau h que
    p11_warm_start_poc.py (memes h_min/x_p/w_flat par seed), SEULE la
    coordonnee x[0] pilote la porte h, mais TOUT le vecteur bouge via A.
  - Convergence et bascule de verification jugees sur x[0] SEUL (c'est la
    coordonnee de decision -- les 9999 autres representent l'etat d'un vrai
    solveur qui doit converger autour, pas l'objet de la decision).
  - Lecture M4R, warm start (x0 = b_guess*X_WARM*e0, reste a 0), pipeline
    couple (verification rapide + bascule) : code IDENTIQUE a
    p11_coupled_pipeline_poc.py, seule la dynamique du solveur change.

PRE-VOL (convention du projet, structure pas p-hacking) : pour chaque
coupling_scale teste, verifier sur un sous-echantillon que BLIND reste
soluble au budget ET que le plateau reste piegeux (creux sous tolerance
avant arrivee) -- sinon, ecarter ce coupling_scale et le documenter, ne
pas forcer un resultat.

CRITERE PRE-FIXE : a coupling_scale calibre, la campagne complete (60 seeds,
comme partout ailleurs dans P11) donne-t-elle (a) une accuracy de lecture
M4R proche de 0.817 (la valeur scalaire deja etablie -- la lecture ne
dependant pas de D, elle ne DEVRAIT pas changer) et (b) un gain reel net en
temps (le -96% en iterations se traduit-il, ICI, nativement, en un vrai
gain de temps CPU, sans modele de cout separe -- le temps mesure EST le
cout complet du solve D-dimensionnel) ?

Statut : exploratoire, hors preprint, aucune modification de dynamics.py.
Guardian doit rester 14/14. Sorties : figures/p11_native_coupled_mechanism_poc.csv + .png
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
import p11_warm_start_poc as pw  # noqa: E402 -- reutilise m4r_read, T_READ=30/B_E=0.3 (via coupled import)
import p11_coupled_pipeline_poc as pc  # noqa: E402 -- met pw.T_READ=30, pw.B_E=0.3 ; fournit SEEDS

FIG = ROOT / "figures"
SEEDS = pc.SEEDS
D = 10_000
ETA = pw.ETA
MAX_ITER = pw.MAX_ITER
SUCCESS_TOL = pw.SUCCESS_TOL
X_TARGET = pw.X_TARGET
X_WARM = pw.X_WARM
W_RAMP = pw.W_RAMP
N_CHECK = pc.N_CHECK
CHECK_TOL = pc.CHECK_TOL
COUPLING_SCALES = [0.0, 0.02, 0.05, 0.1]
PREFLIGHT_SEEDS = list(range(6))


def make_coupling(D, coupling_scale, seed=999):
    rng = np.random.RandomState(800_000 + seed)
    main = np.ones(D)
    off = rng.standard_normal(D - 1) * coupling_scale
    return sparse.diags([off, main, off], offsets=[-1, 0, 1], format="csr")


def make_problem_D(seed, b, D):
    """Meme derivation h_min/w_flat/x_p que pw.make_problem -- seule la
    cible et l'etat initial deviennent des vecteurs de taille D."""
    rng = np.random.RandomState(70000 + seed)
    w_flat = rng.uniform(0.02, 0.03)
    t_c = rng.uniform(700.0, 1400.0)
    x_p_mag = rng.uniform(0.9, 1.3)
    h_min = 2.0 * w_flat / (ETA * x_p_mag * t_c)
    target = np.zeros(D)
    target[0] = b * X_TARGET
    return {"x_p": b * x_p_mag, "w_flat": w_flat, "h_min": h_min, "target": target}


def gate_h(x0, pb):
    d = abs(x0 - pb["x_p"]) - pb["w_flat"]
    if d <= 0:
        return pb["h_min"]
    if d >= W_RAMP:
        return 1.0
    s = d / W_RAMP
    return pb["h_min"] + (1.0 - pb["h_min"]) * s * s * (3.0 - 2.0 * s)


def solve_D(pb, x0, A, max_iter=MAX_ITER):
    """Convergence jugee sur x[0] (coordonnee de decision) -- tout le
    vecteur x evolue sous A, genuinement couple."""
    x = x0.copy()
    for t in range(max_iter):
        if abs(x[0] - pb["target"][0]) < SUCCESS_TOL:
            return t, x[0]
        h = gate_h(x[0], pb)
        x = x - ETA * (A @ (x - pb["target"])) * h
    return max_iter, x[0]


def solve_coupled_D(pb, b_guess, A):
    """Warm start + verification rapide + bascule -- generalisation directe
    de solve_coupled() de p11_coupled_pipeline_poc.py."""
    x = np.zeros(D)
    x[0] = b_guess * X_WARM
    total_iters = 0
    for t in range(N_CHECK):
        if abs(x[0] - pb["target"][0]) < SUCCESS_TOL:
            return total_iters + t
        h = gate_h(x[0], pb)
        x = x - ETA * (A @ (x - pb["target"])) * h
    total_iters += N_CHECK

    assumed_target0 = b_guess * X_TARGET
    if abs(x[0] - assumed_target0) > CHECK_TOL:
        b_guess = -b_guess
        x = np.zeros(D)
        x[0] = b_guess * X_WARM
        total_iters += 0

    remaining = MAX_ITER - total_iters
    for t in range(remaining):
        if abs(x[0] - pb["target"][0]) < SUCCESS_TOL:
            return total_iters + t
        h = gate_h(x[0], pb)
        x = x - ETA * (A @ (x - pb["target"])) * h
    return MAX_ITER


def preflight(coupling_scale):
    """Le plateau reste-t-il soluble ET piegeux sous ce couplage ? (convention
    du projet -- structure verifiee AVANT campagne, pas apres coup)."""
    A = make_coupling(D, coupling_scale)
    bad = 0
    for seed in PREFLIGHT_SEEDS:
        b = 1 if (seed % 2 == 0) else -1
        pb = make_problem_D(seed, b, D)
        x0 = np.zeros(D)
        it, x0_final = solve_D(pb, x0, A, max_iter=MAX_ITER)
        soluble = it < MAX_ITER
        # creux : re-simule en trace courte pour verifier que le residu
        # a bien touche la zone du plateau avant d'arriver (piege reel)
        x = x0.copy()
        min_gap_to_plateau = np.inf
        for t in range(min(it + 1, MAX_ITER)):
            gap = abs(abs(x[0] - pb["x_p"]) - pb["w_flat"])
            min_gap_to_plateau = min(min_gap_to_plateau, gap)
            if abs(x[0] - pb["target"][0]) < SUCCESS_TOL:
                break
            h = gate_h(x[0], pb)
            x = x - ETA * (A @ (x - pb["target"])) * h
        piege = min_gap_to_plateau < 0.05  # le trajet est bien passe pres du plateau
        if not (soluble and piege):
            bad += 1
    return bad == 0, bad


def run_all_D(seeds, A):
    blind_iters, warm_iters, coupled_iters, correct = [], [], [], []
    for seed in seeds:
        b = 1 if (seed % 2 == 0) else -1
        pb = make_problem_D(seed, b, D)

        x0_blind = np.zeros(D)
        it_blind, _ = solve_D(pb, x0_blind, A)
        blind_iters.append(it_blind)

        b_guess = pw.m4r_read(seed, b)
        correct.append(int(b_guess == b))

        x0_warm = np.zeros(D)
        x0_warm[0] = b_guess * X_WARM
        it_warm, _ = solve_D(pb, x0_warm, A)
        warm_iters.append(it_warm)

        it_coupled = solve_coupled_D(pb, b_guess, A)
        coupled_iters.append(it_coupled)

    return (np.array(blind_iters), np.array(warm_iters), np.array(coupled_iters),
            np.array(correct))


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"=== P11 RESERVE 1 : mecanisme NATIF en D={D} genuinement couple ===\n")

    print("[pre-vol] le plateau reste-t-il soluble ET piegeux sous couplage reel ?")
    valid_scales = []
    for cs in COUPLING_SCALES:
        ok, bad = preflight(cs)
        status = "OK" if ok else f"ECHEC ({bad}/{len(PREFLIGHT_SEEDS)} casses)"
        print(f"  coupling_scale={cs:<6} -> {status}")
        if ok:
            valid_scales.append(cs)
    if not valid_scales:
        print("\n[pre-vol] AUCUN coupling_scale teste ne preserve un piege soluble+piegeux.")
        print("  -> le mecanisme ne peut pas etre teste nativement a cette structure de "
              "couplage ; reserve 1 reste OUVERTE, a retenter avec une structure differente.")
        return 1
    print(f"\n[pre-vol] {len(valid_scales)}/{len(COUPLING_SCALES)} couplages valides : {valid_scales}\n")

    rows = []
    for cs in valid_scales:
        print(f"--- coupling_scale={cs} (D={D}, 60 seeds, campagne complete) ---")
        tlab = time.time()
        A = make_coupling(D, cs)
        t_run0 = time.perf_counter()
        blind, warm, coupled, correct = run_all_D(SEEDS, A)
        wall = time.perf_counter() - t_run0
        acc = float(correct.mean())
        print(f"  accuracy lecture M4R          = {acc:.3f} (reference scalaire : 0.817)")
        print(f"  iterations : BLIND={blind.mean():.0f}  WARM={warm.mean():.0f}  "
              f"COUPLED={coupled.mean():.0f}")
        print(f"  temps CPU MESURE (campagne complete, {len(SEEDS)} seeds x 3 conditions) "
              f"= {wall:.2f}s")
        # temps par seed pour chaque strategie, mesure independamment pour le bilan
        t_blind0 = time.perf_counter()
        for seed in SEEDS[:10]:
            b = 1 if (seed % 2 == 0) else -1
            pb = make_problem_D(seed, b, D)
            solve_D(pb, np.zeros(D), A)
        t_blind_per_seed = (time.perf_counter() - t_blind0) / 10
        t_coupled0 = time.perf_counter()
        for seed in SEEDS[:10]:
            b = 1 if (seed % 2 == 0) else -1
            pb = make_problem_D(seed, b, D)
            b_guess = pw.m4r_read(seed, b)
            solve_coupled_D(pb, b_guess, A)
        t_coupled_per_seed = (time.perf_counter() - t_coupled0) / 10
        gain_pct = 100 * (t_blind_per_seed - t_coupled_per_seed) / t_blind_per_seed
        print(f"  temps reel/seed : BLIND={t_blind_per_seed*1000:.2f}ms  "
              f"COUPLED(+lecture)={t_coupled_per_seed*1000:.2f}ms  "
              f"({gain_pct:+.0f}% vs blind)")
        verdict = "GAIN NET" if gain_pct > 5 else ("PERTE NETTE" if gain_pct < -5 else "quasi neutre")
        print(f"  -> {verdict}  [{time.time()-tlab:.0f}s]\n")
        rows.append((cs, acc, blind.mean(), warm.mean(), coupled.mean(),
                     t_blind_per_seed, t_coupled_per_seed, gain_pct))

    print("=== VERDICT RESERVE 1 ===")
    for cs, acc, bm, wm, cm, tb, tc, pct in rows:
        acc_ok = abs(acc - 0.817) < 0.10
        print(f"  coupling_scale={cs}: accuracy {'tient' if acc_ok else 'DEVIE'} "
              f"({acc:.3f} vs 0.817 scalaire), gain reel {pct:+.0f}%")

    with (FIG / "p11_native_coupled_mechanism_poc.csv").open("w", encoding="utf-8") as f:
        f.write("coupling_scale,accuracy,blind_iters,warm_iters,coupled_iters,"
                "t_blind_per_seed_s,t_coupled_per_seed_s,gain_pct\n")
        for r in rows:
            f.write(",".join(f"{x:.6f}" if isinstance(x, float) else str(x) for x in r) + "\n")
    print(f"\n[csv] {FIG / 'p11_native_coupled_mechanism_poc.csv'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
        ax = axes[0]
        cs_vals = [r[0] for r in rows]
        accs = [r[1] for r in rows]
        ax.plot(cs_vals, accs, "o-", color="#9467bd")
        ax.axhline(0.817, ls="--", c="gray", label="reference scalaire (D=1)")
        ax.set_xlabel("coupling_scale")
        ax.set_ylabel("accuracy lecture M4R")
        ax.set_title(f"Accuracy de lecture en D={D} genuinement couple")
        ax.legend()
        ax.grid(alpha=0.3)
        ax = axes[1]
        pcts = [r[7] for r in rows]
        ax.bar([str(c) for c in cs_vals], pcts, color="#2ca02c")
        ax.axhline(0, c="k", lw=0.8)
        ax.set_xlabel("coupling_scale")
        ax.set_ylabel("gain reel vs blind (%)")
        ax.set_title(f"Gain reel NATIF en D={D} (pas de modele de cout separe)")
        ax.grid(axis="y", alpha=0.3)
        fig.suptitle("P11 reserve 1 -- le mecanisme tient-il en D genuinement couple ?", fontsize=10)
        plt.tight_layout()
        plt.savefig(FIG / "p11_native_coupled_mechanism_poc.png", dpi=140)
        print(f"[png] {FIG / 'p11_native_coupled_mechanism_poc.png'}")
    except Exception as e:
        print(f"[png] skipped: {e}")

    print(f"\nWall time: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
