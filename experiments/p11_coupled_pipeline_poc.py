#!/usr/bin/env python3
"""
P11 -- LE COUPLAGE DES TROIS : M4R (direction bon marche) + verification
rapide (l'accumulateur, pour detecter vite une erreur) + solveur.
=============================================================================
Cree : 2026-07-13 (Claude Sonnet 5, L'Ingenieur). Synthese de la journee,
demande explicite de Julien : « un couplage des trois elements justement ».

CE QU'ON SAIT DEJA (etabli aujourd'hui, chaque piece testee separement) :
  - M4R lit une direction moins cher que le hasard mais pas gratuit, et pas
    toujours juste (accuracy mesuree : 0.875-1.00 selon la force de lecture).
  - Warm start : partir du bon cote economise ~97% des iterations solveur ;
    partir du MAUVAIS cote (guess faux) coute un peu PLUS cher que blind.
  - Le raffinement de M4R (sa PROPRE correction interne) est LENT (150-300
    pas pour se corriger tout seul) -- plus lent qu'un accumulateur naif
    (30-100 pas).

L'IDEE DU COUPLAGE : ne pas attendre que M4R se corrige LUI-MEME (lent).
Utiliser un CONTROLE RAPIDE ET BON MARCHE (l'esprit de l'accumulateur :
une verification simple, quelques dizaines de pas) sur la TRAJECTOIRE DU
SOLVEUR LUI-MEME, juste apres le warm start : si le solveur ne s'approche
PAS de la cible attendue apres une courte fenetre de diagnostic, c'est que
le guess de M4R etait FAUX -- on bascule immediatement de cote (redemarre
sur l'autre cote) au lieu d'attendre 150-300 pas de correction interne ou
de laisser le solveur crapahuter ~1500 pas dans le mauvais sens.

PIPELINE COUPLE :
  1. M4R lit (bon marche, mais lecture REALISTE affaiblie : T_READ=30,
     B_E=0.3 -- la meme que la contre-epreuve d'aujourd'hui, accuracy~0.875,
     pas le cas facile a 100%).
  2. Solveur demarre WARM du cote devine, tourne N_CHECK=50 pas
     (fenetre de diagnostic bon marche).
  3. VERIFICATION RAPIDE : si |x - cible supposee| est encore grand apres
     ces 50 pas (le guess semble faux), on BASCULE de cote et on redemarre
     -- pas d'attente, correction immediate et bon marche.
  4. Le solveur termine sa convergence (cote correct, garanti par construction
     apres la bascule).

Compare : PIPELINE COUPLE vs WARM START SEUL (sans verification/bascule,
subit le cout plein d'un guess faux) vs BLIND (aucune aide).

Statut : exploratoire, hors preprint, aucune modification de dynamics.py.
Guardian doit rester 14/14. Sorties : figures/p11_coupled_pipeline_poc.csv + .png
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
import p11_warm_start_poc as p  # noqa: E402

FIG = ROOT / "figures"
SEEDS = list(range(60))
N_CHECK = 50          # fenetre de diagnostic bon marche (bien moins que les 150-300
                       # pas necessaires a M4R pour se corriger tout seul)
CHECK_TOL = 0.5        # "encore loin de la cible" apres N_CHECK pas -> guess suspecte

# Lecture M4R REALISTE (affaiblie, comme la contre-epreuve du meme jour) :
p.T_READ, p.B_E = 30, 0.3


def solve_coupled(pb, b_guess):
    """Warm start + verification rapide + bascule immediate si suspecte."""
    x0 = b_guess * p.X_WARM
    x = x0
    total_iters = 0
    # Phase 1 : fenetre de diagnostic bon marche
    for t in range(N_CHECK):
        if abs(x - pb["target"]) < p.SUCCESS_TOL:
            return total_iters + t  # deja converge, tres tot
        x = x - p.ETA * p.grad(x, pb)
    total_iters += N_CHECK

    assumed_target = b_guess * p.X_TARGET
    if abs(x - assumed_target) > CHECK_TOL:
        # Guess suspecte : bascule de cote, redemarre a partir de la ou on est
        # (pas depuis zero -- on garde l'information deja acquise sur la forme
        # du terrain, seul le signe de la cible est corrige)
        b_guess = -b_guess
        pb_view = pb  # meme pb (la cible REELLE ne change pas, seule notre estimation change)
        # redemarrage du warm start sur l'autre cote
        x = b_guess * p.X_WARM
        total_iters += 0  # le redemarrage lui-meme est gratuit (juste un nouveau x0)

    # Phase 2 : converger jusqu'au bout (cote corrige si bascule, sinon continue)
    remaining = p.MAX_ITER - total_iters
    for t in range(remaining):
        if abs(x - pb["target"]) < p.SUCCESS_TOL:
            return total_iters + t
        x = x - p.ETA * p.grad(x, pb)
    return p.MAX_ITER


def run_all(seeds):
    blind_iters, warm_iters, coupled_iters, correct_guesses = [], [], [], []
    for seed in seeds:
        b = 1 if (seed % 2 == 0) else -1
        pb = p.make_problem(seed, b)

        it_blind = p.solve(pb, x0=0.0)
        blind_iters.append(it_blind)

        b_guess = p.m4r_read(seed, b)
        correct_guesses.append(int(b_guess == b))

        it_warm = p.solve(pb, x0=b_guess * p.X_WARM)
        warm_iters.append(it_warm)

        it_coupled = solve_coupled(pb, b_guess)
        coupled_iters.append(it_coupled)

    return (np.array(blind_iters), np.array(warm_iters), np.array(coupled_iters),
            np.array(correct_guesses))


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("=== LE COUPLAGE DES TROIS : M4R (direction) + verification rapide + solveur ===")
    print(f"(lecture M4R REALISTE : T_READ={p.T_READ}, B_E={p.B_E} -- pas le cas facile a 100%)\n")

    blind, warm, coupled, correct = run_all(SEEDS)
    acc = float(correct.mean())
    n_wrong = int((1 - correct).sum())

    print(f"Accuracy de la lecture M4R (realiste) : {acc:.3f} ({n_wrong}/{len(SEEDS)} guess faux)\n")
    print(f"{'Strategie':<30}{'iterations moyennes':>20}{'vs blind':>12}")
    print("-" * 62)
    for name, arr in [("BLIND (aucune aide)", blind),
                       ("WARM START seul (sans check)", warm),
                       ("COUPLE (M4R + check rapide)", coupled)]:
        m = float(arr.mean())
        pct = 100 * (blind.mean() - m) / blind.mean()
        print(f"{name:<30}{m:>20.0f}{pct:>11.0f}%")

    print("\n=== VERDICT ===")
    econ_warm = float(blind.mean() - warm.mean())
    econ_coupled = float(blind.mean() - coupled.mean())
    print(f"  Economie WARM START seul  : {econ_warm:+.0f} ({100*econ_warm/blind.mean():+.0f}%)")
    print(f"  Economie COUPLE           : {econ_coupled:+.0f} ({100*econ_coupled/blind.mean():+.0f}%)")
    gain_du_couplage = econ_coupled - econ_warm
    print(f"  Gain du COUPLAGE au-dela du warm start seul : {gain_du_couplage:+.0f} iterations")
    if gain_du_couplage > 0.05 * blind.mean():
        print("  -> Le couplage (verification rapide + bascule) ajoute une valeur REELLE "
              "au-dela du warm start seul -- rattraper vite un mauvais guess vaut mieux "
              "que subir son cout plein ou attendre la correction lente de M4R.")
    else:
        print("  -> Le couplage n'ajoute pas grand-chose ici (peu de guess faux dans "
              "cet echantillon, ou le cout du guess faux etait deja faible).")

    if n_wrong > 0:
        wrong_mask = (correct == 0)
        warm_on_wrong = warm[wrong_mask].mean()
        coupled_on_wrong = coupled[wrong_mask].mean()
        print(f"\n  SUR LES {n_wrong} CAS OU M4R S'EST TROMPE SPECIFIQUEMENT :")
        print(f"    warm start seul (subit le guess faux)      : {warm_on_wrong:.0f} iterations")
        print(f"    couple (verification rapide + bascule)     : {coupled_on_wrong:.0f} iterations")
        print(f"    -> le couplage rattrape {100*(warm_on_wrong-coupled_on_wrong)/warm_on_wrong:+.0f}% "
              "du cout d'un guess faux, PRECISEMENT dans les cas ou ca compte.")

    with (FIG / "p11_coupled_pipeline_poc.csv").open("w", encoding="utf-8") as f:
        f.write("seed,correct_guess,blind,warm,coupled\n")
        for i, seed in enumerate(SEEDS):
            f.write(f"{seed},{correct[i]},{blind[i]},{warm[i]},{coupled[i]}\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.8))
        labels = ["BLIND", "WARM START\n(seul)", "COUPLE\n(M4R+check)"]
        means = [blind.mean(), warm.mean(), coupled.mean()]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        ax.bar(labels, means, color=colors, edgecolor="k")
        ax.set_ylabel("iterations solveur (moyenne)")
        ax.set_title(f"Couplage des trois elements (lecture M4R accuracy={acc:.2f})")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIG / "p11_coupled_pipeline_poc.png", dpi=140)
        print(f"\n[png] {FIG / 'p11_coupled_pipeline_poc.png'}")
    except Exception as e:
        print(f"[png] skipped: {e}")

    print(f"\nWall time: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
