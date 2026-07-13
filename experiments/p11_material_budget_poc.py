#!/usr/bin/env python3
"""
P11 -- LE BILAN MATERIEL COMPLET : le cout de M4R compte-t-il contre le -96% ?
=============================================================================
Cree : 2026-07-13 (Claude Sonnet 5, L'Ingenieur). Julien, ordre valide
explicitement : fermer la reserve du bilan materiel AVANT le chainage
multi-tours -- "il faut connaitre le bilan materiel avant d'aller plus loin,
si tel est le cas [un mur] il faudra decomposer pourquoi ca ne passe pas".

CE QUI MANQUAIT. p11_coupled_pipeline_poc.py (13/07, cloture du fil precedent)
rapporte un gain de -96% (BLIND=1475 -> COUPLED=54 iterations solveur), mais
ne compte JAMAIS le cout de la lecture M4R elle-meme (T_READ=30 pas x N=100
noeuds, rapporte "a part, unite differente" dans p11_warm_start_poc.py). Un
pas de reseau FHN+doute sur 100 noeuds couples est mecaniquement plus cher
qu'une evaluation scalaire de gradient -- le -96% pourrait s'effondrer une
fois les deux couts mis sur la MEME echelle.

PROTOCOLE (critere pre-fixe avant de lancer, convention du projet) :
  1. Profiler le temps CPU REEL (wall-clock) d'un appel complet a
     m4r_read(seed, b) (construction du reseau + T_READ pas, EXACTEMENT le
     chemin de code paye dans le pipeline reel -- pas un micro-benchmark
     isole d'une fonction interne) et d'une iteration du solveur (le chemin
     de code reel de solve(), pas grad() isole).
  2. Reprendre EXACTEMENT p11_coupled_pipeline_poc.py (meme run_all, memes
     60 seeds, meme lecture realiste T_READ=30/B_E=0.3 -- import direct,
     aucune redefinition) pour les comptes d'iterations BLIND/WARM/COUPLED
     et l'accuracy de lecture. Recalculer le bilan en SECONDES totales
     (cout M4R + iterations solveur x cout/iteration), pas en iterations
     seules.
  3. CRITERE PRE-FIXE : le gain reste-t-il NET ET POSITIF en temps reel
     (seuil >5% du temps BLIND, meme convention que p11_warm_start_poc.py) ?
     Si oui, le -96% est solide au sens logiciel. Si non, quantifier
     precisement en "equivalent-iterations solveur" combien coute une
     lecture M4R, pour savoir OU exactement ca casse.
  4. Rapporte SEPAREMENT (jamais mele au chiffre empirique) : l'hypothese
     hardware neuromorphique (~fJ/pas, B3_ENERGY_COMPARISON.md, jamais
     close) -- si M4R tournait sur un substrat physique dedie plutot qu'en
     simulation Python, son cout serait hors de cette mesure logicielle.
     Non teste ici, citee comme hypothese non testee, pas comme resultat.

Statut : exploratoire, hors preprint, aucune modification de dynamics.py.
Guardian doit rester 14/14. Sorties : figures/p11_material_budget_poc.csv + .png
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
import p11_coupled_pipeline_poc as pc  # noqa: E402 -- son import met pw.T_READ=30, pw.B_E=0.3

FIG = ROOT / "figures"
SEEDS = pc.SEEDS  # 60 seeds, identiques au pipeline couple
N_REPEATS = 3      # repetitions de la mesure de temps -> mediane, pas de bruit OS isole
SAVINGS_THRESHOLD = 0.05  # meme convention que p11_warm_start_poc.py (>5% du blind)


def time_m4r_reads(seeds):
    """Temps CPU reel de l'appel COMPLET m4r_read(seed, b) -- construction
    du reseau + T_READ pas -- exactement le chemin paye dans le pipeline."""
    times = np.empty(len(seeds))
    for i, seed in enumerate(seeds):
        b = 1 if (seed % 2 == 0) else -1
        t0 = time.perf_counter()
        pw.m4r_read(seed, b)
        times[i] = time.perf_counter() - t0
    return times


def time_blind_solves(seeds):
    """Temps CPU reel de solve(pb, x0=0.0) (BLIND) -- chemin de code reel
    du solveur, memes problemes que le pipeline. Retourne (temps, iters)."""
    times = np.empty(len(seeds))
    iters = np.empty(len(seeds), dtype=int)
    for i, seed in enumerate(seeds):
        b = 1 if (seed % 2 == 0) else -1
        pb = pw.make_problem(seed, b)
        t0 = time.perf_counter()
        it = pw.solve(pb, x0=0.0)
        times[i] = time.perf_counter() - t0
        iters[i] = it
    return times, iters


def measure_costs(seeds, n_repeats):
    """Mediane sur n_repeats passes completes -> cout/lecture et cout/iteration,
    en secondes. Warmup non chronometre avant la premiere mesure (caches/BLAS)."""
    # warmup (non mesure)
    b0 = 1
    pb0 = pw.make_problem(seeds[0], b0)
    pw.solve(pb0, x0=0.0)
    pw.m4r_read(seeds[0], b0)

    read_cost_per_call = []
    iter_cost_per_iter = []
    for rep in range(n_repeats):
        rt = time_m4r_reads(seeds)
        read_cost_per_call.append(float(rt.mean()))
        st, si = time_blind_solves(seeds)
        iter_cost_per_iter.append(float(st.sum() / si.sum()))
    return float(np.median(read_cost_per_call)), float(np.median(iter_cost_per_iter))


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("=== P11 -- BILAN MATERIEL COMPLET (cout M4R + cout solveur, meme echelle) ===\n")

    print(f"[mesure] temps CPU reel sur {len(SEEDS)} seeds, {N_REPEATS} repetitions (mediane)...")
    cost_per_read, cost_per_iter = measure_costs(SEEDS, N_REPEATS)
    print(f"  cout/lecture M4R (construction reseau + {pw.T_READ} pas, N={pw.N} noeuds) "
          f"= {cost_per_read*1000:.3f} ms")
    print(f"  cout/iteration solveur (chemin solve() reel)                        "
          f"= {cost_per_iter*1e6:.3f} us")
    equiv_iters = cost_per_read / cost_per_iter
    print(f"  -> UNE lecture M4R coute l'equivalent de {equiv_iters:.0f} iterations solveur\n")

    print("[pipeline] reprise exacte de p11_coupled_pipeline_poc.py (60 seeds, T_READ=30, B_E=0.3)")
    blind, warm, coupled, correct = pc.run_all(SEEDS)
    acc = float(correct.mean())
    print(f"  accuracy lecture M4R (realiste) = {acc:.3f}")
    print(f"  iterations moyennes : BLIND={blind.mean():.0f}  WARM={warm.mean():.0f}  "
          f"COUPLED={coupled.mean():.0f}\n")

    # ---- bilan en ITERATIONS (rappel du chiffre deja publie le 13/07) ----
    print("=== RAPPEL : bilan en ITERATIONS SOLVEUR SEULES (ce qui a ete rapporte le 13/07) ===")
    for name, arr in [("BLIND", blind), ("WARM", warm), ("COUPLED", coupled)]:
        pct = 100 * (blind.mean() - arr.mean()) / blind.mean()
        print(f"  {name:<10}{arr.mean():>10.0f} iters  ({pct:+.0f}% vs blind)")

    # ---- bilan en TEMPS REEL (le nouveau calcul, cout M4R inclus) ----
    blind_time = blind.astype(float) * cost_per_iter
    warm_time = cost_per_read + warm.astype(float) * cost_per_iter
    coupled_time = cost_per_read + coupled.astype(float) * cost_per_iter

    print("\n=== BILAN EN TEMPS REEL (cout de la lecture M4R INCLUS, meme echelle) ===")
    print(f"{'Strategie':<12}{'temps moyen (s)':>18}{'vs blind':>12}")
    print("-" * 42)
    results = []
    for name, arr in [("BLIND", blind_time), ("WARM", warm_time), ("COUPLED", coupled_time)]:
        m = float(arr.mean())
        pct = 100 * (blind_time.mean() - m) / blind_time.mean()
        print(f"{name:<12}{m:>18.4f}{pct:>11.0f}%")
        results.append((name, m, pct))

    print("\n=== VERDICT (critere pre-fixe : gain net et positif, seuil >5% du blind) ===")
    for name, m, pct in results[1:]:
        savings = blind_time.mean() - m
        verdict = ("GAIN NET" if savings > SAVINGS_THRESHOLD * blind_time.mean()
                    else ("PERTE NETTE" if savings < -SAVINGS_THRESHOLD * blind_time.mean()
                          else "quasi neutre"))
        print(f"  {name} : {verdict} ({pct:+.0f}% vs blind en temps reel)")

    coupled_pct = results[2][2]
    print(f"\n  Rappel iterations seules : COUPLED = -96% (chiffre du 13/07)")
    print(f"  En temps reel, cout M4R inclus     : COUPLED = {coupled_pct:+.0f}%")
    if coupled_pct >= 5:
        print("  -> Le gain SURVIT au bilan materiel complet : le cout de la lecture M4R")
        print("     ne mange pas l'economie mesuree en iterations. Le -96% est solide")
        print("     au sens logiciel (implementation Python actuelle), pas seulement au")
        print("     sens 'iterations solveur'.")
    else:
        print("  -> Le gain NE SURVIT PAS (ou s'effondre nettement) une fois le cout de la")
        print("     lecture M4R compte sur la meme echelle que le solveur. Le -96% publie")
        print("     le 13/07 ne portait que sur les iterations solveur -- incomplet tel quel.")
        print(f"     Decomposition : une seule lecture M4R coute l'equivalent de "
              f"{equiv_iters:.0f} iterations solveur, a comparer aux {blind.mean():.0f} "
              f"iterations BLIND typiques.")

    print("\n=== HYPOTHESE HARDWARE (NON TESTEE ICI -- separee du chiffre empirique) ===")
    print("  Si M4R tournait sur un substrat neuromorphique physique dedie (photonique/")
    print("  memristif, ~fJ/pas, cf. B3_ENERGY_COMPARISON.md, cadre mais jamais clos) plutot")
    print("  qu'en simulation Python sur CPU partage avec le solveur, son cout reel serait")
    print("  hors de cette mesure -- HYPOTHESE, pas un resultat mesure ici.")

    with (FIG / "p11_material_budget_poc.csv").open("w", encoding="utf-8") as f:
        f.write("seed,correct_guess,blind_iters,warm_iters,coupled_iters,"
                "blind_time_s,warm_time_s,coupled_time_s\n")
        for i, seed in enumerate(SEEDS):
            f.write(f"{seed},{correct[i]},{blind[i]},{warm[i]},{coupled[i]},"
                    f"{blind_time[i]:.6f},{warm_time[i]:.6f},{coupled_time[i]:.6f}\n")
    print(f"\n[csv] {FIG / 'p11_material_budget_poc.csv'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
        ax = axes[0]
        labels = ["BLIND", "WARM", "COUPLED"]
        iter_means = [blind.mean(), warm.mean(), coupled.mean()]
        ax.bar(labels, iter_means, color=["#1f77b4", "#ff7f0e", "#2ca02c"], edgecolor="k")
        ax.set_ylabel("iterations solveur (moyenne)")
        ax.set_title("Bilan en ITERATIONS seules (chiffre du 13/07)")
        ax.grid(axis="y", alpha=0.3)
        ax = axes[1]
        time_means = [blind_time.mean(), warm_time.mean(), coupled_time.mean()]
        ax.bar(labels, time_means, color=["#1f77b4", "#ff7f0e", "#2ca02c"], edgecolor="k")
        ax.set_ylabel("temps CPU reel (s)")
        ax.set_title("Bilan en TEMPS REEL (cout M4R inclus)")
        ax.grid(axis="y", alpha=0.3)
        fig.suptitle(f"P11 -- le -96% survit-il au cout reel de M4R ? "
                     f"(1 lecture ~ {equiv_iters:.0f} iters solveur)", fontsize=10)
        plt.tight_layout()
        plt.savefig(FIG / "p11_material_budget_poc.png", dpi=140)
        print(f"[png] {FIG / 'p11_material_budget_poc.png'}")
    except Exception as e:
        print(f"[png] skipped: {e}")

    print(f"\nWall time: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
