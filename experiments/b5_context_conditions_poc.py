#!/usr/bin/env python3
"""
B5 -- QUI EST RESPONSABLE ? Le doute, ou l'architecture FHN+lattice elle-meme ?
=============================================================================
Cree : 2026-07-13 (Claude Sonnet 5, L'Ingenieur). Suite directe de
`b5_context_reinjection_poc.py` (meme jour), qui a trouve un resultat
surprenant : sur NARMA10, une regression lineaire sur le CONTEXTE SEUL
(u[t-9..t] brut, aucun reservoir) fait NRMSE=0.829 -- MEILLEUR que
Mem4ristor FULL avec ou sans contexte (1.88-1.94). Le reservoir de M4R
fait donc PIRE que ne rien faire sur cette tache.

QUESTION LAISSEE OUVERTE (demande de Julien : « 1 et une fois trouve on
passera a 2 ») : est-ce le DOUTE (la dynamique u qui pilote l'exploration/
anti-synchronisation) qui degrade l'information, ou l'architecture
FHN+lattice ELLE-MEME (le couplage spatial D_eff, independamment de tout
mecanisme de doute) ? `reservoir_narma10_poc.py` (B1, 06/07) definit deja
3 conditions comparables :
  - FULL      : doute actif (epsilon_u=0.02) -- le M4R complet.
  - FROZEN_U  : doute gele (epsilon_u=0) -- couplage FHN+lattice SANS le
                mecanisme de doute (u fixe a sa valeur initiale).
  - DECOUPLE  : D=0 -- noeuds FHN independants, AUCUN couplage spatial du
                tout (baseline la plus simple, chaque noeud isole).
Si FROZEN_U (couplage sans doute) bat deja le contexte seul (0.829), le
doute est le coupable specifique. Si FROZEN_U est AUSSI pire que le
contexte seul, le probleme est le COUPLAGE FHN lui-meme (le doute n'est
qu'un facteur aggravant, pas la cause premiere). Si meme DECOUPLE (noeuds
FHN isoles, sans aucun couplage) est pire que le contexte seul, le
probleme remonte a la dynamique FHN elle-meme (le noeud individuel,
independamment de tout reseau).

Protocole : reutilise integralement rc.run_reservoir (FULL/FROZEN_U/
DECOUPLE), meme contexte brut (u[t-9..t]) que b5_context_reinjection_poc.py,
memes seeds/scales. Contexte ajoute au niveau du readout uniquement,
aucune nouvelle simulation reseau au-dela de ce que B1/B5 faisaient deja.

Statut : exploratoire, hors preprint, aucune modification de dynamics.py.
Guardian doit rester 14/14. Sorties : figures/b5_context_conditions_poc.csv + .png
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
import reservoir_narma10_poc as rc  # noqa: E402

FIG = ROOT / "figures"
N = rc.SIZE * rc.SIZE
SEEDS = list(range(8))
K_CONTEXT = 10
CONDITIONS = ["FULL", "FROZEN_U", "DECOUPLE"]
N_BOOT = 10000
RNG_BOOT = np.random.RandomState(20260713)


def build_context_features(u_in, k_context):
    n = len(u_in)
    feats = np.zeros((n, k_context))
    feats[:, 0] = u_in
    for lag in range(1, k_context):
        feats[lag:, lag] = u_in[:-lag]
    return feats


def ridge_nrmse_generic(feature_blocks, target):
    X = np.hstack([np.ones((target.shape[0], 1))] + feature_blocks)
    tr = slice(rc.T_WASH, rc.T_WASH + rc.T_TRAIN)
    te = slice(rc.T_WASH + rc.T_TRAIN, rc.T_WASH + rc.T_TRAIN + rc.T_TEST)
    Xtr, Ytr = X[tr], target[tr]
    Xte, Yte = X[te], target[te]
    F = Xtr.shape[1]
    W = np.linalg.solve(Xtr.T @ Xtr + rc.RIDGE_REG * np.eye(F), Xtr.T @ Ytr)
    pred = Xte @ W
    return float(np.sqrt(np.mean((pred - Yte) ** 2) / (np.var(Yte) + 1e-12)))


def boot_ci(a, ref):
    """IC bootstrap sur (a - ref), ref = scalaire (ex. contexte seul)."""
    d = np.asarray(a, float) - ref
    n = len(d)
    m = np.empty(N_BOOT)
    for k in range(N_BOOT):
        m[k] = d[RNG_BOOT.randint(0, n, n)].mean()
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_steps = rc.T_WASH + rc.T_TRAIN + rc.T_TEST

    results = {c: {"nc": [], "c": []} for c in CONDITIONS}
    context_only = []
    rows = []

    print(f"{'seed':>5}" + "".join(f"{c+'_nc':>12}{c+'_ctx':>12}" for c in CONDITIONS) + f"{'ctx_seul':>10}")
    print("-" * (5 + 24 * len(CONDITIONS) + 10))
    for seed in SEEDS:
        u_in, y = rc.make_narma10(n_steps, seed=seed)
        ctx = build_context_features(u_in, K_CONTEXT)
        rng_mask = np.random.default_rng(1000 + seed)
        w_in_base = rng_mask.uniform(-1.0, 1.0, N)

        row_vals = {"seed": seed}
        for cond in CONDITIONS:
            best_nc, best_c = np.inf, np.inf
            for scale in rc.INPUT_SCALES:
                states = rc.run_reservoir(u_in, w_in_base * scale, cond, seed)
                nrmse_nc = ridge_nrmse_generic([states], y)
                nrmse_c = ridge_nrmse_generic([states, ctx], y)
                best_nc = min(best_nc, nrmse_nc)
                best_c = min(best_c, nrmse_c)
            results[cond]["nc"].append(best_nc)
            results[cond]["c"].append(best_c)
            row_vals[f"{cond}_nc"] = best_nc
            row_vals[f"{cond}_c"] = best_c

        nrmse_ctx_only = ridge_nrmse_generic([ctx], y)
        context_only.append(nrmse_ctx_only)
        row_vals["context_only"] = nrmse_ctx_only

        rows.append(row_vals)
        print(f"{seed:>5}" + "".join(f"{row_vals[f'{c}_nc']:>12.4f}{row_vals[f'{c}_c']:>12.4f}" for c in CONDITIONS)
              + f"{nrmse_ctx_only:>10.4f}")

    ctx_only_arr = np.array(context_only)
    ctx_only_mean = float(ctx_only_arr.mean())

    print("\n=== RESUME (NRMSE NARMA10, plus bas = mieux) ===")
    print(f"  Contexte SEUL (aucun reservoir) : {ctx_only_mean:.4f} +/- {ctx_only_arr.std():.4f}")
    for cond in CONDITIONS:
        nc = np.array(results[cond]["nc"])
        c = np.array(results[cond]["c"])
        print(f"  {cond:<10} sans contexte : {nc.mean():.4f} +/- {nc.std():.4f}   "
              f"avec contexte : {c.mean():.4f} +/- {c.std():.4f}")

    print("\n=== VERDICT : qui est responsable ? ===")
    for cond in CONDITIONS:
        nc = np.array(results[cond]["nc"])
        d, lo, hi = boot_ci(nc, ctx_only_mean)
        beats = "BAT le contexte seul" if hi < 0 else ("PIRE que le contexte seul" if lo > 0 else "statistiquement egal au contexte seul")
        print(f"  {cond:<10} vs contexte seul : delta = {d:+.4f} CI[{lo:+.4f},{hi:+.4f}]  -> {beats}")

    frozen_nc = np.array(results["FROZEN_U"]["nc"])
    decouple_nc = np.array(results["DECOUPLE"]["nc"])
    full_nc = np.array(results["FULL"]["nc"])
    print()
    if np.mean(frozen_nc) < ctx_only_mean and np.mean(decouple_nc) < ctx_only_mean:
        print("  -> FROZEN_U et DECOUPLE battent DEJA le contexte seul : le probleme est "
              "SPECIFIQUEMENT le DOUTE (FULL), pas l'architecture FHN+lattice.")
    elif np.mean(decouple_nc) < ctx_only_mean <= np.mean(frozen_nc):
        print("  -> DECOUPLE (noeuds isoles) bat le contexte seul, mais FROZEN_U (couplage SANS "
              "doute) ne le bat pas : c'est le COUPLAGE SPATIAL (D_eff) qui degrade, pas le doute "
              "ni le noeud FHN seul -- le doute (FULL) n'est qu'un facteur aggravant secondaire.")
    elif np.mean(decouple_nc) >= ctx_only_mean:
        print("  -> MEME DECOUPLE (noeuds FHN isoles, sans reseau ni doute) est pire ou egal au "
              "contexte seul : le probleme remonte a la dynamique FHN ELLE-MEME, independamment "
              "de tout couplage ou de tout doute -- le noeud individuel n'est deja pas un bon "
              "encodeur lineaire de son historique d'entree sur cette tache.")

    with (FIG / "b5_context_conditions_poc.csv").open("w", encoding="utf-8") as f:
        cols = ["seed"] + [f"{c}_nc" for c in CONDITIONS] + [f"{c}_c" for c in CONDITIONS] + ["context_only"]
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(f"{r[k]:.6f}" if isinstance(r[k], float) else str(r[k]) for k in cols) + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        labels = []
        means = []
        colors = []
        color_map = {"FULL": "#2ca02c", "FROZEN_U": "#ff7f0e", "DECOUPLE": "#7f7f7f"}
        for cond in CONDITIONS:
            labels += [f"{cond}\nsans ctx", f"{cond}\n+ctx"]
            means += [np.mean(results[cond]["nc"]), np.mean(results[cond]["c"])]
            colors += [color_map[cond], color_map[cond]]
        labels.append("contexte\nseul")
        means.append(ctx_only_mean)
        colors.append("#1f77b4")
        ax.bar(labels, means, color=colors, edgecolor="k")
        ax.axhline(1.0, ls=":", c="red", label="NRMSE=1 (predire la moyenne)")
        ax.axhline(ctx_only_mean, ls="--", c="#1f77b4", alpha=0.6, label="contexte seul (reference)")
        ax.set_ylabel("NARMA10 NRMSE"); ax.legend(fontsize=7)
        ax.set_title("Qui degrade l'info : le doute, le couplage, ou le noeud FHN ?")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(FIG / "b5_context_conditions_poc.png", dpi=140)
        print(f"\n[png] {FIG / 'b5_context_conditions_poc.png'}")
    except Exception as e:
        print(f"[png] skipped: {e}")

    print(f"\nWall time: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
