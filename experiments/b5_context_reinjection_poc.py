#!/usr/bin/env python3
"""
B5 -- L'ANALOGIE LLM : la "memoire" d'un LLM entre deux requetes vient de la
REPRISE DU CONTEXTE (le prompt re-injecte l'historique), pas d'un etat
interne persistant. Meme analogie pour Mem4ristor ?
=============================================================================
Cree : 2026-07-13 (Claude Sonnet 5, L'Ingenieur). Question de Julien : « on
connait le probleme de memoire de M4R, mais si l'on fait l'analogie a un
LLM, il n'en a pas non plus entre chaque requete -- c'est la reprise du
contexte qui fait le travail. Est-ce qu'une analogie similaire peut etre
appliquee a M4R ? »

UNE NUANCE A DEMELER AVANT DE TESTER (elle change ce qu'on teste) : un LLM
sans contexte redemarre a ZERO a CHAQUE appel (pas d'etat entre appels du
tout) -- la "memoire" est PUREMENT EXTERNE (le prompt re-presente
explicitement l'historique). Mem4ristor, lui, a DEJA un etat interne qui
persiste et evolue en CONTINU (v, w, u, u_c ne sont jamais remis a zero
entre les pas) -- ce n'est pas "pas de memoire du tout", c'est "l'etat
persistant existant n'encode pas assez l'historique utile a NARMA10" (B5,
08/07 : ESN 0.351 vs M4R 1.942 sur la meme mesure). Le probleme de M4R
n'est donc pas structurellement identique a celui d'un LLM entre deux
appels -- mais la SOLUTION de l'analogie (re-injecter explicitement
l'historique plutot que de compter sur la memoire interne) reste testable
telle quelle, et c'est exactement ce que ce script fait.

LE TEST HONNETE : NARMA10 (`rc.make_narma10`) definit y[t+1] a partir de
u[t-9] et u[t] EXPLICITEMENT (formule connue). Le protocole standard
(B1/B5) ne donne au readout QUE l'etat courant du reservoir v(t) -- le
reservoir doit avoir "retenu" u[t-9..t] dans sa dynamique recurrente pour
que le readout lineaire y accede. Ici, EN PLUS de v(t), le readout recoit
EXPLICITEMENT la fenetre brute u[t-9..t] (10 valeurs), exactement comme un
LLM re-recoit le texte de la conversation au lieu de compter sur un etat
cache. Teste sur M4R (FULL) ET sur l'ESN (meme augmentation, pour voir si
le contexte aide TOUT LE MONDE ou SPECIFIQUEMENT le modele a memoire
faible) + un controle CONTEXTE SEUL (regression lineaire sur la fenetre
brute, SANS aucun reservoir) pour cadrer le plafond atteignable par un
readout purement lineaire sur une tache qui a un terme non-lineaire
(u[t-9]*u[t]).

Protocole : reutilise integralement rc.make_narma10, rc.run_reservoir
(Mem4ristor FULL), run_esn (b5_esn_comparison.py) -- MEMES seeds, MEMES
grilles d'hyperparametres, AUCUNE nouvelle simulation reseau : le contexte
est ajoute au niveau du READOUT uniquement (comme un LLM, le contexte
n'est pas dans les poids du "modele", il est dans ce qu'on lui montre a
l'inference). Guardian doit rester 14/14 (aucun claim touche).

Sorties : figures/b5_context_reinjection_poc.csv + .png
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
import b5_esn_comparison as esn  # noqa: E402

FIG = ROOT / "figures"
N = rc.SIZE * rc.SIZE
SEEDS = list(range(8))
K_CONTEXT = 10  # meme ordre que NARMA10 (u[t-9]..u[t])
N_BOOT = 10000
RNG_BOOT = np.random.RandomState(20260713)


def build_context_features(u_in, k_context):
    """feats[t, lag] = u_in[t-lag] pour lag=0..k_context-1 (0 = u[t], padding
    a 0 pour t<lag -- sans consequence, T_WASH=300 >> k_context)."""
    n = len(u_in)
    feats = np.zeros((n, k_context))
    feats[:, 0] = u_in
    for lag in range(1, k_context):
        feats[lag:, lag] = u_in[:-lag]
    return feats


def ridge_nrmse_generic(feature_blocks, target):
    """feature_blocks : liste d'arrays (n_steps, k) a concatener (avec le
    biais). Identique a rc.ridge_nrmse mais generalise au nombre de blocs."""
    X = np.hstack([np.ones((target.shape[0], 1))] + feature_blocks)
    tr = slice(rc.T_WASH, rc.T_WASH + rc.T_TRAIN)
    te = slice(rc.T_WASH + rc.T_TRAIN, rc.T_WASH + rc.T_TRAIN + rc.T_TEST)
    Xtr, Ytr = X[tr], target[tr]
    Xte, Yte = X[te], target[te]
    F = Xtr.shape[1]
    W = np.linalg.solve(Xtr.T @ Xtr + rc.RIDGE_REG * np.eye(F), Xtr.T @ Ytr)
    pred = Xte @ W
    return float(np.sqrt(np.mean((pred - Yte) ** 2) / (np.var(Yte) + 1e-12)))


def boot_ci_paired(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    m = np.empty(N_BOOT)
    for k in range(N_BOOT):
        m[k] = d[RNG_BOOT.randint(0, n, n)].mean()
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_steps = rc.T_WASH + rc.T_TRAIN + rc.T_TEST

    m4_nocontext, m4_context = [], []
    esn_nocontext, esn_context = [], []
    context_only = []
    rows = []

    print(f"{'seed':>5}{'M4R':>9}{'M4R+ctx':>10}{'ESN':>9}{'ESN+ctx':>10}{'ctx_seul':>10}")
    print("-" * 55)
    for seed in SEEDS:
        u_in, y = rc.make_narma10(n_steps, seed=seed)
        ctx = build_context_features(u_in, K_CONTEXT)
        rng_mask = np.random.default_rng(1000 + seed)
        w_in_base = rng_mask.uniform(-1.0, 1.0, N)

        # --- Mem4ristor FULL : meilleur input_scale, avec et sans contexte ---
        best_nc, best_c = np.inf, np.inf
        for scale in rc.INPUT_SCALES:
            states = rc.run_reservoir(u_in, w_in_base * scale, "FULL", seed)
            nrmse_nc = ridge_nrmse_generic([states], y)
            nrmse_c = ridge_nrmse_generic([states, ctx], y)
            best_nc = min(best_nc, nrmse_nc)
            best_c = min(best_c, nrmse_c)
        m4_nocontext.append(best_nc)
        m4_context.append(best_c)

        # --- ESN : meilleur (rho, iscale, leak), avec et sans contexte ---
        e_best_nc, e_best_c = np.inf, np.inf
        for rho in esn.ESN_RHO:
            W, w_in = esn.make_esn(seed, rho)
            for iscale in esn.ESN_ISCALE:
                for leak in esn.ESN_LEAK:
                    states = esn.run_esn(u_in, W, w_in, iscale, leak)
                    nrmse_nc = ridge_nrmse_generic([states], y)
                    nrmse_c = ridge_nrmse_generic([states, ctx], y)
                    e_best_nc = min(e_best_nc, nrmse_nc)
                    e_best_c = min(e_best_c, nrmse_c)
        esn_nocontext.append(e_best_nc)
        esn_context.append(e_best_c)

        # --- Contexte SEUL (aucun reservoir, plafond d'un readout lineaire pur) ---
        nrmse_ctx_only = ridge_nrmse_generic([ctx], y)
        context_only.append(nrmse_ctx_only)

        rows.append((seed, best_nc, best_c, e_best_nc, e_best_c, nrmse_ctx_only))
        print(f"{seed:>5}{best_nc:>9.4f}{best_c:>10.4f}{e_best_nc:>9.4f}{e_best_c:>10.4f}{nrmse_ctx_only:>10.4f}")

    m4_nc, m4_c = np.array(m4_nocontext), np.array(m4_context)
    esn_nc, esn_c = np.array(esn_nocontext), np.array(esn_context)
    ctx_only = np.array(context_only)

    print("\n=== RESUME (NRMSE NARMA10, plus bas = mieux) ===")
    print(f"  Mem4ristor FULL, SANS contexte : {m4_nc.mean():.4f} +/- {m4_nc.std():.4f}")
    print(f"  Mem4ristor FULL, AVEC contexte : {m4_c.mean():.4f} +/- {m4_c.std():.4f}")
    print(f"  ESN,             SANS contexte : {esn_nc.mean():.4f} +/- {esn_nc.std():.4f}")
    print(f"  ESN,             AVEC contexte : {esn_c.mean():.4f} +/- {esn_c.std():.4f}")
    print(f"  Contexte SEUL (pas de reservoir): {ctx_only.mean():.4f} +/- {ctx_only.std():.4f}")

    print("\n=== VERDICT ===")
    d_m4, lo_m4, hi_m4 = boot_ci_paired(m4_nc, m4_c)
    print(f"(1) M4R : contexte aide-t-il ? delta(sans-avec) = {d_m4:+.4f} CI[{lo_m4:+.4f},{hi_m4:+.4f}]")
    if lo_m4 > 0:
        print(f"    -> OUI, gain REEL et significatif ({100*d_m4/m4_nc.mean():.0f}% de reduction NRMSE).")
    elif hi_m4 < 0:
        print("    -> Le contexte AGGRAVE M4R (inattendu, a investiguer).")
    else:
        print("    -> Pas de gain significatif (IC couvre 0).")

    d_esn, lo_esn, hi_esn = boot_ci_paired(esn_nc, esn_c)
    print(f"(2) ESN : contexte aide-t-il AUSSI ? delta(sans-avec) = {d_esn:+.4f} CI[{lo_esn:+.4f},{hi_esn:+.4f}]")
    if lo_esn > 0:
        print(f"    -> Le contexte aide l'ESN AUSSI ({100*d_esn/esn_nc.mean():.0f}% de reduction) -- "
              "ce n'est pas specifique a la faiblesse de memoire de M4R, c'est une aide generale.")
    else:
        print("    -> Pas de gain significatif pour l'ESN -- son etat interne couvrait deja cette info.")

    d_gap_nc, lo_nc, hi_nc = boot_ci_paired(m4_nc, esn_nc)
    d_gap_c, lo_c, hi_c = boot_ci_paired(m4_c, esn_c)
    print(f"(3) L'ECART M4R-ESN se referme-t-il avec le contexte ? "
          f"sans contexte : {d_gap_nc:+.4f} CI[{lo_nc:+.4f},{hi_nc:+.4f}] -> "
          f"avec contexte : {d_gap_c:+.4f} CI[{lo_c:+.4f},{hi_c:+.4f}]")
    if abs(d_gap_c) < abs(d_gap_nc) * 0.5:
        print("    -> L'ecart se REFERME nettement (>50%) -- l'analogie LLM tient : la "
              "re-injection externe compense une bonne partie de la faiblesse de memoire "
              "INTERNE de M4R.")
    elif d_gap_c < d_gap_nc - 0.05:
        print("    -> L'ecart se REDUIT partiellement mais ne se referme pas -- le contexte "
              "aide M4R, mais une partie du desavantage n'est PAS une question de memoire "
              "(traitement non-lineaire du reservoir lui-meme).")
    else:
        print("    -> L'ecart NE se referme PAS -- meme avec le meme historique explicite que "
              "l'ESN, M4R reste derriere : le probleme n'est pas QUE la memoire.")

    print(f"(4) Plafond d'un readout purement lineaire sur le contexte seul : {ctx_only.mean():.4f} "
          f"(NARMA10 a un terme non-lineaire u[t-9]*u[t] -- un readout lineaire pur sur "
          f"des entrees brutes ne peut structurellement pas le capturer parfaitement).")

    with (FIG / "b5_context_reinjection_poc.csv").open("w", encoding="utf-8") as f:
        f.write("seed,m4_nocontext,m4_context,esn_nocontext,esn_context,context_only\n")
        for r in rows:
            f.write(",".join(f"{x:.6f}" if isinstance(x, float) else str(x) for x in r) + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

        ax = axes[0]
        labels = ["M4R\nsans ctx", "M4R\n+ctx", "ESN\nsans ctx", "ESN\n+ctx", "ctx\nseul"]
        means = [m4_nc.mean(), m4_c.mean(), esn_nc.mean(), esn_c.mean(), ctx_only.mean()]
        stds = [m4_nc.std(), m4_c.std(), esn_nc.std(), esn_c.std(), ctx_only.std()]
        colors = ["#2ca02c", "#98df8a", "#1f77b4", "#aec7e8", "#7f7f7f"]
        ax.bar(labels, means, yerr=stds, color=colors, edgecolor="k", capsize=4)
        ax.axhline(1.0, ls=":", c="red", label="NRMSE=1 (predire la moyenne)")
        ax.set_ylabel("NARMA10 NRMSE"); ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)
        ax.set_title("L'analogie LLM : la reinjection de contexte aide-t-elle ?")

        ax = axes[1]
        x = np.arange(len(SEEDS))
        ax.plot(x, m4_nc, "s--", c="#2ca02c", label="M4R sans ctx")
        ax.plot(x, m4_c, "s-", c="#2ca02c", label="M4R +ctx")
        ax.plot(x, esn_nc, "o--", c="#1f77b4", label="ESN sans ctx")
        ax.plot(x, esn_c, "o-", c="#1f77b4", label="ESN +ctx")
        ax.axhline(1.0, ls=":", c="red")
        ax.set_xlabel("seed"); ax.set_ylabel("NRMSE"); ax.set_title("Apparie par seed")
        ax.legend(fontsize=6.5); ax.grid(alpha=0.3)

        fig.suptitle("B5 -- reinjection de contexte (analogie LLM) sur NARMA10", fontsize=11)
        plt.tight_layout()
        plt.savefig(FIG / "b5_context_reinjection_poc.png", dpi=140)
        print(f"\n[png] {FIG / 'b5_context_reinjection_poc.png'}")
    except Exception as e:
        print(f"[png] skipped: {e}")

    print(f"\nWall time: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
