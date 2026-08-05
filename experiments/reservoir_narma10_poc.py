#!/usr/bin/env python3
"""
B1 -- Reservoir computing POC : le doute a-t-il une valeur computationnelle ?

Hypothese falsifiable
---------------------
En reservoir computing, un reservoir SYNCHRONISE est inutile : toutes les
lectures sont colineaires, le rang effectif s'effondre, la couche de sortie
n'a rien a exploiter. Le preprint montre que geler le doute (FROZEN_U,
epsilon_u=0) fait exploser la synchronie (~0.75) tandis que le doute actif
(FULL) la maintient proche de 0.

PREDICTION : si u maintient reellement une diversite EXPLOITABLE, alors
FULL doit battre FROZEN_U en erreur de prediction (NRMSE) sur NARMA10.
Si FROZEN_U egalise ou gagne, la diversite ne se traduit pas en capacite
de calcul -> resultat negatif, rapporte tel quel.

Protocole
---------
- Reservoir  : Mem4Network lattice 10x10 (N=100), heretic_ratio=0 pour isoler u.
- Entree     : masque aleatoire W_in (par noeud) x u_in(t), injecte comme I_stimulus
               (step() accepte deja un vecteur de taille N, dynamics.py:256).
- Lecture    : etats augmentes [1, v(t)] -> ridge regression (readout lineaire entraine).
- Tache      : NARMA10 (standard : memoire d'ordre 10 + non-linearite).
- Conditions : FULL (epsilon_u=0.02), FROZEN_U (epsilon_u=0), DECOUPLE (D=0, baseline).
- Comparaison appariee : meme seed reseau, meme W_in, meme u_in, meme bruit
  (seul epsilon_u ou D change entre conditions).
- Fairness   : balayage d'input_scale, meilleur NRMSE retenu par condition.
- Diagnostic : rang effectif (participation ratio) des etats -> explique le POURQUOI.

Sortie : figures/reservoir_narma10_poc.csv + .png + resume console.

Cree : 2026-07-06 (Claude Fable 5, L'Ingenieur -- piste B1 de docs/FUTURE_WORK.md).
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

CSV_PATH = ROOT / "figures" / "reservoir_narma10_poc.csv"
PNG_PATH = ROOT / "figures" / "reservoir_narma10_poc.png"

SIZE = 10                 # lattice 10x10 -> N = 100
T_WASH = 300              # washout (Echo State Property : oublier l'init)
T_TRAIN = 1500
T_TEST = 800
SEEDS = [0, 1, 2, 3, 4]
INPUT_SCALES = [0.1, 0.3, 0.6]
RIDGE_REG = 1e-6
CONDITIONS = ["FULL", "FROZEN_U", "DECOUPLE"]


# --------------------------------------------------------------------------
# NARMA10 : benchmark reservoir standard (memoire d'ordre 10 + non-linearite)
# --------------------------------------------------------------------------
def make_narma10(n_steps: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 0.5, n_steps)          # entree i.i.d. dans [0, 0.5]
    y = np.zeros(n_steps)
    for t in range(9, n_steps - 1):
        y[t + 1] = (0.3 * y[t]
                    + 0.05 * y[t] * np.sum(y[t - 9:t + 1])
                    + 1.5 * u[t - 9] * u[t]
                    + 0.1)
    return u, y


# --------------------------------------------------------------------------
# Faire tourner le reservoir Mem4ristor, collecter les etats v(t)
# --------------------------------------------------------------------------
def run_reservoir(u_in: np.ndarray, w_in: np.ndarray, condition: str,
                  seed: int) -> np.ndarray:
    net = Mem4Network(size=SIZE, heretic_ratio=0.0, seed=seed)
    if condition == "FROZEN_U":
        net.model.cfg["doubt"]["epsilon_u"] = 0.0          # u fige (ablation canonique)
    elif condition == "DECOUPLE":
        net.model.cfg["coupling"]["D"] = 0.0               # pas de couplage : baseline
        net.model.D_eff = 0.0

    N = net.N
    states = np.zeros((len(u_in), N))
    for t, ui in enumerate(u_in):
        net.step(I_stimulus=w_in * ui)                     # masque d'entree par noeud
        states[t] = net.model.v
    return states


# --------------------------------------------------------------------------
# Ridge readout + NRMSE
# --------------------------------------------------------------------------
def ridge_nrmse(states: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    # etats augmentes : [1, v_1..v_N]
    X = np.hstack([np.ones((states.shape[0], 1)), states])
    tr = slice(T_WASH, T_WASH + T_TRAIN)
    te = slice(T_WASH + T_TRAIN, T_WASH + T_TRAIN + T_TEST)
    Xtr, Ytr = X[tr], target[tr]
    Xte, Yte = X[te], target[te]

    F = Xtr.shape[1]
    W = np.linalg.solve(Xtr.T @ Xtr + RIDGE_REG * np.eye(F), Xtr.T @ Ytr)
    pred = Xte @ W
    nrmse = float(np.sqrt(np.mean((pred - Yte) ** 2) / (np.var(Yte) + 1e-12)))

    # rang effectif (participation ratio) des etats d'entrainement centres
    Xc = states[tr] - states[tr].mean(axis=0, keepdims=True)
    sv = np.linalg.svdvals(Xc)
    eff_rank = float((sv.sum() ** 2) / (np.sum(sv ** 2) + 1e-12))
    return nrmse, eff_rank


# --------------------------------------------------------------------------
def main() -> int:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_steps = T_WASH + T_TRAIN + T_TEST

    rows = []
    # meilleur NRMSE par (condition, seed) sur le balayage d'input_scale
    best = {c: {"nrmse": [], "eff_rank": [], "scale": []} for c in CONDITIONS}

    print(f"{'cond':<10}{'seed':>5}{'scale':>7}{'NRMSE':>10}{'eff_rank':>10}")
    print("-" * 45)
    for seed in SEEDS:
        u_in, y = make_narma10(n_steps, seed=seed)
        rng_mask = np.random.default_rng(1000 + seed)
        w_in_base = rng_mask.uniform(-1.0, 1.0, SIZE * SIZE)   # meme masque / conditions
        for cond in CONDITIONS:
            best_nrmse, best_rank, best_scale = np.inf, np.nan, np.nan
            for scale in INPUT_SCALES:
                states = run_reservoir(u_in, w_in_base * scale, cond, seed)
                nrmse, eff_rank = ridge_nrmse(states, y)
                rows.append((cond, seed, scale, nrmse, eff_rank))
                if nrmse < best_nrmse:
                    best_nrmse, best_rank, best_scale = nrmse, eff_rank, scale
            best[cond]["nrmse"].append(best_nrmse)
            best[cond]["eff_rank"].append(best_rank)
            best[cond]["scale"].append(best_scale)
            print(f"{cond:<10}{seed:>5}{best_scale:>7.2f}{best_nrmse:>10.4f}{best_rank:>10.2f}")

    # ---- resume ----
    print("\n=== RESUME (meilleur input_scale par condition, moyenne sur seeds) ===")
    print(f"{'condition':<10}{'NRMSE mean':>12}{'NRMSE std':>11}{'eff_rank':>10}")
    summary = {}
    for cond in CONDITIONS:
        arr = np.array(best[cond]["nrmse"])
        rk = np.array(best[cond]["eff_rank"])
        summary[cond] = (arr.mean(), arr.std(), rk.mean())
        print(f"{cond:<10}{arr.mean():>12.4f}{arr.std():>11.4f}{rk.mean():>10.2f}")

    full_m = summary["FULL"][0]
    froz_m = summary["FROZEN_U"][0]
    dec_m = summary["DECOUPLE"][0]
    print("\n=== VERDICT (honnete : on compare AUSSI au decouple D=0) ===")
    # 1) Parmi les reservoirs couples, le doute aide-t-il ?
    if full_m < froz_m:
        gain = 100.0 * (froz_m - full_m) / froz_m
        print(f"[couples] FULL < FROZEN_U ({full_m:.4f} vs {froz_m:.4f}, {gain:.1f}% mieux) : "
              f"le doute previent le dommage de synchronisation du couplage.")
    else:
        print(f"[couples] FULL >= FROZEN_U ({full_m:.4f} vs {froz_m:.4f}) : "
              f"pas de gain du doute -- resultat negatif.")
    # 2) Mais le couplage lui-meme apporte-t-il quelque chose vs pas de couplage ?
    if full_m < dec_m:
        print(f"[net] FULL < DECOUPLE ({full_m:.4f} vs {dec_m:.4f}) : "
              f"le couplage+doute bat les noeuds isoles -> valeur computationnelle NETTE.")
    else:
        print(f"[net] DECOUPLE <= FULL ({dec_m:.4f} vs {full_m:.4f}) : le decouplage gagne. "
              f"Le doute repare un probleme cause par le couplage, sans le surpasser. "
              f"Il faut une tache/un regime ou le couplage est NECESSAIRE.")
    if min(full_m, froz_m, dec_m) > 1.0:
        print("[caveat] Tous les NRMSE > 1.0 : mauvais reservoir dans l'absolu (memoire "
              "trop courte pour NARMA10). Contraste reel mais sur fond de faible performance.")

    # ---- CSV ----
    with CSV_PATH.open("w", encoding="utf-8") as f:
        f.write("condition,seed,input_scale,nrmse,eff_rank\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]:.6f},{r[4]:.4f}\n")
    print(f"\n[csv] {CSV_PATH}")

    # ---- figure ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
        conds = CONDITIONS
        means = [summary[c][0] for c in conds]
        stds = [summary[c][1] for c in conds]
        ranks = [summary[c][2] for c in conds]
        colors = ["#2ca02c", "#d62728", "#7f7f7f"]
        ax1.bar(conds, means, yerr=stds, color=colors, edgecolor="k", capsize=5)
        ax1.set_ylabel("NARMA10 NRMSE (lower = better)")
        ax1.set_title("Reservoir performance")
        ax1.grid(axis="y", alpha=0.3)
        ax2.bar(conds, ranks, color=colors, edgecolor="k")
        ax2.set_ylabel("Effective rank (participation ratio)")
        ax2.set_title("Reservoir richness (why)")
        ax2.grid(axis="y", alpha=0.3)
        fig.suptitle(f"Doubt as a reservoir resource -- Mem4ristor lattice N={SIZE*SIZE}, "
                     f"{len(SEEDS)} seeds", fontsize=11)
        plt.tight_layout()
        plt.savefig(PNG_PATH, dpi=140)
        print(f"[png] {PNG_PATH}")
    except Exception as e:
        print(f"[png] skipped: {e}")

    print(f"\nWall time: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
