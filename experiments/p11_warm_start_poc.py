#!/usr/bin/env python3
"""
P11 -- WARM START CONTINU : M4R donne-t-il une direction qui fait economiser
un solveur tiers, AVANT tout calcul ?
=============================================================================
Cree : 2026-07-13 (Claude Sonnet 5, L'Ingenieur). Idee de Julien apres une
longue discussion honnete sur ce que M4R fait vraiment : « imaginons que
des le depart M4R donne la direction... warm start continu, j'aurais du
te donner ca en premier ». Different de P11 (12/07, meme jour precedent) :
la, M4R disait QUAND s'arreter un solveur qui demarre toujours au meme
point connu (x0=2, cible x*=0 fixe). ICI, la cible est AMBIGUE EN SIGNE
(b in {-1,+1} inconnu a l'avance) -- le vrai test d'un "warm start" : le
solveur beneficie-t-il d'un DEMARRAGE INFORME plutot que neutre ?

CONSTRUCTION (reutilise le grad/h EXACT de p11_universal_stopping_poc.py,
seule generalisation : cible non-nulle et miroir par b) :
  - cible = b * X_TARGET (X_TARGET=2.0, comme la distance x0=2 -> x*=0 de P11).
  - plateau EXACTEMENT a mi-chemin, miroir par b : x_p = b * X_P (X_P~1.0-1.3,
    memes plages que P11, h_min/w_flat derives IDENTIQUEMENT).
  - grad(x) = (x - cible) * h(x) -- generalisation directe de grad(x)=x*h(x)
    de P11 (qui est le cas cible=0).
  - BLIND : x0 = 0 (neutre, aucune info) -- doit TOUJOURS traverser le
    plateau quel que soit b (symetrique par construction).
  - WARM (M4R) : M4R fait D'ABORD une lecture bon marche (groupe de 30
    noeuds, pulse signe par b, T_READ pas, readout interference deja
    valide aujourd'hui) -> b_guess. x0 = b_guess * X_WARM (X_WARM=1.5,
    au-dela du plateau+rampe dans TOUS les cas -- verifie : x_p_max=1.3 +
    w_flat_max=0.03 + W_RAMP=0.06 = 1.39 < 1.5).
    - Si b_guess CORRECT : demarre DEJA au-dela du piege -> traversee
      courte et directe jusqu'a la cible, plateau evite ENTIEREMENT.
    - Si b_guess FAUX : demarre du COTE OPPOSE a la cible -> doit
      traverser TOUT (retour a zero, PUIS le vrai plateau du bon cote)
      -- penalite honnete, plus couteux que BLIND. Pas un dejeuner
      gratuit si M4R se trompe.

Cout compte SEPAREMENT et honnetement (pas de taux de change invente
entre "pas reseau M4R" et "iterations solveur", des unites differentes) :
  - iterations solveur (BLIND vs WARM, ponderees par l'accuracy REELLE de
    M4R mesuree, pas supposee) ;
  - cout de la lecture M4R (T_READ x N noeuds-pas), rapporte a part.

Replication INTEGREE des le premier lancement (lecon du Condorcet du
13/07, meme jour) : deux plages de graines disjointes (0-29 et 100-129).

Statut : exploratoire, hors preprint, aucune modification de dynamics.py.
Guardian doit rester 14/14. Sorties : figures/p11_warm_start_poc.csv + .png
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

# --- Solveur (generalise depuis p11_universal_stopping_poc.py) ---
ETA = 0.05
MAX_ITER = 3000
SUCCESS_TOL = 0.05
X_TARGET = 2.0
X_WARM = 1.5           # au-dela du plateau+rampe dans tous les cas testes
W_RAMP = 0.06

# --- Lecture M4R (reprend le protocole etabli aujourd'hui) ---
SIDE, N = 10, 100
GROUP_SIZE = 30
B_E = 0.8
T_READ = 300

SEED_RANGES = {"original (0-29)": list(range(30)), "replication (100-129)": list(range(100, 130))}


def make_problem(seed, b):
    """Plateau miroir par b -- meme derivation h_min/w_flat que P11."""
    rng = np.random.RandomState(70000 + seed)
    w_flat = rng.uniform(0.02, 0.03)
    t_c = rng.uniform(700.0, 1400.0)
    x_p_mag = rng.uniform(0.9, 1.3)
    h_min = 2.0 * w_flat / (ETA * x_p_mag * t_c)
    return {"x_p": b * x_p_mag, "w_flat": w_flat, "h_min": h_min, "target": b * X_TARGET}


def grad(x, pb):
    d = abs(x - pb["x_p"]) - pb["w_flat"]
    if d <= 0:
        h = pb["h_min"]
    elif d >= W_RAMP:
        h = 1.0
    else:
        s = d / W_RAMP
        h = pb["h_min"] + (1.0 - pb["h_min"]) * s * s * (3.0 - 2.0 * s)
    return (x - pb["target"]) * h


def solve(pb, x0, max_iter=MAX_ITER):
    x = x0
    for t in range(max_iter):
        if abs(x - pb["target"]) < SUCCESS_TOL:
            return t
        x = x - ETA * grad(x, pb)
    return max_iter


def build_group(seed):
    rng = np.random.RandomState(90000 + seed)
    mask_nodes = rng.choice(N, size=GROUP_SIZE, replace=False)
    mask = np.zeros(N, dtype=bool)
    mask[mask_nodes] = True
    idle = ~mask
    return mask, idle


def m4r_read(seed, b):
    mask, idle = build_group(seed)
    net = Mem4Network(size=SIDE, heretic_ratio=0.0, seed=seed * 10 + 1,
                       adjacency_matrix=make_lattice_adj(SIDE, periodic=True))
    m = net.model
    m.cfg['complex_doubt']['enabled'] = True
    stim = np.zeros(N)
    stim[mask] = b * B_E
    for _ in range(T_READ):
        net.step(I_stimulus=stim)
    diff = m.u_c[idle].mean() - m.u_c[mask].mean()
    guess = 1 if float(np.real(diff)) >= 0 else -1
    return guess


def run_condition(seeds):
    blind_iters, warm_iters, guesses_correct = [], [], []
    for seed in seeds:
        b = 1 if (seed % 2 == 0) else -1  # alterne les deux signes, pas de biais
        pb = make_problem(seed, b)

        it_blind = solve(pb, x0=0.0)
        blind_iters.append(it_blind)

        b_guess = m4r_read(seed, b)
        guesses_correct.append(int(b_guess == b))
        x0_warm = b_guess * X_WARM
        it_warm = solve(pb, x0=x0_warm)
        warm_iters.append(it_warm)

    return (np.array(blind_iters), np.array(warm_iters), np.array(guesses_correct))


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("=== WARM START CONTINU : M4R donne-t-il une direction qui economise un solveur tiers ? ===\n")

    rows = []
    for label, seeds in SEED_RANGES.items():
        tlab = time.time()
        blind, warm, correct = run_condition(seeds)
        acc = float(correct.mean())
        mean_blind = float(blind.mean())
        mean_warm = float(warm.mean())
        savings = mean_blind - mean_warm
        m4r_cost = T_READ * N
        print(f"-- {label} (n={len(seeds)}) --")
        print(f"  Accuracy de la lecture M4R (b_guess == b vrai) : {acc:.3f}")
        print(f"  Iterations solveur BLIND (x0=0)          : {mean_blind:.0f} +/- {blind.std():.0f}")
        print(f"  Iterations solveur WARM (M4R-informe)    : {mean_warm:.0f} +/- {warm.std():.0f}")
        print(f"  Economie moyenne (iterations solveur)    : {savings:+.0f}")
        print(f"  Cout de la lecture M4R (a part, unite differente) : {m4r_cost} noeuds-pas")
        print(f"  [{time.time()-tlab:.0f}s]\n")
        rows.append((label, len(seeds), acc, mean_blind, mean_warm, savings, m4r_cost))

    print("=== VERDICT ===")
    for label, n, acc, mb, mw, sav, cost in rows:
        verdict = "GAIN NET" if sav > 0.05 * mb else ("PERTE NETTE" if sav < -0.05 * mb else "quasi neutre")
        print(f"  {label} : {verdict} (economie {sav:+.0f} sur {mb:.0f} iterations blind, "
              f"soit {100*sav/mb:+.0f}%), accuracy lecture={acc:.2f}")

    consistent = all(r[5] > 0.05 * r[3] for r in rows)
    if consistent:
        print("\n  -> Le warm start economise des iterations sur les DEUX plages de graines -- "
              "coherent, pas un artefact d'echantillonnage isole.")
    else:
        print("\n  -> Resultat NON coherent entre les deux plages de graines -- a traiter comme "
              "la lecon du Condorcet du meme jour : ne pas conclure a un gain sans confirmation.")

    with (FIG / "p11_warm_start_poc.csv").open("w", encoding="utf-8") as f:
        f.write("seed_range,n,accuracy,mean_blind,mean_warm,savings,m4r_cost_node_steps\n")
        for r in rows:
            f.write(",".join(f"{x:.6f}" if isinstance(x, float) else str(x) for x in r) + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.8))
        labels = [r[0] for r in rows]
        x = np.arange(len(rows))
        w = 0.35
        ax.bar(x - w / 2, [r[3] for r in rows], w, label="BLIND (x0=0)", color="#1f77b4")
        ax.bar(x + w / 2, [r[4] for r in rows], w, label="WARM (M4R-informe)", color="#2ca02c")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("iterations solveur (moyenne)")
        ax.set_title("Warm start continu : M4R fait-il gagner des iterations au solveur ?")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIG / "p11_warm_start_poc.png", dpi=140)
        print(f"\n[png] {FIG / 'p11_warm_start_poc.png'}")
    except Exception as e:
        print(f"[png] skipped: {e}")

    print(f"\nWall time: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
