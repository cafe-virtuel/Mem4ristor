#!/usr/bin/env python3
"""
MANDAT NON-LIVRABLE : fonder theoriquement lambda2_crit ~ 2.31
Etape 4 : CONSOLIDATION — invariance en taille N + plus de seeds.

Le crucial experiment (etape 3) a montre qu'a degre k fixe, lambda2 n'a aucun
effet sur le regime. Cette etape enfonce le clou par le test le plus impitoyable
contre lambda2 : l'INVARIANCE EN TAILLE.

Mecanisme du test :
  Pour un ANNEAU de degre k, lambda2 ~ 1/N^2 -> s'effondre quand N grandit
  (0.12 a N=100 -> ~0.03 a N=200 -> ~0.008 a N=400), alors que <k> = k reste
  CONSTANT. Pour un RANDOM-REGULAR de degre k, lambda2 ~ k-2sqrt(k-1) reste
  quasi CONSTANT en N.

  - Si lambda2 etait causal : l'anneau, dont lambda2 -> 0, devrait devenir de
    plus en plus FONCTIONNEL avec N (il s'eloigne du seuil 2.31 par le bas).
  - Si <k> est causal : anneau et random-regular restent dans le MEME regime a
    tout N (leur degre est identique).

On teste k=4 (sous le seuil de densite, attendu fonctionnel) et k=8 (au-dessus,
attendu dead), aux tailles N = 100, 200, 400, avec 6 seeds.

Protocole regime : endogene, best-of-two-norms H_cog (identique etapes 2-3).
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

import numpy as np
import networkx as nx
from scipy.linalg import eigh

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mem4ristor.core import Mem4Network
from mem4ristor.metrics import calculate_cognitive_entropy

STEPS = 3000
TAIL_FRAC = 0.25
SEEDS = [42, 123, 777, 17, 256, 1337]
I_STIM = 0.0
HERETIC_RATIO = 0.15
SIGMA_V = 0.05
DEAD_THRESHOLD = 0.10
NORMS = ["uniform", "degree_linear"]

K_VALUES = [4, 8]
N_VALUES = [100, 200, 400]


def fiedler_value(adj: np.ndarray) -> float:
    deg = adj.sum(axis=1)
    L = np.diag(deg) - adj
    vals = eigh(L, eigvals_only=True)
    return float(vals[1]) if len(vals) > 1 else 0.0


def run_regime(adj: np.ndarray, seed: int) -> float:
    best = -1.0
    for norm in NORMS:
        net = Mem4Network(size=int(np.sqrt(adj.shape[0])),
                          heretic_ratio=HERETIC_RATIO, seed=seed,
                          adjacency_matrix=adj.copy(), coupling_norm=norm,
                          cold_start=True)
        net.model.cfg["noise"]["sigma_v"] = SIGMA_V
        for _ in range(STEPS):
            net.step(I_stimulus=I_STIM)
        h = float(calculate_cognitive_entropy(net.v))
        if h > best:
            best = h
    return best


def main() -> int:
    t0 = time.time()
    rows = []
    print("=" * 92)
    print("CONSOLIDATION : invariance en N (anneau vs random-regular, degre fixe)")
    print(f"steps={STEPS} endogene seeds={len(SEEDS)} seuil dead H_cog<{DEAD_THRESHOLD}")
    print("=" * 92)
    print("Prediction si <k> cause : anneau et rr MEME dead_frac a tout N,")
    print("  meme si lambda2(anneau) -> 0 quand N grandit.")
    print("-" * 92)
    print(f"{'k':>3} {'model':>8} {'N':>5} {'lambda2':>9} {'dead_frac':>10} {'note'}")
    print("-" * 92)

    for k in K_VALUES:
        for model in ["ring", "rr"]:
            prev_l2 = None
            for Nn in N_VALUES:
                l2s, deads = [], []
                for seed in SEEDS:
                    if model == "ring":
                        G = nx.watts_strogatz_graph(Nn, k, 0.0, seed=seed)
                    else:
                        G = nx.random_regular_graph(k, Nn, seed=seed)
                    adj = nx.to_numpy_array(G, dtype=float)
                    l2s.append(fiedler_value(adj))
                    h = run_regime(adj, seed)
                    deads.append(1 if h < DEAD_THRESHOLD else 0)
                    rows.append({"k": k, "model": model, "N": Nn, "seed": seed,
                                 "lambda2": l2s[-1], "dead": deads[-1]})
                l2m = np.mean(l2s)
                note = ""
                if model == "ring" and prev_l2 is not None:
                    note = f"lambda2 x{l2m/prev_l2:.2f} vs N precedent"
                prev_l2 = l2m
                print(f"{k:>3} {model:>8} {Nn:>5} {l2m:>9.4f} {np.mean(deads):>10.3f} {note}")
            print("-" * 92)

    csvp = HERE / "consolidation_invariance_N.csv"
    with csvp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["k", "model", "N", "seed",
                                          "lambda2", "dead"])
        w.writeheader(); w.writerows(rows)

    print("\nLECTURE CLE (k=8) :")
    print("  Si l'anneau reste dead_frac~1 alors que son lambda2 s'effondre vers 0,")
    print("  lambda2 ne PEUT PAS etre la cause. Le regime suit le degre, pas le spectre.")
    print(f"\n[csv] {csvp}")
    print(f"Wall time: {time.time()-t0:.1f}s | {len(rows)} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
