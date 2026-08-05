#!/usr/bin/env python3
"""
MANDAT NON-LIVRABLE : fonder theoriquement lambda2_crit ~ 2.31
Etape 3 : CRUCIAL EXPERIMENT — decoupler <k> et lambda2.

Question : la mort cognitive est-elle causee par la connectivite algebrique
lambda2 (claim du preprint) ou par la densite <k> (degre moyen) ?

Ces deux grandeurs covarient dans BA et ER, donc les donnees existantes ne
peuvent PAS les distinguer. Watts-Strogatz les decouple : a degre k FIXE, le
taux de recablage p balaie lambda2 de ~0 (anneau, p=0) a grand (p=1), tout en
gardant <k> = k EXACTEMENT constant (WS preserve le nombre d'aretes).

  - Si le regime depend de p (donc de lambda2) a k fixe -> lambda2 CAUSE.
  - Si le regime ne depend QUE de k (constant en p)      -> <k> (densite) CAUSE.

On ajoute un point random-regular (rr) par k = lambda2 maximal atteignable a
degre fixe (borne de Ramanujan ~ k - 2*sqrt(k-1)).

Cas le plus discriminant : k=10 ou 12 (densite au-dessus du seuil ~8).
  anneau p=0 : lambda2 ~ 0.4  << 2.31  -> lambda2 predit FONCTIONNEL
  random-reg : lambda2 ~ 4-5  >> 2.31  -> lambda2 predit DEAD
  Si les deux donnent le MEME regime -> c'est <k>, pas lambda2.

Protocole regime : identique a l'etape 2 (endogene, best-of-two-norms H_cog).
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

N = 100
STEPS = 3000
TAIL_FRAC = 0.25
SEEDS = [42, 123, 777, 17, 256]
I_STIM = 0.0
HERETIC_RATIO = 0.15
SIGMA_V = 0.05
DEAD_THRESHOLD = 0.10
NORMS = ["uniform", "degree_linear"]

K_VALUES = [6, 8, 10, 12]
P_VALUES = [0.0, 0.2, 0.5, 1.0]     # WS rewiring : anneau -> quasi-aleatoire


def to_adj(G) -> np.ndarray:
    return nx.to_numpy_array(G, dtype=float)


def fiedler_value(adj: np.ndarray) -> float:
    deg = adj.sum(axis=1)
    L = np.diag(deg) - adj
    vals = eigh(L, eigvals_only=True)
    return float(vals[1]) if len(vals) > 1 else 0.0


def run_regime(adj: np.ndarray, seed: int) -> float:
    """best_H_cog sur les deux normalisations (endogene). > seuil = fonctionnel."""
    best = -1.0
    for norm in NORMS:
        net = Mem4Network(size=10, heretic_ratio=HERETIC_RATIO, seed=seed,
                          adjacency_matrix=adj.copy(), coupling_norm=norm,
                          cold_start=True)
        net.model.cfg["noise"]["sigma_v"] = SIGMA_V
        tail0 = int(STEPS * (1 - TAIL_FRAC))
        for t in range(STEPS):
            net.step(I_stimulus=I_STIM)
        h = float(calculate_cognitive_entropy(net.v))
        if h > best:
            best = h
    return best


def main() -> int:
    t0 = time.time()
    rows = []
    print("=" * 88)
    print("CRUCIAL EXPERIMENT : <k> fixe, lambda2 variable (Watts-Strogatz + random-regular)")
    print(f"N={N} steps={STEPS} endogene seeds={len(SEEDS)}  seuil dead H_cog<{DEAD_THRESHOLD}")
    print("=" * 88)
    print("Si dead_frac constant par ligne (k) -> <k> CAUSE. Si varie avec lambda2 -> lambda2 CAUSE.")
    print("-" * 88)
    print(f"{'k':>3} {'model':>8} {'p':>5} {'lambda2':>9} {'H_cog_med':>10} {'dead_frac':>10}")
    print("-" * 88)

    for k in K_VALUES:
        specs = [("WS", p) for p in P_VALUES] + [("rr", None)]
        for model, p in specs:
            l2s, hcs, deads = [], [], []
            for seed in SEEDS:
                if model == "WS":
                    G = nx.watts_strogatz_graph(N, k, p, seed=seed)
                else:
                    G = nx.random_regular_graph(k, N, seed=seed)
                adj = to_adj(G)
                l2 = fiedler_value(adj)
                h = run_regime(adj, seed)
                l2s.append(l2); hcs.append(h)
                deads.append(1 if h < DEAD_THRESHOLD else 0)
                rows.append({"k": k, "model": model,
                             "p": (p if p is not None else -1),
                             "seed": seed, "lambda2": l2, "best_h_cog": h,
                             "dead": deads[-1]})
            pstr = f"{p:.1f}" if p is not None else "  -"
            print(f"{k:>3} {model:>8} {pstr:>5} {np.mean(l2s):>9.3f} "
                  f"{np.median(hcs):>10.3f} {np.mean(deads):>10.3f}")
        print("-" * 88)

    csvp = HERE / "crucial_kfixe_lambda2_variable.csv"
    with csvp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["k", "model", "p", "seed",
                                          "lambda2", "best_h_cog", "dead"])
        w.writeheader(); w.writerows(rows)

    print("\nLECTURE :")
    print("  A chaque k, lambda2 va de ~0.4 (WS p=0, anneau) a ~k-2sqrt(k-1) (rr).")
    print("  Le seuil canonique 2.31 est traverse a l'interieur de plusieurs lignes.")
    print("  -> Regarder si dead_frac SAUTE quand lambda2 franchit 2.31 (=> lambda2)")
    print("     ou reste PLAT sur toute la ligne (=> <k>).")
    print(f"\n[csv] {csvp}")
    print(f"Wall time: {time.time()-t0:.1f}s | {len(rows)} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
