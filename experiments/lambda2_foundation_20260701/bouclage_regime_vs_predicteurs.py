#!/usr/bin/env python3
"""
MANDAT NON-LIVRABLE : fonder theoriquement lambda2_crit ~ 2.31
Etape 6 : BOUCLAGE — le regime s'aligne-t-il sur <1/deg> (champ moyen) plutot
que sur lambda2, a travers TOUTES les familles ?

On mesure dead_frac (endogene, best-of-two-norms H_cog) et, pour chaque
topologie, tous les predicteurs candidats : lambda2, <k>, k_median, <1/deg>
(-> degre harmonique). Un seul set, un seul N, un seul protocole -> on peut
comparer honnetement quel predicteur unifie les familles sur UNE frontiere.

Prediction (mecanisme champ-moyen etabli a l'etape 5) :
  dead_frac est une fonction propre et monotone de <1/deg> (ou k_harm),
  identique pour BA / ER / regulier / anneau ; lambda2 les eparpille.
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

N = 200
STEPS = 3000
TAIL_FRAC = 0.25
SEEDS = [42, 123, 777, 17, 256]
I_STIM = 0.0
HERETIC_RATIO = 0.15
SIGMA_V = 0.05
DEAD_THRESHOLD = 0.10
NORMS = ["uniform", "degree_linear"]


def fiedler(adj):
    d = adj.sum(1)
    return float(np.sort(eigh(np.diag(d) - adj, eigvals_only=True))[1])


def run_regime(adj, seed):
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


def graphs():
    specs = []
    for m in [2, 3, 4, 5, 6]:
        specs.append((f"BA m={m}", "BA", lambda s, m=m: nx.barabasi_albert_graph(N, m, seed=s)))
    for p in [0.03, 0.05, 0.08]:
        specs.append((f"ER p={p}", "ER", lambda s, p=p: nx.erdos_renyi_graph(N, p, seed=s)))
    for k in [4, 6, 8]:
        specs.append((f"rr k={k}", "rr", lambda s, k=k: nx.random_regular_graph(k, N, seed=s)))
    for k in [4, 6, 8]:
        specs.append((f"ring k={k}", "ring", lambda s, k=k: nx.watts_strogatz_graph(N, k, 0.0, seed=s)))
    return specs


def main():
    t0 = time.time()
    rows = []
    print("=" * 100)
    print(f"BOUCLAGE : regime vs predicteurs (N={N}, endogene, {len(SEEDS)} seeds)")
    print("=" * 100)
    print(f"{'topo':<11}{'fam':>5}{'lambda2':>9}{'k_mean':>8}{'k_med':>7}"
          f"{'<1/deg>':>9}{'k_harm':>8}{'dead_frac':>10}")
    print("-" * 100)
    for name, fam, gen in graphs():
        l2s, kms, kmeds, invs, deads = [], [], [], [], []
        for s in SEEDS:
            G = gen(s)
            adj = nx.to_numpy_array(G, dtype=float)
            deg = adj.sum(1)
            l2s.append(fiedler(adj)); kms.append(deg.mean())
            kmeds.append(np.median(deg))
            invs.append(np.mean(1.0 / np.where(deg < 1, 1, deg)))
            h = run_regime(adj, s)
            deads.append(1 if h < DEAD_THRESHOLD else 0)
            rows.append({"name": name, "fam": fam, "seed": s,
                         "lambda2": l2s[-1], "k_mean": kms[-1],
                         "k_med": kmeds[-1], "inv_deg": invs[-1],
                         "dead": deads[-1]})
        l2, km, kmed, inv, df = map(np.mean, (l2s, kms, kmeds, invs, deads))
        print(f"{name:<11}{fam:>5}{l2:>9.3f}{km:>8.2f}{kmed:>7.0f}"
              f"{inv:>9.4f}{1/inv:>8.2f}{df:>10.3f}")

    csvp = HERE / "bouclage_regime_vs_predicteurs.csv"
    with csvp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["name", "fam", "seed", "lambda2",
                                          "k_mean", "k_med", "inv_deg", "dead"])
        w.writeheader(); w.writerows(rows)

    # Qualite de separation : pour chaque predicteur, le meilleur seuil
    # (minimise le nb d'instances mal classees) et le nb d'erreurs.
    import itertools
    inst = rows
    def sep_quality(key, invert=False):
        xs = sorted(set(r[key] for r in inst))
        best_err = len(inst) + 1
        best_thr = None
        cands = xs + [(a + b) / 2 for a, b in zip(xs, xs[1:])]
        for thr in cands:
            # dead si x > thr (ou x < thr si invert)
            err = 0
            for r in inst:
                pred = (r[key] > thr) if not invert else (r[key] < thr)
                if int(pred) != r["dead"]:
                    err += 1
            if err < best_err:
                best_err, best_thr = err, thr
        return best_err, best_thr

    print("-" * 100)
    print("Qualite de separation dead/live (nb d'instances mal classees, plus bas = mieux) :")
    for key, inv in [("lambda2", False), ("k_mean", False),
                     ("k_med", False), ("inv_deg", True)]:
        err, thr = sep_quality(key, invert=inv)
        sense = "<" if inv else ">"
        print(f"  {key:<10}: {err:2d}/{len(inst)} erreurs  (seuil dead {sense} {thr:.4f})")
    print("  (inv_deg : dead quand <1/deg> < seuil = degre eleve = champ moyen)")
    print(f"\n[csv] {csvp}")
    print(f"Wall time: {time.time()-t0:.1f}s | {len(rows)} instances")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
