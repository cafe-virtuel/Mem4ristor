#!/usr/bin/env python3
"""
AUDIT FABLE 2026-11-11 -- D5 : l'argument (ii) de preprint.tex:353 -- l'invariance en taille --
rejoue sur graphes FIGES.

Le preprint affirme : pour un anneau k=8, lambda2 tombe de 0.118 a 0.0074 quand N passe de 100
a 400, "yet the dead fraction stays at 1.0 throughout, while a k=4 ring stays functional".
D1 a montre que ces runs tournaient avec le recablage par defaut du coeur.

Questions, ecrites AVANT la mesure :
  Q1. (fidelite) Recablage actif, memes 6 graines : retrouve-t-on le CSV publie
      (consolidation_invariance_N.csv) ?
  Q2. Recablage coupe, 30 graines : l'anneau k=8 est-il mort a N=100, 200, 400 ?
      Et le regulier aleatoire k=8 ? Si l'anneau vit et que l'aleatoire meurt, a degre egal,
      c'est la structure qui decide et l'argument (ii) tombe. Si les deux meurent, il tient.

Protocole : celui de consolidation_invariance_N.py (3000 pas, cold start, I=0, hr=0.15,
best-of-two-norms H_cog < 0.10).
Usage : uv run --frozen python experiments/fable_audit_20261111/d5_invariance_N_static.py
"""
from __future__ import annotations
import csv, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import networkx as nx
from scipy.linalg import eigh

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))
from d2_lambda2_static import wilson  # noqa: E402

STEPS = 3000
PUB_SEEDS = [42, 123, 777, 17, 256, 1337]
NEW_SEEDS = list(range(2000, 2024))
K_VALUES = [4, 8]
N_VALUES = [100, 200, 400]
NORMS = ["uniform", "degree_linear"]


def one(job):
    k, model, Nn, seed, rewire = job
    from mem4ristor.core import Mem4Network
    from mem4ristor.metrics import calculate_cognitive_entropy
    G = nx.watts_strogatz_graph(Nn, k, 0.0, seed=seed) if model == "ring" else nx.random_regular_graph(k, Nn, seed=seed)
    adj = nx.to_numpy_array(G, dtype=float)
    l2 = float(eigh(np.diag(adj.sum(axis=1)) - adj, eigvals_only=True)[1])
    best = -1.0
    for norm in NORMS:
        net = Mem4Network(size=int(np.sqrt(Nn)), heretic_ratio=0.15, seed=seed,
                          adjacency_matrix=adj.copy(), coupling_norm=norm, cold_start=True,
                          rewire_threshold=(0.8 if rewire else 999.0))
        net.model.cfg["noise"]["sigma_v"] = 0.05
        for _ in range(STEPS):
            net.step(I_stimulus=0.0)
        best = max(best, float(calculate_cognitive_entropy(net.v)))
    return dict(k=k, model=model, N=Nn, seed=seed, rewire=int(rewire), lambda2=l2,
                best_h_cog=best, dead=int(best < 0.10))


def main():
    t0 = time.time()
    jobs = [(k, m, Nn, s, True) for k in K_VALUES for m in ("ring", "rr") for Nn in N_VALUES for s in PUB_SEEDS]
    jobs += [(k, m, Nn, s, False) for k in K_VALUES for m in ("ring", "rr") for Nn in N_VALUES
             for s in PUB_SEEDS + NEW_SEEDS]
    with Pool(12) as pool:
        rows = pool.map(one, jobs, chunksize=2)
    with open(HERE / "d5_invariance_N_static.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    pub = {}
    for r in csv.DictReader(open(ROOT / "experiments" / "lambda2_foundation_20260701" / "consolidation_invariance_N.csv")):
        pub.setdefault((int(r["k"]), r["model"], int(r["N"])), []).append(int(r["dead"]))
    print("=" * 100)
    print("D5 -- invariance en taille : publie (recable) / rejoue recable (6 graines) / FIGE (30 graines)")
    print("=" * 100)
    print(f"{'k':>3} {'modele':>6} {'N':>4} {'lambda2':>8} | {'publie':>6} {'rejoue':>6} | {'FIGE dead':>9} {'IC95 Wilson':>13}")
    ok = True
    for k in K_VALUES:
        for m in ("ring", "rr"):
            for Nn in N_VALUES:
                on = [r for r in rows if r["k"] == k and r["model"] == m and r["N"] == Nn and r["rewire"] == 1]
                off = [r for r in rows if r["k"] == k and r["model"] == m and r["N"] == Nn and r["rewire"] == 0]
                p = float(np.mean(pub[(k, m, Nn)])); o = float(np.mean([r["dead"] for r in on]))
                ok &= abs(p - o) < 1e-12
                x = sum(r["dead"] for r in off); lo, hi = wilson(x, len(off))
                print(f"{k:>3} {m:>6} {Nn:>4} {np.mean([r['lambda2'] for r in off]):>8.4f} | {p:>6.2f} {o:>6.2f} | "
                      f"{x / len(off):>9.2f} [{lo:.2f},{hi:.2f}]")
        print("-" * 100)
    print(f"fidelite du rejeu recable contre le CSV publie : {'OK' if ok else 'ECART'}")
    print(f"{len(rows)} cellules, {time.time() - t0:.0f} s. CSV : d5_invariance_N_static.csv")


if __name__ == "__main__":
    main()
