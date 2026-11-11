#!/usr/bin/env python3
"""
AUDIT FABLE 2026-11-11 -- D2 : lambda2 a degre fixe, sur graphes FIGES, 30 graines neuves.

Suite de D1. D1 a etabli (fidelite 0.00e+00 contre le CSV publie) que l'experience
cruciale tournait avec le recablage par defaut : 76-93 % des aretes changees, lambda2
de fin ~ celui d'un graphe regulier aleatoire quel que soit le graphe de depart.

Questions, ecrites AVANT la mesure :
  Q1. Recablage coupe, la fraction 'dead' depend-elle de lambda2 a degre fixe ?
      (preprint.tex:353 affirme : "At each fixed degree the dead/functional outcome
      is flat in lambda2".)  Critere : Spearman(lambda2, dead) par k, 30 graines x 7 graphes.
  Q2. Que mesure 'dead' sur les tensions ? Point fixe FHN a I=0 : v* ~ -1.285 ; la
      frontiere de classe la plus proche est a -1.2, soit 0.085 de distance. Si les runs
      'functional' ont un ecart-type de v de cet ordre et aucun spike, alors H_cog mesure
      le chevauchement d'une frontiere par du bruit, pas une diversite d'etats.
      Mesures : sd(v) final, etendue, fraction de noeuds ayant spike (v>0) sur la queue.

Protocole : celui de crucial_kfixe_lambda2_variable.py, sauf rewire_threshold=999 et graines.
Usage : uv run --frozen python experiments/fable_audit_20261111/d2_lambda2_static.py
"""
from __future__ import annotations
import csv, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import networkx as nx
from scipy.linalg import eigh
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src"))

N = 100
STEPS = 3000
TAIL = 750
SEEDS = list(range(1000, 1030))
I_STIM = 0.0
HERETIC_RATIO = 0.15
SIGMA_V = 0.05
DEAD_THRESHOLD = 0.10
NORMS = ["uniform", "degree_linear"]
K_VALUES = [6, 8, 10]
SPECS = [("WS", 0.0), ("WS", 0.05), ("WS", 0.1), ("WS", 0.2), ("WS", 0.5), ("WS", 1.0), ("rr", -1.0)]


def fiedler(adj):
    L = np.diag(adj.sum(axis=1)) - adj
    return float(eigh(L, eigvals_only=True)[1])


def one(job):
    k, model, p, seed, rewire = job
    from mem4ristor.core import Mem4Network
    from mem4ristor.metrics import calculate_cognitive_entropy
    if model == "WS":
        G = nx.watts_strogatz_graph(N, k, p, seed=seed)
    else:
        G = nx.random_regular_graph(k, N, seed=seed)
    adj = nx.to_numpy_array(G, dtype=float)
    best = None
    for norm in NORMS:
        net = Mem4Network(size=10, heretic_ratio=HERETIC_RATIO, seed=seed,
                          adjacency_matrix=adj.copy(), coupling_norm=norm, cold_start=True,
                          rewire_threshold=(0.8 if rewire else 999.0))
        net.model.cfg["noise"]["sigma_v"] = SIGMA_V
        vmax = np.full(N, -np.inf)
        for t in range(STEPS):
            net.step(I_stimulus=I_STIM)
            if t >= STEPS - TAIL:
                vmax = np.maximum(vmax, net.v)
        v = net.v.copy(); u = net.model.u
        r = dict(h=float(calculate_cognitive_entropy(v)), norm=norm, sd=float(v.std()),
                 vmin=float(v.min()), vmax=float(v.max()), spike=float((vmax > 0).mean()),
                 below=float((v < -1.2).mean()), u_hi=float((u > 0.8).mean()),
                 u_mid=float(((u >= 0.2) & (u <= 0.8)).mean()))
        if best is None or r["h"] > best["h"]:
            best = r
    best.update(k=k, model=model, p=p, seed=seed, rewire=int(rewire), lambda2=fiedler(adj),
                dead=int(best["h"] < DEAD_THRESHOLD))
    return best


def wilson(x, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    ph = x / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    t0 = time.time()
    jobs = [(k, m, p, s, False) for k in K_VALUES for (m, p) in SPECS for s in SEEDS]
    with Pool(12) as pool:
        rows = pool.map(one, jobs, chunksize=4)
    with open(HERE / "d2_lambda2_static.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print("=" * 112)
    print("D2 -- graphes FIGES (recablage coupe), 30 graines neuves par cellule")
    print("=" * 112)
    print(f"{'k':>3} {'modele':>9} {'lambda2':>8} | {'dead':>5} {'IC95 Wilson':>13} | {'sd(v)':>6} "
          f"{'v_min':>6} {'v_max':>6} {'spike':>6} {'v<-1.2':>7} | {'u mid':>6} {'u>.8':>5}")
    for k in K_VALUES:
        for (m, p) in SPECS:
            s = [r for r in rows if r["k"] == k and r["model"] == m and r["p"] == p]
            x = sum(r["dead"] for r in s); lo, hi = wilson(x, len(s))
            med = lambda key: float(np.median([r[key] for r in s]))
            name = f"{m}{'' if p < 0 else f' p={p:.2f}'}"
            print(f"{k:>3} {name:>9} {np.mean([r['lambda2'] for r in s]):>8.3f} | {x / len(s):>5.2f} "
                  f"[{lo:.2f},{hi:.2f}]   | {med('sd'):>6.3f} {med('vmin'):>6.2f} {med('vmax'):>6.2f} "
                  f"{np.mean([r['spike'] for r in s]):>6.3f} {np.mean([r['below'] for r in s]):>7.2f} | "
                  f"{np.mean([r['u_mid'] for r in s]):>6.2f} {np.mean([r['u_hi'] for r in s]):>5.2f}")
        s = [r for r in rows if r["k"] == k]
        l2 = [r["lambda2"] for r in s]
        rho_d, p_d = spearmanr(l2, [r["dead"] for r in s])
        rho_h, p_h = spearmanr(l2, [r["h"] for r in s])
        print(f"    k={k} : Spearman(lambda2, dead) = {rho_d:+.3f} (p={p_d:.1e}) ; "
              f"Spearman(lambda2, H_cog) = {rho_h:+.3f} (p={p_h:.1e}) ; n={len(s)}")
        print("-" * 112)
    print(f"{len(rows)} cellules (x2 normes), {time.time() - t0:.0f} s. CSV : d2_lambda2_static.csv")


if __name__ == "__main__":
    main()
