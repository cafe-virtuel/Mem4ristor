#!/usr/bin/env python3
"""
AUDIT FABLE 2026-11-11 -- D1 : le recablage par defaut, et ou vit u.

Protocole : EXACTEMENT celui de
  experiments/lambda2_foundation_20260701/crucial_kfixe_lambda2_variable.py
(N=100, 3000 pas, cold start, I_stim=0, heretic_ratio=0.15, sigma_v=0.05,
 best-of-two-norms H_cog, memes 5 graines) -- l'experience qui a refute lambda2.

Constat de lecture (pas encore une mesure) : Mem4Network.step() appelle
_doubt_driven_rewire() a CHAQUE pas des qu'une matrice d'adjacence est fournie,
avec rewire_threshold=0.8 par defaut. Aucun script de ce dossier ne le desactive.
Un echange d'aretes preservant les degres conserve <k> et detruit lambda2.

Questions, ecrites AVANT la mesure :
  Q1. Combien d'echanges d'aretes pendant ces runs ? Que vaut lambda2 en fin de run ?
  Q2. Le verdict dead/functional change-t-il si l'on coupe le recablage
      (rewire_threshold=999) ? Si oui, la refutation de lambda2 est confondue.
      Si non, elle tient -- et il faut le dire aussi nettement.
  Q3. Ou vit u en fin de run : fraction < 0.2, dans [0.2, 0.8], > 0.8 ?

Usage : uv run --frozen python experiments/fable_audit_20261111/d1_rewire_and_u.py
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

N = 100
STEPS = 3000
SEEDS = [42, 123, 777, 17, 256]
I_STIM = 0.0
HERETIC_RATIO = 0.15
SIGMA_V = 0.05
DEAD_THRESHOLD = 0.10
NORMS = ["uniform", "degree_linear"]
K_VALUES = [6, 8, 10, 12]
SPECS = [("WS", 0.0), ("WS", 0.2), ("WS", 1.0), ("rr", -1.0)]


def fiedler(adj):
    adj = np.asarray(adj, dtype=float)
    L = np.diag(adj.sum(axis=1)) - adj
    return float(eigh(L, eigvals_only=True)[1])


def one(job):
    k, model, p, seed, norm, rewire = job
    from mem4ristor.core import Mem4Network
    from mem4ristor.metrics import calculate_cognitive_entropy
    if model == "WS":
        G = nx.watts_strogatz_graph(N, k, p, seed=seed)
    else:
        G = nx.random_regular_graph(k, N, seed=seed)
    adj = nx.to_numpy_array(G, dtype=float)
    l2_0 = fiedler(adj)
    net = Mem4Network(size=10, heretic_ratio=HERETIC_RATIO, seed=seed,
                      adjacency_matrix=adj.copy(), coupling_norm=norm,
                      cold_start=True,
                      rewire_threshold=(0.8 if rewire else 999.0))
    net.model.cfg["noise"]["sigma_v"] = SIGMA_V
    for _ in range(STEPS):
        net.step(I_stimulus=I_STIM)
    u = net.model.u
    A_end = np.asarray(net.adjacency_matrix, dtype=float)
    return dict(k=k, model=model, p=p, seed=seed, norm=norm, rewire=int(rewire),
                l2_start=l2_0, l2_end=fiedler(A_end),
                rewires=int(net.rewire_count),
                frac_edges_changed=float(np.abs(A_end - adj).sum() / max(adj.sum(), 1) / 2),
                h_cog=float(calculate_cognitive_entropy(net.v)),
                u_mean=float(u.mean()), u_lo=float((u < 0.2).mean()),
                u_mid=float(((u >= 0.2) & (u <= 0.8)).mean()), u_hi=float((u > 0.8).mean()))


def main():
    t0 = time.time()
    jobs = [(k, m, p, s, n, rw) for k in K_VALUES for (m, p) in SPECS
            for s in SEEDS for n in NORMS for rw in (True, False)]
    with Pool(12) as pool:
        rows = pool.map(one, jobs, chunksize=4)
    with open(HERE / "d1_rewire_and_u.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print("=" * 110)
    print("D1 -- protocole de crucial_kfixe_lambda2_variable.py, recablage ON (defaut) vs OFF")
    print("dead = best-of-two-norms H_cog < 0.10 (definition d'origine). 5 graines.")
    print("=" * 110)
    print(f"{'k':>3} {'modele':>7} {'l2_debut':>8} | {'ON: echanges':>12} {'aretes chg':>10} {'l2_fin':>7} "
          f"{'dead':>5} | {'OFF: dead':>9} | {'u<.2':>5} {'u mid':>6} {'u>.8':>5}  (u : OFF, moy. 2 normes)")
    for k in K_VALUES:
        for (m, p) in SPECS:
            sel = [r for r in rows if r["k"] == k and r["model"] == m and r["p"] == p]
            out = {}
            for rw in (1, 0):
                s2 = [r for r in sel if r["rewire"] == rw]
                dead = []
                for seed in SEEDS:
                    best = max(r["h_cog"] for r in s2 if r["seed"] == seed)
                    dead.append(best < DEAD_THRESHOLD)
                out[rw] = dict(dead=np.mean(dead),
                               rew=np.mean([r["rewires"] for r in s2]),
                               chg=np.mean([r["frac_edges_changed"] for r in s2]),
                               l2e=np.mean([r["l2_end"] for r in s2]),
                               l2s=np.mean([r["l2_start"] for r in s2]),
                               lo=np.mean([r["u_lo"] for r in s2]),
                               mid=np.mean([r["u_mid"] for r in s2]),
                               hi=np.mean([r["u_hi"] for r in s2]))
            name = f"{m}{'' if p < 0 else f' p={p:.1f}'}"
            print(f"{k:>3} {name:>7} {out[1]['l2s']:>8.3f} | {out[1]['rew']:>12.1f} {out[1]['chg']:>10.2f} "
                  f"{out[1]['l2e']:>7.3f} {out[1]['dead']:>5.2f} | {out[0]['dead']:>9.2f} | "
                  f"{out[0]['lo']:>5.2f} {out[0]['mid']:>6.2f} {out[0]['hi']:>5.2f}")
        print("-" * 110)
    print(f"{len(rows)} runs, {time.time() - t0:.0f} s. CSV : d1_rewire_and_u.csv")


if __name__ == "__main__":
    main()
