#!/usr/bin/env python3
"""
AUDIT FABLE 2026-11-11 -- D4 : ou vit u dans le protocole central (I_stim=0.5, degree_linear,
heretiques 15 %), recablage coupe, a l'horizon du papier (6000 pas) et a horizon long (40000) ?

Question, ecrite AVANT la mesure : u est-il une grandeur GRADUEE (une part notable des noeuds
entre 0.2 et 0.8, et qui bouge) ou un VERROU a deux positions (plancher / plafond) ?
Le temoignage du 28/04 resume l'intuition ainsi : "proche de 0 le reseau se synchronise,
proche de 1 il se polarise, au milieu il vit".

Usage : uv run --frozen python experiments/fable_audit_20261111/d4_u_occupancy.py
"""
from __future__ import annotations
import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src"))
SEEDS = [42, 123, 777, 17, 256]


def one(job):
    topo, steps, seed = job
    from mem4ristor.core import Mem4Network
    from mem4ristor.graph_utils import make_ba, make_lattice_adj
    adj = make_ba(100, 3, seed=42) if topo == "ba_m3" else make_lattice_adj(10)
    net = Mem4Network(adjacency_matrix=adj.copy(), heretic_ratio=0.15, seed=seed,
                      coupling_norm="degree_linear", rewire_threshold=999.0)
    half = steps // 2
    U = []
    for t in range(steps):
        net.step(I_stimulus=0.5)
        if t >= half and t % 20 == 0:
            U.append(net.model.u.copy())
    U = np.array(U)
    side = U > 0.5
    return dict(topo=topo, steps=steps, lo=float((U < 0.2).mean()),
                mid=float(((U >= 0.2) & (U <= 0.8)).mean()), hi=float((U > 0.8).mean()),
                clamp=float((U >= 0.999).mean()),
                movers=float((side.any(axis=0) & (~side).any(axis=0)).mean()),
                sd_t=float(np.median(U.std(axis=0))))


def main():
    jobs = [(t, st, s) for t in ("ba_m3", "lattice") for st in (6000, 40000) for s in SEEDS]
    with Pool(4) as pool:
        rows = pool.map(one, jobs)
    print("D4 -- occupation de u, seconde moitie du run, 5 graines, recablage coupe")
    print(f"{'topo':>8} {'pas':>6} | {'u<0.2':>6} {'0.2-0.8':>8} {'u>0.8':>6} {'u=1.0':>6} | "
          f"{'noeuds traversant 0.5':>22} {'sd temporel median':>19}")
    for topo in ("ba_m3", "lattice"):
        for st in (6000, 40000):
            s = [r for r in rows if r["topo"] == topo and r["steps"] == st]
            m = lambda k: float(np.mean([r[k] for r in s]))
            print(f"{topo:>8} {st:>6} | {m('lo'):>6.2f} {m('mid'):>8.2f} {m('hi'):>6.2f} {m('clamp'):>6.2f} | "
                  f"{m('movers'):>22.2f} {m('sd_t'):>19.4f}")


if __name__ == "__main__":
    main()
