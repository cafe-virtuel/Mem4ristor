#!/usr/bin/env python3
"""
AUDIT FABLE 2026-11-11 -- D3 : le resultat central (FULL vs FROZEN_U) survit-il au
retrait du recablage ?

Constat de lecture : p2_sigma_social_ablation.run_one() construit le reseau avec une
matrice d'adjacence et le rewire_threshold par defaut (0.8). Dans le bras FULL, u monte
et le graphe est recable ; dans le bras FROZEN_U (epsilon_u=0, u reste a 0.05) il ne
l'est jamais. Les deux bras different donc par DEUX choses : la modulation du couplage
par le doute ET le recablage. L'ablation publiee ne les separe pas.

Question, ecrite AVANT la mesure : en coupant le recablage dans le bras FULL (sans
toucher au script d'origine : on neutralise la methode dans le processus), la
synchronie reste-t-elle basse ?  Si oui, le resultat central tient -- et il est PLUS
propre qu'avant. Si elle remonte, c'est le recablage qui portait une part de l'effet.

Protocole : celui de b4_ablation_robustness.py (run_one, I_STIM=0.5, 1000+5000 pas,
degree_linear), ses 10 graines canoniques, BA m=3 (seed 42) et lattice 10x10.
Usage : uv run --frozen python experiments/fable_audit_20261111/d3_central_ablation_rewire.py
"""
from __future__ import annotations
import csv, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))


def one(job):
    topo, cond, rewire, seed = job
    import p2_sigma_social_ablation as ab
    from mem4ristor.topology import Mem4Network
    from mem4ristor.graph_utils import make_ba, make_lattice_adj
    adj = make_ba(100, 3, seed=42) if topo == "ba_m3" else make_lattice_adj(10)
    counter = {"n": 0}
    orig = Mem4Network._doubt_driven_rewire
    if rewire:
        def wrapped(self):
            before = self.rewire_count
            orig(self)
            counter["n"] += self.rewire_count - before
        Mem4Network._doubt_driven_rewire = wrapped
    else:
        Mem4Network._doubt_driven_rewire = lambda self: None
    try:
        res = ab.run_one(adj, cond, seed)
    finally:
        Mem4Network._doubt_driven_rewire = orig
    return dict(topo=topo, cond=cond, rewire=int(rewire), seed=seed,
                h_cont=float(res[1]), sync=float(res[2]), rewires=counter["n"])


def main():
    import p2_sigma_social_ablation as ab
    t0 = time.time()
    jobs = [(t, c, rw, s) for t in ("ba_m3", "lattice") for (c, rw) in
            (("FULL", True), ("FULL", False), ("FROZEN_U", True)) for s in ab.SEEDS]
    with Pool(4) as pool:
        rows = pool.map(one, jobs, chunksize=1)
    with open(HERE / "d3_central_ablation_rewire.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("=" * 96)
    print("D3 -- ablation centrale, avec et sans recablage dans le bras FULL (10 graines canoniques)")
    print("=" * 96)
    print(f"{'topo':>8} {'bras':>22} | {'sync moy':>9} {'min':>7} {'max':>7} | {'H_cont':>7} | {'echanges/run':>12}")
    for topo in ("ba_m3", "lattice"):
        for (c, rw, name) in (("FULL", 1, "FULL (publie, recabl.)"), ("FULL", 0, "FULL sans recablage"),
                              ("FROZEN_U", 1, "FROZEN_U")):
            s = [r for r in rows if r["topo"] == topo and r["cond"] == c and r["rewire"] == rw]
            sy = np.array([r["sync"] for r in s])
            print(f"{topo:>8} {name:>22} | {sy.mean():>9.4f} {sy.min():>7.4f} {sy.max():>7.4f} | "
                  f"{np.mean([r['h_cont'] for r in s]):>7.3f} | {np.mean([r['rewires'] for r in s]):>12.1f}")
        a = np.array([r["sync"] for r in rows if r["topo"] == topo and r["cond"] == "FULL" and r["rewire"] == 0])
        z = np.array([r["sync"] for r in rows if r["topo"] == topo and r["cond"] == "FROZEN_U"])
        d = z - a
        print(f"         FROZEN_U - FULL(sans recablage) : {d.mean():+.4f} ; d de Cohen apparie = {d.mean() / d.std(ddof=1):.2f} ; "
              f"separation complete : {a.max() < z.min()}")
        print("-" * 96)
    print(f"{len(rows)} runs, {time.time() - t0:.0f} s. CSV : d3_central_ablation_rewire.csv")


if __name__ == "__main__":
    main()
