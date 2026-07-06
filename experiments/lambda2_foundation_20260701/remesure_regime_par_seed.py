#!/usr/bin/env python3
"""
MANDAT NON-LIVRABLE : fonder theoriquement lambda2_crit ~ 2.31
Etape 2 : RE-MESURE PROPRE du regime, PAR SEED, code actuel.

Pourquoi : le 2.31 canonique repose sur p2_edge_betweenness.csv, dont les labels
de regime sont RECOPIES A LA MAIN (dict REGIME hardcode) depuis limit02, par TYPE
de topologie (pas par seed), avec une metrique ambigue (H_cont vs H_cog) et
figes AVANT le changement de bruit AUDIT-024 (1er mai). Cette etape remplace ce
raccourci par une mesure dynamique reelle, par instance de graphe, avec le code
d'aujourd'hui. Double objectif :
  (A) reetablir la frontiere de separation sur des bases propres ;
  (B) produire le nuage (lambda2, mean_v) pour tester l'hypothese :
      dead zone = distribution de v translatee sous le bord de bin -1.2.

Definition du regime (fidele a l'originale) :
  Pour chaque graphe, on teste les DEUX normalisations (uniform, degree_linear).
  best_H_cog = max des deux. dead_zone <=> best_H_cog < 0.10.
  (H_cog = entropie cognitive 5 bins, la metrique que le 2.31 pretend separer.)

Protocole canonique : ENDOGENE (I_stim=0), cold_start, N=100, 3000 pas,
sigma_v=0.05 (bruit canonique — essentiel : sans lui, aucune diversite).

Sortie : figures du dossier lambda2_foundation_20260701/
  remesure_regime_par_seed.csv  (une ligne par topo x seed x norm)
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
from scipy.linalg import eigh

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mem4ristor.core import Mem4Network
from mem4ristor.graph_utils import make_ba, make_er, make_lattice_adj
from mem4ristor.metrics import (
    calculate_cognitive_entropy, calculate_continuous_entropy,
)

N = 100
STEPS = 3000
TAIL_FRAC = 0.25
SEEDS = [42, 123, 777, 17, 256, 1337, 99, 314]
I_STIM = 0.0
HERETIC_RATIO = 0.15
SIGMA_V = 0.05
DEAD_THRESHOLD = 0.10
NORMS = ["uniform", "degree_linear"]

# Familles pour couvrir lambda2 de ~0.02 a ~7.7
TOPOS = (
    [(f"BA m={m}", "ba", {"m": m}) for m in [1, 2, 3, 4, 5, 6, 7, 8, 10]]
    + [(f"ER p={p}", "er", {"p": p}) for p in [0.05, 0.08, 0.12]]
    + [("Lattice 10x10", "lattice", {})]
)


def build_adj(kind: str, seed: int, **kw) -> np.ndarray:
    if kind == "ba":
        return make_ba(N, kw["m"], seed)
    if kind == "er":
        return make_er(N, kw["p"], seed)
    if kind == "lattice":
        return make_lattice_adj(10, periodic=True)
    raise ValueError(kind)


def fiedler_value(adj: np.ndarray) -> float:
    deg = adj.sum(axis=1)
    L = np.diag(deg) - adj
    vals = eigh(L, eigvals_only=True)
    return float(vals[1]) if len(vals) > 1 else 0.0


def run_one(adj: np.ndarray, norm: str, seed: int) -> dict:
    net = Mem4Network(
        size=10, heretic_ratio=HERETIC_RATIO, seed=seed,
        adjacency_matrix=adj.copy(), coupling_norm=norm, cold_start=True,
    )
    net.model.cfg["noise"]["sigma_v"] = SIGMA_V

    tail0 = int(STEPS * (1 - TAIL_FRAC))
    v_tail = []
    for t in range(STEPS):
        net.step(I_stimulus=I_STIM)
        if t >= tail0:
            v_tail.append(net.v.copy())
    v_tail = np.array(v_tail)          # (T_tail, N)
    v_final = v_tail[-1]

    return {
        "h_cog": float(calculate_cognitive_entropy(v_final)),
        "h_cont": float(calculate_continuous_entropy(v_final, bins=100)),
        "mean_v": float(np.mean(v_tail)),
        "std_v": float(np.mean(np.std(v_tail, axis=1))),
        "frac_bin0": float(np.mean(v_final <= -1.2)),
        "u_mean": float(net.model.u.mean()),
    }


def main() -> int:
    t0 = time.time()
    rows = []
    # Agrege par (topo, seed) : best_H_cog sur les deux normalisations
    per_instance = {}   # (topo, seed) -> dict

    print("=" * 92)
    print("RE-MESURE DU REGIME PAR SEED (endogene, code actuel)")
    print(f"N={N} steps={STEPS} I_stim={I_STIM} sigma_v={SIGMA_V} seeds={len(SEEDS)}")
    print("=" * 92)

    for name, kind, kw in TOPOS:
        for seed in SEEDS:
            adj = build_adj(kind, seed, **kw)
            l2 = fiedler_value(adj)
            best = {"h_cog": -1.0}
            for norm in NORMS:
                r = run_one(adj, norm, seed)
                rows.append({"topology": name, "seed": seed, "norm": norm,
                             "lambda2": l2, **r})
                if r["h_cog"] > best["h_cog"]:
                    best = {**r, "norm": norm}
            dead = int(best["h_cog"] < DEAD_THRESHOLD)
            per_instance[(name, seed)] = {
                "topology": name, "seed": seed, "lambda2": l2,
                "best_h_cog": best["h_cog"], "best_norm": best["norm"],
                "mean_v": best["mean_v"], "std_v": best["std_v"],
                "h_cont": best["h_cont"], "u_mean": best["u_mean"],
                "dead": dead,
            }
        # ligne d'avancement par topologie
        subs = [per_instance[(name, s)] for s in SEEDS]
        l2m = np.mean([x["lambda2"] for x in subs])
        hcm = np.mean([x["best_h_cog"] for x in subs])
        nd = sum(x["dead"] for x in subs)
        print(f"  {name:<14} lambda2~{l2m:6.3f}  best_H_cog~{hcm:5.3f}  "
              f"dead {nd}/{len(SEEDS)}")

    # ---- CSV detaille
    csv1 = HERE / "remesure_regime_par_seed.csv"
    with csv1.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "topology", "seed", "norm", "lambda2", "h_cog", "h_cont",
            "mean_v", "std_v", "frac_bin0", "u_mean"])
        w.writeheader()
        w.writerows(rows)

    # ---- Frontiere de separation PAR SEED (best-of-two-norms)
    inst = list(per_instance.values())
    dead_l2 = sorted([x["lambda2"] for x in inst if x["dead"] == 1])
    live_l2 = sorted([x["lambda2"] for x in inst if x["dead"] == 0])

    print("\n" + "=" * 92)
    print("FRONTIERE DE SEPARATION (par instance de graphe, best-of-two-norms)")
    print("=" * 92)
    print(f"  n instances = {len(inst)}  |  dead = {len(dead_l2)}  live = {len(live_l2)}")
    if dead_l2 and live_l2:
        max_live = max(live_l2)
        min_dead = min(dead_l2)
        print(f"  plus grand lambda2 FONCTIONNEL : {max_live:.3f}")
        print(f"  plus petit  lambda2 DEAD       : {min_dead:.3f}")
        if max_live < min_dead:
            mid = (max_live + min_dead) / 2
            print(f"  -> SEPARATION COMPLETE. gap=({max_live:.3f}, {min_dead:.3f}) "
                  f"midpoint={mid:.3f}")
        else:
            # chevauchement : compter la zone de melange
            lo = min_dead
            hi = max_live
            mix = [x for x in inst if lo <= x["lambda2"] <= hi]
            print(f"  -> CHEVAUCHEMENT sur [{lo:.3f}, {hi:.3f}] : "
                  f"{len(mix)} instances mixtes. Frontiere NON nette par seed.")
        # comparaison au 2.31 canonique
        print(f"  Rappel canonique : gap (2.126, 2.504), midpoint 2.31")

    print(f"\n[csv] {csv1}")
    print(f"Wall time: {time.time() - t0:.1f}s | {len(rows)} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
