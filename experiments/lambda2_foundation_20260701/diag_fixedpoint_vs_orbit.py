#!/usr/bin/env python3
"""
MANDAT NON-LIVRABLE : fonder theoriquement lambda2_crit ~ 2.31
Etape 1 (v2) : POINT FIXE ou ORBITE + QUELLE grandeur bascule a 2.3 ?

CORRECTION v1 -> v2 :
  v1 tournait en DRIVEN (I_stim=0.5) : mauvais regime (le forcage antagoniste
  sature u a 1.0 -> couplage repulsif -> diversite forcee). Le regime canonique
  du 2.31 (limit02_topology_sweep.py) est ENDOGENE (I_stim=0.0).
  v1 mesurait spatial_std : mauvaise grandeur. Le regime "dead zone" est defini
  par H_cog (bins discrets [-1.2,-0.4,0.4,1.2]), pas par std(v).

HYPOTHESE testee ici :
  dead zone = etat synchrone stabilise sur le point fixe bas du FHN v* ~ -1.286.
  Comme -1.286 < -1.2 (bord du bin le plus bas), tout tombe dans un seul bin
  cognitif -> H_cog = 0. Le seuil lambda2_crit = couplage minimal qui stabilise
  cet etat synchrone. Si c'est un POINT FIXE (temporal_std ~ 0 en deterministe),
  on lineraise autour d'un point fixe (Jacobien) et on EVITE Floquet.

Protocole : modele canonique, ENDOGENE (I_stim=0), degree_linear, cold_start.
On balaie BA m dans {2,3,4,5,8,10} (le sweep du preprint) + mesure de lambda2.

Diagnostics sur la queue (derniers 25 %) :
  - H_cog        : entropie cognitive 5 bins (la metrique du regime). Dead -> ~0.
  - H_cont       : entropie continue 100 bins. Decroit-elle en douceur (crossover) ?
  - mean_v       : ou se cale la distribution ? Dead -> proche de v* ~ -1.286.
  - spatial_std  : largeur de la distribution de v.
  - temporal_std : mouvement dans le temps. Point fixe -> ~0.
  - frac_bin0    : fraction des noeuds dans le bin le plus bas (v <= -1.2).
                   Dead -> proche de 1.0 (tout dans un seul bin).
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

import numpy as np
from scipy.linalg import eigh

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mem4ristor.core import Mem4Network
from mem4ristor.graph_utils import make_ba
from mem4ristor.metrics import (
    calculate_cognitive_entropy, calculate_continuous_entropy,
)

N = 100
STEPS = 3000
TAIL_FRAC = 0.25
SEED = 42
I_STIM = 0.0            # ENDOGENE (regime canonique du 2.31)
HERETIC_RATIO = 0.15
M_VALUES = [2, 3, 4, 5, 8, 10]


def fiedler_value(adj: np.ndarray) -> float:
    deg = adj.sum(axis=1)
    L = np.diag(deg) - adj
    vals = eigh(L, eigvals_only=True)
    return float(vals[1]) if len(vals) > 1 else 0.0


def run_one(adj: np.ndarray, sigma_v: float) -> dict:
    net = Mem4Network(
        size=10, heretic_ratio=HERETIC_RATIO, seed=SEED,
        adjacency_matrix=adj.copy(), coupling_norm="degree_linear",
        cold_start=True,
    )
    net.model.cfg["noise"]["sigma_v"] = sigma_v

    v_hist = np.empty((STEPS, N))
    for t in range(STEPS):
        net.step(I_stimulus=I_STIM)
        v_hist[t] = net.v

    tail0 = int(STEPS * (1 - TAIL_FRAC))
    v_tail = v_hist[tail0:]                          # (T_tail, N)
    v_final = v_hist[-1]

    h_cog = float(calculate_cognitive_entropy(v_final))
    h_cont = float(calculate_continuous_entropy(v_final, bins=100))
    mean_v = float(np.mean(v_tail))
    spatial_std = float(np.mean(np.std(v_tail, axis=1)))
    temporal_std = float(np.mean(np.std(v_tail, axis=0)))
    frac_bin0 = float(np.mean(v_final <= -1.2))
    u_mean = float(net.model.u.mean())

    return {
        "h_cog": h_cog, "h_cont": h_cont, "mean_v": mean_v,
        "spatial_std": spatial_std, "temporal_std": temporal_std,
        "frac_bin0": frac_bin0, "u_mean": u_mean,
    }


def main() -> int:
    print("=" * 100)
    print("DIAGNOSTIC v2 : nature de la dead zone (ENDOGENE, metrique H_cog)")
    print(f"N={N} steps={STEPS} I_stim={I_STIM} degree_linear cold_start seed={SEED}")
    print("point fixe bas du FHN attendu : v* ~ -1.286  (bord bin le plus bas = -1.2)")
    print("=" * 100)
    hdr = (f"{'m':>3} {'lambda2':>8} {'noise':>6} "
           f"{'H_cog':>7} {'H_cont':>7} {'mean_v':>7} {'sp_std':>7} "
           f"{'tmp_std':>8} {'frac<-1.2':>9} {'u_mean':>7}")
    print(hdr)
    print("-" * len(hdr))

    for m in M_VALUES:
        adj = make_ba(N, m, SEED)
        l2 = fiedler_value(adj)
        for label, sigma_v in [("noisy", 0.05), ("det", 0.0)]:
            r = run_one(adj, sigma_v)
            print(f"{m:>3} {l2:>8.3f} {label:>6} "
                  f"{r['h_cog']:>7.3f} {r['h_cont']:>7.3f} {r['mean_v']:>7.3f} "
                  f"{r['spatial_std']:>7.3f} {r['temporal_std']:>8.4f} "
                  f"{r['frac_bin0']:>9.2f} {r['u_mean']:>7.3f}")
    print("=" * 100)
    print("Lecture : la bascule H_cog=0 coincide-t-elle avec mean_v passant sous -1.2")
    print("  et frac<-1.2 -> 1.0 ? Et temporal_std(det) -> 0 = point fixe ?")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
