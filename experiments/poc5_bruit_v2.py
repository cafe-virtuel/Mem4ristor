#!/usr/bin/env python3
"""
POC #5 **v2** — bruit filtré, 5 seeds — 2026-06-12 (Claude Code / Fable)

Réponse au PROBLÈME MODÉRÉ 1 de la contre-expertise du 2026-06-03 :
le POC #5 de Session 013 (poc245_batch.py::poc5_bruit) tournait sur n=1 seed
(seed=42) — aucune valeur statistique. v2 : 5 seeds, mêmes conditions
(m∈{3,6}, D∈{0.0,0.15,0.5}, bruit brownien 1/f², injecté sur tous les nœuds).

Question inchangée : le réseau FILTRE-t-il le bruit (cross_corr basse,
AC@lag5 haute) ou le TRANSMET-il passivement (cross_corr haute) ?
Verdict Session 013 (1 seed) : PASSIVE 6/6, cross_corr≈+0.92.

Sorties : figures/poc5_bruit_v2_raw.csv, _agg.csv (nouveaux fichiers).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mem4ristor.core import Mem4Network
from mem4ristor.metrics import calculate_continuous_entropy

import poc245_batch as base  # helpers v1 non modifiés

SEEDS = [42, 123, 777, 456, 999]
SIGMA_NOISE = 0.8            # identique v1

OUTPUT_RAW = PROJECT_ROOT / "figures" / "poc5_bruit_v2_raw.csv"
OUTPUT_AGG = PROJECT_ROOT / "figures" / "poc5_bruit_v2_agg.csv"


def run_one(m, D, seed):
    adj = base.make_ba_adj(m, base.N, seed)
    net = Mem4Network(adjacency_matrix=adj, heretic_ratio=base.HERETIC, seed=seed,
                      coupling_norm=base.COUPLING_NORM, cold_start=True)
    net.model.cfg["coupling"]["D"] = D
    net.model.D_eff = D

    rng = np.random.RandomState(seed)
    white = rng.normal(0, SIGMA_NOISE, base.STEPS)
    brownian = np.cumsum(white) * 0.1
    brownian = brownian - np.mean(brownian)

    v_mean_hist, h_hist = [], []
    for step in range(base.STEPS):
        net.step(I_stimulus=brownian[step])
        v_mean_hist.append(np.mean(net.v))
        h_hist.append(calculate_continuous_entropy(net.v, bins=100))

    v_mean_tail = base.tail(np.array(v_mean_hist))
    h_tail = base.tail(np.array(h_hist))
    drive_tail = base.tail(brownian)
    return {
        "H_cont": float(np.mean(h_tail)),
        "H_std": float(np.std(h_tail)),
        "LZ_v_mean": base.lz76_trace(v_mean_tail),
        "cross_corr_drive": base.tracking_corr(v_mean_tail, drive_tail),
        "ac_lag5": float(np.corrcoef(v_mean_tail[:-5], v_mean_tail[5:])[0, 1]),
        "ac_lag50": float(np.corrcoef(v_mean_tail[:-50], v_mean_tail[50:])[0, 1]),
    }


def main():
    t0 = time.time()
    rows = []
    for m in base.M_VALUES:
        for D in base.D_VALUES:
            for seed in SEEDS:
                res = run_one(m, D, seed)
                rows.append({"m": m, "D": D, "seed": seed, **res})
                print(f"  m={m} D={D:.2f} seed={seed:3d} : cc_drive={res['cross_corr_drive']:+.3f} "
                      f"H={res['H_cont']:.3f} ac5={res['ac_lag5']:+.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_RAW, index=False)

    agg = (df.groupby(["m", "D"])
             .agg(cc_drive_mean=("cross_corr_drive", "mean"),
                  cc_drive_std=("cross_corr_drive", "std"),
                  H_cont_mean=("H_cont", "mean"), H_cont_std=("H_cont", "std"),
                  ac5_mean=("ac_lag5", "mean"), ac50_mean=("ac_lag50", "mean"),
                  lz_mean=("LZ_v_mean", "mean"), n_seeds=("seed", "count"))
             .reset_index())
    agg.to_csv(OUTPUT_AGG, index=False)

    print("\n" + "=" * 70)
    print("POC #5 v2 — transmission du bruit (5 seeds)")
    print("=" * 70)
    for _, r in agg.iterrows():
        verdict = "PASSIVE (transmet)" if r["cc_drive_mean"] > 0.5 else (
                  "FILTRE" if r["cc_drive_mean"] < 0.2 else "INTERMEDIAIRE")
        print(f"  m={int(r['m'])} D={r['D']:.2f} : cc_drive={r['cc_drive_mean']:+.3f}±{r['cc_drive_std']:.3f} "
              f"({int(r['n_seeds'])} seeds) -> {verdict}")
    print(f"\nCSV : {OUTPUT_RAW}\n      {OUTPUT_AGG}")
    print(f"Wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
