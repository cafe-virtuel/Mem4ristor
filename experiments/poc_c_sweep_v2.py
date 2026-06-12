#!/usr/bin/env python3
"""
POC C SWEEP **v2** — 2026-06-12 (Claude Code / Fable, session autonome)

Réponse aux 3 problèmes de la contre-expertise du 2026-06-03 (AUDIT C20,
SYNAPSE MEM4-COWORK-AUDIT-C20-2026-06-03) sur poc_c_sweep.py (v1) :

  1. PROBLÈME CRITIQUE 2 : la fréquence v1 était mesurée par zero-crossing
     (surestime ~5×, compte les harmoniques FHN). → v2 : pic FFT (f_fft).
  2. PROBLÈME MAJEUR 1 : le classifieur binaire INTRINSIC/F_DRIVE discrétise
     un continuum. → v2 : on rapporte des MÉTRIQUES CONTINUES :
       - f_fft brute (pas de classement)
       - drive_power_frac : fraction de la puissance spectrale dans la bande
         de F_DRIVE (±1 bin) — mesure continue de l'entraînement.
  3. PROBLÈME CRITIQUE 1 : LZ v1 sur v_mean (T,1) incomparable aux exps
     principales. → v2 : LZ_state sur la matrice d'état (T,N) en plus.

Grille identique à v1 pour comparabilité : n_pivots 1..10, m∈{3,6},
D∈{0.0,0.15,0.5}, 5 seeds, 3000 pas, tail 25%. Le winner_fft (f_fft plus
proche de F_DRIVE=0.05 que de F_INTRINSIC_FFT=0.002) n'est donné qu'à titre
indicatif — la conclusion doit se lire sur les courbes continues.

Sorties : figures/poc_c_sweep_v2_raw.csv, _agg.csv, .png (nouveaux fichiers).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mem4ristor.core import Mem4Network
from mem4ristor.metrics import calculate_temporal_lz_complexity

# Réutilise la sélection de pivots et la topologie de la v1 (non modifiée)
import poc_c_sweep as v1

N = v1.N
STEPS = v1.STEPS
TAIL_FRAC = v1.TAIL_FRAC
SEEDS = v1.SEEDS
M_VALUES = v1.M_VALUES
D_VALUES = v1.D_VALUES
A = v1.A
F_DRIVE = v1.F_DRIVE
F_INTRINSIC_FFT = 0.002          # POC1 v2 (contre-expertise) : vraie f endogène FFT
COUPLING_NORM = v1.COUPLING_NORM
N_PIVOTS_SWEEP = v1.N_PIVOTS_SWEEP

OUTPUT_RAW = PROJECT_ROOT / "figures" / "poc_c_sweep_v2_raw.csv"
OUTPUT_AGG = PROJECT_ROOT / "figures" / "poc_c_sweep_v2_agg.csv"
FIGURE = PROJECT_ROOT / "figures" / "poc_c_sweep_v2.png"


def fft_peak_and_drive_power(v_1d: np.ndarray, dt: float = 1.0):
    """Pic FFT (cycles/step) + fraction de puissance dans la bande F_DRIVE ±1 bin."""
    x = v_1d - np.mean(v_1d)
    n = len(x)
    if n < 8 or np.std(x) < 1e-12:
        return 0.0, 0.0
    power = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(n, d=dt)
    power[0] = 0.0
    total = float(power.sum())
    if total <= 0:
        return 0.0, 0.0
    f_fft = float(freqs[int(np.argmax(power))])
    df = freqs[1] - freqs[0]
    band = (freqs >= F_DRIVE - 1.5 * df) & (freqs <= F_DRIVE + 1.5 * df)
    drive_power_frac = float(power[band].sum() / total)
    return f_fft, drive_power_frac


def lz_state(v_matrix: np.ndarray) -> float:
    """LZ sur la matrice d'état (T,N) — comparable aux expériences principales."""
    if np.std(v_matrix) < 1e-12:
        return 0.0
    return float(calculate_temporal_lz_complexity(v_matrix))


def run_one(adj, D, seed, n_pivots):
    pivot_set = v1.select_pivots(adj, n_pivots, seed)
    net = Mem4Network(adjacency_matrix=adj, heretic_ratio=0.0, seed=seed,
                      coupling_norm=COUPLING_NORM, cold_start=True)
    net.model.cfg["coupling"]["D"] = D
    net.model.D_eff = D

    tail_start = int(STEPS * (1 - TAIL_FRAC))
    v_mean_hist = []
    v_tail_snaps = []
    for step in range(STEPS):
        i_stim = np.zeros(N)
        drive_t = A * np.sin(2 * np.pi * F_DRIVE * step)
        for p in pivot_set:
            i_stim[p] = drive_t
        net.step(I_stimulus=i_stim)
        v_mean_hist.append(float(np.mean(net.v)))
        if step >= tail_start:
            v_tail_snaps.append(net.v.copy())

    v_mean_tail = np.array(v_mean_hist[tail_start:])
    v_state_tail = np.array(v_tail_snaps)

    f_fft, drive_pf = fft_peak_and_drive_power(v_mean_tail)
    ac50 = (float(np.corrcoef(v_mean_tail[:-50], v_mean_tail[50:])[0, 1])
            if len(v_mean_tail) > 50 else 0.0)
    winner_fft = ("F_DRIVE" if abs(f_fft - F_DRIVE) < abs(f_fft - F_INTRINSIC_FFT)
                  else "INTRINSIC")
    return {
        "f_fft": f_fft,
        "drive_power_frac": drive_pf,
        "lz_state": lz_state(v_state_tail),
        "ac50": ac50,
        "winner_fft": winner_fft,
    }


def main():
    t0 = time.time()
    rows = []
    total = len(N_PIVOTS_SWEEP) * len(M_VALUES) * len(D_VALUES) * len(SEEDS)
    done = 0
    for n_pivots in N_PIVOTS_SWEEP:
        for m in M_VALUES:
            for D in D_VALUES:
                for seed in SEEDS:
                    adj_s = v1.make_ba_adj(m, seed=seed)
                    res = run_one(adj_s, D, seed, n_pivots)
                    rows.append({"n_pivots": n_pivots, "m": m, "D": D, "seed": seed, **res})
                    done += 1
                    if done % 30 == 0 or done == total:
                        el = time.time() - t0
                        print(f"  [{done:3d}/{total}] n_pivots={n_pivots:2d} m={m} D={D:.2f} "
                              f"f_fft={res['f_fft']:.4f} drive_pf={res['drive_power_frac']:.3f} "
                              f"lz_state={res['lz_state']:.2f} ({el:.0f}s, ~{el/done*(total-done):.0f}s rest)")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_RAW, index=False)
    print(f"\nRaw CSV: {OUTPUT_RAW} ({len(df)} rows)")

    agg = (df.groupby(["n_pivots", "m", "D"])
             .agg(f_fft_mean=("f_fft", "mean"), f_fft_std=("f_fft", "std"),
                  drive_pf_mean=("drive_power_frac", "mean"),
                  drive_pf_std=("drive_power_frac", "std"),
                  lz_state_mean=("lz_state", "mean"),
                  ac50_mean=("ac50", "mean"),
                  fdrive_wins=("winner_fft", lambda s: int((s == "F_DRIVE").sum())),
                  n_seeds=("seed", "count"))
             .reset_index())
    agg.to_csv(OUTPUT_AGG, index=False)
    print(f"Agg CSV: {OUTPUT_AGG} ({len(agg)} rows)")

    print("\n" + "=" * 76)
    print("POC C SWEEP v2 — drive_power_frac (continu) par condition")
    print("=" * 76)
    for m in M_VALUES:
        for D in D_VALUES:
            sub = agg[(agg["m"] == m) & (agg["D"] == D)].sort_values("n_pivots")
            print(f"\n  m={m}  D={D}:")
            for _, r in sub.iterrows():
                bar = "#" * int(r["drive_pf_mean"] * 20)
                print(f"    {int(r['n_pivots']):2d}%  drive_pf={r['drive_pf_mean']:.3f}±{r['drive_pf_std']:.3f} "
                      f"[{bar:<20s}] f_fft={r['f_fft_mean']:.4f} lz_state={r['lz_state_mean']:.2f} "
                      f"FFT-wins={int(r['fdrive_wins'])}/{int(r['n_seeds'])}")

    fig, axes = plt.subplots(len(M_VALUES), len(D_VALUES), figsize=(13, 7),
                             sharex=True, sharey=True)
    for i, m in enumerate(M_VALUES):
        for j, D in enumerate(D_VALUES):
            ax = axes[i][j]
            sub = agg[(agg["m"] == m) & (agg["D"] == D)].sort_values("n_pivots")
            ax.errorbar(sub["n_pivots"], sub["drive_pf_mean"], yerr=sub["drive_pf_std"],
                        marker="o", color="darkorange", label="drive_power_frac")
            ax2 = ax.twinx()
            ax2.plot(sub["n_pivots"], sub["lz_state_mean"], marker="s", ms=3,
                     color="steelblue", alpha=0.6, label="LZ_state")
            ax2.set_ylim(0, 2.5)
            if j < len(D_VALUES) - 1:
                ax2.set_yticklabels([])
            ax.set_ylim(0, 1.0)
            ax.set_title(f"m={m}  D={D}", fontsize=10)
            ax.grid(alpha=0.3)
            if j == 0:
                ax.set_ylabel("drive_power_frac")
            if i == len(M_VALUES) - 1:
                ax.set_xlabel("% pivots")
    fig.suptitle("POC C Sweep v2 — entraînement continu (FFT) au lieu du classifieur binaire\n"
                 "orange = fraction de puissance à F_DRIVE | bleu = LZ_state (T,N)", fontsize=11)
    plt.tight_layout()
    plt.savefig(FIGURE, dpi=150, bbox_inches="tight")
    print(f"\nFigure: {FIGURE}")
    print(f"Wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
