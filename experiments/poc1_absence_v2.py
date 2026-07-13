#!/usr/bin/env python3
"""
POC #1 v2 — Test d'absence : le reseau a-t-il son propre rythme ?

CORRECTIONS vs poc1_test_of_absence.py :

1. LZ mesure sur la MATRICE D'ETAT (T, N) — meme metrique que les
   experiences principales (C13, regime LZ < 0.85). La v1 calculait
   LZ sur v_mean (T, 1), une metrique incomparable.

2. Frequence dominante par FFT au lieu du zero-crossing rate.
   Le zero-crossing a un CV de ~55% entre seeds — la FFT donne une
   estimation spectralement resolue.

3. Synchronie par paires ajoutee (calculate_pairwise_synchrony) pour
   comparer directement avec la metrique primaire des experiences
   principales (synchronie = metrique PRIMAIRE, reglee cardinale).

4. LZ de v_mean conserve sous le nom lz_vmean pour traçabilite et
   pour montrer l'ecart avec LZ_state.

5. D=0 identifie explicitement comme "topologie sans effet" dans les
   resultats (couplage nul => m=3 et m=6 identiques par construction).
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
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mem4ristor.core import Mem4Network
from mem4ristor.metrics import (
    calculate_continuous_entropy,
    calculate_temporal_lz_complexity,
    calculate_pairwise_synchrony,
)

# ── Parametres (identiques a poc1_test_of_absence.py) ──────────────────────
N         = 100
STEPS     = 5000
TAIL_FRAC = 0.20          # 1000 derniers pas
SEEDS     = [42, 123, 777, 17, 256]
M_VALUES  = [3, 6]
D_VALUES  = [0.0, 0.15, 0.5]
COUPLING_NORM = "degree_linear"

OUTPUT_RAW = PROJECT_ROOT / "figures" / "poc1_v2_raw.csv"
OUTPUT_AGG = PROJECT_ROOT / "figures" / "poc1_v2_agg.csv"
FIGURE     = PROJECT_ROOT / "figures" / "poc1_v2.png"


# ── Helpers ────────────────────────────────────────────────────────────────
def autocorr(x: np.ndarray, lag: int) -> float:
    if len(x) <= lag or np.std(x) < 1e-12:
        return 0.0
    x0 = x[:-lag] - np.mean(x[:-lag])
    x1 = x[lag:]  - np.mean(x[lag:])
    den = np.std(x[:-lag]) * np.std(x[lag:])
    return float(np.mean(x0 * x1) / den) if den > 1e-12 else 0.0


def dominant_freq_zc(x: np.ndarray) -> float:
    """Zero-crossing rate (conserve pour comparaison — methode v1)."""
    if np.std(x) < 1e-12:
        return 0.0
    zc = np.sum(np.diff(np.sign(x - np.mean(x))) != 0)
    return float(zc / (2.0 * len(x)))


def dominant_freq_fft(x: np.ndarray) -> float:
    """Frequence dominante par FFT (pic de puissance sur les frequences > 0)."""
    if np.std(x) < 1e-12:
        return 0.0
    n = len(x)
    xc = x - np.mean(x)
    power = np.abs(np.fft.rfft(xc)) ** 2
    freqs = np.fft.rfftfreq(n)           # cycles par pas
    if len(freqs) < 2:
        return 0.0
    # Ignorer la composante DC (index 0)
    return float(freqs[1:][np.argmax(power[1:])])


# ── Run unique ─────────────────────────────────────────────────────────────
def run_one(m: int, seed: int, D: float) -> dict:
    import networkx as nx
    np.random.seed(seed)
    G   = nx.barabasi_albert_graph(N, m, seed=seed)
    adj = nx.to_numpy_array(G)

    net = Mem4Network(
        adjacency_matrix=adj,
        heretic_ratio=0.0,
        seed=seed,
        coupling_norm=COUPLING_NORM,
        cold_start=True,
    )
    net.model.cfg["coupling"]["D"] = D
    net.model.D_eff = D

    # Accumuler la matrice d'etat COMPLETE (T, N) pour le LZ correct
    v_hist_matrix = np.empty((STEPS, N), dtype=np.float32)
    v_mean_hist   = np.empty(STEPS, dtype=np.float64)
    h_hist        = np.empty(STEPS, dtype=np.float64)

    for step in range(STEPS):
        net.step(I_stimulus=0.0)           # AUCUN drive
        v_hist_matrix[step] = net.v
        v_mean_hist[step]   = float(np.mean(net.v))
        h_hist[step]        = calculate_continuous_entropy(net.v, bins=100)

    tail_s   = int(STEPS * (1 - TAIL_FRAC))
    v_matrix = v_hist_matrix[tail_s:]     # (1000, N)
    v_mean   = v_mean_hist[tail_s:]       # (1000,)
    h_tail   = h_hist[tail_s:]

    # ── Metriques correctes ─────────────────────────────────────────────
    lz_state = float(calculate_temporal_lz_complexity(v_matrix))   # (T, N) — CORRECT
    lz_vmean = float(calculate_temporal_lz_complexity(           # (T, 1) — v1 (comparaison)
        v_mean.reshape(-1, 1)))

    sync     = float(calculate_pairwise_synchrony(v_matrix))

    f_fft    = dominant_freq_fft(v_mean)
    f_zc     = dominant_freq_zc(v_mean)

    ac50     = autocorr(v_mean, 50)
    ac100    = autocorr(v_mean, 100)

    H_mean   = float(np.mean(h_tail))
    u_mean   = float(net.model.u.mean())

    return {
        "m": m, "seed": seed, "D": D,
        # METRIQUE PRINCIPALE
        "lz_state":  lz_state,    # LZ sur matrice (T,N) — comparable aux exps principales
        # METRIQUE V1 (reference)
        "lz_vmean":  lz_vmean,    # LZ sur v_mean (T,1) — ce que v1 mesurait
        # FREQUENCE
        "f_fft":     f_fft,       # FFT — robuste
        "f_zc":      f_zc,        # Zero-crossing — v1 (pour comparaison)
        # SYNCHRONIE (metrique primaire des exps principales)
        "sync":      sync,
        # PERIODICITE
        "ac50":      ac50,
        "ac100":     ac100,
        # ENTROPIE
        "H_cont":    H_mean,
        "u_mean":    u_mean,
    }


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    rows = []
    total = len(D_VALUES) * len(M_VALUES) * len(SEEDS)
    done  = 0

    for D in D_VALUES:
        for m in M_VALUES:
            for seed in SEEDS:
                try:
                    r = run_one(m, seed, D)
                    rows.append(r)
                except Exception as e:
                    import traceback
                    print(f"  ERROR m={m} seed={seed} D={D}: {e}")
                    traceback.print_exc()
                done += 1
                if done % 5 == 0 or done == total:
                    print(f"  [{done:2d}/{total}]  m={m}  D={D}  seed={seed}  "
                          f"({time.time()-t0:.0f}s)")

    df_raw = pd.DataFrame(rows)
    df_raw.to_csv(OUTPUT_RAW, index=False)
    print(f"\nRaw CSV: {OUTPUT_RAW}  ({len(df_raw)} rows)")

    # ── Agregation ─────────────────────────────────────────────────────────
    agg = df_raw.groupby(["m", "D"]).agg(
        lz_state_mean =("lz_state",  "mean"),
        lz_state_std  =("lz_state",  "std"),
        lz_vmean_mean =("lz_vmean",  "mean"),
        lz_vmean_std  =("lz_vmean",  "std"),
        f_fft_mean    =("f_fft",     "mean"),
        f_fft_std     =("f_fft",     "std"),
        f_zc_mean     =("f_zc",      "mean"),
        f_zc_std      =("f_zc",      "std"),
        sync_mean     =("sync",      "mean"),
        sync_std      =("sync",      "std"),
        ac50_mean     =("ac50",      "mean"),
        ac50_std      =("ac50",      "std"),
        H_mean        =("H_cont",    "mean"),
        H_std         =("H_cont",    "std"),
        count         =("seed",      "count"),
    ).reset_index()

    agg.to_csv(OUTPUT_AGG, index=False)
    print(f"Agg CSV: {OUTPUT_AGG}")

    # ── Resultats console ──────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("VERDICT POC #1 v2 — RYTHME SPONTANE (aucun drive)")
    print("Metrique LZ : MATRICE D'ETAT (T,N) — comparable aux experiences principales")
    print("Seuil structure : LZ_state < 0.85 (regime structure dans les exps principales)")
    print("=" * 78)

    for m in M_VALUES:
        print(f"\n  Topologie BA m={m} :")
        for D in D_VALUES:
            sub = agg[(agg["m"] == m) & (agg["D"] == D)]
            if sub.empty:
                continue
            r = sub.iloc[0]

            d0_note = " [D=0 : topologie sans effet, identique a m=3]" if D == 0.0 and m == 6 else ""

            # Verdict
            lz_s   = r["lz_state_mean"]
            f_fft  = r["f_fft_mean"]
            f_fft_std = r["f_fft_std"]
            sync   = r["sync_mean"]
            ac50   = r["ac50_mean"]

            structured  = lz_s < 0.85
            periodic    = ac50 > 0.3
            low_sync    = sync < 0.1    # synchronie faible = diversite maintenue
            cv_freq     = f_fft_std / f_fft if f_fft > 1e-6 else float("inf")
            reproducible = cv_freq < 0.3   # CV < 30% sur les seeds

            if structured:
                verdict = "STRUCTURE (LZ_state < 0.85)"
            else:
                verdict = "NON STRUCTURE (LZ_state >= 0.85)"
            if periodic:
                verdict += " + PERIODIQUE"
            if reproducible:
                verdict += " + freq REPRODUCTIBLE"
            else:
                verdict += f" + freq VARIABLE (CV={cv_freq:.0%})"

            print(
                f"    D={D}  "
                f"LZ_state={lz_s:.3f}+/-{r['lz_state_std']:.3f}  "
                f"[v1 LZ_vmean={r['lz_vmean_mean']:.3f}]  "
                f"f_fft={f_fft:.4f}+/-{f_fft_std:.4f}  "
                f"[v1 f_zc={r['f_zc_mean']:.4f}]  "
                f"sync={sync:.4f}  ac50={ac50:.3f}"
                f"{d0_note}"
            )
            print(f"    => {verdict}")

    # ── Figure ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.patch.set_facecolor("#1e1e2e")
    colors_D = {0.0: "#4fc3f7", 0.15: "#ff8a65", 0.5: "#b5ff40"}
    markers_m = {3: "o", 6: "s"}
    labels = {3: "BA m=3", 6: "BA m=6"}

    metrics = [
        ("lz_state_mean", "lz_state_std",  "LZ_state (matrice T,N) — CORRECT",  0, 0, 0.85, "LZ < 0.85 = structure"),
        ("lz_vmean_mean", "lz_vmean_std",  "LZ_vmean (v_mean 1D) — v1 FAUTIF",  0, 1, None, None),
        ("f_fft_mean",    "f_fft_std",     "Frequence dominante FFT (cycles/pas)",1, 0, None, None),
        ("f_zc_mean",     "f_zc_std",      "Freq zero-crossing v1 (cycles/pas)", 1, 1, None, None),
        ("sync_mean",     "sync_std",       "Synchronie pairwise",               0, 2, 0.1,  "sync < 0.1 = diversite"),
        ("ac50_mean",     "ac50_std",       "Autocorrelation lag=50",            1, 2, 0.3,  "ac50 > 0.3 = periodique"),
    ]

    for (col, col_std, title, ri, ci, thresh, thresh_label) in metrics:
        ax = axes[ri][ci]
        ax.set_facecolor("#252535")
        for m in M_VALUES:
            for D in D_VALUES:
                sub = agg[(agg["m"] == m) & (agg["D"] == D)]
                if sub.empty:
                    continue
                ax.errorbar(
                    D, sub[col].values[0], yerr=sub[col_std].values[0],
                    color=colors_D[D], marker=markers_m[m], ms=10, lw=2,
                    label=f"{labels[m]}, D={D}", capsize=4,
                )
        if thresh is not None:
            ax.axhline(thresh, color="#ffcc00", ls="--", alpha=0.6,
                       label=thresh_label)
        ax.set_title(title, color="white", fontsize=9)
        ax.set_xlabel("D", color="white")
        ax.tick_params(colors="white")
        ax.set_facecolor("#252535")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    plt.suptitle(
        "POC #1 v2 — Rythme spontane (aucun drive)\n"
        "Gauche : metriques corrigees (LZ_state, FFT) | Droite : metriques v1 pour comparaison",
        color="white", fontsize=11, y=1.01,
    )
    plt.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURE, dpi=150, bbox_inches="tight", facecolor="#1e1e2e")
    print(f"\nFigure: {FIGURE}")
    print(f"Wall time: {time.time() - t0:.1f}s")

    # ── Comparaison directe v1 vs v2 ───────────────────────────────────────
    print("\n" + "=" * 78)
    print("COMPARAISON v1 (fautive) vs v2 (correcte)")
    print("=" * 78)
    print(f"  {'m':>4} {'D':>5} | {'LZ_vmean(v1)':>14} | {'LZ_state(v2)':>14} | {'f_zc(v1)':>10} | {'f_fft(v2)':>10}")
    print("  " + "-" * 65)
    for m in M_VALUES:
        for D in D_VALUES:
            sub = agg[(agg["m"] == m) & (agg["D"] == D)]
            if sub.empty:
                continue
            r = sub.iloc[0]
            print(f"  {m:>4} {D:>5.2f} | "
                  f"{r['lz_vmean_mean']:>7.3f}+/-{r['lz_vmean_std']:.3f} | "
                  f"{r['lz_state_mean']:>7.3f}+/-{r['lz_state_std']:.3f} | "
                  f"{r['f_zc_mean']:>7.4f}+/-{r['f_zc_std']:.4f} | "
                  f"{r['f_fft_mean']:>7.4f}+/-{r['f_fft_std']:.4f}")

    print("\nNote : LZ_state < 0.85 = regime structure (seuil des experiences principales)")
    print("       Si LZ_state >= 0.85 partout => l'oscillateur intrinseque")
    print("       n'est PAS plus structure qu'un signal aleatoire — C20 a revoir.")


if __name__ == "__main__":
    main()
