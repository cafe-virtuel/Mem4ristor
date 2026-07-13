#!/usr/bin/env python3
"""
POC C SWEEP - Seuil de bascule : oscillateur intrinseque -> drive externe.

Question : Entre 1 et 10 pivots (1% a 10% de N=100), a partir de combien
de noeuds pilotes le reseau bascule de INTRINSIC vers F_DRIVE ?

Contexte Session 013 :
- 1 pivot  (POC D, hub seul)    : INTRINSIC 6/6
- 10 pivots (POC C, mix hubs+random) : F_DRIVE 3/6 conditions
=> Seuil de bascule entre 1 et 10 pivots. Ce script l'identifie exactement.

Selection des pivots (identique a pocC_multi_pivots pour coherence) :
  - moitie depuis les noeuds de plus haut degre
  - moitie depuis un echantillon aleatoire (seed-controlled)
  - pour n_pivots=1 : uniquement le hub principal

Conditions : M in [3, 6], D in [0.0, 0.15, 0.5], 5 seeds, STEPS=3000.
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
from mem4ristor.metrics import calculate_temporal_lz_complexity

# ── Parametres ─────────────────────────────────────────────────────────────
N = 100
STEPS = 3000
TAIL_FRAC = 0.25
SEEDS = [42, 123, 777, 999, 314]        # 5 seeds
M_VALUES = [3, 6]
D_VALUES = [0.0, 0.15, 0.5]
A = 0.8
F_DRIVE = 0.05
F_INTRINSIC = 0.01                       # etabli en Session 013 POC 1
COUPLING_NORM = "degree_linear"

N_PIVOTS_SWEEP = list(range(1, 11))      # 1 % a 10 % de N=100

OUTPUT_RAW = PROJECT_ROOT / "figures" / "poc_c_sweep_raw.csv"
OUTPUT_AGG = PROJECT_ROOT / "figures" / "poc_c_sweep_agg.csv"
FIGURE     = PROJECT_ROOT / "figures" / "poc_c_sweep.png"


# ── Helpers ────────────────────────────────────────────────────────────────
def make_ba_adj(m, seed):
    import networkx as nx
    G = nx.barabasi_albert_graph(N, m, seed=seed)
    return nx.to_numpy_array(G)


def tail(arr):
    s = int(len(arr) * (1 - TAIL_FRAC))
    return arr[s:]


def select_pivots(adj, n_pivots, seed):
    """Meme logique que pocC_multi_pivots : moitie hubs, moitie aleatoire."""
    degrees = adj.sum(axis=1)
    sorted_idx = np.argsort(degrees)[::-1]
    np.random.seed(seed)

    if n_pivots == 1:
        return [int(sorted_idx[0])]

    n_hub = n_pivots // 2
    hub_idx = list(sorted_idx[:n_hub])
    n_rand = n_pivots - n_hub
    rand_idx = list(np.random.choice(N, n_rand, replace=False))
    pivot_set = list(set(hub_idx + rand_idx))[:n_pivots]

    # Si la deduplication a reduit le compte, combler avec d'autres noeuds
    pool = [i for i in range(N) if i not in pivot_set]
    np.random.shuffle(pool)
    while len(pivot_set) < n_pivots and pool:
        pivot_set.append(pool.pop(0))

    return pivot_set[:n_pivots]


def zero_crossing_freq(v_1d):
    centered = v_1d - np.mean(v_1d)
    zc = np.sum(np.diff(np.sign(centered)) != 0)
    return zc / (2.0 * len(v_1d))


def lz76(v_1d):
    if np.std(v_1d) < 1e-12:
        return 0.0
    return float(calculate_temporal_lz_complexity(v_1d.reshape(-1, 1)))


# ── Run unique ─────────────────────────────────────────────────────────────
def run_one(adj, D, seed, n_pivots):
    pivot_set = select_pivots(adj, n_pivots, seed)

    net = Mem4Network(
        adjacency_matrix=adj,
        heretic_ratio=0.0,
        seed=seed,
        coupling_norm=COUPLING_NORM,
        cold_start=True,
    )
    net.model.cfg["coupling"]["D"] = D
    net.model.D_eff = D

    v_mean_hist = []
    for step in range(STEPS):
        i_stim = np.zeros(N)
        drive_t = A * np.sin(2 * np.pi * F_DRIVE * step)
        for p in pivot_set:
            i_stim[p] = drive_t
        net.step(I_stimulus=i_stim)
        v_mean_hist.append(float(np.mean(net.v)))

    v_mean_tail = tail(np.array(v_mean_hist))
    f_v = zero_crossing_freq(v_mean_tail)

    # AC@lag50 (signature oscillateur intrinseque : haute autocorrelation)
    if len(v_mean_tail) > 50:
        ac50 = float(np.corrcoef(v_mean_tail[:-50], v_mean_tail[50:])[0, 1])
    else:
        ac50 = 0.0

    lz_val = lz76(v_mean_tail)

    dist_drive     = abs(f_v - F_DRIVE)
    dist_intrinsic = abs(f_v - F_INTRINSIC)
    winner = "F_DRIVE" if dist_drive < dist_intrinsic else "INTRINSIC"

    return {
        "f_v": f_v,
        "ac50": ac50,
        "lz": lz_val,
        "winner": winner,
        "dist_drive": dist_drive,
        "dist_intrinsic": dist_intrinsic,
    }


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    rows = []
    total = len(N_PIVOTS_SWEEP) * len(M_VALUES) * len(D_VALUES) * len(SEEDS)
    done = 0

    for n_pivots in N_PIVOTS_SWEEP:
        for m in M_VALUES:
            adj = make_ba_adj(m, seed=0)   # topologie fixe par m (seed graph sera re-tire par seed de simulation)
            for D in D_VALUES:
                for seed in SEEDS:
                    adj_s = make_ba_adj(m, seed=seed)   # graphe different par seed (comme Session 013)
                    res = run_one(adj_s, D, seed, n_pivots)
                    rows.append({
                        "n_pivots": n_pivots,
                        "m": m,
                        "D": D,
                        "seed": seed,
                        "f_v": res["f_v"],
                        "ac50": res["ac50"],
                        "lz": res["lz"],
                        "winner": res["winner"],
                        "dist_drive": res["dist_drive"],
                        "dist_intrinsic": res["dist_intrinsic"],
                    })
                    done += 1
                    if done % 15 == 0 or done == total:
                        elapsed = time.time() - t0
                        eta = elapsed / done * (total - done)
                        print(f"  [{done:3d}/{total}]  n_pivots={n_pivots:2d}  m={m}  D={D:.2f}  "
                              f"winner={res['winner']:<10s}  f_v={res['f_v']:.4f}  "
                              f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    df_raw = pd.DataFrame(rows)
    df_raw.to_csv(OUTPUT_RAW, index=False)
    print(f"\nRaw CSV: {OUTPUT_RAW}  ({len(df_raw)} rows)")

    # ── Agregation ─────────────────────────────────────────────────────────
    agg_rows = []
    for n_pivots in N_PIVOTS_SWEEP:
        for m in M_VALUES:
            for D in D_VALUES:
                sub = df_raw[
                    (df_raw["n_pivots"] == n_pivots) &
                    (df_raw["m"] == m) &
                    (df_raw["D"] == D)
                ]
                if sub.empty:
                    continue
                n = len(sub)
                f_drive_wins = int((sub["winner"] == "F_DRIVE").sum())
                f_drive_frac = f_drive_wins / n
                agg_rows.append({
                    "n_pivots":      n_pivots,
                    "pct_pivots":    n_pivots,      # N=100 donc n_pivots == pct
                    "m":             m,
                    "D":             D,
                    "n_seeds":       n,
                    "f_drive_wins":  f_drive_wins,
                    "f_drive_frac":  f_drive_frac,
                    "f_v_mean":      sub["f_v"].mean(),
                    "f_v_std":       sub["f_v"].std(),
                    "ac50_mean":     sub["ac50"].mean(),
                    "lz_mean":       sub["lz"].mean(),
                    "winner_majority": "F_DRIVE" if f_drive_frac > 0.5 else "INTRINSIC",
                })

    df_agg = pd.DataFrame(agg_rows)
    df_agg.to_csv(OUTPUT_AGG, index=False)
    print(f"Agg CSV: {OUTPUT_AGG}  ({len(df_agg)} rows)")

    # ── Resultats console ──────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("SWEEP POC C — Tipping point INTRINSIC -> F_DRIVE")
    print("=" * 72)
    for m in M_VALUES:
        for D in D_VALUES:
            sub = df_agg[(df_agg["m"] == m) & (df_agg["D"] == D)].sort_values("n_pivots")
            print(f"\n  m={m}  D={D}:")
            tipping = None
            for _, row in sub.iterrows():
                np_ = int(row["n_pivots"])
                frac = row["f_drive_frac"]
                bar = "#" * int(frac * 10) + "." * (10 - int(frac * 10))
                marker = ""
                if frac >= 0.5 and tipping is None:
                    tipping = np_
                    marker = " <<< BASCULE"
                print(f"    {np_:2d}%  [{bar}]  F_DRIVE {int(row['f_drive_wins'])}/{int(row['n_seeds'])}  "
                      f"frac={frac:.2f}  f_v={row['f_v_mean']:.4f}  ac50={row['ac50_mean']:+.3f}{marker}")
            if tipping is not None:
                print(f"  => SEUIL DE BASCULE : {tipping}% ({tipping} pivots)")
            else:
                print(f"  => INTRINSIC domine jusqu'a 10% (pas de bascule detectee)")

    # ── Figure ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        len(M_VALUES), len(D_VALUES),
        figsize=(13, 7),
        sharey=True,
        sharex=True,
    )
    fig.patch.set_facecolor("#1e1e2e")

    for i, m in enumerate(M_VALUES):
        for j, D in enumerate(D_VALUES):
            ax = axes[i][j]
            ax.set_facecolor("#252535")

            sub = df_agg[(df_agg["m"] == m) & (df_agg["D"] == D)].sort_values("n_pivots")
            xs = sub["n_pivots"].values
            fracs = sub["f_drive_frac"].values

            # Barres colorees selon majorite
            bar_colors = ["#ff8a65" if f >= 0.5 else "#4fc3f7" for f in fracs]
            ax.bar(xs, fracs, color=bar_colors, alpha=0.85, zorder=2)

            # Ligne 50 %
            ax.axhline(0.5, color="#b5ff40", ls="--", lw=1.5, alpha=0.8, zorder=3)

            # Seuil de bascule
            tipping_rows = sub[sub["f_drive_frac"] >= 0.5]
            if not tipping_rows.empty:
                tp = int(tipping_rows.iloc[0]["n_pivots"])
                ax.axvline(tp - 0.5, color="#b5ff40", ls=":", lw=2, alpha=0.7, zorder=4)
                ax.text(tp - 0.4, 0.95, f"T={tp}%", color="#b5ff40",
                        fontsize=8, va="top", zorder=5)

            ax.set_ylim(0, 1.05)
            ax.set_xlim(0.5, 10.5)
            ax.set_xticks(list(range(1, 11)))
            ax.set_xticklabels([f"{x}%" for x in range(1, 11)],
                               color="white", fontsize=7)
            ax.tick_params(colors="white")
            ax.set_title(f"m={m}  D={D}", color="white", fontsize=10, pad=4)

            if j == 0:
                ax.set_ylabel("Frac. F_DRIVE (5 seeds)", color="white", fontsize=9)
            if i == len(M_VALUES) - 1:
                ax.set_xlabel("% pivots drives", color="white", fontsize=9)

            for spine in ax.spines.values():
                spine.set_edgecolor("#444")

    plt.suptitle(
        "POC C Sweep — Seuil de bascule INTRINSIC → F_DRIVE\n"
        "Orange = F_DRIVE majoritaire | Bleu = INTRINSIC | Pointille vert = 50%",
        color="white", fontsize=11, y=1.01,
    )
    plt.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURE, dpi=150, bbox_inches="tight", facecolor="#1e1e2e")
    print(f"\nFigure: {FIGURE}")
    print(f"Wall time total: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
