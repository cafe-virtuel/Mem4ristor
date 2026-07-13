#!/usr/bin/env python3
"""
POC Option A - Adversarial local drive (single pivot node).

DESIGN ADVERSARIAL :
- Un SEUL noeud "pivot" (idx 0) recoit I_pivot(t) = A*sin(2*pi*f*t).
- Tous les autres noeuds recoivent I = 0 (endogenous only).
- Question : le reste du reseau suit-il le pivot OU s'auto-organise-t-il
  sur sa propre dynamique ?
  - SUIVI du pivot (tracking_corr pivot<->others haute) = CONVERGENCE
  - AUTO-ORGANISATION (LZ preserve, tracking_corr faible) = LOCK-IN

POURQUOI C'EST ADVERSARIAL :
- Le reseau a le choix : il peut amplifier le signal du pivot (convergence)
  ou l'ignorer et tomber sur sa propre frequence propre (lock-in sur
  un etat interne).
- Le drive est NON TRIVIAL : un seul noeud sur 100 est pilote. C'est
  l'analogue du test MoE ou un seul expert recoit le signal et le routeur
  doit decider de le propager ou pas.

API EXPLOITEE :
- Mem4ristorV3.step(I_stimulus=array) accepte deja un array de taille N
  (ligne 258-262 de dynamics.py). Pas de patch code source.
- I_eff[heretic_mask] *= -1.0 : si le pivot est heretique, le drive est
  inverse. On choisit PIVOT_IDX = 0 et on GARANTIT qu'il n'est pas
  heretique (en pratique, on force heretic_ratio=0.0 dans cette POC).
"""

from __future__ import annotations
import sys, os, time
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
from mem4ristor.metrics import calculate_continuous_entropy, calculate_temporal_lz_complexity

# Parameters
N = 100
STEPS = 3000
TAIL_FRAC = 0.25
SEEDS = [42, 123, 777]
PIVOT_IDX = 0
HERETIC = 0.0
COUPLING_NORM = "degree_linear"

# Local signal on pivot only
A_DRIVE = 0.8
F_DRIVE = 0.05

# Test cells
M_VALUES = [3, 6]
D_VALUES = [0.0, 0.15, 0.5]

OUTPUT_RAW = PROJECT_ROOT / "figures" / "poc_lockin_local_raw.csv"
OUTPUT_AGG = PROJECT_ROOT / "figures" / "poc_lockin_local_agg.csv"
FIGURE = PROJECT_ROOT / "figures" / "poc_lockin_local_vs_convergence.png"


def lz76_trace(v_1d: np.ndarray) -> float:
    """LZ76 complexity of a 1D voltage trace."""
    if np.std(v_1d) < 1e-12:
        return 0.0
    return calculate_temporal_lz_complexity(v_1d.reshape(-1, 1))


def tracking_corr(v_pivot: np.ndarray, v_others_mean: np.ndarray) -> float:
    """Pearson correlation between pivot and mean-of-others in the tail window."""
    if np.std(v_pivot) < 1e-12 or np.std(v_others_mean) < 1e-12:
        return 0.0
    return float(np.corrcoef(v_pivot, v_others_mean)[0, 1])


def run_one(N, m, seed, D, coupling_norm, steps, tail_frac):
    import networkx as nx
    np.random.seed(seed)
    G = nx.barabasi_albert_graph(N, m, seed=seed)
    adj = nx.to_numpy_array(G)

    net = Mem4Network(adjacency_matrix=adj, heretic_ratio=HERETIC, seed=seed,
                      coupling_norm=coupling_norm, cold_start=True)
    net.model.cfg["coupling"]["D"] = D
    net.model.D_eff = D

    v_pivot_history = []
    v_others_mean_history = []
    v_network_mean_history = []
    h_trace = []

    for step in range(steps):
        # LOCAL drive: only the pivot gets the sinus. All others get 0.
        i_stim_array = np.zeros(N)
        i_stim_array[PIVOT_IDX] = A_DRIVE * np.sin(2 * np.pi * F_DRIVE * step)
        net.step(I_stimulus=i_stim_array)

        v_pivot_history.append(net.v[PIVOT_IDX])
        v_others_mean_history.append(np.mean(np.delete(net.v, PIVOT_IDX)))
        v_network_mean_history.append(np.mean(net.v))
        h_trace.append(calculate_continuous_entropy(net.v, bins=100))

    v_pivot_hist = np.array(v_pivot_history)
    v_others_hist = np.array(v_others_mean_history)
    v_network_mean_hist = np.array(v_network_mean_history)
    h_trace = np.array(h_trace)

    tail_start = int(len(h_trace) * (1 - tail_frac))
    h_tail_mean = float(np.mean(h_trace[tail_start:]))
    lz_pivot_tail = lz76_trace(v_pivot_hist[tail_start:])
    lz_others_tail = lz76_trace(v_others_hist[tail_start:])

    tracking = tracking_corr(v_pivot_hist[tail_start:], v_others_hist[tail_start:])
    tracking_err = float(np.sqrt(np.mean(
        (v_pivot_hist[tail_start:] - v_others_hist[tail_start:]) ** 2
    )))

    # Frequency analysis: zero-crossing rate of network mean in tail
    tail = v_network_mean_hist[tail_start:]
    if np.std(tail) > 1e-12:
        zero_crossings = np.sum(np.diff(np.sign(tail - np.mean(tail))) != 0)
        steps_in_tail = len(tail)
        dominant_freq_cycles_per_step = zero_crossings / (2.0 * steps_in_tail)
    else:
        dominant_freq_cycles_per_step = 0.0
    freq_match = abs(dominant_freq_cycles_per_step - F_DRIVE) < (F_DRIVE * 0.5)

    u_mean = float(net.model.u.mean())

    return {
        "N": N, "m": m, "seed": seed, "D": D,
        "H_cont": h_tail_mean,
        "LZ_pivot": lz_pivot_tail,
        "LZ_others": lz_others_tail,
        "tracking_corr": tracking,
        "tracking_err": tracking_err,
        "dom_freq": dominant_freq_cycles_per_step,
        "freq_match_drive": bool(freq_match),
        "u_mean": u_mean,
    }


def run_batch(N, m, D, seeds, coupling_norm, steps, tail_frac):
    results = []
    for s in seeds:
        try:
            r = run_one(N, m, s, D, coupling_norm, steps, tail_frac)
            results.append(r)
        except Exception as e:
            import traceback
            print(f"  ERROR for (m={m}, seed={s}, D={D}): {e}")
            traceback.print_exc()
            results.append({"N": N, "m": m, "seed": s, "D": D,
                            "H_cont": np.nan, "LZ_pivot": np.nan, "LZ_others": np.nan,
                            "tracking_corr": np.nan, "tracking_err": np.nan,
                            "dom_freq": np.nan, "freq_match_drive": np.nan, "u_mean": np.nan})
    return results


def main():
    print(f"POC Option A - local pivot drive, N={N}, M={M_VALUES}, D={D_VALUES}, {len(SEEDS)} seeds")
    print(f"Drive: A={A_DRIVE}, f={F_DRIVE} (period={1/F_DRIVE:.0f} steps), steps={STEPS}")
    t0 = time.time()
    rows = []
    for D in D_VALUES:
        for m in M_VALUES:
            print(f"  D={D}, m={m} ...", end=" ", flush=True)
            batch = run_batch(N, m, D, SEEDS, COUPLING_NORM, STEPS, TAIL_FRAC)
            rows.extend(batch)
            print(f"done ({len(batch)} runs, {time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_RAW, index=False)
    print(f"\nRaw CSV: {OUTPUT_RAW}  ({len(df)} rows)")

    agg = df.groupby(["N", "m", "D"]).agg(
        H_cont_mean=("H_cont", "mean"),
        H_cont_std=("H_cont", "std"),
        LZ_pivot_mean=("LZ_pivot", "mean"),
        LZ_pivot_std=("LZ_pivot", "std"),
        LZ_others_mean=("LZ_others", "mean"),
        LZ_others_std=("LZ_others", "std"),
        track_corr_mean=("tracking_corr", "mean"),
        track_corr_std=("tracking_corr", "std"),
        track_err_mean=("tracking_err", "mean"),
        track_err_std=("tracking_err", "std"),
        dom_freq_mean=("dom_freq", "mean"),
        freq_match_frac=("freq_match_drive", "mean"),
        u_mean_mean=("u_mean", "mean"),
        count=("seed", "count"),
    ).reset_index()
    agg.to_csv(OUTPUT_AGG, index=False)
    print(f"Agg CSV: {OUTPUT_AGG}")

    # Verdict
    print("\n" + "=" * 70)
    print(f"VERDICT POC (LOCAL drive, f_drive={F_DRIVE} cycles/step)")
    print("=" * 70)
    for m in M_VALUES:
        for D in D_VALUES:
            sub = agg[(agg["m"] == m) & (agg["D"] == D)]
            if len(sub) == 0:
                continue
            tc = sub["track_corr_mean"].values[0]
            te = sub["track_err_mean"].values[0]
            lz = sub["LZ_others_mean"].values[0]
            df_ = sub["dom_freq_mean"].values[0]
            fm = sub["freq_match_frac"].values[0]
            H = sub["H_cont_mean"].values[0]
            if tc > 0.5 and fm > 0.5:
                verdict = "CONVERGENCE (tracks drive)"
            elif tc < 0.2 and fm < 0.3:
                verdict = "LOCK-IN (own dynamics)"
            elif lz < 0.85:
                verdict = "STRUCTURED (undecided)"
            else:
                verdict = "INCOHERENT"
            print(f"  m={m}  D={D}  track_corr={tc:+.3f}  track_err={te:.3f}  "
                  f"LZ_others={lz:.3f}  dom_freq={df_:.4f}  freq_match={fm:.2f}  "
                  f"H={H:.2f}  => {verdict}")

    # Figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.patch.set_facecolor("#1e1e2e")
    colors_D = {0.0: "#4fc3f7", 0.15: "#ff8a65", 0.5: "#b5ff40"}
    markers_m = {3: "o", 6: "s"}

    # Top-left: LZ_others vs D
    ax = axes[0, 0]
    for m in M_VALUES:
        for D in D_VALUES:
            sub = agg[(agg["m"] == m) & (agg["D"] == D)]
            if len(sub) == 0:
                continue
            ax.errorbar(D, sub["LZ_others_mean"].values[0],
                        yerr=sub["LZ_others_std"].values[0],
                        color=colors_D[D], marker=markers_m[m], ms=12, lw=2,
                        label=f"m={m}, D={D}", capsize=3)
    ax.set_xlabel("D", color="white")
    ax.set_ylabel("LZ76 (mean of others)", color="white")
    ax.set_title("Temporal structure vs coupling", color="white")
    ax.set_facecolor("#252535")
    ax.grid(True, alpha=0.3)
    ax.tick_params(colors="white")
    ax.axhline(0.85, color="#ffcc00", ls="--", alpha=0.5, label="LZ=0.85 (regime boundary)")

    # Top-right: tracking_corr vs D
    ax = axes[0, 1]
    for m in M_VALUES:
        for D in D_VALUES:
            sub = agg[(agg["m"] == m) & (agg["D"] == D)]
            if len(sub) == 0:
                continue
            ax.errorbar(D, sub["track_corr_mean"].values[0],
                        yerr=sub["track_corr_std"].values[0],
                        color=colors_D[D], marker=markers_m[m], ms=12, lw=2,
                        label=f"m={m}, D={D}", capsize=3)
    ax.set_xlabel("D", color="white")
    ax.set_ylabel("Tracking correlation (pivot vs mean others)", color="white")
    ax.set_title("Does the network follow the pivot?", color="white")
    ax.set_facecolor("#252535")
    ax.grid(True, alpha=0.3)
    ax.tick_params(colors="white")
    ax.axhline(0.5, color="#b5ff40", ls="--", alpha=0.5, label="tracking threshold")
    ax.set_ylim(-0.1, 1.05)

    # Bottom-left: tracking_err vs D
    ax = axes[1, 0]
    for m in M_VALUES:
        for D in D_VALUES:
            sub = agg[(agg["m"] == m) & (agg["D"] == D)]
            if len(sub) == 0:
                continue
            ax.errorbar(D, sub["track_err_mean"].values[0],
                        yerr=sub["track_err_std"].values[0],
                        color=colors_D[D], marker=markers_m[m], ms=12, lw=2,
                        label=f"m={m}, D={D}", capsize=3)
    ax.set_xlabel("D", color="white")
    ax.set_ylabel("Tracking error (RMSE)", color="white")
    ax.set_title("Tracking error (lower = better)", color="white")
    ax.set_facecolor("#252535")
    ax.grid(True, alpha=0.3)
    ax.tick_params(colors="white")

    # Bottom-right: scatter LZ vs tracking_corr
    ax = axes[1, 1]
    for m in M_VALUES:
        for D in D_VALUES:
            sub = agg[(agg["m"] == m) & (agg["D"] == D)]
            if len(sub) == 0:
                continue
            ax.scatter(sub["LZ_others_mean"].values[0],
                       sub["track_corr_mean"].values[0],
                       color=colors_D[D], marker=markers_m[m], s=200,
                       label=f"m={m}, D={D}")
    ax.set_xlabel("LZ76 (mean of others)", color="white")
    ax.set_ylabel("Tracking correlation", color="white")
    ax.set_title("LOCK-IN DIAGNOSTIC\n(low LZ + low tracking = lock-in / low LZ + high tracking = convergence)",
                 color="white")
    ax.set_facecolor("#252535")
    ax.grid(True, alpha=0.3)
    ax.tick_params(colors="white")
    ax.axhline(0.5, color="#b5ff40", ls="--", alpha=0.3)
    ax.axvline(0.85, color="#ffcc00", ls="--", alpha=0.3)
    ax.annotate("LOCK-IN ZONE", xy=(0.5, 0.0), xytext=(0.4, -0.05),
                color="#ff5252", fontsize=10, ha="center",
                arrowprops=dict(arrowstyle="->", color="#ff5252"))
    ax.annotate("CONVERGENCE ZONE", xy=(0.5, 0.9), xytext=(0.4, 0.95),
                color="#b5ff40", fontsize=10, ha="center",
                arrowprops=dict(arrowstyle="->", color="#b5ff40"))

    plt.suptitle("POC Option A - Local pivot drive: does the network follow the signal?",
                 color="white", fontsize=14, y=1.00)
    plt.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURE, dpi=150, bbox_inches="tight")
    print(f"Figure: {FIGURE}")
    print(f"\nTotal wall time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
