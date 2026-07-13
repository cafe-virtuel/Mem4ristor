#!/usr/bin/env python3
"""
POC #1 - Test of absence: does the network have its OWN rhythm?

POC #1 INVERSION RADICALE :
Au lieu de demander "est-ce que le reseau suit un drive externe ?",
on coupe TOUT drive et on observe ce que fait le reseau TOUT SEUL.

HYPOTHESE (implicite au reframe Julien) :
Si la zone dense m>=6 est une "phase de resolution" (convergence vers
la solution), alors le reseau devrait avoir un COMPORTEMENT COHERENT
SPONTANE (oscillations propres, structure temporelle, etc.) meme
sans aucune entree.

CE QU'ON ATTEND SI LA ZONE DENSE EST "STRUCTUREE" :
- A m=6, D=0.50*u : oscillations regulieres (LZ < 0.85)
- Frequence propre stable d'un seed a l'autre
- H_cont moderee (ni zero ni chaos)

CE QU'ON ATTEND SI LA ZONE DENSE EST "NEUTRE" :
- Oscillations erratiques (LZ ~ 1.1 comme D=0)
- Frequence propre non-reproductible
- H_cont elevee (chaos)

DESIGN :
- N=100, steps=5000 (long run pour stabiliser)
- I_stimulus = 0 partout, TOUT LE TEMPS
- 3 protocoles x 2 topologies x 5 seeds = 30 runs
- Metriques : H_cont, LZ_others, dom_freq, auto-correlation
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
STEPS = 5000
TAIL_FRAC = 0.20  # metric window = last 20% (2500 steps = ~125 cycles at f=0.05)
SEEDS = [42, 123, 777, 17, 256]
HERETIC = 0.0
COUPLING_NORM = "degree_linear"

# Test cells
M_VALUES = [3, 6]
D_VALUES = [0.0, 0.15, 0.5]

OUTPUT_RAW = PROJECT_ROOT / "figures" / "poc1_absence_raw.csv"
OUTPUT_AGG = PROJECT_ROOT / "figures" / "poc1_absence_agg.csv"
FIGURE = PROJECT_ROOT / "figures" / "poc1_absence.png"


def lz76_trace(v_1d: np.ndarray) -> float:
    if np.std(v_1d) < 1e-12:
        return 0.0
    return calculate_temporal_lz_complexity(v_1d.reshape(-1, 1))


def autocorrelation_at_lag(x: np.ndarray, lag: int) -> float:
    """Normalized autocorrelation at a given lag. Range: [-1, 1]."""
    if len(x) <= lag:
        return 0.0
    if np.std(x) < 1e-12:
        return 0.0
    x0 = x[:-lag] - np.mean(x[:-lag])
    x1 = x[lag:] - np.mean(x[lag:])
    num = np.mean(x0 * x1)
    den = np.std(x[:-lag]) * np.std(x[lag:])
    if den < 1e-12:
        return 0.0
    return float(num / den)


def dominant_freq(x: np.ndarray) -> float:
    """Zero-crossing rate as rough freq estimator (cycles per step)."""
    if np.std(x) < 1e-12:
        return 0.0
    zc = np.sum(np.diff(np.sign(x - np.mean(x))) != 0)
    return float(zc / (2.0 * len(x)))


def run_one(N, m, seed, D, coupling_norm, steps, tail_frac):
    import networkx as nx
    np.random.seed(seed)
    G = nx.barabasi_albert_graph(N, m, seed=seed)
    adj = nx.to_numpy_array(G)

    net = Mem4Network(adjacency_matrix=adj, heretic_ratio=HERETIC, seed=seed,
                      coupling_norm=coupling_norm, cold_start=True)
    net.model.cfg["coupling"]["D"] = D
    net.model.D_eff = D

    v_mean_history = []
    h_trace = []

    for step in range(steps):
        # NO DRIVE. I_stimulus = 0 everywhere, always.
        net.step(I_stimulus=0.0)
        v_mean_history.append(np.mean(net.v))
        h_trace.append(calculate_continuous_entropy(net.v, bins=100))

    v_mean_hist = np.array(v_mean_history)
    h_trace = np.array(h_trace)

    tail_start = int(len(h_trace) * (1 - tail_frac))
    tail_v = v_mean_hist[tail_start:]
    tail_h = h_trace[tail_start:]

    H_mean = float(np.mean(tail_h))
    LZ_v = lz76_trace(tail_v)
    f_dom = dominant_freq(tail_v)
    # Autocorrelation at 50 steps (~ 1 expected period if f~0.02)
    ac_lag50 = autocorrelation_at_lag(tail_v, 50)
    ac_lag100 = autocorrelation_at_lag(tail_v, 100)

    u_mean = float(net.model.u.mean())

    return {
        "N": N, "m": m, "seed": seed, "D": D,
        "H_cont": H_mean,
        "LZ_v_mean": LZ_v,
        "dom_freq": f_dom,
        "ac_lag50": ac_lag50,
        "ac_lag100": ac_lag100,
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
                            "H_cont": np.nan, "LZ_v_mean": np.nan,
                            "dom_freq": np.nan, "ac_lag50": np.nan, "ac_lag100": np.nan,
                            "u_mean": np.nan})
    return results


def main():
    print(f"POC #1 - Test of absence (no drive), N={N}, M={M_VALUES}, D={D_VALUES}, {len(SEEDS)} seeds")
    print(f"STEPS={STEPS} (long run for spontaneous regime), tail_frac={TAIL_FRAC}")
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
        H_mean=("H_cont", "mean"),
        H_std=("H_cont", "std"),
        LZ_mean=("LZ_v_mean", "mean"),
        LZ_std=("LZ_v_mean", "std"),
        fdom_mean=("dom_freq", "mean"),
        fdom_std=("dom_freq", "std"),
        ac50_mean=("ac_lag50", "mean"),
        ac50_std=("ac_lag50", "std"),
        ac100_mean=("ac_lag100", "mean"),
        ac100_std=("ac_lag100", "std"),
        u_mean_mean=("u_mean", "mean"),
        count=("seed", "count"),
    ).reset_index()
    agg.to_csv(OUTPUT_AGG, index=False)
    print(f"Agg CSV: {OUTPUT_AGG}")

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT POC #1 - SPONTANEOUS RHYTHM (no drive)")
    print("=" * 70)
    for m in M_VALUES:
        for D in D_VALUES:
            sub = agg[(agg["m"] == m) & (agg["D"] == D)]
            if len(sub) == 0:
                continue
            H = sub["H_mean"].values[0]
            LZ = sub["LZ_mean"].values[0]
            f = sub["fdom_mean"].values[0]
            fstd = sub["fdom_std"].values[0]
            ac50 = sub["ac50_mean"].values[0]
            # Reproducibility: low f_std across seeds = stable intrinsic rhythm
            reproducible = fstd < 0.005
            # Regularity: high ac50 = periodic
            periodic = ac50 > 0.3
            # Structure: LZ < 0.85 = structured
            structured = LZ < 0.85
            if reproducible and periodic and structured:
                verdict = "INTRINSIC CLOCK (structured + periodic + reproducible)"
            elif structured and periodic:
                verdict = "STRUCTURED but non-reproducible rhythm"
            elif reproducible and not periodic:
                verdict = "STABLE but non-periodic (fixed-point-like)"
            else:
                verdict = "NO INTRINSIC RHYTHM (chaotic)"
            print(f"  m={m}  D={D}  H={H:.2f}  LZ={LZ:.3f}  "
                  f"f_dom={f:.4f}+/-{fstd:.4f}  ac50={ac50:+.3f}  "
                  f"=> {verdict}")

    # Figure: 4 panels
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.patch.set_facecolor("#1e1e2e")
    colors_D = {0.0: "#4fc3f7", 0.15: "#ff8a65", 0.5: "#b5ff40"}
    markers_m = {3: "o", 6: "s"}

    # Top-left: LZ
    ax = axes[0, 0]
    for m in M_VALUES:
        for D in D_VALUES:
            sub = agg[(agg["m"] == m) & (agg["D"] == D)]
            if len(sub) == 0:
                continue
            ax.errorbar(D, sub["LZ_mean"].values[0], yerr=sub["LZ_std"].values[0],
                        color=colors_D[D], marker=markers_m[m], ms=12, lw=2,
                        label=f"m={m}, D={D}", capsize=3)
    ax.set_xlabel("D", color="white")
    ax.set_ylabel("LZ76 (network mean voltage)", color="white")
    ax.set_title("Spontaneous temporal structure (no drive)", color="white")
    ax.set_facecolor("#252535")
    ax.grid(True, alpha=0.3)
    ax.tick_params(colors="white")
    ax.axhline(0.85, color="#ffcc00", ls="--", alpha=0.5)

    # Top-right: dominant frequency
    ax = axes[0, 1]
    for m in M_VALUES:
        for D in D_VALUES:
            sub = agg[(agg["m"] == m) & (agg["D"] == D)]
            if len(sub) == 0:
                continue
            ax.errorbar(D, sub["fdom_mean"].values[0], yerr=sub["fdom_std"].values[0],
                        color=colors_D[D], marker=markers_m[m], ms=12, lw=2,
                        label=f"m={m}, D={D}", capsize=3)
    ax.set_xlabel("D", color="white")
    ax.set_ylabel("Dominant frequency (cycles/step)", color="white")
    ax.set_title("Intrinsic frequency reproducibility", color="white")
    ax.set_facecolor("#252535")
    ax.grid(True, alpha=0.3)
    ax.tick_params(colors="white")

    # Bottom-left: autocorrelation at lag 50
    ax = axes[1, 0]
    for m in M_VALUES:
        for D in D_VALUES:
            sub = agg[(agg["m"] == m) & (agg["D"] == D)]
            if len(sub) == 0:
                continue
            ax.errorbar(D, sub["ac50_mean"].values[0], yerr=sub["ac50_std"].values[0],
                        color=colors_D[D], marker=markers_m[m], ms=12, lw=2,
                        label=f"m={m}, D={D}", capsize=3)
    ax.set_xlabel("D", color="white")
    ax.set_ylabel("Autocorrelation at lag=50", color="white")
    ax.set_title("Periodicity (high = oscillating)", color="white")
    ax.set_facecolor("#252535")
    ax.grid(True, alpha=0.3)
    ax.tick_params(colors="white")
    ax.axhline(0.3, color="#b5ff40", ls="--", alpha=0.3, label="periodic threshold")

    # Bottom-right: H_cont
    ax = axes[1, 1]
    for m in M_VALUES:
        for D in D_VALUES:
            sub = agg[(agg["m"] == m) & (agg["D"] == D)]
            if len(sub) == 0:
                continue
            ax.errorbar(D, sub["H_mean"].values[0], yerr=sub["H_std"].values[0],
                        color=colors_D[D], marker=markers_m[m], ms=12, lw=2,
                        label=f"m={m}, D={D}", capsize=3)
    ax.set_xlabel("D", color="white")
    ax.set_ylabel("H_cont (bits)", color="white")
    ax.set_title("Spontaneous diversity", color="white")
    ax.set_facecolor("#252535")
    ax.grid(True, alpha=0.3)
    ax.tick_params(colors="white")

    plt.suptitle("POC #1 - Test of absence: does the network have its own rhythm?",
                 color="white", fontsize=14, y=1.00)
    plt.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURE, dpi=150, bbox_inches="tight")
    print(f"Figure: {FIGURE}")
    print(f"\nTotal wall time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
