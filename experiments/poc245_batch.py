#!/usr/bin/env python3
"""
POCs #2, #4, #5 - Batch run.

POC #2 - LE JUGE : Deux reseaux Mem4ristor independants (seeds differents)
        recoivent le MEME drive externe global. Est-ce qu'ils convergent
        sur la meme reponse ?
        - Convergence des reponses = "langage commun" (reframe valide)
        - Divergence = chacun lock-in sur sa propre solution (reframe subjectif)

POC #4 - LE MENTEUR : Deux pivots (noeuds 0 et 1) recoivent des drives OPPOSES
        (signe inverse). Le reseau converge-t-il vers le BON signal (rejet
        du menteur) ou moyenne-t-il les deux (compromis degrade) ?
        - track_corr(pivot_0, others) >> track_corr(pivot_1, others) = debruitage
        - track_corr symetrique = compromis (lock-in sur moyenne)

POC #5 - DECORRELATION PAR BRUIT : Le drive est un BRUIT BLANC FILTRE
        (gaussien filtre passe-bas) au lieu d'un sinus coherent. Le reseau
        devient-il plus coherent (filtre passe-bas endogene) ou plus
        chaotique (amplification du bruit) ?
        - H_cont DECROIT = filtrage (reframe valide : le reseau debruite
          vers son regime propre)
        - H_cont CROIT = amplification du bruit (le reseau est passif)
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

# Shared parameters
N = 100
STEPS = 3000
TAIL_FRAC = 0.25
SEEDS_BASE = [42, 123, 777]
HERETIC = 0.0
COUPLING_NORM = "degree_linear"
M_VALUES = [3, 6]
D_VALUES = [0.0, 0.15, 0.5]

OUTPUT_RAW = PROJECT_ROOT / "figures" / "poc245_raw.csv"
OUTPUT_AGG = PROJECT_ROOT / "figures" / "poc245_agg.csv"
FIGURE = PROJECT_ROOT / "figures" / "poc245.png"


def lz76_trace(v_1d: np.ndarray) -> float:
    if np.std(v_1d) < 1e-12:
        return 0.0
    return calculate_temporal_lz_complexity(v_1d.reshape(-1, 1))


def tracking_corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def make_ba_adj(m, N, seed):
    import networkx as nx
    np.random.seed(seed)
    G = nx.barabasi_albert_graph(N, m, seed=seed)
    return nx.to_numpy_array(G)


def build_net(adj, D, seed):
    net = Mem4Network(adjacency_matrix=adj, heretic_ratio=HERETIC, seed=seed,
                      coupling_norm=coupling_norm_global(), cold_start=True)
    net.model.cfg["coupling"]["D"] = D
    net.model.D_eff = D
    return net


_coupling_norm_cache = None
def coupling_norm_global():
    global _coupling_norm_cache
    if _coupling_norm_cache is None:
        _coupling_norm_cache = COUPLING_NORM
    return _coupling_norm_cache


def tail(arr, frac=TAIL_FRAC):
    s = int(len(arr) * (1 - frac))
    return arr[s:]


# =============================================================================
# POC #2 - LE JUGE (two independent networks, same drive)
# =============================================================================
def poc2_juge():
    """Two networks, same sinusoidal drive (GLOBAL), compare their responses."""
    A = 0.8
    F = 0.05
    rows = []
    for D in D_VALUES:
        for m in M_VALUES:
            # Run two independent networks (different seeds)
            traces = []
            for seed in SEEDS_BASE[:2]:  # use 2 seeds
                adj = make_ba_adj(m, N, seed)
                net = Mem4Network(adjacency_matrix=adj, heretic_ratio=HERETIC, seed=seed,
                                  coupling_norm=COUPLING_NORM, cold_start=True)
                net.model.cfg["coupling"]["D"] = D
                net.model.D_eff = D
                v_mean_hist = []
                for step in range(STEPS):
                    drive = A * np.sin(2 * np.pi * F * step)
                    net.step(I_stimulus=drive)
                    v_mean_hist.append(np.mean(net.v))
                traces.append(np.array(tail(v_mean_hist)))
            # Compare the two traces
            cross_corr = tracking_corr(traces[0], traces[1])
            # Each trace's self-LZ
            lz_a = lz76_trace(traces[0])
            lz_b = lz76_trace(traces[1])
            rows.append({
                "poc": "POC2_juge", "m": m, "D": D, "n_seeds": 2,
                "cross_corr": cross_corr,
                "LZ_seed1": lz_a,
                "LZ_seed2": lz_b,
                "LZ_mean": (lz_a + lz_b) / 2,
            })
    return rows


# =============================================================================
# POC #4 - LE MENTEUR (two pivots, opposite drives) - 5 seeds for stats
# =============================================================================
def poc4_menteur():
    """Two pivots (idx 0 and 1) with opposite drives. 5 seeds for stats."""
    A = 0.8
    F = 0.05
    rows = []
    for D in D_VALUES:
        for m in M_VALUES:
            for seed in SEEDS_BASE:  # 5 seeds for variance estimation
                adj = make_ba_adj(m, N, seed)
                net = Mem4Network(adjacency_matrix=adj, heretic_ratio=HERETIC, seed=seed,
                                  coupling_norm=COUPLING_NORM, cold_start=True)
                net.model.cfg["coupling"]["D"] = D
                net.model.D_eff = D

                v0_hist, v1_hist, v_others_hist, h_hist = [], [], [], []
                for step in range(STEPS):
                    drive_t = A * np.sin(2 * np.pi * F * step)
                    i_stim = np.zeros(N)
                    i_stim[0] = drive_t
                    i_stim[1] = -drive_t
                    net.step(I_stimulus=i_stim)
                    v0_hist.append(net.v[0])
                    v1_hist.append(net.v[1])
                    v_others_hist.append(np.mean(net.v[2:]))
                    h_hist.append(calculate_continuous_entropy(net.v, bins=100))

                v0 = tail(np.array(v0_hist))
                v1 = tail(np.array(v1_hist))
                v_others = tail(np.array(v_others_hist))
                h_tail = tail(np.array(h_hist))

                tc0 = tracking_corr(v0, v_others)
                tc1 = tracking_corr(v1, v_others)
                H = float(np.mean(h_tail))

                rows.append({
                    "poc": "POC4_menteur", "m": m, "D": D, "seed": seed,
                    "track_corr_pivot0": tc0,
                    "track_corr_pivot1": tc1,
                    "track_corr_diff": tc0 - tc1,
                    "H_cont": H,
                })
    # Aggregate by (m, D)
    df = pd.DataFrame(rows)
    agg = df.groupby(["m", "D"]).agg(
        tc0_mean=("track_corr_pivot0", "mean"),
        tc0_std=("track_corr_pivot0", "std"),
        tc1_mean=("track_corr_pivot1", "mean"),
        tc1_std=("track_corr_pivot1", "std"),
        diff_mean=("track_corr_diff", "mean"),
        diff_std=("track_corr_diff", "std"),
        H_mean=("H_cont", "mean"),
    ).reset_index()
    out = []
    for _, r in agg.iterrows():
        diff = r["diff_mean"]
        if abs(diff) > 0.3:
            decision = "FOLLOW_TRUTH" if diff > 0 else "FOLLOW_LIAR"
        elif r["tc0_mean"] > 0.3 and r["tc1_mean"] > 0.3:
            decision = "FOLLOW_BOTH"
        else:
            decision = "FOLLOW_NEITHER"
        out.append({
            "poc": "POC4_menteur", "m": int(r["m"]), "D": r["D"], "n_seeds": 5,
            "track_corr_pivot0": r["tc0_mean"],
            "track_corr_pivot1": r["tc1_mean"],
            "track_corr_diff": diff,
            "diff_std": r["diff_std"],
            "H_cont": r["H_mean"],
            "decision": decision,
        })
    return out


# =============================================================================
# POC C - DRIVE MULTI-NOEUDS (10% of nodes driven, same sinus)
# =============================================================================
def pocC_multi_pivots():
    """10% of nodes (idx 0..9) receive the SAME sinusoidal drive.
    Does the network follow when more nodes carry the signal?"""
    A = 0.8
    F = 0.05
    N_PIVOTS = 10  # 10% of N=100
    rows = []
    for D in D_VALUES:
        for m in M_VALUES:
            for seed in SEEDS_BASE:
                adj = make_ba_adj(m, N, seed)
                # Identify the top-N_PIVOTS highest-degree nodes (hubs + neighbors)
                degrees = adj.sum(axis=1)
                # Mix: half from highest-degree, half random (avoids all-hubs case)
                np.random.seed(seed)
                sorted_idx = np.argsort(degrees)[::-1]
                hub_idx = sorted_idx[:N_PIVOTS // 2]
                random_idx = np.random.choice(N, N_PIVOTS // 2, replace=False)
                pivot_set = list(set(list(hub_idx) + list(random_idx)))[:N_PIVOTS]

                net = Mem4Network(adjacency_matrix=adj, heretic_ratio=HERETIC, seed=seed,
                                  coupling_norm=COUPLING_NORM, cold_start=True)
                net.model.cfg["coupling"]["D"] = D
                net.model.D_eff = D

                pivot_v_hist, others_hist, v_mean_hist = [], [], []
                for step in range(STEPS):
                    i_stim = np.zeros(N)
                    drive_t = A * np.sin(2 * np.pi * F * step)
                    for p in pivot_set:
                        i_stim[p] = drive_t
                    net.step(I_stimulus=i_stim)
                    pivot_v_hist.append(np.mean(net.v[pivot_set]))
                    others_hist.append(np.mean(np.delete(net.v, pivot_set)))
                    v_mean_hist.append(np.mean(net.v))

                piv_v = tail(np.array(pivot_v_hist))
                oth_v = tail(np.array(others_hist))
                v_mean = tail(np.array(v_mean_hist))
                tc = tracking_corr(piv_v, oth_v)

                # Frequency of v_mean
                zc = np.sum(np.diff(np.sign(v_mean - np.mean(v_mean))) != 0)
                f_v = zc / (2.0 * len(v_mean))
                choices = {"F_DRIVE": abs(f_v - F), "INTRINSIC": abs(f_v - 0.01)}
                winner = min(choices, key=choices.get)

                rows.append({
                    "poc": "POCC_multi", "m": m, "D": D, "seed": seed,
                    "n_pivots": N_PIVOTS,
                    "tc_pivots_vs_others": tc,
                    "f_v_mean": f_v,
                    "winner": winner,
                })
    df = pd.DataFrame(rows)
    agg = df.groupby(["m", "D"]).agg(
        tc_mean=("tc_pivots_vs_others", "mean"),
        tc_std=("tc_pivots_vs_others", "std"),
        f_mean=("f_v_mean", "mean"),
        f_std=("f_v_mean", "std"),
        winner_mode=("winner", lambda x: x.mode().iloc[0] if not x.mode().empty else "MIXED"),
    ).reset_index()
    out = []
    for _, r in agg.iterrows():
        out.append({
            "poc": "POCC_multi", "m": int(r["m"]), "D": r["D"], "n_seeds": 5,
            "n_pivots": N_PIVOTS,
            "tc": r["tc_mean"], "tc_std": r["tc_std"],
            "f_v_mean": r["f_mean"], "f_std": r["f_std"],
            "winner": r["winner_mode"],
        })
    return out


# =============================================================================
# POC D - HUB OSCILLANT (drive on the topological hub)
# =============================================================================
def pocD_hub():
    """Drive on the highest-degree node (the hub). Does centrality help propagation?"""
    A = 0.8
    F = 0.05
    rows = []
    for D in D_VALUES:
        for m in M_VALUES:
            for seed in SEEDS_BASE:
                adj = make_ba_adj(m, N, seed)
                # Hub = highest-degree node
                degrees = adj.sum(axis=1)
                hub_idx = int(np.argmax(degrees))
                hub_degree = int(degrees[hub_idx])

                net = Mem4Network(adjacency_matrix=adj, heretic_ratio=HERETIC, seed=seed,
                                  coupling_norm=COUPLING_NORM, cold_start=True)
                net.model.cfg["coupling"]["D"] = D
                net.model.D_eff = D

                v_hub_hist, v_others_hist, v_mean_hist = [], [], []
                for step in range(STEPS):
                    i_stim = np.zeros(N)
                    drive_t = A * np.sin(2 * np.pi * F * step)
                    i_stim[hub_idx] = drive_t
                    net.step(I_stimulus=i_stim)
                    v_hub_hist.append(net.v[hub_idx])
                    v_others_hist.append(np.mean(np.delete(net.v, hub_idx)))
                    v_mean_hist.append(np.mean(net.v))

                v_hub = tail(np.array(v_hub_hist))
                v_others = tail(np.array(v_others_hist))
                v_mean = tail(np.array(v_mean_hist))
                tc = tracking_corr(v_hub, v_others)

                # Frequency of v_mean
                zc = np.sum(np.diff(np.sign(v_mean - np.mean(v_mean))) != 0)
                f_v = zc / (2.0 * len(v_mean))
                choices = {"F_DRIVE": abs(f_v - F), "INTRINSIC": abs(f_v - 0.01)}
                winner = min(choices, key=choices.get)

                rows.append({
                    "poc": "POCD_hub", "m": m, "D": D, "seed": seed,
                    "hub_idx": hub_idx, "hub_degree": hub_degree,
                    "tc_hub_vs_others": tc,
                    "f_v_mean": f_v,
                    "winner": winner,
                })
    df = pd.DataFrame(rows)
    agg = df.groupby(["m", "D"]).agg(
        tc_mean=("tc_hub_vs_others", "mean"),
        tc_std=("tc_hub_vs_others", "std"),
        f_mean=("f_v_mean", "mean"),
        f_std=("f_v_mean", "std"),
        hub_degree_mean=("hub_degree", "mean"),
        winner_mode=("winner", lambda x: x.mode().iloc[0] if not x.mode().empty else "MIXED"),
    ).reset_index()
    out = []
    for _, r in agg.iterrows():
        out.append({
            "poc": "POCD_hub", "m": int(r["m"]), "D": r["D"], "n_seeds": 5,
            "hub_degree": r["hub_degree_mean"],
            "tc": r["tc_mean"], "tc_std": r["tc_std"],
            "f_v_mean": r["f_mean"], "f_std": r["f_std"],
            "winner": r["winner_mode"],
        })
    return out


# =============================================================================
# POC B - DEUX FREQUENCES CONCURRENTES (no common language because different freqs)
# =============================================================================
def pocB_two_freqs():
    """Two pivots with different frequencies f1=0.05 and f2=0.12.
    Does the network follow one, both, or its own (f~0.01)?"""
    A = 0.8
    F1, F2 = 0.05, 0.12
    rows = []
    for D in D_VALUES:
        for m in M_VALUES:
            for seed in SEEDS_BASE:  # 5 seeds
                adj = make_ba_adj(m, N, seed)
                net = Mem4Network(adjacency_matrix=adj, heretic_ratio=HERETIC, seed=seed,
                                  coupling_norm=COUPLING_NORM, cold_start=True)
                net.model.cfg["coupling"]["D"] = D
                net.model.D_eff = D

                v0_hist, v1_hist, v_others_hist, v_mean_hist = [], [], [], []
                for step in range(STEPS):
                    i_stim = np.zeros(N)
                    i_stim[0] = A * np.sin(2 * np.pi * F1 * step)  # pivot 0
                    i_stim[1] = A * np.sin(2 * np.pi * F2 * step)  # pivot 1, different freq
                    net.step(I_stimulus=i_stim)
                    v0_hist.append(net.v[0])
                    v1_hist.append(net.v[1])
                    v_others_hist.append(np.mean(net.v[2:]))
                    v_mean_hist.append(np.mean(net.v))

                v_others = tail(np.array(v_others_hist))
                v_mean = tail(np.array(v_mean_hist))
                tc0 = tracking_corr(tail(np.array(v0_hist)), v_others)
                tc1 = tracking_corr(tail(np.array(v1_hist)), v_others)

                # Check if v_mean contains either f1 or f2 or the intrinsic f
                # Simple: zero-crossing rate of v_mean
                zc = np.sum(np.diff(np.sign(v_mean - np.mean(v_mean))) != 0)
                f_v_mean = zc / (2.0 * len(v_mean))

                # Which is closest?
                dist_to_f1 = abs(f_v_mean - F1)
                dist_to_f2 = abs(f_v_mean - F2)
                dist_to_intrinsic = abs(f_v_mean - 0.01)  # POC #1 finding
                choices = {"F1": dist_to_f1, "F2": dist_to_f2, "INTRINSIC": dist_to_intrinsic}
                winner = min(choices, key=choices.get)

                rows.append({
                    "poc": "POCB_2freqs", "m": m, "D": D, "seed": seed,
                    "tc0": tc0, "tc1": tc1,
                    "f_v_mean": f_v_mean,
                    "winner": winner,
                })
    df = pd.DataFrame(rows)
    agg = df.groupby(["m", "D"]).agg(
        tc0_mean=("tc0", "mean"),
        tc0_std=("tc0", "std"),
        tc1_mean=("tc1", "mean"),
        tc1_std=("tc1", "std"),
        f_mean=("f_v_mean", "mean"),
        f_std=("f_v_mean", "std"),
        winner_mode=("winner", lambda x: x.mode().iloc[0] if not x.mode().empty else "MIXED"),
    ).reset_index()
    out = []
    for _, r in agg.iterrows():
        out.append({
            "poc": "POCB_2freqs", "m": int(r["m"]), "D": r["D"], "n_seeds": 5,
            "tc0": r["tc0_mean"], "tc0_std": r["tc0_std"],
            "tc1": r["tc1_mean"], "tc1_std": r["tc1_std"],
            "f_v_mean": r["f_mean"], "f_std": r["f_std"],
            "winner": r["winner_mode"],
        })
    return out


# =============================================================================
# POC #5 - DECORRELATION PAR BRUIT (white noise filtered)
# =============================================================================
def poc5_bruit():
    """Filtered white noise as drive. Does the network denoise?"""
    rows = []
    SIGMA_NOISE = 0.8  # comparable amplitude to sinus POCs
    for D in D_VALUES:
        for m in M_VALUES:
            seed = SEEDS_BASE[0]
            adj = make_ba_adj(m, N, seed)
            net = Mem4Network(adjacency_matrix=adj, heretic_ratio=HERETIC, seed=seed,
                              coupling_norm=COUPLING_NORM, cold_start=True)
            net.model.cfg["coupling"]["D"] = D
            net.model.D_eff = D

            # Generate filtered noise: low-pass via cumulative sum
            rng = np.random.RandomState(seed)
            white = rng.normal(0, SIGMA_NOISE, STEPS)
            # Low-pass: cumulative sum gives brownian noise (1/f^2 spectrum)
            brownian = np.cumsum(white) * 0.1  # scale down
            brownian = brownian - np.mean(brownian)  # zero-mean

            v_mean_hist, h_hist = [], []
            for step in range(STEPS):
                # Inject the filtered noise as drive on ALL nodes (so it's a real
                # exogenous perturbation, not a local pivot). Mean of the drive is
                # ~0 so we don't trivially bias the mean.
                net.step(I_stimulus=brownian[step])
                v_mean_hist.append(np.mean(net.v))
                h_hist.append(calculate_continuous_entropy(net.v, bins=100))

            v_mean_tail = tail(np.array(v_mean_hist))
            h_tail = tail(np.array(h_hist))
            H = float(np.mean(h_tail))
            H_std = float(np.std(h_tail))
            LZ = lz76_trace(v_mean_tail)
            # How much does the network's variance RESEMBLE the drive?
            drive_tail = tail(brownian)
            cross_corr_drive = tracking_corr(v_mean_tail, drive_tail)
            # Autocorrelation at small lag (if denoising, AC should be high - smooth)
            ac_lag5 = float(np.corrcoef(v_mean_tail[:-5], v_mean_tail[5:])[0, 1]) if len(v_mean_tail) > 5 else 0.0
            ac_lag50 = float(np.corrcoef(v_mean_tail[:-50], v_mean_tail[50:])[0, 1]) if len(v_mean_tail) > 50 else 0.0

            rows.append({
                "poc": "POC5_bruit", "m": m, "D": D, "n_seeds": 1,
                "H_cont": H,
                "H_std": H_std,
                "LZ_v_mean": LZ,
                "cross_corr_drive": cross_corr_drive,
                "ac_lag5": ac_lag5,
                "ac_lag50": ac_lag50,
            })
    return rows


def main():
    t0 = time.time()
    all_rows = []
    print("POC #2 - LE JUGE ...")
    all_rows.extend(poc2_juge())
    print(f"  done ({time.time()-t0:.0f}s)")
    print("POC #4 - LE MENTEUR (5 seeds) ...")
    all_rows.extend(poc4_menteur())
    print(f"  done ({time.time()-t0:.0f}s)")
    print("POC B - DEUX FREQUENCES CONCURRENTES (5 seeds) ...")
    all_rows.extend(pocB_two_freqs())
    print(f"  done ({time.time()-t0:.0f}s)")
    print("POC C - DRIVE MULTI-NOEUDS (10 pivots, 5 seeds) ...")
    all_rows.extend(pocC_multi_pivots())
    print(f"  done ({time.time()-t0:.0f}s)")
    print("POC D - HUB OSCILLANT (drive on highest-degree node, 5 seeds) ...")
    all_rows.extend(pocD_hub())
    print(f"  done ({time.time()-t0:.0f}s)")
    print("POC #5 - DECORRELATION PAR BRUIT ...")
    all_rows.extend(poc5_bruit())
    print(f"  done ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_RAW, index=False)
    print(f"\nRaw CSV: {OUTPUT_RAW}  ({len(df)} rows)")

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT POC #2 - LE JUGE (common language between two networks)")
    print("=" * 70)
    p2 = df[df["poc"] == "POC2_juge"]
    for _, row in p2.iterrows():
        if row["cross_corr"] > 0.5:
            verdict = "COMMON LANGUAGE (reframe: collective coherence)"
        elif row["cross_corr"] > 0.2:
            verdict = "PARTIAL agreement"
        else:
            verdict = "NO common language (each locks in on its own solution)"
        print(f"  m={int(row['m'])}  D={row['D']}  cross_corr={row['cross_corr']:+.3f}  "
              f"LZ_mean={row['LZ_mean']:.3f}  => {verdict}")

    print("\n" + "=" * 70)
    print("VERDICT POC #4 - LE MENTEUR (rejection of contradictory drive)")
    print("=" * 70)
    p4 = df[df["poc"] == "POC4_menteur"]
    for _, row in p4.iterrows():
        print(f"  m={int(row['m'])}  D={row['D']}  tc_pivot0={row['track_corr_pivot0']:+.3f}  "
              f"tc_pivot1={row['track_corr_pivot1']:+.3f}  diff={row['track_corr_diff']:+.3f}  "
              f"H={row['H_cont']:.2f}  => {row['decision']}")

    print("\n" + "=" * 70)
    print("VERDICT POC B - DEUX FREQUENCES CONCURRENTES (F1=0.05 vs F2=0.12 vs intrinsic~0.01)")
    print("=" * 70)
    pB = df[df["poc"] == "POCB_2freqs"]
    for _, row in pB.iterrows():
        print(f"  m={int(row['m'])}  D={row['D']}  tc_F1={row['tc0']:+.3f}+/-{row['tc0_std']:.3f}  "
              f"tc_F2={row['tc1']:+.3f}+/-{row['tc1_std']:.3f}  "
              f"f_v_mean={row['f_v_mean']:.4f}+/-{row['f_std']:.4f}  "
              f"=> winner: {row['winner']}")

    print("\n" + "=" * 70)
    print("VERDICT POC C - DRIVE MULTI-NOEUDS (10 pivots carrying same drive)")
    print("=" * 70)
    pC = df[df["poc"] == "POCC_multi"]
    for _, row in pC.iterrows():
        print(f"  m={int(row['m'])}  D={row['D']}  n_pivots={int(row['n_pivots'])}  "
              f"tc={row['tc']:+.3f}+/-{row['tc_std']:.3f}  "
              f"f_v_mean={row['f_v_mean']:.4f}+/-{row['f_std']:.4f}  "
              f"=> winner: {row['winner']}")

    print("\n" + "=" * 70)
    print("VERDICT POC D - HUB OSCILLANT (drive on highest-degree node)")
    print("=" * 70)
    pD = df[df["poc"] == "POCD_hub"]
    for _, row in pD.iterrows():
        print(f"  m={int(row['m'])}  D={row['D']}  hub_degree={row['hub_degree']:.0f}  "
              f"tc={row['tc']:+.3f}+/-{row['tc_std']:.3f}  "
              f"f_v_mean={row['f_v_mean']:.4f}+/-{row['f_std']:.4f}  "
              f"=> winner: {row['winner']}")

    print("\n" + "=" * 70)
    print("VERDICT POC #5 - DECORRELATION PAR BRUIT (denoising test)")
    print("=" * 70)
    p5 = df[df["poc"] == "POC5_bruit"]
    for _, row in p5.iterrows():
        if row["H_cont"] < 1.5 and row["ac_lag50"] > 0.5:
            verdict = "DENOISING (low H, high AC = smoothing)"
        elif row["H_cont"] > 3.0:
            verdict = "AMPLIFIES NOISE (high H)"
        else:
            verdict = "PASSIVE (no denoising, no amplification)"
        print(f"  m={int(row['m'])}  D={row['D']}  H={row['H_cont']:.2f}+/-{row['H_std']:.2f}  "
              f"LZ={row['LZ_v_mean']:.3f}  cross_corr_drive={row['cross_corr_drive']:+.3f}  "
              f"ac50={row['ac_lag50']:+.3f}  => {verdict}")

    # Figure: 3 panels, one per POC
    fig, axes = plt.subplots(3, 2, figsize=(13, 14))
    fig.patch.set_facecolor("#1e1e2e")
    colors_D = {0.0: "#4fc3f7", 0.15: "#ff8a65", 0.5: "#b5ff40"}
    markers_m = {3: "o", 6: "s"}

    # POC #2 - cross_corr and LZ_mean
    ax = axes[0, 0]
    for m in M_VALUES:
        for D in D_VALUES:
            sub = p2[(p2["m"] == m) & (p2["D"] == D)]
            if len(sub) == 0:
                continue
            ax.bar(f"m{m}D{D}", sub["cross_corr"].values[0],
                   color=colors_D[D])
    ax.set_ylabel("Cross-correlation between 2 networks", color="white")
    ax.set_title("POC #2 - LE JUGE: common language?", color="white")
    ax.set_facecolor("#252535")
    ax.tick_params(colors="white", rotation=45)
    ax.axhline(0.5, color="#b5ff40", ls="--", alpha=0.5)

    ax = axes[0, 1]
    for m in M_VALUES:
        for D in D_VALUES:
            sub = p2[(p2["m"] == m) & (p2["D"] == D)]
            if len(sub) == 0:
                continue
            ax.bar(f"m{m}D{D}", sub["LZ_mean"].values[0],
                   color=colors_D[D])
    ax.set_ylabel("LZ76 (mean of 2 networks)", color="white")
    ax.set_title("POC #2 - LZ of the responses", color="white")
    ax.set_facecolor("#252535")
    ax.tick_params(colors="white", rotation=45)

    # POC #4 - track_corr pivot0 vs pivot1
    ax = axes[1, 0]
    x = np.arange(len(p4))
    w = 0.35
    ax.bar(x - w/2, p4["track_corr_pivot0"].values, w, label="pivot 0 (truth)", color="#b5ff40")
    ax.bar(x + w/2, p4["track_corr_pivot1"].values, w, label="pivot 1 (liar)", color="#ff5252")
    ax.set_xticks(x)
    ax.set_xticklabels([f"m{int(r['m'])}D{r['D']}" for _, r in p4.iterrows()], rotation=45, color="white")
    ax.set_ylabel("Tracking correlation (pivot vs others)", color="white")
    ax.set_title("POC #4 - LE MENTEUR: who wins?", color="white")
    ax.legend()
    ax.set_facecolor("#252535")
    ax.tick_params(colors="white")
    ax.axhline(0.5, color="white", ls="--", alpha=0.3)

    ax = axes[1, 1]
    for m in M_VALUES:
        for D in D_VALUES:
            sub = p4[(p4["m"] == m) & (p4["D"] == D)]
            if len(sub) == 0:
                continue
            ax.bar(f"m{m}D{D}", sub["track_corr_diff"].values[0],
                   color=["#b5ff40" if v > 0 else "#ff5252"
                          for v in [sub["track_corr_diff"].values[0]]])
    ax.set_ylabel("tc_pivot0 - tc_pivot1 (positive = truth wins)", color="white")
    ax.set_title("POC #4 - Asymmetry (truth vs liar)", color="white")
    ax.set_facecolor("#252535")
    ax.tick_params(colors="white", rotation=45)
    ax.axhline(0, color="white", ls="-", alpha=0.5)

    # POC #5 - H and AC
    ax = axes[2, 0]
    for m in M_VALUES:
        for D in D_VALUES:
            sub = p5[(p5["m"] == m) & (p5["D"] == D)]
            if len(sub) == 0:
                continue
            ax.bar(f"m{m}D{D}", sub["H_cont"].values[0],
                   color=colors_D[D])
    ax.set_ylabel("H_cont (bits, under filtered noise)", color="white")
    ax.set_title("POC #5 - DECORRELATION: H under noise", color="white")
    ax.set_facecolor("#252535")
    ax.tick_params(colors="white", rotation=45)

    ax = axes[2, 1]
    for m in M_VALUES:
        for D in D_VALUES:
            sub = p5[(p5["m"] == m) & (p5["D"] == D)]
            if len(sub) == 0:
                continue
            ax.bar(f"m{m}D{D}", sub["ac_lag50"].values[0],
                   color=colors_D[D])
    ax.set_ylabel("Autocorrelation at lag=50", color="white")
    ax.set_title("POC #5 - Smoothing (high AC = low-pass filter)", color="white")
    ax.set_facecolor("#252535")
    ax.tick_params(colors="white", rotation=45)
    ax.axhline(0.5, color="#b5ff40", ls="--", alpha=0.3)

    plt.suptitle("POCs #2 #4 #5 - Three adversarial tests of the reframe",
                 color="white", fontsize=14, y=1.00)
    plt.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURE, dpi=150, bbox_inches="tight")
    print(f"\nFigure: {FIGURE}")
    print(f"Total wall time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
