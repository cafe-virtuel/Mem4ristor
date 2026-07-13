"""Hermes smoke tests 2026-06-12 - regard neuf de chercheur.

Reproduction rapide (5-15 min) des 4 vagues de tests executes par Hermes
(posture "chercheur qui decouvre" sur Mem4ristor v6.0.0).

Usage :
    /c/Users/julch/AppData/Local/Programs/Python/Python313/python.exe \\
        experiments/hermes_smoke_20260612/hermes_smoke.py

Resultats (12/06/2026) :
    - C01 PASS (4.00 vs 4.06 bits, tolerance 0.08)
    - C04 PASS (ratio 16-30x avec tail=200, ordre de grandeur tenu)
    - "Dead zone" H_cog=0 = ARTEFACT METRIQUE (H_stable=3.5+ sur BA m=3-5)
    - dt=0.10 fait +0.85 bits vs dt=0.05 (H_stable pas stable au-dela de 0.05)
    - Range des bins de H_stable ([-3,3] default) tronque les queues -> gonfle H
    - AC@lag50 = 0.68 reel vs 0.01 shuffle (claim [20] TIENT)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import numpy as np
import networkx as nx
from mem4ristor.core import Mem4Network
from mem4ristor.metrics import calculate_continuous_entropy, calculate_cognitive_entropy
from scipy.stats import ttest_1samp

SEEDS = [42, 123, 777, 17, 256, 1337, 99, 314, 2024, 888]
N_STEPS = 3000


def banner(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def test1_c01():
    banner("TEST 1 : H_stable lattice 10x10 (C01 = 4.06 +/- 0.08)")
    rs = []
    for seed in SEEDS:
        net = Mem4Network(size=10, heretic_ratio=0.15, seed=seed,
                          cold_start=True, boundary="periodic")
        for _ in range(N_STEPS):
            net.step(I_stimulus=0.5)
        rs.append(net.calculate_entropy())
    rs = np.array(rs)
    print(f"  Result: {rs.mean():.4f} +/- {rs.std()/np.sqrt(10):.4f} bits  (delta={abs(rs.mean()-4.06):.4f})")
    return rs


def test2_c04(tail=200):
    banner(f"TEST 2 : sync FULL vs FROZEN (C04 ~90x, tail={tail})")
    sync_full, sync_frozen = [], []
    for seed in SEEDS[:8]:
        for mode in ["FULL", "FROZEN"]:
            net = Mem4Network(size=10, heretic_ratio=0.15, seed=seed,
                              cold_start=True, boundary="periodic")
            if mode == "FROZEN":
                net.model.cfg["doubt"]["epsilon_u"] = 0.0
            for _ in range(N_STEPS):
                net.step(I_stimulus=0.5)
            traj = np.zeros((tail, 100))
            for k in range(tail):
                net.step(I_stimulus=0.5)
                traj[k] = net.v
            corrs = []
            for i in range(100):
                for j in range(i + 1, 100):
                    corrs.append(np.corrcoef(traj[:, i], traj[:, j])[0, 1])
            (sync_full if mode == "FULL" else sync_frozen).append(np.mean(corrs))
    full = np.mean(sync_full)
    froz = np.mean(sync_frozen)
    print(f"  FULL={full:.4f}  FROZEN={froz:.4f}  ratio={froz/max(full, 1e-9):.1f}x")


def test3_dt_sensitivity():
    banner("TEST 3 : dt-sensitivity H_stable (sensibilite au pas)")
    for dt_test in [0.01, 0.03, 0.05, 0.08, 0.10]:
        rs = []
        for seed in SEEDS[:5]:
            net = Mem4Network(size=10, heretic_ratio=0.15, seed=seed,
                              cold_start=True, boundary="periodic")
            net.model.cfg["dynamics"]["dt"] = dt_test
            steps = int(N_STEPS * 0.05 / dt_test)
            for _ in range(steps):
                net.step(I_stimulus=0.5)
            rs.append(net.calculate_entropy())
        rs = np.array(rs)
        print(f"  dt={dt_test:.2f}: H={rs.mean():.3f} +/- {rs.std()/np.sqrt(5):.3f} bits")


def test4_i_stim_sweep():
    banner("TEST 4 : H_stable vs I_stim (la frontiere du claim)")
    for I_test in [0.0, 0.1, 0.25, 0.5, 1.0]:
        rs = []
        for seed in SEEDS[:8]:
            net = Mem4Network(size=10, heretic_ratio=0.15, seed=seed,
                              cold_start=True, boundary="periodic")
            for _ in range(N_STEPS):
                net.step(I_stimulus=I_test)
            rs.append(net.calculate_entropy())
        rs = np.array(rs)
        print(f"  I_stim={I_test:.2f}: H={rs.mean():.3f} +/- {rs.std()/np.sqrt(8):.3f} bits")


def test5_full_vs_frozen_at_i0():
    banner("TEST 5 : FULL vs FROZEN a I=0 (la diversite survit-elle ?)")
    for I_test in [0.0, 0.5]:
        for mode in ["FULL", "FROZEN"]:
            rs, us = [], []
            for seed in SEEDS:
                net = Mem4Network(size=10, heretic_ratio=0.15, seed=seed,
                                  cold_start=True, boundary="periodic")
                if mode == "FROZEN":
                    net.model.cfg["doubt"]["epsilon_u"] = 0.0
                for _ in range(N_STEPS):
                    net.step(I_stimulus=I_test)
                rs.append(net.calculate_entropy())
                us.append(net.model.u.mean())
            rs, us = np.array(rs), np.array(us)
            print(f"  I={I_test} {mode:6s}: H={rs.mean():.3f} bits  u_mean={us.mean():.3f}")


def test6_ac_lag50():
    banner("TEST 6 : AC@lag50 (claim [20]) - test statistique")
    for D_test in [0.0, 0.15]:
        acs_real, acs_shuf = [], []
        for seed in SEEDS:
            net = Mem4Network(size=10, heretic_ratio=0.15, seed=seed,
                              cold_start=True, boundary="periodic")
            net.model.cfg["coupling"]["D"] = D_test
            net.model.D_eff = D_test / np.sqrt(net.N)
            traj = np.zeros(N_STEPS)
            for k in range(N_STEPS):
                net.step(I_stimulus=0.0)
                traj[k] = net.v.mean()
            traj = traj[N_STEPS // 2:]
            traj = traj - np.linspace(traj[0], traj[-1], len(traj))
            acs_real.append(np.corrcoef(traj[:-50], traj[50:])[0, 1])
            shuf = traj.copy()
            np.random.shuffle(shuf)
            acs_shuf.append(np.corrcoef(shuf[:-50], shuf[50:])[0, 1])
        ar = np.array(acs_real); ash = np.array(acs_shuf)
        t, p = ttest_1samp(ar, 0.5)
        print(f"  D={D_test}: real={ar.mean():.3f}+/-{ar.std()/np.sqrt(10):.3f} "
              f"(t={t:.2f}, p={p:.4f}) | shuffle={ash.mean():.3f}")


def test7_dead_zone_metric():
    banner("TEST 7 : 'Dead zone' BA m=5 - artefact H_cog ?")
    for m in [1, 3, 5, 7, 10]:
        Hs, Hc = [], []
        for seed in SEEDS:
            G = nx.barabasi_albert_graph(100, m, seed=seed)
            net = Mem4Network(adjacency_matrix=nx.to_numpy_array(G),
                              heretic_ratio=0.15, seed=seed)
            for _ in range(N_STEPS):
                net.step(I_stimulus=0.5)
            Hs.append(calculate_continuous_entropy(net.v, bins=100))
            Hc.append(calculate_cognitive_entropy(net.v))
        print(f"  m={m:2d}: H_stable={np.mean(Hs):.3f}  H_cog={np.mean(Hc):.3f}  "
              f"(ratio H_stable/H_cog={np.mean(Hs)/max(np.mean(Hc), 1e-3):.1f}x)")


def test8_h_range_bins():
    banner("TEST 8 : sensibilite de H_stable au range des bins (default [-3,3])")
    for rng in [(-3, 3), (-5, 5), (-10, 10)]:
        rs = []
        for seed in SEEDS:
            G = nx.barabasi_albert_graph(100, 3, seed=seed)
            net = Mem4Network(adjacency_matrix=nx.to_numpy_array(G),
                              heretic_ratio=0.15, seed=seed)
            for _ in range(N_STEPS):
                net.step(I_stimulus=0.5)
            rs.append(calculate_continuous_entropy(net.v, bins=100,
                                                    v_min=rng[0], v_max=rng[1]))
        rs = np.array(rs)
        print(f"  range={rng}: H={rs.mean():.3f} +/- {rs.std()/np.sqrt(10):.3f} bits")


if __name__ == "__main__":
    test1_c01()
    test2_c04()
    test3_dt_sensitivity()
    test4_i_stim_sweep()
    test5_full_vs_frozen_at_i0()
    test6_ac_lag50()
    test7_dead_zone_metric()
    test8_h_range_bins()
    print()
    print("=" * 70)
    print("FIN HERMES SMOKE - voir experiments/hermes_smoke_20260612/RAPPORT_HERMES.md")
    print("=" * 70)
