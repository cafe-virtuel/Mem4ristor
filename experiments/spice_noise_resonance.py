#!/usr/bin/env python3
"""
SPICE noise / mismatch resonance test on the BA m=5 dead zone.

Hypothesis: the deterministic dead zone confirmed in §3nonies might be
escaped by realistic hardware imperfections:
  (a) thermal Johnson noise on each integrator (modeled with trnoise())
  (b) capacitor mismatch (CMOS Monte Carlo, +/-5% sigma)

Sweeps noise amplitude eta_v in {0.0, 0.01, 0.03, 0.10, 0.30} for the
three normalizations (uniform / degree_linear / spectral) on the SAME
BA m=5 N=64 graph used in spice_dead_zone_test.py.

Bonus: a Monte Carlo capacitor-mismatch run (sigma_C = 5%) at the noise
level that maximizes H_stable. Tells us whether real CMOS variability
helps the network escape the consensus attractor.

Created: 2026-04-19 (Claude Opus 4.7, hardware track)
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

NGSPICE = Path("D:/ANTIGRAVITY/ngspice-46_64/Spice64/bin/ngspice_con.exe")
RESULTS = ROOT / "experiments" / "spice" / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

from limit02_topology_sweep import make_ba  # noqa: E402
from spice_dead_zone_test import (  # noqa: E402
    PHYS, N, M_BA, HERETIC_RATIO, T_END, DT, SEED, TAIL_FRAC,
    compute_scale_factors, make_heretic_mask, parse_wrdata,
    cognitive_entropy, h_stable,
)

NORMS = ["uniform", "degree_linear", "spectral"]
NOISE_LEVELS = [0.0, 0.01, 0.03, 0.10, 0.30]
MISMATCH_SIGMA = 0.05      # 5% RMS capacitor variation (CMOS-typical)
MISMATCH_TRIALS = 3        # Monte Carlo runs (noise level reused from phase 1 best)


# ---------- Netlist generation with trnoise() and capacitor mismatch ----------

def generate_netlist(adj: np.ndarray, scale: np.ndarray, init_v: np.ndarray,
                     t_end: float, dt: float, tag: str,
                     noise_amp: float, c_values: np.ndarray | None = None) -> Path:
    """Build coupled-FHN+doubt netlist with optional trnoise and per-node capacitance.

    noise_amp:  RMS amplitude for trnoise on each B_dv (units: A on a 1F cap = V/s).
                trnoise(NA NT NALPHA NAMP) -> NA = noise density (V/sqrtHz),
                NT = sample time. We sample at dt.
    c_values:   per-node capacitance vector. If None, all 1F (no mismatch).
    """
    n = adj.shape[0]
    D_uniform = PHYS["D"] / np.sqrt(n)
    if c_values is None:
        c_values = np.ones(n)

    L = []
    L.append(f"* Mem4ristor v3 noise/mismatch test ({tag})")
    L.append(f".title Mem4ristor SPICE noise={noise_amp:g} {tag}")
    L.append("")
    for k, v in PHYS.items():
        L.append(f".param {k}={v:g}")
    L.append(f".param D_uni={D_uniform:.8g}")
    L.append("")

    L.append("* State capacitors (C_v with mismatch; C_w, C_u nominal)")
    for i in range(n):
        L.append(f"C_v{i} v{i} 0 {c_values[i]:.6f} IC={init_v[i]:.6f}")
        L.append(f"C_w{i} w{i} 0 1 IC=0.0")
        L.append(f"C_u{i} u{i} 0 1 IC={PHYS['sigma_base']:.4f}")
    L.append("")

    L.append("* Behavioral dynamics + optional trnoise current injection")
    for i in range(n):
        nbrs = np.where(adj[i] > 0)[0]
        lap = " + ".join(f"(V(v{j}) - V(v{i}))" for j in nbrs) if len(nbrs) else "0"
        L.append(
            f"B_dv{i} 0 v{i} I = V(v{i}) - V(v{i})*V(v{i})*V(v{i})/v_cubic_divisor "
            f"- V(w{i}) "
            f"+ D_uni*{scale[i]:.8g}*(tanh(3.14159265*(0.5 - V(u{i}))) + leak_delta)*({lap}) "
            f"- alpha*tanh(V(v{i}))"
        )
        if noise_amp > 0:
            # trnoise(NA NT NALPHA NAMP) is only valid as an independent
            # current/voltage source, NOT inside a B-source expression.
            # Use Iname pos neg dc 0 trnoise(...). Convention: I from 0->v means
            # positive current INTO node v -> +dV/dt.
            L.append(
                f"I_eta{i} 0 v{i} dc 0 trnoise({noise_amp:g} {dt:g} 0 0)"
            )
        L.append(f"B_dw{i} 0 w{i} I = eps*(V(v{i}) + a - b*V(w{i}))")
        L.append(f"B_du{i} 0 u{i} I = eps_u*(sigma_base - V(u{i}))")
    L.append("")

    save_nodes = " ".join(f"v(v{i})" for i in range(n))
    L.append(f".save {save_nodes}")
    # Slightly looser tolerances + Gear method help with stochastic dynamics
    method = "gear" if noise_amp > 0 else "trap"
    L.append(f".options method={method} reltol=5e-3 abstol=1e-5 itl4=300")
    L.append(f".tran {dt:g} {t_end:g} 0 0.5 uic")
    L.append("")
    L.append(".control")
    L.append("run")
    out = (RESULTS / f"{tag}.dat").as_posix()
    L.append(f"wrdata {out} {save_nodes}")
    L.append("quit")
    L.append(".endc")
    L.append(".end")

    path = RESULTS / f"{tag}.cir"
    path.write_text("\n".join(L), encoding="utf-8")
    return path


def run_ngspice(netlist_path: Path) -> float:
    if not NGSPICE.exists():
        sys.exit(f"ngspice not found at {NGSPICE}")
    t0 = time.time()
    res = subprocess.run(
        [str(NGSPICE), "-b", str(netlist_path)],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    dt = time.time() - t0
    if res.returncode != 0:
        print("STDOUT:", res.stdout[-800:])
        print("STDERR:", res.stderr[-500:])
        sys.exit(f"ngspice failed (rc={res.returncode}) on {netlist_path.name}")
    return dt


# ---------- Main ----------

def main() -> int:
    print("=" * 84)
    print(f"SPICE noise/mismatch resonance test  (BA m={M_BA}, N={N})")
    print("=" * 84)

    adj = make_ba(N, m=M_BA, seed=SEED)
    rng = np.random.RandomState(SEED)
    init_v = rng.uniform(-1.0, 1.0, N)
    print(f"adjacency: edges={int(adj.sum()/2)}, "
          f"deg min/mean/max = {int(adj.sum(axis=1).min())}/"
          f"{adj.sum(axis=1).mean():.1f}/{int(adj.sum(axis=1).max())}")
    print()

    # ---- Phase 1: noise sweep ----
    print(f"PHASE 1 — noise sweep (no mismatch)")
    print(f"{'norm':<14} " + " ".join(f"{'eta='+f'{a:g}':>10}" for a in NOISE_LEVELS))
    print("-" * 84)

    table = {}  # (norm, eta) -> H
    runtimes = {}
    for norm in NORMS:
        scale = compute_scale_factors(adj, norm)
        row = []
        for eta in NOISE_LEVELS:
            tag = f"noise_BA_m{M_BA}_N{N}_{norm}_eta{eta:g}"
            path = generate_netlist(adj, scale, init_v, T_END, DT, tag, noise_amp=eta)
            secs = run_ngspice(path)
            runtimes[(norm, eta)] = secs
            _, v_sp = parse_wrdata(RESULTS / f"{tag}.dat")
            H = h_stable(v_sp)
            table[(norm, eta)] = H
            row.append(H)
        cells = " ".join(f"{H:>10.3f}" for H in row)
        print(f"{norm:<14} {cells}")

    total_t = sum(runtimes.values())
    print(f"\n  total ngspice runtime: {total_t:.1f}s "
          f"(mean {total_t/len(runtimes):.1f}s/run)")

    # ---- Phase 2: capacitor mismatch Monte Carlo at the best (norm, eta) ----
    # Pick the (norm, eta) that maximized H from phase 1.
    # Tie-break: prefer higher eta (more "interesting" stochastic regime).
    best_key = max(table, key=lambda k: (table[k], k[1]))
    best_norm, best_eta = best_key
    print()
    print(f"PHASE 2 — capacitor mismatch Monte Carlo "
          f"(sigma_C={MISMATCH_SIGMA*100:g}%, reusing best eta={best_eta:g})")
    print(f"  reference: norm={best_norm}, eta={best_eta}, H_no_mismatch={table[best_key]:.3f}")

    scale = compute_scale_factors(adj, best_norm)
    mismatch_H = []
    for trial in range(MISMATCH_TRIALS):
        mc_rng = np.random.RandomState(SEED + 1000 + trial)
        c_vals = np.clip(mc_rng.normal(1.0, MISMATCH_SIGMA, N), 0.5, 1.5)
        tag = f"mismatch_BA_m{M_BA}_N{N}_{best_norm}_eta{best_eta:g}_trial{trial}"
        path = generate_netlist(adj, scale, init_v, T_END, DT, tag,
                                noise_amp=best_eta, c_values=c_vals)
        secs = run_ngspice(path)
        _, v_sp = parse_wrdata(RESULTS / f"{tag}.dat")
        H = h_stable(v_sp)
        mismatch_H.append(H)
        print(f"  trial {trial}: H={H:.3f}  (ngspice {secs:.1f}s)")

    H_mismatch = float(np.mean(mismatch_H))
    H_no_mismatch = table[best_key]
    delta = H_mismatch - H_no_mismatch
    print(f"\n  mean H with mismatch:    {H_mismatch:.3f} +- {np.std(mismatch_H):.3f}")
    print(f"  H without mismatch:      {H_no_mismatch:.3f}")
    print(f"  delta (mismatch effect): {delta:+.3f}")

    # ---- Verdict ----
    print()
    print("=" * 84)
    h_max_overall = max(table.values())
    norm_h_max = max(k for k in table if table[k] == h_max_overall)
    print(f"H_max overall: {h_max_overall:.3f} at {norm_h_max}")
    if h_max_overall > 0.30:
        print(f"=> Noise/mismatch RESCAPES the dead zone (stochastic resonance confirmed in SPICE).")
    elif h_max_overall > 0.10:
        print(f"=> Partial escape — noise modulates dynamics but consensus still dominates.")
    else:
        print(f"=> Dead zone is robust to noise/mismatch up to eta={max(NOISE_LEVELS)}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
