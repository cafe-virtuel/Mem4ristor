#!/usr/bin/env python3
"""
SPICE vs Python validation for Mem4ristor v3 — Hardware feasibility check.

Generates a minimal NxN coupled FHN+doubt netlist programmatically, runs it with
ngspice 46, parses the trajectories, runs the IDENTICAL equations in Python
(plain Euler, no heretics/noise/hysteresis to match SPICE), and produces a
side-by-side comparison figure.

Goal: prove the Python dynamics are reproducible in analog hardware. This is
the figure that should appear in any future hardware paper.

Created: 2026-04-19 (Claude Opus 4.7, P4 hardware validation track)
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# --- Paths ---
ROOT = Path(__file__).resolve().parent.parent
NGSPICE = Path("D:/ANTIGRAVITY/ngspice-46_64/Spice64/bin/ngspice_con.exe")
RESULTS = ROOT / "experiments" / "spice" / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)


# --- Shared physics (one source of truth) ---
PHYS = dict(
    a=0.7, b=0.8, eps=0.08, alpha=0.15,
    eps_u=0.02, sigma_base=0.05,
    D=0.15, leak_delta=0.05,
)

# --- Simulation grid ---
N_SIDE = 4              # 4x4 = 16 units (small enough for fast ngspice tran)
T_END = 50.0
DT = 0.05
SAMPLE_NODES = [0, 5, 10, 15]   # diagonal-ish sample for the figure


# ---------- Netlist generation ----------

def generate_netlist(n_side: int, t_end: float, dt: float, seed: int = 42) -> str:
    """Build a coupled-FHN+doubt netlist with periodic 4-neighbor lattice.

    Uses direct integrators: a B-source as current (CUR=f) on a 1F capacitor,
    so dV/dt = f exactly. The previous RC-integrator pattern (R + B-voltage)
    silently adds a -v damping term and does NOT integrate the FHN equation.
    """
    rng = np.random.RandomState(seed)
    n_total = n_side * n_side
    d_eff = PHYS["D"] / np.sqrt(n_total)

    L = []
    L.append(f"* Mem4ristor v3 - {n_side}x{n_side} validation netlist (direct integrator)")
    L.append(".title Mem4ristor SPICE vs Python validation")
    L.append("")
    L.append(".param a={:g}".format(PHYS["a"]))
    L.append(".param b={:g}".format(PHYS["b"]))
    L.append(".param eps={:g}".format(PHYS["eps"]))
    L.append(".param alpha_cog={:g}".format(PHYS["alpha"]))
    L.append(".param eps_u={:g}".format(PHYS["eps_u"]))
    L.append(".param sigma_base={:g}".format(PHYS["sigma_base"]))
    L.append(".param D_eff={:g}".format(d_eff))
    L.append(".param leak_delta={:g}".format(PHYS["leak_delta"]))
    L.append("")

    # Capacitors (one per state variable). 1F + IC means V(node) integrates dV/dt = I_in.
    init_v = rng.uniform(-1.0, 1.0, n_total)
    L.append("* State capacitors (1F so I = dV/dt)")
    for i in range(n_total):
        L.append(f"C_v{i} v{i} 0 1 IC={init_v[i]:.6f}")
        L.append(f"C_w{i} w{i} 0 1 IC=0.0")
        L.append(f"C_u{i} u{i} 0 1 IC={PHYS['sigma_base']:.4f}")
    L.append("")

    # B-sources as current injectors: I from 0 -> node = +dV/dt at node.
    L.append("* Behavioral dynamics (B-source current injection)")
    for i in range(n_total):
        row, col = divmod(i, n_side)
        nbrs = [
            ((row - 1) % n_side) * n_side + col,
            ((row + 1) % n_side) * n_side + col,
            row * n_side + ((col - 1) % n_side),
            row * n_side + ((col + 1) % n_side),
        ]
        lap = " + ".join(f"(V(v{n}) - V(v{i}))" for n in nbrs)
        L.append(
            f"B_dv{i} 0 v{i} I = V(v{i}) - V(v{i})*V(v{i})*V(v{i})/5 "
            f"- V(w{i}) "
            f"+ D_eff * (tanh(3.14159*(0.5 - V(u{i}))) + leak_delta) * ({lap}) "
            f"- alpha_cog*tanh(V(v{i}))"
        )
        L.append(f"B_dw{i} 0 w{i} I = eps * (V(v{i}) + a - b * V(w{i}))")
        L.append(f"B_du{i} 0 u{i} I = eps_u * (sigma_base - V(u{i}))")
    L.append("")

    # Use .save to limit output size to nodes we need + use wrdata for clean ascii
    save_nodes = " ".join(f"v(v{i})" for i in range(n_total))
    L.append(f".save {save_nodes}")
    # Stiff-aware integration. Note: avoid pow() on signed quantities (ngspice
    # Jacobian misbehaves on negative bases) — we expand x^3 as x*x*x above.
    L.append(".options method=trap reltol=1e-3 abstol=1e-6 itl4=200")
    L.append(f".tran {dt:g} {t_end:g} 0 {dt:g} uic")
    L.append("")
    L.append(".control")
    L.append("run")
    out = (RESULTS / f"spice_trace_{n_side}x{n_side}.dat").as_posix()
    L.append(f"wrdata {out} {save_nodes}")
    L.append("quit")
    L.append(".endc")
    L.append(".end")

    return "\n".join(L), init_v


def run_ngspice(netlist_path: Path) -> None:
    if not NGSPICE.exists():
        sys.exit(f"ngspice not found at {NGSPICE}")
    print(f"  -> ngspice -b {netlist_path.name}")
    res = subprocess.run(
        [str(NGSPICE), "-b", str(netlist_path)],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    if res.returncode != 0:
        print("STDOUT:", res.stdout[-500:])
        print("STDERR:", res.stderr[-500:])
        sys.exit(f"ngspice failed (rc={res.returncode})")


def parse_wrdata(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """ngspice wrdata writes columns: t v1 t v2 t v3 ... (time repeated)."""
    raw = np.loadtxt(path)
    if raw.ndim == 1:
        raw = raw[None, :]
    # Columns: t0, v0, t1, v1, t2, v2, ...
    t = raw[:, 0]
    n_signals = raw.shape[1] // 2
    v = np.column_stack([raw[:, 2 * k + 1] for k in range(n_signals)])
    return t, v


# ---------- Python reference (identical equations) ----------

def python_reference(n_side: int, t_end: float, dt: float, init_v: np.ndarray):
    """Plain Euler integration with the EXACT same dynamics as the netlist."""
    n_total = n_side * n_side
    d_eff = PHYS["D"] / np.sqrt(n_total)
    a, b, eps = PHYS["a"], PHYS["b"], PHYS["eps"]
    alpha = PHYS["alpha"]
    eps_u, sigma_base = PHYS["eps_u"], PHYS["sigma_base"]
    delta = PHYS["leak_delta"]

    # Build adjacency for periodic 4-neighbor lattice
    adj = np.zeros((n_total, n_total))
    for i in range(n_total):
        row, col = divmod(i, n_side)
        for j in [
            ((row - 1) % n_side) * n_side + col,
            ((row + 1) % n_side) * n_side + col,
            row * n_side + ((col - 1) % n_side),
            row * n_side + ((col + 1) % n_side),
        ]:
            adj[i, j] = 1.0
    deg = adj.sum(axis=1)

    v = init_v.copy()
    w = np.zeros(n_total)
    u = np.full(n_total, sigma_base)

    n_steps = int(round(t_end / dt)) + 1
    t_hist = np.zeros(n_steps)
    v_hist = np.zeros((n_steps, n_total))
    v_hist[0] = v

    for k in range(1, n_steps):
        # Laplacian-style coupling: D_eff * f(u) * sum_j A_ij * (v_j - v_i)
        coupling = adj @ v - deg * v
        kernel = np.tanh(np.pi * (0.5 - u)) + delta
        dv = v - v ** 3 / 5.0 - w + d_eff * kernel * coupling - alpha * np.tanh(v)
        dw = eps * (v + a - b * w)
        du = eps_u * (sigma_base - u)
        v = v + dv * dt
        w = w + dw * dt
        u = u + du * dt
        t_hist[k] = k * dt
        v_hist[k] = v

    return t_hist, v_hist


# ---------- Main ----------

def main() -> int:
    print(f"=== SPICE vs Python validation (N={N_SIDE}x{N_SIDE}) ===")

    netlist, init_v = generate_netlist(N_SIDE, T_END, DT)
    netlist_path = RESULTS / f"validation_{N_SIDE}x{N_SIDE}.cir"
    netlist_path.write_text(netlist, encoding="utf-8")
    print(f"  netlist: {netlist_path}")

    run_ngspice(netlist_path)

    spice_data = RESULTS / f"spice_trace_{N_SIDE}x{N_SIDE}.dat"
    t_sp, v_sp = parse_wrdata(spice_data)
    print(f"  spice samples: {t_sp.size}, signals: {v_sp.shape[1]}")

    t_py, v_py = python_reference(N_SIDE, T_END, DT, init_v)
    print(f"  python samples: {t_py.size}")

    # Resample SPICE onto Python grid for clean RMS comparison
    v_sp_resampled = np.column_stack([
        np.interp(t_py, t_sp, v_sp[:, i]) for i in range(v_sp.shape[1])
    ])
    rms_per_node = np.sqrt(np.mean((v_sp_resampled - v_py) ** 2, axis=0))
    rms_global = float(np.sqrt(np.mean(rms_per_node ** 2)))
    final_dv_max = float(np.max(np.abs(v_sp_resampled[-1] - v_py[-1])))
    print(f"  RMS(v) per node: mean={rms_per_node.mean():.4e}, max={rms_per_node.max():.4e}")
    print(f"  RMS global = {rms_global:.4e}, final |dv|_max = {final_dv_max:.4e}")

    # --- Figure ---
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, idx in zip(axes.flat, SAMPLE_NODES):
        ax.plot(t_py, v_py[:, idx], lw=1.6, color="C0", label="Python (Euler)")
        ax.plot(t_sp, v_sp[:, idx], lw=1.0, ls="--", color="C3", label="SPICE (ngspice 46)")
        ax.set_title(f"unit {idx}  (RMS={rms_per_node[idx]:.2e})", fontsize=10)
        ax.set_ylabel("v")
        ax.grid(alpha=0.3)
    axes[1, 0].set_xlabel("time")
    axes[1, 1].set_xlabel("time")
    axes[0, 0].legend(fontsize=8, loc="best")
    fig.suptitle(
        f"Mem4ristor v3 SPICE/Python validation — {N_SIDE}×{N_SIDE} lattice "
        f"(global RMS = {rms_global:.2e})",
        fontsize=12,
    )
    fig.tight_layout()

    out_fig = FIGURES / "spice_vs_python_validation.png"
    fig.savefig(out_fig, dpi=140)
    print(f"  figure: {out_fig}")

    # Pass/fail thresholds (analog hardware feasibility):
    # - global RMS < 0.05 (5% of typical |v| amplitude ~1)
    # - final |dv|_max < 0.10
    ok = (rms_global < 0.05) and (final_dv_max < 0.10)
    verdict = "PASS" if ok else "FAIL"
    print(f"\n  >>> Hardware feasibility: {verdict}")
    print(f"      (global RMS<0.05 and final |dv|_max<0.10)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
