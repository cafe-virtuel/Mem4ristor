#!/usr/bin/env python3
"""
P4.20 — HfO₂ memristor compact model : neurone (A), synapse (B), combiné (A+B).

Instead of ideal capacitors (1F) or fixed coupling weights, we introduce a
Yakopcic-inspired smooth memristor model suitable for ngspice B-sources:

    G(x) = Ron*x + Roff*(1-x)       # Memristive conductance
    dx/dt = k_s * tanh(kv*V) * x*(1-x)   # Smooth state equation (no conditionals)
      - tanh(kv*V)  : drive direction (SET if V>0, RESET if V<0)
      - x*(1-x)     : natural [0,1] barrier (Bernstein form)

Parameters calibrated for HfO₂ filamentary switching:
    Ron  = 100 Ω   (filament fully formed)
    Roff = 16000 Ω (filament ruptured, Ron/Roff ~ 160:1, typical HfO₂)
    k_s  = 0.02    (switching speed, normalized to our FHN timescale)
    kv   = 3.0     (voltage sensitivity)
    x0   = 0.5     (initial state, mid-point)

Three sub-experiments
─────────────────────
(A) Memristive neuron  : x_v{i} modulates neuron excitability (alpha term).
    The local firing threshold becomes state-dependent — a physical model of
    neuronal fatigue/potentiation. Lattice 4×4, T=100, verify FHN survives.

(B) Memristive synapse : each edge i→j has its own state x_ij.
    G_ij(x_ij) replaces the fixed coupling weight. Anti-Hebbian effect:
    larger voltage difference → SET → stronger coupling (disagreeing nodes
    reinforce their connection). Tested on BA m=5 N=64 (canonical dead zone).

(A+B) Combined         : both neuron and synapse memristors active.
    BA m=5 N=16 (smaller for compute budget).

Created: 2026-04-19 (Antigravity, P4.20)
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

NGSPICE = Path("D:/ANTIGRAVITY/ngspice-46_64/Spice64/bin/ngspice_con.exe")
RESULTS = ROOT / "experiments" / "spice" / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

from spice_dead_zone_test import (  # noqa: E402
    PHYS, SEED, TAIL_FRAC, parse_wrdata, h_stable, compute_scale_factors,
)

# ─── Memristor HfO₂ parameters ────────────────────────────────────────────────
MEM = dict(
    Ron  = 100.0,      # Omega  -- ON resistance (filament formed)
    Roff = 16000.0,    # Omega  -- OFF resistance (filament ruptured)
    k_s  = 0.002,      # Switching speed -- 10x slower: let FHN establish first
    kv   = 3.0,        # Voltage sensitivity of switching
    x0   = 0.5,        # Initial state (symmetric)
)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_ba(n: int, m: int, seed: int) -> np.ndarray:
    return nx.to_numpy_array(nx.barabasi_albert_graph(n, m, seed=seed))


def make_lattice(n: int) -> np.ndarray:
    side = int(np.sqrt(n))
    G = nx.grid_2d_graph(side, side, periodic=True)
    return nx.to_numpy_array(G)


def run_ngspice(netlist_path: Path) -> float:
    if not NGSPICE.exists():
        sys.exit(f"ngspice not found at {NGSPICE}")
    t0 = time.time()
    res = subprocess.run(
        [str(NGSPICE), "-b", str(netlist_path)],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    elapsed = time.time() - t0
    if res.returncode != 0:
        print("STDERR:", res.stderr[-600:])
        sys.exit(f"ngspice failed on {netlist_path.name}")
    # Sanity: check output dat exists and is non-empty
    tag = netlist_path.stem
    dat = RESULTS / f"{tag}.dat"
    # For dead_zone prefix files the dat is different, handled by caller
    return elapsed


# ─── (A) Memristive neuron ─────────────────────────────────────────────────────

def netlist_A(adj: np.ndarray, init_v: np.ndarray, t_end: float, dt: float,
              tag: str) -> Path:
    """FHN with memristive excitability: alpha_eff(i) = alpha * G(x_v{i}) / G_mid.
    The neuron's firing threshold is modulated by its own memristive state.
    dx_v/dt = k_s * tanh(kv * V(v{i})) * x_v*(1-x_v)
    G(x_v) = Ron*x_v + Roff*(1-x_v)
    alpha_eff = alpha * G(x_v) / G_mid    [where G_mid = G(0.5)]
    """
    n = adj.shape[0]
    D_uniform = PHYS["D"] / np.sqrt(n)
    norm = "degree_linear"
    scale = compute_scale_factors(adj, norm)

    Ron, Roff, k_s, kv, x0 = MEM["Ron"], MEM["Roff"], MEM["k_s"], MEM["kv"], MEM["x0"]
    G_mid = Ron * 0.5 + Roff * 0.5   # G at x=0.5, the normalisation reference

    L = [f"* P4.20 Approach A — Memristive neuron ({tag})"]
    L.append(f".title Mem4ristor P4.20-A {tag}")
    L.append("")
    for k, v in PHYS.items():
        L.append(f".param {k}={v:g}")
    L.append(f".param D_uni={D_uniform:.8g}")
    L.append(f".param Ron={Ron:g} Roff={Roff:g} k_s={k_s:g} kv={kv:g}")
    L.append(f".param G_mid={G_mid:g}")
    L.append("")

    # State capacitors: FHN + memristor state x_v per neuron
    L.append("* Capacitors: FHN states + memristor state x_v{i}")
    for i in range(n):
        L.append(f"C_v{i}  v{i}  0 1 IC={init_v[i]:.6f}")
        L.append(f"C_w{i}  w{i}  0 1 IC=0.0")
        L.append(f"C_u{i}  u{i}  0 1 IC={PHYS['sigma_base']:.4f}")
        L.append(f"C_xv{i} xv{i} 0 1 IC={x0:.4f}")
    L.append("")

    L.append("* Dynamics")
    for i in range(n):
        nbrs = np.where(adj[i] > 0)[0]
        lap = " + ".join(f"(V(v{j})-V(v{i}))" for j in nbrs) if len(nbrs) else "0"
        # Effective conductance normalised to G_mid (so alpha_eff = alpha at x=0.5)
        G_expr = f"(Ron*V(xv{i}) + Roff*(1-V(xv{i})))"
        alpha_eff = f"alpha*{G_expr}/G_mid"
        L.append(
            f"B_dv{i} 0 v{i} I = V(v{i}) - V(v{i})*V(v{i})*V(v{i})/v_cubic_divisor "
            f"- V(w{i}) "
            f"+ D_uni*{scale[i]:.8g}*(tanh(3.14159265*(0.5-V(u{i})))+leak_delta)*({lap}) "
            f"- ({alpha_eff})*tanh(V(v{i}))"
        )
        L.append(f"B_dw{i}  0 w{i}  I = eps*(V(v{i})+a-b*V(w{i}))")
        L.append(f"B_du{i}  0 u{i}  I = eps_u*(sigma_base-V(u{i}))")
        # Memristor state: driven by local membrane potential
        L.append(
            f"B_dxv{i} 0 xv{i} I = k_s*tanh(kv*V(v{i}))*V(xv{i})*(1-V(xv{i}))"
        )
    L.append("")

    save_v = " ".join(f"v(v{i})" for i in range(n))
    save_x = " ".join(f"v(xv{i})" for i in range(n))
    L.append(f".save {save_v} {save_x}")
    L.append(".options method=gear reltol=5e-3 abstol=1e-5 itl4=500")
    L.append(f".tran {dt:g} {t_end:g} 0 0.5 uic")
    L.append(".control")
    L.append("run")
    dat = (RESULTS / f"{tag}.dat").as_posix()
    L.append(f"wrdata {dat} {save_v}")
    L.append("quit")
    L.append(".endc")
    L.append(".end")

    path = RESULTS / f"{tag}.cir"
    path.write_text("\n".join(L), encoding="utf-8")
    return path


# ─── (B) Memristive synapse ────────────────────────────────────────────────────

def netlist_B(adj: np.ndarray, init_v: np.ndarray, t_end: float, dt: float,
              tag: str) -> Path:
    """Synaptic conductances are memristive: G_ij(x_ij) replaces fixed weight.
    Coupling term: G_ij * (V(v_j) - V(v_i))
    dx_ij/dt = k_s * tanh(kv * (V(v_j) - V(v_i))) * x_ij*(1-x_ij)
    Anti-Hebbian: large voltage difference (disagreement) SETS the synapse
    (strong coupling), promoting information exchange over the link.
    """
    n = adj.shape[0]
    edges = [(i, j) for i in range(n) for j in np.where(adj[i] > 0)[0] if i < j]
    edge_idx = {(i, j): k for k, (i, j) in enumerate(edges)}

    Ron, Roff, k_s, kv, x0 = MEM["Ron"], MEM["Roff"], MEM["k_s"], MEM["kv"], MEM["x0"]
    # Characteristic coupling: D_uniform / G_mid so D_uniform * G/G_mid is the effective D
    G_mid = Ron * 0.5 + Roff * 0.5

    L = [f"* P4.20 Approach B — Memristive synapse ({tag})"]
    L.append(f".title Mem4ristor P4.20-B {tag}")
    L.append("")
    for k, v in PHYS.items():
        L.append(f".param {k}={v:g}")
    L.append(f".param Ron={Ron:g} Roff={Roff:g} k_s={k_s:g} kv={kv:g}")
    L.append(f".param G_mid={G_mid:g}")
    L.append("")

    # FHN state caps + one state cap per synaptic edge
    L.append("* FHN state capacitors")
    for i in range(n):
        L.append(f"C_v{i} v{i} 0 1 IC={init_v[i]:.6f}")
        L.append(f"C_w{i} w{i} 0 1 IC=0.0")
        L.append(f"C_u{i} u{i} 0 1 IC={PHYS['sigma_base']:.4f}")

    L.append("")
    L.append("* Synaptic memristor state capacitors")
    for (ei, ej) in edges:
        k_e = edge_idx[(ei, ej)]
        L.append(f"C_xs{k_e} xs{k_e} 0 1 IC={x0:.4f}")
    L.append("")

    # Memristive coupling contribution per node
    # For node i: I_mem_i = sum_j G_ij * (V(v_j) - V(v_i)) * scale_factor
    # Rewrite using D_uniform / G_mid as base coupling, then multiply by G_ij
    D_uniform = PHYS["D"] / np.sqrt(n)
    scale = compute_scale_factors(adj, "degree_linear")
    # Precompute edge-to-node contribution map
    node_coupling = {i: [] for i in range(n)}
    for (ei, ej) in edges:
        k_e = edge_idx[(ei, ej)]
        node_coupling[ei].append((ej, k_e))
        node_coupling[ej].append((ei, k_e))

    L.append("* FHN dynamics with memristive coupling")
    for i in range(n):
        cterms = []
        for j, k_e in node_coupling[i]:
            G_expr = f"(Ron*V(xs{k_e})+Roff*(1-V(xs{k_e})))"
            cterms.append(f"{D_uniform:.8g}*{scale[i]:.8g}*(tanh(3.14159265*(0.5-V(u{i})))+leak_delta)*({G_expr}/G_mid)*(V(v{j})-V(v{i}))")
        coup = " + ".join(cterms) if cterms else "0"
        L.append(
            f"B_dv{i} 0 v{i} I = V(v{i}) - V(v{i})*V(v{i})*V(v{i})/v_cubic_divisor "
            f"- V(w{i}) + {coup} - alpha*tanh(V(v{i}))"
        )
        L.append(f"B_dw{i} 0 w{i} I = eps*(V(v{i})+a-b*V(w{i}))")
        L.append(f"B_du{i} 0 u{i} I = eps_u*(sigma_base-V(u{i}))")

    L.append("")
    L.append("* Synaptic memristor state dynamics")
    for (ei, ej) in edges:
        k_e = edge_idx[(ei, ej)]
        Vdiff = f"(V(v{ej})-V(v{ei}))"
        L.append(
            f"B_xs{k_e} 0 xs{k_e} I = k_s*tanh(kv*{Vdiff})*V(xs{k_e})*(1-V(xs{k_e}))"
        )
    L.append("")

    save_v = " ".join(f"v(v{i})" for i in range(n))
    L.append(f".save {save_v}")
    L.append(".options method=gear reltol=5e-3 abstol=1e-5 itl4=500")
    L.append(f".tran {dt:g} {t_end:g} 0 0.5 uic")
    L.append(".control")
    L.append("run")
    dat = (RESULTS / f"{tag}.dat").as_posix()
    L.append(f"wrdata {dat} {save_v}")
    L.append("quit")
    L.append(".endc")
    L.append(".end")

    path = RESULTS / f"{tag}.cir"
    path.write_text("\n".join(L), encoding="utf-8")
    return path


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_wrdata_local(path: Path, n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Parse ngspice wrdata output where each node contributes (t_i, v_i) column pair.
    Unlike parse_wrdata in spice_dead_zone_test, this handles N_NODES * 2 columns."""
    raw = np.loadtxt(path)
    if raw.ndim == 1:
        raw = raw[None, :]
    # Format: [t0, v0(t), t0, v1(t), t0, v2(t), ...] per row
    t = raw[:, 0]
    v_cols = [raw[:, 2 * k + 1] for k in range(n_nodes)]
    return t, np.column_stack(v_cols)



def main() -> int:
    rng = np.random.RandomState(SEED)
    results = {}
    t_global = time.time()

    # ── (A) Memristive neuron — 4×4 lattice ───────────────────────────────────
    print("\n" + "=" * 72)
    print("(A) Memristive neuron — 4x4 toroidal lattice")
    print("=" * 72)
    N_A, T_A, DT_A = 16, 100.0, 0.05
    adj_A = make_lattice(N_A)
    iv_A = rng.uniform(-1.0, 1.0, N_A)
    tag_A = "p420_A_neuron_lattice4x4"
    path_A = netlist_A(adj_A, iv_A, T_A, DT_A, tag_A)
    t_A = run_ngspice(path_A)
    _, v_A = parse_wrdata_local(RESULTS / f"{tag_A}.dat", N_A)
    H_A = h_stable(v_A)
    results["A_lattice4x4"] = H_A
    print(f"  H_stable = {H_A:.3f}   ({t_A:.1f}s)")
    print(f"  FHN survives memristive neuron: {'YES' if H_A > 0.05 else 'COLLAPSED'}")

    # ── (B) Memristive synapse — BA m=5 N=64 ──────────────────────────────────
    print("\n" + "=" * 72)
    print("(B) Memristive synapse — BA m=5 N=64 (canonical dead zone)")
    print("=" * 72)
    N_B, T_B, DT_B = 64, 150.0, 0.05
    adj_B = make_ba(N_B, m=5, seed=SEED)
    iv_B = rng.uniform(-1.0, 1.0, N_B)
    tag_B = "p420_B_synapse_BAm5_N64"
    path_B = netlist_B(adj_B, iv_B, T_B, DT_B, tag_B)
    t_B = run_ngspice(path_B)
    _, v_B = parse_wrdata_local(RESULTS / f"{tag_B}.dat", N_B)
    H_B = h_stable(v_B)
    results["B_BAm5_N64"] = H_B
    print(f"  H_stable = {H_B:.3f}   ({t_B:.1f}s)")
    print(f"  Dead zone broken by memristive synapse: {'YES' if H_B > 0.20 else 'NO'}")

    # ── (A+B) Combined — BA m=5 N=16 ──────────────────────────────────────────
    print("\n" + "=" * 72)
    print("(A+B) Combined memristive neuron + synapse — BA m=5 N=16")
    print("=" * 72)
    N_AB, T_AB, DT_AB = 16, 150.0, 0.05
    adj_AB = make_ba(N_AB, m=5, seed=SEED)
    iv_AB = rng.uniform(-1.0, 1.0, N_AB)
    tag_AB = "p420_AB_combined_BAm5_N16"

    # Combined netlist: write from scratch to avoid patching fragility
    Ron, Roff, k_s, kv, x0 = MEM["Ron"], MEM["Roff"], MEM["k_s"], MEM["kv"], MEM["x0"]
    G_mid = Ron * 0.5 + Roff * 0.5
    D_uniform_ab = PHYS["D"] / np.sqrt(N_AB)
    scale_ab = compute_scale_factors(adj_AB, "degree_linear")
    edges_ab = [(i, j) for i in range(N_AB) for j in np.where(adj_AB[i] > 0)[0] if i < j]
    eidx_ab  = {(i, j): k for k, (i, j) in enumerate(edges_ab)}
    node_coup_ab = {i: [] for i in range(N_AB)}
    for (ei, ej) in edges_ab:
        k_e = eidx_ab[(ei, ej)]
        node_coup_ab[ei].append((ej, k_e))
        node_coup_ab[ej].append((ei, k_e))

    L = [f"* P4.20 A+B combined ({tag_AB})"]
    L.append(f".title Mem4ristor P4.20-AB {tag_AB}")
    L.append("")
    for k, v in PHYS.items():
        L.append(f".param {k}={v:g}")
    L.append(f".param Ron={Ron:g} Roff={Roff:g} k_s={k_s:g} kv={kv:g}")
    L.append(f".param G_mid={G_mid:g}")
    L.append("")
    # Capacitors: FHN + neuron memristor xv + synaptic memristors xs
    for i in range(N_AB):
        L.append(f"C_v{i}  v{i}  0 1 IC={iv_AB[i]:.6f}")
        L.append(f"C_w{i}  w{i}  0 1 IC=0.0")
        L.append(f"C_u{i}  u{i}  0 1 IC={PHYS['sigma_base']:.4f}")
        L.append(f"C_xv{i} xv{i} 0 1 IC={x0:.4f}")
    for (ei, ej) in edges_ab:
        k_e = eidx_ab[(ei, ej)]
        L.append(f"C_xs{k_e} xs{k_e} 0 1 IC={x0:.4f}")
    L.append("")
    # Dynamics
    for i in range(N_AB):
        cterms = []
        for j, k_e in node_coup_ab[i]:
            G_syn = f"(Ron*V(xs{k_e})+Roff*(1-V(xs{k_e})))"
            cterm = (f"{D_uniform_ab:.8g}*{scale_ab[i]:.8g}"
                     f"*(tanh(3.14159265*(0.5-V(u{i})))+leak_delta)"
                     f"*({G_syn}/G_mid)*(V(v{j})-V(v{i}))")
            cterms.append(cterm)
        coup = " + ".join(cterms) if cterms else "0"
        G_neu = f"(Ron*V(xv{i})+Roff*(1-V(xv{i})))"
        L.append(
            f"B_dv{i} 0 v{i} I = V(v{i}) - V(v{i})*V(v{i})*V(v{i})/v_cubic_divisor "
            f"- V(w{i}) + {coup} "
            f"- alpha*({G_neu}/G_mid)*tanh(V(v{i}))"
        )
        L.append(f"B_dw{i}  0 w{i}  I = eps*(V(v{i})+a-b*V(w{i}))")
        L.append(f"B_du{i}  0 u{i}  I = eps_u*(sigma_base-V(u{i}))")
        L.append(f"B_dxv{i} 0 xv{i} I = k_s*tanh(kv*V(v{i}))*V(xv{i})*(1-V(xv{i}))")
    for (ei, ej) in edges_ab:
        k_e = eidx_ab[(ei, ej)]
        L.append(f"B_xs{k_e} 0 xs{k_e} I = k_s*tanh(kv*(V(v{ej})-V(v{ei})))*V(xs{k_e})*(1-V(xs{k_e}))")
    L.append("")
    save_v = " ".join(f"v(v{i})" for i in range(N_AB))
    L.append(f".save {save_v}")
    L.append(".options method=gear reltol=5e-3 abstol=1e-5 itl4=500")
    L.append(f".tran {DT_AB:g} {T_AB:g} 0 0.5 uic")
    L.append(".control")
    L.append("run")
    dat_ab = (RESULTS / f"{tag_AB}.dat").as_posix()
    L.append(f"wrdata {dat_ab} {save_v}")
    L.append("quit")
    L.append(".endc")
    L.append(".end")
    path_AB = RESULTS / f"{tag_AB}.cir"
    path_AB.write_text("\n".join(L), encoding="utf-8")

    t_AB = run_ngspice(path_AB)
    _, v_AB = parse_wrdata_local(RESULTS / f"{tag_AB}.dat", N_AB)
    H_AB = h_stable(v_AB)
    results["AB_BAm5_N16"] = H_AB
    print(f"  H_stable = {H_AB:.3f}   ({t_AB:.1f}s)")
    synergy = H_AB - max(H_A, H_B)
    print(f"  Synergy (H_AB - max(H_A, H_B)): {synergy:+.3f}")

    # ── Summary figure ─────────────────────────────────────────────────────────
    labels = ["(A) Neuron\n4x4 lattice", "(B) Synapse\nBA m=5 N=64", "(A+B) Combined\nBA m=5 N=16"]
    Hs = [H_A, H_B, H_AB]
    colors = ["#4e79a7", "#f28e2b", "#e15759"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, Hs, color=colors, width=0.55, edgecolor="white", linewidth=1.2)
    for bar, h in zip(bars, Hs):
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.3f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("H_stable (Shannon entropy, 5 bins)", fontsize=11)
    ax.set_title("P4.20 — HfO₂ memristor model: neuron vs synapse vs combined\n"
                 "Yakopcic smooth model, degree_linear norm", fontsize=10)
    ax.set_ylim(0, max(Hs) * 1.25 + 0.1)
    ax.grid(axis="y", alpha=0.3)

    if synergy > 0.05:
        ax.annotate(f"Synergy: +{synergy:.3f}", xy=(2, H_AB), xytext=(1.6, H_AB + 0.1),
                    fontsize=9, color="#e15759",
                    arrowprops=dict(arrowstyle="->", color="#e15759"))

    fig.tight_layout()
    out = FIGURES / "p420_hfo2_memristor.png"
    fig.savefig(out, dpi=140)
    print(f"\nFigure: {out}")

    # ── CSV ────────────────────────────────────────────────────────────────────
    csv_path = FIGURES / "p420_hfo2_memristor.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("experiment,topology,N,H_stable\n")
        f.write(f"A_neuron,lattice_4x4,{N_A},{H_A:.4f}\n")
        f.write(f"B_synapse,BAm5,{N_B},{H_B:.4f}\n")
        f.write(f"AB_combined,BAm5,{N_AB},{H_AB:.4f}\n")

    print("\n" + "=" * 72)
    print("SUMMARY — P4.20 HfO2 memristor model")
    for k, v in results.items():
        print(f"  {k:<22} H = {v:.3f}")
    print(f"  synergy A+B vs best single: {synergy:+.3f}")
    print(f"  total elapsed: {time.time()-t_global:.1f}s")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
