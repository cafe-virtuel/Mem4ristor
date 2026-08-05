#!/usr/bin/env python3
"""
SPICE dead-zone test — does BA m=5 collapse in analog hardware too?

Generates a BA m=5 N=64 netlist with three coupling normalizations
(uniform, degree_linear, spectral) and compares SPICE vs Python H_stable.

Hypothesis: if H_stable collapses on BA m=5 in SPICE *just as it does in
Python*, the dead zone is intrinsic to the dynamics (not a numerical
artifact of Euler integration). That settles a Paper 2 question.

Runs deterministic (no noise, no V5, no V4 rewire, I_stimulus=0). Entropy
comes purely from the initial random voltage distribution and the
Levitating Sigmoid + heretic flip dynamics. This is a *structural* probe.

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

# --- Shared physics (matches Mem4Network defaults) ---
PHYS = dict(
    a=0.7, b=0.8, eps=0.08, alpha=0.15,
    eps_u=0.02, sigma_base=0.05,
    D=0.15, leak_delta=0.05,
    v_cubic_divisor=5.0,
)

# --- Configuration ---
N = 64                  # Network size (small enough for ngspice tran)
M_BA = 5                # BA preferential-attachment parameter (canonical dead zone)
HERETIC_RATIO = 0.15
T_END = 150.0           # Equivalent to 3000 Python steps at dt=0.05
DT = 0.05
SEED = 42
TAIL_FRAC = 0.25
NORMS = ["uniform", "degree_linear", "spectral"]


# ---------- Coupling weights (mirror Mem4Network._compute_coupling_weights) ----------

def eigenvector_centrality(adj: np.ndarray, max_iter: int = 200, tol: float = 1e-8) -> np.ndarray:
    n = adj.shape[0]
    c = np.ones(n) / np.sqrt(n)
    for _ in range(max_iter):
        c_new = adj @ c
        norm = np.linalg.norm(c_new)
        if norm < 1e-15:
            deg = adj.sum(axis=1)
            c_new = np.maximum(deg, 1.0)
            norm = np.linalg.norm(c_new)
        c_new = c_new / norm
        if np.linalg.norm(c_new - c) < tol:
            break
        c = c_new
    c = np.maximum(c_new, 1e-8)
    c = c / np.mean(c)  # mean = 1
    return c


def compute_scale_factors(adj: np.ndarray, norm: str) -> np.ndarray:
    """Per-node scale factor that multiplies the Laplacian term, matching
    Mem4Network.step lines 854-859: scale_factors = (node_weights * D) / (D/sqrt(N)).
    """
    n = adj.shape[0]
    if norm == "uniform":
        return np.ones(n)

    deg = np.maximum(adj.sum(axis=1), 1.0)
    if norm == "degree_linear":
        raw = 1.0 / deg
    elif norm == "spectral":
        raw = 1.0 / eigenvector_centrality(adj)
    else:
        raise ValueError(f"unknown norm {norm}")

    target_mean = 1.0 / np.sqrt(n)
    node_weights = raw * target_mean / np.mean(raw)
    # scale_factors = (node_weights * D) / (D / sqrt(N)) = node_weights * sqrt(N)
    return node_weights * np.sqrt(n)


# ---------- Heretic mask (mirrors Mem4ristorV3.__init__ uniform_placement) ----------

def make_heretic_mask(n: int, ratio: float, seed: int) -> np.ndarray:
    """Match the structured uniform placement in core.py:181-193."""
    rng = np.random.RandomState(seed)
    if ratio <= 0:
        return np.zeros(n, dtype=bool)
    step = max(int(1.0 / ratio), 1)
    target = int(n * ratio)
    ids = []
    for i in range(0, n, step):
        if len(ids) < target:
            block_end = min(i + step, n)
            ids.append(rng.randint(i, block_end))
    mask = np.zeros(n, dtype=bool)
    mask[ids] = True
    return mask


# ---------- Netlist generation ----------

def generate_netlist(adj: np.ndarray, scale: np.ndarray, heretic: np.ndarray,
                     init_v: np.ndarray, t_end: float, dt: float, tag: str) -> tuple[str, Path]:
    """Build coupled-FHN+doubt netlist for the given adjacency / norm.

    Uses direct integrators (C with B-source as current). Heretic flip is on
    the I_stimulus term, which is zero here — keeps the netlist faithful but
    inactive (matches Python with I_stimulus=0).
    """
    n = adj.shape[0]
    D_uniform = PHYS["D"] / np.sqrt(n)

    L = []
    L.append(f"* Mem4ristor v3 dead-zone test ({tag}) - BA m={M_BA} N={n}")
    L.append(f".title Mem4ristor SPICE dead-zone {tag}")
    L.append("")
    for k, v in PHYS.items():
        L.append(f".param {k}={v:g}")
    L.append(f".param D_uni={D_uniform:.8g}")
    L.append("")

    # State capacitors: 1F + IC means V(node) integrates dV/dt = I_in
    L.append("* State capacitors (1F so I = dV/dt)")
    for i in range(n):
        L.append(f"C_v{i} v{i} 0 1 IC={init_v[i]:.6f}")
        L.append(f"C_w{i} w{i} 0 1 IC=0.0")
        L.append(f"C_u{i} u{i} 0 1 IC={PHYS['sigma_base']:.4f}")
    L.append("")

    # Behavioral dynamics. For each node i:
    #   laplacian_i = sum_j A_ij*(v_j - v_i)
    #   coupling_i  = D_uni * scale[i] * (tanh(pi*(0.5 - u_i)) + leak_delta) * laplacian_i
    #   dv_i = v_i - v_i^3/v_cubic_divisor - w_i + coupling_i - alpha*tanh(v_i)
    #          (heretic flip on I_stimulus, but I_stimulus=0 here)
    L.append("* Behavioral dynamics (B-source current injection)")
    for i in range(n):
        nbrs = np.where(adj[i] > 0)[0]
        if len(nbrs) == 0:
            lap = "0"
        else:
            lap = " + ".join(f"(V(v{j}) - V(v{i}))" for j in nbrs)
        L.append(
            f"B_dv{i} 0 v{i} I = V(v{i}) - V(v{i})*V(v{i})*V(v{i})/v_cubic_divisor "
            f"- V(w{i}) "
            f"+ D_uni*{scale[i]:.8g}*(tanh(3.14159265*(0.5 - V(u{i}))) + leak_delta)*({lap}) "
            f"- alpha*tanh(V(v{i}))"
        )
        L.append(f"B_dw{i} 0 w{i} I = eps*(V(v{i}) + a - b*V(w{i}))")
        # NOTE (Audit Manus 2026-04-25 — Faille C): B_du omits the sigma_social term
        # present in the full Python model: du/dt = (eps_u/tau_u)*(k_u*sigma_local + sigma_base - u).
        # This is a deliberate simplification for a STRUCTURAL dead-zone probe (I_stimulus=0,
        # no noise). The full metacognitive coupling requires an additional B-source measuring
        # |mean(V(neighbors)) - V(v_i)|, which is out of scope here.
        # Consequence: hardware feasibility validated for FHN + autonomous doubt only.
        L.append(f"B_du{i} 0 u{i} I = eps_u*(sigma_base - V(u{i}))")
    L.append("")

    save_nodes = " ".join(f"v(v{i})" for i in range(n))
    L.append(f".save {save_nodes}")
    L.append(".options method=trap reltol=5e-3 abstol=1e-5 itl4=300")
    # Sample at coarser interval to keep the dat file small (T=150, dt_out=0.5 -> 300 samples)
    L.append(f".tran {dt:g} {t_end:g} 0 0.5 uic")
    L.append("")
    L.append(".control")
    L.append("run")
    out = (RESULTS / f"dead_zone_{tag}.dat").as_posix()
    L.append(f"wrdata {out} {save_nodes}")
    L.append("quit")
    L.append(".endc")
    L.append(".end")

    netlist = "\n".join(L)
    path = RESULTS / f"dead_zone_{tag}.cir"
    path.write_text(netlist, encoding="utf-8")
    return netlist, path


def run_ngspice(netlist_path: Path) -> float:
    if not NGSPICE.exists():
        sys.exit(f"ngspice not found at {NGSPICE}")
    print(f"  -> ngspice -b {netlist_path.name}")
    t0 = time.time()
    res = subprocess.run(
        [str(NGSPICE), "-b", str(netlist_path)],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    dt = time.time() - t0
    if res.returncode != 0:
        print("STDOUT:", res.stdout[-800:])
        print("STDERR:", res.stderr[-500:])
        sys.exit(f"ngspice failed (rc={res.returncode})")
    return dt


def parse_wrdata(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = np.loadtxt(path)
    if raw.ndim == 1:
        raw = raw[None, :]
    t = raw[:, 0]
    n_signals = raw.shape[1] // 2
    v = np.column_stack([raw[:, 2 * k + 1] for k in range(n_signals)])
    return t, v


# ---------- Python deterministic reference (no noise, no V5, no V4) ----------

def python_reference(adj: np.ndarray, scale: np.ndarray, init_v: np.ndarray,
                     t_end: float, dt: float):
    n = adj.shape[0]
    D_uniform = PHYS["D"] / np.sqrt(n)
    a, b, eps = PHYS["a"], PHYS["b"], PHYS["eps"]
    alpha = PHYS["alpha"]
    eps_u, sigma_base = PHYS["eps_u"], PHYS["sigma_base"]
    delta = PHYS["leak_delta"]
    cubic = PHYS["v_cubic_divisor"]
    deg = adj.sum(axis=1)

    v = init_v.copy()
    w = np.zeros(n)
    u = np.full(n, sigma_base)

    n_steps = int(round(t_end / dt)) + 1
    t_hist = np.zeros(n_steps)
    v_hist = np.zeros((n_steps, n))
    v_hist[0] = v

    for k in range(1, n_steps):
        coupling_lap = adj @ v - deg * v
        kernel = np.tanh(np.pi * (0.5 - u)) + delta
        coupling = D_uniform * scale * kernel * coupling_lap
        dv = v - v ** 3 / cubic - w + coupling - alpha * np.tanh(v)
        dw = eps * (v + a - b * w)
        # NOTE: sigma_social term omitted here to match the SPICE netlist (see B_du comment).
        # This is the truncated-doubt model used for the structural dead-zone probe only.
        du = eps_u * (sigma_base - u)
        v = v + dv * dt
        w = w + dw * dt
        u = u + du * dt
        t_hist[k] = k * dt
        v_hist[k] = v

    return t_hist, v_hist


# ---------- Entropy on cognitive bins (matches Mem4Network.calculate_entropy) ----------

def cognitive_entropy(v: np.ndarray) -> float:
    """5-bin Shannon entropy on KIMI cognitive thresholds [-1.2, -0.4, 0.4, 1.2].
    Corrected 2026-04-25 (Audit Manus): replaced obsolete pre-KIMI bins
    [-1.5, -0.8, 0.8, 1.5] with current KIMI bins matching calculate_cognitive_entropy()
    in src/mem4ristor/metrics.py.
    """
    bin_edges = [-np.inf, -1.2, -0.4, 0.4, 1.2, np.inf]
    counts, _ = np.histogram(v, bins=bin_edges)
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def h_stable(v_hist: np.ndarray, tail_frac: float = TAIL_FRAC) -> float:
    n_steps = v_hist.shape[0]
    tail = int(n_steps * (1 - tail_frac))
    return float(np.mean([cognitive_entropy(v_hist[k]) for k in range(tail, n_steps)]))


# ---------- Main ----------

def main() -> int:
    print("=" * 80)
    print(f"SPICE dead-zone test  (BA m={M_BA}, N={N}, t_end={T_END}, dt={DT})")
    print("=" * 80)

    adj = make_ba(N, m=M_BA, seed=SEED)
    deg = adj.sum(axis=1)
    print(f"adjacency: edges={int(adj.sum()/2)}, "
          f"deg min/mean/max = {int(deg.min())}/{deg.mean():.1f}/{int(deg.max())}")

    rng = np.random.RandomState(SEED)
    init_v = rng.uniform(-1.0, 1.0, N)
    heretic = make_heretic_mask(N, HERETIC_RATIO, SEED)
    print(f"heretics: {int(heretic.sum())}/{N} ({100*heretic.sum()/N:.1f}%)")
    print()

    print(f"{'norm':<16} {'spice_t':>9} {'H_spice':>8} {'H_python':>9}  {'|delta|':>8}")
    print("-" * 80)

    rows = []
    for norm in NORMS:
        scale = compute_scale_factors(adj, norm)
        tag = f"BA_m{M_BA}_N{N}_{norm}"

        _, netlist_path = generate_netlist(adj, scale, heretic, init_v, T_END, DT, tag)
        spice_secs = run_ngspice(netlist_path)

        t_sp, v_sp = parse_wrdata(RESULTS / f"dead_zone_{tag}.dat")
        t_py, v_py = python_reference(adj, scale, init_v, T_END, DT)

        h_sp = h_stable(v_sp)
        h_py = h_stable(v_py)
        delta = abs(h_sp - h_py)
        rows.append((norm, spice_secs, h_sp, h_py, delta))
        print(f"{norm:<16} {spice_secs:>8.1f}s {h_sp:>8.3f} {h_py:>9.3f}  {delta:>8.3f}")

    print("-" * 80)
    # Verdict
    py_dead_zone = all(r[3] < 0.10 for r in rows)
    sp_dead_zone = all(r[2] < 0.10 for r in rows)
    print()
    print(f"Python dead zone present (all H<0.10): {py_dead_zone}")
    print(f"SPICE  dead zone present (all H<0.10): {sp_dead_zone}")
    if py_dead_zone and sp_dead_zone:
        print("=> Dead zone is INTRINSIC to the dynamics (confirmed in analog).")
    elif py_dead_zone and not sp_dead_zone:
        print("=> Dead zone is a NUMERICAL ARTIFACT (analog hardware would escape it).")
    else:
        print("=> Mixed result — see per-norm rows.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
