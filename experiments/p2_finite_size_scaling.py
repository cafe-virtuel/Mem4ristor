#!/usr/bin/env python3
"""
P2-7 — Finite-size scaling of the dead zone transition (PROJECT_STATUS §10 P2 item 7).

Motivation:
  The dead zone at BA m>=5 was identified on N=100. Is the critical threshold
  m_crit(N) or equivalently lambda2_crit(N) stable as N grows?

  If lambda2_crit is independent of N → thermodynamic limit exists → publishable
  scaling law.
  If lambda2_crit shifts with N → finite-size effect → the dead zone may
  disappear or narrow at large N.

Protocol:
  - N ∈ {100, 400, 1600} (6400 if time permits — skip if N=1600 already slow)
  - For each N: BA m ∈ {1, 2, 3, 4, 5, 6, 8, 10} with degree_linear norm
  - 3 seeds, 2000 steps (reduced for large N), TAIL_FRAC=0.25
  - Compute H_stable (continuous 100-bin entropy) and lambda2 per instance
  - Plot lambda2 vs H_stable for each N, overlay critical lambda2_crit
    (defined as lambda2 where H_stable drops below 0.1)
  - Use scipy.sparse for N > 1000 (auto via Mem4Network if N>1000)

Output:
  figures/p2_finite_size_scaling.png   (lambda2 vs H, one curve per N)
  figures/p2_finite_size_scaling.csv   (raw data)

Created: 2026-04-24 (Claude Sonnet 4.6, P2-7).
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mem4ristor.topology import Mem4Network          # noqa: E402
from mem4ristor.metrics import calculate_continuous_entropy  # noqa: E402

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

# N sizes to test (6400 requires ~40 min, skip unless time allows)
N_SIZES = [100, 400, 1600]

BA_MS = [1, 2, 3, 4, 5, 6, 8, 10]

STEPS_BY_N = {100: 3000, 400: 2000, 1600: 1000}
SEEDS = [0, 1, 2]  # TODO(A2): increase to [0, 1, 2, 3, 4] for publication — heavy (N=1600 runs)
TAIL_FRAC = 0.25
TRACE_STRIDE = 10

H_THRESHOLD = 0.10   # below this = dead zone

FIG_PATH = ROOT / "figures" / "p2_finite_size_scaling.png"
CSV_PATH = ROOT / "figures" / "p2_finite_size_scaling.csv"


# ──────────────────────────────────────────────────────────────────────────────
# BA adjacency generator (numpy, works for all N)
# ──────────────────────────────────────────────────────────────────────────────

def make_ba_adjacency(n: int, m: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    adj = np.zeros((n, n), dtype=np.float32)  # float32 to save RAM at N=1600
    for i in range(m + 1):
        for j in range(i + 1, m + 1):
            adj[i, j] = adj[j, i] = 1.0
    degrees = adj.sum(axis=1)
    for new_node in range(m + 1, n):
        probs = degrees[:new_node] / degrees[:new_node].sum()
        targets = rng.choice(new_node, size=m, replace=False, p=probs)
        for t in targets:
            adj[new_node, t] = adj[t, new_node] = 1.0
        degrees = adj.sum(axis=1)
    return adj.astype(np.float64)


def compute_lambda2(adj: np.ndarray) -> float:
    """Fiedler value via scipy sparse eigsh for efficiency."""
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import eigsh
        n = adj.shape[0]
        deg = adj.sum(axis=1)
        L = np.diag(deg) - adj
        L_sp = csr_matrix(L)
        vals = eigsh(L_sp, k=2, which='SM', return_eigenvectors=False)
        return float(sorted(vals)[1])
    except Exception:
        # Fallback: dense
        deg = adj.sum(axis=1)
        L = np.diag(deg) - adj
        evals = np.sort(np.linalg.eigvalsh(L))
        return float(evals[1]) if len(evals) > 1 else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Core runner
# ──────────────────────────────────────────────────────────────────────────────

def run_one(n: int, m: int, seed: int) -> dict:
    steps = STEPS_BY_N.get(n, 1000)
    adj = make_ba_adjacency(n, m, seed)
    lambda2 = compute_lambda2(adj)

    net = Mem4Network(
        adjacency_matrix=adj,
        heretic_ratio=0.15,
        coupling_norm='degree_linear',
        seed=seed,
    )

    v_history: list[np.ndarray] = []
    for step in range(steps):
        net.step(I_stimulus=0.0)
        if step % TRACE_STRIDE == 0:
            v_history.append(net.model.v.copy())

    v_arr = np.array(v_history)
    cut = int(len(v_history) * (1.0 - TAIL_FRAC))
    v_tail = v_arr[cut:].flatten()
    H_stable = calculate_continuous_entropy(v_tail)

    return {
        "N": n, "m": m, "seed": seed,
        "lambda2": lambda2,
        "H_stable": H_stable,
        "steps": steps,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    all_rows: list[dict] = []
    total = len(N_SIZES) * len(BA_MS) * len(SEEDS)
    count = 0

    for n in N_SIZES:
        for m in BA_MS:
            for seed in SEEDS:
                t_run = time.time()
                r = run_one(n, m, seed)
                dt = time.time() - t_run
                count += 1
                print(
                    f"[{count:>3}/{total}] N={n:>4}  m={m:>2}  seed={seed}  "
                    f"l2={r['lambda2']:6.3f}  H={r['H_stable']:5.3f}  ({dt:.1f}s)"
                )
                all_rows.append(r)

    # ── CSV ───────────────────────────────────────────────────────────────
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["N", "m", "seed", "lambda2", "H_stable", "steps"]
        )
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n[csv] {CSV_PATH}")

    # ── Aggregate: mean over seeds ────────────────────────────────────────
    # agg[N][m] = (mean_lambda2, mean_H, std_H)
    agg: dict[int, dict[int, tuple]] = {n: {} for n in N_SIZES}
    for n in N_SIZES:
        for m in BA_MS:
            rows_nm = [r for r in all_rows if r["N"] == n and r["m"] == m]
            l2s = [r["lambda2"] for r in rows_nm]
            hs  = [r["H_stable"] for r in rows_nm]
            agg[n][m] = (float(np.mean(l2s)), float(np.mean(hs)), float(np.std(hs, ddof=1)))

    # ── Critical lambda2: find lambda2 where H drops below threshold ───────
    print(f"\n=== Critical lambda2 (H_threshold={H_THRESHOLD}) by N ===")
    l2_crits: dict[int, float] = {}
    for n in N_SIZES:
        # Sort by lambda2
        pairs = sorted([(agg[n][m][0], agg[n][m][1]) for m in BA_MS])
        l2_arr = np.array([p[0] for p in pairs])
        H_arr  = np.array([p[1] for p in pairs])
        # Find transition: last m where H > threshold, interpolate
        above = H_arr >= H_THRESHOLD
        if above.any() and (~above).any():
            # Interpolate between last above and first below
            idx_last_above = np.where(above)[0][-1]
            idx_first_below = np.where(~above)[0][0]
            if idx_first_below > idx_last_above:
                l2a, Ha = l2_arr[idx_last_above], H_arr[idx_last_above]
                l2b, Hb = l2_arr[idx_first_below], H_arr[idx_first_below]
                # Linear interpolation to find where H=threshold
                t = (H_THRESHOLD - Ha) / (Hb - Ha)
                l2_crit = l2a + t * (l2b - l2a)
            else:
                l2_crit = float("nan")
        elif above.all():
            l2_crit = float("inf")   # never dead zone at this N
        else:
            l2_crit = float("nan")   # always dead zone
        l2_crits[n] = l2_crit
        print(f"  N={n:>4}: lambda2_crit ≈ {l2_crit:.3f}")

    # Stability check
    crits = [l2_crits[n] for n in N_SIZES if not np.isnan(l2_crits[n]) and not np.isinf(l2_crits[n])]
    if len(crits) >= 2:
        spread = max(crits) - min(crits)
        print(f"\n  Spread of lambda2_crit: {spread:.3f} (< 0.5 → stable)")
        if spread < 0.5:
            print("  → STABLE TRANSITION — scaling law publishable")
        else:
            print("  → FINITE-SIZE EFFECT — transition shifts with N")

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n{'N':>5}  {'m':>3}  {'lambda2':>8}  {'H_mean':>8}  {'H_std':>7}")
    print("-" * 40)
    for n in N_SIZES:
        for m in BA_MS:
            l2, H, Hs = agg[n][m]
            marker = " ← DEAD" if H < H_THRESHOLD else ""
            print(f"{n:>5}  {m:>3}  {l2:8.3f}  {H:8.4f}  {Hs:7.4f}{marker}")
        print()

    # ── Figure ────────────────────────────────────────────────────────────
    colors_N = {100: "#1f77b4", 400: "#2ca02c", 1600: "#d62728", 6400: "#9467bd"}
    markers_N = {100: "o", 400: "s", 1600: "^", 6400: "D"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: lambda2 vs H_stable (scatter by N)
    ax1 = axes[0]
    for n in N_SIZES:
        l2s = [agg[n][m][0] for m in BA_MS]
        Hs  = [agg[n][m][1] for m in BA_MS]
        Hstds = [agg[n][m][2] for m in BA_MS]
        ax1.errorbar(l2s, Hs, yerr=Hstds,
                     fmt=markers_N[n] + "-",
                     color=colors_N[n], capsize=4,
                     linewidth=1.8, markersize=7,
                     label=f"N={n}")
        # Annotate m values
        for m, l2, H in zip(BA_MS, l2s, Hs):
            ax1.annotate(f"m={m}", (l2, H), fontsize=6,
                         xytext=(3, 3), textcoords="offset points")

    ax1.axhline(H_THRESHOLD, ls="--", color="k", lw=1.5, alpha=0.6,
                label=f"H threshold = {H_THRESHOLD}")
    ax1.set_xlabel("lambda2 (Fiedler value)", fontsize=11)
    ax1.set_ylabel("H_stable (continuous 100-bin, mean over 3 seeds)", fontsize=10)
    ax1.set_title("Finite-size scaling: lambda2 vs H_stable\nby network size N", fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=9)

    # Panel 2: lambda2_crit vs N (stability plot)
    ax2 = axes[1]
    ns_valid = [n for n in N_SIZES
                if not np.isnan(l2_crits[n]) and not np.isinf(l2_crits[n])]
    l2c_valid = [l2_crits[n] for n in ns_valid]
    if len(ns_valid) >= 2:
        ax2.plot(ns_valid, l2c_valid, "ko-", markersize=10, linewidth=2)
        for n, l2c in zip(ns_valid, l2c_valid):
            ax2.annotate(f"N={n}\nl2c={l2c:.2f}", (n, l2c),
                         fontsize=8, ha="left",
                         xytext=(5, 5), textcoords="offset points")
        # Check if flat (publishable scaling)
        if len(ns_valid) >= 3:
            r_fit, p_fit = stats.pearsonr(np.log10(ns_valid), l2c_valid)
            ax2.set_title(
                f"lambda2_crit(N) — stability check\n"
                f"r(log N, l2c) = {r_fit:+.3f}, p = {p_fit:.3e}\n"
                f"{'STABLE → scaling law' if abs(r_fit) < 0.5 else 'SHIFTS → finite-size effect'}",
                fontsize=9,
            )
        else:
            ax2.set_title("lambda2_crit(N) — stability check", fontsize=9)
    else:
        ax2.text(0.5, 0.5, "Not enough data for stability plot",
                 ha="center", va="center", transform=ax2.transAxes)
        ax2.set_title("lambda2_crit(N)", fontsize=9)

    ax2.set_xlabel("N (network size)", fontsize=11)
    ax2.set_ylabel("lambda2_crit", fontsize=11)
    ax2.grid(alpha=0.3)

    fig.suptitle(
        "P2-7 — Finite-size scaling of the dead zone transition\n"
        "(BA m sweep, degree_linear norm, 3 seeds per point)",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=140)
    print(f"\n[png] {FIG_PATH}")
    print(f"Total wall time: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
