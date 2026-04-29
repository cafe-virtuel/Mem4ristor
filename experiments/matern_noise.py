#!/usr/bin/env python3
"""
[12] Bruit spatialement correle (Matern) — dead zone BA m=5
============================================================
Question : Est-ce qu'un bruit dont la correlation suit la geometrie du graphe
(bruit Matern) brise la dead zone differemment qu'un bruit independant ?

Hypothese :
  - Bruit non-correle (baseline) : chaque noeud recoit un bruit independant.
    -> Stochastic resonance classique.
  - Bruit Matern ell court (ell=1) : voisins immediats partageant le meme bruit.
    -> Perturbations localement coherentes.
  - Bruit Matern ell long (ell=4) : vastes clusters partageant le meme bruit.
    -> Perturbations globalement coherentes sur des sous-graphes.

Si le bruit Matern brise la dead zone a amplitude PLUS FAIBLE que le bruit
independant, c'est une propriete utile pour le hardware (les memristors ont
naturellement des correlations spatiales par proximity sur le chip).

Protocole :
  - Topologie : BA m=5, N=100 (dead zone confirmee)
  - 4 types de bruit : uncorrelated / Matern exp (ell=1) / Matern exp (ell=3)
                        / Matern gaussien (ell=3)
  - Amplitudes : eta in {0.1, 0.2, 0.3, 0.5}
  - 5 seeds, 2000 steps (stride=10 -> T=200 snapshots)
  - Metrique : H_cont (100-bin continuous entropy) -- la seule fiable en dead zone

Outputs :
  figures/matern_noise.csv         -- resultats bruts
  figures/matern_noise_summary.csv -- synthese par type x amplitude
  figures/matern_noise.png         -- heatmap H_cont(eta, bruit_type)

Created: 2026-04-30 (Claude Sonnet 4.6 via Antigravity, item [12])
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mem4ristor.topology import Mem4Network          # noqa: E402
from mem4ristor.metrics import calculate_continuous_entropy  # noqa: E402
from mem4ristor.graph_utils import make_ba            # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

N_NODES      = 100
BA_M         = 5           # dead zone topology
STEPS        = 2000
TRACE_STRIDE = 10          # T = 200 snapshots
WARMUP_FRAC  = 0.25
N_SEEDS      = 5

ETAS = [0.1, 0.2, 0.3, 0.5]   # noise amplitude

NOISE_TYPES = [
    # (name, kind, ell)
    ("uncorrelated",    "uncorr",   None),
    ("matern_exp_l1",   "exp",      1.0),
    ("matern_exp_l3",   "exp",      3.0),
    ("matern_gauss_l3", "gauss",    3.0),
]

FIG_DIR  = ROOT / "figures"
CSV_PATH = FIG_DIR / "matern_noise.csv"
SUM_PATH = FIG_DIR / "matern_noise_summary.csv"
FIG_PATH = FIG_DIR / "matern_noise.png"

# ─────────────────────────────────────────────────────────────────────────────
# Matern covariance builders
# ─────────────────────────────────────────────────────────────────────────────

def graph_distances(adj: np.ndarray) -> np.ndarray:
    """Shortest-path distances (unweighted). Disconnected pairs -> inf."""
    A = (adj > 0).astype(float)
    return shortest_path(csr_matrix(A), directed=False, unweighted=True)


def matern_cholesky(dist: np.ndarray, kind: str, ell: float,
                    eps: float = 1e-4) -> np.ndarray:
    """
    Build matrix L s.t. noise = L @ randn(N) ~ N(0, C).
    Uses spectral decomposition + eigenvalue clipping for PD guarantee
    (graph distances don't always produce PD Matern matrices).
    Disconnected pairs (dist=inf) treated as zero correlation.
    """
    N = dist.shape[0]
    D = np.where(np.isinf(dist), 1e6, dist)   # inf -> very large distance

    if kind == "exp":
        C = np.exp(-D / ell)
    elif kind == "gauss":
        C = np.exp(-(D ** 2) / (2 * ell ** 2))
    else:
        raise ValueError(kind)

    # Symmetrize + regularize
    C = (C + C.T) / 2.0
    C += eps * np.eye(N)

    # Spectral decomposition -- clip negative eigenvalues for PD guarantee
    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals = np.maximum(eigvals, eps)   # clip to small positive
    # L = V * sqrt(lambda) -- so that L @ L.T = C (approximately)
    return eigvecs * np.sqrt(eigvals)[np.newaxis, :]


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_one(adj: np.ndarray, dist: np.ndarray, noise_kind: str, ell,
            eta: float, seed: int) -> float:
    """Run one cell, return H_cont on the stable tail."""
    rng = np.random.RandomState(seed + 1000)

    net = Mem4Network(adjacency_matrix=adj, heretic_ratio=0.15,
                      coupling_norm='degree_linear', seed=seed)

    # Pre-compute Cholesky factor if correlated
    if noise_kind != "uncorr":
        L = matern_cholesky(dist, noise_kind, ell)
    else:
        L = None

    snapshots: list[np.ndarray] = []
    warmup = int(STEPS * WARMUP_FRAC)

    for step in range(STEPS):
        # Sample spatially correlated noise
        if L is not None:
            noise = eta * (L @ rng.randn(N_NODES))
        else:
            noise = eta * rng.randn(N_NODES)

        # Inject as per-node stimulus offset
        net.step(I_stimulus=0.0)
        # Add noise directly to voltages (post-step perturbation)
        net.model.v += noise * net.model.dt

        if step >= warmup and step % TRACE_STRIDE == 0:
            snapshots.append(net.model.v.copy())

    v_arr = np.array(snapshots).flatten()
    return float(calculate_continuous_entropy(v_arr))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    total = len(NOISE_TYPES) * len(ETAS) * N_SEEDS
    print(f"[12] Matern noise : {len(NOISE_TYPES)} types x "
          f"{len(ETAS)} etas x {N_SEEDS} seeds = {total} runs")
    print(f"BA m={BA_M}, N={N_NODES}, steps={STEPS}, "
          f"stride={TRACE_STRIDE} -> T={STEPS//TRACE_STRIDE}\n")

    # Pre-compute adjacency + distances per seed
    print("Pre-computing graph distances for 5 seeds...", flush=True)
    adjs, dists = [], []
    for s in range(N_SEEDS):
        adj = make_ba(N_NODES, BA_M, s)
        adjs.append(adj)
        dists.append(graph_distances(adj))
    print("  done.\n")

    rows: list[dict] = []
    agg: dict = {}   # (noise_name, eta) -> list of H_cont

    run_idx = 0
    for noise_name, noise_kind, ell in NOISE_TYPES:
        for eta in ETAS:
            key = (noise_name, eta)
            agg[key] = []
            for seed in range(N_SEEDS):
                run_idx += 1
                h = run_one(adjs[seed], dists[seed], noise_kind, ell,
                            eta, seed)
                agg[key].append(h)
                rows.append({
                    "noise_type": noise_name,
                    "noise_kind": noise_kind,
                    "ell":        ell if ell is not None else "N/A",
                    "eta":        eta,
                    "seed":       seed,
                    "H_cont":     h,
                })
                elapsed = time.time() - t0
                pct     = run_idx / total
                eta_t   = (elapsed / pct) * (1 - pct) if pct > 0 else 0
                print(f"[{run_idx:3d}/{total}] {noise_name:22s} "
                      f"eta={eta:.1f} seed={seed} "
                      f"| H_cont={h:.3f} | ETA {eta_t:.0f}s", flush=True)

    # ── CSV raw ──────────────────────────────────────────────────────────────
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[csv] {CSV_PATH}  ({len(rows)} lignes)")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("SYNTHESE -- H_cont (mean +/- std, 5 seeds)")
    print("="*70)
    print(f"{'type':22s}  {'eta':5s}  mean    std    escape?")
    print("-"*70)

    # Baseline dead zone (no noise)
    print(f"{'no_noise':22s}  {'0.0':5s}  ~1.40   --     NO  (baseline §3septvicies)")

    summary_rows = []
    for noise_name, noise_kind, ell in NOISE_TYPES:
        for eta in ETAS:
            key  = (noise_name, eta)
            data = np.array(agg[key])
            mean = data.mean()
            std  = data.std(ddof=1)
            # Escape threshold: H_cont > 2.0 bits (well above dead zone ~1.4)
            escaped = "YES" if mean > 2.0 else ("~" if mean > 1.7 else "NO ")
            print(f"{noise_name:22s}  {eta:.1f}    {mean:.3f}  {std:.3f}  {escaped}")
            summary_rows.append({
                "noise_type": noise_name, "eta": eta,
                "H_cont_mean": mean, "H_cont_std": std,
                "escaped": escaped.strip(),
            })

    with SUM_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"[csv] {SUM_PATH}")

    # ── Figure : heatmap H_cont(noise_type, eta) ─────────────────────────────
    noise_names = [n for n, _, _ in NOISE_TYPES]
    H_matrix = np.array([
        [np.mean(agg[(n, e)]) for e in ETAS]
        for n in noise_names
    ])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Heatmap
    ax = axes[0]
    im = ax.imshow(H_matrix, aspect='auto', cmap='RdYlGn',
                   vmin=1.0, vmax=5.0)
    ax.set_xticks(range(len(ETAS)))
    ax.set_xticklabels([f"eta={e}" for e in ETAS])
    ax.set_yticks(range(len(noise_names)))
    ax.set_yticklabels(noise_names)
    ax.set_title("[12] H_cont (dead zone escape)\nGreen = diverse, Red = dead zone")
    plt.colorbar(im, ax=ax, label="H_cont (bits)")
    for i in range(len(noise_names)):
        for j in range(len(ETAS)):
            ax.text(j, i, f"{H_matrix[i,j]:.2f}",
                    ha='center', va='center', fontsize=9,
                    color='black' if 1.5 < H_matrix[i,j] < 4.0 else 'white')

    # Line plot
    ax = axes[1]
    colors_n = ['#666666', '#1f77b4', '#ff7f0e', '#9467bd']
    for idx, (noise_name, _, _) in enumerate(NOISE_TYPES):
        means = [np.mean(agg[(noise_name, e)]) for e in ETAS]
        stds  = [np.std(agg[(noise_name, e)], ddof=1) for e in ETAS]
        ax.errorbar(ETAS, means, yerr=stds, marker='o',
                    label=noise_name, color=colors_n[idx], capsize=5)
    ax.axhline(1.40, ls='--', color='red', alpha=0.5, label='dead zone baseline (~1.40)')
    ax.axhline(2.00, ls='--', color='green', alpha=0.5, label='escape threshold (2.0)')
    ax.set_xlabel("Noise amplitude (eta)")
    ax.set_ylabel("H_cont (bits)")
    ax.set_title("H_cont vs eta by noise type\n(BA m=5 dead zone)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"[12] Matern Spatially Correlated Noise — BA m={BA_M}, N={N_NODES}, "
        f"{N_SEEDS} seeds",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=140)
    print(f"\n[png] {FIG_PATH}")
    print(f"\nTotal wall time: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
