#!/usr/bin/env python3
# @TODO: n=3 seeds (directional result) — relancer à n=10 avant soumission journal (voir preprint.tex Limitations)
"""
Piste A1 — Résonance Stochastique Dirigée (2026-04-24)

Hypothèse : La dead zone (BA m=5) est un état métastable profond.
Un bruit thermique ciblé sur les hubs ou les hérétiques peut induire
une résonance stochastique suffisante pour en sortir — sans noyer le
signal global comme le fait le bruit homogène.

Modes de bruit testés :
  - UNIFORM   : σ_v identique pour tous les nœuds (baseline)
  - HUB       : σ_v ∝ deg(i)^0.5, normalisé à la même énergie totale
  - HERETIC   : σ_v > 0 uniquement sur les nœuds hérétiques (η=0.15)
  - ZERO      : σ_v = 0 (contrôle déterministe)

Métriques : H_cont (100-bin), H_cog (5-bin KIMI), complexité LZ temporelle.
Sweep : amplitude de bruit global σ ∈ [0, 0.50] × 4 modes × 3 seeds.

Script    : experiments/p2_stochastic_resonance_directed.py
Figures   : figures/p2_stochastic_resonance_directed.png
CSV       : figures/p2_stochastic_resonance_directed.csv

Référence : PROJECT_STATUS.md §P2-AUDIT Piste A1
"""
import sys, os, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from mem4ristor.core import Mem4Network
from mem4ristor.graph_utils import make_ba
from mem4ristor.metrics import (
    calculate_cognitive_entropy,
    calculate_temporal_lz_complexity,
)

# ── Paramètres ────────────────────────────────────────────────────────────────
N        = 100
M_BA     = 5          # dead zone cible
STEPS    = 3000
WARM_UP  = int(STEPS * 0.25)   # ignore les STEPS premiers
SEEDS    = [42, 123, 777]
I_STIM   = 0.0        # régime endogène (heretics inactive — documented no-op)
SIGMAS   = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
MODES    = ['ZERO', 'UNIFORM', 'HUB', 'HERETIC']
RECORD_HISTORY_EVERY = 10   # pour LZ (T ≈ 225 snapshots)




def build_sigma_vec(mode, sigma, degrees, heretic_mask):
    """Construit le vecteur σ_v per-nœud selon le mode."""
    N = len(degrees)
    if mode == 'ZERO' or sigma == 0.0:
        return np.zeros(N)

    if mode == 'UNIFORM':
        return np.full(N, sigma)

    if mode == 'HUB':
        # σ_i ∝ sqrt(deg(i)), renormalisé pour que RMS = sigma
        raw = np.sqrt(degrees.astype(float))
        rms_raw = np.sqrt(np.mean(raw ** 2))
        if rms_raw == 0:
            return np.full(N, sigma)
        return raw * (sigma / rms_raw)

    if mode == 'HERETIC':
        # bruit uniquement sur les nœuds hérétiques
        # amplifié pour conserver la même énergie totale en attente
        vec = np.zeros(N)
        n_her = heretic_mask.sum()
        if n_her == 0:
            return np.full(N, sigma)
        # même énergie totale : N * sigma^2 = n_her * sigma_her^2
        sigma_her = sigma * np.sqrt(N / n_her)
        vec[heretic_mask] = sigma_her
        return vec

    raise ValueError(f"Unknown mode: {mode}")


def run_one(adj, mode, sigma, seed):
    net = Mem4Network(
        adjacency_matrix=adj.copy(),
        heretic_ratio=0.15,
        coupling_norm='uniform',   # uniform -> dead zone confirmee
        seed=seed,
    )
    # sigma_v_vec est toujours passe a step() -> cfg['noise']['sigma_v'] ignore
    degrees = adj.sum(axis=1)
    heretic_mask = net.model.heretic_mask
    sigma_vec = build_sigma_vec(mode, sigma, degrees, heretic_mask)

    v_history = []
    for step in range(STEPS):
        net.step(I_stimulus=I_STIM, sigma_v_vec=sigma_vec)
        if step >= WARM_UP and step % RECORD_HISTORY_EVERY == 0:
            v_history.append(net.v.copy())

    v_tail = np.array(v_history)   # shape (T, N)
    h_cont = float(np.mean([
        net.model.calculate_entropy()   # current v at end
        for _ in [None]                 # single snapshot — use queue
    ]))
    # Re-compute on final snapshot directly
    from mem4ristor.metrics import calculate_continuous_entropy
    h_cont = float(np.mean([calculate_continuous_entropy(v) for v in v_tail]))
    h_cog  = float(np.mean([calculate_cognitive_entropy(v) for v in v_tail]))
    lz     = calculate_temporal_lz_complexity(v_tail) if len(v_tail) > 1 else 0.0

    return h_cont, h_cog, lz


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 80)
    print("Piste A1 — Résonance Stochastique DIRIGÉE")
    print(f"BA m={M_BA} N={N} | coupling_norm=uniform (dead zone) | I_stim={I_STIM}")
    print(f"Modes: {MODES}")
    print(f"Sigma: {SIGMAS}")
    print(f"Seeds: {SEEDS}")
    print("=" * 80)

    t0 = time.time()
    rows = []

    for mode in MODES:
        print(f"\n{'-'*60}")
        print(f"MODE: {mode}")
        print(f"  {'sigma':>6}  {'H_cont':>8}  {'H_cog':>8}  {'LZ':>8}")
        for sigma in SIGMAS:
            h_c_list, h_k_list, lz_list = [], [], []
            for seed in SEEDS:
                adj = make_ba(N, M_BA, seed)
                h_c, h_k, lz = run_one(adj, mode, sigma, seed)
                h_c_list.append(h_c)
                h_k_list.append(h_k)
                lz_list.append(lz)
            h_c_m  = np.mean(h_c_list)
            h_k_m  = np.mean(h_k_list)
            lz_m   = np.mean(lz_list)
            star = " ***" if h_k_m > 0.3 else ""
            print(f"  s={sigma:.2f}  H_cont={h_c_m:.4f}  H_cog={h_k_m:.4f}  LZ={lz_m:.4f}{star}")
            rows.append({
                'mode': mode, 'sigma': sigma,
                'h_cont_mean': h_c_m, 'h_cont_std': np.std(h_c_list),
                'h_cog_mean': h_k_m,  'h_cog_std': np.std(h_k_list),
                'lz_mean': lz_m,       'lz_std': np.std(lz_list),
            })

    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed:.1f}s")

    # ── CSV ──────────────────────────────────────────────────────────────────
    import csv, pathlib
    fig_dir = pathlib.Path(__file__).resolve().parents[1] / 'figures'
    fig_dir.mkdir(exist_ok=True)
    csv_path = fig_dir / 'p2_stochastic_resonance_directed.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV sauvegardé : {csv_path}")

    # ── Figure ───────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        colors = {'ZERO': 'gray', 'UNIFORM': 'steelblue', 'HUB': 'darkorange', 'HERETIC': 'crimson'}
        metrics = [
            ('h_cont_mean', 'H_cont (100-bin)', 'Shannon entropy (bits)'),
            ('h_cog_mean',  'H_cog (5-bin KIMI)', 'Cognitive entropy (bits)'),
            ('lz_mean',     'LZ complexity', 'Normalized LZ complexity'),
        ]

        for ax, (key, title, ylabel) in zip(axes, metrics):
            for mode in MODES:
                xs = [r['sigma'] for r in rows if r['mode'] == mode]
                ys = [r[key] for r in rows if r['mode'] == mode]
                ax.plot(xs, ys, marker='o', label=mode, color=colors[mode])
            ax.set_xlabel('σ (noise amplitude)')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)

        fig.suptitle(
            f'Piste A1 — Directed Stochastic Resonance\n'
            f'BA m={M_BA}, coupling=uniform (dead zone), I_stim={I_STIM}, {len(SEEDS)} seeds',
            fontsize=11
        )
        plt.tight_layout()
        png_path = fig_dir / 'p2_stochastic_resonance_directed.png'
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        print(f"Figure sauvegardée : {png_path}")
    except Exception as e:
        print(f"[matplotlib non disponible ou erreur] {e}")
