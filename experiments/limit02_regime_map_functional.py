#!/usr/bin/env python3
"""
LIMIT-02 Regime Map -- FUNCTIONAL metrics (A5, 2026-07-08).

Regenere la "regime map" du preprint (Table tab:ba_m_sweep) : pour chaque
BA m, comparer les deux normalisations extremes uniform (gamma=0) et
degree_linear (gamma=1), et localiser les deux transitions de regime
(uniform->deg-linear a bas m ; entree en dead zone a haut m).

CHANGEMENT A5. La table historique jugeait le regime par H_cog (5 bins,
artefact). On re-mesure sur la metrique FONCTIONNELLE du papier : la
synchronie de paires (Pearson), deja "primary metric" de tab:ablations.
  - sync ~ 1 : consensus (dead zone fonctionnelle)
  - sync ~ 0 : trajectoires independantes (diversite maintenue)
LZ garde la distinction structure/chaos ; H_cont/H_cog en reference.
(H_cont seul induit en erreur en endogene : il confond bruit decorrele et
 diversite -- cf. limit02_alpha_sweep.py, decouverte 08/07.)

Protocole : N=100, 10 seeds, 3000 steps, I_stim=0 (endogene), COLD START.
Cree : 2026-07-08 (Opus 4.8, A5). Source de tab:ba_m_sweep (metrique fonctionnelle).
"""
import sys, os, time, csv
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from mem4ristor.core import Mem4Network
from mem4ristor.metrics import (
    calculate_continuous_entropy,
    calculate_cognitive_entropy,
    calculate_pairwise_synchrony,
    calculate_temporal_lz_complexity,
)
from scipy.linalg import eigh

N = 100
STEPS = 3000
TAIL_FRAC = 0.25
SEEDS = [42, 123, 777, 17, 256, 1337, 99, 314, 2024, 888]
I_STIM = 0.0
M_TARGETS = [1, 2, 3, 4, 5, 6, 7, 8, 10, 15]
NORMS = [("uniform", "uniform"), ("degree_linear", "deg_linear")]


def make_ba(n, m, seed):
    rng = np.random.RandomState(seed)
    adj = np.zeros((n, n), dtype=float)
    for i in range(m + 1):
        for j in range(i + 1, m + 1):
            adj[i, j] = adj[j, i] = 1.0
    degrees = np.sum(adj, axis=1)
    for new_node in range(m + 1, n):
        probs = degrees[:new_node] / degrees[:new_node].sum()
        targets = rng.choice(new_node, size=min(m, new_node), replace=False, p=probs)
        for t in targets:
            adj[new_node, t] = adj[t, new_node] = 1.0
        degrees = np.sum(adj, axis=1)
    return adj


def fiedler(adj):
    d = adj.sum(1)
    return float(np.sort(eigh(np.diag(d) - adj, eigvals_only=True))[1])


def run_one(adj, norm, seed, steps=STEPS):
    net = Mem4Network(size=10, heretic_ratio=0.15, seed=seed,
                      adjacency_matrix=adj.copy(), coupling_norm=norm,
                      cold_start=True)
    tail_start = int(steps * (1 - TAIL_FRAC))
    v_tail = []
    for step in range(steps):
        net.step(I_stimulus=I_STIM)
        if step >= tail_start:
            v_tail.append(net.v.copy())
    V = np.asarray(v_tail)
    V_s = V[::10]
    return {
        'sync': calculate_pairwise_synchrony(V),
        'lz': calculate_temporal_lz_complexity(V[::5]),
        'h_cont': float(np.mean([calculate_continuous_entropy(v) for v in V_s])),
        'h_cog': float(np.mean([calculate_cognitive_entropy(v) for v in V_s])),
    }


if __name__ == '__main__':
    print("=" * 92)
    print("REGIME MAP -- metriques FONCTIONNELLES (sync/LZ), uniform vs degree_linear")
    print(f"N={N} | m={M_TARGETS} | {len(SEEDS)} seeds | cold start | I_stim=0")
    print("=" * 92)
    t0 = time.time()
    rows = []
    print(f"{'m':>3} {'lambda2':>8} | {'sync_uni':>9} {'sync_dl':>9} | "
          f"{'lz_uni':>7} {'lz_dl':>7} | {'winner (lower sync)':>20}")
    print("-" * 92)

    for m in M_TARGETS:
        agg = {'uniform': {'sync': [], 'lz': [], 'h_cont': [], 'h_cog': []},
               'degree_linear': {'sync': [], 'lz': [], 'h_cont': [], 'h_cog': []}}
        l2s = []
        for seed in SEEDS:
            adj = make_ba(N, m, seed)
            l2s.append(fiedler(adj))
            for norm, _ in NORMS:
                r = run_one(adj, norm, seed)
                for k in agg[norm]:
                    agg[norm][k].append(r[k])
        row = {'m': m, 'lambda2': float(np.mean(l2s))}
        for norm, tag in NORMS:
            for k in ['sync', 'lz', 'h_cont', 'h_cog']:
                row[f'{tag}_{k}_mean'] = float(np.mean(agg[norm][k]))
                row[f'{tag}_{k}_std'] = float(np.std(agg[norm][k]))
        rows.append(row)
        su, sd = row['uniform_sync_mean'], row['deg_linear_sync_mean']
        winner = 'deg_linear' if sd < su - 0.03 else ('uniform' if su < sd - 0.03 else 'approx equal')
        print(f"{m:>3} {row['lambda2']:>8.3f} | {su:>9.3f} {sd:>9.3f} | "
              f"{row['uniform_lz_mean']:>7.3f} {row['deg_linear_lz_mean']:>7.3f} | {winner:>20}")

    out = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'figures',
                                       'limit02_regime_map_functional.csv'))
    fields = ['m', 'lambda2']
    for _, tag in NORMS:
        for k in ['sync', 'lz', 'h_cont', 'h_cog']:
            fields += [f'{tag}_{k}_mean', f'{tag}_{k}_std']
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\n[csv] {out}")
    print(f"Elapsed: {time.time()-t0:.1f}s")
