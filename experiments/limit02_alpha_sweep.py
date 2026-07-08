#!/usr/bin/env python3
"""
LIMIT-02 Power-Law Normalization Sweep -- FUNCTIONAL metrics (A5, 2026-07-08).

Teste D/deg(i)^gamma pour gamma in [0,1] sur BA m in {2..10} : un exposant
intermediaire franchit-il la dead zone ? (resultat negatif du preprint,
Table tab:alpha_sweep).

CHANGEMENT A5 (backlog docs/FUTURE_WORK.md). Avant : le regime etait juge par
H_cog (5 bins, artefact) et H_cont (100 bins). PROBLEME decouvert le 08/07 :
en regime endogene H_cont CONFOND diversite fonctionnelle et bruit decorrele
(a gamma=0 haut m, H_cont monte a ~3.2 avec variance ~0.5 = comportement
bimodal par seed, pas de la structure). H_cont n'est donc PAS un remplacant
propre de H_cog ici. On re-mesure le regime sur la metrique FONCTIONNELLE du
papier : la synchronie de paires (Pearson sur les trajectoires v(t)), deja la
"primary metric" de tab:ablations et le coeur du resultat FROZEN_U (A2).
  - synchronie ~ 1  : consensus (dead zone fonctionnelle)
  - synchronie ~ 0  : trajectoires independantes (diversite maintenue)
On garde LZ (structured vs chaotic) pour distinguer diversite structuree d'un
chaos decorrele, et H_cont/H_cog pour reference/continuite.

Protocole : N=100, 10 seeds, 3000 steps, I_stim=0 (endogene, heretiques
inactifs), COLD START (v=w=0, conforme au protocole revendique, cf. A4/L109).
Cree : 2026-04-10 (Antigravity). Re-mesure fonctionnelle : 2026-07-08 (Opus 4.8, A5).
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

N_BA   = 100
STEPS  = 3000
TAIL_FRAC = 0.25
SEEDS  = [42, 123, 777, 17, 256, 1337, 99, 314, 2024, 888]  # n=10 (set canonique)
I_STIM = 0.0
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
M_TARGETS = [2, 3, 4, 5, 6, 8, 10]


def make_ba(n, m, seed):
    rng = np.random.RandomState(seed)
    adj = np.zeros((n, n), dtype=float)
    for i in range(m + 1):
        for j in range(i + 1, m + 1):
            adj[i, j] = adj[j, i] = 1.0
    degrees = np.sum(adj, axis=1)
    for new_node in range(m + 1, n):
        probs = degrees[:new_node] / degrees[:new_node].sum()
        targets = rng.choice(new_node, size=m, replace=False, p=probs)
        for t in targets:
            adj[new_node, t] = adj[t, new_node] = 1.0
        degrees = np.sum(adj, axis=1)
    return adj


def run_experiment(adj, alpha, seed, steps=STEPS):
    net = Mem4Network(
        adjacency_matrix=adj.copy(), heretic_ratio=0.15,
        coupling_norm='degree_power', seed=seed, cold_start=True,
    )
    net.degree_power_alpha = alpha
    net._compute_coupling_weights()

    tail_start = int(steps * (1 - TAIL_FRAC))
    v_tail = []
    for step in range(steps):
        net.step(I_stimulus=I_STIM)
        if step >= tail_start:
            v_tail.append(net.v.copy())
    V = np.asarray(v_tail)          # (T_tail=750, N)
    V_s = V[::10]                    # sous-echantillon pour entropie
    sync = calculate_pairwise_synchrony(V)
    lz = calculate_temporal_lz_complexity(V[::5])   # 150 pts, O(n^2) borne
    h_cont = float(np.mean([calculate_continuous_entropy(v) for v in V_s]))
    h_cog = float(np.mean([calculate_cognitive_entropy(v) for v in V_s]))
    return sync, lz, h_cont, h_cog


if __name__ == '__main__':
    print("=" * 92)
    print("LIMIT-02 gamma-SWEEP -- metriques FONCTIONNELLES (sync/LZ) + entropies")
    print(f"N={N_BA} | m={M_TARGETS} | gamma={ALPHAS} | {len(SEEDS)} seeds | cold start | I_stim=0")
    print("=" * 92)
    t0 = time.time()
    results = {}

    for m in M_TARGETS:
        print(f"\n--- BA m={m} ---")
        print(f"  {'gamma':>5}  {'sync':>14}  {'LZ':>8}  {'H_cont':>8}  {'H_cog':>7}")
        for alpha in ALPHAS:
            syncs, lzs, hconts, hcogs = [], [], [], []
            for seed in SEEDS:
                adj = make_ba(N_BA, m, seed)
                s, lz, hc, hg = run_experiment(adj, alpha, seed)
                syncs.append(s); lzs.append(lz); hconts.append(hc); hcogs.append(hg)
            results[(m, alpha)] = {
                'sync_mean': float(np.mean(syncs)), 'sync_std': float(np.std(syncs)),
                'lz_mean': float(np.mean(lzs)), 'lz_std': float(np.std(lzs)),
                'h_cont_mean': float(np.mean(hconts)), 'h_cont_std': float(np.std(hconts)),
                'h_cog_mean': float(np.mean(hcogs)),
            }
            r = results[(m, alpha)]
            print(f"  {alpha:>5.1f}  {r['sync_mean']:>7.3f}+/-{r['sync_std']:.3f}"
                  f"  {r['lz_mean']:>8.3f}  {r['h_cont_mean']:>8.3f}  {r['h_cog_mean']:>7.3f}")

    # --- CSV ---------------------------------------------------------------
    out = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'figures',
                                       'limit02_alpha_sweep.csv'))
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['m', 'alpha', 'sync_mean', 'sync_std', 'lz_mean', 'lz_std',
                    'h_cont_mean', 'h_cont_std', 'h_cog_mean', 'n_seeds'])
        for (m, alpha), r in sorted(results.items()):
            w.writerow([m, alpha, r['sync_mean'], r['sync_std'], r['lz_mean'],
                        r['lz_std'], r['h_cont_mean'], r['h_cont_std'],
                        r['h_cog_mean'], len(SEEDS)])
    print(f"\n[csv] {out}")

    # --- Verdict fonctionnel : la synchronie MINIMALE atteignable par gamma -
    # (le regime fonctionnel = synchronie basse ; la dead zone = synchronie haute
    #  qu'aucun gamma ne fait retomber).
    print(f"\n{'='*92}")
    print("VERDICT (metrique fonctionnelle) : synchronie MIN sur gamma par m")
    print("  regime fonctionnel si un gamma amene sync bas ; dead si sync reste haute")
    print(f"{'m':>4}  {'min sync (gamma*)':>22}  {'LZ au gamma*':>12}  {'H_cog au gamma*':>15}")
    for m in M_TARGETS:
        best_a = min(ALPHAS, key=lambda a: results[(m, a)]['sync_mean'])
        r = results[(m, best_a)]
        print(f"{m:>4}  {r['sync_mean']:>10.3f} (g={best_a:.1f})       "
              f"{r['lz_mean']:>8.3f}      {r['h_cog_mean']:>10.3f}")
    print(f"\nElapsed: {time.time()-t0:.1f}s")
