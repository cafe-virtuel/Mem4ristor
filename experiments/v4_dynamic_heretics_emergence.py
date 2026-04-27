"""
V4 Dynamic Heretics — Emergence Experiment
===========================================
Mesure l'émergence des hérétiques dynamiques (u_i >= 0.8 × 100 steps → bascule)
et compare la dynamique V3 (hérétiques statiques) vs V4 (hérétiques dynamiques).

Métriques relevées :
- Nombre d'hérétiques dynamiques en fonction du temps
- Synchrony (mean |v_i|) avant/après premières bascules
- Entropie cognitive H au cours du temps

Usage :
    python experiments/v4_dynamic_heretics_emergence.py
"""
import sys
import os
import numpy as np
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mem4ristor.topology import Mem4Network

# --- Config ---
N_STEPS       = 5000
N_SEEDS       = 5
GRID_SIZE     = 10   # 10x10 = 100 nodes
I_STIM        = 0.3
HERETIC_RATIO = 0.15

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_scenario(dynamic_enabled: bool, seed: int) -> dict:
    net = Mem4Network(size=GRID_SIZE, heretic_ratio=HERETIC_RATIO, seed=seed)
    net.model.cfg['coupling']['dynamic_heretics'] = {
        'enabled':        dynamic_enabled,
        'u_threshold':    0.8,
        'steps_required': 100,
    }

    timeline = {
        'step':             [],
        'dynamic_heretics': [],
        'total_heretics':   [],
        'synchrony':        [],
        'entropy':          [],
    }

    for t in range(N_STEPS):
        net.step(I_stimulus=I_STIM)

        if t % 50 == 0:
            synchrony = float(np.mean(np.abs(net.model.v)))
            entropy   = net.calculate_entropy()
            timeline['step'].append(t)
            timeline['dynamic_heretics'].append(net.model.dynamic_heretic_count)
            timeline['total_heretics'].append(int(np.sum(net.model.heretic_mask)))
            timeline['synchrony'].append(synchrony)
            timeline['entropy'].append(entropy)

    return timeline


def main():
    results = {'v3': [], 'v4': []}

    for seed in range(N_SEEDS):
        print(f"[V3] seed={seed}...", flush=True)
        results['v3'].append(run_scenario(dynamic_enabled=False, seed=seed))
        print(f"[V4] seed={seed}...", flush=True)
        results['v4'].append(run_scenario(dynamic_enabled=True, seed=seed))

    # --- Agrégation et export CSV ---
    steps = results['v3'][0]['step']

    rows = []
    for t_idx, t in enumerate(steps):
        row = {'step': t}
        for version in ('v3', 'v4'):
            for metric in ('dynamic_heretics', 'total_heretics', 'synchrony', 'entropy'):
                vals = [results[version][s][metric][t_idx] for s in range(N_SEEDS)]
                row[f'{version}_{metric}_mean'] = float(np.mean(vals))
                row[f'{version}_{metric}_std']  = float(np.std(vals))
        rows.append(row)

    csv_path = os.path.join(OUTPUT_DIR, 'v4_dynamic_heretics_emergence.csv')
    fieldnames = list(rows[0].keys())
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV saved -> {csv_path}")

    # --- Résumé terminal ---
    last = rows[-1]
    print(f"\n=== RÉSULTATS FINAUX (step {N_STEPS}, n={N_SEEDS} seeds) ===")
    print(f"V3  total_heretics : {last['v3_total_heretics_mean']:.1f} ± {last['v3_total_heretics_std']:.1f}")
    print(f"V4  total_heretics : {last['v4_total_heretics_mean']:.1f} ± {last['v4_total_heretics_std']:.1f}")
    print(f"V4  dynamic_born   : {last['v4_dynamic_heretics_mean']:.1f} ± {last['v4_dynamic_heretics_std']:.1f}")
    print(f"V3  synchrony      : {last['v3_synchrony_mean']:.4f} ± {last['v3_synchrony_std']:.4f}")
    print(f"V4  synchrony      : {last['v4_synchrony_mean']:.4f} ± {last['v4_synchrony_std']:.4f}")
    print(f"V3  entropy H      : {last['v3_entropy_mean']:.4f} ± {last['v3_entropy_std']:.4f}")
    print(f"V4  entropy H      : {last['v4_entropy_mean']:.4f} ± {last['v4_entropy_std']:.4f}")


if __name__ == '__main__':
    main()
