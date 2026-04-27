"""
V4 Dynamic Heretics — Exploration régime u_thr > 0.9
======================================================
Objectif : trouver la frontière de suppression de la cascade.

Questions :
  1. À partir de quel u_thr la cascade s'effondre-t-elle ?
  2. Quelle est l'amplitude typique de u_i observée dans la dynamique ?
  3. Y a-t-il un régime de transition graduelle (10-90%) ?

Différences avec v4_parametric_sweep.py :
  - Grille u_thr fine dans [0.90, 2.00]
  - Seulement 3 steps_required représentatifs (l'effet est déjà caractérisé)
  - N_SEEDS = 5 pour meilleure statistique dans la zone de transition
  - Diagnostic u_max : on enregistre le max de u_i observé sur toute la simulation

Usage :
    python experiments/v4_high_uthr_sweep.py
"""
import sys
import os
import numpy as np
import csv
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from mem4ristor.topology import Mem4Network

# --- Paramètres ---
U_THRESHOLDS   = [0.90, 0.92, 0.94, 0.96, 0.98, 1.00, 1.05, 1.10, 1.20, 1.50, 2.00]
STEPS_REQUIRED = [10, 50, 200]
N_SEEDS        = 5
N_STEPS        = 3000
GRID_SIZE      = 10
I_STIM         = 0.3
HERETIC_RATIO  = 0.15

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_one(u_threshold: float, steps_req: int, seed: int) -> dict:
    net = Mem4Network(size=GRID_SIZE, heretic_ratio=HERETIC_RATIO, seed=seed)
    net.model.cfg['coupling']['dynamic_heretics'] = {
        'enabled':        True,
        'u_threshold':    u_threshold,
        'steps_required': steps_req,
    }
    N = net.N
    initial_heretics = int(np.sum(net.model.heretic_mask))

    t_first = -1
    u_max_global = 0.0
    heretic_pct_timeline = []

    for t in range(N_STEPS):
        net.step(I_stimulus=I_STIM)

        # Diagnostic u_max : max absolu de u observé sur tous les nœuds
        u_vals = np.abs(net.model.u)
        u_max_global = max(u_max_global, float(np.max(u_vals)))

        dyn = net.model.dynamic_heretic_count
        if t_first == -1 and dyn > 0:
            t_first = t

        if t % 10 == 0:
            heretic_pct_timeline.append(int(np.sum(net.model.heretic_mask)) / N)

    heretic_pct_final  = int(np.sum(net.model.heretic_mask)) / N
    dynamic_born_final = net.model.dynamic_heretic_count
    entropy_final      = float(net.calculate_entropy())
    synchrony_final    = float(np.mean(np.abs(net.model.v)))

    cascade_size = heretic_pct_final - (initial_heretics / N)
    t_sat = -1
    if cascade_size >= 0.05:
        target = (initial_heretics / N) + 0.9 * cascade_size
        for idx, pct in enumerate(heretic_pct_timeline):
            if pct >= target:
                t_sat = idx * 10
                break

    return {
        'heretic_pct_final':  heretic_pct_final,
        'dynamic_born_final': dynamic_born_final,
        'initial_heretics':   initial_heretics,
        't_first':            t_first,
        't_sat':              t_sat,
        'entropy_final':      entropy_final,
        'synchrony_final':    synchrony_final,
        'u_max_observed':     u_max_global,
    }


def main():
    total_runs = len(U_THRESHOLDS) * len(STEPS_REQUIRED) * N_SEEDS
    print(f"Sweep haute zone : {len(U_THRESHOLDS)} seuils x {len(STEPS_REQUIRED)} durees x {N_SEEDS} seeds = {total_runs} runs")
    print(f"N_STEPS={N_STEPS}, grid={GRID_SIZE}x{GRID_SIZE}={GRID_SIZE**2} noeuds\n")

    rows = []
    run_idx = 0
    t0 = time.time()

    for u_thr in U_THRESHOLDS:
        for steps_req in STEPS_REQUIRED:
            seed_results = []
            for seed in range(N_SEEDS):
                run_idx += 1
                r = run_one(u_thr, steps_req, seed)
                seed_results.append(r)

            def agg(key):
                vals = [r[key] for r in seed_results if r[key] != -1]
                valid_mean = float(np.mean(vals)) if vals else -1.0
                all_vals = [r[key] for r in seed_results]
                return float(np.mean(all_vals)), float(np.std(all_vals)), valid_mean

            row = {'u_threshold': u_thr, 'steps_required': steps_req}
            for metric in ('heretic_pct_final', 'dynamic_born_final',
                           'initial_heretics', 't_first', 't_sat',
                           'entropy_final', 'synchrony_final', 'u_max_observed'):
                mean_v, std_v, valid_mean = agg(metric)
                row[f'{metric}_mean'] = round(mean_v, 4)
                row[f'{metric}_std']  = round(std_v, 4)
                if metric in ('t_first', 't_sat'):
                    row[f'{metric}_valid_mean'] = round(valid_mean, 1)

            rows.append(row)

            elapsed = time.time() - t0
            pct_done = run_idx / total_runs
            eta = (elapsed / pct_done) * (1 - pct_done) if pct_done > 0 else 0
            print(f"[{run_idx:3d}/{total_runs}] u_thr={u_thr:.2f} steps={steps_req:4d} "
                  f"| heretic%={row['heretic_pct_final_mean']:.3f} "
                  f"u_max={row['u_max_observed_mean']:.3f} "
                  f"H={row['entropy_final_mean']:.3f} "
                  f"| ETA {eta:.0f}s", flush=True)

    csv_path = os.path.join(OUTPUT_DIR, 'v4_high_uthr_sweep.csv')
    fieldnames = list(rows[0].keys())
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV -> {csv_path}")

    # --- Résumé : frontière de phase (steps=50 fixé) ---
    print("\n" + "="*70)
    print("FRONTIERE DE CASCADE (steps_required=50, n=5 seeds)")
    print(f"{'u_thr':>6} | {'heretic%':>9} | {'u_max_obs':>10} | {'H_final':>8} | {'t_first':>8} | {'cascade?':>9}")
    print("-" * 70)
    for u_thr in U_THRESHOLDS:
        r = next(x for x in rows if x['u_threshold'] == u_thr and x['steps_required'] == 50)
        pct = r['heretic_pct_final_mean']
        u_max = r['u_max_observed_mean']
        h = r['entropy_final_mean']
        tf = r['t_first_valid_mean']
        cascade = "OUI" if pct > 0.20 else "NON"
        tf_str = f"{tf:.0f}" if tf >= 0 else "--"
        print(f"{u_thr:>6.2f} | {pct:>9.3f} | {u_max:>10.3f} | {h:>8.3f} | {tf_str:>8} | {cascade:>9}")

    print("\n" + "="*70)
    print("DIAGNOSTIC u_max vs u_thr (montre pourquoi la cascade s'arrete)")
    print("Si u_max_observed < u_thr : le seuil est inatteignable => cascade impossible")
    print("="*70)
    uthr_seen = []
    for r in rows:
        if r['steps_required'] == 50 and r['u_threshold'] not in uthr_seen:
            uthr_seen.append(r['u_threshold'])
            u_max = r['u_max_observed_mean']
            u_thr = r['u_threshold']
            gap = u_max - u_thr
            reachable = "ATTEIGNABLE" if gap >= 0 else "INACCESSIBLE"
            print(f"  u_thr={u_thr:.2f} | u_max_obs={u_max:.3f} | gap={gap:+.3f} => {reachable}")

    print("\n" + "="*70)
    print("CARTE COMPLETE heretic% (tous steps_required)")
    header = f"{'u_thr':>6} | " + " | ".join(f"s={s:>3}" for s in STEPS_REQUIRED)
    print(header)
    print("-" * len(header))
    for u_thr in U_THRESHOLDS:
        line = f"{u_thr:>6.2f} | "
        cells = []
        for steps_req in STEPS_REQUIRED:
            r = next(x for x in rows if x['u_threshold'] == u_thr and x['steps_required'] == steps_req)
            pct = r['heretic_pct_final_mean']
            cells.append(f"{pct:.3f}")
        line += " | ".join(cells)
        print(line)


if __name__ == '__main__':
    main()
