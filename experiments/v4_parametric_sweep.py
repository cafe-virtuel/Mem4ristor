"""
V4 Dynamic Heretics — Parametric Sweep
=======================================
Sweep 2D : u_threshold x steps_required
Objectif : trouver la règle de fonctionnement (frontière de phase)

Métriques :
- heretic_pct_final  : % du réseau devenu hérétique à t_final
- dynamic_born_final : nombre d'hérétiques nés dynamiquement
- t_first            : step de première bascule (-1 si aucune)
- t_sat              : step où heretic_pct atteint 90% du max (-1 si cascade < 5%)
- entropy_final      : entropie H à t_final
- synchrony_final    : synchronie moyenne à t_final

Usage :
    python experiments/v4_parametric_sweep.py
"""
import sys
import os
import numpy as np
import csv
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from mem4ristor.topology import Mem4Network

# --- Paramètres du sweep ---
U_THRESHOLDS   = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
STEPS_REQUIRED = [10, 25, 50, 100, 200, 500]
N_SEEDS        = 3
N_STEPS        = 3000
GRID_SIZE      = 10   # 10x10 = 100 nodes
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
    prev_dynamic = 0
    heretic_pct_timeline = []

    for t in range(N_STEPS):
        net.step(I_stimulus=I_STIM)

        dyn = net.model.dynamic_heretic_count
        if t_first == -1 and dyn > 0:
            t_first = t

        if t % 10 == 0:
            heretic_pct_timeline.append(int(np.sum(net.model.heretic_mask)) / N)

    heretic_pct_final  = int(np.sum(net.model.heretic_mask)) / N
    dynamic_born_final = net.model.dynamic_heretic_count
    entropy_final      = float(net.calculate_entropy())
    synchrony_final    = float(np.mean(np.abs(net.model.v)))

    # t_sat : step où la cascade atteint 90% de sa valeur finale
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
    }


def main():
    total_runs = len(U_THRESHOLDS) * len(STEPS_REQUIRED) * N_SEEDS
    print(f"Sweep 2D : {len(U_THRESHOLDS)} seuils x {len(STEPS_REQUIRED)} durees x {N_SEEDS} seeds = {total_runs} runs")
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

            # Agréger sur les seeds
            def agg(key):
                vals = [r[key] for r in seed_results if r[key] != -1]
                valid_mean = float(np.mean(vals)) if vals else -1.0
                all_vals = [r[key] for r in seed_results]
                return float(np.mean(all_vals)), float(np.std(all_vals)), valid_mean

            row = {
                'u_threshold':          u_thr,
                'steps_required':       steps_req,
            }
            for metric in ('heretic_pct_final', 'dynamic_born_final',
                           'initial_heretics', 't_first', 't_sat',
                           'entropy_final', 'synchrony_final'):
                mean_v, std_v, valid_mean = agg(metric)
                row[f'{metric}_mean'] = round(mean_v, 4)
                row[f'{metric}_std']  = round(std_v,  4)
                if metric in ('t_first', 't_sat'):
                    row[f'{metric}_valid_mean'] = round(valid_mean, 1)

            rows.append(row)

            elapsed = time.time() - t0
            pct_done = run_idx / total_runs
            eta = (elapsed / pct_done) * (1 - pct_done) if pct_done > 0 else 0
            print(f"[{run_idx:3d}/{total_runs}] u_thr={u_thr:.1f} steps={steps_req:4d} "
                  f"| heretic%={row['heretic_pct_final_mean']:.2f} "
                  f"t_first={row['t_first_valid_mean']:.0f} "
                  f"H={row['entropy_final_mean']:.3f} "
                  f"| ETA {eta:.0f}s", flush=True)

    # Export CSV
    csv_path = os.path.join(OUTPUT_DIR, 'v4_parametric_sweep.csv')
    fieldnames = list(rows[0].keys())
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV saved -> {csv_path}")

    # --- Résumé analytique : frontière de phase ---
    print("\n" + "="*70)
    print("CARTE DE PHASE : heretic_pct_final_mean")
    print("="*70)
    header = f"{'u_thr':>6} | " + " | ".join(f"s={s:>4}" for s in STEPS_REQUIRED)
    print(header)
    print("-" * len(header))
    for u_thr in U_THRESHOLDS:
        row_data = [r for r in rows if r['u_threshold'] == u_thr]
        line = f"{u_thr:>6.1f} | "
        cells = []
        for steps_req in STEPS_REQUIRED:
            r = next(x for x in row_data if x['steps_required'] == steps_req)
            pct = r['heretic_pct_final_mean']
            cells.append(f"{pct:>5.2f}")
        line += " | ".join(cells)
        print(line)

    print("\nLEGENDE : valeur = fraction finale du reseau devenu hereticique (0.15 = base statique)")

    print("\n" + "="*70)
    print("CARTE DE PHASE : entropy_final_mean (V3 baseline ~ 3.41)")
    print("="*70)
    print(header)
    print("-" * len(header))
    for u_thr in U_THRESHOLDS:
        row_data = [r for r in rows if r['u_threshold'] == u_thr]
        line = f"{u_thr:>6.1f} | "
        cells = []
        for steps_req in STEPS_REQUIRED:
            r = next(x for x in row_data if x['steps_required'] == steps_req)
            h = r['entropy_final_mean']
            cells.append(f"{h:>5.3f}")
        line += " | ".join(cells)
        print(line)

    print("\n" + "="*70)
    print("CARTE DE PHASE : t_first_valid_mean (step de 1ere bascule, -1=jamais)")
    print("="*70)
    print(header)
    print("-" * len(header))
    for u_thr in U_THRESHOLDS:
        row_data = [r for r in rows if r['u_threshold'] == u_thr]
        line = f"{u_thr:>6.1f} | "
        cells = []
        for steps_req in STEPS_REQUIRED:
            r = next(x for x in row_data if x['steps_required'] == steps_req)
            tf = r['t_first_valid_mean']
            cells.append(f"{tf:>5.0f}" if tf >= 0 else "   --")
        line += " | ".join(cells)
        print(line)


if __name__ == '__main__':
    main()
