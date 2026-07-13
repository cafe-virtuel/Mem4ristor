#!/usr/bin/env python3
"""
Mem4ristor Parameter Sweep — H_cont Maximization (Serial, Checkpointed)
=========================================================================
D (coupling)     : 0.05 → 0.50  step 0.05  (10 values)
sigma_v (noise)  : 0.01 → 0.20  step 0.02  (10 values)
heretic_ratio    : 0.05 → 0.30  step 0.05  (6 values)

Topologies  : BA m=3, BA m=5
Network sizes: N=400 (20×20), N=1600 (40×40)
Metric      : H_cont (continuous entropy, 100-bin, measured post-warmup)

Checkpoint  : CSV written after every 50 combinations
              Resume support via checking existing CSV rows
"""

import sys, os, time, csv, io
from datetime import datetime

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..')
CSV_PATH    = os.path.join(RESULTS_DIR, 'WORK_LOG_SWEEP_data.csv')
MD_PATH     = os.path.join(RESULTS_DIR, 'WORK_LOG_SWEEP.md')
LOCK_PATH   = os.path.join(RESULTS_DIR, 'sweep_checkpoint.lock')

# ── Parameter grids ────────────────────────────────────────────────────────────
D_VALUES       = [round(0.05 + i * 0.05, 4) for i in range(10)]   # 0.05 … 0.50
SIGMA_V_VALUES = [round(0.01 + i * 0.02, 4) for i in range(10)]  # 0.01 … 0.19
HERETIC_VALUES = [round(0.05 + i * 0.05, 4) for i in range(6)]    # 0.05 … 0.30
TOPOLOGIES     = [('BA_m3', 3), ('BA_m5', 5)]
SIZES          = [(400, 20), (1600, 40)]
SEEDS          = [42, 123, 777]

# ── Simulation params ───────────────────────────────────────────────────────────
STEPS    = 400   # measurement window; sample every 4th → 100 independent samples
WARMUP   = 200   # settle
I_STIM   = 0.0   # cold-start

# ── CSV fields ────────────────────────────────────────────────────────────────
CSV_FIELDS = ['topology','N','m','D','sigma_v','heretic_ratio',
              'H_cont','H_cont_std','H_cont_min','H_cont_max',
              'seeds','n_runs','elapsed_s']

def now():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def checkpoint(results_list, start_time, total, run_id):
    """Write partial CSV and print progress."""
    elapsed = time.time() - start_time
    pct = run_id / total * 100
    rate = run_id / elapsed if elapsed > 0 else 0
    eta = (total - run_id) / rate if rate > 0 else 0

    # Write CSV
    with open(CSV_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in results_list:
            writer.writerow(row)

    print(f"  [{now()}] checkpoint {run_id}/{total} ({pct:.1f}%) | "
          f"ETA {eta/60:.1f} min | {elapsed:.0f}s elapsed")

    # Flush stdout so parent sees it
    sys.stdout.flush()

# ── Main ────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # ── Lazy import to avoid startup overhead in multiprocessing ──────────────
    from mem4ristor.core import Mem4Network
    from mem4ristor.graph_utils import make_ba
    from mem4ristor.metrics import calculate_continuous_entropy

    total_combos = (
        len(D_VALUES) * len(SIGMA_V_VALUES) * len(HERETIC_VALUES) *
        len(TOPOLOGIES) * len(SIZES)
    )
    total_runs = total_combos * len(SEEDS)

    print(f"[{now()}] SWEEP START")
    print(f"  Combinations : {total_combos}")
    print(f"  Total runs   : {total_runs}")
    print(f"  D_range       : {D_VALUES}")
    print(f"  sigma_v_range : {SIGMA_V_VALUES}")
    print(f"  heretic_range : {HERETIC_VALUES}")

    # ── Load existing results for resume ──────────────────────────────────────
    existing_keys = set()
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, 'r') as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)
        for r in existing_rows:
            key = (r['topology'], int(r['N']), float(r['D']),
                   float(r['sigma_v']), float(r['heretic_ratio']))
            existing_keys.add(key)
        results_so_far = existing_rows
        print(f"  Resuming: {len(existing_keys)} combos already in CSV")
    else:
        results_so_far = []

    start_time = time.time()
    run_id = 0

    # ── Run loop ───────────────────────────────────────────────────────────────
    for topo_name, m in TOPOLOGIES:
        for N, size in SIZES:
            for D in D_VALUES:
                for sigma_v in SIGMA_V_VALUES:
                    for hr in HERETIC_VALUES:
                        key = (topo_name, N, D, sigma_v, hr)
                        if key in existing_keys:
                            continue  # skip already done

                        h_vals = []
                        for seed in SEEDS:
                            t0 = time.time()

                            try:
                                adj = make_ba(N, m, seed=seed)
                            except Exception as e:
                                print(f"    make_ba failed N={N} m={m} seed={seed}: {e}")
                                continue

                            net = Mem4Network(
                                adjacency_matrix=adj,
                                heretic_ratio=hr,
                                seed=seed,
                                coupling_norm='degree_linear'
                            )
                            net.model.cfg['coupling']['D'] = D
                            net.model.cfg['noise']['sigma_v'] = sigma_v

                            # Warmup
                            for _ in range(WARMUP):
                                net.step(I_stimulus=I_STIM)

                            # Measurement — sample every 4th step
                            h_samples = []
                            for step_idx in range(STEPS):
                                net.step(I_stimulus=I_STIM)
                                if step_idx % 4 == 0:
                                    h_samples.append(calculate_continuous_entropy(net.v, bins=100))

                            h = float(np.mean(h_samples))
                            h_vals.append(h)
                            elapsed = time.time() - t0

                        if h_vals:
                            h_mean = float(np.mean(h_vals))
                            h_std  = float(np.std(h_vals)) if len(h_vals) > 1 else 0.0
                            h_min  = float(np.min(h_vals))
                            h_max  = float(np.max(h_vals))
                            elapsed = time.time() - start_time
                        else:
                            continue  # skip failed combos

                        row = {
                            'topology':      topo_name,
                            'N':            N,
                            'm':            m,
                            'D':            D,
                            'sigma_v':      sigma_v,
                            'heretic_ratio': hr,
                            'H_cont':       round(h_mean, 6),
                            'H_cont_std':   round(h_std, 6),
                            'H_cont_min':   round(h_min, 6),
                            'H_cont_max':   round(h_max, 6),
                            'seeds':        '-'.join(str(s) for s in SEEDS),
                            'n_runs':       len(h_vals),
                            'elapsed_s':    round(elapsed, 1),
                        }
                        results_so_far.append(row)
                        run_id += 1

                        if run_id % 20 == 0:
                            checkpoint(results_so_far, start_time, total_combos, run_id)

    total_elapsed = time.time() - start_time
    print(f"\n[{now()}] SWEEP COMPLETE — {total_elapsed:.1f}s total, {run_id} combos")

    # Sort and write final CSV
    results_so_far.sort(key=lambda r: r['H_cont'], reverse=True)
    with open(CSV_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(results_so_far)

    # ── Build Markdown report ─────────────────────────────────────────────────
    md_lines = []
    md_lines.append("# WORK_LOG_SWEEP — Mem4ristor H_cont Parameter Sweep\n")
    md_lines.append(f"**Generated**   : {now()}")
    md_lines.append(f"**Combinations** : {run_id}")
    md_lines.append(f"**Wall time**    : {total_elapsed:.1f}s\n")
    md_lines.append("## Parameter Grids\n")
    md_lines.append(f"- D (coupling)    : {D_VALUES}")
    md_lines.append(f"- sigma_v         : {SIGMA_V_VALUES}")
    md_lines.append(f"- heretic_ratio   : {HERETIC_VALUES}")
    md_lines.append(f"- Seeds           : {SEEDS}")
    md_lines.append(f"- STEPS           : {STEPS} (sampled every 4th = {STEPS//4} samples)")
    md_lines.append(f"- WARMUP          : {WARMUP}\n")

    # Optimal per topology × N
    md_lines.append("## Optimal H_cont by Topology × Size\n")
    md_lines.append("| topology | N | m | D | sigma_v | heretic_ratio | H_cont_mean | H_cont_std |")
    md_lines.append("|---|---|---|---|---|---|---|---|")
    for topo_name, m in TOPOLOGIES:
        for N, _ in SIZES:
            subset = [r for r in results_so_far if r['topology'] == topo_name and int(r['N']) == N]
            if subset:
                best = max(subset, key=lambda r: r['H_cont'])
                md_lines.append(
                    f"| {best['topology']} | {best['N']} | {best['m']} | "
                    f"{best['D']} | {best['sigma_v']} | {best['heretic_ratio']} | "
                    f"{best['H_cont']} | {best['H_cont_std']} |"
                )

    # Top 30 overall
    md_lines.append("\n## Top 30 Parameter Combinations (by H_cont)\n")
    top30 = sorted(results_so_far, key=lambda r: r['H_cont'], reverse=True)[:30]
    md_lines.append("| rank | topology | N | D | sigma_v | heretic_ratio | H_cont | std |")
    md_lines.append("|---|---|---|---|---|---|---|---|")
    for rank, row in enumerate(top30, 1):
        md_lines.append(
            f"| {rank} | {row['topology']} | {row['N']} | {row['D']} | "
            f"{row['sigma_v']} | {row['heretic_ratio']} | {row['H_cont']} | {row['H_cont_std']} |"
        )

    # Scaling analysis (D × sigma_v at fixed heretic)
    md_lines.append("\n## Scaling Analysis (N=400 vs N=1600)\n")
    for topo_name, m in TOPOLOGIES:
        md_lines.append(f"### {topo_name} (m={m})\n")
        md_lines.append("| D | sigma_v | heretic_ratio | H_cont(N=400) | H_cont(N=1600) | ratio |")
        md_lines.append("|---|---|---|---|---|---|")
        for D in D_VALUES:
            for sigma_v in [SIGMA_V_VALUES[0], SIGMA_V_VALUES[4], SIGMA_V_VALUES[-1]]:
                for hr in HERETIC_VALUES:
                    r400 = [r for r in results_so_far
                            if r['topology'] == topo_name and int(r['N']) == 400
                            and float(r['D']) == D and float(r['sigma_v']) == sigma_v
                            and float(r['heretic_ratio']) == hr]
                    r1600 = [r for r in results_so_far
                             if r['topology'] == topo_name and int(r['N']) == 1600
                             and float(r['D']) == D and float(r['sigma_v']) == sigma_v
                             and float(r['heretic_ratio']) == hr]
                    if r400 and r1600:
                        h400 = float(r400[0]['H_cont'])
                        h1600 = float(r1600[0]['H_cont'])
                        ratio = h1600 / h400 if abs(h400) > 1e-9 else float('nan')
                        md_lines.append(f"| {D} | {sigma_v} | {hr} | {h400:.6f} | {h1600:.6f} | {ratio:.4f} |")

    # Full CSV block
    md_lines.append("\n## Raw CSV Data\n")
    md_lines.append("```csv")
    md_lines.append(','.join(CSV_FIELDS))
    for row in results_so_far:
        md_lines.append(','.join(str(row[k]) for k in CSV_FIELDS))
    md_lines.append("```\n")
    md_lines.append(f"\n*Generated by mem4ristor parameter sweep at {now()}*")

    with open(MD_PATH, 'w') as f:
        f.write('\n'.join(md_lines))

    print(f"Results written to {CSV_PATH}")
    print(f"Report written to {MD_PATH}")