"""
dt Sensitivity Analysis (DeepSeek review item 7)
=================================================
Objectif : vérifier que les résultats clés (H_cog, synchrony) ne dépendent pas
du pas d'intégration dt de l'intégrateur Euler.

Protocole :
  - 4 valeurs de dt : 0.01, 0.02, 0.05 (référence), 0.10
  - Temps total simulé FIXÉ à T=150 unités (= N_steps × dt)
    → steps : 15000 / 7500 / 3000 / 1500
  - 3 topologies représentatives :
      BA m=3 (lambda2~1.4, régime fonctionnel)
      BA m=5 (lambda2~3.0, frontière critique)
      BA m=8 (lambda2~5.9, dead zone confirmée)
  - 5 seeds, I_stim=0.0 (dynamique endogène, plus sensible aux artefacts)
  - Warmup = 25% du temps total (fixé en temps, pas en steps)

Métriques :
  H_cog     : entropie cognitive (la claim principale)
  H_cont    : entropie continue (proxy de la variance de v)
  sync      : synchronie pairwise moyenne
  v_mean    : voltage moyen (drift check)
  u_mean    : doute moyen (drift check)

Usage :
    python experiments/dt_sensitivity.py
"""
import sys
import os
import time
import csv
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from mem4ristor.topology import Mem4Network
from mem4ristor.metrics import (
    calculate_cognitive_entropy,
    calculate_continuous_entropy,
    calculate_pairwise_synchrony,
)

# --- Paramètres ---
DTS            = [0.01, 0.02, 0.05, 0.10]
T_TOTAL        = 150.0   # temps simulé total (unités FHN), identique pour tous les dt
WARMUP_FRAC    = 0.25    # fraction warmup (ignorée dans les métriques)
N_SEEDS        = 5
I_STIM         = 0.3     # condition principale des expériences Mem4ristor
GRID_SIZE      = 10      # 10x10 = 100 nœuds

# Topologies : (nom, m pour BA)
TOPOLOGIES = [
    ('ba_m3_functional',  3),   # fonctionnel, lambda2~1.4
    ('ba_m5_critical',    5),   # frontière, lambda2~3.0
    ('ba_m8_dead_zone',   8),   # dead zone, lambda2~5.9
]

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

REF_DT = 0.05   # référence pour calcul des écarts


def run_one(topo_name: str, m: int, dt: float, seed: int) -> dict:
    n_steps  = int(round(T_TOTAL / dt))
    warmup_n = int(round(n_steps * WARMUP_FRAC))
    measure_n = n_steps - warmup_n

    net = Mem4Network(size=GRID_SIZE, heretic_ratio=0.15, seed=seed,
                      coupling_norm='degree_linear')
    # Injecter le m de BA via adjacency directe
    from mem4ristor.graph_utils import make_ba
    adj = make_ba(GRID_SIZE * GRID_SIZE, m, seed)
    net = Mem4Network(adjacency_matrix=adj, heretic_ratio=0.15, seed=seed,
                      coupling_norm='degree_linear')
    # Surcharger dt dans le modèle
    net.model.cfg['dynamics']['dt'] = dt
    net.model.dt = dt

    v_snaps = []
    for step in range(n_steps):
        net.step(I_stimulus=I_STIM)
        if step >= warmup_n and step % max(1, n_steps // 300) == 0:
            v_snaps.append(net.model.v.copy())

    v_arr   = np.array(v_snaps)
    h_cog   = float(np.mean([calculate_cognitive_entropy(v) for v in v_arr[::5]]))
    h_cont  = float(calculate_continuous_entropy(v_arr.flatten()))
    sync    = float(calculate_pairwise_synchrony(v_arr))
    v_mean  = float(np.mean(v_arr))
    u_mean  = float(np.mean(net.model.u))

    return {
        'topo':     topo_name,
        'm':        m,
        'dt':       dt,
        'n_steps':  n_steps,
        'seed':     seed,
        'h_cog':    h_cog,
        'h_cont':   h_cont,
        'sync':     sync,
        'v_mean':   v_mean,
        'u_mean':   u_mean,
    }


def main():
    total = len(TOPOLOGIES) * len(DTS) * N_SEEDS
    print(f"dt sensitivity : {len(TOPOLOGIES)} topos x {len(DTS)} dt x {N_SEEDS} seeds = {total} runs")
    print(f"T_total={T_TOTAL} | warmup={WARMUP_FRAC*100:.0f}% | I_stim={I_STIM}\n")

    rows = []
    run_idx = 0
    t0 = time.time()

    for topo_name, m in TOPOLOGIES:
        for dt in DTS:
            n_steps = int(round(T_TOTAL / dt))
            seed_rows = []
            for seed in range(N_SEEDS):
                run_idx += 1
                r = run_one(topo_name, m, dt, seed)
                seed_rows.append(r)
                rows.append(r)

            elapsed = time.time() - t0
            pct = run_idx / total
            eta = (elapsed / pct) * (1 - pct) if pct > 0 else 0
            h_mean = np.mean([r['h_cog']  for r in seed_rows])
            s_mean = np.mean([r['sync']   for r in seed_rows])
            print(f"[{run_idx:3d}/{total}] {topo_name:22s} dt={dt:.2f} n={n_steps:6d} "
                  f"| H_cog={h_mean:.4f} sync={s_mean:.4f} | ETA {eta:.0f}s", flush=True)

    # Export CSV
    csv_path = os.path.join(OUTPUT_DIR, 'dt_sensitivity.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV -> {csv_path}")

    # --- Tableau de synthèse : mean over seeds, par topo x dt ---
    print("\n" + "="*80)
    print("SYNTHESE : H_cog (mean over seeds) — reference = dt=0.05")
    print("="*80)
    print(f"{'topo':22s} | " + " | ".join(f"dt={dt:.2f}" for dt in DTS) + " | max_delta_from_ref")
    print("-" * 80)

    for topo_name, m in TOPOLOGIES:
        ref_rows = [r for r in rows if r['topo'] == topo_name and abs(r['dt'] - REF_DT) < 1e-9]
        h_ref = np.mean([r['h_cog'] for r in ref_rows])

        cells = []
        deltas = []
        for dt in DTS:
            dt_rows = [r for r in rows if r['topo'] == topo_name and abs(r['dt'] - dt) < 1e-9]
            h = np.mean([r['h_cog'] for r in dt_rows])
            delta = abs(h - h_ref)
            deltas.append(delta)
            cells.append(f"{h:.4f}")
        max_delta = max(deltas)
        flag = " *** ATTENTION" if max_delta > 0.05 else " OK"
        print(f"{topo_name:22s} | " + " | ".join(cells) + f" | {max_delta:.4f}{flag}")

    print("\n" + "="*80)
    print("SYNTHESE : synchrony (mean over seeds)")
    print("="*80)
    print(f"{'topo':22s} | " + " | ".join(f"dt={dt:.2f}" for dt in DTS) + " | max_delta_from_ref")
    print("-" * 80)

    for topo_name, m in TOPOLOGIES:
        ref_rows = [r for r in rows if r['topo'] == topo_name and abs(r['dt'] - REF_DT) < 1e-9]
        s_ref = np.mean([r['sync'] for r in ref_rows])

        cells = []
        deltas = []
        for dt in DTS:
            dt_rows = [r for r in rows if r['topo'] == topo_name and abs(r['dt'] - dt) < 1e-9]
            s = np.mean([r['sync'] for r in dt_rows])
            delta = abs(s - s_ref)
            deltas.append(delta)
            cells.append(f"{s:.4f}")
        max_delta = max(deltas)
        flag = " *** ATTENTION" if max_delta > 0.05 else " OK"
        print(f"{topo_name:22s} | " + " | ".join(cells) + f" | {max_delta:.4f}{flag}")

    print("\n" + "="*80)
    print("VERDICT")
    print("="*80)
    print("Seuil critique : variation > 0.05 entre dt=0.01 et dt=0.10 = artefact suspect")
    print("Si toutes les metriques sont stables -> integration Euler dt=0.05 validee")
    print(f"\nTotal wall time: {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
