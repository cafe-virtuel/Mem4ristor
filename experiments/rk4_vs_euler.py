"""
RK4 vs Euler — Validation de l'intégrateur (DeepSeek review item 8)
=====================================================================
Objectif : vérifier que l'intégration Euler (dt=0.05) ne crée pas d'artefacts
numériques dans les régimes critiques de Mem4ristor.

Protocole :
  1. Implémenter un intégrateur RK4 standalone reproduisant les équations FHN
     de dynamics.py (dv, dw, du) avec couplage réseau fixé.
  2. Comparer Euler vs RK4 à dt=0.05 (même pas) sur 2000 steps.
  3. Tester 3 conditions critiques :
       (a) FULL (dynamics normales)
       (b) FROZEN_U (tau_u=10000 — u gelé, claim principal de Paper 1)
       (c) BA m=5 (frontière dead zone, régime le plus sensible)
  4. Métriques : divergence de trajectoire (MAE(v)), H_cog, synchrony, surge ratio.

Bruit : même eta fixé au début de chaque step pour les deux intégrateurs
(Euler-Maruyama — approche standard pour SDE stochastiques faibles).

Usage :
    python experiments/rk4_vs_euler.py
"""
import sys, os, time, csv
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from mem4ristor.topology import Mem4Network
from mem4ristor.graph_utils import make_ba
from mem4ristor.metrics import calculate_cognitive_entropy, calculate_pairwise_synchrony

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Paramètres ---
DT       = 0.05
N_WARM   = 1000   # warmup Euler (partagé — on copie l'état après)
N_STEPS  = 2000   # steps de comparaison
N_SEEDS  = 5
I_STIM   = 0.3
GRID     = 10     # 10x10 = 100 noeuds

# Paramètres FHN (identiques à config.yaml / dynamics.py defaults)
A         = 0.7
B         = 0.8
EPS       = 0.08
ALPHA     = 0.15
VCD       = 5.0    # v_cubic_divisor
EPS_U     = 0.1
K_U       = 1.0
SIG_B     = 0.1
TAU_U     = 5.0
U_CLAMP   = (0.0, 1.0)
D_EFF_FAC = 0.15   # D avant normalisation /sqrt(N)
SIG_STEEP = 10.0
SOC_LEAK  = 0.1
SIG_V     = 0.05   # bruit sigma_v par défaut
# plasticity off pour simplifier la comparaison RK4
LAM_LEARN = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Dérivées FHN (extraites de dynamics.py)
# ─────────────────────────────────────────────────────────────────────────────

def fhn_derivatives(v, w, u, laplacian_v, eta, I_stim, heretic_mask,
                    D_eff, tau_u=TAU_U, frozen_u=False):
    u_filter  = np.tanh(SIG_STEEP * (0.5 - u)) + SOC_LEAK
    I_coup    = D_eff * u_filter * laplacian_v
    I_eff     = np.full(len(v), I_stim)
    I_eff[heretic_mask] *= -1.0
    I_ext     = I_eff + I_coup

    sigma_s   = np.abs(laplacian_v)
    alpha_s   = 2.0
    eps_u_eff = EPS_U * np.clip(1.0 + alpha_s * sigma_s, 1.0, 5.0)

    dv = v - (v**3) / VCD - w + I_ext - ALPHA * np.tanh(v) + eta
    dw = EPS * (v + A - B * w)
    if frozen_u:
        du = np.zeros_like(u)
    else:
        du = (eps_u_eff * (K_U * sigma_s + SIG_B - u)) / tau_u

    return dv, dw, du


def compute_laplacian(adj_norm, v):
    """Laplacien normalisé : W_norm @ v - v."""
    return adj_norm @ v - v


def euler_step(v, w, u, adj_norm, heretic_mask, D_eff, rng, I_stim, frozen_u=False):
    lv   = compute_laplacian(adj_norm, v)
    eta  = rng.normal(0, SIG_V, len(v))
    dv, dw, du = fhn_derivatives(v, w, u, lv, eta, I_stim, heretic_mask, D_eff, frozen_u=frozen_u)
    v_new = np.clip(v + dv * DT, -100, 100)
    w_new = np.clip(w + dw * DT, -100, 100)
    u_new = np.clip(u + du * DT, *U_CLAMP)
    return v_new, w_new, u_new


def rk4_step(v, w, u, adj_norm, heretic_mask, D_eff, rng, I_stim, frozen_u=False):
    # eta fixé une fois par step (Euler-Maruyama pour la partie stochastique)
    eta = rng.normal(0, SIG_V, len(v))

    def deriv(v_, w_, u_):
        lv_ = compute_laplacian(adj_norm, v_)
        return fhn_derivatives(v_, w_, u_, lv_, eta, I_stim, heretic_mask, D_eff, frozen_u=frozen_u)

    k1v, k1w, k1u = deriv(v, w, u)
    k2v, k2w, k2u = deriv(v + 0.5*DT*k1v, w + 0.5*DT*k1w, u + 0.5*DT*k1u)
    k3v, k3w, k3u = deriv(v + 0.5*DT*k2v, w + 0.5*DT*k2w, u + 0.5*DT*k2u)
    k4v, k4w, k4u = deriv(v + DT*k3v,     w + DT*k3w,     u + DT*k3u)

    v_new = np.clip(v + (DT/6)*(k1v + 2*k2v + 2*k3v + k4v), -100, 100)
    w_new = np.clip(w + (DT/6)*(k1w + 2*k2w + 2*k3w + k4w), -100, 100)
    u_new = np.clip(u + (DT/6)*(k1u + 2*k2u + 2*k3u + k4u), *U_CLAMP)
    return v_new, w_new, u_new


def get_adj_norm(adj):
    """Normalisation degree_linear (identique à topology.py)."""
    deg = adj.sum(axis=1, keepdims=True)
    deg = np.where(deg == 0, 1.0, deg)
    return adj / deg


def run_comparison(adj, heretic_mask, seed, I_stim, frozen_u=False, label=""):
    rng = np.random.RandomState(seed)
    N = adj.shape[0]
    D_eff = D_EFF_FAC / np.sqrt(N)
    adj_norm = get_adj_norm(adj)

    # --- Warmup Euler commun ---
    v = np.zeros(N); w = np.zeros(N); u = np.full(N, 0.5)
    for _ in range(N_WARM):
        v, w, u = euler_step(v, w, u, adj_norm, heretic_mask, D_eff, rng, I_stim, frozen_u)

    # --- Copie état initial post-warmup ---
    v0, w0, u0 = v.copy(), w.copy(), u.copy()
    rng_state0 = rng.get_state()

    # --- Euler forward ---
    ve, we, ue = v0.copy(), w0.copy(), u0.copy()
    rng.set_state(rng_state0)
    snaps_e = []
    for t in range(N_STEPS):
        ve, we, ue = euler_step(ve, we, ue, adj_norm, heretic_mask, D_eff, rng, I_stim, frozen_u)
        if t % 10 == 0:
            snaps_e.append(ve.copy())

    # --- RK4 forward (même état initial, même seed) ---
    vr, wr, ur = v0.copy(), w0.copy(), u0.copy()
    rng.set_state(rng_state0)
    snaps_r = []
    for t in range(N_STEPS):
        vr, wr, ur = rk4_step(vr, wr, ur, adj_norm, heretic_mask, D_eff, rng, I_stim, frozen_u)
        if t % 10 == 0:
            snaps_r.append(vr.copy())

    arr_e = np.array(snaps_e)
    arr_r = np.array(snaps_r)

    h_euler  = float(np.mean([calculate_cognitive_entropy(s) for s in arr_e[::5]]))
    h_rk4    = float(np.mean([calculate_cognitive_entropy(s) for s in arr_r[::5]]))
    sy_euler = float(calculate_pairwise_synchrony(arr_e))
    sy_rk4   = float(calculate_pairwise_synchrony(arr_r))
    mae_v    = float(np.mean(np.abs(arr_e - arr_r)))  # divergence trajectoire

    return {'label': label, 'seed': seed, 'frozen_u': frozen_u,
            'h_euler': h_euler, 'h_rk4': h_rk4,
            'sy_euler': sy_euler, 'sy_rk4': sy_rk4,
            'mae_v': mae_v}


def main():
    print("RK4 vs Euler — dt=0.05, N_warm=1000, N_steps=2000")
    print(f"Seeds={N_SEEDS}, I_stim={I_STIM}\n")
    t0 = time.time()

    results = []

    # Conditions à tester
    conditions = [
        ('ba_m3_FULL',       3, False),
        ('ba_m3_FROZEN_U',   3, True),
        ('ba_m5_FULL',       5, False),
        ('ba_m5_FROZEN_U',   5, True),
    ]

    for label, m, frozen_u in conditions:
        seed_rows = []
        for seed in range(N_SEEDS):
            adj = make_ba(GRID*GRID, m, seed=42)  # structure fixe
            # Masque hérétique déterministe
            N = adj.shape[0]
            step = max(int(1.0 / 0.15), 1)
            rng_h = np.random.RandomState(seed)
            heretic_ids = []
            for i in range(0, N, step):
                block_end = min(i + step, N)
                heretic_ids.append(rng_h.randint(i, block_end))
            heretic_mask = np.zeros(N, dtype=bool)
            heretic_mask[heretic_ids] = True

            r = run_comparison(adj, heretic_mask, seed=seed, I_stim=I_STIM,
                               frozen_u=frozen_u, label=label)
            seed_rows.append(r)
            results.append(r)

        mae_m  = np.mean([r['mae_v']    for r in seed_rows])
        h_e_m  = np.mean([r['h_euler']  for r in seed_rows])
        h_r_m  = np.mean([r['h_rk4']   for r in seed_rows])
        sy_e_m = np.mean([r['sy_euler'] for r in seed_rows])
        sy_r_m = np.mean([r['sy_rk4']  for r in seed_rows])
        print(f"{label:20s} | MAE(v)={mae_m:.4f} | "
              f"H euler={h_e_m:.4f} rk4={h_r_m:.4f} delta={abs(h_e_m-h_r_m):.4f} | "
              f"sync euler={sy_e_m:.4f} rk4={sy_r_m:.4f}", flush=True)

    # CSV
    csv_path = os.path.join(OUTPUT_DIR, 'rk4_vs_euler.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nCSV -> {csv_path}")

    # --- Surge ratio : FROZEN_U / FULL ---
    print("\n" + "="*70)
    print("CLAIM PRINCIPAL : synchrony surge FROZEN_U / FULL")
    print("="*70)
    for m in [3, 5]:
        full_e  = np.mean([r['sy_euler'] for r in results if r['label'] == f'ba_m{m}_FULL'])
        froz_e  = np.mean([r['sy_euler'] for r in results if r['label'] == f'ba_m{m}_FROZEN_U'])
        full_r  = np.mean([r['sy_rk4']  for r in results if r['label'] == f'ba_m{m}_FULL'])
        froz_r  = np.mean([r['sy_rk4']  for r in results if r['label'] == f'ba_m{m}_FROZEN_U'])
        surge_e = (froz_e / full_e - 1) * 100 if full_e > 1e-9 else float('nan')
        surge_r = (froz_r / full_r - 1) * 100 if full_r > 1e-9 else float('nan')
        print(f"  BA m={m}: Euler surge = {surge_e:+.1f}%  |  RK4 surge = {surge_r:+.1f}%  "
              f"|  delta = {abs(surge_e - surge_r):.1f}pp")

    # --- Verdict H_cog ---
    print("\n" + "="*70)
    print("VERDICT H_cog (classification dead zone)")
    print("="*70)
    for label, m, frozen_u in conditions:
        h_e = np.mean([r['h_euler'] for r in results if r['label'] == label])
        h_r = np.mean([r['h_rk4']  for r in results if r['label'] == label])
        delta = abs(h_e - h_r)
        flag = "OK" if delta < 0.05 else "ATTENTION"
        print(f"  {label:20s} | Euler={h_e:.4f} RK4={h_r:.4f} delta={delta:.4f} {flag}")

    print(f"\nTotal: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
