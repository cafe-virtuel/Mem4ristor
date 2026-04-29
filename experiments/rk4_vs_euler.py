"""
RK4 vs Euler -- Validation de l'integrateur (DeepSeek review item 8)
=====================================================================
Objectif : verifier que l'integration Euler (dt=0.05) ne cree pas d'artefacts
numeriques dans les regimes critiques de Mem4ristor.

Protocole :
  1. Implémenter un intégrateur RK4 standalone reproduisant les équations FHN
     de dynamics.py (dv, dw, du) avec couplage réseau fixé.
  2. Comparer Euler vs RK4 à dt=0.05 (même pas) sur 2000 steps.
  3. Tester 4 conditions critiques :
       (a) FULL     BA m=3 et m=5
       (b) FROZEN_U BA m=3 et m=5 (claim principal de Paper 1)
  4. Metriques : divergence de trajectoire (MAE(v)), H_cog, synchrony, surge ratio.

Bruit : meme eta fixé une fois par step pour les deux intégrateurs
(Euler-Maruyama -- approche standard pour SDE faibles).

Plasticite : ACTIVE (lambda_learn=0.05) -- validation complète du systeme reel.
  La plasticite affecte dw (terme dw_learning) via sigma_social et innovation_mask.
  Elle n'affecte pas dv (la variable la plus sensible numeriquement).

PARAMETRES : alignés sur config.yaml + dynamics.py defaults (correction 2026-04-29)
  - sigmoid_steepness = pi    (etait 10.0 -- erreur de transcription)
  - social_leakage    = 0.01  (etait 0.1)
  - epsilon_u         = 0.02  (etait 0.1)
  - sigma_baseline    = 0.05  (etait 0.1)
  - tau_u             = 10.0  (etait 5.0)

Usage :
    python experiments/rk4_vs_euler.py
"""
import sys, os, time, csv
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from mem4ristor.graph_utils import make_ba
from mem4ristor.metrics import calculate_cognitive_entropy, calculate_pairwise_synchrony

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Parametres de simulation ---
DT       = 0.05
N_WARM   = 1000   # warmup Euler (partage -- on copie l'etat apres)
N_STEPS  = 2000   # steps de comparaison
N_SEEDS  = 5
I_STIM   = 0.3
GRID     = 10     # 10x10 = 100 noeuds

# Parametres FHN -- ALIGNES sur config.yaml + dynamics.py defaults
A         = 0.7
B         = 0.8
EPS       = 0.08
ALPHA     = 0.15
VCD       = 5.0         # v_cubic_divisor
EPS_U     = 0.02        # epsilon_u          (etait 0.1 -- corrige 2026-04-29)
K_U       = 1.0
SIG_B     = 0.05        # sigma_baseline     (etait 0.1 -- corrige 2026-04-29)
TAU_U     = 10.0        # tau_u              (etait 5.0 -- corrige 2026-04-29)
U_CLAMP   = (0.0, 1.0)
D_EFF_FAC = 0.15        # D avant normalisation /sqrt(N)
SIG_STEEP = np.pi       # sigmoid_steepness  (etait 10.0 -- corrige 2026-04-29)
SOC_LEAK  = 0.01        # social_leakage     (etait 0.1 -- corrige 2026-04-29)
SIG_V     = 0.05        # sigma_v bruit
ALPHA_S   = 2.0         # alpha_surprise
SUR_CAP   = 5.0         # surprise_cap

# Plasticite -- ACTIVE (validation complete du systeme reel)
LAM_LEARN = 0.05        # lambda_learn (config.yaml default)
TAU_PLAST = 1000        # tau_plasticity
W_SAT     = 2.0         # w_saturation


# ─────────────────────────────────────────────────────────────────────────────
# Derivees FHN + plasticite -- reproduisent fidelement dynamics.py step()
# ─────────────────────────────────────────────────────────────────────────────

def fhn_derivatives(v, w, u, laplacian_v, eta, I_stim, heretic_mask,
                    D_eff, frozen_u=False):
    """
    Retourne (dv, dw, du) avec plasticite active.
    Identique a dynamics.py step() sans hysteresis (innovation_mask = u > 0.5).
    """
    sigma_s   = np.abs(laplacian_v)
    u_filter  = np.tanh(SIG_STEEP * (0.5 - u)) + SOC_LEAK
    I_coup    = D_eff * u_filter * laplacian_v
    I_eff     = np.full(len(v), I_stim)
    I_eff[heretic_mask] *= -1.0
    I_ext     = I_eff + I_coup

    # FHN core
    dv = v - (v**3) / VCD - w + I_ext - ALPHA * np.tanh(v) + eta

    # Plasticite (lambda_learn = 0 -> desactivee, 0.05 -> systeme reel)
    innovation_mask = (u > 0.5).astype(float)
    w_ratio     = w / W_SAT
    sat_factor  = np.clip(1.0 - w_ratio**2, 0.0, 1.0)
    dw_learn    = LAM_LEARN * sigma_s * innovation_mask * sat_factor - w / TAU_PLAST
    dw_fhn      = EPS * (v + A - B * w)
    dw          = dw_fhn + dw_learn

    # Doute
    eps_u_eff = EPS_U * np.clip(1.0 + ALPHA_S * sigma_s, 1.0, SUR_CAP)
    if frozen_u:
        du = np.zeros_like(u)
    else:
        du = (eps_u_eff * (K_U * sigma_s + SIG_B - u)) / TAU_U

    return dv, dw, du


def compute_laplacian(adj_norm, v):
    """Laplacien normalise degree_linear : W_norm @ v - v."""
    return adj_norm @ v - v


def euler_step(v, w, u, adj_norm, heretic_mask, D_eff, rng, I_stim, frozen_u=False):
    lv   = compute_laplacian(adj_norm, v)
    eta  = rng.normal(0, SIG_V, len(v))
    dv, dw, du = fhn_derivatives(v, w, u, lv, eta, I_stim, heretic_mask, D_eff,
                                  frozen_u=frozen_u)
    v_new = np.clip(v + dv * DT, -100, 100)
    w_new = np.clip(w + dw * DT, -100, 100)
    u_new = np.clip(u + du * DT, *U_CLAMP)
    return v_new, w_new, u_new


def rk4_step(v, w, u, adj_norm, heretic_mask, D_eff, rng, I_stim, frozen_u=False):
    """
    RK4 fixed-step avec eta fixe une fois par step (Euler-Maruyama pour la SDE).
    Pour la partie deterministe (dv_det, dw, du), RK4 est exact a l'ordre 4.
    Le terme eta est traite comme un forcing additif constant sur le step,
    ce qui est la convention standard pour les SDE faibles.
    """
    eta = rng.normal(0, SIG_V, len(v))

    def deriv(v_, w_, u_):
        lv_ = compute_laplacian(adj_norm, v_)
        return fhn_derivatives(v_, w_, u_, lv_, eta, I_stim, heretic_mask, D_eff,
                                frozen_u=frozen_u)

    k1v, k1w, k1u = deriv(v, w, u)
    k2v, k2w, k2u = deriv(v + 0.5*DT*k1v, w + 0.5*DT*k1w, u + 0.5*DT*k1u)
    k3v, k3w, k3u = deriv(v + 0.5*DT*k2v, w + 0.5*DT*k2w, u + 0.5*DT*k2u)
    k4v, k4w, k4u = deriv(v + DT*k3v,     w + DT*k3w,     u + DT*k3u)

    v_new = np.clip(v + (DT/6)*(k1v + 2*k2v + 2*k3v + k4v), -100, 100)
    w_new = np.clip(w + (DT/6)*(k1w + 2*k2w + 2*k3w + k4w), -100, 100)
    u_new = np.clip(u + (DT/6)*(k1u + 2*k2u + 2*k3u + k4u), *U_CLAMP)
    return v_new, w_new, u_new


def get_adj_norm(adj):
    """Normalisation degree_linear -- identique a topology.py."""
    deg = adj.sum(axis=1, keepdims=True)
    deg = np.where(deg == 0, 1.0, deg)
    return adj / deg


def run_comparison(adj, heretic_mask, seed, I_stim, frozen_u=False, label=""):
    rng = np.random.RandomState(seed)
    N = adj.shape[0]
    D_eff = D_EFF_FAC / np.sqrt(N)
    adj_norm = get_adj_norm(adj)

    # --- Warmup Euler commun (etat partage) ---
    v = np.zeros(N); w = np.zeros(N); u = np.full(N, 0.5)
    for _ in range(N_WARM):
        v, w, u = euler_step(v, w, u, adj_norm, heretic_mask, D_eff, rng, I_stim,
                              frozen_u=frozen_u)

    # --- Snapshot etat initial post-warmup ---
    v0, w0, u0  = v.copy(), w.copy(), u.copy()
    rng_state0  = rng.get_state()

    # --- Euler forward ---
    ve, we, ue = v0.copy(), w0.copy(), u0.copy()
    rng.set_state(rng_state0)
    snaps_e = []
    for t in range(N_STEPS):
        ve, we, ue = euler_step(ve, we, ue, adj_norm, heretic_mask, D_eff, rng, I_stim,
                                 frozen_u=frozen_u)
        if t % 10 == 0:
            snaps_e.append(ve.copy())

    # --- RK4 forward (meme etat initial, meme seed RNG) ---
    vr, wr, ur = v0.copy(), w0.copy(), u0.copy()
    rng.set_state(rng_state0)
    snaps_r = []
    for t in range(N_STEPS):
        vr, wr, ur = rk4_step(vr, wr, ur, adj_norm, heretic_mask, D_eff, rng, I_stim,
                                frozen_u=frozen_u)
        if t % 10 == 0:
            snaps_r.append(vr.copy())

    arr_e = np.array(snaps_e)
    arr_r = np.array(snaps_r)

    h_euler   = float(np.mean([calculate_cognitive_entropy(s) for s in arr_e[::5]]))
    h_rk4     = float(np.mean([calculate_cognitive_entropy(s) for s in arr_r[::5]]))
    sy_euler  = float(calculate_pairwise_synchrony(arr_e))
    sy_rk4    = float(calculate_pairwise_synchrony(arr_r))
    mae_v     = float(np.mean(np.abs(arr_e - arr_r)))
    max_mae_v = float(np.max(np.abs(arr_e - arr_r)))

    return {
        'label': label, 'seed': seed, 'frozen_u': frozen_u,
        'h_euler': h_euler, 'h_rk4': h_rk4,
        'sy_euler': sy_euler, 'sy_rk4': sy_rk4,
        'mae_v': mae_v, 'max_mae_v': max_mae_v,
    }


def main():
    print("=" * 70)
    print("RK4 vs Euler -- dt=0.05, N_warm=1000, N_steps=2000, PLASTICITE=ON")
    print("Params alignes sur config.yaml + dynamics.py (corrige 2026-04-29)")
    print("SIG_STEEP=pi (%.4f), SOC_LEAK=%.2f, EPS_U=%.3f, TAU_U=%.1f" % (
          SIG_STEEP, SOC_LEAK, EPS_U, TAU_U))
    print("LAM_LEARN=%.2f (actif), TAU_PLAST=%d, W_SAT=%.1f" % (
          LAM_LEARN, TAU_PLAST, W_SAT))
    print("Seeds=%d, I_stim=%.1f" % (N_SEEDS, I_STIM))
    print("=" * 70 + "\n")
    t0 = time.time()

    results = []

    conditions = [
        ('ba_m3_FULL',     3, False),
        ('ba_m3_FROZEN_U', 3, True),
        ('ba_m5_FULL',     5, False),
        ('ba_m5_FROZEN_U', 5, True),
    ]

    for label, m, frozen_u in conditions:
        seed_rows = []
        for seed in range(N_SEEDS):
            adj = make_ba(GRID * GRID, m, seed=42)  # structure fixe, 100 noeuds
            N   = adj.shape[0]
            # Masque heretique deterministe (15%, uniforme)
            step_h = max(int(1.0 / 0.15), 1)
            rng_h  = np.random.RandomState(seed)
            heretic_ids = []
            for i in range(0, N, step_h):
                block_end = min(i + step_h, N)
                heretic_ids.append(rng_h.randint(i, block_end))
            heretic_mask = np.zeros(N, dtype=bool)
            heretic_mask[heretic_ids] = True

            r = run_comparison(adj, heretic_mask, seed=seed, I_stim=I_STIM,
                               frozen_u=frozen_u, label=label)
            seed_rows.append(r)
            results.append(r)

        mae_m   = np.mean([r['mae_v']     for r in seed_rows])
        maxm_m  = np.mean([r['max_mae_v'] for r in seed_rows])
        h_e_m   = np.mean([r['h_euler']   for r in seed_rows])
        h_r_m   = np.mean([r['h_rk4']    for r in seed_rows])
        sy_e_m  = np.mean([r['sy_euler']  for r in seed_rows])
        sy_r_m  = np.mean([r['sy_rk4']   for r in seed_rows])
        print("%22s | MAE_mean=%.4f MAE_max=%.4f | H euler=%.4f rk4=%.4f D=%.4f | sync euler=%.4f rk4=%.4f" % (
              label, mae_m, maxm_m, h_e_m, h_r_m, abs(h_e_m - h_r_m), sy_e_m, sy_r_m), flush=True)

    # CSV
    csv_path = os.path.join(OUTPUT_DIR, 'rk4_vs_euler_plasticity_on.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print("\nCSV -> %s" % csv_path)

    # --- Surge ratio FROZEN_U / FULL ---
    print("\n" + "=" * 70)
    print("CLAIM PRINCIPAL : synchrony surge FROZEN_U / FULL")
    print("=" * 70)
    for m in [3, 5]:
        full_e  = np.mean([r['sy_euler'] for r in results if r['label'] == 'ba_m%d_FULL' % m])
        froz_e  = np.mean([r['sy_euler'] for r in results if r['label'] == 'ba_m%d_FROZEN_U' % m])
        full_r  = np.mean([r['sy_rk4']  for r in results if r['label'] == 'ba_m%d_FULL' % m])
        froz_r  = np.mean([r['sy_rk4']  for r in results if r['label'] == 'ba_m%d_FROZEN_U' % m])
        surge_e = (froz_e / full_e - 1) * 100 if full_e > 1e-9 else float('nan')
        surge_r = (froz_r / full_r - 1) * 100 if full_r > 1e-9 else float('nan')
        delta_pp = abs(surge_e - surge_r)
        flag = "OK" if delta_pp < 10 else "ATTENTION"
        print("  BA m=%d: Euler surge=%+.1f%%  RK4 surge=%+.1f%%  delta=%.1fpp  [%s]" % (
              m, surge_e, surge_r, delta_pp, flag))

    # --- Verdict H_cog ---
    print("\n" + "=" * 70)
    print("VERDICT H_cog (seuil : Delta > 0.05 = artefact potentiel)")
    print("=" * 70)
    all_h_deltas = []
    for label, m, frozen_u in conditions:
        h_e = np.mean([r['h_euler'] for r in results if r['label'] == label])
        h_r = np.mean([r['h_rk4']  for r in results if r['label'] == label])
        delta = abs(h_e - h_r)
        all_h_deltas.append(delta)
        flag = "OK" if delta < 0.05 else "*** ATTENTION"
        print("  %22s | Euler=%.4f RK4=%.4f Delta=%.4f  [%s]" % (
              label, h_e, h_r, delta, flag))

    # --- Verdict global ---
    print("\n" + "=" * 70)
    print("VERDICT GLOBAL (plasticite=ON)")
    print("=" * 70)
    max_h_delta = max(all_h_deltas)
    surge_deltas = []
    for m in [3, 5]:
        full_e = np.mean([r['sy_euler'] for r in results if r['label'] == 'ba_m%d_FULL' % m])
        froz_e = np.mean([r['sy_euler'] for r in results if r['label'] == 'ba_m%d_FROZEN_U' % m])
        full_r = np.mean([r['sy_rk4']  for r in results if r['label'] == 'ba_m%d_FULL' % m])
        froz_r = np.mean([r['sy_rk4']  for r in results if r['label'] == 'ba_m%d_FROZEN_U' % m])
        if full_e > 1e-9 and full_r > 1e-9:
            surge_deltas.append(abs((froz_e / full_e - 1) - (froz_r / full_r - 1)) * 100)
    max_surge_delta = max(surge_deltas) if surge_deltas else float('nan')

    print("  Max Delta(H_cog) Euler vs RK4      : %.4f  [%s]" % (
          max_h_delta, "OK" if max_h_delta < 0.05 else "ATTENTION"))
    print("  Max Delta(surge ratio) Euler vs RK4 : %.1fpp  [%s]" % (
          max_surge_delta, "OK" if max_surge_delta < 10 else "ATTENTION"))

    if max_h_delta < 0.05 and max_surge_delta < 10:
        print("\n  -> EULER dt=0.05 VALIDE (plasticite=ON) -- pas d'artefact detecte.")
    else:
        print("\n  -> ATTENTION -- artefacts potentiels detectes, reverifier.")

    print("\nTotal: %.1fs" % (time.time() - t0))


if __name__ == '__main__':
    main()
