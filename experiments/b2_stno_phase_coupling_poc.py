#!/usr/bin/env python3
"""
B2/B5/B6 — Portabilite du mecanisme du doute vers un substrat spintronique (STNO) — 2026-07-09
Claude Code (Opus 4.8) / Julien Chauvin — suite de SPINTRONIC_PATHWAY.md, "prochaine etape la
moins couteuse" : verifier qu'un couplage module par le desaccord peut inverser de signe sur un
reseau d'oscillateurs de phase. PAS un simulateur LLG/micromagnetique complet.

MODELE : reduction phase-amplitude de Slavin-Tiberkevich, la reduction standard utilisee dans la
litterature STNO pour les questions de synchronisation de reseau (pas l'ingenierie d'un dispositif
unique) — equivalente a un modele de Kuramoto en champ local sur graphe. C'est le meme niveau
d'abstraction que la reduction Kuramoto utilisee par Hong & Strogatz, Phys. Rev. Lett. 106, 054102
(2011) pour montrer que des oscillateurs "contrarians" (couplage de signe negatif) suppriment la
synchronisation globale — c'est exactement le mecanisme qu'on cherche a porter ici : le "doute" de
Mem4ristor (u_filter = tanh(pi*(0.5-u))) EST un mecanisme conformiste/contrarian a bascule.

CE QUI EST PORTE A L'IDENTIQUE (memes equations, memes constantes que dynamics.py) :
  - u_filter = tanh(pi*(0.5-u)) + social_leakage (identique)
  - du = epsilon_u_adaptive * (k_u*sigma_social_for_u + sigma_baseline - u) / tau_u (identique)
  - epsilon_u_adaptive = epsilon_u * clip(1 + alpha_surprise*sigma_social, 1, surprise_cap) (identique)
  - Convention de bruit Euler-Maruyama (eta / sqrt(dt), meme convention qu'AUDIT-024)
  - FROZEN_U implemente EXACTEMENT comme p2_sigma_social_ablation.py : sigma_social_override=0
    dans l'equation de u SEULEMENT (la physique du couplage reste inchangee).

CALIBRATION (documentee, pas cachee — meme esprit que B1d, "1ere calibration ratee, corrigee") :
  Premier essai (sigma_omega=0.15, K=1.2, gain capteur=1) : effet CORRECT en signe (le doute
  reduit R) mais FAIBLE (Cohen d~0.85). Diagnostic : sigma_social = |L_phi_i| est un couplage de
  Kuramoto normalise par degre, BORNE dans [-1,1] par construction (moyenne de sin(.), effet
  d'annulation type-CLT sur les noeuds a fort degre) — contrairement a laplacian_v = coupling_input@v-v
  du modele FHN, qui n'est PAS borne (v n'est pas restreint au cercle unite). Consequence : u ne
  depasse jamais ~0.18 dans ce regime — il ne franchit JAMAIS le seuil de bascule de polarite u=0.5
  (tanh(pi*(0.5-u)) ne change de signe qu'au-dela). Le mecanisme ne peut donc montrer que sa
  modulation "douce" (affaiblissement d'amplitude), pas son effet contrarian qualitatif.
  CORRECTION : ajout d'un gain sur le CAPTEUR de desaccord qui alimente u (sigma_social_for_u =
  gain_u * sigma_social), SANS toucher au canal de couplage physique lui-meme — c'est exactement
  le pattern deja utilise par `sigma_social_override` dans p2_sigma_social_ablation.py (decoupler
  la perception du desaccord de la force de couplage reelle est deja une brique du modele
  original, pas un ajout ad hoc). A gain_u=1 (capteur "brut", aucune interpretation), a gain_u=5
  (capteur amplifie, u franchit effectivement 0.5) — LES DEUX conditions sont rapportees, pas
  seulement la plus favorable.

CE QUI EST NOUVEAU (specifique au substrat de phase) :
  - Couplage de Kuramoto normalise par degre : L_phi_i = (1/deg_i) * sum_j A_ij sin(phi_j - phi_i).
  - Desordre de frequence naturelle omega_i ~ N(omega0, sigma_omega) : variabilite de fabrication
    STNO documentee (theme deja rencontre pour le photonique, PHOTONIC_PATHWAY.md §4quater).
  - Regime de couplage calibre pres de la transition de synchronisation (sigma_omega=0.30,
    K=1.0) — sweep documente dans le journal de session, pas un choix arbitraire : c'est la ou
    R_FROZEN est ni trivialement =1 ni trivialement =bruit (R~1/sqrt(N)), donc ou l'effet du
    doute est mesurable sans etre un artefact de plafond/plancher.

QUESTION : le meme mecanisme reduit-il la synchronisation (parametre d'ordre de Kuramoto R) sur
ce substrat, comme il le fait sur le substrat FHN (b4_ablation_robustness.py, Cohen d=9.4 BA m=3
/ 4.7 lattice) ?

PROTOCOLE : BA m=3 et lattice 10x10 (memes topologies que B4), N=100, 10 seeds canoniques,
WARM_UP=1000 + STEPS=3000, 2 conditions (FULL, FROZEN_U) x 2 gains de capteur (1, 5).
Sorties : figures/b2_stno_phase_coupling_poc.csv / _agg.csv / .png
"""
import csv
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'src'))

from mem4ristor.graph_utils import make_ba, make_lattice_adj

SEEDS = [42, 123, 777, 17, 256, 1337, 99, 314, 2024, 888]   # set canonique
N = 100
DT = 0.05
WARM_UP = 1000
STEPS = 3000

# --- Oscillateur (regime calibre, cf. docstring) ---
OMEGA0 = 1.0
SIGMA_OMEGA = 0.30
K_COUPLING = 1.0
SIGMA_PHASE = 0.05

# --- Doute : IDENTIQUE a dynamics.py (aucun reglage de ses propres constantes) ---
EPSILON_U = 0.02
K_U = 1.0
SIGMA_BASELINE = 0.05
TAU_U = 10.0
ALPHA_SURPRISE = 2.0
SURPRISE_CAP = 5.0
SOCIAL_LEAKAGE = 0.01

GAINS_U = [1.0, 5.0]   # 1 = capteur brut (mecanisme "as-is") ; 5 = capteur amplifie (u franchit 0.5)


def run_one(adj, seed, condition, gain_u):
    rng = np.random.RandomState(seed)
    n = adj.shape[0]
    deg = adj.sum(axis=1)
    deg_safe = np.where(deg > 0, deg, 1.0)

    omega = OMEGA0 + rng.normal(0, SIGMA_OMEGA, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    u = np.full(n, SIGMA_BASELINE)

    R_traj = []
    u_traj = []
    steps_total = WARM_UP + STEPS

    for t in range(steps_total):
        sin_diff = np.sin(phi[None, :] - phi[:, None])     # [i,j] = phi_j - phi_i
        L_phi = (adj * sin_diff).sum(axis=1) / deg_safe    # analogue laplacien_v, borne [-1,1]
        sigma_social = np.abs(L_phi)

        if condition == 'FROZEN_U':
            sigma_social_for_u = np.zeros(n)
        else:
            sigma_social_for_u = sigma_social * gain_u     # gain du CAPTEUR seulement

        u_filter = np.tanh(np.pi * (0.5 - u)) + SOCIAL_LEAKAGE

        eta = rng.normal(0, SIGMA_PHASE, n) / np.sqrt(DT)
        dphi = omega + K_COUPLING * u_filter * L_phi + eta   # couplage PHYSIQUE non amplifie

        sigma_safe = np.clip(sigma_social_for_u, 0.0, 100.0)
        eps_adapt = EPSILON_U * np.clip(1.0 + ALPHA_SURPRISE * sigma_safe, 1.0, SURPRISE_CAP)
        du = eps_adapt * (K_U * sigma_social_for_u + SIGMA_BASELINE - u) / TAU_U

        phi = phi + dphi * DT
        u = np.clip(u + du * DT, 0.0, 1.0)

        if t >= WARM_UP:
            R_traj.append(float(np.abs(np.mean(np.exp(1j * phi)))))
            u_traj.append(float(u.mean()))

    return {
        'R_mean': float(np.mean(R_traj)),
        'R_std': float(np.std(R_traj)),
        'u_mean': float(np.mean(u_traj)),
        'u_max': float(np.max(u_traj)),
    }


def bootstrap_ci(a, b, n_boot=5000, seed=0):
    """IC bootstrap sur (mean(a) - mean(b)), meme esprit que b4_ablation_robustness.py."""
    rng = np.random.RandomState(seed)
    diffs = np.empty(n_boot)
    na, nb = len(a), len(b)
    for k in range(n_boot):
        sa = a[rng.randint(0, na, na)]
        sb = b[rng.randint(0, nb, nb)]
        diffs[k] = sb.mean() - sa.mean()
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def main():
    t0 = time.time()
    topologies = {
        'BA_m3': make_ba(N, 3, seed=42),
        'LATTICE_10x10': make_lattice_adj(10, periodic=True),
    }
    conditions = ['FULL', 'FROZEN_U']

    rows = []
    total = len(topologies) * len(conditions) * len(GAINS_U) * len(SEEDS)
    done = 0
    for topo_name, adj in topologies.items():
        for cond in conditions:
            gains = GAINS_U if cond == 'FULL' else [1.0]   # gain n'a pas de sens pour FROZEN_U (sigma_social_for_u=0)
            for gain in gains:
                for seed in SEEDS:
                    r = run_one(adj, seed, cond, gain)
                    rows.append({'topology': topo_name, 'condition': cond, 'gain_u': gain, 'seed': seed, **r})
                    done += 1
                sub = [r for r in rows if r['topology'] == topo_name and r['condition'] == cond and r['gain_u'] == gain]
                print(f"{topo_name:15s} {cond:10s} gain={gain:3.0f} : "
                      f"R={np.mean([r['R_mean'] for r in sub]):.4f}"
                      f"+/-{np.std([r['R_mean'] for r in sub]):.4f}  "
                      f"u_mean={np.mean([r['u_mean'] for r in sub]):.4f}  "
                      f"u_max={np.mean([r['u_max'] for r in sub]):.4f}  "
                      f"[{done}/{total}, {time.time()-t0:.0f}s]")

    fig_dir = HERE.parent / 'figures'
    fig_dir.mkdir(exist_ok=True)
    raw_path = fig_dir / 'b2_stno_phase_coupling_poc.csv'
    with open(raw_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

    # Agregation + Cohen's d + IC bootstrap (meme esprit que b4_ablation_robustness.py)
    agg = []
    print("\n" + "=" * 86)
    print("VERDICT — le doute reduit-il la synchronisation sur le substrat de phase (STNO) ?")
    print("=" * 86)
    for topo_name in topologies:
        frozen = np.array([r['R_mean'] for r in rows if r['topology'] == topo_name and r['condition'] == 'FROZEN_U'])
        for gain in GAINS_U:
            full = np.array([r['R_mean'] for r in rows if r['topology'] == topo_name
                              and r['condition'] == 'FULL' and r['gain_u'] == gain])
            diff = frozen.mean() - full.mean()
            pooled_std = np.sqrt((full.var(ddof=1) + frozen.var(ddof=1)) / 2)
            cohen_d = diff / pooled_std if pooled_std > 1e-12 else float('nan')
            ci_lo, ci_hi = bootstrap_ci(full, frozen, seed=hash((topo_name, gain)) % (2**31))
            agg.append({
                'topology': topo_name, 'gain_u': gain,
                'R_FULL_mean': float(full.mean()), 'R_FULL_std': float(full.std()),
                'R_FROZEN_mean': float(frozen.mean()), 'R_FROZEN_std': float(frozen.std()),
                'diff_FROZEN_minus_FULL': float(diff), 'diff_ci_lo': ci_lo, 'diff_ci_hi': ci_hi,
                'cohen_d': float(cohen_d), 'n_seeds': len(full),
            })
            print(f"{topo_name:15s} gain={gain:3.0f} : R_FULL={full.mean():.4f}+/-{full.std():.4f}  "
                  f"R_FROZEN={frozen.mean():.4f}+/-{frozen.std():.4f}  "
                  f"diff={diff:+.4f} CI[{ci_lo:+.4f},{ci_hi:+.4f}]  Cohen_d={cohen_d:+.2f}")

    agg_path = fig_dir / 'b2_stno_phase_coupling_poc_agg.csv'
    with open(agg_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=agg[0].keys())
        w.writeheader(); w.writerows(agg)

    # Figure
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for ax, topo_name in zip(axes, topologies):
            frozen_a = next(a for a in agg if a['topology'] == topo_name and a['gain_u'] == GAINS_U[0])
            labels = ['FROZEN_U'] + [f'FULL (gain={g:.0f})' for g in GAINS_U]
            means = [frozen_a['R_FROZEN_mean']] + [
                next(a for a in agg if a['topology'] == topo_name and a['gain_u'] == g)['R_FULL_mean'] for g in GAINS_U]
            stds = [frozen_a['R_FROZEN_std']] + [
                next(a for a in agg if a['topology'] == topo_name and a['gain_u'] == g)['R_FULL_std'] for g in GAINS_U]
            ax.bar(labels, means, yerr=stds, color=['gray', 'steelblue', 'crimson'], capsize=4)
            ax.set_ylabel('Parametre d\'ordre de Kuramoto R')
            ax.set_title(topo_name)
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.3, axis='y')
        fig.suptitle('Portabilite du mecanisme du doute -> substrat STNO (oscillateurs de phase)\n'
                     'FROZEN_U = controle (u fige) ; FULL = doute actif, a 2 gains de capteur',
                     fontsize=10)
        plt.tight_layout()
        png = fig_dir / 'b2_stno_phase_coupling_poc.png'
        plt.savefig(png, dpi=150, bbox_inches='tight')
        print(f"\nFigure : {png}")
    except Exception as e:
        print(f"[matplotlib] {e}")

    print(f"\nCSV : {raw_path}\n      {agg_path}")
    print(f"Wall time : {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
