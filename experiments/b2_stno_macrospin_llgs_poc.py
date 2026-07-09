#!/usr/bin/env python3
"""
B2/B5/B6 — Macrospin LLGS complet (le vrai vecteur d'aimantation, pas une reduction) — 2026-07-09
Claude Code (Opus 4.8) / Julien Chauvin — palier choisi explicitement par Julien apres verif
materiel (RTX 3070 8Go, Ryzen 7 5800H, 32Go RAM — largement suffisant pour ce niveau, PAS
besoin de GPU ici) : macrospin LLGS maintenant, micromagnetisme complet (mumax3) reporte a
une decision separee.

CE QUI CHANGE VS LES 2 POCs PRECEDENTS : b2_stno_phase_coupling_poc.py (Kuramoto) et
b2_stno_amplitude_phase_poc.py (Slavin-Tiberkevich, amplitude+phase scalaires) sont tous les
deux des REDUCTIONS PHENOMENOLOGIQUES. Ici, l'aimantation `m_i` est un vrai vecteur unite 3D
integre par l'equation de Landau-Lifshitz-Gilbert-Slonczewski (LLGS) explicite -- la meme
equation que celle que Slavin-Tiberkevich reduisent formellement en amplitude+phase. On ne
suppose plus la forme de l'oscillateur, on la fait emerger de la precession vectorielle.

MODELE (par noeud i, m_i in R^3, |m_i|=1, polariseur p=z fixe) :
  gamma_prime = gamma/(1+alpha^2)
  H_eff_i = (H_k*m_i,z + h_ext_i)*z_hat + K_coupling*u_filter_i*S_i + bruit_thermique_i
  S_i = (1/deg_i) * sum_j A_ij (m_j - m_i)                    [champ de couplage, vectoriel]
  dm_i/dt = -gamma_prime*(m_i x H_eff_i) - gamma_prime*alpha*(m_i x (m_i x H_eff_i))
            + gamma_prime*beta*(m_i x (m_i x p)) - gamma_prime*alpha*beta*(m_i x p)
  [Slonczewski Damping-like torque uniquement ; le terme "field-like" (plus petit,
  souvent neglige en premiere approche) est omis -- simplification assumee, pas cachee.]
  sigma_social_i = |S_i| ; u_filter_i, du IDENTIQUES a dynamics.py (aucun reglage propre).

VERIFICATION PHYSIQUE PREALABLE (avant ce script, pas dans le CSV) :
  - Isole (sans couplage), a alpha=0.02, H_ext=1.0, H_k=0.3, un tilt initial converge vers
    un CONE DE PRECESSION STABLE (pas un point fixe, pas un renversement complet) des que
    beta depasse un seuil ; angle de cone continument ajustable par beta (0.005->0.6 deg,
    0.025->20 deg env., 0.04->133 deg = renversement). Comportement STT qualitativement
    correct (regime "oscillateur" distinct du regime "commutation").
  - La frequence de precession depend de H_k*m_z (verifie : 1.00 a H_k=0, 1.26 a H_k=0.3,
    1.56 a H_k=0.6) -- une non-isochronicite emerge NATURELLEMENT de l'anisotropie, sans
    qu'on ait besoin d'ajouter un terme phenomenologique comme le N*p du POC precedent.
    Bonne nouvelle de coherence : ce que Slavin-Tiberkevich modelisent par un parametre
    ad hoc a une origine physique claire ici.

DECOUVERTE DE CALIBRATION (documentee, pas cachee) : un test minimal a 2 macrospins couples
directement montre que CETTE geometrie de couplage (champ effectif vectoriel, couple
gyroscopique) verrouille les oscillateurs en **ANTIPHASE** (dphi -> pi), pas en phase comme
les 2 POCs precedents. C'est un phenomene reel et documente pour les oscillateurs gyrotropes
couples (la nature du canal de couplage -- dipolaire vs electrique -- determine le signe
effectif du verrouillage dans la litterature STNO reelle). Consequence : le parametre d'ordre
de Kuramoto standard R = |mean(exp(i*phi))| reste plancher (~1/sqrt(N)) meme quand les
oscillateurs sont parfaitement organises -- il faut mesurer le bon objet : le parametre
d'ordre au 2e harmonique R2 = |mean(exp(2i*phi))|, standard pour detecter un etat a 2 clusters
(antiphase), non ambigu (verifie sur 2 oscillateurs : dphi=pi -> R2=1, R=0).

DECOUVERTE TOPOLOGIQUE (non cherchee, notee) : sur LATTICE (graphe BIPARTITE, se prete a un
damier antiphase globalement coherent), R2 organise fortement (jusqu'a 0.94 en FROZEN_U).
Sur BA m=3 (graphe NON bipartite, cycles impairs), le couplage antiphase est FRUSTRE --
R2 reste bas (~0.15-0.20) dans TOUTES les conditions, doute ou pas. **3e mecanisme
independant (apres B1/B4) ou BA scale-free se comporte differemment de lattice** -- ici pas
comme "cas le plus sensible au doute" mais comme "cas ou aucun ordre n'emerge du tout" sous
cette geometrie de couplage precise. Rapporte tel quel, pas force.

QUESTION (sur LATTICE, ou un ordre existe reellement) : le doute reduit-il R2 comme il
reduit R dans les 2 POCs precedents ?

PROTOCOLE : LATTICE 10x10 (ordre reel) + BA m=3 (controle de frustration), N=100,
10 seeds canoniques, gain_u in {1 (brut), 3 (calibre)}, dt=0.01 (stable jusqu'a gain=8
verifie en calibration).
Sorties : figures/b2_stno_macrospin_llgs_poc.csv / _agg.csv / .png
"""
import csv
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'src'))

from mem4ristor.graph_utils import make_ba, make_lattice_adj

SEEDS = [42, 123, 777, 17, 256, 1337, 99, 314, 2024, 888]
N = 100
DT = 0.01
WARM_UP = 3000
STEPS = 5000

# --- Macrospin LLGS (regime calibre, cf. docstring) ---
GAMMA = 1.0
ALPHA = 0.02       # amortissement de Gilbert, ordre de grandeur typique metal ferromagnetique
H_EXT = 1.0        # champ de biais le long de z
H_K = 0.3          # anisotropie uniaxiale (donne la non-isochronicite naturelle)
BETA = 0.025       # magnitude du couple de Slonczewski (cone de precession ~20 deg isole)
P_POLARIZER = np.array([0., 0., 1.])
SIGMA_H = 0.05     # desordre de champ de biais (variabilite de fabrication)
K_COUPLING = 5.0
SIGMA_THERMAL = 0.01
THETA0 = 0.3       # tilt initial (rad)

# --- Doute : IDENTIQUE a dynamics.py (aucun reglage specifique) ---
EPSILON_U = 0.02
K_U = 1.0
SIGMA_BASELINE = 0.05
TAU_U = 10.0
ALPHA_SURPRISE = 2.0
SURPRISE_CAP = 5.0
SOCIAL_LEAKAGE = 0.01

GAINS_U = [1.0, 3.0]   # 1 = capteur brut ; 3 = capteur calibre (effet quasi maximal, cf. sweep)


def run_one(adj, seed, condition, gain_u):
    rng = np.random.RandomState(seed)
    n = adj.shape[0]
    deg = adj.sum(axis=1)
    deg_safe = np.where(deg > 0, deg, 1.0)

    h_ext_i = H_EXT + rng.normal(0, SIGMA_H, n)
    phi0 = rng.uniform(0, 2 * np.pi, n)
    m = np.stack([np.sin(THETA0) * np.cos(phi0),
                  np.sin(THETA0) * np.sin(phi0),
                  np.full(n, np.cos(THETA0))], axis=1)
    u = np.full(n, SIGMA_BASELINE)
    gp = GAMMA / (1.0 + ALPHA ** 2)

    R2_traj, u_traj = [], []
    steps_total = WARM_UP + STEPS

    for t in range(steps_total):
        diff = m[None, :, :] - m[:, None, :]                    # [i,j,:] = m_j - m_i
        S = np.einsum('ij,ijk->ik', adj, diff) / deg_safe[:, None]
        sigma_social = np.linalg.norm(S, axis=1)

        sigma_social_for_u = np.zeros(n) if condition == 'FROZEN_U' else sigma_social * gain_u
        u_filter = np.tanh(np.pi * (0.5 - u)) + SOCIAL_LEAKAGE

        hz = H_K * m[:, 2] + h_ext_i
        Heff = np.zeros((n, 3))
        Heff[:, 2] = hz
        Heff = Heff + (u_filter[:, None] * K_COUPLING) * S
        thermal = rng.normal(0, SIGMA_THERMAL, (n, 3)) / np.sqrt(DT)
        Heff = Heff + thermal

        mxH = np.cross(m, Heff)
        mxmxH = np.cross(m, mxH)
        mxp = np.cross(m, P_POLARIZER)
        mxmxp = np.cross(m, mxp)
        dm = (-gp * mxH - gp * ALPHA * mxmxH
              + gp * BETA * mxmxp - gp * ALPHA * BETA * mxp)

        sigma_safe = np.clip(sigma_social_for_u, 0.0, 100.0)
        eps_adapt = EPSILON_U * np.clip(1.0 + ALPHA_SURPRISE * sigma_safe, 1.0, SURPRISE_CAP)
        du = eps_adapt * (K_U * sigma_social_for_u + SIGMA_BASELINE - u) / TAU_U

        m = m + dm * DT
        m = m / np.linalg.norm(m, axis=1, keepdims=True)
        u = np.clip(u + du * DT, 0.0, 1.0)

        if not np.all(np.isfinite(m)):
            raise OverflowError(
                f"Divergence Euler (condition={condition}, gain={gain_u}, seed={seed}, pas={t})")

        if t >= WARM_UP:
            phi = np.arctan2(m[:, 1], m[:, 0])
            R2_traj.append(float(np.abs(np.mean(np.exp(2j * phi)))))
            u_traj.append(float(u.mean()))

    return {
        'R2_mean': float(np.mean(R2_traj)), 'R2_std': float(np.std(R2_traj)),
        'u_mean': float(np.mean(u_traj)), 'u_max': float(np.max(u_traj)),
    }


def bootstrap_ci(a, b, n_boot=5000, seed=0):
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
        'LATTICE_10x10': make_lattice_adj(10, periodic=True),   # bipartite -> ordre antiphase possible
        'BA_m3': make_ba(N, 3, seed=42),                        # non-bipartite -> frustration attendue
    }

    rows = []
    total = len(topologies) * (1 + len(GAINS_U)) * len(SEEDS)
    done = 0

    for topo_name, adj in topologies.items():
        for seed in SEEDS:
            r = run_one(adj, seed, 'FROZEN_U', 1.0)
            rows.append({'topology': topo_name, 'condition': 'FROZEN_U', 'gain_u': None, 'seed': seed, **r})
            done += 1
        sub = [r for r in rows if r['topology'] == topo_name and r['condition'] == 'FROZEN_U']
        print(f"{topo_name:15s} FROZEN_U : R2={np.mean([r['R2_mean'] for r in sub]):.4f}  "
              f"[{done}/{total}, {time.time()-t0:.0f}s]")

        for gain in GAINS_U:
            for seed in SEEDS:
                r = run_one(adj, seed, 'FULL', gain)
                rows.append({'topology': topo_name, 'condition': 'FULL', 'gain_u': gain, 'seed': seed, **r})
                done += 1
            sub = [r for r in rows if r['topology'] == topo_name and r['condition'] == 'FULL' and r['gain_u'] == gain]
            print(f"{topo_name:15s} FULL gain={gain:.0f} : R2={np.mean([r['R2_mean'] for r in sub]):.4f}  "
                  f"u_mean={np.mean([r['u_mean'] for r in sub]):.4f}  u_max={np.mean([r['u_max'] for r in sub]):.4f}  "
                  f"[{done}/{total}, {time.time()-t0:.0f}s]")

    fig_dir = HERE.parent / 'figures'
    fig_dir.mkdir(exist_ok=True)
    raw_path = fig_dir / 'b2_stno_macrospin_llgs_poc.csv'
    with open(raw_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

    agg = []
    print("\n" + "=" * 92)
    print("VERDICT — macrospin LLGS complet : le doute reduit-il l'ordre antiphase (R2) ?")
    print("=" * 92)
    for topo_name in topologies:
        frozen = np.array([r['R2_mean'] for r in rows if r['topology'] == topo_name and r['condition'] == 'FROZEN_U'])
        for gain in GAINS_U:
            full = np.array([r['R2_mean'] for r in rows if r['topology'] == topo_name
                              and r['condition'] == 'FULL' and r['gain_u'] == gain])
            diff = frozen.mean() - full.mean()
            pooled_std = np.sqrt((full.var(ddof=1) + frozen.var(ddof=1)) / 2)
            cohen_d = diff / pooled_std if pooled_std > 1e-12 else float('nan')
            ci_lo, ci_hi = bootstrap_ci(full, frozen, seed=hash((topo_name, gain)) % (2**31))
            agg.append({
                'topology': topo_name, 'gain_u': gain,
                'R2_FULL_mean': float(full.mean()), 'R2_FULL_std': float(full.std()),
                'R2_FROZEN_mean': float(frozen.mean()), 'R2_FROZEN_std': float(frozen.std()),
                'diff_FROZEN_minus_FULL': float(diff), 'diff_ci_lo': ci_lo, 'diff_ci_hi': ci_hi,
                'cohen_d': float(cohen_d), 'n_seeds': len(full),
            })
            print(f"{topo_name:15s} gain={gain:.0f} : R2_FULL={full.mean():.4f}+/-{full.std():.4f}  "
                  f"R2_FROZEN={frozen.mean():.4f}+/-{frozen.std():.4f}  "
                  f"diff={diff:+.4f} CI[{ci_lo:+.4f},{ci_hi:+.4f}]  Cohen_d={cohen_d:+.2f}")

    agg_path = fig_dir / 'b2_stno_macrospin_llgs_poc_agg.csv'
    with open(agg_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=agg[0].keys())
        w.writeheader(); w.writerows(agg)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for ax, topo_name in zip(axes, topologies):
            frozen_a = next(a for a in agg if a['topology'] == topo_name and a['gain_u'] == GAINS_U[0])
            labels = ['FROZEN_U'] + [f'FULL (gain={g:.0f})' for g in GAINS_U]
            means = [frozen_a['R2_FROZEN_mean']] + [
                next(a for a in agg if a['topology'] == topo_name and a['gain_u'] == g)['R2_FULL_mean'] for g in GAINS_U]
            stds = [frozen_a['R2_FROZEN_std']] + [
                next(a for a in agg if a['topology'] == topo_name and a['gain_u'] == g)['R2_FULL_std'] for g in GAINS_U]
            ax.bar(labels, means, yerr=stds, color=['gray', 'steelblue', 'crimson'], capsize=4)
            ax.set_ylabel("Parametre d'ordre au 2e harmonique R2 (antiphase)")
            ax.set_title(topo_name)
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.3, axis='y')
        fig.suptitle('Macrospin LLGS complet — le doute reduit-il le verrouillage antiphase ?\n'
                     'LATTICE = ordre reel possible (bipartite) ; BA m3 = frustre (non-bipartite)',
                     fontsize=10)
        plt.tight_layout()
        png = fig_dir / 'b2_stno_macrospin_llgs_poc.png'
        plt.savefig(png, dpi=150, bbox_inches='tight')
        print(f"\nFigure : {png}")
    except Exception as e:
        print(f"[matplotlib] {e}")

    print(f"\nCSV : {raw_path}\n      {agg_path}")
    print(f"Wall time : {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
