#!/usr/bin/env python3
"""
B2/B5/B6 — Generalisation amplitude+phase (Slavin-Tiberkevich) du POC STNO — 2026-07-09
Claude Code (Opus 4.8) / Julien Chauvin — suite immediate de b2_stno_phase_coupling_poc.py.
Julien : "tu m'as mis l'eau a la bouche je veux voir ce que ca donne" -> palier suivant.

POURQUOI CE PALIER : b2_stno_phase_coupling_poc.py utilisait un modele de Kuramoto pur
(phase seule, amplitude fixee a 1). C'est le cas limite ISOCHRONE d'un modele plus complet
et plus fidele a la litterature STNO : l'oscillateur auto-entretenu non-lineaire de
Slavin & Tiberkevich (IEEE Trans. Magn. 2009 "Nonlinear Auto-Oscillator Theory of Microwave
Generation by Spin-Polarized Current"), qui derive formellement de l'equation LLGS complete
et qui EST le modele utilise par le domaine pour les questions de synchronisation de reseau.
Sa signature physique centrale, absente du modele Kuramoto pur : le DECALAGE DE FREQUENCE
NON-LINEAIRE (non-isochronicite) omega(p) = omega0 + N*p, ou p = |a|^2 est la puissance
d'oscillation. C'est precisement ce que Slavin-Tiberkevich identifient comme LA difference
qualitative entre un STNO et un oscillateur "conventionnel" (isochrone). Un test de
portabilite qui ignore ce terme n'a testé qu'un cas particulier, pas le regime STNO reel.

MODELE (par nœud i, amplitude complexe a_i, p_i=|a_i|^2) :
  da_i/dt = [ (Gamma_plus - Gamma_minus*(1+Q*p_i)) + i*(omega_i + N_nonlin*p_i) ] * a_i
            + K_coupling * u_filter_i * S_i + bruit complexe
  S_i = (1/deg_i) * sum_j A_ij (a_j - a_i)     (couplage complexe, generalise sin(phi_j-phi_i)
                                                 du POC precedent — reactif ET dissipatif)
  sigma_social_i = |S_i| ; u_filter_i, du IDENTIQUES a dynamics.py (aucun reglage propre).
  Sans couplage ni bruit : p -> p* = (Gamma_plus/Gamma_minus - 1)/Q (verifie numeriquement,
  cf. calibration ci-dessous), omega -> omega0 + N_nonlin*p* (frequence d'equilibre decalee).

VERIFICATION DE CONTINUITE (fait avant ce script, pas dans le CSV) : a N_nonlin=0 et
sigma_omega/K_coupling dans le meme regime, ce modele reproduit qualitativement le POC
Kuramoto precedent (le doute reduit R, effet croissant avec le gain du capteur) — ce n'est
PAS un nouveau mecanisme, c'est une generalisation qui contient l'ancien modele comme cas limite.

CALIBRATION NUMERIQUE (documentee, pas cachee — meme esprit que B1d et le stiffness proof
Euler du 1er mai) : a dt=0.01 (echelle de temps naturelle Gamma_minus=1), gain_u=10 et
N_nonlin>=10 font DIVERGER l'integration Euler explicite (overflow, confirme non-physique
par un test a dt decroissant : dt=0.005 et en-dessous restent finis et convergent vers la
meme valeur a dt plus fin). CORRECTION : dt=0.005 pour toute la campagne (marge de securite
sur toute la plage de N_nonlin testee), pas de RK adaptatif (incoherent avec le bruit
stochastique, meme reserve que dynamics.py:411 sur RK45).

QUESTION : le mecanisme (identique a dynamics.py) reduit-il la synchronisation quand on
ajoute la signature physique manquante du modele precedent (non-isochronicite) ? Reste-t-il
mesurable a mesure que N_nonlin croit (litterature : STNO a vortex = MOINS non-isochrones
que les STNO a mode uniforme, valeur precise non trouvee par recherche web -> teste sur une
plage 0-10, pas une valeur unique affirmee).

PROTOCOLE : BA m=3 et lattice 10x10 (memes topologies que B4/POC precedent), N=100,
10 seeds canoniques, N_nonlin in {0, 3, 10}, gain_u in {1 (brut), 10 (calibre)} pour FULL,
FROZEN_U par N_nonlin (gain sans effet, sigma_social_for_u=0).
Sorties : figures/b2_stno_amplitude_phase_poc.csv / _agg.csv / .png
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
DT = 0.005                 # cf. calibration numerique (docstring) — Euler diverge a dt=0.01
WARM_UP = 4000
STEPS = 8000

# --- Oscillateur auto-entretenu (Slavin-Tiberkevich), regime calibre ---
GAMMA_MINUS = 1.0
GAMMA_PLUS = 1.2            # p* isole = (Gamma_plus/Gamma_minus - 1)/Q = 0.2
Q = 1.0
OMEGA0 = 1.0
SIGMA_OMEGA = 0.15
K_COUPLING = 0.3
SIGMA_NOISE = 0.02
N_NONLIN_VALUES = [0.0, 3.0, 10.0]   # decalage de frequence non-lineaire (non-isochronicite)

# --- Doute : IDENTIQUE a dynamics.py (aucun reglage specifique) ---
EPSILON_U = 0.02
K_U = 1.0
SIGMA_BASELINE = 0.05
TAU_U = 10.0
ALPHA_SURPRISE = 2.0
SURPRISE_CAP = 5.0
SOCIAL_LEAKAGE = 0.01

GAINS_U = [1.0, 10.0]        # 1 = capteur brut ; 10 = capteur calibre (u franchit 0.5)


def run_one(adj, seed, condition, gain_u, n_nonlin):
    rng = np.random.RandomState(seed)
    n = adj.shape[0]
    deg = adj.sum(axis=1)
    deg_safe = np.where(deg > 0, deg, 1.0)

    omega = OMEGA0 + rng.normal(0, SIGMA_OMEGA, n)
    a = 0.05 * (rng.randn(n) + 1j * rng.randn(n))     # depart pres de 0, croissance auto-entretenue
    u = np.full(n, SIGMA_BASELINE)

    R_traj, p_traj, u_traj = [], [], []
    steps_total = WARM_UP + STEPS

    for t in range(steps_total):
        diff = a[None, :] - a[:, None]              # [i,j] = a_j - a_i
        S = (adj * diff).sum(axis=1) / deg_safe      # couplage complexe, generalise L_phi
        sigma_social = np.abs(S)

        sigma_social_for_u = np.zeros(n) if condition == 'FROZEN_U' else sigma_social * gain_u
        u_filter = np.tanh(np.pi * (0.5 - u)) + SOCIAL_LEAKAGE

        p = np.abs(a) ** 2
        growth = GAMMA_PLUS - GAMMA_MINUS * (1.0 + Q * p)
        domega = omega + n_nonlin * p
        eta = (rng.normal(0, SIGMA_NOISE, n) + 1j * rng.normal(0, SIGMA_NOISE, n)) / np.sqrt(DT)
        da = (growth + 1j * domega) * a + K_COUPLING * u_filter * S + eta

        sigma_safe = np.clip(sigma_social_for_u, 0.0, 100.0)
        eps_adapt = EPSILON_U * np.clip(1.0 + ALPHA_SURPRISE * sigma_safe, 1.0, SURPRISE_CAP)
        du = eps_adapt * (K_U * sigma_social_for_u + SIGMA_BASELINE - u) / TAU_U

        a = a + da * DT
        u = np.clip(u + du * DT, 0.0, 1.0)

        if not np.all(np.isfinite(a)):
            raise OverflowError(
                f"Divergence Euler (condition={condition}, gain={gain_u}, N_nonlin={n_nonlin}, "
                f"seed={seed}, pas={t}) — voir calibration numerique dans le docstring.")

        if t >= WARM_UP:
            phi = np.angle(a)
            R_traj.append(float(np.abs(np.mean(np.exp(1j * phi)))))
            p_traj.append(float(p.mean()))
            u_traj.append(float(u.mean()))

    return {
        'R_mean': float(np.mean(R_traj)), 'R_std': float(np.std(R_traj)),
        'p_mean': float(np.mean(p_traj)),
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
        'BA_m3': make_ba(N, 3, seed=42),
        'LATTICE_10x10': make_lattice_adj(10, periodic=True),
    }

    rows = []
    total = len(topologies) * len(N_NONLIN_VALUES) * (1 + len(GAINS_U)) * len(SEEDS)
    done = 0

    for topo_name, adj in topologies.items():
        for n_nonlin in N_NONLIN_VALUES:
            # FROZEN_U : gain sans effet (sigma_social_for_u=0), une seule fois par (topo, n_nonlin)
            for seed in SEEDS:
                r = run_one(adj, seed, 'FROZEN_U', 1.0, n_nonlin)
                rows.append({'topology': topo_name, 'n_nonlin': n_nonlin, 'condition': 'FROZEN_U',
                             'gain_u': None, 'seed': seed, **r})
                done += 1
            sub = [r for r in rows if r['topology'] == topo_name and r['n_nonlin'] == n_nonlin
                   and r['condition'] == 'FROZEN_U']
            print(f"{topo_name:15s} N_nonlin={n_nonlin:5.1f} FROZEN_U : "
                  f"R={np.mean([r['R_mean'] for r in sub]):.4f}  "
                  f"p={np.mean([r['p_mean'] for r in sub]):.4f}  [{done}/{total}, {time.time()-t0:.0f}s]")

            for gain in GAINS_U:
                for seed in SEEDS:
                    r = run_one(adj, seed, 'FULL', gain, n_nonlin)
                    rows.append({'topology': topo_name, 'n_nonlin': n_nonlin, 'condition': 'FULL',
                                 'gain_u': gain, 'seed': seed, **r})
                    done += 1
                sub = [r for r in rows if r['topology'] == topo_name and r['n_nonlin'] == n_nonlin
                       and r['condition'] == 'FULL' and r['gain_u'] == gain]
                print(f"{topo_name:15s} N_nonlin={n_nonlin:5.1f} FULL gain={gain:5.1f} : "
                      f"R={np.mean([r['R_mean'] for r in sub]):.4f}  "
                      f"p={np.mean([r['p_mean'] for r in sub]):.4f}  "
                      f"u_mean={np.mean([r['u_mean'] for r in sub]):.4f}  "
                      f"u_max={np.mean([r['u_max'] for r in sub]):.4f}  "
                      f"[{done}/{total}, {time.time()-t0:.0f}s]")

    fig_dir = HERE.parent / 'figures'
    fig_dir.mkdir(exist_ok=True)
    raw_path = fig_dir / 'b2_stno_amplitude_phase_poc.csv'
    with open(raw_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

    agg = []
    print("\n" + "=" * 92)
    print("VERDICT — le doute reduit-il R quand on ajoute amplitude + non-isochronicite (STNO reel) ?")
    print("=" * 92)
    for topo_name in topologies:
        for n_nonlin in N_NONLIN_VALUES:
            frozen = np.array([r['R_mean'] for r in rows if r['topology'] == topo_name
                                and r['n_nonlin'] == n_nonlin and r['condition'] == 'FROZEN_U'])
            for gain in GAINS_U:
                full = np.array([r['R_mean'] for r in rows if r['topology'] == topo_name
                                  and r['n_nonlin'] == n_nonlin and r['condition'] == 'FULL'
                                  and r['gain_u'] == gain])
                diff = frozen.mean() - full.mean()
                pooled_std = np.sqrt((full.var(ddof=1) + frozen.var(ddof=1)) / 2)
                cohen_d = diff / pooled_std if pooled_std > 1e-12 else float('nan')
                ci_lo, ci_hi = bootstrap_ci(full, frozen, seed=hash((topo_name, n_nonlin, gain)) % (2**31))
                agg.append({
                    'topology': topo_name, 'n_nonlin': n_nonlin, 'gain_u': gain,
                    'R_FULL_mean': float(full.mean()), 'R_FULL_std': float(full.std()),
                    'R_FROZEN_mean': float(frozen.mean()), 'R_FROZEN_std': float(frozen.std()),
                    'diff_FROZEN_minus_FULL': float(diff), 'diff_ci_lo': ci_lo, 'diff_ci_hi': ci_hi,
                    'cohen_d': float(cohen_d), 'n_seeds': len(full),
                })
                print(f"{topo_name:15s} N_nonlin={n_nonlin:5.1f} gain={gain:5.1f} : "
                      f"R_FULL={full.mean():.4f}+/-{full.std():.4f}  R_FROZEN={frozen.mean():.4f}+/-{frozen.std():.4f}  "
                      f"diff={diff:+.4f} CI[{ci_lo:+.4f},{ci_hi:+.4f}]  Cohen_d={cohen_d:+.2f}")

    agg_path = fig_dir / 'b2_stno_amplitude_phase_poc_agg.csv'
    with open(agg_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=agg[0].keys())
        w.writeheader(); w.writerows(agg)

    # Figure : Cohen's d vs N_nonlin, pour les 2 topologies, au gain calibre
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for ax, gain in zip(axes, GAINS_U):
            for topo_name, color in zip(topologies, ('steelblue', 'crimson')):
                ds = [next(a for a in agg if a['topology'] == topo_name and a['n_nonlin'] == nn
                           and a['gain_u'] == gain)['cohen_d'] for nn in N_NONLIN_VALUES]
                ax.plot(N_NONLIN_VALUES, ds, marker='o', color=color, label=topo_name)
            ax.axhline(0, color='gray', ls='--', alpha=0.5)
            ax.set_xlabel('N_nonlin (non-isochronicite)')
            ax.set_ylabel("Cohen's d (FROZEN_U - FULL)")
            ax.set_title(f'gain capteur = {gain:.0f}')
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=8)
        fig.suptitle('Robustesse du mecanisme du doute a la non-isochronicite (signature STNO)',
                     fontsize=10)
        plt.tight_layout()
        png = fig_dir / 'b2_stno_amplitude_phase_poc.png'
        plt.savefig(png, dpi=150, bbox_inches='tight')
        print(f"\nFigure : {png}")
    except Exception as e:
        print(f"[matplotlib] {e}")

    print(f"\nCSV : {raw_path}\n      {agg_path}")
    print(f"Wall time : {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
