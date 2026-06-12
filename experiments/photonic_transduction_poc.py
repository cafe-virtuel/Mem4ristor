#!/usr/bin/env python3
"""
POC Transduction Photonique — Vague 2 (dossier hardware) — 2026-06-12
Claude Code (Fable) / Julien Chauvin — piste V6 "Memristors Photoniques" (notée 05/05/2026)

CONTEXTE HARDWARE : la piste V6 propose de remplacer le stimulus électrique
I_stimulus par une stimulation lumineuse (fibre optique -> matériau photosensible :
GST Ge2Sb2Te5, VO2, WO3). Réf. expérimentale : neurones photoniques à spikes
tout-optiques sur GST (Feldmann et al., Nature 569, 208-214, 2019).

QUESTION : le régime fonctionnel (BA m=3) et la dead zone (BA m=5) survivent-ils
quand I_stimulus est délivré optiquement, c'est-à-dire avec le BRUIT DE PHOTONS ?

MODÈLE DE TRANSDUCTION (minimal, honnête) :
  - Chaque nœud reçoit à chaque pas un nombre de photons k_i(t) ~ Poisson(Λ),
    où Λ = budget de photons par nœud et par pas de temps.
  - I_i(t) = I_nominal · k_i(t)/Λ  (transduction linéaire, photodétection idéale).
  - Bruit relatif imposé par la physique : σ/I = 1/sqrt(Λ) (bruit de grenaille).
  - Les hérétiques gardent leur inversion de polarité (canal optique propre,
    inversion faite à la photodétection) — l'inversion est appliquée par le
    réseau lui-même (masque hérétique), comme dans le protocole électrique.
  - Contrôle : Λ = DETERMINISTIC (pas de bruit de photons) = protocole standard.

CE QUE DIT LE RÉSULTAT :
  - Si les signatures (H_cog(m=3) > 0, H_cog(m=5) ~ 0, sync basse) tiennent jusqu'à
    Λ faible -> la lecture optique est viable et Λ_min est un CHIFFRE DE
    DIMENSIONNEMENT pour le dossier de faisabilité (budget optique par nœud).
  - La conversion en puissance physique exige de fixer l'échelle temporelle réelle
    du pas dt — non fixée par le modèle ; on rapporte donc Λ (photons/nœud/pas),
    sans unité de puissance. (À 1550 nm, E_photon ≈ 1.28e-19 J ; la conversion
    W = Λ·E_photon/Δt_physique sera faite dans le dossier hardware.)

PROTOCOLE : BA m=3 (fonctionnel) et BA m=5 (dead zone), N=100, I_nominal=0.5,
coupling degree_linear, heretic_ratio=0.15, WARM_UP=1000 + STEPS=3000,
10 seeds canoniques. Λ ∈ {3, 10, 30, 100, 300, 1000} + DETERMINISTIC.
Métriques : H_cont, H_cog, synchronie pairwise.

Sorties : figures/photonic_transduction_poc.csv / _agg.csv / .png
"""
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'src'))

from mem4ristor.core import Mem4Network
from mem4ristor.graph_utils import make_ba
from mem4ristor.metrics import (
    calculate_cognitive_entropy,
    calculate_continuous_entropy,
    calculate_pairwise_synchrony,
)

SEEDS = [42, 123, 777, 17, 256, 1337, 99, 314, 2024, 888]   # set canonique Table 1
M_VALUES = [3, 5]                  # fonctionnel / dead zone
LAMBDAS = [3, 10, 30, 100, 300, 1000]   # photons / nœud / pas
I_NOMINAL = 0.5
N = 100
WARM_UP = 1000
STEPS = 3000
HERETIC = 0.15
COUPLING_NORM = 'degree_linear'


def run_one(adj, seed, lam):
    """lam = budget de photons par nœud/pas ; None = contrôle déterministe."""
    net = Mem4Network(adjacency_matrix=adj.copy(), heretic_ratio=HERETIC,
                      seed=seed, coupling_norm=COUPLING_NORM)
    rng_phot = np.random.RandomState(seed + 77000)   # RNG photonique séparé

    def stimulus():
        if lam is None:
            return I_NOMINAL
        # k ~ Poisson(lam) par nœud, transduction linéaire normalisée
        return I_NOMINAL * rng_phot.poisson(lam, N) / lam

    for _ in range(WARM_UP):
        net.step(I_stimulus=stimulus())

    snaps = []
    for _ in range(STEPS):
        net.step(I_stimulus=stimulus())
        snaps.append(net.v.copy())
    v_s = np.array(snaps)

    return {
        'h_cont': float(np.mean([calculate_continuous_entropy(v) for v in v_s[::10]])),
        'h_cog': float(np.mean([calculate_cognitive_entropy(v) for v in v_s[::10]])),
        'sync': float(calculate_pairwise_synchrony(v_s)),
    }


def main():
    import csv
    t0 = time.time()
    conditions = [None] + LAMBDAS    # None = DETERMINISTIC en tête (référence)
    rows = []
    total = len(M_VALUES) * len(conditions) * len(SEEDS)
    done = 0

    for m in M_VALUES:
        adj = make_ba(N, m, seed=42)     # topologie fixe (cohérent ablation C04)
        for lam in conditions:
            label = 'DETERMINISTIC' if lam is None else str(lam)
            for seed in SEEDS:
                r = run_one(adj, seed, lam)
                rows.append({'m': m, 'lambda_photons': label, 'seed': seed, **r})
                done += 1
            sub = [r for r in rows if r['m'] == m and r['lambda_photons'] == label]
            print(f"m={m} Lambda={label:>13s} : "
                  f"H_cont={np.mean([r['h_cont'] for r in sub]):.3f}"
                  f"±{np.std([r['h_cont'] for r in sub]):.3f}  "
                  f"H_cog={np.mean([r['h_cog'] for r in sub]):.4f}  "
                  f"sync={np.mean([r['sync'] for r in sub]):+.4f}  "
                  f"[{done}/{total}, {time.time()-t0:.0f}s]")

    fig_dir = HERE.parent / 'figures'
    raw_path = fig_dir / 'photonic_transduction_poc.csv'
    with open(raw_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

    # Agrégation
    agg = []
    for m in M_VALUES:
        for lam in conditions:
            label = 'DETERMINISTIC' if lam is None else str(lam)
            sub = [r for r in rows if r['m'] == m and r['lambda_photons'] == label]
            agg.append({
                'm': m, 'lambda_photons': label, 'n_seeds': len(sub),
                'h_cont_mean': float(np.mean([r['h_cont'] for r in sub])),
                'h_cont_std': float(np.std([r['h_cont'] for r in sub])),
                'h_cog_mean': float(np.mean([r['h_cog'] for r in sub])),
                'h_cog_std': float(np.std([r['h_cog'] for r in sub])),
                'sync_mean': float(np.mean([r['sync'] for r in sub])),
                'sync_std': float(np.std([r['sync'] for r in sub])),
                'shot_noise_rel': 0.0 if lam is None else 1.0 / np.sqrt(lam),
            })
    agg_path = fig_dir / 'photonic_transduction_poc_agg.csv'
    with open(agg_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=agg[0].keys())
        w.writeheader(); w.writerows(agg)

    # Verdict console
    print("\n" + "=" * 76)
    print("VERDICT — signatures vs budget de photons (référence = DETERMINISTIC)")
    print("=" * 76)
    for m in M_VALUES:
        ref = next(a for a in agg if a['m'] == m and a['lambda_photons'] == 'DETERMINISTIC')
        regime = 'FONCTIONNEL' if m == 3 else 'DEAD ZONE'
        print(f"\nBA m={m} ({regime}) — réf : H_cont={ref['h_cont_mean']:.3f} "
              f"H_cog={ref['h_cog_mean']:.4f} sync={ref['sync_mean']:+.4f}")
        for lam in LAMBDAS:
            a = next(x for x in agg if x['m'] == m and x['lambda_photons'] == str(lam))
            d_hcont = a['h_cont_mean'] - ref['h_cont_mean']
            d_sync = a['sync_mean'] - ref['sync_mean']
            ok = abs(d_hcont) < 0.15 and abs(d_sync) < 0.05
            print(f"  Lambda={lam:>5d} (bruit rel. {100/np.sqrt(lam):4.1f}%) : "
                  f"dH_cont={d_hcont:+.3f}  dH_cog={a['h_cog_mean']-ref['h_cog_mean']:+.4f}  "
                  f"dsync={d_sync:+.4f}  -> {'OK signatures intactes' if ok else 'DEVIATION'}")

    # Figure
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        x_lams = LAMBDAS
        for m, color in zip(M_VALUES, ('steelblue', 'crimson')):
            for ax, key, label in zip(axes,
                                      ('h_cont_mean', 'h_cog_mean', 'sync_mean'),
                                      ('H_cont (bits)', 'H_cog (bits)', 'Pairwise sync')):
                ys = [next(a for a in agg if a['m'] == m and a['lambda_photons'] == str(l))[key]
                      for l in x_lams]
                es = [next(a for a in agg if a['m'] == m and a['lambda_photons'] == str(l))[key.replace('mean', 'std')]
                      for l in x_lams]
                ref = next(a for a in agg if a['m'] == m and a['lambda_photons'] == 'DETERMINISTIC')
                ax.errorbar(x_lams, ys, yerr=es, marker='o', color=color,
                            label=f"BA m={m}" + (" (dead zone)" if m == 5 else " (fonctionnel)"))
                ax.axhline(ref[key], color=color, ls='--', alpha=0.4)
                ax.set_xscale('log')
                ax.set_xlabel('Budget de photons Λ (photons/nœud/pas)')
                ax.set_ylabel(label)
                ax.grid(alpha=0.3)
        axes[0].legend(fontsize=8)
        fig.suptitle('Transduction photonique — signatures vs budget de photons\n'
                     'pointillés = contrôle déterministe (stimulus électrique idéal)',
                     fontsize=11)
        plt.tight_layout()
        png = fig_dir / 'photonic_transduction_poc.png'
        plt.savefig(png, dpi=150, bbox_inches='tight')
        print(f"\nFigure : {png}")
    except Exception as e:
        print(f"[matplotlib] {e}")

    print(f"CSV : {raw_path}\n      {agg_path}")
    print(f"Wall time : {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
