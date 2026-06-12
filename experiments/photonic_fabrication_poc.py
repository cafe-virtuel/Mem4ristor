#!/usr/bin/env python3
"""
POC Variabilite de fabrication optique — Vague 2, etape 4 du PHOTONIC_PATHWAY
2026-06-12 — Claude Code (Fable) / Julien Chauvin

SUITE DE : photonic_coupling_poc.py (tout-optique OK a Lambda=10),
photonic_gst_transduction_poc.py (saturation+inertie OK),
photonic_event_poc.py (evenements transmis).

QUESTION (le dernier mur physique) : dans une puce reelle, chaque guide d'onde
a ses propres PERTES D'INSERTION — un gain statique different par noeud, fige a
la fabrication. Contrairement au bruit temporel (que u filtre), c'est une
heterogeneite STRUCTURELLE : des noeuds "faibles" permanents, pseudo-heretiques
involontaires. Les regimes survivent-ils ?

MODELE : pour chaque realisation de puce (= seed), on tire UNE FOIS par noeud :
    t_i ~ Normal(1, sigma_fab), clippe a [0.1, +inf)   (transmission relative)
  - Canal stimulus : photons k ~ Poisson(Lambda * t_stim_i * I_target/I_REF)
    -> la perte reduit le flux AVANT photodetection (physique correcte).
  - Canal couplage : l_v_i -> l_v_i * t_coup_i, puis shot noise Poisson(Lambda)/Lambda.
  - t_stim et t_coup independants (deux jeux de guides distincts).
Le systeme est TOUT-OPTIQUE (Lambda=10 sur les deux canaux = budget valide).

SWEEP : sigma_fab in {0, 0.05, 0.10, 0.20, 0.30, 0.50}. A 0.50, ~16% des
guides perdent plus de la moitie du signal — fabrication tres mediocre.
References : ELEC (deterministe, canaux parfaits) + FULLOPT sigma_fab=0
(= condition FULLOPT_10 du POC couplage, comparabilite directe).

PROTOCOLE : BA m=3 + m=5, N=100, I_nominal=0.5, heretic_ratio=0.15,
degree_linear, WARM_UP=1000 + STEPS=3000, 10 seeds canoniques
(1 seed = 1 realisation de puce ET 1 realisation dynamique). 140 runs.

Sorties : figures/photonic_fabrication_poc.csv / _agg.csv / .png
(prints ASCII uniquement — regle du 12/06)
"""
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'src'))

from mem4ristor.core import Mem4Network
from mem4ristor.graph_utils import make_ba
from mem4ristor.metrics import (
    calculate_cognitive_entropy,
    calculate_continuous_entropy,
    calculate_pairwise_synchrony,
)

SEEDS = [42, 123, 777, 17, 256, 1337, 99, 314, 2024, 888]
M_VALUES = [3, 5]
SIGMA_FAB = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]
LAM = 10
I_NOMINAL = 0.5
I_REF = 1.0
N = 100
WARM_UP = 1000
STEPS = 3000
HERETIC = 0.15
COUPLING_NORM = 'degree_linear'


class FabricatedPhotonicNet(Mem4Network):
    """Tout-optique avec pertes d'insertion statiques par noeud.

    Meme patron que PhotonicCouplingNet (photonic_coupling_poc.py), verifie
    identique bit a bit au parent quand canaux parfaits. Ajout : t_coup (gains
    statiques) sur le couplage avant le shot noise.
    """

    def __init__(self, *args, lam_coupling=None, t_coup=None, phot_seed=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.lam_coupling = lam_coupling
        self.t_coup = t_coup
        self._rng_phot_coup = np.random.RandomState(phot_seed + 88000)

    def step(self, I_stimulus=0.0, sigma_v_vec=None, sigma_social_override=None):
        self._doubt_driven_rewire()
        if self._weights_dirty:
            self._rebuild_laplacian()
            self._compute_coupling_weights()
        l_v = -(self.L @ self.v)
        if self.coupling_norm != 'uniform':
            D = self.model.cfg['coupling']['D']
            if D > 0:
                uniform_D_eff = D / np.sqrt(self.N)
                scale_factors = (self.node_weights * D) / uniform_D_eff
                l_v = l_v * scale_factors
            else:
                l_v = np.zeros_like(l_v)
        if self.t_coup is not None:
            l_v = l_v * self.t_coup            # pertes d'insertion statiques
        if self.lam_coupling is not None:
            k = self._rng_phot_coup.poisson(self.lam_coupling, self.N)
            l_v = l_v * (k / self.lam_coupling)  # shot noise
        if self.adjacency_matrix is not None:
            self.model._adj_matrix = self.adjacency_matrix
        self.model.step(I_stimulus, l_v, sigma_v_vec=sigma_v_vec,
                        sigma_social_override=sigma_social_override)


def run_one(adj, seed, sigma_fab, electrical=False):
    rng_fab = np.random.RandomState(seed + 55000)
    if electrical:
        t_stim = t_coup = None
    else:
        t_stim = np.clip(rng_fab.normal(1.0, sigma_fab, N), 0.1, None)
        t_coup = np.clip(rng_fab.normal(1.0, sigma_fab, N), 0.1, None)

    net = FabricatedPhotonicNet(
        adjacency_matrix=adj.copy(), heretic_ratio=HERETIC, seed=seed,
        coupling_norm=COUPLING_NORM,
        lam_coupling=None if electrical else LAM,
        t_coup=t_coup, phot_seed=seed)
    rng_stim = np.random.RandomState(seed + 77000)

    def stimulus():
        if electrical:
            return I_NOMINAL
        lam_vec = LAM * t_stim * I_NOMINAL / I_REF
        k = rng_stim.poisson(lam_vec)
        return I_REF * k / LAM      # moyenne = t_stim * I_NOMINAL

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
    conds = [('ELEC', None)] + [(f'fab{s:g}', s) for s in SIGMA_FAB]
    rows = []
    total = len(M_VALUES) * len(conds) * len(SEEDS)
    done = 0
    for m in M_VALUES:
        adj = make_ba(N, m, seed=42)
        for label, s_fab in conds:
            for seed in SEEDS:
                r = run_one(adj, seed, s_fab if s_fab is not None else 0.0,
                            electrical=(label == 'ELEC'))
                rows.append({'m': m, 'condition': label,
                             'sigma_fab': '' if s_fab is None else s_fab,
                             'seed': seed, **r})
                done += 1
            sub = [r for r in rows if r['m'] == m and r['condition'] == label]
            print(f"m={m} {label:>8s} : "
                  f"H_cont={np.mean([r['h_cont'] for r in sub]):.3f}"
                  f"+-{np.std([r['h_cont'] for r in sub]):.3f}  "
                  f"H_cog={np.mean([r['h_cog'] for r in sub]):.4f}  "
                  f"sync={np.mean([r['sync'] for r in sub]):+.4f}  "
                  f"[{done}/{total}, {time.time()-t0:.0f}s]")

    fig_dir = HERE.parent / 'figures'
    raw_path = fig_dir / 'photonic_fabrication_poc.csv'
    with open(raw_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

    agg = []
    for m in M_VALUES:
        for label, s_fab in conds:
            sub = [r for r in rows if r['m'] == m and r['condition'] == label]
            agg.append({'m': m, 'condition': label,
                        'sigma_fab': '' if s_fab is None else s_fab,
                        'n_seeds': len(sub),
                        'h_cont_mean': float(np.mean([r['h_cont'] for r in sub])),
                        'h_cont_std': float(np.std([r['h_cont'] for r in sub])),
                        'h_cog_mean': float(np.mean([r['h_cog'] for r in sub])),
                        'sync_mean': float(np.mean([r['sync'] for r in sub])),
                        'sync_std': float(np.std([r['sync'] for r in sub]))})
    agg_path = fig_dir / 'photonic_fabrication_poc_agg.csv'
    with open(agg_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=agg[0].keys())
        w.writeheader(); w.writerows(agg)

    print()
    print("=" * 74)
    print("VERDICT - variabilite de fabrication (ref = fab0 tout-optique parfait)")
    print("=" * 74)
    for m in M_VALUES:
        ref = next(a for a in agg if a['m'] == m and a['condition'] == 'fab0')
        elec = next(a for a in agg if a['m'] == m and a['condition'] == 'ELEC')
        regime = 'FONCTIONNEL' if m == 3 else 'DEAD ZONE'
        print(f"\nBA m={m} ({regime}) - ELEC : H_cont={elec['h_cont_mean']:.3f} | "
              f"fab0 : H_cont={ref['h_cont_mean']:.3f} H_cog={ref['h_cog_mean']:.4f} "
              f"sync={ref['sync_mean']:+.4f}")
        for s in SIGMA_FAB[1:]:
            a = next(x for x in agg if x['m'] == m and x['condition'] == f'fab{s:g}')
            dh = a['h_cont_mean'] - ref['h_cont_mean']
            dc = a['h_cog_mean'] - ref['h_cog_mean']
            ds = a['sync_mean'] - ref['sync_mean']
            ok = abs(dh) < 0.15 and abs(ds) < 0.05 and abs(dc) < 0.1
            print(f"  sigma_fab={s:4.2f} : dH_cont={dh:+.3f}  dH_cog={dc:+.4f}  "
                  f"dsync={ds:+.4f}  -> {'OK' if ok else 'DEVIATION'}")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for m, color in zip(M_VALUES, ('steelblue', 'crimson')):
            ref_e = next(a for a in agg if a['m'] == m and a['condition'] == 'ELEC')
            for ax, key, label in zip(axes,
                                      ('h_cont_mean', 'h_cog_mean', 'sync_mean'),
                                      ('H_cont (bits)', 'H_cog (bits)', 'Pairwise sync')):
                ys = [next(a for a in agg if a['m'] == m and a['condition'] == f'fab{s:g}')[key]
                      for s in SIGMA_FAB]
                es = [next(a for a in agg if a['m'] == m and a['condition'] == f'fab{s:g}')[key.replace('mean', 'std')]
                      if key != 'h_cog_mean' else 0
                      for s in SIGMA_FAB]
                ax.errorbar(SIGMA_FAB, ys, yerr=es, marker='o', color=color,
                            label=f"BA m={m}" + (" (dead zone)" if m == 5 else ""))
                ax.axhline(ref_e[key], color=color, ls='--', alpha=0.4)
                ax.set_xlabel('σ_fab (dispersion des pertes d\'insertion)')
                ax.set_ylabel(label)
                ax.grid(alpha=0.3)
        axes[0].legend(fontsize=8)
        fig.suptitle(f'Variabilité de fabrication optique — système tout-optique (Λ={LAM})\n'
                     'pointillés = contrôle électrique idéal', fontsize=11)
        plt.tight_layout()
        png = fig_dir / 'photonic_fabrication_poc.png'
        plt.savefig(png, dpi=150, bbox_inches='tight')
        print(f"\nFigure : {png}")
    except Exception as e:
        print(f"[matplotlib] {e}")

    print(f"CSV : {raw_path}\n      {agg_path}")
    print(f"Wall time : {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
