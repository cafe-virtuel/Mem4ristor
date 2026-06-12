#!/usr/bin/env python3
"""
POC Couplage Photonique — Vague 2, étape 1 du PHOTONIC_PATHWAY — 2026-06-12
Claude Code (Fable) / Julien Chauvin

SUITE DE : experiments/photonic_transduction_poc.py (stimulus optique : favorable,
Λ_min ≈ 10). Ici on attaque le cas le plus dur : le COUPLAGE optique.

QUESTION : si le signal de couplage inter-nœuds (le terme laplacien, c'est-à-dire
ce que chaque nœud perçoit de ses voisins) transite par la lumière, les régimes
survivent-ils au bruit de grenaille sur ce canal ?

POURQUOI C'EST PLUS DUR QUE LE STIMULUS :
  - Le couplage est le MÉCANISME du modèle (la polarité modulée par u agit dessus).
  - σ_social = |laplacien| pilote la dynamique du doute u : le bruit photonique
    sur le couplage contamine donc AUSSI la perception du désaccord local.
  - Le couplage porte un signe (attractif/répulsif) : on modélise une détection
    différentielle (deux rails optiques ou détection cohérente) — le signe est
    préservé, la magnitude porte le bruit : l_v -> l_v · Poisson(Λc)/Λc par nœud.

CONDITIONS :
  - COUPLING-ONLY : couplage optique (Λc balayé), stimulus électrique parfait.
  - FULL-OPTICAL  : couplage ET stimulus optiques au même budget Λ.
  - Contrôle DETERMINISTIC = protocole standard.

PROTOCOLE : BA m=3 (fonctionnel) + m=5 (dead zone), N=100, I_nominal=0.5,
heretic_ratio=0.15, degree_linear, WARM_UP=1000 + STEPS=3000, 10 seeds canoniques,
Λ ∈ {3, 10, 30, 100, 300, 1000}.

IMPLÉMENTATION : sous-classe de Mem4Network qui réplique step() (topology.py:290)
en insérant la transduction photonique sur l_v APRÈS le scaling degree_linear —
le l_v bruité se propage naturellement à σ_social (réaliste : le nœud perçoit
le signal optique bruité, pas le signal idéal). Le code parent n'est PAS modifié.

Sorties : figures/photonic_coupling_poc.csv / _agg.csv / .png
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

SEEDS = [42, 123, 777, 17, 256, 1337, 99, 314, 2024, 888]
M_VALUES = [3, 5]
LAMBDAS = [3, 10, 30, 100, 300, 1000]
I_NOMINAL = 0.5
N = 100
WARM_UP = 1000
STEPS = 3000
HERETIC = 0.15
COUPLING_NORM = 'degree_linear'


class PhotonicCouplingNet(Mem4Network):
    """Mem4Network dont le canal de couplage est optique (shot noise sur l_v).

    Réplique Mem4Network.step (src/mem4ristor/topology.py:290) à l'identique,
    en ajoutant la transduction photonique sur l_v. Hypothèse hardware :
    détection différentielle -> le signe de l_v est préservé.
    """

    def __init__(self, *args, lam_coupling=None, phot_seed=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.lam_coupling = lam_coupling
        self._rng_phot_coup = np.random.RandomState(phot_seed + 88000)

    def step(self, I_stimulus=0.0, sigma_v_vec=None, sigma_social_override=None):
        self._doubt_driven_rewire()
        if self._weights_dirty:
            self._rebuild_laplacian()
            self._compute_coupling_weights()
        # use_stencil=False pour les graphes explicites (BA) — chemin identique au parent
        l_v = -(self.L @ self.v)
        if self.coupling_norm != 'uniform':
            D = self.model.cfg['coupling']['D']
            if D > 0:
                uniform_D_eff = D / np.sqrt(self.N)
                scale_factors = (self.node_weights * D) / uniform_D_eff
                l_v = l_v * scale_factors   # même associativité que le parent (bit à bit)
            else:
                l_v = np.zeros_like(l_v)
        # --- TRANSDUCTION PHOTONIQUE DU COUPLAGE (seule addition vs parent) ---
        if self.lam_coupling is not None:
            k = self._rng_phot_coup.poisson(self.lam_coupling, self.N)
            l_v = l_v * (k / self.lam_coupling)
        if self.adjacency_matrix is not None:
            self.model._adj_matrix = self.adjacency_matrix
        self.model.step(I_stimulus, l_v, sigma_v_vec=sigma_v_vec,
                        sigma_social_override=sigma_social_override)


def run_one(adj, seed, lam_coupling, lam_stimulus):
    net = PhotonicCouplingNet(adjacency_matrix=adj.copy(), heretic_ratio=HERETIC,
                              seed=seed, coupling_norm=COUPLING_NORM,
                              lam_coupling=lam_coupling, phot_seed=seed)
    rng_stim = np.random.RandomState(seed + 77000)

    def stimulus():
        if lam_stimulus is None:
            return I_NOMINAL
        return I_NOMINAL * rng_stim.poisson(lam_stimulus, N) / lam_stimulus

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
    # (label, lam_coupling, lam_stimulus) — None = canal parfait
    conditions = [('DETERMINISTIC', None, None)]
    conditions += [(f'COUPLING_{l}', l, None) for l in LAMBDAS]
    conditions += [(f'FULLOPT_{l}', l, l) for l in LAMBDAS]

    rows = []
    total = len(M_VALUES) * len(conditions) * len(SEEDS)
    done = 0
    for m in M_VALUES:
        adj = make_ba(N, m, seed=42)
        for label, lc, ls in conditions:
            for seed in SEEDS:
                r = run_one(adj, seed, lc, ls)
                rows.append({'m': m, 'condition': label, 'seed': seed, **r})
                done += 1
            sub = [r for r in rows if r['m'] == m and r['condition'] == label]
            print(f"m={m} {label:>15s} : "
                  f"H_cont={np.mean([r['h_cont'] for r in sub]):.3f}"
                  f"±{np.std([r['h_cont'] for r in sub]):.3f}  "
                  f"H_cog={np.mean([r['h_cog'] for r in sub]):.4f}  "
                  f"sync={np.mean([r['sync'] for r in sub]):+.4f}  "
                  f"[{done}/{total}, {time.time()-t0:.0f}s]")

    fig_dir = HERE.parent / 'figures'
    raw_path = fig_dir / 'photonic_coupling_poc.csv'
    with open(raw_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

    agg = []
    for m in M_VALUES:
        for label, lc, ls in conditions:
            sub = [r for r in rows if r['m'] == m and r['condition'] == label]
            agg.append({
                'm': m, 'condition': label,
                'lam_coupling': '' if lc is None else lc,
                'lam_stimulus': '' if ls is None else ls,
                'n_seeds': len(sub),
                'h_cont_mean': float(np.mean([r['h_cont'] for r in sub])),
                'h_cont_std': float(np.std([r['h_cont'] for r in sub])),
                'h_cog_mean': float(np.mean([r['h_cog'] for r in sub])),
                'sync_mean': float(np.mean([r['sync'] for r in sub])),
                'sync_std': float(np.std([r['sync'] for r in sub])),
            })
    agg_path = fig_dir / 'photonic_coupling_poc_agg.csv'
    with open(agg_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=agg[0].keys())
        w.writeheader(); w.writerows(agg)

    print("\n" + "=" * 78)
    print("VERDICT — couplage optique (référence = DETERMINISTIC)")
    print("=" * 78)
    for m in M_VALUES:
        ref = next(a for a in agg if a['m'] == m and a['condition'] == 'DETERMINISTIC')
        regime = 'FONCTIONNEL' if m == 3 else 'DEAD ZONE'
        print(f"\nBA m={m} ({regime}) — réf : H_cont={ref['h_cont_mean']:.3f} "
              f"H_cog={ref['h_cog_mean']:.4f} sync={ref['sync_mean']:+.4f}")
        for prefix in ('COUPLING', 'FULLOPT'):
            print(f"  [{prefix}]")
            for l in LAMBDAS:
                a = next(x for x in agg if x['m'] == m and x['condition'] == f'{prefix}_{l}')
                d_hcont = a['h_cont_mean'] - ref['h_cont_mean']
                d_sync = a['sync_mean'] - ref['sync_mean']
                ok = abs(d_hcont) < 0.15 and abs(d_sync) < 0.05
                print(f"    Lambda={l:>5d} : dH_cont={d_hcont:+.3f}  "
                      f"dH_cog={a['h_cog_mean']-ref['h_cog_mean']:+.4f}  "
                      f"dsync={d_sync:+.4f}  -> {'OK' if ok else 'DEVIATION'}")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
        for row_i, prefix in enumerate(('COUPLING', 'FULLOPT')):
            for m, color in zip(M_VALUES, ('steelblue', 'crimson')):
                ref = next(a for a in agg if a['m'] == m and a['condition'] == 'DETERMINISTIC')
                for ax, key, label in zip(axes[row_i],
                                          ('h_cont_mean', 'h_cog_mean', 'sync_mean'),
                                          ('H_cont (bits)', 'H_cog (bits)', 'Pairwise sync')):
                    ys = [next(a for a in agg if a['m'] == m and a['condition'] == f'{prefix}_{l}')[key]
                          for l in LAMBDAS]
                    ax.plot(LAMBDAS, ys, marker='o', color=color,
                            label=f"BA m={m}" + (" (dead zone)" if m == 5 else ""))
                    ax.axhline(ref[key], color=color, ls='--', alpha=0.4)
                    ax.set_xscale('log')
                    ax.set_ylabel(label)
                    ax.grid(alpha=0.3)
                    title = 'Couplage optique seul' if prefix == 'COUPLING' else 'Tout-optique (stimulus + couplage)'
                    ax.set_title(f'{title} — {label}', fontsize=9)
                    if row_i == 1:
                        ax.set_xlabel('Λ (photons/nœud/pas)')
        axes[0][0].legend(fontsize=8)
        fig.suptitle('Couplage photonique — étape 1 du PHOTONIC_PATHWAY\n'
                     'pointillés = contrôle déterministe', fontsize=11)
        plt.tight_layout()
        png = fig_dir / 'photonic_coupling_poc.png'
        plt.savefig(png, dpi=150, bbox_inches='tight')
        print(f"\nFigure : {png}")
    except Exception as e:
        print(f"[matplotlib] {e}")

    print(f"CSV : {raw_path}\n      {agg_path}")
    print(f"Wall time : {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
