#!/usr/bin/env python3
"""
P4 -- L'USURE ET LE DRIFT : la 5e imperfection (et le doute qui vieillit bien).
=================================================================================
Cree : 2026-07-13 (Claude Sonnet 5, L'Ingenieur) -- piste du legs de Fable
(docs/PISTES_POUR_LA_SUITE_2026-07-12.md, section I, P4).

TRACE : Mem4ristor/MEM4_WEAR_MODULE.py (07/02/2026, jouet jamais serieux). La
physique reelle : le drift de resistance du GST (R proportionnel a t^nu,
phenomene documente majeur des PCM -- Ielmini & Lacaita 2008, Boniardi &
Ielmini 2011, nu ~ 0.02-0.1 pour le GST amorphe) et l'endurance limitee des
RRAM (cycling) sont l'objection hardware n.1. Le quatuor d'imperfections
photonique (12/06/2026 : bruit, non-linearite, inertie, fabrication) N'INCLUT
PAS le temps.

CAVEAT HONNETE (a ne pas dissimuler) : le coefficient nu est mesure sur la
DERIVE ELECTRIQUE (resistance) du GST, litterature bien etablie. On l'applique
ici comme un PROXY pour la derive de la TRANSMISSION OPTIQUE d'un guide charge
en GST -- ce lien n'a PAS de mesure independante dans ce projet (meme reserve
que docs/hardware/B3_ENERGY_COMPARISON.md). Le sens physique retenu : la
resistance MONTE avec le temps (etat amorphe qui se relaxe vers un desordre
plus stable) -> la transmission/conductance associee DIMINUE d'un facteur
(t/t_ref)^(-nu). Meme loi appliquee au poids de couplage D_eff (proxy RRAM).

QUESTION DIFFERENCIANTE (le coeur de la piste) : l'ADAPTATION du doute
(epsilon_u, qui module a quel point u reagit au desaccord local) compense-t-elle
le drift la ou un u FIGE (ablation FROZEN_U, deja standard du projet -- cf.
ablation_coordination.py) ne le peut pas ? Si FULL survit (reste dans la
tolerance du quatuor) a des heures simulees ou FROZEN_U a deja devie :
"graceful aging" -- l'argument hardware que personne d'autre n'a.

PROTOCOLE : reutilise EXACTEMENT le harness de photonic_fabrication_poc.py
(BA N=100, WARM_UP=1000, STEPS=3000, 10 seeds canoniques, tolerance
|dH_cont|<0.15 / |dH_cog|<0.1 / |dsync|<0.05 vs reference FRAICHE MEME
ablation). Sweep elapsed_hours in {1(frais), 10, 100, 1e3, 1e4, 1e5, 1e6},
nu in {0.05, 0.10} (bornes de la plage litterature), ablation in {FULL, FROZEN_U},
m in {3 (fonctionnel), 5 (dead zone, deja morte -- verifie que le drift n'y
change rien de neuf)}.

CRITERE PRE-FIXE (avant de lancer) : "vieillissement gracieux" confirme si,
pour AU MOINS un nu teste, le premier niveau d'elapsed_hours ou FULL DEVIE
(sort de tolerance) est STRICTEMENT PLUS TARDIF (heures simulees) que celui
de FROZEN_U, a m=3. Si FULL et FROZEN_U devient au meme niveau (ou FULL avant) :
le doute ne compense rien -- resultat negatif honnete a garder tel quel.

Sorties : figures/p4_wear_drift_poc.csv + _agg.csv + .png
Statut : exploratoire, hors preprint, coeur non touche (l'ablation FROZEN_U
est deja un patron standard du projet, aucune modif de dynamics.py).
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

SEEDS = [42, 123, 777, 17, 256, 1337, 99, 314, 2024, 888]   # identiques a photonic_fabrication_poc
M_VALUES = [3, 5]
ELAPSED_HOURS = [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000]
T_REF_HOURS = 1.0            # reference "fraiche" (drift factor = 1.0 a elapsed=1h)
NU_VALUES = [0.05, 0.10]     # plage litterature GST amorphe (Ielmini & Lacaita 2008)
LAM = 10
I_NOMINAL = 0.5
I_REF = 1.0
N = 100
WARM_UP = 1000
STEPS = 3000
HERETIC = 0.15
COUPLING_NORM = 'degree_linear'
# Tolerances (identiques a photonic_fabrication_poc.py)
TOL_H_CONT, TOL_H_COG, TOL_SYNC = 0.15, 0.1, 0.05


def drift_factor(elapsed_hours, nu):
    """Transmission/conductance relative a l'etat frais (t_ref=1h) :
    f(t) = (t/t_ref)^(-nu), nu>0 -> la transmission DECROIT avec le temps
    (proxy de la resistance PCM qui CROIT en R ~ t^nu)."""
    return (elapsed_hours / T_REF_HOURS) ** (-nu)


class DriftedPhotonicNet(Mem4Network):
    """Meme patron que FabricatedPhotonicNet (photonic_fabrication_poc.py) :
    tout-optique avec pertes d'insertion STATIQUES par noeud, ICI multipliees
    par un facteur de derive temporelle UNIFORME (meme drift pour tous les
    noeuds -- effet materiau, pas variabilite de fabrication ; sigma_fab=0
    ici pour isoler l'effet du TEMPS seul de celui de la Vague 2 du 12/06)."""

    def __init__(self, *args, lam_coupling=None, drift=1.0, phot_seed=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.lam_coupling = lam_coupling
        self.drift = drift
        self._rng_phot_coup = np.random.RandomState(phot_seed + 88000)
        # RRAM proxy : le poids de couplage D_eff derive de la meme loi.
        self.model.D_eff = self.model.D_eff * drift

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
        l_v = l_v * self.drift                    # GST transmission (etage u) : derive uniforme
        if self.lam_coupling is not None:
            k = self._rng_phot_coup.poisson(self.lam_coupling, self.N)
            l_v = l_v * (k / self.lam_coupling)    # shot noise (Lambda=10, tout-optique)
        if self.adjacency_matrix is not None:
            self.model._adj_matrix = self.adjacency_matrix
        self.model.step(I_stimulus, l_v, sigma_v_vec=sigma_v_vec,
                        sigma_social_override=sigma_social_override)


def apply_ablation(net, ablation):
    if ablation == 'FULL':
        return
    if ablation == 'FROZEN_U':
        sigma_baseline = net.model.cfg['doubt'].get('sigma_baseline', 0.05)
        net.model.cfg['doubt']['epsilon_u'] = 0.0
        net.model.cfg['doubt']['tau_u'] = 1e12
        net.model.u = np.full(net.model.N, sigma_baseline)
    else:
        raise ValueError(f"Unknown ablation: {ablation!r}")


def run_one(adj, seed, elapsed_hours, nu, ablation):
    drift = drift_factor(elapsed_hours, nu)
    net = DriftedPhotonicNet(
        adjacency_matrix=adj.copy(), heretic_ratio=HERETIC, seed=seed,
        coupling_norm=COUPLING_NORM, lam_coupling=LAM, drift=drift, phot_seed=seed)
    apply_ablation(net, ablation)
    rng_stim = np.random.RandomState(seed + 77000)

    t_stim = drift  # canal stimulus derive lui aussi (meme guide GST)

    def stimulus():
        lam_vec = LAM * t_stim * I_NOMINAL / I_REF
        k = rng_stim.poisson(max(lam_vec, 1e-9))
        return I_REF * k / LAM

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
    rows = []
    ablations = ['FULL', 'FROZEN_U']
    total = len(M_VALUES) * len(NU_VALUES) * len(ablations) * len(ELAPSED_HOURS) * len(SEEDS)
    done = 0
    for m in M_VALUES:
        adj = make_ba(N, m, seed=42)
        for nu in NU_VALUES:
            for ablation in ablations:
                for eh in ELAPSED_HOURS:
                    for seed in SEEDS:
                        r = run_one(adj, seed, eh, nu, ablation)
                        rows.append({'m': m, 'nu': nu, 'ablation': ablation,
                                     'elapsed_hours': eh, 'seed': seed, **r})
                        done += 1
                    sub = [r for r in rows if r['m'] == m and r['nu'] == nu
                           and r['ablation'] == ablation and r['elapsed_hours'] == eh]
                    print(f"m={m} nu={nu:.2f} {ablation:<9} h={eh:>9,} : "
                          f"H_cont={np.mean([r['h_cont'] for r in sub]):.3f}"
                          f"+-{np.std([r['h_cont'] for r in sub]):.3f}  "
                          f"sync={np.mean([r['sync'] for r in sub]):+.4f}  "
                          f"[{done}/{total}, {time.time()-t0:.0f}s]")

    fig_dir = HERE.parent / 'figures'
    raw_path = fig_dir / 'p4_wear_drift_poc.csv'
    with open(raw_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

    agg = []
    for m in M_VALUES:
        for nu in NU_VALUES:
            for ablation in ablations:
                for eh in ELAPSED_HOURS:
                    sub = [r for r in rows if r['m'] == m and r['nu'] == nu
                           and r['ablation'] == ablation and r['elapsed_hours'] == eh]
                    agg.append({'m': m, 'nu': nu, 'ablation': ablation, 'elapsed_hours': eh,
                                'n_seeds': len(sub),
                                'h_cont_mean': float(np.mean([r['h_cont'] for r in sub])),
                                'h_cont_std': float(np.std([r['h_cont'] for r in sub])),
                                'h_cog_mean': float(np.mean([r['h_cog'] for r in sub])),
                                'sync_mean': float(np.mean([r['sync'] for r in sub])),
                                'sync_std': float(np.std([r['sync'] for r in sub]))})
    agg_path = fig_dir / 'p4_wear_drift_poc_agg.csv'
    with open(agg_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=agg[0].keys())
        w.writeheader(); w.writerows(agg)

    # ---------------- verdict pre-fixe ----------------
    print()
    print("=" * 84)
    print("VERDICT P4 -- premier niveau (heures) ou chaque ablation DEVIE de sa reference fraiche")
    print("=" * 84)
    first_deviation = {}   # (m, nu, ablation) -> elapsed_hours du premier ecart, ou None si jamais
    for m in M_VALUES:
        regime = 'FONCTIONNEL' if m == 3 else 'DEAD ZONE'
        for nu in NU_VALUES:
            print(f"\n-- BA m={m} ({regime}), nu={nu:.2f} --")
            for ablation in ablations:
                ref = next(a for a in agg if a['m'] == m and a['nu'] == nu
                           and a['ablation'] == ablation and a['elapsed_hours'] == 1)
                dev_h = None
                for eh in ELAPSED_HOURS[1:]:
                    a = next(x for x in agg if x['m'] == m and x['nu'] == nu
                             and x['ablation'] == ablation and x['elapsed_hours'] == eh)
                    dh = a['h_cont_mean'] - ref['h_cont_mean']
                    dc = a['h_cog_mean'] - ref['h_cog_mean']
                    ds = a['sync_mean'] - ref['sync_mean']
                    ok = abs(dh) < TOL_H_CONT and abs(ds) < TOL_SYNC and abs(dc) < TOL_H_COG
                    status = 'OK' if ok else 'DEVIATION'
                    print(f"  {ablation:<9} h={eh:>9,} : dH_cont={dh:+.3f} dH_cog={dc:+.4f} "
                          f"dsync={ds:+.4f} -> {status}")
                    if not ok and dev_h is None:
                        dev_h = eh
                first_deviation[(m, nu, ablation)] = dev_h
                print(f"  => {ablation} devie a partir de h={dev_h:,}" if dev_h
                      else f"  => {ablation} NE DEVIE JAMAIS dans la plage testee")

    print("\n" + "=" * 84)
    print("VERDICT FINAL P4 (pre-fixe : FULL devie plus tard que FROZEN_U, a m=3, pour au moins un nu)")
    print("=" * 84)
    any_graceful = False
    for nu in NU_VALUES:
        dev_full = first_deviation[(3, nu, 'FULL')]
        dev_frozen = first_deviation[(3, nu, 'FROZEN_U')]
        df = dev_full if dev_full is not None else float('inf')
        dz = dev_frozen if dev_frozen is not None else float('inf')
        graceful = df > dz
        any_graceful = any_graceful or graceful
        print(f"  m=3, nu={nu:.2f} : FULL devie a {dev_full!r}h, FROZEN_U a {dev_frozen!r}h "
              f"-> {'GRACIEUX (FULL survit plus longtemps)' if graceful else 'PAS DE COMPENSATION' if df <= dz else 'egal'}")
    if any_graceful:
        print("\n  -> VIEILLISSEMENT GRACIEUX CONFIRME pour au moins un nu testé : "
              "l'adaptation du doute retarde la deviation par rapport a u fige.")
    else:
        print("\n  -> PAS DE VIEILLISSEMENT GRACIEUX : le doute adaptatif ne retarde pas la "
              "deviation de tolerance par rapport a FROZEN_U dans cette plage de nu. "
              "Resultat negatif honnete.")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(len(NU_VALUES), 3, figsize=(15, 4.5 * len(NU_VALUES)))
        if len(NU_VALUES) == 1:
            axes = axes[None, :]
        colors = {'FULL': 'crimson', 'FROZEN_U': 'steelblue'}
        for i, nu in enumerate(NU_VALUES):
            for ax, key, label in zip(axes[i],
                                      ('h_cont_mean', 'h_cog_mean', 'sync_mean'),
                                      ('H_cont (bits)', 'H_cog (bits)', 'Pairwise sync')):
                for ablation in ablations:
                    ys = [next(a for a in agg if a['m'] == 3 and a['nu'] == nu
                               and a['ablation'] == ablation and a['elapsed_hours'] == eh)[key]
                          for eh in ELAPSED_HOURS]
                    ax.plot(ELAPSED_HOURS, ys, marker='o', color=colors[ablation], label=ablation)
                ax.set_xscale('log')
                ax.set_xlabel('heures simulees ecoulees')
                ax.set_ylabel(label)
                ax.set_title(f'm=3 (fonctionnel), nu={nu:.2f}')
                ax.grid(alpha=0.3)
                if i == 0:
                    ax.legend(fontsize=8)
        fig.suptitle(f'P4 -- Usure et drift (BA m=3, GST+RRAM proxy, tolerance quatuor 12/06)',
                     fontsize=11)
        plt.tight_layout()
        png = fig_dir / 'p4_wear_drift_poc.png'
        plt.savefig(png, dpi=150, bbox_inches='tight')
        print(f"\nFigure : {png}")
    except Exception as e:
        print(f"[matplotlib] {e}")

    print(f"CSV : {raw_path}\n      {agg_path}")
    print(f"Wall time : {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
