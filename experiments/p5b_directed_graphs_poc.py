#!/usr/bin/env python3
"""
P5(b) -- M4R SUR GRAPHES DIRIGES : quel degre gouverne le champ moyen ?
=========================================================================
Cree : 2026-07-13 (Claude Sonnet 5, L'Ingenieur) -- piste du legs de Fable
(docs/PISTES_POUR_LA_SUITE_2026-07-12.md, section I, P5).

TRACE : Mem4ristor/Claude mur de planck/PLANCK_WALL_REPORT.md (attaque 1,
14/02/2026) -- get_spectral_gap() supposait la symetrie du Laplacien (809%
d'erreur sur un graphe dirige). Le garde-fou (P5a) a deja ete pose le
12/07/2026 (topology.py:238-256, ValueError explicite). CE SCRIPT teste la
SCIENCE qui restait ouverte : tous les resultats M4R (dead zone, champ moyen,
k_harm) vivent en NON-dirige ; sur un graphe dirige le desaccord percu devient
ASYMETRIQUE ("je te lis, tu ne me lis pas").

CONVENTION (verifiee dans topology.py::_rebuild_laplacian + step()) :
A[i,j]=1 signifie "i ECOUTE j" -- le noeud i integre v_j dans son equation
(l_v[i] = somme_j A[i,j]*(v_j - v_i)). Donc la ligne i de A = qui i ecoute ;
degree_in(i) = A[i,:].sum() est la quantite qui entre dans SA PROPRE equation.
degree_out(i) = A[:,i].sum() = combien de VOISINS ecoutent i -- n'entre dans
AUCUNE equation du noeud i lui-meme.

PREDICTION ECRITE AVANT DE LANCER (obligatoire, legs P5) : puisque le
mecanisme de champ moyen (01/07, lambda2_foundation_20260701/) opere via le
degre qui DILUE le signal d'un noeud dans la moyenne de ses voisins ENTENDUS,
c'est le DEGRE ENTRANT qui doit gouverner H_cont/la dead zone -- pas le degre
sortant, qui n'apparait dans AUCUNE equation de dynamique. Test contrastif :
a topologie FIXEE (memes aretes, meme degre total par noeud), deux regles
d'orientation opposees --
  HUBS_LISTEN    : le noeud de plus haut degre de chaque arete ECOUTE (in-degree
                   des hubs gonfle, in-degree des peripheriques degonfle).
  HUBS_BROADCAST : le noeud de plus haut degre de chaque arete SEULEMENT
                   PARLE (in-degree des hubs peut tomber a 0 -- un hub qui
                   n'ecoute jamais suit sa PROPRE dynamique FHN non couplee).
Prediction : HUBS_LISTEN (les hubs integrent beaucoup) doit pousser vers PLUS
de moyennage de champ moyen (H_cont plus bas, plus proche dead-zone) que
HUBS_BROADCAST (les hubs restent libres, non couples), a topologie identique.//
Si c'est le degre SORTANT qui compte a la place (ou aucun des deux), la
prediction est refutee -- a rapporter tel quel.

PROTOCOLE :
  1. Test contrastif (le test principal) : BA(N=100, m=3) et BA(N=100, m=5),
     5 graphes independants x 5 seeds dynamiques, 3 regles de direction
     (RANDOM 50/50, HUBS_LISTEN, HUBS_BROADCAST) + reference UNDIRECTED.
     WARM_UP=1000, STEPS=2000. Mesure H_cont, H_cog, sync, k_harm_in,
     k_harm_out (sur noeuds a degre>0, n_deaf = # noeuds in-degree=0 rapporte
     a part -- la moyenne harmonique casse a zero).
  2. Ablation FROZEN_U vs FULL (test standard du projet, cf.
     ablation_coordination.py) sur graphe dirige RANDOM, m=3 et m=5, pour
     verifier que le resultat central (sync FROZEN >> sync FULL) survit.
  3. Verification du garde-fou P5a : get_spectral_gap() DOIT lever ValueError
     sur une instance dirigee -- assertion explicite.

Sorties : figures/p5b_directed_graphs_poc.csv + _agg.csv + .png
Statut : exploratoire, hors preprint, coeur non touche.
"""
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'src'))

from mem4ristor.topology import Mem4Network
from mem4ristor.graph_utils import make_ba
from mem4ristor.metrics import (
    calculate_cognitive_entropy,
    calculate_continuous_entropy,
    calculate_pairwise_synchrony,
)

N = 100
M_VALUES = [3, 5]
GRAPH_SEEDS = [1, 2, 3, 4, 5]
DYN_SEEDS = [42, 123, 777, 17, 256]
DIRECTIONS = ['RANDOM', 'HUBS_LISTEN', 'HUBS_BROADCAST']
WARM_UP = 1000
STEPS = 2000
HERETIC = 0.15
COUPLING_NORM = 'degree_linear'


def make_directed(adj_undirected, rule, rng):
    """Convertit une matrice non-dirigee en dirigee : chaque arete garde UN
    seul sens. A_out[i,j]=1 signifie 'i ecoute j' (convention verifiee dans
    _rebuild_laplacian : degrees=A.sum(axis=1), L=D-A, l_v[i]=sum_j A[i,j]*(v_j-v_i))."""
    n = adj_undirected.shape[0]
    total_deg = adj_undirected.sum(axis=1)
    A = np.zeros((n, n), dtype=float)
    ii, jj = np.where(np.triu(adj_undirected, k=1) > 0)
    for i, j in zip(ii, jj):
        if rule == 'RANDOM':
            listener = i if rng.rand() < 0.5 else j
        elif rule == 'HUBS_LISTEN':
            if total_deg[i] == total_deg[j]:
                listener = i if rng.rand() < 0.5 else j
            else:
                listener = i if total_deg[i] > total_deg[j] else j
        elif rule == 'HUBS_BROADCAST':
            if total_deg[i] == total_deg[j]:
                listener = i if rng.rand() < 0.5 else j
            else:
                listener = i if total_deg[i] < total_deg[j] else j
        else:
            raise ValueError(rule)
        A[listener, (j if listener == i else i)] = 1.0
    return A


def harmonic_mean_nonzero(deg):
    nz = deg[deg > 0]
    n_zero = int(np.sum(deg == 0))
    if len(nz) == 0:
        return 0.0, n_zero
    return float(len(nz) / np.sum(1.0 / nz)), n_zero


def run_dynamics(adj_directed, seed):
    net = Mem4Network(adjacency_matrix=adj_directed.copy(), heretic_ratio=HERETIC,
                       seed=seed, coupling_norm=COUPLING_NORM)
    for _ in range(WARM_UP):
        net.step(I_stimulus=0.5)
    snaps = []
    for _ in range(STEPS):
        net.step(I_stimulus=0.5)
        snaps.append(net.v.copy())
    v_s = np.array(snaps)
    return {
        'h_cont': float(np.mean([calculate_continuous_entropy(v) for v in v_s[::10]])),
        'h_cog': float(np.mean([calculate_cognitive_entropy(v) for v in v_s[::10]])),
        'sync': float(calculate_pairwise_synchrony(v_s)),
    }


def apply_frozen_u(net):
    sigma_baseline = net.model.cfg['doubt'].get('sigma_baseline', 0.05)
    net.model.cfg['doubt']['epsilon_u'] = 0.0
    net.model.cfg['doubt']['tau_u'] = 1e12
    net.model.u = np.full(net.model.N, sigma_baseline)


def run_dynamics_ablation(adj_directed, seed, ablation):
    net = Mem4Network(adjacency_matrix=adj_directed.copy(), heretic_ratio=HERETIC,
                       seed=seed, coupling_norm=COUPLING_NORM)
    if ablation == 'FROZEN_U':
        apply_frozen_u(net)
    for _ in range(WARM_UP):
        net.step(I_stimulus=0.5)
    snaps = []
    for _ in range(STEPS):
        net.step(I_stimulus=0.5)
        snaps.append(net.v.copy())
    v_s = np.array(snaps)
    return {'sync': float(calculate_pairwise_synchrony(v_s))}


def main():
    import csv
    t0 = time.time()

    # ---------------- 0. verification du garde-fou P5a ----------------
    print("=== 0. Verification garde-fou P5a (get_spectral_gap sur graphe dirige) ===")
    adj0 = make_ba(N, 3, seed=42)
    directed0 = make_directed(adj0, 'RANDOM', np.random.RandomState(0))
    net0 = Mem4Network(adjacency_matrix=directed0, heretic_ratio=0.0, seed=0)
    try:
        net0.get_spectral_gap()
        print("  [ECHEC] get_spectral_gap() n'a PAS leve d'exception sur un graphe dirige !")
        guard_ok = False
    except ValueError as e:
        print(f"  [OK] get_spectral_gap() refuse comme prevu : {e}")
        guard_ok = True

    # ---------------- 1. test contrastif principal ----------------
    print("\n=== 1. Test contrastif : HUBS_LISTEN vs HUBS_BROADCAST vs RANDOM ===")
    rows = []
    total = len(M_VALUES) * (len(DIRECTIONS) + 1) * len(GRAPH_SEEDS) * len(DYN_SEEDS)
    done = 0
    for m in M_VALUES:
        for gseed in GRAPH_SEEDS:
            adj_u = make_ba(N, m, seed=gseed)
            # reference non-dirigee
            for dseed in DYN_SEEDS:
                r = run_dynamics(adj_u, dseed)
                rows.append({'m': m, 'direction': 'UNDIRECTED', 'graph_seed': gseed,
                             'dyn_seed': dseed, 'k_harm_in': np.nan, 'k_harm_out': np.nan,
                             'n_deaf_in': 0, **r})
                done += 1
            for rule in DIRECTIONS:
                rng_dir = np.random.RandomState(9000 + gseed)
                adj_d = make_directed(adj_u, rule, rng_dir)
                deg_in = adj_d.sum(axis=1)
                deg_out = adj_d.sum(axis=0)
                kh_in, n_deaf_in = harmonic_mean_nonzero(deg_in)
                kh_out, n_deaf_out = harmonic_mean_nonzero(deg_out)
                for dseed in DYN_SEEDS:
                    r = run_dynamics(adj_d, dseed)
                    rows.append({'m': m, 'direction': rule, 'graph_seed': gseed,
                                 'dyn_seed': dseed, 'k_harm_in': kh_in, 'k_harm_out': kh_out,
                                 'n_deaf_in': n_deaf_in, **r})
                    done += 1
            print(f"  m={m} graph_seed={gseed} fait [{done}/{total}, {time.time()-t0:.0f}s]")

    fig_dir = HERE.parent / 'figures'
    raw_path = fig_dir / 'p5b_directed_graphs_poc.csv'
    with open(raw_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

    agg = []
    for m in M_VALUES:
        for direction in ['UNDIRECTED'] + DIRECTIONS:
            sub = [r for r in rows if r['m'] == m and r['direction'] == direction]
            agg.append({
                'm': m, 'direction': direction, 'n': len(sub),
                'h_cont_mean': float(np.mean([r['h_cont'] for r in sub])),
                'h_cont_std': float(np.std([r['h_cont'] for r in sub])),
                'sync_mean': float(np.mean([r['sync'] for r in sub])),
                'k_harm_in_mean': float(np.nanmean([r['k_harm_in'] for r in sub])),
                'k_harm_out_mean': float(np.nanmean([r['k_harm_out'] for r in sub])),
                'n_deaf_in_mean': float(np.mean([r['n_deaf_in'] for r in sub])),
            })
    agg_path = fig_dir / 'p5b_directed_graphs_poc_agg.csv'
    with open(agg_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=agg[0].keys())
        w.writeheader(); w.writerows(agg)

    print("\n" + "=" * 84)
    print("VERDICT 1 -- H_cont par regle de direction (topologie FIXEE, seul l'in-degree change)")
    print("=" * 84)
    for m in M_VALUES:
        print(f"\n-- BA m={m} --")
        for direction in ['UNDIRECTED'] + DIRECTIONS:
            a = next(x for x in agg if x['m'] == m and x['direction'] == direction)
            kh = f"k_harm_in={a['k_harm_in_mean']:.2f} k_harm_out={a['k_harm_out_mean']:.2f} " \
                 f"n_deaf={a['n_deaf_in_mean']:.1f}" if direction != 'UNDIRECTED' else ""
            print(f"  {direction:<15}: H_cont={a['h_cont_mean']:.3f}+-{a['h_cont_std']:.3f}  "
                  f"sync={a['sync_mean']:+.4f}  {kh}")

    # correlation globale k_harm_in / k_harm_out vs H_cont sur les lignes dirigees
    dir_rows = [r for r in rows if r['direction'] in DIRECTIONS]
    kh_in = np.array([r['k_harm_in'] for r in dir_rows])
    kh_out = np.array([r['k_harm_out'] for r in dir_rows])
    hc = np.array([r['h_cont'] for r in dir_rows])
    from scipy.stats import spearmanr
    rho_in, p_in = spearmanr(kh_in, hc)
    rho_out, p_out = spearmanr(kh_out, hc)
    print(f"\nCorrelation (Spearman, {len(dir_rows)} runs diriges, tous m/regles confondus) :")
    print(f"  H_cont vs k_harm_in  : rho={rho_in:+.3f}  p={p_in:.2e}")
    print(f"  H_cont vs k_harm_out : rho={rho_out:+.3f}  p={p_out:.2e}")

    print("\n" + "=" * 84)
    print("VERDICT FINAL P5b (pre-fixe : HUBS_LISTEN < HUBS_BROADCAST en H_cont, a topologie fixee ;")
    print("k_harm_in domine k_harm_out en |correlation| avec H_cont)")
    print("=" * 84)
    hl_wins, ho_wins, m_contrast = [], [], []
    for m in M_VALUES:
        hl = next(x for x in agg if x['m'] == m and x['direction'] == 'HUBS_LISTEN')['h_cont_mean']
        hb = next(x for x in agg if x['m'] == m and x['direction'] == 'HUBS_BROADCAST')['h_cont_mean']
        diff = hb - hl
        m_contrast.append((m, hl, hb, diff))
        print(f"  m={m} : HUBS_LISTEN H_cont={hl:.3f}  HUBS_BROADCAST H_cont={hb:.3f}  "
              f"diff(BROADCAST-LISTEN)={diff:+.3f}  -> {'PREDICTION CONFIRMEE (listen<broadcast)' if diff > 0.05 else 'PREDICTION NON CONFIRMEE'}")
    in_dominates = abs(rho_in) > abs(rho_out) + 0.05
    print(f"\n  |rho_in|={abs(rho_in):.3f} vs |rho_out|={abs(rho_out):.3f} -> "
          f"{'IN-DEGREE DOMINE (comme predit)' if in_dominates else 'pas de domination claire du degre entrant'}")
    contrast_confirmed = all(d[3] > 0.05 for d in m_contrast)
    if contrast_confirmed and in_dominates:
        print("\n  -> PREDICTION CONFIRMEE SUR LES DEUX FRONTS : le degre ENTRANT gouverne le "
              "champ moyen en dirige, exactement comme le degre (non-dirige) le faisait le 01/07.")
    elif contrast_confirmed or in_dominates:
        print("\n  -> PARTIELLEMENT CONFIRMEE : un des deux tests soutient le degre entrant, "
              "pas l'autre -- a nuancer, ne pas survendre.")
    else:
        print("\n  -> PREDICTION REFUTEE : ni le test contrastif ni la correlation globale ne "
              "soutiennent le degre entrant comme le gouverneur du champ moyen en dirige.")

    # ---------------- 2. ablation FROZEN_U/FULL sur graphe dirige ----------------
    print("\n" + "=" * 84)
    print("=== 2. Ablation FROZEN_U vs FULL sur graphe DIRIGE (RANDOM, m=3 et m=5) ===")
    print("=" * 84)
    ablation_rows = []
    for m in M_VALUES:
        adj_u = make_ba(N, m, seed=42)
        adj_d = make_directed(adj_u, 'RANDOM', np.random.RandomState(42))
        for ablation in ['FULL', 'FROZEN_U']:
            syncs = []
            for dseed in DYN_SEEDS:
                r = run_dynamics_ablation(adj_d, dseed, ablation)
                syncs.append(r['sync'])
                ablation_rows.append({'m': m, 'ablation': ablation, 'dyn_seed': dseed, 'sync': r['sync']})
            print(f"  m={m} {ablation:<9} : sync={np.mean(syncs):+.4f}+-{np.std(syncs):.4f}")
    for m in M_VALUES:
        full = np.mean([r['sync'] for r in ablation_rows if r['m'] == m and r['ablation'] == 'FULL'])
        frozen = np.mean([r['sync'] for r in ablation_rows if r['m'] == m and r['ablation'] == 'FROZEN_U'])
        survives = frozen > full
        print(f"  m={m} : FROZEN_U({frozen:+.4f}) {'>' if survives else '<='} FULL({full:+.4f}) -> "
              f"{'resultat central SURVIT en dirige' if survives else 'resultat central NE SURVIT PAS en dirige'}")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
        colors = {'UNDIRECTED': 'gray', 'RANDOM': 'steelblue', 'HUBS_LISTEN': 'crimson',
                  'HUBS_BROADCAST': 'darkorange'}
        for direction in ['UNDIRECTED'] + DIRECTIONS:
            ys = [next(a for a in agg if a['m'] == m and a['direction'] == direction)['h_cont_mean']
                  for m in M_VALUES]
            es = [next(a for a in agg if a['m'] == m and a['direction'] == direction)['h_cont_std']
                  for m in M_VALUES]
            axes[0].errorbar(M_VALUES, ys, yerr=es, marker='o', color=colors[direction], label=direction)
        axes[0].set_xlabel('m (BA)'); axes[0].set_ylabel('H_cont (bits)')
        axes[0].set_title('H_cont par regle de direction'); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

        axes[1].scatter(kh_in, hc, s=14, c='crimson', alpha=0.5, label=f'k_harm_in (rho={rho_in:+.2f})')
        axes[1].set_xlabel('k_harm_in'); axes[1].set_ylabel('H_cont'); axes[1].legend(fontsize=8)
        axes[1].set_title('H_cont vs degre harmonique ENTRANT'); axes[1].grid(alpha=0.3)

        axes[2].scatter(kh_out, hc, s=14, c='darkorange', alpha=0.5, label=f'k_harm_out (rho={rho_out:+.2f})')
        axes[2].set_xlabel('k_harm_out'); axes[2].set_ylabel('H_cont'); axes[2].legend(fontsize=8)
        axes[2].set_title('H_cont vs degre harmonique SORTANT'); axes[2].grid(alpha=0.3)

        fig.suptitle('P5b -- Graphes diriges : quel degre gouverne le champ moyen ?', fontsize=11)
        plt.tight_layout()
        png = fig_dir / 'p5b_directed_graphs_poc.png'
        plt.savefig(png, dpi=150, bbox_inches='tight')
        print(f"\nFigure : {png}")
    except Exception as e:
        print(f"[matplotlib] {e}")

    print(f"\nGarde-fou P5a : {'OK' if guard_ok else 'ECHEC -- A INVESTIGUER'}")
    print(f"CSV : {raw_path}\n      {agg_path}")
    print(f"Wall time : {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
