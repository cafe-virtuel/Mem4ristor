#!/usr/bin/env python3
"""
MANDAT NON-LIVRABLE : fonder theoriquement lambda2_crit ~ 2.31
Etape 5 : LE POURQUOI (partie cinematique).

Mecanisme propose : la mort cognitive est un effet de CHAMP MOYEN par
echantillonnage. Le couplage (normalise degree_linear) tire chaque noeud i vers
la moyenne de ses voisins :  cible_i = (A v)_i / deg_i = <v des voisins de i>.

  cible_i est une moyenne de deg_i echantillons de v. Si v a une variance V,
  alors Var(cible_i) ~ V / deg_i. La DISPERSION des cibles A TRAVERS LES NOEUDS
  vaut donc  Var_i(cible_i) ~ V * <1/deg>  (moyenne harmonique inverse du degre).

  - Beaucoup de voisins (deg grand) -> <1/deg> petit -> toutes les cibles ~
    identiques (= moyenne globale) -> attraction coherente vers UN point ->
    consensus / mort cognitive.
  - Peu de voisins -> <1/deg> grand -> cibles dispersees, propres a chaque noeud
    -> attractions incoherentes -> la diversite survit.

Ce mecanisme ne depend QUE du nombre de voisins (echantillonnage), PAS de leur
arrangement (lambda2). Et il est domine par les noeuds de BAS degre : dans un
scale-free (BA), la nuee de noeuds peripheriques (deg = m) garde <1/deg> eleve
meme si quelques hubs gonflent le degre MOYEN -> explique pourquoi BA survit a
un <k> plus eleve qu'un graphe regulier (l'anomalie de l'hetetogeneite).

TEST (cinematique, aucune simulation) : sur un etat de test v i.i.d. fixe,
mesurer Var_i(cible_i) et verifier :
  (1) Var_i(cible) ~ V * <1/deg>  (loi 1/k)  -> correlation quasi parfaite ;
  (2) Var_i(cible) NON correle a lambda2 ;
  (3) anneau et random-regular de MEME degre -> MEME Var_i(cible) (meme si
      leurs lambda2 different de 100x).
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

import numpy as np
import networkx as nx
from scipy.linalg import eigh
from scipy.stats import pearsonr

HERE = Path(__file__).resolve().parent
N = 200
V_DRAWS = 40          # nombre de tirages de v_test moyennes
GRAPH_SEEDS = [1, 2, 3, 4]


def fiedler(adj):
    d = adj.sum(1)
    return float(np.sort(eigh(np.diag(d) - adj, eigvals_only=True))[1])


def target_dispersion(adj, v_draws):
    """Var a travers les noeuds de la cible de couplage <v_vois>, moyennee sur v."""
    deg = adj.sum(1)
    deg_safe = np.where(deg < 1, 1, deg)
    vals = []
    for v in v_draws:
        targets = (adj @ v) / deg_safe
        vals.append(np.var(targets))
    return float(np.mean(vals))


def graphs():
    specs = []
    for m in [1, 2, 3, 4, 5, 6, 8, 10]:
        specs.append((f"BA m={m}", "BA", lambda s, m=m: nx.barabasi_albert_graph(N, m, seed=s)))
    for p in [0.03, 0.05, 0.08, 0.12]:
        specs.append((f"ER p={p}", "ER", lambda s, p=p: nx.erdos_renyi_graph(N, p, seed=s)))
    for k in [4, 6, 8, 10]:
        specs.append((f"rr k={k}", "rr", lambda s, k=k: nx.random_regular_graph(k, N, seed=s)))
    for k in [4, 6, 8, 10]:
        specs.append((f"ring k={k}", "ring", lambda s, k=k: nx.watts_strogatz_graph(N, k, 0.0, seed=s)))
    return specs


def main():
    rng = np.random.default_rng(0)
    v_draws = [rng.normal(0, 1, N) for _ in range(V_DRAWS)]
    V = float(np.mean([np.var(v) for v in v_draws]))

    rows = []
    print("=" * 96)
    print(f"POURQUOI (cinematique) : dispersion des cibles de couplage  (N={N}, V(v)={V:.3f})")
    print("=" * 96)
    print(f"{'topo':<12}{'fam':>5}{'k_mean':>8}{'k_med':>7}{'<1/deg>':>9}"
          f"{'lambda2':>9}{'Var_cible':>11}{'V*<1/deg>':>11}")
    print("-" * 96)
    for name, fam, gen in graphs():
        km, kmed, inv, l2, vc = [], [], [], [], []
        for s in GRAPH_SEEDS:
            G = gen(s)
            adj = nx.to_numpy_array(G, dtype=float)
            deg = adj.sum(1)
            km.append(deg.mean()); kmed.append(np.median(deg))
            inv.append(np.mean(1.0 / np.where(deg < 1, 1, deg)))
            l2.append(fiedler(adj)); vc.append(target_dispersion(adj, v_draws))
        km, kmed, inv, l2, vc = map(np.mean, (km, kmed, inv, l2, vc))
        rows.append({"name": name, "fam": fam, "k_mean": km, "k_med": kmed,
                     "inv": inv, "l2": l2, "vc": vc})
        print(f"{name:<12}{fam:>5}{km:>8.2f}{kmed:>7.0f}{inv:>9.4f}"
              f"{l2:>9.3f}{vc:>11.4f}{V*inv:>11.4f}")

    # Correlations
    inv_a = np.array([r["inv"] for r in rows])
    l2_a = np.array([r["l2"] for r in rows])
    vc_a = np.array([r["vc"] for r in rows])
    r_inv, _ = pearsonr(inv_a, vc_a)
    r_l2, _ = pearsonr(l2_a, vc_a)
    # regression Var_cible = a*<1/deg>
    a = float(np.sum(inv_a * vc_a) / np.sum(inv_a * inv_a))

    print("-" * 96)
    print(f"Var_cible vs <1/deg> : r = {r_inv:+.4f}   (prediction champ-moyen : ~ +1.00)")
    print(f"Var_cible vs lambda2 : r = {r_l2:+.4f}   (prediction : ~0, aucun role)")
    print(f"Pente Var_cible / <1/deg> = {a:.3f}   (theorie : V = {V:.3f})")

    # Test anneau vs rr a meme degre
    print("\nAnneau vs random-regular a MEME degre (lambda2 tres differents) :")
    print(f"{'k':>3}{'lambda2_ring':>14}{'lambda2_rr':>12}{'Vc_ring':>10}{'Vc_rr':>10}")
    for k in [4, 6, 8, 10]:
        rr = next(r for r in rows if r["name"] == f"rr k={k}")
        rg = next(r for r in rows if r["name"] == f"ring k={k}")
        print(f"{k:>3}{rg['l2']:>14.3f}{rr['l2']:>12.3f}{rg['vc']:>10.4f}{rr['vc']:>10.4f}")
    print("  -> lambda2 diffile de ~100x, mais Var_cible ~ identique (fixee par k).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
