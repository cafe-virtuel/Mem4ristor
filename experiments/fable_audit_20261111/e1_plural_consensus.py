#!/usr/bin/env python3
"""
AUDIT FABLE 2026-11-11 -- E1 : consensus local, pluralite globale.
Le doute sur les NOEUDS (coeur actuel) contre le doute sur les LIENS (le dipole de la genese).

POURQUOI. La genese (19/08/2025) decrit un MEMRISTOR a 4 etats : un dipole, une synapse --
un composant qui se trouve ENTRE deux unites et dont l'etat depend de la tension a ses
bornes. Le coeur a place la variable de doute u SUR les unites, pilotee par la somme nette
|somme_j (v_j - v_i)|. Deux consequences structurelles, lisibles dans les equations :
(a) un noeud pris entre deux camps opposes voit une somme nulle, donc aucun doute ;
(b) un noeud ne peut pas garder ses liens d'accord et couper ses liens de desaccord.
L'intuition d'origine -- "une diversite de perspectives qui tient ensemble sans s'ecraser
en consensus" -- demande exactement (b).

TACHE (figee avant toute mesure). K=3 groupes ; chaque noeud recoit un courant continu
I_i = mu_groupe + bruit (mu = 0.3 / 0.7 / 1.1, sigma = 0.25 : les groupes se recouvrent).
Le reseau doit rendre a chaque noeud la valeur de SON groupe : moyenner avec les siens,
pas avec les autres. Deux graphes : grille 12x12 en 3 bandes (la structure n'est QUE dans
l'entree) ; SBM 3x40 (la structure est aussi dans le graphe).

LECTURES ("langages"), sur les MEMES runs :
  inst  : les 5 classes cognitives de l'instantane final de v (la lecture habituelle du projet)
  mean  : niveau continu <v_i> moyenne dans le temps, k-means K=3   <-- PRIMAIRE
  edge  : composantes connexes des liens de confiance (u_ij < 0.5)   (bras 'edge*' seulement)
Metrique : ARI contre la verite (1 = groupes retrouves, 0 = hasard) ; + RMSE de <v_i> a la
valeur ideale du groupe, rapportee a celle du bras sans couplage (< 1 = le reseau aide).

BRAS : none | fixed | node (= coeur, gate 0.0e+00) | edge | edge_noflip | oracle, chacun
balaye sur c = D_eff*<k> dans {0.1, 0.3, 1, 3, 10} ; et pour chaque bras 'edge*' deux
temoins a doute FIGE : 'frozen' (le motif appris, conditions initiales neuves) et
'shuffled' (memes valeurs, permutees entre liens -- meme NIVEAU, motif detruit).

CRITERES, ECRITS AVANT LA MESURE. c choisi par bras sur les graines 0-9, verdict sur 10-29.
  H1  ARI_mean(edge) - ARI_mean(fixed) > 0, IC95 bootstrap apparie excluant 0, sur les 2 graphes
  H2  ARI_mean(edge) - ARI_mean(node)  > 0, idem
  H3  ARI_mean(frozen) - ARI_mean(shuffled) > 0 : le MOTIF porte l'effet, pas le niveau
  (edge_noflip : memes tests, en secondaire.)   Tout autre resultat = echec, rapporte tel quel.

Usage : uv run --frozen python experiments/fable_audit_20261111/e1_plural_consensus.py
"""
from __future__ import annotations
import csv, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import networkx as nx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fable_sim import (simulate, kmeans1d, ari, components, cognitive_bins)  # noqa: E402

K = 3
MU = np.array([0.3, 0.7, 1.1])
SIGMA_IN = 0.25
STEPS, REC_FROM, REC_EVERY = 20000, 12000, 4
C_GRID = [0.1, 0.3, 1.0, 3.0, 10.0]
SEEDS = list(range(30))
TUNE, TEST = list(range(10)), list(range(10, 30))
GRAPHS = ["grille", "sbm"]


def build(graph, seed):
    if graph == "grille":
        G = nx.grid_2d_graph(12, 12)
        nodes = sorted(G.nodes())
        adj = nx.to_numpy_array(G, nodelist=nodes, dtype=float)
        groups = np.array([c // 4 for (_, c) in nodes])
    else:
        G = nx.stochastic_block_model([40, 40, 40],
                                      [[0.25, 0.04, 0.04], [0.04, 0.25, 0.04], [0.04, 0.04, 0.25]],
                                      seed=1000 + seed)
        adj = nx.to_numpy_array(G, nodelist=range(120), dtype=float)
        groups = np.repeat(np.arange(3), 40)
    return adj, groups


def read(out, groups, ideal, raw_rmse=None):
    m = out["rec"].mean(axis=0)
    N = len(groups)
    if not np.all(np.isfinite(m)):
        return dict(ari_mean=0.0, ari_inst=0.0, rmse=np.nan, ari_edge=np.nan, cut_in=np.nan,
                    cut_out=np.nan, u_sat=np.nan)
    r = dict(ari_mean=ari(kmeans1d(m, K), groups),
             ari_inst=ari(cognitive_bins(out["v"]), groups),
             rmse=float(np.sqrt(np.mean((m - ideal[groups]) ** 2))),
             ari_edge=np.nan, cut_in=np.nan, cut_out=np.nan, u_sat=np.nan)
    if out["u_e"] is not None:
        iu, ju, ue = out["iu"], out["ju"], out["u_e"]
        same = groups[iu] == groups[ju]
        r["ari_edge"] = ari(components(N, iu, ju, ue < 0.5), groups)
        r["cut_in"] = float((ue[same] >= 0.5).mean())
        r["cut_out"] = float((ue[~same] >= 0.5).mean()) if np.any(~same) else np.nan
    else:
        r["u_sat"] = float((out["u_n"] > 0.8).mean())
    return r


def job(args):
    graph, seed, ideal = args
    adj, groups = build(graph, seed)
    N = adj.shape[0]
    kmean = adj.sum() / N
    I = MU[groups] + np.random.RandomState(5000 + seed).normal(0, SIGMA_IN, N)
    rows = []
    kw = dict(steps=STEPS, rec_from=REC_FROM, rec_every=REC_EVERY)

    def add(arm, c, out):
        r = read(out, groups, ideal)
        r.update(graph=graph, seed=seed, arm=arm, c=c)
        rows.append(r)

    add("none", 0.0, simulate(adj, I, 0.0, "none", seed, **kw))
    for c in C_GRID:
        D = c / kmean * np.sqrt(N)          # D_eff = D/sqrt(N) = c/<k>
        add("fixed", c, simulate(adj, I, D, "fixed", seed, **kw))
        add("node", c, simulate(adj, I, D, "node", seed, **kw))
        add("oracle", c, simulate(adj, I, D, "oracle", seed, groups=groups, **kw))
        for mode, nf in (("edge", False), ("edge_noflip", True)):
            out = simulate(adj, I, D, mode, seed, **kw)
            add(mode, c, out)
            ue = out["u_e"]
            add(mode + ":frozen", c, simulate(adj, I, D, "edge_frozen", 10000 + seed,
                                              u_edge=ue, noflip=nf, **kw))
            perm = np.random.RandomState(7000 + seed).permutation(len(ue))
            add(mode + ":shuffled", c, simulate(adj, I, D, "edge_frozen", 10000 + seed,
                                                u_edge=ue[perm], noflip=nf, **kw))
    return rows


def boot_ci(d, n=10000, seed=0):
    d = np.asarray(d, dtype=float)
    rng = np.random.RandomState(seed)
    bs = d[rng.randint(0, len(d), size=(n, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    t0 = time.time()
    ideal = np.array([simulate(np.zeros((4, 4)), mu, 0.0, "none", 99, steps=STEPS,
                               rec_from=REC_FROM, rec_every=REC_EVERY)["rec"].mean() for mu in MU])
    print(f"valeur ideale <v> par groupe (noeud isole, entree propre) : {np.round(ideal, 3)}")
    jobs = [(g, s, ideal) for g in GRAPHS for s in SEEDS]
    with Pool(12) as pool:
        res = pool.map(job, jobs, chunksize=1)
    rows = [r for rr in res for r in rr]
    keys = ["graph", "seed", "arm", "c", "ari_mean", "ari_inst", "rmse", "ari_edge", "cut_in", "cut_out", "u_sat"]
    with open(HERE / "e1_plural_consensus.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)

    def val(graph, arm, c, seeds, key):
        return np.array([r[key] for s in seeds for r in rows
                         if r["graph"] == graph and r["arm"] == arm and r["c"] == c and r["seed"] == s])

    arms = ["fixed", "node", "edge", "edge:frozen", "edge:shuffled",
            "edge_noflip", "edge_noflip:frozen", "edge_noflip:shuffled", "oracle"]
    for graph in GRAPHS:
        print("\n" + "=" * 118)
        print(f"GRAPHE : {graph}   (ARI lecture 'mean', moyenne sur les 30 graines, par c = D_eff*<k>)")
        print("=" * 118)
        raw = val(graph, "none", 0.0, SEEDS, "ari_mean").mean()
        raw_rmse = val(graph, "none", 0.0, SEEDS, "rmse")
        print(f"sans couplage : ARI_mean = {raw:.3f} ; ARI_inst = {val(graph, 'none', 0.0, SEEDS, 'ari_inst').mean():.3f}"
              f" ; RMSE = {raw_rmse.mean():.3f}")
        print(f"{'bras':>22} | " + " ".join(f"c={c:<5}" for c in C_GRID) + " | c* | ARI_mean test [IC95]"
              "      ARI_inst  RMSE/brut  ARI_edge  cut_in cut_out u_sat")
        best = {}
        for arm in arms:
            prof = [val(graph, arm, c, SEEDS, "ari_mean").mean() for c in C_GRID]
            cstar = C_GRID[int(np.argmax([val(graph, arm, c, TUNE, "ari_mean").mean() for c in C_GRID]))]
            best[arm] = cstar
            a = val(graph, arm, cstar, TEST, "ari_mean")
            m, lo, hi = boot_ci(a)
            rm = np.nanmean(val(graph, arm, cstar, TEST, "rmse") / val(graph, "none", 0.0, TEST, "rmse"))
            f = lambda key: np.nanmean(val(graph, arm, cstar, TEST, key)) if np.any(np.isfinite(val(graph, arm, cstar, TEST, key))) else float("nan")
            print(f"{arm:>22} | " + " ".join(f"{p:7.3f}" for p in prof) + f" | {cstar:<4}| {m:.3f} [{lo:.3f},{hi:.3f}]"
                  f"   {val(graph, arm, cstar, TEST, 'ari_inst').mean():8.3f} {rm:9.3f} {f('ari_edge'):9.3f} "
                  f"{f('cut_in'):7.2f} {f('cut_out'):7.2f} {f('u_sat'):5.2f}")
        print("-" * 118)
        print("Differences appariees sur les graines de test (chaque bras a SON c*), ARI lecture 'mean' :")
        pairs = [("H1  edge - fixed", "edge", "fixed"), ("H2  edge - node", "edge", "node"),
                 ("H3  edge:frozen - edge:shuffled", "edge:frozen", "edge:shuffled"),
                 ("H1' edge_noflip - fixed", "edge_noflip", "fixed"), ("H2' edge_noflip - node", "edge_noflip", "node"),
                 ("H3' noflip:frozen - noflip:shuffled", "edge_noflip:frozen", "edge_noflip:shuffled"),
                 ("     node - fixed", "node", "fixed"), ("     oracle - fixed", "oracle", "fixed")]
        for name, a, b in pairs:
            d = val(graph, a, best[a], TEST, "ari_mean") - val(graph, b, best[b], TEST, "ari_mean")
            m, lo, hi = boot_ci(d)
            verdict = "OUI" if lo > 0 else ("NON (inverse)" if hi < 0 else "non concluant")
            print(f"  {name:<38} {m:+.3f}  IC95 [{lo:+.3f}, {hi:+.3f}]   -> {verdict}")
    print(f"\n{len(rows)} runs, {time.time() - t0:.0f} s. CSV : e1_plural_consensus.csv")


if __name__ == "__main__":
    main()
