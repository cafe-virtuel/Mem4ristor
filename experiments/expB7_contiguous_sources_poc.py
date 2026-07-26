#!/usr/bin/env python3
"""
EXPERIENCE B7 (volet 2 -- LE TEST DECISIF) : la topologie sert-elle QUAND il y a
une structure spatiale a exploiter ?

CE QUE LE VOLET 1 A ETABLI (commit cb3dc1c). Sur la tache trompeuse actuelle, le
desaccord local mean(|L v|) et la dispersion globale std(v) sont UNE SEULE
lecture (r = 0.981 sur les signaux normalises) parce que la tache n'a AUCUNE
structure spatiale : Moran's I du stimulus = -0.010, et le reseau n'en fabrique
pas (Moran's I de v <= 0.022). Autrement dit la question "la topologie
sert-elle ?" n'a jamais ete posee dans des conditions ou elle POURRAIT servir.

CE VOLET LA POSE. Meme tache, memes effectifs (26 leurres / 14 verites), memes
amplitudes, meme signe, meme graphe (tore 10x10) : seule la POSITION des
capteurs change. Les leurres forment un bloc contigu, les verites un autre
(Moran's I du stimulus +0.677, mesure au volet 1).

LE CONTROLE EST STRUCTUREL, PAS UN MONDE DE PLUS. Permuter aleatoirement les
etiquettes de noeuds sur le monde CONTIGU detruit la contiguite sans changer
aucune statistique globale -- et redonne exactement un placement aleatoire des
memes effectifs, c'est-a-dire le monde RANDOM. Le monde RANDOM EST donc le null
de permutation du monde CLUSTERED ; les comparer est deja le test controle.

PREDICTION, POSEE AVANT DE MESURER (et deja inscrite dans le commit cb3dc1c) :
    (LAPLACIAN - STD) doit etre <= 0 dans RANDOM (etabli : -0.07 a -0.10, IC
    touchant 0) et STRICTEMENT POSITIF dans CLUSTERED.
Si elle echoue -- si le desaccord local n'aide pas davantage la ou la topologie
porte enfin de l'information -- alors la topologie n'est pas necessaire au signal
d'arret, point final : c'est un 5e retrecissement de la revendication, a ecrire
tel quel.

GATE OBLIGATOIRE, EVALUE AVANT DE REGARDER LES SIGNAUX. La variante contigue
doit rester TROMPEUSE, sinon les deux mondes ne sont pas comparables et le test
ne vaut rien :
    (g1) la verite finit par gagner en budget illimite (acc_final eleve)
    (g2) le basculement faux->juste est TARDIF (flip_time > 100)
    (g3) la justesse a tout instant dans la fenetre reste basse (~0.40 comme
         dans RANDOM) -- sinon la tache est devenue facile

SORTIES : figures/expB7_contiguous_sources_poc.csv + .png
Cree : 2026-07-26 (Claude Opus 5, L'Ingenieur) -- suite de expB7 volet 1.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "scratch"))
import deceptive_task_poc as dp  # noqa: E402
import expB7_spatial_structure_diagnostic_poc as b7d  # noqa: E402
from mem4ristor.graph_utils import make_lattice_adj  # noqa: E402

dp.MAX_BUDGET = 2000
MAXB = dp.MAX_BUDGET
N = dp.N
SIDE = dp.SIDE
WARMUP = dp.WARMUP

SEEDS_TRAIN = list(range(20))
SEEDS_TEST = list(range(20, 60))       # 40 graines reservees
T_PULSE = 700
# K = nombre de PATCHS par role. K=1 : un bloc unique (structure maximale) ;
# K grand : dispersion, on retombe sur la tache d'origine. Le premier run de ce
# script (K=1 seul) a CASSE la tache -- acc_final 0.50 contre 0.78 en disperse :
# concentrees, les 14 sources de verite saturent leur region et ne remontent plus
# dans le readout global mean(v). D'ou le balayage.
K_GRID = [1, 2, 3, 5]
WORLDS = ["RANDOM"] + [f"CLUSTERED_K{k}" for k in K_GRID]
STOP_SIGNALS = ["LAPLACIAN", "STD", "TEMPORAL"]
DROP_GRID = [0.15, 0.30, 0.45, 0.60]
PROBE_TIMES = [100, 350, 690]
GATE_ACC_FINAL_MIN = 0.70   # declare AVANT mesure : seuil de solvabilite
# REGLE DE METHODE, POSEE AVANT DE MESURER : le choix de K se fait sur le SEUL
# gate de solvabilite, sur les SEULES graines TRAIN, et sans jamais regarder
# l'ecart LAPLACIAN - STD. Calibrer une tache jusqu'a obtenir le resultat voulu
# serait l'oracle par run (corrige le 26/07) sous un autre nom.

CSV_PATH = ROOT / "figures" / "expB7_contiguous_sources_poc.csv"
PNG_PATH = ROOT / "figures" / "expB7_contiguous_sources_poc.png"


def make_deceptive_multicluster(rng, k):
    """Meme tache que dp.make_deceptive, capteurs groupes en k PATCHS par role.

    Memes effectifs (26 leurres / 14 verites), memes amplitudes, meme signe,
    meme graphe : seule la POSITION change. k=1 reproduit le bloc unique.
    Allocation en tourniquet autour de 2k centres tires au hasard, pour que les
    patchs se partagent le tore au lieu qu'un seul centre serve tout le monde.
    """
    adj = make_lattice_adj(SIDE, periodic=True)
    dstar = rng.choice([-1, 1])
    centers = rng.choice(N, size=2 * k, replace=False)
    d_centers, t_centers = centers[:k], centers[k:]
    rem_d = [dp.N_DISTRACT // k + (1 if i < dp.N_DISTRACT % k else 0) for i in range(k)]
    rem_t = [dp.N_TRUE // k + (1 if i < dp.N_TRUE % k else 0) for i in range(k)]

    taken = set()

    def nearest_free(c):
        best, best_d = None, None
        for j in range(N):
            if j in taken:
                continue
            d = b7d.torus_dist(c, j)
            if best_d is None or d < best_d:
                best, best_d = j, d
        return best

    d_nodes, t_nodes = [], []
    while sum(rem_d) + sum(rem_t) > 0:
        for i in range(k):
            if rem_d[i] > 0:
                j = nearest_free(d_centers[i])
                taken.add(j)
                d_nodes.append(j)
                rem_d[i] -= 1
            if rem_t[i] > 0:
                j = nearest_free(t_centers[i])
                taken.add(j)
                t_nodes.append(j)
                rem_t[i] -= 1

    stim_on = np.zeros(N)
    stim_on[np.array(d_nodes, dtype=int)] = -dstar * dp.E_DISTRACT
    stim_on[np.array(t_nodes, dtype=int)] = +dstar * dp.E_TRUE
    stim_off = np.zeros(N)
    stim_off[np.array(t_nodes, dtype=int)] = +dstar * dp.E_TRUE
    return adj, stim_on, stim_off, dstar


def build_task(world, seed):
    rng = np.random.RandomState(3000 + seed)
    if world == "RANDOM":
        return dp.make_deceptive(rng)
    return make_deceptive_multicluster(rng, int(world.split("_K")[1]))


def flip_time(dec, dstar):
    correct = (dec == dstar)
    for t in range(len(dec)):
        if np.all(correct[t:]):
            return t + 1
    return MAXB + 1


def main() -> int:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    adj_ref = make_lattice_adj(SIDE, periodic=True)
    print("=" * 100)
    print("EXPERIENCE B7 volet 2 -- LA TOPOLOGIE SERT-ELLE QUAND IL Y A DE QUOI L'EXPLOITER ?")
    print(f"{len(SEEDS_TRAIN)} graines TRAIN (seuil par monde x signal) + "
          f"{len(SEEDS_TEST)} TEST | T_pulse={T_PULSE}")
    print("=" * 100)

    store = {}
    acc_grid = {}
    gate = {w: dict(acc_final=[], flip=[], always=[], moran_stim=[], moran_v=[])
            for w in WORLDS}

    def run(world, group, seeds):
        for idx, seed in enumerate(seeds):
            adj, s_on, s_off, dstar = build_task(world, seed)
            sigs, dec, probes = b7d.simulate(adj, s_on, s_off, seed * 10 + 1, T_PULSE)
            store[(world, group, idx)] = dict(sigs=sigs, dec=dec, dstar=dstar)
            for sname in STOP_SIGNALS:
                for drop in DROP_GRID:
                    c = b7d.stop_at(sigs[sname], drop)
                    acc_grid.setdefault((world, sname, drop, group), []).append(
                        int(dp.dec_at(dec, c) == dstar))
            if group == "train":     # le gate se juge sur TRAIN, jamais sur TEST
                gate[world]["acc_final"].append(int(dec[-1] == dstar))
                gate[world]["flip"].append(flip_time(dec, dstar))
                gate[world]["always"].append(
                    float(np.mean(dec[WARMUP:T_PULSE] == dstar)))
                gate[world]["moran_stim"].append(b7d.morans_i(s_on, adj_ref))
                gate[world]["moran_v"].append(
                    float(np.mean([b7d.morans_i(probes[t], adj_ref)
                                   for t in PROBE_TIMES])))
        print(f"  [{world}/{group}] {len(seeds)} graines ({time.time() - t0:.0f}s)")

    # --- phase 1 : le gate sur TRAIN, tous les niveaux de structure -----
    for world in WORLDS:
        run(world, "train", SEEDS_TRAIN)

    print("\nGATE SUR TRAIN -- QUELLE STRUCTURE LA TACHE SUPPORTE-T-ELLE ?")
    print("(choix de K sur ce seul tableau ; l'ecart LAPLACIAN-STD n'est pas regarde ici)")
    print(f"{'monde':<16}{'acc_final':>11}{'flip moyen':>12}{'%bascule':>10}"
          f"{'juste a tout inst.':>20}{'Moran I stim':>14}{'Moran I v':>11}")
    for w in WORLDS:
        g = gate[w]
        pct = 100.0 * float(np.mean([f <= MAXB for f in g["flip"]]))
        print(f"{w:<16}{np.mean(g['acc_final']):>11.2f}{np.mean(g['flip']):>12.0f}"
              f"{pct:>9.0f}%{np.mean(g['always']):>20.2f}"
              f"{np.mean(g['moran_stim']):>+14.3f}{np.mean(g['moran_v']):>+11.3f}")

    ref_always = float(np.mean(gate["RANDOM"]["always"]))
    eligible = [f"CLUSTERED_K{k}" for k in K_GRID
                if float(np.mean(gate[f"CLUSTERED_K{k}"]["acc_final"])) >= GATE_ACC_FINAL_MIN
                and float(np.mean(gate[f"CLUSTERED_K{k}"]["flip"])) > 100
                and abs(float(np.mean(gate[f"CLUSTERED_K{k}"]["always"])) - ref_always) < 0.20]
    print(f"  seuil de solvabilite declare : acc_final >= {GATE_ACC_FINAL_MIN:.2f} "
          f"(RANDOM = {np.mean(gate['RANDOM']['acc_final']):.2f})")
    if not eligible:
        print("  -> AUCUN niveau de structure ne preserve la tache. La contiguite des")
        print("     sources et la solvabilite sont incompatibles sur ce readout global :")
        print("     le test decisif ne peut PAS etre construit ainsi. A ecrire tel quel.")
        gate_ok = False
        world_c = f"CLUSTERED_K{K_GRID[0]}"
    else:
        gate_ok = True
        # structure MAXIMALE parmi les mondes solubles = le plus petit K eligible
        world_c = min(eligible, key=lambda w: int(w.split("_K")[1]))
        print(f"  -> monde retenu : {world_c} "
              f"(Moran I stimulus {np.mean(gate[world_c]['moran_stim']):+.3f})")

    # --- phase 2 : la mesure sur les graines reservees ------------------
    for world in ("RANDOM", world_c):
        run(world, "test", SEEDS_TEST)

    WORLDS_CMP = ["RANDOM", world_c]
    chosen = {(w, s): max(DROP_GRID,
                          key=lambda d: np.mean(acc_grid[(w, s, d, "train")]))
              for w in WORLDS_CMP for s in STOP_SIGNALS}

    print("\nJUSTESSE A L'ARRET (seuil regle par monde x signal sur TRAIN, mesure sur TEST)")
    print(f"{'monde':<16}" + "".join(f"{s:>16}" for s in STOP_SIGNALS))
    table = {}
    for w in WORLDS_CMP:
        line = f"{w:<16}"
        for s in STOP_SIGNALS:
            a = np.array(acc_grid[(w, s, chosen[(w, s)], "test")], float)
            table[(w, s)] = a
            line += f"{a.mean():>16.2f}"
        print(line)
    print("  (seuils retenus : " + ", ".join(
        f"{w}/{s}={chosen[(w, s)]:.2f}" for w in WORLDS_CMP for s in STOP_SIGNALS) + ")")

    print("\nLA PREDICTION : (LAPLACIAN - STD) <= 0 dans RANDOM, > 0 dans le monde structure")
    deltas = {}
    for w in WORLDS_CMP:
        d, lo, hi = b7d.boot_ci(table[(w, "LAPLACIAN")] - table[(w, "STD")])
        deltas[w] = (d, lo, hi)
        verdict = ("LAPLACIAN superieur" if lo > 0 else
                   "LAPLACIAN inferieur" if hi < 0 else "non tranche")
        print(f"  {w:<12} LAPLACIAN - STD = {d:+.2f} CI[{lo:+.2f},{hi:+.2f}]  -> {verdict}")
    # effet d'interaction : le gain du laplacien change-t-il entre les deux mondes ?
    inter = ((table[(world_c, "LAPLACIAN")] - table[(world_c, "STD")]).mean()
             - (table[("RANDOM", "LAPLACIAN")] - table[("RANDOM", "STD")]).mean())
    boot = []
    rng_b = np.random.RandomState(20260727)
    n = len(SEEDS_TEST)
    for _ in range(10_000):
        i1 = rng_b.randint(0, n, n)
        i2 = rng_b.randint(0, n, n)
        boot.append(
            (table[(world_c, "LAPLACIAN")][i1] - table[(world_c, "STD")][i1]).mean()
            - (table[("RANDOM", "LAPLACIAN")][i2] - table[("RANDOM", "STD")][i2]).mean())
    lo_i, hi_i = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    print(f"\n  INTERACTION (le gain du laplacien change-t-il avec la structure ?)")
    print(f"  (LAP-STD)_CLUSTERED - (LAP-STD)_RANDOM = {inter:+.2f} CI[{lo_i:+.2f},{hi_i:+.2f}]")

    print("\n--- VERDICT ---")
    if not gate_ok:
        print("  GATE NON FRANCHI -- pas de verdict sur la topologie.")
    elif deltas[world_c][1] > 0 and lo_i > 0:
        print("  PREDICTION TENUE : la ou la tache porte enfin une structure spatiale,")
        print("  le desaccord LOCAL bat la dispersion globale, et l'ecart entre les deux")
        print("  mondes est lui-meme significatif. La topologie n'etait pas inutile --")
        print("  elle etait inexploitable sur la tache d'origine.")
    elif deltas[world_c][1] > 0:
        print("  PARTIELLEMENT TENUE : le laplacien gagne dans le monde structure, mais")
        print("  l'interaction entre mondes n'est pas tranchee. A repliquer avant d'y croire.")
    else:
        print("  PREDICTION REFUTEE : meme la ou la topologie porte de l'information")
        print(f"  (Moran's I du stimulus {np.mean(gate[world_c]['moran_stim']):+.2f}), le")
        print("  desaccord local ne bat pas une simple dispersion globale. Le signal")
        print("  d'arret n'a PAS besoin de la topologie.")
        print("  -> 5e retrecissement de la revendication, a ecrire tel quel.")

    rows = []
    for w in WORLDS:
        in_cmp = w in WORLDS_CMP
        for s in STOP_SIGNALS:
            rows.append(dict(world=w, stop_signal=s,
                             drop_threshold=chosen[(w, s)] if in_cmp else "",
                             acc_test=float(table[(w, s)].mean()) if in_cmp else "",
                             n_seeds_test=len(SEEDS_TEST) if in_cmp else 0,
                             gate_acc_final=float(np.mean(gate[w]["acc_final"])),
                             gate_flip_mean=float(np.mean(gate[w]["flip"])),
                             gate_always_correct=float(np.mean(gate[w]["always"])),
                             moran_stim=float(np.mean(gate[w]["moran_stim"])),
                             moran_v=float(np.mean(gate[w]["moran_v"]))))
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"\n[csv] {CSV_PATH}")
    make_figure(table, deltas, gate, inter, (lo_i, hi_i), WORLDS_CMP)
    print(f"[png] {PNG_PATH}")
    print(f"\nWall time: {time.time() - t0:.1f}s")
    return 0


def make_figure(table, deltas, gate, inter, inter_ci, worlds_cmp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    colors = {"LAPLACIAN": "#d62728", "STD": "#1f77b4", "TEMPORAL": "#7f7f7f"}

    ax = axes[0]
    xs = np.arange(len(worlds_cmp))
    for i, s in enumerate(STOP_SIGNALS):
        ax.bar(xs + (i - 1) * 0.27, [table[(w, s)].mean() for w in worlds_cmp], 0.26,
               color=colors[s], edgecolor="k", label=s)
    ax.axhline(0.5, ls=":", c="gray")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{w}\nMoran I "
                        f"{np.mean(gate[w]['moran_stim']):+.2f}" for w in worlds_cmp],
                       fontsize=8)
    ax.set_ylabel("accuracy at stop (40 held-out seeds)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Does topology help when there is structure?", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    vals = [deltas[w][0] for w in worlds_cmp]
    errs = [[deltas[w][0] - deltas[w][1] for w in worlds_cmp],
            [deltas[w][2] - deltas[w][0] for w in worlds_cmp]]
    ax.bar(range(len(worlds_cmp)), vals, yerr=errs, capsize=5,
           color=["#7f7f7f", "#2ca02c"], edgecolor="k")
    ax.axhline(0, c="k", lw=1)
    ax.set_xticks(range(len(worlds_cmp)))
    ax.set_xticklabels(worlds_cmp, fontsize=8)
    ax.set_ylabel("accuracy( mean|Lv| ) - accuracy( std(v) )")
    ax.set_title(f"The prediction\ninteraction {inter:+.2f} "
                 f"CI[{inter_ci[0]:+.2f},{inter_ci[1]:+.2f}]", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[2]
    labels = ["accuracy at\nunlimited budget", "correct at any\ninstant in window",
              "Moran I\nstimulus", "Moran I\nfield v"]
    keys = ["acc_final", "always", "moran_stim", "moran_v"]
    w_ = 0.8 / max(1, len(WORLDS))
    cmap = ["#7f7f7f", "#2ca02c", "#1f77b4", "#d62728", "#9467bd"]
    for i, w in enumerate(WORLDS):
        ax.bar(np.arange(len(keys)) + (i - (len(WORLDS) - 1) / 2) * w_,
               [np.mean(gate[w][k]) for k in keys], w_,
               color=cmap[i % len(cmap)], edgecolor="k", label=w)
    ax.axhline(0, c="k", lw=0.8)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_title("Gate: is it still the same task?", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Experiment B7 (part 2) -- asking the topology question where the "
                 "topology could actually matter", fontsize=11)
    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=140)


if __name__ == "__main__":
    raise SystemExit(main())
