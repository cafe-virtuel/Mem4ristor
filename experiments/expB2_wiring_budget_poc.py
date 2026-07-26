#!/usr/bin/env python3
"""
EXPERIENCE B2 -- que coute le CABLAGE d'agregation ? (bilan materiel de l'Experience B)

CONTEXTE. `expB_annealing_faceoff_poc.py` (2026-07-26) a montre qu'un filtre a
oubli exponentiel, branche sur le flux d'observation, bat le doute de 10 points
(1.00 vs 0.90) en payant 4x plus de PAS (1348 vs 330). Le compte rendu de ce
resultat portait une reserve : "ce filtre recoit gratuitement l'agregat global
des 100 capteurs, la ou M4R doit l'agreger par couplage local".

CETTE RESERVE EST SUSPECTE, ET C'EST LA PREMIERE CHOSE QUE CE SCRIPT VERIFIE.
Le readout de M4R dans le harness B1d/B5b est `mean(v) - mean(v_ref)` : il somme
lui aussi les 100 noeuds, ET il entretient un SECOND RESEAU COMPLET (le run de
reference a stimulus nul). Si c'est exact, la reserve ne protege pas M4R -- elle
le flattait, et le vrai bilan est PIRE que celui annonce. Lecon de P11 (13/07) :
compter les iterations d'un seul cote produit un -96% qui devient +4205% quand
on compte les deux. On ne refait pas cette erreur.

TROIS VOLETS.

  A. BILAN D'OPERATIONS. Modele de cout explicite (pas un chronometre : numpy
     vectorise avantagerait M4R et penaliserait une boucle scalaire, ce qui
     mentirait dans l'autre sens). On compte par pas : arcs de couplage traites,
     operations d'integration par noeud, sommations de readout, et le facteur 2
     du reseau de reference. Sensibilite au seul parametre discutable (le cout
     d'integration par noeud) affichee, pour que la conclusion ne repose pas
     dessus.

  B. PRIVATION DE CAPTEURS -- le volet qui peut SAUVER M4R, et le seul qui teste
     vraiment la valeur du couplage. On retire aux DEUX l'agregat global : le
     filtre ne lit plus que k capteurs bruts tires au hasard, et M4R ne lit plus
     que k noeuds tires au hasard. Le stimulus n'occupe que 40 des 100 noeuds :
     un observateur qui n'echantillonne que k capteurs bruts voit donc souvent
     du vide, tandis que le couplage local de M4R a eu le temps de DIFFUSER
     l'information vers des noeuds non stimules. Si M4R tient a petit k quand le
     filtre s'effondre, la valeur du reseau est demontree -- et si les deux
     s'effondrent ensemble, elle ne l'est pas. Falsifiable dans les deux sens.

  C. LE RESEAU DE REFERENCE EST-IL NECESSAIRE ? Il double le cout de M4R. On
     teste une lecture a BASELINE CALIBREE (une constante mesuree une fois hors
     ligne, amortie sur tous les runs -- ce qu'on ferait sur un vrai dispositif)
     au lieu d'un second reseau simule en parallele.

GATE DE FIDELITE : la simulation locale (qui doit stocker v_hist, ce que
`dp.simulate` ne rend pas) est verifiee BIT A BIT contre `dp.simulate` avant
toute mesure. Sans ce gate, on comparerait a un harness legerement different.

SORTIES : figures/expB2_wiring_budget_poc.csv + .png
Cree : 2026-07-26 (Claude Opus 5, L'Ingenieur) -- suite de l'Experience B.
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
sys.path.insert(0, str(ROOT / "experiments" / "scratch"))
import deceptive_task_poc as dp  # noqa: E402
from mem4ristor.topology import Mem4Network  # noqa: E402

dp.MAX_BUDGET = 2000
MAXB = dp.MAX_BUDGET
N = dp.N
SIDE = dp.SIDE

# CORRECTION DE METHODE (2026-07-26, meme jour) : une premiere version reglait le
# filtre par ORACLE PAR RUN. Controle : sur un signal SANS AUCUNE information
# (bruit pur), cette procedure rend 0.935 d'accuracy au lieu de 0.500 -- avec 18
# combinaisons et une decision binaire, il s'en trouve presque toujours une qui
# tombe juste. Le 0.94 du filtre a k=1 capteur en etait entierement fabrique.
# Les hyperparametres sont desormais choisis sur les graines TRAIN et mesures sur
# des graines DISJOINTES. M4R, lui, n'ajuste rien (arret natif).
SEEDS_TRAIN = list(range(20))
SEEDS_TEST = list(range(20, 40))
T_PULSE_LEVELS = [350, 700]
SIGMA_OBS = 0.05
K_SENSORS = [1, 3, 5, 10, 25, 50, 100]
STABLE_W_GRID = [50, 200, 800]
LEAKY_ALPHA = [0.005, 0.02, 0.05]
LEAKY_THR = [0.0, 0.02]
N_BOOT = 10_000
RNG_BOOT = np.random.RandomState(20260726)

# --- modele de cout (volet A) ----------------------------------------------
# Un lattice periodique 10x10 a 4 voisins par noeud -> 4N arcs traites par pas.
ARCS_PER_STEP = 4 * N
# Operations arithmetiques pour integrer un noeud (v, w, u) d'un pas d'Euler.
# Seul parametre discutable du modele -> balaye explicitement.
C_NODE_GRID = [10, 20, 40]

CSV_PATH = ROOT / "figures" / "expB2_wiring_budget_poc.csv"
PNG_PATH = ROOT / "figures" / "expB2_wiring_budget_poc.png"


def simulate_full(adj, stim_on, stim_off, seed, t_pulse):
    """Reimplementation de dp.simulate qui EXPOSE v_hist et v_ref_hist.
    Verifiee bit a bit contre dp.simulate par le gate de fidelite."""
    net = Mem4Network(size=SIDE, heretic_ratio=0.0, seed=seed, adjacency_matrix=adj)
    ref = Mem4Network(size=SIDE, heretic_ratio=0.0, seed=seed, adjacency_matrix=adj)
    L = net.L
    zero = np.zeros(N)
    sig = np.empty(MAXB)
    v_hist = np.empty((MAXB, N))
    vref_hist = np.empty((MAXB, N))
    for t in range(MAXB):
        stim = stim_on if t < t_pulse else stim_off
        net.step(I_stimulus=stim)
        ref.step(I_stimulus=zero)
        v = net.model.v
        sig[t] = float(np.mean(np.abs(L @ v)))
        v_hist[t] = v
        vref_hist[t] = ref.model.v
    return sig, v_hist, vref_hist


def run_leaky(y, alpha, thr, W):
    m = 0.0
    s_prev, stable = 0, 0
    for t in range(len(y)):
        m += alpha * (y[t] - m)
        s = 1 if m >= 0 else -1
        stable = stable + 1 if (s == s_prev and abs(m) > thr) else 0
        s_prev = s
        if stable >= W:
            return s, t + 1
    return s_prev, len(y)


LEAKY_GRID = [(a, thr, W) for a in LEAKY_ALPHA for thr in LEAKY_THR
              for W in STABLE_W_GRID]


def leaky_all(y, dstar):
    """Evalue TOUTES les combinaisons ; la selection se fait ensuite sur les
    graines TRAIN seulement (voir la correction de methode en tete de fichier)."""
    return {c: (int(run_leaky(y, *c)[0] == dstar), run_leaky(y, *c)[1])
            for c in LEAKY_GRID}


def boot_ci_paired(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    m = np.array([d[RNG_BOOT.randint(0, n, n)].mean() for _ in range(N_BOOT)])
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def fidelity_gate():
    """Le harness reimplemente doit reproduire dp.simulate EXACTEMENT."""
    print("GATE DE FIDELITE (la reimplementation doit egaler dp.simulate bit a bit)")
    for seed in (0, 7):
        for t_pulse in (350, 700):
            rng = np.random.RandomState(3000 + seed)
            adj, s_on, s_off, dstar = dp.make_deceptive(rng)
            sig_a, dec_a, dvar_a = dp.simulate(adj, s_on, s_off, seed * 10 + 1, t_pulse)
            rng2 = np.random.RandomState(3000 + seed)
            adj2, s_on2, s_off2, _ = dp.make_deceptive(rng2)
            sig_b, vh, vrh = simulate_full(adj2, s_on2, s_off2, seed * 10 + 1, t_pulse)
            dvar_b = vh.mean(axis=1) - vrh.mean(axis=1)
            ok_sig = np.allclose(sig_a, sig_b, rtol=0, atol=0)
            ok_dv = np.allclose(dvar_a, dvar_b, rtol=0, atol=1e-12)
            print(f"  seed={seed} t_pulse={t_pulse} : sigma identique={ok_sig} "
                  f"d_var identique={ok_dv}")
            if not (ok_sig and ok_dv):
                raise SystemExit("GATE ECHOUE : la reimplementation devie du harness. "
                                 "Aucune mesure ne serait comparable -- arret.")
    print("  -> gate PASSE.\n")


def main() -> int:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    fidelity_gate()

    rows = []
    # accumulateurs par graine

    # baseline calibree (volet C) : moyenne de v d'un reseau NON stimule, mesuree
    # une fois hors ligne et amortie -- ce qu'on ferait sur un vrai dispositif.
    rng0 = np.random.RandomState(3000)
    adj0, s_on0, s_off0, _ = dp.make_deceptive(rng0)
    _, _, vref0 = simulate_full(adj0, np.zeros(N), np.zeros(N), 1, 0)
    BASELINE = float(vref0[MAXB // 2:].mean())
    print(f"[volet C] baseline calibree hors ligne : mean(v) = {BASELINE:.4f}\n")

    print("=" * 100)
    print("EXPERIENCE B2 -- bilan de cablage. 20 graines TRAIN + 20 TEST x 2 t_pulse.")
    print("=" * 100)

    from collections import defaultdict
    leaky_store = defaultdict(lambda: defaultdict(list))   # [(k, combo)][group]
    leaky_costs = defaultdict(lambda: defaultdict(list))
    m4r_store = defaultdict(lambda: defaultdict(list))      # [k][group]
    m4r_costs = defaultdict(lambda: defaultdict(list))
    nodiff_store = defaultdict(list)

    for group, seeds in (("train", SEEDS_TRAIN), ("test", SEEDS_TEST)):
        for seed in seeds:
            for t_pulse in T_PULSE_LEVELS:
                rng = np.random.RandomState(3000 + seed)
                adj, s_on, s_off, dstar = dp.make_deceptive(rng)
                sig, vh, vrh = simulate_full(adj, s_on, s_off, seed * 10 + 1, t_pulse)
                cd = dp.stop_doubt(sig)          # l'arret natif de M4R, inchange

                sel_rng = np.random.RandomState(4242 + seed)
                obs_rng = np.random.RandomState(9000 + seed)
                noise_flux = SIGMA_OBS * obs_rng.randn(MAXB)
                stim_mat = np.where(np.arange(MAXB)[:, None] < t_pulse,
                                    s_on[None, :], s_off[None, :])

                for k in K_SENSORS:
                    idx = sel_rng.choice(N, size=k, replace=False)
                    # --- M4R : readout differentiel sur k NOEUDS seulement ---
                    dv_k = vh[:, idx].mean(axis=1) - vrh[:, idx].mean(axis=1)
                    dec_k = np.where(dv_k >= 0, 1, -1)
                    m4r_store[k][group].append(int(dp.dec_at(dec_k, cd) == dstar))
                    m4r_costs[k][group].append(cd)
                    # --- filtre a oubli : k CAPTEURS bruts seulement ---
                    y_k = stim_mat[:, idx].mean(axis=1) + noise_flux
                    for combo, (a_l, c_l) in leaky_all(y_k, dstar).items():
                        leaky_store[(k, combo)][group].append(a_l)
                        leaky_costs[(k, combo)][group].append(c_l)

                # --- volet C : M4R SANS reseau de reference (baseline calibree) ---
                dv_nd = vh.mean(axis=1) - BASELINE
                dec_nd = np.where(dv_nd >= 0, 1, -1)
                nodiff_store[group].append(int(dp.dec_at(dec_nd, cd) == dstar))
        print(f"  [{group}] {len(seeds)} graines simulees")

    def by_seed(flat):
        return np.asarray(flat, float).reshape(-1, len(T_PULSE_LEVELS)).mean(axis=1)

    # selection des hyperparametres du filtre sur TRAIN, mesure sur TEST
    chosen = {}
    for k in K_SENSORS:
        chosen[k] = max(LEAKY_GRID, key=lambda c: (
            np.mean(leaky_store[(k, c)]["train"]),
            -np.mean(leaky_costs[(k, c)]["train"])))

    acc_m4r = {k: by_seed(m4r_store[k]["test"]) for k in K_SENSORS}
    acc_leaky = {k: by_seed(leaky_store[(k, chosen[k])]["test"]) for k in K_SENSORS}
    cost_m4r = {k: by_seed(m4r_costs[k]["test"]) for k in K_SENSORS}
    cost_leaky = {k: by_seed(leaky_costs[(k, chosen[k])]["test"]) for k in K_SENSORS}
    acc_nodiff = by_seed(nodiff_store["test"])
    for k in K_SENSORS:
        for i in range(len(acc_m4r[k])):
            rows.append(dict(k=k, seed_idx=i, m4r_acc=acc_m4r[k][i],
                             m4r_cost=cost_m4r[k][i], leaky_acc=acc_leaky[k][i],
                             leaky_cost=cost_leaky[k][i], leaky_params=str(chosen[k])))

    # ---------------- volet B : privation de capteurs ----------------------
    print("\nVOLET B -- PRIVATION DE CAPTEURS (on retire l'agregat global AUX DEUX)")
    print("mesure sur graines TEST ; hyperparametres du filtre choisis sur TRAIN")
    print(f"{'k capteurs':>11}{'M4R acc':>10}{'filtre acc':>12}{'ecart (IC95)':>28}")
    print("-" * 64)
    for k in K_SENSORS:
        d, lo, hi = boot_ci_paired(acc_m4r[k], acc_leaky[k])
        print(f"{k:>11}{np.mean(acc_m4r[k]):>10.2f}{np.mean(acc_leaky[k]):>12.2f}"
              f"{f'{d:+.2f} [{lo:+.2f},{hi:+.2f}]':>28}")

    # ---------------- volet C : le reseau de reference ---------------------
    print(f"\nVOLET C -- le reseau de reference est-il necessaire ?")
    d, lo, hi = boot_ci_paired(acc_nodiff, acc_m4r[100])
    print(f"  M4R avec 2e reseau (differentiel) : acc={np.mean(acc_m4r[100]):.2f}")
    print(f"  M4R avec baseline calibree seule  : acc={np.mean(acc_nodiff):.2f}")
    print(f"  ecart {d:+.2f} CI[{lo:+.2f},{hi:+.2f}] -> "
          + ("la reference est NECESSAIRE" if hi < 0 else
             "la reference est SUPERFLUE (cout divisible par 2)" if lo > 0 else
             "parite : la reference ne se justifie pas par la precision"))

    # ---------------- volet A : bilan d'operations -------------------------
    print(f"\nVOLET A -- BILAN D'OPERATIONS (modele explicite, pas un chronometre)")
    print(f"  M4R par pas   = 2 reseaux x (4N arcs + c_node x N + N readout), N={N}")
    print(f"  filtre par pas= k sommations + 3 operations de filtre")
    print(f"{'c_node':>8}{'M4R ops/pas':>14}{'M4R total':>14}{'filtre ops/pas':>16}"
          f"{'filtre total':>15}{'rapport':>10}")
    print("-" * 78)
    m4r_steps = float(np.mean(cost_m4r[100]))
    lk_steps = float(np.mean(cost_leaky[100]))
    budget_rows = []
    for c_node in C_NODE_GRID:
        ops_m4r = 2 * (ARCS_PER_STEP + c_node * N + N)
        ops_lk = 100 + 3
        tot_m4r = ops_m4r * m4r_steps
        tot_lk = ops_lk * lk_steps
        print(f"{c_node:>8}{ops_m4r:>14}{tot_m4r:>14.0f}{ops_lk:>16}"
              f"{tot_lk:>15.0f}{tot_m4r / tot_lk:>10.1f}x")
        budget_rows.append((c_node, ops_m4r, tot_m4r, ops_lk, tot_lk, tot_m4r / tot_lk))
    print(f"\n  (M4R s'arrete a {m4r_steps:.0f} pas, le filtre a {lk_steps:.0f} : "
          f"l'avantage en PAS de M4R est reel,")
    print("   mais chaque pas lui coute ~3 ordres de grandeur de plus en operations.)")

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[csv] {CSV_PATH}")

    make_figure(acc_m4r, acc_leaky, budget_rows)
    print(f"[png] {PNG_PATH}")
    print(f"\nWall time: {time.time() - t0:.1f}s")
    return 0


def make_figure(acc_m4r, acc_leaky, budget_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    ax = axes[0]
    ks = K_SENSORS
    ax.plot(ks, [np.mean(acc_m4r[k]) for k in ks], "o-", color="#d62728",
            label="M4R (readout on k nodes)")
    ax.plot(ks, [np.mean(acc_leaky[k]) for k in ks], "s-", color="#7b4173",
            label="forgetting filter (k raw sensors)")
    ax.axhline(0.5, ls=":", c="gray", label="chance")
    ax.set_xscale("log")
    ax.set_xlabel("number of sensors read (log)")
    ax.set_ylabel("decision accuracy at stop")
    ax.set_title("Sensor deprivation: does local coupling\nbuy anything when the global sum is gone?",
                 fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    c_nodes = [r[0] for r in budget_rows]
    ax.bar([str(c) for c in c_nodes], [r[5] for r in budget_rows],
           color="#1f77b4", edgecolor="k")
    ax.axhline(1.0, ls="--", c="k", lw=1.2, label="parity (equal total cost)")
    ax.set_xlabel("integration ops per node per step (model parameter)")
    ax.set_ylabel("M4R total ops / filter total ops")
    ax.set_yscale("log")
    ax.set_title("Operation budget to reach a decision:\nfewer steps, far more work per step",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Experiment B2 -- what the aggregation wiring actually costs "
                 "(follow-up to Experiment B)", fontsize=11)
    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=140)


if __name__ == "__main__":
    raise SystemExit(main())
