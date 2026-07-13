#!/usr/bin/env python3
"""
B1d -- ECONOMIE D'ECHELLE : le gain du doute survit-il avec BEAUCOUP MOINS
de noeuds ?
=============================================================================
Cree : 2026-07-13 (Claude Sonnet 5, L'Ingenieur). Demande de Julien, apres
une longue discussion honnete sur ce que M4R fait vraiment : « on sait qu'il
est meilleur pour de la recherche pure quand on part de rien et que la
contrainte n'est pas le temps ; son defaut majeur c'est que contraint il
est meme pire que ce qui se fait ; je crois que le second axe [economie
materielle, pas performance brute] est peut-etre a tester ».

Le seul terrain ou M4R gagne REELLEMENT, replique sur des MOIS et
plusieurs substrats (B1d/B5b/P11/P12/pont LLM) : la decision sous
incertitude a horizon inconnu, quand converger tot est un piege. Question
d'ECONOMIE posee ici, jamais testee : cet avantage survit-il avec
BEAUCOUP MOINS de noeuds -- N=100 est-il necessaire, ou l'effet tient-il
a N=49, 25, meme 9 ? En materiel reel, N = nombre de dispositifs physiques
(memristors, oscillateurs) -- si le gain du doute survit a un N bien plus
petit, c'est un vrai argument d'economie materielle (moins de composants
pour le meme comportement gagnant). S'il s'effondre en dessous d'un
certain N, ca borne la reponse : "M4R est economique" serait faux en
dessous de ce seuil.

Protocole : reprend `deceptive_task_poc.py` (B1d) A L'IDENTIQUE (meme
piege PULSE, meme readout differentiel, memes seuils DOUBT_DROP/CONV_THR/
CONV_W, memes T_PULSE_LEVELS testes) -- seul SIDE (donc N) varie, et
N_DISTRACT/N_TRUE sont recalcules en gardant les MEMES PROPORTIONS que
l'original (N_DISTRACT/N=0.26, N_TRUE/N=0.14, ratio verite/leurre=0.538).
SIDE in {3, 5, 7, 10} -> N in {9, 25, 49, 100}. A SIDE=3, N_TRUE arrondi a
1 seul noeud -- cas limite honnetement signale, pas cache.

Statut : exploratoire, hors preprint, aucune modification de dynamics.py.
Guardian doit rester 14/14. Sorties : figures/b1d_scale_economy_poc.csv + .png
"""
from __future__ import annotations

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
from mem4ristor.topology import Mem4Network  # noqa: E402
from mem4ristor.graph_utils import make_lattice_adj  # noqa: E402

FIG = ROOT / "figures"
MAX_BUDGET = 3000
WARMUP = 30
DOUBT_DROP = 0.30
CONV_W = 50
CONV_THR = 0.02
SEEDS = list(range(12))
E_TRUE = 0.6
E_DISTRACT = 1.0
T_PULSE_LEVELS = [350, 700]
SIDES = [3, 5, 7, 10]

# Proportions EXACTES de l'original (SIDE=10, N=100 : N_DISTRACT=26, N_TRUE=14)
FRAC_DISTRACT = 26 / 100
FRAC_TRUE = 14 / 100


def counts_for_side(side):
    n = side * side
    n_distract = max(1, round(n * FRAC_DISTRACT))
    n_true = max(1, round(n * FRAC_TRUE))
    return n, n_distract, n_true


def make_deceptive(rng, n, n_distract, n_true):
    adj = make_lattice_adj(int(np.sqrt(n)), periodic=True)
    dstar = rng.choice([-1, 1])
    nodes = rng.choice(n, size=n_distract + n_true, replace=False)
    d_nodes, t_nodes = nodes[:n_distract], nodes[n_distract:]
    stim_on = np.zeros(n)
    stim_on[d_nodes] = -dstar * E_DISTRACT
    stim_on[t_nodes] = +dstar * E_TRUE
    stim_off = np.zeros(n)
    stim_off[t_nodes] = +dstar * E_TRUE
    return adj, stim_on, stim_off, dstar


def simulate(side, n, adj, stim_on, stim_off, seed, t_pulse):
    net = Mem4Network(size=side, heretic_ratio=0.0, seed=seed, adjacency_matrix=adj)
    ref = Mem4Network(size=side, heretic_ratio=0.0, seed=seed, adjacency_matrix=adj)
    L = net.L
    zero = np.zeros(n)
    sig = np.empty(MAX_BUDGET)
    d_var = np.empty(MAX_BUDGET)
    dec = np.empty(MAX_BUDGET, dtype=int)
    for t in range(MAX_BUDGET):
        stim = stim_on if t < t_pulse else stim_off
        net.step(I_stimulus=stim)
        ref.step(I_stimulus=zero)
        v = net.model.v
        sig[t] = float(np.mean(np.abs(L @ v)))
        d = float(np.mean(v) - np.mean(ref.model.v))
        d_var[t] = d
        dec[t] = 1 if d >= 0 else -1
    return sig, dec, d_var


def stop_doubt(sig):
    peak = float(np.max(sig[:WARMUP + 20]))
    thr = DOUBT_DROP * peak
    for t in range(WARMUP, len(sig)):
        if sig[t] < thr:
            return t + 1
    return len(sig)


def stop_conv(d_var):
    for t in range(WARMUP + CONV_W, len(d_var)):
        if abs(d_var[t] - d_var[t - CONV_W]) < CONV_THR:
            return t + 1
    return len(d_var)


def flip_time(dec, dstar):
    correct = (dec == dstar)
    for t in range(len(dec)):
        if np.all(correct[t:]):
            return t + 1
    return MAX_BUDGET + 1


def dec_at(dec, t):
    return int(dec[min(int(t), MAX_BUDGET) - 1])


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = []
    print("=== ECONOMIE D'ECHELLE : le gain du doute (B1d) survit-il a N petit ? ===\n")
    print(f"{'SIDE':>5}{'N':>5}{'n_distr':>8}{'n_true':>7}{'T_pulse':>8}"
          f"{'%bascule':>10}{'acc_DOUTE':>11}{'acc_CONV':>10}{'gain':>8}{'acc_FIN':>9}")
    print("-" * 80)

    summary = {}
    for side in SIDES:
        n, n_distract, n_true = counts_for_side(side)
        for t_pulse in T_PULSE_LEVELS:
            acc_d, acc_c, acc_fin, flips = [], [], [], []
            for seed in SEEDS:
                rng = np.random.RandomState(3000 + seed)
                adj, stim_on, stim_off, dstar = make_deceptive(rng, n, n_distract, n_true)
                sig, dec, d_var = simulate(side, n, adj, stim_on, stim_off, seed * 10 + 1, t_pulse)
                ft = flip_time(dec, dstar)
                cd = stop_doubt(sig)
                cc = stop_conv(d_var)
                a_d = int(dec_at(dec, cd) == dstar)
                a_c = int(dec_at(dec, cc) == dstar)
                a_f = int(dec[-1] == dstar)
                acc_d.append(a_d); acc_c.append(a_c); acc_fin.append(a_f); flips.append(ft)
            ad, ac, af = float(np.mean(acc_d)), float(np.mean(acc_c)), float(np.mean(acc_fin))
            gain = ad - ac
            pct_flip = 100.0 * np.mean([f <= MAX_BUDGET for f in flips])
            summary[(side, t_pulse)] = (n, n_distract, n_true, pct_flip, ad, ac, gain, af)
            print(f"{side:>5}{n:>5}{n_distract:>8}{n_true:>7}{t_pulse:>8}"
                  f"{pct_flip:>9.0f}%{ad:>11.2f}{ac:>10.2f}{gain:>+8.2f}{af:>9.2f}")
            rows.append((side, n, n_distract, n_true, t_pulse, pct_flip, ad, ac, gain, af))

    print("\n=== VERDICT : l'avantage du doute est-il ECONOMIQUE (petit N) ? ===")
    for t_pulse in T_PULSE_LEVELS:
        print(f"\n  -- T_pulse={t_pulse} --")
        gains_by_side = {}
        for side in SIDES:
            n, nd, nt, pf, ad, ac, gain, af = summary[(side, t_pulse)]
            gains_by_side[side] = gain
            trustworthy = af >= 0.6 and pf > 30
            flag = "" if trustworthy else "  [regime non-trompeur ou peu fiable a cette taille]"
            print(f"    SIDE={side:<3} N={n:<4} gain_doute={gain:+.2f}{flag}")
        n_min_100 = SIDES[-1]
        gain_full = gains_by_side[n_min_100]
        surviving = [s for s in SIDES if gains_by_side[s] > 0.10]
        if len(surviving) == len(SIDES):
            print(f"    -> L'avantage SURVIT a TOUTES les tailles testees, y compris N={SIDES[0]**2} "
                  "-- economie materielle plausible : le gain ne depend pas d'un grand nombre de noeuds.")
        elif surviving:
            min_n_ok = min(s * s for s in surviving)
            print(f"    -> L'avantage SURVIT seulement a partir de N>={min_n_ok} -- il y a un SEUIL "
                  "de taille minimal, l'economie materielle a une limite basse.")
        else:
            print("    -> L'avantage ne survit a AUCUNE des tailles reduites testees -- "
                  "N=100 (ou plus) semble necessaire, PAS d'economie materielle a esperer ici.")

    with (FIG / "b1d_scale_economy_poc.csv").open("w", encoding="utf-8") as f:
        f.write("side,n,n_distract,n_true,t_pulse,pct_flip,acc_doute,acc_conv,gain,acc_final\n")
        for r in rows:
            f.write(",".join(f"{x:.6f}" if isinstance(x, float) else str(x) for x in r) + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
        ns = [s * s for s in SIDES]
        for ax, t_pulse in zip(axes, T_PULSE_LEVELS):
            gains = [summary[(s, t_pulse)][6] for s in SIDES]
            ad_vals = [summary[(s, t_pulse)][4] for s in SIDES]
            ac_vals = [summary[(s, t_pulse)][5] for s in SIDES]
            ax.plot(ns, ad_vals, "o-", c="#d62728", label="acc DOUTE")
            ax.plot(ns, ac_vals, "s-", c="#1f77b4", label="acc CONVERGENCE")
            ax2 = ax.twinx()
            ax2.plot(ns, gains, "^--", c="#2ca02c", alpha=0.6, label="gain")
            ax2.axhline(0, ls=":", c="gray", alpha=0.5)
            ax.set_xlabel("N (nombre de noeuds)"); ax.set_ylabel("accuracy")
            ax2.set_ylabel("gain (DOUTE-CONV)", color="#2ca02c")
            ax.set_title(f"T_pulse={t_pulse}"); ax.set_ylim(-0.05, 1.05)
            ax.legend(loc="lower left", fontsize=7); ax2.legend(loc="upper right", fontsize=7)
            ax.grid(alpha=0.3)
        fig.suptitle("B1d -- l'avantage du doute survit-il a un N petit (economie materielle) ?", fontsize=11)
        plt.tight_layout()
        plt.savefig(FIG / "b1d_scale_economy_poc.png", dpi=140)
        print(f"\n[png] {FIG / 'b1d_scale_economy_poc.png'}")
    except Exception as e:
        print(f"[png] skipped: {e}")

    print(f"\nWall time: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
