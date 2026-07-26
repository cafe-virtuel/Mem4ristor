#!/usr/bin/env python3
"""
EXPERIENCE B4 -- la latence de decision, ancree dans le temps PHYSIQUE
et decomposee par rapport a la duree du leurre.

CONTEXTE. Apres les corrections de la journee (expB, expB2, expB3-bis), une
seule grandeur appartient encore en propre a M4R : il tranche en 309 pas contre
1348 pour le filtre a oubli, soit 4.4x moins. Tout le reste (energie, operations)
s'est retourne, ou depend du substrat des DEUX cotes.
Ce script attaque ce dernier chiffre par les deux endroits ou il peut ceder.

  1. UN DETAIL QUI CLOCHE, ET QUI DECIDE DE TOUT. Le leurre dure T_pulse = 350 ou
     700 pas. M4R s'arrete en moyenne a 309. Il tranche donc AVANT la fin du
     leurre, et il a raison 9 fois sur 10. Deux lectures s'excluent :
       (a) il extrait vraiment la verite MINORITAIRE (14 capteurs a 0.6) pendant
           que le leurre MAJORITAIRE (26 capteurs a 1.0) domine encore la moyenne
           -- ce serait le resultat le plus fort du projet sur cette tache ;
       (b) le chiffre agrege melange les deux niveaux de leurre et masque un
           comportement bien plus banal.
     On decompose donc PAR T_pulse, et on mesure l'accuracy CONDITIONNELLE au
     fait d'avoir tranche avant la fin du leurre. C'est le test qui separe (a)
     de (b), et il peut tuer le resultat.

  2. L'ANCRAGE PHYSIQUE. "4.4x moins de pas" ne dit rien tant qu'un pas n'a pas
     de duree. On ancre dt avec les trois familles de
     `docs/hardware/B3_ENERGY_COMPARISON.md` pour obtenir des temps ABSOLUS.
     Note importante sur ce que l'ancrage change et ne change pas : les deux
     decideurs observent LE MEME flux, donc subissent le meme dt -- l'ancrage ne
     modifie pas le RAPPORT 4.4, il donne l'echelle a laquelle ce rapport se
     joue (gagner 3 ns et gagner 3 us ne se valent pas dans une application).
     Le filtre a oubli est un RC : sa constante de temps se choisit librement, il
     n'impose aucun plancher ; c'est M4R qui a un dt minimal impose par la
     physique de son dispositif. On affiche donc aussi ce plancher.

Selection des hyperparametres du filtre sur graines TRAIN, mesure sur graines
TEST disjointes (correction de methode du 2026-07-26 -- l'oracle par run
fabriquait 0.935 d'accuracy sur du bruit pur).

SORTIES : figures/expB4_latency_anchored_poc.csv + .png
Cree : 2026-07-26 (Claude Opus 5, L'Ingenieur) -- suite de expB3-bis.
"""
from __future__ import annotations

import csv
import sys
import time
from collections import defaultdict
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

dp.MAX_BUDGET = 2000
MAXB = dp.MAX_BUDGET
N = dp.N

SEEDS_TRAIN = list(range(20))
SEEDS_TEST = list(range(20, 40))
T_PULSE_LEVELS = [350, 700]
SIGMA_OBS = 0.05
STABLE_W_GRID = [50, 200, 800]
LEAKY_ALPHA = [0.005, 0.02, 0.05]
LEAKY_THR = [0.0, 0.02]
LEAKY_GRID = [(a, t, w) for a in LEAKY_ALPHA for t in LEAKY_THR for w in STABLE_W_GRID]
N_BOOT = 10_000
RNG_BOOT = np.random.RandomState(20260726)

# dt physique par pas de modele -- docs/hardware/B3_ENERGY_COMPARISON.md, section 2.
# (ordres de grandeur de brique elementaire, pas des mesures systeme)
DT_PHYS = {
    "STNO vortex (v)":       (2.25e-12, 22.5e-12),
    "Photonique GST (u)":    (225e-12, 449e-12),
    "Neuristor Mott (v)":    (2.25e-9, 4.49e-9),
}

CSV_PATH = ROOT / "figures" / "expB4_latency_anchored_poc.csv"
PNG_PATH = ROOT / "figures" / "expB4_latency_anchored_poc.png"


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


def boot_ci(v):
    v = np.asarray(v, float)
    n = len(v)
    m = np.array([v[RNG_BOOT.randint(0, n, n)].mean() for _ in range(N_BOOT)])
    return float(np.mean(v)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def fmt_time(s):
    for unit, scale in (("ps", 1e-12), ("ns", 1e-9), ("us", 1e-6), ("ms", 1e-3)):
        if s < scale * 1000:
            return f"{s / scale:.2f} {unit}"
    return f"{s:.2e} s"


def main() -> int:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    store = defaultdict(lambda: defaultdict(list))    # [(t_pulse, combo)][group]
    costs = defaultdict(lambda: defaultdict(list))
    m4r = defaultdict(lambda: defaultdict(list))      # [t_pulse][group] -> acc
    m4r_cost = defaultdict(lambda: defaultdict(list))
    rows = []

    print("=" * 100)
    print("EXPERIENCE B4 -- latence de decision ancree, decomposee par duree de leurre")
    print("=" * 100)

    for group, seeds in (("train", SEEDS_TRAIN), ("test", SEEDS_TEST)):
        for seed in seeds:
            for t_pulse in T_PULSE_LEVELS:
                rng = np.random.RandomState(3000 + seed)
                adj, s_on, s_off, dstar = dp.make_deceptive(rng)
                sig, dec, d_var = dp.simulate(adj, s_on, s_off, seed * 10 + 1, t_pulse)
                cd = dp.stop_doubt(sig)
                m4r[t_pulse][group].append(int(dp.dec_at(dec, cd) == dstar))
                m4r_cost[t_pulse][group].append(cd)

                obs_rng = np.random.RandomState(9000 + seed)
                y = (np.where(np.arange(MAXB) < t_pulse, s_on.mean(), s_off.mean())
                     + SIGMA_OBS * obs_rng.randn(MAXB))
                for c in LEAKY_GRID:
                    d_, k_ = run_leaky(y, *c)
                    store[(t_pulse, c)][group].append(int(d_ == dstar))
                    costs[(t_pulse, c)][group].append(k_)

    chosen = {tp: max(LEAKY_GRID, key=lambda c: (np.mean(store[(tp, c)]["train"]),
                                                 -np.mean(costs[(tp, c)]["train"])))
              for tp in T_PULSE_LEVELS}

    # ---- 1. LE TEST QUI PEUT TUER LE RESULTAT --------------------------
    print("\n1. M4R TRANCHE-T-IL AVANT LA FIN DU LEURRE ? (graines TEST)")
    print(f"{'T_pulse':>8}{'M4R arret':>11}{'% avant fin leurre':>20}"
          f"{'acc si avant':>14}{'acc si apres':>14}{'filtre arret':>14}{'acc filtre':>12}")
    print("-" * 96)
    summary = {}
    for tp in T_PULSE_LEVELS:
        cds = np.array(m4r_cost[tp]["test"], float)
        accs = np.array(m4r[tp]["test"], float)
        before = cds < tp
        acc_before = accs[before].mean() if before.any() else float("nan")
        acc_after = accs[~before].mean() if (~before).any() else float("nan")
        c = chosen[tp]
        lk_cost = float(np.mean(costs[(tp, c)]["test"]))
        lk_acc = float(np.mean(store[(tp, c)]["test"]))
        print(f"{tp:>8}{cds.mean():>11.0f}{100 * before.mean():>19.0f}%"
              f"{acc_before:>14.2f}{acc_after:>14.2f}{lk_cost:>14.0f}{lk_acc:>12.2f}")
        summary[tp] = dict(m4r_cost=cds.mean(), frac_before=before.mean(),
                           acc_before=acc_before, acc_after=acc_after,
                           acc=accs.mean(), lk_cost=lk_cost, lk_acc=lk_acc)
        rows.append(dict(t_pulse=tp, m4r_cost=cds.mean(), m4r_acc=accs.mean(),
                         frac_before_lure_ends=before.mean(), acc_if_before=acc_before,
                         acc_if_after=acc_after, leaky_cost=lk_cost, leaky_acc=lk_acc,
                         leaky_params=str(c)))

    print("\n  Lecture : le leurre MAJORITAIRE (26 capteurs a 1.0) domine la moyenne tant")
    print("  qu'il dure ; la verite MINORITAIRE (14 capteurs a 0.6) est presente DES t=0.")
    print("  Trancher juste avant la fin du leurre = extraire le signal minoritaire.")
    n_before = sum(summary[tp]["frac_before"] for tp in T_PULSE_LEVELS) / 2
    if n_before > 0.5:
        print(f"  -> M4R tranche avant la fin du leurre dans {100 * n_before:.0f}% des cas.")
    else:
        print(f"  -> M4R attend surtout la fin du leurre ({100 * n_before:.0f}% avant) :")
        print("     l'avantage de latence viendrait alors du seuil d'arret, pas d'une")
        print("     extraction precoce. A dire tel quel.")

    # ---- 2. ANCRAGE PHYSIQUE -------------------------------------------
    m4r_steps = float(np.mean([summary[tp]["m4r_cost"] for tp in T_PULSE_LEVELS]))
    lk_steps = float(np.mean([summary[tp]["lk_cost"] for tp in T_PULSE_LEVELS]))
    print(f"\n2. ANCRAGE PHYSIQUE (dt par famille, B3 section 2)")
    print(f"   M4R {m4r_steps:.0f} pas | filtre {lk_steps:.0f} pas | "
          f"rapport {lk_steps / m4r_steps:.2f}x")
    print("   Les deux observent le MEME flux, donc le meme dt : l'ancrage ne change pas")
    print("   le rapport, il donne l'echelle a laquelle il se joue.")
    print(f"{'famille':<24}{'dt/pas':>22}{'decision M4R':>22}{'decision filtre':>22}")
    print("-" * 90)
    for fam, (lo, hi) in DT_PHYS.items():
        for dt in (lo, hi):
            print(f"{fam:<24}{fmt_time(dt):>22}{fmt_time(m4r_steps * dt):>22}"
                  f"{fmt_time(lk_steps * dt):>22}")
            rows.append(dict(t_pulse="", m4r_cost=m4r_steps, m4r_acc="",
                             frac_before_lure_ends="", acc_if_before="", acc_if_after="",
                             leaky_cost=lk_steps, leaky_acc="",
                             leaky_params=f"{fam} dt={dt:.2e} "
                                          f"tM4R={m4r_steps * dt:.3e} tLK={lk_steps * dt:.3e}"))

    print("\n   Reserve, formulee avec soin car elle est facile a dire de travers :")
    print("   le rapport 4.4 est PUREMENT ALGORITHMIQUE -- les deux decideurs observent le")
    print("   meme flux au meme dt, aucun ne peut trancher avant d'avoir vu assez de signal,")
    print("   et 'assez' est fixe par la tache, pas par la technologie. L'ancrage ne fait")
    print("   qu'en donner l'echelle. Ce que le dispositif change vraiment est ailleurs :")
    print("   M4R impose un dt PLANCHER (il lui faut resoudre sa propre dynamique), donc il")
    print("   ne peut pas traiter un flux plus rapide que son substrat, alors qu'un RC")
    print("   s'accorde librement au rythme du flux. La contrainte pese sur M4R seul.")

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[csv] {CSV_PATH}")
    make_figure(summary, m4r_steps, lk_steps)
    print(f"[png] {PNG_PATH}")
    print(f"\nWall time: {time.time() - t0:.1f}s")
    return 0


def make_figure(summary, m4r_steps, lk_steps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    ax = axes[0]
    x = np.arange(len(T_PULSE_LEVELS))
    w = 0.35
    ax.bar(x - w / 2, [summary[tp]["m4r_cost"] for tp in T_PULSE_LEVELS], w,
           color="#d62728", edgecolor="k", label="M4R (doubt)")
    ax.bar(x + w / 2, [summary[tp]["lk_cost"] for tp in T_PULSE_LEVELS], w,
           color="#7b4173", edgecolor="k", label="forgetting filter")
    for i, tp in enumerate(T_PULSE_LEVELS):
        ax.hlines(tp, i - 0.5, i + 0.5, colors="k", linestyles="--", lw=1.5)
        ax.text(i, tp + 25, f"lure ends ({tp})", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"lure = {tp} steps" for tp in T_PULSE_LEVELS])
    ax.set_ylabel("steps before deciding (held-out seeds)")
    ax.set_title("Does the decision come before the lure ends?", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    fams = list(DT_PHYS)
    ypos = np.arange(len(fams))
    for i, fam in enumerate(fams):
        lo, hi = DT_PHYS[fam]
        ax.barh(i - 0.18, (hi - lo) * m4r_steps, left=lo * m4r_steps, height=0.32,
                color="#d62728", edgecolor="k")
        ax.barh(i + 0.18, (hi - lo) * lk_steps, left=lo * lk_steps, height=0.32,
                color="#7b4173", edgecolor="k")
    ax.set_yticks(ypos)
    ax.set_yticklabels(fams, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("absolute decision time (s, log) -- bar spans the dt range of B3")
    ax.set_title("Same 4.4x ratio, three orders of magnitude apart\n"
                 "(red = M4R, purple = filter)", fontsize=10)
    ax.grid(axis="x", alpha=0.3, which="both")

    fig.suptitle("Experiment B4 -- decision latency: the one quantity that survived the day, "
                 "anchored in physical time", fontsize=11)
    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=140)


if __name__ == "__main__":
    raise SystemExit(main())
