#!/usr/bin/env python3
"""
EXPERIENCE B6-bis -- QU'EST-CE QUI PORTE L'INFORMATION D'ARRET ?
(tranche les deux conditions laissees ouvertes par expB5-bis)

CONTEXTE (expB5-bis, 2026-07-26). Sur la niche trompeuse, le reseau ne detient
pas la verite minoritaire en permanence -- a instant fixe il est a 0.10, sous le
hasard -- mais a l'instant que son doute choisit il est a 0.90, contre 0.46 pour
des instants de meme distribution tires sans lien au run. L'arret est donc
INFORME. Restaient deux questions explicitement notees comme non tranchees :

  (i)  le signal d'arret natif est mean(|L v|), une lecture du DESACCORD LOCAL.
       Meme dans la condition DECOUPLE (D=0), cette lecture spatiale subsiste --
       c'est pourquoi "le couplage ne sert a rien" ne se deduisait PAS du
       resultat. Il faut un arret qui n'ait acces a AUCUNE information spatiale
       pour separer ce qui vient de la topologie de ce qui vient d'ailleurs.
  (ii) les comparaisons FULL vs DECOUPLE (+0.00 CI[-0.20,+0.20]) et FULL vs
       FROZEN_U (+0.15 CI[-0.10,+0.40]) n'etaient pas tranchees a 20 graines.

Les deux se traitent d'un coup, parce que les trois signaux d'arret se calculent
depuis la MEME simulation : le seul cout supplementaire est le nombre de graines.

TROIS SIGNAUX D'ARRET, a information spatiale DECROISSANTE :
    LAPLACIAN  mean(|L v|)          desaccord LOCAL -- le signal natif, connait
                                    la topologie (qui est voisin de qui)
    STD        std(v)               dispersion GLOBALE -- spatial mais AVEUGLE a
                                    la topologie : permuter les noeuds ne change
                                    rien
    TEMPORAL   |d mean(v)/dt| lisse  PUREMENT TEMPOREL -- aucune information
                                    spatiale d'aucune sorte
Chacun est passe dans la MEME regle d'arret (retombee sous une fraction du pic,
la regle de `dp.stop_doubt`), et cette fraction est reglee PAR SIGNAL sur les
graines d'entrainement puis figee -- sinon le seuil historique 0.30, choisi pour
le signal natif, avantagerait le natif par construction.

CE QUE CHAQUE ISSUE SIGNIFIERAIT, pose avant de mesurer :
  - TEMPORAL suffit      -> l'arret informe ne doit RIEN au spatial ; c'est une
                            propriete de la dynamique temporelle du reseau.
  - STD suffit, pas TEMPORAL -> il faut du spatial, mais pas la topologie.
  - seul LAPLACIAN marche -> le desaccord LOCAL est le mecanisme, et la
                            structure de voisinage compte vraiment.

SORTIES : figures/expB6_stop_signal_ablation_poc.csv + .png
Cree : 2026-07-26 (Claude Opus 5, L'Ingenieur) -- suite de expB5-bis.
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

SEEDS_TRAIN = list(range(20))          # reglent le seuil de retombee, par signal
SEEDS_TEST = list(range(20, 60))       # 40 graines : le double de expB5-bis
T_PULSE = 700
CONDITIONS = ["FULL", "DECOUPLE", "FROZEN_U", "DECOUPLE_FROZEN"]
STOP_SIGNALS = ["LAPLACIAN", "STD", "TEMPORAL"]
DROP_GRID = [0.15, 0.30, 0.45, 0.60]   # fraction du pic ; 0.30 = valeur historique
SMOOTH_W = 25
WARMUP = dp.WARMUP
N_BOOT = 10_000
RNG_BOOT = np.random.RandomState(20260726)

CSV_PATH = ROOT / "figures" / "expB6_stop_signal_ablation_poc.csv"
PNG_PATH = ROOT / "figures" / "expB6_stop_signal_ablation_poc.png"


def make_net(adj, seed, condition):
    net = Mem4Network(size=SIDE, heretic_ratio=0.0, seed=seed, adjacency_matrix=adj)
    if "DECOUPLE" in condition:
        net.model.cfg["coupling"]["D"] = 0.0
        net.model.D_eff = 0.0
    if "FROZEN" in condition:
        net.model.cfg["doubt"]["epsilon_u"] = 0.0
    return net


def smooth(x, w=SMOOTH_W):
    return np.convolve(np.asarray(x, float), np.ones(w) / w, mode="same")


def simulate(adj, stim_on, stim_off, seed, t_pulse, condition):
    net = make_net(adj, seed, condition)
    ref = make_net(adj, seed, condition)
    L = net.L
    zero = np.zeros(N)
    sig_lap = np.empty(MAXB)
    sig_std = np.empty(MAXB)
    mean_v = np.empty(MAXB)
    d_var = np.empty(MAXB)
    for t in range(MAXB):
        stim = stim_on if t < t_pulse else stim_off
        net.step(I_stimulus=stim)
        ref.step(I_stimulus=zero)
        v = net.model.v
        sig_lap[t] = float(np.mean(np.abs(L @ v)))     # desaccord LOCAL (topologie)
        sig_std[t] = float(np.std(v))                  # dispersion GLOBALE
        mean_v[t] = float(np.mean(v))
        d_var[t] = mean_v[t] - float(np.mean(ref.model.v))
    # purement temporel : amplitude de variation de la moyenne, lissee
    sig_tmp = smooth(np.abs(np.diff(mean_v, prepend=mean_v[0])))
    return dict(LAPLACIAN=sig_lap, STD=sig_std, TEMPORAL=sig_tmp), d_var


def stop_at(sig, drop):
    """Regle de dp.stop_doubt, avec la fraction de retombee en parametre."""
    peak = float(np.max(sig[:WARMUP + 20]))
    thr = drop * peak
    for t in range(WARMUP, len(sig)):
        if sig[t] < thr:
            return t + 1
    return len(sig)


def boot_ci(v):
    v = np.asarray(v, float)
    n = len(v)
    m = np.array([v[RNG_BOOT.randint(0, n, n)].mean() for _ in range(N_BOOT)])
    return float(np.mean(v)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> int:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    # store[(cond, signal, drop)][group] = liste (acc, stop) par graine
    acc = {}
    stops = {}
    decs = {}
    dstars = {}
    always = {}
    print("=" * 100)
    print("EXPERIENCE B6-bis -- qu'est-ce qui porte l'information d'arret ?")
    print(f"{len(SEEDS_TRAIN)} graines TRAIN (reglage du seuil) + {len(SEEDS_TEST)} TEST "
          f"| {len(CONDITIONS)} conditions x {len(STOP_SIGNALS)} signaux")
    print("=" * 100)

    for group, seeds in (("train", SEEDS_TRAIN), ("test", SEEDS_TEST)):
        for si, seed in enumerate(seeds):
            rng = np.random.RandomState(3000 + seed)
            adj, s_on, s_off, dstar = dp.make_deceptive(rng)
            dstars[(group, si)] = dstar
            for cond in CONDITIONS:
                sigs, d_var = simulate(adj, s_on, s_off, seed * 10 + 1, T_PULSE, cond)
                dec = np.where(d_var >= 0, 1, -1)
                decs[(group, si, cond)] = dec
                # JUSTESSE TEMPORELLE MOYENNE : fraction des instants de la fenetre
                # trompeuse ou la decision courante est deja correcte, INDEPENDAMMENT
                # de tout arret.
                # Ajoutee pour verifier un soupcon ne d'un chiffre du premier run : le
                # null permute vaut 0.94 en FROZEN_U contre 0.55 en FULL, ce qui
                # semblait dire que le doute DEGRADE la lecture instantanee et que son
                # arret ne fait que rattraper. MESURE -> SOUPCON FAUX : la justesse a
                # tout instant est ~0.40 dans les QUATRE conditions (FULL - FROZEN_U =
                # +0.01, IC couvre 0). L'ecart des nulls venait d'un DEFAUT DE LA
                # BASELINE, pas d'un effet : les arrets de FROZEN_U sont bien plus
                # disperses (jusqu'a 1530) et tombent donc souvent APRES la fin du
                # leurre (700), la ou la decision est facile. Le null permute n'est
                # comparable qu'entre conditions de meme distribution d'arrets --
                # limite a retenir si on le reutilise.
                always.setdefault((cond, group), []).append(
                    float(np.mean(dec[WARMUP:T_PULSE] == dstar)))
                for sname in STOP_SIGNALS:
                    for drop in DROP_GRID:
                        c = stop_at(sigs[sname], drop)
                        acc.setdefault((cond, sname, drop, group), []).append(
                            int(dp.dec_at(dec, c) == dstar))
                        stops.setdefault((cond, sname, drop, group), []).append(c)
        print(f"  [{group}] {len(seeds)} graines simulees "
              f"({time.time() - t0:.0f}s)")

    # seuil choisi sur TRAIN, par (condition, signal)
    chosen = {}
    for cond in CONDITIONS:
        for sname in STOP_SIGNALS:
            chosen[(cond, sname)] = max(
                DROP_GRID, key=lambda d: np.mean(acc[(cond, sname, d, "train")]))

    print("\nRESULTATS SUR GRAINES TEST (seuil de retombee regle sur TRAIN, par signal)")
    print(f"{'condition':<18}" + "".join(f"{s:>22}" for s in STOP_SIGNALS))
    print(f"{'':<18}" + "".join(f"{'acc (arret permute)':>22}" for _ in STOP_SIGNALS))
    print("-" * 86)
    rng_null = np.random.RandomState(4242)
    table, nulls = {}, {}
    for cond in CONDITIONS:
        line = f"{cond:<18}"
        for sname in STOP_SIGNALS:
            d = chosen[(cond, sname)]
            a = np.array(acc[(cond, sname, d, "test")], float)
            st = np.array(stops[(cond, sname, d, "test")], int)
            null = []
            for _ in range(200):
                perm = rng_null.permutation(len(st))
                null.append(np.mean([
                    decs[("test", i, cond)][min(st[perm[i]], MAXB) - 1] == dstars[("test", i)]
                    for i in range(len(st))]))
            table[(cond, sname)] = a
            nulls[(cond, sname)] = float(np.mean(null))
            line += f"{a.mean():>13.2f} ({np.mean(null):.2f})"
        print(line)

    print("\n(i) QU'EST-CE QUI PORTE L'INFORMATION D'ARRET ? (condition FULL, 40 graines)")
    for sname in STOP_SIGNALS:
        a = table[("FULL", sname)]
        m, lo, hi = boot_ci(a)
        print(f"  {sname:<10} acc={m:.2f} CI[{lo:.2f},{hi:.2f}]  "
              f"vs arret permute {nulls[('FULL', sname)]:.2f}  "
              f"(seuil retenu {chosen[('FULL', sname)]})")
    d_ls, lo_ls, hi_ls = boot_ci(table[("FULL", "LAPLACIAN")] - table[("FULL", "STD")])
    d_lt, lo_lt, hi_lt = boot_ci(table[("FULL", "LAPLACIAN")] - table[("FULL", "TEMPORAL")])
    print(f"  LAPLACIAN - STD      = {d_ls:+.2f} CI[{lo_ls:+.2f},{hi_ls:+.2f}]")
    print(f"  LAPLACIAN - TEMPORAL = {d_lt:+.2f} CI[{lo_lt:+.2f},{hi_lt:+.2f}]")
    if lo_lt > 0 and lo_ls > 0:
        print("  -> le DESACCORD LOCAL est le mecanisme : ni la dispersion globale ni le")
        print("     purement temporel ne le remplacent. La topologie compte.")
    elif lo_lt > 0:
        print("  -> il faut du SPATIAL (le purement temporel echoue), mais la dispersion")
        print("     globale suffit : la topologie de voisinage n'est pas necessaire.")
    else:
        print("  -> l'arret informe ne doit RIEN au spatial : un signal purement temporel")
        print("     fait aussi bien. C'est une propriete de la dynamique, pas du reseau.")

    print("\nJUSTESSE TEMPORELLE MOYENNE -- fraction des instants de la fenetre trompeuse")
    print("ou la decision est DEJA correcte, sans aucun arret (40 graines TEST) :")
    for cond in CONDITIONS:
        a = np.array(always[(cond, "test")], float)
        m, lo, hi = boot_ci(a)
        best = max(table[(cond, s2)].mean() for s2 in STOP_SIGNALS)
        print(f"  {cond:<18} justesse a tout instant = {m:.2f} CI[{lo:.2f},{hi:.2f}]   "
              f"| meilleur arret = {best:.2f}   | gain de l'arret = {best - m:+.2f}")
    d, lo, hi = boot_ci(np.array(always[("FULL", "test")], float)
                        - np.array(always[("FROZEN_U", "test")], float))
    print(f"  FULL - FROZEN_U sur la justesse a tout instant = {d:+.2f} "
          f"CI[{lo:+.2f},{hi:+.2f}]")
    if hi < 0:
        print("  -> LE DOUTE DEGRADE LA LECTURE INSTANTANEE : gele, le reseau est deja")
        print("     correct la plupart du temps. Le signal d'arret ne fait alors que")
        print("     RATTRAPER une instabilite que le doute a lui-meme introduite.")
        print("     A rapporter tel quel : c'est un cout du doute, pas un benefice.")
    elif lo > 0:
        print("  -> le doute AMELIORE aussi la lecture instantanee, pas seulement l'arret.")
    else:
        print("  -> non tranche sur la justesse a tout instant.")

    print("\n(ii) LES DEUX COMPARAISONS NON TRANCHEES A 20 GRAINES, REPRISES A 40")
    for other in ("DECOUPLE", "FROZEN_U"):
        d, lo, hi = boot_ci(table[("FULL", "LAPLACIAN")] - table[(other, "LAPLACIAN")])
        verdict = ("FULL superieur" if lo > 0 else "FULL inferieur" if hi < 0
                   else "toujours non tranche")
        print(f"  FULL - {other:<16} = {d:+.2f} CI[{lo:+.2f},{hi:+.2f}]  -> {verdict}")

    rows = []
    for cond in CONDITIONS:
        for sname in STOP_SIGNALS:
            rows.append(dict(condition=cond, stop_signal=sname,
                             drop_threshold=chosen[(cond, sname)],
                             acc_test=float(table[(cond, sname)].mean()),
                             acc_permuted_null=nulls[(cond, sname)],
                             n_seeds_test=len(SEEDS_TEST)))
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[csv] {CSV_PATH}")
    make_figure(table, nulls)
    print(f"[png] {PNG_PATH}")
    print(f"\nWall time: {time.time() - t0:.1f}s")
    return 0


def make_figure(table, nulls):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    colors = {"LAPLACIAN": "#d62728", "STD": "#1f77b4", "TEMPORAL": "#7f7f7f"}

    ax = axes[0]
    xs = np.arange(len(STOP_SIGNALS))
    vals = [table[("FULL", s)].mean() for s in STOP_SIGNALS]
    errs = [[vals[i] - boot_ci(table[("FULL", s)])[1] for i, s in enumerate(STOP_SIGNALS)],
            [boot_ci(table[("FULL", s)])[2] - vals[i] for i, s in enumerate(STOP_SIGNALS)]]
    ax.bar(xs - 0.19, vals, 0.38, yerr=errs, capsize=4,
           color=[colors[s] for s in STOP_SIGNALS], edgecolor="k", label="native stop")
    ax.bar(xs + 0.19, [nulls[("FULL", s)] for s in STOP_SIGNALS], 0.38,
           color="white", edgecolor="k", hatch="///", label="permuted stop times")
    ax.axhline(0.5, ls=":", c="gray")
    ax.set_xticks(xs)
    ax.set_xticklabels(["local disagreement\nmean|Lv|\n(knows topology)",
                        "global spread\nstd(v)\n(blind to topology)",
                        "purely temporal\n|d mean(v)/dt|\n(no spatial info)"], fontsize=7.5)
    ax.set_ylabel("accuracy at stop (40 held-out seeds)")
    ax.set_ylim(0, 1.05)
    ax.set_title("What carries the stopping information?", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    xs = np.arange(len(CONDITIONS))
    for i, s in enumerate(STOP_SIGNALS):
        ax.bar(xs + (i - 1) * 0.27, [table[(c, s)].mean() for c in CONDITIONS], 0.26,
               color=colors[s], edgecolor="k", label=s)
    ax.axhline(0.5, ls=":", c="gray")
    ax.set_xticks(xs)
    ax.set_xticklabels([c.replace("_", "\n") for c in CONDITIONS], fontsize=8)
    ax.set_ylabel("accuracy at stop")
    ax.set_ylim(0, 1.05)
    ax.set_title("Ablation x stopping signal, 40 held-out seeds", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Experiment B6-bis -- separating what the stop signal needs: "
                 "topology, spatial spread, or neither", fontsize=11)
    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=140)


if __name__ == "__main__":
    raise SystemExit(main())
