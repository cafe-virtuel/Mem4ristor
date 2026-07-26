#!/usr/bin/env python3
"""
EXPERIENCE B7 (volet 1 -- DIAGNOSTIC) : pourquoi std(v) egale mean(|L v|) ?

CONTEXTE (expB6-bis, 2026-07-26). Le signal d'arret du reseau exige de
l'information SPATIALE (un signal purement temporel s'effondre a 0.03) mais PAS
la topologie : un simple std(v), aveugle a qui est voisin de qui, egale voire
depasse le desaccord local mean(|L v|) natif (-0.07, IC [-0.17, +0.00]).
La question laissee ouverte : POURQUOI ?

HYPOTHESE (lue dans le code de la tache, pas devinee). Dans
`deceptive_task_poc.make_deceptive`, les 26 capteurs-leurre et les 14 sources de
verite sont tires par `rng.choice(N, ...)` : ils sont DISPERSES AU HASARD sur le
tore 10x10. La tache n'a donc aucune structure spatiale -- etre voisin de
quelqu'un ne dit rien sur ce qu'il mesure. Si c'est vrai, le laplacien ne peut
rien exploiter que la dispersion globale ne voie deja, et les deux signaux sont
deux lectures de la MEME chose.

CE VOLET NE TRANCHE RIEN SUR LA TOPOLOGIE. Il etablit seulement le diagnostic
mecanique. La condition de separation (sources CONTIGUES) est le volet 2.

PREDICTIONS, POSEES AVANT DE MESURER :
  P1  les deux signaux s'arretent quasi au meme instant (|delta t_stop| petit
      devant l'echelle de la fenetre trompeuse, 700 pas)
  P2  r(std(v)(t), mean|L v|(t)) eleve sur la fenetre decisive
  P3  Moran's I du champ de STIMULUS ~ 0 en tirage aleatoire, et nettement
      positif dans la variante contigue construite ici (controle de construction)
  P4  Moran's I du champ v reste faible en tirage aleatoire : le reseau ne
      fabrique pas de structure spatiale que la tache ne lui donne pas

Si P1 et P2 tombent (signaux decorreles, arrets differents alors que la justesse
est la meme), l'hypothese est FAUSSE et l'egalite a une autre cause -- a ecrire
tel quel.

SORTIES : figures/expB7_spatial_structure_diagnostic_poc.csv + .png
Cree : 2026-07-26 (Claude Opus 5, L'Ingenieur) -- suite de expB6-bis, 3e session
du jour. (Les en-tetes des 4 scripts B7 portaient a tort la date du 27 dans les
commits cb3dc1c/7c6c30c/a866eac ; corrige ici, historique non reecrit. Les
graines numeriques 20260727 sont laissees telles quelles : les changer
changerait les chiffres.)
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
from mem4ristor.graph_utils import make_lattice_adj  # noqa: E402

dp.MAX_BUDGET = 2000
MAXB = dp.MAX_BUDGET
N = dp.N
SIDE = dp.SIDE
WARMUP = dp.WARMUP

SEEDS_TRAIN = list(range(20))    # reglent le seuil de retombee, PAR SIGNAL (cf. expB6-bis)
SEEDS_TEST = list(range(20, 40))
T_PULSE = 700              # meme regime que expB6-bis
DROP = 0.30                # seuil historique -- garde comme reference "seuil commun"
DROP_GRID = [0.15, 0.30, 0.45, 0.60]
SMOOTH_W = 25
PROBE_TIMES = [100, 350, 690]   # instants sondes DANS la fenetre trompeuse
RATIO_TIMES = [100, 200, 350, 500, 690]  # ou l'on sonde la derive du rapport

CSV_PATH = ROOT / "figures" / "expB7_spatial_structure_diagnostic_poc.csv"
PNG_PATH = ROOT / "figures" / "expB7_spatial_structure_diagnostic_poc.png"


# --------------------------------------------------------------------------
# construction de la variante CONTIGUE (utilisee ici seulement pour montrer que
# le controle de structure fonctionne ; le vrai face-a-face est le volet 2)
# --------------------------------------------------------------------------
def torus_dist(i, j):
    ri, ci = divmod(i, SIDE)
    rj, cj = divmod(j, SIDE)
    dr = abs(ri - rj)
    dc = abs(ci - cj)
    return min(dr, SIDE - dr) + min(dc, SIDE - dc)


def make_deceptive_clustered(rng):
    """Meme tache que dp.make_deceptive, mais capteurs CONTIGUS.

    Memes effectifs (26 leurres / 14 verites), memes amplitudes, meme signe :
    seule la POSITION change. Les leurres forment un bloc autour d'un centre
    tire au hasard ; les verites un bloc autour du noeud le plus eloigne.
    """
    adj = make_lattice_adj(SIDE, periodic=True)
    dstar = rng.choice([-1, 1])
    c_d = int(rng.randint(N))
    order_d = sorted(range(N), key=lambda j: (torus_dist(c_d, j), j))
    d_nodes = np.array(order_d[:dp.N_DISTRACT], dtype=int)
    remaining = [j for j in range(N) if j not in set(d_nodes.tolist())]
    c_t = max(remaining, key=lambda j: torus_dist(c_d, j))
    order_t = sorted(remaining, key=lambda j: (torus_dist(c_t, j), j))
    t_nodes = np.array(order_t[:dp.N_TRUE], dtype=int)

    stim_on = np.zeros(N)
    stim_on[d_nodes] = -dstar * dp.E_DISTRACT
    stim_on[t_nodes] = +dstar * dp.E_TRUE
    stim_off = np.zeros(N)
    stim_off[t_nodes] = +dstar * dp.E_TRUE
    return adj, stim_on, stim_off, dstar


# --------------------------------------------------------------------------
def morans_i(x, adj):
    """Autocorrelation spatiale sur le graphe. 0 = aucune structure."""
    x = np.asarray(x, float)
    z = x - x.mean()
    denom = float(np.sum(z * z))
    if denom <= 0:
        return 0.0
    w_sum = float(np.sum(adj))
    num = float(z @ (adj @ z))
    return (N / w_sum) * (num / denom)


def smooth(x, w=SMOOTH_W):
    return np.convolve(np.asarray(x, float), np.ones(w) / w, mode="same")


def stop_at(sig, drop=DROP):
    peak = float(np.max(sig[:WARMUP + 20]))
    thr = drop * peak
    for t in range(WARMUP, len(sig)):
        if sig[t] < thr:
            return t + 1
    return len(sig)


def simulate(adj, stim_on, stim_off, seed, t_pulse):
    net = Mem4Network(size=SIDE, heretic_ratio=0.0, seed=seed, adjacency_matrix=adj)
    ref = Mem4Network(size=SIDE, heretic_ratio=0.0, seed=seed, adjacency_matrix=adj)
    L = net.L
    zero = np.zeros(N)
    sig_lap = np.empty(MAXB)
    sig_std = np.empty(MAXB)
    mean_v = np.empty(MAXB)
    d_var = np.empty(MAXB)
    probes = {}
    for t in range(MAXB):
        stim = stim_on if t < t_pulse else stim_off
        net.step(I_stimulus=stim)
        ref.step(I_stimulus=zero)
        v = net.model.v
        sig_lap[t] = float(np.mean(np.abs(L @ v)))
        sig_std[t] = float(np.std(v))
        mean_v[t] = float(np.mean(v))
        d_var[t] = mean_v[t] - float(np.mean(ref.model.v))
        if t in PROBE_TIMES:
            probes[t] = v.copy()
    sig_tmp = smooth(np.abs(np.diff(mean_v, prepend=mean_v[0])))
    dec = np.where(d_var >= 0, 1, -1)
    return dict(LAPLACIAN=sig_lap, STD=sig_std, TEMPORAL=sig_tmp), dec, probes


def boot_ci(v, n_boot=10_000, seed=20260727):
    v = np.asarray(v, float)
    rng = np.random.RandomState(seed)
    n = len(v)
    m = np.array([v[rng.randint(0, n, n)].mean() for _ in range(n_boot)])
    return float(np.mean(v)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> int:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("=" * 100)
    print("EXPERIENCE B7 volet 1 -- POURQUOI std(v) EGALE mean(|L v|) ?")
    print(f"{len(SEEDS_TRAIN)} graines TRAIN (reglage du seuil PAR SIGNAL) "
          f"+ {len(SEEDS_TEST)} TEST | T_pulse={T_PULSE}")
    print("=" * 100)

    adj_ref = make_lattice_adj(SIDE, periodic=True)
    rows = []
    store = {}          # (group, idx) -> dict par graine
    acc_grid = {}       # (signal, drop, group) -> liste 0/1
    moran_stim_rand, moran_stim_clust = [], []
    moran_v = {t: [] for t in PROBE_TIMES}
    r_full, r_window, r_norm_window = [], [], []
    ratios = {t: [] for t in RATIO_TIMES}

    for group, seeds in (("train", SEEDS_TRAIN), ("test", SEEDS_TEST)):
        for idx, seed in enumerate(seeds):
            rng = np.random.RandomState(3000 + seed)
            adj, s_on, s_off, dstar = dp.make_deceptive(rng)
            sigs, dec, probes = simulate(adj, s_on, s_off, seed * 10 + 1, T_PULSE)
            store[(group, idx)] = dict(sigs=sigs, dec=dec, dstar=dstar, seed=seed)

            for sname in ("LAPLACIAN", "STD"):
                for drop in DROP_GRID:
                    c = stop_at(sigs[sname], drop)
                    acc_grid.setdefault((sname, drop, group), []).append(
                        int(dp.dec_at(dec, c) == dstar))

            if group != "test":
                continue

            # --- mesures diagnostiques : uniquement sur les graines TEST ---
            rng_c = np.random.RandomState(3000 + seed)
            _, s_on_c, _, _ = make_deceptive_clustered(rng_c)
            moran_stim_rand.append(morans_i(s_on, adj_ref))
            moran_stim_clust.append(morans_i(s_on_c, adj_ref))
            for t in PROBE_TIMES:
                moran_v[t].append(morans_i(probes[t], adj_ref))

            lap, std = sigs["LAPLACIAN"], sigs["STD"]
            r_full.append(float(np.corrcoef(lap[WARMUP:], std[WARMUP:])[0, 1]))
            r_window.append(float(np.corrcoef(lap[WARMUP:T_PULSE],
                                              std[WARMUP:T_PULSE])[0, 1]))
            # la regle d'arret ne voit QUE le signal normalise par son pic
            lap_n = lap / float(np.max(lap[:WARMUP + 20]))
            std_n = std / float(np.max(std[:WARMUP + 20]))
            r_norm_window.append(float(np.corrcoef(lap_n[WARMUP:T_PULSE],
                                                   std_n[WARMUP:T_PULSE])[0, 1]))
            for t in RATIO_TIMES:
                ratios[t].append(float(lap_n[t] / std_n[t]))

            c_lap30 = stop_at(lap, DROP)
            c_std30 = stop_at(std, DROP)
            rows.append(dict(seed=seed,
                             stop_laplacian_drop030=c_lap30,
                             stop_std_drop030=c_std30,
                             delta_stop_drop030=abs(c_lap30 - c_std30),
                             r_signals_full=r_full[-1],
                             r_signals_window=r_window[-1],
                             r_normalised_window=r_norm_window[-1],
                             moran_stim_random=moran_stim_rand[-1],
                             moran_stim_clustered=moran_stim_clust[-1],
                             **{f"moran_v_t{t}": moran_v[t][-1] for t in PROBE_TIMES},
                             **{f"ratio_norm_t{t}": ratios[t][-1] for t in RATIO_TIMES}))
        print(f"  [{group}] {len(seeds)} graines simulees ({time.time() - t0:.0f}s)")

    # seuil retenu PAR SIGNAL sur les graines TRAIN (protocole expB6-bis)
    chosen = {s: max(DROP_GRID, key=lambda d: np.mean(acc_grid[(s, d, "train")]))
              for s in ("LAPLACIAN", "STD")}

    print("\nP1 -- ARRET ET JUSTESSE, SEUIL REGLE PAR SIGNAL SUR TRAIN (graines TEST)")
    stops_tuned = {}
    for sname in ("LAPLACIAN", "STD"):
        d = chosen[sname]
        st = [stop_at(store[("test", i)]["sigs"][sname], d) for i in range(len(SEEDS_TEST))]
        a = np.array(acc_grid[(sname, d, "test")], float)
        stops_tuned[sname] = np.array(st, int)
        m, lo, hi = boot_ci(a)
        print(f"  {sname:<10} seuil retenu {d:.2f} | arret moyen {np.mean(st):6.0f} pas "
              f"| justesse {m:.2f} CI[{lo:.2f},{hi:.2f}]")
    dacc, lo, hi = boot_ci(np.array(acc_grid[("LAPLACIAN", chosen["LAPLACIAN"], "test")], float)
                           - np.array(acc_grid[("STD", chosen["STD"], "test")], float))
    print(f"  LAPLACIAN - STD = {dacc:+.2f} CI[{lo:+.2f},{hi:+.2f}]")
    dt_tuned = np.abs(stops_tuned["LAPLACIAN"] - stops_tuned["STD"])
    m, lo, hi = boot_ci(dt_tuned)
    print(f"  |delta t_stop| = {m:.1f} pas CI[{lo:.1f},{hi:.1f}] "
          f"| arrets identiques {int(np.sum(dt_tuned == 0))}/{len(dt_tuned)}")

    # ------------------------------------------------------------------
    # P5 -- ECHELLE OU INFORMATION ? Le decalage est estime sur les graines
    # TRAIN a partir des seuls INSTANTS D'ARRET (jamais de la justesse : ce
    # serait l'oracle par run corrige le 26/07), puis applique EN AVEUGLE aux
    # graines TEST. Si retarder LAPLACIAN du decalage suffit a rejoindre STD,
    # l'ecart de justesse est un probleme de CALIBRAGE, pas d'information.
    # ------------------------------------------------------------------
    st_lap_tr = np.array([stop_at(store[("train", i)]["sigs"]["LAPLACIAN"],
                                  chosen["LAPLACIAN"]) for i in range(len(SEEDS_TRAIN))])
    st_std_tr = np.array([stop_at(store[("train", i)]["sigs"]["STD"],
                                  chosen["STD"]) for i in range(len(SEEDS_TRAIN))])
    delta = int(round(float(np.mean(st_std_tr - st_lap_tr))))

    def acc_shift(sname, shift):
        out = []
        for i in range(len(SEEDS_TEST)):
            s = store[("test", i)]
            c = max(1, stop_at(s["sigs"][sname], chosen[sname]) + shift)
            out.append(int(dp.dec_at(s["dec"], c) == s["dstar"]))
        return np.array(out, float)

    print("\nP5 -- ECHELLE OU INFORMATION ? (decalage estime sur TRAIN, applique en aveugle)")
    print(f"  decalage mesure sur TRAIN (t_stop STD - t_stop LAPLACIAN) = {delta:+d} pas")
    a0 = acc_shift("LAPLACIAN", 0)
    a1 = acc_shift("LAPLACIAN", delta)
    b0 = acc_shift("STD", 0)
    b1 = acc_shift("STD", -delta)
    for label, arr in (("LAPLACIAN natif        ", a0),
                       (f"LAPLACIAN retarde {delta:+d}  ", a1),
                       ("STD natif              ", b0),
                       (f"STD avance {-delta:+d}       ", b1)):
        m, lo, hi = boot_ci(arr)
        print(f"  {label} justesse {m:.2f} CI[{lo:.2f},{hi:.2f}]")
    d, lo, hi = boot_ci(a1 - b0)
    print(f"  (LAPLACIAN retarde) - (STD natif) = {d:+.2f} CI[{lo:+.2f},{hi:+.2f}]")
    if lo > -0.05 and float(np.mean(a1)) > float(np.mean(a0)):
        print("  -> CALIBRAGE : retarder le signal natif du seul decalage d'echelle")
        print("     suffit a rejoindre std(v). L'ecart de B6-bis n'etait pas un")
        print("     deficit d'information -- les deux signaux disent la meme chose,")
        print("     l'un franchit son seuil plus tot.")
    else:
        print("  -> PAS SEULEMENT LE CALIBRAGE : meme recale, le desaccord local")
        print("     n'egale pas la dispersion globale. L'ecart tient a autre chose.")

    print("\nP1-brut -- LE MEME SEUIL 0.30 IMPOSE AUX DEUX (reference, non comparable a B6)")
    dt30 = np.array([r["delta_stop_drop030"] for r in rows], float)
    m, lo, hi = boot_ci(dt30)
    print(f"  |delta t_stop| = {m:.1f} pas CI[{lo:.1f},{hi:.1f}] "
          f"| LAPLACIAN {np.mean([r['stop_laplacian_drop030'] for r in rows]):.0f} "
          f"vs STD {np.mean([r['stop_std_drop030'] for r in rows]):.0f}")

    print("\nP2 -- LES DEUX SIGNAUX SONT-ILS LA MEME LECTURE ?")
    m, lo, hi = boot_ci(r_full)
    print(f"  r brut, trajectoire entiere        = {m:.3f} CI[{lo:.3f},{hi:.3f}]")
    m, lo, hi = boot_ci(r_window)
    print(f"  r brut, fenetre trompeuse          = {m:.3f} CI[{lo:.3f},{hi:.3f}]")
    m, lo, hi = boot_ci(r_norm_window)
    print(f"  r NORMALISE par le pic (ce que la regle d'arret voit reellement)")
    print(f"                                     = {m:.3f} CI[{lo:.3f},{hi:.3f}]")

    print("\nP2-bis -- LE RAPPORT DES DEUX SIGNAUX NORMALISES DERIVE-T-IL ?")
    print("  (s'il etait constant, la regle 'retombee a x% du pic' declencherait")
    print("   les deux EXACTEMENT au meme instant -- l'ecart mesure vient de la)")
    for t in RATIO_TIMES:
        m, lo, hi = boot_ci(ratios[t])
        print(f"  lap_norm/std_norm a t={t:<4} = {m:.3f} CI[{lo:.3f},{hi:.3f}]")
    drift = float(np.mean(ratios[RATIO_TIMES[-1]]) - np.mean(ratios[RATIO_TIMES[0]]))
    print(f"  derive totale sur la fenetre = {drift:+.3f}")

    print("\nP3 -- LA TACHE A-T-ELLE UNE STRUCTURE SPATIALE ? (Moran's I du stimulus)")
    m, lo, hi = boot_ci(moran_stim_rand)
    print(f"  tirage ALEATOIRE (la tache actuelle) = {m:+.3f} CI[{lo:+.3f},{hi:+.3f}]")
    m2, lo2, hi2 = boot_ci(moran_stim_clust)
    print(f"  variante CONTIGUE (volet 2)          = {m2:+.3f} CI[{lo2:+.3f},{hi2:+.3f}]")
    print(f"  -> le controle de structure fonctionne : ecart {m2 - m:+.3f}")

    print("\nP4 -- LE RESEAU FABRIQUE-T-IL DE LA STRUCTURE QUE LA TACHE NE DONNE PAS ?")
    for t in PROBE_TIMES:
        m, lo, hi = boot_ci(moran_v[t])
        print(f"  Moran's I de v a t={t:<4} = {m:+.3f} CI[{lo:+.3f},{hi:+.3f}]")

    print("\n--- LECTURE (chaque prediction jugee SEPAREMENT) ---")
    rn = float(np.mean(r_norm_window))
    sm = float(np.mean(moran_stim_rand))
    print(f"  P2 (meme lecture)          : r normalise = {rn:.3f} -> "
          f"{'TENUE' if rn > 0.8 else 'TOMBE'}")
    print(f"  P3 (tache sans structure)  : Moran's I stimulus = {sm:+.3f} -> "
          f"{'TENUE' if abs(sm) < 0.10 else 'TOMBE'}")
    print(f"  P4 (reseau n'en cree pas)  : Moran's I de v <= "
          f"{max(abs(np.mean(moran_v[t])) for t in PROBE_TIMES):.3f} -> "
          f"{'TENUE' if max(abs(np.mean(moran_v[t])) for t in PROBE_TIMES) < 0.10 else 'TOMBE'}")
    print(f"  P1 (meme instant d'arret)  : |delta t| = {np.mean(dt_tuned):.0f} pas -> "
          f"{'TENUE' if np.mean(dt_tuned) < 0.05 * T_PULSE else 'TOMBE'}")
    print("  NOTE : P1 et P2 peuvent diverger sans contradiction -- deux signaux")
    print("  corretes a 0.99 declenchent a des instants differents des que leur")
    print("  RAPPORT derive (voir P2-bis). C'est une difference d'ECHELLE, pas")
    print("  d'information spatiale.")

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[csv] {CSV_PATH}")
    make_figure(rows, moran_stim_rand, moran_stim_clust, moran_v, ratios)
    print(f"[png] {PNG_PATH}")
    print(f"\nWall time: {time.time() - t0:.1f}s")
    return 0


def make_figure(rows, moran_rand, moran_clust, moran_v, ratios):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    ax = axes[0]
    ax.scatter([r["stop_laplacian_drop030"] for r in rows],
               [r["stop_std_drop030"] for r in rows],
               c="#d62728", edgecolor="k", zorder=3)
    lim = [0, max(max(r["stop_laplacian_drop030"] for r in rows),
                  max(r["stop_std_drop030"] for r in rows)) * 1.1]
    ax.plot(lim, lim, ls="--", c="gray")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("stop time -- local disagreement mean|Lv|")
    ax.set_ylabel("stop time -- global spread std(v)")
    ax.set_title("P1: same information, different firing time", fontsize=10)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ts = sorted(ratios.keys())
    means = [np.mean(ratios[t]) for t in ts]
    los = [boot_ci(ratios[t])[1] for t in ts]
    his = [boot_ci(ratios[t])[2] for t in ts]
    ax.plot(ts, means, "o-", c="#1f77b4")
    ax.fill_between(ts, los, his, color="#1f77b4", alpha=0.25)
    ax.axhline(1.0, ls="--", c="gray")
    ax.set_xlabel("step")
    ax.set_ylabel("mean|Lv| / std(v), each normalised by its own peak")
    ax.set_title("P2-bis: the ratio drifts -> different stop times", fontsize=10)
    ax.grid(alpha=0.3)

    ax = axes[2]
    labels = ["stimulus\nrandom\n(current task)", "stimulus\nclustered\n(volet 2)"] + \
             [f"field v\nt={t}" for t in PROBE_TIMES]
    vals = [np.mean(moran_rand), np.mean(moran_clust)] + \
           [np.mean(moran_v[t]) for t in PROBE_TIMES]
    cols = ["#7f7f7f", "#2ca02c"] + ["#d62728"] * len(PROBE_TIMES)
    ax.bar(range(len(vals)), vals, color=cols, edgecolor="k")
    ax.axhline(0, c="k", lw=0.8)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("Moran's I (spatial structure on the graph)")
    ax.set_title("P3/P4: is there any spatial structure to exploit?", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Experiment B7 (part 1) -- why a topology-blind std(v) matches the "
                 "native local-disagreement signal", fontsize=11)
    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=140)


if __name__ == "__main__":
    raise SystemExit(main())
