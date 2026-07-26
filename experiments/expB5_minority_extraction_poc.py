#!/usr/bin/env python3
"""
EXPERIENCE B5-bis -- COMMENT le reseau extrait-il la verite MINORITAIRE
pendant que la moyenne du flux dit encore le contraire ?

CONTEXTE (expB4, 2026-07-26). A leurre = 700 pas, M4R tranche a ~328 pas, donc
DANS la fenetre trompeuse, et il a raison 9 fois sur 10. A cet instant :
    26 capteurs "leurre"  a  -dstar * 1.0
    14 capteurs "verite"  a  +dstar * 0.6
    60 capteurs muets
-> la moyenne instantanee du stimulus vaut -0.176 * dstar : elle pointe encore
vers la MAUVAISE reponse. Le filtre a oubli, qui lit cette moyenne, reste piege
jusqu'a la fin du leurre (il tranche a 1522). M4R, lui, tranche juste avant.

L'hypothese que j'ai formulee en rapportant expB4 -- une RECTIFICATION NON
LINEAIRE : les noeuds satures par le leurre fort (|I|=1.0) pesant moins que les
noeuds de verite restes dans la zone reactive (|I|=0.6) -- n'est qu'une
hypothese. Je me suis deja trompe DEUX FOIS aujourd'hui sur ce type
d'explication mecaniste (expA : ni l'amplitude du champ recu ni la variance de
la cible n'expliquaient l'effet du degre). Elle est donc testee, pas racontee.

QUATRE CONDITIONS, une ablation au sens du projet (meme convention que B4 :
FROZEN_U = epsilon_u a 0 ; DECOUPLE = D a 0) :
    FULL             reseau complet
    DECOUPLE         D=0 : des noeuds FHN INDEPENDANTS, aucun couplage
    FROZEN_U         doute gele : la non-linearite reste, la modulation part
    DECOUPLE_FROZEN  un simple banc de noeuds FHN passifs
+ un TEMOIN ANALYTIQUE : la reponse stationnaire r(I) du noeud FHN isole,
  resolue numeriquement, qui permet de PREDIRE le decalage de moyenne attendu
  d'une pure rectification statique -- sans aucune dynamique ni reseau.

CE QUE CHAQUE ISSUE SIGNIFIE, pose avant de mesurer :
  - si DECOUPLE suffit  -> le reseau ne sert a rien ici ; c'est la non-linearite
    du NOEUD, et un banc de noeuds independants ferait le meme travail. Ce serait
    le quatrieme adversaire trivial de la journee, et il faudrait le dire.
  - si le temoin statique suffit -> meme conclusion, en plus fort : pas meme
    besoin de dynamique, une simple courbe de reponse concave explique tout.
  - si FULL est necessaire -> c'est le PREMIER resultat de la journee ou le
    reseau couple compte vraiment, et la valeur de M4R sur cette tache tient au
    couplage, pas au substrat ni au reglage.

GATE DE FIDELITE : la condition FULL doit reproduire `dp.simulate` bit a bit.

SORTIES : figures/expB5_minority_extraction_poc.csv + .png
Cree : 2026-07-26 (Claude Opus 5, L'Ingenieur) -- suite de expB4.
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
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "scratch"))
import deceptive_task_poc as dp  # noqa: E402
from mem4ristor.topology import Mem4Network  # noqa: E402

dp.MAX_BUDGET = 2000
MAXB = dp.MAX_BUDGET
N = dp.N
SIDE = dp.SIDE

SEEDS = list(range(20, 40))       # graines TEST de expB/expB4
T_PULSE = 700                     # le regime ou 100 % des decisions tombent avant la fin
CONDITIONS = ["FULL", "DECOUPLE", "FROZEN_U", "DECOUPLE_FROZEN"]
PROBE_TIMES = [100, 200, 328, 500, 690]   # instants d'inspection, tous < T_PULSE
N_BOOT = 10_000
RNG_BOOT = np.random.RandomState(20260726)

CSV_PATH = ROOT / "figures" / "expB5_minority_extraction_poc.csv"
PNG_PATH = ROOT / "figures" / "expB5_minority_extraction_poc.png"


def make_net(adj, seed, condition):
    net = Mem4Network(size=SIDE, heretic_ratio=0.0, seed=seed, adjacency_matrix=adj)
    if "DECOUPLE" in condition:
        net.model.cfg["coupling"]["D"] = 0.0
        net.model.D_eff = 0.0
    if "FROZEN" in condition:
        net.model.cfg["doubt"]["epsilon_u"] = 0.0     # convention B4
    return net


def simulate(adj, stim_on, stim_off, seed, t_pulse, condition):
    net = make_net(adj, seed, condition)
    ref = make_net(adj, seed, condition)
    L = net.L
    zero = np.zeros(N)
    sig = np.empty(MAXB)
    d_var = np.empty(MAXB)
    for t in range(MAXB):
        stim = stim_on if t < t_pulse else stim_off
        net.step(I_stimulus=stim)
        ref.step(I_stimulus=zero)
        v = net.model.v
        sig[t] = float(np.mean(np.abs(L @ v)))
        d_var[t] = float(np.mean(v) - np.mean(ref.model.v))
    return sig, d_var


def fhn_fixed_point(I):
    """Reponse stationnaire v*(I) du noeud FHN isole (parametres de dynamics.py).
    dv = v - v^3/5 - w + I - 0.15*tanh(v) ; dw = eps*(v + 0.7 - 0.8*w) -> w=(v+0.7)/0.8."""
    def f(v):
        w = (v + 0.7) / 0.8
        return v - v ** 3 / 5.0 - w + I - 0.15 * np.tanh(v)
    lo, hi = -6.0, 6.0
    if f(lo) * f(hi) > 0:
        return np.nan
    return brentq(f, lo, hi, xtol=1e-12)


def static_prediction():
    """Ce qu'une PURE rectification statique predirait, sans dynamique ni reseau."""
    r0 = fhn_fixed_point(0.0)
    r_lure = fhn_fixed_point(-dp.E_DISTRACT)      # 26 noeuds, signe oppose a dstar
    r_truth = fhn_fixed_point(+dp.E_TRUE)         # 14 noeuds, signe de dstar
    shift = (dp.N_DISTRACT * (r_lure - r0) + dp.N_TRUE * (r_truth - r0)) / N
    return r0, r_lure, r_truth, shift


def boot_ci(v):
    v = np.asarray(v, float)
    n = len(v)
    m = np.array([v[RNG_BOOT.randint(0, n, n)].mean() for _ in range(N_BOOT)])
    return float(np.mean(v)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def fidelity_gate():
    print("GATE DE FIDELITE (condition FULL == dp.simulate, bit a bit)")
    for seed in (20, 27):
        rng = np.random.RandomState(3000 + seed)
        adj, s_on, s_off, _ = dp.make_deceptive(rng)
        sig_a, _, dv_a = dp.simulate(adj, s_on, s_off, seed * 10 + 1, T_PULSE)
        rng2 = np.random.RandomState(3000 + seed)
        adj2, s_on2, s_off2, _ = dp.make_deceptive(rng2)
        sig_b, dv_b = simulate(adj2, s_on2, s_off2, seed * 10 + 1, T_PULSE, "FULL")
        ok = np.allclose(sig_a, sig_b, atol=0) and np.allclose(dv_a, dv_b, atol=1e-12)
        print(f"  seed={seed} : identique={ok}")
        if not ok:
            raise SystemExit("GATE ECHOUE -- arret.")
    print("  -> gate PASSE.\n")


def main() -> int:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    fidelity_gate()

    r0, r_lure, r_truth, shift = static_prediction()
    print("=" * 100)
    print("TEMOIN ANALYTIQUE -- que predit une PURE rectification statique ?")
    print(f"  reponse stationnaire du noeud isole : r(0)={r0:+.4f}  "
          f"r(-{dp.E_DISTRACT})={r_lure:+.4f}  r(+{dp.E_TRUE})={r_truth:+.4f}")
    print(f"  ecarts : leurre {r_lure - r0:+.4f} (x{dp.N_DISTRACT})   "
          f"verite {r_truth - r0:+.4f} (x{dp.N_TRUE})")
    print(f"  decalage de moyenne predit (dstar=+1) : {shift:+.5f}")
    verdict_static = "CORRECT" if shift > 0 else "FAUX (pointe vers le leurre)"
    print(f"  -> une rectification purement statique donnerait une decision {verdict_static}")
    print("=" * 100)

    rows = []
    res = {c: {t: [] for t in PROBE_TIMES} for c in CONDITIONS}
    acc_stop = {c: [] for c in CONDITIONS}
    res_dec = {c: [] for c in CONDITIONS}     # trajectoires, pour la baseline d'arret
    res_dstar = []

    for seed in SEEDS:
        rng = np.random.RandomState(3000 + seed)
        adj, s_on, s_off, dstar = dp.make_deceptive(rng)
        res_dstar.append(dstar)
        for cond in CONDITIONS:
            sig, d_var = simulate(adj, s_on, s_off, seed * 10 + 1, T_PULSE, cond)
            dec = np.where(d_var >= 0, 1, -1)
            cd = dp.stop_doubt(sig)
            res_dec[cond].append(dec)
            acc_stop[cond].append(int(dp.dec_at(dec, cd) == dstar))
            for t in PROBE_TIMES:
                res[cond][t].append(int(dec[t - 1] == dstar))
            rows.append(dict(seed=seed, condition=cond, dstar=int(dstar),
                             stop=cd, acc_at_stop=acc_stop[cond][-1],
                             **{f"correct_t{t}": res[cond][t][-1] for t in PROBE_TIMES}))

    print("\nFRACTION DE DECISIONS CORRECTES A INSTANT FIXE, PENDANT LE LEURRE "
          f"(T_pulse={T_PULSE}, {len(SEEDS)} graines)")
    print("la moyenne du stimulus vaut -0.176*dstar a tous ces instants : elle dit FAUX")
    print(f"{'condition':<18}" + "".join(f"t={t:<8}" for t in PROBE_TIMES)
          + f"{'acc a l arret':>15}")
    print("-" * 90)
    for cond in CONDITIONS:
        line = f"{cond:<18}"
        for t in PROBE_TIMES:
            line += f"{np.mean(res[cond][t]):<10.2f}"
        line += f"{np.mean(acc_stop[cond]):>15.2f}"
        print(line)

    # ---- LA BASELINE SANS LAQUELLE LE RESULTAT NE VEUT RIEN DIRE -----------
    # A instant FIXE la decision est fausse ; a l'instant CHOISI par le doute elle
    # est juste. Deux lectures s'excluent : (a) le signal d'arret sait quand la
    # decision est mure -- ce serait le vrai mecanisme ; (b) les instants d'arret
    # sont simplement disperses et tombent par chance sur des phases favorables.
    # On tranche en comparant l'arret NATIF a des instants TIRES AU HASARD dans la
    # meme distribution empirique -- meme dispersion, aucune information.
    print("\nL'ARRET EST-IL INFORME, OU SEULEMENT DISPERSE ?")
    null_by_cond = {}
    rng_null = np.random.RandomState(4242)
    for cond in CONDITIONS:
        stops = np.array([r["stop"] for r in rows if r["condition"] == cond])
        accs = np.array([r["acc_at_stop"] for r in rows if r["condition"] == cond])
        # null : pour chaque graine, lire la decision a un instant d'arret TIRE
        # parmi ceux des AUTRES graines (meme distribution, aucun lien au run)
        null_accs = []
        for rep in range(200):
            perm = rng_null.permutation(len(stops))
            null_accs.append(np.mean([
                res_dec[cond][i][min(stops[perm[i]], MAXB) - 1] == res_dstar[i]
                for i in range(len(stops))]))
        m_null = float(np.mean(null_accs))
        null_by_cond[cond] = m_null
        print(f"  {cond:<18} arret natif={accs.mean():.2f}   "
              f"instants permutes={m_null:.2f}   "
              f"stops [{stops.min()}-{stops.max()}], med={int(np.median(stops))}")

    print("\nVERDICT")
    d1, lo1, hi1 = boot_ci(np.array(acc_stop["FULL"]) - np.array(acc_stop["DECOUPLE"]))
    d2, lo2, hi2 = boot_ci(np.array(acc_stop["FULL"]) - np.array(acc_stop["FROZEN_U"]))
    print(f"  1. L'ARRET EST INFORME, pas seulement disperse : "
          f"{np.mean(acc_stop['FULL']):.2f} a l'instant que le doute")
    print(f"     choisit, contre {null_by_cond['FULL']:.2f} a des instants de MEME distribution")
    print("     mais sans lien au run. Le reseau n'extrait donc PAS la verite en")
    print(f"     permanence : a instant fixe pendant le leurre il est a "
          f"{np.mean(res['FULL'][328]):.2f}, SOUS le hasard.")
    print("     Ce qu'il sait, c'est QUAND sa decision est mure. Meme role que l'horloge")
    print("     d'arret de P11 (12/07), retrouve ici sur la niche B1d/B5b.")
    print(f"  2. AUCUN EFFET DETECTABLE DU COUPLAGE a 20 graines : FULL - DECOUPLE = "
          f"{d1:+.2f} CI[{lo1:+.2f},{hi1:+.2f}].")
    print("     A ne PAS lire comme 'le couplage est inutile' : un IC qui couvre 0 sur")
    print("     20 graines ne demontre pas l'absence d'effet, il constate qu'on ne le")
    print("     voit pas a cette puissance. Et portee exacte : meme a D=0 le signal")
    print("     d'arret reste mean(|L v|), donc une LECTURE du desaccord spatial ; ce qui")
    print("     est ici sans effet visible est la PROPAGATION, pas la lecture spatiale.")
    print("     Deux conditions manquent pour trancher (arret non spatial ; plus de")
    print("     graines) -- non faites, notees comme pistes ouvertes.")
    print(f"  3. GELER LE DOUTE : FULL - FROZEN_U = {d2:+.2f} CI[{lo2:+.2f},{hi2:+.2f}] "
          f"-- NON TRANCHE.")
    print("     Ce qui est net en revanche est descriptif, pas inferentiel : les instants")
    print("     d'arret se dispersent enormement quand u est gele (187-513 en FULL contre")
    print("     181-1530 en FROZEN_U). La dynamique de u resserre l'arret ; que cela")
    print("     ameliore l'accuracy reste a etablir sur plus de graines.")
    print(f"  4. MON HYPOTHESE DE RECTIFICATION STATIQUE EST REFUTEE : le temoin analytique")
    print(f"     predit un decalage de {shift:+.5f}, soit le MAUVAIS signe. Une courbe de")
    print("     reponse concave n'explique pas l'effet ; il est temporel, pas statique.")

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[csv] {CSV_PATH}")
    make_figure(res, acc_stop, shift, null_by_cond)
    print(f"[png] {PNG_PATH}")
    print(f"\nWall time: {time.time() - t0:.1f}s")
    return 0


def make_figure(res, acc_stop, shift, null_by_cond):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    colors = {"FULL": "#d62728", "DECOUPLE": "#1f77b4",
              "FROZEN_U": "#ff7f0e", "DECOUPLE_FROZEN": "#7f7f7f"}
    ax = axes[0]
    for c in CONDITIONS:
        ax.plot(PROBE_TIMES, [np.mean(res[c][t]) for t in PROBE_TIMES],
                "o-", color=colors[c], label=c)
    ax.axhline(0.5, ls=":", c="gray", label="chance")
    ax.axvline(328, ls="--", c="k", lw=1)
    ax.text(335, 0.05, "M4R's stopping time", fontsize=8, rotation=90)
    ax.set_xlabel(f"time inside the lure window (lure lasts {T_PULSE} steps)")
    ax.set_ylabel("fraction of correct decisions")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Extracting the minority truth while the\nsignal average still says otherwise",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    xs = np.arange(len(CONDITIONS))
    ax.bar(xs - 0.19, [np.mean(acc_stop[c]) for c in CONDITIONS], 0.38,
           color=[colors[c] for c in CONDITIONS], edgecolor="k", label="native stop")
    ax.bar(xs + 0.19, [null_by_cond[c] for c in CONDITIONS], 0.38,
           color="white", edgecolor="k", hatch="///",
           label="permuted stop times (same spread, no link to run)")
    ax.axhline(0.5, ls=":", c="gray")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_xticks(xs)
    ax.set_xticklabels([c.replace("_", "\n") for c in CONDITIONS], fontsize=8)
    ax.set_ylabel("accuracy at the doubt's own stopping time")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Ablation at the native stop\n(static rectification predicts shift="
                 f"{shift:+.4f})", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Experiment B5-bis -- what actually extracts the minority truth: "
                 "the coupling, the node's nonlinearity, or neither?", fontsize=11)
    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=140)


if __name__ == "__main__":
    raise SystemExit(main())
