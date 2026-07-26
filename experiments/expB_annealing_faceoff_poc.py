#!/usr/bin/env python3
"""
EXPERIENCE B -- face-a-face avec le recuit simule et l'exploration standard.

CONTEXTE (audit externe "Google DeepMind" joue par Gemini, 2026-07-17, point 2 ;
mis au programme par Julien, execute le 2026-07-26).
  Question de l'audit : "quelle classe computationnelle bat une exploration
  purement aleatoire ?" -- le SEUL adversaire cite que le projet n'ait jamais
  affronte. La niche etablie du doute (B1d/B5b) est la DECISION SOUS
  CONVERGENCE-PIEGE A HORIZON INCONNU : un leurre nombreux domine tot, une verite
  persistante gagne tard, donc converger tot = se tromper.

  On reprend le harness EXACT de B1d/B5b (`scratch/deceptive_task_poc.py`) et on
  ajoute les adversaires cites.

########################################################################
# CORRECTION DE METHODE (2026-07-26, meme jour, avant toute citation)  #
########################################################################
Une premiere version reglait chaque adversaire "a son maximum par ORACLE PAR
RUN" -- le meilleur des N combinaisons d'hyperparametres, choisi en connaissant
la bonne reponse, run par run. C'etait cense etre genereux envers l'adversaire.
C'etait en realite de la FABRICATION DE RESULTAT : un controle sur un signal
SANS AUCUNE INFORMATION (bruit pur) donne alors 0.935 d'accuracy au lieu de
0.500, parce qu'avec 18 combinaisons et une decision binaire, il s'en trouve
presque toujours une qui tombe juste par hasard.
Le biais allait CONTRE M4R, qui n'ajuste rien : il utilise son arret natif.

Methode corrigee, appliquee ici : les hyperparametres de chaque adversaire sont
choisis UNE FOIS sur les graines d'ENTRAINEMENT (accuracy moyenne), puis
evalues sur des graines DISJOINTES. L'adversaire garde son meilleur reglage
global -- ce qui reste genereux -- mais ne peut plus deviner la reponse.
Le controle "bruit pur" est rejoue a chaque execution et imprime : si la
selection train/test rend 0.5 sur du bruit, la procedure est saine.

RESERVE SUR UN RESULTAT EXISTANT : `b5b_deceptive_exploration.py` (08/07) regle
l'ESN par le meme oracle par run (grille de 6 combinaisons, moins diverse donc
moins exploitable). Le biais y joue en faveur de l'ESN, donc contre M4R : la
conclusion de B5b ("le doute bat les arrets naifs de l'ESN") en sort
CONSERVATIVE, pas invalidee. A re-verifier si elle doit etre citee au chiffre.
########################################################################

LES ADVERSAIRES, traduits fidelement dans un probleme d'ARRET :
  - SA      recuit simule : energie -s*S_t, flips acceptes avec proba
            exp(-dH/T), refroidissement en 1/t (SANS horizon connu -- un
            refroidissement geometrique exigerait de connaitre T_max, ce que la
            tache interdit). C'est l'argument meme du recuit : la temperature
            l'empeche de se figer sur un minimum local, ET LE LEURRE EST UN
            MINIMUM LOCAL. Adversaire serieux, bien choisi par l'audit.
  - EPS     epsilon-greedy a deux bras, valeurs Q en moyenne exponentielle.
  - NOISE   bruit stochastique pur + arret naif : le controle "l'aleatoire seul
            suffit-il ?".
  - LEAKY   filtre a oubli exponentiel : epsilon-greedy PRIVE de son epsilon.
            Sert a isoler le facteur -- si LEAKY egale EPS, ce qui bat le doute
            n'est pas l'exploration mais l'oubli.
  + FIXED   meilleur budget fixe (la baseline qui a deja tue une partie de la
            niche le 08/07), gardee comme juge de paix.
  La PATIENCE (fenetre de stabilite avant arret) appartient a la grille de
  chacun : la figer reviendrait a leur interdire d'attendre la fin du leurre.

DEUX VOLETS D'INFORMATION, parce qu'un seul mentirait :
  volet SIGNAL : les adversaires lisent la variable de decision de M4R.
      Le doute est-il une meilleure REGLE D'ARRET, a information identique ?
  volet FLUX   : ils lisent directement le flux d'observation. Le reservoir
      sert-il a quelque chose ? (Le cout reel de ce cablage est chiffre a part,
      dans `expB2_wiring_budget_poc.py`.)

CONTROLE DE LA RESERVE BACKLOG 8 : B1d/B5b utilisaient un readout INSTANTANE et
  P6b (12/07) a montre que la reponse FHN est adaptative. Tout est rejoue en
  readout lisse, les deux lectures cote a cote.

SORTIES : figures/expB_annealing_faceoff_poc.csv + .png
Cree : 2026-07-26 (Claude Opus 5, L'Ingenieur) -- Experience B du backlog DeepMind.
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

SEEDS_TRAIN = list(range(20))       # servent A CHOISIR les hyperparametres
SEEDS_TEST = list(range(20, 40))    # servent A MESURER -- disjointes
T_PULSE_LEVELS = [350, 700]
SMOOTH_W = 25
SIGMA_OBS = 0.05
STABLE_W_GRID = [50, 200, 800]      # la patience appartient a la grille
N_BOOT = 10_000
RNG_BOOT = np.random.RandomState(20260726)

CSV_PATH = ROOT / "figures" / "expB_annealing_faceoff_poc.csv"
PNG_PATH = ROOT / "figures" / "expB_annealing_faceoff_poc.png"

SA_T0 = [0.5, 2.0, 8.0]
SA_TAU = [50.0, 200.0, 800.0]
SA_TSTOP = [0.05, 0.2]
EPS_EPS = [0.05, 0.2]
EPS_ALPHA = [0.01, 0.05]
EPS_THR = [0.02, 0.1]
LEAKY_ALPHA = [0.005, 0.02, 0.05]
LEAKY_THR = [0.0, 0.02]
NOISE_SIGMA = [0.05, 0.3]


# --- adversaires : chacun retourne (decision, cout) -------------------------
def run_sa(y, T0, tau, t_stop, W, seed):
    rng = np.random.RandomState(seed)
    s = 1 if rng.rand() < 0.5 else -1
    S, stable = 0.0, 0
    for t in range(len(y)):
        S += y[t]
        T = T0 / (1.0 + t / tau)
        dH = 2.0 * s * S
        if dH < 0 or rng.rand() < np.exp(-min(dH / max(T, 1e-9), 50.0)):
            s, stable = -s, 0
        else:
            stable += 1
        if T < t_stop and stable >= W:
            return s, t + 1
    return s, len(y)


def run_eps_greedy(y, eps, alpha, thr, W, seed):
    rng = np.random.RandomState(seed)
    Q = {1: 0.0, -1: 0.0}
    stable = 0
    for t in range(len(y)):
        a = (1 if Q[1] >= Q[-1] else -1) if rng.rand() > eps else (1 if rng.rand() < 0.5 else -1)
        Q[a] += alpha * (a * y[t] - Q[a])
        best = 1 if Q[1] >= Q[-1] else -1
        stable = stable + 1 if abs(Q[1] - Q[-1]) > thr else 0
        if stable >= W:
            return best, t + 1
    return (1 if Q[1] >= Q[-1] else -1), len(y)


def run_leaky(y, alpha, thr, W, seed):
    """epsilon-greedy prive de son epsilon : oubli exponentiel + seuil, zero alea.
    Sert a distinguer 'une classe d'exploration nous bat' de 'un passe-bas nous bat'."""
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


def smooth(x, w=SMOOTH_W):
    return np.asarray(x, float) if w <= 1 else np.convolve(np.asarray(x, float),
                                                           np.ones(w) / w, mode="same")


def boot_ci_paired(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    m = np.array([d[RNG_BOOT.randint(0, n, n)].mean() for _ in range(N_BOOT)])
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


ADVERSARIES = {
    "SA": (run_sa, [(a, b, c, w) for a in SA_T0 for b in SA_TAU
                    for c in SA_TSTOP for w in STABLE_W_GRID]),
    "EPS": (run_eps_greedy, [(a, b, c, w) for a in EPS_EPS for b in EPS_ALPHA
                             for c in EPS_THR for w in STABLE_W_GRID]),
    "NOISE": (run_leaky, None),   # place tenue, remplacee ci-dessous
    "LEAKY": (run_leaky, [(a, b, w) for a in LEAKY_ALPHA for b in LEAKY_THR
                          for w in STABLE_W_GRID]),
}


def run_noise(y, sigma, W, seed):
    rng = np.random.RandomState(seed)
    S = 0.0
    s_prev, stable = 0, 0
    for t in range(len(y)):
        S += y[t]
        s = 1 if (S + sigma * rng.randn()) >= 0 else -1
        stable = stable + 1 if s == s_prev else 0
        s_prev = s
        if stable >= W:
            return s, t + 1
    return s_prev, len(y)


ADVERSARIES["NOISE"] = (run_noise, [(a, w) for a in NOISE_SIGMA for w in STABLE_W_GRID])


def sanity_control():
    """La procedure de selection rend-elle 0.5 sur un signal SANS information ?
    C'est ce controle qui a fait tomber l'oracle-par-run (0.935 sur du bruit)."""
    rng = np.random.RandomState(1234)
    n = 120
    ys, ds = [], []
    for _ in range(n):
        ds.append(1 if rng.rand() < 0.5 else -1)
        ys.append(0.05 * rng.randn(MAXB))
    half = n // 2
    print("CONTROLE DE PROCEDURE -- signal sans aucune information (bruit pur)")
    for name, (runner, grid) in ADVERSARIES.items():
        # oracle par run (la methode FAUTIVE, gardee comme temoin)
        orc = np.mean([max(int(runner(ys[i], *p, 7)[0] == ds[i]) for p in grid)
                       for i in range(n)])
        # selection train -> evaluation test (la methode CORRIGEE)
        best = max(grid, key=lambda p: np.mean(
            [int(runner(ys[i], *p, 7)[0] == ds[i]) for i in range(half)]))
        tst = np.mean([int(runner(ys[i], *best, 7)[0] == ds[i]) for i in range(half, n)])
        print(f"  {name:<6} oracle-par-run={orc:.3f} (faux)   "
              f"selection train/test={tst:.3f} (attendu ~0.5)")
    print()


def main() -> int:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("=" * 100)
    print("EXPERIENCE B -- M4R vs recuit simule / epsilon-greedy / bruit / oubli")
    print(f"harness B1d/B5b exact | {len(SEEDS_TRAIN)} graines TRAIN + "
          f"{len(SEEDS_TEST)} graines TEST (disjointes) | budget max {MAXB}")
    print("=" * 100)
    sanity_control()

    # store[(name, volet, readout, combo)][group] = liste d'accuracies
    store = defaultdict(lambda: defaultdict(list))
    costs = defaultdict(lambda: defaultdict(list))
    m4r = defaultdict(lambda: defaultdict(list))     # [(kind, readout)][group]
    m4r_cost = defaultdict(lambda: defaultdict(list))
    fixed_b = defaultdict(lambda: defaultdict(list))  # [(B, readout)][group]
    BUDGET_GRID = list(range(100, MAXB, 100))
    rows = []

    for group, seeds in (("train", SEEDS_TRAIN), ("test", SEEDS_TEST)):
        for seed in seeds:
            for t_pulse in T_PULSE_LEVELS:
                rng = np.random.RandomState(3000 + seed)
                adj, s_on, s_off, dstar = dp.make_deceptive(rng)
                sig, dec, d_var = dp.simulate(adj, s_on, s_off, seed * 10 + 1, t_pulse)

                obs_rng = np.random.RandomState(9000 + seed)
                base = np.where(np.arange(MAXB) < t_pulse, s_on.mean(), s_off.mean())
                y_flux = base + SIGMA_OBS * obs_rng.randn(MAXB)

                for readout in ("inst", "smooth"):
                    dv, sg = (d_var, sig) if readout == "inst" else (smooth(d_var), smooth(sig))
                    dec_r = np.where(dv >= 0, 1, -1)
                    cd, cc = dp.stop_doubt(sg), dp.stop_conv(dv)
                    m4r[("DOUBT", readout)][group].append(int(dp.dec_at(dec_r, cd) == dstar))
                    m4r[("CONV", readout)][group].append(int(dp.dec_at(dec_r, cc) == dstar))
                    m4r_cost[("DOUBT", readout)][group].append(cd)
                    m4r_cost[("CONV", readout)][group].append(cc)
                    for B in BUDGET_GRID:
                        fixed_b[(B, readout)][group].append(int(dp.dec_at(dec_r, B) == dstar))

                    for volet, y in (("signal", dv),
                                     ("flux", y_flux if readout == "inst" else smooth(y_flux))):
                        for name, (runner, grid) in ADVERSARIES.items():
                            for combo in grid:
                                d_, c_ = runner(y, *combo, seed * 31 + 7)
                                store[(name, volet, readout, combo)][group].append(
                                    int(d_ == dstar))
                                costs[(name, volet, readout, combo)][group].append(c_)
                rows.append(dict(group=group, seed=seed, t_pulse=t_pulse, dstar=int(dstar)))
        print(f"  [{group}] {len(seeds)} graines simulees")

    # --- selection sur TRAIN, mesure sur TEST ------------------------------
    chosen = {}
    for name, (runner, grid) in ADVERSARIES.items():
        for volet in ("signal", "flux"):
            for readout in ("inst", "smooth"):
                best = max(grid, key=lambda c: (
                    np.mean(store[(name, volet, readout, c)]["train"]),
                    -np.mean(costs[(name, volet, readout, c)]["train"])))
                chosen[(name, volet, readout)] = best

    print("\n" + "=" * 100)
    print("HYPERPARAMETRES CHOISIS SUR LES GRAINES TRAIN (puis figes)")
    for (name, volet, readout), c in sorted(chosen.items()):
        if readout == "inst":
            print(f"  {name:<6} {volet:<7} -> {c}")

    for readout in ("inst", "smooth"):
        lbl = "INSTANTANE (protocole B1d/B5b)" if readout == "inst" \
              else f"LISSE (fenetre {SMOOTH_W}) -- controle de la reserve backlog 8"
        print("\n" + "=" * 100)
        print(f"RESULTATS SUR LES GRAINES TEST (disjointes) -- readout {lbl}")
        print("-" * 100)
        print(f"{'methode':<16}{'accuracy':>10}{'cout moyen (pas)':>20}")
        print(f"{'M4R_DOUBT':<16}{np.mean(m4r[('DOUBT', readout)]['test']):>10.2f}"
              f"{np.mean(m4r_cost[('DOUBT', readout)]['test']):>20.0f}")
        print(f"{'M4R_CONV':<16}{np.mean(m4r[('CONV', readout)]['test']):>10.2f}"
              f"{np.mean(m4r_cost[('CONV', readout)]['test']):>20.0f}")
        for volet in ("signal", "flux"):
            for name in ADVERSARIES:
                c = chosen[(name, volet, readout)]
                print(f"{name + '_' + volet:<16}"
                      f"{np.mean(store[(name, volet, readout, c)]['test']):>10.2f}"
                      f"{np.mean(costs[(name, volet, readout, c)]['test']):>20.0f}")
        bB, bacc = max(((B, float(np.mean(fixed_b[(B, readout)]["train"]))) for B in BUDGET_GRID),
                       key=lambda x: x[1])
        print(f"{'FIXED (B=' + str(bB) + ')':<16}"
              f"{np.mean(fixed_b[(bB, readout)]['test']):>10.2f}{bB:>20}")

        print(f"\n  face-a-face sur TEST (IC bootstrap apparie, {len(SEEDS_TEST)} graines) :")
        base = per_seed(m4r[("DOUBT", readout)]["test"])
        for volet in ("signal", "flux"):
            for name in ADVERSARIES:
                c = chosen[(name, volet, readout)]
                opp = per_seed(store[(name, volet, readout, c)]["test"])
                d, lo, hi = boot_ci_paired(base, opp)
                verdict = ("M4R gagne" if lo > 0 else "M4R PERD" if hi < 0
                           else "parite (IC couvre 0)")
                print(f"    doute vs {name + '_' + volet:<13} : {d:+.2f} "
                      f"CI[{lo:+.2f},{hi:+.2f}]  -> {verdict}")

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        out = []
        for readout in ("inst", "smooth"):
            out.append(dict(readout=readout, method="M4R_DOUBT",
                            acc=float(np.mean(m4r[("DOUBT", readout)]["test"])),
                            cost=float(np.mean(m4r_cost[("DOUBT", readout)]["test"])),
                            params=""))
            out.append(dict(readout=readout, method="M4R_CONV",
                            acc=float(np.mean(m4r[("CONV", readout)]["test"])),
                            cost=float(np.mean(m4r_cost[("CONV", readout)]["test"])),
                            params=""))
            for volet in ("signal", "flux"):
                for name in ADVERSARIES:
                    c = chosen[(name, volet, readout)]
                    out.append(dict(readout=readout, method=f"{name}_{volet}",
                                    acc=float(np.mean(store[(name, volet, readout, c)]["test"])),
                                    cost=float(np.mean(costs[(name, volet, readout, c)]["test"])),
                                    params=str(c)))
        w = csv.DictWriter(f, fieldnames=["readout", "method", "acc", "cost", "params"])
        w.writeheader()
        w.writerows(out)
    print(f"\n[csv] {CSV_PATH}")

    make_figure(m4r, m4r_cost, store, costs, chosen, fixed_b, BUDGET_GRID)
    print(f"[png] {PNG_PATH}")
    print(f"\nWall time: {time.time() - t0:.1f}s")
    return 0


def per_seed(flat):
    """Les listes sont indexees (seed, t_pulse) ; on regroupe par graine pour que
    l'IC apparie porte sur des unites independantes."""
    a = np.asarray(flat, float).reshape(-1, len(T_PULSE_LEVELS))
    return a.mean(axis=1)


def make_figure(m4r, m4r_cost, store, costs, chosen, fixed_b, BUDGET_GRID):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["M4R_DOUBT", "M4R_CONV"] + [f"{n}_{v}" for v in ("signal", "flux")
                                         for n in ADVERSARIES]
    colors = {"M4R_DOUBT": "#d62728", "M4R_CONV": "#ff9896",
              "SA_signal": "#1f77b4", "EPS_signal": "#aec7e8",
              "NOISE_signal": "#c5dbef", "LEAKY_signal": "#7b4173",
              "SA_flux": "#2ca02c", "EPS_flux": "#98df8a",
              "NOISE_flux": "#d5efd0", "LEAKY_flux": "#c994c7"}

    def val(name, readout, what="acc"):
        src = store if "_" in name and not name.startswith("M4R") else None
        if name.startswith("M4R"):
            kind = name.split("_")[1]
            d = m4r if what == "acc" else m4r_cost
            return float(np.mean(d[(kind, readout)]["test"]))
        n, v = name.rsplit("_", 1)
        c = chosen[(n, v, readout)]
        d = store if what == "acc" else costs
        return float(np.mean(d[(n, v, readout, c)]["test"]))

    fig, axes = plt.subplots(1, 3, figsize=(19.5, 5.4))
    for ax, readout, title in zip(axes[:2], ("inst", "smooth"),
                                  ("instantaneous readout (B1d/B5b protocol)",
                                   f"smoothed readout (window {SMOOTH_W})")):
        ax.bar(range(len(names)), [val(n, readout) for n in names],
               color=[colors[n] for n in names], edgecolor="k")
        bB, _ = max(((B, float(np.mean(fixed_b[(B, readout)]["train"]))) for B in BUDGET_GRID),
                    key=lambda x: x[1])
        ax.axhline(float(np.mean(fixed_b[(bB, readout)]["test"])), ls="--", c="k", lw=1.2,
                   label=f"best fixed budget (B={bB})")
        ax.axhline(0.5, ls=":", c="gray", label="chance")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=7.5)
        ax.set_ylabel("decision accuracy at stop (held-out seeds)")
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    ax = axes[2]
    for n in names:
        ax.scatter(val(n, "inst", "cost"), val(n, "inst"), s=120,
                   color=colors[n], edgecolors="k", zorder=5)
        ax.annotate(n.replace("_", " "), (val(n, "inst", "cost"), val(n, "inst")),
                    textcoords="offset points", xytext=(7, 5), fontsize=7.5)
    ax.axhline(0.5, ls=":", c="gray")
    ax.set_xlabel("cost at stop (steps) -- lower is better")
    ax.set_ylabel("decision accuracy at stop")
    ax.set_title("Accuracy vs cost in STEPS\n(true wiring cost: see expB2)", fontsize=10)
    ax.grid(alpha=0.3)

    fig.suptitle("Experiment B -- doubt vs simulated annealing / epsilon-greedy / noise / "
                 "forgetting filter, on the deceptive-decision niche\n"
                 "hyperparameters selected on training seeds, measured on DISJOINT seeds",
                 fontsize=10.5)
    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=140)


if __name__ == "__main__":
    raise SystemExit(main())
