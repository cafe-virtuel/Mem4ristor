#!/usr/bin/env python3
"""
EXPERIENCE B -- face-a-face avec le recuit simule et l'exploration standard.

CONTEXTE (audit externe "Google DeepMind" joue par Gemini, 2026-07-17, point 2 ;
mis au programme par Julien, execute le 2026-07-26).
  Question de l'audit : "quelle classe computationnelle bat une exploration
  purement aleatoire ?" -- c'est le SEUL adversaire cite par l'audit que le
  projet n'ait jamais affronte. La niche etablie du doute (B1d/B5b, 07-08/07)
  est la DECISION SOUS CONVERGENCE-PIEGE A HORIZON INCONNU : un leurre nombreux
  domine tot, une verite persistante gagne tard, donc converger tot = se tromper.
  Le doute y bat les arrets naifs, mais EGALE deja le meilleur budget fixe.

  On reprend le harness EXACT de B1d/B5b (`scratch/deceptive_task_poc.py`) --
  meme flux, memes graines, meme lecture differentielle, meme metrique -- et on
  ajoute les adversaires cites, chacun regle a SON MAXIMUM par oracle (la regle
  de loyaute deja appliquee a l'ESN dans B5b : on ne bat pas un homme de paille).

LES TROIS ADVERSAIRES, traduits fidelement dans un probleme d'ARRET :
  - SA      recuit simule : etat de decision s, energie -s*S_t (S = evidence
            cumulee), flips acceptes avec proba exp(-dH/T), refroidissement en
            1/t (SANS horizon connu -- un refroidissement geometrique exigerait
            de connaitre T_max, ce que la tache interdit). Arret = gel (s stable
            + T basse). C'est l'argument meme du recuit : la temperature l'empeche
            de se figer sur un minimum local, et LE LEURRE EST UN MINIMUM LOCAL.
            Adversaire serieux, bien choisi par l'audit.
  - EPS     epsilon-greedy a deux bras (+1 / -1), recompense a*y_t, valeurs Q en
            moyenne exponentielle, arret quand l'ecart |Q+ - Q-| tient au-dessus
            d'un seuil pendant W pas.
  - NOISE   bruit stochastique pur + arret naif : signe(S_t + bruit), arret des
            que le signe tient W pas. Le controle "l'aleatoire seul suffit-il ?".
  + FIXED   meilleur budget fixe (politique non-adaptative) -- la baseline qui a
            deja tue une partie de la niche le 08/07, gardee comme juge de paix.

DEUX VOLETS D'INFORMATION, parce qu'un seul mentirait :
  volet SIGNAL : les adversaires lisent la variable de decision de M4R (d_var).
      Question isolee : le doute est-il une meilleure REGLE D'ARRET que le recuit
      sur un signal identique ?
  volet FLUX   : les adversaires lisent directement le flux d'observation
      (moyenne du stimulus + bruit d'observation). Question brutale : le
      reservoir M4R sert-il a quelque chose, ou un explorateur standard branche
      sur le flux fait-il mieux et moins cher ? Reserve a enoncer avec le
      resultat : cet observateur recoit GRATUITEMENT l'agregat global des 100
      capteurs, la ou M4R doit l'agreger par couplage local. S'il gagne, ce
      n'est pas une surprise -- ce qui compte est l'ecart et le cout.

CONTROLE DE LA RESERVE OUVERTE (backlog tache 8) : B1d/B5b utilisaient un readout
  INSTANTANE, et P6b (12/07) a montre que la reponse FHN est adaptative -- d'ou
  une reserve inscrite au backlog : "re-verifier B1d/B5b au readout lisse avant
  citation". On la ferme ici : tout est rejoue avec un readout lisse (moyenne
  glissante), et les deux lectures sont rapportees cote a cote.

Prediction honnete posee AVANT la mesure (le mandat l'exige) : incertaine. Si le
  recuit bat le doute, la niche se retrecit encore et c'est ecrit tel quel.

SORTIES : figures/expB_annealing_faceoff_poc.csv + .png
Cree : 2026-07-26 (Claude Opus 5, L'Ingenieur) -- Experience B du backlog DeepMind.
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

dp.MAX_BUDGET = 2000                # identique a B5b (t_pulse <= 700, marge suffisante)
MAXB = dp.MAX_BUDGET

N = dp.N
SEEDS = list(range(40))             # B5b en utilisait 15 ; monte a 40 pour trancher
                                    # les IC qui frolaient 0 au premier run correct
T_PULSE_LEVELS = [350, 700]         # meme regime trompeur que B5b
SMOOTH_W = 25                       # readout lisse (controle de la reserve backlog 8)
SIGMA_OBS = 0.05                    # bruit d'observation du volet FLUX (= sigma_v du reseau)
# Fenetre de stabilite exigee avant de s'arreter. CE PARAMETRE EST LA PATIENCE de
# l'adversaire : le fixer en dur revient a lui interdire d'attendre la fin du leurre.
# Un premier run (2026-07-26) le fixait a 50 et donnait EPS/NOISE a 0.00 sur 15/15
# graines -- pas 0.5 : SYSTEMATIQUEMENT faux, la signature d'un adversaire bride, pas
# d'un adversaire mauvais. Il appartient donc a la grille, comme tout le reste.
STABLE_W_GRID = [50, 200, 800]
N_BOOT = 10_000
RNG_BOOT = np.random.RandomState(20260726)

CSV_PATH = ROOT / "figures" / "expB_annealing_faceoff_poc.csv"
PNG_PATH = ROOT / "figures" / "expB_annealing_faceoff_poc.png"

# grilles d'hyperparametres -- chaque adversaire est pris a SON MAXIMUM (oracle)
SA_T0 = [0.5, 2.0, 8.0]
SA_TAU = [50.0, 200.0, 800.0]
SA_TSTOP = [0.05, 0.2]
EPS_EPS = [0.05, 0.2]
EPS_ALPHA = [0.01, 0.05]
EPS_THR = [0.02, 0.1]
LEAKY_ALPHA = [0.005, 0.02, 0.05]
LEAKY_THR = [0.0, 0.02]
NOISE_SIGMA = [0.05, 0.3]


# --- adversaires : chacun retourne (decision_finale, cout) ------------------
def run_sa(y, T0, tau, t_stop, W, seed):
    """Recuit simule sur l'evidence cumulee. Refroidissement en 1/t (sans horizon)."""
    rng = np.random.RandomState(seed)
    s = 1 if rng.rand() < 0.5 else -1
    S = 0.0
    stable = 0
    for t in range(len(y)):
        S += y[t]
        T = T0 / (1.0 + t / tau)
        dH = 2.0 * s * S                      # energie -s*S ; flip s -> -s
        if dH < 0 or rng.rand() < np.exp(-min(dH / max(T, 1e-9), 50.0)):
            s = -s
            stable = 0
        else:
            stable += 1
        if T < t_stop and stable >= W:
            return s, t + 1
    return s, len(y)


def run_eps_greedy(y, eps, alpha, thr, W, seed):
    """epsilon-greedy a deux bras (+1 / -1), recompense a*y_t."""
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
    """LE TEST QUI ISOLE LE FACTEUR (ajoute apres un premier run, 2026-07-26).

    Un simple filtre a OUBLI EXPONENTIEL + seuil : aucune exploration, aucune
    temperature, aucun tirage aleatoire. C'est epsilon-greedy prive de son
    epsilon -- il ne garde que la moyenne exponentielle des Q.

    Raison d'etre : dans le premier run correct, l'adversaire qui bat le doute
    est epsilon-greedy (alpha = 0.01-0.05), tandis que l'integrateur PUR (NOISE,
    qui somme depuis t=0) echoue completement. La difference entre les deux n'est
    pas l'exploration -- c'est que l'un OUBLIE le leurre et l'autre non. Si LEAKY
    egale epsilon-greedy, alors la reponse au point 2 de l'audit n'est pas "une
    classe d'exploration nous bat" mais "un filtre a oubli nous bat", ce qui n'est
    pas la meme phrase du tout. (Meme motif que le 13/07 : sur le raffinement de
    P11, un accumulateur naif corrigeait plus vite que M4R.)"""
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


def run_noise(y, sigma, W, seed):
    """Bruit stochastique pur + arret naif : le controle 'l'aleatoire seul suffit-il ?'."""
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


def best_by_oracle(runner, grids, y, dstar, seed):
    """Regle l'adversaire a SON MAXIMUM (meilleure accuracy, puis cout le plus bas)
    -- la meme regle de loyaute que B5b applique a l'ESN."""
    best = None
    for params in grids:
        dec, cost = runner(y, *params, seed)
        acc = int(dec == dstar)
        score = (acc, -cost)
        if best is None or score > best[0]:
            best = (score, dict(acc=acc, cost=cost, params=params))
    return best[1]


def smooth(x, w=SMOOTH_W):
    if w <= 1:
        return np.asarray(x, float)
    k = np.ones(w) / w
    return np.convolve(np.asarray(x, float), k, mode="same")


def boot_ci_paired(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    m = np.array([d[RNG_BOOT.randint(0, n, n)].mean() for _ in range(N_BOOT)])
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> int:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = []

    # la patience (W) fait partie de la grille de CHAQUE adversaire -- cf. STABLE_W_GRID
    sa_grid = [(a, b, c, w) for a in SA_T0 for b in SA_TAU for c in SA_TSTOP
               for w in STABLE_W_GRID]
    eps_grid = [(a, b, c, w) for a in EPS_EPS for b in EPS_ALPHA for c in EPS_THR
                for w in STABLE_W_GRID]
    noise_grid = [(a, w) for a in NOISE_SIGMA for w in STABLE_W_GRID]
    leaky_grid = [(a, b, w) for a in LEAKY_ALPHA for b in LEAKY_THR for w in STABLE_W_GRID]

    METHODS = ["M4R_DOUBT", "M4R_CONV",
               "SA_signal", "EPS_signal", "NOISE_signal", "LEAKY_signal",
               "SA_flux", "EPS_flux", "NOISE_flux", "LEAKY_flux"]
    acc = {readout: {k: [] for k in METHODS} for readout in ("inst", "smooth")}
    cost = {readout: {k: [] for k in METHODS} for readout in ("inst", "smooth")}
    fixed_budget = {readout: {} for readout in ("inst", "smooth")}
    BUDGET_GRID = list(range(100, MAXB, 100))
    for readout in ("inst", "smooth"):
        fixed_budget[readout] = {B: [] for B in BUDGET_GRID}

    print("=" * 104)
    print("EXPERIENCE B -- M4R vs recuit simule / epsilon-greedy / bruit pur "
          "(reponse DeepMind pt 2)")
    print(f"harness B1d/B5b exact | {len(SEEDS)} graines x t_pulse {T_PULSE_LEVELS} "
          f"| budget max {MAXB}")
    print("chaque adversaire regle a SON MAXIMUM par oracle ; deux volets "
          "d'information ; readout instantane ET lisse")
    print("=" * 104)

    for seed in SEEDS:
        per = {r: {k: [] for k in METHODS} for r in ("inst", "smooth")}
        perc = {r: {k: [] for k in METHODS} for r in ("inst", "smooth")}
        for t_pulse in T_PULSE_LEVELS:
            rng = np.random.RandomState(3000 + seed)
            adj, stim_on, stim_off, dstar = dp.make_deceptive(rng)

            # --- M4R : une seule simulation, reutilisee par tous les volets ---
            sig, dec, d_var = dp.simulate(adj, stim_on, stim_off, seed * 10 + 1, t_pulse)

            # flux d'observation brut vu par le volet FLUX
            obs_rng = np.random.RandomState(9000 + seed)
            base = np.where(np.arange(MAXB) < t_pulse, stim_on.mean(), stim_off.mean())
            y_flux = base + SIGMA_OBS * obs_rng.randn(MAXB)

            for readout in ("inst", "smooth"):
                if readout == "inst":
                    dv, sg = d_var, sig
                else:
                    dv, sg = smooth(d_var), smooth(sig)
                dec_r = np.where(dv >= 0, 1, -1)

                cd, cc = dp.stop_doubt(sg), dp.stop_conv(dv)
                per[readout]["M4R_DOUBT"].append(int(dp.dec_at(dec_r, cd) == dstar))
                per[readout]["M4R_CONV"].append(int(dp.dec_at(dec_r, cc) == dstar))
                perc[readout]["M4R_DOUBT"].append(cd)
                perc[readout]["M4R_CONV"].append(cc)

                for volet, y in (("signal", dv), ("flux", y_flux if readout == "inst"
                                                  else smooth(y_flux))):
                    for name, runner, grid in (("SA", run_sa, sa_grid),
                                               ("EPS", run_eps_greedy, eps_grid),
                                               ("NOISE", run_noise, noise_grid),
                                               ("LEAKY", run_leaky, leaky_grid)):
                        r = best_by_oracle(runner, grid, y, dstar, seed * 31 + 7)
                        per[readout][f"{name}_{volet}"].append(r["acc"])
                        perc[readout][f"{name}_{volet}"].append(r["cost"])

                for B in BUDGET_GRID:
                    fixed_budget[readout][B].append(int(dp.dec_at(dec_r, B) == dstar))

                rows.append(dict(seed=seed, t_pulse=t_pulse, dstar=int(dstar),
                                 readout=readout,
                                 **{k: per[readout][k][-1] for k in METHODS},
                                 **{f"cost_{k}": perc[readout][k][-1] for k in METHODS}))

        for readout in ("inst", "smooth"):
            for k in METHODS:
                acc[readout][k].append(float(np.mean(per[readout][k])))
                cost[readout][k].append(float(np.mean(perc[readout][k])))
        print(f"  seed {seed:>2} : doute={np.mean(per['inst']['M4R_DOUBT']):.2f} "
              f"SA_sig={np.mean(per['inst']['SA_signal']):.2f} "
              f"SA_flux={np.mean(per['inst']['SA_flux']):.2f} "
              f"EPS_flux={np.mean(per['inst']['EPS_flux']):.2f} "
              f"NOISE_flux={np.mean(per['inst']['NOISE_flux']):.2f}")

    # --- resume ------------------------------------------------------------
    for readout in ("inst", "smooth"):
        lbl = "INSTANTANE (protocole B1d/B5b)" if readout == "inst" \
              else f"LISSE (fenetre {SMOOTH_W}) -- controle de la reserve backlog 8"
        print("\n" + "=" * 104)
        print(f"RESULTATS -- readout {lbl}")
        print("-" * 104)
        print(f"{'methode':<16}{'accuracy':>10}{'cout moyen (pas)':>20}")
        for k in METHODS:
            print(f"{k:<16}{np.mean(acc[readout][k]):>10.2f}{np.mean(cost[readout][k]):>20.0f}")
        bB, bacc = max(((B, float(np.mean(fixed_budget[readout][B]))) for B in BUDGET_GRID),
                       key=lambda x: x[1])
        print(f"{'FIXED (best B)':<16}{bacc:>10.2f}{bB:>20}")

        print(f"\n  face-a-face (IC bootstrap apparie sur les {len(SEEDS)} graines) :")
        for opp in ["SA_signal", "EPS_signal", "NOISE_signal", "LEAKY_signal",
                    "SA_flux", "EPS_flux", "NOISE_flux", "LEAKY_flux"]:
            d, lo, hi = boot_ci_paired(acc[readout]["M4R_DOUBT"], acc[readout][opp])
            verdict = ("M4R gagne" if lo > 0 else
                       "M4R PERD" if hi < 0 else "parite (IC couvre 0)")
            print(f"    doute vs {opp:<12} : {d:+.2f} CI[{lo:+.2f},{hi:+.2f}]  -> {verdict}")

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[csv] {CSV_PATH}")

    make_figure(acc, cost, fixed_budget, BUDGET_GRID, METHODS)
    print(f"[png] {PNG_PATH}")
    print(f"\nWall time: {time.time() - t0:.1f}s")
    return 0


def make_figure(acc, cost, fixed_budget, BUDGET_GRID, METHODS):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(19.5, 5.4))
    colors = {"M4R_DOUBT": "#d62728", "M4R_CONV": "#ff9896",
              "SA_signal": "#1f77b4", "EPS_signal": "#aec7e8", "NOISE_signal": "#c5dbef",
              "LEAKY_signal": "#7b4173",
              "SA_flux": "#2ca02c", "EPS_flux": "#98df8a", "NOISE_flux": "#d5efd0",
              "LEAKY_flux": "#c994c7"}
    for ax, readout, title in zip(axes[:2], ("inst", "smooth"),
                                  ("instantaneous readout (B1d/B5b protocol)",
                                   f"smoothed readout (window {SMOOTH_W})")):
        means = [np.mean(acc[readout][k]) for k in METHODS]
        ax.bar(range(len(METHODS)), means,
               color=[colors[k] for k in METHODS], edgecolor="k")
        bB, bacc = max(((B, float(np.mean(fixed_budget[readout][B]))) for B in BUDGET_GRID),
                       key=lambda x: x[1])
        ax.axhline(bacc, ls="--", c="k", lw=1.2, label=f"best fixed budget (B={bB})")
        ax.axhline(0.5, ls=":", c="gray", label="chance")
        ax.set_xticks(range(len(METHODS)))
        ax.set_xticklabels([k.replace("_", "\n") for k in METHODS], fontsize=7.5)
        ax.set_ylabel("decision accuracy at stop")
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    # Panneau 3 : le plan accuracy-vs-cout. Sans lui la figure ment par omission --
    # le resultat central n'est pas "M4R perd 10 points" mais "M4R perd 10 points
    # ET coute 4x moins de pas". Les deux moities doivent etre visibles ensemble.
    ax = axes[2]
    for k in METHODS:
        ax.scatter(np.mean(cost["inst"][k]), np.mean(acc["inst"][k]), s=120,
                   color=colors[k], edgecolors="k", zorder=5)
        ax.annotate(k.replace("_", " "), (np.mean(cost["inst"][k]), np.mean(acc["inst"][k])),
                    textcoords="offset points", xytext=(7, 5), fontsize=7.5)
    bB, bacc = max(((B, float(np.mean(fixed_budget["inst"][B]))) for B in BUDGET_GRID),
                   key=lambda x: x[1])
    ax.scatter(bB, bacc, s=120, marker="s", color="k", zorder=5)
    ax.annotate(f"best fixed budget", (bB, bacc), textcoords="offset points",
                xytext=(7, -12), fontsize=7.5)
    ax.axhline(0.5, ls=":", c="gray")
    ax.set_xlabel("cost at stop (steps) -- lower is better")
    ax.set_ylabel("decision accuracy at stop")
    ax.set_title("Accuracy vs cost (instantaneous readout):\n"
                 "what beats doubt pays ~4x the steps", fontsize=10)
    ax.grid(alpha=0.3)

    fig.suptitle("Experiment B -- doubt vs simulated annealing / epsilon-greedy / pure noise "
                 "on the deceptive-decision niche\n"
                 "(signal volet = adversaries read M4R's decision variable; "
                 "flux volet = adversaries read the raw observation stream)", fontsize=10.5)
    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=140)


if __name__ == "__main__":
    raise SystemExit(main())
