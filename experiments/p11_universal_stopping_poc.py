#!/usr/bin/env python3
"""
P11 (legs de Fable) -- L'horloge de deliberation comme module d'arret UNIVERSEL.
=================================================================================
Cree : 2026-07-12 (Claude Fable 5, L'Ingenieur) -- piste P11 de
docs/PISTES_POUR_LA_SUITE_2026-07-12.md. Premiere fois que le signal M4R est
teste comme critere d'arret d'un systeme TIERS (un solveur iteratif qui n'a
rien a voir avec M4R).

PROBLEME UNIVERSEL. 'Quand s'arreter ?' (solveurs, MCMC, early-exit). Une
tolerance fixe sur le residu se fait pieger par les PLATEAUX TROMPEURS (residu
minuscule, mais on n'est PAS arrive) ; B5b a montre que |Lv| est une 'horloge
de deliberation intrinseque' sur les taches M4R -- ici on la branche en
SIDE-CAR sur un solveur exterieur.

PROTOCOLE.
  - Solveur tiers : descente de gradient x <- x - eta*f'(x), eta=0.05,
    budget max 2000 iterations, cible x*=0, depart x0=2.
  - Famille FACILE (12 problemes) : f'(x) = x (quadratique pure) -- converger
    vite, s'arreter tot est juste.
  - Famille TROMPEUSE (12 problemes) : f'(x) = x * h(x) avec
    h(x) = 1 - (1-h_min)*exp(-(x-x_p)^2/(2 w^2)) -- un PLATEAU (pente ~h_min)
    entre le depart et le vrai minimum : le residu s'effondre sur le plateau
    (piege les tolerances), puis REMONTE a la sortie (residu non monotone,
    ce qui piege AUSSI l'early stopping a patience), puis convergence vraie.
    h_min, x_p, w tires par seed (profondeur et duree de plateau variees =
    l'horizon est inconnu).
    CALIBRATION (revisions apres lancements 1-2, structure pas p-hacking) :
    (1) au 1er lancement les plateaux ne descendaient PAS sous la grille des
    tolerances (r_plateau ~ 0.02 > tol_min=0.002 -> TOL global reussissait
    tout en 138 iters : tache non piegeuse). (2) au 2e, la gaussienne etroite
    etait parfois ENJAMBEE par le pas discret du solveur (2/12 creux non
    mordants). Revision finale : plateau PLAT a rampes douces (smoothstep,
    largeur de rampe 0.06 -- deceleration progressive garantie, pas
    d'enjambement) ; la profondeur h_min est DERIVEE de la duree de traversee
    cible T_c tiree dans [700, 1400] iters : h_min = 2*w_flat/(ETA*x_p*T_c),
    d'ou un creux r = 2*w_flat/(ETA*T_c) dans [0.00057, 0.0017] -- sous
    TOUTES les tolerances de la grille, et traversee toujours faisable au
    budget. Le pre-vol durci exige que chaque trompeur (i) atteigne x* au
    budget max et (ii) morde la tolerance la plus fine, sinon campagne
    annulee.
  - Succes d'un arret : |x_stop| < 0.05 (on est arrive au vrai minimum).
  - REGLES D'ARRET (chacune UN hyperparametre GLOBAL sur le melange des 24
    problemes -- personne ne se regle par probleme) :
      TOL      : arret quand r_t = |f'(x_t)| < tol, tol dans {0.02,0.01,0.005,0.002}
      PATIENCE : arret quand r_t > (1-1%%)*r_{t-K} (moins de 1%% d'amelioration
                 en K iterations), K dans {50,100,200,400,800,1600} -- l'early
                 stopping standard (Keras/PyTorch), l'adversaire fort a
                 memoire (grille etendue a 1600 au lancement 3 pour couvrir
                 les durees de plateau : loyaute envers l'adversaire).
      FIXED    : budget fixe B dans {200..2000 par 200} -- l'adversaire B5b.
      M4R_U    : side-car M4R (lattice 10x10, constantes du coeur PAR DEFAUT),
                 stimulus antagoniste +/- C_IN*r_t (moitie des noeuds +, moitie -,
                 le conflit est proportionnel au residu), k_net pas de reseau
                 par iteration du solveur ; arret quand u_mean retombe sous
                 frac * pic-roulant-causal. Grille globale : k_net x frac.
      M4R_SIG  : meme side-car, horloge sur mean|Lv| lisse (50 pas reseau) --
                 le capteur rapide (lecon P6 : le brut reagit, u est lent).
  - COUT compte honnetement : iterations solveur utilisees ; le side-car
    consomme EN PLUS k_net*100 noeuds-pas par iteration (rapporte tel quel --
    la lecon B1c 'le doute sur-reflechit quand c'est facile' est attendue ici).

CRITERES PRE-FIXES (avant de voir un chiffre) :
  - La niche existe si M4R_U ou M4R_SIG bat TOL en taux de succes global
    (IC bootstrap apparie > 0) grace aux trompeurs, SANS s'effondrer sur les
    faciles (succes faciles >= 90%%).
  - Adversaires forts : prediction honnete (pattern B5b, 4e occurrence
    attendue) -- PATIENCE et FIXED bien regles feront aussi bien ou mieux ;
    la valeur du side-car serait alors sa GENERICITE (aucun acces au residu
    requis, seulement un flux de conflit), pas sa superiorite.
  - Verdict negatif possible et informatif : si u est trop LENT pour suivre
    le solveur (tau_eff ~ milliers de pas reseau, cf. P1), M4R_U rendra des
    arrets tardifs/jamais -- a rapporter tel quel.

Statut : exploratoire, hors preprint, coeur non touche.
Sorties : figures/p11_universal_stopping_poc{,_agg}.csv + .png
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

CSV_PATH = ROOT / "figures" / "p11_universal_stopping_poc.csv"
AGG_PATH = ROOT / "figures" / "p11_universal_stopping_poc_agg.csv"
PNG_PATH = ROOT / "figures" / "p11_universal_stopping_poc.png"

SIDE, N = 10, 100
ETA = 0.05
MAX_ITER = 2000
X0 = 2.0
SUCCESS_TOL = 0.05
SEEDS = list(range(12))
C_IN = 3.0
TOLS = [0.02, 0.01, 0.005, 0.002]
PATIENCE_KS = [50, 100, 200, 400, 800, 1600]
FIXED_BUDGETS = list(range(200, MAX_ITER + 1, 200))
KNETS = [2, 5, 10]
FRACS = [0.3, 0.5, 0.7]
SIG_SMOOTH = 50


W_RAMP = 0.06


def make_problem(kind, seed):
    rng = np.random.RandomState(70000 + seed)
    if kind == "easy":
        return {"kind": kind}
    w_flat = rng.uniform(0.02, 0.03)
    t_c = rng.uniform(700.0, 1400.0)       # duree de traversee cible (iters)
    x_p = rng.uniform(0.9, 1.3)
    h_min = 2.0 * w_flat / (ETA * x_p * t_c)
    return {"kind": kind, "h_min": h_min, "x_p": x_p, "w_flat": w_flat}


def grad(x, pb):
    if pb["kind"] == "easy":
        return x
    d = abs(x - pb["x_p"]) - pb["w_flat"]
    if d <= 0:
        h = pb["h_min"]
    elif d >= W_RAMP:
        h = 1.0
    else:
        s = d / W_RAMP
        h = pb["h_min"] + (1.0 - pb["h_min"]) * s * s * (3.0 - 2.0 * s)
    return x * h


def solve_traj(pb):
    """Trajectoire complete du solveur (x_t, r_t) sur MAX_ITER iterations."""
    xs = np.empty(MAX_ITER + 1)
    rs = np.empty(MAX_ITER + 1)
    x = X0
    for t in range(MAX_ITER + 1):
        r = abs(grad(x, pb))
        xs[t] = x
        rs[t] = r
        x = x - ETA * grad(x, pb)
    return xs, rs


def sidecar_traces(rs, seed, k_net, adj):
    """Side-car M4R alimente par le residu ; retourne u_mean(t) et sig(t)
    echantillonnes a chaque iteration solveur (apres k_net pas reseau)."""
    net = Mem4Network(size=SIDE, heretic_ratio=0.0, seed=seed, adjacency_matrix=adj)
    L = net.L
    half = np.zeros(N)
    half[: N // 2] = 1.0
    half[N // 2:] = -1.0
    u_tr = np.empty(MAX_ITER + 1)
    sig_tr = np.empty(MAX_ITER + 1)
    for t in range(MAX_ITER + 1):
        stim = C_IN * rs[t] * half
        for _ in range(k_net):
            net.step(I_stimulus=stim)
        u_tr[t] = float(np.mean(net.model.u))
        sig_tr[t] = float(np.mean(np.abs(L @ net.model.v)))
    # lissage causal du capteur rapide (SIG_SMOOTH pas reseau ~ /k_net iters)
    w = max(1, SIG_SMOOTH // k_net)
    csum = np.cumsum(sig_tr)
    sig_sm = np.empty_like(sig_tr)
    for t in range(len(sig_tr)):
        lo = max(0, t - w + 1)
        tot = csum[t] - (csum[lo - 1] if lo > 0 else 0.0)
        sig_sm[t] = tot / (t - lo + 1)
    return u_tr, sig_sm


def stop_rolling(signal, frac, warm=20):
    peak = float(np.max(signal[:warm])) if warm > 0 else 0.0
    for t in range(warm, len(signal)):
        peak = max(peak, float(signal[t]))
        if peak > 0 and signal[t] < frac * peak:
            return t
    return len(signal) - 1


def stop_tol(rs, tol):
    idx = np.argmax(rs < tol)
    return int(idx) if rs[idx] < tol else MAX_ITER


def stop_patience(rs, K, delta=0.01):
    for t in range(K, len(rs)):
        if rs[t] > (1.0 - delta) * rs[t - K]:
            return t
    return MAX_ITER


def boot_ci_paired(a, b, n_boot=10000, seed=20260712):
    rng = np.random.RandomState(seed)
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    m = np.empty(n_boot)
    for k in range(n_boot):
        m[k] = d[rng.randint(0, n, n)].mean()
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> int:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    adj = make_lattice_adj(SIDE, periodic=True).astype(float)

    problems = ([("easy", s) for s in SEEDS] + [("trap", s) for s in SEEDS])

    # ------ pre-vol durci : les 12 trompeurs doivent etre solubles ET piegeux ------
    pb_e = make_problem("easy", 0)
    xs_e, rs_e = solve_traj(pb_e)
    print(f"[pre-vol] facile : r(400)={rs_e[400]:.4f}, |x|(2000)={abs(xs_e[-1]):.4f}")
    bad = 0
    rs_t = None
    for s in SEEDS:
        pb_t = make_problem("trap", s)
        xs_t, rs_here = solve_traj(pb_t)
        if s == 0:
            rs_t = rs_here
        soluble = abs(xs_t[-1]) < SUCCESS_TOL
        # le creux doit mordre la tolerance la plus fine AVANT d'etre arrive
        creux = rs_here[:np.argmax(np.abs(xs_t) < SUCCESS_TOL) or MAX_ITER].min()
        piege = creux < min(TOLS)
        if not (soluble and piege):
            bad += 1
            print(f"  [pre-vol] trap seed {s}: soluble={soluble} "
                  f"(|x|fin={abs(xs_t[-1]):.3f}), creux={creux:.5f} piege={piege}")
    if bad:
        print(f"[pre-vol] {bad}/12 trompeurs mal structures -> campagne annulee, "
              f"recalibrer la famille (pas les regles).")
        return 1
    print(f"[pre-vol] OK : 12/12 trompeurs solubles au budget max ET creux du "
          f"residu sous tol_min={min(TOLS)}.")

    # ---------------- simulation ----------------
    store = {}
    rows = []
    done = 0
    for kind, seed in problems:
        pb = make_problem(kind, seed)
        xs, rs = solve_traj(pb)
        entry = {"rules": {}}
        for tol in TOLS:
            st = stop_tol(rs, tol)
            entry["rules"][("TOL", tol)] = (st, int(abs(xs[st]) < SUCCESS_TOL), 0)
        for K in PATIENCE_KS:
            st = stop_patience(rs, K)
            entry["rules"][("PATIENCE", K)] = (st, int(abs(xs[st]) < SUCCESS_TOL), 0)
        for B in FIXED_BUDGETS:
            entry["rules"][("FIXED", B)] = (B, int(abs(xs[B]) < SUCCESS_TOL), 0)
        for k_net in KNETS:
            u_tr, sig_sm = sidecar_traces(rs, seed * 10 + 1, k_net, adj)
            for frac in FRACS:
                st_u = stop_rolling(u_tr, frac)
                entry["rules"][("M4R_U", (k_net, frac))] = (
                    st_u, int(abs(xs[st_u]) < SUCCESS_TOL), st_u * k_net * N)
                st_s = stop_rolling(sig_sm, frac)
                entry["rules"][("M4R_SIG", (k_net, frac))] = (
                    st_s, int(abs(xs[st_s]) < SUCCESS_TOL), st_s * k_net * N)
        store[(kind, seed)] = entry
        for (rule, param), (st, ok, extra) in entry["rules"].items():
            rows.append((kind, seed, rule, str(param).replace(",", ";"), st, ok, extra))
        done += 1
        if done % 6 == 0:
            print(f"  [{done}/{len(problems)}] {time.time()-t0:.0f}s")

    # ---------------- choix GLOBAL par regle ----------------
    RULES = [("TOL", TOLS), ("PATIENCE", PATIENCE_KS), ("FIXED", FIXED_BUDGETS),
             ("M4R_U", [(k, f) for k in KNETS for f in FRACS]),
             ("M4R_SIG", [(k, f) for k in KNETS for f in FRACS])]
    keys = sorted(store.keys())
    best = {}
    print("\n=== CHOIX GLOBAL (succes moyen sur les 24 problemes, tie-break cout) ===")
    for rule, params in RULES:
        scored = []
        for p in params:
            oks = [store[k]["rules"][(rule, p)][1] for k in keys]
            costs = [store[k]["rules"][(rule, p)][0] for k in keys]
            scored.append((np.mean(oks), -np.mean(costs), p))
        scored.sort(reverse=True)
        best[rule] = scored[0][2]
        print(f"  {rule:<9} -> param={scored[0][2]}  (succes={scored[0][0]:.3f}, "
              f"cout solveur moyen={-scored[0][1]:.0f} iters)")

    # ---------------- tableau par famille ----------------
    print(f"\n{'regle':<10}{'succes easy':>12}{'succes trap':>12}{'succes tous':>12}"
          f"{'cout iters':>11}{'cout reseau (noeuds-pas)':>26}")
    print("-" * 85)
    agg_rows = []
    for rule, _ in RULES:
        p = best[rule]
        res = {"easy": [], "trap": []}
        costs, extras = [], []
        for (kind, seed) in keys:
            st, ok, extra = store[(kind, seed)]["rules"][(rule, p)]
            res[kind].append(ok)
            costs.append(st)
            extras.append(extra)
        se, st_, sa = np.mean(res["easy"]), np.mean(res["trap"]), np.mean(res["easy"] + res["trap"])
        print(f"{rule:<10}{se:>12.2f}{st_:>12.2f}{sa:>12.2f}{np.mean(costs):>11.0f}"
              f"{np.mean(extras):>26.0f}")
        agg_rows.append((rule, str(p).replace(",", ";"), se, st_, sa,
                         np.mean(costs), np.mean(extras)))

    # ---------------- verdict ----------------
    print("\n=== VERDICT P11 (criteres pre-fixes) ===")

    def okvec(rule):
        p = best[rule]
        return [store[k]["rules"][(rule, p)][1] for k in keys]

    d1, lo1, hi1 = boot_ci_paired(okvec("M4R_SIG"), okvec("TOL"))
    print(f"  1. M4R_SIG - TOL      = {d1:+.3f} CI[{lo1:+.3f},{hi1:+.3f}] -> "
          f"{'side-car bat la tolerance naive' if lo1 > 0 else ('TOL devant' if hi1 < 0 else 'parite')}")
    d1u, lo1u, hi1u = boot_ci_paired(okvec("M4R_U"), okvec("TOL"))
    print(f"     M4R_U   - TOL      = {d1u:+.3f} CI[{lo1u:+.3f},{hi1u:+.3f}]")
    d2, lo2, hi2 = boot_ci_paired(okvec("M4R_SIG"), okvec("PATIENCE"))
    print(f"  2. M4R_SIG - PATIENCE = {d2:+.3f} CI[{lo2:+.3f},{hi2:+.3f}] -> "
          f"{'side-car bat l early-stopping standard' if lo2 > 0 else ('PATIENCE devant' if hi2 < 0 else 'parite (pattern B5b attendu)')}")
    d3, lo3, hi3 = boot_ci_paired(okvec("M4R_SIG"), okvec("FIXED"))
    print(f"  3. M4R_SIG - FIXED    = {d3:+.3f} CI[{lo3:+.3f},{hi3:+.3f}] -> "
          f"{'side-car bat le budget fixe' if lo3 > 0 else ('FIXED devant' if hi3 < 0 else 'parite (pattern B5b attendu)')}")
    ease = [store[k]["rules"][("M4R_SIG", best["M4R_SIG"])][1]
            for k in keys if k[0] == "easy"]
    print(f"  4. Faciles non sacrifies (>=0.90) : M4R_SIG easy = {np.mean(ease):.2f}")
    net_cost = np.mean([store[k]["rules"][("M4R_SIG", best["M4R_SIG"])][2] for k in keys])
    print(f"  5. Cout du side-car (jamais gratuit) : ~{net_cost:.0f} noeuds-pas reseau "
          f"par probleme, EN PLUS du solveur.")
    # 6. le duel B5b : a SUCCES EGAL, qui coute le moins d'iterations solveur ?
    print("  6. Couts (iters solveur) a comparer A SUCCES EGAL :")
    for rule in ["TOL", "PATIENCE", "FIXED", "M4R_U", "M4R_SIG"]:
        p = best[rule]
        oks = [store[k]["rules"][(rule, p)][1] for k in keys]
        costs = [store[k]["rules"][(rule, p)][0] for k in keys]
        print(f"     {rule:<9} succes={np.mean(oks):.2f}  cout={np.mean(costs):>6.0f}")

    # ---------------- CSV ----------------
    with CSV_PATH.open("w", encoding="utf-8") as f:
        f.write("kind,seed,rule,param,stop_iter,success,network_node_steps\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    with AGG_PATH.open("w", encoding="utf-8") as f:
        f.write("rule,best_param_global,success_easy,success_trap,success_all,"
                "mean_stop_iter,mean_network_node_steps\n")
        for r in agg_rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print(f"\n[csv] {CSV_PATH}\n[csv] {AGG_PATH}")

    # ---------------- figure ----------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
        ax = axes[0]
        ax.semilogy(rs_e, label="facile (seed 0)", c="#2ca02c")
        ax.semilogy(rs_t, label="trompeur (seed 0)", c="#d62728")
        for tol in TOLS:
            ax.axhline(tol, ls=":", c="gray", lw=0.6)
        ax.set_xlabel("iteration")
        ax.set_ylabel("residu |f'(x)| (log)")
        ax.set_title("Le piege : le residu s'effondre au plateau puis remonte")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax = axes[1]
        u_tr, sig_sm = sidecar_traces(rs_t, 1, best["M4R_SIG"][0], adj)
        ax2 = ax.twinx()
        ax.plot(sig_sm, c="#d62728", label="mean|Lv| lisse (side-car)")
        ax2.plot(u_tr, c="#9467bd", label="u_mean (side-car)")
        ax.set_xlabel("iteration solveur")
        ax.set_ylabel("mean|Lv| lisse", color="#d62728")
        ax2.set_ylabel("u_mean", color="#9467bd")
        ax.set_title("Les horloges du side-car sur le trompeur (seed 0)")
        ax.grid(alpha=0.3)
        ax = axes[2]
        names = [r[0] for r in agg_rows]
        for i, (nm, key, col) in enumerate([("easy", 2, "#2ca02c"), ("trap", 3, "#d62728")]):
            ax.bar(np.arange(len(names)) + (i - 0.5) * 0.35,
                   [r[key] for r in agg_rows], width=0.35, color=col, label=nm)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=7)
        ax.set_ylabel("taux de succes")
        ax.set_title("Succes par famille (hyperparametres globaux)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
        fig.suptitle("P11 -- l'horloge M4R comme critere d'arret d'un solveur tiers",
                     fontsize=11)
        plt.tight_layout()
        plt.savefig(PNG_PATH, dpi=140)
        print(f"[png] {PNG_PATH}")
    except Exception as e:
        print(f"[png] skipped: {e}")

    print(f"\nWall time: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
