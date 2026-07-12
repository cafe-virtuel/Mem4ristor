#!/usr/bin/env python3
"""
P6(b) -- LA COUCHE D'ABSTENTION : construire le compas, apres avoir repare la boussole.
========================================================================================
Cree : 2026-07-12 (Claude Fable 5, L'Ingenieur) -- suite directe de P6(a)
(doubt_calibration_poc.py, commit 04ea50a) : la piste P6 du legs = la Couche
d'Abstention Calibree de Julien (PEPIT_LOG 11/06/2026).

CE QUE P6(a) AVAIT TROUVE : u n'est pas naivement calibre, il est INVERSE
(r=-0.29 a B=800 : u haut -> reussite) ; le capteur brut |Lv| passe le critere
a decision precoce ; et un COLLATERAL non tranche -- l'accuracy globale se
degradait avec le budget (89 -> 72 -> 36 %%).

LE COLLATERAL EST TRANCHE (diagnostic du 12/07, decompose par groupe de noeuds) :
c'est un ARTEFACT DE READOUT, pas une sur-deliberation du reseau.
  1. La reponse FHN a un courant constant est ADAPTATIVE : depolarisation
     transitoire massive (les leurres a +1.0 montent de +3.2/noeud) puis
     rebond SOUS la baseline en ~300-400 pas (la variable lente w surcompense).
     Le piege B1d ne trompe que pendant son transitoire.
  2. Le signal verite en regime (~-0.03 sur d_all) est du meme ordre que le
     bruit de decorrelation net/ref (~+/-0.05) : la lecture INSTANTANEE de
     P6(a) produisait des labels quasi aleatoires selon l'instant (les essais
     LOYAUX tombaient a 50 %% a B=800 -- impossible si le readout etait sain).
REVISION STRUCTURELLE (leçon P12 le meme jour : la diode integre) : les labels
sont reconstruits sur un readout LISSE (moyenne des W_READ=200 derniers pas de
d(t) avant l'instant de decision B). La physique adaptative reste (c'est la
tache) ; le bruit de decorrelation se moyenne.

CE QUE P6(b) MESURE (criteres pre-fixes avant de voir un chiffre) :
  - L'inversion de u tient-elle sur des labels PROPRES ? (comparaison loyale :
    conf_u_naif = 1-u vs conf_u_inv = u, memes essais, memes labels)
  - Les signaux d'abstention candidats, tous causaux, lus a l'instant B :
      conf_u_naif = 1 - u_mean(B)         (le naif de P6a)
      conf_u_inv  = u_mean(B)             (le compas INVERSE : le conflit
                                           accompagne la verite qui resiste)
      conf_sig    = 1 - sig(B)/pic_causal (le capteur |Lv| norme de P6a)
      conf_tcons  = min(1, t_consensus/B) (la regle qualitative de Julien :
                                           un consensus venu VITE est suspect ;
                                           t_consensus = 1er t ou sig < 0.3*pic
                                           causal ; jamais atteint -> 1.0)
      conf_stab   = stabilite de la decision lissee sur les 400 derniers pas
                    (une decision qui vient de flipper est suspecte)
      COMPOSITE   = regression logistique sur les 5 signaux standardises,
                    VALIDATION CROISEE GROUPEE PAR SEED (6 folds x 4 seeds --
                    les 5 t_pulse d'un meme seed partagent le masque de noeuds,
                    les separer fuiterait). Score = probabilite predite.
  - Metriques : r point-biserial, reliability (quantiles, IC Wilson),
    courbe risque-couverture, gain d'abstention @50%% et @70%% de couverture.
  - CRITERE DE SUCCES (pre-fixe) : un compas est UTILISABLE si r_pb > 0.15 ET
    gain @50%% > +3 pts, sur les labels propres. Le composite doit battre le
    meilleur signal simple pour justifier son cout, sinon on garde le simple.

Tache et essais : IDENTIQUES a P6(a) (b1d.make_deceptive, memes seeds, 5
t_pulse x 24 seeds = 120 essais, T_SIM=1600, budgets {400, 800, 1600}).

Statut : exploratoire, hors preprint, coeur non touche.
Sorties : figures/p6b_abstention_poc{_raw,_agg}.csv + .png
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
from mem4ristor.topology import Mem4Network  # noqa: E402
import deceptive_task_poc as b1d  # noqa: E402

RAW_CSV = ROOT / "figures" / "p6b_abstention_poc_raw.csv"
AGG_CSV = ROOT / "figures" / "p6b_abstention_poc_agg.csv"
PNG = ROOT / "figures" / "p6b_abstention_poc.png"

SIDE, N = 10, 100
T_SIM = 1600
BUDGETS = [400, 800, 1600]
T_PULSES = [0, 150, 350, 700, 1200]
SEEDS = list(range(24))
W_READ = 200                    # lissage du readout (revision structurelle)
W_STAB = 400                    # fenetre de stabilite de la decision
CONS_FRAC = 0.30                # seuil de consensus pour t_consensus
N_BINS = 5
N_FOLDS = 6
SIGNALS = ["conf_u_naif", "conf_u_inv", "conf_sig", "conf_tcons", "conf_stab"]


def simulate_record(adj, stim_on, stim_off, seed, t_pulse):
    """Trajectoires sig=|Lv| moyen, u_mean, d (readout differentiel brut)."""
    net = Mem4Network(size=SIDE, heretic_ratio=0.0, seed=seed, adjacency_matrix=adj)
    ref = Mem4Network(size=SIDE, heretic_ratio=0.0, seed=seed, adjacency_matrix=adj)
    L = net.L
    zero = np.zeros(N)
    sig = np.empty(T_SIM)
    u_mean = np.empty(T_SIM)
    d_raw = np.empty(T_SIM)
    for t in range(T_SIM):
        stim = stim_on if t < t_pulse else stim_off
        net.step(I_stimulus=stim)
        ref.step(I_stimulus=zero)
        v = net.model.v
        sig[t] = float(np.mean(np.abs(L @ v)))
        u_mean[t] = float(np.mean(net.model.u))
        d_raw[t] = float(np.mean(v) - np.mean(ref.model.v))
    return sig, u_mean, d_raw


def rolling_mean(x, w):
    csum = np.cumsum(x)
    out = np.empty_like(x)
    for t in range(len(x)):
        lo = max(0, t - w + 1)
        tot = csum[t] - (csum[lo - 1] if lo > 0 else 0.0)
        out[t] = tot / (t - lo + 1)
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((center - half) / denom, (center + half) / denom)


def point_biserial(conf, correct):
    conf = np.asarray(conf, float)
    correct = np.asarray(correct, float)
    if conf.std() < 1e-12 or correct.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(conf, correct)[0, 1])


def risk_coverage(conf, correct):
    order = np.argsort(-np.asarray(conf))
    cs = np.asarray(correct, float)[order]
    n = len(cs)
    cov, acc = [], []
    for k in range(1, 11):
        m = max(1, int(round(n * k / 10)))
        cov.append(m / n)
        acc.append(float(cs[:m].mean()))
    return cov, acc


def logistic_cv(X, y, groups, n_folds=N_FOLDS, lr=0.3, iters=400):
    """Regression logistique maison, CV groupee : score hors-fold pour chaque
    essai. X standardise DANS le fold d'entrainement (pas de fuite)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    uniq = np.unique(groups)
    rng = np.random.RandomState(20260712)
    perm = rng.permutation(len(uniq))
    folds = np.array_split(uniq[perm], n_folds)
    scores = np.full(len(y), np.nan)
    for fold in folds:
        test = np.isin(groups, fold)
        train = ~test
        mu = X[train].mean(axis=0)
        sd = X[train].std(axis=0)
        sd[sd < 1e-12] = 1.0
        Xtr = (X[train] - mu) / sd
        Xte = (X[test] - mu) / sd
        Xtr = np.hstack([np.ones((Xtr.shape[0], 1)), Xtr])
        Xte = np.hstack([np.ones((Xte.shape[0], 1)), Xte])
        w = np.zeros(Xtr.shape[1])
        ytr = y[train]
        for _ in range(iters):
            p = 1.0 / (1.0 + np.exp(-Xtr @ w))
            grad = Xtr.T @ (p - ytr) / len(ytr)
            w -= lr * grad
        scores[test] = 1.0 / (1.0 + np.exp(-(Xte @ w)))
    return scores


def main() -> int:
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    rows = []
    n_total = len(T_PULSES) * len(SEEDS)
    done = 0
    print(f"P6(b) -- {n_total} essais (memes que P6a), readout LISSE W={W_READ}, "
          f"budgets {BUDGETS}")
    for t_pulse in T_PULSES:
        for seed in SEEDS:
            rng = np.random.RandomState(3000 + seed)
            adj, stim_on, stim_off, dstar = b1d.make_deceptive(rng)
            sig, u_mean, d_raw = simulate_record(adj, stim_on, stim_off,
                                                 seed * 10 + 1, t_pulse)
            d_sm = rolling_mean(d_raw, W_READ)
            dec_sm = np.where(d_sm >= 0, 1, -1)
            # pic causal et t_consensus (roulants, comme P12)
            peak_run = np.maximum.accumulate(sig)
            below = sig < CONS_FRAC * peak_run
            below[:30] = False                       # warmup court (comme B1d)
            t_cons = int(np.argmax(below)) + 1 if below.any() else T_SIM + 1
            row = {"t_pulse": t_pulse, "seed": seed, "dstar": dstar,
                   "t_consensus": t_cons}
            for B in BUDGETS:
                i = B - 1
                row[f"correct_B{B}"] = int(dec_sm[i] == dstar)
                row[f"conf_u_naif_B{B}"] = 1.0 - float(u_mean[i])
                row[f"conf_u_inv_B{B}"] = float(u_mean[i])
                row[f"conf_sig_B{B}"] = 1.0 - min(1.0, float(sig[i] / max(peak_run[i], 1e-12)))
                row[f"conf_tcons_B{B}"] = min(1.0, t_cons / B)
                lo = max(0, i - W_STAB + 1)
                row[f"conf_stab_B{B}"] = float(np.mean(dec_sm[lo:i + 1] == dec_sm[i]))
            rows.append(row)
            done += 1
        print(f"  t_pulse={t_pulse:>5} fait  [{done}/{n_total}, {time.time()-t0:.0f}s]")

    # ---------------- analyses ----------------
    agg_lines = []
    results = {}
    groups = np.array([r["seed"] for r in rows])
    print("\n=== ACCURACY DE BASE (labels au readout lisse) ===")
    for B in BUDGETS:
        correct = np.array([r[f"correct_B{B}"] for r in rows])
        by_tp = {tp: np.mean([r[f"correct_B{B}"] for r in rows if r["t_pulse"] == tp])
                 for tp in T_PULSES}
        detail = "  ".join(f"tp{tp}:{100*a:.0f}%" for tp, a in by_tp.items())
        print(f"  B={B:>5}: global {100*correct.mean():5.1f}%   ({detail})")

    print("\n=== SIGNAUX (r point-biserial + abstention) ===")
    for B in BUDGETS:
        correct = np.array([r[f"correct_B{B}"] for r in rows])
        base_acc = correct.mean()
        print(f"\n-- B={B} (base {100*base_acc:.1f}%) --")
        X = np.column_stack([[r[f"{s}_B{B}"] for r in rows] for s in SIGNALS])
        comp = logistic_cv(X, correct, groups)
        for name, conf in ([(s, np.array([r[f"{s}_B{B}"] for r in rows]))
                            for s in SIGNALS] + [("COMPOSITE_CV", comp)]):
            rpb = point_biserial(conf, correct)
            cov, acc_rc = risk_coverage(conf, correct)
            acc50, acc70 = acc_rc[4], acc_rc[6]
            results[(B, name)] = (rpb, base_acc, acc50, acc70, cov, acc_rc)
            print(f"  {name:<13} r_pb={rpb:+.3f}  abst@50%: {100*base_acc:.1f}->"
                  f"{100*acc50:.1f}% ({100*(acc50-base_acc):+.1f})  "
                  f"@70%: ->{100*acc70:.1f}% ({100*(acc70-base_acc):+.1f})")
            agg_lines.append((B, name, f"{rpb:+.4f}", f"{base_acc:.4f}",
                              f"{acc50:.4f}", f"{acc70:.4f}"))

    # ---------------- verdict ----------------
    print("\n" + "=" * 76)
    print("VERDICT P6(b) -- criteres pre-fixes : utilisable si r_pb>0.15 ET "
          "gain@50% > +3 pts")
    print("=" * 76)
    for B in BUDGETS:
        usable = []
        for name in SIGNALS + ["COMPOSITE_CV"]:
            rpb, base, acc50, acc70, _, _ = results[(B, name)]
            if rpb > 0.15 and (acc50 - base) > 0.03:
                usable.append((acc50 - base, rpb, name))
        usable.sort(reverse=True)
        if usable:
            g, r, n = usable[0]
            print(f"  B={B:>5}: UTILISABLE(S) : "
                  + ", ".join(f"{n}(+{100*g:.1f}pts)" for g, r, n in usable))
        else:
            print(f"  B={B:>5}: AUCUN compas ne passe le critere.")
    ru_naif = results[(800, "conf_u_naif")][0]
    ru_inv = results[(800, "conf_u_inv")][0]
    print(f"\n  Question P6a re-posee sur labels propres (B=800) : "
          f"r(1-u)={ru_naif:+.3f} vs r(u)={ru_inv:+.3f}")
    if ru_inv > 0.15 and ru_naif < 0:
        print("  -> l'INVERSION TIENT : le doute residuel accompagne la justesse "
              "(il signale le conflit, pas l'erreur).")
    elif ru_naif > 0.15:
        print("  -> l'inversion de P6a NE TIENT PAS sur labels propres : elle "
              "etait un artefact du readout instantane.")
    else:
        print("  -> ni u ni son inverse ne sont un compas fiable seul sur "
              "labels propres.")

    # ---------------- sorties ----------------
    with RAW_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with AGG_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["budget", "signal", "r_pb", "base_acc", "acc_cov50", "acc_cov70"])
        w.writerows(agg_lines)
    print(f"\n[csv] {RAW_CSV}\n[csv] {AGG_CSV}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
        palette = {"conf_u_naif": "#9467bd", "conf_u_inv": "#d62728",
                   "conf_sig": "#1f77b4", "conf_tcons": "#2ca02c",
                   "conf_stab": "#ff7f0e", "COMPOSITE_CV": "#000000"}
        for ax, B in zip(axes, BUDGETS):
            for name in SIGNALS + ["COMPOSITE_CV"]:
                _, base, _, _, cov, acc_rc = results[(B, name)]
                ax.plot([100 * c for c in cov], [100 * a for a in acc_rc],
                        "o-", ms=3, lw=1.4, color=palette[name], label=name)
            ax.axhline(100 * results[(B, "conf_sig")][1], ls=":", c="gray",
                       label="base (sans abstention)")
            ax.set_xlabel("couverture (%)")
            ax.set_ylabel("accuracy sur les gardees (%)")
            ax.set_title(f"B={B}")
            ax.grid(alpha=0.3)
            if B == BUDGETS[0]:
                ax.legend(fontsize=6.5)
        fig.suptitle("P6(b) -- Couche d'Abstention : risque-couverture des compas "
                     f"(labels au readout lisse W={W_READ}, {n_total} essais)",
                     fontsize=11)
        plt.tight_layout()
        plt.savefig(PNG, dpi=140)
        print(f"[png] {PNG}")
    except Exception as e:
        print(f"[png] skipped: {e}")

    print(f"\nWall time: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
