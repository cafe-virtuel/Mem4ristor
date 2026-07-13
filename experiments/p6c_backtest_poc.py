#!/usr/bin/env python3
"""
P6(c) -- LE BACKTEST 0 EUR : la recette de l'abstention transporte-t-elle
vers un domaine aux statistiques DIFFERENTES ?
=========================================================================
Cree : 2026-07-13 (Claude Sonnet 5, L'Ingenieur) -- piste du legs de Fable
(docs/PISTES_POUR_LA_SUITE_2026-07-12.md, section I, P6, volet (c)).

TRACE : `PEPIT_LOG.md` ligne 66 (11/06/2026, idee de Julien) -- "ne decide
pas, decide quand ne pas decider" (paris sportifs preenregistres, reponses
LLM, investissement virtuel). P6(a)/P6(b) (12/07) ont deja construit et
valide le COMPAS COMPOSITE (u, |Lv|, t_consensus, stabilite, regression
logistique en CV groupee) sur LA tache B1d -- +38.3 pts a B=400, +25.0 pts
a B=800 @50% couverture. Reste EXPLICITEMENT non fait : le backtest sur un
VRAI domaine hors B1d, avec l'avertissement de la piste elle-meme : "le
composite doit etre RE-APPRIS par domaine -- ne transporter que la recette
(signaux + CV), pas les poids."

DOMAINE CHOISI (0 euro, aucune donnee externe) : investissement virtuel
synthetique -- decision LONG/SHORT sur un marche a "faux breakout" (leurre
de momentum, nombreux et fort, MAIS eteint apres une duree INCONNUE)
suivi d'une tendance vraie (peu de capteurs, faible, mais persistante).
C'est structurellement la MEME FAMILLE que B1d (leurre puis verite qui
resiste), et c'est volontaire -- le point n'est PAS de changer de
mecanisme, c'est de changer les STATISTIQUES : contrairement a B1d
(N_DISTRACT/N_TRUE/E_TRUE/E_DISTRACT FIXES), ici CHAQUE essai tire ses
propres parametres de marche (nombre de capteurs, forces, duree du faux
breakout) -- un regime GENUINEMENT different a chaque tirage, comme un
vrai backtest sur des episodes de marche varies.

PROTOCOLE :
  - N=100 lattice, budget de decision B=800 (zone de signal la plus forte,
    P6a/P6b), 60 essais (parametres de marche tires independamment).
  - 5 signaux IDENTIQUES a P6b (conf_u_naif, conf_u_inv, conf_sig,
    conf_tcons, conf_stab), COMPOSITE = regression logistique en CV
    groupee (recette reutilisee TELLE QUELLE depuis p6b_abstention_poc.py).
  - CRITERE DE SUCCES : IDENTIQUE a P6b (r_pb>0.15 ET gain@50%>+3pts) --
    meme barre, domaine different.
  - CONTROLE "les poids ne transportent pas" : le MEILLEUR signal SIMPLE
    trouve sur B1d (conf_u_inv, "le doute accompagne la verite qui
    resiste") est re-teste ICI SANS modification -- s'il echoue alors que
    le COMPOSITE reussit, ca illustre concretement que la recette
    (signaux+CV) generalise mieux qu'un signal fige.

Sorties : figures/p6c_backtest_poc.csv + .png
Statut : exploratoire, hors preprint, coeur non touche.
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
sys.path.insert(0, str(ROOT / "experiments"))
from mem4ristor.topology import Mem4Network  # noqa: E402
from mem4ristor.graph_utils import make_lattice_adj  # noqa: E402
import p6b_abstention_poc as p6b  # noqa: E402  (recette : logistic_cv, rolling_mean, wilson, SIGNALS)

CSV_PATH = ROOT / "figures" / "p6c_backtest_poc.csv"
PNG_PATH = ROOT / "figures" / "p6c_backtest_poc.png"

SIDE, N = 10, 100
T_SIM = 1600
BUDGET = 800
TRIALS = 60
W_READ = 200
W_STAB = 400
CONS_FRAC = 0.30
SIGNALS = p6b.SIGNALS   # ["conf_u_naif", "conf_u_inv", "conf_sig", "conf_tcons", "conf_stab"]


def make_market_episode(rng):
    """Un episode de marche : parametres TIRES (pas fixes comme B1d) --
    regime different a chaque essai, comme un vrai backtest."""
    adj = make_lattice_adj(SIDE, periodic=True)
    dstar = rng.choice([-1, 1])
    n_fake = rng.integers(15, 36)
    n_trend = rng.integers(8, 21)
    e_fake = rng.uniform(0.7, 1.3)
    e_trend = rng.uniform(0.4, 0.8)
    t_fake_end = rng.uniform(100, 1400)      # duree du faux breakout INCONNUE
    nodes = rng.choice(N, size=n_fake + n_trend, replace=False)
    fake_nodes, trend_nodes = nodes[:n_fake], nodes[n_fake:]
    stim_on = np.zeros(N)
    stim_on[fake_nodes] = -dstar * e_fake
    stim_on[trend_nodes] = dstar * e_trend
    stim_off = np.zeros(N)
    stim_off[trend_nodes] = dstar * e_trend
    return adj, stim_on, stim_off, dstar, t_fake_end


def simulate_record(adj, stim_on, stim_off, seed, t_fake_end):
    net = Mem4Network(size=SIDE, heretic_ratio=0.0, seed=seed, adjacency_matrix=adj)
    ref = Mem4Network(size=SIDE, heretic_ratio=0.0, seed=seed, adjacency_matrix=adj)
    L = net.L
    zero = np.zeros(N)
    sig = np.empty(T_SIM); u_mean = np.empty(T_SIM); d_raw = np.empty(T_SIM)
    for t in range(T_SIM):
        stim = stim_on if t < t_fake_end else stim_off
        net.step(I_stimulus=stim); ref.step(I_stimulus=zero)
        v = net.model.v
        sig[t] = float(np.mean(np.abs(L @ v)))
        u_mean[t] = float(np.mean(net.model.u))
        d_raw[t] = float(np.mean(v) - np.mean(ref.model.v))
    return sig, u_mean, d_raw


def main() -> int:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"P6(c) -- backtest 0 euro, {TRIALS} episodes de marche synthetiques "
          f"(parametres tires par essai, B={BUDGET})")

    rows = []
    for trial in range(TRIALS):
        rng = np.random.default_rng(5000 + trial)
        adj, stim_on, stim_off, dstar, t_fake_end = make_market_episode(rng)
        sig, u_mean, d_raw = simulate_record(adj, stim_on, stim_off, trial * 7 + 3, t_fake_end)
        d_sm = p6b.rolling_mean(d_raw, W_READ)
        i = BUDGET - 1
        dec_sm = 1 if d_sm[i] >= 0 else -1
        peak_run = np.maximum.accumulate(sig)
        below = sig < CONS_FRAC * peak_run
        below[:30] = False
        t_cons = int(np.argmax(below)) + 1 if below.any() else T_SIM + 1
        lo = max(0, i - W_STAB + 1)
        row = {
            "trial": trial, "dstar": dstar, "t_fake_end": t_fake_end,
            "correct": int(dec_sm == dstar),
            "conf_u_naif": 1.0 - float(u_mean[i]),
            "conf_u_inv": float(u_mean[i]),
            "conf_sig": 1.0 - min(1.0, float(sig[i] / max(peak_run[i], 1e-12))),
            "conf_tcons": min(1.0, t_cons / BUDGET),
            "conf_stab": float(np.mean((d_sm[lo:i + 1] >= 0) == (dec_sm >= 0))),
        }
        rows.append(row)
        if (trial + 1) % 15 == 0:
            print(f"  {trial+1}/{TRIALS} episodes  [{time.time()-t0:.0f}s]")

    correct = np.array([r["correct"] for r in rows])
    win_rate_always = correct.mean()
    print(f"\n=== ALWAYS-TRADE (100% couverture) : win-rate = {100*win_rate_always:.1f}% "
          f"({int(correct.sum())}/{TRIALS}) ===")

    X = np.column_stack([[r[s] for r in rows] for s in SIGNALS])
    groups = np.arange(TRIALS)   # 1 episode = 1 groupe (pas de seed partagee entre episodes ici)
    comp = p6b.logistic_cv(X, correct.astype(float), groups, n_folds=6)

    def risk_coverage_gain(conf, label):
        cov, acc_rc = p6b.risk_coverage(conf, correct)
        rpb = p6b.point_biserial(conf, correct)
        acc50, acc70 = acc_rc[4], acc_rc[6]
        print(f"  {label:<16} r_pb={rpb:+.3f}  win-rate@50%: {100*win_rate_always:.1f}->"
              f"{100*acc50:.1f}% ({100*(acc50-win_rate_always):+.1f})  "
              f"@70%: ->{100*acc70:.1f}% ({100*(acc70-win_rate_always):+.1f})")
        return rpb, acc50, acc70

    print("\n=== SIGNAUX (recette P6b reutilisee, poids RE-APPRIS ici) ===")
    results = {}
    for name, conf in ([(s, np.array([r[s] for r in rows])) for s in SIGNALS]
                        + [("COMPOSITE_CV", comp)]):
        rpb, acc50, acc70 = risk_coverage_gain(conf, name)
        results[name] = (rpb, acc50, acc70)

    print("\n" + "=" * 84)
    print("VERDICT P6(c) -- meme critere que P6b : utilisable si r_pb>0.15 ET gain@50%>+3pts")
    print("=" * 84)
    usable = []
    for name, (rpb, acc50, acc70) in results.items():
        if rpb > 0.15 and (acc50 - win_rate_always) > 0.03:
            usable.append((acc50 - win_rate_always, rpb, name))
    usable.sort(reverse=True)
    if usable:
        print("  UTILISABLE(S) : " + ", ".join(f"{n}(+{100*g:.1f}pts, r={r:+.2f})" for g, r, n in usable))
    else:
        print("  AUCUN signal ne passe le critere sur ce domaine.")

    comp_rpb, comp_acc50, _ = results["COMPOSITE_CV"]
    naif_rpb, naif_acc50, _ = results["conf_u_inv"]
    print(f"\n  Le signal individuel gagnant sur B1d (conf_u_inv, \"le doute accompagne\n"
          f"  la verite qui resiste\") ICI SANS MODIFICATION : r_pb={naif_rpb:+.3f}, "
          f"gain@50%={100*(naif_acc50-win_rate_always):+.1f} pts")
    print(f"  Le COMPOSITE RE-APPRIS sur ce domaine : r_pb={comp_rpb:+.3f}, "
          f"gain@50%={100*(comp_acc50-win_rate_always):+.1f} pts")
    if comp_rpb > 0.15 and (comp_acc50 - win_rate_always) > 0.03:
        if naif_rpb <= 0.15 or (naif_acc50 - win_rate_always) <= 0.03:
            print("\n  -> LA RECETTE TRANSPORTE, LE SIGNAL FIGE NON : le composite RE-APPRIS "
                  "reussit ici alors que le meilleur signal individuel de B1d, applique tel "
                  "quel, ne passe pas le critere -- exactement la lecon annoncee par la piste.")
        else:
            print("\n  -> LES DEUX TRANSPORTENT : le composite ET le signal simple de B1d "
                  "fonctionnent ici -- le domaine n'est peut-etre pas assez different pour "
                  "trancher la question 'recette vs poids'.")
    else:
        print("\n  -> LA RECETTE NE TRANSPORTE PAS NON PLUS ICI : resultat negatif honnete -- "
              "le compas composite reste specifique a B1d, pas encore un outil general.")

    with CSV_PATH.open("w", encoding="utf-8") as f:
        f.write("trial,dstar,t_fake_end,correct," + ",".join(SIGNALS) + "\n")
        for r in rows:
            f.write(f"{r['trial']},{r['dstar']},{r['t_fake_end']:.1f},{r['correct']}," +
                    ",".join(f"{r[s]:.4f}" for s in SIGNALS) + "\n")
    print(f"\n[csv] {CSV_PATH}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
        ax = axes[0]
        palette = {"conf_u_naif": "#9467bd", "conf_u_inv": "#d62728", "conf_sig": "#1f77b4",
                   "conf_tcons": "#2ca02c", "conf_stab": "#ff7f0e", "COMPOSITE_CV": "#000000"}
        for name in list(SIGNALS) + ["COMPOSITE_CV"]:
            conf = comp if name == "COMPOSITE_CV" else np.array([r[name] for r in rows])
            cov, acc_rc = p6b.risk_coverage(conf, correct)
            ax.plot([100 * c for c in cov], [100 * a for a in acc_rc], "o-", ms=3,
                     lw=1.4 if name == "COMPOSITE_CV" else 1.0, color=palette[name], label=name)
        ax.axhline(100 * win_rate_always, ls=":", c="gray", label="ALWAYS-TRADE")
        ax.set_xlabel("couverture (%)"); ax.set_ylabel("win-rate sur les positions gardees (%)")
        ax.set_title(f"P6(c) -- risque/couverture, backtest 0 euro ({TRIALS} episodes)")
        ax.legend(fontsize=6.5); ax.grid(alpha=0.3)
        ax = axes[1]
        names = list(SIGNALS) + ["COMPOSITE_CV"]
        gains = [100 * (results[n][1] - win_rate_always) for n in names]
        ax.bar(names, gains, color=[palette[n] for n in names])
        ax.axhline(3, ls="--", c="k", alpha=0.5, label="seuil de succes (+3pts)")
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("gain @50% couverture (pts)")
        ax.set_title("Gain d'abstention par signal"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
        fig.suptitle("P6(c) -- le compas composite transporte-t-il vers un nouveau domaine ?", fontsize=11)
        plt.tight_layout()
        plt.savefig(PNG_PATH, dpi=140)
        print(f"[png] {PNG_PATH}")
    except Exception as e:
        print(f"[png] skipped: {e}")

    print(f"\nWall time: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
