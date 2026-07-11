#!/usr/bin/env python3
"""
P6(a) -- CALIBRATION DU DOUTE : l'IA sait-elle qu'elle ne sait pas ?
=====================================================================
Cree : 2026-07-12 (Claude Fable 5, L'Ingenieur) -- premiere marche de la piste
P6 du legs (docs/PISTES_POUR_LA_SUITE_2026-07-12.md) = la Couche d'Abstention
Calibree de Julien (PEPIT_LOG 11/06/2026, jamais testee). Avant de construire
un garde-fou qui "decide quand ne pas decider", il faut verifier le prerequis
que personne n'a jamais mesure : LE DOUTE DU RESEAU EST-IL CALIBRE ? Quand il
est haut au moment de la decision, le reseau se trompe-t-il vraiment plus
souvent ?

PROTOCOLE (concu pour etre loyal envers la question, pas envers la reponse) :
  - Tache : la decision trompeuse de B1d (deceptive_task_poc.py, protocole
    REPRIS A L'IDENTIQUE : leurre nombreux+pulse vs verite persistante,
    readout differentiel vs reseau de reference stim=0). On y AJOUTE des
    essais LOYAUX (t_pulse=0 : verite seule, decision facile) pour avoir un
    eventail de difficultes -- sans variete, pas de calibration mesurable.
  - Confiance mesuree A BUDGET FIXE, PAS a l'arret adaptatif : la regle
    d'arret du doute s'arrete PAR CONSTRUCTION quand sig ~ 0.30*pic, donc le
    doute residuel a l'arret est quasi constant -- le mesurer la serait un
    artefact circulaire. A budget fixe B, les essais faciles ont un doute
    retombe, les essais pieges une tension residuelle : c'est LA le signal.
  - Deux signaux candidats, mesures aux memes instants :
      conf_u   = 1 - u_mean(B)          (la variable de doute du COEUR)
      conf_sig = 1 - sig(B)/pic         (le capteur de desaccord |Lv| norme)
  - Trois budgets de decision B in {400, 800, 1600} : a budget court les
    pieges longs sont encore faux (le signal doit le dire), a budget long
    presque tout est juste (la calibration doit suivre).
  - Mesures : reliability diagram (5 bins par quantiles, accuracy + IC
    Wilson), correlation point-biseriale conf<->correct, et la courbe
    RISQUE-COUVERTURE de la prediction selective (garder les k% les plus
    confiants ; si le signal est bon, l'accuracy monte quand la couverture
    baisse -- c'est exactement l'Abstention Calibree).

PREDICTION ecrite avant de lancer (garde-fou du 07/07) : conf_u devrait etre
positivement liee a la justesse (monotone). Si elle est PLATE, u n'est pas
calibre et l'Abstention Calibree exigera un signal composite -- resultat
rapporte tel quel dans les deux cas (un garde-fou mal calibre est pire que
pas de garde-fou).

Statut : exploratoire, hors preprint, coeur non touche.
Sorties : figures/doubt_calibration_poc{_raw,_agg}.csv + .png
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
import deceptive_task_poc as b1d  # noqa: E402  (make_deceptive : protocole identique)

RAW_CSV = ROOT / "figures" / "doubt_calibration_poc_raw.csv"
AGG_CSV = ROOT / "figures" / "doubt_calibration_poc_agg.csv"
PNG = ROOT / "figures" / "doubt_calibration_poc.png"

SIDE, N = 10, 100
T_SIM = 1600
BUDGETS = [400, 800, 1600]
T_PULSES = [0, 150, 350, 700, 1200]     # 0 = essai LOYAL (verite seule)
SEEDS = list(range(24))                  # 5 x 24 = 120 essais
N_BINS = 5
WARMUP = 30


def simulate_record(adj, stim_on, stim_off, seed, t_pulse):
    """Comme b1d.simulate mais enregistre AUSSI u_mean(t). Budget T_SIM."""
    net = Mem4Network(size=SIDE, heretic_ratio=0.0, seed=seed, adjacency_matrix=adj)
    ref = Mem4Network(size=SIDE, heretic_ratio=0.0, seed=seed, adjacency_matrix=adj)
    L = net.L
    zero = np.zeros(N)
    sig = np.empty(T_SIM)
    u_mean = np.empty(T_SIM)
    dec = np.empty(T_SIM, dtype=int)
    for t in range(T_SIM):
        stim = stim_on if t < t_pulse else stim_off
        net.step(I_stimulus=stim)
        ref.step(I_stimulus=zero)
        v = net.model.v
        sig[t] = float(np.mean(np.abs(L @ v)))
        u_mean[t] = float(np.mean(net.model.u))
        d = float(np.mean(v) - np.mean(ref.model.v))
        dec[t] = 1 if d >= 0 else -1
    return sig, u_mean, dec


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
    """Accuracy sur les k% les plus confiants, k = 10..100."""
    order = np.argsort(-np.asarray(conf))
    correct_sorted = np.asarray(correct, float)[order]
    cov, acc = [], []
    n = len(correct_sorted)
    for k in range(1, 11):
        m = max(1, int(round(n * k / 10)))
        cov.append(m / n)
        acc.append(float(correct_sorted[:m].mean()))
    return cov, acc


def main() -> int:
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    rows = []
    n_total = len(T_PULSES) * len(SEEDS)
    done = 0
    print(f"Calibration du doute -- {n_total} essais (5 t_pulse x 24 seeds), "
          f"T_SIM={T_SIM}, budgets {BUDGETS}")
    for t_pulse in T_PULSES:
        for seed in SEEDS:
            rng = np.random.RandomState(3000 + seed)          # meme convention que B1d
            adj, stim_on, stim_off, dstar = b1d.make_deceptive(rng)
            sig, u_mean, dec = simulate_record(adj, stim_on, stim_off,
                                               seed * 10 + 1, t_pulse)
            peak = float(np.max(sig[:WARMUP + 20]))
            row = {"t_pulse": t_pulse, "seed": seed, "dstar": dstar}
            for B in BUDGETS:
                row[f"correct_B{B}"] = int(dec[B - 1] == dstar)
                row[f"conf_u_B{B}"] = 1.0 - float(u_mean[B - 1])
                row[f"conf_sig_B{B}"] = 1.0 - min(1.0, float(sig[B - 1]) / max(peak, 1e-12))
            rows.append(row)
            done += 1
        print(f"  t_pulse={t_pulse:>5} fait  [{done}/{n_total}, {time.time()-t0:.0f}s]")

    # ---------------- analyses ----------------
    agg_lines = []
    print("\n=== RELIABILITY (bins par quantiles de confiance, acc + IC Wilson) ===")
    results = {}
    for B in BUDGETS:
        correct = np.array([r[f"correct_B{B}"] for r in rows])
        base_acc = correct.mean()
        print(f"\n-- budget de decision B={B} (accuracy globale {100*base_acc:.1f}%) --")
        for signal in ["conf_u", "conf_sig"]:
            conf = np.array([r[f"{signal}_B{B}"] for r in rows])
            rpb = point_biserial(conf, correct)
            qs = np.quantile(conf, np.linspace(0, 1, N_BINS + 1))
            qs[0] -= 1e-9
            accs = []
            print(f"  {signal:<9} r_pb={rpb:+.3f}   bins (conf croissante) :")
            for i in range(N_BINS):
                mask = (conf > qs[i]) & (conf <= qs[i + 1])
                nb = int(mask.sum())
                kb = int(correct[mask].sum())
                lo, hi = wilson(kb, nb)
                acc_b = kb / nb if nb else float("nan")
                accs.append(acc_b)
                print(f"    bin{i+1} conf({qs[i]:+.3f},{qs[i+1]:+.3f}] n={nb:>3} "
                      f"acc={100*acc_b:5.1f}% IC[{100*lo:.0f},{100*hi:.0f}]")
                agg_lines.append((B, signal, i + 1, nb, f"{acc_b:.4f}", f"{lo:.4f}", f"{hi:.4f}"))
            mono = all(accs[i] <= accs[i + 1] + 0.02 for i in range(len(accs) - 1))
            cov, acc_rc = risk_coverage(conf, correct)
            acc50 = acc_rc[4]      # 50% de couverture
            results[(B, signal)] = (rpb, mono, base_acc, acc50, cov, acc_rc)
            print(f"    -> monotone (tol 2 pts) : {'OUI' if mono else 'NON'} ; "
                  f"selective a 50% couverture : {100*acc50:.1f}% "
                  f"(vs {100*base_acc:.1f}% sans abstention)")

    # ---------------- verdict ----------------
    print("\n" + "=" * 76)
    print("VERDICT P6(a) -- le doute est-il un estimateur de confiance utilisable ?")
    print("=" * 76)
    for B in BUDGETS:
        for signal in ["conf_u", "conf_sig"]:
            rpb, mono, base, acc50, _, _ = results[(B, signal)]
            gain = acc50 - base
            verdict = ("CALIBRE et UTILE" if (rpb > 0.15 and gain > 0.03) else
                       ("signal present mais faible" if rpb > 0.05 else
                        "PAS de signal (plat)"))
            print(f"  B={B:>5} {signal:<9}: r={rpb:+.3f}, monotone={'oui' if mono else 'non'}, "
                  f"abstention@50% {100*base:.1f}->{100*acc50:.1f}% ({100*gain:+.1f} pts) "
                  f"=> {verdict}")
    print("\n  Rappel du critere (fixe avant run) : utilisable si r_pb>0.15 ET")
    print("  gain d'abstention a 50% > +3 pts. Sinon : le dire, et chercher un")
    print("  signal composite avant de construire la Couche d'Abstention.")

    # ---------------- sorties ----------------
    with RAW_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with AGG_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["budget", "signal", "bin", "n", "acc", "ci_lo", "ci_hi"])
        w.writerows(agg_lines)
    print(f"\n[csv] {RAW_CSV}\n[csv] {AGG_CSV}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
        colors = {"conf_u": "#d62728", "conf_sig": "#1f77b4"}
        # panel 1 : reliability a B=800 (le budget intermediaire, le plus parlant)
        ax = axes[0]
        B = 800
        for signal in ["conf_u", "conf_sig"]:
            conf = np.array([r[f"{signal}_B{B}"] for r in rows])
            correct = np.array([r[f"correct_B{B}"] for r in rows])
            qs = np.quantile(conf, np.linspace(0, 1, N_BINS + 1))
            qs[0] -= 1e-9
            xs, ys, los, his = [], [], [], []
            for i in range(N_BINS):
                mask = (conf > qs[i]) & (conf <= qs[i + 1])
                if mask.sum() == 0:
                    continue
                xs.append(conf[mask].mean())
                kb, nb = int(correct[mask].sum()), int(mask.sum())
                lo, hi = wilson(kb, nb)
                ys.append(kb / nb)
                los.append(kb / nb - lo)
                his.append(hi - kb / nb)
            ax.errorbar(xs, ys, yerr=[los, his], fmt="o-", color=colors[signal],
                        capsize=4, label=signal)
        ax.set_xlabel("confiance (1 - doute)")
        ax.set_ylabel("taux de bonne decision")
        ax.set_title(f"Reliability diagram (B={B})")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        # panels 2-3 : risque-couverture aux 3 budgets
        for j, signal in enumerate(["conf_u", "conf_sig"]):
            ax = axes[1 + j]
            for B, ls in zip(BUDGETS, ["-", "--", ":"]):
                _, _, base, _, cov, acc_rc = results[(B, signal)]
                ax.plot([100 * c for c in cov], [100 * a for a in acc_rc],
                        ls, marker="o", ms=3, label=f"B={B} (base {100*base:.0f}%)")
            ax.set_xlabel("couverture (% des decisions gardees)")
            ax.set_ylabel("accuracy sur les gardees (%)")
            ax.set_title(f"Abstention par {signal}")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        fig.suptitle("P6(a) -- Le doute sait-il qu'il ne sait pas ? "
                     f"({n_total} essais loyaux+pieges, tache B1d)", fontsize=11)
        plt.tight_layout()
        plt.savefig(PNG, dpi=140)
        print(f"[png] {PNG}")
    except Exception as e:
        print(f"[png] skipped: {e}")

    print(f"\nWall time: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
