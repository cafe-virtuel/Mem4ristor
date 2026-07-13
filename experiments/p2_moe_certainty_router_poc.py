#!/usr/bin/env python3
"""
P2 -- LE MoE PAR CERTITUDE : M4R comme routeur physique (le legs, piste P2).
=============================================================================
Cree : 2026-07-13 (Claude Sonnet 5, L'Ingenieur) -- piste du legs de Fable
(docs/PISTES_POUR_LA_SUITE_2026-07-12.md, section I).

TRACE : Mem4ristor/MEM4_MOE_CONCEPT.md (02/02/2026) -- l'idee fondatrice du
bicameralisme : M4R ne route pas par SUJET (MoE classique) mais par
CERTITUDE -- une sentinelle bon marche toujours allumee qui n'escalade au
"Sage" couteux (GPU/LLM) que quand le doute monte. Jamais quantifiee.

CE QUE CETTE PISTE REUTILISE (au lieu de reinventer) : la tache et le compas
COMPOSITE de P6b (experiments/p6b_abstention_poc.py, deja valides : +38.3 pts
a B=400, +25.0 pts a B=800, a 50% de couverture). P6b mesurait "accuracy vs
couverture" -- P2 est LA MEME MECANIQUE relue en "accuracy vs cout" : refuser
de decider = ESCALADER au resolveur couteux, coverage = fraction traitee par
la sentinelle SEULE.

TEST MINIMAL (piste P2, texte du legs) : un flux de taches melangees ; condition
A = tout va au modele couteux ; condition B = M4R traite tout, escalade au
couteux seulement si le score d'incertitude depasse un seuil. Mesurer accuracy
ET cout (pas de calcul simules). Le seuil doit etre calibre par VALIDATION
(risque explicite du legs : "sinon on refait le piege du budget fixe a
l'envers").

MODELE DE COUT (honnete, pas invente) : cout mesure en unites de B_cheap
(le budget reseau de la sentinelle, B=400 pas -- le point ou P6b a mesure le
signal le plus fort, r(u)=+0.74 raw). Le cout du resolveur "couteux" (GPU/LLM)
n'a AUCUNE mesure fiable dans ce projet (cf. docs/hardware/B3_ENERGY_COMPARISON.md :
"B3 est cadre, pas clos") -- au lieu d'inventer UN chiffre, on BALAIE le ratio
rho = cout_couteux / B_cheap in {2,5,10,25,50,100} et on rapporte ou la
comparaison bascule, plutot que de cacher l'hypothese derriere un seul nombre.
Le resolveur couteux est modelise comme ORACLE (accuracy=1.0 quand consulte) --
hypothese explicite, pas cachee : un GPU/LLM couteux est suppose largement plus
fiable que la sentinelle a mi-transitoire.

PROTOCOLE ANTI-FUITE (le risque du legs) : le score composite (regression
logistique sur les 5 signaux P6b) est deja hors-fold par construction (CV
groupee par seed, 6 folds). Mais le CHOIX du niveau de couverture (= seuil
d'escalade) est un second hyperparametre -- il est choisi sur un split CALIB
(18 seeds / 24) qui minimise le cout total sous contrainte accuracy >= 90%,
puis EVALUE tel quel sur un split HOLDOUT disjoint (6 seeds / 24) jamais vu
au moment du choix. Le nombre rapporte comme "verdict" est le nombre HOLDOUT.

CRITERE DE SUCCES (pre-fixe avant de lancer) : le routage M4R est UTILE si,
au HOLDOUT, sa courbe cout/accuracy DOMINE a la fois :
  (a) ALWAYS-CHEAP (la sentinelle seule, sans escalade) en accuracy, ET
  (b) ALL-EXPENSIVE (tout couteux) en cout, pour au moins un niveau de rho
      teste, a accuracy >= 90% (le seuil retenu en CALIB).
Si (a) echoue : le signal composite ne generalise pas hors CV interne.
Si (b) echoue a TOUS les rho testes : la sentinelle ne vaut jamais son cout
propre (B_cheap) face au couteux direct -- resultat negatif honnete a garder.

Sorties : figures/p2_moe_certainty_router_poc.csv + .png
Statut : exploratoire, hors preprint, coeur non touche.
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
import deceptive_task_poc as b1d          # noqa: E402
import p6b_abstention_poc as p6b          # noqa: E402

CSV_PATH = ROOT / "figures" / "p2_moe_certainty_router_poc.csv"
PNG_PATH = ROOT / "figures" / "p2_moe_certainty_router_poc.png"

B_CHEAP = 400                       # budget de la sentinelle (le point le plus informatif, P6a/P6b)
SEEDS = list(range(24))
T_PULSES = p6b.T_PULSES             # [0, 150, 350, 700, 1200] -- reuse EXACT de P6a/P6b (5 niveaux, dont le cas facile 0)
CALIB_SEEDS = list(range(18))       # 18/24 seeds -- choix du seuil
HOLDOUT_SEEDS = list(range(18, 24)) # 6/24 seeds -- jamais vus au choix du seuil
RHO_LEVELS = [2, 5, 10, 25, 50, 100]  # cout_couteux / B_cheap, balaye (pas invente)
TARGET_ACC = 0.90                   # contrainte d'accuracy en CALIB


def collect_trials():
    """Reproduit exactement les 120 essais de P6a/P6b (5 t_pulse x 24 seeds),
    mais ne lit que B=B_CHEAP. Reutilise p6b.simulate_record + rolling_mean."""
    rows = []
    t0 = time.time()
    n_total = len(T_PULSES) * len(SEEDS)
    done = 0
    for t_pulse in T_PULSES:
        for seed in SEEDS:
            rng = np.random.RandomState(3000 + seed)
            adj, stim_on, stim_off, dstar = b1d.make_deceptive(rng)
            sig, u_mean, d_raw = p6b.simulate_record(
                adj, stim_on, stim_off, seed * 10 + 1, t_pulse)
            d_sm = p6b.rolling_mean(d_raw, p6b.W_READ)
            i = B_CHEAP - 1
            dec_sm = 1 if d_sm[i] >= 0 else -1
            peak_run = np.maximum.accumulate(sig)
            below = sig < p6b.CONS_FRAC * peak_run
            below[:30] = False
            t_cons = int(np.argmax(below)) + 1 if below.any() else b1d.MAX_BUDGET + 1
            lo = max(0, i - p6b.W_STAB + 1)
            row = {
                "t_pulse": t_pulse, "seed": seed, "dstar": dstar,
                "correct": int(dec_sm == dstar),
                "conf_u_naif": 1.0 - float(u_mean[i]),
                "conf_u_inv": float(u_mean[i]),
                "conf_sig": 1.0 - min(1.0, float(sig[i] / max(peak_run[i], 1e-12))),
                "conf_tcons": min(1.0, t_cons / B_CHEAP),
                "conf_stab": float(np.mean(
                    (p6b.rolling_mean(d_raw, p6b.W_READ)[lo:i + 1] >= 0) ==
                    (dec_sm >= 0))),
            }
            rows.append(row)
            done += 1
        print(f"  t_pulse={t_pulse:>5} fait  [{done}/{n_total}, {time.time()-t0:.0f}s]")
    return rows


def composite_score(rows):
    """Meme composite que P6b : regression logistique sur les 5 signaux,
    CV groupee par seed -> score hors-fold pour chaque essai."""
    X = np.column_stack([[r[s] for r in rows] for s in p6b.SIGNALS])
    y = np.array([r["correct"] for r in rows], dtype=float)
    groups = np.array([r["seed"] for r in rows])
    return p6b.logistic_cv(X, y, groups)


def cost_accuracy_at_coverage(rows, comp, seed_subset, coverage):
    """A un niveau de couverture donne (fraction traitee par la sentinelle
    SEULE, le reste escalade), retourne (accuracy, cout_normalise_hors_rho)
    ou cout_normalise_hors_rho = fraction escaladee (a multiplier par rho
    ensuite, + 1.0 pour B_cheap toujours paye)."""
    idx = [k for k, r in enumerate(rows) if r["seed"] in seed_subset]
    if not idx:
        return None, None
    sub_comp = comp[idx]
    sub_correct = np.array([rows[k]["correct"] for k in idx])
    order = np.argsort(-sub_comp)          # confiance decroissante
    n = len(idx)
    n_keep = max(1, int(round(n * coverage)))
    kept = order[:n_keep]                  # traites par la sentinelle SEULE
    escalated_frac = 1.0 - (n_keep / n)
    acc_kept = float(sub_correct[kept].mean())
    overall_acc = (n_keep / n) * acc_kept + escalated_frac * 1.0  # oracle sur l'escalade
    return overall_acc, escalated_frac


def pick_coverage_on_calib_for(rows, comp, calib_seeds):
    """Balaie la couverture, choisit celle qui MINIMISE le cout total (fraction
    escaladee) sous contrainte accuracy>=TARGET_ACC, sur calib_seeds uniquement."""
    best = None
    for coverage in np.linspace(0.05, 1.0, 20):
        acc, esc = cost_accuracy_at_coverage(rows, comp, calib_seeds, coverage)
        if acc is None:
            continue
        if acc >= TARGET_ACC:
            if best is None or esc < best[1]:
                best = (coverage, esc, acc)
    if best is None:
        # aucune couverture n'atteint TARGET_ACC en CALIB -> tout escalader (coverage=0)
        return 0.0
    return best[0]


def pick_coverage_on_calib(rows, comp):
    return pick_coverage_on_calib_for(rows, comp, CALIB_SEEDS)


def main() -> int:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"P2 -- MoE par certitude. {len(T_PULSES)*len(SEEDS)} essais (B1d canonique), "
          f"B_cheap={B_CHEAP}, CALIB={len(CALIB_SEEDS)} seeds, HOLDOUT={len(HOLDOUT_SEEDS)} seeds")
    rows = collect_trials()
    comp = composite_score(rows)

    base_acc_calib, _ = cost_accuracy_at_coverage(rows, comp, CALIB_SEEDS, 1.0)
    base_acc_holdout, _ = cost_accuracy_at_coverage(rows, comp, HOLDOUT_SEEDS, 1.0)
    print(f"\nALWAYS-CHEAP (coverage=1.0, pas d'escalade) : "
          f"CALIB acc={100*base_acc_calib:.1f}%  HOLDOUT acc={100*base_acc_holdout:.1f}%")

    coverage_star = pick_coverage_on_calib(rows, comp)
    acc_calib, esc_calib = cost_accuracy_at_coverage(rows, comp, CALIB_SEEDS, coverage_star)
    acc_holdout, esc_holdout = cost_accuracy_at_coverage(rows, comp, HOLDOUT_SEEDS, coverage_star)
    print(f"\nSeuil choisi en CALIB (coverage={coverage_star:.2f}, cible acc>={100*TARGET_ACC:.0f}%) :")
    print(f"  CALIB   : acc={100*acc_calib:.1f}%  escalade={100*esc_calib:.1f}%")
    print(f"  HOLDOUT : acc={100*acc_holdout:.1f}%  escalade={100*esc_holdout:.1f}%  <- verdict")

    print(f"\n{'rho':>6}{'A:ALL-EXP cost':>16}{'B:M4R cost':>14}{'B<A?':>8}"
          f"{'B acc':>10}{'A acc':>8}{'C:CHEAP acc':>14}")
    rows_out = []
    b_beats_a_at = []
    for rho in RHO_LEVELS:
        cost_A = rho                      # ALL-EXPENSIVE : rho*B_cheap / B_cheap = rho, acc=1.0
        cost_B = 1.0 + esc_holdout * rho  # M4R-ROUTER : sentinelle toujours payee + escalade
        b_beats_a = cost_B < cost_A
        if b_beats_a and acc_holdout >= TARGET_ACC:
            b_beats_a_at.append(rho)
        print(f"{rho:>6}{cost_A:>16.2f}{cost_B:>14.2f}{'OUI' if b_beats_a else 'non':>8}"
              f"{100*acc_holdout:>9.1f}%{100.0:>7.1f}%{100*base_acc_holdout:>13.1f}%")
        rows_out.append((rho, cost_A, cost_B, acc_holdout, 1.0, base_acc_holdout, b_beats_a))

    print("\n" + "=" * 78)
    print("VERDICT P2 (pre-fixe : domine ALWAYS-CHEAP en accuracy ET ALL-EXPENSIVE")
    print("en cout a accuracy>=90%, sur HOLDOUT jamais vu au choix du seuil)")
    print("=" * 78)
    beats_cheap = acc_holdout > base_acc_holdout + 0.03
    print(f"  (a) M4R-ROUTER vs ALWAYS-CHEAP (HOLDOUT) : "
          f"{100*acc_holdout:.1f}% vs {100*base_acc_holdout:.1f}% -> "
          f"{'BAT' if beats_cheap else 'NE BAT PAS'} la sentinelle seule")
    if acc_holdout < TARGET_ACC:
        print(f"  (b) accuracy HOLDOUT ({100*acc_holdout:.1f}%) N'ATTEINT PAS la cible CALIB "
              f"({100*TARGET_ACC:.0f}%) -> le seuil choisi en CALIB NE GENERALISE PAS tel quel.")
        print("  -> NON CONCLUANT : le compas ne transporte pas son critere CALIB vers HOLDOUT.")
    elif b_beats_a_at:
        print(f"  (b) M4R-ROUTER bat ALL-EXPENSIVE en cout pour rho >= {min(b_beats_a_at)} "
              f"(sur {len(b_beats_a_at)}/{len(RHO_LEVELS)} niveaux testes)")
        if beats_cheap:
            print("  -> UTILE : le routage par certitude domine les deux baselines sur HOLDOUT.")
        else:
            print("  -> PARTIEL : bat ALL-EXPENSIVE en cout mais pas ALWAYS-CHEAP en accuracy "
                  "(escalade trop genereuse, gain marginal).")
    else:
        print(f"  (b) M4R-ROUTER ne bat ALL-EXPENSIVE a AUCUN rho teste ({RHO_LEVELS}) : "
              "le cout d'escalade (B_cheap toujours paye + fraction escaladee) "
              "depasse l'economie, meme au ratio le plus favorable.")
        print("  -> NON UTILE tel quel a ces rho : le seuil choisi escalade trop pour rester "
              "moins cher que tout-couteux dans cette gamme de ratio.")

    # ---------------- robustesse : le near-miss est-il un artefact du split ? ----------------
    # 30 essais HOLDOUT = granularite de ~3.3 pts -- un seul essai separe 86.7% de 90.0%.
    # On repete le protocole COMPLET (choix du seuil en CALIB, mesure en HOLDOUT) sur
    # N_SPLITS partitions aleatoires disjointes 18/6, pour voir si le near-miss est stable
    # ou un coup de mauvais sort de CE split precis. Le split original (seeds 0-17/18-23)
    # reste le verdict pre-fixe ; ceci est un controle de stabilite, pas une re-election du seuil.
    print("\n" + "=" * 78)
    print("CONTROLE DE ROBUSTESSE -- 8 partitions CALIB/HOLDOUT aleatoires (18/6 seeds)")
    print("=" * 78)
    rng_split = np.random.RandomState(20260713)
    all_seeds = np.array(SEEDS)
    rob_accs, rob_gen_ok, rob_beats_cheap = [], [], []
    for k in range(8):
        perm = rng_split.permutation(all_seeds)
        calib_k, holdout_k = list(perm[:18]), list(perm[18:])
        cov_k = pick_coverage_on_calib_for(rows, comp, calib_k)
        acc_h, _ = cost_accuracy_at_coverage(rows, comp, holdout_k, cov_k)
        acc_cheap_h, _ = cost_accuracy_at_coverage(rows, comp, holdout_k, 1.0)
        rob_accs.append(acc_h)
        rob_gen_ok.append(acc_h >= TARGET_ACC)
        rob_beats_cheap.append(acc_h > acc_cheap_h + 0.03)
        print(f"  split {k}: HOLDOUT acc={100*acc_h:.1f}%  (cheap seul={100*acc_cheap_h:.1f}%)  "
              f"generalise>=90%: {'oui' if acc_h >= TARGET_ACC else 'non'}")
    rob_accs = np.array(rob_accs)
    print(f"\n  Sur 9 splits (l'original + 8 aleatoires) : HOLDOUT acc moyenne={100*np.mean([acc_holdout]+list(rob_accs)):.1f}% "
          f"+-{100*np.std([acc_holdout]+list(rob_accs)):.1f}  ;  "
          f"atteint la cible 90% dans {sum([acc_holdout>=TARGET_ACC]+rob_gen_ok)}/9 cas  ;  "
          f"bat ALWAYS-CHEAP (+3pts) dans {sum([beats_cheap]+rob_beats_cheap)}/9 cas")

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rho", "cost_A_all_expensive", "cost_B_m4r_router",
                    "acc_B_holdout", "acc_A", "acc_C_always_cheap_holdout", "B_beats_A_cost"])
        w.writerows(rows_out)
    print(f"\n[csv] {CSV_PATH}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
        axes[0].plot(RHO_LEVELS, [r[1] for r in rows_out], "o-", color="#7f7f7f", label="A: ALL-EXPENSIVE")
        axes[0].plot(RHO_LEVELS, [r[2] for r in rows_out], "o-", color="#d62728", label="B: M4R-ROUTER")
        axes[0].axhline(1.0, ls=":", color="#2ca02c", label="C: ALWAYS-CHEAP (cout=1)")
        axes[0].set_xlabel("rho = cout_couteux / B_cheap"); axes[0].set_ylabel("cout total (unites B_cheap)")
        axes[0].set_title("Cout vs ratio de prix"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
        axes[1].bar(["A\nALL-EXP", "B\nM4R-ROUTER", "C\nALWAYS-CHEAP"],
                     [1.0, acc_holdout, base_acc_holdout],
                     color=["#7f7f7f", "#d62728", "#2ca02c"])
        axes[1].axhline(TARGET_ACC, ls="--", color="k", alpha=0.5, label=f"cible CALIB {TARGET_ACC:.0%}")
        axes[1].set_ylabel("accuracy (HOLDOUT)"); axes[1].set_ylim(0, 1.05)
        axes[1].set_title("Accuracy par condition (HOLDOUT)"); axes[1].legend(fontsize=8)
        fig.suptitle(f"P2 -- MoE par certitude (coverage*={coverage_star:.2f}, "
                     f"B_cheap={B_CHEAP}, 6 seeds HOLDOUT)", fontsize=11)
        plt.tight_layout()
        plt.savefig(PNG_PATH, dpi=140)
        print(f"[png] {PNG_PATH}")
    except Exception as e:
        print(f"[png] skipped: {e}")

    print(f"\nWall time: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
