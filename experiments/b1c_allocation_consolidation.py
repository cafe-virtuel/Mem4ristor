#!/usr/bin/env python3
"""
CONSOLIDATION B1c -- robustesse de l'allocation par le doute sur un flux (seeds x topologies).

Contexte. doubt_compute_allocation_poc.py (2026-07-07, LATTICE+ring+BA melanges, 6 seeds) a etabli,
sur une tache LOYALE (se stabiliser = avoir juste) :
  (1) l'allocation adaptative (DOUTE ou CONVERGENCE) ECRASE l'uniforme (ACT tient) ;
  (2) le DOUTE ne bat PAS le controle CONVERGENCE trivial (aussi precis, ~3.5x plus cher : il
      sur-reflechit). C'est le pendant "tache loyale" de B1d (tache trompeuse, ou le doute gagne).
Reserve : 6 seeds ; melange de topos dans une seule condition. Ce script consolide sur :
  - 18 seeds (au lieu de 6),
  - la topologie de BASE des familles EVIDENCE/CONTRADICTION declinee sur LATTICE / BA / ER
    (la famille TOPOLOGIE reste volontairement sparse/BA : c'est sa raison d'etre),
  - IC bootstrap (apparie par seed) sur l'ecart DOUTE - CONVERGENCE au budget serre (0.75x).

Message a verrouiller (pendant de B1d) : sur tache loyale, le doute NE bat PAS la convergence,
robustement aux seeds et a la topologie de base -> la valeur du doute est CONDITIONNELLE.

simulate / stop_doubt / stop_conv / allocate_and_score / decision_at : IMPORTES du POC.
Sortie : figures/b1c_allocation_consolidation.csv + .png + verdict.
Cree : 2026-07-08 (Claude Opus 4.8) -- consolidation Volet B1 (docs/FUTURE_WORK.md).
"""
from __future__ import annotations
import sys, time
from pathlib import Path
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from mem4ristor.graph_utils import make_lattice_adj, make_ba, make_er  # noqa: E402
import doubt_compute_allocation_poc as apoc  # noqa: E402

CSV = ROOT / "figures" / "b1c_allocation_consolidation.csv"
SUMMARY_CSV = ROOT / "figures" / "b1c_allocation_summary.csv"   # acc par (base_topo, seed) -> capstone
PNG = ROOT / "figures" / "b1c_allocation_consolidation.png"

N = apoc.N
SEEDS = list(range(18))
N_PER_FAMILY = apoc.N_PER_FAMILY
N_BOOT = 10000
RNG_BOOT = np.random.RandomState(20260708)
BUDGET_TIGHT = 0.75              # niveau de budget serre pour la comparaison decisive
BASE_TOPOS = ["LATTICE", "BA_m3", "ER_p06"]

def base_adj(name, seed):
    if name == "LATTICE":
        return make_lattice_adj(apoc.SIDE, periodic=True)
    if name == "BA_m3":
        return make_ba(N, m=3, seed=seed)
    if name == "ER_p06":
        return make_er(N, p=0.06, seed=seed)
    raise ValueError(name)

def make_problem(family, rng, seed, base):
    """Comme apoc.make_problem, mais EVIDENCE/CONTRADICTION posees sur la topologie 'base'
    (au lieu du lattice cable en dur). La famille TOPOLOGIE garde ring/BA (son objet propre)."""
    if family == "EVIDENCE":
        adj = base
        n_sens = rng.randint(1, 4); E = rng.uniform(0.25, 0.5); sign = rng.choice([-1, 1])
        nodes = rng.choice(N, size=n_sens, replace=False)
        stim = np.zeros(N); stim[nodes] = sign * E
        return adj, stim, sign, f"EVID(n={n_sens})"
    if family == "CONTRADICTION":
        adj = base
        n_plus = rng.randint(3, 6); n_minus = n_plus - rng.randint(1, 2); E = 1.0
        nodes = rng.choice(N, size=n_plus + n_minus, replace=False)
        stim = np.zeros(N); stim[nodes[:n_plus]] = +E; stim[nodes[n_plus:]] = -E
        return adj, stim, +1, f"CONTRA(+{n_plus}/-{n_minus})"
    if family == "TOPOLOGIE":
        return apoc.make_problem("TOPOLOGIE", rng, seed)
    raise ValueError(family)

def run_seed(seed, base_name):
    rng = np.random.RandomState(2000 + seed)
    families = (["EVIDENCE"] * N_PER_FAMILY + ["CONTRADICTION"] * N_PER_FAMILY +
                ["TOPOLOGIE"] * N_PER_FAMILY)
    rng.shuffle(families)
    base = base_adj(base_name, seed)
    probs = []
    for idx, fam in enumerate(families):
        adj, stim, dstar, label = make_problem(fam, rng, seed * 100 + idx, base)
        sig, dec, d_var = apoc.simulate(adj, stim, seed * 100 + idx)
        c_doubt = apoc.stop_doubt(sig); c_conv = apoc.stop_conv(d_var)
        correct = (dec == dstar); oracle = apoc.MAX_BUDGET
        for t in range(len(dec)):
            if np.all(correct[t:]):
                oracle = t + 1; break
        probs.append(dict(fam=fam, label=label, dstar=dstar, dec=dec,
                          c_doubt=c_doubt, c_conv=c_conv, oracle=oracle))
    return len(families), probs

def boot_ci(diffs):
    diffs = np.asarray(diffs, dtype=float); n = len(diffs)
    means = np.empty(N_BOOT)
    for b in range(N_BOOT):
        means[b] = diffs[RNG_BOOT.randint(0, n, n)].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

def main():
    CSV.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = []
    summary_rows = []   # (base_topo, seed, acc_doubt, acc_conv, acc_uniform) -> reproductible capstone
    summary = {}   # base -> dict(d, u, c, gap_ci, cost_d, cost_c)
    print(f"{'base_topo':<10}{'DOUTE':>8}{'UNIFORME':>10}{'CONV':>8}{'gapD-C':>9}{'CI95(D-C)':>16}"
          f"{'cost_D':>8}{'cost_C':>8}  (budget 0.75x, 18 seeds)")
    print("-" * 85)
    for base_name in BASE_TOPOS:
        # 1) simuler les flux, recolter couts auto-termines
        seed_data = []
        tot_oracle = 0.0; nprob = 0
        for seed in SEEDS:
            K, probs = run_seed(seed, base_name)
            seed_data.append((K, probs))
            for p in probs:
                tot_oracle += p["oracle"]; nprob += 1
        mean_oracle = tot_oracle / nprob
        K = seed_data[0][0]
        B_total = BUDGET_TIGHT * K * mean_oracle
        # 2) scorer chaque condition, par seed (pour l'IC apparie)
        acc_d, acc_u, acc_c = [], [], []
        cost_d, cost_c = [], []
        for (Kx, probs) in seed_data:
            sd, ad = apoc.allocate_and_score(probs, B_total, "DOUTE")
            su, au = apoc.allocate_and_score(probs, B_total, "UNIFORME")
            sc, ac = apoc.allocate_and_score(probs, B_total, "CONVERGENCE")
            acc_d.append(sd / Kx); acc_u.append(su / Kx); acc_c.append(sc / Kx)
            cost_d.append(np.mean([p["c_doubt"] for p in probs]))
            cost_c.append(np.mean([p["c_conv"] for p in probs]))
            for p in probs:
                rows.append((base_name, p["fam"], p["oracle"], p["c_doubt"], p["c_conv"]))
        for sd_i, seed in enumerate(SEEDS):
            summary_rows.append((base_name, seed, acc_d[sd_i], acc_c[sd_i], acc_u[sd_i]))
        gaps = np.array(acc_d) - np.array(acc_c)
        lo, hi = boot_ci(gaps)
        d, u, c = np.mean(acc_d), np.mean(acc_u), np.mean(acc_c)
        cd, cc = np.mean(cost_d), np.mean(cost_c)
        summary[base_name] = dict(d=d, u=u, c=c, gap=float(np.mean(gaps)), lo=lo, hi=hi, cd=cd, cc=cc)
        print(f"{base_name:<10}{d:>8.2f}{u:>10.2f}{c:>8.2f}{np.mean(gaps):>+9.2f}"
              f"  [{lo:+.2f},{hi:+.2f}]{cd:>8.0f}{cc:>8.0f}")

    print("\n=== VERDICT consolidation B1c (honnete) ===")
    print("Attendu (tache LOYALE, pendant de B1d) : (1) DOUTE et CONV >> UNIFORME (ACT tient) ;")
    print("(2) le DOUTE NE bat PAS la CONVERGENCE (gap D-C <= 0 ou IC couvrant 0) et coute plus cher.\n")
    act_ok = True; not_better = True
    for base_name in BASE_TOPOS:
        s = summary[base_name]
        act = s["d"] > s["u"] * 1.1 and s["c"] > s["u"] * 1.1
        # "ne bat pas" = borne haute de l'IC du gap ne franchit pas nettement 0 (<= +0.05)
        nb = s["hi"] <= 0.05
        act_ok = act_ok and act; not_better = not_better and nb
        cost_ratio = s["cd"] / max(s["cc"], 1e-9)
        print(f"  {base_name:<10} ACT: D={s['d']:.2f} C={s['c']:.2f} >> U={s['u']:.2f} "
              f"({'oui' if act else 'NON'}) | gap D-C {s['gap']:+.2f} CI[{s['lo']:+.2f},{s['hi']:+.2f}] "
              f"({'doute ne gagne pas' if nb else 'doute gagne ?!'}) | doute {cost_ratio:.1f}x plus cher")
    print("\n  --> " + (
        "CONSOLIDE : sur 18 seeds x 3 topos de base, l'allocation adaptative ecrase l'uniforme, et le"
        "\n      doute ne bat pas la convergence triviale (il sur-reflechit). Pendant loyal de B1d confirme."
        if (act_ok and not_better) else
        "PARTIEL : un des deux volets ne tient pas sur toutes les topologies -- voir tableau."))

    with CSV.open("w", encoding="utf-8") as f:
        f.write("base_topo,family,oracle,c_doubt,c_conv\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    with SUMMARY_CSV.open("w", encoding="utf-8") as f:
        f.write("base_topo,seed,acc_doubt,acc_conv,acc_uniform\n")
        for r in summary_rows:
            f.write(f"{r[0]},{r[1]},{r[2]:.4f},{r[3]:.4f},{r[4]:.4f}\n")
    print(f"\n[csv] {CSV}\n[csv] {SUMMARY_CSV}")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
        xs = np.arange(len(BASE_TOPOS)); w = 0.25
        for i, (lab, key, c) in enumerate([("DOUTE", "d", "#d62728"),
                                           ("UNIFORME", "u", "#7f7f7f"),
                                           ("CONVERGENCE", "c", "#1f77b4")]):
            axes[0].bar(xs + (i - 1) * w, [summary[t][key] for t in BASE_TOPOS], w,
                        color=c, edgecolor="k", label=lab)
        axes[0].set_xticks(xs); axes[0].set_xticklabels(BASE_TOPOS)
        axes[0].set_ylabel("fraction resolue (budget 0.75x)")
        axes[0].set_title("Allocation : adaptatif >> uniforme, doute ~ convergence")
        axes[0].legend(); axes[0].grid(axis="y", alpha=0.3); axes[0].set_ylim(0, 1.05)
        gaps = [summary[t]["gap"] for t in BASE_TOPOS]
        los = [summary[t]["gap"] - summary[t]["lo"] for t in BASE_TOPOS]
        his = [summary[t]["hi"] - summary[t]["gap"] for t in BASE_TOPOS]
        axes[1].bar(xs, gaps, 0.5, yerr=[los, his], capsize=5, color="#9467bd", edgecolor="k")
        axes[1].axhline(0, color="k", lw=0.8)
        axes[1].set_xticks(xs); axes[1].set_xticklabels(BASE_TOPOS)
        axes[1].set_ylabel("gap DOUTE - CONVERGENCE (IC95)")
        axes[1].set_title("Sur tache loyale, le doute ne gagne pas")
        axes[1].grid(axis="y", alpha=0.3)
        fig.suptitle("Consolidation B1c : tache loyale -> le doute n'ajoute rien (18 seeds x 3 topos)",
                     fontsize=11)
        plt.tight_layout(); plt.savefig(PNG, dpi=140)
        print(f"[png] {PNG}")
    except Exception as e:
        print(f"[png] skipped: {e}")
    print(f"\nWall time: {time.time()-t0:.1f}s")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
