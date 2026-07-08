#!/usr/bin/env python3
"""
CONSOLIDATION B1b -- robustesse de l'utilite du watchdog natif (seeds x topologies).

Contexte. watchdog_multimodal_poc.py (2026-07-07) a valide, sur LATTICE / 5 seeds, que le cycle
FOU<->SAGE natif (dynamics.py:363, opt-in) : (1) est UTILE -- validite 0.97 vs hasard 0.35 ;
(2) sa couverture superieure vient du KICK, pas de la "nativite" (WATCHDOG ~ BICAMERAL_KICK).
Reserve gravee : "seeds 0-4, lattice 10x10, E=1.0". Ce script consolide sur :
  - 30 seeds (au lieu de 5),
  - 3 topologies : LATTICE (regulier), BA m=3 (scale-free), ER (aleatoire),
  - IC bootstrap sur les deux verdicts (utilite vs hasard ; couverture kick vs natif).

La contrainte multi-modale (A=noeud 0 -> +, B=noeud 99 -> -, interface libre) et toutes les
metriques (valid_frac, coverage, sharpness, consec_dist) sont IMPORTEES du POC : seule la
topologie injectee (adjacency_matrix) et le nombre de seeds changent.

Sortie : figures/b1b_watchdog_consolidation.csv + .png + verdict.
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
from mem4ristor.topology import Mem4Network  # noqa: E402
from mem4ristor.graph_utils import make_lattice_adj, make_ba, make_er  # noqa: E402
import watchdog_multimodal_poc as wpoc  # noqa: E402  (helpers : consolidate/analyse/stim/consts)

CSV = ROOT / "figures" / "b1b_watchdog_consolidation.csv"
PNG = ROOT / "figures" / "b1b_watchdog_consolidation.png"

N = wpoc.N
SIDE = wpoc.SIDE
SEEDS = list(range(30))
N_BOOT = 10000
RNG_BOOT = np.random.RandomState(20260708)
CONDS = ["BICAMERAL", "BICAMERAL_KICK", "WATCHDOG", "HASARD", "ATTRACTIF"]

def make_topologies(seed):
    return {
        "LATTICE": make_lattice_adj(SIDE, periodic=True),
        "BA_m3": make_ba(N, m=3, seed=seed),
        "ER_p06": make_er(N, p=0.06, seed=seed),
    }

def run_external(condition, seed, adj):
    """Copie fidele de wpoc.run_external mais avec adjacency_matrix injectee.
    Reutilise wpoc.consolidate (pin u=U_SAGE, step avec wpoc.stim) -> meme mecanique."""
    net = Mem4Network(size=SIDE, heretic_ratio=0.0, seed=seed, adjacency_matrix=adj)
    rng = np.random.RandomState(1000 + seed)
    sols = []
    for _c in range(wpoc.N_CYCLES):
        if condition == "BICAMERAL":
            net.model.cfg["doubt"]["epsilon_u"] = 0.02
            for _ in range(wpoc.T_FOU):
                net.step(I_stimulus=wpoc.stim)
        elif condition == "BICAMERAL_KICK":
            net.model.cfg["doubt"]["epsilon_u"] = 0.02
            net.model.u[:] = wpoc.U_FOU
            for _ in range(wpoc.T_FOU):
                net.step(I_stimulus=wpoc.stim)
        elif condition == "HASARD":
            net.model.v[:] = rng.uniform(-1.5, 1.5, N)
            net.model.w[:] = rng.uniform(0.0, 1.0, N)
        elif condition == "ATTRACTIF":
            pass
        sols.append(wpoc.consolidate(net, wpoc.T_SAGE))
    return sols

def run_watchdog(seed, adj):
    """Copie fidele de wpoc.run_watchdog mais avec adjacency_matrix injectee.
    u est gere par le coeur (jamais ecrit ici)."""
    net = Mem4Network(size=SIDE, heretic_ratio=0.0, seed=seed, adjacency_matrix=adj)
    net.model.cfg["doubt"]["epsilon_u"] = 0.02
    net.model.cfg["consolidation_watchdog"] = {
        "enabled": True, "t_explore": wpoc.T_FOU, "t_consolidate": wpoc.T_SAGE,
        "u_sage": wpoc.U_SAGE, "u_fou": wpoc.U_FOU,
    }
    sols = []
    prev = False
    steps = 0
    max_steps = wpoc.N_CYCLES * (wpoc.T_FOU + wpoc.T_SAGE) + 2 * (wpoc.T_FOU + wpoc.T_SAGE)
    while len(sols) < wpoc.N_CYCLES and steps < max_steps:
        net.step(I_stimulus=wpoc.stim)
        steps += 1
        now = bool(net.model.watchdog_consolidating)
        if prev and not now:
            sols.append(net.model.v.copy())
        prev = now
    return sols

def run(condition, seed, adj):
    if condition == "WATCHDOG":
        return run_watchdog(seed, adj)
    return run_external(condition, seed, adj)

def boot_ci(vec):
    vec = np.asarray(vec, dtype=float)
    n = len(vec)
    means = np.empty(N_BOOT)
    for b in range(N_BOOT):
        means[b] = vec[RNG_BOOT.randint(0, n, n)].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

def main():
    CSV.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = []
    topo_names = ["LATTICE", "BA_m3", "ER_p06"]
    # data[topo][cond] = {"valid":[...], "cov":[...], "consec":[...]}
    data = {t: {c: {"valid": [], "cov": [], "consec": []} for c in CONDS} for t in topo_names}

    print(f"{'topo':<9}{'cond':<16}{'valid_frac':>12}{'coverage':>10}{'consec_d':>10}  (moy 30 seeds)")
    print("-" * 70)
    for topo in topo_names:
        for seed in SEEDS:
            adjs = make_topologies(seed)
            adj = adjs[topo]
            for cond in CONDS:
                vf, cov, sh, cd = wpoc.analyse(run(cond, seed, adj))
                data[topo][cond]["valid"].append(vf)
                data[topo][cond]["cov"].append(cov)
                data[topo][cond]["consec"].append(cd)
                rows.append((topo, cond, seed, vf, cov, sh, cd))
        for cond in CONDS:
            vf = np.mean(data[topo][cond]["valid"])
            cov = np.mean(data[topo][cond]["cov"])
            cd = np.mean(data[topo][cond]["consec"])
            print(f"{topo:<9}{cond:<16}{vf:>12.2f}{cov:>10.1f}{cd:>10.2f}")

    print("\n=== VERDICT consolidation B1b (honnete) ===")
    print("Q1 : le watchdog natif est-il UTILE (validite >> hasard) sur les 3 topologies ?")
    all_useful = True
    for topo in topo_names:
        wd = data[topo]["WATCHDOG"]["valid"]; ha = data[topo]["HASARD"]["valid"]
        diff = np.array(wd) - np.array(ha)
        lo, hi = boot_ci(diff)
        useful = lo > 0
        all_useful = all_useful and useful
        print(f"  {topo:<9} valid WATCHDOG={np.mean(wd):.2f} vs HASARD={np.mean(ha):.2f} "
              f"| ecart {np.mean(diff):+.2f} CI[{lo:+.2f},{hi:+.2f}] -> "
              f"{'UTILE' if useful else 'non concluant'}")

    print("\nQ2 : la couverture vient-elle du KICK (WATCHDOG ~ BICAMERAL_KICK, tous deux > BICAMERAL) ?")
    kick_story = True
    for topo in topo_names:
        wd = np.array(data[topo]["WATCHDOG"]["cov"], dtype=float)
        bk = np.array(data[topo]["BICAMERAL_KICK"]["cov"], dtype=float)
        bi = np.array(data[topo]["BICAMERAL"]["cov"], dtype=float)
        lo_wb, hi_wb = boot_ci(wd - bk)          # WATCHDOG - BIC_KICK : attendu ~0
        lo_ki, hi_ki = boot_ci(bk - bi)          # BIC_KICK - BICAMERAL : attendu > 0
        equiv = lo_wb <= 0 <= hi_wb              # les deux formes de kick equivalentes
        kick_gain = lo_ki > 0                    # le kick releve la couverture au-dessus du bruit-driven
        kick_story = kick_story and equiv and kick_gain
        print(f"  {topo:<9} cov WATCHDOG={wd.mean():.1f} BIC_KICK={bk.mean():.1f} BICAMERAL={bi.mean():.1f}"
              f" | (WD-BK)CI[{lo_wb:+.1f},{hi_wb:+.1f}] (BK-BI)CI[{lo_ki:+.1f},{hi_ki:+.1f}]"
              f" -> {'kick=source' if (equiv and kick_gain) else 'nuance'}")

    print("\n  --> " + (
        "CONSOLIDE : sur 30 seeds x 3 topos, le watchdog natif est utile (validite > hasard) et sa"
        "\n      couverture vient du kick internalise (WATCHDOG ~ BIC_KICK > BICAMERAL)."
        if (all_useful and kick_story) else
        "PARTIEL : un des deux verdicts ne tient pas sur toutes les topologies -- voir tableau."))

    with CSV.open("w", encoding="utf-8") as f:
        f.write("topo,condition,seed,valid_frac,coverage,sharpness,consec_dist\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]:.4f},{r[4]},{r[5]:.4f},{r[6]:.4f}\n")
    print(f"\n[csv] {CSV}")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        colors = {"BICAMERAL": "#2ca02c", "BICAMERAL_KICK": "#98df8a", "WATCHDOG": "#d62728",
                  "HASARD": "#7f7f7f", "ATTRACTIF": "#1f77b4"}
        xs = np.arange(len(topo_names)); w = 0.16
        for i, cond in enumerate(CONDS):
            vals = [np.mean(data[t][cond]["valid"]) for t in topo_names]
            axes[0].bar(xs + (i - 2) * w, vals, w, color=colors[cond], edgecolor="k", label=cond)
        axes[0].set_xticks(xs); axes[0].set_xticklabels(topo_names)
        axes[0].set_ylabel("valid_frac"); axes[0].set_title("Validite par topologie (30 seeds)")
        axes[0].legend(fontsize=7); axes[0].grid(axis="y", alpha=0.3)
        for i, cond in enumerate(CONDS):
            vals = [np.mean(data[t][cond]["cov"]) for t in topo_names]
            axes[1].bar(xs + (i - 2) * w, vals, w, color=colors[cond], edgecolor="k", label=cond)
        axes[1].set_xticks(xs); axes[1].set_xticklabels(topo_names)
        axes[1].set_ylabel("coverage (sol. distinctes)"); axes[1].set_title("Couverture par topologie")
        axes[1].legend(fontsize=7); axes[1].grid(axis="y", alpha=0.3)
        fig.suptitle("Consolidation B1b : watchdog natif utile + couverture pilotee par le kick "
                     "(30 seeds x 3 topos)", fontsize=11)
        plt.tight_layout(); plt.savefig(PNG, dpi=140)
        print(f"[png] {PNG}")
    except Exception as e:
        print(f"[png] skipped: {e}")
    print(f"\nWall time: {time.time()-t0:.1f}s")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
