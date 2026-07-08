#!/usr/bin/env python3
"""
B4 -- ROBUSTESSE STATISTIQUE du resultat central (ablation FROZEN_U -> surge de synchronie).

Contexte. Le resultat le plus robuste et le moins parametrise du preprint (mene l'abstract depuis
A2) est l'ablation : geler u (FROZEN_U, epsilon_u=0) fait SURGIR la synchronie de Pearson (le
reseau se synchronise et "meurt" cognitivement), la ou u actif la maintient ~0. Le preprint le
chiffre en "~90-fold" (lattice) / "~24-fold" (BA). B4 (docs/FUTURE_WORK.md) demande de remplacer
la "complete separation" / le ratio-point par un IC honnete.

DECOUVERTE de cadrage (mesuree ici) : le RATIO FROZEN/FULL est une statistique FRAGILE car la
synchronie FULL ~ 0 et CHEVAUCHE zero (parfois negative sur BA). Diviser par ~0 donne des ratios
absurdes (jusqu'a ~1e9 sur un seed). Le resultat honnete n'est PAS un ratio mais :
  - un SAUT vers un niveau de synchronie eleve : difference FROZEN - FULL avec IC bootstrap,
  - une SEPARATION (max FULL vs min FROZEN) et une taille d'effet (Cohen d) enormes,
  - le ratio est rapporte MAIS explicitement signale comme instable (a ne pas mettre en avant).

Mesure sur 30 seeds x 2 topologies (LATTICE 10x10, BA m=3), en reutilisant TEL QUEL
p2_sigma_social_ablation.run_one(adj, condition, seed) (memes params : I_STIM=0.5, STEPS=5000,
WARM_UP=1000). On capture la synchronie (central) et H_cont (effondrement de diversite, bonus).

Sortie : figures/b4_ablation_robustness.csv + .png + verdict.
Cree : 2026-07-08 (Claude Opus 4.8) -- B4 robustesse statistique (docs/FUTURE_WORK.md).
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
from mem4ristor.graph_utils import make_ba, make_lattice_adj  # noqa: E402
import p2_sigma_social_ablation as ab  # noqa: E402  (run_one reutilise tel quel)

CSV = ROOT / "figures" / "b4_ablation_robustness.csv"
SUMMARY_CSV = ROOT / "figures" / "b4_ablation_summary.csv"   # 1 ligne/topo : Cohen d, diff, separation (Guardian)
PNG = ROOT / "figures" / "b4_ablation_robustness.png"

# 30 seeds : les 10 canoniques (Table 1) + 20 nouveaux -> le sous-ensemble canonique reste
# identifiable, et on quadruple l'echantillon pour un IC honnete.
SEEDS = ab.SEEDS + list(range(3001, 3021))
N_BOOT = 10000
RNG = np.random.RandomState(20260708)
TOPOS = ["LATTICE", "BA_m3"]

def make_topo(name):
    if name == "LATTICE":
        return make_lattice_adj(10, periodic=True)
    if name == "BA_m3":
        return make_ba(100, 3, seed=42)   # topo BA canonique du Paper 2 (seed fixe)
    raise ValueError(name)

def boot_mean_ci(vec):
    vec = np.asarray(vec, float); n = len(vec)
    m = np.empty(N_BOOT)
    for b in range(N_BOOT):
        m[b] = vec[RNG.randint(0, n, n)].mean()
    return float(vec.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

def boot_paired_diff_ci(a, b):
    """IC bootstrap de mean(a-b), apparie par seed."""
    d = np.asarray(a, float) - np.asarray(b, float); n = len(d)
    m = np.empty(N_BOOT)
    for k in range(N_BOOT):
        m[k] = d[RNG.randint(0, n, n)].mean()
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

def boot_ratio_ci(num, den):
    """IC bootstrap du ratio mean(num)/mean(den), apparie. Instable si mean(den) ~ 0."""
    num = np.asarray(num, float); den = np.asarray(den, float); n = len(num)
    r = np.empty(N_BOOT)
    for k in range(N_BOOT):
        idx = RNG.randint(0, n, n)
        dm = den[idx].mean()
        r[k] = num[idx].mean() / dm if abs(dm) > 1e-6 else np.nan
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.nanmedian(r)), float(np.nanpercentile(r, 2.5)), float(np.nanpercentile(r, 97.5))

def cohen_d_paired(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else float("inf")

def main():
    CSV.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = []
    results = {}
    print(f"Seeds: {len(SEEDS)} ({len(ab.SEEDS)} canoniques + {len(SEEDS)-len(ab.SEEDS)} nouveaux)")
    print(f"I_STIM={ab.I_STIM} STEPS={ab.STEPS} WARM_UP={ab.WARM_UP}\n")

    for topo in TOPOS:
        adj = make_topo(topo)
        full_sync, froz_sync, full_hc, froz_hc = [], [], [], []
        for seed in SEEDS:
            _, hcont_f, sync_f, _, _ = ab.run_one(adj, "FULL", seed)
            _, hcont_z, sync_z, _, _ = ab.run_one(adj, "FROZEN_U", seed)
            full_sync.append(sync_f); froz_sync.append(sync_z)
            full_hc.append(hcont_f); froz_hc.append(hcont_z)
            rows.append((topo, seed, sync_f, sync_z, hcont_f, hcont_z))
        results[topo] = dict(fs=np.array(full_sync), zs=np.array(froz_sync),
                             fh=np.array(full_hc), zh=np.array(froz_hc))

    print(f"{'topo':<9}{'FULL sync (CI)':>24}{'FROZEN sync (CI)':>24}{'diff (CI)':>22}")
    print("-" * 79)
    summary = {}
    for topo in TOPOS:
        r = results[topo]
        fm, flo, fhi = boot_mean_ci(r["fs"])
        zm, zlo, zhi = boot_mean_ci(r["zs"])
        dm, dlo, dhi = boot_paired_diff_ci(r["zs"], r["fs"])
        summary[topo] = dict(fm=fm, flo=flo, fhi=fhi, zm=zm, zlo=zlo, zhi=zhi,
                             dm=dm, dlo=dlo, dhi=dhi)
        print(f"{topo:<9}{f'{fm:.4f} [{flo:.4f},{fhi:.4f}]':>24}"
              f"{f'{zm:.3f} [{zlo:.3f},{zhi:.3f}]':>24}"
              f"{f'{dm:.3f} [{dlo:.3f},{dhi:.3f}]':>22}")

    print("\n=== VERDICT B4 (honnete) ===")
    summary_out = []   # 1 ligne/topo pour le Guardian (Cohen d, diff, separation)
    for topo in TOPOS:
        r = results[topo]; s = summary[topo]
        # separation complete ?
        max_full = float(r["fs"].max()); min_froz = float(r["zs"].min())
        sep = min_froz > max_full
        d = cohen_d_paired(r["zs"], r["fs"])
        rmed, rlo, rhi = boot_ratio_ci(r["zs"], r["fs"])
        # effondrement de diversite (H_cont)
        dh, dhlo, dhhi = boot_paired_diff_ci(r["zh"], r["fh"])
        summary_out.append((topo, s["fm"], s["zm"], s["dm"], s["dlo"], s["dhi"],
                            d, int(sep), max_full, min_froz))
        print(f"\n  [{topo}]")
        print(f"    Synchronie : FULL {s['fm']:.4f} -> FROZEN {s['zm']:.3f} ; "
              f"difference {s['dm']:+.3f} CI[{s['dlo']:+.3f},{s['dhi']:+.3f}] (Cohen d={d:.1f})")
        print(f"    Separation : max(FULL)={max_full:.4f}  min(FROZEN)={min_froz:.3f}  -> "
              f"{'COMPLETE' if sep else 'partielle'}")
        print(f"    Ratio FROZEN/FULL (FRAGILE, den~0) : mediane bootstrap {rmed:.0f}x "
              f"CI[{rlo:.0f}x,{rhi:.0f}x]  <- NE PAS mettre en avant (FULL chevauche 0)")
        print(f"    Bonus H_cont (diversite) : difference FROZEN-FULL {dh:+.3f} "
              f"CI[{dhlo:+.3f},{dhhi:+.3f}] (effondrement quand u gele)")
    print("\n  --> RECOMMANDATION PREPRINT : exprimer le resultat central comme un SAUT de synchronie")
    print("      (difference + separation complete + Cohen d), PAS comme un ratio 'x90' : le")
    print("      denominateur FULL ~ 0 (parfois negatif) rend le ratio instable et non defendable.")

    with CSV.open("w", encoding="utf-8") as f:
        f.write("topo,seed,full_sync,frozen_sync,full_hcont,frozen_hcont\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]:.6f},{r[3]:.6f},{r[4]:.6f},{r[5]:.6f}\n")
    with SUMMARY_CSV.open("w", encoding="utf-8") as f:
        f.write("topo,full_sync_mean,frozen_sync_mean,diff_mean,diff_ci_lo,diff_ci_hi,"
                "cohen_d,separation_complete,max_full,min_frozen\n")
        for r in summary_out:
            f.write(f"{r[0]},{r[1]:.6f},{r[2]:.6f},{r[3]:.6f},{r[4]:.6f},{r[5]:.6f},"
                    f"{r[6]:.4f},{r[7]},{r[8]:.6f},{r[9]:.6f}\n")
    print(f"\n[csv] {CSV}\n[csv] {SUMMARY_CSV}")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
        for ax, topo in zip(axes, TOPOS):
            r = results[topo]
            # nuage par seed FULL vs FROZEN + barres moyennes avec IC
            ax.scatter(np.zeros(len(r["fs"])) + 0.05 * RNG.randn(len(r["fs"])), r["fs"],
                       s=16, c="#1f77b4", alpha=0.6, label="FULL (u actif)")
            ax.scatter(np.ones(len(r["zs"])) + 0.05 * RNG.randn(len(r["zs"])), r["zs"],
                       s=16, c="#d62728", alpha=0.6, label="FROZEN_U (u gele)")
            s = summary[topo]
            ax.errorbar([0], [s["fm"]], yerr=[[s["fm"]-s["flo"]], [s["fhi"]-s["fm"]]],
                        fmt="o", c="k", capsize=5, ms=8)
            ax.errorbar([1], [s["zm"]], yerr=[[s["zm"]-s["zlo"]], [s["zhi"]-s["zm"]]],
                        fmt="o", c="k", capsize=5, ms=8)
            ax.axhline(0, color="gray", lw=0.8, ls=":")
            ax.set_xticks([0, 1]); ax.set_xticklabels(["FULL", "FROZEN_U"])
            ax.set_ylabel("Pairwise synchrony (Pearson)")
            ax.set_title(f"{topo} (30 seeds)"); ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=8)
        fig.suptitle("B4 : ablation FROZEN_U -> surge de synchronie. Le resultat est un SAUT "
                     "(separation complete), pas un ratio instable.", fontsize=10.5)
        plt.tight_layout(); plt.savefig(PNG, dpi=140)
        print(f"[png] {PNG}")
    except Exception as e:
        print(f"[png] skipped: {e}")
    print(f"\nWall time: {time.time()-t0:.1f}s")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
