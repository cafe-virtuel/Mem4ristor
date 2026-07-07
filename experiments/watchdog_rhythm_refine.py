#!/usr/bin/env python3
"""
POC B1b (raffinement) -- CONFIRMER LE SOMMET T_SAGE ~ 400.

Le sweep grossier (watchdog_rhythm_sweep.py) montrait un pic ETROIT de couverture ET de
validite a T_SAGE=400 (effondrement des deux cotes : 300->0.66 validite, 600->0.28). Ici on
raffine T_SAGE in {350,400,450,500} sur DEUX lignes T_FOU (300=reference, 500=optimum grossier),
avec 8 seeds pour sortir des barres d'erreur et voir si le sommet 400 est significatif ou un
artefact de maillage.

Reutilise run_watchdog / analyse de watchdog_rhythm_sweep (meme probleme multi-modal, memes
metriques). Sortie : figures/watchdog_rhythm_refine.csv + .png (couverture & validite vs T_SAGE,
une courbe par T_FOU, avec ecart-type inter-seed).
Cree : 2026-07-07 (Claude Opus 4.8).
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
sys.path.insert(0, str(ROOT / "experiments"))
from watchdog_rhythm_sweep import run_watchdog, analyse  # noqa: E402  (reutilisation)

CSV = ROOT / "figures" / "watchdog_rhythm_refine.csv"
PNG = ROOT / "figures" / "watchdog_rhythm_refine.png"

T_FOU_LINES = [300, 500]
T_SAGE_FINE = [350, 400, 450, 500]
SEEDS = list(range(8))

def main():
    CSV.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    # data[tf][ts] = (cov_mean, cov_std, val_mean, val_std)
    data = {tf: {} for tf in T_FOU_LINES}
    rows = []
    print(f"{'T_FOU':>6}{'T_SAGE':>7}{'valid':>8}{'cover':>7}{'cov_sd':>7}{'consec':>8}")
    print("-" * 43)
    for tf in T_FOU_LINES:
        for ts in T_SAGE_FINE:
            vfs, covs, cons = [], [], []
            for seed in SEEDS:
                vf, cov, sh, cd = analyse(run_watchdog(seed, tf, ts))
                vfs.append(vf); covs.append(cov); cons.append(cd)
                rows.append((tf, ts, seed, vf, cov, sh, cd))
            cm, cs = float(np.mean(covs)), float(np.std(covs))
            vm, vs = float(np.mean(vfs)), float(np.std(vfs))
            data[tf][ts] = (cm, cs, vm, vs)
            print(f"{tf:>6}{ts:>7}{vm:>8.2f}{cm:>7.1f}{cs:>7.1f}{np.mean(cons):>8.2f}")

    print("\n=== VERDICT raffinement (honnete) ===")
    for tf in T_FOU_LINES:
        best_ts = max(T_SAGE_FINE, key=lambda ts: data[tf][ts][0] if data[tf][ts][2] >= 0.9 else -1)
        cm, cs, vm, vs = data[tf][best_ts]
        print(f"  T_FOU={tf} : sommet couverture (validite>=0.9) a T_SAGE={best_ts} "
              f"-> {cm:.1f}+/-{cs:.1f} (validite {vm:.2f}).")
        # 400 est-il le pic, ou un plateau/decalage ?
        c400 = data[tf][400][0]; c450 = data[tf][450][0]; c350 = data[tf][350][0]
        if c400 >= c450 and c400 >= c350:
            gap = c400 - max(c350, c450)
            sd = data[tf][400][1]
            verdict = "PIC net" if gap > sd else "sommet mais dans le bruit (plateau 350-450)"
            print(f"    -> 400 est le sommet local ({verdict} ; ecart au 2e {gap:+.1f} vs sd {sd:.1f}).")
        else:
            top = 350 if c350 > c450 else 450
            print(f"    -> le sommet a GLISSE vers T_SAGE={top} (400 n'est pas le max sur cette ligne).")

    with CSV.open("w", encoding="utf-8") as f:
        f.write("t_fou,t_sage,seed,valid_frac,coverage,sharpness,consec_dist\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]:.4f},{r[4]},{r[5]:.4f},{r[6]:.4f}\n")
    print(f"\n[csv] {CSV}")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
        colors = {300: "#1f77b4", 500: "#d62728"}
        for tf in T_FOU_LINES:
            cm = [data[tf][ts][0] for ts in T_SAGE_FINE]
            cs = [data[tf][ts][1] for ts in T_SAGE_FINE]
            vm = [data[tf][ts][2] for ts in T_SAGE_FINE]
            vsd = [data[tf][ts][3] for ts in T_SAGE_FINE]
            axes[0].errorbar(T_SAGE_FINE, cm, yerr=cs, marker="o", capsize=4,
                             color=colors[tf], label=f"T_FOU={tf}")
            axes[1].errorbar(T_SAGE_FINE, vm, yerr=vsd, marker="o", capsize=4,
                             color=colors[tf], label=f"T_FOU={tf}")
        axes[0].axvline(400, ls="--", c="grey", alpha=0.6)
        axes[1].axvline(400, ls="--", c="grey", alpha=0.6)
        axes[0].set_title("Coverage (valid distinct sol.) vs T_SAGE")
        axes[1].set_title("Valid fraction vs T_SAGE"); axes[1].axhline(0.9, ls=":", c="k", alpha=0.4)
        for ax in axes:
            ax.set_xlabel("T_SAGE (consolidation)"); ax.grid(alpha=0.3); ax.legend()
            ax.set_xticks(T_SAGE_FINE)
        fig.suptitle("Raffinement du rythme : le sommet T_SAGE~400 est-il reel ? (8 seeds)", fontsize=11)
        plt.tight_layout(); plt.savefig(PNG, dpi=140)
        print(f"[png] {PNG}")
    except Exception as e:
        print(f"[png] skipped: {e}")
    print(f"\nWall time: {time.time()-t0:.1f}s")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
