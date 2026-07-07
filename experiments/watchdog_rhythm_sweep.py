#!/usr/bin/env python3
"""
POC B1b (suite) -- SWEEP DU RYTHME du watchdog natif (T_FOU x T_SAGE).

Le watchdog validé (watchdog_multimodal_poc.py) tournait à un rythme FIXE (T_FOU=300,
T_SAGE=400) et couvrait ~6 solutions valides distinctes. Question : jusqu'ou la
COUVERTURE monte-t-elle, et existe-t-il un rythme OPTIMAL ?

Hypotheses :
  - T_FOU court  -> exploration insuffisante -> couverture faible.
  - T_FOU long   -> derive -> peut casser la validite (A>0.4 / B<-0.4).
  - T_SAGE court -> consolidation incomplete -> solutions floues / invalides.
  - T_SAGE long  -> sur-consolidation -> retombe sur la MEME interface -> couverture basse.
  => on s'attend a un OPTIMUM de couverture a rythme intermediaire, sous contrainte de validite.

On ne mesure que la condition WATCHDOG (natif, u pilote de l'interieur par dynamics.py).
Meme probleme multi-modal (A=+E coin 0, B=-E coin 99). Metriques : valid_frac, coverage,
sharpness, consec_dist (voir watchdog_multimodal_poc.py).

Grille : T_FOU in {100,200,300,500,800} x T_SAGE in {150,300,400,600}. Couverture plafonnee
a N_CYCLES=12 (nb d'echantillons). Sortie : figures/watchdog_rhythm_sweep.csv + 2 heatmaps.
Cree : 2026-07-07 (Claude Opus 4.8) -- suite de la validation B1b.
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
from mem4ristor.topology import Mem4Network  # noqa: E402

CSV = ROOT / "figures" / "watchdog_rhythm_sweep.csv"
PNG = ROOT / "figures" / "watchdog_rhythm_sweep.png"

SIDE, N = 10, 100
N_CYCLES = 12
U_SAGE, U_FOU = 0.05, 0.9
E = 1.0
SEEDS = [0, 1, 2, 3, 4]
A, B = 0, N - 1
stim = np.zeros(N); stim[A] = +E; stim[B] = -E
DECIDED = 0.4

T_FOU_GRID = [100, 200, 300, 500, 800]
T_SAGE_GRID = [150, 300, 400, 600]

def sig(v):
    s = np.zeros(N, dtype=int); s[v > DECIDED] = 1; s[v < -DECIDED] = -1
    return tuple(s)

def is_valid(v):
    return v[A] > DECIDED and v[B] < -DECIDED

def sharpness(v):
    return float(np.mean(np.abs(v) > DECIDED))

def run_watchdog(seed, t_fou, t_sage):
    """Cycle FOU<->SAGE pilote DE L'INTERIEUR. On n'ecrit jamais net.model.u ici."""
    net = Mem4Network(size=SIDE, heretic_ratio=0.0, seed=seed)
    net.model.cfg["doubt"]["epsilon_u"] = 0.02
    net.model.cfg["consolidation_watchdog"] = {
        "enabled": True, "t_explore": t_fou, "t_consolidate": t_sage,
        "u_sage": U_SAGE, "u_fou": U_FOU,
    }
    sols = []
    prev = False
    steps = 0
    max_steps = N_CYCLES * (t_fou + t_sage) + 2 * (t_fou + t_sage)
    while len(sols) < N_CYCLES and steps < max_steps:
        net.step(I_stimulus=stim)
        steps += 1
        now = bool(net.model.watchdog_consolidating)
        if prev and not now:
            sols.append(net.model.v.copy())
        prev = now
    return sols

def analyse(sols):
    if not sols:
        return 0.0, 0, 0.0, 0.0
    valid = [s for s in sols if is_valid(s)]
    valid_frac = len(valid) / len(sols)
    coverage = len({sig(s) for s in valid})
    sharp = float(np.mean([sharpness(s) for s in sols]))
    if len(sols) > 1:
        consec = float(np.mean([np.linalg.norm(sols[i] - sols[i - 1]) / np.sqrt(N)
                                for i in range(1, len(sols))]))
    else:
        consec = 0.0
    return valid_frac, coverage, sharp, consec

def main():
    CSV.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    # grilles moyennees sur seeds : [i_fou, j_sage]
    cov_grid = np.zeros((len(T_FOU_GRID), len(T_SAGE_GRID)))
    val_grid = np.zeros_like(cov_grid)
    con_grid = np.zeros_like(cov_grid)
    rows = []
    print(f"{'T_FOU':>6}{'T_SAGE':>7}{'valid':>8}{'cover':>7}{'consec':>8}")
    print("-" * 36)
    for i, tf in enumerate(T_FOU_GRID):
        for j, ts in enumerate(T_SAGE_GRID):
            vfs, covs, cons = [], [], []
            for seed in SEEDS:
                vf, cov, sh, cd = analyse(run_watchdog(seed, tf, ts))
                vfs.append(vf); covs.append(cov); cons.append(cd)
                rows.append((tf, ts, seed, vf, cov, sh, cd))
            cov_grid[i, j] = np.mean(covs)
            val_grid[i, j] = np.mean(vfs)
            con_grid[i, j] = np.mean(cons)
            print(f"{tf:>6}{ts:>7}{val_grid[i,j]:>8.2f}{cov_grid[i,j]:>7.1f}{con_grid[i,j]:>8.2f}")

    # meilleur rythme = couverture max PARMI ceux valides (valid_frac >= 0.9)
    valid_mask = val_grid >= 0.9
    masked_cov = np.where(valid_mask, cov_grid, -1.0)
    bi, bj = np.unravel_index(np.argmax(masked_cov), masked_cov.shape)
    print("\n=== VERDICT rythme (honnete) ===")
    print(f"Couverture MAX sous contrainte validite>=0.9 : {cov_grid[bi,bj]:.1f} solutions distinctes")
    print(f"  a T_FOU={T_FOU_GRID[bi]}, T_SAGE={T_SAGE_GRID[bj]} (validite {val_grid[bi,bj]:.2f}).")
    print(f"  Reference validee (T_FOU=300, T_SAGE=400) : couverture ~6.0.")
    cov_max_overall = cov_grid.max()
    print(f"  Couverture max TOUTES cases (meme peu valides) : {cov_max_overall:.1f} "
          f"(plafond N_CYCLES={N_CYCLES}).")

    # lecture des tendances marginales
    cov_by_fou = cov_grid.mean(axis=1); cov_by_sage = cov_grid.mean(axis=0)
    val_by_sage = val_grid.mean(axis=0)
    print("\n  Tendance T_FOU (couverture moy.) :",
          ", ".join(f"{tf}:{c:.1f}" for tf, c in zip(T_FOU_GRID, cov_by_fou)))
    print("  Tendance T_SAGE (couverture moy.):",
          ", ".join(f"{ts}:{c:.1f}" for ts, c in zip(T_SAGE_GRID, cov_by_sage)))
    print("  Tendance T_SAGE (validite moy.) :",
          ", ".join(f"{ts}:{v:.2f}" for ts, v in zip(T_SAGE_GRID, val_by_sage)))
    if cov_by_fou[-1] > cov_by_fou[0] * 1.1:
        print("  -> plus de FOU = plus de couverture (l'exploration paye).")
    if cov_by_sage[0] > cov_by_sage[-1] * 1.1:
        print("  -> plus de SAGE = MOINS de couverture (sur-consolidation -> meme interface).")

    with CSV.open("w", encoding="utf-8") as f:
        f.write("t_fou,t_sage,seed,valid_frac,coverage,sharpness,consec_dist\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]:.4f},{r[4]},{r[5]:.4f},{r[6]:.4f}\n")
    print(f"\n[csv] {CSV}")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
        for ax, grid, lab, cmap in [
            (axes[0], cov_grid, "Coverage (valid distinct sol.)", "viridis"),
            (axes[1], val_grid, "Valid fraction", "magma"),
        ]:
            im = ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap)
            ax.set_xticks(range(len(T_SAGE_GRID))); ax.set_xticklabels(T_SAGE_GRID)
            ax.set_yticks(range(len(T_FOU_GRID))); ax.set_yticklabels(T_FOU_GRID)
            ax.set_xlabel("T_SAGE (consolidation)"); ax.set_ylabel("T_FOU (exploration)")
            ax.set_title(lab)
            for ii in range(len(T_FOU_GRID)):
                for jj in range(len(T_SAGE_GRID)):
                    ax.text(jj, ii, f"{grid[ii,jj]:.1f}", ha="center", va="center",
                            color="w", fontsize=8)
            fig.colorbar(im, ax=ax, shrink=0.85)
        axes[0].scatter([bj], [bi], marker="*", s=260, c="red", edgecolor="w",
                        label="optimum valide", zorder=5)
        axes[0].legend(loc="lower right", fontsize=8)
        fig.suptitle("Watchdog natif : sweep du rythme FOU x SAGE (couverture & validite)", fontsize=11)
        plt.tight_layout(); plt.savefig(PNG, dpi=140)
        print(f"[png] {PNG}")
    except Exception as e:
        print(f"[png] skipped: {e}")
    print(f"\nWall time: {time.time()-t0:.1f}s")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
