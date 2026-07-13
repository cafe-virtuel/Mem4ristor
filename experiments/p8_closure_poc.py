#!/usr/bin/env python3
"""
P8 -- FERMER LE RESIDU SYNC-FULL : un tau plus fin + une analyse d'ordre.
=============================================================================
Cree : 2026-07-13 (Claude Sonnet 5, L'Ingenieur) -- suite de
`p8_colored_noise_rk45_poc.py` (12/07). Reste documente par la piste :
"le residu sync-FULL (+6.6% au tau le plus fin, critere strict 2sd=0.0018
NON atteint) -- fermeture demanderait tau <= 0.0016 (cout x4/pas) OU une
analyse d'ordre."

CE SCRIPT FAIT LES DEUX, a cout raisonnable :
  1. UN point supplementaire, tau=0.0025 (~2.5x plus fin que le tau=0.00625
     deja teste -- pas le tau=0.0016 le plus couteux (x4), un compromis
     cout/information), RK45_OU SEULEMENT (EULER_WHITE deja connu, pas
     recalcule), memes 4 seeds, memes 2 ablations (FULL/FROZEN_U), meme
     gate de fidelite et meme RHS reduit (reutilise make_rhs/make_ou_interp/
     rk45_run de p8_colored_noise_rk45_poc.py par import -- pas de
     duplication).
  2. ANALYSE D'ORDRE : avec les 5 points tau desormais disponibles
     {0.4, 0.1, 0.025, 0.00625, 0.0025}, on ajuste log|r(tau)| = log(C) +
     p*log(tau) (regression log-log) sur le residu r(tau) = sync_RK45_OU(tau)
     - sync_EULER_WHITE, et on extrapole r(tau->0). Egalement rapporte : les
     ordres LOCAUX entre paires consecutives (ratio r(tau_i)/r(tau_i+1) vs
     le ratio des tau, 4x ou 2.5x) -- pour voir si la convergence est un
     POWER LAW propre (ordre constant) ou decelere (signe d'un plancher
     residuel independant de la couleur du bruit).

CRITERE PRE-FIXE (avant de lancer le nouveau tau) : le residu FERME si, au
tau=0.0025, |sync_RK45_OU - sync_EULER_WHITE| <= 2*sd(EULER_WHITE)=0.0018 EN
VALEUR ABSOLUE. Si non : l'analyse d'ordre doit dire EXPLICITEMENT si
l'extrapolation tau->0 suggere un residu qui s'annule (juste besoin de
tau encore plus fin) ou un PLANCHER non-nul (effet independant de
l'integrateur/couleur -- a rapporter comme decouverte, pas comme echec a
cacher).

Statut : exploratoire, hors preprint, coeur non touche, aucun chiffre
canonique modifie (meme perimetre que P8 original).
Sorties : figures/p8_closure_poc.csv + .png
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
import p8_colored_noise_rk45_poc as p8  # noqa: E402  (build_model, rk45_run, observables, SEEDS, CONDITIONS)

CSV_PATH = ROOT / "figures" / "p8_closure_poc.csv"
PNG_PATH = ROOT / "figures" / "p8_closure_poc.png"
AGG_EXISTING = ROOT / "figures" / "p8_colored_noise_rk45_poc_agg.csv"

TAU_NEW = 0.0025    # ~2.5x plus fin que 0.00625, compromis cout/information


def read_existing_agg():
    """Relit les 4 tau deja mesures (12/07) -- pas de recalcul, juste la lecture."""
    rows = []
    with AGG_EXISTING.open() as f:
        header = f.readline().strip().split(",")
        for line in f:
            parts = line.strip().split(",")
            d = dict(zip(header, parts))
            rows.append(d)
    return rows


def main() -> int:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"=== 1. NOUVEAU POINT tau={TAU_NEW} (RK45_OU seulement, {len(p8.SEEDS)} seeds x "
          f"{len(p8.CONDITIONS)} ablations) ===")
    new_rows = []
    for seed in p8.SEEDS:
        for cond in p8.CONDITIONS:
            frozen = cond == "FROZEN_U"
            _, mm, aa = p8.build_model(seed)
            traj = p8.rk45_run(mm, aa, frozen, seed, TAU_NEW)
            if traj is None:
                print(f"  seed={seed} {cond} -> DIVERGE/ECHEC")
                continue
            h, sync = p8.observables(traj)
            new_rows.append((seed, cond, h, sync))
            print(f"  seed={seed} {cond} : H_cont={h:.4f} sync={sync:.4f}  [{time.time()-t0:.0f}s]")

    agg_new = {}
    for cond in p8.CONDITIONS:
        hs = np.array([r[2] for r in new_rows if r[1] == cond])
        ss = np.array([r[3] for r in new_rows if r[1] == cond])
        agg_new[cond] = (float(hs.mean()), float(hs.std()), float(ss.mean()), float(ss.std()))
        print(f"  [{cond}] tau={TAU_NEW} : H_cont={hs.mean():.4f}+-{hs.std():.4f}  "
              f"sync={ss.mean():.4f}+-{ss.std():.4f}")

    # ---------------- assembler la serie complete (5 tau) ----------------
    existing = read_existing_agg()

    def get_existing(integrator, tau_str, cond, field):
        for r in existing:
            if r["integrator"] == integrator and r["tau"] == tau_str and r["condition"] == cond:
                return float(r[field])
        raise KeyError((integrator, tau_str, cond, field))

    print("\n=== 2. CRITERE PRE-FIXE : le residu ferme-t-il au nouveau tau ? ===")
    for cond in p8.CONDITIONS:
        sync_white = get_existing("EULER_WHITE", "", cond, "sync_mean")
        sd_white = get_existing("EULER_WHITE", "", cond, "sync_std")
        sync_new = agg_new[cond][2]
        resid = sync_new - sync_white
        tol = 2 * sd_white
        closed = abs(resid) <= tol
        print(f"  [{cond}] tau={TAU_NEW} : sync_RK45_OU={sync_new:.4f}  sync_EULER_WHITE={sync_white:.4f}  "
              f"residu={resid:+.4f}  tol(2sd)={tol:.4f}  -> {'FERME' if closed else 'TOUJOURS HORS TOLERANCE'}")

    # ---------------- 3. analyse d'ordre (FULL, l'observable qui resistait) ----------------
    print("\n=== 3. ANALYSE D'ORDRE (sync FULL, le residu qui resistait) ===")
    taus_all = [0.4, 0.1, 0.025, 0.00625, TAU_NEW]
    sync_white_full = get_existing("EULER_WHITE", "", "FULL", "sync_mean")
    sync_rk45_full = [get_existing("RK45_OU", "0.4", "FULL", "sync_mean"),
                       get_existing("RK45_OU", "0.1", "FULL", "sync_mean"),
                       get_existing("RK45_OU", "0.025", "FULL", "sync_mean"),
                       get_existing("RK45_OU", "0.00625", "FULL", "sync_mean"),
                       agg_new["FULL"][2]]
    resid = [s - sync_white_full for s in sync_rk45_full]
    print(f"  {'tau':>10}{'sync_RK45_OU':>14}{'residu r(tau)':>16}")
    for t, s, r in zip(taus_all, sync_rk45_full, resid):
        print(f"  {t:>10.5f}{s:>14.4f}{r:>16.5f}")

    print("\n  Ordres LOCAUX entre paires consecutives (ratio residu vs ratio tau) :")
    orders_local = []
    for i in range(len(taus_all) - 1):
        rt = resid[i] / resid[i + 1] if resid[i + 1] != 0 else float("inf")
        tt = taus_all[i] / taus_all[i + 1]
        p_local = np.log(abs(rt)) / np.log(tt) if rt > 0 else float("nan")
        orders_local.append(p_local)
        print(f"    tau {taus_all[i]}->{taus_all[i+1]} (x{tt:.2f}) : "
              f"r={resid[i]:.5f}->{resid[i+1]:.5f}  ordre local p={p_local:.3f}")

    # regression log-log GLOBALE (tous les points ou le residu garde son signe)
    log_tau = np.log(taus_all)
    log_r = np.log(np.abs(resid))
    A = np.vstack([log_tau, np.ones_like(log_tau)]).T
    p_fit, log_c = np.linalg.lstsq(A, log_r, rcond=None)[0]
    pred = A @ np.array([p_fit, log_c])
    ss_res = np.sum((log_r - pred) ** 2)
    ss_tot = np.sum((log_r - log_r.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    print(f"\n  Regression log-log GLOBALE (5 points) : ordre ajuste p={p_fit:.3f}, R^2={r2:.4f}")
    c_fit = np.exp(log_c)
    resid_extrap_at_1e5 = c_fit * (1e-5 ** p_fit)
    print(f"  Extrapolation a tau=1e-5 (bien au-dela de ce qui est teste) : "
          f"residu predit ~ {resid_extrap_at_1e5:.6f}")

    deceleration = orders_local[-1] < orders_local[0] * 0.7
    sd_white_full = get_existing("EULER_WHITE", "", "FULL", "sync_std")
    tol_full = 2 * sd_white_full
    last_resid_ok = abs(resid[-1]) <= tol_full
    print("\n=== VERDICT P8 (fermeture du residu) ===")
    if last_resid_ok:
        print(f"  -> RESIDU FERME au nouveau tau={TAU_NEW} : |{resid[-1]:.5f}| <= 2sd={tol_full:.5f}. "
              "La reserve numerique tombe sur sync FULL aussi (pas seulement H_cont).")
    elif not deceleration and p_fit > 0.5:
        print(f"  -> PAS ENCORE FERME mais la regression est propre (R^2={r2:.3f}, ordre~{p_fit:.2f}, "
              "pas de deceleration marquee) : le residu continue de decroitre en loi de puissance, "
              "l'extrapolation soutient qu'un tau encore plus fin (0.0016 comme prevu par la piste, "
              "ou au-dela) fermerait le residu -- juste une question de cout de calcul, pas d'un "
              "biais structurel.")
    else:
        print(f"  -> PAS FERME et la convergence DECELERE (ordre local passe de "
              f"{orders_local[0]:.2f} a {orders_local[-1]:.2f}) : le residu ne suit PAS un simple "
              "power-law jusqu'a ce tau -- signe possible d'un PLANCHER independant de la couleur "
              "du bruit / de l'integrateur (par ex. l'ecart entre le RHS REDUIT teste ici et le "
              "pipeline step() complet, ou un biais numerique de la regression a peu de points). "
              "A rapporter tel quel : le critere strict 2sd reste HORS d'atteinte a ce budget de "
              "calcul, et l'analyse d'ordre ne permet PAS d'affirmer qu'un tau plus fin suffirait "
              "seul a le fermer.")

    with CSV_PATH.open("w", encoding="utf-8") as f:
        f.write("tau,sync_rk45_ou_full,residu_full,ordre_local\n")
        for i, (t, s, r) in enumerate(zip(taus_all, sync_rk45_full, resid)):
            ol = orders_local[i] if i < len(orders_local) else ""
            f.write(f"{t},{s:.6f},{r:.6f},{ol}\n")
        f.write(f"\n# fit global: p={p_fit:.4f} R2={r2:.4f} c={c_fit:.6e}\n")
    print(f"\n[csv] {CSV_PATH}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.5, 5))
        ax.loglog(taus_all, np.abs(resid), "o", ms=8, color="#d62728", label="|residu| mesure")
        tt = np.logspace(np.log10(min(taus_all) * 0.5), np.log10(max(taus_all) * 1.5), 50)
        ax.loglog(tt, c_fit * tt ** p_fit, "--", color="#1f77b4",
                  label=f"fit log-log (p={p_fit:.2f}, R2={r2:.2f})")
        ax.axhline(tol_full, ls=":", c="gray", label=f"tolerance 2sd={tol_full:.4f}")
        ax.set_xlabel("tau (bruit OU, log)"); ax.set_ylabel("|sync_RK45_OU - sync_EULER_WHITE| (log)")
        ax.set_title("P8 -- fermeture du residu sync-FULL : analyse d'ordre")
        ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
        plt.tight_layout()
        plt.savefig(PNG_PATH, dpi=140)
        print(f"[png] {PNG_PATH}")
    except Exception as e:
        print(f"[png] skipped: {e}")

    print(f"\nWall time: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
