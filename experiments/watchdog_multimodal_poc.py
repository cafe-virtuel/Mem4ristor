#!/usr/bin/env python3
"""
POC B1b -- VALIDATION DE L'UTILITE DU WATCHDOG NATIF (cycle FOU<->SAGE dans dynamics.py).

Contexte. Le 2026-07-07 un watchdog de consolidation opt-in a ete ajoute au coeur
(dynamics.py:363). Il fait NATIVEMENT ce que bicameral_multimodal_poc.py fait DE L'EXTERIEUR :
alterner exploration (FOU, u haut) et consolidation (SAGE, u=0.05). Diagnostic acquis :
le cycle MARCHE (u parcourt [0.05, 0.9]). Question OUVERTE : son UTILITE. Produit-il des
solutions AUSSI BONNES (voire meilleures) que le pilotage externe, et mieux que le hasard ?

Ce script ajoute une 4e condition WATCHDOG au protocole multi-modal identique
(deux capteurs opposes forts : A=+E coin 0, B=-E coin 99 ; toute solution valide respecte
A>0.4 et B<-0.4 ; l'interface entre domaines + et - est libre => famille de solutions valides).

Conditions comparees (memes metriques, meme probleme, memes seeds) :
  BICAMERAL : FOU (epsilon_u=0.02, u libre) -> SAGE (u=0.05) pilotes DE L'EXTERIEUR.
  WATCHDOG  : IDENTIQUE mais pilote DE L'INTERIEUR par dynamics.py (KICK u=0.9 en debut de FOU).
  HASARD    : reinit aleatoire -> SAGE. (exploration = bruit)
  ATTRACTIF : SAGE continu. (coince sur une interface)

La comparaison decisive est BICAMERAL (externe, bruit-driven) vs WATCHDOG (natif, kick-driven) :
c'est le meme rythme, seule change la SOURCE de la remontee du doute.

Metriques (par run, moyennees sur seeds) : valid_frac, coverage, sharpness, consec_dist
(voir bicameral_multimodal_poc.py pour les definitions).

Sortie : figures/watchdog_multimodal_poc.csv + .png + verdict.
Cree : 2026-07-07 (Claude Opus 4.8) -- validation B1b (docs/FUTURE_WORK.md).
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

CSV = ROOT / "figures" / "watchdog_multimodal_poc.csv"
PNG = ROOT / "figures" / "watchdog_multimodal_poc.png"

SIDE, N = 10, 100
N_CYCLES = 12
T_FOU, T_SAGE = 300, 400
U_SAGE = 0.05
U_FOU = 0.9                   # amplitude du KICK natif (= defaut watchdog)
E = 1.0                       # evidence FORTE : contrainte nette A>0, B<0
SEEDS = [0, 1, 2, 3, 4]
A, B = 0, N - 1
stim = np.zeros(N); stim[A] = +E; stim[B] = -E
DECIDED = 0.4                 # |v| > DECIDED => noeud "decide"

def sig(v):
    s = np.zeros(N, dtype=int); s[v > DECIDED] = 1; s[v < -DECIDED] = -1
    return tuple(s)

def is_valid(v):
    return v[A] > DECIDED and v[B] < -DECIDED

def sharpness(v):
    return float(np.mean(np.abs(v) > DECIDED))

def consolidate(net, T):
    """Consolidation pilotee de l'exterieur (u pin a U_SAGE, doute coupe)."""
    net.model.cfg["doubt"]["epsilon_u"] = 0.0
    net.model.u[:] = U_SAGE
    for _ in range(T):
        net.model.u[:] = U_SAGE
        net.step(I_stimulus=stim)
    return net.model.v.copy()

def run_external(condition, seed):
    """BICAMERAL / BICAMERAL_KICK / HASARD / ATTRACTIF : exploration + consolidation externes."""
    net = Mem4Network(size=SIDE, heretic_ratio=0.0, seed=seed)
    rng = np.random.RandomState(1000 + seed)
    sols = []
    for _c in range(N_CYCLES):
        if condition == "BICAMERAL":
            # FOU bruit-driven : u remonte seul depuis le consensus (via le bruit), pas de kick.
            net.model.cfg["doubt"]["epsilon_u"] = 0.02
            for _ in range(T_FOU):
                net.step(I_stimulus=stim)
        elif condition == "BICAMERAL_KICK":
            # CONTROLE : meme rythme externe, mais KICK u=U_FOU en debut de FOU (comme le watchdog).
            # Isole l'effet du kick de l'effet "natif". Doit egaler WATCHDOG si le coeur est fidele.
            net.model.cfg["doubt"]["epsilon_u"] = 0.02
            net.model.u[:] = U_FOU
            for _ in range(T_FOU):
                net.step(I_stimulus=stim)
        elif condition == "HASARD":
            net.model.v[:] = rng.uniform(-1.5, 1.5, N)
            net.model.w[:] = rng.uniform(0.0, 1.0, N)
        elif condition == "ATTRACTIF":
            pass
        sols.append(consolidate(net, T_SAGE))
    return sols

def run_watchdog(seed):
    """WATCHDOG : le cycle FOU<->SAGE est pilote DE L'INTERIEUR par dynamics.py.
    On n'ecrit JAMAIS net.model.u ici : c'est le coeur qui gere u. On echantillonne
    une solution consolidee a chaque fin de phase SAGE (transition consolidating True->False)."""
    net = Mem4Network(size=SIDE, heretic_ratio=0.0, seed=seed)
    net.model.cfg["doubt"]["epsilon_u"] = 0.02
    net.model.cfg["consolidation_watchdog"] = {
        "enabled": True, "t_explore": T_FOU, "t_consolidate": T_SAGE,
        "u_sage": U_SAGE, "u_fou": U_FOU,
    }
    sols = []
    prev_consolidating = False
    steps = 0
    max_steps = N_CYCLES * (T_FOU + T_SAGE) + 2 * (T_FOU + T_SAGE)  # marge de securite
    while len(sols) < N_CYCLES and steps < max_steps:
        net.step(I_stimulus=stim)
        steps += 1
        now = bool(net.model.watchdog_consolidating)
        if prev_consolidating and not now:
            # fin d'une phase SAGE : v est le consolide propre (le KICK ne touche que u)
            sols.append(net.model.v.copy())
        prev_consolidating = now
    if len(sols) < N_CYCLES:
        print(f"  [warn] WATCHDOG seed={seed}: {len(sols)}/{N_CYCLES} solutions "
              f"en {steps} pas (cycle non atteint ?)")
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

def run(condition, seed):
    if condition == "WATCHDOG":
        return run_watchdog(seed)
    return run_external(condition, seed)

def main():
    CSV.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    conds = ["BICAMERAL", "BICAMERAL_KICK", "WATCHDOG", "HASARD", "ATTRACTIF"]
    agg = {c: {"valid": [], "cov": [], "sharp": [], "consec": []} for c in conds}
    print(f"{'condition':<15}{'seed':>5}{'valid_frac':>12}{'coverage':>10}{'sharpness':>11}{'consec_d':>10}")
    print("-" * 63)
    rows = []
    for seed in SEEDS:
        for cond in conds:
            vf, cov, sh, cd = analyse(run(cond, seed))
            agg[cond]["valid"].append(vf); agg[cond]["cov"].append(cov)
            agg[cond]["sharp"].append(sh); agg[cond]["consec"].append(cd)
            rows.append((cond, seed, vf, cov, sh, cd))
            print(f"{cond:<15}{seed:>5}{vf:>12.2f}{cov:>10}{sh:>11.2f}{cd:>10.2f}")

    print(f"\n{'condition':<15}{'valid_frac':>12}{'coverage':>10}{'sharpness':>11}{'consec_d':>10}")
    print("-" * 58)
    summ = {}
    for c in conds:
        vf = np.mean(agg[c]["valid"]); cov = np.mean(agg[c]["cov"])
        sh = np.mean(agg[c]["sharp"]); cd = np.mean(agg[c]["consec"])
        summ[c] = (vf, cov, sh, cd)
        print(f"{c:<15}{vf:>12.2f}{cov:>10.1f}{sh:>11.2f}{cd:>10.2f}")

    bi, bk = summ["BICAMERAL"], summ["BICAMERAL_KICK"]
    wd, ha, at = summ["WATCHDOG"], summ["HASARD"], summ["ATTRACTIF"]
    print("\n=== VERDICT B1b (honnete) ===")
    print("Q1 : le watchdog NATIF vaut-il le pilotage EXTERNE, et bat-il le hasard ?")
    print(f"  VALIDITE   : WATCHDOG={wd[0]:.2f}  BICAMERAL={bi[0]:.2f}  BIC_KICK={bk[0]:.2f}  HASARD={ha[0]:.2f}  ATTRACTIF={at[0]:.2f}")
    print(f"  COUVERTURE : WATCHDOG={wd[1]:.1f}  BICAMERAL={bi[1]:.1f}  BIC_KICK={bk[1]:.1f}  HASARD={ha[1]:.1f}  ATTRACTIF={at[1]:.1f}")
    print(f"  STRUCTURE  : WATCHDOG={wd[3]:.2f}  BICAMERAL={bi[3]:.2f}  BIC_KICK={bk[3]:.2f}  HASARD={ha[3]:.2f}  (dist. consec.)")

    print("\n  Lecture Q1 (utilite vs hasard) :")
    if wd[0] >= bi[0] * 0.9 and wd[0] > ha[0] * 1.2:
        print("  -> UTILE : le cycle natif tient la validite au niveau du pilotage externe,")
        print("     et bien au-dessus du hasard. Le KICK remplace le bruit sans perte.")
    elif wd[0] > ha[0] * 1.2:
        print(f"  -> PARTIEL : le natif bat le hasard mais reste sous l'externe ({wd[0]:.2f} vs {bi[0]:.2f}).")
    else:
        print("  -> NON CONCLUANT : le cycle natif ne se distingue pas du hasard sur la validite.")

    print("\nQ2 : la couverture superieure vient-elle du KICK ou du 'natif' ? (controle BIC_KICK)")
    # Si WATCHDOG ~ BIC_KICK : c'est le KICK (l'ingredient), et le coeur le reproduit fidelement.
    # Si WATCHDOG >> BICAMERAL mais ~ BIC_KICK : couverture+ = effet du kick, pas de la nativite.
    rel = abs(wd[1] - bk[1]) / max(bk[1], 1e-9)
    print(f"  COUVERTURE : WATCHDOG={wd[1]:.1f}  vs  BIC_KICK={bk[1]:.1f}  (ecart relatif {rel*100:.0f}%)")
    if wd[1] > bi[1] * 1.15 and rel <= 0.25:
        print("  -> La couverture+ vient du KICK : BIC_KICK (externe+kick) egale WATCHDOG (natif+kick),")
        print("     tous deux au-dessus du BICAMERAL bruit-driven. Le watchdog INTERNALISE le kick")
        print("     dans le coeur (fidelement) -- c'est ca son apport : plus besoin de piloter u dehors.")
    elif wd[1] > bk[1] * 1.15:
        print("  -> Le natif fait MIEUX que le kick externe : effet de timing propre au cycle interne.")
    elif wd[1] < bk[1] * 0.85:
        print("  -> Le natif fait MOINS que le kick externe : le cycle interne perd en exploration.")
    else:
        print("  -> Kick externe et natif equivalents.")

    with CSV.open("w", encoding="utf-8") as f:
        f.write("condition,seed,valid_frac,coverage,sharpness,consec_dist\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]:.4f},{r[3]},{r[4]:.4f},{r[5]:.4f}\n")
    print(f"\n[csv] {CSV}")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
        colors = {"BICAMERAL": "#2ca02c", "BICAMERAL_KICK": "#98df8a", "WATCHDOG": "#d62728",
                  "HASARD": "#7f7f7f", "ATTRACTIF": "#1f77b4"}
        for ax, (idx, lab) in zip(axes, [(0, "Valid fraction"),
                                         (1, "Valid distinct solutions (coverage)"),
                                         (3, "Consecutive distance")]):
            ax.bar(conds, [summ[c][idx] for c in conds],
                   color=[colors[c] for c in conds], edgecolor="k")
            ax.set_title(lab); ax.grid(axis="y", alpha=0.3); ax.tick_params(axis="x", rotation=20)
        fig.suptitle("Watchdog natif vs pilotage externe : le cycle FOU<->SAGE interne est-il utile ?",
                     fontsize=11)
        plt.tight_layout(); plt.savefig(PNG, dpi=140)
        print(f"[png] {PNG}")
    except Exception as e:
        print(f"[png] skipped: {e}")
    print(f"\nWall time: {time.time()-t0:.1f}s")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
