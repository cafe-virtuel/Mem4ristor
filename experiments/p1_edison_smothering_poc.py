#!/usr/bin/env python3
"""
P1 (legs de Fable) -- L'attaque symetrique d'Edison : l'etouffement SAGE.
==========================================================================
Cree : 2026-07-12 (Claude Fable 5, L'Ingenieur) -- piste P1 de
docs/PISTES_POUR_LA_SUITE_2026-07-12.md.

CONTEXTE. Contre-audit Edison (03/02/2026, 'Analyse de KIMI V2.md', ancien
dossier D:/ANTIGRAVITY/Mem4ristor) : l'arbitre a hysteresis Sage/Fou (Schmitt
trigger theta_low=0.35 / theta_high=0.65, cfg['hysteresis']) a ete valide
contre le scintillement, mais Edison a demontre l'attaque symetrique : **un
signal maintenu dans la bande morte (u in [0.36, 0.64]) verrouille le systeme
en SAGE pour toujours** ('deni de service par etouffement -- le Fou n'est
jamais active', donc innovation_mask=0, donc plasticite w jamais activee).
Sa V5b (timeout + exploration forcee epsilon-greedy) n'a JAMAIS ete
implementee. Le watchdog natif du coeur (commit 06cb6a9, 07/07) resout le
verrouillage FOU -- personne n'a verifie s'il immunise aussi contre
l'etouffement SAGE. Et le coeur possede une DEFENSE DORMANTE jamais evaluee
contre cette attaque : 'fatigue_rate' (les seuils effectifs convergent vers
0.5 avec le temps passe dans l'etat -- un timeout adaptatif V5b avant l'heure,
present dans _update_hysteresis mais a 0.0 par defaut).

PROTOCOLE. L'adversaire pilote le desaccord percu via sigma_social_override
(la voie d'attaque exacte d'Edison : u relaxe vers K_u*sigma + baseline, donc
sigma(t) = 0.45 + 0.10*sin(2*pi*t/200) maintient u dans ~[0.40, 0.60], au
coeur de la bande morte, sans jamais franchir 0.65). 4 conditions x 5 seeds.
DUREE : 12000 pas -- le lancement 1 (3000 pas) a montre que la dynamique de u
est LENTE (tau_eff = tau_u/eps_adapt ~ 3000-5000 pas selon sigma) : a 3000
pas, u n'avait pas encore ATTEINT la bande morte (verrouillage trivial) ni
franchi 0.65 dans le controle positif. 12000 pas ~ 2-4 tau_eff + 17 cycles
complets du watchdog (300+400) :
  HONEST_HIGH     : sigma=1.0 constant, coeur par defaut -> controle positif
                    (le trigger doit basculer FOU : u -> ~1.0 > 0.65).
  ATTACK          : attaque Edison, coeur par defaut -> reproduction du
                    verrouillage (FOU jamais active).
  ATTACK_FATIGUE  : attaque + fatigue_rate=0.003 (defense dormante du coeur :
                    shift de 0.15 en ~1000 pas -> theta_high effectif descend
                    vers 0.5, sous le u impose par l'adversaire).
  ATTACK_WATCHDOG : attaque + consolidation_watchdog aux defauts du coeur
                    (300/400, u_sage=0.05, u_fou=0.9) -> le kick force
                    u=0.9 >= 0.65 au debut de chaque exploration ; l'hysteresis
                    doit MAINTENIR le mode FOU tant que u > 0.35 meme quand
                    l'adversaire ramene u vers 0.5.

CRITERES PRE-FIXES (avant de voir un chiffre) :
  - trigger sain        : HONEST_HIGH  fou_frac > 0.50
  - attaque reproduite  : ATTACK       fou_frac < 0.01 et aucun noeud jamais FOU
  - fatigue immunise    : ATTACK_FATIGUE  fou_frac > 0.10
  - watchdog immunise   : ATTACK_WATCHDOG fou_frac > 0.15
  Si NI la fatigue NI le watchdog n'immunisent -> implementer le timeout V5b
  d'Edison dans le watchdog et re-tester (prevu par la piste P1).

Statut : exploratoire, hors preprint, coeur non touche (defenses testees via
cfg, mecanismes deja presents). Sorties : figures/p1_edison_smothering_poc.csv
+ _agg.csv + .png
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
from mem4ristor.topology import Mem4Network  # noqa: E402

CSV_PATH = ROOT / "figures" / "p1_edison_smothering_poc.csv"
AGG_PATH = ROOT / "figures" / "p1_edison_smothering_poc_agg.csv"
PNG_PATH = ROOT / "figures" / "p1_edison_smothering_poc.png"

SIDE, N = 10, 100
T_SIM = 12000
SEEDS = [0, 1, 2, 3, 4]
ATTACK_MEAN = 0.45
ATTACK_AMP = 0.10
ATTACK_PERIOD = 200
FATIGUE_RATE = 0.003
CONDITIONS = ["HONEST_HIGH", "ATTACK", "ATTACK_FATIGUE", "ATTACK_WATCHDOG"]


def sigma_of_t(t: int, condition: str) -> float:
    if condition == "HONEST_HIGH":
        return 1.0
    return ATTACK_MEAN + ATTACK_AMP * np.sin(2.0 * np.pi * t / ATTACK_PERIOD)


def run(condition: str, seed: int):
    net = Mem4Network(size=SIDE, heretic_ratio=0.0, seed=seed)
    if condition == "ATTACK_FATIGUE":
        net.model.cfg['hysteresis']['fatigue_rate'] = FATIGUE_RATE
    if condition == "ATTACK_WATCHDOG":
        net.model.cfg['consolidation_watchdog'] = {
            'enabled': True, 't_explore': 300, 't_consolidate': 400,
            'u_sage': 0.05, 'u_fou': 0.9,
        }
    fou_frac = np.empty(T_SIM)
    u_tr = np.empty(T_SIM)
    ever_fou = np.zeros(N, dtype=bool)
    for t in range(T_SIM):
        s = sigma_of_t(t, condition)
        net.step(sigma_social_override=np.full(N, s))
        ms = net.model.mode_state
        fou_frac[t] = float(ms.mean())
        ever_fou |= ms
        u_tr[t] = float(net.model.u.mean())
    in_band = float(np.mean((u_tr >= 0.35) & (u_tr <= 0.65)))
    t_first = int(np.argmax(fou_frac > 0)) + 1 if np.any(fou_frac > 0) else T_SIM + 1
    return {
        "fou_frac_time": float(fou_frac.mean()),
        "fou_frac_late": float(fou_frac[T_SIM // 2:].mean()),
        "pct_nodes_ever_fou": float(ever_fou.mean()),
        "t_first_fou": t_first,
        "u_in_deadband_frac": in_band,
        "fou_trace": fou_frac,
        "u_trace": u_tr,
    }


def main() -> int:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = []
    traces = {}
    print(f"P1 -- attaque d'Edison (etouffement SAGE), {len(CONDITIONS)} conditions "
          f"x {len(SEEDS)} seeds, T={T_SIM} pas\n")
    print(f"{'condition':<17}{'seed':>5}{'fou_frac':>10}{'fou_late':>10}"
          f"{'%ever_fou':>11}{'t_1er_fou':>11}{'u_in_band':>11}")
    print("-" * 76)
    for cond in CONDITIONS:
        for seed in SEEDS:
            r = run(cond, seed)
            rows.append((cond, seed, r["fou_frac_time"], r["fou_frac_late"],
                         r["pct_nodes_ever_fou"], r["t_first_fou"],
                         r["u_in_deadband_frac"]))
            if seed == 0:
                traces[cond] = (r["fou_trace"], r["u_trace"])
            print(f"{cond:<17}{seed:>5}{r['fou_frac_time']:>10.3f}"
                  f"{r['fou_frac_late']:>10.3f}{r['pct_nodes_ever_fou']:>11.2f}"
                  f"{r['t_first_fou']:>11}{r['u_in_deadband_frac']:>11.2f}")

    # ---------------- agregats + verdict ----------------
    agg = {}
    for cond in CONDITIONS:
        vals = [r for r in rows if r[0] == cond]
        agg[cond] = {
            "fou_frac": np.mean([v[2] for v in vals]),
            "ever_fou": np.mean([v[4] for v in vals]),
            "never": all(v[5] > T_SIM for v in vals),
        }

    print("\n=== VERDICT P1 (criteres pre-fixes) ===")
    ok_trigger = agg["HONEST_HIGH"]["fou_frac"] > 0.50
    print(f"  1. Trigger sain (HONEST_HIGH fou_frac>{0.50}) : "
          f"{agg['HONEST_HIGH']['fou_frac']:.3f} -> {'OK' if ok_trigger else 'ECHEC'}")
    reproduced = agg["ATTACK"]["fou_frac"] < 0.01 and agg["ATTACK"]["never"]
    print(f"  2. Attaque reproduite (ATTACK fou_frac<0.01, jamais FOU) : "
          f"{agg['ATTACK']['fou_frac']:.4f}, jamais_fou={agg['ATTACK']['never']} "
          f"-> {'VERROUILLAGE CONFIRME' if reproduced else 'attaque inefficace'}")
    fat = agg["ATTACK_FATIGUE"]["fou_frac"]
    print(f"  3. Defense 'fatigue' dormante (>0.10) : {fat:.3f} -> "
          f"{'IMMUNISE' if fat > 0.10 else 'insuffisante'}")
    wd = agg["ATTACK_WATCHDOG"]["fou_frac"]
    print(f"  4. Defense watchdog natif (>0.15) : {wd:.3f} -> "
          f"{'IMMUNISE' if wd > 0.15 else 'insuffisant'}")
    if fat <= 0.10 and wd <= 0.15:
        print("  -> AUCUNE defense native ne suffit : implementer le timeout V5b")
        print("     d'Edison dans le watchdog (piste P1, etape 2) et re-tester.")
    else:
        winners = []
        if wd > 0.15:
            winners.append("watchdog (07/07)")
        if fat > 0.10:
            winners.append("fatigue (dormante depuis V4)")
        print(f"  -> La V5b d'Edison est couverte nativement par : {', '.join(winners)}.")

    # ---------------- CSV ----------------
    with CSV_PATH.open("w", encoding="utf-8") as f:
        f.write("condition,seed,fou_frac_time,fou_frac_late,pct_nodes_ever_fou,"
                "t_first_fou,u_in_deadband_frac\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    with AGG_PATH.open("w", encoding="utf-8") as f:
        f.write("condition,fou_frac_time_mean,pct_nodes_ever_fou_mean,never_fou_all_seeds\n")
        for cond in CONDITIONS:
            f.write(f"{cond},{agg[cond]['fou_frac']:.6f},{agg[cond]['ever_fou']:.6f},"
                    f"{agg[cond]['never']}\n")
    print(f"\n[csv] {CSV_PATH}\n[csv] {AGG_PATH}")

    # ---------------- figure ----------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        colors = {"HONEST_HIGH": "#2ca02c", "ATTACK": "#d62728",
                  "ATTACK_FATIGUE": "#ff7f0e", "ATTACK_WATCHDOG": "#1f77b4"}
        for cond in CONDITIONS:
            fou, u = traces[cond]
            axes[0].plot(fou, color=colors[cond], lw=1.0, label=cond)
            axes[1].plot(u, color=colors[cond], lw=1.0, label=cond)
        axes[0].set_ylabel("fraction de noeuds en mode FOU")
        axes[0].set_title("L'attaque d'Edison : le Fou est-il etouffe ? (seed 0)")
        axes[0].grid(alpha=0.3)
        axes[0].legend(fontsize=8)
        axes[1].axhspan(0.35, 0.65, color="gray", alpha=0.15, label="bande morte")
        axes[1].set_ylabel("u moyen")
        axes[1].set_xlabel("pas")
        axes[1].grid(alpha=0.3)
        axes[1].legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(PNG_PATH, dpi=140)
        print(f"[png] {PNG_PATH}")
    except Exception as e:
        print(f"[png] skipped: {e}")

    print(f"\nWall time: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
