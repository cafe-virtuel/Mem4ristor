#!/usr/bin/env python3
"""
P11 -- WARM START : LE PRENDRE EN DEFAUT (deux contre-epreuves).
=============================================================================
Cree : 2026-07-13 (Claude Sonnet 5, L'Ingenieur). Demande explicite de
Julien apres le resultat spectaculaire (+97%) de `p11_warm_start_poc.py` :
« bien sur que l'on pousse au max de ce qui peut etre teste et on fait
aussi tout pour le prendre en defaut ». Deux angles adversariaux :

CONTRE-EPREUVE 1 -- LE HASARD FAIT-IL DEJA LE TRAVAIL ? Le piege est
TRES asymetrique (bien deviner = quasi gratuit ~45 iters ; mal deviner =
coute a peine plus cher que BLIND, cf. sanity check du POC precedent :
+11 a +37 iters sur ~1000-1700). Si un tirage a PILE OU FACE (aucune
information de M4R) recupere deja une grosse part de l'economie, le
"+97%" est en partie un artefact de la structure du piege, pas de
l'intelligence de M4R. Mesure : economie(hasard 50/50) vs economie(M4R).
La VRAIE contribution de M4R = economie(M4R) - economie(hasard).

CONTRE-EPREUVE 2 -- LE WARM START SURVIT-IL SI LA POSITION DU PIEGE EST
MOINS PREVISIBLE ? X_WARM=1.5 a ete choisi pour etre TOUJOURS au-dela du
piege dans la plage x_p_mag in [0.9, 1.3] du POC original -- un choix
"informe" qui suppose qu'on connait cette plage a l'avance. Ici, x_p_mag
elargi a [0.9, 1.9] : le warm start atterrit PARFOIS a l'interieur ou
pres du piege (verifie explicitement, pas suppose). Teste si l'economie
s'effondre ou se degrade gracieusement.

Statut : exploratoire, hors preprint, aucune modification de dynamics.py.
Guardian doit rester 14/14. Sorties : figures/p11_warm_start_stress_poc.csv
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
import p11_warm_start_poc as p  # noqa: E402

FIG = ROOT / "figures"
SEEDS = list(range(60))


def make_problem_wide(seed, b, x_p_max):
    rng = np.random.RandomState(70000 + seed)
    w_flat = rng.uniform(0.02, 0.03)
    t_c = rng.uniform(700.0, 1400.0)
    x_p_mag = rng.uniform(0.9, x_p_max)
    h_min = 2.0 * w_flat / (p.ETA * x_p_mag * t_c)
    return {"x_p": b * x_p_mag, "w_flat": w_flat, "h_min": h_min, "target": b * p.X_TARGET}


def contre_epreuve_1_hasard():
    print("=== CONTRE-EPREUVE 1 : le hasard fait-il deja le travail ? ===\n")
    rng_coin = np.random.RandomState(42)
    blind_c, warm_c, correct_c = [], [], []
    for seed in SEEDS:
        b = 1 if (seed % 2 == 0) else -1
        pb = p.make_problem(seed, b)
        it_blind = p.solve(pb, x0=0.0)
        b_guess = rng_coin.choice([-1, 1])
        correct_c.append(int(b_guess == b))
        it_warm = p.solve(pb, x0=b_guess * p.X_WARM)
        blind_c.append(it_blind)
        warm_c.append(it_warm)
    blind_c, warm_c, correct_c = map(np.array, (blind_c, warm_c, correct_c))

    p.T_READ, p.B_E = 300, 0.8
    blind_m, warm_m, correct_m = p.run_condition(SEEDS)

    econ_c = float(blind_c.mean() - warm_c.mean())
    econ_m = float(blind_m.mean() - warm_m.mean())
    pct_c = 100 * econ_c / blind_c.mean()
    pct_m = 100 * econ_m / blind_m.mean()
    added_value = econ_m - econ_c
    excess_acc = float(correct_m.mean() - correct_c.mean())

    print(f"  HASARD (piece de monnaie) : accuracy={correct_c.mean():.3f}  "
          f"blind={blind_c.mean():.0f}  warm={warm_c.mean():.0f}  economie={econ_c:+.0f} ({pct_c:+.0f}%)")
    print(f"  M4R (lecture forte)       : accuracy={correct_m.mean():.3f}  "
          f"blind={blind_m.mean():.0f}  warm={warm_m.mean():.0f}  economie={econ_m:+.0f} ({pct_m:+.0f}%)")
    print(f"\n  VALEUR AJOUTEE de M4R au-dela du hasard : {added_value:+.0f} iterations "
          f"(exces d'accuracy : {excess_acc:+.3f})")
    if added_value > 0.2 * econ_m:
        print("  -> M4R ajoute une valeur REELLE et substantielle au-dela du hasard -- "
              "le '+97%' n'est PAS qu'un artefact de structure, mais il faut le "
              "presenter avec sa part hasard/part M4R, pas comme un chiffre unique.")
    else:
        print("  -> La quasi-totalite de l'economie vient de la structure du piege, "
              "pas de M4R -- le '+97%' serait trompeur presente seul.")
    return dict(econ_hasard=econ_c, pct_hasard=pct_c, econ_m4r=econ_m, pct_m4r=pct_m,
                added_value=added_value, excess_acc=excess_acc)


def contre_epreuve_2_position_incertaine():
    print("\n=== CONTRE-EPREUVE 2 : le warm start survit-il a une position de piege moins previsible ? ===\n")
    p.T_READ, p.B_E = 300, 0.8
    blind_w, warm_w, correct_w = [], [], []
    n_inside = 0
    for seed in SEEDS:
        b = 1 if (seed % 2 == 0) else -1
        pb = make_problem_wide(seed, b, x_p_max=1.9)
        it_blind = p.solve(pb, x0=0.0)
        b_guess = p.m4r_read(seed, b)
        correct_w.append(int(b_guess == b))
        if abs(abs(pb["x_p"]) - p.X_WARM) < pb["w_flat"] + p.W_RAMP:
            n_inside += 1
        it_warm = p.solve(pb, x0=b_guess * p.X_WARM)
        blind_w.append(it_blind)
        warm_w.append(it_warm)
    blind_w, warm_w, correct_w = map(np.array, (blind_w, warm_w, correct_w))
    econ = float(blind_w.mean() - warm_w.mean())
    pct = 100 * econ / blind_w.mean()
    print(f"  x_p_mag elargi a [0.9, 1.9] (vs [0.9, 1.3] original) -- X_WARM=1.5 peut "
          f"atterrir DANS le piege")
    print(f"  accuracy lecture M4R : {correct_w.mean():.3f}")
    print(f"  cas ou le warm start atterrit DANS le piege : {n_inside}/{len(SEEDS)}")
    print(f"  blind={blind_w.mean():.0f}  warm={warm_w.mean():.0f}  economie={econ:+.0f} ({pct:+.0f}%)")
    if pct > 20:
        print("  -> Degrade GRACIEUSEMENT (moins spectaculaire que le cas favorable, "
              "mais toujours un gain net substantiel), pas d'effondrement.")
    else:
        print("  -> L'avantage S'EFFONDRE quand la position du piege est moins previsible -- "
              "le '+97%' dependait d'une hypothese fragile (connaitre la plage du piege a l'avance).")
    return dict(accuracy=float(correct_w.mean()), n_inside=n_inside, n_total=len(SEEDS),
                blind=float(blind_w.mean()), warm=float(warm_w.mean()), econ=econ, pct=pct)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    r1 = contre_epreuve_1_hasard()
    r2 = contre_epreuve_2_position_incertaine()

    with (FIG / "p11_warm_start_stress_poc.csv").open("w", encoding="utf-8") as f:
        f.write("test,metric,value\n")
        for k, v in r1.items():
            f.write(f"hasard_vs_m4r,{k},{v}\n")
        for k, v in r2.items():
            f.write(f"position_incertaine,{k},{v}\n")
    print(f"\n[csv] {FIG / 'p11_warm_start_stress_poc.csv'}")
    print(f"\nWall time: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
