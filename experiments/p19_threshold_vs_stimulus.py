#!/usr/bin/env python3
"""
P19 -- Le seuil de desynchronisation est-il une propriete du RESEAU ou du FORCAGE ?
        Et pourquoi synchronie et complexite LZ ne decrochent-elles pas au meme endroit ?
(29 juillet 2026, Claude Opus 5 -- les deux questions laissees ouvertes par P18.)

------------------------------------------------------------------------------------------
CE QUE P18 A ETABLI (meme jour)
------------------------------------------------------------------------------------------
En regime force (I_stim = 0.5), la synchronie s'effondre entre u = 0.30 (filtre +0.567,
sync 0.42) et u = 0.45 (filtre +0.166, sync 0.087) : la transition a lieu PENDANT QUE LE
COUPLAGE EST ENCORE ATTRACTIF. Ce n'est pas le signe qui desynchronise, c'est l'absence
d'attraction forte. En regime endogene (I_stim = 0), AUCUNE transition : les huit conditions
testees donnent une synchronie <= 0.008.

Deux lectures restaient possibles, et elles n'ont pas les memes consequences :
  (L1) le seuil appartient au FORCAGE. Le stimulus commun synchronise ; l'attraction ne fait
       que l'amplifier. Alors le seuil doit SE DEPLACER avec I_stim -- plus le forcage est
       fort, moins il faut d'attraction pour que le reseau se verrouille, donc le seuil
       descend vers le repulsif.
  (L2) le seuil appartient au RESEAU (topologie, degre, FHN). Alors il ne bouge pas, et
       I_stim ne fait qu'allumer ou eteindre le phenomene.

Et un second fait de P18 restait sans explication : la synchronie decroche vers filtre +0.17,
mais la complexite LZ ne decroche qu'au-dela de |filtre| ~ 0.55 (1.55 -> 1.11). DEUX EFFETS
DU MEME PARAMETRE, A DEUX SEUILS DIFFERENTS, jamais separes jusqu'ici.

------------------------------------------------------------------------------------------
LE DISPOSITIF
------------------------------------------------------------------------------------------
Harnais ablation_coordination (BA m=3, N=100, degree_linear, 3000 pas). Balayage croise :
  I_stim  in {0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0}      (7 valeurs)
  u fige  in {0.05 ... 0.95}                            (11 valeurs, grille resserree autour
                                                         de la transition vue par P18)
u est FIGE partout (epsilon_u = 0, tau_u = 1e12) : on cartographie le couplage seul, sans
l'adaptativite -- P17 a montre qu'elle ne change pas le resultat en regime force.

MI-TRANSITION, definie avant de regarder : pour une observable y(u) sur la grille, on prend
y_mid = (min(y) + max(y)) / 2 et on interpole LINEAIREMENT la valeur de u_filter ou y croise
y_mid. C'est une definition purement descriptive, appliquee identiquement aux deux
observables ; elle ne suppose aucune forme de courbe.

------------------------------------------------------------------------------------------
CRITERES, ECRITS AVANT LA MESURE (29/07/2026)
------------------------------------------------------------------------------------------
G1  (INSTRUMENT) max|u_fin - u_init| < 1e-12 partout. Sinon tous les verdicts sont suspendus.

F1  "LE SEUIL APPARTIENT AU FORCAGE" (lecture L1)
    Soit f_c(I) le filtre ou la synchronie croise 0.15 (frontiere de quadrant du papier).
    ACCEPTEE si, sur les six regimes forces (I_stim > 0), f_c est DECROISSANT en I_stim sur
    au moins 4 des 5 intervalles consecutifs, ET si f_c(0.1) - f_c(1.0) >= 0.10.
    REJETEE si l'etendue totale de f_c est < 0.10 : le seuil serait alors une propriete du
    RESEAU (lecture L2), et I_stim un simple interrupteur.
    Le sens de la prediction est ecrit AVANT : f_c doit DESCENDRE quand I_stim monte.

F2  "LES DEUX OBSERVABLES DECROCHENT A DES ENDROITS DIFFERENTS"
    ACCEPTEE si |f_mid(LZ) - f_mid(synchronie)| >= 0.15 sur au moins 4 des 6 regimes forces.
    Si elle passe, "u desynchronise" et "u structure les trajectoires" sont DEUX effets
    distincts, et le preprint les traite aujourd'hui comme un seul (tab:ablations rapporte
    les deux colonnes cote a cote comme une seule ablation).

F3  (CONTROLE, replication du fait endogene de P18 sur grille fine et graines neuves)
    ACCEPTEE si, a I_stim = 0, l'etendue de la synchronie sur toute la grille de u est
    < 0.05. C'est-a-dire : sans forcage, le couplage ne produit AUCUNE transition.

Replication : les cellules des deux I_stim extremes (0.1 et 1.0) sont rejouees sur les
graines 3041-3050, jamais utilisees (P17 a pris 3021-3030, P18 3031-3040).
Aucun seuil ne sera deplace apres coup. Aucun .tex n'est touche.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import ablation_coordination as ac  # noqa: E402
from mem4ristor.topology import Mem4Network  # noqa: E402
from mem4ristor.metrics import (  # noqa: E402
    calculate_pairwise_synchrony,
    calculate_temporal_lz_complexity,
)

SEEDS_CANON = list(range(10))
SEEDS_REPLI = list(range(3041, 3051))
I_GRID = [0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
U_GRID = [0.05, 0.15, 0.25, 0.325, 0.40, 0.45, 0.50, 0.55, 0.65, 0.80, 0.95]
I_REPLI = [0.1, 1.0]
SEUIL_SYNC = 0.15
CSV_OUT = ROOT / "figures" / "p19_threshold_vs_stimulus.csv"


def u_filter(u: float) -> float:
    return float(np.tanh(np.pi * (0.5 - u)) + 0.01)


def run_cell(seed: int, i_stim: float, u_val: float) -> dict:
    adj = ac.make_ba_adjacency(ac.N_NODES, ac.BA_M, seed)
    net = Mem4Network(adjacency_matrix=adj, heretic_ratio=ac.HERETIC_RATIO,
                      coupling_norm="degree_linear", seed=seed)
    net.model.cfg["doubt"]["epsilon_u"] = 0.0
    net.model.cfg["doubt"]["tau_u"] = 1e12
    net.model.u = np.full(net.model.N, float(u_val))
    u0 = net.model.u.copy()

    snaps = []
    for step in range(ac.STEPS):
        net.step(I_stimulus=i_stim)
        if step % ac.TRACE_STRIDE == 0:
            snaps.append(net.model.v.copy())
    v_hist = np.array(snaps)
    cut = int(len(snaps) * (1.0 - ac.TAIL_FRAC))
    return {
        "synchrony": float(calculate_pairwise_synchrony(v_hist[cut:])),
        "lz_full": float(calculate_temporal_lz_complexity(v_hist)),
        "u_drift": float(np.max(np.abs(net.model.u - u0))),
    }


def cross_at(y: list, level: float):
    """Filtre ou la courbe y(u) croise `level`, par interpolation lineaire.

    La grille U_GRID est croissante en u, donc DECROISSANTE en u_filter et (en regime force)
    decroissante en synchronie. On parcourt dans l'ordre de la grille et on prend le PREMIER
    croisement rencontre. Retourne None si la courbe ne croise jamais le niveau.
    """
    f = [u_filter(u) for u in U_GRID]
    for i in range(len(y) - 1):
        y0, y1 = y[i], y[i + 1]
        if (y0 - level) * (y1 - level) <= 0 and y0 != y1:
            t = (level - y0) / (y1 - y0)
            return f[i] + t * (f[i + 1] - f[i])
    return None


def mid_transition(y: list):
    lo, hi = min(y), max(y)
    if hi - lo < 1e-9:
        return None
    return cross_at(y, (lo + hi) / 2.0)


def main() -> int:
    t0 = time.time()
    print("=" * 96)
    print("P19 -- le seuil appartient-il au FORCAGE ou au RESEAU ? "
          "et les deux observables decrochent-elles ensemble ?")
    print("=" * 96)
    print("Grille u (11) -> filtre : " +
          "  ".join("%.2f:%+.2f" % (u, u_filter(u)) for u in U_GRID))
    print("Grille I_stim (7) : %s" % I_GRID)
    print("Total : %d cellules x %d graines, + replication sur I=%s (graines 3041-3050)\n"
          % (len(I_GRID) * len(U_GRID), len(SEEDS_CANON), I_REPLI))

    res: dict = {}
    for i_stim in I_GRID:
        print("[I_stim = %.2f]" % i_stim, end="", flush=True)
        for u in U_GRID:
            res[(i_stim, u, "canon")] = [run_cell(s, i_stim, u) for s in SEEDS_CANON]
            print(".", end="", flush=True)
        if i_stim in I_REPLI:
            for u in U_GRID:
                res[(i_stim, u, "repli")] = [run_cell(s, i_stim, u) for s in SEEDS_REPLI]
                print("r", end="", flush=True)
        print(" ok")

    drift = max(r["u_drift"] for lst in res.values() for r in lst)
    g1 = drift < 1e-12

    def curve(i_stim, col, grp="canon"):
        return [float(np.mean([r[col] for r in res[(i_stim, u, grp)]])) for u in U_GRID]

    # ------------------------------------------------------------------ tableau
    print("\n" + "=" * 96)
    print("SYNCHRONIE (lignes = I_stim, colonnes = u fige / filtre)")
    print("=" * 96)
    print("  I_stim " + "".join("%8.2f" % u for u in U_GRID))
    print("  filtre " + "".join("%+8.2f" % u_filter(u) for u in U_GRID))
    for i_stim in I_GRID:
        print("  %6.2f " % i_stim + "".join("%8.3f" % v for v in curve(i_stim, "synchrony")))
    print("\nCOMPLEXITE LZ")
    print("  I_stim " + "".join("%8.2f" % u for u in U_GRID))
    for i_stim in I_GRID:
        print("  %6.2f " % i_stim + "".join("%8.3f" % v for v in curve(i_stim, "lz_full")))

    # ------------------------------------------------------------------ F1
    print("\n" + "=" * 96)
    print("VERDICTS (criteres ecrits avant la mesure)")
    print("=" * 96)
    print("G1 -- instrument : max|u_fin - u_init| = %.3e -> %s"
          % (drift, "PASSE" if g1 else "ECHOUE"))

    forces = [i for i in I_GRID if i > 0]
    fc = {i: cross_at(curve(i, "synchrony"), SEUIL_SYNC) for i in forces}
    print("\nF1 -- filtre ou la synchronie croise %.2f :" % SEUIL_SYNC)
    for i in forces:
        print("   I_stim %.2f  ->  f_c = %s" % (i, "%+.3f" % fc[i] if fc[i] is not None
                                                else "jamais atteint"))
    vals = [fc[i] for i in forces if fc[i] is not None]
    if len(vals) == len(forces):
        dec = sum(1 for a, b in zip(vals, vals[1:]) if b <= a)
        etendue = max(vals) - min(vals)
        ecart = fc[0.1] - fc[1.0]
        f1 = dec >= 4 and ecart >= 0.10
        print("   decroissant sur %d/5 intervalles ; f_c(0.1) - f_c(1.0) = %+.3f ; "
              "etendue %.3f" % (dec, ecart, etendue))
        print("   F1 'le seuil appartient au FORCAGE' -> %s"
              % ("ACCEPTEE" if f1 else "REJETEE"))
        if not f1 and etendue < 0.10:
            print("   -> lecture L2 : le seuil est une propriete du RESEAU, I_stim n'est"
                  " qu'un interrupteur.")
    else:
        print("   F1 NON EVALUABLE : le seuil n'est pas atteint dans %d regime(s)."
              % sum(1 for i in forces if fc[i] is None))

    # ------------------------------------------------------------------ F2
    print("\nF2 -- mi-transition de chaque observable (en unites de filtre) :")
    n_sep = 0
    for i in forces:
        fs = mid_transition(curve(i, "synchrony"))
        fl = mid_transition(curve(i, "lz_full"))
        if fs is None or fl is None:
            print("   I_stim %.2f  sync %s   LZ %s   -> non evaluable"
                  % (i, fs, fl))
            continue
        d = abs(fl - fs)
        n_sep += d >= 0.15
        print("   I_stim %.2f  sync %+.3f   LZ %+.3f   ecart %.3f %s"
              % (i, fs, fl, d, "<- distincts" if d >= 0.15 else ""))
    print("   F2 'deux effets, deux seuils' -> %s (%d/6 regimes)"
          % ("ACCEPTEE" if n_sep >= 4 else "rejetee", n_sep))

    # ------------------------------------------------------------------ F3
    c0 = curve(0.0, "synchrony")
    etendue0 = max(c0) - min(c0)
    f3 = etendue0 < 0.05
    print("\nF3 -- controle endogene (I_stim = 0) : etendue de la synchronie sur toute la"
          " grille = %.4f -> %s" % (etendue0, "ACCEPTEE" if f3 else "rejetee"))

    # ------------------------------------------------------------------ replication
    print("\nREPLICATION sur graines 3041-3050 (jamais utilisees) :")
    for i in I_REPLI:
        fc_r = cross_at(curve(i, "synchrony", "repli"), SEUIL_SYNC)
        print("   I_stim %.2f  f_c canon %s  vs  replication %s"
              % (i, "%+.3f" % fc[i] if fc[i] is not None else "n/a",
                 "%+.3f" % fc_r if fc_r is not None else "jamais atteint"))

    print("\n  Portee : verdicts %s." % ("VALIDES" if g1 else "SUSPENDUS"))

    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with CSV_OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["i_stim", "u_frozen", "u_filter", "group", "seed",
                    "synchrony", "lz_full", "u_drift"])
        for (i_stim, u, grp), lst in res.items():
            seeds = SEEDS_CANON if grp == "canon" else SEEDS_REPLI
            for s, r in zip(seeds, lst):
                w.writerow(["%.2f" % i_stim, "%.3f" % u, "%.6f" % u_filter(u), grp, s,
                            "%.10f" % r["synchrony"], "%.10f" % r["lz_full"],
                            "%.3e" % r["u_drift"]])
    print("  CSV -> %s" % CSV_OUT)
    print("  Duree : %.1f s" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
