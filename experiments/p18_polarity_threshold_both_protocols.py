#!/usr/bin/env python3
"""
P18 -- Quelle est la version MINIMALE du mecanisme anti-synchronisation ?
(29 juillet 2026, Claude Opus 5. Demande de Julien : "le preprint on s'en fout, on fait de la
recherche" -> la question n'est pas de defendre un tableau, c'est de savoir de quoi le
mecanisme a besoin.)

------------------------------------------------------------------------------------------
CE QUE P17 A ETABLI (meme jour, protocole FORCE uniquement)
------------------------------------------------------------------------------------------
Un u FIGE a 0.997 -- constante, aucune adaptativite -- desynchronise autant que le doute
adaptatif : synchronie +0.0030 contre +0.0023, 10/10 puis 10/10 sur graines neuves.
L'heterogeneite entre noeuds n'y change rien (ecart 0.004).
=> en regime force, c'est la POLARITE du couplage qui desynchronise, pas la DYNAMIQUE de u.

Deux choses restaient ouvertes, et ce sont elles qu'on mesure ici :
  (a) le regime ENDOGENE (I_stim = 0) n'a pas ete teste. C'est le regime ou les heretiques
      sont inactifs (I_eff = -0 = 0) et ou le seul brise-symetrie est le Cold Start ;
  (b) on ne sait pas COMBIEN de repulsion il faut. u = 0.997 donne u_filter = -0.906, la
      valeur la plus extreme accessible. Un repulsif FAIBLE suffirait-il ?

------------------------------------------------------------------------------------------
LE DISPOSITIF
------------------------------------------------------------------------------------------
Harnais ablation_coordination (BA m=3, N=100, degree_linear, 3000 pas), DEUX protocoles :
ENDOGENE (I_stim = 0) et FORCE (I_stim = 0.5).

Bras : FULL (u adaptatif) + sept valeurs de u FIGE, choisies pour encadrer le changement de
signe du filtre u_filter = tanh(pi*(0.5 - u)) + 0.01 :

    u = 0.05  -> u_filter = +0.898   attractif fort   (l'ablation du preprint)
    u = 0.30  -> u_filter = +0.596   attractif moyen
    u = 0.45  -> u_filter = +0.165   attractif FAIBLE
    u = 0.55  -> u_filter = -0.145   repulsif FAIBLE     <- le miroir de 0.45
    u = 0.70  -> u_filter = -0.567   repulsif moyen
    u = 0.85  -> u_filter = -0.826   repulsif fort
    u = u_bar -> mesure dans CHAQUE regime, jamais emprunte a l'autre (regle I4)

Graines : 0-9 (canoniques) + 3031-3040 pour la replication. 3021-3030 ont ete consommees
par P17 ce matin ; on ne s'en ressert pas.

------------------------------------------------------------------------------------------
CRITERES, ECRITS AVANT LA MESURE (29/07/2026)
------------------------------------------------------------------------------------------
G1  (INSTRUMENT) max|u_fin - u_init| < 1e-12 dans tous les bras figes. Si ce controle tombe,
    tous les verdicts sont suspendus : l'ablation ne ferait pas ce que son nom dit.

E1  "LE NIVEAU SUFFIT AUSSI EN ENDOGENE"
    ACCEPTEE si, en ENDOGENE, le bras u = u_bar atteint A LA FOIS synchronie <= 0.15 ET
    C_LZ < 1.6 (les deux frontieres de quadrant du papier, fig:phase_space), sur >= 8/10
    graines canoniques ET >= 8/10 graines de replication.
    Si elle passe, un couplage repulsif CONSTANT occupe le demi-plan structure dans les DEUX
    protocoles -- ce que le papier attribue au seul modele complet.

E2  "LA FRONTIERE EST LE CHANGEMENT DE SIGNE, PAS L'INTENSITE"  (prediction, posee avant)
    On classe les sept valeurs de u par ordre croissant et on cherche entre quelles deux
    valeurs consecutives la synchronie chute le plus.
    ACCEPTEE si cette plus grande marche tombe entre u = 0.45 et u = 0.55 -- l'intervalle qui
    contient le zero du filtre -- DANS LES DEUX PROTOCOLES.
    REJETEE sinon : si la marche est ailleurs, c'est l'INTENSITE qui gouverne, ce qui
    contredirait P16 ("le signe tranche, la force non", 28/07) et devrait etre dit tel quel.

E3  "UN REPULSIF FAIBLE SUFFIT"
    ACCEPTEE si u = 0.55 (u_filter = -0.145, six fois plus faible que le -0.906 de FULL)
    atteint deja synchronie <= 0.15 sur >= 8/10 graines, dans les deux protocoles.
    Son miroir attractif u = 0.45 (u_filter = +0.165, meme intensite, signe oppose) sert de
    controle : si les deux se ressemblent, le signe ne tranche pas et E2 tombe avec E3.

Aucun seuil ne sera deplace apres coup. Aucun fichier .tex n'est touche par ce script.

REFERENCE PERIMEE, RAPPELEE POUR MEMOIRE ET NON UTILISEE COMME GATE :
figures/scratch/ablation_coordination.csv (26/04/2026) precede le fix de bruit du 01/05
(AUDIT-024). Ses valeurs endogenes sont affichees en tete d'execution A TITRE INDICATIF,
avec la mention "perime" -- elles ne conditionnent rien. Voir P17 pour le detail.
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
SEEDS_REPLI = list(range(3031, 3041))   # jamais utilisees (P17 a consomme 3021-3030)
PROTOCOLS = [("ENDOGENE", 0.0), ("FORCE", 0.5)]
U_GRID = [0.05, 0.30, 0.45, 0.55, 0.70, 0.85]   # u_bar ajoute par regime

SEUIL_SYNC = 0.15     # frontiere de quadrant du papier
SEUIL_LZ = 1.6        # frontiere de quadrant du papier
GATE_FRAC = 8         # sur 10
CSV_OUT = ROOT / "figures" / "p18_polarity_threshold.csv"
CSV_REF_PERIME = ROOT / "figures" / "scratch" / "ablation_coordination.csv"


def u_filter(u: float) -> float:
    return float(np.tanh(np.pi * (0.5 - u)) + 0.01)


def run_arm(seed: int, i_stim: float, u_frozen=None) -> dict:
    adj = ac.make_ba_adjacency(ac.N_NODES, ac.BA_M, seed)
    net = Mem4Network(adjacency_matrix=adj, heretic_ratio=ac.HERETIC_RATIO,
                      coupling_norm="degree_linear", seed=seed)
    if u_frozen is not None:
        net.model.cfg["doubt"]["epsilon_u"] = 0.0
        net.model.cfg["doubt"]["tau_u"] = 1e12
        net.model.u = np.full(net.model.N, float(u_frozen))
    u0 = net.model.u.copy()

    snaps = []
    for step in range(ac.STEPS):
        net.step(I_stimulus=i_stim)
        if step % ac.TRACE_STRIDE == 0:
            snaps.append(net.model.v.copy())

    v_hist = np.array(snaps)
    cut = int(len(snaps) * (1.0 - ac.TAIL_FRAC))
    v_tail = v_hist[cut:]
    return {
        "synchrony": float(calculate_pairwise_synchrony(v_tail)),
        "lz_full": float(calculate_temporal_lz_complexity(v_hist)),
        "u_final_mean": float(net.model.u.mean()),
        "u_drift": float(np.max(np.abs(net.model.u - u0))),
    }


def show_stale_reference() -> None:
    if not CSV_REF_PERIME.exists():
        return
    acc: dict = {}
    with CSV_REF_PERIME.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["regime"] == "ENDOGENOUS":
                acc.setdefault(row["ablation"], []).append(
                    (float(row["synchrony"]), float(row["lz_full"])))
    print("\n[reference PERIMEE -- CSV du 26/04, anterieur au fix de bruit du 01/05 ; "
          "affichee pour memoire, ne conditionne RIEN]")
    for a, v in acc.items():
        arr = np.array(v)
        print("   ENDOGENE %-11s sync %+.3f   LZ %.3f" % (a, arr[:, 0].mean(), arr[:, 1].mean()))


def main() -> int:
    t0 = time.time()
    print("=" * 92)
    print("P18 -- de quoi l'anti-synchronisation a-t-elle BESOIN ? (les deux protocoles)")
    print("=" * 92)
    print("Filtre u_filter = tanh(pi*(0.5-u)) + 0.01 :")
    for u in U_GRID:
        print("   u = %.2f  ->  u_filter = %+.3f  (%s)"
              % (u, u_filter(u), "attractif" if u_filter(u) > 0 else "repulsif"))
    show_stale_reference()

    all_seeds = SEEDS_CANON + SEEDS_REPLI
    res: dict = {}          # (proto, arm) -> {seed: r}
    u_bars: dict = {}

    for proto, i_stim in PROTOCOLS:
        print("\n[%s] FULL ..." % proto)
        res[(proto, "FULL")] = {s: run_arm(s, i_stim, None) for s in all_seeds}
        u_bar = float(np.mean([res[(proto, "FULL")][s]["u_final_mean"] for s in SEEDS_CANON]))
        u_bars[proto] = u_bar
        print("   u_bar MESURE dans ce regime = %.4f  (u_filter = %+.4f)"
              % (u_bar, u_filter(u_bar)))
        for u in U_GRID + [u_bar]:
            name = "U_BAR" if u == u_bar else "U_%.2f" % u
            print("[%s] %s ..." % (proto, name))
            res[(proto, name)] = {s: run_arm(s, i_stim, u) for s in all_seeds}

    # ---------------------------------------------------------------- G1
    print("\n" + "=" * 92)
    print("G1 -- CONTROLE D'INSTRUMENT (le gel gele-t-il ?)")
    print("=" * 92)
    drift = max(r["u_drift"] for (p, a), d in res.items() if a != "FULL" for r in d.values())
    g1 = drift < 1e-12
    print("  max|u_fin - u_init| sur tous les bras figes = %.3e -> %s"
          % (drift, "PASSE" if g1 else "ECHOUE"))

    # ---------------------------------------------------------------- tableaux
    def m(proto, arm, col, seeds):
        return float(np.mean([res[(proto, arm)][s][col] for s in seeds]))

    def n_ok(proto, arm, seeds):
        return sum(1 for s in seeds
                   if res[(proto, arm)][s]["synchrony"] <= SEUIL_SYNC
                   and res[(proto, arm)][s]["lz_full"] < SEUIL_LZ)

    for proto, _ in PROTOCOLS:
        print("\n" + "=" * 92)
        print("RESULTATS -- %s" % proto)
        print("=" * 92)
        print("  %-9s %9s %10s %10s %9s %9s   %s"
              % ("bras", "u_filter", "sync(0-9)", "sync(repl)", "LZ", "LZ(repl)",
                 "quadrant structure ?"))
        arms = ["FULL"] + ["U_%.2f" % u for u in U_GRID] + ["U_BAR"]
        for arm in arms:
            uf = ("adaptatif" if arm == "FULL"
                  else "%+.3f" % u_filter(u_bars[proto] if arm == "U_BAR"
                                          else float(arm[2:])))
            print("  %-9s %9s   %+.4f    %+.4f   %.4f    %.4f     %d/10 puis %d/10"
                  % (arm, uf,
                     m(proto, arm, "synchrony", SEEDS_CANON),
                     m(proto, arm, "synchrony", SEEDS_REPLI),
                     m(proto, arm, "lz_full", SEEDS_CANON),
                     m(proto, arm, "lz_full", SEEDS_REPLI),
                     n_ok(proto, arm, SEEDS_CANON), n_ok(proto, arm, SEEDS_REPLI)))

    # ---------------------------------------------------------------- E1
    print("\n" + "=" * 92)
    print("VERDICTS (criteres ecrits avant la mesure)")
    print("=" * 92)
    e1 = (n_ok("ENDOGENE", "U_BAR", SEEDS_CANON) >= GATE_FRAC
          and n_ok("ENDOGENE", "U_BAR", SEEDS_REPLI) >= GATE_FRAC)
    print("  E1 'le niveau suffit AUSSI en endogene' (sync<=%.2f ET LZ<%.1f) : %d/10 puis %d/10"
          "  -> %s" % (SEUIL_SYNC, SEUIL_LZ, n_ok("ENDOGENE", "U_BAR", SEEDS_CANON),
                       n_ok("ENDOGENE", "U_BAR", SEEDS_REPLI), "ACCEPTEE" if e1 else "rejetee"))

    # ---------------------------------------------------------------- E2
    e2_ok = True
    for proto, _ in PROTOCOLS:
        vals = [(u, m(proto, "U_%.2f" % u, "synchrony", SEEDS_CANON)) for u in U_GRID]
        marches = [(vals[i][1] - vals[i + 1][1], vals[i][0], vals[i + 1][0])
                   for i in range(len(vals) - 1)]
        chute, u_lo, u_hi = max(marches)
        dans = (u_lo, u_hi) == (0.45, 0.55)
        e2_ok &= dans
        print("  E2 [%s] plus grande chute de synchronie : %+.4f entre u=%.2f et u=%.2f  -> %s"
              % (proto, chute, u_lo, u_hi, "au changement de SIGNE" if dans else "AILLEURS"))
    print("  E2 'la frontiere est le changement de signe' -> %s"
          % ("ACCEPTEE" if e2_ok else "REJETEE"))

    # ---------------------------------------------------------------- E3
    e3_ok = True
    for proto, _ in PROTOCOLS:
        n55 = sum(1 for s in SEEDS_CANON
                  if res[(proto, "U_0.55")][s]["synchrony"] <= SEUIL_SYNC)
        n45 = sum(1 for s in SEEDS_CANON
                  if res[(proto, "U_0.45")][s]["synchrony"] <= SEUIL_SYNC)
        e3_ok &= n55 >= GATE_FRAC
        print("  E3 [%s] repulsif FAIBLE (u=0.55, filtre %+.3f) : %d/10 sous %.2f"
              "   |  controle miroir attractif (u=0.45, filtre %+.3f) : %d/10"
              % (proto, u_filter(0.55), n55, SEUIL_SYNC, u_filter(0.45), n45))
    print("  E3 'un repulsif faible suffit' -> %s" % ("ACCEPTEE" if e3_ok else "rejetee"))

    print("\n  Portee : verdicts %s." % ("VALIDES" if g1 else "SUSPENDUS (G1 est tombe)"))
    print("  Rappel : u=u_bar est une valeur MESUREE SUR FULL. Ce script ne dit pas que la")
    print("  dynamique de u est inutile -- il dit que ce qui desynchronise est le NIVEAU")
    print("  qu'elle atteint. Personne ne peut poser ce niveau a la main dans un dispositif.")

    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with CSV_OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["protocol", "arm", "u_frozen", "u_filter", "group", "seed",
                    "synchrony", "lz_full", "u_drift"])
        for (proto, arm), d in res.items():
            if arm == "FULL":
                uval, uf = "", ""
            else:
                uval = u_bars[proto] if arm == "U_BAR" else float(arm[2:])
                uf = "%.6f" % u_filter(uval)
                uval = "%.6f" % uval
            for s, r in d.items():
                w.writerow([proto, arm, uval, uf,
                            "canon" if s in SEEDS_CANON else "replication", s,
                            "%.10f" % r["synchrony"], "%.10f" % r["lz_full"],
                            "%.3e" % r["u_drift"]])
    print("\n  CSV -> %s" % CSV_OUT)
    print("  Duree : %.1f s" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
