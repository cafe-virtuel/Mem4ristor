#!/usr/bin/env python3
"""
P20b -- FULL est-il vraiment DESYNCHRONISE, ou est-ce DEUX CAMPS EQUILIBRES ?
(29 juillet 2026, Claude Opus 5 -- trou dans mon propre design de P20, comble ici.)

------------------------------------------------------------------------------------------
POURQUOI CE SCRIPT EXISTE
------------------------------------------------------------------------------------------
P20 a mesure la structure en groupes de huit cellules a u FIGE -- et a oublie la seule qui
compte pour le preprint : FULL, u adaptatif. Le trou est le mien ; il est comble ici avec un
critere ecrit AVANT la mesure, comme les autres.

Ce que P20 a etabli et qui rend la question urgente : des que le couplage n'est plus
franchement attractif, le reseau se SCINDE EN DEUX GROUPES en anti-phase (r_intra ~ +0.6 a
+0.7, r_inter ~ -0.24 a -0.57), et cela dans TOUTES les cellules testees -- le creux comme la
bande. La synchronie moyenne de Pearson, elle, ne distingue pas :
    (a) "aucune structure, noeuds independants"          r_intra ~ 0, r_inter ~ 0
    (b) "deux camps opposes, equilibres"                 r_intra > 0, r_inter < 0
Les deux donnent une moyenne globale proche de zero.

Or le preprint ecrit, Section trajectoires : "r ~ 0: independent". Et son resultat central est
que FULL tient une synchronie ~ 0.03 la ou FROZEN_U monte a 0.75. Si FULL est en realite un
etat a deux camps equilibres, alors "independent" est une lecture ABUSIVE de la meme mesure --
sans qu'aucun chiffre publie soit faux.

C'est la meme ambiguite que le preprint avait lui-meme identifiee pour l'entropie instantanee
("conflates structured diversity with random disorder", Section 3) et resolue en introduisant
la complexite LZ. La question est de savoir si son remplacant en souffre aussi.

------------------------------------------------------------------------------------------
LE DISPOSITIF
------------------------------------------------------------------------------------------
Meme harnais et meme analyse que P20 (partition par le signe du premier vecteur propre de la
matrice de correlation, fenetre de queue). Quatre cellules :

  FULL a I_stim = 0.5   <- le regime exact de tab:ablations
  FULL a I_stim = 1.0   <- le regime de la bande de P19
  u = 0.05 a I_stim = 0.5  (l'ablation du preprint, controle "un seul camp")
  u = 0.95 a I_stim = 0.5  (repulsif fort fige, controle deja vu en P20 a I=1.0)

Graines 0-9, replication sur 3061-3070 (jamais utilisees : P17 a pris 3021-3030, P18
3031-3040, P19 3041-3050, P20 3051-3060).

------------------------------------------------------------------------------------------
CRITERES, ECRITS AVANT LA MESURE (29/07/2026)
------------------------------------------------------------------------------------------
G2  (CODE) decomposition exacte a 1e-9 pres, comme en P20. Sinon tout est suspendu.

D1  "FULL EST UN ETAT A DEUX CAMPS, PAS UN ETAT INDEPENDANT"
    ACCEPTEE si, pour FULL a I_stim = 0.5 : r_intra >= +0.30 ET r_inter <= -0.10,
    sur >= 8/10 graines canoniques ET >= 8/10 en replication.

D2  "FULL EST VRAIMENT DECORRELE"  (l'hypothese inverse, ecrite en meme temps)
    ACCEPTEE si r_intra < +0.15 sur >= 8/10 graines, memes conditions.

    D1 et D2 sont mutuellement exclusives. Si AUCUNE ne passe (r_intra entre 0.15 et 0.30),
    le resultat est NON TRANCHE et sera rapporte tel quel -- aucun des deux seuils ne bougera.

D3  "SI FULL EST A DEUX CAMPS, C'EST L'EQUILIBRE DES TAILLES QUI ANNULE LA MOYENNE"
    Evaluee seulement si D1 passe. ACCEPTEE si le desequilibre min(n1,n2)/N >= 0.40
    (camps de tailles comparables) sur >= 8/10 graines.
    -> alors la synchronie ~ 0 de FULL ne signifie PAS l'independance : elle signifie deux
       camps de taille comparable dont les correlations positives et negatives se compensent.

Aucun .tex n'est touche. Ce script ne modifie aucun chiffre publie.
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
import p20_resync_band_cluster_structure as p20  # noqa: E402  (analyse reutilisee TELLE QUELLE)
from mem4ristor.topology import Mem4Network  # noqa: E402

SEEDS_CANON = list(range(10))
SEEDS_REPLI = list(range(3061, 3071))
CELLS = [
    (0.5, None, "FULL (regime de Table 1)"),
    (1.0, None, "FULL (regime de la bande)"),
    (0.5, 0.05, "u=0.05 (ablation du preprint)"),
    (0.5, 0.95, "u=0.95 (repulsif fort fige)"),
]
CSV_OUT = ROOT / "figures" / "p20b_full_camps_or_independent.csv"


def run_cell(seed: int, i_stim: float, u_val) -> dict:
    adj = ac.make_ba_adjacency(ac.N_NODES, ac.BA_M, seed)
    net = Mem4Network(adjacency_matrix=adj, heretic_ratio=ac.HERETIC_RATIO,
                      coupling_norm="degree_linear", seed=seed)
    if u_val is not None:
        net.model.cfg["doubt"]["epsilon_u"] = 0.0
        net.model.cfg["doubt"]["tau_u"] = 1e12
        net.model.u = np.full(net.model.N, float(u_val))

    snaps = []
    for step in range(ac.STEPS):
        net.step(I_stimulus=i_stim)
        if step % ac.TRACE_STRIDE == 0:
            snaps.append(net.model.v.copy())
    v_hist = np.array(snaps)
    cut = int(len(snaps) * (1.0 - ac.TAIL_FRAC))
    out = p20.analyse(v_hist[cut:])
    out["u_final_mean"] = float(net.model.u.mean())
    return out


def main() -> int:
    t0 = time.time()
    print("=" * 100)
    print("P20b -- FULL : deux camps equilibres, ou noeuds independants ?")
    print("=" * 100)

    res = {}
    for i_stim, u_val, label in CELLS:
        print("[%-30s]" % label, end="", flush=True)
        res[(i_stim, u_val, "canon")] = [run_cell(s, i_stim, u_val) for s in SEEDS_CANON]
        res[(i_stim, u_val, "repli")] = [run_cell(s, i_stim, u_val) for s in SEEDS_REPLI]
        print(" ok")

    err = max(abs(r["r_global"] - r["r_recomp"]) for lst in res.values() for r in lst)
    g2 = err < 1e-9
    print("\nG2 -- decomposition exacte : ecart max %.3e -> %s"
          % (err, "PASSE" if g2 else "ECHOUE"))

    def col(cell, grp, key):
        return np.array([r[key] for r in res[(cell[0], cell[1], grp)]], dtype=float)

    print("\n" + "=" * 100)
    print("STRUCTURE MESUREE (10 graines canoniques)")
    print("=" * 100)
    print("  %-32s %9s %9s %9s %8s %8s"
          % ("cellule", "r_global", "r_intra", "r_inter", "n_min", "u final"))
    for i_stim, u_val, label in CELLS:
        c = (i_stim, u_val)
        print("  %-32s %+9.3f %+9.3f %+9.3f %8.1f %8.3f"
              % ("%s  I=%.1f" % (label, i_stim),
                 np.nanmean(col(c, "canon", "r_global")),
                 np.nanmean(col(c, "canon", "r_intra")),
                 np.nanmean(col(c, "canon", "r_inter")),
                 np.nanmean(col(c, "canon", "n_min")),
                 np.nanmean(col(c, "canon", "u_final_mean"))))

    print("\n" + "=" * 100)
    print("VERDICTS (criteres ecrits avant la mesure)")
    print("=" * 100)
    cf = (0.5, None)
    d1_n = {g: int(((col(cf, g, "r_intra") >= 0.30)
                    & (col(cf, g, "r_inter") <= -0.10)).sum()) for g in ("canon", "repli")}
    d2_n = {g: int((col(cf, g, "r_intra") < 0.15).sum()) for g in ("canon", "repli")}
    d1 = d1_n["canon"] >= 8 and d1_n["repli"] >= 8
    d2 = d2_n["canon"] >= 8 and d2_n["repli"] >= 8
    print("  D1 'FULL est a DEUX CAMPS'   : %d/10 puis %d/10  -> %s"
          % (d1_n["canon"], d1_n["repli"], "ACCEPTEE" if d1 else "rejetee"))
    print("  D2 'FULL est DECORRELE'      : %d/10 puis %d/10  -> %s"
          % (d2_n["canon"], d2_n["repli"], "ACCEPTEE" if d2 else "rejetee"))
    if not d1 and not d2:
        print("  -> NON TRANCHE (r_intra entre 0.15 et 0.30). Seuils non deplaces.")

    if d1:
        d3_n = {g: int((col(cf, g, "desequilibre") >= 0.40).sum())
                for g in ("canon", "repli")}
        d3 = d3_n["canon"] >= 8 and d3_n["repli"] >= 8
        print("  D3 'camps de tailles comparables' : %d/10 puis %d/10  -> %s"
              % (d3_n["canon"], d3_n["repli"], "ACCEPTEE" if d3 else "rejetee"))
        if d3:
            print("  -> la synchronie ~ 0 de FULL ne signifie PAS l'independance des noeuds :")
            print("     elle signifie deux camps de taille comparable qui se compensent.")

    print("\n  Portee : verdicts %s." % ("VALIDES" if g2 else "SUSPENDUS"))

    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with CSV_OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["i_stim", "u_frozen", "label", "group", "seed", "r_global", "r_intra",
                    "r_inter", "n_min", "desequilibre", "u_final_mean"])
        for i_stim, u_val, label in CELLS:
            for grp, seeds in (("canon", SEEDS_CANON), ("repli", SEEDS_REPLI)):
                for s, r in zip(seeds, res[(i_stim, u_val, grp)]):
                    w.writerow(["%.2f" % i_stim,
                                "" if u_val is None else "%.3f" % u_val, label, grp, s,
                                "%.8f" % r["r_global"], "%.8f" % r["r_intra"],
                                "%.8f" % r["r_inter"], r["n_min"],
                                "%.4f" % r["desequilibre"], "%.6f" % r["u_final_mean"]])
    print("  CSV -> %s" % CSV_OUT)
    print("  Duree : %.1f s" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
