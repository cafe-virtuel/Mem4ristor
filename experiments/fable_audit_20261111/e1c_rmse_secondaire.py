#!/usr/bin/env python3
"""
AUDIT FABLE 2026-11-11 -- E1c : la metrique SECONDAIRE de E1 (RMSE de <v_i> a la valeur vraie du
groupe, rapportee au bras sans couplage), relue sur le CSV de E1. Aucune simulation nouvelle.

Statut : EXPLORATOIRE. La RMSE etait declaree avant la mesure comme metrique secondaire, mais le
choix de c PAR la RMSE (et non par l'ARI) est decide apres avoir vu E1 et E1b. A confirmer sur
des graines neuves avant toute citation.
Pourquoi la regarder : l'ARI passe par un k-means, insensible a l'echelle -- un reseau qui ecrase
toutes les valeurs vers un quasi-consensus garde un bon ARI. La RMSE, elle, demande que chaque
noeud tienne la valeur de SON groupe.
Usage : uv run --frozen python experiments/fable_audit_20261111/e1c_rmse_secondaire.py
"""
import csv, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from e1_plural_consensus import boot_ci, C_GRID, TUNE, TEST

rows = list(csv.DictReader(open(HERE / "e1_plural_consensus.csv", encoding="utf-8")))
def val(g, arm, c, seeds):
    d = {int(r["seed"]): float(r["rmse"]) for r in rows if r["graph"] == g and r["arm"] == arm and float(r["c"]) == c}
    return np.array([d[s] for s in seeds])
ARMS = ["fixed", "node", "edge", "edge_noflip", "edge_noflip:frozen", "edge_noflip:shuffled", "oracle"]
for g in ("grille", "sbm"):
    print(f"\n{g} -- RMSE / RMSE sans couplage (graines de test) ; < 1 = le reseau aide")
    print(f"{'bras':>22} | " + " ".join(f"c={c:<5}" for c in C_GRID) + " | c* (choisi sur 0-9)")
    best = {}
    for a in ARMS:
        prof = [np.nanmean(val(g, a, c, TEST) / val(g, "none", 0.0, TEST)) for c in C_GRID]
        tune = [np.nanmean(val(g, a, c, TUNE) / val(g, "none", 0.0, TUNE)) for c in C_GRID]
        best[a] = C_GRID[int(np.nanargmin(tune))]
        print(f"{a:>22} | " + " ".join(f"{p:7.3f}" for p in prof) + f" | {best[a]}")
    for a, b in (("edge_noflip", "fixed"), ("edge_noflip:frozen", "edge_noflip:shuffled"), ("node", "fixed"), ("oracle", "fixed")):
        d = (val(g, a, best[a], TEST) - val(g, b, best[b], TEST)) / val(g, "none", 0.0, TEST)
        m, lo, hi = boot_ci(d)
        print(f"   {a} - {b} (chacun a son c*) : {m:+.3f} IC95 [{lo:+.3f}, {hi:+.3f}]   (negatif = {a} meilleur)")
