"""
V5 — Couplage non-local par similarite de doute

Idee : en plus du couplage physique (voisins sur la grille), chaque noeud
       "ressent" aussi les noeuds qui lui ressemblent par leur niveau de doute,
       peu importe ou ils se trouvent dans le reseau.

       C'est comme si des neurones tres incertains se "parlaient" entre eux
       a travers le reseau, meme s'ils ne sont pas connectes physiquement.

       sigma_u = rayon de similarite :
         petit (0.05) -> seuls les noeuds quasi-identiques en u se voient
         moyen (0.10) -> communautes moderement ouvertes
         grand (0.50) -> presque tout le monde voit tout le monde

Ce script compare :
  - V4 pur                    : pas de couplage virtuel
  - sigma_u faible (0.05)     : communautes tres serrees
  - sigma_u moyen  (0.10)     : valeur recommandee
  - sigma_u large  (0.50)     : quasi champ moyen global

Et teste deux forces de couplage : D_meta = 0.05 et D_meta = 0.10.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import numpy as np
from mem4ristor.core import Mem4Network
from mem4ristor.metrics import calculate_continuous_entropy

STEPS  = 3000
WARMUP = int(STEPS * 0.75)
SEEDS  = [42, 123, 777]
SIZE   = 10
I_STIM = 0.5

CONDITIONS = [
    {"label": "V4 pur                  ", "enabled": False, "D_meta": 0.00, "sigma_u": 0.10},
    {"label": "sigma=0.05  D=0.05 (serre) ", "enabled": True,  "D_meta": 0.05, "sigma_u": 0.05},
    {"label": "sigma=0.10  D=0.05 (moyen) ", "enabled": True,  "D_meta": 0.05, "sigma_u": 0.10},
    {"label": "sigma=0.50  D=0.05 (large) ", "enabled": True,  "D_meta": 0.05, "sigma_u": 0.50},
    {"label": "sigma=0.05  D=0.10 (serre) ", "enabled": True,  "D_meta": 0.10, "sigma_u": 0.05},
    {"label": "sigma=0.10  D=0.10 (moyen) ", "enabled": True,  "D_meta": 0.10, "sigma_u": 0.10},
    {"label": "sigma=0.50  D=0.10 (large) ", "enabled": True,  "D_meta": 0.10, "sigma_u": 0.50},
]

print("=" * 72)
print("  V5 — COUPLAGE NON-LOCAL PAR SIMILARITE DE DOUTE")
print(f"  Reseau {SIZE}x{SIZE} | {STEPS} steps | derniers 25% | {len(SEEDS)} seeds | I_stim={I_STIM}")
print("=" * 72)
print(f"\n  {'Condition':<42}  {'H_stable':>10}  {'u_spread':>10}")
print(f"  {'-'*66}")

results = []
for cond in CONDITIONS:
    H_runs, u_spread_runs = [], []

    for seed in SEEDS:
        net = Mem4Network(size=SIZE, heretic_ratio=0.15, seed=seed)
        net.model.cfg['nonlocal_coupling'] = {
            'enabled': cond['enabled'],
            'D_meta':  cond['D_meta'],
            'sigma_u': cond['sigma_u'],
        }
        H_win, spread_win = [], []
        for step in range(STEPS):
            net.step(I_stimulus=I_STIM)
            if step >= WARMUP:
                H_win.append(calculate_continuous_entropy(net.model.v))
                # u_spread = ecart-type de u -> mesure si des communautes de doute se forment
                # Si u_spread augmente, les noeuds divergent en doute -> communautes distinctes
                spread_win.append(net.model.u.std())
        H_runs.append(np.mean(H_win))
        u_spread_runs.append(np.mean(spread_win))

    r = {
        "label":    cond["label"],
        "H":        np.mean(H_runs),
        "H_std":    np.std(H_runs),
        "u_spread": np.mean(u_spread_runs),
        "enabled":  cond["enabled"],
        "D_meta":   cond["D_meta"],
        "sigma_u":  cond["sigma_u"],
    }
    results.append(r)
    print(f"  {r['label']}  {r['H']:>6.2f}±{r['H_std']:.2f}  {r['u_spread']:>10.5f}")

# ── Verdict ────────────────────────────────────────────────────────────────
print(f"\n{'=' * 72}")
print("  VERDICT")
print(f"{'=' * 72}\n")

H_v4      = results[0]["H"]
spread_v4 = results[0]["u_spread"]
best      = max(results, key=lambda x: x["H"])

print(f"  Referentiel V4 pur : H = {H_v4:.2f} bits  |  u_spread = {spread_v4:.5f}")
print(f"  Meilleur resultat  : {best['label'].strip()} -> H = {best['H']:.2f} bits")
print(f"  Delta H            : {best['H'] - H_v4:+.2f} bits\n")

for r in results[1:]:
    delta = r["H"] - H_v4
    delta_spread = r["u_spread"] - spread_v4
    signe = "+" if delta >= 0 else ""

    if delta > 0.10:
        effet_H = "GAIN"
    elif delta < -0.10:
        effet_H = "perte"
    else:
        effet_H = "neutre"

    if delta_spread > 0.005:
        effet_spread = "communautes emergent"
    elif delta_spread < -0.005:
        effet_spread = "u se concentre"
    else:
        effet_spread = "u inchange"

    print(f"  sigma={r['sigma_u']:.2f} D={r['D_meta']:.2f} | "
          f"H={r['H']:.2f} ({signe}{delta:.2f}, {effet_H}) | "
          f"u_spread: {effet_spread} ({delta_spread:+.5f})")

print()
# Question cle : est-ce que le couplage virtuel cree des communautes stables ?
max_spread = max(r["u_spread"] for r in results[1:])
if max_spread > spread_v4 + 0.005:
    print("  --> Des communautes de doute emergent (u_spread augmente).")
    print("      Les noeuds se regroupent spontanement par niveau d'incertitude.")
else:
    print("  --> Pas de communautes de doute distinctes detectees.")
    print("      Le couplage virtuel homogeneise u plutot que de le fragmenter.")

print(f"\n{'=' * 72}")
print("  Pour reproduire : python experiments/p2_nonlocal_coupling.py")
print(f"{'=' * 72}\n")
