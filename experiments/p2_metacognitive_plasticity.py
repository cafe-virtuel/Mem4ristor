"""
V5 — Plasticite Metacognitive : epsilon per-noeud module par le doute u

Idee : dans le cerveau, un neurone incertain ralentit son traitement
       (il "reflechit" plus). On reproduit ca ici : les noeuds avec
       beaucoup de doute (u pres de 1) ont un epsilon reduit et reagissent
       moins vite. Les noeuds tres certains (u pres de 0) ont un epsilon
       eleve et sont plus "impulsifs".

Ce script compare 4 conditions sur un reseau 10x10 :
  - V4 pur        : epsilon fixe, pas de plasticite metacognitive
  - META faible   : alpha_meta=0.25 (modulation douce)
  - META normal   : alpha_meta=0.50 (parametres defaut V5)
  - META fort     : alpha_meta=1.00 (modulation maximale)

Metriques :
  H_stable   = diversite des etats (plus c'est haut, plus le reseau pense
               de facon variee — c'est ce qu'on veut maximiser)
  u_mean     = doute moyen (proche de 1 = tous les noeuds sont tres incertains)
  epsilon_spread = ecart-type des epsilon_i (mesure a quel point les vitesses
                   divergent entre noeuds — zero en V4, positif en V5)
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import numpy as np
from mem4ristor.core import Mem4Network
from mem4ristor.metrics import calculate_continuous_entropy

# ── Parametres communs ─────────────────────────────────────────────────────
STEPS   = 3000
WARMUP  = int(STEPS * 0.75)   # on analyse seulement les 25% finaux (regime stable)
SEEDS   = [42, 123, 777]
SIZE    = 10                   # reseau 10x10 = 100 noeuds
I_STIM  = 0.5
ETA     = 0.08                 # epsilon de base

CONDITIONS = [
    {"label": "V4 pur       (alpha=0.00)", "alpha_meta": 0.00, "enabled": False},
    {"label": "META faible  (alpha=0.25)", "alpha_meta": 0.25, "enabled": True},
    {"label": "META normal  (alpha=0.50)", "alpha_meta": 0.50, "enabled": True},
    {"label": "META fort    (alpha=1.00)", "alpha_meta": 1.00, "enabled": True},
]

# ── Barre de titre ─────────────────────────────────────────────────────────
print("=" * 70)
print("  V5 — PLASTICITE METACOGNITIVE")
print(f"  Reseau {SIZE}x{SIZE} | {STEPS} steps | derniers 25% | {len(SEEDS)} seeds")
print("=" * 70)
print(f"\n  {'Condition':<38}  {'H_stable':>10}  {'u_mean':>8}  {'eps_spread':>12}")
print(f"  {'-'*68}")

results = []

for cond in CONDITIONS:
    H_runs, u_runs, eps_runs = [], [], []

    for seed in SEEDS:
        net = Mem4Network(size=SIZE, heretic_ratio=0.15, seed=seed)
        # Injection de la config V5 directement dans le modele
        net.model.cfg['metacognitive'] = {
            'enabled':    cond['enabled'],
            'alpha_meta': cond['alpha_meta'],
            'epsilon_min': 0.01,
        }

        H_window, u_window, eps_window = [], [], []

        for step in range(STEPS):
            net.step(I_stimulus=I_STIM)

            if step >= WARMUP:
                H_window.append(calculate_continuous_entropy(net.model.v))
                u_window.append(net.model.u.mean())

                # Calcul de l'ecart-type des epsilon_i (V5 uniquement)
                if cond['enabled']:
                    alpha = cond['alpha_meta']
                    eps_i = ETA * (1.0 + alpha * (0.5 - net.model.u))
                    eps_i = np.maximum(eps_i, 0.01)
                    eps_window.append(eps_i.std())
                else:
                    eps_window.append(0.0)  # V4 : epsilon uniforme partout

        H_runs.append(np.mean(H_window))
        u_runs.append(np.mean(u_window))
        eps_runs.append(np.mean(eps_window))

    H_mean  = np.mean(H_runs)
    H_std   = np.std(H_runs)
    u_mean  = np.mean(u_runs)
    eps_mean = np.mean(eps_runs)

    results.append({
        "label": cond["label"], "alpha": cond["alpha_meta"],
        "H": H_mean, "H_std": H_std, "u": u_mean, "eps_spread": eps_mean
    })

    print(f"  {cond['label']:<38}  {H_mean:>6.2f}±{H_std:.2f}  {u_mean:>7.3f}  {eps_mean:>12.5f}")

# ── Interpretation humaine ─────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("  LECTURE DES RESULTATS")
print(f"{'=' * 70}")

H_v4 = results[0]["H"]
print(f"\n  Referentiel : V4 pur -> H_stable = {H_v4:.2f} bits")
print()

for r in results[1:]:
    delta = r["H"] - H_v4
    signe = "+" if delta >= 0 else ""
    sens  = "amelioration" if delta > 0.05 else ("perte" if delta < -0.05 else "stable")
    print(f"  alpha={r['alpha']:.2f} | H = {r['H']:.2f} bits ({signe}{delta:.2f}) | {sens}")
    print(f"         u_moyen = {r['u']:.3f}  |  ecart epsilon = {r['eps_spread']:.5f}")
    print()

# Verdict final
best = max(results, key=lambda x: x["H"])
print(f"  Meilleure diversite : {best['label'].strip()} -> H = {best['H']:.2f} bits")

v4_H  = results[0]["H"]
best_H = best["H"]
if best_H > v4_H + 0.1:
    print("\n  --> La plasticite metacognitive AMELIORE la diversite du reseau.")
    print("      Les noeuds incertains qui ralentissent creent plus de variete.")
elif best_H < v4_H - 0.1:
    print("\n  --> La plasticite metacognitive REDUIT la diversite.")
    print("      Ralentir les noeuds douteux les isole trop du reste.")
else:
    print("\n  --> La plasticite metacognitive n'a pas d'effet majeur sur H_stable.")
    print("      L'organisation du reseau est robuste a ce type de modulation.")

print(f"\n{'=' * 70}")
print("  Pour reproduire : python experiments/p2_metacognitive_plasticity.py")
print(f"{'=' * 70}\n")
