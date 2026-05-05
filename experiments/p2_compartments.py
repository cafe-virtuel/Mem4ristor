"""
V5 — Compartimentalisation Dynamique (Sous-Personnalites)

Idee : le reseau se divise spontanement en K groupes selon le niveau de doute u.
       Les noeuds "certains" forment un groupe, les noeuds "douteux" un autre.
       A l'interieur de chaque groupe, un couplage attractif supplémentaire
       pousse les noeuds a se synchroniser entre eux (-> sous-personnalite coherente).
       En mode 'full', les groupes se repoussent mutuellement en plus.

       K=2 : deux sous-personnalites (certain vs douteux)
       K=3 : trois sous-personnalites (certain / mitige / douteux)

Ce script compare :
  - V4 pur                       : pas de compartimentation
  - K=2  attraction  gamma=0.05  : deux groupes, couplage intra leger
  - K=2  attraction  gamma=0.10  : deux groupes, couplage intra moyen
  - K=2  full        gamma=0.05  : deux groupes, intra+inter
  - K=2  full        gamma=0.10  : deux groupes, intra+inter force
  - K=3  attraction  gamma=0.10  : trois groupes, couplage intra moyen
  - K=3  full        gamma=0.10  : trois groupes, intra+inter

Et mesure :
  - H_stable   : diversite globale (entropie sur v)
  - v_between  : ecart entre la moyenne de v de chaque groupe -> specialisation reussie ?
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
    {"label": "V4 pur                        ", "enabled": False, "K": 2,  "gamma": 0.00, "mode": "attraction"},
    {"label": "K=2  attra  gamma=0.05        ", "enabled": True,  "K": 2,  "gamma": 0.05, "mode": "attraction"},
    {"label": "K=2  attra  gamma=0.10        ", "enabled": True,  "K": 2,  "gamma": 0.10, "mode": "attraction"},
    {"label": "K=2  full   gamma=0.05        ", "enabled": True,  "K": 2,  "gamma": 0.05, "mode": "full"},
    {"label": "K=2  full   gamma=0.10        ", "enabled": True,  "K": 2,  "gamma": 0.10, "mode": "full"},
    {"label": "K=3  attra  gamma=0.10        ", "enabled": True,  "K": 3,  "gamma": 0.10, "mode": "attraction"},
    {"label": "K=3  full   gamma=0.10        ", "enabled": True,  "K": 3,  "gamma": 0.10, "mode": "full"},
]

print("=" * 76)
print("  V5 — COMPARTIMENTALISATION DYNAMIQUE (SOUS-PERSONNALITES)")
print(f"  Reseau {SIZE}x{SIZE} | {STEPS} steps | derniers 25% | {len(SEEDS)} seeds | I_stim={I_STIM}")
print("=" * 76)
print(f"\n  {'Condition':<46}  {'H_stable':>10}  {'v_between':>10}")
print(f"  {'-'*70}")

results = []
for cond in CONDITIONS:
    H_runs, vb_runs = [], []

    for seed in SEEDS:
        net = Mem4Network(size=SIZE, heretic_ratio=0.15, seed=seed)
        net.model.cfg['compartments'] = {
            'enabled': cond['enabled'],
            'K':       cond['K'],
            'gamma':   cond['gamma'],
            'mode':    cond['mode'],
        }
        H_win, vb_win = [], []
        for step in range(STEPS):
            net.step(I_stimulus=I_STIM)
            if step >= WARMUP:
                H_win.append(calculate_continuous_entropy(net.model.v))

                # v_between : ecart-type des moyennes de v par groupe
                # Si les groupes ont des etats tres differents -> v_between eleve
                K = cond['K']
                u_ranks = np.argsort(np.argsort(net.model.u))
                labels  = np.minimum((u_ranks * K) // net.model.N, K - 1)
                group_means = [net.model.v[labels == k].mean()
                               for k in range(K) if (labels == k).sum() > 0]
                vb_win.append(np.std(group_means) if len(group_means) > 1 else 0.0)

        H_runs.append(np.mean(H_win))
        vb_runs.append(np.mean(vb_win))

    r = {
        "label":     cond["label"],
        "H":         np.mean(H_runs),
        "H_std":     np.std(H_runs),
        "v_between": np.mean(vb_runs),
        "enabled":   cond["enabled"],
        "K":         cond["K"],
        "gamma":     cond["gamma"],
        "mode":      cond["mode"],
    }
    results.append(r)
    print(f"  {r['label']}  {r['H']:>6.2f}±{r['H_std']:.2f}  {r['v_between']:>10.4f}")

# ── Verdict ────────────────────────────────────────────────────────────────
print(f"\n{'=' * 76}")
print("  VERDICT")
print(f"{'=' * 76}\n")

H_v4    = results[0]["H"]
vb_v4   = results[0]["v_between"]
best    = max(results, key=lambda x: x["H"])
best_vb = max(results, key=lambda x: x["v_between"])

print(f"  Referentiel V4 pur : H = {H_v4:.2f} bits  |  v_between = {vb_v4:.4f}")
print(f"  Meilleur H         : {best['label'].strip()} -> H = {best['H']:.2f} bits  (delta {best['H']-H_v4:+.2f})")
print(f"  Meilleure separat. : {best_vb['label'].strip()} -> v_between = {best_vb['v_between']:.4f}  (delta {best_vb['v_between']-vb_v4:+.4f})\n")

for r in results[1:]:
    delta_H  = r["H"] - H_v4
    delta_vb = r["v_between"] - vb_v4

    if delta_H > 0.10:
        effet_H = "GAIN"
    elif delta_H < -0.10:
        effet_H = "perte"
    else:
        effet_H = "neutre"

    if delta_vb > 0.05:
        effet_vb = "groupes bien separes"
    elif delta_vb > 0.01:
        effet_vb = "separation legere"
    else:
        effet_vb = "groupes homogenes"

    print(f"  K={r['K']} {r['mode']:<10} gamma={r['gamma']:.2f} | "
          f"H={r['H']:.2f} ({delta_H:+.2f}, {effet_H}) | "
          f"v_between: {effet_vb} ({delta_vb:+.4f})")

print()
# Question cle : est-ce que la compartimentation cree des sous-personnalites ET augmente H ?
best_combo = max(results[1:], key=lambda x: x["H"] + 2 * (x["v_between"] - vb_v4))
delta_best = best_combo["H"] - H_v4
if delta_best > 0.10:
    print(f"  --> SUCCES : la compartimentation K={best_combo['K']} ({best_combo['mode']}) "
          f"cree des sous-personnalites ET gagne {delta_best:+.2f} bits.")
    print("      Les groupes specialises amplifient la diversite globale.")
elif delta_best > -0.05:
    print(f"  --> NEUTRE : la compartimentation cree des groupes mais H reste stable.")
    print("      La specialisation ne nuit pas — piste a explorer avec d'autres params.")
else:
    print(f"  --> ECHEC : la compartimentation synchronise trop les noeuds -> perte de H.")
    print("      Les sous-personnalites s'effondrent en etats uniformes.")

print(f"\n{'=' * 76}")
print("  Pour reproduire : python experiments/p2_compartments.py")
print(f"{'=' * 76}\n")
