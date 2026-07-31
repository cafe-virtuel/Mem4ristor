"""
V5 — Combinaison : Metacognition + Compartimentalisation

Teste si les deux features V5 validees separement donnent des gains additifs :
  - Metacog seul    : alpha_meta=-0.5 -> +0.79 bits (mesure)
  - Compart. seul   : K=3 full gamma=0.10 -> +0.15 bits (mesure)
  - Metacog+Compart : gains additifs ou antagonistes ?

Hypothese optimiste    : +0.94 bits (additivite parfaite)
Hypothese conservatrice: interaction non lineaire (interference ou synergie)
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import numpy as np
import csv
from mem4ristor.core import Mem4Network
from mem4ristor.metrics import calculate_continuous_entropy

STEPS  = 4000
WARMUP = int(STEPS * 0.75)
SEEDS  = [42, 123, 777, 1234, 5678, 9999, 314, 2718, 8765, 4321]
SIZE   = 10
I_STIM = 0.5

CONDITIONS = [
    {
        "label": "V4 pur",
        "meta":  {"enabled": False},
        "comp":  {"enabled": False, "K": 3, "gamma": 0.10, "mode": "full"},
    },
    {
        "label": "Metacog seul (alpha=-0.5)",
        "meta":  {"enabled": True, "alpha_meta": -0.5, "epsilon_min": 0.01},
        "comp":  {"enabled": False, "K": 3, "gamma": 0.10, "mode": "full"},
    },
    {
        "label": "Compart. seul (K=3 full)",
        "meta":  {"enabled": False},
        "comp":  {"enabled": True, "K": 3, "gamma": 0.10, "mode": "full"},
    },
    {
        "label": "Metacog + Compart. (combi)",
        "meta":  {"enabled": True, "alpha_meta": -0.5, "epsilon_min": 0.01},
        "comp":  {"enabled": True, "K": 3, "gamma": 0.10, "mode": "full"},
    },
]

print("=" * 76)
print("  V5 — COMBINAISON METACOGNITION + COMPARTIMENTALISATION")
print(f"  Reseau {SIZE}x{SIZE} | {STEPS} steps | derniers 25% | {len(SEEDS)} seeds | I_stim={I_STIM}")
print("=" * 76)
print(f"\n  {'Condition':<38}  {'H_stable':>10}  {'Delta_V4':>9}")
print(f"  {'-'*62}")

results = []
for cond in CONDITIONS:
    H_runs = []
    for seed in SEEDS:
        net = Mem4Network(size=SIZE, heretic_ratio=0.15, seed=seed)
        net.model.cfg['metacognitive'] = dict(cond['meta'])
        net.model.cfg['compartments']  = dict(cond['comp'])
        H_win = []
        for step in range(STEPS):
            net.step(I_stimulus=I_STIM)
            if step >= WARMUP:
                H_win.append(calculate_continuous_entropy(net.model.v))
        H_runs.append(np.mean(H_win))

    r = {
        "label": cond["label"],
        "H":     np.mean(H_runs),
        "H_std": np.std(H_runs),
        "meta_enabled": cond["meta"]["enabled"],
        "comp_enabled": cond["comp"]["enabled"],
    }
    results.append(r)
    print(f"  {r['label']:<38}  {r['H']:>6.2f}±{r['H_std']:.2f}  ", end="")
    if results[0]["H"] > 0:
        delta = r["H"] - results[0]["H"]
        print(f"{delta:>+8.2f}")
    else:
        print("       —")

# Verdict
print(f"\n{'=' * 76}")
print("  VERDICT")
print(f"{'=' * 76}\n")

H_v4       = results[0]["H"]
H_meta     = results[1]["H"]
H_comp     = results[2]["H"]
H_combi    = results[3]["H"]
delta_meta = H_meta - H_v4
delta_comp = H_comp - H_v4
delta_combi = H_combi - H_v4
expected_additive = delta_meta + delta_comp

print(f"  V4 pur              : H = {H_v4:.4f} bits")
print(f"  Metacog seul        : H = {H_meta:.4f} bits  (delta {delta_meta:+.4f})")
print(f"  Compart. seul       : H = {H_comp:.4f} bits  (delta {delta_comp:+.4f})")
print(f"  Combinaison         : H = {H_combi:.4f} bits  (delta {delta_combi:+.4f})")
print(f"\n  Somme des gains individuels : {expected_additive:+.4f} bits")
print(f"  Gain reel de la combinaison : {delta_combi:+.4f} bits")
synergy = delta_combi - expected_additive
if synergy > 0.05:
    verdict = "SYNERGIE — la combinaison surpasse la somme des parties !"
elif synergy > -0.05:
    verdict = "ADDITIVITE — les gains s'additionnent approximativement."
elif synergy > -0.20:
    verdict = "INTERFERENCE PARTIELLE — la combinaison perd une partie des gains."
else:
    verdict = "INTERFERENCE FORTE — les features se neutralisent mutuellement."
print(f"\n  --> {verdict}")
print(f"      Synergisme : {synergy:+.4f} bits (positif=synergie, negatif=interference)")

# Sauvegarder CSV
out_dir = os.path.join(os.path.dirname(__file__), '../figures')
os.makedirs(out_dir, exist_ok=True)
csv_path = os.path.join(out_dir, 'p2_v5_combination.csv')
with open(csv_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['label', 'H_stable', 'H_std', 'delta_v4', 'meta_enabled', 'comp_enabled'])
    for r in results:
        w.writerow([r['label'], round(r['H'], 6), round(r['H_std'], 6),
                    round(r['H'] - results[0]['H'], 6),
                    r['meta_enabled'], r['comp_enabled']])
print(f"\n  CSV -> {csv_path}")

print(f"\n{'=' * 76}")
print("  Pour reproduire : python experiments/p2_v5_combination.py")
print(f"{'=' * 76}\n")
