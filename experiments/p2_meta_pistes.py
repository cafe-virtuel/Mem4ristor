"""
V5 — Plasticite Metacognitive : exploration de deux pistes

PISTE A : meme logique (douteux=lent), mais stimulus = 0
  -> u est moins sature (~0.5-0.7 au lieu de ~0.99)
  -> la modulation epsilon_i est plus etalee entre les noeuds
  -> est-ce que la diversite augmente davantage ?

PISTE B : logique inversee (douteux=rapide), stimulus = 0.5
  -> alpha_meta negatif : epsilon AUGMENTE quand u est eleve
  -> intuition : un noeud incertain "s'emballe" plus facilement
  -> est-ce que ca amplifie la diversite ?

Comparaison dans les deux cas avec le V4 pur (referentiel).
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
EPS    = 0.08   # epsilon de base

# ══════════════════════════════════════════════════════════════════════════
# PISTE A : I_stim = 0  (u moins sature)
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("  PISTE A : I_stim=0  (u moins sature -> modulation plus etalee)")
print("=" * 70)
print(f"\n  {'Condition':<36}  {'H_stable':>10}  {'u_mean':>8}  {'eps_spread':>12}")
print(f"  {'-'*66}")

conds_A = [
    {"label": "V4 pur  (alpha=0.00)",  "alpha": 0.00, "enabled": False},
    {"label": "META    (alpha=0.25)",  "alpha": 0.25, "enabled": True},
    {"label": "META    (alpha=0.50)",  "alpha": 0.50, "enabled": True},
    {"label": "META    (alpha=1.00)",  "alpha": 1.00, "enabled": True},
]

results_A = []
for cond in conds_A:
    H_runs, u_runs, eps_runs = [], [], []
    for seed in SEEDS:
        net = Mem4Network(size=SIZE, heretic_ratio=0.15, seed=seed)
        net.model.cfg['metacognitive'] = {
            'enabled': cond['enabled'], 'alpha_meta': cond['alpha'], 'epsilon_min': 0.01
        }
        H_win, u_win, eps_win = [], [], []
        for step in range(STEPS):
            net.step(I_stimulus=0.0)   # <-- stimulus zero
            if step >= WARMUP:
                H_win.append(calculate_continuous_entropy(net.model.v))
                u_win.append(net.model.u.mean())
                if cond['enabled']:
                    eps_i = EPS * (1.0 + cond['alpha'] * (0.5 - net.model.u))
                    eps_win.append(np.maximum(eps_i, 0.01).std())
                else:
                    eps_win.append(0.0)
        H_runs.append(np.mean(H_win))
        u_runs.append(np.mean(u_win))
        eps_runs.append(np.mean(eps_win))

    r = {"label": cond["label"], "alpha": cond["alpha"],
         "H": np.mean(H_runs), "H_std": np.std(H_runs),
         "u": np.mean(u_runs), "eps": np.mean(eps_runs)}
    results_A.append(r)
    print(f"  {r['label']:<36}  {r['H']:>6.2f}±{r['H_std']:.2f}  {r['u']:>7.3f}  {r['eps']:>12.5f}")

# ══════════════════════════════════════════════════════════════════════════
# PISTE B : logique inversee (douteux=rapide), I_stim = 0.5
# ══════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print("  PISTE B : logique inversee — douteux=RAPIDE (alpha negatif)")
print("=" * 70)
print(f"\n  {'Condition':<36}  {'H_stable':>10}  {'u_mean':>8}  {'eps_spread':>12}")
print(f"  {'-'*66}")

conds_B = [
    {"label": "V4 pur  (alpha= 0.00)", "alpha":  0.00, "enabled": False},
    {"label": "INV     (alpha=-0.25)", "alpha": -0.25, "enabled": True},
    {"label": "INV     (alpha=-0.50)", "alpha": -0.50, "enabled": True},
    {"label": "INV     (alpha=-1.00)", "alpha": -1.00, "enabled": True},
]

results_B = []
for cond in conds_B:
    H_runs, u_runs, eps_runs = [], [], []
    for seed in SEEDS:
        net = Mem4Network(size=SIZE, heretic_ratio=0.15, seed=seed)
        net.model.cfg['metacognitive'] = {
            'enabled': cond['enabled'], 'alpha_meta': cond['alpha'], 'epsilon_min': 0.01
        }
        H_win, u_win, eps_win = [], [], []
        for step in range(STEPS):
            net.step(I_stimulus=0.5)  # <-- stimulus normal
            if step >= WARMUP:
                H_win.append(calculate_continuous_entropy(net.model.v))
                u_win.append(net.model.u.mean())
                if cond['enabled']:
                    eps_i = EPS * (1.0 + cond['alpha'] * (0.5 - net.model.u))
                    eps_win.append(np.maximum(eps_i, 0.01).std())
                else:
                    eps_win.append(0.0)
        H_runs.append(np.mean(H_win))
        u_runs.append(np.mean(u_win))
        eps_runs.append(np.mean(eps_win))

    r = {"label": cond["label"], "alpha": cond["alpha"],
         "H": np.mean(H_runs), "H_std": np.std(H_runs),
         "u": np.mean(u_runs), "eps": np.mean(eps_runs)}
    results_B.append(r)
    print(f"  {r['label']:<36}  {r['H']:>6.2f}±{r['H_std']:.2f}  {r['u']:>7.3f}  {r['eps']:>12.5f}")

# ══════════════════════════════════════════════════════════════════════════
# VERDICT
# ══════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print("  VERDICT COMPARE")
print(f"{'=' * 70}\n")

ref_A = results_A[0]["H"]
ref_B = results_B[0]["H"]
best_A = max(results_A, key=lambda x: x["H"])
best_B = max(results_B, key=lambda x: x["H"])

print(f"  PISTE A (I_stim=0, douteux=lent) :")
print(f"    Referentiel V4 = {ref_A:.2f} bits")
print(f"    Meilleur META  = {best_A['H']:.2f} bits  ({best_A['label'].strip()})")
dA = best_A['H'] - ref_A
print(f"    Delta          = {dA:+.2f} bits  -> ", end="")
if dA > 0.10:
    print("GAIN reel — la piste A fonctionne !")
elif dA < -0.10:
    print("perte — ralentir les noeuds nuit meme sans stimulus.")
else:
    print("neutre — u reste trop uniforme meme a I_stim=0.")

print(f"\n  PISTE B (I_stim=0.5, douteux=rapide) :")
print(f"    Referentiel V4 = {ref_B:.2f} bits")
print(f"    Meilleur INV   = {best_B['H']:.2f} bits  ({best_B['label'].strip()})")
dB = best_B['H'] - ref_B
print(f"    Delta          = {dB:+.2f} bits  -> ", end="")
if dB > 0.10:
    print("GAIN reel — emballer les noeuds incertains amplifie la diversite !")
elif dB < -0.10:
    print("perte — emballer les noeuds incertains les desynchronise trop.")
else:
    print("neutre — la logique inversee ne change pas le resultat.")

print(f"\n  u_moyen piste A (sans stimulus) : {results_A[0]['u']:.3f}  "
      f"vs piste B (avec stimulus) : {results_B[0]['u']:.3f}")
print("  (si u_moyen piste A < piste B, u est plus etale -> modulation plus visible)")

print(f"\n{'=' * 70}\n")
