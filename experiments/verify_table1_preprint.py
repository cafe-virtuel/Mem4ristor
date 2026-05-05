"""
Verification du Tableau 1 de preprint.tex
Reproduit : H_stable et mean_doubt pour lattices 4x4, 10x10, 25x25

Parametres testes :
  - I_stimulus = 0.0  (hypothese texte §4.1)
  - I_stimulus = 0.5  (hypothese legende Tableau 1)
  3000 steps, derniers 25%, 10 seeds, eta=0.15, dt=0.05

Note : passe de 3 a 10 seeds (2026-05-05) pour robustesse statistique.
L'ecart-type sur n=3 n'est pas fiable ; n=10 permet une estimation credible.
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from mem4ristor.core import Mem4Network
from mem4ristor.metrics import calculate_continuous_entropy

STEPS    = 3000
WARMUP   = int(STEPS * 0.75)   # derniers 25%
SEEDS    = [42, 123, 777, 17, 256, 1337, 99, 314, 2024, 888]  # n=10
SIZES    = [4, 10, 25]          # 4x4=16, 10x10=100, 25x25=625
ETA      = 0.15
I_VALUES = [0.0, 0.5]

print(f"{'='*65}")
print(f"  VERIFICATION TABLEAU 1 — {STEPS} steps, derniers 25%, n={len(SEEDS)} seeds")
print(f"{'='*65}")

for I_stim in I_VALUES:
    print(f"\n>>> I_stimulus = {I_stim}")
    print(f"  {'Size':>6}  {'N':>5}  {'H_stable':>12}  {'std':>7}  {'doubt_mean':>11}")
    print(f"  {'-'*55}")
    for size in SIZES:
        H_runs  = []
        u_runs  = []
        for seed in SEEDS:
            net = Mem4Network(size=size, heretic_ratio=ETA, seed=seed)
            v_history = []
            u_history = []
            for step in range(STEPS):
                net.step(I_stimulus=I_stim)
                if step >= WARMUP:
                    v_history.append(net.model.v.copy())
                    u_history.append(net.model.u.mean())
            # H_stable = moyenne des entropies instantanees sur la fenetre
            H_vals = [calculate_continuous_entropy(v) for v in v_history]
            H_runs.append(np.mean(H_vals))
            u_runs.append(np.mean(u_history))

        H_mean = np.mean(H_runs)
        H_std  = np.std(H_runs)
        u_mean = np.mean(u_runs)
        N      = size * size
        print(f"  {size:>2}x{size:<2}   {N:>5}  {H_mean:>8.2f} bits  ±{H_std:.2f}  u={u_mean:.3f}")

print(f"\n{'='*65}")
print("Valeurs attendues dans preprint.tex (legende Tableau 1, I_stim=0.5, n=10) :")
print("  4x4  (N=16)  : H = 3.22 +/- 0.14,  u = 0.993")
print("  10x10(N=100) : H = 4.06 +/- 0.08,  u = 0.983")
print("  25x25(N=625) : H = 4.28 +/- 0.06,  u = 0.974")
print(f"{'='*65}")
