#!/usr/bin/env python3
"""
LIMIT-04 Investigation: Long-Term Stability Drift
Date: 2026-03-21 | Investigator: Claude Opus 4.6

See experiments/entropy_sweep/README.md for context.
Run: python limit04_stability.py
"""
import numpy as np, time, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from mem4ristor.core import Mem4Network

def measure_drift(size=10, dt=0.05, n_steps=10000, seed=42, I_stim=0.0):
    net = Mem4Network(size=size, heretic_ratio=0.15, seed=seed)
    net.model.cfg['dynamics']['dt'] = dt; net.model.dt = dt
    win = 500; n_win = n_steps // win
    wins = []
    for w in range(n_win):
        ent = []
        for s in range(win):
            net.step(I_stim)
            if s % 10 == 0: ent.append(net.calculate_entropy())
        wins.append(np.mean(ent))
    q = n_win // 4
    h1, h4 = np.mean(wins[:q]), np.mean(wins[-q:])
    drift = abs(h4 - h1) / max(h1, 0.01) * 100
    return {'H_first': h1, 'H_last': h4, 'drift_pct': drift,
            'diverged': np.any(~np.isfinite(net.model.v))}

if __name__ == '__main__':
    seeds = [42, 137, 314]
    print("LIMIT-04: Euler Stability Drift\n")
    print(f"  {'dt':>6s} | {'H_first':>8s} | {'H_last':>8s} | {'drift%':>8s}")
    print("  " + "-" * 40)
    for dt in [0.01, 0.02, 0.05, 0.07, 0.10, 0.15, 0.20, 0.50]:
        res = [measure_drift(dt=dt, n_steps=5000, seed=s) for s in seeds]
        print(f"  {dt:>6.3f} | {np.mean([r['H_first'] for r in res]):>8.4f} | "
              f"{np.mean([r['H_last'] for r in res]):>8.4f} | "
              f"{np.mean([r['drift_pct'] for r in res]):>7.1f}%")
    print("\nExtended run (20000 steps, dt=0.05):")
    for s in seeds:
        r = measure_drift(dt=0.05, n_steps=20000, seed=s)
        print(f"  Seed {s}: H_first={r['H_first']:.4f}, H_last={r['H_last']:.4f}, drift={r['drift_pct']:.1f}%")
