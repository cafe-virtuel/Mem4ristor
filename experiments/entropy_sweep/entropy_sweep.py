#!/usr/bin/env python3
"""
LIMIT-05 Investigation: Systematic Parameter Sweep for Maximum Entropy H
=========================================================================
Goal: Either find conditions where H ≈ 1.94 (preprint claim) or establish
the true maximum achievable entropy with the current model.

Entropy uses 5 cognitive bins: [-inf, -1.5, -0.8, 0.8, 1.5, inf]
Theoretical max H = log2(5) ≈ 2.3219 (perfect uniform across 5 bins)
Preprint claims H ≈ 1.94 attractor
Current empirical max found ≈ 1.56

Strategy:
  Phase 1: Coarse sweep over key parameters (D, heretic_ratio, sigma_v, epsilon, I_stimulus)
  Phase 2: Fine sweep around best candidates from Phase 1
  Phase 3: Extended runs (more steps) on top candidates to check for late-emerging attractors
  Phase 4: Network size & boundary condition effects

Date: 2026-03-21
Investigator: Claude Opus 4.6 (Anthropic) — Cafe Virtuel session
"""

import numpy as np
import time
import itertools
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from mem4ristor.core import Mem4ristorV3, Mem4Network

def run_single(params, n_steps=2000, seeds=(42, 137, 314)):
    results = []
    for seed in seeds:
        cfg = {
            'dynamics': {'a': 0.7, 'b': 0.8, 'epsilon': params.get('epsilon', 0.08),
                         'alpha': params.get('alpha', 0.15), 'v_cubic_divisor': params.get('v_cubic_divisor', 5.0),
                         'dt': params.get('dt', 0.05), 'lambda_learn': params.get('lambda_learn', 0.05),
                         'tau_plasticity': params.get('tau_plasticity', 1000), 'w_saturation': params.get('w_saturation', 2.0)},
            'coupling': {'D': params.get('D', 0.15), 'heretic_ratio': params.get('heretic_ratio', 0.15),
                         'uniform_placement': params.get('uniform_placement', True)},
            'doubt': {'epsilon_u': params.get('epsilon_u', 0.02), 'k_u': params.get('k_u', 1.0),
                      'sigma_baseline': params.get('sigma_baseline', 0.05), 'u_clamp': [0.0, 1.0],
                      'tau_u': params.get('tau_u', 1.0), 'alpha_surprise': params.get('alpha_surprise', 2.0),
                      'surprise_cap': params.get('surprise_cap', 5.0)},
            'noise': {'sigma_v': params.get('sigma_v', 0.05), 'use_rtn': params.get('use_rtn', False)}
        }
        size = params.get('size', 10)
        net = Mem4Network(size=size, heretic_ratio=params.get('heretic_ratio', 0.15),
                         seed=seed, boundary=params.get('boundary', 'periodic'))
        net.model.cfg = net.model._deep_merge(net.model.cfg, cfg)
        net.model.D_eff = cfg['coupling']['D'] / np.sqrt(net.N)
        net.model.lambda_learn = cfg['dynamics']['lambda_learn']
        net.model.tau_plasticity = cfg['dynamics']['tau_plasticity']
        net.model.w_saturation = cfg['dynamics']['w_saturation']
        I_stim = params.get('I_stimulus', 0.0)
        entropies, max_h, best_step = [], 0.0, 0
        for step_i in range(n_steps):
            net.step(I_stim)
            if step_i % 10 == 0:
                h = net.calculate_entropy()
                entropies.append(h)
                if h > max_h: max_h, best_step = h, step_i
        results.append({'seed': seed, 'max_h': max_h, 'final_h': net.calculate_entropy(),
                        'best_step': best_step, 'distribution': net.get_state_distribution()})
    return {'mean_max_h': np.mean([r['max_h'] for r in results]),
            'best_max_h': max(r['max_h'] for r in results),
            'mean_final_h': np.mean([r['final_h'] for r in results]), 'details': results}

def main():
    t0 = time.time()
    print(f"LIMIT-05 INVESTIGATION | Theoretical max = {np.log2(5):.4f} | Claim = 1.94\n")
    param_grid = {'D': [0.01, 0.05, 0.10, 0.15, 0.25, 0.50, 1.0, 2.0, 5.0],
                  'heretic_ratio': [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50],
                  'sigma_v': [0.01, 0.05, 0.10, 0.20, 0.50],
                  'I_stimulus': [0.0, 0.1, 0.5, 1.0, 2.0]}
    best = []
    for D in param_grid['D']:
        for hr in param_grid['heretic_ratio']:
            r = run_single({'D': D, 'heretic_ratio': hr}, n_steps=1500)
            best.append((r['best_max_h'], r['mean_max_h'], {'D': D, 'heretic_ratio': hr}, r))
    best.sort(key=lambda x: x[0], reverse=True)
    print("Phase 1 TOP 5:")
    for h, hm, p, _ in best[:5]: print(f"  H={h:.4f} | D={p['D']}, hr={p['heretic_ratio']}")
    top_D, top_hr = best[0][2]['D'], best[0][2]['heretic_ratio']
    for sv in param_grid['sigma_v']:
        for I_s in param_grid['I_stimulus']:
            r = run_single({'D': top_D, 'heretic_ratio': top_hr, 'sigma_v': sv, 'I_stimulus': I_s}, n_steps=1500)
            best.append((r['best_max_h'], r['mean_max_h'], {'D': top_D, 'heretic_ratio': top_hr, 'sigma_v': sv, 'I_stimulus': I_s}, r))
    best.sort(key=lambda x: x[0], reverse=True)
    print(f"\nGlobal best transient: H = {best[0][0]:.4f} | params = {best[0][2]}")
    print(f"Time: {time.time()-t0:.1f}s")

if __name__ == '__main__':
    main()
