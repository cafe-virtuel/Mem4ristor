#!/usr/bin/env python3
"""
LIMIT-05 Part 2: Stability Analysis
====================================
Measures SUSTAINED (attractor) entropy vs transient peaks.
The preprint claims H ≈ 1.94 as an ATTRACTOR, not a transient peak.

Date: 2026-03-21
Investigator: Claude Opus 4.6 (Anthropic) — Cafe Virtuel session
"""

import numpy as np
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from mem4ristor.core import Mem4ristorV3, Mem4Network

def run_stability(params, n_steps=5000, seeds=(42, 137, 314, 256, 999)):
    all_trajectories = []
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
        trajectory = []
        for step_i in range(n_steps):
            net.step(params.get('I_stimulus', 0.0))
            if step_i % 5 == 0: trajectory.append(net.calculate_entropy())
        all_trajectories.append(trajectory)
    trajectories = np.array(all_trajectories)
    n_samples = trajectories.shape[1]
    last_quarter = trajectories[:, int(n_samples * 0.75):]
    return {'sustained_h': np.mean(last_quarter), 'sustained_std': np.std(last_quarter),
            'peak_h': np.max(trajectories), 'mean_trajectory': np.mean(trajectories, axis=0)}

def main():
    print("LIMIT-05 STABILITY ANALYSIS")
    print(f"Theoretical max: {np.log2(5):.4f}\n")
    configs = {
        'DEFAULT (paper)': {'D': 0.15, 'heretic_ratio': 0.15, 'sigma_v': 0.05, 'I_stimulus': 0.0},
        'Weak D + stimulus': {'D': 0.01, 'heretic_ratio': 0.15, 'sigma_v': 0.05, 'I_stimulus': 1.0},
        'Best transient': {'D': 0.01, 'heretic_ratio': 0.05, 'sigma_v': 0.1, 'I_stimulus': 1.0},
        'Moderate + noise': {'D': 0.10, 'heretic_ratio': 0.15, 'sigma_v': 0.20, 'I_stimulus': 0.5},
        'Strong coupling': {'D': 0.50, 'heretic_ratio': 0.15, 'sigma_v': 0.05, 'I_stimulus': 0.0},
        'High heretics': {'D': 0.15, 'heretic_ratio': 0.50, 'sigma_v': 0.10, 'I_stimulus': 0.0},
        'Slow doubt': {'D': 0.05, 'heretic_ratio': 0.20, 'sigma_v': 0.15, 'I_stimulus': 0.5,
                       'epsilon_u': 0.005, 'k_u': 0.5, 'tau_u': 5.0},
    }
    print(f"{'Config':<30s} | {'Sustained H':>12s} | {'± std':>8s} | {'Peak H':>8s}")
    print("-" * 70)
    for name, params in configs.items():
        r = run_stability(params, n_steps=5000)
        print(f"{name:<30s} | {r['sustained_h']:>12.4f} | {r['sustained_std']:>8.4f} | {r['peak_h']:>8.4f}")

if __name__ == '__main__':
    main()
