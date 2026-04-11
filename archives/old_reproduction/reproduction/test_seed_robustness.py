"""
Test robustness of Mem4ristor v2.0.4 to random seed variation.
Verifies that entropy statistics are stable across different random seeds.

Author: Antigravity (Session 8)
Date: 2025-12-28
"""

import numpy as np
import sys
import os

# Inject src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.append(os.path.dirname(__file__))

from reference_impl import Mem4ristorV2

def test_seed_robustness(n_seeds: int = 10, n_steps: int = 5000) -> bool:
    """
    Run verification with multiple random seeds and check statistical stability.
    
    Args:
        n_seeds: Number of different seeds to test
        n_steps: Number of simulation steps per seed
    
    Returns:
        bool: True if coefficient of variation < 10%, False otherwise
    
    Pass Criterion:
        Entropy CV (std/mean) must be < 0.10 for stability
    """
    seeds = [42, 123, 456, 789, 1337, 2048, 9999, 12345, 54321, 77777][:n_seeds]
    results = []
    
    print(f"\n=== SEED ROBUSTNESS TEST ===")
    print(f"Testing {n_seeds} different random seeds...")
    print(f"Steps per seed: {n_steps}")
    print(f"Stimulus: I = 1.1 (bias phase)")
    print()
    
    for idx, seed in enumerate(seeds, 1):
        model = Mem4ristorV2()
        model.rng = np.random.RandomState(seed)
        model._initialize_params(100)
        
        # Run with bias stimulus
        for _ in range(n_steps):
            model.step(I_stimulus=1.1)
        
        # Measure final metrics
        H_final = model.calculate_entropy()
        states = model.get_states()
        n_occupied = len(np.unique(states))
        v_std = np.std(model.v)
        u_mean = np.mean(model.u)
        
        results.append({
            'seed': seed,
            'H_final': H_final,
            'n_occupied': n_occupied,
            'v_std': v_std,
            'u_mean': u_mean
        })
        
        print(f"Seed {idx:2d} ({seed:5d}): H = {H_final:.4f}, States = {n_occupied}, σ(v) = {v_std:.4f}")
    
    # Statistical Analysis
    H_vals = np.array([r['H_final'] for r in results])
    H_mean = np.mean(H_vals)
    H_std = np.std(H_vals)
    H_min = np.min(H_vals)
    H_max = np.max(H_vals)
    
    print(f"\n{'='*50}")
    print(f"STATISTICAL SUMMARY")
    print(f"{'='*50}")
    print(f"Entropy Mean:   {H_mean:.4f}")
    print(f"Entropy Std:    {H_std:.4f}")
    print(f"Entropy Range:  [{H_min:.4f}, {H_max:.4f}]")
    
    # Coefficient of Variation
    cv = (H_std / H_mean * 100) if H_mean > 0 else float('inf')
    print(f"Coeff. Var.:    {cv:.2f}%")
    print()
    
    # Pass/Fail Decision
    threshold_pct = 10.0
    if cv < threshold_pct:
        print(f"✅ [PASS] CV = {cv:.2f}% < {threshold_pct}% threshold")
        print(f"Conclusion: Model is ROBUST to random seed variation")
        verdict = True
    else:
        print(f"❌ [FAIL] CV = {cv:.2f}% >= {threshold_pct}% threshold")
        print(f"Conclusion: Model shows EXCESSIVE sensitivity to seed")
        verdict = False
    
    print(f"{'='*50}\n")
    return verdict


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Mem4ristor seed robustness')
    parser.add_argument('--seeds', type=int, default=10, help='Number of seeds to test')
    parser.add_argument('--steps', type=int, default=5000, help='Steps per simulation')
    
    args = parser.parse_args()
    
    passed = test_seed_robustness(n_seeds=args.seeds, n_steps=args.steps)
    
    sys.exit(0 if passed else 1)
