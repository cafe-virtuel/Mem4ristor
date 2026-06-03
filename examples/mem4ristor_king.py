"""
WARNING: EXPERIMENTAL CODE - Not production-ready.

Mem4ristor v3.1 "The Philosopher King" - Experimental Extension.

Known Issues (documented in audit 2026-02-16):
  1. Temporary state mutation (self.u, self.D_eff) is fragile and not thread-safe
  2. super().step(I_stimulus) called without coupling_input, so social coupling
     is absent in standalone mode - the martial law mechanism has no effect
  3. No input sanitization guards (unlike the canonical Mem4ristorV3 in core.py)

This file is preserved for research exploration and transparency.
It should NOT be imported by production code or used in benchmarks.

To use: import directly from this file, not from the mem4ristor package.
"""
import numpy as np
from typing import Optional, Dict

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from mem4ristor.core import Mem4ristorV3, Mem4Network


class Mem4ristorKing(Mem4ristorV3):
    """
    Mem4ristor v3.1 (The Philosopher King) - EXPERIMENTAL.

    Concepts:
    1. Dynamic Constitution (Martial Law):
       Resolves 'Analysis Paralysis' via a Frustration variable (Phi).
       If error persists, Phi rises -> Force Consensus (u=0).

    2. Metacognitive Metabolism (Autopoiesis):
       Self-regulation of physical constants (epsilon, noise) based on Boredom (Low Entropy).
       "A bored system becomes paranoid/manic to feel alive."
    """
    def __init__(self, config: Optional[Dict] = None, seed: int = 42):
        super().__init__(config, seed)

        # 1. Dynamic Constitution
        self.frustration_phi = 0.0
        self.decay_phi = 0.95
        self.gain_phi = 0.05
        self.martial_threshold = 0.8

        # 2. Metacognition
        self.boredom_index = 0.0
        self.base_epsilon = self.cfg['dynamics']['epsilon']
        self.base_sigma_v = self.cfg['noise'].get('sigma_v', 0.05)

    def metacognitive_metabolism(self):
        """
        Adjusts physical parameters based on internal vitality (Entropy).
        Second-order homeostasis.
        """
        H = self.calculate_entropy()
        self.boredom_index = np.clip(1.0 - (H / 1.5), 0, 1)

        target_epsilon = self.base_epsilon + (0.10 * self.boredom_index)
        current_eps = self.cfg['dynamics']['epsilon']
        self.cfg['dynamics']['epsilon'] += 0.01 * (target_epsilon - current_eps)

        target_noise = self.base_sigma_v + (0.15 * self.boredom_index)
        current_sigma = self.cfg['noise']['sigma_v']
        self.cfg['noise']['sigma_v'] += 0.01 * (target_noise - current_sigma)

    def step(self, I_stimulus: float = 0.0, target_vector: Optional[np.ndarray] = None) -> Dict:
        """
        Advanced Step with Meta-Control Loops.
        """
        if target_vector is None:
            target_vector = np.full(self.N, 1.5)

        current_error = np.mean((self.v - target_vector)**2)

        if current_error > 0.5:
            self.frustration_phi = np.clip(self.frustration_phi + self.gain_phi, 0, 1.2)
        else:
            self.frustration_phi *= self.decay_phi

        martial_law_active = self.frustration_phi > self.martial_threshold

        if martial_law_active:
            u_meta_coupling = np.zeros_like(self.u)
            D_meta = self.D_eff * 2.5
        else:
            u_meta_coupling = self.u
            D_meta = self.D_eff

        backup_u = self.u.copy()
        backup_D = self.D_eff

        if martial_law_active:
            self.u = u_meta_coupling
            self.D_eff = D_meta

        self.metacognitive_metabolism()

        # NOTE: coupling_input is not passed here - see WARNING at top of file
        super().step(I_stimulus)

        if martial_law_active:
            self.u = backup_u
            self.D_eff = backup_D

        return {
            'martial_law': martial_law_active,
            'frustration': self.frustration_phi,
            'boredom': self.boredom_index,
            'epsilon': self.cfg['dynamics']['epsilon'],
            'noise': self.cfg['noise']['sigma_v']
        }
