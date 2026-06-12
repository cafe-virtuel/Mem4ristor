#!/usr/bin/env python3
"""
Dead Zone check — 2026-06-12 (Claude Code / Fable, session autonome)

QUESTION : le scaling Euler-Maruyama du bruit (commit 818cf67, 1er mai) qui
multiplie le bruit effectif par 1/sqrt(dt) ≈ 4.47 a-t-il DÉTRUIT la dead zone ?
Contexte : claim [12] (Matern) = tout bruit η ≥ 0.1 brise la dead zone ; or le
sigma_v par défaut (0.05) devient effectivement ~0.22 avec le nouveau scaling.

MÉTHODE : BA m=5 N=100 (dead zone canonique, λ₂≈2.9) + BA m=3 (contrôle vivant),
FULL, I_stim=0.5, 3 seeds, 3000 pas après warmup 1000. Mesure H_cont / H_cog / sync.
À exécuter dans DEUX worktrees : HEAD (nouveau bruit) et 0fdeee0 (ancien bruit).

LECTURE : ancien code, dead zone attendue : H_cont ≈ 1.4 (cf. S01).
Si HEAD donne H_cont ≈ 3+ sur BA m=5, la dead zone n'existe plus au bruit par défaut.
"""
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'src'))

from mem4ristor.core import Mem4Network
from mem4ristor.graph_utils import make_ba
from mem4ristor.metrics import (
    calculate_cognitive_entropy,
    calculate_continuous_entropy,
    calculate_pairwise_synchrony,
)

SEEDS = [42, 123, 777]
I_STIM = 0.5
WARM_UP = 1000
STEPS = 3000
N = 100

if __name__ == '__main__':
    print(f"Dead Zone check — repo: {HERE.parent}")
    for m in (3, 5):
        adj = make_ba(N, m, seed=42)
        hcont_l, hcog_l, sync_l = [], [], []
        for seed in SEEDS:
            net = Mem4Network(adjacency_matrix=adj.copy(), heretic_ratio=0.15,
                              seed=seed, coupling_norm='degree_linear')
            for _ in range(WARM_UP):
                net.step(I_stimulus=I_STIM)
            snaps = []
            for _ in range(STEPS):
                net.step(I_stimulus=I_STIM)
                snaps.append(net.v.copy())
            v_s = np.array(snaps)
            hcont_l.append(float(np.mean([calculate_continuous_entropy(v) for v in v_s[::10]])))
            hcog_l.append(float(np.mean([calculate_cognitive_entropy(v) for v in v_s[::10]])))
            sync_l.append(calculate_pairwise_synchrony(v_s))
        print(f"BA m={m} : H_cont={np.mean(hcont_l):.3f}±{np.std(hcont_l):.3f}  "
              f"H_cog={np.mean(hcog_l):.3f}±{np.std(hcog_l):.3f}  "
              f"sync={np.mean(sync_l):.3f}±{np.std(sync_l):.3f}")
