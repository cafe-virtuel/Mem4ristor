# Scientific Limitations Investigation Suite

**Date**: 2026-03-21
**Investigator**: Claude Opus 4.6 (Anthropic) — Cafe Virtuel session
**Coordinated by**: Julien Chauvin

## Scripts

| Script | Limitation | Finding |
|:---|:---|:---|
| `entropy_sweep.py` | LIMIT-05: H ≈ 1.94 claim | FALSE — transient peaks reach 2.31, stable H ≈ 0.92 |
| `stability_analysis.py` | LIMIT-05: sustained entropy | Best sustained H = 1.48 ± 0.66 (weak D + stimulus) |
| `limit02_scalefree.py` | LIMIT-02: scale-free strangulation | CONFIRMED — H collapses to 0.00 on BA networks, V4 rewiring ineffective |
| `limit04_stability.py` | LIMIT-04: Euler drift | NUANCED — "drift" is transient convergence, system stable after ~2000 steps at dt≤0.05 |

## How to Reproduce

```bash
cd experiments/entropy_sweep
python entropy_sweep.py        # ~4 min — LIMIT-05 parameter sweep
python stability_analysis.py   # ~3 min — LIMIT-05 attractor analysis
python limit02_scalefree.py    # ~1 min — LIMIT-02 scale-free test
python limit04_stability.py    # ~1 min — LIMIT-04 stability characterization
```

Requires: numpy, scipy, pyyaml (from project requirements.txt)

## Key Conclusions

1. **LIMIT-05** (H ≈ 1.94): The preprint claim is definitively false as an attractor. The system passes through high-entropy transients but settles at H ≈ 0.92 with default parameters.

2. **LIMIT-02** (Scale-free): Hub strangulation is catastrophic (H → 0). The V4 doubt-driven rewiring performs thousands of reconnections but has zero effect on entropy because synchronization occurs in ~100 steps, before rewiring can intervene. Fundamental redesign needed (degree-normalized coupling suggested).

3. **LIMIT-04** (Stability): The original "drift > 5%" finding was measuring transient convergence, not instability. At dt=0.05, the system reaches steady state by ~2000-3000 steps and remains stable (std ≈ 0.016) for at least 20000 steps.
