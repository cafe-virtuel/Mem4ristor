# CHANGELOG

---

## v3.1.0 → v3.2.0

**Date**: 2026-04-10
**Contributors**: Claude Opus 4.6 (Implementation + Investigation), Antigravity (Audit + Consolidation)
**Orchestrator**: Julien Chauvin

### New Features

| Feature | Detail |
|:---|:---|
| **V5 Hysteresis** | Dead-zone latching [θ_low=0.35, θ_high=0.65] with Schmitt-trigger transitions + watchdog fatigue. Mode state array per neuron. 3 new tests. |
| **Sparse CSR Backend** | Auto-conversion to `scipy.sparse.csr_matrix` for N > 1000. Memory reduction 455× at N=5000, sparse eigsh for spectral gap. Transparent to API. |
| **Degree-Normalized Coupling** | 4 modes: `uniform`, `degree`, `degree_linear`, `degree_log`. `degree_linear` (D/deg(i)) recovers 86% of lattice performance on BA m=3 networks. |
| **Demo Pipeline** | `examples/demo_applied.py`: 4 scenarios (sensory, hysteresis, scale-free sparse, phase diversity), 5 PNG outputs. |

### Scientific Investigations

| Investigation | Result |
|:---|:---|
| **LIMIT-05 (H ≈ 1.94)** | FALSE. Stable attractor H ≈ 0.92 ± 0.04. Transient peaks ≈ 2.31. 800+ parameter sweeps. |
| **LIMIT-04 (Euler stability)** | NUANCED. dt ≤ 0.05 stable to 20,000+ steps. "Drift" was transient convergence, not instability. |
| **LIMIT-02 (Scale-free strangulation)** | PARTIALLY RESOLVED. `degree_linear` works on BA m=3, HK, WS, ER sparse. Does NOT work on BA m=1/5/10, Config Models, ER dense. 3 regimes identified. Optimal normalization depends on degree heterogeneity × path redundancy interaction. |

### Multi-Topology Validation (13 topologies, 3 seeds, 3000 steps)

Three distinct normalization regimes discovered:
1. **`degree_linear` wins**: BA m=3, Holme-Kim, Watts-Strogatz, ER sparse
2. **`uniform` wins**: BA m=1 (trees), BA m=10, all Configuration Models
3. **Neither works**: BA m=5, ER dense (p=0.12)

### Files Modified (v3.2.0)

- `src/mem4ristor/core.py` — V5 Hysteresis (4 patches), Sparse CSR (8 patches), coupling_norm parameter
- `tests/test_v5_hysteresis.py` — 3 tests activated (were xfail)
- `examples/demo_applied.py` — New
- `experiments/limit02_norm_sweep.py` — New (degree normalization sweep)
- `experiments/limit02_topology_sweep.py` — New (13-topology validation)
- `experiments/entropy_sweep/` — New (LIMIT-02/04/05 investigation scripts)
- `docs/limitations.md` — Updated with all investigation results
- `docs/preprint.tex` — Claims corrected (H values, degree normalization)
- `README.md` — Rewritten for v3.2.0
- `VERSION` — v3.2.0
- `pyproject.toml` — version 3.2.0
- `PROJECT_STATUS.md` — Complete session logs

### Cleanup (v3.2.0)

- Removed `test_error.txt`, `test_output.txt` (orphan debug files)
- Removed `tests/test_philosopher_king.py.broken` (documented in `failures/philosopher_king_removal.log`)

---

## v2.9.3 → v3.0.0

## Migration Summary

**Date**: 2026-02-16
**Auditor**: Claude Opus 4.6 (Automated Scientific Audit)
**Orchestrator**: Julien Chauvin
**Reason**: Scientific rigor audit identified overclaims, test weaknesses, and inconsistencies.

---

## Philosophy of Changes

This migration follows a single principle: **claims must match evidence**.
Every modification below either (a) corrects a factual inconsistency, (b) strengthens a weak test, or (c) recalibrates language to match what the code actually proves.

---

## Structural Changes

### 1. Core Engine: V3 Levitating Sigmoid becomes the canonical implementation
- **Before**: `core.py` contained `Mem4ristorV2` (linear kernel `1-2u`), `mem4ristor_v3.py` contained `Mem4ristorV3` (Levitating Sigmoid `tanh(π(0.5-u)) + δ` + Inhibition Plasticity)
- **After**: `core.py` contains `Mem4ristorV3` as the main class, incorporating all V2 security guards + V3 innovations
- **Rationale**: V3's smooth sigmoid eliminates the dead zone at u=0.5 where V2's linear kernel made coupling noise-dominated (LIMIT-01). Plasticity adds structural memory of dissidence.

### 2. Mem4ristorKing moved to experimental/
- **Before**: `src/mem4ristor/mem4ristor_king.py`
- **After**: `experimental/mem4ristor_king.py` with WARNING header
- **Rationale**: King has design issues (temporary state mutation, missing coupling_input passthrough) that make it unsuitable for production code. Kept as documented exploration.

### 3. Old mem4ristor_v3.py removed
- **Rationale**: Its code is now merged into `core.py`. Keeping it would create confusion.

---

## Claim Corrections (Preprint & Documentation)

| Original Claim | Correction | Justification |
|:---|:---|:---|
| "Law of 15%" | "Empirical Threshold of 15%" | Fails on scale-free networks (own LIMIT-02) |
| "topologically invariant" | "validated on regular lattices (2D grid, Small-World, Random)" | Scale-free counterexample exists |
| "CCC Validation" | "CCC Illustration" | No raw CCC data used; parameters hand-tuned to match known outcomes |
| Δt = 0.1 (preprint Limitations) | Δt = 0.05 (consistent everywhere) | Code default is 0.05; 0.1 was stale reference |
| Δt = 0.01 recommended (preprint) | Δt = 0.05 standard, 0.01 high-precision | Harmonized across code, config, and docs |

---

## Test Suite Hardening

| Test | Issue | Fix |
|:---|:---|:---|
| `test_nan_injection_survial` | Accepted `nan_count <= 1` | Changed to `nan_count == 0` |
| `test_manus_v2.py` (5 tests) | Used `print()` instead of `assert` | Converted to proper assertions |
| `test_snr_validity` | Didn't calculate SNR | Renamed and rewritten with actual SNR computation |
| `test_spatial_clustering` | Arbitrary threshold (<5) | Threshold justified by block size math |
| `test_two_unit_network_symmetry` | Tolerance of 1.0 | Reduced to 0.3 (reasonable for noise) |
| `test_float32_vs_float64` | Didn't test float32 ops | Removed (was testing nothing) |
| `test_fuzzing` | 50 iterations | Increased to 200, added input tracking |

---

## Benchmark Corrections

### benchmark_kuramoto.py
- **Before**: Mem4ristor on 2D grid vs Kuramoto all-to-all (topology mismatch)
- **After**: Both models use same adjacency matrix for fair comparison
- **Before**: Voltage→phase mapping via linear rescaling (physically invalid for FHN)
- **After**: Comparison uses entropy and variance directly (topology-agnostic metrics)

---

## Files Modified

- `src/mem4ristor/core.py` — Rewritten: V3 engine + V2 guards
- `src/mem4ristor/__init__.py` — Updated exports
- `src/mem4ristor/config.yaml` — Added V3 parameters
- `src/mem4ristor/mem4ristor_v3.py` — Removed (merged into core)
- `src/mem4ristor/mem4ristor_king.py` — Moved to experimental/
- `experimental/mem4ristor_king.py` — Added with WARNING
- `tests/test_kernel.py` — Hardened
- `tests/test_robustness.py` — Hardened
- `tests/test_fuzzing.py` — Hardened
- `tests/test_manus_v2.py` — Converted to proper assertions
- `experiments/benchmark_kuramoto.py` — Corrected methodology
- `docs/preprint.tex` — Claims recalibrated
- `README.md` — Updated for V3
- `pyproject.toml` — Version bump
- `VERSION` — 3.0.0
- `CITATION.cff` — Updated
- `CAFE-VIRTUEL-LIMITATIONS.md` — Updated status table
