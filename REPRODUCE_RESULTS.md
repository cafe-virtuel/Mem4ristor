# Reproducibility Guide: Mem4ristor Project

This document describes how to reproduce all results presented in the
accompanying scientific paper (`docs/papers/preprint/preprint.tex`).

> Mis à jour le 2026-10-19 : alignement intégral des chemins, scripts et
> tableaux de valeurs sur le preprint V6.0.0 (`docs/papers/preprint/preprint.tex`)
> et sur les garde-fous automatisés (`tools/preprint_guardian.py`, `tools/tex_guardian.py`).

---

## 1. Quick Start (5 Minutes)

```bash
git clone https://github.com/cafe-virtuel/Mem4ristor.git
cd Mem4ristor
pip install -e .
python tools/preprint_guardian.py
python tools/tex_guardian.py
```

Or run a quick 10x10 lattice simulation under the canonical Cold Start Protocol:

```bash
python -c "
from mem4ristor.core import Mem4Network
net = Mem4Network(size=10, heretic_ratio=0.15, seed=42, cold_start=True)
for _ in range(3000):
    net.step(I_stimulus=0.5)
print('Entropy:', net.calculate_entropy(bins=100))
"
```

Expected output across 10 seeds (`experiments/verify_table1_preprint.py`, claim `C01`):
`H_stable = 4.09 +/- 0.19 bits` (sustained attractor entropy on 10x10 lattice).

---

## 2. Environment Setup

### Requirements

- Python 3.10+
- NumPy, SciPy, NetworkX, PyYAML, Pandas, Matplotlib, tqdm
- NGSpice 46+ (optional, for SPICE hardware validation)
- LaTeX / `latexmk` (optional, for paper compilation)

```bash
pip install -r requirements.txt
pip install -e .
```

### Verification

```bash
python -m pytest
python tools/preprint_guardian.py
python tools/tex_guardian.py
```

All tests and all 22 automated claim checks should pass with exit code 0.

---

## 3. Directory Structure

```
Mem4ristor/
  src/mem4ristor/
    core.py            -- Mem4Network & Mem4ristorV3 compatibility facade
    dynamics.py        -- FHN + constitutional doubt ODE implementation
    topology.py        -- Graph coupling & sparse CSR Laplacian
    metrics.py         -- Entropy, synchrony, LZ76 complexity, MI
    graph_utils.py     -- BA, lattice, ER graph generators
    config.yaml        -- Default parameters
    mem4ristor_v26.va  -- Verilog-A hardware model
  spice/
    mem4ristor_coupled_3x3.cir  -- 3x3 SPICE netlist
  experiments/
    verify_table1_preprint.py   -- Table 1 & Table 7 lattice benchmarks (C01-C03, C07b)
    ablation_coordination.py    -- Table 3 coordination ablations (C14-C17)
    b4_ablation_robustness.py   -- 30-seed FULL vs FROZEN_U separation (C18)
    limit02_*.py                -- Table 4 & Table 5 BA regime / gamma sweeps
    fss_lz_sweep.py             -- Table 6 adaptive D / LZ regime sweep (C19, C22)
    spice_*.py                  -- SPICE validation scripts (C11, C20, C21)
  tools/
    preprint_guardian.py        -- Automated CSV-to-claim verifier (22/22 claims)
    tex_guardian.py             -- Automated LaTeX-to-CSV anchor & dead-value verifier
  tests/
    test_scientific_regression.py
    test_robustness.py
    test_kernel.py
    ...
  docs/papers/preprint/preprint.tex -- Main preprint manuscript
  REPRODUCE_RESULTS.md              -- This file
```

---

## 4. Core Results Reproduction

### 4.1 Sustained Attractor Diversity on Lattice (Table 1 / Claims C01–C03, C07b)

```bash
python experiments/verify_table1_preprint.py
```

Produces: `figures/p2_table1_lattice.csv` and `figures/p2_table1_sync.csv`
(Cold Start Protocol `v = w = 0`, `I_stimulus = 0.5`, 10 seeds, 3000 steps, last 25% window):

| Topology       | H_stable (bits) | Pairwise Synchrony | Claim      |
|----------------|-----------------|--------------------|------------|
| Lattice 4x4    | 3.21 +/- 0.14   | 0.088 +/- 0.056    | C02        |
| Lattice 10x10  | 4.09 +/- 0.19   | 0.036 +/- 0.013    | C01, C07b  |
| Lattice 25x25  | 4.40 +/- 0.08   | 0.014 +/- 0.004    | C03        |

### 4.2 Component Ablation (Table 3 / Claims C04, C14–C18)

```bash
python experiments/ablation_coordination.py
python experiments/p2_sigma_social_ablation.py
python experiments/b4_ablation_robustness.py
```

Produces: `figures/ablation_coordination_summary.csv`, `figures/p2_sigma_social_ablation.csv`,
and `figures/b4_ablation_summary.csv`.

Key metrics from paper Table 3 (`tab:ablations`, BA `m=3`, `N=100`, `degree_linear`, `I_stim=0.5`, 10 seeds, 3000 steps):

| Configuration         | Synchrony       | LZ Complexity   | Claims   |
|-----------------------|-----------------|-----------------|----------|
| Full model            | 0.002 +/- 0.009 | 1.096 +/- 0.022 | C14, C16 |
| No heretics           | 0.034 +/- 0.023 | 1.154 +/- 0.026 | —        |
| No Levitating Sigmoid | 0.003 +/- 0.008 | 1.069 +/- 0.030 | —        |
| Frozen u              | 0.697 +/- 0.085 | 1.602 +/- 0.005 | C15, C17 |

On the 5000-step canonical seed harness (`experiments/p2_sigma_social_ablation.py`, claim `C04`),
pairwise synchrony moves from `0.007` (FULL) to `0.658` (FROZEN_U), with complete separation
across 30 seeds (`Cohen's d = 9.43` on BA `m=3`, claim `C18`).

### 4.3 Barabasi-Albert Coupling-Degree Regime Map (Table 4 / Claim C05)

```bash
python experiments/limit02_Degree_Linear_Dead_Zone.py
python experiments/lambda2_crit_regression.py
```

Produces: `figures/limit02_regime_map_functional.csv` and `figures/lambda2_crit_regression.csv`.

Key findings from paper Table 4 (`tab:ba_m_sweep`, Cold Start Protocol, `I_stimulus=0`, 10 seeds):

| m  | lambda_2 | H_cog (uni) | H_cog (deg_linear) | H_cont (deg_linear) | Sync (deg_linear) | Regime               |
|----|----------|-------------|--------------------|---------------------|-------------------|----------------------|
| 1  | 0.02     | 0.67        | 0.80               | 3.09                | -0.00             | both functional      |
| 2  | 0.63     | 0.27        | 0.71               | 3.14                | 0.00              | degree-linear        |
| 3  | 1.40     | 0.01        | 0.68               | 3.14                | 0.01              | degree-linear        |
| 4  | 2.23     | 0.00        | 0.62               | 3.20                | 0.03              | degree-linear        |
| 5  | 3.08     | 0.00        | 0.32               | 3.33                | 0.13              | degree-linear (weak) |
| 6  | 4.01     | 0.00        | 0.03               | 2.85                | 0.10              | dead-zone onset      |
| 7  | 4.91     | 0.00        | 0.00               | 2.60                | -0.00             | dead zone            |

Note: The regime boundary is governed by the harmonic coupling degree (`k_harm ~ 6`, mean-field
averaging of heretic variance); the separation at `lambda_2 ~ 2.31` on the initial BA/ER dataset
(`C05`) is a correlational proxy because `lambda_2` co-varies monotonically with `m` on BA graphs.

### 4.4 Power-Law Exponent & Adaptive Coupling Sweeps (Tables 5 & 6 / Claims C19, C22)

```bash
python experiments/p2_delta_sweep.py        # delta sweep
python experiments/p2_tau_u_bifurcation.py  # tau_u sweep
python experiments/fss_lz_sweep.py          # adaptive D / LZ regime sweep (default --nseeds 5)
```

---

## 5. SPICE Hardware Validation

### 5.1 Prerequisites

Install NGSpice 46+:
- Windows: http://ngspice.sourceforge.net/spice-46.html
- Linux: `apt install ngspice` or build from source
- macOS: `brew install ngspice`

### 5.2 Run 3x3 Network

```bash
ngspice -b spice/mem4ristor_coupled_3x3.cir
```

Output: terminal voltage traces for all 9 nodes.

### 5.3 Python-to-SPICE Validation (Claim C20)

```bash
python experiments/spice_validation.py
```

Generates: `figures/spice_vs_python_validation.png` and `figures/spice_validation_summary.csv`.

Expected: global RMS error `0.009689` (`~9.7e-3`, `< 1%` of typical `|v|`, claim `C20`).

### 5.4 50-Seed Monte Carlo Campaign (Claim C21)

```bash
python experiments/spice_mismatch_50seeds.py
```

Generates: `figures/spice_50seeds_validation.png` and `figures/spice_50seeds_summary.csv`.

Expected (from paper Table 8, `tab:spice_mc`, claim `C21`):

| Condition              | H_cont          |
|------------------------|-----------------|
| Dead zone (m=5)        | 1.38 +/- 0.12 bits |
| Functional (noise)     | 4.30 +/- 0.08 bits |
| Functional (mismatch)  | 4.33 +/- 0.04 bits |

---

## 6. Paper Compilation

### 6.1 Requirements

```bash
pip install latexmk # or: apt install texlive-latex-extra
```

### 6.2 Compile

```bash
cd docs/papers/preprint
latexmk -pdf preprint.tex
```

Or manually:

```bash
cd docs/papers/preprint
pdflatex preprint.tex
bibtex preprint
pdflatex preprint.tex
pdflatex preprint.tex
```

### 6.3 Expected Output

`docs/papers/preprint/preprint.pdf` -- Full 28-page paper with all sections, tables, and
references.

---

## 7. Parameter Reference

All default parameters (from `src/mem4ristor/config.yaml`):

| Group      | Parameter      | Value |
|------------|----------------|-------|
| Dynamics   | a              | 0.7   |
|            | b              | 0.8   |
|            | epsilon        | 0.08  |
|            | alpha          | 0.15  |
|            | dt             | 0.05  |
| Coupling   | D              | 0.15  |
|            | delta          | 0.01  |
|            | heretic_ratio  | 0.15  |
| Doubt      | epsilon_u      | 0.02  |
|            | tau_u          | 10.0  |
|            | k_u            | 1.0   |
|            | sigma_baseline | 0.05  |
| Meta-doubt | alpha_surprise | 2.0   |
|            | surprise_cap   | 5.0   |
| Noise      | sigma_v        | 0.05  |

---

## 8. Known Limitations

See `docs/CLAIMS_REGISTER.md` and `docs/BILAN_FORCES_FAIBLESSES.md` for the complete,
up-to-date register of verified claims, limitations, and negative results.

1. **Entropy calibration:** Python simulations at default parameters under zero drive
   produce voltages predominantly in `[-3.2, -1.3]`, placing nodes in `bin1` of the legacy
   5-state cognitive entropy (`H_cog`). Continuous entropy (`H_stable`, 100 bins) and
   binning-independent pairwise Pearson synchrony (`r_bar`) are used as the primary metrics.

2. **Coupling-degree dead zone vs. spectral gap:** Degree-controlled experiments on static graphs
   (`experiments/fable_audit_20261111/`, scripts `d2` and `d5`; the earlier runs in
   `experiments/lambda2_foundation_20260701/` had the model's rewiring active and did not
   control `lambda_2`) show that the dead zone on dense graphs is
   caused by mean-field averaging above harmonic degree `k_harm ~ 6`, not by the algebraic
   connectivity `lambda_2` itself.

3. **Integration scheme:** Default results use Euler-Maruyama (`dt = 0.05` with `sqrt(dt)`
   stochastic scaling); RK4 (`use_rk4=True`) is available and tested for trajectory consistency.

---

## 9. Citation

If you use these results, please cite:

```bibtex
@software{mem4ristor_zenodo,
  author       = {Julien Chauvin},
  title        = {Sustained Attractor Diversity in Doubt-Modulated FitzHugh-Nagumo Networks:
                  Spectral Dead Zone and Chimera States on Scale-Free Topologies},
  year         = {2026},
  publisher    = {Zenodo},
  version      = {V4.0.0},
  doi          = {10.5281/zenodo.18620596},
  url          = {https://doi.org/10.5281/zenodo.18620596},
  note         = {Concept DOI; resolves to the latest deposit (V4.0.0, DOI 10.5281/zenodo.19986042).
                  The code in this repository is V6.0.0.}
}
```

---

*Last updated: 2026-10-19*
