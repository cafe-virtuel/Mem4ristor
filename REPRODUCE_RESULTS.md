# Reproducibility Guide: Mem4ristor Project

This document describes how to reproduce all results presented in the
accompanying scientific paper (WORK_LOG_PAPER.tex).

---

## 1. Quick Start (5 Minutes)

```bash
git clone https://github.com/cafe-virtuel/mem4ristor-v2.git
cd mem4ristor-v2
pip install -e .
python -c "
from mem4ristor.core import Mem4Network
net = Mem4Network(size=10, heretic_ratio=0.15, seed=42)
for _ in range(1000):
    net.step(I_stimulus=0.0)
print('Entropy:', net.calculate_entropy())
"
```

Expected output: `Entropy: ~3.79 bits` (sustained attractor entropy on
10x10 lattice).

---

## 2. Environment Setup

### Requirements

- Python 3.9+
- NumPy, SciPy, NetworkX, PyYAML, Matplotlib
- NGSpice 46+ (for SPICE hardware validation)
- LaTeX (for paper compilation)

```bash
pip install numpy scipy networkx pyyaml matplotlib pytest
```

### Verification

```bash
pytest tests/test_scientific_regression.py -v
```

All tests should pass with exit code 0.

---

## 3. Directory Structure

```
mem4ristor-v2/
  src/mem4ristor/
 core.py          -- Mem4Network high-level API
    dynamics.py       -- FHN + doubt ODE implementation
    metrics.py        -- Entropy, synchrony, LZ complexity
    topology.py       -- BA, lattice, ER graph generators
    config.yaml       -- Default parameters
    mem4ristor_v26.va -- Verilog-A hardware model
  spice/
    mem4ristor_coupled_3x3.cir  -- 3x3 SPICE netlist
  experiments/
    p2_*.py           -- Paper2 experiment scripts
    spice_*.py        -- SPICE validation scripts
  tests/
    test_scientific_regression.py
    test_robustness.py
    test_kernel.py
    ...
 WORK_LOG_PAPER.tex -- This paper
  REPRODUCE_RESULTS.md -- This file
```

---

## 4. Core Results Reproduction

### 4.1 Sustained Attractor Diversity on Lattice

```python
from mem4ristor.core import Mem4Network
import numpy as np

# 10x10 lattice, Cold Start,7 seeds
Hs = []
for seed in range(7):
    net = Mem4Network(size=10, heretic_ratio=0.15, seed=seed)
    for _ in range(3000):
        net.step(I_stimulus=0.0)
    # Last 25% of steps
    H = net.calculate_entropy()
    Hs.append(H)

print(f"H_stable = {np.mean(Hs):.2f} +/- {np.std(Hs):.2f} bits")
# Expected: 3.79 +/- 0.14 bits
```

### 4.2 Component Ablation

```bash
python experiments/p2_sigma_social_ablation.py
```

Produces: `figures/p2_sigma_social_ablation.png`

Key metrics (from paper Table 3):

| Configuration      | Synchrony   | LZ Complexity |
|-------------------|-------------|---------------|
| Full model        | 0.031       | 1.069         |
| No heretics       | 0.069       | 1.134         |
| No sigmoid | 0.039       | 1.073         |
| Frozen u          | 0.751       | 1.635         |

### 4.3 Barabasi-Albert Phase Transition

```bash
python experiments/p2_fiedler_phase_diagram.py
```

Produces: `figures/fiedler_phase_diagram.png`

Key findings (from paper Table 4):

| m  | H_uniform | H_deg_linear | lambda_2 |
|----|-----------|--------------|----------|
| 3  | 0.00      | 0.83         | 1.41     |
| 4  | 0.00      | 0.23         | 2.21     |
| 5  | 0.02      | 0.00         | 2.99     |

### 4.4 Power-Law Exponent Sweep

```bash
python experiments/p2_delta_sweep.py # delta sweep
python experiments/p2_tau_u_bifurcation.py  # tau_u sweep
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

### 5.3 Python-to-SPICE Validation

```bash
python experiments/spice/spice_validation.py
```

Generates: `figures/spice_vs_python_validation.png`

Expected: global RMS error ~9.7e-3 (< 1% of typical |v|).

### 5.450-Seed Monte Carlo Campaign

```bash
python experiments/spice/spice_mismatch_50seeds.py
```

Generates: `figures/spice_50seeds_validation.png`

Expected (from paper Table 6):

| Condition              | H_cont |
|------------------------|-----------|
| Dead zone (m=5)        | 1.38 bits |
| Functional (noise)     | 4.30 bits |
| Functional (mismatch)  | 4.33 bits |

---

## 6. Paper Compilation

###6.1 Requirements

```bash
pip install latexmk # or: apt install texlive-latex-extra
```

### 6.2 Compile

```bash
cd mem4ristor-v2
latexmk -pdf WORK_LOG_PAPER.tex
```

Or manually:

```bash
pdflatex WORK_LOG_PAPER.tex
bibtex WORK_LOG_PAPER
pdflatex WORK_LOG_PAPER.tex
pdflatex WORK_LOG_PAPER.tex
```

### 6.3 Expected Output

`WORK_LOG_PAPER.pdf` -- Full paper with all sections, tables, and
references.

---

## 7. Parameter Reference

All default parameters (from `src/mem4ristor/config.yaml`):

| Group | Parameter         | Value   |
|-------------|-------------------|---------|
| Dynamics | a                 | 0.7 |
|             | b                 | 0.8     |
|             | epsilon           | 0.08    |
|             | alpha_self        | 0.15    |
|             | dt                | 0.05    |
| Coupling    | D                 | 0.15    |
|             | delta             | 0.01    |
|             | heretic_ratio     | 0.15    |
| Doubt       | epsilon_u         | 0.02    |
|             | tau_u             | 10.0    |
|             | k_u               | 1.0     |
|             | sigma_baseline    | 0.05    |
| Meta-doubt  | alpha_surprise    | 2.0     |
|             | surprise_cap      | 5.0     |
| Noise       | sigma_v           | 0.05    |

---

## 8. Known Limitations

1. **Entropy calibration:** Python simulations at default parameters
   produce voltages predominantly in [-3.2, -1.3], placing all nodes in
   bin1 of the 5-state cognitive entropy (H_cog ~ 0). This is a
   calibration artifact, not a diversity collapse. Use continuous
   entropy (H_stable, 100 bins) as the primary metric.

2. **Finite-size effects:** The phase transition at m ~ 5 was observed
   on N = 100 BA networks. Finite-size effects may shift this boundary
   at larger N.

3. **Euler integration:** All results use first-order Euler integration
   with dt = 0.05. Quantitative values may shift under higher-order
   schemes (RK4).

---

## 9. Citation

If you use these results, please cite:

```bibtex
@software{mem4ristor_v3,
  author       = {Julien Chauvin},
  title        = {Mem4ristor V3: Neuromorphic Cognitive Architecture},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18620596},
  url          = {https://doi.org/10.5281/zenodo.18620596}
}
```

---

*Last updated: 2026-05-30*
