# Mem4ristor V3: Neuromorphic Cognitive Architecture

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18620596.svg)](https://doi.org/10.5281/zenodo.18620596)
[![Tests](https://github.com/cafe-virtuel/mem4ristor-v2/actions/workflows/test.yml/badge.svg)](https://github.com/cafe-virtuel/mem4ristor-v2/actions/workflows/test.yml)

**Mem4ristor V3** is a computational implementation of extended FitzHugh-Nagumo dynamics designed to investigate emergent critical states in neuromorphic networks. This research code focuses on the role of "Constitutional Doubt" ($u$) and "Structural Heretics" in preventing consensus collapse in scale-free and lattice networks.

> **Status**: v3.2.1 (Post-Audit — corrected doubt equation, n=5 statistics, Paper 2 draft)

## 🔬 Key Scientific Features

*   **Constitutional Doubt ($u$):** A dynamic state variable that modulates coupling polarity based on local uncertainty, enabling repulsive social coupling when doubt is high.
*   **Structural Heretics:** A subset of nodes with inverted stimulus perception, critical for maintaining global diversity (Empirical Threshold of 15%, validated on regular lattices).
*   **Levitating Sigmoid Coupling:** Smooth repulsive coupling via $\tanh(\pi(0.5-u)) + \delta$, eliminating the dead zone at $u=0.5$.
*   **Adaptive Meta-Doubt (v3.1.0):** Doubt learning rate accelerates proportionally to social surprise, providing social-driven meta-plasticity.
*   **Degree-Normalized Coupling (v3.2.0):** Per-node coupling $D/\deg(i)$ breaks hub strangulation on scale-free networks, recovering 86% of lattice-level diversity on Barabási-Albert networks.
*   **V5 Hysteresis (v3.2.0):** Dead-zone latching with Schmitt-trigger-style transitions and watchdog fatigue for memristor-realistic dynamics.
*   **Sparse CSR Backend (v3.2.0):** Automatic scipy.sparse conversion for $N > 1000$, yielding 455× memory reduction at $N=5000$.
*   **Scale-Invariant Dynamics:** Normalized coupling strength ($D_{eff} = D/\sqrt{N}$) ensures consistent behavior across network sizes ($N=10$ to $N=2500$).

## 🚀 Installation

```bash
git clone https://github.com/cafe-virtuel/mem4ristor-v2.git
cd mem4ristor-v2
pip install -e .
```

*Note: The `-e` flag installs in editable mode, allowing you to modify source code without reinstalling.*

## 💻 Usage

### Quick Start (Python API)

```python
from mem4ristor.core import Mem4Network

# Initialize a lattice network (N=100, 15% Heretics)
net = Mem4Network(size=10, heretic_ratio=0.15, seed=42)

# Run simulation for 1000 steps (Cold Start: no external stimulus)
for step in range(1000):
    net.step(I_stimulus=0.0)

# Calculate final entropy (measure of diversity)
print(f"Final System Entropy: {net.calculate_entropy():.4f}")
# See docs/limitations.md for expected values and binning notes
```

### Scale-Free Networks (v3.2.0)

```python
import networkx as nx
import numpy as np

# Create a Barabási-Albert scale-free network
G = nx.barabasi_albert_graph(100, 3, seed=42)
adj = nx.to_numpy_array(G)

# Use degree-linear normalization to prevent hub strangulation
net = Mem4Network(adjacency_matrix=adj, heretic_ratio=0.15,
                  coupling_norm='degree_linear', seed=42)

for step in range(3000):
    net.step(I_stimulus=0.0)

print(f"Scale-Free Entropy: {net.calculate_entropy():.4f}")
# Expected: H_stable ≈ 0.83 ± 0.07 (86% recovery of lattice performance)
```

### Running Demos

```bash
# Full demo: sensory pipeline, hysteresis comparison, scale-free sparse, phase diversity
python examples/demo_applied.py
# Produces 5 PNG files
```

## ⚙️ Configuration

The model is highly configurable via `src/mem4ristor/config.yaml`. You can adjust:

*   **Dynamics:** `a`, `b`, `epsilon` (FHN parameters)
*   **Coupling:** `D` (Strength), `heretic_ratio`, `coupling_norm` (`uniform`, `degree_linear`)
*   **Doubt:** `epsilon_u`, `u_clamp`, `alpha_surprise` (meta-doubt gain)
*   **Hysteresis (v3.2.0):** `enabled`, `theta_low`, `theta_high`, `fatigue_rate`
*   **Noise:** `sigma_v`

## 🧪 Testing

The repository includes a comprehensive test suite using `pytest`.

```bash
# Run all tests
pytest

# Run only robustness tests
pytest tests/test_robustness.py

# Run scientific regression tests (v3.2.0)
pytest tests/test_scientific_regression.py
```

## 📂 Repository Structure

### Core source — `src/mem4ristor/`

| File | Role |
|---|---|
| `dynamics.py` | **Heart of the model** — FHN + doubt ODE, `epsilon_u_adaptive`, Levitating Sigmoid, heretic polarity, sparse CSR backend |
| `core.py` | High-level `Mem4Network` API facade — wraps `dynamics.py`, exposes `step()`, `sigma_social_override` hook |
| `config.py` + `config.yaml` | Default parameters (FHN a/b/ε, coupling D, doubt τ_u/ε_u/α_surprise, hysteresis thresholds) |
| `topology.py` | Graph utilities — Barabási-Albert, lattice, ER generators; Laplacian; degree normalization |
| `metrics.py` | Shannon entropy H_cog (5-bin), continuous H_cont (100-bin), pairwise synchrony, spatial mutual information |
| `sensory.py` | Sensory input frontend (visual/auditory preprocessing pipeline) |
| `arena.py` | Multi-network arena for population-level experiments |
| `cortex.py` | Cortex-level abstraction (hierarchical coupling) |
| `graph_utils.py` | NetworkX helpers, community detection, NMI computation |
| `hierarchy.py` | Hierarchical topology builder |
| `inception.py` | Cold-start protocol implementation |
| `symbiosis.py` | Experimental symbiotic coupling between networks |
| `viz.py` | Visualization helpers (phase portraits, entropy traces) |
| `mem4ristor_v26.va` | Verilog-A hardware model (synchronized with `dynamics.py` v3.2) |
| `benchmarks/engine.py` | Benchmark harness for throughput and reproducibility |

### Tests — `tests/`

| File | What it validates |
|---|---|
| `test_scientific_regression.py` | Core scientific claims — H_cog, sync, topology phase transition |
| `test_sigma_social_override.py` | `sigma_social_override` hook correctness (ablation API) |
| `test_robustness.py` | Noise tolerance, large N (N=1200), integration stability |
| `test_adversarial.py` | Adversarial parameter inputs, edge cases |
| `test_kernel.py` | Levitating Sigmoid kernel, polarity switching |
| `test_v5_hysteresis.py` | Schmitt-trigger hysteresis and fatigue mechanism |
| `test_coordination_metrics.py` | Pairwise synchrony, mutual information metrics |
| `test_manus_v2.py` | Manus v2 compatibility regression |
| `test_fuzzing.py` | Random parameter fuzzing (stability boundary) |
| `test_v4_extensions.py` | V4 extension hooks (reserved for dynamic heretics) |
| `exploits/` | Adversarial stress tests — topology attacks, stiffness, chaos |

### Experiments — `experiments/`

**Paper 2 experiments** (all produce CSV + PNG in `figures/`):

| Script | Produces | Description |
|---|---|---|
| `p2_sigma_social_ablation.py` | `p2_sigma_social_ablation.*` | FULL / SS_NOISE / SS_STATIC / FROZEN_U ablation (n=5 seeds) |
| `p2_spatial_mutual_information.py` | `p2_spatial_mutual_information.*` | I(v_i; v_j) vs hop distance, 4 conditions |
| `p2_tau_u_bifurcation.py` | `p2_tau_u_bifurcation.*` | τ_u sweep, forced regime (n=5 seeds) |
| `p2_tau_u_bifurcation_endogenous.py` | `p2_tau_u_bifurcation_endogenous.*` | τ_u sweep, I_stim=0 |
| `p2_stochastic_resonance_topology.py` | `p2_stochastic_resonance_topology.*` | SR curve by topology (n=3 — TODO n=5) |
| `p2_finite_size_scaling.py` | `p2_finite_size_scaling.*` | N=25→1600 scaling (n=3 — TODO n=5) |
| `p2_edge_betweenness_analysis.py` | `p2_edge_betweenness.*` | Edge betweenness vs diversity |
| `p2_delta_sweep.py` | `p2_delta_sweep.*` | δ offset sweep (Levitating Sigmoid) |
| `p2_directed_coupling.py` | `p2_directed_coupling.*` | Asymmetric / directed topology experiment |
| `p2_stochastic_resonance_directed.py` | `p2_stochastic_resonance_directed.*` | SR on directed graphs |
| `p2_doubt_community_detection.py` | `p2_doubt_community_detection.*` | NMI between structural and u-functional partitions |

**SPICE validation** (Paper B):

| Script | Description |
|---|---|
| `spice_mismatch_50seeds.py` | 50-seed MC campaign, BA m=5, Cohen's d=20.78 |
| `spice_noise_calibration.py` | Derives η=0.5 ↔ σ_equiv=0.0044 (270× gap) |
| `spice_mismatch_sweep.py` | RMS error sweep vs σ_noise |
| `spice_validation.py` | SPICE vs Python side-by-side on 4×4 lattice |
| `spice_p420_hfo2_memristor.py` | HfO₂ memristor SPICE characterization |

### Scientific documents — `docs/`

| File | Description |
|---|---|
| `preprint.tex` / `preprint.pdf` | **Paper 1** — "Frustrated Synchronization in Doubt-Modulated FHN Networks" (14 pages, v3.2.1 corrected) |
| `paper_B/paper_B.tex` / `.pdf` | **Paper B** — SPICE validation + topological dead zone (hardware bridge) |
| `paper_2/paper_2.tex` / `.pdf` | **Paper 2** — "The Doubt Variable as Anti-Synchronization Filter" (7 pages, draft) |
| `limitations.md` | Scientific truth table — confirmed claims, known limits, reconciliation of binning artifacts |
| `academic_history.md` | Full chronological history of the project |
| `theoretical_anchoring.md` | Mathematical foundations and literature connections |

### Project-level files

| File | Description |
|---|---|
| `PROJECT_STATUS.md` | **Living document** — current state, all experimental results, open questions |
| `CHANGELOG_V3.md` | Version history with scientific changes per release |
| `HACKER_GUIDE.md` | Deep dive into the codebase for contributors |
| `CITATION.cff` | Machine-readable citation metadata |
| `ai_contributors.json` | AI collaboration log (Kimi, Edison, Manus, Claude) |
| `docs/limitations.md` | Truth table of scientific claims (confirmed / refuted / conditional) |

### Figures — `figures/`

All figures are auto-generated by their corresponding experiment script. Key files:

| Figure | Generated by |
|---|---|
| `p2_sigma_social_ablation.png` | `experiments/p2_sigma_social_ablation.py` |
| `p2_tau_u_bifurcation.png` | `experiments/p2_tau_u_bifurcation.py` |
| `fiedler_phase_diagram.png` | `experiments/fiedler_phase_diagram.py` |
| `spice_50seeds_validation.png` | `experiments/spice_mismatch_50seeds.py` |
| `p2_doubt_community_detection.png` | `experiments/p2_doubt_community_detection.py` |

## 📜 Citation

If you use this code in your research, please cite the associated dataset/preprint:

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

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
