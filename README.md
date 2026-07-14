# Mem4ristor V6: Spatiotemporal Chaos & Chimera States

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19700749.svg)](https://doi.org/10.5281/zenodo.19700749)
[![Tests](https://github.com/cafe-virtuel/Mem4ristor/actions/workflows/test.yml/badge.svg)](https://github.com/cafe-virtuel/Mem4ristor/actions/workflows/test.yml)

**Mem4ristor V6** is a computational implementation of extended FitzHugh-Nagumo dynamics designed to investigate emergent critical states in neuromorphic networks. The model uses "Constitutional Doubt" ($u$) and "Structural Heretics" to maintain phase diversity in scale-free networks, generating robust Chimera states without requiring stochastic noise.

> **Status**: V6.0.0 — arXiv Ready. See `PROJECT_STATUS.md` for the full scientific claims register and `docs/limitations.md` for the honest truth table of what has and has not been proven.

---
:fire: **[CLICK HERE TO REPRODUCE THE CHIMERA STATE IN 5 MINUTES](REPRODUCE_IN_5_MINUTES.md)** :fire:

---

## :microscope: Key Scientific Features

*   **Constitutional Doubt ($u$):** A dynamic state variable that modulates coupling polarity based on local uncertainty, enabling repulsive social coupling when doubt is high.
*   **Structural Heretics:** A subset of nodes with inverted stimulus perception. These act as causal structural walls that prevent consensus collapse.
*   **Levitating Sigmoid Coupling:** Smooth repulsive coupling via $\tanh(\pi(0.5-u)) + \delta$, eliminating the dead zone at $u=0.5$.
*   **Degree-Normalized Coupling:** Per-node coupling $D/\deg(i)$ prevents hub strangulation on Barabási-Albert networks.
*   **Robust Chimera States:** Phase-locked macroscopic state coexisting with spatial diversity, validated across homogeneous and random initial conditions.
*   **Sparse CSR Backend:** Automatic `scipy.sparse` conversion for $N > 1000$, yielding massive memory reductions.
*   **ART (Adaptive Reset Threshold):** Per-node reset threshold modulated by doubt history, preventing fatigue collapse.
*   **Metacognitive Plasticity:** Per-node learning rate modulated by local doubt, enabling adaptive memory consolidation.
*   **Compartimentalised Dynamics:** Multiple sub-personalities coexisting within the same network.
*   **Non-Local Coupling by Doubt Similarity:** Coupling between topologically distant nodes when their doubt states are correlated.

## :rocket: Installation

```bash
git clone https://github.com/cafe-virtuel/Mem4ristor.git
cd Mem4ristor
pip install -e .
```

*Note: The `-e` flag installs in editable mode, allowing you to modify source code without reinstalling.*

## :computer: Usage

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

### Scale-Free Networks

```python
import networkx as nx

# Create a Barabási-Albert scale-free network
G = nx.barabasi_albert_graph(100, 3, seed=42)
adj = nx.to_numpy_array(G)

# Use degree-linear normalization to prevent hub strangulation
net = Mem4Network(adjacency_matrix=adj, heretic_ratio=0.15,
                  coupling_norm='degree_linear', seed=42)

for step in range(3000):
    net.step(I_stimulus=0.0)

print(f"Scale-Free Entropy: {net.calculate_entropy():.4f}")
# Expected: H_stable ~ 0.83 (86% recovery of lattice performance)
```

### Running Demos

```bash
# Chimera demo
python experiments/demo_chimera.py
# Produces demo_chimera_output.png

# Full applied demo (sensory pipeline, hysteresis, scale-free sparse, phase diversity)
python examples/demo_applied.py
# Produces 5 PNG files
```

## :gear: Configuration

The model is highly configurable via `src/mem4ristor/config.yaml`. You can adjust:

*   **Dynamics:** `a`, `b`, `epsilon` (FHN parameters)
*   **Coupling:** `D` (Strength), `heretic_ratio`, `coupling_norm` (`uniform`, `degree_linear`, `spectral`)
*   **Doubt:** `epsilon_u`, `u_clamp`, `alpha_surprise` (meta-doubt gain)
*   **Hysteresis (V5):** `enabled`, `theta_low`, `theta_high`, `fatigue_rate`
*   **ART (V6):** `art_enabled`, `art_threshold`, `art_window`
*   **Noise:** `sigma_v`

## :test_tube: Testing

The repository includes a comprehensive test suite using `pytest`.

```bash
# Run all tests
pytest

# Run only robustness tests
pytest tests/test_robustness.py

# Run scientific regression tests
pytest tests/test_scientific_regression.py
```

## :open_file_folder: Repository Structure

### Core source — `src/mem4ristor/`

| File | Role |
|---|---|
| `dynamics.py` | **Heart of the model** — FHN + doubt ODE, epsilon_u_adaptive, Levitating Sigmoid, heretic polarity, ART, metacognitive plasticity, sparse CSR backend |
| `core.py` | High-level `Mem4Network` API facade — wraps `dynamics.py`, exposes `step()`, `sigma_social_override` hook |
| `config.py` + `config.yaml` | Default parameters (FHN a/b/eps, coupling D, doubt tau_u/eps_u/alpha_surprise, hysteresis thresholds, ART config) |
| `topology.py` | Graph utilities — Barabási-Albert, lattice, ER generators; Laplacian; degree normalization |
| `metrics.py` | Shannon entropy H_cog (5-bin), continuous H_cont (100-bin), pairwise synchrony, spatial mutual information |
| `sensory.py` | Sensory input frontend (visual/auditory preprocessing pipeline) |
| `cortex.py` | Cortex-level abstraction (hierarchical coupling) |
| `graph_utils.py` | NetworkX helpers, community detection, NMI computation |
| `inception.py` | Cold-start protocol implementation |
| `symbiosis.py` | CreativeProjector (Phase 4) + SymbioticSwarm |
| `viz.py` | Visualization helpers (phase portraits, entropy traces) |
| `sonification.py` | Audio output from doubt dynamics |
| `benchmarks/engine.py` | Benchmark harness for throughput and reproducibility |

### Experimental modules — `experimental/`

| File | Role |
|---|---|
| `hierarchy.py` | Hierarchical chimera orchestration |
| `arena.py` | Gladiator-style network competition |
| `mem4ristor_king.py` | "Philosopher King" — martial law, metacognition |
| `demo_*.py` | Standalone demonstrations (sensory, hysteresis, arena, swarm, etc.) |

### Tests — `tests/`

| File | What it validates |
|---|---|
| `test_scientific_regression.py` | Core scientific claims — H_cog, sync, topology phase transition |
| `test_sigma_social_override.py` | `sigma_social_override` hook correctness (ablation API) |
| `test_robustness.py` | Noise tolerance, large N (N=1200), integration stability |
| `test_adversarial.py` | Adversarial parameter inputs, edge cases |
| `test_kernel.py` | Levitating Sigmoid kernel, polarity switching |
| `test_v5_hysteresis.py` | Schmitt-trigger hysteresis and fatigue mechanism |
| `test_v5_art.py` | ART adaptive reset threshold (V6 core feature) |
| `test_v5_compartments.py` | Compartimentalised dynamics |
| `test_v5_metacognitive.py` | Metacognitive plasticity |
| `test_v5_nonlocal_coupling.py` | Non-local coupling by doubt similarity |
| `test_coordination_metrics.py` | Pairwise synchrony, mutual information metrics |
| `test_manus_v2.py` | Manus v2 compatibility regression |
| `test_fuzzing.py` | Random parameter fuzzing (stability boundary) |
| `exploits/` | Adversarial stress tests — topology attacks, stiffness, chaos |

### Experiments — `experiments/`

**Note:** Historical, exploratory, and verification scripts have been organized into the `experiments/scratch/` directory to keep the root clean and maintain focus on the core modules.

**Paper 1 experiments** (produce CSV + PNG in `figures/`, located in `experiments/scratch/`):

| Script | Produces | Description |
|---|---|---|
| `experiments/scratch/p2_sigma_social_ablation.py` | `p2_sigma_social_ablation.*` | FULL / SS_NOISE / SS_STATIC / FROZEN_U ablation (n=5 seeds) |
| `experiments/scratch/p2_spatial_mutual_information.py` | `p2_spatial_mutual_information.*` | I(v_i; v_j) vs hop distance, 4 conditions |
| `experiments/scratch/p2_tau_u_bifurcation.py` | `p2_tau_u_bifurcation.*` | tau_u sweep, forced regime (n=5 seeds) |
| `experiments/scratch/p2_tau_u_bifurcation_endogenous.py` | `p2_tau_u_bifurcation_endogenous.*` | tau_u sweep, I_stim=0 |
| `experiments/scratch/p2_stochastic_resonance_topology.py` | `p2_stochastic_resonance_topology.*` | SR curve by topology |
| `experiments/scratch/p2_finite_size_scaling.py` | `p2_finite_size_scaling.*` | N=25 to 1600 scaling |
| `experiments/scratch/p2_edge_betweenness_analysis.py` | `p2_edge_betweenness.*` | Edge betweenness vs diversity |
| `experiments/scratch/p2_delta_sweep.py` | `p2_delta_sweep.*` | delta offset sweep (Levitating Sigmoid) |
| `experiments/scratch/p2_doubt_community_detection.py` | `p2_doubt_community_detection.*` | NMI between structural and u-functional partitions |
| `experiments/scratch/p2_art_benchmark.py` | `p2_art_benchmark.*` | ART benchmark under sustained cognitive load |
| `experiments/scratch/p2_v5_combination.py` | `p2_v5_combination.*` | Metacognitive plasticity + compartments combined |
| `experiments/scratch/p2_compartments.py` | `p2_compartments.*` | Compartimentalised dynamics exploration |
| `experiments/scratch/p2_nonlocal_coupling.py` | `p2_nonlocal_coupling.*` | Non-local coupling by doubt similarity |

**SPICE validation** (Paper B, located in `experiments/scratch/`):

| Script | Description |
|---|---|
| `experiments/scratch/spice_mismatch_50seeds.py` | 50-seed MC campaign, BA m=5, Cohen's d validation |
| `experiments/scratch/spice_noise_calibration.py` | Derives eta=0.5 equivalence to sigma_equiv |
| `experiments/scratch/spice_mismatch_sweep.py` | RMS error sweep vs sigma_noise |
| `experiments/scratch/spice_validation.py` | SPICE vs Python side-by-side on 4x4 lattice |
| `experiments/scratch/spice_art_kirchhoff.py` | Kirchhoff law validation of ART in SPICE |
| `experiments/scratch/spice_p420_hfo2_memristor.py` | HfO2 memristor SPICE characterization |

**Binder cumulant / Finite-size scaling** (V6, located in `experiments/scratch/`):

| Script | Description |
|---|---|
| `experiments/scratch/v6_binder_cumulant_u4.py` | Binder U4 cumulant FSS for spectral phase transition |
| `experiments/scratch/run_heroic_1600.py` | Large-scale validation N=1600 |
| `experiments/scratch/run_heroic_800.py` | Medium-scale validation N=800 |

### Scientific documents — `docs/`

| File | Description |
|---|---|
| `preprint.tex` / `preprint.pdf` | **Paper 1** — "Sustained Attractor Diversity in Doubt-Modulated FHN Networks" (14 pages) |
| `paper_B/paper_B.tex` / `.pdf` | **Paper B** — SPICE validation + topological dead zone (hardware bridge) |
| `paper_2/paper_2.tex` / `.pdf` | **Paper 2** — Draft on doubt variable as anti-synchronization filter |
| `limitations.md` | **Scientific truth table** — confirmed claims, known limits, reconciled binning artifacts |
| `academic_history.md` | Full chronological history of the project |
| `theoretical_anchoring.md` | Mathematical foundations and literature connections |
| `scientific_report_v26.md` | Full scientific documentation |
| `unified_specification_v23.md` | Unified model specification |

### Project-level files

| File | Description |
|---|---|
| `PROJECT_STATUS.md` | **Living document** — current state, all experimental results, open questions |
| `REPRODUCE_IN_5_MINUTES.md` | **5-minute visual proof** — start here |
| `CHANGELOG_V3.md` | Version history with scientific changes per release |
| `HACKER_GUIDE.md` | Deep dive into the codebase for contributors |
| `CITATION.cff` | Machine-readable citation metadata |
| `ai_contributors.json` | AI collaboration log (Kimi, Edison, Manus, Claude) |
| `docs/limitations.md` | Truth table of scientific claims |

### Figures — `figures/`

All figures are auto-generated by their corresponding experiment script. Key files:

| Figure | Generated by |
|---|---|
| `p2_sigma_social_ablation.png` | `experiments/scratch/p2_sigma_social_ablation.py` |
| `p2_tau_u_bifurcation.png` | `experiments/scratch/p2_tau_u_bifurcation.py` |
| `fiedler_phase_diagram.png` | `experiments/scratch/fiedler_phase_diagram.py` |
| `spice_50seeds_validation.png` | `experiments/scratch/spice_mismatch_50seeds.py` |
| `p2_doubt_community_detection.png` | `experiments/scratch/p2_doubt_community_detection.py` |
| `v6_binder_cumulant.png` | `experiments/scratch/v6_binder_cumulant_u4.py` |
| `p2_art_benchmark.png` | `experiments/scratch/p2_art_benchmark.py` |
| `spice_art_kirchhoff.png` | `experiments/scratch/spice_art_kirchhoff.py` |

## :scroll: Citation

If you use this code in your research, please cite:

```bibtex
@software{mem4ristor_v6,
  author       = {Julien Chauvin},
  title        = {Mem4ristor V6: Spatiotemporal Chaos and Chimera States in Doubt-Modulated FitzHugh-Nagumo Networks},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.19700749},
  url          = {https://doi.org/10.5281/zenodo.19700749}
}
```

## :page_facing_up: License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## :handshake: Let's Connect

I am an independent researcher looking for an **arXiv endorsement** (e.g., in `nlin.AO` or `cs.NE`) to publish the preprint associated with this code.

If the demo intrigues you, I would be honored to discuss the methodology with you. You can reach out directly via:
* **Email:** contact@cafevirtuel.org
* **X / Twitter:** [@Jusyl80](https://x.com/Jusyl80)

Julien Chauvin