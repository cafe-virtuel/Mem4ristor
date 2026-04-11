# Mem4ristor V3: Neuromorphic Cognitive Architecture

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18620597.svg)](https://doi.org/10.5281/zenodo.18620597)
[![Tests](https://github.com/Jusyl236/mem4ristor-v2/actions/workflows/test.yml/badge.svg)](https://github.com/Jusyl236/mem4ristor-v2/actions/workflows/test.yml)

**Mem4ristor V3** is a computational implementation of extended FitzHugh-Nagumo dynamics designed to investigate emergent critical states in neuromorphic networks. This research code focuses on the role of "Constitutional Doubt" ($u$) and "Structural Heretics" in preventing consensus collapse in scale-free and lattice networks.

> **Status**: v3.2.0 (Stable Research Release — Degree-Normalized Coupling + V5 Hysteresis + Sparse CSR)

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
git clone https://github.com/Jusyl236/mem4ristor-v2.git
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

# Run simulation for 1000 steps
for step in range(1000):
    net.step(I_stimulus=0.5)

# Calculate final entropy (measure of diversity)
print(f"Final System Entropy: {net.calculate_entropy():.4f}")
# Expected: H_stable ≈ 0.92 ± 0.04 on default lattice configuration
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
    net.step(I_stimulus=0.5)

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

*   `src/mem4ristor/`: Core package source code.
*   `tests/`: Unit, robustness, adversarial, and scientific regression tests.
*   `experiments/`: Benchmark scripts, parameter sweeps, and SPICE netlists.
*   `experimental/`: Non-production modules (Philosopher King).
*   `docs/`: Scientific documentation, preprint, and limitations truth table.
*   `failures/`: Archived failure logs (nothing is erased).

## 📜 Citation

If you use this code in your research, please cite the associated dataset/preprint:

```bibtex
@software{mem4ristor_v3,
  author       = {Julien Chauvin},
  title        = {Mem4ristor V3: Neuromorphic Cognitive Architecture},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18620597},
  url          = {https://doi.org/10.5281/zenodo.18620597}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
