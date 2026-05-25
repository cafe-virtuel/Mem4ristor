# Mem4ristor V6: Reproduce in 5 Minutes

Welcome. If you are a researcher in complex systems, non-linear dynamics, or neuromorphic engineering, this repository contains the code for a spatiotemporal chaos model (Chimera state) using FitzHugh-Nagumo oscillators on a scale-free network, driven by intrinsic plasticity.

We know your time is valuable. You do not need to read the 2,000+ lines of documentation right now.

**Run this 1-minute demo to see the mathematical proof visually.**

## Quick Start (Linux, macOS, Windows)

```bash
# 1. Clone the repository
git clone https://github.com/cafe-virtuel/Mem4ristor.git
cd Mem4ristor

# 2. Install dependencies (requires Python 3.9+)
pip install -e .

# 3. Run the visual proof
python experiments/demo_chimera.py
```

### What you will see:

The script generates `demo_chimera_output.png` which contains 3 plots:

1. **Time Dynamics**: Shows the frustrated synchronization between the majority (normal nodes) and the 15% structural "heretics".
2. **Phase Dispersion**: A polar plot of the Kuramoto phase space (v, w) demonstrating a macroscopic phase-locked state coexisting with spatial dispersion.
3. **The Graph**: The underlying Barabási-Albert (m=3) topology.

## Why this matters?

This model demonstrates that **"heretic" nodes (forced with an inverted stimulus) act as causal structural walls**. The surrounding network bombards them with information, forcing them to resist, which shatters the dead-zone consensus and maintains continuous diversity. The full scientific claims and known limitations are documented in `docs/limitations.md`.

## Scientific integrity note

This repository is actively maintained and includes an honest **limitations register** (`docs/limitations.md`). Several early claims have been revised:

* The 15% heretic threshold is not universal — it requires degree-linear coupling normalization on scale-free networks.
* The isolated-node stability claim in the original preprint was corrected: the fixed point is a stable spiral at default parameters.
* Noise is required to sustain diversity under cold-start conditions (I_stimulus=0).

The `PROJECT_STATUS.md` file maintains a full register of all scientific investigations, confirmed findings, and revised claims.

## Citation

If this work is useful to you, please cite:

```bibtex
@software{mem4ristor_v6,
  author = {Julien Chauvin},
  title = {Mem4ristor V6: Spatiotemporal Chaos and Chimera States in Doubt-Modulated FitzHugh-Nagumo Networks},
  year = {2026},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.19700749},
  url = {https://doi.org/10.5281/zenodo.19700749}
}
```

## Let us Connect

I am an independent researcher and I am currently looking for an **arXiv endorsement** (e.g., in `nlin.AO` or `cs.NE`) to publish the preprint associated with this code.

If the demo intrigues you, I would be honored to discuss the methodology with you. You can reach out directly via:
* **Email:** contact@cafevirtuel.org
* **X / Twitter:** [@Jusyl80](https://x.com/Jusyl80)

Julien Chauvin