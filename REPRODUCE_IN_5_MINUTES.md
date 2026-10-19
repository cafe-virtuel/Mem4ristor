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

This model demonstrates that **"heretic" nodes (forced with an inverted stimulus) act as causal structural walls**, while the dynamic doubt variable $u$ acts as an active anti-synchronization filter (`FULL` vs `FROZEN_U`, complete separation across 30 seeds, `Cohen's d = 9.43`). The full scientific claims, automated verification rules, and known limitations are documented in [`docs/CLAIMS_REGISTER.md`](docs/CLAIMS_REGISTER.md) and [`docs/BILAN_FORCES_FAIBLESSES.md`](docs/BILAN_FORCES_FAIBLESSES.md) (superseding the earlier `docs/limitations.md`).

## Scientific integrity note

This repository is actively maintained and includes an automated **22-claim verification suite** (`python tools/preprint_guardian.py` and `python tools/tex_guardian.py`) alongside an honest **claims and limitations register** (`docs/CLAIMS_REGISTER.md`, `docs/BILAN_FORCES_FAIBLESSES.md`). Several early claims have been revised or refuted by our own controls:

* The dead-zone boundary on dense networks is governed by the harmonic coupling degree ($k_{\text{harm}} \approx 6$, mean-field averaging), not by the spectral gap $\lambda_2$ ($\lambda_2 \approx 2.31$ is a correlational proxy on BA/ER graphs).
* The 15% heretic threshold is not universal — it requires degree-linear coupling normalization on scale-free networks.
* The isolated-node stability claim in the original preprint was corrected: the fixed point is a stable spiral at default parameters ($\alpha_{\text{crit}} = 0.2955$).
* Noise is required to sustain diversity under cold-start conditions ($I_{\text{stimulus}}=0$).

The `PROJECT_STATUS.md` file maintains a full register of all scientific investigations, confirmed findings, and revised claims.

## Citation

If this work is useful to you, please cite:

```bibtex
@software{mem4ristor_zenodo,
  author = {Julien Chauvin},
  title = {Sustained Attractor Diversity in Doubt-Modulated FitzHugh-Nagumo Networks:
           Spectral Dead Zone and Chimera States on Scale-Free Topologies},
  year = {2026},
  publisher = {Zenodo},
  version = {V4.0.0},
  doi = {10.5281/zenodo.18620596},
  url = {https://doi.org/10.5281/zenodo.18620596},
  note = {Concept DOI; resolves to the latest deposit. Latest deposited version is
          V4.0.0 (2026-05-02, DOI 10.5281/zenodo.19986042). The code in this
          repository is V6.0.0 and has not been deposited.}
}
```

## Let us Connect

I am an independent researcher and I am currently looking for an **arXiv endorsement** (e.g., in `nlin.AO` or `cs.NE`) to publish the preprint associated with this code.

If the demo intrigues you, I would be honored to discuss the methodology with you. You can reach out directly via:
* **Email:** contact@cafevirtuel.org
* **X / Twitter:** [@Jusyl80](https://x.com/Jusyl80)

Julien Chauvin