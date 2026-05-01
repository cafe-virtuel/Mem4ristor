# 🚀 Mem4ristor V4: Reproduce in 5 Minutes

Welcome. If you are a researcher in complex systems, non-linear dynamics, or neuromorphic engineering, this repository contains the code for a mathematically audited **spatiotemporal chaos model** (Chimera state) using FitzHugh-Nagumo oscillators on a scale-free network, driven by intrinsic plasticity.

We know your time is valuable. You don't need to read the 2,000+ lines of documentation right now. 

**Run this 1-minute demo to see the mathematical proof visually.**

## Quick Start (Linux, macOS, Windows)

```bash
# 1. Clone the repository (optional: navigate to your Desktop first)
# cd ~/Desktop 
git clone https://github.com/cafe-virtuel/mem4ristor.git
cd mem4ristor

# 2. Install dependencies (requires Python 3.9+)
pip install -e .

# 3. Run the visual proof
python experiments/demo_chimera.py
```

### What you will see:
The script generates `demo_chimera_output.png` which contains 3 plots:
1. **Time Dynamics**: Shows the frustrated synchronization between the majority (normal nodes) and the 15% structural "heretics".
2. **Phase Dispersion**: A polar plot of the Kuramoto phase space ($v, w$) demonstrating a macroscopic phase-locked state ($R \approx 0.61$) coexisting with spatial dispersion.
3. **The Graph**: The underlying Barabási-Albert ($m=3$) topology.

## Why this matters?
This model mathematically demonstrates that **"heretic" nodes (forced with an inverted stimulus) act as causal structural walls**. The surrounding network bombards them with information (proven via Transfer Entropy), forcing them to resist, which shatters the dead-zone consensus and maintains continuous diversity without requiring stochastic noise.

## Let's Connect
I am an independent researcher and I am currently looking for an **arXiv endorsement** (e.g., in `nlin.AO` or `cs.NE`) to publish the preprint associated with this code.

If the demo intrigues you, I would be honored to discuss the methodology with you. You can reach out directly via:
* **Email:** contact@cafevirtuel.org
* **X / Twitter:** [@Jusyl80](https://x.com/Jusyl80)

Julien Chauvin
