# EDISON Review: Glassy Dynamics + Active Inference Gaps

**Date:** 2026-06-01
**Source:** EDISON observations on preprint
**Session:** Hermes autonomous research — TEST_HERMES
**Status:** NEW OBSERVATIONS (not documented in preprint)

---

## Finding 1: Glassy Dynamics Fingerprint

### Observation (EDISON)

The variance of H_stable and LZ show opposite behaviors as a function of lambda2:

| Zone | lambda2 range | H_stable mean | H_stable raw_std | LZ mean | LZ raw_std |
|------|-------------|---------------|-----------------|---------|------------|
| SPARSE | 1.3-1.8 | 3.557 | **0.379** | 0.827 | **0.056** |
| CRITICAL | 2.0-2.8 | 3.245 | **0.544** | 0.817 | **0.076** |
| DENSE | 3.0-9.7 | 2.631 | **0.265** | 0.744 | **0.090** |

**Key pattern:**
- H_stable mean decreases monotonically (consolidation)
- **But H_stable variance peaks at the critical zone (0.544), then drops in dense regime (0.265)**
- LZ mean decreases monotonically (structural consolidation)
- **LZ variance increases monotonically with lambda2 (0.056 -> 0.076 -> 0.090)**

This diverging variance fingerprint — variance of one metric increasing while the other decreases — is a characteristic signature of **glassy dynamics** (slow dynamics, history-dependence, ergodicity breaking near the transition). It is NOT discussed in the current preprint.

### Why This Matters

The preprint characterizes the entropy depression as a "spectral crossover" without mentioning the dynamic glass-like behavior that precedes it. The critical zone (lambda2 ~ 2-3) is where the system exhibits:

1. **Maximum metastability** — H_stable variance highest at critical
2. **Divergent timescales** — LZ variance increases with connectivity
3. **Aging-like behavior** — the system does not reach a unique steady state in finite time

This is relevant because:
- It deepens the scientific story: the network exhibits slow relaxation, not just a sharp transition
- It connects to the physics literature on glasses and complex systems
- It provides a new observable (variance ratio) that could be measured experimentally

### EDISON Assessment

**"Interesting and not explicitly discussed"** — confirmed. This is a genuine new finding that:

1. Is supported by the campaign_j data (campaign_j_agg.csv)
2. Does not require any new simulation — it is already in the existing data
3. Requires only a variance analysis to formalize

### Recommended Actions

1. **Do NOT claim glassy dynamics without a dedicated analysis.** The fingerprint is suggestive but not conclusive.

2. **Add a note in the Discussion** acknowledging that the variance behavior is suggestive of glass-like dynamics, pending dedicated analysis.

3. **Measurement to add** in future work:
   - Compute Var(H_stable) and Var(LZ) as a function of waiting time
   - Measure aging exponent (test for t^(-alpha) decay)
   - Test for ergodicity breaking (compare time-averaged vs ensemble-averaged observables)

4. **For the current preprint**: Add a sentence in Discussion along lines of:
   > "The critical zone also exhibits maximum run-to-run variability in H_stable (raw_std=0.544) combined with monotonically increasing LZ variability (raw_std=0.056 to 0.090), a fingerprint suggestive of glass-like dynamics that warrants dedicated investigation."

---

## Finding 2: Active Inference Framing — Rhetorical Stretch

### Observation (EDISON)

Section 6.3 of the preprint ("Thermodynamic Viability and Active Inference") claims a connection to Friston's Free Energy Principle. However, the paper explicitly admits:

> "This dynamic can be formally interpreted as a form of Active Inference: the node alters its effective coupling with the environment — the network topology — to minimize future prediction errors, **without invoking an explicit generative model**." (line 450, emphasis added)

### What Active Inference Requires vs What Is Present

| FEP Component | Present in preprint? |
|---------------|---------------------|
| Generative model of world | **NO** — explicitly absent |
| Variational inference (ELBO) | **NO** — no posterior approximation |
| Free energy as objective | **NO** — only \|Laplacian\| as proxy |
| Action on environment | PARTIAL — coupling changes |
| Active inference loop | **NO** — simple feedback |

### EDISON Assessment

**"A rhetorical stretch"** — confirmed. The mechanism described is:

1. **Homeostatic regulation** — a node adjusts coupling based on local disagreement
2. **Feedback control** — u responds to |Laplacian| like a thermostat responds to temperature
3. **Adaptive coupling** — polarity modulation based on state-dependent signal

This is a well-understood control theory concept. Calling it "Active Inference" when there is no generative model, no variational inference, and no free energy minimization is not accurate.

### The Mechanism Is Still Interesting

The doubt-modulated coupling mechanism is worth publishing on its own terms:
- Polarity-modulated anti-synchronization is clearly demonstrated
- The FEP framing is unnecessary for the scientific contribution
- Removing it strengthens the paper by eliminating an overclaim

### Recommended Actions

1. **Remove "Active Inference" from the title and abstract.** The term creates expectations that are not met.

2. **Downgrade Section 6.3** from "Active Inference" to a more accurate description like:
   - "Homeostatic coupling regulation"
   - "Prediction-error-driven coupling modulation"
   - "Adaptive polarity control"

3. **Keep the physical interpretation** but frame it correctly:
   - The Laplacian magnitude acts as a local conflict signal (not a sensory prediction error in the FEP sense)
   - Nodes adjust coupling to reduce local disagreement (not to minimize variational free energy)
   - The mechanism is closer to "gradient descent on local disagreement" than to FEP

4. **If the reviewer raises this**: the authors can acknowledge that the FEP framing was aspirational and that the mechanism is more accurately described as homeostatic regulation.

---

## Combined Recommendation

Both observations share a common theme: **the preprint claims more than it demonstrates**.

| Issue | Current claim | EDISON recommendation |
|-------|--------------|----------------------|
| Glassy dynamics | Not mentioned | Add acknowledgment + future work |
| Active Inference | Claimed | Downgrade to "homeostatic regulation" |

The paper's core contribution (doubt-modulated anti-synchronization, spectral connectivity boundary) is solid. These two adjustments would remove the overclaims and make the paper more defensible.

---

**EDISON sign-off: 2026-06-01**
**Source data:** figures/campaign_j_agg.csv (Campaign J, 1800 sims, N=100/200/400)
**Script for reproduction:** Re-run variance analysis on campaign_j_agg.csv grouped by lambda2 zone