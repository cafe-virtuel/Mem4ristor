# Executive Summary
**Overall Scientific Integrity Score: 8.5/10.** The Mem4ristor v3.2.0 project represents a substantial, methodologically rigorous investigation into frustrated synchronization. The discovery of a thermodynamic escape from topological dead zones via Johnson-Nyquist noise is a compelling, high-impact result, well-supported by robust analog (ngspice) and digital validation. However, several inconsistencies and methodological nuances remain. The three most critical items to address prior to Paper 2 submission are:
1. **Requalifying the Endogenous tau_u Bifurcation (A6)**: The bifurcation exists, but the cognitive diversity (H_cog) metric conflates sub-threshold oscillations with true state diversity under $I_{stim}=0$. The heretic mechanism is verified to be a stimulus-response mechanism, mathematically inactive under $I_{stim}=0$.
2. **Clarifying the FROZEN_U Synchronized Diversity Artifact (A5)**: The high $H_{cog}$ observed when freezing $u$ is a spatial binning artifact of coherent limit-cycle oscillations (nodes spread along the FHN cycle but moving synchronously). The narrative must clearly distinguish "spurious spatial spread" from "genuine coordinate diversity".
3. **Addressing the Inconsistent Community Detection Baseline (A7)**: The NMI signal for doubt-driven community detection is weak, and the project documentation is out of sync with the generated CSV data. The baseline permutation test reveals that NMI > 0.25 occurs by chance due to the fragmentation of the doubt partition by singleton heretics.

---

# Detailed Audit Findings

## A1. Paper B claim-by-claim verification
🟢 **CONFIRMED**
- **Runnable experiments**: All key claims in Paper B are backed by runnable experiments (`spice_mismatch_sweep.py`, `spice_19ter_robustness.py`, `spice_50seeds_validation.py`, `spice_noise_calibration.py`). 
- **Figures**: All referenced figures exist and match the text.
- **Statistics**: The reported Cohen's $d = 20.78$ ($p < 10^{-63}$) for noise-only escape and $d = 0.19$ ($p = 0.35$) for noise+CMOS mismatch are exactly reproducible via the 50-seed Monte Carlo script.
- **270x calibration gap**: Validated. The $\sigma_{equiv}$ calculation in `spice_noise_calibration.py` is physically sound (integrating voltage variance on an RC unit).
- **$\lambda_{2,crit} \approx 2.5$ threshold**: Confirmed by `p2_stochastic_resonance_topology.py` (Item 10). It is a topology-agnostic structural boundary.
- **SS_STATIC ablation**: Confirmed by `p2_sigma_social_ablation.py`. Replaces the live local disagreement signal with its temporal mean, confirming the hardware simplification does not alter the fundamental dynamic regime.

## A2. Paper B methodology gaps
🟠 **WEAK**
- **Seed counts**: The n=50 seeds for SPICE validation is excellent. However, many Python topology investigations (e.g. `limit02_topology_sweep.py`) still rely on n=3 seeds. While sufficient for deterministic bounds, evaluating stochastic resonance on n=3 is underpowered.
- **Topology generalization**: BA m=5 N=64 is used for SPICE, whereas BA m=3 N=100 is used for Python. This asymmetry weakens direct cross-comparison, though the finite-size scaling script (`p2_finite_size_scaling.py`) establishes that $\lambda_2$ is stable with $N$, mitigating the impact.
- **Statistical Tests**: Confidence intervals are reported correctly for the 50-seed SPICE analysis, but absent in earlier experiments.

## A3. Preprint (Paper 1) consistency with current code
🟡 **INCONSISTENCY**
- **Equations vs Implementation**: The implementation correctly incorporates the $10.0$ factor for $\tau_u$ and the $0.01$ $\delta$ parameter (Levitating Sigmoid), matching the updated preprint. The plasticity decay term ($-w/\tau_{plast}$) is universally active, which is now documented in the preprint.
- **Poincaré-Bendixson**: Retained only for the single isolated oscillator in the $\alpha > \alpha_{crit}$ regime, correctly reflecting the numerical stability of the $3N$ coupled system.
- **15% Heretic Ratio**: The limitation notes that 15% is not universal, which contradicts the bolder claims of earlier drafts. The preprint appropriately qualifies this.

## A4. New code correctness (added 2026-04-25)
🟢 **CONFIRMED**
- **`sigma_social_override`**: Implemented correctly in `dynamics.py` and `topology.py`. It explicitly overrides the signal *only* in the $du$ equation, leaving the $I_{coup}$ and plasticity terms properly coupled to the true Laplacian.
- **Guard and bounds**: The guard `np.clip(sigma_social_for_u, 0, 100)` is correctly applied.
- **`core.py` facade**: Re-exports `Mem4Network`, meaning downstream calls to `step()` safely receive the `sigma_social_override` parameter.
- **Warmup phase**: The manual Laplacian computation in `p2_sigma_social_ablation.py` flawlessly replicates the logic in `topology.py`, including the `degree_linear` normalization scaling.

## A5. Experiment C (sigma_social ablation) interpretation
🟡 **INCONSISTENCY**
- **FULL vs SS_NOISE bias**: The 1000-step warmup is sufficient to reach steady-state RMS, meaning SS_NOISE is calibrated against a valid baseline.
- **Rectified Gaussian**: Using absolute values of a normal distribution for SS_NOISE alters the mean relative to the true $\sigma_{social}$ distribution, but empirically produces $\Delta H_{cog} < 2\%$, confirming the network is insensitive to the exact noise shape so long as it prevents $u$ from decaying to $\sigma_{baseline}$.
- **FROZEN_U Artifact**: The $H_{cog}=0.99$ and $\mathrm{sync}=0.78$ for FROZEN_U is a **measurement artifact**. With $u$ frozen, the nodes fall into a globally synchronized FHN limit-cycle oscillation. Because the nodes are spread along the phase of the cycle, any instantaneous spatial snapshot shows high diversity ($H_{cog}$), but their trajectories are highly correlated ($\mathrm{sync}=0.78$). This is not "cognitive diversity" but rather a "spurious spatial spread." The project documentation acknowledges this ("diversité cognitive spurieuse"), but it must be made clearer in the paper.

## A6. Experiment D (endogenous bifurcation) interpretation
🔴 **CRITICAL**
- **Heretic No-Op**: With $I_{stim}=0.0$, the heretic flip `I_eff[mask] *= -1.0` is an exact mathematical no-op. The heretics still receive identical coupling ($I_{coup}$) to normal nodes. Therefore, the "endogenous" regime does not test the structural asymmetry mechanism at all.
- **$H_{cog} \approx 0$ Artifact**: The $H_{cog} \approx 0$ result throughout the $I_{stim}=0$ sweep is a binning artifact. All nodes converge near the excitable fixed point $v^* \approx -1.29$, falling entirely within state bin 1 ($v < -1.2$). $H_{cont}$ reveals significant sub-threshold continuous diversity (ranging from 0.77 to 3.08 bits), confirming the bifurcation occurs, but it is a bifurcation in sub-threshold noise coordination, not cognitive state diversity.

## A7. NMI community detection
🔴 **CRITICAL**
- **Inconsistent Documentation**: `PROJECT_STATUS.md` claims BA m=3 seed=777 is highly significant ($z=+2.82, p=0.002$). However, the generated CSV file (`p2_doubt_community_detection.csv`) reports $z=+0.73, p=0.238$ for this seed. The only significant seed in the CSV is lattice seed=123.
- **Baseline Permutation**: The bootstrap permutation of the doubt partition is implemented correctly (`a_shuffled = rng.permutation(a)`).
- **Singletons Problem**: The presence of heretics ($u=1.0$) with zero variance forces their Pearson correlation with other nodes to near-zero, creating artifactual singleton communities. This artificially inflates the random baseline NMI, burying any real signal.
- **Louvain on Signed Edges**: The script applies Louvain to a doubt-affinity graph with negative edge weights (`C_u` passed directly). NetworkX's Louvain implementation assumes non-negative weights; negative weights can corrupt the modularity optimization.

## A8. Test suite audit
🟢 **CONFIRMED**
- **Test Suite Execution**: 78 PASS, 2 XFAIL, 0 FAIL.
- **Test Coverage**: Core mechanisms, boundaries, NaN handling, and scientific regression properties are thoroughly tested.
- **Missing Test**: A test for `sigma_social_override` was missing. A new test suite `test_sigma_social_override.py` was created and successfully passed (6/6 tests), verifying the override affects only the doubt equation and correctly clamps values.
