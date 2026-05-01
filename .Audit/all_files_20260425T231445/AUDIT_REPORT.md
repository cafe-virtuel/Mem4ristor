# Executive Summary
The scientific integrity of Mem4ristor v3.2.0 is fundamentally sound. The underlying Python implementation, SPICE-to-Python calibration, and topological findings (λ₂_crit ≈ 2.5) are robust and reproducible. The most critical items to address before Paper 2 submission are: (1) A silent equation mismatch between the preprint and code regarding an adaptive `epsilon_u` term (the "surprise" multiplier) missing from the preprint's doubt equation; (2) The Paper B claim of `ΔH_cog < 0.09%` for SS_STATIC is stale relative to the $n=5$ dataset ($0.115\%$); and (3) The $H_{cog} \approx 0$ endogenous result (I_STIM=0) is entirely an artifact of the 5-bin physiological thresholds (all voltage activity remains below -1.2, despite continuous variations measured by $H_{cont}$).

## A1. Paper B claim-by-claim verification
- 🟢 **Runnable scripts**: SPICE data generation and calibration scripts exist and function correctly.
- 🟢 **Figures**: All referenced figures (`p420_hfo2_memristor.png`, `spice_mismatch_sweep_continuous.png`, `spice_50seeds_validation.png`) are present.
- 🟢 **Statistics**: Cohen's $d = 20.78$ ($p < 10^{-63}$, $n=50$) is exactly reproducible from the `spice_50seeds_validation.csv` (A vs B mean 1.38 vs 4.30).
- 🟢 **270× calibration gap**: The `spice_noise_calibration.py` derives $\eta=0.5 \leftrightarrow \sigma_{\text{equiv}}=0.0044$. The $1.2$ digital noise tested is indeed $\sim 272\times$ the equivalent amplitude. The derivation accurately measures standard deviation of numerical increments.
- 🟡 **λ₂_crit ≈ 2.5 threshold**: This is an eyeballed midpoint rather than a rigorously fitted threshold. The exact crossing falls between BA $m=3$ ($\lambda_2=1.41$) and BA $m=5$ ($\lambda_2=3.10$). The phase diagram CSV confirms this gap.
- 🟠 **§2 scope note**: The paper claims $\Delta H_{\text{cog}} < 0.09\%$ for the SS_STATIC ablation. In the latest $n=5$ CSV (`p2_sigma_social_ablation.csv`), the absolute $\Delta$ is 0.0027 bits, which is $0.115\%$ of $\log_2 5$. The claim is slightly stale but qualitatively valid.

## A2. Paper B methodology gaps
- 🟢 **Seed counts**: 50 seeds for the SPICE MC campaign is well-powered for the reported effect sizes. Python experiments use $n=3$ or $n=5$, which is acceptable for large deterministic divergences but limits precise statistical bounds.
- 🟡 **Topology asymmetry**: SPICE validation relies on BA $m=5$ $N=64$ (computationally bound), whereas Python primarily uses BA $m=3$ $N=100$. This prevents direct point-by-point superposition, though the phenomenological comparisons remain sound.

## A3. Preprint (Paper 1) consistency with current code
- 🔴 **Equation mismatch**: The preprint defines $du_i/dt$ (Eq. 2) with a fixed $\varepsilon_u / \tau_u$ multiplier. The actual implementation in `dynamics.py` (L228) computes an `epsilon_u_adaptive` term scaled by `np.clip(1.0 + alpha_surprise * sigma_social_safe, 1.0, 5.0)`. This critical adaptive mechanism is missing from the core preprint equations (though referenced in an appendix context).
- 🟢 **Poincaré-Bendixson**: The preprint correctly footnotes that PB theorem only applies to $\alpha > \alpha_{\text{crit}} \approx 0.296$ (planar autonomous systems), invalidating it for the default sub-Hopf excitable regime.
- 🟢 **15% Heretic Ratio**: The preprint explicitly identifies parameter dependencies and scaling laws without overclaiming universality, in alignment with `PROJECT_STATUS.md`.

## A4. New code correctness (v3.2.0)
- 🟢 `sigma_social_override` acts strictly on the $du$ equation via `sigma_social_for_u` (L231). The main $I_{coup}$ and plasticity variables use the true un-overridden `laplacian_v`.
- 🟢 The guard `np.clip(sigma_social_for_u, 0, 100)` is correctly applied to both adaptive scaling and the base $du$ term.
- 🟢 The API facade in `core.py` successfully passes `sigma_social_override` via `Mem4Network.step()`.

## A5. Experiment C (sigma_social ablation) interpretation
- 🟡 **Warmup bias**: The 1000-step warmup captures transient, non-stationary dynamics. This slightly biases the RMS estimation used for SS_NOISE and SS_STATIC, but not enough to invalidate the qualitative conclusion (which relies on order-of-magnitude differences against FROZEN_U).
- 🟡 **Rectified Gaussian mean**: A full-wave rectified Gaussian $|N(0, \sigma)|$ possesses an identical RMS to the underlying unrectified Gaussian, but its mean is lower ($\sim 0.798 \sigma$). The metric preserves power but alters the DC offset of the injected signal.
- 🟢 **FROZEN_U hypersynchrony**: The data confirms that FROZEN_U results in hyper-synchronization (sync = 0.730). The resulting $H_{cog} = 1.07$ is indeed a measurement artifact of synchronized nodes oscillating coherently across boundaries, not true cognitive diversity. `PROJECT_STATUS` accurately notes this spurious diversity.

## A6. Experiment D (endogenous bifurcation) interpretation
- 🟢 **Heretic flip**: In `dynamics.py`, the stimulus flip $I_{eff}[heretic\_mask] \times= -1.0$ is executed before coupling is added. For $I_{stim}=0$, $I_{eff}=0$, so heretics are functionally identical to consensus nodes.
- 🔴 **H_cog ≈ 0 artifact**: The reported $H_{cog} \approx 0$ across the endogenous sweep is entirely a binning artifact. Phase-space tracking shows voltages oscillating narrowly between $-1.36$ and $-1.23$, permanently confined to the lowest $H_{cog}$ bin ($v < -1.2$). $H_{cont}$ successfully detects variations in this regime, but the 5-bin physiological metric collapses.

## A7. NMI community detection
- 🟢 **Bootstrap implementation**: The `nmi_baseline_random` function correctly holds the structural partition fixed while permuting the doubt partition, generating an appropriate null distribution.
- 🟠 **Singletons limitation**: Nodes with $u=1.0$ (heretics) have zero variance and correlate with nothing, artificially inflating the partition count and compromising Louvain optimization. Using trace correlations forces heretics into functional isolation regardless of structural embedding.
- 🟢 **Matrix preparation**: The pipeline extracts $|C_u|$ to pass positive weights to NetworkX's Louvain implementation, successfully averting the modularity corruption caused by negative Pearson correlations.

## A8. Test suite audit
- 🟢 Tests ran with 84 passing, 2 explicitly expected failures (xfail), and 0 unexpected failures. The `sigma_social_override` hook is comprehensively covered.
