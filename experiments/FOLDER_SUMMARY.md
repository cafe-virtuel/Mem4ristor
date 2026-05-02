# experiments/ — Résumé du dossier
> Mis à jour : 2026-05-02 10:26 | Par : Ronde Perpétuelle (mode summaries)
> **Ne pas modifier la table manuellement** — utiliser --mode summaries pour mettre à jour.

## Rôle du dossier
Scripts d'expériences scientifiques Mem4ristor. Classés par tier de pertinence (1=finding original, 2=validation, 3=robustesse, 4=hardware).

## Contenu

| Fichier | Description | Tier | Statut |
|---------|-------------|------|--------|
| `ablation_coordination.py` | Ablation study with trajectory-based coordination metrics (P1.5bis). | — | 🆕 nouveau |
| `ablation_coordination_topology.py` | Piste A — Bimodality analysis of ENDOGENOUS FULL synchrony (P1.5bis follow-up). | — | 🆕 nouveau |
| `ablation_coordination_topology_sweep.py` | Piste D — Multi-topology universality of coordination metrics (PROJECT_STATUS §3novedecies-bis D). | — | 🆕 nouveau |
| `ablation_minimality.py` | Ablation / Minimality Study for Mem4ristor v3.2.0 (KIMI critique #4). | — | 🆕 nouveau |
| `attack_resilience.py` | import numpy as np | — | 🆕 nouveau |
| `benchmark_kuramoto.py` | Benchmark: Mem4ristor V3 vs Kuramoto (Standard & Frustrated) | — | 🆕 nouveau |
| `benchmark_variability.py` | import numpy as np | — | 🆕 nouveau |
| `bimodality_50seeds.py` | Piste F — Bimodalité ENDOGENOUS FULL sur 50 seeds + Hartigan dip test | — | 🆕 nouveau |
| `cohen_u3.py` | Cohen U3 -- Overlap coefficient entre distributions FROZEN_U vs FULL | — | 🆕 nouveau |
| `demo_chimera.py` | Mem4ristor V4 - Chimera State Demonstration | demo | 🆕 nouveau |
| `dt_sensitivity.py` | dt Sensitivity Analysis (DeepSeek review item 7) | — | 🆕 nouveau |
| `dt_sensitivity_figure.py` | Génère figures/dt_sensitivity.png depuis figures/dt_sensitivity.csv | — | 🆕 nouveau |
| `event_phase_transition.py` | [13] Event-Driven Phase Transition — Peripheral Node High-Amplitude Forcing | 1 | 🆕 nouveau |
| `fiedler_phase_diagram.py` | lambda2 vs H_stable phase diagram — visualizing the topological transition. | — | 🆕 nouveau |
| `forcing_sweep_frozen_u.py` | Piste E — Forcing sweep FULL vs FROZEN_U (PROJECT_STATUS §3novedecies-bis piste E). | — | 🆕 nouveau |
| `generate_spice_grid.py` | import numpy as np | — | 🆕 nouveau |
| `heretic_ratio_sweep_coordination.py` | Piste C — Heretic ratio sweep under forcing (P1.5bis follow-up). | — | 🆕 nouveau |
| `lambda2_crit_regression.py` | Formalisation de lambda2_crit -- Regression logistique multi-sources. | — | 🆕 nouveau |
| `limit02_alpha_sweep.py` | LIMIT-02 Power-Law Normalization Sweep (v2) - Using core.py degree_power mode. | — | 🆕 nouveau |
| `limit02_ba_m_sweep.py` | LIMIT-02 BA Attachment Parameter Sweep — The m=5 Mystery. | — | 🆕 nouveau |
| `limit02_norm_sweep.py` | LIMIT-02 Normalization Sweep — Degree-based coupling on Barabási-Albert networks. | — | 🆕 nouveau |
| `limit02_topology_sweep.py` | LIMIT-02 Multi-Topology Sweep — Validate degree_linear across diverse networks. | — | 🆕 nouveau |
| `lyapunov_numerical.py` | Vérification Numérique de Stabilité (Lyapunov) | — | 🆕 nouveau |
| `lz_per_node.py` | [11] LZ Complexity Per Node — FULL vs FROZEN_U | 1 | 🆕 nouveau |
| `matern_noise.py` | [12] Bruit spatialement correle (Matern) — dead zone BA m=5 | 2 | 🆕 nouveau |
| `p2_delta_sweep.py` | Piste A5 -- Sweep delta de la Levitating Sigmoid (2026-04-24) | — | 🆕 nouveau |
| `p2_directed_coupling.py` | Piste A3 -- Couplage Asymetrique et Graphes Diriges (2026-04-24) | — | 🆕 nouveau |
| `p2_doubt_community_detection.py` | Item 12 -- Doubt-Driven Community Detection (2026-04-24) | — | 🆕 nouveau |
| `p2_edge_betweenness_analysis.py` | P2-9 — Edge betweenness + diameter vs lambda2 (PROJECT_STATUS §10 P2 item 9). | — | 🆕 nouveau |
| `p2_finite_size_scaling.py` | P2-7 — Finite-size scaling of the dead zone transition (PROJECT_STATUS §10 P2 item 7). | — | 🆕 nouveau |
| `p2_sigma_social_ablation.py` | Piste C — Ablation sigma_social → bruit pur (2026-04-25) | — | 🆕 nouveau |
| `p2_spatial_mutual_information.py` | Piste A4 -- Information Mutuelle Spatio-Temporelle (2026-04-24) | — | 🆕 nouveau |
| `p2_stochastic_resonance_directed.py` | Piste A1 — Résonance Stochastique Dirigée (2026-04-24) | — | 🆕 nouveau |
| `p2_stochastic_resonance_topology.py` | Item 10 -- Stochastic Resonance x Topology (2026-04-25) | — | 🆕 nouveau |
| `p2_tau_u_bifurcation.py` | Piste A2 -- Bifurcation tau_u (Dynamique Temporelle du Doute) (2026-04-24) | — | 🆕 nouveau |
| `p2_tau_u_bifurcation_endogenous.py` | Piste D — Bifurcation tau_u en Regime ENDOGENE (I_STIM = 0.0) (2026-04-25) | — | 🆕 nouveau |
| `phase_diagram.py` | Phase Diagram : H(heretic_ratio, D) | — | 🆕 nouveau |
| `phase_space_coordination.py` | Piste B — 2D phase diagram of coordination (synchrony × LZ complexity). | — | 🆕 nouveau |
| `protocole_bicameral.py` | import numpy as np | — | 🆕 nouveau |
| `reviewer2_chimera_comparison.py` | Reviewer Defense - Chimera State Comparison with Abrams-Strogatz (2004) | 1 | 🆕 nouveau |
| `reviewer2_critical_exponents.py` | Reviewer Defense - Critical Exponents of the Spectral Dead Zone Transition | 2 | 🆕 nouveau |
| `reviewer2_finite_size_scaling.py` | Reviewer 2 Defense - Finite Size Scaling & Thermodynamic Limit | 3 | 🆕 nouveau |
| `reviewer2_initial_conditions.py` | Reviewer 2 Defense - Initial Conditions (Symmetry Breaking) | 3 | 🆕 nouveau |
| `reviewer2_kuramoto.py` | Reviewer 2 Defense - Kuramoto Order Parameter | 2 | 🆕 nouveau |
| `reviewer2_linear_stability.py` | Reviewer Defense - Linear Stability Analysis of the Isolated FHN Node | 2 | 🆕 nouveau |
| `reviewer2_sigmoid_robustness.py` | Reviewer 2 Defense - Sigmoid Robustness (No Fine-Tuning) | 3 | 🆕 nouveau |
| `reviewer2_spectral_gap.py` | Reviewer 2 Defense - Spectral Gap (Fiedler Value) | 3 | 🆕 nouveau |
| `reviewer2_stiffness_proof.py` | Reviewer 2 Defense - Stiffness & Euler Stability | 3 | 🆕 nouveau |
| `reviewer2_transfer_entropy.py` | Reviewer 2 Defense - Causal Illusion (Transfer Entropy) | 2 | 🆕 nouveau |
| `reviewer2_traveling_waves.py` | Reviewer 2 Defense - Zero-Lag Blindness & Traveling Waves | 3 | 🆕 nouveau |
| `rk4_vs_euler.py` | RK4 vs Euler -- Validation de l'integrateur (DeepSeek review item 8) | — | 🆕 nouveau |
| `run_spice_test.py` | import subprocess | — | 🆕 nouveau |
| `scale_test_10k.py` | import sys | — | 🆕 nouveau |
| `spectral_normalization_test.py` | Test spectral normalization on the topological dead zone. | — | 🆕 nouveau |
| `spice_19ter_robustness.py` | SPICE mismatch robustness — P4.19ter | — | 🆕 nouveau |
| `spice_dead_zone_test.py` | SPICE dead-zone test — does BA m=5 collapse in analog hardware too? | — | 🆕 nouveau |
| `spice_mismatch_50seeds.py` | 50-seed Monte Carlo validation at 3 critical points (KIMI critique #3). | — | 🆕 nouveau |
| `spice_mismatch_cmos_realistic.py` | CMOS-realistic mismatch sweep — P4.19 rebutted against KIMI critique #2. | — | 🆕 nouveau |
| `spice_mismatch_reanalyze.py` | Re-analyze cached P4.19 sweep with the new Phase 5 entropy metrics. | — | 🆕 nouveau |
| `spice_mismatch_sweep.py` | SPICE mismatch sweep — characterize the escape curve from the dead zone. | — | 🆕 nouveau |
| `spice_noise_calibration.py` | Faille A (Audit Manus 2026-04-25) -- Calibration eta_SPICE <-> sigma_Python | — | 🆕 nouveau |
| `spice_noise_resonance.py` | SPICE noise / mismatch resonance test on the BA m=5 dead zone. | — | 🆕 nouveau |
| `spice_p420_hfo2_memristor.py` | P4.20 — HfO₂ memristor compact model : neurone (A), synapse (B), combiné (A+B). | — | 🆕 nouveau |
| `spice_validation.py` | SPICE vs Python validation for Mem4ristor v3 — Hardware feasibility check. | — | 🆕 nouveau |
| `v4_dynamic_heretics_emergence.py` | V4 Dynamic Heretics — Emergence Experiment | — | 🆕 nouveau |
| `v4_high_uthr_figure.py` | V4 High u_thr — Figure de synthèse | — | 🆕 nouveau |
| `v4_high_uthr_sweep.py` | V4 Dynamic Heretics — Exploration régime u_thr > 0.9 | — | 🆕 nouveau |
| `v4_parametric_sweep.py` | V4 Dynamic Heretics — Parametric Sweep | — | 🆕 nouveau |
| `verify_quick_start.py` | import sys | — | 🆕 nouveau |

## Fichiers supprimés depuis la dernière mise à jour

- `> Mis à jour : 2026-05-02 10:26` — présent dans l'ancienne table, introuvable sur disque

## Prochaines expériences prévues

_(à compléter manuellement ou via --mode summaries après ajout de scripts)_

> Généré par brain_watcher.py --mode summaries le 2026-05-02 10:26

> Généré par brain_watcher.py --mode summaries le 2026-05-02 10:26