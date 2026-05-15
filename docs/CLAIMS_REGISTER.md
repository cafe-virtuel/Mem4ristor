# CLAIMS REGISTER — MEM4RISTOR Preprint

> Principe : tout claim numérique publié dans preprint.tex doit avoir un script de vérification listé ici.
> Zéro valeur sans script reproductible. Mis à jour à chaque session.
> Dernière mise à jour : 2026-05-15 (Session C — SPICE Kirchhoff)

---

## Claims Primaires (n = 10 seeds)

| # | Claim | Valeur | Script | Seeds | Dernière exécution | Statut |
|---|-------|--------|--------|-------|-------------------|--------|
| C01 | H_stable lattice 10×10 | 4.06 ± 0.08 bits | `experiments/verify_table1_preprint.py` | 10 | 2026-05-06 | ✅ vérifié |
| C02 | H_stable lattice 4×4 | 3.22 ± 0.14 bits | `experiments/verify_table1_preprint.py` | 10 | 2026-05-06 | ✅ vérifié |
| C03 | H_stable lattice 25×25 | 4.28 ± 0.06 bits | `experiments/verify_table1_preprint.py` | 10 | 2026-05-06 | ✅ vérifié |
| C04 | sync FULL vs FROZEN (+985%) | FULL=0.067, FROZEN=0.730 | `experiments/p2_sigma_social_ablation.py` | 10 | 2026-05-06 | ✅ vérifié |
| C05 | λ₂_crit midpoint | 2.31 (intervalle 2.13–2.50) | `experiments/p2_edge_betweenness.py` (régression logistique) | 36 obs | 2026-04-27 | ✅ vérifié |
| C06 | α_crit Hopf | 0.295 (preprint claim 0.296, écart 0.3%) | `experiments/reviewer2_linear_stability.py` | 1 | 2026-05-02 | ✅ vérifié |
| C07 | pairwise synchrony lattice FULL | 0.031 ± 0.034 | `experiments/verify_table1_preprint.py` | 10 | 2026-05-06 | ✅ vérifié |
| C08 | MI FROZEN_U / FULL ratio (BA m=3) | 2.2× (0.870 → 1.958) | `experiments/p2_spatial_mutual_information.py` | 3 ⚠️ | 2026-04-24 | ⚠️ 3 seeds |
| C09 | ART hard H_min_post vs V4 | +0.40 bits (3.12 vs 2.72) | `experiments/p2_art_benchmark.py` | 10 | 2026-05-11 | ✅ vérifié |
| C10 | Combi metacog+compart | +0.49 bits additif, synergie ≈ 0 | `experiments/p2_v5_combination.py` | 10 | 2026-05-11 | ✅ vérifié |
| C11 | ART soft H_min_post circuit SPICE | ratio SPICE/V4=1.490 = Python/V4=1.490 (accord parfait) | `experiments/spice_art_kirchhoff.py` | 1 (seed=42) | 2026-05-15 | ✅ vérifié |

---

## Claims Secondaires / Exploratoires (n = 3–5 seeds)

Ces claims sont documentés comme directionnels dans preprint.tex §8 (Limitations).
Replication à n=10 planifiée pour révision future.

| # | Claim | Valeur | Script | Seeds | Statut |
|---|-------|--------|--------|-------|--------|
| S01 | Stochastic resonance dead zone escape | H: ~1.40 → ~3.3–3.7 bits à η=0.1 | `experiments/p2_stochastic_resonance_directed.py` | 3 | ⚠️ directional |
| S02 | SR topology variants | Tous types brisent dead zone à η=0.1+ | `experiments/p2_stochastic_resonance_topology.py` | 5 | ⚠️ directional |
| S03 | γ sweep optimal γ*(m) | γ*(m=2)=0.7, γ*(m=3)=0.9 | `experiments/limit02_alpha_sweep.py` | 3 | ⚠️ directional |
| S04 | ART hard SPICE vs Python divergence | SPICE ratio=4.43 vs Python=1.49 (rétroaction implicite trap > Euler) | `experiments/spice_art_kirchhoff.py` | 1 (seed=42) | 🔍 exploratoire |

---

## Claims Preprint Supprimés / Corrigés

| Claim supprimé | Raison | Session |
|----------------|--------|---------|
| Section Floquet (|μ|~10³–10⁴) | Jacobien donne |μ|≈0.20 — contradictoire | 2026-05-06 |
| τ_u = 12.5 | Erreur — valeur correcte est 10.0 | 2026-05-06 |
| "36 configurations" | Doublement faux — correct : 12 topologies × 3 seeds | 2026-05-06 |
| "complete separation" logistic regression (3 seeds) | Corrigé → 10 seeds + Albert & Anderson 1984 | 2026-05-06 |

---

## Processus de mise à jour

Avant tout push Zenodo ou soumission journal :
1. Vérifier que tous les claims ✅ ont leur script qui tourne sans erreur
2. Relancer les claims ⚠️ à n=10 (voir @TODO dans les scripts concernés)
3. Mettre à jour les dates "Dernière exécution" après chaque vérification
4. Ajouter tout nouveau claim numérique dans ce registre avant de l'insérer dans le preprint

Ce fichier peut être scanné par la Ronde Perpétuelle (brain_watcher.py).
