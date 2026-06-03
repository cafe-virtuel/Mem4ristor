# CLAIMS REGISTER — MEM4RISTOR Preprint

> Principe : tout claim numérique publié dans preprint.tex doit avoir un script de vérification listé ici.
> Zéro valeur sans script reproductible. Mis à jour à chaque session.
> Dernière mise à jour : 2026-06-03 (Claude Cowork — C20 marqué À RÉVISER après contre-expertise POC #1 v2)

---

## Claims Primaires (n = 10 seeds)

| # | Claim | Valeur | Script | Seeds | Dernière exécution | Statut |
|---|-------|--------|--------|-------|-------------------|--------|
| C01 | H_stable lattice 10×10 | 4.06 ± 0.08 bits | `experiments/verify_table1_preprint.py` | 10 | 2026-05-06 | ✅ vérifié |
| C02 | H_stable lattice 4×4 | 3.22 ± 0.14 bits | `experiments/verify_table1_preprint.py` | 10 | 2026-05-06 | ✅ vérifié |
| C03 | H_stable lattice 25×25 | 4.28 ± 0.06 bits | `experiments/verify_table1_preprint.py` | 10 | 2026-05-06 | ✅ vérifié |
| C04 | sync FULL vs FROZEN (+985%) | FULL=0.067, FROZEN=0.730 | `experiments/p2_sigma_social_ablation.py` | 10 | 2026-05-06 | ✅ vérifié |
| C05 | λ₂_crit midpoint (EBC) | 2.31 (intervalle 2.13–2.50) | `experiments/p2_edge_betweenness.py` (régression logistique sur EBC) | 36 obs | 2026-04-27 | ✅ vérifié — voir note (*) |
| C06 | α_crit Hopf | 0.295 (preprint claim 0.296, écart 0.3%) | `experiments/reviewer2_linear_stability.py` | 1 | 2026-05-02 | ✅ vérifié |
| C07 | pairwise synchrony BA m=3 FORCED FULL | 0.031 ± 0.034 | `experiments/ablation_coordination_topology_sweep.py` (BA_m3 FORCED FULL) | 10 | 2026-05-06 | ✅ vérifié — **topologie corrigée** (était "lattice", vraie = BA m=3) |
| C07b | pairwise synchrony lattice 10×10 FULL | 0.0197 ± 0.0142 | `experiments/verify_table1_preprint.py` (lattice_10x10) | 10 | 2026-05-06 | ✅ vérifié — **valeur réelle lattice** (différente de C07 = BA m=3) |
| C08 | MI FROZEN_U / FULL ratio (lattice distance=1) | 2.25× (0.870 → 1.958) | `experiments/p2_spatial_mutual_information.py` (lattice) | 5 | 2026-04-24 | ✅ vérifié — **topologie/seeds corrigés** (était "BA m=3 3 seeds", vraie = lattice 5 seeds) |
| C08b | MI FROZEN_U / FULL ratio (BA m=3 distance=1) | 1.84× (1.031 → 1.894) | `experiments/p2_spatial_mutual_information.py` (ba_m3) | 4-5 | 2026-04-24 | ✅ vérifié — **ratio BA m=3 séparé** |
| C09 | ART hard H_min_post vs V4 | +0.40 bits (3.12 vs 2.72) | `experiments/p2_art_benchmark.py` | 10 | 2026-05-11 | ✅ vérifié |
| C10 | Combi metacog+compart | +0.49 bits additif, synergie ≈ 0 | `experiments/p2_v5_combination.py` | 10 | 2026-05-11 | ✅ vérifié |
| C11 | ART soft H_min_post circuit SPICE | ratio SPICE/V4=1.490 = Python/V4=1.490 (accord parfait) | `experiments/spice_art_kirchhoff.py` | 1 (seed=42) | 2026-05-15 | ✅ vérifié |
| ~~C12~~ | ~~Transition thermodynamique du FSS via Binder U4~~ | ~~λ2_crit ≈ 2.31, convergence vers minimum U4~~ | ~~`experiments/v6_binder_cumulant_u4.py`~~ | ~~40~~ | ~~2026-05-19~~ | **🚨 INFIRMEE 2026-06-01** — U4 plat (0.6641–0.6666, variation 0.38%), pas de minimum. Section Binder FSS retirée du preprint V6.0.0 par AUDIT-017. Voir ligne supprimés ci-dessous. |
| C13 | LZ76 regime classification (adaptive D(u)=0.50·u) | LZ=0.88→0.65 (-27%) à m=6 ; structuré jusqu'à m=10 (LZ=0.58) | `experiments/fss_lz_sweep.py` | 5 (re-productible avec 3 seeds aussi) | 2026-06-01 | ✅ vérifié — voir note LZ |
| ⚠️ C20 | **Intrinsic oscillator — narrow-band endogenous rhythm** | ~~f≈0.01 cycles/step (période 70–100 pas) ; LZ spontané > 1.0 partout ; AC@lag50 = +0.6 à +0.7 ; reproductible sur 5 seeds × 2 topologies (m=3, m=6) × 3 D~~ | `experiments/poc1_test_of_absence.py` → **remplacé par `poc1_absence_v2.py`** | 5 | 2026-06-03 | **⚠️ À RÉVISER — contre-expertise Claude Cowork 2026-06-03.** (1) LZ mesuré sur v_mean (T,1) ≠ LZ sur matrice (T,N) des exps principales — incomparable. LZ_state correct : 0.73 à D=0 (structuré trivial — FHN découplés), 1.71–2.19 à D>0 (chaotique, > 0.85). (2) Fréquence surestimée 5× : f_fft≈0.002, pas 0.01. (3) À D=0 seulement : oscillation FHN indépendante, topologie sans effet. (4) C20 tel que publié est indéfendable. Ce qui reste valide : AC@lag50 positif (0.57–0.74) = persistance temporelle réelle. Voir `figures/poc1_v2_raw.csv` + `poc1_v2.png`. |

---

## Claims Secondaires / Exploratoires (n = 3–5 seeds)

Ces claims sont documentés comme directionnels dans preprint.tex §8 (Limitations).
Replication à n=10 planifiée pour révision future.

| # | Claim | Valeur | Script | Seeds | Statut |
|---|-------|--------|--------|-------|--------|
| C21 | **POC C Sweep — seuil de bascule INTRINSIC → F_DRIVE** | Bascule seulement à D=0.5 : m=3 à **8%** (3/5 seeds), m=6 à **10%** (3/5 seeds). À D=0.0 et D=0.15 : INTRINSIC dominant sur toute la plage 1–10%. Transition graduelle (percolation), pas abrupte. AC@lag50 décroît monotonement avec n_pivots dans tous les cas. | `experiments/poc_c_sweep.py` | 5 | 2026-06-03 | ✅ vérifié — session Claude Cowork 2026-06-03. Voir CSV `figures/poc_c_sweep_agg.csv` + figure `figures/poc_c_sweep.png` |
| S01 | Stochastic resonance dead zone escape | H: ~1.40 → ~3.3–3.7 bits à η=0.1 | `experiments/p2_stochastic_resonance_directed.py` | 3 | ⚠️ directional |
| S02 | SR topology variants | Tous types brisent dead zone à η=0.1+ | `experiments/p2_stochastic_resonance_topology.py` | 5 | ⚠️ directional |
| S03 | γ sweep optimal γ*(m) | γ*(m=2)=0.7, γ*(m=3)=0.9 | `experiments/limit02_alpha_sweep.py` | 3 | ⚠️ directional |
| S04 | ART hard SPICE vs Python divergence | SPICE ratio=4.43 vs Python=1.49 (rétroaction implicite trap > Euler) | `experiments/spice_art_kirchhoff.py` | 1 (seed=42) | 🔍 exploratoire |

---

## Claims Preprint Supprimés / Corrigés

| Claim supprimé | Raison | Session |
|----------------|--------|---------|
| Section Floquet (\|μ\|~10³–10⁴) | Jacobien donne \|μ\|≈0.20 — contradictoire | 2026-05-06 |
| τ_u = 12.5 | Erreur — valeur correcte est 10.0 | 2026-05-06 |
| "36 configurations" | Doublement faux — correct : 12 topologies × 3 seeds | 2026-05-06 |
| "complete separation" logistic regression (3 seeds) | Corrigé → 10 seeds + Albert & Anderson 1984 | 2026-05-06 |
| **C12 — Transition thermodynamique FSS via Binder U4** | **INFIRMEE 2026-06-01** — U4 plat (0.6641–0.6666, variation 0.38%), pas de minimum. CSV `v6_binder_cumulant_U4.csv` ne contient que 4 bins lambda2 (9 sims), pas 40. Section Binder FSS retirée du preprint V6.0.0 par AUDIT-017 (31 mai 2026). | 2026-06-01 |
| **C05b — λ₂_crit combined (3.14, CI95 [1.93, 4.85])** | **Non-canonique** — valeur alternative existe dans `lambda2_crit_regression.csv` mais n'est pas publiée. La valeur canonique preprint est l'EBC-only = 2.31. Conservée comme mention secondaire pour traçabilité. | 2026-06-01 |

---

## Notes d'audit Hermes M3 (2026-06-01)

**C05 note (\*)** : Le CSV `lambda2_crit_regression.csv` contient DEUX estimations :
- **EBC-only** (régression logistique sur la métrique EBC) : midpoint = 2.3149, gap [2.126, 2.504], n=36 → **CANONIQUE préprint**
- **Combined** (EBC + autres métriques) : midpoint = 3.140, CI95 [1.928, 4.845], n=58, σ_boot=0.746 → valeur alternative non publiée

La valeur 2.31 est plus conservative et soutenue par un IC resserré. Le 3.14 a un IC très large (3 unités de lambda2) qui reflete l'instabilité de la regression combinée. **Recommandation preprint : garder 2.31 (EBC).**

**C07 / C07b** : Le claim "pairwise synchrony = 0.031 ± 0.034" refere a BA m=3 en regime FORCED (I_stim=0.5), PAS au lattice. La valeur lattice 10×10 = 0.0197 ± 0.0142 est 36% plus faible. Le preprint.tex (lignes 241 & 375) doit clarifier la topologie.

---

## Processus de mise à jour

Avant tout push Zenodo ou soumission journal :
1. Vérifier que tous les claims ✅ ont leur script qui tourne sans erreur
2. Relancer les claims ⚠️ à n=10 (voir @TODO dans les scripts concernés)
3. Mettre à jour les dates "Dernière exécution" après chaque vérification
4. Ajouter tout nouveau claim numérique dans ce registre avant de l'insérer dans le preprint

Ce fichier peut être scanné par la Ronde Perpétuelle (brain_watcher.py).
