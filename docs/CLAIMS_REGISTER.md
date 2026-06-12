# CLAIMS REGISTER — MEM4RISTOR Preprint

> Principe : tout claim numérique publié dans preprint.tex doit avoir un script de vérification listé ici.
> Zéro valeur sans script reproductible. Mis à jour à chaque session.
> Dernière mise à jour : 2026-06-12 (Claude Code/Fable — AUDIT-024 détecté ET résolu : CSV pré-mai régénérés avec le code actuel, preprint corrigé (Option A validée Julien) ; C20 reformulé ; C21 révisé)
>
> ✅ **AUDIT-024 RÉSOLU (2026-06-12, Option A validée par Julien)** : le commit `818cf67`
> (1er mai) avait introduit le scaling Euler-Maruyama du bruit (×4.47 effectif), rendant les
> CSV pré-mai non reproductibles. **Les 3 CSV canoniques (C01/C04/C08) ont été régénérés avec
> le code actuel et le preprint corrigé** (abstract, table MI, conclusion, limitations +
> convention de bruit explicitée dans les Méthodes). Guardian 13/13. Voir AUDIT_LOG AUDIT-024.

---

## Claims Primaires (n = 10 seeds)

| # | Claim | Valeur | Script | Seeds | Dernière exécution | Statut |
|---|-------|--------|--------|-------|-------------------|--------|
| C01 | H_stable lattice 10×10 | 4.06 ± 0.08 bits (preprint, verify_table1 10 seeds) ; delta sweep δ=0 : 4.157 | `experiments/verify_table1_preprint.py` + `p2_delta_sweep.py` | 10 / 3 | 2026-06-12 | ✅ **RÉGÉNÉRÉ 12/06 (Option A)** : `p2_delta_sweep.csv` refait avec le code actuel (4.157). La valeur preprint 4.06±0.08 vient de verify_table1 (CSV 27/05, code actuel) — elle TIENT, aucune correction preprint nécessaire pour C01 |
| C02 | H_stable lattice 4×4 | 3.22 ± 0.14 bits | `experiments/verify_table1_preprint.py` | 10 | 2026-05-06 | ✅ vérifié |
| C03 | H_stable lattice 25×25 | 4.28 ± 0.06 bits | `experiments/verify_table1_preprint.py` | 10 | 2026-05-06 | ✅ vérifié |
| C04 | sync FULL vs FROZEN (~×90) | FULL=0.0072, FROZEN=0.6582 | `experiments/p2_sigma_social_ablation.py` | **10** | 2026-06-12 | ✅ **RÉGÉNÉRÉ 12/06 (Option A), puis monté à 10 seeds** : FULL strictement stable (0.0072 aux deux tailles), FROZEN 0.6513→0.6582. Preprint corrigé (« +985% » → « ~90-fold, 0.007 → 0.658 »). Anciennes valeurs 0.067/0.730 = ancien bruit (pré-818cf67). Voir AUDIT-024 |
| C05 | λ₂_crit midpoint (EBC) | 2.31 (intervalle 2.13–2.50) | `experiments/p2_edge_betweenness.py` (régression logistique sur EBC) | 36 obs | 2026-04-27 | ✅ vérifié — voir note (*) |
| C06 | α_crit Hopf | 0.295 (preprint claim 0.296, écart 0.3%) | `experiments/reviewer2_linear_stability.py` | 1 | 2026-05-02 | ✅ vérifié |
| C07 | pairwise synchrony BA m=3 FORCED FULL | 0.031 ± 0.034 | `experiments/ablation_coordination_topology_sweep.py` (BA_m3 FORCED FULL) | 10 | 2026-05-06 | ✅ vérifié — **topologie corrigée** (était "lattice", vraie = BA m=3) |
| C07b | pairwise synchrony lattice 10×10 FULL | 0.0197 ± 0.0142 | `experiments/verify_table1_preprint.py` (lattice_10x10) | 10 | 2026-05-06 | ✅ vérifié — **valeur réelle lattice** (différente de C07 = BA m=3) |
| C08 | MI FROZEN_U / FULL ratio (lattice distance=1) | **2.85×** (0.6015 → 1.7118) | `experiments/p2_spatial_mutual_information.py` (lattice) | **10** | 2026-06-12 | ✅ **RÉGÉNÉRÉ 12/06 (Option A), puis monté à 10 seeds** : ratio très stable (2.84× à 3 seeds → 2.85× à 10 seeds). Preprint à jour. Nuance honnête : NO_SIGMOID ≈ FULL (la décorrélation vient de u) ; decay BA négatifs notés. Ancien : 2.25× (CSV 24/04, ancien bruit) |
| C08b | MI FROZEN_U / FULL ratio (BA m=3 distance=1) | **3.41×** (0.4875 → 1.6625) | `experiments/p2_spatial_mutual_information.py` (ba_m3) | **10** | 2026-06-12 | ✅ **RÉGÉNÉRÉ 12/06 (Option A), puis monté à 10 seeds** (3.37× à 3 seeds → 3.41×). Preprint corrigé (était 1.84×, ancien bruit). Mappé Guardian |
| C09 | ART hard H_min_post vs V4 | +0.40 bits (3.12 vs 2.72) | `experiments/p2_art_benchmark.py` | 10 | 2026-05-11 | ✅ vérifié |
| C10 | Combi metacog+compart | +0.49 bits additif, synergie ≈ 0 | `experiments/p2_v5_combination.py` | 10 | 2026-05-11 | ✅ vérifié |
| C11 | ART soft H_min_post circuit SPICE | ratio SPICE/V4=1.490 = Python/V4=1.490 (accord parfait) | `experiments/spice_art_kirchhoff.py` | 1 (seed=42) | 2026-05-15 | ✅ vérifié |
| ~~C12~~ | ~~Transition thermodynamique du FSS via Binder U4~~ | ~~λ2_crit ≈ 2.31, convergence vers minimum U4~~ | ~~`experiments/v6_binder_cumulant_u4.py`~~ | ~~40~~ | ~~2026-05-19~~ | **🚨 INFIRMEE 2026-06-01** — U4 plat (0.6641–0.6666, variation 0.38%), pas de minimum. Section Binder FSS retirée du preprint V6.0.0 par AUDIT-017. Voir ligne supprimés ci-dessous. |
| C13 | LZ76 regime classification (adaptive D(u)=0.50·u) | LZ=0.88→0.65 (-27%) à m=6 ; structuré jusqu'à m=10 (LZ=0.58) | `experiments/fss_lz_sweep.py` | 5 (re-productible avec 3 seeds aussi) | 2026-06-01 | ✅ vérifié — voir note LZ |
| C20 | **Persistance temporelle endogène (révisé)** | AC@lag50 = +0.57 à +0.74 sans drive externe (I_stim=0), reproductible 5 seeds × 2 topologies (m=3, m=6) × 3 D. Le réseau maintient une cohérence temporelle réelle à 50 pas, sans stimulus. | `experiments/poc1_absence_v2.py` | 5 | 2026-06-12 | ✅ **REFORMULÉ 2026-06-12 (Claude Code/Fable)** suite contre-expertise 03/06 : les composantes fréquence (« f≈0.01 » — réel : f_fft≈0.002) et LZ (mesuré sur v_mean, incomparable ; LZ_state>0.85 partout sauf D=0 trivial) sont **retirées du claim**. Ne reste que la persistance AC@lag50, validée par la contre-expertise. L'appellation « intrinsic oscillator » est abandonnée (pas de bande étroite démontrée). Voir `figures/poc1_v2_raw.csv` + `poc1_v2.png` |

---

## Claims Secondaires / Exploratoires (n = 3–5 seeds)

Ces claims sont documentés comme directionnels dans preprint.tex §8 (Limitations).
Replication à n=10 planifiée pour révision future.

| # | Claim | Valeur | Script | Seeds | Statut |
|---|-------|--------|--------|-------|--------|
| C21 | **POC C Sweep v2 — entraînement graduel, PAS de bascule (révisé)** | Avec la métrique FFT correcte : la fréquence dominante reste endogène (f_fft≈0.002) dans 59/60 conditions jusqu'à 10% de pivots. L'entraînement est CONTINU : drive_power_frac croît monotonement avec n_pivots et avec D (max 0.343±0.127 à m=6 D=0.5 10%). Aucun seuil de bascule ≤ 10%. | `experiments/poc_c_sweep_v2.py` | 5 | 2026-06-12 | ⚠️ **RÉVISÉ 2026-06-12 (Claude Code/Fable)** : la « bascule à 8%/10% » de la v1 était un artefact du zero-crossing (harmoniques FHN) + classifieur binaire — exactement le PROBLÈME MAJEUR 1 de la contre-expertise du 03/06. v2 = FFT + drive_power_frac (continu) + LZ_state (T,N). La v1 reste archivée (`poc_c_sweep.py`, `figures/poc_c_sweep_*`). Voir `figures/poc_c_sweep_v2_agg.csv` + `poc_c_sweep_v2.png` |
| S01 | Stochastic resonance dead zone escape | H_cont monotone : 2.88 (σ=0) → 3.16 (σ=0.1) → 4.28 (σ=0.5) en UNIFORM, BA m=5 ; H_cog reste ≈0 ; HERETIC sature à 3.33 | `experiments/p2_stochastic_resonance_directed.py` | **10** | ✅ régénéré 12/06 (10 seeds, code actuel). Ancien (3 seeds, ancien bruit) : « 1.40 → 3.3-3.7 à η=0.1 » — la baseline monte car le bruit de fond par défaut est ×4.47 plus fort (AUDIT-024). Effet directionnel (bruit ↑ H) : CONFIRMÉ |
| S02 | SR topology variants | H monte avec σ sur les 7 topologies ; évasion H_cog>0 à σ≥0.3–0.5 (denses) ; σ*=1.2 partout (saturation de la grille) | `experiments/p2_stochastic_resonance_topology.py` | **10** | ✅ régénéré 12/06 (10 seeds, code actuel). Ancien : « brisent dead zone à η=0.1+ » — seuil décalé vers σ≥0.3 avec le nouveau bruit (AUDIT-024). Effet qualitatif : CONFIRMÉ |
| S03 | γ sweep (régime endogène I_stim=0) | **Résultat négatif renforcé** : aucun γ∈[0,1] ne franchit la dead zone ; m=2 : γ*=1.0, H_cog=0.59 ; m≥3 : H_cog≤0.01. H_cont décline doucement (3.9→2.3 bits) | `experiments/limit02_alpha_sweep.py` (+ CSV ajouté `figures/limit02_alpha_sweep.csv`) | **10** | ✅ régénéré 12/06 (10 seeds, code actuel). ⚠️ L'ancien détail « γ*(m=2)=0.7, γ*(m=3)=0.9, sur-correction » NE SURVIT PAS au nouveau bruit — retiré du preprint (table, narration, contribution (4), conclusion). Le claim central (normalisation γ insuffisante) se RENFORCE |
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
