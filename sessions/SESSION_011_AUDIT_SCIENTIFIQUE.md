# SESSION 011 — Audit Scientifique Froid (Phase 2/3)

**Date** : 2026-06-01
**Agent** : Hermes (M3)
**Type** : EDISON stop-rule sur 12 claims primaires + 4 secondaires + 1 finding EDISON (glassy)
**Mandat** : Julien Chauvin — Phase 2 du "auditer totalement TEST_HERMES"

---

## 0. Resume executif

**12 claims auditees. Resultats :**

| Categorie | Count | Verdict |
|-----------|-------|---------|
| CONFORMES (delta < tolerance) | 8 | C01, C02, C03, C04, C06, C09, C10, C11 |
| DISCREPANCY mineure (delta > tolerance mais recoverable) | 1 | **C07** (topologie erronee + valeur lattice reelle differente) |
| DISCREPANCY majeure / Claim infirmee | 1 | **C12** (Binder U4 plat, claim "minimum" non observe) |
| Claim a verifier mais script manquant | 1 | C05 (script `p2_edge_betweenness.py` absent, mais CSV `lambda2_crit_regression.csv` existe) |
| AVERTISSEMENT (n=3 seeds, pas 10) | 1 | C08 (MI 5 seeds lattice, pas 3 seeds BA m=3 comme claim) |
| Verifie via CSV mais condition experimentale non confirmee | 1 | S04 (SPICE vs Python ratio = infirme SPICE=4.43) |

**Plus 1 finding EDISON partiellement verifie** (glassy dynamics) — pattern qualitatif OK, valeurs quantitatives 5-10x differentes.

**Risque global** : le preprint est globalement defensible. **2 corrections obligatoires** (C07, C12) + **1 reaffirmation a faire** (glassy dans Discussion).

---

## 1. Resultats detailles par claim

### C01 : H_stable lattice 10×10
- **Valeur claim** : 4.06 ± 0.08 bits, 10 seeds
- **CSV source** : `figures/p2_table1_lattice.csv`
- **Valeur observee** : 4.0560 ± 0.0849 bits
- **Statut** : ✅ **CONFORME**

### C02 : H_stable lattice 4×4
- **Valeur claim** : 3.22 ± 0.14 bits
- **CSV source** : `figures/p2_table1_lattice.csv`
- **Valeur observee** : 3.2246 ± 0.1409 bits
- **Statut** : ✅ **CONFORME**

### C03 : H_stable lattice 25×25
- **Valeur claim** : 4.28 ± 0.06 bits
- **CSV source** : `figures/p2_table1_lattice.csv`
- **Valeur observee** : 4.2767 ± 0.0587 bits
- **Statut** : ✅ **CONFORME**

### C04 : sync FULL vs FROZEN (+985%)
- **Valeur claim** : FULL=0.067, FROZEN=0.730, +985%
- **CSV source** : `figures/p2_sigma_social_ablation.csv`
- **Valeur observee** : FULL=0.0673±0.025, FROZEN=0.7302±0.078, delta = +985.3%
- **Statut** : ✅ **CONFORME**

### C05 : λ₂_crit midpoint = 2.31 (2.13–2.50)
- **Valeur claim** : 2.31 (intervalle 2.13–2.50), 36 obs
- **Script** : `experiments/p2_edge_betweenness.py` ⚠️ **MANQUANT** dans le working tree (renomme probablement en `p2_edge_betweenness_analysis.py`)
- **CSV source** : `figures/lambda2_crit_regression.csv` (regression logistique via EBC)
- **Valeur observee** : `lambda2_crit_ebc_midpoint = 2.3149`, `gap_lo_ebc = 2.1261`, `gap_hi_ebc = 2.5037` — **CONFORME pour EBC seul**
- **AVERTISSEMENT** : Il existe aussi une valeur `lambda2_crit_combined = 3.14` (CI95 [1.93, 4.85]) qui utilise un dataset plus large (58 obs vs 36). Si la regression combinee est canonique, le claim "2.31" est trop optimiste.
- **Statut** : ⚠️ **CONFORME sur EBC, mais valeur alternative (3.14) existe et pourrait etre plus robuste** — a clarifier avec l'agent qui a fait `p2_edge_betweenness_analysis.py`

### C06 : α_crit Hopf = 0.295 (preprint claim 0.296, ecart 0.3%)
- **Valeur claim** : 0.295 (ecart 0.3% avec 0.296 du preprint)
- **CSV source** : `figures/reviewer2_linear_stability.csv` (201 lignes, α ∈ [0, ~0.3])
- **Valeur observee** : `re_lambda_max` NEGATIF jusqu'a α=0.292, POSITIF a α=0.295. **alpha_crit observe = 0.295 (binary search dans le CSV)**
- **Statut** : ✅ **CONFORME** (0.295 vs 0.296 = 0.3% d'ecart, dans la tolerance)

### C07 : pairwise synchrony lattice FULL = 0.031 ± 0.034  🚨 DISCREPANCY
- **Valeur claim** : 0.031 ± 0.034, lattice FULL
- **CSV source cherche** : `figures/p2_table1_sync.csv` (lattice 10×10) = **0.0197 ± 0.0142** — ne correspond PAS
- **Vraie source du 0.031** : `figures/ablation_coordination_topo_sweep.csv` ligne **BA_m3 FORCED FULL** = 0.0307 ± 0.0340
- **DISCREPANCY majeure** :
  1. **Mauvaise topologie** : 0.031 ± 0.034 = BA m=3 en regime FORCED (I_stim=0.5), pas lattice
  2. **Mauvaise valeur pour le lattice** : la vraie valeur lattice 10×10 = 0.0197 ± 0.0142 (~36% plus faible)
- **Origine** : confusion entre `p2_table1_sync.csv` (lattice) et `ablation_coordination_topo_sweep.csv` (BA m=3 forced) dans le Claims Register
- **Impact** : si le lecteur reporte C07 sur la table lattice du preprint, la claim est numeriquement fausse d'environ 0.011 et topologiquement fausse
- **Action** : **OBLIGATOIRE** — corriger l'intitule C07 dans CLAIMS_REGISTER.md en "BA m=3 FORCED FULL = 0.031 ± 0.034" OU recalculer la valeur lattice reelle (0.0197 ± 0.0142) et l'aligner avec le preprint

### C08 : MI FROZEN_U / FULL ratio (BA m=3) = 2.2× (0.870 → 1.958)
- **Valeur claim** : 2.2×, BA m=3, 3 seeds ⚠️
- **CSV source** : `figures/p2_spatial_mutual_information.csv`
- **Valeur observee** : 
  - Lattice FULL distance=1 : MI = 0.870
  - Lattice FROZEN_U distance=1 : MI = 1.958
  - **Ratio = 1.958 / 0.870 = 2.25×** ✅
  - **Mais** : c'est sur **lattice**, pas BA m=3. Sur BA m=3, FULL=1.031, FROZEN=1.894, ratio = 1.84× (different)
  - Et c'est 5 seeds (lattice) ou 4-5 seeds (BA m=3) — pas 3
- **Statut** : ⚠️ **CONFORME sur le ratio, mais topologie reelle = lattice, seeds = 5 (pas 3)**
- **Action** : reclamer C08 "lattice distance=1, 5 seeds" — la topologie et le nombre de seeds dans le Claims Register sont errones

### C09 : ART hard H_min_post vs V4 = +0.40 bits
- **Valeur claim** : +0.40 bits (3.12 vs 2.72), 10 seeds
- **CSV source** : `figures/p2_art_benchmark.csv`
- **Valeur observee** : V4 H_min_post=2.7151, ART hard=3.1234, delta = +0.4083
- **Statut** : ✅ **CONFORME**

### C10 : Combi metacog+compart = +0.49 bits additif
- **Valeur claim** : +0.49 bits additif, synergie ≈ 0, 10 seeds
- **CSV source** : `figures/p2_v5_combination.csv`
- **Valeur observee** : V4=4.0346, Metacog=4.5154 (delta=+0.4808), Compart=4.0491 (delta=+0.0145), Combi=4.5249 (delta=+0.4903)
- **Synergie** = 0.4903 - 0.4808 - 0.0145 = -0.005 bits (négligeable)
- **Statut** : ✅ **CONFORME** (additivite quasi-parfaite)

### C11 : ART soft H_min_post circuit SPICE = ratio 1.490 = 1.490
- **Valeur claim** : ratio SPICE/V4=1.490 = Python/V4=1.490, seed=42
- **CSV source** : `figures/spice_art_kirchhoff.csv`
- **Valeur observee** :
  - V4 pur : H_min_post_spice=0.569, H_min_post_python=0.569, delta=0%
  - ART soft : H_min_post_spice=0.8476, H_min_post_python=0.8476, delta=0%
  - ART hard : H_min_post_spice=2.5219, H_min_post_python=0.8476, delta=197.54% ⚠️
  - **Ratio ART_soft/V4** : SPICE = 0.8476/0.569 = 1.490 ✅ ; Python = 0.8476/0.569 = 1.490 ✅
- **Statut** : ✅ **CONFORME sur C11** (l'accord parfait ART soft)
- **Note** : S04 (ART hard) montre que SPICE diverge fortement du Python (197% vs 0%) — claim exploratoire preserve

### C12 : Transition thermodynamique du FSS via Binder U4  🚨 CLAIM INFIRMEE
- **Valeur claim** : λ2_crit ≈ 2.31, **convergence** du cumulant de Binder U4, 40 sims
- **CSV source** : `figures/v6_binder_cumulant_U4.csv` (4 bins, 9 sims TOTALES) + `figures/v6_binder_cumulant_raw.csv` (15 sims)
- **Valeur observee** :
  - U4 range = 0.6641 - 0.6666 (variation 0.0025, soit **0.38% de variation**)
  - U4_mean par bin lambda2 : 1.27→0.6659, 1.94→0.6641, 2.68→0.6664, 5.03→0.6666
  - **PAS de minimum**, **PAS de convergence vers 2/3 (deja a 2/3 partout)**, **PAS de transition observee**
- **Confirmation EDISON/SYNAPSE 16 mai** : "U4 plat partout, claim 'minimum' infirmee"
- **Mensonge dans le Claims Register** : 40 sims annoncees, 9-15 reelles (datasource : figures/)
- **Statut** : 🚨 **CLAIM INFIRMEE** — la section Binder FSS du preprint doit etre retirees ou reformulee en "U4 est constant (=2/3) sur toute la plage λ2 ∈ [1.3, 5.0], pas de signature de transition de phase thermodynamique"
- **Origine** : v6_binder_cumulant_u4.py a ete execute avec un nombre de sims insuffisant (mauvais plan factoriel) ou les resultats ont ete perdus. La campagne J (1800 sims) a montre la meme platitude avec 10x plus de donnees, ce qui a conduit a AUDIT-017 (mai 31) qui a declare l'arret.
- **Action** : **CRITIQUE** — verifier que le preprint V6 n'utilise plus la section Binder FSS. Si oui, C12 doit etre supprimee du Claims Register, pas marquee "verifiee".

### S01-S03 : Claims secondaires ⚠️
- **S01, S02, S03** : non verifiees dans cette passe (sweep stochastique et gamma sweep). Scripts existent. Statut maintenu a "directional" dans le Claims Register.

### S04 : ART hard SPICE vs Python divergence = 4.43 vs 1.49
- **Valeur claim** : SPICE ratio=4.43 vs Python=1.49 (retroaction implicite trap > Euler)
- **CSV source** : `figures/spice_art_kirchhoff.csv`
- **Valeur observee** : ART hard SPICE H_min_post=2.5219 vs Python=0.8476. **Ratio SPICE/V4 = 2.5219/0.569 = 4.43** ✅, **Ratio Python/V4 = 0.8476/0.569 = 1.49** ✅
- **Statut** : ✅ **CONFORME** sur les ratios, mais claim preserve comme "exploratoire"

---

## 2. Finding EDISON 2026-06-01 — Glassy dynamics (verifie)

### Claim EDISON
La variance de H_stable et LZ montrent des patterns opposes en fonction de lambda2 :
- H_stable variance PEAK dans la zone critique (raw_std=0.544)
- LZ variance MONOTONE CROISSANT avec lambda2 (0.056 → 0.076 → 0.090)

### Verification
Code : `figures/campaign_j_agg.csv` agrege par zone lambda2 (SPARSE <=1.8, CRITICAL ∈[2.0,2.8], DENSE >=3.0), N=1800 sims

| Zone | n_seeds | H_mean | **H_std** | LZ_mean | **LZ_std** |
|------|---------|--------|-----------|---------|------------|
| SPARSE | 242 | 3.6087 | **0.0609** | 0.8257 | **0.0078** |
| CRITICAL | 383 | 3.2435 | **0.1044** | 0.8126 | **0.0106** |
| DENSE | 1165 | 2.6697 | **0.0721** | 0.7558 | **0.0195** |

### Verdict
- **Pattern qualitatif CONFIRME** : H_std peak en CRITICAL (0.104 > 0.061 et 0.072) ✅, LZ_std monotone croissant (0.008 → 0.011 → 0.020) ✅
- **Mais** : les **valeurs numeriques** d'EDISON (0.379, 0.544, 0.265 pour H_std) sont **5-10x superieures** aux miennes (0.061, 0.104, 0.072)
- **Hypothese** : EDISON a probablement utilise `campaign_j_raw.csv` (non agrege, 248 KB) au lieu de `campaign_j_agg.csv` (agrege par bin, 21 KB), et a calcule le std de H_stable ACROSS bins plutot qu'au sein d'un bin. Cela expliquerait un std 5-10x plus grand (variance BETWEEN bins > variance WITHIN bin).
- **Recommandation** : Le pattern glassy **est robuste dans la direction**, mais la formulation EDISON est imprecise. Reformulation suggeree : "Variance between-zone de H_stable (0.07 a 0.10 selon la zone) avec peak dans la zone critique, combinee a LZ_std monotone croissant, suggere une dynamique de type vitreux."
- **Statut** : ⚠️ **PATTERN CONFIRME, FORMULATION A REPRENDRE** — la discussion preprint devrait ajouter une mention courte.

---

## 3. Synthese globale

### Verdict par claim

| Claim | Statut | Commentaire |
|-------|--------|-------------|
| C01 | ✅ CONFORME | 4.06 vs 4.06 (delta < 0.01) |
| C02 | ✅ CONFORME | 3.22 vs 3.22 |
| C03 | ✅ CONFORME | 4.28 vs 4.28 |
| C04 | ✅ CONFORME | +985% calcule = +985% claim |
| C05 | ⚠️ CONFORME sur EBC, valeur combined existe (3.14) | Script manquant, mais CSV present |
| C06 | ✅ CONFORME | alpha_crit = 0.295 observe |
| **C07** | 🚨 **DISCREPANCY** | Topologie erronee (BA au lieu de lattice) + valeur lattice reelle differente (0.0197) |
| C08 | ⚠️ CONFORME sur ratio (2.25x) mais topologie et seeds errones | ratio lattice distance=1, 5 seeds |
| C09 | ✅ CONFORME | +0.4083 bits |
| C10 | ✅ CONFORME | +0.4903 bits, synergie = -0.005 |
| C11 | ✅ CONFORME | ART soft 1.490 = 1.490 |
| **C12** | 🚨 **CLAIM INFIRMEE** | U4 plat 0.6641-0.6666, pas de minimum, 9-15 sims (pas 40) |
| S04 | ✅ CONFORME | 4.43 vs 1.49 confirmes |

### Actions obligatoires (3)

1. **C07** : Corriger le Claims Register + le preprint.tex (lignes 241 & 375) — soit changer la topologie (BA m=3 forced), soit utiliser la vraie valeur lattice 0.0197 ± 0.0142.

2. **C12** : Supprimer la ligne C12 du Claims Register. Verifier que la section Binder FSS du preprint a bien ete retirees ou reformulee (AUDIT-017 du 31 mai a fait l'arret mais a-t-il ete pousse en PDF ?). Verifier aussi que le PDF recompile du 1er juin ne contient plus la claim.

3. **C05** : Clarifier si la valeur canonique est 2.31 (EBC) ou 3.14 (combined). Le Claims Register dit 2.31, mais le CSV montre qu'une valeur combined 3.14 existe avec un IC plus large. Lequel est publie dans le preprint ?

### Recommandations fortes (2)

4. **Glassy dynamics** : Ajouter une mention dans la Discussion du preprint (1-2 phrases) sur la variance comportementale. Les donnees supportent le pattern qualitatif. Cela renforce la section "spectral crossover" deja en place.

5. **C08** : Corriger la topologie (lattice au lieu de BA m=3) et le nombre de seeds (5 au lieu de 3) dans le Claims Register.

---

## 4. Ce qui n'a pas ete verifie dans cette passe

- **S01, S02, S03** : claims secondaires sur stochastic resonance et gamma sweep — non auditees ici (sweep long, 80+ runs)
- **Scripts p2_* 30+ sans CSV/PNG dans figures/** : a investiguer en Phase 3 (chasse aux approches)
- **3 papiers LaTeX** (preprint, paper_2, paper_B) — coherence inter-papiers non verifiee

---

## 5. Pour aller plus loin (Phase 3 - chasse aux approches nouvelles)

**Scripts qui meritent attention particuliere :**

A. `experiments/campaign_j_binder_lz.py` (16 KB, mai 31) — source du `campaign_j_agg.csv` qui a infirme C12. **A regarder en priorite**.

B. `experiments/fss_lz_sweep.py` (20 KB, mai 30) — sweep LZ76 le plus recent. Probablement le "regime map" qui remplace C12.

C. `experiments/p2_v5_final_best.py` (12 KB, mai 31) — protocole V5 optimal (D(u)=0.50*u + alpha_meta=-4.0). Titre "FINAL_BEST" est un signal.

D. Branche `feat/kimi-p419-continuous-entropy` (5 commits) — 10 commits, 25-26 avril, corrections metriques. A voir si des fixes metriques n'ont pas ete reportes dans main.

E. `experiments/fss_lambda2_sweep_v2.csv` (271 lignes) — sweep lambda2 le plus recent, plus de donnees que le v6_binder_cumulant.

F. `docs/papers/paper_2/` — paper 2 (Doubt Variable as Anti-Synchronization Filter) qui pourrait avoir des resultats non reportes dans le preprint.

---

*Hermes — 2026-06-01, session 011, modele M3, EDISON stop-rule applique*
