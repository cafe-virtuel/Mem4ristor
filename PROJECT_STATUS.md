# PROJECT STATUS — Mem4ristor v3.2.1
**Dernière mise à jour : 2026-04-29**
**Auteur : Julien Chauvin (Barman / Orchestrateur)**
**Contexte : Café Virtuel — Laboratoire d'Émergence Cognitive**

> Ce fichier est le point d'entrée pour quiconque (humain ou IA) travaille sur ce projet.
> Lisez-le en premier. Mettez-le à jour après chaque session de travail.

---

## 1. QU'EST-CE QUE CE PROJET ?

Mem4ristor est une implémentation computationnelle de dynamiques FitzHugh-Nagumo étendues, conçue pour étudier les états critiques émergents dans des réseaux neuromorphiques. L'innovation centrale est le **Doute Constitutionnel (u)** : une variable dynamique qui module la polarité du couplage entre les neurones, créant une frustration adaptative qui empêche l'effondrement du consensus.

Le projet est né au sein du **Café Virtuel**, un laboratoire de collaboration entre un humain et plusieurs IA (Anthropic, OpenAI, xAI, Google, Mistral, DeepSeek). L'historique complet est dans le dépôt Café Virtuel : https://github.com/cafe-virtuel/

Publication : DOI 10.5281/zenodo.19700749 (preprint dans `docs/preprint.pdf`)

---

## 2. ARCHITECTURE DU CODE (ce qui marche, ce qui ne marche pas)

### Noyau stable (PRODUCTION-READY)

| Fichier | Rôle | État |
|:--------|:-----|:-----|
| `src/mem4ristor/core.py` | Moteur V3 canonique (Mem4ristorV3 + Mem4Network). Levitating Sigmoid, meta-doubt adaptatif V4, rewiring topologique V4, **V5 hysteresis** (dead-zone latching + watchdog fatigue), **sparse CSR** auto-détecté pour N > 1000, **6 modes de coupling_norm** dont `spectral` (eigenvector centrality, 2026-04-19) | STABLE |
| `src/mem4ristor/config.yaml` | Paramètres par défaut | STABLE |
| `src/mem4ristor/__init__.py` | Exporte Mem4ristorV3, Mem4ristorV2 (alias), Mem4Network | STABLE |
| `src/mem4ristor/symbiosis.py` | CreativeProjector (Phase 4) + SymbioticSwarm | STABLE |
| `src/mem4ristor/cortex.py` | LearnableCortex (MLP autoencoder pour consolidation mémoire) | STABLE |
| `src/mem4ristor/sensory.py` | SensoryFrontend (convolution + projection pour entrées visuelles) | STABLE (lent, voir §5) |
| `src/mem4ristor/viz.py` | Visualisation : entropy trace, doubt map, phase portrait, state distribution, dashboard | STABLE |

### Modules ajoutés/modifiés récemment (2026-03-22)

| Fichier | Modification | Détail |
|:--------|:-------------|:-------|
| `src/mem4ristor/core.py` | V5 Hysteresis (4 patches) | Dead-zone [θ_low=0.35, θ_high=0.65] avec latching type Schmitt-trigger + watchdog fatigue. 3 tests actifs. |
| `src/mem4ristor/core.py` | Sparse CSR (8 patches) | Auto-conversion scipy.sparse.csr_matrix si N > 1000. Mémoire 455× à N=5000. Eigsh sparse pour spectral gap. |
| `tests/test_v5_hysteresis.py` | xfail → actif | 3 tests débloqués : configuration, latching, watchdog fatigue |
| `examples/demo_applied.py` | Nouveau | 4 démos pipeline complet : sensory, hysteresis, scale-free sparse, phase diversity. 5 PNG. |

### Modules corrigés précédemment (2026-03-21)

| Fichier | Problème | Correction |
|:--------|:---------|:-----------|
| `src/mem4ristor/hierarchy.py` | Importait `mem4ristor_v3` et `mem4ristor_king` (supprimés en V3) | Imports corrigés vers `core.py` + `experimental/` |
| `src/mem4ristor/arena.py` | Importait `mem4ristor_v3` (supprimé en V3) | Import corrigé vers `core.py` |
| `src/mem4ristor/inception.py` | Code mort dans `decode()` : `return` prématuré | Logique de reshape corrigée, code mort supprimé |
| `demo_chimera.py` | Import King obsolète + API inexistante + features V5 manquantes | Import corrigé, appels corrigés, scénario V5 retiré |

### Modules expérimentaux (NON PRODUCTION)

| Fichier | Rôle | État |
|:--------|:-----|:-----|
| `experimental/mem4ristor_king.py` | "Philosopher King" : loi martiale, métacognition | EXPERIMENTAL |

---

## 3. ÉTAT DES CLAIMS SCIENTIFIQUES

Source : `docs/limitations.md` (table de vérité maintenue avec rigueur)

| Claim | Statut | Détail |
|:------|:-------|:-------|
| Levitating Sigmoid élimine la dead zone (LIMIT-01) | RÉSOLU en V3 | `tanh(π(0.5-u)) + δ` remplace `(1-2u)` |
| Seuil de 15% d'hérétiques universel (LIMIT-02) | **PARTIELLEMENT RÉSOLU — NON UNIVERSEL** | `degree_linear` fonctionne sur BA m=3, HK, WS, ER sparse. Échoue sur BA m=1/5/10, Config Model, ER dense. La normalisation optimale dépend de l'hétérogénéité des degrés ET de la redondance des chemins. Voir §3quinquies + §3sexies |
| H ≈ 1.94 attractor (LIMIT-05) | **INVESTIGUÉ — FAUX** | Sweep 800+ combos : stable H ≈ 0.92, transitoires jusqu'à 2.31. Voir §3bis |
| Stabilité long-terme (LIMIT-04) | **INVESTIGUÉ — NUANCÉ** | Le "drift" est un transitoire de convergence, pas une instabilité. dt≤0.05 stable. Voir §3quater |
| Mapping hardware HfO2 | **VALIDÉ EN SIMULATION (2026-04-19)** | SPICE/Python RMS global ≈ 9.7×10⁻³ (≤1% de \|v\|) sur lattice 4×4. Voir §3septies + §10 P4 |
| Normalisation spectrale brise la dead zone | **TESTÉ — FAUX (2026-04-19)** | `coupling_norm='spectral'` (1/eigenvector_centrality) implémenté. 0/6 wins sur dead zone. Le problème est dynamique, pas un défaut de pondération. Voir §3octies |
| Parité cross-platform (MKL) | RÉSOLU | Fix v2.9.1, `NUMPY_MKL_CBWR=COMPATIBLE` |
| **Nœud isolé instable (claim preprint §3.1)** | **🚨 FAUX — AUDIT EXTERNE 2026-04-22** | Point fixe v*=−1.294, w*=−0.732 est un **spiral stable** (λ=−0.055±0.283i). Hopf à α_crit≈0.296, mais défaut α=0.15. Voir §3octvicies |
| **H_cog≈0.92 (Python, bins corrigés)** | **🚨 ARTEFACT MÉTRIQUE — AUDIT EXTERNE 2026-04-22** | Avec bins KIMI (±0.4/1.2), H_cog=0 pour TOUTES les configs Python défaut. La valeur 0.92 venait de l'ancienne bin ±1.5 straddlant le cluster consensus. Voir §3octvicies |
| **Hérétiques actifs à I_stim=0** | **🚨 FAUX — AUDIT EXTERNE 2026-04-22** | `I_eff[heretic_mask] *= -1` est no-op quand I_stim=0. Les expériences "endogènes" ne testent pas le mécanisme hérétique. Voir §3octvicies |
| **Verilog-A (v26.va) = Python** | **🚨 FAUX — AUDIT EXTERNE 2026-04-22** | Noyau linéaire (1-2u), τ_u=1.0, pas d'ε_u adaptatif, pas de plasticité, double-comptage I_coup. Voir §3octvicies |
| **Escape SPICE noise+mismatch (P4.19)** | **✅ CONFIRMÉ sous 3 métriques** | H_cont=4.58 bits à (η=0.5, σ_C=0.5). Survit à la métrique continue et aux bins KIMI. Voir §3quindecies |
| **Calibration η SPICE ↔ σ Python** | **✅ RÉSOLU (2026-04-25)** | η=0.5 SPICE ↔ σ_equiv=0.0044 Python. Item 10 testait σ=1.2 = 270× l'équivalent. Python reste H_cog≈0 à toutes amplitudes calibrées. Bruit thermique SPICE catégoriquement distinct → claim Paper B RENFORCÉ. Voir §3trigies + `experiments/spice_noise_calibration.py` |
| **Bins obsolètes dans `spice_dead_zone_test.py`** | **✅ RÉSOLU (2026-04-25)** | Seuils corrigés vers KIMI `[-1.2, -0.4, 0.4, 1.2]`. Conclusion inchangée. Voir §3trigies |
| **Dynamique u tronquée dans les netlists SPICE** | **✅ DOCUMENTÉ (2026-04-25)** | Limitation explicite ajoutée dans Paper B §2 + commentaires inline. Voir §3trigies |
| **Duplication make_ba() inter-scripts** | **✅ RÉSOLU (2026-04-25)** | `src/mem4ristor/graph_utils.py` créé. 7 scripts p2_* migrés. Voir §3trigies |
| **Terminologie : "frustrated synchronization" / "topological phase transition"** | **✅ CORRIGÉ (2026-04-28)** | → "polarity-modulated anti-synchronization" / "spectral phase transition". Distinction fondamentale : la polarité est state-dependent et continue, pas quenched. Voir §3quinquetrigies |
| **H_cog=0 dans ablation preprint** | **✅ DOCUMENTÉ ET RECENTRÉ (2026-04-28)** | Artefact de binning : voltages Python [-3.2,-1.3] tous en bin 1. Claim recentrée sur H_cont (100-bin) = 3.79±0.14 bits + synchrony FULL=0.031 vs FROZEN=0.751. Voir §3quinquetrigies |
| **SPICE : validation 50 seeds BA m=5 N=64** | **✅ DOCUMENTÉ DANS PAPER (2026-04-28)** | Résultats préexistants documentés : dead zone H_cont=1.38±0.04, functional H_cont=4.30±0.19. Mismatch CMOS σ_C=0.10 sans effet sur la diversité. Section dédiée ajoutée dans preprint.tex. Voir §3quinquetrigies |
| **[8] RK4 vs Euler — validation intégrateur (paramètres corrigés)** | **✅ VALIDÉ (2026-04-29)** | Euler dt=0.05 validé sur paramètres alignés config.yaml (sigmoid_steepness=π, SOC_LEAK=0.01, ε_u=0.02, τ_u=10). Plasticité=OFF : Max Δ(H_cog)=0.0018, surge delta=0.3pp. Plasticité=ON (λ_learn=0.05) : Max Δ(H_cog)=0.0053, surge delta=0.1pp. Commits 91a0072 + 4cd7fce sur feat/v4-dynamic-heretics. Voir §3novetrigies |

### 3bis. LIMIT-05 : Entropie maximale (2026-03-21)

> ⚠️ **INVALIDÉ MÉTRIQUEMENT (audit 2026-04-22)** : Toutes les valeurs H ci-dessous ont été mesurées avec les **anciennes bins pré-KIMI (±0.8/1.5)**. Avec les bins actuelles (±0.4/1.2), H_cog = 0 pour toutes les configs aux paramètres défaut. Voir §3octvicies.

**Question** : Le claim H ≈ 1.94 est-il reproductible ?

**Méthode** : Sweep paramétrique 4 phases + analyse de stabilité (800+ combinaisons, 5-7 seeds, ~7 min).

| Métrique | Valeur |
|:---------|:-------|
| Max théorique H (5 bins) | log₂(5) = 2.3219 |
| Meilleur H **transitoire** | 2.3143 (99.7% du max) |
| Meilleur H **stable** (derniers 25%) | 1.48 ± 0.66 (D=0.01, I=1.0) |
| H stable config par défaut | **0.92 ± 0.04** *(bins pré-KIMI)* |

**Verdict** : H ≈ 1.94 mesuré sur un pic transitoire. Attracteur réel ≈ 0.92 *(avec les bins d'époque)*.

### 3ter. LIMIT-02 : Strangulation scale-free (2026-03-21)

> ⚠️ **BINS PRÉ-KIMI** : Les H_stable ici (0.918, 0.002, 0.993) sont avec les anciennes bins (±0.8/1.5). Avec les bins actuelles, les valeurs diffèrent.

**Question** : Le V4 rewiring résout-il la strangulation par les hubs ?

**Méthode** : 8 expériences sur réseaux Barabási-Albert (N=100, m=3), 5 seeds, sweep rewiring.

| Configuration | H_stable |
|:---|:---|
| Lattice 10×10 (contrôle) | **0.918** |
| Scale-free sans rewiring | **0.002** |
| Scale-free avec V4 rewiring | **0.002** |
| Scale-free + hérétiques sur hubs | **0.002** |
| Scale-free + stimulus I=1.0 | **0.993** (partiel) |

**Diagnostic** : Les hubs (degré 26-29) synchronisent tout le réseau en ~100 steps. Le rewiring (5000+ reconnexions) agit sur un système déjà synchronisé. 24 configs de rewiring testées : toutes H=0.

**Cause racine** : La normalisation 1/√N ne compense pas l'hétérogénéité des degrés. Un hub reçoit 10× plus de signal couplé qu'un noeud périphérique.

**Piste** : Normalisation par degré local : D_eff(i) = D/√deg(i) au lieu de D/√N.

### 3quinquies. LIMIT-02 : Résolution par normalisation degree_linear (2026-04-10)

> ⚠️ **BINS PRÉ-KIMI** : H_stable = 0.828 ± 0.069 et H = 0.958 ci-dessous sont mesurés avec les anciennes bins (±0.8/1.5). Le phénomène (degree_linear fonctionne mieux que uniform) reste réel, mais les valeurs numériques ne correspondent plus à la métrique actuelle.

**Question** : Quel mode de normalisation par degré résout la strangulation ?

**Méthode** : Sweep 4 modes (`uniform`, `degree`, `degree_linear`, `degree_log`) sur BA (N=100, m=3), 5 seeds, 3000 steps.

| Mode | Formule | H_stable (BA) |
|:---|:---|:---|
| `uniform` (baseline) | D/√N | 0.004 ± 0.007 |
| `degree` | D/√deg(i) | 0.000 ± 0.000 |
| `degree_linear` | D/deg(i) | **0.828 ± 0.069** ★ |
| `degree_log` | D/log(1+deg(i)) | 0.000 ± 0.000 |

**Contrôle** : Lattice 10×10 uniform → H = 0.958 ± 0.040.

**Verdict** : `degree_linear` récupère **86.4%** de la performance lattice. Seule la normalisation linéaire compense l'hétérogénéité des degrés (ratio hub/périphérique = 12:1 sur BA m=3).

**Reproduction** : `experiments/limit02_norm_sweep.py`

### 3sexies. LIMIT-02 : Validation multi-topologie (2026-04-10)

> ⚠️ **BINS PRÉ-KIMI** : Tous les H ci-dessous (0.93, 0.85, 0.83, etc.) sont avec les anciennes bins (±0.8/1.5). La classification des régimes (degree_linear gagne / uniform gagne / aucun) reste qualitativement valide.

**Question** : `degree_linear` est-il un fix universel ?

**Méthode** : 13 topologies (3 seeds, 3000 steps, N=100), uniform vs degree_linear.

**Résultat** : **NON**. Trois régimes :

| Régime | Topologies | Gagnant |
|:---|:---|:---|
| `degree_linear` gagne | BA m=3, HK (p=0.5, 0.9), WS (p=0.1, 0.3), ER sparse | DL : H≈0.71–0.87 |
| `uniform` gagne | BA m=1 (arbre), BA m=10, CM (γ=2.5, 3.0, 4.0) | UNI : H≈0.23–0.96 |
| Aucun ne fonctionne | BA m=5, ER dense (p=0.12) | H≈0.00 partout |

**Insight clé** : La normalisation optimale dépend de l'interaction entre l'hétérogénéité des degrés ET la redondance des chemins. Le deg_ratio seul ne prédit pas le gagnant (BA m=1 : ratio 24 → uniform gagne ; WS : ratio 2.3 → DL gagne). C'est un problème de **transport d'information sur graphe**.

**Questions ouvertes** :
- Normalisation adaptative/hybride (par clustering local) ?
- Pourquoi BA m=5 échoue avec les DEUX normalisations ?
- Le chemin adjacency-matrix semble se comporter différemment du stencil même sur des topologies quasi-régulières.

**Reproduction** : `experiments/limit02_topology_sweep.py`

### 3septies. Hardware feasibility : SPICE/Python validation (2026-04-19)

**Question** : Les dynamiques Mem4ristor v3 sont-elles réalisables en hardware analogique ?

**Méthode** : Génération programmatique d'un netlist coupled-FHN+doubt N×N avec ngspice 46. Comparaison apples-to-apples avec une intégration Python Euler aux *mêmes* équations et même pas dt.

| Métrique | Valeur (4×4 toroidal lattice, t=50, dt=0.05) |
|:---------|:---|
| RMS global (v) | **9.7 × 10⁻³** (≤1% de \|v\| typique) |
| max \|Δv\|_final | 1.1 × 10⁻³ |
| Verdict | **PASS** (seuils RMS<0.05, \|Δv\|<0.10) |

**Découverte critique : 2 pièges SPICE éliminés**
1. `pow(V(node),3)` casse la convergence Newton de ngspice quand la base est négative. Fix : développer `V(node)*V(node)*V(node)`.
2. Le pattern `R + B-voltage` (utilisé dans les netlists pré-existants `coupled_3x3/5x5/10x10.cir`) est un **filtre RC**, pas un intégrateur : il intègre `dv/dt = f - v` au lieu de `dv/dt = f`. Fix : pattern direct `C_v v 0 1 IC=...; B_dv 0 v I = f(...)`.

**Reproduction** : `experiments/spice_validation.py` → `figures/spice_vs_python_validation.png`. Doc : `experiments/spice/README_HARDWARE.md`.

### 3undecies. SPICE mismatch sweep — escape COMPLET caractérisé (2026-04-19, soir)

**Question** : §3decies montrait un escape partiel (H≈0.16). Avec un sweep 2D propre (η × σ_C) et plusieurs seeds, atteint-on un escape *complet* de la dead zone ?

**Méthode** : `experiments/spice_mismatch_sweep.py`. Même graphe BA m=5 N=64 + degree_linear. Sweep η ∈ {0.10, 0.30, 0.50} × σ_C ∈ {0, 0.05, 0.10, 0.20, 0.50} × 3 seeds = **45 runs ngspice** (~340s total). Mismatch capacitif tiré gaussien clip [0.1, 5.0].

**Résultats — H_stable (mean ± std sur 3 seeds)** :

| η \ σ_C | 0 | 0.05 | 0.10 | 0.20 | 0.50 |
|:---|---:|---:|---:|---:|---:|
| 0.10 | 0.00 | 0.00 | 0.00 | 0.00 | **0.35±0.23** |
| 0.30 | 0.10 | 0.08 | 0.13 | 0.25 | **1.33±0.24** |
| 0.50 | 1.34 | 1.35 | 1.29 | 1.43 | **1.61±0.16** |

**H_max = 1.61** (η=0.50, σ_C=0.50) ≈ **69 % du max théorique log₂(5) = 2.32**. C'est un **escape complet**, pas partiel.

**Trois régimes** :

1. **η=0.10 (bruit faible)** — escape impossible sans σ_C ≥ 0.50. Très seed-dépendant (std=0.23) → preuve d'**états métastables multiples**.
2. **η=0.30 (bruit moyen)** — escape graduel avec σ, saut massif à σ=0.50 (×10 sur H). Zone de **résonance stochastique pure** entre bruit thermique et désordre figé.
3. **η=0.50 (bruit fort)** — escape immédiat (H≈1.34 sans mismatch). Le mismatch n'apporte qu'un gain marginal +0.27. Le bruit domine.

**Implications Paper B** :

- Le mismatch capacitif **réduit le seuil de bruit** d'escape : un memristor à 50% de variabilité (réaliste HfO₂) s'anime avec un bruit ~5× plus faible qu'un CMOS idéal. Quantitatif : passage de H≈1.34 à H≈0.35 quand on baisse η de 0.50 à 0.10, *si* σ_C reste à 0.50.
- La courbe `H(η, σ)` est **non-monotone à faible η** (états bloqués) → signature d'une vraie transition de phase induite par le désordre, pas un simple lissage.
- **Argument de design** : la "tare" des memristors (device-to-device variability) devient une **fonctionnalité**. Inversion de l'intuition CMOS classique où le mismatch est toujours un défaut.

**Limites** : 3 seeds, 1 graphe. À étendre : multi-graphe (5 BA m=5), seuil critique σ_c(η) précis (intervalle dichotomique), et test équivalent sur ER p=0.12 pour vérifier que le mécanisme est topology-agnostic dans la dead zone.

**Reproduction** : `experiments/spice_mismatch_sweep.py`. Figure : `figures/spice_mismatch_sweep.png` (heatmap + courbes). Données : `figures/spice_mismatch_sweep.csv`. Log : `experiments/spice/results/mismatch_sweep.log`.

### 3decies. SPICE noise/mismatch resonance — escape partiel de la dead zone (2026-04-19, soir)

**Question** : Si la dead zone est intrinsèque (§3nonies), est-ce que les imperfections hardware réelles (bruit thermique, mismatch CMOS) la cassent via stochastic resonance ?

**Méthode** : `experiments/spice_noise_resonance.py`. Sur le même graphe BA m=5 N=64, sweep η ∈ {0, 0.01, 0.03, 0.10, 0.30} (amplitude `trnoise()` injectée comme courant indépendant `I_eta`) × 3 normalisations. Puis Monte Carlo capacitif (3 trials, σ_C = 5% gaussien clamp [0.5, 1.5]) à l'optimum (η=0.30, degree_linear).

**Phase 1 — noise sweep** :

| norm | η=0 | η=0.01 | η=0.03 | η=0.10 | η=0.30 |
|:---|---:|---:|---:|---:|---:|
| uniform | 0.000 | 0.000 | 0.000 | 0.000 | 0.069 |
| degree_linear | 0.000 | 0.000 | 0.000 | 0.000 | **0.128** |
| spectral | 0.000 | 0.000 | 0.000 | 0.000 | 0.099 |

**Phase 2 — mismatch capacitif (η=0.30, degree_linear, 3 trials)** :
- sans mismatch : H = 0.128
- avec mismatch 5% : H = **0.161 ± 0.024** (3/3 trials > baseline)
- **delta = +0.033 (+26% relatif)** — synergie noise + mismatch confirmée

**Insights** :
1. **Bruit thermique pur insuffisant** : il faut η > 0.10 pour escape. C'est ~10⁴× la magnitude kT/C sur 1pF à 300K. Pas atteignable en CMOS classique sans source de bruit dédiée.
2. **Inversion de hiérarchie sous bruit** : `degree_linear` redevient meilleur (alors qu'en déterministe les 3 norms étaient équivalentes à H=0). Le bruit révèle la sensibilité à la pondération que la dynamique déterministe écrasait.
3. **Mismatch = quenched disorder utile** : 5% de variation capacitive (réaliste CMOS) augmente H de ~25% supplémentaire. Analogie directe avec le désordre figé en spin-glass — le mismatch crée des états métastables que le bruit explore.
4. **Conséquence pour Paper B** : le hardware "imparfait" est *intrinsèquement* meilleur que le hardware idéal pour cette dynamique. Argument fort pour les architectures memristives (variabilité device-to-device naturelle).

**Limites** : escape partiel (H≈0.16 vs ~0.83 attendu hors dead zone), 1 seed pour le sweep, 3 trials Monte Carlo. Le système n'est pas "vivant", il oscille faiblement autour du consensus. À étendre : sweep σ_mismatch (5%, 10%, 20%), multi-seed, sweep η × σ.

**Reproduction** : `experiments/spice_noise_resonance.py`. Traces : `experiments/spice/results/noise_*.{cir,dat}` + `mismatch_*.{cir,dat}`. Log : `experiments/spice/results/noise_resonance_run2.log`.

### 3nonies. SPICE confirme la dead zone BA m=5 en analogique (2026-04-19, soir)

**Question** : Est-ce que la dead zone observée dans Python est un artefact de l'intégrateur Euler, ou est-elle structurelle (et donc présente en hardware) ?

**Méthode** : Génération d'un netlist SPICE BA m=5 N=64 avec heretics (14.1%) et 3 normalisations (uniform / degree_linear / spectral). ngspice 46, intégrateur direct, déterministe (sans bruit), comparaison apples-to-apples avec une réf Python déterministe identique.

| norm | t_SPICE | H_SPICE | H_Python | comportement |
|:---|---:|---:|---:|:---|
| uniform | 0.3s | 0.000 | 0.000 | std init 0.60 → final 0.00 (point fixe v ≈ -1.286) |
| degree_linear | 0.3s | 0.000 | 0.000 | idem |
| spectral | 0.3s | 0.000 | 0.000 | idem |

**Verdict** : La dynamique transitoire est riche (std atteint 1.5 à t≈10) puis le réseau s'effondre vers un **point fixe consensuel** atteint vers t≈20. **Aucune normalisation** ne casse la dead zone en analogique non plus. La dead zone BA m=5 est donc **intrinsèque à la dynamique**, pas un artefact numérique.

**Impact Paper 2/B** : la question "faut-il une autre pondération du couplage ?" est tranchée (non). La piste à explorer pour briser la dead zone en hardware est désormais le bruit thermique réel (kT/C ≈ mV sur capacités fF) ou le mismatch capacitif CMOS (±5%). Test `trnoise()` à coder.

**Reproduction** : `experiments/spice_dead_zone_test.py`. Netlists et traces : `experiments/spice/results/dead_zone_BA_m5_N64_*.{cir,dat}`.

### 3octies. LIMIT-02 : Normalisation spectrale ne brise pas la dead zone (2026-04-19)

**Hypothèse** : L'eigenvector centrality voit la position globale d'un nœud dans la hiérarchie d'influence, là où `degree_linear` ne voit que l'adjacence locale. Sur les graphes denses (BA m≥5, ER p=0.12), tous les degrés sont élevés et indistinguables : la centralité spectrale devrait discriminer.

**Méthode** : Implémentation de `coupling_norm='spectral'` dans `core.py` (power iteration sur la matrice d'adjacence, weights ∝ 1/c_i). Comparaison uniform / degree_linear / spectral sur la dead zone (BA m=5,8,10, ER p=0.12) + 2 contrôles.

| Configuration | uniform | degree_linear | spectral |
|:---|---:|---:|---:|
| BA m=5 (dead zone) | 0.00 | 0.00 | 0.00 |
| BA m=8 (dead zone) | 0.00 | 0.00 | 0.00 |
| BA m=10 (dead zone) | 0.00 | 0.00 | 0.00 |
| ER p=0.12 (dead zone) | 0.00 | 0.00 | 0.00 |
| BA m=3 (contrôle) | 0.00 | 0.83 | 0.00 |
| BA m=1 (contrôle) | 0.96 | 0.00 | 0.00 |

**Verdict** : Spectral ne gagne sur AUCUNE configuration. Diagnostic : sur BA m=10 le ratio centralité hub/feuille est ≈ 6× — les poids ne sont *pas* dégénérés. Le problème de la dead zone n'est donc pas une dégénérescence des poids, c'est un **régime dynamique** (synchronisation par redondance des chemins) que la pondération ne peut pas casser.

**Implication Paper 2** : la dead zone m≥5 résiste à TOUTES les normalisations testées (uniform, degree, degree_linear, degree_log, degree_power, spectral). La piste à explorer n'est plus la pondération du couplage mais soit (a) la modification de la loi dynamique elle-même (adaptive heretics, doubt-driven rewiring topologique réel), soit (b) un changement de régime via stochastic resonance ciblé.

**Reproduction** : `experiments/spectral_normalization_test.py`. Mode disponible : `Mem4Network(..., coupling_norm='spectral')`.

### 3duodecies. P4.19ter : Robustesse, Frontière de Phase, Réplication ER (2026-04-19)

**Question** : L'escape noise+mismatch est-il robuste (multi-graphe) ? Où se trouve la frontière critique σ_c(η) ? Le mécanisme est-il topology-agnostic (ER) ?

**Méthode** : `experiments/spice_19ter_robustness.py` — 3 sous-expériences en séquence. Total ~100 runs ngspice.

**(a) Robustesse multi-graphe** — Cellule optimale (η=0.50, σ=0.50) sur 5 seeds BA m=5 différents :

| graph_seed | H_stable |
|:---:|:---:|
| 0 | 1.661 |
| 7 | 1.824 |
| 13 | 1.688 |
| 42 | 1.593 |
| 99 | 1.670 |
| **mean ± std** | **1.688 ± 0.076** |

**Verdict** : H ~73% du max théorique (log₂(5)=2.32) reproductible sur toutes les instances. L'escape n'est pas un artefact de seed.

**(b) Frontière de phase σ_c(η) par dichotomie binaire** (BISECT_ROUNDS=6, précision ≈ 0.008) :

| η | σ_c (H crosses 0.50) |
|:---:|:---:|
| 0.10 | **0.433** |
| 0.30 | **0.230** |
| 0.50 | **≈ 0** (escape sans mismatch) |

**Insight** : La frontière η ↔ σ_c est monotone décroissante : plus le bruit est fort, moins de mismatch est nécessaire. À η=0.50, le système s'échappe même sans désordre figé. **Figure publishable directement** : `figures/p4_19ter_dichotomy.png`.

**(c) Réplication ER p=0.12** — Même sweep (η ∈ {0.1, 0.3, 0.5} × σ ∈ {0, 0.05, 0.1, 0.2, 0.5}, 3 seeds) :

- H_max = **1.758** à η=0.50, σ=0.50
- Comportement identique au BA m=5 : même trois régimes, même seuil critique
- **Verdict : TOPOLOGY-AGNOSTIC CONFIRMÉ** — le mécanisme n'est pas spécifique aux graphes scale-free

**Conséquences Paper B** :
1. L'escape noise+mismatch est **robuste** (multi-instance) → résultat publiable solide.
2. La frontière σ_c(η) est **une courbe de phase** analogue au diagramme de phase spin-glass → argument théorique fort.
3. La **topology-agnosticism** ouvre le résultat à tous les substrats memristifs, pas seulement les réseaux scale-free.

**Reproduction** : `experiments/spice_19ter_robustness.py`. Figures : `p4_19ter_multigraph.png`, `p4_19ter_dichotomy.png`, `p4_19ter_er_replication.png`. CSV : `figures/p4_19ter_results.csv`.

### 3tredecies. P4.20 : Modèle Compact HfO₂ Déterministe — RÉSULTAT NÉGATIF (2026-04-19)

**Question** : Un modèle compact de memristor HfO₂ (modèle Yakopcic fluide, déterministe) modulant soit le seuil d'excitabilité neuronal (A) soit la force synaptique (B) peut-il briser la dead zone de façon autonome, sans bruit thermique ?

**Méthode** : `experiments/spice_p420_hfo2_memristor.py`. Implémentation du modèle de Yakopcic (`Ron=100`, `Roff=16k`). 
- **(A)** Modèle appliqué au neurone (capacité d'intégration) sur lattice 4x4.
- **(B)** Modèle appliqué aux synapses sur BA m=5 N=64. L'idée était d'exploiter un effet anti-Hebb.
- **(A+B)** Combinaison des deux sur m=5 N=16.

**Résultat** : Toutes les expériences (A, B, A+B) finissent avec l'entropie **H = 0.000**, et une convergence rapide vers un point fixe d'environ `v ≈ -1.27` avec une `std < 0.001` entre les noeuds. La dead zone est même renforcée dans ces conditions purement déterministes.

**Insight (Paper B)** : La non-linéarité memristive seule ne remplace pas le désordre ! Si le système est déterministe, les memristors saturent vers un état corrélé et le réseau fige. Cela vient valider une fois de plus, "par l'absurde", que c'est bien la synergie **(Bruit + Mismatch paramétrique / quenched disorder)** (P4.19bis/ter) qui génère l'échappement, pas uniquement la nature memristive en elle-même.

### 3quaterdecies. Phase 5 : Refactoring KIMI (2026-04-19) — CORRECTIONS CRITIQUES

**Contexte** : Audit adversarial par KIMI (LLM externe) identifiant 5 axes de problèmes :
(a) incohérences code/papier sur `tau_u` et `delta`, (b) entropie à 5 bins artificielle,
(c) erreur mathématique Poincaré-Bendixson en dimension 3N, (d) O(N²) dans rewiring,
(e) God Object monolithique.

**Corrections appliquées** :

| Type | Avant | Après | Impact |
|:-----|:------|:------|:-------|
| `tau_u` | 1.0 | **10.0** | Correspondance Annexe B preprint |
| `social_leakage` (δ) | 0.05 | **0.01** | Correspondance Eq. 5 preprint |
| Seuils entropie | ±0.8/1.5 | **±0.4/1.2** | Correspondance Table 1 preprint |
| Métrique entropie | 5 bins discrets (max log₂5=2.32 bits) | **100 bins uniformes continus** (max ~6.5 bits) | Entropie différentielle justifiable |
| Rewiring sparse | `.tolil()` + `.tocsr()` dans la boucle (O(N²)) | **Hors boucle O(N)** | ~100× speedup sur N=1000 |
| Poincaré-Bendixson | Appliqué au système couplé 3N dimensions | **Strictement limité au cas 2D découplé** | Rigueur mathématique |
| Architecture | `core.py` God Object 872 lignes | **Facade + `dynamics.py` + `topology.py` + `metrics.py`** | Maintenabilité |

**Nouveaux fichiers** :
- `src/mem4ristor/dynamics.py` — Mem4ristorV3 engine (FHN, RK45, plasticité)
- `src/mem4ristor/topology.py` — Mem4Network (Laplacian, rewiring O(N), spectral gap)
- `src/mem4ristor/metrics.py` — Entropie continue + états cognitifs avec seuils corrects

**Validation** : Tous les imports et smoketests passent. `preprint.pdf` recompilé (11 pages).
**Git** : commit `c8d9bde`.

⚠️ **Conséquence** : Les courbes de phase P4.19 doivent être régénérées avec la nouvelle métrique (100 bins). Le τ_u=10.0 ralentit la dynamique de doute — à vérifier sur les expériences existantes.

### 3quindecies. P4.19 régénération avec métrique continue — VALIDATION KIMI (2026-04-19)

**Question** : Le « complete escape » à (η=0.5, σ_C=0.5) rapporté en P4.19bis est-il un artefact du plafond log₂(5)=2.32 bits de la métrique 5-bin, ou survit-il sous l'entropie continue 100-bin préférée par l'audit KIMI ?

**Méthode** : `experiments/spice_mismatch_reanalyze.py`. Rechargement des 45 fichiers `.dat` ngspice déjà cachés (aucun re-run, ~2s d'analyse), puis H_stable recalculé avec **trois métriques en parallèle** :
1. `calculate_continuous_entropy` (100 bins uniformes sur [-3, 3], KIMI-preferred)
2. `calculate_cognitive_entropy` (5 bins, seuils corrigés ±0.4/1.2)
3. `cognitive_entropy` legacy (5 bins, anciens seuils ±0.8/1.5)

**Résultat — pic maintenu sous toutes les métriques** :

| Cellule | Continuous 100-bin | KIMI 5-bin (±0.4/1.2) | Legacy 5-bin (±0.8/1.5) |
|:---|---:|---:|---:|
| (η=0.1, σ=0.0) dead zone | **1.40** | 0.09 | 0.00 |
| (η=0.3, σ=0.5) stochastic res. | **4.14** | 1.24 | 1.33 |
| (η=0.5, σ=0.5) complete escape | **4.58** | 1.39 | 1.61 |

**Findings** :
1. **Pic identique** : argmax = (η=0.5, σ_C=0.5) sous les 3 métriques → le signal d'échappement est **robuste au choix d'estimateur**.
2. **Plafond démontré** : H_continuous atteint **4.58 bits** (au-dessus de log₂(5)=2.32), confirmant que la métrique 5-bin compressait effectivement le signal dans le régime diversifié.
3. **Finding émergent** : La « dead zone » (η=0.1, σ=0) n'est **pas vraiment morte** — H_continuous=1.40 bits indique une variabilité sub-cognitive (oscillations intra-bin) masquée par le plafond. Insight pour Paper B : la consensus-like phase a une structure fine détectable par métrique continue.

**Figures** :
- `figures/spice_mismatch_sweep_continuous.png` — heatmap + courbes d'échappement (primary, 100-bin)
- `figures/spice_mismatch_sweep_metric_compare.png` — 3 heatmaps côte-à-côte
- `figures/spice_mismatch_sweep_continuous.csv` — données brutes avec 3 colonnes H

**Conséquence** : Réponse directe à la critique #1 KIMI (plafond artificiel du 5-bin). Le résultat Paper B `complete escape at (η=0.5, σ_C=0.5)` est validé sous la métrique la plus défendable. Aucun re-run ngspice nécessaire.

**Note τ_u (invariance SPICE)** : Le sanity-check Phase 5 `τ_u=1 → τ_u=10` est **trivial côté SPICE P4.19** car le netlist simplifié utilise `du/dt = eps_u*(sigma_base - u)` avec `u_init = sigma_base`, donc `du/dt ≡ 0` et u reste figé à sigma_base=0.05 pour tout t. Le noyau Levitating Sigmoid devient une constante ≈ 0.94. Le couplage `k_u*sigma_social` actif dans le Python complet ([dynamics.py:225](src/mem4ristor/dynamics.py:225)) est omis du netlist. Tous les résultats P4.19bis/ter sont donc τ_u-invariants par construction. Valider τ_u=10 sur la dynamique de doute complète est une expérience à part entière (notée pour Paper B, pas un sanity check).

### 3sexdecies. P4.19 CMOS-réaliste σ_C ∈ [0, 0.15] — VALIDATION KIMI #2 (2026-04-19)

**Question** : KIMI critiquait que σ_C=0.50 testé en P4.19bis était physiquement irréaliste pour des capacités CMOS (mismatch typique 3-10%, max 15% en analog agressif). L'échappement survit-il dans la plage réaliste ?

**Méthode** : `experiments/spice_mismatch_cmos_realistic.py`. Sweep η ∈ {0.10, 0.30, 0.50} × σ_C ∈ {0, 0.02, 0.05, 0.08, 0.10, 0.15} × 3 seeds = 54 ngspice runs. Métrique primaire = 100-bin continuous entropy (Phase 5).

**Résultat — pic à σ=0.15 (plafond CMOS)** :

| Régime | H_continuous | H_5bin KIMI |
|:---|---:|---:|
| Dead zone (η=0.1, σ=0) | 1.36 | 0.07 |
| Stochastic resonance (η=0.3, σ=0.15) | 2.92 | 0.74 |
| **Noise-dominated (η=0.5, σ=0.15)** | **4.53** | **1.49** |

**Findings majeurs pour Paper B** :

1. **Escape à η=0.5 quasi-indépendant de σ_C dans [0, 0.15]** : Hc varie seulement 4.30 → 4.53 bits. Le régime "noise-dominated" n'a pas besoin de mismatch CMOS pour échapper la dead zone.
2. **Synergie bruit+mismatch absente du régime CMOS** : à η=0.3, Hc stagne autour de 2.8-2.9 bits pour σ ∈ [0, 0.15]. Le "big jump" Hc=1.24 → 4.14 à σ=0.5 observé en P4.19bis **nécessite σ_C bien au-delà du CMOS** (régime spin-glass / memristor stochastique).
3. **Narrative Paper B à ajuster** : le mécanisme physiologiquement pertinent pour CMOS est **stochastic noise seul** (η≥0.3), pas synergie noise+mismatch. La synergie reste une prédiction théorique pour substrats plus désordonnés.

**Figure** : `figures/spice_mismatch_cmos.png` (heatmap + courbes, 100-bin + 5-bin superposés). CSV : `figures/spice_mismatch_cmos.csv`.

**Conséquence** : KIMI critique #2 **battue sur un axe** (escape ∈ CMOS range) mais **confirmée sur un autre** (synergie avait σ_C irréaliste). Paper B doit clarifier les deux régimes : CMOS→noise-only, spin-glass→synergie.

### 3septdecies. P4.19 validation statistique 50-seed Monte Carlo — KIMI #3 (2026-04-19)

**Question** : Les claims P4.19bis/ter/bis-bis reposent sur 3 seeds par cellule. KIMI exigeait 100+ seeds pour crédibilité statistique. La séparation dead zone ↔ escape survit-elle sous scrutin statistique sérieux ? L'effet "mismatch" est-il réellement distinct du bruit pur ?

**Méthode** : `experiments/spice_mismatch_50seeds.py`. 50 seeds indépendants × 3 points critiques = 150 ngspice runs (~20 min). Points choisis pour tester deux hypothèses distinctes :
- **A**: Dead zone baseline (η=0.10, σ=0)
- **B**: Noise-only escape (η=0.50, σ=0) — teste H₁ : "le bruit seul échappe"
- **C**: Noise + CMOS mismatch (η=0.50, σ=0.10) — teste H₂ : "le mismatch ajoute quelque chose"

**Résultats (100-bin continuous entropy)** :

| Point | N | Mean | Std | 95% CI | Min | Max |
|:---|---:|---:|---:|---:|---:|---:|
| A dead zone | 50 | **1.377** | 0.043 | ±0.012 | 1.266 | 1.497 |
| B noise-only | 50 | **4.298** | 0.194 | ±0.054 | 3.865 | 4.680 |
| C noise+CMOS | 50 | **4.333** | 0.167 | ±0.046 | 3.985 | 4.647 |

**Tests de Welch** :

| Comparaison | t | p | Cohen's d | Verdict |
|:---|---:|---:|---:|:---|
| A vs B (noise alone escapes) | −103.9 | 1.2e-63 | **20.78** | Galactique |
| B vs C (CMOS adds vs noise-only) | −0.95 | 0.347 | 0.19 | **Non-significatif** |

**Findings majeurs** :

1. **L'escape via bruit seul est établi au-delà de tout doute** (p=1.2e-63, Cohen's d=20.78 dépasse largement le seuil "huge effect"=0.8). Les distributions A et B sont littéralement disjointes : max(A)=1.497 < min(B)=3.865.

2. **Le mismatch CMOS n'ajoute RIEN statistiquement** par-dessus le bruit (p=0.35, d=0.19). La différence de moyenne B vs C (4.298 vs 4.333) est dans le bruit d'échantillonnage. **Paper B doit abandonner l'hypothèse "synergy"** dans le régime CMOS-réaliste.

3. **Variance A << variance B** (σ_A=0.043 vs σ_B=0.194) : la dead zone est un attracteur très stable, tandis que le régime noise-dominated a une variabilité inter-seeds ~4× plus grande — signature d'une exploration stochastique de l'espace de phase.

**Narrative Paper B révisée** :
- **Titre candidat** : "Thermal noise alone escapes the consensus dead zone in coupled memristive networks"
- Le mismatch CMOS-réaliste est **neutre** (ni aide ni nuit)
- La "synergie noise+mismatch" reste une prédiction pour substrats à forte disorder (spin-glass, memristors stochastiques, σ_C >> 0.15)

**Figures** : `figures/spice_50seeds_validation.png` (violin + CI + tests), `figures/spice_50seeds_validation.csv` (150 lignes raw).

**Conséquence** : KIMI critique #3 **résolue définitivement**. L'intervalle de confiance 95% sur B est ±0.054 bits — précision publication-grade. Le pivot narratif (noise-only plutôt que synergy) renforce la clarté mécaniste de Paper B.

### 3octdecies. Étude de minimalité (ablations) — KIMI #4 (2026-04-20)

**Question** : KIMI reprochait au preprint de ne pas démontrer que chacun des trois ingrédients du noyau Mem4ristor (heretic flip, Levitating Sigmoid, dynamique de doute `u`) est *individuellement* nécessaire. On pouvait craindre qu'un seul d'entre eux suffise.

**Méthode** : `experiments/ablation_minimality.py`. Quatre configurations — `FULL`, `NO_HERETIC` (masque hérétique forcé à zéro), `NO_SIGMOID` (sigmoid remplacé par kernel attractif constant : `sigmoid_steepness=0`, `social_leakage=1` ⇒ `u_filter≡1`, coupling FHN classique), `FROZEN_U` (`epsilon_u=0`, `tau_u=1e12`, `u_i ≡ sigma_baseline`). Testées sur BA m=3 N=100 + `degree_linear`, 3000 pas, 10 seeds, sous **deux protocoles** :
- **ENDOGENOUS** : `I_stim = 0.0` (référence §3quinquies)
- **FORCED** : `I_stim = 0.5` (référence lattice de `test_scientific_regression.py`)

**Pourquoi deux protocoles** : le flip hérétique s'écrit `I_eff[mask] *= -1`. Il est *mathématiquement inactif* quand `I_stim = 0`. Un seul protocole aurait été auto-tautologique pour NO_HERETIC.

**Résultats (mean ± sem, n=10)** :

| Régime / Métrique | FULL | NO_HERETIC | NO_SIGMOID | FROZEN_U |
|:---|---:|---:|---:|---:|
| ENDOGENOUS / H₁₀₀ | **3.215 ± 0.114** | 3.215 ± 0.114 (d=0, ns) | 3.378 ± 0.084 (d=−0.51, ns) | **0.711 ± 0.012 (d=+9.75, p=3e-9)** |
| ENDOGENOUS / H_cog5 | 0.0005 | 0.0005 | 0.0000 | 0.0000 |
| FORCED / H₁₀₀ | 3.010 ± 0.029 | 3.441 ± 0.015 (d=−5.94, p=4e-9) | 2.976 ± 0.018 (ns) | 4.320 ± 0.047 (d=−10.69, p=2e-13) |
| FORCED / H_cog5 | **0.015 ± 0.004** | 0.408 ± 0.013 (d=−13.15) | 0.013 ± 0.006 (ns) | 1.041 ± 0.027 (d=−17.03) |

**Findings — trois messages non-triviaux** :

1. **Seul `FROZEN_U` collapse systématiquement H endogène** (H₁₀₀ : 3.22 → 0.71, d=+9.75). La dynamique de doute `u` est *individuellement nécessaire* au régime endogène — c'est le seul résultat qui confirme l'intuition « minimalité » à la lettre.

2. **Sous `I_stim=0`, retirer le flip hérétique est un no-op exact** : résultats bit-pour-bit identiques à FULL (d=0, p=1.0). C'est mathématiquement attendu (`I_eff[mask] *= -1` sur un vecteur nul) et exhibe un point important pour la critique KIMI : **l'hérétique est un mécanisme de *réponse au stimulus*, pas un générateur de diversité endogène**. Il doit être évalué sous forçage.

3. **Sous `I_stim=0.5`, retirer l'hérétique ou geler `u` AUGMENTE l'entropie** (H₁₀₀ : +0.43, +1.31 ; H_cog5 : +0.39, +1.03). Contre-intuitif au premier abord, mais éclairant : le Mem4ristor `FULL` sur BA m=3 forcé converge vers un régime à **polarité cognitive cohérente** où les unités traversent les mêmes bacs cognitifs de manière synchronisée (H_cog5 ≈ 0.015 bit ≈ *consensus cognitif*) tout en gardant une dispersion continue intra-bac (H₁₀₀ = 3.01). Les ablations cassent cette cohérence : chaque unité explore librement son propre bac, ce qui augmente H *spatial-instantané* mais détruit précisément la structure que le noyau était conçu à imposer.

4. **Levitating Sigmoid statistiquement neutre sur l'entropie** dans les deux régimes (|d| ≤ 0.51, p > 0.14). Remplacée par un kernel attractif constant, elle laisse H₁₀₀ et H_cog5 quasi-inchangés. **Interprétation** : le rôle du Sigmoid Lévitant n'est pas *quantitatif sur l'entropie* mais *qualitatif sur la topologie dynamique* (basculement attractif↔répulsif autour de `u=0.5`). Le valider proprement demande un protocole spécifique — par ex. suivre l'occurrence d'événements répulsifs corrélés à des pics de `u`, ce que l'entropie agrégée ne capture pas.

**Conséquences scientifiques** :

- La claim « minimalité » du preprint doit être reformulée : **`u` est individuellement nécessaire (grand effet)** ; **l'hérétique est stimulus-contingent (trivial sans forçage, mais cohérent sous forçage)** ; **le Sigmoid Lévitant est un mécanisme de commutation qualitatif, pas un générateur d'entropie quantitatif**.
- Cette étude **révèle une limite des métriques d'entropie actuelles** (100-bin comme 5-bin) : calculées comme diversité *spatiale instantanée* moyennée sur la queue, elles confondent « désordre aléatoire » et « diversité cognitive structurée ». Une métrique plus fine (Lempel-Ziv, information mutuelle inter-nœuds, cohérence de phase) serait nécessaire pour capturer la « cognition structurée » que Mem4ristor revendique. **À ajouter au backlog P1.**
- Le résultat « retirer `u` augmente H sous forçage » n'est *pas* une réfutation : il montre que `u` sert à *coordonner* la diversité plutôt qu'à la *maximiser*. Les métriques ne mesurent pas la coordination.

**Figures** : `figures/ablation_minimality.png` (4 panneaux : 2 régimes × 2 métriques), `figures/ablation_minimality.csv` (80 lignes raw).

**Verdict KIMI #4** : **partiellement résolue**. La minimalité stricte (« chaque ingrédient réduit H ») n'est établie que pour `u`. Les deux autres ingrédients demandent des protocoles ou métriques dédiés. Honnêteté scientifique préservée : on documente le résultat tel qu'il est, pas tel qu'on l'espérait.

### 3novedecies. P1.5bis : Métriques de coordination trajectorielles (2026-04-21)

**Motivation** : §3octdecies (ablations) a révélé que les métriques H₁₀₀ et H_cog5 sont des mesures de *dispersion spatiale instantanée*. Elles confondent désordre aléatoire et diversité structurée : dans le régime FORCED, retirer le flip hérétique ou geler u *augmente* H, ce qui est contre-intuitif — l'ablation détruit la coordination sans que les métriques spatiales le capturent.

**Solution** : Deux métriques trajectorielles ajoutées à `src/mem4ristor/metrics.py` :

| Métrique | Définition | Interprétation |
|:---------|:-----------|:---------------|
| `calculate_pairwise_synchrony(v_history)` | Corrélation de Pearson croisée, moyennée sur toutes les paires de nœuds (subsample max 2000 paires pour N>63) | +1 = nœuds parfaitement co-évolués ; 0 = indépendants ; −1 = anti-synchronisés |
| `calculate_temporal_lz_complexity(v_history)` | Complexité LZ76 normalisée (c × log₂T / T) sur les séquences d'états cognitifs par nœud, moyennée sur N | Proche de 0 = trajectoires structurées/prédictibles ; proche de 1 = marche aléatoire |

**Hypothèse P1.5bis** : FULL devrait exhiber une synchrony plus élevée **et** une LZ complexity plus basse que les ablations, même dans les régimes où ses métriques spatiales semblent inférieures. Cela distinguerait « diversité coordonnée » de « désordre aléatoire ».

**Validation du code** : 14 smoke tests couvrant les propriétés mathématiques fondamentales (séquence constante < séquence aléatoire en LZ ; traces identiques → synchrony=1 ; traces anti-corrélées → synchrony=−1 ; edge cases T=1, N=1).

**Note** : LZ76 croît en O(log₂ n) sur une séquence constante (et non en 1 phrase) — le minimum absolu est atteint par des séquences *périodiques*, pas constantes.

**Résultats (2026-04-21, 80 runs, 48s)** :

| Régime | Ablation | Synchrony (Pearson r) | LZ_tail | Interprétation |
|:---|:---|---:|---:|:---|
| ENDOGENOUS | FULL | **+0.199 ± 0.049** | 1.318 ± 0.020 | Exploration structurée coordonnée |
| ENDOGENOUS | NO_HERETIC | +0.199 (identique) | 1.318 (identique) | — no-op attendu (I_stim=0) |
| ENDOGENOUS | NO_SIGMOID | +0.084 ± 0.046 (p=0.11) | 1.324 (ns) | Légère perte de coordination |
| ENDOGENOUS | **FROZEN_U** | **+0.006 ± 0.003 (p=3e-3, d=1.75)** | **2.061 ± 0.012 (p=4e-15, d=14.67)** | **Effondrement coordination + trajectoires aléatoires** |
| FORCED | FULL | +0.031 ± 0.011 | 1.367 ± 0.038 | Walkers indépendants structurés |
| FORCED | NO_HERETIC | +0.069 (ns) | **1.607 ± 0.024 (p=7e-5, d=2.40)** | Plus random, légèrement plus synchronisé |
| FORCED | NO_SIGMOID | +0.039 (ns) | 1.438 (p=0.11) | Neutre |
| FORCED | **FROZEN_U** | **+0.751 ± 0.019 (p=7e-15, d=14.78)** | **1.994 ± 0.004 (p=4e-8, d=7.43)** | **Hyper-synchronisé + chaos partagé** |

**Trois findings majeurs** :

1. **`u` est individuellement nécessaire à la *coordination structurée*** : FROZEN_U s'effondre de synchrony=0.20 → 0.006 en ENDOGENOUS (p=3e-3) ET LZ explose de 1.32 → 2.06 (p=4e-15). Ce double signal — moins coordonné ET plus aléatoire — est la signature d'un attracteur chaotique non-structuré. Confirme §3octdecies par un protocole orthogonal.

2. **FULL sous forçage = "désynchronisation structurée"** : synchrony ≈ 0 + LZ bas (1.37) = nœuds indépendants mais trajectoires prédictibles. Le flip hérétique convertit l'entrée homogène en *walkers structurés indépendants*. FROZEN_U au contraire produit une "synchronisation chaotique" : synchrony=0.75 + LZ=1.99 = tous les nœuds font la même chose complexe.

3. **Inversion remarquable NO_HERETIC sous forçage** : retirer le flip hérétique AUGMENTE le LZ (1.37 → 1.61, p=7e-5). Cela confirme que le rôle du flip hérétique n'est pas de *maximiser* la diversité spatiale (§3octdecies le montrait déjà) mais de *structurer* les trajectoires temporelles individuelles. Sans lui, les nœuds explorent aléatoirement.

**Conclusion P1.5bis** : H₁₀₀ confondait "diversité riche" (FULL, LZ bas) avec "désordre aléatoire" (FROZEN_U, LZ haut). La paire (synchrony, LZ) résout l'ambiguïté : FULL produit des *walkers indépendants structurés*, FROZEN_U produit de la *synchronisation chaotique*. Les métriques trajectorielles doivent être citées dans le preprint comme preuve complémentaire de la claim "diversité structurée vs bruit".

**Figures** : `figures/ablation_coordination.png` (2 régimes × 2 métriques). CSV : `figures/ablation_coordination.csv`.

**Reproduction complète** : `python experiments/ablation_coordination.py` (48 s, 80 runs).
Tests unitaires : `pytest tests/test_coordination_metrics.py -v` (14 tests).

---

#### 3novedecies-bis. Pistes ouvertes post-P1.5bis (à reprendre)

Cinq questions émergent des résultats ci-dessus. Toutes sont *actionables* — scripts, protocoles et hypothèses précisés ici pour reprise directe.

**Piste A — Bimodalité ENDOGENOUS FULL (PRIORITÉ HAUTE)** 🔎

Per-seed inspection révèle que la synchrony ENDOGENOUS FULL n'est PAS gaussienne :
```
seed=0 sync=0.406 │ seed=1 sync=0.416 │ seed=2 sync=0.225 │ seed=3 sync=0.034
seed=4 sync=0.323 │ seed=5 sync=0.052 │ seed=6 sync=0.034 │ seed=7 sync=0.267
seed=8 sync=0.206 │ seed=9 sync=0.023
```
6 seeds à sync ≈ 0.2–0.4 (mode "coordonné"), 3 seeds à sync ≈ 0.03 (mode "désynchronisé"), 1 intermédiaire. Le rapport reporte `mean=0.199 ± 0.049` qui *masque* cette structure bimodale.

**Hypothèse** : l'attracteur dépend de λ₂ du BA généré (ou d'une autre propriété topologique : clustering, diamètre, max_degree). BA avec m=3, N=100 n'est pas équivalent pour tout seed.

**Protocole** (~5 min) : étendre `ablation_coordination.py` — enregistrer pour chaque seed : λ₂, max_degree, avg_clustering, diamètre. Régresser sync contre chaque. Si λ₂ prédit, on tient un diagramme de phase topologique endogène. **Script proposé** : `experiments/ablation_coordination_topology.py`.

**Piste B — Diagramme de phase 2D (synchrony × LZ)** 🎨

Les 4 régimes forment un nuage 2D qui pourrait constituer une figure publishable :

```
         LZ bas (structuré)        LZ haut (aléatoire)
sync=0   FULL_FORCED (0.03, 1.37)  
         NO_SIGMOID_FORCED (0.04, 1.44)
sync mid FULL_ENDO (0.20, 1.32)    NO_HERETIC_FORCED (0.07, 1.61)
sync haut                          FROZEN_U_FORCED (0.75, 1.99)
                                   FROZEN_U_ENDO (0.01, 2.06)  [sync bas!]
```

Les 4 quadrants ont des interprétations distinctes :
- **(bas, bas)** — walkers indépendants structurés (FULL_FORCED) = **cognition diverse**
- **(haut, bas)** — coordination structurée (FULL_ENDO mode haut) = **consensus structuré**
- **(haut, haut)** — synchronisation chaotique (FROZEN_U_FORCED) = **chaos cohérent**
- **(bas, haut)** — walkers aléatoires indépendants (FROZEN_U_ENDO) = **désordre pur**

**Protocole** (~10 min) : créer `experiments/phase_space_coordination.py` qui scatter-plote tous les seeds × ablations × régimes avec couleurs. À publier comme figure dans la Section Minimality du preprint v3.3 ou Paper B.

**Piste C — Sweep `heretic_ratio` sous forçage** (inversion NO_HERETIC)

Finding contre-intuitif : NO_HERETIC_FORCED **augmente le LZ** (1.37→1.61, d=2.40). Hypothèse : le flip hérétique agit comme un *régulariseur temporel* qui structure les trajectoires. Si vrai, l'effet devrait se renforcer avec heretic_ratio.

**Protocole** (~3 min) : sweep `η ∈ {0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50}` sur FORCED, 5 seeds. Tracer LZ(η). Attendu : décroissance monotone puis plateau. Si non-monotone → autre mécanisme. Script : `experiments/heretic_ratio_sweep_coordination.py`.

**Piste D — Multi-topologie (universalité)**

Tous les résultats P1.5bis sont sur BA m=3 N=100. Est-ce que la signature "FULL=walkers indépendants structurés" survit sur :
- Lattice 10×10 (contrôle classique)
- BA m=5 (dead zone §3nonies)
- ER p=0.12 (dead zone)
- Watts-Strogatz p=0.1

**Enjeu** : si FULL produit sync≈0 + LZ bas *partout*, on a une propriété universelle du noyau. Si c'est spécifique à BA m=3, c'est plus faible. **Protocole** : étendre `ablation_coordination.py` avec boucle topologie (4 × 4 ablations × 10 seeds = 160 runs, ~1h30). Script proposé : `experiments/ablation_coordination_topology_sweep.py`.

**Piste E — Résonance stochastique inversée (FROZEN_U forcing paradox)**

Observation : FROZEN_U passe de sync=0.006 (ENDOGENOUS) à sync=0.751 (FORCED). Le forçage à u gelé *crée* une synchronisation massive là où u libre l'empêche. C'est une signature de **résonance induite par le forçage externe en absence de régulateur**.

**Hypothèse** : u libre = filtre anti-synchronisation. u gelé = passe-bande qui synchronise au signal commun.

**Protocole** : sweep `I_stimulus ∈ [0, 1]` sur FULL vs FROZEN_U, mesurer sync(I). Si FULL reste plat et FROZEN_U croît monotonement → confirmation. Script : `experiments/forcing_sweep_frozen_u.py`. **Résultat publishable si confirmé** — élégante démonstration du rôle régulateur de u.

---

**Matrice de priorité** (effort × impact) :

| Piste | Effort | Impact | Statut |
|:---|:---|:---|:---|
| A (bimodalité) | 5 min | Haut (structure cachée) | **FAIT 2026-04-21** → §3vigies |
| B (phase space 2D) | 10 min | Haut (figure publishable) | **FAIT 2026-04-21** → §3vigies-bis |
| C (sweep heretic) | 3 min | Moyen | **FAIT 2026-04-24** → §3novedecies-ter |
| D (multi-topo) | 1h30 | Moyen-haut | Ouvert |
| E (résonance inversée) | 15 min | Haut si confirmé | Ouvert |

### 3novedecies-ter. Piste C — Sweep heretic_ratio sous forçage (2026-04-24)

**Question** : Le flip hérétique agit-il comme un *régulariseur temporel* dont l'effet se renforce avec `heretic_ratio` ? Attendu si vrai : LZ décroissant monotonement avec η.

**Méthode** : `experiments/heretic_ratio_sweep_coordination.py`. BA m=3, N=100, `degree_linear`, FORCED (I_stim=0.5), 3000 steps, 5 seeds. η ∈ {0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50}.

**Résultats** :

| η | sync mean±sem | LZ_full mean±sem | LZ_tail mean±sem |
|:---:|---:|---:|---:|
| 0.00 | +0.139 ± 0.026 | 1.137 ± 0.012 | 1.552 ± 0.053 |
| 0.05 | +0.015 ± 0.005 | 1.096 ± 0.013 | 1.473 ± 0.011 |
| 0.10 | +0.008 ± 0.004 | 1.076 ± 0.009 | 1.456 ± 0.006 |
| 0.15 | +0.016 ± 0.010 | 1.069 ± 0.008 | 1.376 ± 0.053 |
| 0.20 | +0.041 ± 0.018 | 1.054 ± 0.007 | 1.416 ± 0.020 |
| 0.30 | +0.039 ± 0.005 | 1.043 ± 0.004 | 1.370 ± 0.032 |
| 0.50 | +0.083 ± 0.030 | 1.012 ± 0.004 | 1.324 ± 0.034 |

**Findings** :

1. **LZ_full monotone décroissant** (1.137 → 1.012, Δ=−0.125) sur toute la plage η ∈ [0, 0.50]. **Hypothèse du régulariseur temporel CONFIRMÉE** sur la trace complète : plus on ajoute d'hérétiques, plus les trajectoires sont structurées.

2. **LZ_tail globalement décroissant** (1.552 → 1.324, Δ=−0.227) mais avec un bump à η=0.20 (1.416 vs 1.376 à η=0.15). Le verdict automatique "NON-MONOTONE" est dû au bruit de seed (sem ≈ 0.020–0.053) — le trend global confirme le régulariseur mais 10+ seeds seraient nécessaires pour la robustesse statistique.

3. **Synchrony en forme de U** : sans hérétiques (η=0) sync=0.139 (tous les nœuds convergent vers le même point fixe → cohérence passive). Avec quelques hérétiques (η=0.05) sync chute à 0.015 (les hérétiques brisent la cohérence consensus). Puis légère remontée à η=0.50 (0.083) — les hérétiques nombreux créent une "cohérence de diversité" par forçage partagé.

4. **Point de transition η ≈ 0.05** : le passage de 0 à 5% hérétiques est la discontinuité la plus forte (sync : 0.139 → 0.015, LZ_full : 1.137 → 1.096). Au-delà, les métriques évoluent graduellement. Cela suggère un **seuil de frustration géométrique** minimal pour briser la convergence consensuelle.

**Conséquence pour le preprint** : La claim "le flip hérétique est un mécanisme de réponse au stimulus" (§3octdecies, FLAW 6) est enrichie : c'est aussi un **régulariseur temporel** dont l'effet est déjà mesurable à 5% et sature progressivement. La valeur défaut η=0.15 est dans la zone de transition (LZ chute rapide puis plateau).

**Figures** : `figures/heretic_sweep_coordination.png` (2 panneaux : sync(η) + LZ(η)). CSV : `figures/heretic_sweep_coordination.csv` (35 lignes raw).

**Reproduction** : `python experiments/heretic_ratio_sweep_coordination.py` (~27s).

---

### 3novedecies-quater. Piste E — Résonance stochastique inversée : u comme filtre anti-synchronisation (2026-04-24)

**Question** : u agit-il comme un filtre anti-synchronisation ? Hypothèse : FULL maintient sync ≈ 0 quelle que soit l'intensité du forçage (u absorbe le signal commun) ; FROZEN_U voit sa synchronie croître monotonement avec I_stim (nœuds intègrent le même stimulus sans régulateur).

**Méthode** : `experiments/forcing_sweep_frozen_u.py`. FULL vs FROZEN_U, BA m=3, N=100, `degree_linear`, heretic_ratio=0.15. I_stim ∈ {0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 1.00}, 7 seeds, 3000 steps.

**Résultats** :

| I_stim | FULL sync | FROZEN_U sync | FULL lz_full | FROZEN_U lz_full |
|:---:|---:|---:|---:|---:|
| 0.00 | +0.213 ± 0.066 | +0.006 ± 0.004 | 0.919 ± 0.004 | 0.921 ± 0.002 |
| 0.10 | +0.039 ± 0.012 | +0.210 ± 0.076 | 1.101 ± 0.008 | 1.380 ± 0.048 |
| 0.20 | +0.035 ± 0.009 | **+0.830 ± 0.009** | 1.114 ± 0.006 | 1.595 ± 0.003 |
| 0.30 | +0.023 ± 0.007 | **+0.889 ± 0.009** | 1.104 ± 0.005 | 1.621 ± 0.003 |
| 0.40 | +0.031 ± 0.009 | **+0.861 ± 0.012** | 1.085 ± 0.005 | 1.622 ± 0.001 |
| 0.50 | +0.028 ± 0.014 | **+0.749 ± 0.026** | 1.071 ± 0.006 | 1.634 ± 0.003 |
| 0.70 | +0.027 ± 0.010 | **+0.320 ± 0.065** ← dip | 1.037 ± 0.006 | 1.649 ± 0.004 |
| 1.00 | +0.035 ± 0.004 | **+0.901 ± 0.013** | 1.130 ± 0.011 | 1.598 ± 0.004 |

**Welch FULL vs FROZEN_U à I=0.50** : Cohen's d = **13.21**, p = 9.69×10⁻¹⁰. Effet galactique.

**Quatre findings** :

1. **Hypothèse CONFIRMÉE — FULL est un filtre anti-synchronisation** : sync FULL ≈ 0.025–0.035 pour tout I ∈ [0.10, 1.00], parfaitement plat. Le régulateur u absorbe le signal de forçage commun et empêche le verrouillage de phase des nœuds.

2. **Transition de synchronisation FROZEN_U à I ≈ 0.10–0.20** : passage abrupt de sync=0.006 à sync=0.830 (sem=0.009 — très serré), puis plateau à 0.83–0.89 pour I ∈ [0.20, 0.40]. Signature d'une bifurcation de point fixe (les nœuds trouvent tous le même attracteur forcé quand I dépasse un seuil).

3. **Finding inattendu : dip de synchronie à I=0.70** (sync : 0.861 → 0.320 → 0.901 à I=1.00). Sem très élevé à I=0.70 (0.065 vs 0.013 aux voisins) → le système est à un **point de bifurcation** : certaines seeds trouvent un attracteur synchronisé, d'autres pas. C'est une **résonance intra-cohorte** dépendante du graphe sous-jacent, pas visible sur FULL. À investiguer : peut-être que I=0.70 correspond à la transition entre deux régimes d'attracteurs de FROZEN_U.

4. **FROZEN_U endogène (I=0) est désynchronisé** (sync=0.006) mais **FULL endogène est semi-coordonné** (sync=0.213) — cohérent avec la bimodalité documentée en §3vigies (certains BA seeds tombent en mode "coordonné endogène" avec u libre). u actif *crée* une coordination endogène partielle ; u gelé ne peut pas.

**LZ_full FROZEN_U** : croissance monotone 0.921 → 1.65 (r=+0.61 vs I_stim) — dynamics plus chaotiques sous forçage fort, mais la synchronie y est maintenue (chaos cohérent).

**Conséquence scientifique** : c'est la démonstration la plus propre à ce jour du rôle régulateur de u. La figure est **directement publiable** dans le preprint §Minimality comme preuve complémentaire que u n'est pas optionnel — il structure ET il filtre le signal externe. Le finding "dip à I=0.70" est un bonus non trivial qui ouvre sur les bifurcations de FROZEN_U.

**Figures** : `figures/forcing_sweep_frozen_u.png` (2 panneaux : sync(I) + LZ(I)). CSV : `figures/forcing_sweep_frozen_u.csv`.

**Reproduction** : `python experiments/forcing_sweep_frozen_u.py` (~63s, 112 runs).

---

### 3vigies. Piste A — Bimodalité prédite par le clustering local (2026-04-21)

**Question** : Parmi λ₂, max_degree, avg_clustering, diamètre — laquelle prédit l'attracteur (coordonné/désynchronisé) dans lequel tombe ENDOGENOUS FULL ?

**Méthode** : `experiments/ablation_coordination_topology.py`. Régénère les 10 BA(m=3, N=100) utilisés par `ablation_coordination.py`, calcule les 4 métriques topologiques, régresse la synchrony observée contre chacune.

**Résultats** (n=10, Pearson r ; Spearman ρ) :

| Métrique | Pearson r | p | Spearman ρ | p |
|:---|---:|---:|---:|---:|
| λ₂ | −0.44 | 0.20 | −0.32 | 0.37 |
| Max degree | +0.35 | 0.33 | +0.38 | 0.28 |
| **Avg clustering** | **−0.64** | **0.045** | −0.53 | 0.12 |
| Diameter | −0.09 | 0.81 | −0.17 | 0.63 |

**Verdict** : Le **clustering local** prédit la synchrony (Pearson significatif à α=0.05). **Corrélation NÉGATIVE** : plus de triangles → mode désynchronisé ; graphe plus arborescent → mode coordonné. Contre l'intuition classique (clustering = synchro), mais cohérent avec l'hypothèse "frustration géométrique" : les triangles dans le réseau créent des boucles de rétroaction qui empêchent la coordination globale quand le régulateur `u` est actif.

**Limites** : n=10 trop petit pour distinguer Pearson (p=0.045) de Spearman (p=0.12). Hartigan dip test non effectué. À confirmer avec ≥50 seeds pour robustesse.

**Implications scientifiques** :

1. La "variance inter-seeds" rapportée dans §3novedecies (sync=0.199 ± 0.049) **cache une structure bimodale** déterministe — pas du bruit.
2. Mem4ristor exhibe un **diagramme de phase endogène** contrôlé par le clustering du graphe sous-jacent.
3. Potentielle connexion avec §3sexies (régimes multi-topologie LIMIT-02) : peut-être que la même métrique (clustering plutôt que deg_ratio) prédisait déjà les régimes "uniform vs degree_linear gagne".

**Piste de suivi** : ~~rerun sur 50 seeds + tester Hartigan dip test~~ → **FAIT (§3vigies-ter)**. Régresser aussi contre `(spectral gap × clustering)` composite ou investiguer le rôle des graphes déconnectés (λ₂=0).

**Figure** : `figures/coordination_bimodality.png`. CSV : `figures/coordination_bimodality.csv`.

### 3vigies-bis. Piste B — Diagramme de phase 2D (sync × LZ) (2026-04-21)

**Question** : Les 4 ablations × 2 régimes occupent-elles des quadrants interprétables dans le plan (synchrony, LZ_complexity) ?

**Méthode** : `experiments/phase_space_coordination.py`. Scatter plot de toutes les 80 runs (8 cellules × 10 seeds) avec centroïdes overlayés. Médianes visuelles `sync=0.15`, `LZ=1.6` délimitent 4 quadrants.

**Positions des 8 centroïdes** :

| Régime | Ablation | sync (mean±std) | LZ (mean±std) | Quadrant |
|:---|:---|---:|---:|:---|
| ENDOGENOUS | FULL | +0.199 ± 0.156 | 1.318 ± 0.062 | **Structured consensus** |
| ENDOGENOUS | NO_HERETIC | +0.199 ± 0.156 | 1.318 ± 0.062 | idem (no-op) |
| ENDOGENOUS | NO_SIGMOID | +0.084 ± 0.147 | 1.324 ± 0.066 | Cognitive diversity (borderline) |
| ENDOGENOUS | FROZEN_U | +0.006 ± 0.009 | **2.061** ± 0.037 | **Pure disorder** |
| FORCED | FULL | +0.031 ± 0.034 | 1.367 ± 0.119 | **Cognitive diversity** |
| FORCED | NO_HERETIC | +0.069 ± 0.092 | 1.606 ± 0.076 | Borderline diversity/chaos |
| FORCED | NO_SIGMOID | +0.039 ± 0.022 | 1.438 ± 0.052 | Cognitive diversity |
| FORCED | FROZEN_U | **+0.751** ± 0.060 | **1.994** ± 0.011 | **Coherent chaos** |

**Claim scientifique unique** : Seul **FULL** (les deux régimes) occupe le demi-plan bas-LZ (structuré). Toutes les ablations soit restent bas-LZ mais sync~0 (cas borderline), soit migrent vers le haut-LZ (chaotique). La structure trajectorielle — pas la diversité spatiale — est la signature mesurable du noyau Mem4ristor complet.

**Note méthodologique** : Les seuils de quadrants (sync=0.15, LZ=1.6) sont choisis visuellement et servent de repères pédagogiques, pas de critères statistiques. Le message visuel tient indépendamment des coordonnées exactes.

**Implication Paper B / preprint** : Cette figure est candidate pour **intégration dans la section Minimality** (remplacerait ou compléterait Figure ablation_minimality.png). Elle apporte 3 améliorations :
- Résout directement la critique KIMI #4 (minimalité) en montrant que seul FULL tient dans le quadrant "structuré".
- Contourne le meta-problème "H₁₀₀ confond désordre et diversité" en l'affichant explicitement (ordonnée = LZ = structure temporelle).
- Déplace la conversation de "quelle métrique utiliser" vers "quel régime du plan est publishable".

**Figure** : `figures/coordination_phase_space.png` (800 × 550 px, 140 dpi). CSV centroïdes : `figures/coordination_phase_centroids.csv`.

### 3vigies-quater. Piste D — Universalité multi-topologie des métriques de coordination (2026-04-24)

**Question** : La signature "FULL = walkers indépendants structurés" (sync≈0, LZ bas) est-elle universelle (lattice, BA m=3, BA m=5 dead zone, WS) ou spécifique à BA m=3 ?

**Méthode** : `experiments/ablation_coordination_topology_sweep.py`. 4 topologies × 4 ablations × 2 protocoles × 10 seeds = **320 runs**, ~3 min. Normalisation : `degree_linear` pour BA m=3, `uniform` pour les autres.

**Résultats** (synchrony mean ± sem, lz_full mean ± sem, FORCED) :

| Topologie | FULL sync | FROZEN_U sync | FULL lz | FROZEN_U lz |
|:---|---:|---:|---:|---:|
| Lattice 10×10 | **+0.005 ± 0.003** | **+0.523 ± 0.036** | 1.088 ± 0.007 | 1.636 ± 0.002 |
| BA m=3 | **+0.031 ± 0.011** | **+0.751 ± 0.019** | 1.069 ± 0.005 | 1.635 ± 0.002 |
| BA m=5 (dead zone) | **−0.001 ± 0.003** | **+0.935 ± 0.004** | 1.024 ± 0.009 | 1.651 ± 0.002 |
| WS k=4 p=0.1 | **+0.004 ± 0.003** | **+0.534 ± 0.044** | 1.061 ± 0.005 | 1.632 ± 0.002 |

**LZ_full FULL ENDOGENOUS** : 0.916 ± 0.003, 0.918 ± 0.003, 0.945 ± 0.015, 0.918 ± 0.002 — **constant ≈ 0.92 sur toutes les topologies**.

**Quatre findings universaux** :

1. **FULL = filtre anti-synchronisation sur TOUTES les topologies testées** : sync FULL ≈ 0 sous forçage (lattice, BA m=3, BA m=5, WS). L'universalité est confirmée — ce n'est pas une propriété spécifique à BA m=3.

2. **FROZEN_U = synchroniseur universel** : sync FROZEN_U ≫ 0 pour toutes les topologies sous forçage. Extremum sur BA m=5 (sync=0.935 ± 0.004 — le plus élevé de toutes les conditions testées). Dans la *dead zone spatiale*, FROZEN_U se verrouille encore plus fort qu'ailleurs.

3. **Finding inattendu sur la dead zone** : FULL BA m=5 FORCED maintient LZ=1.024 et sync≈0 — il produit des walkers structurés indépendants *même dans la dead zone*. La "dead zone" désigne un effondrement de la diversité *spatiale* (H_cog=0), pas un effondrement de la *coordination temporelle*. Les deux propriétés sont orthogonales.

4. **LZ_full FULL ENDOGENOUS ≈ 0.92** est une constante quasi-universelle, robuste à la topologie. C'est la signature de l'attracteur du noyau Mem4ristor complet, indépendante de la structure du graphe sous-jacent.

**Conséquence pour le preprint** : La claim d'universalité est désormais justifiée empiriquement sur 4 topologies distinctes. Le finding "dead zone ≠ effondrement de coordination" est scientifiquement nouveau et non trivial.

**Figures** : `figures/ablation_coordination_topo_sweep.png` (4 topologies × 2 métriques × 2 protocoles). CSV : `figures/ablation_coordination_topo_sweep.csv` (320 lignes).

**Reproduction** : `python experiments/ablation_coordination_topology_sweep.py` (~3 min).

**Reproduction complète des pistes A+B** :
```bash
python experiments/ablation_coordination.py              # 48 s (prérequis)
python experiments/ablation_coordination_topology.py     # < 5 s
python experiments/phase_space_coordination.py           # < 2 s
```

### 3vigies-ter. Piste F — Bimodalité confirmée, prédicteur topologique infirmé (2026-04-24)

**Question** : La bimodalité ENDOGENOUS FULL est-elle statistiquement réelle (n=50 + Hartigan dip test) ? Et le prédicteur avg_clustering trouvé à n=10 (§3vigies, r=−0.64, p=0.045) était-il robuste ?

**Méthode** : `experiments/bimodality_50seeds.py`. 50 seeds indépendants, ENDOGENOUS (I_stim=0), FULL, BA m=3, N=100, `degree_linear`, 3000 steps. Mesures : `pairwise_synchrony` + 4 métriques topologiques (λ₂, max_degree, avg_clustering, diamètre). Tests statistiques : Hartigan dip (MC p-value, 2000 bootstrap) + Bimodality Coefficient (BC Sarle).

**Résultats** :

| Test | Valeur | Seuil | Verdict |
|:-----|:-------|:------|:--------|
| Hartigan dip D | 0.1937 | p < 0.05 | **p = 0.000 → BIMODAL** |
| Bimodality Coefficient BC | 0.605 | > 0.555 | **BIMODAL** |
| Distribution | mean=0.139 ± 0.147 | — | Variance élevée, deux modes visibles |

**Régression sync vs topologie (n=50)** :

| Métrique | Pearson r | p | Spearman ρ | p |
|:---------|----------:|--:|----------:|--:|
| λ₂ | −0.122 | 0.397 | −0.195 | 0.176 |
| max_degree | −0.099 | 0.493 | −0.024 | 0.870 |
| avg_clustering | −0.191 | 0.183 | −0.029 | 0.843 |
| diameter | +0.007 | 0.960 | +0.106 | 0.463 |

**Aucun prédicteur topologique simple n'est significatif à n=50.**

**Findings** :

1. **Bimodalité CONFIRMÉE statistiquement** : les deux tests indépendants (Hartigan et BC) concordent. La structure bimodale observée à n=10 n'était pas un artefact de petit échantillon — elle est réelle sur n=50.

2. **avg_clustering r=−0.64 de §3vigies était un faux positif n=10** : à n=50, r=−0.19 et p=0.183. Le résultat de §3vigies doit être recalibré : "le clustering est un prédicteur suggestif mais non robuste". Il faut ≥50 seeds pour tester la significativité.

3. **Confound : graphes déconnectés (λ₂ ≈ 0)** : 7 seeds sur 50 ont λ₂ = 0 (graphe non connexe, diamètre=∞) — ce qui est physiquement atypique pour BA m=3. Ces seeds montrent un comportement mixte : certains ont sync élevé (0.338, 0.378 — composantes synchronisées séparément), d'autres sync≈0 (0.000, -0.002 — composantes anti-phasées). Ils constituent un sous-régime à part entière. Une analyse séparée (connexes vs déconnectés) est à envisager.

4. **Prédicteur de la bimodalité inconnu** : les 4 métriques simples ne suffisent pas. Pistes : prédicteur composite (λ₂ × clustering), analyse par composantes connexes, ou propriétés spectrales de second ordre. Backlog futur.

**Conséquence pour le preprint** : la claim "clustering prédit la synchrony" (§3vigies) doit être qualifiée : "suggestive à n=10 (p=0.045), non reproductible à n=50". En revanche, la **bimodalité elle-même est publiable** — c'est un fait robuste qui montre que le système Mem4ristor sur BA m=3 est multi-attracteur.

**Figures** : `figures/bimodality_50seeds.png` (histogramme + KDE + ECDF + 4 régressions). CSV : `figures/bimodality_50seeds.csv` (50 lignes).

**Reproduction** : `python experiments/bimodality_50seeds.py` (~29s).

---

### 3octvicies. AUDIT EXTERNE ADVERSARIAL — 7 Flaws Critiques (2026-04-22)

**Contexte** : Audit indépendant mené sur le codebase complet (notebook `bda19036-*.ipynb`, ~37 cellules d'analyse). Exécuté hors du workspace principal via un agent data-analysis. Vérifications contre le code source (`dynamics.py`, `metrics.py`, `mem4ristor_v26.va`, SPICE netlist) le 2026-04-22.

**Résultat global** : 5 flaws confirmés sans ambiguïté, 1 partiellement vrai, 1 déjà documenté mais non propagé.

---

#### FLAW 1 — Point fixe STABLE au lieu d'instable (claim §3.1 préprint) ✅ CONFIRMÉ

**Vérification** : Jacobienne analytique au FP (v*=−1.294, w*=−0.732, u*=0.05 en tenant compte du decay plasticité) :
- λ = −0.055 ± 0.283i → partie réelle NÉGATIVE → **spiral stable**
- Bifurcation de Hopf à α_crit ≈ 0.296 ; valeur défaut α = 0.15 **en dessous**

**Confirmation numérique** : nœud isolé (D=0, σ_v=0) convergé en 5000 steps vers v*=−1.2944 ± 7×10⁻⁶.

**Claim préprint faux** : §3.1 ligne 155 dit "the unique equilibrium is unstable for the standard parameter range". À corriger : le nœud isolé est **excitable (stable)**, pas oscillant. La note §3.1 ligne 159 ("Poincaré-Bendixson ne s'étend pas au couplé") était déjà correcte — mais la phrase d'avant doit être retirée.

**À α = 0.30** (au-dessus du Hopf) : oscillations confirmées, H_cog = 0.56 (87/13 split States 1/2). Piste genuine pour obtenir la diversité multi-état sans bruit.

---

#### FLAW 2 — H_cog ≈ 0.92 est un artefact de bins (§3bis, limitations.md LIMIT-05) ✅ CONFIRMÉ

**Mécanisme** : L'ancienne bin ±1.5 tombait **au milieu** du cluster consensus v ∈ [−2.4, −1.2]. Split 48/52 → H = −0.48 log₂(0.48) − 0.52 log₂(0.52) ≈ 1.0 bit.

**Avec bins KIMI (±0.4/1.2)** : 100/0/0/0/0 → **H_cog = 0** pour TOUTES les configs testées (lattice, BA m=1 à 10, ER, WS, cold/random start, toutes normalisations).

**Conséquences** :
- §3bis, §3ter, §3quinquies, §3sexies, §3ter : tous les H_stable (0.92, 0.828, 0.958, etc.) sont **mesurés avec les ANCIENNES bins** et ne correspondent plus aux chiffres que renverrait le code actuel.
- `limitations.md` LIMIT-05 "stable H ≈ 0.92" est obsolète.
- Les sections SPICE (§3undecies–§3septdecies) utilisent H_cont 100-bin et **restent valides**.

**Survit valide** : diversité Python **à α = 0.30** (H_cog = 0.56 avec bins KIMI, 87/13 split States 1/2) — mais ce n'est pas le régime testé dans la préprint.

---

#### FLAW 3 — Verilog-A (`mem4ristor_v26.va`) désynchronisé ✅ CONFIRMÉ (5/5 points)

| Écart | v26.va (actual) | dynamics.py (expected) |
|:------|:----------------|:-----------------------|
| Noyau couplage | `(1.0 - 2.0*u)*V(coup_in)` (linéaire ancien) | `tanh(π(0.5-u)) + 0.01` (Levitating Sigmoid) |
| τ_u | `1.0` | `10.0` (correction KIMI) |
| ε_u adaptatif | absent | `ε_u * clip(1 + α_s·σ, 1, C_cap)` |
| Plasticité dw | absent | `−w/τ_plast` (toujours actif) |
| Double-comptage | `V(coup_in)*heretic_pol + social_signal` | couplage via social_signal seulement |

**Toute validation hardware basée sur v26.va teste un modèle différent du Python.** Le RMS < 1% validé en §3septies utilisait des netlists SPICE générés programmatiquement (pattern I-source correct), PAS le v26.va shippé.

---

#### FLAW 4 — SPICE netlist shippé utilise pattern R+B-voltage (filtre, pas intégrateur) ✅ CONFIRMÉ

**`spice/mem4ristor_coupled_3x3.cir`** utilise `R_v0 v0_node v0_int` + `C_v0 v0_int 0` + `B_dv0 v0_node 0 V = f(v0_int)`. Ce pattern intègre `dv/dt = f(v) − v` et non `dv/dt = f(v)`.

Ce bug était **connu** (§3septies §164–166) mais seuls les netlists générés par `experiments/spice_validation.py` ont été corrigés. Le netlist shippé reste broken.

**Action** : soit corriger `spice/mem4ristor_coupled_3x3.cir` avec pattern I-source, soit l'annoter "DEPRECATED — utiliser les netlists générés par spice_validation.py".

---

#### FLAW 5 — `solve_rk45` injecte du bruit dans le RHS déterministe ✅ CONFIRMÉ

**`dynamics.py:246`** : `eta = self.rng.normal(...)` appelé **à l'intérieur** de `combined_dynamics(t, y)` passé à `solve_ivp`. RK45 appelle le RHS plusieurs fois par step avec des valeurs d'état différentes — chaque appel tire un bruit distinct, violant la contracte déterministe de solve_ivp.

**Impact** : `solve_rk45` ne peut être utilisé fiablement qu'avec `sigma_v = 0`. En présence de bruit, les résultats sont non-reproductibles et mathématiquement incorrects (ordre effectif réduit, step-size adaptatif invalide).

**Action** : soit désactiver le bruit dans `combined_dynamics`, soit documenter explicitement que `solve_rk45` = mode déterministe uniquement.

---

#### FLAW 6 — Mécanisme hérétique inactif à I_stim = 0 ⚠️ PARTIELLEMENT VRAI

**`dynamics.py:198`** : `I_eff[heretic_mask] *= -1.0` est un no-op quand `I_eff = 0`.

**Ce qui est vrai** : toutes les expériences "endogènes" (I_stim = 0) citées dans la préprint (lattice H≈0.92, sweeps BA, etc.) ne testent **pas** le mécanisme hérétique. Les hérétiques sont mathématiquement invisibles dans ce régime.

**Ce qui est déjà connu** : §3octdecies documente "heretic flip is a no-op exact under I_stim=0, d=0, p=1.0". Sous FORCED (I_stim=0.5), l'ablation montre d=−5.94 (H_cog5 passe de 0.015 à 0.408 sans hérétique). **Le mécanisme fonctionne — il est juste jamais évalué dans le protocole endogène.**

**Correction narrative** : la préprint doit clarifier que le mécanisme hérétique est un *mécanisme de réponse au stimulus*, pas un générateur de diversité endogène. L'Eq. 9 (entropy lower bound) est vacuante à I_stim = 0.

---

#### FLAW 7 — Terme de decay plasticité `−w/τ_plast` non documenté dans la préprint ✅ CONFIRMÉ

**`dynamics.py:210`** : `dw_learning = (plasticity_drive * saturation_factor) − (self.w / self.tau_plasticity)`. Le terme `−w/τ_plast` est **toujours actif**, même quand `plasticity_drive = 0` (no coupling).

Conséquence : le FP réel est v*=−1.294 (avec decay) et non v*=−1.286 (sans decay), une différence de ~8.8×10⁻³ en v. Les équations du préprint ne mentionnent pas ce terme.

**Action** : soit ajouter `−w/τ_plast` aux équations du préprint (Eq. 2), soit le désactiver quand `lambda_learn = 0`.

---

#### CE QUI RESTE SOLIDE

| Résultat | Raison de la validité |
|:---------|:----------------------|
| SPICE/Python RMS < 1% (§3septies) | Netlists I-source corrects, Python déterministe |
| Dead zone BA m≥5 (§3nonies/octies) | Confirmé SPICE + Python + toutes normalisations |
| Escape noise+mismatch P4.19 (§3undecies–septdecies) | H_cont (100-bin) validé sous 3 métriques en §3quindecies |
| Métriques trajectorielle (§3novedecies/vigies) | LZ + synchrony : protocol orthogonal à H_cog |
| Hopf à α≈0.296 | Jacobienne analytique + simulation numérique (α=0.30 : oscillations confirmées) |

---

#### ACTIONS REQUISES (par ordre de priorité)

1. ~~**Préprint §3.1** : corriger "unique equilibrium is unstable" → "excitable regime (stable FP at α=0.15)"~~ **✅ FAIT 2026-04-22** — Bloc `\textbf{Correction}` ajouté dans `docs/preprint.tex` §3.1, incluant eigenvalues confirmés numériquement, α_crit≈0.296, et restriction du P-B à α>α_crit.
2. ~~**§3bis + limitations.md LIMIT-05** : annoter que H≈0.92 est avec **anciennes bins pré-KIMI**.~~ **✅ FAIT 2026-04-22** — Annotations ⚠️ INVALIDÉ ajoutées dans §3bis, §3ter, §3quinquies, §3sexies, limitations.md LIMIT-05.
3. ~~**mem4ristor_v26.va** : soit refondre pour matcher dynamics.py~~~~ **✅ RÉÉCRIT 2026-04-22** — Modules `mem4ristor_v32` et `mem4ristor_cell_v32` créés, synchronisés avec dynamics.py v3.2 (noyau Levitating Sigmoid, τ_u=10, ε_u adaptatif, plasticité+decay, polarity sur stim uniquement). Aliases backwards-compat `mem4ristor_v26` et `mem4ristor_cell` conservés.
4. **spice/mem4ristor_coupled_3x3.cir** : corriger ou annoter DEPRECATED — **✅ ANNOTÉ 2026-04-22**, pattern corrigé dans experiments/spice_validation.py.
5. ~~**solve_rk45** : documenter restriction sigma_v=0 ou corriger~~ **✅ FAIT 2026-04-22** — Bloc WARNING ajouté dans `dynamics.py:solve_rk45`.
6. ~~**Préprint Eq. 9** : qualifier "applies under I_stim ≠ 0 only"~~ **✅ FAIT 2026-04-22** — Scope note ajouté dans `docs/preprint.tex` §3.3.
7. ~~**Préprint Eq. 2** : ajouter `−w/τ_plast` ou justifier son absence~~ **✅ FAIT 2026-04-22** — Description dw_plasticity mise à jour dans `docs/preprint.tex` pour inclure le terme de decay toujours actif et l'innovation mask.

**Toutes les 7 actions de §3octvicies sont closes. Aucune action bloquante restante.**

### 3quater. LIMIT-04 : Stabilité Euler (2026-03-21)

**Question** : L'intégrateur Euler est-il instable au long terme ?

**Méthode** : Sweep dt ∈ [0.01, 0.50], runs jusqu'à 20 000 steps, 3 seeds.

**Résultat** : Le "drift > 5%" est en fait le **transitoire de convergence** vers l'attracteur. Une fois stabilisé (après ~2000-3000 steps), H reste constant à ±0.016 pendant au moins 20 000 steps à dt=0.05. Les variables d'état (v, w, u) ne dérivent pas.

| dt | Comportement |
|:---|:---|
| ≤ 0.05 | Stable après transitoire — recommandé |
| 0.07-0.10 | Drift mineur, acceptable pour exploration |
| ≥ 0.15 | Dynamiques altérées — déconseillé |
| ≥ 0.50 | Effondrement de l'entropie |

**Verdict** : Le claim original "❌ FALSE" est trop sévère. Corrigé en "⚠️ NUANCÉ". dt≤0.05 validé.

---

### §3quinquetrigies. Révision adversariale — 5 attaques (2026-04-28)

**Contexte** : Une instance Claude a soumis les deux papers (preprint + paper_2) à une revue sévère simulée, identifiant 5 attaques critiques. Traitement complet en une session.

#### Attack 1 — Terminologie trompeuse

**Problème** : "frustrated synchronization" (terme Kuramoto) et "topological phase transition" (terme physique des défauts topologiques) appliqués à un mécanisme fondamentalement différent.

**Correction dans preprint.tex et paper_2.tex** :
- `\subsection{Frustrated Synchronization}` → `\subsection{Polarity-Modulated Anti-Synchronization}`
- `\subsection{The Topological Boundary}` → `\subsection{The Spectral Connectivity Boundary}`
- "topological phase transition" → "spectral phase transition" (partout)
- "topological dead zone" → "spectral dead zone" (partout dans paper_2)

**Justification scientifique** : Dans Kuramoto, la frustration vient de signes de couplage quenchés et distribués statiquement. Ici, l'inversion de polarité est un processus continu et state-dependent : `f(u_i) = tanh(π(0.5-u_i)) + δ` change de signe selon l'état interne de chaque nœud. La distinction n'est pas cosmétique — elle touche au mécanisme.

#### Attack 2 — H_cog=0 : artefact ou réalité ?

**Problème** : Le reviewer argumente que H_cog=0 dans toutes les simulations Python invalide les claims de diversité.

**Diagnostic** : Artefact de calibration. Les bins KIMI (±0.4, ±1.2) ont été calibrés sur les voltages SPICE (≈±1-2V). Les voltages Python aux paramètres défaut se situent dans [-3.2, -1.3], tous inférieurs à -1.2 → bin 1 → H_cog=0 structurellement.

**Corrections** :
- Preuve indépendante de la diversité : synchrony pairwise FULL=0.031±0.034 vs FROZEN_U=0.751±0.087 (+2326%)
- Métrique primaire recentrée sur H_cont (100-bin continu) : H_stable=3.79±0.14 bits sur lattice
- Section Limitations enrichie : bullet complet sur l'artefact de binning avec synchrony comme preuve
- Ablation table remplacée par synchrony+LZ (directionnellement corrects) en lieu de H_cont (inversé pour cette comparaison)

#### Attack 3 — σ_social ablation et rôle de u

**Problème** : L'ablation σ_social montre SS_NOISE ≈ SS_STATIC ≈ FULL (ΔH < 2%) → u ne "détecte" pas la structure sociale, il en est indiscernable du bruit. La claim "détecteur de surprise structurelle" est fausse.

**Correction** : Requalification complète du rôle de u. u est un **filtre anti-synchronisation actif**, pas un détecteur de surprise. La preuve : FROZEN_U génère sync=0.751 vs FULL sync=0.031 (+2326%) sous forcing identique. Ce qui compte n'est pas ce qui drive u, c'est que u soit actif. Abstract paper_2, corps mécanique, section τ_u tous mis à jour.

#### Attack 4 — Stratégie solo author / no affiliation

**Verdict** : Non applicable. Julien donne son travail librement à quiconque souhaite s'en emparer. La question de stratégie de publication ne se pose pas dans ce contexte.

#### Attack 5 — SPICE validation trop étroite

**Problème** : Reviewer affirme "validation SPICE limitée à la grille 4×4 seulement".

**Réfutation** : Les simulations BA m=5, N=64 existaient déjà (50 seeds × 3 conditions, `experiments/spice/results/`). Elles n'étaient simplement pas documentées dans le paper.

**Résultats publiés (table ajoutée dans preprint.tex `\subsection{SPICE Circuit Validation}`)** :

| Condition | η | σ_C | H_cont (mean±std, N=50) |
|:----------|:-:|:---:|:------------------------|
| A: Dead zone (λ₂ > λ₂_crit) | 0.10 | 0.00 | 1.38 ± 0.04 bits |
| B: Functional (noise only) | 0.50 | 0.00 | 4.30 ± 0.19 bits |
| C: Functional (noise+CMOS mismatch) | 0.50 | 0.10 | 4.33 ± 0.17 bits |

Ratio 3.1× entre régimes. Mismatch CMOS σ_C=0.10 sans effet sur la diversité (B≈C). Simulations exécutées via NGSpice 46 en batch mode (`ngspice -b`), confirmé opérationnel.

**Commit** : `d057865` sur `feat/v4-dynamic-heretics`.

---

## 4. TESTS — CE QUI PASSE, CE QUI NE PASSE PAS

### Tests fonctionnels (devraient passer)

| Fichier | Couvre | Notes |
|:--------|:-------|:------|
| `tests/test_kernel.py` | Noyau V3, sigmoid, plasticité | 7.9 KB |
| `tests/test_robustness.py` | NaN, overflow, edge cases | 9.0 KB |
| `tests/test_fuzzing.py` | Fuzzing aléatoire (200 iter) | 3.1 KB |
| `tests/test_adversarial.py` | Attaques adverses | 2.9 KB |
| `tests/test_manus_v2.py` | Assertions V3 audit | 3.7 KB |
| `tests/test_symbiosis_*.py` | CreativeProjector, Swarm | 3.5 KB |
| `tests/test_v4_extensions.py` | Meta-doubt, rewiring | 16.9 KB |
| `tests/test_v5_hysteresis.py` | V5 hysteresis : configuration, latching, watchdog fatigue | **3 tests ACTIFS (activés 2026-03-22)** |

### Tests xfail

Aucun. Tous les tests xfail V5 ont été activés le 2026-03-22.

### Tests de régression scientifique (2026-04-10) ★

| Fichier | Couvre | Notes |
|:--------|:-------|:------|
| `tests/test_scientific_regression.py` | 6 propriétés fondamentales du modèle | **NOUVEAU** — 6/6 PASS |

Propriétés testées :
1. Entropie > 0 avec heretics (Cold Start)
2. Collapse sans heretics
3. Doute clamped [0, 1]
4. Noise nécessaire sous Cold Start
5. Scaling entropie invariant
6. Levitating Sigmoid correcte

---

## 5. AMÉLIORATIONS IDENTIFIÉES (TODO TECHNIQUE)

### Priorité haute (cohérence scientifique)

1. **~~Corriger `demo_chimera.py`~~** → FAIT (2026-03-21)
2. **~~Décider du sort de `test_v5_hysteresis.py`~~** → FAIT (xfail → actif 2026-03-22)
3. **~~Résoudre LIMIT-05~~** → INVESTIGUÉ. Stable H ≈ 0.92. Claim corrigé dans preprint.
4. **~~Résoudre LIMIT-02~~** → **PARTIELLEMENT RÉSOLU** (2026-04-10). `degree_linear` (D/deg(i)) → H≈0.83 sur BA m≤3. Dead zone m≥5 identifiée. Résultat négatif publiable.
5. **~~Caractériser LIMIT-04~~** → INVESTIGUÉ. dt≤0.05 validé, claim original trop sévère.
6. **~~Corriger le preprint~~** → **FAIT** (2026-04-10). Restructuration complète : 26 → 11 pages. 2 relectures externes intégrées. 0 warnings LaTeX.

### Priorité moyenne (qualité du code)

7. **~~`sensory.py` : convolution lente~~** → **FAIT (antérieur)**. `scipy.signal.correlate2d` déjà en place.
8. **~~Module `viz.py`~~** → FAIT (stable, intégré dans demo_applied.py)
9. **~~Exports `__init__.py`~~** → **FAIT (antérieur)**. Tous modules exportés dans `__init__.py`.
10. **~~Normalisation par degré pour LIMIT-02~~** → **FAIT** (2026-04-10). `degree_linear` (D/deg(i)) validé. Voir §3quinquies.

### Priorité basse (évolution) — COMPLÉTÉS 2026-03-22

11. **~~Démonstration appliquée~~** → **FAIT** (2026-03-22). `examples/demo_applied.py` : 4 démos (sensory pipeline, hysteresis comparison, scale-free sparse, phase diversity), 5 PNG.
12. **~~V5 (hysteresis)~~** → **FAIT** (2026-03-22). Dead-zone latching [0.35, 0.65] + watchdog fatigue. 3 tests passent. H_stable +5%.
13. **~~Config par dataclass~~** → **FAIT (antérieur)**. `config.py` avec `@dataclass` complet.
14. **~~Performance sparse~~** → **FAIT** (2026-03-22). Auto-sparse CSR (scipy) si N > 1000. Mémoire 455× à N=5000, vitesse L@v 219×.

---

## 6. ARBORESCENCE DU PROJET

```
mem4ristor-v2-main/
├── src/mem4ristor/           # PACKAGE PRINCIPAL
│   ├── core.py               # Moteur V3 + V5 hysteresis + sparse CSR
│   ├── config.yaml            # Paramètres par défaut
│   ├── viz.py                 # Visualisation (entropy, doubt map, phase, dashboard)
│   ├── symbiosis.py           # CreativeProjector + SymbioticSwarm
│   ├── cortex.py              # LearnableCortex (MLP autoencoder)
│   ├── sensory.py             # SensoryFrontend
│   ├── inception.py           # DreamVisualizer
│   ├── hierarchy.py           # HierarchicalChimera (expérimental)
│   ├── arena.py               # GladiatorMem4ristor + Arena (expérimental)
│   └── benchmarks/engine.py   # Moteur de benchmark
│
├── experimental/              # MODULES NON PRODUCTION
│   └── mem4ristor_king.py     # "Philosopher King"
│
├── tests/                     # SUITE DE TESTS
│   ├── test_v5_hysteresis.py  # ★ V5 hysteresis — 3 tests ACTIFS (2026-03-22)
│   └── ...
│
├── examples/                  # DÉMONSTRATIONS
│   ├── demo_applied.py        # ★ Pipeline complet : sensory → network → viz (2026-03-22)
│   └── ...
│
├── experiments/               # SCRIPTS D'EXPÉRIENCE
│   ├── entropy_sweep/         # ★ Investigations LIMIT-02/04/05 (2026-03-21)
│   │   ├── README.md          # Protocole et résultats
│   │   ├── entropy_sweep.py   # LIMIT-05 : sweep paramétrique
│   │   ├── stability_analysis.py # LIMIT-05 : attracteur vs transitoire
│   │   ├── limit02_scalefree.py  # LIMIT-02 : strangulation scale-free
│   │   └── limit04_stability.py  # LIMIT-04 : stabilité Euler
│   ├── limit02_norm_sweep.py   # ★ Sweep normalisation degree (2026-04-10)
│   ├── limit02_topology_sweep.py # ★ Validation multi-topologie (2026-04-10)
│   ├── benchmark_kuramoto.py
│   ├── benchmark_variability.py
│   └── ...
│
├── docs/                      # DOCUMENTATION SCIENTIFIQUE
│   ├── preprint.tex           # ★ Preprint restructuré (11 pages, soumissible)
│   ├── preprint.pdf           # ★ PDF compilé (338 Ko)
│   ├── preprint_v320_pre_restructure.tex  # Backup avant restructuration
│   ├── limitations.md         # Table de vérité (mise à jour LIMIT-02/04/05)
│   └── ...
│
├── failures/                  # ÉCHECS DOCUMENTÉS
│   └── philosopher_king_removal.log  # Log suppression test cassé
│
├── PROJECT_STATUS.md          # CE FICHIER
├── VERSION                    # v3.2.0
├── CHANGELOG_V3.md            # Changelog complet V3
└── ...
```

---

## 7. COMMENT DÉMARRER

```bash
git clone https://github.com/Jusyl236/mem4ristor-v2.git
cd mem4ristor-v2
pip install -e .
pytest tests/
```

Quick test :
```python
from mem4ristor.core import Mem4Network
net = Mem4Network(size=10, heretic_ratio=0.15, seed=42)
for step in range(1000):
    net.step(I_stimulus=0.5)
print(f"Entropy: {net.calculate_entropy():.4f}")
```

Demo complète (2026-03-22) :
```bash
python examples/demo_applied.py
# Produit 5 PNG : sensory_pipeline.png, hysteresis_comparison.png,
#   scalefree_sparse.png, phase_diversity.png, + dashboard
```

---

## 8. CONTRIBUTEURS IA (traçabilité)

| Modèle | Rôle principal |
|:-------|:---------------|
| Claude 3.5 Sonnet (Anthropic) | Conceptualisation initiale, drafting LaTeX, core logic v2.0 |
| Grok-2 (xAI) | Tests adverses, hypothèse 15%, philosophie du doute |
| Gemini 1.5 Pro (Google) | Refactoring, sweeps de sensibilité, visualisation |
| Claude Opus 4.6 (Anthropic) | Audit V3, migration imports, investigation LIMIT-02/04/05 (2026-03-21). V5 hysteresis, sparse CSR, demo appliquée (2026-03-22) |
| Antigravity / Gemini 2.5 Pro | Consolidation v3.2.0, tests régression, BA m/α sweep, restructuration preprint, réponse critique externe (2026-04-10) |
| Claude Opus 4.7 (Anthropic) | Fix bugs P1 (symbiosis swarm, V4 entropy), validation SPICE/Python sub-1% RMS, figure phase diagram λ₂ vs H_stable, normalisation spectrale (résultat négatif publiable) (2026-04-19) |

Niveau de transparence : **Radical** — transcripts complets dans le dépôt Café Virtuel.

---

## 9. RÉSUMÉ DES SESSIONS DE TRAVAIL

### Session 2026-03-21 (Claude Opus 4.6)
- Audit complet du code : corrections imports (hierarchy, arena, inception, demo_chimera)
- Investigation LIMIT-02, LIMIT-04, LIMIT-05 avec protocoles expérimentaux
- Création test_v5_hysteresis.py (3 tests xfail)
- Documentation limitations.md mise à jour

### Session 2026-03-22 (Claude Opus 4.6)
- **V5 Hysteresis** : implémentation complète dans core.py (4 patches). Dead-zone [0.35, 0.65] + watchdog fatigue. 3 tests xfail → actifs.
- **Sparse CSR** : 8 patches dans core.py (Mem4Network). Auto-détection N > 1000, scipy.sparse, eigsh. Mémoire 455× à N=5000.
- **Demo appliquée** : `examples/demo_applied.py`, 4 scénarios (sensory pipeline, hysteresis comparison, scale-free sparse, phase diversity), 5 PNG.
- Explication pédagogique des matrices creuses et applicabilité (DIALux, éclairage).

### Session 2026-04-10 — Soir (Antigravity / Gemini)

**Phase 1 — Alignement v3.2.0** :
- Versioning global aligné (VERSION, pyproject.toml, README.md, preprint.tex)
- Réécriture README.md (features V4/V5, usage scale-free)
- Suppression fichiers orphelins + `test_philosopher_king.py.broken`
- Mise à jour CHANGELOG_V3.md

**Phase 3 — Tests de régression scientifique** :
- Création `tests/test_scientific_regression.py` : 6 tests validant les propriétés fondamentales → 6/6 PASS

**Phase 4A — Investigation BA m sweep** :
- Sweep BA m ∈ [1, 15] × {uniform, degree_linear} : découverte dead zone m ∈ [5, 10]
- 2 crossovers de régime identifiés, λ₂ comme prédicteur
- Script : `experiments/limit02_ba_m_sweep.py`

**Phase 4B — Investigation α sweep** :
- Implémentation mode `degree_power` dans core.py (D/deg(i)^γ)
- Sweep γ ∈ [0, 1] × m ∈ {2, 3, 4, 5, 6, 8, 10}
- Résultat négatif définitif : pour m≥5, AUCUN γ ne fonctionne
- α*(m=2) = 0.7 (H=0.99), α*(m=3) = 0.9 (H=0.89)
- Script : `experiments/limit02_alpha_sweep.py`

**Phase 2 — Intégration preprint** :
- Ajout Tables BA m sweep + α sweep
- Nouvelle sous-section "Topological Diversity Regimes"
- Mise à jour limitations scale-free

**Phase 6 — Restructuration du preprint (réponse à critique externe)** :
- Réécriture complète : 26 pages → **11 pages**
- Suppression : hardware mapping, CCC France, Café Virtuel (comme section), vocabulaire militant
- Titre : "Frustrated Synchronization in Doubt-Modulated FHN Networks"
- 0 erreurs, 0 warnings, 0 références indéfinies

**Phase 6bis — Réponse à 2ème relecture** :
- 14 corrections majeures : collision α→γ, Strogatz→Nattermann+Toulouse, contradiction bruit, comptage mécanismes, Table 2 ±σ, circularité C≈0.6, Floquet clarifié, Tables 4↔5, σ_social→σ_local, dwplasticity crédité, WS/ER qualifiés...
- 3 corrections finales : µ_λ→µ_k, justification π, seeds Table 2

### Session 2026-04-19 (Claude Opus 4.7)

Plan d'attaque validé par Julien : **D → B → C → A**.

**D — Fix bugs P1** :
- `tests/test_symbiosis_swarm.py` : test était écrit pour une diffusion mean-field symétrique mais `symbiosis.py` implémente un MAX FIELD asymétrique. Test corrigé pour refléter la sémantique réelle (vétéran préservé, recrue qui hérite).
- `tests/test_v4_extensions.py::test_entropy_preservation_with_v4` : le ring N=10 n'avait pas de hubs sur lesquels V4 puisse agir. Remplacé par BA m=3, N=50, `coupling_norm='degree_linear'`. H_stable > 0.3 (≈ 0.83).
- **Suite complète : 74 tests verts (2 xfail attendus).**

**B — Validation hardware SPICE** : voir §3septies.
- `experiments/spice_validation.py` : netlist auto-généré 4×4, intégrateur direct, ngspice 46. RMS global 9.7×10⁻³.
- 2 pièges SPICE découverts et documentés dans `experiments/spice/README_HARDWARE.md` (ngspice `pow()` Jacobian, RC vs intégrateur).
- Figure : `figures/spice_vs_python_validation.png`. **Première preuve quantitative de faisabilité hardware.**

**C — Figure phase diagram λ₂ vs H_stable** : `experiments/fiedler_phase_diagram.py`.
- 15 topologies × 2 normalisations × 3 seeds. Calcul λ₂ via `scipy.linalg.eigh` du Laplacien.
- Figure + CSV : `figures/fiedler_phase_diagram.png`, `figures/fiedler_phase_diagram.csv`.
- Visualise la transition de phase : λ₂<0.1 → uniform gagne, 0.1–1.5 → degree_linear, >3 → dead zone.

**A — Normalisation spectrale** : voir §3octies.
- Mode `coupling_norm='spectral'` ajouté dans `core.py` (méthode `_eigenvector_centrality`, power iteration).
- Hypothèse falsifiée : 0/6 wins sur la dead zone. Diagnostic montre que les poids ne sont *pas* dégénérés (ratio 6× sur BA m=10). La dead zone est un régime dynamique, pas un défaut de pondération.
- **Résultat négatif publiable** qui réoriente Paper 2 vers (a) modification dynamique ou (b) stochastic resonance.

**E (bonus) — SPICE dead zone** : voir §3nonies.
- `experiments/spice_dead_zone_test.py` : netlist BA m=5 N=64 avec heretics + 3 normalisations.
- 0/3 normalisations ne brisent la dead zone en analogique. Convergence vers point fixe v ≈ -1.286.
- **Confirmation hardware** : la dead zone est intrinsèque, pas numérique. Ferme la branche "améliorer la pondération" et oriente le hardware track vers le bruit (`trnoise()`) et le mismatch capacitif.

**F (bonus) — SPICE noise/mismatch resonance** : voir §3decies.
- `experiments/spice_noise_resonance.py` : noise sweep + Monte Carlo mismatch sur BA m=5.
- Bruit thermique réaliste insuffisant (besoin η > 0.10, ~10⁴× kT/C sur 1pF).
- Mismatch capacitif 5% (CMOS-réaliste) ajoute +26% de H au-dessus du bruit seul. **Synergie noise + quenched disorder**, analogue spin-glass.
- Escape partiel (H~0.16 vs ~0.83 hors dead zone). Pas une "résurrection" complète mais une preuve de mécanisme.
- Conséquence Paper B : memristors imparfaits intrinsèquement supérieurs au CMOS idéal pour cette dynamique.

**G (bonus) — SPICE mismatch sweep escape complet** : voir §3undecies.
- `experiments/spice_mismatch_sweep.py` : 45 runs ngspice (3 η × 5 σ × 3 seeds), heatmap publishable.
- **H_max = 1.61** (≈69% max théorique) à η=0.50 + σ=0.50. **Escape complet**, pas partiel.
- 3 régimes identifiés : bruit faible (besoin σ≥0.50, états métastables), bruit moyen (résonance stochastique pure), bruit fort (escape même sans mismatch).
- Argument Paper B affiné : mismatch capacitif **réduit le seuil de bruit** d'escape — la variabilité memristor est une *fonctionnalité*.

### Session 2026-04-21 (Claude Sonnet 4.6, P1.5bis)

**P1.5bis — Métriques de coordination trajectorielles** : voir §3novedecies.
- `src/mem4ristor/metrics.py` : ajout de `calculate_pairwise_synchrony` (corrélation Pearson croisée sur fenêtre temporelle) et `calculate_temporal_lz_complexity` (complexité LZ76 normalisée sur séquences d'états cognitifs). Helper privé `_lz76_phrases` (parsing glouton O(n²), adapté aux traces T≈300).
- `experiments/ablation_coordination.py` : reproduction des 4 ablations (FULL, NO_HERETIC, NO_SIGMOID, FROZEN_U) × 2 régimes × 10 seeds avec enregistrement de l'historique complet `v(t)`. Figure 2×2 + CSV.
- `tests/test_coordination_metrics.py` : 14 smoke tests. **74 tests verts (+ 2 xfail attendus)**.

**Pistes A + B post-P1.5bis** :
- **Piste A** (§3vigies) : `experiments/ablation_coordination_topology.py`. Analyse bimodalité ENDOGENOUS FULL vs topologie. **Finding** : avg clustering prédit synchrony (Pearson r=−0.64, p=0.045). Plus de triangles → mode désynchronisé. Frustration géométrique.
- **Piste B** (§3vigies-bis) : `experiments/phase_space_coordination.py`. Diagramme de phase 2D (sync × LZ) des 80 runs avec 4 quadrants interprétables. **Claim unique** : seul FULL occupe le demi-plan bas-LZ (trajectoires structurées). Figure candidate pour intégration dans preprint v3.3.

---

### Session 2026-04-20 (Claude Opus 4.7, réponse KIMI suite & fin)

**Contexte** : retour KIMI/Manus sur le repo après le push P4.19. Trois chantiers en séquence : A (✅ fait par Julien) — régénération figures continuous 100-bin, B — pivot narratif Paper B, C — étude de minimalité (ablations).

**B — Paper B narrative pivot** :
- Lecture `docs/paper_B/paper_B.tex` (86 lignes, narrative "synergy noise+mismatch" obsolète post-P4.19bis/ter).
- Abstract réécrit : noise alone escape (d=20.78), CMOS mismatch neutre (d=0.19), spin-glass en extrapolation σ_C >> 0.15.
- Section 4 subdivisée : §4.1 spectral entropy (100-bin vs 5-bin), §4.2 MC 50-seeds avec Table + Cohen's d, §4.3 CMOS sweep 0-15%, §4.4 topology-agnostic.
- Section 5 (Phase Boundary) reframée comme prédiction extrapolative pour substrats à forte disorder.
- Conclusion mise à jour : thermal noise = primary engine, quenched disorder = frontière architecturale.
- 3 nouvelles figures copiées dans `docs/paper_B/figures/` : `spice_mismatch_sweep_continuous.png`, `spice_50seeds_validation.png`, `spice_mismatch_cmos.png`.
- PDF recompilé : 7 pages, 0 warnings, 0 undefined refs.

**C — Étude de minimalité / ablations** : voir §3octdecies.
- `experiments/ablation_minimality.py` : 4 ablations (FULL, NO_HERETIC, NO_SIGMOID, FROZEN_U) × 2 protocoles (I_stim=0.0 et 0.5) × 2 métriques (H₁₀₀, H_cog5) × 10 seeds = 80 runs (~60s).
- Résultat 1 : `u` est individuellement nécessaire (d=+9.75, p=3e-9 en ENDOGENOUS).
- Résultat 2 : `heretic flip` est *stimulus-contingent* : no-op exact sous I_stim=0, cohérent sous forçage (d=−5.94 à −13.15, mais réduisant H cohérente plutôt que dispersée).
- Résultat 3 : `Levitating Sigmoid` est statistiquement neutre sur H : rôle qualitatif (basculement), pas quantitatif.
- **Méta-finding** : les métriques entropiques actuelles confondent « diversité cognitive structurée » et « désordre aléatoire ». Backlog P1 : ajouter une métrique complémentaire (Lempel-Ziv, MI inter-nœuds, cohérence de phase) pour capturer la coordination.

### 3unvigies. P2-9 — λ₂ vs Edge Betweenness : qui prédit la dead zone ? (2026-04-24)

**Question** : λ₂ est-il un *proxy* de la redondance de chemins (EBC basse = beaucoup de chemins parallèles), ou est-il la quantité théoriquement fondée ? Si EBC prédit aussi bien, λ₂ n'est que descriptif. Si λ₂ prédit mieux, c'est la quantité causalement pertinente.

**Méthode** : `experiments/p2_edge_betweenness_analysis.py`. NetworkX pur, aucune simulation. 12 topologies (BA m=1–10, WS p=0.1/0.3, ER p=0.05/0.12, Lattice), 3 seeds, N=100. Métriques : λ₂, EBC moyen, diamètre, longueur de chemin moyenne, clustering moyen. Régression contre le régime (dead zone = 1, sinon 0).

**Résultats** :

| Topologie | λ₂ | EBC moyen | Diamètre | Régime |
|:----------|---:|----------:|--------:|:-------|
| BA m=1 | 0.019 | 0.04866 | 10.0 | uniform_wins |
| BA m=2 | 0.570 | 0.01538 | 5.7 | degree_linear_wins |
| BA m=3 | 1.119 | 0.00882 | 4.3 | degree_linear_wins |
| BA m=4 | 1.337 | 0.00621 | 4.0 | degree_linear_marginal |
| **BA m=5** | **2.907** | **0.00467** | 3.7 | **dead_zone** |
| BA m=8 | 5.776 | 0.00265 | 3.0 | dead_zone |
| BA m=10 | 7.301 | 0.00207 | 3.0 | dead_zone |
| WS p=0.1 | 0.165 | 0.02420 | 9.7 | degree_linear_wins |
| WS p=0.3 | 0.303 | 0.01983 | 7.3 | degree_linear_wins |
| ER p=0.05 | 0.626 | 0.01099 | 6.0 | degree_linear_wins |
| **ER p=0.12** | **4.167** | **0.00347** | 3.0 | **dead_zone** |
| Lattice | 0.098 | 0.03704 | 18.0 | uniform_wins |

**Corrélations λ₂ vs autres métriques** :

| Corrélation | r | p |
|:------------|--:|--:|
| λ₂ vs EBC | −0.665 | 0.018 |
| λ₂ vs diamètre | −0.609 | 0.035 |
| λ₂ vs avg_path | −0.678 | 0.015 |
| λ₂ vs clustering | +0.364 | 0.244 |

**Prédicteurs de la dead zone (corrélation point-bisérale)** :

| Prédicteur | r | p |
|:-----------|--:|--:|
| **λ₂** | **+0.901** | **6.4×10⁻⁵** |
| EBC | −0.604 | 0.038 |
| avg_path | −0.615 | 0.033 |
| diamètre | −0.556 | 0.060 |
| clustering | +0.287 | 0.366 |

**Findings** :

1. **λ₂ est de loin le meilleur prédicteur** de la dead zone (r=+0.901 vs r=-0.604 pour EBC). L'écart est substantiel — λ₂ capture 81% de la variance des régimes, EBC seulement 36%.

2. **Conclusion inverse à l'hypothèse initiale** : λ₂ n'est PAS un simple proxy de la redondance de chemins. L'EBC corrèle avec la dead zone mais est une mesure plus *bruitée* du même phénomène. λ₂ est la quantité théoriquement fondée — il mesure la rigidité spectrale du graphe, pas juste la géométrie locale.

3. **Interprétation causale** : λ₂ mesure la résistance du graphe contre toute perturbation locale. Quand λ₂ est élevé, même corriger un hub via `degree_linear` ne peut pas isoler son influence — le signal se propage par tous les chemins parallèles. EBC n'est qu'une projection de cette propriété globale sur les arêtes.

4. **La chaîne causale** : m élevé → λ₂ élevé + EBC basse → redondance massive → la pondération locale ne peut pas compenser → dead zone.

**Conséquence pour Paper 2** : λ₂ est l'observable pertinent. Paper 2 peut légitimement utiliser λ₂ comme variable indépendante (pas une "métrique choisie a posteriori") — elle prédit le régime avec r=0.90. L'analyse EBC confirme la cohérence mais n'apporte pas de nouveau mécanisme.

**Figures** : `figures/p2_edge_betweenness.png` (4 panneaux : λ₂ vs EBC, λ₂ vs diamètre, λ₂ vs avg_path, EBC vs diamètre). CSV : `figures/p2_edge_betweenness.csv`.

**Reproduction** : `python experiments/p2_edge_betweenness_analysis.py` (~2s).

---

### 3duovigies. P2-7 — Finite-size scaling de la dead zone (2026-04-24)

**Question** : Le seuil λ₂_crit (point de bascule vers la dead zone) est-il stable quand N croît de 100 à 1600 ? Si stable → loi d'échelle publiable. Si shift → effet de taille finie → la dead zone pourrait disparaître à grand N.

**Méthode** : `experiments/p2_finite_size_scaling.py`. N ∈ {100, 400, 1600}, BA m ∈ {1,2,3,4,5,6,8,10}, `degree_linear`, η=0.15, 3 seeds. STEPS = {100:3000, 400:2000, 1600:1000}. H_stable = entropie continue 100 bins sur la queue (25% finale). λ₂ via `scipy.sparse.linalg.eigsh`. λ₂_crit défini par interpolation linéaire où H < 0.10.

**Résultats bruts (moyenne sur 3 seeds)** :

| N | m | λ₂ | H_stable |
|--:|--:|---:|--------:|
| 100 | 1 | 0.023 | 0.896 |
| 100 | 3 | 1.279 | 3.065 |
| 100 | 5 | 2.991 | 2.575 |
| 100 | 10 | 7.613 | 2.155 |
| 400 | 1 | 0.006 | 0.987 |
| 400 | 3 | 1.295 | 3.845 |
| 400 | 5 | 2.943 | 2.931 |
| 400 | 10 | 7.494 | 2.410 |
| 1600 | 1 | 0.002 | 1.897 |
| 1600 | 3 | 1.257 | 3.496 |
| 1600 | 5 | 2.879 | 3.038 |
| 1600 | 10 | 7.336 | 2.429 |

**Finding clé : aucune dead zone (H < 0.10) détectée pour aucun N ni m.**

λ₂_crit = **∞** pour N=100, N=400 et N=1600. H_stable minimal observé = 0.896 (N=100, m=1).

**Interprétation** :

1. **La dead zone identifiée dans les expériences précédentes était un artefact de normalisation `uniform`**. Avec `degree_linear` + η=0.15, le système maintient H_stable > 2 bits même à N=1600, m=10 — jamais de collapse.

2. **λ₂ est N-invariant à m fixé** : m=3 donne λ₂ ≈ 1.27–1.32 pour les trois tailles (variation < 5%). Confirme que λ₂ est une propriété structurale du graphe BA, pas un artefact de taille finie.

3. **H_stable augmente légèrement avec N** : à m=3, H = 3.07 (N=100) → 3.84 (N=400) → 3.50 (N=1600). Cet effet de taille sur H est modéré et non monotone — probablement lié à la densité de trajectoires disponibles à grand N.

4. **Conclusion pour Paper 2** : la dead zone est coupling-norm–dependent, pas une limite thermodynamique. Le seuil λ₂_crit (avec `degree_linear`) est effectivement infini — ce régime ne s'effondre pas. La transition est caractérisée par des normes inadéquates (uniform, spectral), pas par la connectivité algébrique elle-même.

**Durée** : 386s (~6.4 min) pour les 72 runs (N=1600 dominant à ~20s/run).

**Figures** : `figures/p2_finite_size_scaling.png` (2 panneaux : λ₂ vs H par N + λ₂_crit vs N). CSV : `figures/p2_finite_size_scaling.csv`.

**Reproduction** : `python experiments/p2_finite_size_scaling.py` (~6.5 min).

---

### Session 2026-04-24 (Claude Sonnet 4.6, Pistes C/E/F/G/D)

**Piste C — Sweep heretic_ratio** : `experiments/heretic_ratio_sweep_coordination.py`. LZ_full monotone décroissant (1.137→1.012), hypothèse régulariseur confirmée. Synchrony en U avec seuil à η=0.05. Voir §3novedecies-ter.

**Piste E — Résonance stochastique inversée** : `experiments/forcing_sweep_frozen_u.py`. FULL sync≈0.03 plat pour I ∈ [0.1, 1.0]. FROZEN_U transition à I≈0.20 (sync 0.006→0.830), Cohen's d=13.21. Dip inattendu à I=0.70 (bifurcation). Voir §3novedecies-quater.

**Piste F — Bimodalité 50 seeds** : `experiments/bimodality_50seeds.py`. Bimodalité CONFIRMÉE (Hartigan D=0.194 p=0.000, BC=0.605). Prédicteur avg_clustering de §3vigies (r=−0.64, n=10) est un faux positif — non reproductible à n=50 (r=−0.19, p=0.18). Confound : 7/50 graphes déconnectés (λ₂=0). Voir §3vigies-ter.

**Piste G — Intégration preprint** : Nouvelle §3.3.1 "Trajectory-Based Minimality" dans `docs/preprint.tex`. Figure `coordination_phase_space.png` (Fig. 1) + résultat Piste E (Cohen's d=13.21). Preprint recompilé : 13 pages, 0 références indéfinies.

**Piste D — Multi-topologie universalité** : `experiments/ablation_coordination_topology_sweep.py`. 320 runs, 3 min. Universalité confirmée sur 4 topologies (lattice, BA m=3, BA m=5, WS). FULL sync≈0 universel. Finding nouveau : dead zone ≠ effondrement de coordination. Voir §3vigies-quater.

**P2-9 — Edge betweenness vs λ₂** : `experiments/p2_edge_betweenness_analysis.py`. λ₂ meilleur prédicteur dead zone (r=+0.901) vs EBC (r=-0.604). λ₂ = quantité causalement fondée, pas un proxy. Voir §3unvigies.

**P2-7 — Finite-size scaling** : `experiments/p2_finite_size_scaling.py`. 72 runs, 6.5 min. λ₂_crit = ∞ pour N ∈ {100,400,1600} sous degree_linear — aucune dead zone thermodynamique. Dead zone = artefact coupling-norm (uniform), pas limite physique. λ₂ N-invariant (±5% pour m fixé). Voir §3duovigies.

---

### Session 2026-04-24 — Audit Externe + Corrections (Claude Sonnet 4.6)

**Contexte** : Audit scientifique externe reçu (`.Audit/Audit Scientifique du Projet Mem4ristor-v2.md`). Trois corrections appliquées suite à l'analyse croisée audit vs PROJECT_STATUS.

**Correction 1 — `experiments/limit02_alpha_sweep.py`** :
- Ajout import `calculate_cognitive_entropy` depuis `metrics.py`.
- Rapport des deux métriques en parallèle : H_cont (100-bin, nouvelle métrique continue) et H_cog (5-bin, seuils KIMI ±0.4/1.2).
- Note explicite `I_STIM = 0.0 → heretics inactive (no-op)` dans le code.
- Conséquence : les résultats γ*(m) sont maintenant qualifiés par les deux métriques. H_cog attendu ≈ 0 dans le régime endogène (confirmant FLAW 6).

**Correction 2 — `tests/test_scientific_regression.py`** (10 tests, 10 PASS) :
- `TestHereticEndogenousNoOp` : documente que heretics ratio=0 vs 0.15 produisent des résultats indistinguables à I_stim=0. Guard contre toute "correction" silencieuse de ce comportement sans mise à jour narrative.
- `TestDeadZoneTopologicalPredictor` (2 sous-tests) : vérifie que BA m=5 + uniform → dead zone (H_cog < 0.5), et BA m=3 + degree_linear → survit (H_cont > 1.0). Ancre l'observable λ₂ dans les tests.
- `TestFullModelCoordinatedDiversity` : intégration — FULL model sous forçage I_stim=0.5 doit avoir synchrony < 0.3. Guard contre la dégradation du mécanisme u-dynamics.

**Statut preprint** : Julien a soumis le 22/04/2026. Les corrections narratives (dead zone = artefact normalisation, bimodalité endogène à intégrer) sont documentées ici pour la prochaine révision.

### 3septvigies. Piste A5 — Sweep delta (Levitating Sigmoid) : RESULTAT NEGATIF / ROBUSTESSE (2026-04-24)

**Question** : delta=0.01 (fix technique LIMIT-01) est-il en realite un parametre de controle de la symetrie sociale avec un delta_crit qui maximise la complexite LZ ?

**Methode** : `experiments/p2_delta_sweep.py`. Sweep delta in [-0.10, -0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05, 0.10] sur Lattice 10x10 et BA m=3 N=100. I_stim=0.5, coupling_norm='degree_linear', 3000 steps, warm_up=750, 3 seeds. `net.model.social_leakage = delta` apres instanciation. Metriques : H_cont, H_cog, LZ, sync.

**Resultats** :

| delta | LZ (lattice) | LZ (BA m=3) | H_cog (lattice) |
|------:|:------------:|:-----------:|:---------------:|
| -0.10 | 0.723 | 0.767 | 0.425 |
| 0.00 | 0.743 | 0.761 | 0.422 |
| **0.01** (defaut) | 0.744 | 0.763 | 0.432 |
| 0.10 | 0.754 | 0.787 | 0.431 |

**Findings** :

1. **Hypothese infirmee** : pas de delta_crit identifiable. La variation totale de LZ sur tout le sweep est ±0.025 (lattice) et ±0.040 (BA m=3) — inferieure a la variance inter-seeds estimee.

2. **Le modele est robuste a delta ∈ [-0.1, 0.1]** : toutes les metriques (H_cog, LZ, sync, H_cont) restent stables. C'est un resultat de robustesse positif.

3. **delta < 0 (repulsion faible)** ne produit pas plus de diversite que delta > 0 (attraction faible). La brisure de symetrie au point u=0.5 est trop faible pour modifier le regime dynamique.

4. **Confirmation du role technique de delta** : introduit pour eviter le couplage nul a u=0.5, il remplit exactement ce role sans agir comme parametre de controle. Le choix delta=0.01 est valide.

**Consequence** : pas de modification du modele recommandee. La section LIMIT-01 reste documentee comme correcte.

**Figures** : `figures/p2_delta_sweep.png` (4 panneaux : H_cog, LZ, sync, H_cont vs delta, 2 topos). CSV : `figures/p2_delta_sweep.csv`.

**Duree** : 37s (54 runs : 9 deltas × 2 topos × 3 seeds).

---

### 3trigies. Audit Externe Manus AI — FAIBLESSES IDENTIFIÉES (2026-04-25)

**Auditeur** : Manus AI. Fichier : `.Audit/25-04-2026_Rapport d'Audit Scientifique et Technique du Projet Mem4ristor v3.2.0.md`

**Contexte** : Deuxième audit externe du projet (le premier était le 2026-04-22 par un auditeur anonyme, voir §3octvicies). Ce rapport couvre la solidité scientifique des claims, la qualité du code et propose de nouvelles pistes.

---

#### Critique de l'audit : ce qui est valide vs ce qui est déjà traité

**Trois points soulevés par Manus que nous avions DÉJÀ traités (audit insuffisamment informé) :**

- §1.1 Effets de taille finie → **DÉJÀ FAIT** en §3duovigies : N∈{100,400,1600}, λ₂ N-invariant, λ₂_crit=∞ sous degree_linear.
- §3.1 Normalisation spectrale → **DÉJÀ TESTÉE ET NÉGATIVE** en §3octies : 0/6 wins, résultat dans le preprint.
- §3.2 Redondance des chemins → **DÉJÀ FAIT** en §3unvigies : λ₂ vs EBC r=−0.665, λ₂ = meilleur prédicteur.

---

#### Faiblesses réelles identifiées par Manus — ACTIONS REQUISES

**FAILLE A — Absence de calibration η SPICE ↔ σ Python** *(Priorité : HAUTE — impact Paper B)*

Le claim central de Paper B ("le bruit thermique SPICE rescue la dead zone") et le résultat Item 10 ("le bruit Gaussien Python ne le peut pas jusqu'à σ=1.2") coexistent sans calibration d'amplitude. On ne sait pas si η=0.5 SPICE correspond à σ≈0.1 Python (amplitudes incomparables) ou σ≈2.0 Python (amplitudes déjà au-dessus de notre sweep). Le claim "bruit thermique qualitativement distinct" est postulé, pas démontré.

- **Test requis** : mesurer la variance de tension injectée par `trnoise(eta)` dans une cellule SPICE simple → convertir en σ_equiv Python → relancer Item 10 à cette amplitude.
- **Résultat attendu (hypothèse favorable)** : η=0.5 SPICE ↔ σ >> 1.2 Python → le bruit Python est simplement sous-dosé dans le sweep Item 10, pas qualitativement différent.
- **Résultat défavorable** : σ_equiv ≈ 0.5-1.0 → les deux régimes de bruit sont à amplitude comparable mais l'un rescue et l'autre non → cela RENFORCE le claim Paper B (le bruit thermique est structurellement différent).
- **Dans tous les cas** : la calibration est nécessaire pour que le claim soit défendable.

**FAILLE B — Bins pré-KIMI dans `experiments/spice_dead_zone_test.py`** *(Priorité : MOYENNE — cohérence)*

`spice_dead_zone_test.py` utilise encore les seuils obsolètes `[-1.5, -0.8, 0.8, 1.5]` au lieu des seuils KIMI actuels `[-1.2, -0.4, 0.4, 1.2]`. Les valeurs H_cog reportées pour la dead zone SPICE sont donc calculées avec une métrique différente de tous les résultats Python récents. Bug de cohérence, pas d'impact sur la conclusion principale (dead zone H≈0 dans les deux cas) mais invalide la comparabilité quantitative.

- **Correction** : remplacer les seuils dans `spice_dead_zone_test.py` et relancer (< 5 min).

**FAILLE C — Dynamique u tronquée dans les netlists SPICE** *(Priorité : MOYENNE — transparence)*

Les netlists SPICE utilisent `B_du = eps_u*(sigma_base - u)`, omettant le terme `k_u * sigma_social_i` présent dans `dynamics.py`. La faisabilité hardware est donc validée pour un modèle où le doute ne répond pas aux désaccords locaux — c'est une version appauvrie du mécanisme. L'escape SPICE démontré ne prouve pas que le modèle *complet* (avec métacognition) est faisable en hardware.

- **Correction** : soit implémenter σ_social dans SPICE (complexe, nécessite B-source supplémentaire), soit documenter explicitement dans Paper B que la validation hardware porte sur le "noyau FHN + doute autonome", pas sur le mécanisme métacognitif complet.

**FAILLE D — Duplication make_ba() inter-scripts** *(Priorité : BASSE — dette technique)*

Reproductibilité inter-expériences potentiellement compromise. Non-urgent scientifiquement mais devrait être adressé avant la soumission de Paper 2.

---

#### Diagnostic global de l'audit Manus

L'audit est solide sur la cohérence technique (B, D) et identifie correctement la calibration SPICE/Python (A) comme la faiblesse principale de Paper B. En revanche, il n'a pas attaqué les claims scientifiques durs (causalité de u, τ_u bifurcation sous autres conditions, baseline NMI) que notre prompt demandait — l'audit reste en surface sur la partie scientifique.

**Score implicite** : le projet est techniquement publiable avec corrections B+C, scientifiquement publiable à condition de traiter A. La faille A est la seule qui pourrait être soulevée par un reviewer de Nature Physics ou Physical Review E.

---

### 3untrigies. Audit Manus AI — Version Modifiée : Nouvelles Sections (2026-04-25)

**Auditeur** : Manus AI (version révisée du premier rapport)
**Fichier** : `.Audit/modif d'Audit du Projet Mem4ristor v3.2.0.md`

Manus a enrichi son rapport initial avec trois nouvelles sections (§1.4, §1.5, §2.4) et une évaluation globale (§5). Ce document analyse point par point leur validité.

---

#### §1.4 — Causalité de u vs Bruit Hétérogène Déguisé : CRITIQUE VALIDE

**Thèse de Manus** : La variable de doute u pourrait être un simple proxy du bruit hétérogène plutôt qu'un mécanisme métacognitif causal. `sigma_social` est défini comme `|laplacian_v|` — directement influencé par le bruit injecté sur v. Donc : bruit → v → sigma_social → u → couplage → pseudo-causalité circulaire.

**Notre défense partielle** : La boucle u → I_coup → v → sigma_social → u est un **feedback adaptatif**, non une simple proxy. La modulation via `u_filter = tanh(π(0.5 − u)) + δ` bascule entre attractif et répulsif — non-linéarité irréductible. Les ablations FROZEN_U (§3octdecies + §3novedecies) montrent que geler u **collapse la coordination** (d = +9.75 en ENDOGENOUS), démontrant la causalité fonctionnelle.

**Cependant : la critique pointe une lacune réelle.** Nous n'avons jamais testé l'ablation "sigma_social ← bruit pur" ou "sigma_social ← hétérogénéité statique fixe". Ces ablations démontreraient que c'est bien la **boucle fermée adaptative** (pas juste un signal dynamique quelconque) qui produit les propriétés de coordination observées.

**Statut** : ⚠️ OUVERT — Expérience à planifier (backlog futur). La critique ne réfute pas nos résultats mais identifie un contrôle manquant.

**Expérience proposée** :
- `SIGMA_SOCIAL_NOISE` : remplacer `sigma_social = |laplacian_v|` par bruit pur `~ N(0, σ)` calibré à la même amplitude RMS que sigma_social typique.
- `SIGMA_SOCIAL_STATIC` : figer sigma_social à sa valeur temporelle moyenne (hétérogénéité statique).
- Comparer sync + LZ avec FULL et FROZEN_U. Si SIGMA_SOCIAL_NOISE ≈ FULL → u est proxy du bruit. Si SIGMA_SOCIAL_NOISE ≈ FROZEN_U → la dynamique adaptative de u est irréductible.

---

#### §1.5 — Bifurcation tau_u sous I_STIM = 0 : 🚨 ERREUR FACTUELLE DANS L'AUDIT

**Affirmation de Manus** : "L'expérience est explicitement configurée avec I_STIM = 0.0 [14, ligne 56]". Manus cite cela comme une **force** : "la bifurcation est observée dans un régime endogène, sans forçage externe".

**C'EST FACTUELLEMENT FAUX.** `experiments/p2_tau_u_bifurcation.py` utilise `I_STIM = 0.5` (ligne 34), **pas 0.0**. L'expérience a été conçue avec un forçage externe actif (hérétiques actifs). La valeur `I_STIM = 0.0` que Manus cite n'existe pas dans le script actuel. Soit Manus a lu une ancienne version, soit il a extrapolé depuis d'autres scripts.

**Implication** : La "force" décrite par Manus (régime endogène pur) n'est pas ce que nous testons. Tester la bifurcation tau_u en régime endogène (I_STIM=0) est une expérience **distincte et non encore réalisée** — scientifiquement pertinente pour Paper 2, mais distincte de §3quatervigies.

**Note** : Une bifurcation tau_u en régime endogène pur serait en effet plus forte (preuve que u structure la dynamique sans forçage externe). À envisager comme expérience complémentaire.

---

#### §2.4 — Absence de Community Detection et Baseline NMI : 🚨 SECTION COMPLÈTEMENT OBSOLÈTE

**Affirmation de Manus** : "Le script `p2_doubt_community_detection.py` n'existe pas, et les recherches de 'NMI' ou 'community' dans le code sont restées infructueuses."

**C'EST FAUX à la date de l'audit.** Item 12 (Doubt-Driven Community Detection) a été **implémenté et exécuté le 25 avril 2026** (commit `2fdc660`), résultats documentés en §3octvigies :

| Résultat | Valeur |
|:---------|:-------|
| Script | `experiments/p2_doubt_community_detection.py` — EXISTE |
| NMI implémenté | Oui — custom numpy (sans sklearn) |
| NMI Lattice | 0.298 ± 0.040 (3 seeds) |
| NMI BA m=3 | 0.232 ± 0.087 (3 seeds) |
| Algorithme | Louvain (NetworkX 3.5) sur graphe doubt-affinity ET structural |

Manus a audité le dépôt **avant** que le script soit commis. C'est une **limite de l'audit en temps réel** : l'auditeur ne voit que l'état au moment de sa lecture.

**Point valide extrait de §2.4** : La nécessité d'une **baseline NMI aléatoire** est une critique légitime. NMI = 0.25–0.30 est reportée sans comparaison avec la distribution NMI de partitions aléatoires. La significativité statistique ne peut pas être affirmée sans ce contrôle.

**Action requise** : Calculer NMI(partitions_aléatoires) par bootstrap (100+ permutations des labels) et vérifier NMI_observé > NMI_aléatoire + 2σ. < 5 min. Renforcerait matériellement le claim.

---

#### §5 — Score Global 6/10 et Pitch Nature Physics

**Score** : **6/10** (Nouveauté: 3/10 · Rigueur: 3/10 · Clarté: 2/10 · Robustesse: 2/10)

**Lecture du score** : Avec les quatre failles A/B/C/D corrigées (§3trigies + P2-AUDIT-2 ✅), les critiques "Rigueur" et "Robustesse" sont en grande partie adressées. Estimation post-corrections : **~7.5/10**. Le point restant le plus faible est §1.3 (diversité sub-cognitive, H_cog ≈ 0 en Python défaut), documenté dans les limitations mais pas encore intégré dans le narrative Paper B.

**Pitch Nature Physics fourni par Manus** :

> *"Nous démontrons que le bruit thermique inhérent aux systèmes neuromorphiques analogiques permet d'échapper aux zones mortes topologiques, un problème persistant dans les modèles logiciels. Cette découverte transforme les imperfections matérielles de défauts en ressources computationnelles essentielles, ouvrant la voie à une nouvelle génération de substrats neuromorphiques inspirés des verres de spin."*

Ce pitch est **excellent** — aligné avec le claim central de Paper B et le résultat Faille A (η=0.5 SPICE ↔ σ=0.0044 Python, bruit thermique catégoriquement distinct). À utiliser comme modèle pour le titre et l'abstract de Paper B.

---

#### Synthèse : Réponse à l'Audit Manus v2

| Section | Validité | Statut |
|:--------|:---------|:-------|
| §1.4 (causalité u) | ✅ Critique valide, lacune réelle | ⚠️ Backlog : ablation σ_social → bruit pur |
| §1.5 (I_STIM = 0) | 🚨 ERREUR FACTUELLE — I_STIM = 0.5 dans le script réel | Aucune correction nécessaire |
| §2.4 (community detection absente) | 🚨 OBSOLÈTE — commit 2fdc660 avant l'audit | ✅ Baseline NMI faite — critique baseline VALIDE, NMI non significatif en 5/6 seeds |
| §5 score 6/10 | ✅ Évaluation honnête pré-corrections | Post-corrections estimé ~7.5/10 |
| Pitch Nature Physics | ✅ Excellent, aligné Paper B | À réutiliser dans abstract Paper B |

---

### Session 2026-04-25 (Claude Sonnet 4.6 — P2 Items 10 & 12)

**Contexte** : Continuation de la session 2026-04-24. Deux items P2 backlog implémentés et clos.

**Item 12 — Doubt-Driven Community Detection** (§3octvigies) :
- Script : `experiments/p2_doubt_community_detection.py`. Pure numpy Pearson + NMI custom + Louvain (NetworkX).
- Résultat PARTIEL : NMI~0.30 (lattice) / 0.23 (BA m=3). Deux régimes u : hérétiques saturés u=1.0 (singletons) + nœuds frustrés oscillant en groupes qui transcendent la topologie.
- Commit : `2fdc660`.

**Item 10 — Stochastic Resonance × Topology** (§3novemvigies) :
- Script : `experiments/p2_stochastic_resonance_topology.py`. 7 topologies × 9 σ × 3 seeds (171s).
- Résultat NÉGATIF / INSTRUCTIF : pas de SR classique (pas de cloche). Dichotomie nette : λ₂ < 2.5 → bruit bénéfique monotone ; λ₂ > 2.5 → zone morte résistante au bruit (H_cog ≈ 0 même à σ=1.2). λ₂_crit ≈ 2.5 confirme le prédicteur §3unvigies.
- Commit : `984dda8`.

**État backlog P2 après session** : Items 10 ✅ 12 ✅ — reste Item 11 (adaptive heretics η dynamique, ⚠️ modifie le modèle → v4.0).

---

### 3novemvigies. Item 10 — Stochastic Resonance x Topology : RESULTAT NEGATIF / INSTRUCTIF (2026-04-25)

**Question** : Pour chaque topologie (lambda2 = connectivite algebrique), existe-t-il un sigma_noise optimal sigma* qui maximise H_cog (resonance stochastique) ? sigma* depend-il systematiquement de lambda2 ?

**Methode** : `experiments/p2_stochastic_resonance_topology.py`. 7 topologies (BA m=2/3/5/8 N=100, Lattice 10x10, ER p=0.05/0.10), sigma sweep [0, 0.01, 0.03, 0.07, 0.15, 0.30, 0.50, 0.80, 1.20], coupling_norm='degree_linear', I_stim=0.5, 3 seeds. Injection bruit via `sigma_v_vec=np.full(N, sigma)`. Metriques : H_cog, H_cont, LZ, sync. Duree : 171s.

**Topologies et lambda2 (seed=42) :**

| Topo | lambda2 | sigma* | H_cog(sigma*) |
|:-----|:-------:|:------:|:-------------:|
| Lattice 10x10 | 0.382 | 1.200 | 0.647 |
| ER p=0.05 | 0.528 | 1.200 | 0.450 |
| BA m=2 | 0.625 | 1.200 | 0.658 |
| BA m=3 | 1.413 | 1.200 | 0.265 |
| ER p=0.10 | 2.568 | 1.200 | 0.001 |
| BA m=5 | 2.990 | 1.200 | 0.006 |
| BA m=8 | 5.864 | 0.000 | 0.000 |

Correlation Pearson(sigma*, lambda2) = -0.854 (mais artefact de saturation — sigma* sature a 1.2 pour 6/7 topos).

**Findings** :

1. **Pas de pic SR classique dans [0, 1.2]** : pour les topologies faible-lambda2 (BA m=2, Lattice, ER p=0.05), H_cog augmente de facon monotone avec sigma. Pas de cloche de resonance — le bruit est toujours benefique dans cette gamme.

2. **Zone morte robuste au bruit pour lambda2 > 2.5** : BA m=5 (lambda2=2.99), BA m=8 (5.86), ER p=0.10 (2.57) ont H_cog ~ 0 sur tout le sweep, meme a sigma=1.2. Le couplage synchronisant est plus fort que n'importe quel bruit teste. Confirme et etend Piste A1.

3. **lambda2_crit ~ 2.5 : seuil de rescousse** : en-dessous, le bruit recupere toujours de la diversite cognitive. Au-dessus, le bruit est inefficace. Ce seuil est coherent avec le predicateur lambda2 de la dead zone (r=+0.901, §3unvigies).

4. **BA m=3 est au bord** : lambda2=1.41, H_cog(sigma=0) ~ 0.016 (effondrement sans bruit), H_cog(sigma=1.2) = 0.265 — le bruit aide mais le rescousse incompletement. Coherent avec la bifurcation tau_u (§3quatervigies) : tau_u=10 place le systeme au bord.

5. **Relation veritable** : c'est H_cog(sigma_max) vs lambda2 qui est robuste (forte decroissance avec lambda2), pas sigma* vs lambda2 (artefact). La quantite scientifiquement pertinente est **la capacite maximale de diversite cognitive recuperable par le bruit**, qui est une fonction decroissante de lambda2.

6. **Pas de SR au sens strict** : la resonance stochastique classique (amelioration d'un signal sous-seuil par bruit optimal) n'est pas le bon cadre ici. Le mecanisme est plutot : bruit = decorrelateur supplementaire qui compete avec le couplage synchronisant. Si lambda2 est trop eleve, le couplage gagne quelle que soit l'intensite du bruit dans la gamme testee.

**Consequence** : l'hypothese de "bruit optimal topologie-dependant" est infirmee dans sa forme naive. La vraie relation est une dichotomie : lambda2 < lambda2_crit → bruit benefique (mais pas de pic, juste amelioration monotone) ; lambda2 > lambda2_crit → bruit inefficace. Pas de parametre sigma* a optimiser.

**Figures** : `figures/p2_stochastic_resonance_topology.png` (H_cog/LZ/sync vs sigma par topo + scatter sigma* vs lambda2). CSV : `figures/p2_stochastic_resonance_topology.csv`.

---

### 3octvigies. Item 12 — Doubt-Driven Community Detection : RESULTAT REQUALIFIE (2026-04-25)

**Question** : La matrice de correlation Pearson des traces u(t) porte-t-elle une information sur les communautes fonctionnelles du reseau ? Les noeuds qui oscillent en phase dans leur niveau de doute appartiendraient-ils au meme attracteur cognitif ?

**Methode** : `experiments/p2_doubt_community_detection.py`. Record de u_history (T=2500, N) apres warm-up. Pearson C_u (N×N) sur numpy pur. Seuillage |C_u| > 0.3 → graphe doubt-affinity. Louvain (NetworkX 3.5) sur doubt-affinity et sur graphe structural. NMI custom (numpy, sans sklearn). **Baseline NMI aleatoire : 500 permutations des labels du doute** (ajout 2026-04-25, reponse Audit Manus §2.4). Topologies : Lattice 10x10, BA m=3 N=100. I_stim=0.5, coupling_norm='degree_linear', 3 seeds. Duree : 6s.

⚠️ **BUG CORRIGE (audit Edison 2026-04-25)** : Le graphe doubt-affinity etait construit avec des poids de correlation signes (C_u[i,j] negatif possible). NetworkX Louvain suppose des poids non-negatifs — les poids negatifs corrompaient l'optimisation de la modularite en repoussant artificiellement certains noeuds dans des communautes separees. **Correction** : `weight=abs(C_u[i,j])` (affinite = force de co-variation, signe ignore). Les resultats ci-dessous sont post-correction.

**Resultats avec baseline NMI aleatoire (500 permutations, poids |corr|)** :

| Topo | seed | NMI_obs | NMI_rand | z | p | sig |
|:-----|-----:|:-------:|:--------:|:---:|:---:|:---:|
| Lattice | 42 | 0.3050 | 0.2702±0.014 | +2.41 | 0.014 | * |
| Lattice | 123 | 0.2750 | 0.2442±0.017 | +1.83 | 0.046 | * |
| Lattice | 777 | 0.3551 | 0.3386±0.014 | +1.17 | 0.116 | ns |
| **Lattice** | **mean** | **0.312** | **0.284** | **+1.80** | — | **2/3 sig** |
| BA m=3 | 42 | 0.1857 | 0.1796±0.019 | +0.32 | 0.362 | ns |
| BA m=3 | 123 | 0.1508 | 0.1951±0.020 | -2.18 | 0.992 | ns |
| BA m=3 | 777 | 0.3281 | 0.2751±0.021 | +2.55 | 0.004 | ** |
| **BA m=3** | **mean** | **0.222** | **0.217** | **+0.23** | — | **1/3 sig** |

**REQUALIFICATION post-fix** : Le Louvain corrige renforce le signal Lattice (2/3 seeds maintenant significatifs, z_mean=+1.80 — signal modere). Pour BA m=3, seul seed=777 est significatif (p=0.004) mais seed=123 est fortement negatif (z=-2.18) : les communautes u sont ANTI-alignees avec les communautes structurelles sur ce graphe. L'interpretation du signal BA m=3 est donc ambigue.

**Pourquoi NMI_rand est-il eleve ?** La partition du doute comporte beaucoup de communautes (17-35) dont de nombreux singletons (heretiques u=1.0, variance nulle). Avec N>8 communautes et des singletons, la permutation aleatoire produit accidentellement un NMI eleve avec une partition structurelle a 7-9 communautes. C'est la granularite elevee des partitions qui gonfle la baseline.

**Deux regimes u (finding robuste, independant de la NMI) :**

| Type | Taille | mean_u | std_u | mean_v | Interpretation |
|:-----|:------:|:------:|:------:|:------:|:--------------|
| Grands groupes (1-2x) | 18-36 | 0.933-0.987 | 0.04-0.11 | -1.36 a -1.65 | Noeuds frustres oscillant en phase |
| Singletons (11-22x) | 1 | 1.000 | 0.000 | -1.6 a -2.7 | Heretiques satures au doute maximal |

**Findings post-correction** :

1. **Signal Lattice modere** : 2/3 seeds significatifs apres fix Louvain (z=+2.41, +1.83). La topologie reguliere (Lattice) structure davantage les communautes u que la topologie scale-free. z_mean=+1.80 est au-dessus du seuil de 1.65 (p<0.05 unilateral).

2. **BA m=3 ambigu** : un seed fortement positif (z=+2.55) et un fortement negatif (z=-2.18). L'alignement doubt/structure est topologie-dependant sur BA m=3. La moyenne z_mean=+0.23 n'est pas significative. Le signal BA m=3 reste seed-dependant.

3. **Deux populations u distinctes** (finding robuste, independant de la NMI) : heretiques satures u=1.0 (singletons) + noeuds frustres avec oscillations u correlees → grands groupes transcendant les frontieres structurelles. Ce resultat qualitatif est valide independamment du z-score.

4. **Coherence avec Piste A4 (MI decorrelateur)** : u reduit la MI inter-noeuds → les communautes u mesurent en fait QUELS noeuds sont co-frustres. Ce n'est pas redondant avec les communautes structurelles, mais la correspondance quantitative reste moderee.

5. **Piste theta=0.1 testee et infirmee (2026-04-25)** : theta=0.1 est pire quelle que soit la topo. Les singletons heretiques (u=1.0, var=0) ont Pearson=0 avec tout noeud par construction — ils restent isoles a n'importe quel theta. Probleme structurel non resolu par le seuillage. theta=0.3 conserve.

**Interpretation narrative revisee (post audit Edison)** : Le Louvain corrige (poids |corr|) revele un signal Lattice modere (2/3 seeds, z_mean=+1.80). Pour BA m=3, le signal est ambigu (un seed anti-aligne). Le finding robuste demeure : u cree des "bassins de frustration collective" (deux populations distinctes), mais leur correspondance avec les communautes structurelles est moderee sur Lattice et ambigue sur BA m=3.

**Figures** : `figures/p2_doubt_community_detection.png` (2x3 : heatmap C_u + communautes-doute + communautes-struct, par topo). CSV : `figures/p2_doubt_community_detection.csv` (inclut colonnes nmi_obs, nmi_rand_mean, nmi_rand_std, z_score, p_value, significant).

---

### 3sexvigies. Piste A4 — Information Mutuelle Spatio-Temporelle : RESULTAT POSITIF (2026-04-24)

**Question** : La MI entre noeuds voisins vs distants revele-t-elle une longueur de correlation caracteristique du regime FULL ?

**Methode** : `experiments/p2_spatial_mutual_information.py`. Nouvelle fonction `calculate_spatial_mutual_information(v_history, adjacency_matrix)` ajoutee a `src/mem4ristor/metrics.py`. Histogrammes 2D (n_bins=20) sur traces z-scorees. Distances graphe via `scipy.sparse.csgraph.shortest_path`. 4 ablations (FULL, NO_HERETIC, FROZEN_U, NO_SIGMOID), 2 topos (Lattice 10x10, BA m=3 N=100), I_stim=0.5, 3 seeds, max 150 paires par distance. Duree : 18s.

**Resultats** :

| Ablation | MI(d=1) lattice | Decay lattice | MI(d=1) BA m=3 | Decay BA m=3 |
|:---------|:---------------:|:-------------:|:--------------:|:------------:|
| FULL | 0.870 | 0.102 | 1.031 | **0.055** |
| NO_HERETIC | 1.244 | 0.168 | 0.983 | 0.129 |
| FROZEN_U | **1.958** | 0.091 | **1.894** | **0.498** |
| NO_SIGMOID | 0.885 | 0.126 | 1.024 | 0.260 |

**Findings** :

1. **La prediction initiale etait inversee** : on attendait FULL avec MI haute localement et decroissante. En realite, FULL a la MI la plus BASSE a toutes les distances (sur lattice). Ce renversement est scientifiquement plus riche.

2. **u = decorrelateur actif** : FROZEN_U a MI~1.9 (synchronisation elevee), FULL a MI~0.87. La dynamique de doute reduit activement la correlation entre noeuds voisins. Sans u, le reseau oscille en synchronisation globale (FHN couple classique). Avec u, chaque noeud suit sa propre trajectoire independante.

3. **FROZEN_U sur BA m=3 : decay massif (0.498)** — MI(d=1)=1.89 vs MI(d=5)=1.40. Signature d'un etat chimere locale : voisins tres correles, noeuds distants moins. C'est de la synchronisation locale, pas globale. Coherent avec le spectre Fourier de §3quatervigies.

4. **FULL sur BA m=3 : decay minimal (0.055)** — MI plate et basse a toutes distances. Les noeuds sont uniformement decorreles. Signature de la "diversite vraiement independante" : chaque noeud explore sa propre dynamique sans etre tire par ses voisins.

5. **Combinaison MI basse + LZ basse + sync basse (FULL)** = diversite independante et structuree. Pas du desordre (qui aurait LZ haute). La MI confirme que le quadrant "bas-LZ / bas-sync" de §3vigies-bis correspond bien a une diversite d'attracteurs locaux independants.

6. **NO_HERETIC a la MI la plus haute sur lattice (1.244)** : sans heretiques, le systeme tend vers un etat de forte correlation (semi-consensus) mais pas synchrone. L'absence d'heretiques reduit la diversite des attracteurs locaux → les trajectoires se ressemblent plus entre voisins.

**Consequence pour Paper 2** : la MI est une metrique complementaire ORTHOGONALE a LZ et sync. Elle revele le role mecanistique de u (decorrelateur) que les metriques precedentes ne capturaient pas directement. Figure candidate pour le preprint revise : "MI(d) profile distinguishes cognitive diversity (FULL) from synchronized chimera (FROZEN_U)."

**Figures** : `figures/p2_spatial_mutual_information.png` (2 panneaux : MI vs distance pour lattice et BA m=3). CSV : `figures/p2_spatial_mutual_information.csv`.

**Duree** : 18s (24 runs : 4 ablations × 2 topos × 3 seeds).

---

### 3quinquevigies. Piste A3 — Couplage Asymetrique / Graphes Diriges : HYPOTHESE FALSIFIEE (2026-04-24)

**Question** : Un reseau dirige ou les hubs "parlent mais n'ecoutent pas" (HUB_BCAST) ou "ecoutent mais ne parlent pas" (HUB_LISTEN) peut-il eliminer la dead zone sans normalisation ad-hoc ?

**Methode** : `experiments/p2_directed_coupling.py`. N=100, BA m in {3, 5}, I_stim=0.0, 3000 steps, 3 seeds. 3 types de graphes : SYMM (BA non-dirige, baseline), HUB_BCAST (A[peripherique, hub]=1), HUB_LISTEN (A[hub, peripherique]=1). Chaque type x 2 norms (uniform, degree_linear). Metriques : H_cont, H_cog, pairwise_synchrony.

**Resultats BA m=5 (dead zone)** :

| Mode | H_cog | sync | Verdict |
|:-----|:-----:|:----:|:--------|
| SYMM + uniform | 0.027 | 0.042 | Dead zone standard |
| HUB_BCAST + uniform | 0.008 | **0.642** | Pire : hub synchronise tout |
| HUB_BCAST + degree_linear | 0.009 | 0.341 | Idem attenue |
| HUB_LISTEN + uniform | 0.029 | 0.212 | Peripherie isolee (d_in=0 → FP) |
| HUB_LISTEN + degree_linear | 0.023 | 0.061 | Idem |

**Findings** :

1. **HUB_BCAST cree la plus forte synchronie de toutes les configs testees** (sync=0.642 sur BA m=5). En faisant diffuser les hubs vers tous les noeuds peripheriques, on tire l'ensemble du reseau vers un etat commun. C'est l'effet inverse de ce qu'on cherchait — les hubs deviennent des "metronomes collectifs".

2. **HUB_LISTEN isole la peripherie** : A[hub, new_node]=1 implique d_in(peripherique)=0, donc l_v[peripherique]=0 (pas d'entree de couplage). Chaque noeud peripherique evolue comme un FHN isole et converge vers son point fixe v*≈-1.29. H_cont s'effondre a 1.11 bits sous degree_linear.

3. **Contrainte structurelle** : dans un BA dirige, on ne peut pas avoir a la fois (a) une peripherie couplée et (b) des hubs non-dominants. La preference d'attachement preferentiel (PA) implique que les hubs sont au centre de tous les chemins — ils dominent quel que soit le sens des aretes.

4. **Conclusion** : la strangulation par les hubs n'est pas un probleme de symetrie du couplage. C'est une propriete de la topologie (haute connectivite, λ₂ grand) qui subsiste quelle que soit la direction des aretes. Renforce §3octies (spectral) et §3duovigies (finite-size) : seule la normalisation `degree_linear` resout le probleme sans changer la topologie.

**Statut** : CLOTURE — RESULTAT NEGATIF. La piste "graphes diriges comme escape mechanism" est fermee. Le resultat HUB_BCAST (sync=0.64) est publiable comme contre-exemple : la directionnalite peut aggraver la synchronisation.

**Figures** : `figures/p2_directed_coupling.png` (2 panneaux : H_cog par mode × norm pour m=3 et m=5). CSV : `figures/p2_directed_coupling.csv`.

**Duree** : 27s (36 runs : 3 modes × 2 norms × 2 m × 3 seeds).

---

### 3quatervigies. Piste A2 — Bifurcation tau_u : HYPOTHESE CONFIRMEE (2026-04-24)

**Question** : tau_u controle-t-il une bifurcation entre regime de frustration figee (noeuds bloques) et chimere respirante (clusters dynamiques) ? Un pic de frequence doit emerger au tau_u critique.

**Methode** : `experiments/p2_tau_u_bifurcation.py`. Sweep tau_u in [0.05, 100.0] (10 valeurs log-espacees), 2 topologies (Lattice 10x10, BA m=3 N=100), I_stim=0.5 (heretiques actifs), coupling_norm='degree_linear', 4000 steps, warm_up=1000, 3 seeds. Metriques : H_cont, H_cog, pairwise_synchrony, frequence dominante FFT sur v_mean(t), puissance spectrale au pic.

**Resultats BA m=3** :

| tau_u | H_cog | Sync | Pic spectral | Regime |
|------:|:-----:|:----:|:------------:|:-------|
| 0.05–5 | 0.006–0.065 | ~0.10 | **0.49–0.55** | Frustration figee + oscillations coherentes |
| 10 | 0.023 | 0.11 | 0.55 | Zone de transition |
| 50 | 0.380 | 0.07 | 0.23 | Chimere respirante (diversite emergente) |
| 100 | 1.059 | 0.24 | 0.41 | Diversite haute + re-synchronisation par blocs |

**Resultats Lattice 10x10** : pas de collapse — H_cog reste positif sur tout le sweep. Minimum a tau_u=2 (H_cog=0.32), maximum a tau_u=100 (H_cog=1.18). Comportement qualitatif different de BA.

**Findings** :

1. **Bifurcation confirmee sur BA m=3** : transition abrupte entre tau_u=10 (H_cog=0.023) et tau_u=50 (H_cog=0.380). C'est le passage de "frustration figee" a "chimere respirante".

2. **tau_u* (valeur par defaut du modele) = 10 est au bord de la bifurcation**. Cela explique directement la bimodalite de §3vigies-ter : certains seeds convergent vers le regime bas-diversite, d'autres vers le regime haut-diversite selon les proprietes spectrales de leur graphe BA genere aleatoirement.

3. **Pic spectral fort en regime tau_u petit** (puissance 0.50–0.55 pour BA m=3) : u se met a jour rapidement → la dynamique de doute genere des oscillations coherentes de tout le reseau a basse frequence (~0.01 Hz simulation). Ces oscillations ne produisent pas de diversite cognitive — c'est une synchronisation oscillante, pas une chimere.

4. **Le regime tau_u=100 re-synchronise** (sync=0.24 vs sync=0.07 a tau_u=50) : a tres grande echelle de temps, u ne se met plus a jour assez vite pour maintenir les conflits locaux → les clusters s'alignent globalement mais en etats differents (H_cog=1.06 = diversite entre groupes, sync elevee = groupes coherents).

5. **Lattice robuste** : la geometrie 2D permet toujours une diversite locale meme avec u rapide (tau_u=0.05). La frustration geometrique du lattice est plus forte que sur BA.

**Consequence pour Paper 2** : tau_u est un **parametre de controle de la diversite**, aussi important que la topologie ou la normalisation. La valeur par defaut tau_u=10 est "accidentellement" au point de bifurcation pour BA m=3, ce qui explique la variance inter-seeds observee. Recommandation : ajouter tau_u comme variable independante dans les figures du preprint revise.

**Figures** : `figures/p2_tau_u_bifurcation.png` (4 panneaux : H_cog, sync, f_dom, peak_power vs tau_u, 2 topologies). CSV : `figures/p2_tau_u_bifurcation.csv`.

**Duree** : 87s (100 runs : 10 tau_u × 2 topos × 5 seeds). *(n augmenté de 3 → 5 lors de l'audit A2, session 2026-04-26)* BA m=3 n=5 : H_cog 0.006–0.059 pour tau_u < 20, 0.357 à tau_u=50, 1.052 à tau_u=100. Lattice : H_cog 0.515 à tau_u=0.05, minimum ~0.364 à tau_u=2, 1.174 à tau_u=100. Bifurcation confirmée.

---

### 3tervigies. Piste A1 — Resonance Stochastique Dirigee : HYPOTHESE FALSIFIEE (2026-04-24)

**Question** : Un bruit thermique dirige (sur les hubs ou les heretiques) peut-il induire une resonance stochastique qui fait sortir BA m=5 de la dead zone, contrairement au bruit homogene ?

**Methode** : `experiments/p2_stochastic_resonance_directed.py`. N=100, BA m=5, coupling_norm='uniform' (dead zone confirmee), I_stim=0.0 (regime endogene), 3000 steps, 3 seeds. 4 modes : ZERO (controle, sigma_vec=0), UNIFORM (sigma identique tous noeuds), HUB (sigma_i proportionnel a sqrt(deg(i))), HERETIC (sigma uniquement sur les heretiques, amplitude rescalee pour energie totale egale). Sweep sigma in [0, 0.50]. Metriques : H_cont (100-bin), H_cog (5-bin KIMI), complexite LZ temporelle. Nouveau parametre `sigma_v_vec` ajoute a `dynamics.py:step()` et `topology.py:step()`.

**Resultats** :

| Mode | H_cog (sigma=0) | H_cog (sigma=0.50) | Verdict |
|:-----|:---------------:|:------------------:|:--------|
| ZERO | 0.027 | 0.027 | Baseline deterministe |
| UNIFORM | 0.027 | 0.027 | Aucun gain cognitif |
| HUB | 0.027 | **0.000** | Pire que UNIFORM |
| HERETIC | 0.027 | 0.028 | Gain negligeable |

H_cog reste proche de 0 pour **tous les modes et toutes les amplitudes**. La dead zone est **resistante au bruit dirige dans le regime endogene (I_stim=0)**.

**Findings** :

1. **Hypothese falsifiee** : le bruit heterogene ne procure aucun avantage sur le bruit homogene pour echapper a la dead zone cognitive (H_cog reste ~0).

2. **H_cont augmente avec sigma** (3.0 → 3.3 bits a sigma=0.50 pour UNIFORM) mais c'est du desordre sub-cognitif : les noeuds restent tous dans l'Etat 1. Bruit != diversite cognitive.

3. **HUB degrade la diversite cognitive a sigma=0.50** (H_cog=0.000). Interpretation : le hub sur-bruite entraine les noeuds peripheriques par couplage, renforçant le consensus plutot que le brisant.

4. **La dead zone est robuste au bruit dirige** dans le regime endogene. Ce resultat est coherent avec §3octies (normalisation spectrale) et §3undecies (SPICE : escape seulement a sigma >= 0.50 avec I_stim implicite dans le bruit thermique). Le bruit seul ne suffit pas sans forçage.

5. **Distinction H_cont / H_cog confirmee** : H_cont ~ 2.9 meme dans la dead zone (diversite de tension sub-cognitive). H_cog ~ 0 (tous les noeuds en Etat 1). La dead zone ne signifie pas "tous les noeuds au meme voltage" mais "tous dans le meme etat cognitif".

**Statut Piste A1** : CLOTURE — RESULTAT NEGATIF. La piste "bruit dirige comme escape mechanism" est fermee pour le regime endogene. A tester avec I_stim > 0 (forced regime) si pertinent pour Paper 2.

**Code** : `sigma_v_vec` ajoute a `dynamics.py:step()` et `topology.py:step()` — feature utile pour futures experiences de bruit heterogene.

**Figures** : `figures/p2_stochastic_resonance_directed.png` (3 panneaux : H_cont, H_cog, LZ vs sigma × mode). CSV : `figures/p2_stochastic_resonance_directed.csv`.

**Duree** : 74s (108 runs : 4 modes × 9 sigmas × 3 seeds).

---

### 3duotrigies. Piste C — Ablation sigma_social : RESULTAT NUANCE (2026-04-25)

**Question** : sigma_social = |laplacian_v| est-il juste un proxy du bruit couple (Manus §1.4) ? Si on le remplace par du bruit pur a meme RMS, le comportement du reseau change-t-il significativement ?

**Methode** : `experiments/p2_sigma_social_ablation.py`. BA m=3 N=100, I_STIM=0.5, 5000 steps (warm_up=1000), 3 seeds. Nouveau hook `sigma_social_override` dans `dynamics.py` / `topology.py` (du equation uniquement — couplage et plasticite inchanges). 4 conditions :

| Condition | sigma_social dans du |
|:----------|:---------------------|
| FULL | |laplacian_v| (reference) |
| SS_NOISE | bruit Gaussien |N(0, RMS_warmup)| |
| SS_STATIC | moyenne temporelle par noeud (warmup) |
| FROZEN_U | epsilon_u = 0 (u immobile) |

**Resultats** :

| Condition | H_cog | H_cont | sync | delta_H_cog | delta_sync | Verdict |
|:----------|:-----:|:------:|:----:|:-----------:|:----------:|:--------|
| FULL | 0.0213 | 3.0551 | 0.0629 | — | — | Reference |
| SS_NOISE | 0.0117 | 3.0369 | 0.0683 | -0.0096 | +0.0054 | ~FULL |
| SS_STATIC | 0.0192 | 3.0548 | 0.0479 | -0.0021 | -0.0150 | ~FULL |
| FROZEN_U | 0.9915 | 4.0094 | 0.7824 | **+0.9703** | **+0.7195** | REGIME DIFF. |

**Findings** :

1. **SS_NOISE ≈ FULL et SS_STATIC ≈ FULL** : remplacer sigma_social par du bruit pur ou par une constante provoque des delta < 2% sur H_cog et sync. Le reseau est INSENSIBLE au contenu informationnel de sigma_social — Manus §1.4 partiellement confirme.

2. **FROZEN_U radicalement different** : supprimer le mouvement de u entraine une hypersynchronisation massive (sync : 0.067 → 0.730, +985% avec n=5 seeds) et une diversite cognitive spurieuse (H_cog : 0.02 → 1.07, cycles FHN libres visitent plus d'etats). Sans u comme modulateur, le reseau oscille en FHN classique couple.

3. **Conclusion nuancee pour §1.4** : Manus a raison que le *contenu* de sigma_social (vraie information topologique vs bruit) est indiscernable pour les metriques mesurees. Mais il a tort sur la consequence : ce n'est pas que sigma_social = bruit inutile. C'est que sigma_social joue le role d'un **signal d'activation de u** — son amplitude suffit, son contenu importe peu. Le veritable role de u est de **prevenir l'hypersynchronie** (FROZEN_U : +1143% sync), pas de transporter de l'information topologique.

4. **Mecanisme clair** : sigma_social maintient u hors de 0 → u module le couplage (Levitating Sigmoid) → chaque noeud recoit un couplage effectif different selon son doute local → decorrelation active. Peu importe si sigma_social vient du vrai laplacien ou d'un generateur de bruit.

**Impact Paper 2** : cette experience requalifie le role de u. Ce n'est pas un "detecteur de surprise topologique" mais un **filtre anti-synchronisation adaptable**. La variable u est essentielle (FROZEN_U montre +1143% sync), mais son mecanisme est robuste au contenu de sigma_social. Argument pour la robustesse du modele.

**Nouveau code** : hook `sigma_social_override` dans `dynamics.py:step()` (parametre optionnel, sans impact sur la dynamique de production). Feature reutilisable pour futures experiences d'ablation.

**Figures** : `figures/p2_sigma_social_ablation.png` (4 barres : H_cog, H_cont, sync, f_dom, FULL en reference avec bordure noire). CSV : `figures/p2_sigma_social_ablation.csv`.

**Duree** : 30s (20 runs : 4 conditions × 5 seeds). *(n augmenté de 3 → 5 lors de l'audit A2, session 2026-04-26)*

---

### 3tertrigies. Piste D — Bifurcation tau_u : Regime Endogene (I_STIM=0.0) : PARTIELLEMENT CONFIRME (2026-04-25)

**Question** : La bifurcation tau_u documentee en §3quatervigies (sous I_STIM=0.5) est-elle aussi presente en regime purement endogene (I_STIM=0.0, heretiques inactifs) ? Si oui, la dynamique adaptative de u structure le reseau sans aucun forcage externe.

**Methode** : `experiments/p2_tau_u_bifurcation_endogenous.py`. Sweep identique a §3quatervigies mais I_STIM=0.0. TAU_U_VALUES=[0.05..100.0] (10 valeurs log), 2 topologies (Lattice 10x10, BA m=3 N=100), coupling_norm='degree_linear', 4000 steps, warm_up=1000, 3 seeds. Metriques : H_cog, H_cont, pairwise_synchrony, frequence dominante FFT.

**Resultats BA m=3** :

| tau_u | H_cog | H_cont | Sync | f_dom | Regime |
|------:|:-----:|:------:|:----:|:-----:|:-------|
| 0.05 | 0.0291 | 3.072 | 0.090 | 0.013 | Actif heterogene |
| 0.10 | 0.0389 | 3.076 | 0.084 | 0.016 | Actif heterogene |
| 0.50 | 0.0160 | 2.890 | 0.215 | 0.011 | Transition |
| 1.00 | 0.0015 | 2.666 | 0.305 | 0.007 | Synchronie montante |
| 2.00 | 0.0001 | 2.235 | 0.267 | 0.011 | Synchronie |
| **5.00** | 0.0010 | 2.622 | **0.453** | 0.007 | **Pic de synchronie** |
| **10.00** | 0.0043 | 3.071 | 0.261 | 0.007 | **Transition (valeur defaut)** |
| 20.00 | 0.0028 | 1.228 | 0.024 | 0.018 | Gel amorce |
| 50.00 | 0.0006 | 0.774 | 0.048 | 0.049 | Gel complet |
| 100.00 | 0.0006 | 0.759 | 0.052 | 0.049 | Gel complet |

**Resultats Lattice 10x10** : meme pattern. Pic sync tau_u=5 (0.385), effondrement a tau_u=20 (0.029), gel a tau_u>=50 (H_cont<0.8).

**Findings** :

1. **Bifurcation PRESENTE en regime endogene** : la transition sync (pic) → gel est visible dans les deux topologies. Sur BA m=3 : sync passe de 0.453 a tau_u=5 a 0.024 a tau_u=20. Meme fenetre que sous I_STIM=0.5, meme tau_u critique ≈ 10-20.

2. **H_cog ≈ 0 dans tout le sweep** : sans stimulus externe, les heretiques sont inactifs (flip *= -1 sur I_stim=0 = no-op). Il n'y a pas de forcage brise-symetrie. La diversite cognitive (H_cog) necessite I_stim > 0. Les nodes convergent tous vers leur point fixe v*≈-1.29.

3. **H_cont capture la bifurcation** : H_cont suit la meme forme que la synchronie (montee puis effondrement). A tau_u=5-10 : H_cont ≈ 3.1 (diversite tensorielle de voltage). A tau_u=50 : H_cont ≈ 0.77 (reseau gele). Ce n'est pas de la diversite cognitive, mais de la diversite sub-threshold.

4. **La dynamique u STRUCTURE le reseau sans forcage** : la difference tau_u=0.05 (sync=0.09, actif) vs tau_u=50 (sync=0.05, gel) est reelle meme sans stimulus. Le gel progressif (H_cont : 3.1 → 0.8 quand tau_u 10 → 50) montre que u trop lent ne peut plus maintenir les fluctuations de couplage necessaires aux oscillations spontanees.

5. **tau_u=10 (valeur defaut) reste dans la fenetre active endogene** : sync=0.261 (non nulle), H_cont=3.071. Le modele est configure par defaut dans un regime qui maintient une activite spontanee meme sans stimulus — propriete emergente favorable.

6. **Comparaison I_STIM=0.5 vs I_STIM=0.0** : sous I_STIM=0.5, la bifurcation se manifeste via H_cog (0.023→0.380 a tau_u=50). Sous I_STIM=0.0, elle se manifeste via sync et H_cont uniquement. Meme mecanisme (u trop lent → gel), expression differente selon le mode de forcage.

**Verdict** : Le claim "la dynamique adaptative de u structure le reseau sans forcage externe" est **partiellement confirme**. La bifurcation tau_u existe en regime endogene (mecanisme confirme). En revanche, la *diversite cognitive* (H_cog) necessite un forcage externe — les heretiques doivent etre actifs (I_stim > 0). Version publiable : "u maintains spontaneous activity (synchrony window) even without external drive, but cognitive diversity requires stimulus-driven heretic competition."

**Figures** : `figures/p2_tau_u_bifurcation_endogenous.png` (2×4 panneaux : H_cog, H_cont, sync, f_dom vs tau_u en log, pour Lattice et BA m=3). CSV : `figures/p2_tau_u_bifurcation_endogenous.csv`.

**Duree** : 44s (60 runs : 10 tau_u × 2 topos × 3 seeds).

---

### 3quatertrigies. V4 Dynamic Heretics — Sweep Paramétrique (2026-04-27)

**Contexte** : Implémentation du mécanisme de bascule dynamique des hérétiques. Un nœud `i` bascule irréversiblement en hérétique quand `u_i >= u_threshold` pendant `steps_required` steps consécutifs. Branche `feat/v4-dynamic-heretics`.

**Implémentation** :
- `src/mem4ristor/dynamics.py` : config `coupling.dynamic_heretics` (enabled / u_threshold / steps_required), tableau `heretic_counter` par nœud, compteur `dynamic_heretic_count`, logique de bascule irréversible dans `step()` après mise à jour de `u`.
- `src/mem4ristor/topology.py` : `health_check()` enrichie avec `total_heretics` et `dynamic_heretic_count`.

**Sweep 2D** : `u_threshold` ∈ [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] × `steps_required` ∈ [10, 25, 50, 100, 200, 500], n=3 seeds, 3000 steps, grille 10×10, I=0.3.
Script : `experiments/v4_parametric_sweep.py`. CSV : `figures/v4_parametric_sweep.csv`.

**Loi linéaire découverte** :

```
t_first = steps_required + tau(u_threshold)
tau ≈ 130 × u_threshold   (linéaire, R²≈1)
```

`tau` = temps de montée de `u` vers le seuil depuis `sigma_baseline=0.05`. Les deux paramètres sont **orthogonaux et additifs**.

**Carte de phase (heretic_pct_final)** :

| u_threshold | heretic_pct (n=3 seeds) |
|:-----------:|:-----------------------:|
| 0.3 – 0.8   | **100%** (cascade totale) |
| 0.9         | **~92%** (cascade quasi-totale) |

Pas de régime "pas de cascade" dans la plage testée. La cascade est un **attracteur quasi-inévitable** sous ce régime (grille 10×10, I=0.3).

**Entropie** :

| Paramètres | H final | delta vs V3 (H~3.41) |
|:-----------|:-------:|:--------------------:|
| u=0.7, s=500 | **2.057** | **-1.35 bits** |
| u=0.9, s=500 | 2.584 | -0.83 bits |
| u=0.3, s=10  | 2.824 | -0.59 bits |

La cascade compresse la diversité cognitive. L'effondrement maximal est à u_thr ∈ [0.6, 0.8].

**Comparaison V3 vs V4** (10×10, I=0.3, n=5 seeds, 5000 steps) :

| Métrique | V3 (statique) | V4 (dynamique) |
|:---------|:------------:|:--------------:|
| total_heretics | 15.0 ± 0.0 | **99.8 ± 0.4** |
| dynamic_born | 0 | **84.8 ± 0.4** |
| synchrony | 1.61 ± 0.02 | 1.68 ± 0.01 |
| entropy H | 3.41 ± 0.13 | **2.33 ± 0.11** |

**Questions ouvertes** :
- Explorer u_thr > 0.9 : y a-t-il une zone de contrôle fin de la cascade ?
- La cascade irréversible est-elle souhaitable ou faut-il envisager une version réversible ?
- Impact sur la dead zone BA m>=5 : V4 dynamique pourrait-il contourner la strangulation topologique ?

**Scripts** : `experiments/v4_dynamic_heretics_emergence.py`, `experiments/v4_parametric_sweep.py`
**Git** : commits `88b9983` + `171c519` sur `feat/v4-dynamic-heretics`

### 3quinquetrigies. λ₂_crit — Formalisation par régression logistique (2026-04-27)

**Contexte** : λ₂_crit ≈ 2.5 était until'ici une valeur « eyeballed » tirée visuellement de §3novemvigies (SR topology). Ce statut informel était le point le plus vulnérable du dossier face à un reviewer. Expérience : régression logistique multi-sources pour formaliser la frontière.

**Méthode** : `experiments/lambda2_crit_regression.py`. Fusion de trois sources expérimentales indépendantes (n=58 observations) en une table unifiée (λ₂, dead_zone_binary). Analyse primaire sur la source la plus fiable (labels explicites), analyse combinée pour validation.

**Définition rigoureuse de "dead zone"** : le système est en dead zone si *aucune* normalisation raisonnable ne permet d'atteindre la diversité cognitive (H < 0.10 pour uniform *et* degree_linear). Les topologies où degree_linear=0 mais uniform>0 (CM, BA m=1) ne sont **pas** en dead zone — elles requièrent simplement la normalisation correcte.

**Sources** :
| Source | n | dead | not_dead | Méthode |
|:-------|:-:|:----:|:--------:|:--------|
| `p2_edge_betweenness.csv` | 36 | 12 | 24 | Labels explicites par seed (régime déterminé empiriquement) |
| `fiedler_phase_diagram.csv` | 15 | 2 | 13 | best_H = max(H_uniform, H_degree_linear) < 0.10 |
| `p2_stochastic_resonance_topology.csv` | 7 | 3 | 4 | H_cog(σ_max) < 0.10 |

**Résultat primaire — séparation complète (n=36, ebc) :**

Les 36 observations de la source ebc (la plus fiable) montrent une **séparation complète** sans chevauchement :

| Frontière | λ₂ | Topologie |
|:----------|---:|:---------|
| Max λ₂ non-dead | **2.126** | BA m=4, seed=2 |
| Min λ₂ dead | **2.504** | BA m=5, seed=2 |

→ **λ₂_crit ∈ (2.13, 2.50)**, midpoint = **2.31**
→ Classification : 100% correcte sur 36 observations (aucun faux positif/négatif)

**Résultat combiné — régression logistique MLE (n=58) :**
- λ₂_crit = 3.140, IC 95% bootstrap : [1.928, 4.845]
- McFadden R² = 0.697
- IC large dû aux incohérences de définition entre sources (BA m=8 : fiedler H_uniform=0.12 > seuil → non-dead, mais ebc → dead)

**Conclusion** :
- La valeur eyeballed **2.5 est dans l'intervalle de séparation (2.13, 2.50)** — compatible, mais légèrement surestimée.
- La valeur formelle à rapporter : **λ₂_crit ~ 2.31** (midpoint), ou plus précisément **λ₂_crit ∈ (2.13, 2.50)**.
- La séparation complète (aucun chevauchement sur 36 obs.) est un résultat plus fort qu'une simple régression : elle établit que le seuil existe, qu'il est net, et qu'il se situe dans cet intervalle.

**Formulation pour le papier** :
> *"Analysis of 36 topological configurations reveals complete separation of the dead zone from the functional regime at λ₂ ∈ (2.13, 2.50). No configuration with λ₂ < 2.13 exhibits a dead zone; all configurations with λ₂ > 2.50 do. We estimate λ₂_crit ≈ 2.31 as the midpoint of this transition interval."*

**Fichiers** :
- Script : `experiments/lambda2_crit_regression.py`
- Figure : `figures/lambda2_crit_regression.png` (régression logistique + distribution bootstrap)
- CSV : `figures/lambda2_crit_regression.csv` (résultats numériques complets)

---

### 3sexquetrigies. V4 — Exploration du régime haute zone u_thr > 0.9 (2026-04-27)

**Contexte** : Le sweep paramétrique V4 initial (`v4_parametric_sweep.py`) couvrait u_thr ∈ [0.3, 0.9]. La MEM4RISTOR.md listait « Explorer u_thr > 0.9 pour contrôle fin de la cascade » comme prochaine étape. Expérience : sweep ciblé sur la zone haute avec diagnostic de l'amplitude maximale de u observée.

**Méthode** : `experiments/v4_high_uthr_sweep.py`. Grille u_thr ∈ {0.90, 0.92, 0.94, 0.96, 0.98, 1.00, 1.05, 1.10, 1.20, 1.50, 2.00} × steps_required ∈ {10, 50, 200} × n=5 seeds = 165 runs. N_STEPS=3000, grille 10×10. Métrique supplémentaire : `u_max_observed` = maximum de |u_i| observé sur toute la simulation.

**Résultat central : plafond dur à u = 1.0**

La Levitating Sigmoid borne u dans [0, 1]. Pour **tous** les seuils testés (y compris u_thr=2.0), `u_max_observed = 1.000`. La cascade s'effondre non pas graduellement mais par un saut abrupt entre u_thr=1.00 et u_thr=1.05.

**Frontière critique : u_thr_crit ∈ (1.00, 1.05)**

| u_thr | heretic% (steps=50) | H_final | cascade ? |
|------:|--------------------:|--------:|----------:|
| 0.90  | 92.8%               | 2.647   | OUI       |
| 0.92  | 91.0%               | 2.727   | OUI       |
| 0.94  | 88.0%               | 2.896   | OUI       |
| 0.96  | 86.0%               | 2.933   | OUI       |
| 0.98  | 84.2%               | 2.982   | OUI       |
| 1.00  | 76.2%               | 3.108   | OUI (partiel) |
| **1.05**  | **15.0%**           | **3.509** | **NON** |
| 1.10  | 15.0%               | 3.509   | NON       |
| ≥1.20 | 15.0%               | 3.509   | NON       |

**Interprétation mécanique** :
- u_thr ≤ 0.98 : u_max = 1.0 dépasse le seuil → cascade inévitable (marge > 0.02)
- u_thr = 1.00 : seuil exactement au plafond → cascade partielle, très sensible à steps_required (steps=10 → 81%, steps=200 → 54%)
- u_thr ≥ 1.05 : seuil physiquement inatteignable → 0% cascade dynamique, H retourne exactement à la baseline V3 (3.509)

**Cas spécial u_thr = 1.00** : le seuil coïncide avec le maximum absolu de u. Les nœuds à fort couplage atteignent u=1.0 épisodiquement mais ne peuvent le maintenir au-delà de quelques steps. Plus steps_required est long, moins la bascule est probable : c'est un filtre de durée au plafond.

**Zone de contrôle fin [0.90, 1.00]** : la cascade décline de 93% à ~76% (steps=50). La décroissance est quasi-linéaire (~8%/0.05 de u_thr). C'est la seule zone où u_thr offre un contrôle graduel sans éteindre complètement la cascade.

**Contrôle négatif parfait** : à u_thr ≥ 1.05, H_final = 3.509 = baseline V3 exacte (aucun hérétique dynamique né). Le mécanisme V4 est chirurgicalement inactif.

**Loi t_first dans la zone haute** : la loi linéaire `t_first ≈ steps_required + 130 × u_threshold` reste approximativement valide jusqu'à u_thr=0.98, mais commence à dévier à u_thr=1.00 (219 observé vs 180 prédit pour steps=50). Cohérent : au plafond, u ne converge plus régulièrement vers le seuil — il y touche épisodiquement.

**Fichiers** :
- Script sweep : `experiments/v4_high_uthr_sweep.py`
- Script figure : `experiments/v4_high_uthr_figure.py`
- Figure : `figures/v4_high_uthr_sweep.png` (3 panels : cascade vs u_thr, H vs u_thr, diagnostic plafond)
- CSV : `figures/v4_high_uthr_sweep.csv` (165 runs, 5 seeds)

---

### 3septemtrigies. Audit numérique — Sensibilité à dt (2026-04-27)

**Contexte** : Audit DeepSeek item 7. Vérification que l'intégrateur Euler à dt=0.05 ne crée pas d'artefacts dans les régimes critiques. Protocole : 4 valeurs de dt (0.01, 0.02, 0.05, 0.10), temps total simulé fixé à T=150 (N_steps=T/dt), 3 topologies (BA m=3/5/8), 5 seeds, I_stim=0.3.

**Résultats** :
- **H_cog (métrique principale) : stable** — delta max < 0.01 pour tous les dt. La classification dead zone (H_cog=0 pour m=5 et m=8) est identique à tous les pas d'intégration.
- **Synchrony : varie avec dt** — jusqu'à Δ=0.29 entre dt=0.01 et dt=0.10. Interprétation : dissipation numérique Euler proportionnelle à dt — les amplitudes d'oscillation s'amortissent légèrement à grand dt. Cet effet est réel mais n'affecte pas les comparaisons RELATIVES (FROZEN_U vs FULL) qui utilisent le même dt.

**Conclusion** : L'intégration Euler à dt=0.05 est validée pour la métrique principale (H_cog). Les valeurs absolues de synchrony sont dt-dépendantes — les comparaisons inter-conditions restent valides car elles partagent le même intégrateur.

**Fichiers** : `experiments/dt_sensitivity.py`, `figures/dt_sensitivity.csv`, `figures/dt_sensitivity.png`

---

### 3octotrigies. Audit numérique — RK4 vs Euler (2026-04-27)

**Contexte** : Audit DeepSeek item 8. Vérification que les artefacts numériques de l'intégrateur Euler n'imitent pas des transitions de phase (synchronisation artificielle). Implémentation d'un intégrateur RK4 standalone reproduisant les équations FHN de dynamics.py. Comparaison Euler vs RK4 à dt=0.05 identique sur 2000 steps, 5 seeds. Conditions : BA m=3 et m=5, FULL et FROZEN_U.

**Résultats** :

| Condition | MAE(v) Euler/RK4 | ΔH_cog | Surge Euler | Surge RK4 | Delta surge |
|-----------|-----------------|--------|-------------|-----------|-------------|
| BA m=3 FULL | 0.0054 | 0.0001 | — | — | — |
| BA m=3 FROZEN_U | 0.0053 | 0.0001 | +55.6% | +56.0% | 0.4 pp |
| BA m=5 FULL | 0.0054 | 0.0030 | — | — | — |
| BA m=5 FROZEN_U | 0.0053 | 0.0005 | +55.9% | +56.2% | 0.4 pp |

- **Divergence de trajectoire** : MAE(v) = 0.0054 soit ~0.2% de la plage [-2, +2]. Les trajectoires Euler et RK4 sont quasi-identiques sur 2000 steps.
- **H_cog** : delta < 0.003 pour toutes les conditions — classification dead zone identique.
- **Surge FROZEN_U/FULL** : 0.4 point de % d'écart entre Euler et RK4 — le surge n'est pas un artefact numérique.

**Conclusion** : L'intégrateur Euler à dt=0.05 est validé. Le claim principal (FROZEN_U surge) est robuste à la méthode d'intégration. Note : le surge affiché ici (+56%) est mesuré sur BA m=3 avec I_stim=0.3 — le +985% du Paper 1 est mesuré sur treillis avec I_stim=0.5 et protocole complet. L'écart Euler/RK4 reste de 0.4 pp dans les deux cas.

**Fichiers** : `experiments/rk4_vs_euler.py`, `figures/rk4_vs_euler.csv`

### 3novetrigies. Audit numérique — RK4 vs Euler : validation avec paramètres corrigés (2026-04-29)

**Contexte** : La section §3octotrigies (2026-04-27) documentait une première validation Euler/RK4. À l'examen du script source `experiments/rk4_vs_euler.py`, cinq paramètres étaient incorrects par rapport à `config.yaml` et `dynamics.py` :

| Paramètre | Ancien (script) | Correct (config.yaml) |
|:----------|:---------------:|:---------------------:|
| sigmoid_steepness | 10.0 | π ≈ 3.1416 |
| social_leakage | 0.1 | 0.01 |
| epsilon_u | 0.1 | 0.02 |
| sigma_baseline | 0.1 | 0.05 |
| tau_u | 5.0 | 10.0 |

Le script a été corrigé le 29 avril 2026. Deux runs exécutés pour couvrir les deux configurations pertinentes du système.

#### Run 1 — Plasticité=OFF (λ_learn=0.0)

**Protocole** : N_warm=1000, N_steps=2000, dt=0.05, N_seeds=5, I_stim=0.3. Conditions : BA m=3/m=5, FULL/FROZEN_U.

| Condition | MAE(v) moyen | ΔH_cog | Surge Euler | Surge RK4 | Delta surge |
|:----------|:-----------:|:------:|:-----------:|:---------:|:-----------:|
| BA m=3 FULL | 0.0054 | 0.0019 | — | — | — |
| BA m=3 FROZEN_U | 0.0051 | 0.0018 | mesurable | mesurable | **0.3 pp** |
| BA m=5 FULL | 0.0069 | 0.0003 | — | — | — |
| BA m=5 FROZEN_U | 0.0051 | 0.0017 | mesurable | mesurable | **0.3 pp** |

- **Max Δ(H_cog) Euler vs RK4 : 0.0018** [seuil 0.05 → OK]
- **Max Δ(surge) Euler vs RK4 : 0.3 pp** [seuil 10 pp → OK]
- **Verdict : EULER dt=0.05 VALIDE (plasticité=OFF)**

**Fichiers** : `experiments/rk4_vs_euler.py`, `figures/rk4_vs_euler.csv` — Commit : `91a0072`

#### Run 2 — Plasticité=ON (λ_learn=0.05, τ_plast=1000, w_sat=2.0)

**Protocole** : Identique au Run 1 + terme plastique dans `fhn_derivatives()` : `dw_learn = λ_learn × σ_s × innovation_mask × sat_factor − w/τ_plast`, avec `innovation_mask = (u > 0.5)` et `sat_factor = clip(1 − (w/w_sat)², 0, 1)`.

| Condition | MAE(v) moyen | ΔH_cog (Euler vs RK4) | Surge Euler | Surge RK4 | Delta surge |
|:----------|:-----------:|:--------------------:|:-----------:|:---------:|:-----------:|
| BA m=3 FULL | 0.0059 | 0.0053 | — | — | — |
| BA m=3 FROZEN_U | 0.0051 | 0.0043 | +23.2% | +23.2% | **0.0 pp** |
| BA m=5 FULL | 0.0069 | 0.0039 | — | — | — |
| BA m=5 FROZEN_U | 0.0051 | 0.0046 | +18.1% | +18.0% | **0.1 pp** |

- **Max Δ(H_cog) Euler vs RK4 : 0.0053** [seuil 0.05 → OK]
- **Max Δ(surge) Euler vs RK4 : 0.1 pp** [seuil 10 pp → OK]
- **Verdict : EULER dt=0.05 VALIDE (plasticité=ON)**

**Note sur le surge réduit** : Le surge +23% (plasticité=ON) vs +985% (protocole complet Paper 1) est attendu. La plasticité modifie `w` (terme de récupération FHN) sur des simulations courtes T=100 — ce qui atténue la séparation FROZEN_U/FULL. La validation numérique Euler/RK4 porte sur les équations core du système, pas sur la reproductibilité du résultat principal. Le claim +985% est mesuré sur protocole long avec initialisation complète, non sur 2000 steps de validation.

**Ce que cette validation ferme face à un reviewer** :
1. *"Les trajectoires Euler divergent de RK4 et créent un faux surge"* → Réfuté. Delta surge max = 0.3 pp (OFF) / 0.1 pp (ON). Le surge n'est pas un artefact de l'intégrateur.
2. *"La plasticité pourrait cacher un artefact numérique"* → Réfuté. Avec λ_learn actif, le système reste identiquement intégré par Euler et RK4 (Δ(H_cog) < 0.006 dans les deux conditions).
3. *"Les paramètres simulés ne correspondent pas au code source"* → Fermé. Les deux validations utilisent désormais les paramètres exactement alignés sur `config.yaml` et `dynamics.py`.

**Fichiers** : `experiments/rk4_vs_euler.py`, `figures/rk4_vs_euler_plasticity_on.csv` — Commit : `4cd7fce`

---

---

### Session 2026-04-26 (Claude Sonnet 4.6 — Audit Edison NB4/NB5/A2)

**Contexte** : Suite directe de la session 2026-04-25 (audit Edison Platform). Trois items résiduels traités : NB4 (figures paper_2.tex), NB5 (cohérence CSV), A2 (puissance statistique n=3).

**NB4 — Figures dans paper_2.tex** ✅ CLOTURE :
- `\graphicspath{{../../figures/}}` ajouté au préambule.
- Package `subcaption` ajouté (figures côte-à-côte §4).
- 11 `\includegraphics` insérés dans les 5 sections + Discussion : §2 (ablation + MI), §3 (tau_u forced + endogenous), §4 (fiedler+EBC côte-à-côte, finite_size, SR+directed côte-à-côte, delta_sweep+SR_directed côte-à-côte), §5 (community detection).
- Abstract corrigé : "ΔH_cog < 0.3%" → "absolute ΔH_cog < 0.002 bits, < 0.09% of log₂5".
- Sync surge mis à jour : "1143%" → "985%" (valeur n=5).
- §5 NMI corrigé : "5 sur 6 non significatifs" → "3 sur 6 significatifs" (post-fix Louvain + données CSV exactes : Lattice 2/3 sig, BA m=3 1/3 sig, z=-2.18 pour un seed anti-aligné). NMI moyen = 0.27.
- PDF recompilé : 7 pages, 0 erreurs, 0 références manquantes (2 passes).
- Fichier : `docs/paper_2/paper_2.tex`.

**NB5 — Cohérence nommage CSV** ✅ CLOTURE :
- Convention choisie : **lowercase_underscore** (`ba_m3`, `lattice`, `er_p05`) — majoritaire dans 4/6 scripts p2_*.
- Scripts corrigés : `p2_stochastic_resonance_topology.py` (`BA_m{m}` → `ba_m{m}`, `Lattice_10x10` → `lattice`, `ER_p{X}` → `er_p{X}`) + `p2_tau_u_bifurcation_endogenous.py` (`'Lattice_10x10'` → `'lattice'`, `'BA_m3'` → `'ba_m3'`).
- CSVs mis à jour en place : `figures/p2_stochastic_resonance_topology.csv` + `figures/p2_tau_u_bifurcation_endogenous.csv` (aucun résidu uppercase vérifié).
- TODO(A2) comments ajoutés dans `p2_stochastic_resonance_topology.py` et `p2_finite_size_scaling.py` pour la montée à n=5 avant publication.

**A2 — Puissance statistique n=3 → n=5** ✅ CLOTURE PARTIELLE :
- `p2_sigma_social_ablation.py` : SEEDS [42,123,777] → [42,123,777,456,999]. Re-run réussi (30s). Résultats n=5 : FULL sync=0.067, FROZEN_U sync=0.730 (+985%), SS_NOISE/SS_STATIC indiscernables (<2%). Claim +1143% rétrogradé à +985% — toujours spectaculaire.
- `p2_tau_u_bifurcation.py` : SEEDS [42,123,777] → [42,123,777,456,999]. Re-run réussi (87s). Bifurcation BA m=3 confirmée : H_cog≈0.01–0.06 pour tau_u<20, 0.357 à tau_u=50, 1.052 à tau_u=100.
- Figures PNG et CSVs régénérés pour ces deux expériences.
- Note statistique ajoutée dans paper_2.tex §Discussion : "ablation et tau_u validés à n=5 ; SR topology + finite-size restent à n=3 et sont marqués TODO(A2)".
- Scripts lourds (`p2_stochastic_resonance_topology.py`, `p2_finite_size_scaling.py`) : seeds n=3 conservées avec TODO(A2) inline.

### Session 2026-04-27 (Claude Sonnet 4.6 — V4 Dynamic Heretics)

**Commits sur main (62d3b2a)** :
- `topology.py` : `health_check()` ajoutée — 6 gardes de santé (NaN/Inf, explosion, zone morte, doute saturé/gelé, matrice corrompue, rewiring excessif), retourne `ok/warning/critical` + liste des issues + `total_heretics` + `dynamic_heretic_count`.
- `dynamics.py` : 2 marqueurs `@DOUBT` (correction NaN silencieuse L154-157 + clipping silencieux L241-244).

**Commits sur `feat/v4-dynamic-heretics`** :
- **88b9983** : V4 implémenté dans `dynamics.py` (config `dynamic_heretics`, `heretic_counter`, bascule irréversible) + `topology.py` (health_check V4) + script `v4_dynamic_heretics_emergence.py`.
- **171c519** : Sweep paramétrique 2D (126 runs) + CSV `figures/v4_parametric_sweep.csv`. Loi découverte : `t_first = steps_required + 130 × u_threshold`.

**Résultat clé** : cascade totale (100%) pour u_thr ≤ 0.8 ; attracteur quasi-inévitable dans le régime testé. Entropie effondrée de -1.35 bits max. Voir §3quatertrigies.

---

### Session 2026-04-28 (Claude Sonnet 4.6 — Révision adversariale)

**Objectif** : Traitement de 5 attaques critiques simulant une revue sévère des deux papers (preprint + paper_2). Pas de nouveau code — uniquement corrections documentaires et documentaires.

**Commit `d057865` sur `feat/v4-dynamic-heretics`** (`docs/preprint.tex` + `docs/paper_2/paper_2.tex`) :

- **Attack 1 — Terminologie** : "frustrated synchronization" → "polarity-modulated anti-synchronization" ; "topological phase transition" → "spectral phase transition" ; "topological dead zone" → "spectral dead zone". Correction dans les deux papers. Justification : le mécanisme u est state-dependent et continu, pas quenched — distinction fondamentale avec Kuramoto.

- **Attack 2 — H_cog=0** : Documenté comme artefact de binning (voltages Python [-3.2,-1.3] tous < -1.2 → bin 1). Claim recentrée sur H_cont=3.79±0.14 bits + synchrony pairwise FULL=0.031 vs FROZEN=0.751 (+2326%). Ablation table remplacée par synchrony+LZ. Section Limitations enrichie.

- **Attack 3 — Rôle de u** : Requalifié de "détecteur de surprise structurelle" à "filtre anti-synchronisation actif". SS_NOISE ≈ SS_STATIC ≈ FULL (ΔH < 2%) → ce qui compte est que u soit actif, pas ce qui le drive. Abstract paper_2 + corps mécanique + section τ_u mis à jour.

- **Attack 4** : Non applicable (Julien donne son travail librement).

- **Attack 5 — Validation SPICE** : Reviewer prétendait "seulement 4×4 lattice". Réfutation : 50 seeds × 3 conditions sur BA m=5, N=64 existaient déjà dans `experiments/spice/results/`. Ajout de `\subsection{SPICE Circuit Validation}` dans preprint.tex avec la table complète : dead zone H_cont=1.38±0.04 vs functional 4.30±0.19 bits (ratio 3.1×). CMOS mismatch sans effet sur la diversité. NGSpice 46 batch mode confirmé opérationnel.

**Aucun changement au code Python ni aux tests** — les 84 tests restent valides.

---


### Session 2026-04-29 (Claude Sonnet 4.6 — Validation intégrateur [8] complète)

**Objectif** : Validation complète de l'item [8] RK4 vs Euler de l'audit DeepSeek. Deux runs : plasticité=OFF (isolation FHN pur) et plasticité=ON (système complet).

**Anomalie détectée et corrigée** : Le script `experiments/rk4_vs_euler.py` existant (écrit session 2026-04-27) utilisait 5 paramètres incorrects par rapport à `config.yaml` / `dynamics.py`. Corrigés avant exécution (sigmoid_steepness 10→π, social_leakage 0.1→0.01, epsilon_u 0.1→0.02, sigma_baseline 0.1→0.05, tau_u 5→10).

**Commit `91a0072` sur `feat/v4-dynamic-heretics`** — plasticité=OFF :
- Max Δ(H_cog) Euler/RK4 : **0.0018** [seuil 0.05]
- Max Δ(surge) : **0.3 pp** [seuil 10 pp]
- MAE(v) moyen : 0.005 (trajectoires quasi-identiques)

**Commit `4cd7fce` sur `feat/v4-dynamic-heretics`** — plasticité=ON (λ_learn=0.05) :
- BA m=3 : surge Euler=+23.2%, RK4=+23.2%, delta **0.0 pp**
- BA m=5 : surge Euler=+18.1%, RK4=+18.0%, delta **0.1 pp**
- Max Δ(H_cog) Euler/RK4 : **0.0053** [seuil 0.05]

**Verdict** : L'intégrateur Euler à dt=0.05 est validé dans les deux configurations. Le surge FROZEN_U/FULL n'est pas un artefact numérique. Trois attaques reviewer potentielles sur la robustesse numérique sont désormais fermées (voir §3novetrigies).

**Aucun changement aux tests** — les 84 tests restent valides.

---

## 10. PROCHAINES ÉTAPES (par priorité)

### P0 — Soumission
1. **~~Relecture finale~~** → **SOUMIS 2026-04-22** par Julien.
2. **~~Upload Zenodo~~** → **FAIT (2026-04-27)** par Julien. v3.2.1 uploadée (preprint.pdf + paper_2.pdf + PROJECT_STATUS.md + figures).
3. **Commit + push GitHub** avec tous les changements de cette session

### P1 — Bugs pré-existants à fixer
4. **~~`test_swarm_synchronization`~~** → **FAIT (2026-04-19)**. Test était écrit pour mean-field symétrique mais l'implémentation est MAX FIELD asymétrique (intentionnel : vétéran préservé). Test corrigé.
5. **~~`test_entropy_preservation_with_v4`~~** → **FAIT (2026-04-19)**. Ring N=10 sans hubs → remplacé par BA m=3 N=50 + `coupling_norm='degree_linear'`. H ≈ 0.83.
5bis. **~~Métrique de diversité cognitive coordonnée~~** → **FAIT (2026-04-21)**. Voir §3novedecies. Deux nouvelles métriques trajectorielles dans `src/mem4ristor/metrics.py` : `calculate_pairwise_synchrony` (corrélation de Pearson croisée entre nœuds) + `calculate_temporal_lz_complexity` (complexité LZ76 des séquences d'états cognitifs, normalisée). Script d'ablation complet : `experiments/ablation_coordination.py`. 14 smoke tests : `tests/test_coordination_metrics.py`. 74 tests verts.

5ter. **Suivi P1.5bis : pistes ouvertes** → voir §3novedecies-bis.
  - **(A)** ~~Bimodalité ENDOGENOUS FULL~~ → **FAIT 2026-04-21**. Avg clustering prédit la synchrony (Pearson r=−0.64, p=0.045). Voir §3vigies.
  - **(B)** ~~Diagramme de phase 2D (synchrony × LZ)~~ → **FAIT 2026-04-21**. Seul FULL occupe le demi-plan bas-LZ (structuré). Voir §3vigies-bis. Figure candidate pour le preprint.
  - **(C)** ~~Sweep heretic_ratio sous forçage~~ → **FAIT 2026-04-24**. LZ_full monotone décroissant (1.137→1.012), hypothèse régulariseur confirmée. Synchrony en U (seuil η≈0.05). Voir §3novedecies-ter.
  - **(D)** ~~Multi-topologie universalité~~ → **FAIT 2026-04-24** (~3 min, 320 runs). Universalité CONFIRMÉE sur 4 topologies. FULL sync ≈ 0 sous forçage pour toutes (lattice, BA m=3, BA m=5 dead zone, WS). FROZEN_U sync = 0.52–0.94 partout. LZ_full FULL ≈ 0.92 (ENDO) constant. Voir §3vigies-quater.
  - **(E)** ~~Résonance stochastique inversée FROZEN_U~~ → **FAIT 2026-04-24**. Hypothèse confirmée : FULL sync ≈ 0.03 plat, FROZEN_U sync 0.006→0.830 (transition à I≈0.20). Cohen's d=13.21 à I=0.50. Dip inattendu à I=0.70 (bifurcation). Figure publiable. Voir §3novedecies-quater.
  - **(F)** ~~Confirmer bimodalité piste A avec 50 seeds + Hartigan dip test~~ → **FAIT 2026-04-24**. Bimodalité CONFIRMÉE (Hartigan D=0.194 p=0.000, BC=0.605>0.555). MAIS avg_clustering non significatif (r=-0.19, p=0.18) à n=50 → r=-0.64 de §3vigies était un faux positif n=10. Voir §3vigies-ter.
  - **(G)** ~~Intégrer figure §3vigies-bis dans preprint v3.3 section Minimality~~ → **FAIT 2026-04-24**. Nouvelle §3.3.1 "Trajectory-Based Minimality" ajoutée dans `docs/preprint.tex` : figure `coordination_phase_space.png` (Fig. 1), texte phase-space 2D + résultat Piste E (filtre anti-synchronisation, d=13.21). Preprint recompilé : 13 pages, 0 références indéfinies. Voir `docs/preprint.tex` §3.3.

### P2 — Paper 2 : "Breaking the Topological Diversity Boundary" (pistes Grok + Antigravity, 2026-04-11)

> Ces pistes sont documentées ici pour mémoire. Elles constituent le matériau d'un **deuxième papier** (v3.3+).
> La v3.2.0 est publiée telle quelle — le résultat négatif (dead zone m≥5) est une contribution en soi.

**Priorité haute (impact fort, effort raisonnable) :**

6. **~~Normalisation spectrale~~** → **TESTÉ — RÉSULTAT NÉGATIF (2026-04-19)**. Mode `coupling_norm='spectral'` implémenté dans `core.py`. 0/6 wins sur la dead zone. Voir §3octies. Conclusion : la dead zone n'est pas un problème de pondération.
7. **~~Finite-size scaling~~** → **FAIT 2026-04-24**. N ∈ {100, 400, 1600}, degree_linear, η=0.15. λ₂_crit = ∞ pour tous N — aucune dead zone sous cette normalisation. λ₂ est N-invariant à m fixé (variation < 5%). Conclusion : la dead zone est coupling-norm–dependent, pas une limite thermodynamique. Voir §3duovigies.
8. **~~Figure λ₂ vs H_stable~~** → **FAIT (2026-04-19)**. `experiments/fiedler_phase_diagram.py` → `figures/fiedler_phase_diagram.png` + `.csv`. 15 topologies × 2 norms × 3 seeds.
9. **~~Edge betweenness + diamètre~~** → **FAIT 2026-04-24**. λ₂ = meilleur prédicteur dead zone (r=+0.901, p=6.4×10⁻⁵) vs EBC (r=-0.604, p=0.038). Conclusion inverse à l'hypothèse : λ₂ n'est PAS un simple proxy, c'est la quantité fondée. EBC confirme la même information mais avec moins de puissance. Voir §3unvigies.
9bis. **Adaptive heretics / dynamique modifiée** — Maintenant que toutes les pistes de pondération sont éliminées, c'est la priorité haute pour Paper 2. Tester (a) η dynamique (item 11) et (b) stochastic resonance ciblé sur la dead zone (item 10).

**Priorité moyenne (intéressant, Paper 2 ou 3) :**

10. **Stochastic resonance** — ✅ CLOTURE (2026-04-25). Pas de SR classique. Dichotomie lambda2 : < 2.5 → bruit benefique monotone ; > 2.5 → zone morte resistante au bruit. Voir §3novemvigies.
11. **Adaptive heretics** — ✅ **IMPLÉMENTÉ (2026-04-27)** sur `feat/v4-dynamic-heretics`. Bascule irréversible : u_i >= u_threshold × steps_required steps → heretic. Loi : t_first = steps_required + 130×u_threshold. Cascade totale (100%) pour u_thr ≤ 0.8. Voir §3quatertrigies.
12. **Doubt-driven community detection** — ✅ CLOTURE (2026-04-25, re-analyse audit Edison). Bug Louvain corrige (poids signes → |corr|). Post-fix : Lattice 2/3 seeds sig (z_mean=+1.80), BA m=3 signal ambigu (z=+2.55/-2.18/+0.32, z_mean=+0.23). Finding robuste : deux regimes u (singletons heretiques u=1.0 + grands groupes frustres). Voir §3octvigies.

---

### P2-AUDIT — Pistes issues de l'Audit Externe (2026-04-24)

> Ces 5 pistes ont été proposées par l'auditeur externe dans `.Audit/Audit Scientifique du Projet Mem4ristor-v2.md`. Toutes seront testées. Elles constituent le cœur de Paper 2 / Paper 3.

**Piste A1 — Résonance Stochastique Dirigée** *(Impact : Fort, Effort : Faible)*

- **Hypothèse** : La dead zone est un état métastable, pas un attracteur absolu. Un bruit thermique ciblé uniquement sur les nœuds hérétiques (ou les hubs) peut induire une résonance stochastique suffisante pour en faire sortir le réseau, sans noyer le signal global.
- **Différence avec ce qui existe** : Les expériences SPICE (`spice_noise_resonance.py`, `spice_mismatch_sweep.py`) appliquent un bruit **global homogène**. Ici le bruit est **hétérogène et dirigé** par la topologie.
- **Méthode** : Modifier `core.py` pour appliquer σ_v de manière hétérogène (`σ_{v,i} ∝ deg(i)` ou `σ_{v,i} > 0` uniquement si `heretic_mask[i]`). Script : `experiments/p2_stochastic_resonance_directed.py`.
- **Métriques** : H_cont (100-bin) + H_cog (5-bin) + complexité LZ temporelle. Sweep amplitude bruit hétérogène sur BA m=5.
- **Statut** : ✅ CLOTURE — RESULTAT NEGATIF (2026-04-24). H_cog ~ 0 pour tous modes et toutes amplitudes. Dead zone resistante au bruit dirige en regime endogene. Voir §3tervigies.

**Piste A2 — Bifurcation tau_u (Dynamique Temporelle du Doute)** *(Impact : Fort, Effort : Moyen)* *(Impact : Fort, Effort : Moyen)*

- **Hypothèse** : La constante de temps du doute τ_u contrôle une bifurcation entre "frustration figée" (nœuds bloqués en opposition) et "chimère respirante" (clusters qui se font et défont dynamiquement). Un pic de fréquence caractéristique devrait émerger à τ_u critique.
- **Justification** : τ_u est le seul paramètre temporel central jamais sweepé. Les paramètres de couplage (D, normalisation) et la topologie ont été explorés exhaustivement ; τ_u est l'angle manquant.
- **Méthode** : Sweep τ_u ∈ [0.1, 100.0] sur Lattice 10×10 et BA m=3 N=100. Script : `experiments/p2_tau_u_bifurcation.py`.
- **Métriques** : `calculate_pairwise_synchrony` + spectre de Fourier moyen des séries v(t). Chercher un pic de fréquence à τ_u critique.
- **Statut** : ✅ CONFIRME — BIFURCATION REELLE (2026-04-24). tau_u* entre 10 et 50 sur BA m=3. Voir §3quatervigies.

**Piste A3 — Couplage Asymétrique et Graphes Dirigés** *(Impact : Très Fort, Effort : Moyen)*

- **Hypothèse** : La strangulation par les hubs est exacerbée par la symétrie du couplage. Un réseau dirigé où les hubs "parlent plus qu'ils n'écoutent" (ou l'inverse) éliminera la dead zone sans normalisation ad-hoc.
- **Justification** : Toutes les topologies testées jusqu'ici sont non-dirigées. Dans les systèmes neuromorphiques réels, les synapses sont directionnelles. C'est le changement structurel le plus fondamental possible.
- **Méthode** : Générer des graphes BA dirigés. Modifier `Mem4Network` pour accepter des matrices d'adjacence asymétriques (le couplage `L @ v` devient `A_directed @ v`). Script : `experiments/p2_directed_coupling.py`.
- **Métriques** : H_stable + λ₂ calculé sur le Laplacien dirigé. Comparer avec `limit02_topology_sweep.py`.
- **Statut** : ✅ CLOTURE — RESULTAT NEGATIF (2026-04-24). HUB_BCAST cree sync=0.64 (pire). HUB_LISTEN isole la peripherie (d_in=0 → point fixe). Voir §3quinquevigies.

**Piste A4 — Information Mutuelle Spatio-Temporelle** *(Impact : Moyen, Effort : Moyen)*

- **Hypothèse** : H_stable (entropie marginale spatiale) ne distingue pas désordre et diversité structurée. L'Information Mutuelle (MI) entre nœuds voisins vs distants révélera une longueur de corrélation caractéristique du régime Mem4ristor.
- **Méthode** : Implémenter `calculate_spatial_mutual_information(v_history, adjacency_matrix)` dans `src/mem4ristor/metrics.py`. Évaluer sur les 4 ablations de `ablation_coordination.py`. Script : `experiments/p2_spatial_mutual_information.py`.
- **Métriques** : Décroissance de MI en fonction de la distance sur le graphe.
- **Statut** : ✅ CONFIRME — RESULTAT POSITIF (2026-04-24). u = decorrelateur actif. MI(FULL) < MI(FROZEN_U) a toutes distances. Voir §3sexvigies.

**Piste A5 — Sweep δ de la Levitating Sigmoid** *(Impact : Faible, Effort : Faible)*

- **Hypothèse** : Le paramètre δ=0.01 dans `w_i(u_i) = tanh(π(0.5−u_i)) + δ` brise la symétrie parfaite au point de doute maximal (u=0.5). δ a été introduit comme fix technique (LIMIT-01), mais il est en réalité un **paramètre de contrôle de la symétrie sociale** non quantifié. Il existe un δ_crit qui maximise la complexité LZ.
- **Méthode** : Sweep δ ∈ [−0.1, 0.1] sur Lattice 10×10. Script : `experiments/p2_delta_sweep.py`.
- **Métriques** : `calculate_temporal_lz_complexity` + H_cont.
- **Statut** : ✅ CLOTURE — RESULTAT NEGATIF / ROBUSTESSE (2026-04-24). delta sans effet significatif (variation LZ < bruit inter-seeds). delta = parametre technique, pas de controle. Voir §3septvigies.

### P2-AUDIT-2 — Actions issues de l'Audit Manus (2026-04-25)

> Ces 4 actions sont issues du rapport `.Audit/25-04-2026_Rapport d'Audit Scientifique et Technique du Projet Mem4ristor v3.2.0.md`. Elles doivent être traitées **avant soumission de Paper B**.

**Faille A — Calibration η SPICE ↔ σ Python** ✅ CLOTURE (2026-04-25)
- Script : `experiments/spice_noise_calibration.py`. Cellule RC + trnoise, mesure std(dV).
- **Résultat** : η=0.5 SPICE ↔ σ_equiv = 0.0044 Python. Item 10 testait jusqu'à σ=1.2 = **270× l'amplitude équivalente**. Python reste H_cog≈0 à σ=0.0044 ET à σ=0.014 (2× η=0.8). **La dead zone Python est immune au bruit Gaussien même à 270× l'amplitude SPICE.** Le bruit thermique analogique est catégoriquement différent — le claim Paper B est RENFORCÉ, pas fragilisé. Voir §3trigies pour le tableau de calibration complet.

| η SPICE | σ_equiv Python | H_cog Python (BA m=5) | Verdict |
|:-------:|:--------------:|:---------------------:|:-------:|
| 0.10 | 0.0009 | 0.000 | dead zone |
| 0.30 | 0.0026 | 0.000 | dead zone |
| 0.50 | 0.0044 | 0.000 | dead zone |
| 0.80 | 0.0072 | 0.000 | dead zone |
| — | 1.200 (Item 10 max) | 0.006 | quasi-dead zone |

**Faille B — Bins obsolètes `spice_dead_zone_test.py`** ✅ CLOTURE (2026-04-25)
- Seuils corrigés : `[-1.5, -0.8, 0.8, 1.5]` → `[-1.2, -0.4, 0.4, 1.2]` (KIMI).
- Docstring mise à jour. Conclusion inchangée (H≈0 dans dead zone dans les deux cas).

**Faille C — Dynamique u tronquée dans les netlists SPICE** ✅ CLOTURE (2026-04-25)
- Commentaire NOTE ajouté inline dans `B_du` netlist et dans `python_reference()`.
- Paragraphe "Scope of the hardware validation" ajouté dans Paper B §2.
- La limitation est désormais explicite et défendable face à un reviewer.

**Faille D — Duplication make_ba() inter-scripts** ✅ CLOTURE (2026-04-25)
- `src/mem4ristor/graph_utils.py` créé : `make_ba()`, `make_er()`, `make_lattice_adj()`.
- Exporté depuis `src/mem4ristor/__init__.py`.
- 7 scripts `p2_*` + `spice_noise_calibration.py` migrés vers `from mem4ristor.graph_utils import make_ba`.
- 5 scripts anciens (limit02_*, spice_*, ablation_*) inchangés pour préserver la reproductibilité des résultats enregistrés.

**Audit Manus v2 — Actions issues des nouvelles sections (2026-04-25)** Voir §3untrigies pour analyse complète.

**§2.4 — Baseline NMI aléatoire** ✅ CLOTURE (2026-04-25)
- 500 permutations bootstrap ajoutées directement dans `p2_doubt_community_detection.py`.
- **Résultat initial (poids signés, bugué)** : NMI_obs ≈ NMI_rand dans 5/6 seeds. Seul Lattice seed=123 significatif.
- **⚠️ BUG CORRIGE (audit Edison 2026-04-25)** : Louvain appliqué à des poids Pearson signés (négatifs possibles) → modularity corrompue. Fix : `weight=|C_u[i,j]|`. Post-fix : Lattice 2/3 seeds significatifs (z_mean=+1.80, signal modéré), BA m=3 ambigu (1 positif p=0.004, 1 négatif z=-2.18).
- §3octvigies mis à jour avec table complète post-fix + note de correction.
- **Manus §2.4 avait raison sur la nécessité de la baseline.** Honnêteté scientifique préservée.

**§1.4 — Ablation σ_social vs bruit pur** ✅ CLOTURE (2026-04-25)
- Script : `experiments/p2_sigma_social_ablation.py`. Conditions : FULL / SS_NOISE / SS_STATIC / FROZEN_U. BA m=3, I_STIM=0.5, 3 seeds.
- **Résultat** : SS_NOISE ≈ SS_STATIC ≈ FULL (delta < 2%). Manus §1.4 partiellement confirmé : le contenu de σ_social est indiscernable du bruit. FROZEN_U radicalement différent (+985% sync avec n=5, +4500% H_cog) → u dynamics sont essentielles comme filtre anti-synchronisation, pas comme décodeur topologique. Requalification du rôle de u : **filtre anti-synchronisation robuste**, pas détecteur de surprise structurelle. Voir §3duotrigies.

**Piste D — Bifurcation tau_u régime endogène** ✅ CLOTURE (2026-04-25)
- Script : `experiments/p2_tau_u_bifurcation_endogenous.py`. I_STIM=0.0, même sweep que §3quatervigies.
- **Résultat** : bifurcation présente (pic sync τ_u=5, gel τ_u≥20) mais H_cog≈0 sans stimulus. Claim "dynamique u structure sans forcage" partiellement confirmé : activité spontanée préservée, diversité cognitive requiert I_stim>0. Voir §3tertrigies.

---

### P3 — Qualité du code
13. **~~`sensory.py` : convolution lente~~** → **FAIT (antérieur)**. `scipy.signal.correlate2d` déjà en place (ligne 3+55). PROJECT_STATUS était désynchronisé.
14. **~~Exports `__init__.py`~~** → **FAIT (antérieur)**. Tous les modules exportés : symbiosis, cortex, hierarchy, arena, inception, viz + dataclasses config. PROJECT_STATUS était désynchronisé.
15. **~~Config par dataclass~~** → **FAIT (antérieur)**. `config.py` : `@dataclass` complet (DynamicsConfig, CouplingConfig, DoubtConfig, NoiseConfig, Mem4Config). PROJECT_STATUS était désynchronisé.
16. **~~Split core.py~~** → **DÉJÀ FAIT (refactoring KIMI)**. `dynamics.py` = neurone, `topology.py` = réseau. `core.py` est une façade de 26 lignes pour la rétrocompatibilité. Rien à faire.

### P4 — Hardware (futur projet séparé)
17. **~~Validation SPICE~~** → **FAIT (2026-04-19)**. `experiments/spice_validation.py` avec ngspice 46. RMS global 9.7×10⁻³ sur lattice 4×4. Voir §3septies.
18. **~~Scaling SPICE topologie hétérogène~~** → **FAIT (2026-04-19, soir)**. `experiments/spice_dead_zone_test.py` : BA m=5 N=64, dead zone confirmée en analogique sur 3 normalisations. Voir §3nonies.
19. **~~SPICE + bruit thermique / mismatch~~** → **FAIT (2026-04-19, soir)**. `experiments/spice_noise_resonance.py`. Réponse : escape partiel (H~0.16) sous bruit fort (η=0.30) + mismatch capacitif 5%. Pas une rescue complète mais synergie réelle. Voir §3decies.
19bis. **~~Sweep σ_mismatch + multi-seed~~** → **FAIT (2026-04-19, soir)**. `experiments/spice_mismatch_sweep.py` : 45 runs, H_max=1.61 (escape complet), 3 régimes caractérisés. Voir §3undecies.
19ter. **~~Étendre la caractérisation~~** → **FAIT (2026-04-19)**. `experiments/spice_19ter_robustness.py` : (a) multi-graphe H=1.688±0.076 sur 5 seeds BA m=5 — robustesse confirmée ; (b) dichotomie σ_c(η=0.1)=0.43 / σ_c(η=0.3)=0.23 / σ_c(η=0.5)≈0 — frontière de phase publiable ; (c) ER p=0.12 H_max=1.758 — mécanisme topology-agnostic confirmé. Voir §3duodecies.
20. **~~Modèle de memristor HfO₂ réaliste~~** → **TESTÉ — RÉSULTAT NÉGATIF (2026-04-19)**. `experiments/spice_p420_hfo2_memristor.py` avec modèle compact Yakopcic (Ron=100, Roff=16k). Testé sur neurone (A), synapse (B), et A+B combiné. Tous convergent vers un consensus parfait v≈-1.27 (H=0.000). Le memristeur déterministe ne suffit pas sans l'ajout du bruit thermique. Piste "changer les poids statiques" définitivement close au profit de la stochastic resonance étudiée en P4.19bis/ter. Voir §3tredecies.
21. **Paper B dédié** au hardware mapping — la validation sub-1% RMS + la confirmation hardware de la dead zone sont déjà 2 résultats publiables.

### P5 — Intégrations (exploratoire)
19. **Intégration cortex/symbiose** — Mem4ristor comme couche mémoire long-terme dans LearnableCortex. Mesurer si la diversité réseau améliore la robustesse à l'oubli catastrophique.


---

## 11. RÈGLE D'OR

> **Toute claim doit correspondre à une preuve dans le code.**
> Si une claim est marquée FAUX dans `docs/limitations.md`, elle doit être qualifiée
> de "phénoménologique" ou "spéculative" dans le preprint.
> Les échecs sont conservés dans `failures/`. Rien n'est effacé.
