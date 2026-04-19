# PROJECT STATUS — Mem4ristor v3.2.0
**Dernière mise à jour : 2026-04-19**
**Auteur : Julien Chauvin (Barman / Orchestrateur)**
**Contexte : Café Virtuel — Laboratoire d'Émergence Cognitive**

> Ce fichier est le point d'entrée pour quiconque (humain ou IA) travaille sur ce projet.
> Lisez-le en premier. Mettez-le à jour après chaque session de travail.

---

## 1. QU'EST-CE QUE CE PROJET ?

Mem4ristor est une implémentation computationnelle de dynamiques FitzHugh-Nagumo étendues, conçue pour étudier les états critiques émergents dans des réseaux neuromorphiques. L'innovation centrale est le **Doute Constitutionnel (u)** : une variable dynamique qui module la polarité du couplage entre les neurones, créant une frustration adaptative qui empêche l'effondrement du consensus.

Le projet est né au sein du **Café Virtuel**, un laboratoire de collaboration entre un humain et plusieurs IA (Anthropic, OpenAI, xAI, Google, Mistral, DeepSeek). L'historique complet est dans le dépôt Café Virtuel : https://github.com/cafe-virtuel/

Publication : DOI 10.5281/zenodo.18620597 (preprint dans `docs/preprint.pdf`)

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

### 3bis. LIMIT-05 : Entropie maximale (2026-03-21)

**Question** : Le claim H ≈ 1.94 est-il reproductible ?

**Méthode** : Sweep paramétrique 4 phases + analyse de stabilité (800+ combinaisons, 5-7 seeds, ~7 min).

| Métrique | Valeur |
|:---------|:-------|
| Max théorique H (5 bins) | log₂(5) = 2.3219 |
| Meilleur H **transitoire** | 2.3143 (99.7% du max) |
| Meilleur H **stable** (derniers 25%) | 1.48 ± 0.66 (D=0.01, I=1.0) |
| H stable config par défaut | **0.92 ± 0.04** |

**Verdict** : H ≈ 1.94 mesuré sur un pic transitoire. Attracteur réel ≈ 0.92.

### 3ter. LIMIT-02 : Strangulation scale-free (2026-03-21)

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

7. **`sensory.py` : convolution lente** — Remplacer par `scipy.signal.correlate2d` (gain ~100x)
8. **~~Module `viz.py`~~** → FAIT (stable, intégré dans demo_applied.py)
9. **Exports `__init__.py`** — Modules symbiosis, cortex, etc. non exportés
10. **~~Normalisation par degré pour LIMIT-02~~** → **FAIT** (2026-04-10). `degree_linear` (D/deg(i)) validé. Voir §3quinquies.

### Priorité basse (évolution) — COMPLÉTÉS 2026-03-22

11. **~~Démonstration appliquée~~** → **FAIT** (2026-03-22). `examples/demo_applied.py` : 4 démos (sensory pipeline, hysteresis comparison, scale-free sparse, phase diversity), 5 PNG.
12. **~~V5 (hysteresis)~~** → **FAIT** (2026-03-22). Dead-zone latching [0.35, 0.65] + watchdog fatigue. 3 tests passent. H_stable +5%.
13. **Config par dataclass** — Remplacer dicts imbriqués. Non critique.
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

---

## 10. PROCHAINES ÉTAPES (par priorité)

### P0 — Soumission
1. **Relecture finale** du PDF par Julien → soumission arXiv ou journal
2. **Upload Zenodo** avec la v3.2.0 (DOI existant à mettre à jour)
3. **Commit + push GitHub** avec tous les changements de cette session

### P1 — Bugs pré-existants à fixer
4. **~~`test_swarm_synchronization`~~** → **FAIT (2026-04-19)**. Test était écrit pour mean-field symétrique mais l'implémentation est MAX FIELD asymétrique (intentionnel : vétéran préservé). Test corrigé.
5. **~~`test_entropy_preservation_with_v4`~~** → **FAIT (2026-04-19)**. Ring N=10 sans hubs → remplacé par BA m=3 N=50 + `coupling_norm='degree_linear'`. H ≈ 0.83.

### P2 — Paper 2 : "Breaking the Topological Diversity Boundary" (pistes Grok + Antigravity, 2026-04-11)

> Ces pistes sont documentées ici pour mémoire. Elles constituent le matériau d'un **deuxième papier** (v3.3+).
> La v3.2.0 est publiée telle quelle — le résultat négatif (dead zone m≥5) est une contribution en soi.

**Priorité haute (impact fort, effort raisonnable) :**

6. **~~Normalisation spectrale~~** → **TESTÉ — RÉSULTAT NÉGATIF (2026-04-19)**. Mode `coupling_norm='spectral'` implémenté dans `core.py`. 0/6 wins sur la dead zone. Voir §3octies. Conclusion : la dead zone n'est pas un problème de pondération.
7. **Finite-size scaling** — Sweep m × γ pour N ∈ {100, 400, 1600, 6400} avec sparse CSR. Tracer λ₂_critique(N). Si transition stable → loi d'échelle publiable.
8. **~~Figure λ₂ vs H_stable~~** → **FAIT (2026-04-19)**. `experiments/fiedler_phase_diagram.py` → `figures/fiedler_phase_diagram.png` + `.csv`. 15 topologies × 2 norms × 3 seeds.
9. **Edge betweenness + diamètre** — Montrer que λ₂ est un proxy de la multipath redundancy, pas la cause directe. Script NetworkX rapide.
9bis. **Adaptive heretics / dynamique modifiée** — Maintenant que toutes les pistes de pondération sont éliminées, c'est la priorité haute pour Paper 2. Tester (a) η dynamique (item 11) et (b) stochastic resonance ciblé sur la dead zone (item 10).

**Priorité moyenne (intéressant, Paper 2 ou 3) :**

10. **Stochastic resonance** — Sweep σ_noise × λ₂ : le bruit optimal dépend-il de la topologie ? Analogie avec spin glasses biologiques.
11. **Adaptive heretics** — η dynamique : nœuds deviennent hérétiques si u_i > 0.8 pendant >100 pas. Auto-régulation. Pourrait supprimer la dead zone sans changer la topologie. ⚠️ Change le modèle fondamentalement → v4.0.
12. **Doubt-driven community detection** — Matrice de doute u(i) comme signal pour détecter des communautés fonctionnelles. Spéculatif mais original.

### P3 — Qualité du code
13. **`sensory.py` : convolution lente** — Remplacer par `scipy.signal.correlate2d` (gain ~100x)
14. **Exports `__init__.py`** — Modules symbiosis, cortex, etc. non exportés
15. **Config par dataclass** — Remplacer dicts imbriqués (non critique)
16. **Split core.py** → `neuron.py` + `network.py` (Phase 5, reportée)

### P4 — Hardware (futur projet séparé)
17. **~~Validation SPICE~~** → **FAIT (2026-04-19)**. `experiments/spice_validation.py` avec ngspice 46. RMS global 9.7×10⁻³ sur lattice 4×4. Voir §3septies.
18. **~~Scaling SPICE topologie hétérogène~~** → **FAIT (2026-04-19, soir)**. `experiments/spice_dead_zone_test.py` : BA m=5 N=64, dead zone confirmée en analogique sur 3 normalisations. Voir §3nonies.
19. **~~SPICE + bruit thermique / mismatch~~** → **FAIT (2026-04-19, soir)**. `experiments/spice_noise_resonance.py`. Réponse : escape partiel (H~0.16) sous bruit fort (η=0.30) + mismatch capacitif 5%. Pas une rescue complète mais synergie réelle. Voir §3decies.
19bis. **Sweep σ_mismatch + multi-seed** — Étendre le test : σ ∈ {5%, 10%, 20%, 50%}, 5 seeds par cellule. Caractériser la courbe d'escape vs désordre. Si H continue à monter avec σ → mécanisme purement de désordre figé (publiable comme "memristor variability is a feature").
20. **Modèle de memristor HfO₂ réaliste** — Remplacer la capacité 1F idéale par un modèle compact memristor (Stanford-PKU, etc.). Mesurer comment l'imperfection hardware module la dynamique.
21. **Paper B dédié** au hardware mapping — la validation sub-1% RMS + la confirmation hardware de la dead zone sont déjà 2 résultats publiables.

### P5 — Intégrations (exploratoire)
19. **Intégration cortex/symbiose** — Mem4ristor comme couche mémoire long-terme dans LearnableCortex. Mesurer si la diversité réseau améliore la robustesse à l'oubli catastrophique.


---

## 11. RÈGLE D'OR

> **Toute claim doit correspondre à une preuve dans le code.**
> Si une claim est marquée FAUX dans `docs/limitations.md`, elle doit être qualifiée
> de "phénoménologique" ou "spéculative" dans le preprint.
> Les échecs sont conservés dans `failures/`. Rien n'est effacé.
