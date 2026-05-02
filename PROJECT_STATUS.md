# PROJECT STATUS — Mem4ristor V4.0.0 (Audited Stable)
**Dernière mise à jour : 2026-05-02**
**Auteur : Julien Chauvin (Barman / Orchestrateur)**
**Contexte : Café Virtuel — Laboratoire d'Émergence Cognitive**

> Ce fichier est le point d'entrée pour quiconque (humain ou IA) travaille sur ce projet.
> Lisez-le en premier. Pour l'historique complet des sessions et investigations :
> → **[PROJECT_HISTORY.md](PROJECT_HISTORY.md)**

---

## 1. QU'EST-CE QUE CE PROJET ?

Mem4ristor est une implémentation computationnelle de dynamiques FitzHugh-Nagumo étendues, conçue pour étudier les états critiques émergents dans des réseaux neuromorphiques. L'innovation centrale est le **Doute Constitutionnel (u)** : une variable dynamique qui module la polarité du couplage entre les neurones, créant une frustration adaptative qui empêche l'effondrement du consensus.

Le projet est né au sein du **Café Virtuel**, un laboratoire de collaboration entre un humain et plusieurs IA (Anthropic, OpenAI, xAI, Google, Mistral, DeepSeek). L'historique complet est dans le dépôt Café Virtuel : https://github.com/cafe-virtuel/

Publication : DOI 10.5281/zenodo.19700749 (preprint dans `docs/preprint.pdf`)

---

## 2. ARCHITECTURE DU CODE


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
| **[6] Cohen U3 — non-chevauchement distributions FROZEN_U vs FULL** | **✅ VALIDÉ (2026-04-29)** | U3=100% (empirique et analytique) dans 3 comparaisons. OVL=0.000000 (distributions strictement disjointes). SPICE d=20.78 (n=50) + Python d=13.21 (n=7) + ablation d=11.44 (n=5). Commit 4a0bd62. Voir §3quadragies |
| **[8] RK4 vs Euler — validation intégrateur (paramètres corrigés)** | **✅ VALIDÉ (2026-04-29)** | Euler dt=0.05 validé sur paramètres alignés config.yaml (sigmoid_steepness=π, SOC_LEAK=0.01, ε_u=0.02, τ_u=10). Plasticité=OFF : Max Δ(H_cog)=0.0018, surge delta=0.3pp. Plasticité=ON (λ_learn=0.05) : Max Δ(H_cog)=0.0053, surge delta=0.1pp. Commits 91a0072 + 4cd7fce sur feat/v4-dynamic-heretics. Voir §3novetrigies |
| **[7] dt sensitivity — H_cog stable, synchrony dépend de dt (physique Euler)** | **✅ VALIDÉ (2026-04-29)** | H_cog : max_delta < 0.01 sur toutes topologies (seuil 0.05) → claim principale robuste. Synchrony sensible au dt (BA m=3 : 0.751→0.462→0.396 pour dt=0.01/0.05/0.10) mais variation reflète la dissipation numérique Euler, cohérente avec [8]. Les comparaisons relatives FULL vs FROZEN_U restent valides. 60 runs, 3 topos × 4 dt × 5 seeds. CSV : figures/dt_sensitivity.csv. Voir §3quadragies-bis |
| **[11] LZ par nœud — hubs plus structurés dans FULL, corrélation absente dans FROZEN_U** | **✅ VALIDÉ (2026-04-29)** | FULL Forcé : LZ_mean=1.101 (BA m=3) / 1.123 (BA m=5). FROZEN_U Forcé : LZ_mean=1.619 / 1.679 (+47-50%). **Finding clé : corrélation degré ↔ LZ dans FULL** : r=-0.564 (BA m=3) / r=-0.716 (BA m=5) — les hubs ont des trajectoires plus structurées. Dans FROZEN_U : r≈0 (p>0.5) — corrélation absente. 40 runs, 22s. CSV : figures/lz_per_node.csv. Voir §3quinquagies |
| **[12] Bruit Matern spatialement corrélé — escape dead zone à η=0.1, structure spatiale non-discriminante** | **✅ VALIDÉ (2026-04-30)** | Tous les types de bruit (non-corrélé / Matern exp ℓ=1 / ℓ=3 / Gauss ℓ=3) brisent la dead zone dès η=0.1 (H_cont : ~1.40 → ~3.3-3.7 bits). La structure spatiale du bruit ne modifie pas le seuil d'escape — c'est l'amplitude qui compte. Bonne nouvelle hardware : les corrélations naturelles des memristors physiques n'empêchent pas l'escape par bruit. 80 runs, 39s. CSV : figures/matern_noise.csv. Voir §3quinquagies-bis |
| **[13] Transition de phase événementielle — nœud périphérique, bifurcation irréversible de l'attracteur** | **✅ VALIDÉ (2026-04-30)** | Idée originale : Julien Chauvin (analogie demande en mariage dans un restaurant). Un nœud périphérique forçant à I≥0.8 pendant ≥50 steps produit dH=+1.20 bits sur BA m=3 (bifurcation positive, "elle dit oui"). Le périphérique fait MIEUX que le hub (I=1.5 T=50 : périph dH=+1.03 vs hub dH=+0.21). Sur BA m=5 (dead zone) : tous les forcings produisent dH<0 — le réseau se dégrade quelle que soit l'amplitude. Le seuil de bifurcation positive n'est pas dans l'amplitude — il est dans la topologie. Classe de phénomène distincte de la dead zone structurelle : transition de phase déclenchée par événement (event-driven phase transition). 240 runs, 155s. Commit 667a2a9. CSV : figures/event_phase_transition.csv. Voir §3quinquagies-ter |


> Détail de chaque investigation : voir **PROJECT_HISTORY.md** § Investigations scientifiques.

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

## 5. ARBORESCENCE DU PROJET

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

## 6. COMMENT DÉMARRER

```bash
git clone https://github.com/cafe-virtuel/Mem4ristor.git
cd Mem4ristor
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

## 7. CONTRIBUTEURS IA (traçabilité)

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


## 8. RÈGLE D'OR

> **Toute claim doit correspondre à une preuve dans le code.**
> Si une claim est marquée FAUX dans `docs/limitations.md`, elle doit être qualifiée
> de «phénoménologique» ou «spéculative» dans le preprint.
> Les échecs sont conservés dans `failures/`. Rien n'est effacé.

---

## 🚀 Future Research: Roadmap V4.1 / V5 (Cognitive Innovations)

Following the 2026-05-01 Professional Audit (ManusAI), the following concepts are prioritized for future exploration:

### 1. Metacognitive Plasticity ($u \to \epsilon$)
- **Concept**: Dynamic modulation of neuronal excitability based on local doubt.
- **Goal**: Uncertain nodes become "slow/prudent" while certain nodes become "fast/impulsive", simulating adaptive cognitive effort.

### 2. Non-Local Topological Coupling (Doubt Similarity)
- **Concept**: Establishing virtual links between nodes with similar doubt levels, regardless of physical proximity.
- **Goal**: Emergence of "communities of thought" and selective information gating.

### 3. Dynamic Compartmentalization (Sub-Personalities)
- **Concept**: Using doubt-driven rewiring to isolate contradictory sub-graphs.
- **Goal**: Preventing global consensus collapse by sequestering "traumatic" or inconsistent information in modular silos.

### 4. Advanced Visualization
- **Concept**: 3D mapping of entropy flows between Hierarchical layers (V1 -> V4 -> PFC).



---

## 9. FIGURES DE RÉFÉRENCE — INVENTAIRE INTERNE (02 mai 2026)

Ces figures existent dans figures/ mais ne sont pas citées dans les documents publiés.
Elles sont documentées ici pour traçabilité et pour répondre aux reviewers si nécessaire.

### Résultats scientifiques confirmés (non publiés)

| Fichier | Expérience | Résultat | Statut |
|---------|-----------|---------|--------|
| event_phase_transition.png / .csv / _summary.csv | [13] Transition de phase événementielle | Nœud périphérique I>=0.8 pendant >=50 steps → dH=+1.20 bits (BA m=3). Périphérique > Hub. | Confirmé, commit 667a2a9 |
| matern_noise.png / .csv / _summary.csv | [12] Bruit Matern spatialement corrélé | Tous types de bruit (i.i.d., Matern exp/Gauss) brisent la dead zone à eta=0.1+. Amplitude seule compte. | Confirmé, commit a8d6ba4 |
| v4_dynamic_heretics_emergence.csv | V4 Dynamic Heretics | Loi linéaire t_first = steps_required + 130 * u_threshold (R²~1). Cascade quasi-inévitable pour u_thr <= 0.8. | Confirmé, commit 88b9983 |

### Analyses de robustesse et défenses reviewer

| Fichier | Analyse | Note |
|---------|---------|------|
| floquet_multipliers.png | Multiplicateurs de Floquet (Section 5.2 preprint) | Ajouté figure preprint.tex 02 mai 2026 |
| lyapunov_analysis.png | Exposant de Lyapunov max vs couplage D (Section 5.2 preprint) | Ajouté figure preprint.tex 02 mai 2026 |
| phase_diagram_corrected.png | Diagramme de phase corrigé (post-audit KIMI) | Remplacé par fiedler_phase_diagram.png comme référence principale |
| phase_diagram_v4_extensions.png | Extensions V4 (dynamic heretics) | Exploratoire, données dans figures/v4_parametric_sweep.csv |
| flop_benchmark.png | Benchmark performances (FLOP/s vs N) | Interne — pas dans les papers |

### Données hardware SPICE

| Fichier | Contenu | Note |
|---------|---------|------|
| p420_hfo2_memristor.csv | Caractérisation SPICE HfO₂ — 50 seeds Monte Carlo | Référencé dans Paper B (in preparation) |
