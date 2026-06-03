# PROJECT STATUS — Mem4ristor V6.0.0 (arXiv Ready)
**Dernière mise à jour : 2026-06-02 (Hermes M3 session 013 — claim [20] Intrinsic Oscillator ajouté)**
**Auteur : Julien Chauvin (Barman / Orchestrateur) & Antigravity (Orchestrateur 2026)**
**Contexte : Café Virtuel — Laboratoire d'Émergence Cognitive**

> ⚠️ **AUDIT HERMES M3 — 2026-06-01 (sessions 010-011)**
> **Statut** : En cours de correction (3 obligatoires, 2 recommandées)
> **Détails** : voir `sessions/SESSION_010_CARTOGRAPHIE_HERMES.md` et `sessions/SESSION_011_AUDIT_SCIENTIFIQUE.md`
> **Resume** : 8/12 claims CONFORMES, 1 DISCREPANCY majeure (C07 topologie erronee), 1 CLAIM INFIRMEE (C12 Binder U4 plat), 1 valeur canonique floue (C05 EBC vs combined), 1 avertissement seeds/topologie (C08), 1 finding glassy qualitativement confirme.

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
| **[14] U clamping avec u_clamp=0.6 — effet NOCIF sur la synchronie** | **✅ AUDITÉ + CORRIGÉ (2026-05-30)** | Le clamping u=0.6 donne H=4.00 (+0.66 bits) mais RE-SYNCHRONISE le reseau (sync=0.53). C'est NOCIF sur la metrique robuste. Le gain H_cont est un artefact du binning accompagne d'un retour au consensus. Scripts : adaptive_D_conclusive_test.py. Voir AUDIT_LOG.md AUDIT-008 + AUDIT-011 |
| **[15] U saturation — mécanisme des deux attracteurs élucidé** | **✅ AUDITÉ (2026-05-30)** | La dynamique u a deux attracteurs stables : D=0 → u→~0.05, D>0 → u→~0.999. La fenêtre "optimale" 0.575-0.625 est instable sans clamping. Mécanisme : feedback négatif sigma_social ↔ sigmoid(u). Script : u_saturation_profile.py. Voir AUDIT_LOG.md AUDIT-007 |
| **[16] D(u) = D_max * u — SWEET SPOT CONFIRMÉ (nuance)** | **✅ AUDIT-015 (2026-05-31)** | D=0.50*u abaisse le seuil FUNCTIONAL a m>=6 (vs m>=7 avec D=0.15). C'est le meilleur protocole: LZ=0.58, H=3.17 a m=10. D=0.15 est suboptimal. D=0 ne produit jamais FUNCTIONAL. Voir AUDIT_LOG.md AUDIT-015 |
| **[17] Transition de phase topologique — seuil PROTOCOLE-DEPENDANT** | **✅ AUDIT-015 (2026-05-31) + DZ2 sweep (2026-06-03) + preprint integrated** | LZ76 desambiguise sync=0: D=0.15 -> seuil m>=7; D=0.50*u -> seuil m>=6 (LZ=0.66 a m=6). D=0 ne produit jamais FUNCTIONAL (LZ~1.1). Le seuil n'est PAS un invariant topologique. **MAJ DZ2 (2026-06-03) :** D=0.50*u ameliore aussi m=1 (frontiere dead zone) : sync=0.72, H=3.70 vs D=0.15 sync=0.51 — meilleur resultat sur tout m. Mais a m>=6 : H->0, sync->-0.01 (collapse total, pas functional). Le mecanisme est optimal POUR LES TOPOLOGIES INTERMEDIAIRES (m=1-5), pas pour les denses. **Preprint mis a jour (2026-06-03)** : nouvelle subsection §3.4.1 "Adaptive Coupling Protocol D(u)=D_max * u" avec table de donnees et recommendation operative. Voir figures/dz2_topological_sweep_agg.csv |
|| **[18] V5 FINAL: D(u)=0.50*u + alpha_meta=-4.0 SWEET SPOT — RESTREINT AUX TOPOLOGIES DENSES** | **✅ AUDIT-018 ARRET (2026-05-31)** | Combination optimale D(u) adaptive + metacognition freeze (alpha=-4.0) — MAINTENANT QUALIFIEE: le mecanisme est LEGITIME pour m>=7 (zone morte V4) mais CHAOTICgenic pour m<=5 (LZ monte a 0.86-0.94). Gains vs V4 D=0.15: m=3 (LZ=0.942 CHAOTIC, pas de structure), m=5 (LZ=0.862 CHAOTIC), m=7 (LZ=0.560 FUNCTIONAL, gain legitime +1.47), m=10 (LZ=0.510 FUNCTIONAL, gain legitime +0.84). Sync safe everywhere (|sync|<0.016). **Le SWEET SPOT est optimal POUR LES TOPOLOGIES DENSES (m>=7), pas universellement.** SECTION BINDER DU PREPRINT INVALIDEE (AUDIT-017) + CHEMIN B ABANDONNE (AUDIT-018). Le manuscrit重构 autour de la depression H_stable (crossover progressif) et de la baisse LZ — pas d'une transition de phase thermodynamique. Scripts: experiments/p2_v5_final_best.py. CSV: figures/p2_v5_final_best.csv. Voir AUDIT_LOG.md AUDIT-017 + AUDIT-018 |
|| **[19] OPTION B COHERENCE FIX + EDISON REVIEW (2026-06-01): Glassy dynamics fingerprint + Active Inference overclaim** | **NEW (2026-06-01)** | (a) Var(H_stable) peaks at critical zone (0.544 vs 0.379 sparse / 0.265 dense). Var(LZ) increases monotonically with lambda2 (0.056 -> 0.076 -> 0.090). Diverging variance pattern = glassy dynamics fingerprint. Not documented in preprint. See docs/papers/recommendations/EDISON_REVIEW_FINDINGS_20260601.md. (b) Section 6.3 claims Active Inference / Friston FEP but admits no generative model (line 450). FEP components absent: no generative model, no variational inference, no free energy minimization. Mechanism = feedback control / homeostatic regulation. "Active Inference" framing is a rhetorical stretch. See EDISON_REVIEW_FINDINGS_20260601.md. |
| **[20] INTRINSIC OSCILLATOR — narrow-band endogenous rhythm f~0.01 cycles/step** | **NEW (2026-06-02) — session 013** | 7 POCs adversariaux executés pour tester le reframe "doute = phase de resolution" (DOUBT-REFRAME-001 de Claude Cowork). Resultat central : le reseau Mem4ristor est un **oscillateur endogene a bande etroite** avec frequence intrinseque f ~ 0.01 cycles/step (periode ~70-100 pas), reproductible sur 5 seeds x 2 topologies (m=3, m=6) x 3 protocoles de couplage (D=0, 0.15, 0.5*u). LZ spontane > 1.0 partout (la "structured dead zone" LZ<0.85 du claim [17] n'est PAS un regime spontane, c'est une REPONSE a l'excitation I_stim=0.5). POC #1 (absence) : AC@lag50 = +0.6 a +0.7, periode stable. POC #4 (menteur) : le reseau ignore ou suit le menteur a m=6 D=0.15 (FOLLOW_LIAR reproduitible, diff=-1.43, marqueur de frontiere). POC B (2 frequences) : INTRINSIC gagne 6/6. POC C (10 pivots) : F_DRIVE gagne 3/6 conditions — seuil de bascule identifie entre 1 et 10% de noeuds pilotes. POC D (hub) : INTRINSIC 6/6, la centralite topologique n'aide PAS. POC #5 (bruit) : PASSIVE 6/6. **Reformulation recommandee** : remplacer le reframe "doute = resolution" par "doute = mecanisme de resistance aux inputs externes, avec seuil de bascule en nombre de porteurs". Le claim [20] est testable, falsifiable, et defendable. Scripts : `experiments/poc1_test_of_absence.py`, `experiments/poc245_batch.py`. CSVs : `figures/poc1_absence_agg.csv`, `figures/poc245_raw.csv`. Write-up : `sessions/SESSION_013_REFRAMING_DOUTE.md`. |


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
| Antigravity (Gemini 3 Flash) | **Run Héroïque N=1600**, validation Binder $U_4$, résolution audit Mistral (19/20), Expérience 008 "Guerre des Phases" (2026-05-16) |
| **Aria, Flux, Sentinel** | **Système MAS 2026** : Veille scientifique, optimisation Chemical Inductor, Stress Test V3 (Synchronisation 100%) (2026-05-16) |

Niveau de transparence : **Radical** — transcripts complets dans le dépôt Café Virtuel.

---


## 8. RÈGLE D'OR

> **Toute claim doit correspondre à une preuve dans le code.**
> Si une claim est marquée FAUX dans `docs/limitations.md`, elle doit être qualifiée
> de «phénoménologique» ou «spéculative» dans le preprint.
> Les échecs sont conservés dans `failures/`. Rien n'est effacé.

---

## 🚀 Roadmap V5 (Cognitive Innovations) — État au 2026-05-05

Suivant l'Audit Professionnel ManusAI (2026-05-01), les trois axes V5 ont été explorés sur la branche `main` (commits 695a403, 145316e, dbf40b4).

### 1. Metacognitive Plasticity ($u \to \epsilon$) — ✅ ARRETE (2026-05-31)
- **Anciens résultats (2026-05-05)**: alpha=-0.5 → H=4.82 bits (+0.79 vs V4 pur 4.03)
- **Resultat FINAL (2026-05-31, 12 seeds + AUDIT-017)**: alpha=-4.0 beneficiaire sur m>=7 (LZ baisse, structure) mais CHAOTICgenic sur m<=5 (LZ monte a 0.86-0.94). Gains vs V4: m=3 (LZ=0.942 CHAOTIC — non-structurel), m=5 (LZ=0.862 CHAOTIC — non-structurel), m=7 (LZ=0.560 FUNCTIONAL, +1.47 bits), m=10 (LZ=0.510 FUNCTIONAL, +0.84 bits).
- **Mecanisme revise (AUDIT-017)**: alpha=-4.0 gele quasi-totalement la plasticite w. L'effet est regime-dependent:
  - m<=5 (sparse): gele dans le chaos — LZ passe de 0.79->0.94. Gain H_cont = gain chaos.
  - m>=7 (dense): gele dans la structure — LZ passe de 0.77->0.51. Gain legitime.
  - Le SWEET SPOT D(u)+alpha=-4.0 est optimal POUR LES TOPOLOGIES DENSES (m>=7), pas universellement.
- **ARRET (AUDIT-018)**: Section Binder du preprint INVALIDEE. Chemin B (correlation LZ-H_stable) ABANDONNE. Manuscrit重构 autour de la depression H_stable (crossover progressif) et de la baisse LZ — pas d'une transition de phase thermodynamique. Voir AUDIT-017 + AUDIT-018.

### 2. Non-Local Topological Coupling (Doubt Similarity) — ❌ ÉCHEC
- **Résultat** : u_spread augmente (+0.01 à +0.03) mais H baisse (-0.08 à -0.23 bits). La synchronisation inter-nœuds similaires étouffe la chimère. Commit 145316e.
- **Conclusion** : V4 pur reste optimal sur cet axe.

### 3. Dynamic Compartmentalization (Sub-Personalities) — ✅ IMPLÉMENTÉ + CLARIFIÉ
- **Ancien résultat (2026-05-05)**: K=3 full gamma=0.10 → H=4.18 bits (+0.15 vs V4 pur)
- **Résultat FINAL (2026-05-31)**: la compartimentalisation est NEUTRE ou NOCIVE en combinaison avec D(u)+alpha=-4.0.
  Sur BA m>=7 (dead zone): K=3 reduit H de -0.30 bits vs D(u)+alpha=-4.0 seul.
  La separation en sous-groupes n'apporte rien quand la plasticite w est geleed.
- **Conclusion**: K=3 etait beneficial en isolation (V4+compart) mais pas en combination avec D(u)+meta.
  K=2 ou K=10 sont a peu pres equivalents a pas de compartiments (within noise).

### 4. Prochaine étape V5 — ✅ RÉSOLUE (2026-05-31)
- **PRIORITÉ initiale**: combiner Metacognitif (alpha=-0.5) + Compartimentalisation (K=3 full)
- **Résultat FINAL**: D(u)=0.50*u + alpha_meta=-4.0 SANS compartimentalisation = OPTIMAL
  Gains vs V4: m=3 (+1.50), m=5 (+1.90), m=7 (+1.47), m=10 (+0.89)
- **Piste V6**: Memristors Photoniques — I_stimulus via fibre optique (matériaux : GST, VO2, WO3)

### 5. Advanced Visualization
- **Concept** : 3D mapping of entropy flows between Hierarchical layers (V1 → V4 → PFC). Pending.



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
| exp_008_guerre_phases.png | Expérience 008 | La minorité organisée (15%) crée une oscillation perpétuelle à l'interface. | Confirmé (Shadow Lab) |
| exp_008_v3_sentinel_proof.png | Expérience 008 V3 | **Stress Test Sentinel** : Synchronisation 1.0000 sans bruit. Preuve de la robustesse structurelle de l'inductance chimique. | VALIDÉ (Mai 2026) |

### Analyses de robustesse et défenses reviewer

| Fichier | Analyse | Note |
|---------|---------|------|
| floquet_multipliers.png | Multiplicateurs de Floquet (Section 5.2 preprint) | Ajouté figure preprint.tex 02 mai 2026 |
| lyapunov_analysis.png | Exposant de Lyapunov max vs couplage D (Section 5.2 preprint) | Ajouté figure preprint.tex 02 mai 2026 |
| phase_diagram_corrected.png | Diagramme de phase corrigé (post-audit KIMI) | Remplacé par fiedler_phase_diagram.png comme référence principale |
| phase_diagram_v4_extensions.png | Extensions V4 (dynamic heretics) | Exploratoire, données dans figures/v4_parametric_sweep.csv |
| flop_benchmark.png | Benchmark performances (FLOP/s vs N) | Interne — pas dans les papers |

### Données hardware SPICE

|| Fichier | Contenu | Note |
|---------|---------|------|
|| p420_hfo2_memristor.csv | Caractérisation SPICE HfO₂ — 50 seeds Monte Carlo | Référencé dans Paper B (in preparation) |

---

## 10. AUDIT HERMES M3 — 2026-06-01 (sessions 010 + 011)

**Methode** : EDISON stop-rule + cartographie structurelle. 12 claims primaries + 4 secondaires + 1 finding glassy auditees contre les CSV source.

### 10.1 Verdict global

| Categorie | Count | Claims |
|-----------|-------|--------|
| ✅ CONFORMES | 8 | C01, C02, C03, C04, C06, C09, C10, C11 |
| 🚨 DISCREPANCY majeure | 1 | **C07** (topologie erronee + valeur lattice reelle) |
| 🚨 CLAIM INFIRMEE | 1 | **C12** (Binder U4 plat, pas de minimum) |
| ⚠️ VALEUR FLoue | 1 | **C05** (EBC=2.31 vs combined=3.14) |
| ⚠️ SEEDS/TOPOLOGIE ERRONES | 1 | **C08** (5 seeds lattice, pas 3 seeds BA m=3) |
| ✅ CONFORME (exploratoire) | 1 | S04 |

**Plus** : 1 finding glassy qualitativement confirme (variance peak CRITICAL, LZ_std monotone croissant).

### 10.2 Corrections obligatoires identifiees

**OBLIGATOIRE #1 — C07 (lattice sync 0.031 vs 0.0197)**
- Claim preprint : "lattice FULL sync = 0.031 ± 0.034"
- Valeur reobservee dans `figures/p2_table1_sync.csv` : lattice 10×10 = **0.0197 ± 0.0142** (~36% plus faible)
- Source du 0.031 = `figures/ablation_coordination_topo_sweep.csv` BA m=3 FORCED FULL (topologie erronee)
- **Action** : Corriger `docs/CLAIMS_REGISTER.md` C07 (topologie) + `docs/papers/preprint/preprint.tex` lignes 241 & 375

**OBLIGATOIRE #2 — C12 (Binder U4 plat)**
- Claim preprint : "U4 convergence vers minimum, transition thermodynamique"
- Valeur reobservee dans `figures/v6_binder_cumulant_U4.csv` : U4 = 0.6641-0.6666 (variation 0.38%, PAS de minimum)
- 9-15 sims reelles (pas 40 annoncees)
- AUDIT-017 du 31 mai avait deja fait l'arret, mais la ligne C12 du Claims Register est toujours marquee "verifiee"
- **Action** : Supprimer la ligne C12 du Claims Register, verifier que `preprint.pdf` recompile du 1er juin ne contient plus la section Binder FSS

**OBLIGATOIRE #3 — C05 (valeur canonique floue)**
- Claim preprint : "λ2_crit = 2.31 (2.13-2.50)"
- CSV `lambda2_crit_regression.csv` montre DEUX valeurs : EBC midpoint = 2.31 (n=36) ET combined = 3.14 (n=58, CI95 [1.93, 4.85])
- **Action** : Decider laquelle est canonique pour le preprint, supprimer l'autre du CSV ou clarifier

### 10.3 Corrections recommandees (non bloquantes)

**RECOMMANDE #4 — C08 (topologie + seeds)**
- Claim : "MI FROZEN/FULL ratio 2.2x, BA m=3, 3 seeds"
- Verite : 5 seeds lattice (pas 3 seeds BA m=3). Ratio 2.25x sur lattice, 1.84x sur BA m=3
- **Action** : Corriger `docs/CLAIMS_REGISTER.md` C08 (topologie=BA m=3 si on garde 3 seeds, ou lattice si on accepte 5 seeds)

**RECOMMANDE #5 — Glassy dynamics dans Discussion**
- Pattern qualitatif confirme (H_std peak CRITICAL, LZ_std monotone croissant)
- Valeurs quantitatives EDISON (0.379, 0.544, 0.265) ne matchent pas l'aggregation (0.061, 0.104, 0.072) — facteur 5-10x. Hypothese : EDISON a utilise raw data non-agregee
- **Action** : Ajouter 1-2 phrases dans la Discussion du preprint V6 sur la variance comportementale. Section "spectral crossover" deja en place pourrait etre etendue.

### 10.4 Inventaire structurel (Session 010)

| Item | Taille/Count | Risque |
|------|--------------|--------|
| Working tree pollue | 80+ deleted, 90+ untracked | Confusion agent suivant |
| SPICE results/ | 11 GB (gitignore) | Verifier si regenerable |
| 3 papiers LaTeX en parallele | preprint + paper_2 + paper_B | Coherence inter-papiers non verifiee |
| 4 branches distantes obsoletes | kimi, v4, spice, v5 | Toutes absorbees ou archivees |
| core.py = 25 lignes | facade vide | Desync avec PROJECT_STATUS §2 |
| Sequence sessions/ | manque 001, 007 | Non-continue |
| 74 scripts sans CSV/PNG | sur 120+ | Non executes ou outputs mal nommes |

### 10.5 Scripts prometteurs (Phase 3, chasse aux approches)

- `experiments/campaign_j_binder_lz.py` (16 KB, mai 31) — source du CSV qui a infirme C12
- `experiments/fss_lz_sweep.py` (20 KB, mai 30) — sweep LZ76 le plus recent, "regime map" qui remplace C12
- `experiments/p2_v5_final_best.py` (12 KB, mai 31) — protocole V5 optimal (D(u)=0.50*u + alpha_meta=-4.0)
- Branche `feat/kimi-p419-continuous-entropy` (5 commits, avril) — corrections metriques non reportees dans main ?

---

**Statut arXiv** : Pas soumissible en l'etat. 3 corrections obligatoires a faire avant submission. Delai estime : 2-4 heures de travail concentrees.