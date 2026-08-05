# Rapport d’audit critique — Mem4ristor

**Dépôt audité :** `cafe-virtuel/Mem4ristor`  
**Branche observée :** `main`  
**Périmètre :** arborescence, code source, expériences, tests, documentation, résultats versionnés, historique Git et configuration  
**Date de l’audit :** 2026-08-05  
**Nature :** analyse scientifique et logicielle critique, sans modification du code

---

## 1. Synthèse exécutive

Mem4ristor est un projet scientifique computationnel fondé sur des réseaux d’oscillateurs FitzHugh–Nagumo enrichis par une variable dynamique de doute `u ∈ [0,1]`. Cette variable module le signe et l’amplitude du couplage social via une fonction sigmoïde dite *Levitating Sigmoid* :

```text
w(u) = tanh(π(0.5 − u)) + δ
```

Le dépôt contient un noyau logiciel réel et relativement structuré, mais aussi plusieurs générations de modèles, des résultats historiques, des extensions exploratoires et des documents contradictoires. Il ne doit pas être considéré comme un instantané homogène ni comme une release scientifique parfaitement reproductible.

### Conclusion scientifique principale

La conclusion la mieux étayée est étroite :

> Dans certains protocoles simulés, un réseau utilisant un doute `u` dynamique devient beaucoup moins synchronisé qu’une version où `u` est gelé (`FROZEN_U`).

Cette conclusion est soutenue par l’ablation FULL/FROZEN_U et par une mesure de synchronie de Pearson indépendante du binning. Elle ne démontre pas, à elle seule, une cognition, une mémoire, une capacité générale de calcul ou un avantage énergétique.

### Ce qui a été réfuté ou fortement réduit

- λ₂≈2,31 n’est plus une cause acceptable de la dead zone ; le projet lui-même le requalifie comme corrélation historique.
- La transition thermodynamique Binder U4 a été infirmée.
- La transition événementielle annoncée avec `dH=+1.20` ne se reproduit pas avec le code actuel.
- L’universalité du ratio de 15 % est abandonnée.
- Le modèle perd contre un ESN sur NARMA10, contre un filtre de Kalman sur Lorenz et contre la recherche aléatoire sur Max-Cut.
- L’économie d’énergie est réfutée à substrat comparable : le modèle est annoncé comme environ 15 fois plus coûteux qu’un filtre RC équivalent.

### Verdict global

Le projet est intéressant comme étude de dynamique de réseaux couplés et comme hypothèse de couplage anti-synchronisant. Il est prématuré de le présenter comme architecture cognitive, processeur neuromorphique utile ou dispositif hardware validé.

**Niveau de confiance global : modéré.** La confiance est élevée sur les incohérences documentaires et les défauts de packaging ; modérée sur les conclusions scientifiques, car plusieurs expériences centrales restent dépendantes de protocoles spécifiques et de données déjà générées.

---

## 2. Cartographie du dépôt

### 2.1 Noyau logiciel

| Fichier | Fonction | Statut d’audit |
|---|---|---|
| `src/mem4ristor/dynamics.py` | Moteur FHN + `u` + bruit + plasticité + extensions | Noyau réel |
| `src/mem4ristor/topology.py` | `Mem4Network`, Laplacien, normalisations, rewiring, sparse | Noyau réel |
| `src/mem4ristor/metrics.py` | Entropies, LZ, synchronie, MI, transfert | Noyau métrique |
| `src/mem4ristor/graph_utils.py` | Générateurs BA, ER, lattice, graphes dirigés | Utilitaire canonique |
| `src/mem4ristor/core.py` | Façade de compatibilité de 25 lignes | Réexport uniquement |
| `src/mem4ristor/config.py` | Dataclasses de configuration | Partiel |
| `src/mem4ristor/config.yaml` | Valeurs par défaut YAML | Partiellement périmé |
| `src/mem4ristor/__init__.py` | API publique | Version périmée |

Le moteur actuel est bien séparé en `dynamics.py`, `topology.py` et `metrics.py`. `core.py` n’est plus le moteur canonique, contrairement à des documents historiques.

### 2.2 Extensions applicatives

Le package exporte également :

- `SensoryFrontend` (`sensory.py`) ;
- `LearnableCortex` (`cortex.py`) ;
- `CreativeProjector` et `SymbioticSwarm` (`symbiosis.py`) ;
- `DreamVisualizer` (`inception.py`) ;
- sonification et visualisations.

Ces modules sont utiles pour les démonstrations mais ne sont pas au même niveau de validation que le noyau dynamique.

### 2.3 Expériences

Les expériences sont organisées par préfixes historiques :

- `p2_*` : Paper 2 et ablations ;
- `b1_*` à `b6_*` : campagnes de validation et de réduction du périmètre ;
- `expA_*`, `expB_*` : mécanismes et usages applicatifs ;
- `p15_*` à `p21_*` : benchmarks et diagnostics ;
- `spice_*` : pont hardware ;
- `v6_*` : Binder/FSS et extensions.

Le dossier `experiments/scratch/` est gitignoré dans `.gitignore:85`, alors que de nombreux documents le citent encore. Cela crée une frontière entre l’état local du chercheur et l’état réellement reproductible par un tiers.

### 2.4 Tests

Les tests couvrent :

- invariants numériques ;
- robustesse NaN/Inf ;
- fuzzing ;
- parity sparse/dense ;
- graphes dirigés ;
- hystérésis ;
- ART ;
- métacognition ;
- compartiments ;
- coupling non local ;
- doute complexe ;
- métriques.

Le dépôt annonce `130 passed + 2 xfail` dans `PROJECT_STATUS.md:371`, mais cette valeur n’a pas pu être confirmée dans l’environnement d’audit : `pytest` et `numpy` n’étaient pas installés dans l’interpréteur disponible.

### 2.5 Documentation et articles

Documents récents et critiques :

- `PROJECT_STATUS.md` ;
- `docs/BILAN_FORCES_FAIBLESSES.md` ;
- `docs/CLAIMS_REGISTER.md` ;
- `AUDIT_LOG.md` ;
- `PROJECT_HISTORY.md`.

Documents périmés ou partiellement périmés :

- `README.md` ;
- `CONTEXT.md` ;
- `docs/limitations.md` ;
- `docs/compendium/COMPENDIUM.md` ;
- `docs/compendium/BRIEF_COMPENDIUM.md` ;
- `docs/unified_specification_v23.md` ;
- `docs/theoretical_anchoring.md` ;
- `REPRODUCE_RESULTS.md`.

### 2.6 Packaging et CI

- `pyproject.toml` annonce la version 6.0.0 et les dépendances principales.
- `requirements.txt` ajoute `pandas` et `tqdm`, qui ne sont pas déclarés dans `pyproject.toml`.
- `.github/workflows/test.yml` exécute uniquement pytest sur Python 3.9–3.11.
- Aucun lint, typecheck, build Docker, compilation LaTeX ou test de reproduction scientifique n’est exécuté automatiquement.

---

## 3. Reconstruction historique des couches

### 3.1 Couche V2 : modulateur linéaire

La première génération utilisait `(1−2u)`, avec une annulation à `u=0.5`. Cette formulation survit dans des documents tels que `docs/unified_specification_v23.md` et `docs/theoretical_anchoring.md`.

Elle portait des claims plus larges : loi universelle des 15 %, invariance topologique, architecture cognitive et avantage hardware.

**Statut : historique et non canonique.**

### 3.2 Couche V3 : Levitating Sigmoid

La fonction linéaire a été remplacée par :

```text
w(u) = tanh(π(0.5 − u)) + δ
```

Le code actuel utilise `δ=0.01` dans `dynamics.py:58-60`. Cette couche constitue la base réelle du modèle actuel.

### 3.3 Couche V3.2 : normalisation topologique

Le problème de strangulation des hubs a conduit à l’introduction de plusieurs normalisations :

- `uniform` ;
- `degree` ;
- `degree_linear` ;
- `degree_log` ;
- `spectral`.

Le résultat réellement soutenu est que le choix de normalisation dépend de la famille de graphe. `degree_linear` aide sur BA m=3 mais n’est pas universel : `docs/limitations.md:84-115` documente des cas où uniform gagne et des cas où aucune normalisation ne fonctionne.

### 3.4 Couche V4 : heretics dynamiques et λ₂

Les « dynamic heretics » deviennent hérétiques après un seuil de doute prolongé. La régression λ₂≈2,31 formalise ensuite une frontière entre régimes.

Cette lecture est progressivement devenue l’axe principal du projet, avant d’être réfutée par des contrôles à degré fixé et à λ₂ variable.

### 3.5 Couche V5 : extensions adaptatives

Le code conserve plusieurs extensions, mais elles sont désactivées par défaut :

- métacognition ;
- coupling non local ;
- compartimentalisation ;
- ART ;
- watchdog.

Leur présence dans le noyau crée un risque de confusion entre le modèle qui a produit le preprint et la plateforme expérimentale actuelle.

### 3.6 Couche V6 : Binder et hardware

Le Binder cumulant a d’abord été interprété comme preuve d’une transition de phase. Le résultat a ensuite été infirmé : U4 est plat, sans minimum convergent. Cette réfutation est documentée dans `docs/CLAIMS_REGISTER.md:43` et `:77`.

La couche hardware a produit des prédictions falsifiables, mais pas une validation expérimentale. La comparaison SPICE/Python ne reproduit pas exactement le même modèle.

### 3.7 Couche récente : réduction du périmètre

Les expériences B5/B6/B7 réduisent fortement l’ambition applicative :

- pas de mémoire compétitive ;
- pas de prédiction générale ;
- pas d’optimisation ;
- pas d’économie énergétique ;
- bénéfice possible et étroit sur le moment de lecture ou d’arrêt dans une tâche synthétique trompeuse.

Cette réduction est probablement la lecture la plus honnête du projet, mais elle n’est pas encore reflétée partout.

---

## 4. Hypothèses principales et statut actuel

| Hypothèse | Évaluation |
|---|---|
| `u` module le couplage | Implémenté et vérifiable dans `dynamics.py:232-234`. Confirmé au niveau logiciel. |
| `u` dynamique anti-synchronise | Fortement soutenu par FULL/FROZEN_U, mais dépendant du protocole, du bruit et de la normalisation. |
| Les heretics empêchent le consensus | Seulement lorsque le stimulus est non nul ; à `I_stimulus=0`, le flip du stimulus est nul. |
| 15 % est universel | Réfuté. Le comportement dépend fortement de la topologie et de la normalisation. |
| λ₂≈2,31 est causal | Réfuté par `docs/audit_externe_neuromorphique_2026-07-06.md:27-38`. |
| Le degré harmonique est causal | Hypothèse actuelle plus plausible, mais sans théorie analytique. |
| Il existe une transition thermodynamique | Réfuté par Binder U4 plat. |
| Le système forme une chimère | Phénomène plausible, mais classification formelle à renforcer. |
| H_cont est une mesure cognitive | Non démontré ; c’est une entropie d’histogramme. |
| Le modèle est une mémoire | Réfuté par la comparaison ESN/NARMA10. |
| Le modèle prédit mieux | Réfuté sur Lorenz. |
| Le modèle optimise mieux | Réfuté sur Max-Cut. |
| Le modèle économise l’énergie | Réfuté contre un RC équivalent à substrat comparable. |
| Le hardware est validé | Non ; il existe des prédictions et POCs, pas une validation expérimentale. |

---

## 5. Erreurs, incohérences et fragilités

### 5.1 Contradictions scientifiques entre documents

`CONTEXT.md:11` affirme encore une dead zone spectrale causée par λ₂≈2,31. `PROJECT_STATUS.md:203-207` affirme explicitement le contraire.

`CONTEXT.md:36` conserve la transition événementielle `+1.20 bits`, alors que `PROJECT_STATUS.md:546` rapporte 0/9 configurations positives et un effet négatif.

`docs/compendium/BRIEF_COMPENDIUM.md:62-86` continue de présenter λ₂≈2,31 et `dH=+1.20` comme résultats centraux.

### 5.2 Sources de vérité externes au dépôt

Les fichiers `.brain/claims_mapping.json`, `.brain/preprint_guardian.py` et `.brain/tex_guardian.py` sont annoncés comme indispensables dans `PROJECT_STATUS.md:138-144`, mais ils ne sont pas versionnés.

Conséquence : les assertions « Guardian 20/20 », « Tex Guardian 15/15 » et « 12/12 sources » ne sont pas auditables par un clone indépendant.

### 5.3 Producteurs manquants ou non régénérables

Le registre `docs/CLAIMS_REGISTER.md:6-13` reconnaît lui-même que plusieurs claims avaient leur seul producteur dans `experiments/scratch/`. Ce dossier est gitignoré par `.gitignore:85`.

Le problème dépasse les CSV : un CSV exact au bit près ne prouve pas que le protocole qui l’a produit est celui annoncé dans le papier. Le registre le reconnaît explicitement pour la synchronie dans `docs/CLAIMS_REGISTER.md:37`.

### 5.4 Versions incohérentes

| Source | Version |
|---|---|
| `pyproject.toml:7` | 6.0.0 |
| `VERSION:1` | V6.0.0 |
| `src/mem4ristor/__init__.py:54` | 4.0.0 |
| `CITATION.cff:8` | 4.0.0 |
| `src/mem4ristor/config.yaml:1` | V4.0.0 |
| `Dockerfile:1` | v2.9.3 |

### 5.5 DOI incohérents

Les documents utilisent au moins les références suivantes :

- `10.5281/zenodo.19700749` dans `README.md:5` ;
- `10.5281/zenodo.18620596` dans `CITATION.cff:9` ;
- `10.5281/zenodo.19986042` dans `PROJECT_STATUS.md:78`.

La distinction « concept DOI » versus DOI de version devrait être explicitement normalisée dans un seul fichier de métadonnées.

### 5.6 Docker cassé

`Dockerfile:24` invoque `experiments/attack_resilience.py`, absent de l’arbre versionné. Le build peut réussir mais le conteneur échoue au démarrage.

### 5.7 Exemple cassé

`examples/demo_swarm.py:9` importe un module inexistant : `mem4ristor.mem4ristor_v3`.

### 5.8 Configuration typée incomplète

`Mem4Config` ne décrit que dynamics, coupling, doubt et noise dans `src/mem4ristor/config.py:85-88`. Les autres sections YAML sont ignorées par `from_dict()`.

### 5.9 Paramètres SPICE différents

`dynamics.py:60` fixe `social_leakage=0.01`, alors que `spice_art_kirchhoff.py:60` fixe `leak_delta=0.05`. Le script précise aussi que son ART différentiel ne reproduit pas le mécanisme multiplicatif actuel (`spice_art_kirchhoff.py:24-27`).

### 5.10 Chemin hardware non portable

`spice_art_kirchhoff.py:50` contient un chemin absolu propre à une machine Windows.

### 5.11 Nettoyage numérique silencieux

`dynamics.py:191-195` remplace NaN et Inf par des valeurs neutres. Cela protège contre les crashes mais peut invalider une expérience sans la faire échouer.

### 5.12 RK45 incohérent avec `step()`

`solve_rk45()` avertit que le bruit est incompatible avec un solveur adaptatif, mais continue malgré tout. Il ignore aussi plusieurs extensions du modèle actuel.

### 5.13 Validation de graphe incomplète

`Mem4Network` vérifie NaN/Inf, mais ne refuse pas au constructeur les matrices non carrées, asymétriques, pondérées négativement ou déconnectées.

### 5.14 Erreurs spectrales masquées

`topology.py:244-249` retourne `0.0` en cas d’exception du solveur sparse. Cela transforme une panne numérique en résultat scientifique plausible.

### 5.15 Générateurs sans validation

`graph_utils.py:14-45` ne vérifie pas les contraintes de `n` et `m`. `make_er()` ne vérifie pas `p`. Les cas pathologiques sont laissés aux erreurs internes de NumPy.

### 5.16 Métrique H_cont dépendante de choix arbitraires

`metrics.py:9-20` utilise 100 bins et une plage fixe [-3,3]. Les valeurs hors plage disparaissent du calcul. La métrique est donc comparable uniquement si le protocole complet est strictement identique.

### 5.17 Tests insuffisamment connectés aux claims

La suite teste surtout des invariants et des seuils larges. Elle ne garantit pas que les valeurs du preprint sont régénérées, ni que les scripts producteurs existent dans un clone propre.

### 5.18 Documentation de reproduction historique

`REPRODUCE_RESULTS.md:4` référence `WORK_LOG_PAPER.tex`, tandis que le dépôt actuel utilise `docs/papers/preprint/preprint.tex`. Il référence également plusieurs scripts sous `experiments/scratch/` qui ne sont pas suivis.

---

## 6. Corrections proposées

### Priorité documentation

1. Remplacer immédiatement `CONTEXT.md`, README et compendium par le cadrage actuel.
2. Ajouter un bandeau `RETRACTED` dans les scripts Binder, λ₂ et transition événementielle.
3. Retirer `docs/limitations.md` des sources de vérité ou le réécrire entièrement.
4. Corriger les chemins de preprint et de scripts dans `README.md` et `REPRODUCE_RESULTS.md`.

### Priorité reproductibilité

5. Versionner les Guardians.
6. Fournir un manifeste machine-readable pour chaque claim.
7. Ajouter une commande de reproduction par claim.
8. Réinclure ou remplacer tout producteur actuellement dépendant de `scratch/`.
9. Capturer commit, environnement, seed, paramètres et hash des sorties.

### Priorité packaging

10. Unifier la version sur `pyproject.toml`.
11. Corriger `__version__` dans `src/mem4ristor/__init__.py:54`.
12. Harmoniser `requirements.txt` avec `pyproject.toml`.
13. Corriger ou supprimer le Dockerfile.
14. Corriger `examples/demo_swarm.py:9`.

### Priorité noyau

15. Unifier `config.yaml`, `config.py` et les defaults internes.
16. Ajouter les dataclasses des extensions ou refuser explicitement les clés inconnues.
17. Ajouter un mode numérique strict.
18. Refuser les graphes invalides à la construction.
19. Propager les erreurs spectrales au lieu de retourner `0.0`.
20. Séparer clairement le sous-modèle RK45 du modèle SDE Euler-Maruyama.

### Priorité scientifique

21. Refaire FULL/FROZEN_U avec un protocole pré-enregistré et beaucoup plus de seeds.
22. Tester `u` dynamique contre un coupling répulsif fixe optimisé.
23. Permuter indépendamment amplitude, temporalité et localisation de `u`.
24. Remplacer les conclusions cognitives par une formulation de dynamique de réseau.
25. Dériver et tester le seuil en degré hors des familles BA/ER.
26. Formaliser la classification chimère.

---

## 7. Fiable versus à revalider

### Relativement fiable

- La formule de couplage adaptatif existe dans le code.
- Le noyau FHN fonctionne comme simulation numérique.
- FULL et FROZEN_U produisent des niveaux de synchronie très différents dans les protocoles étudiés.
- Les normalisations par degré changent fortement les régimes.
- La causalité λ₂ est réfutée par des contrôles internes.
- Binder U4 n’apporte pas de preuve de transition thermodynamique.
- Les limites mémoire, prédiction, optimisation et énergie sont documentées par des résultats négatifs.

### À revalider

- La taille exacte et la généralité de l’effet FULL/FROZEN_U.
- La causalité propre du feedback de `u`.
- Le seuil de degré harmonique.
- La désignation de chimère.
- La robustesse au bruit, au pas `dt`, au solveur et au warmup.
- Les comparaisons hardware.
- Le retard de récupération sur des seeds indépendants et des dispositifs non jumeaux.
- Toute affirmation d’utilité aval.

### À considérer comme retiré

- Dead zone spectrale causale.
- Binder comme transition de phase.
- Transition événementielle `+1.20 bits`.
- Universalité des 15 %.
- Économie énergétique.
- Mémoire ou prédiction générale.
- Supériorité en optimisation.
- Bénéfice topologique du signal d’arrêt.

---

## 8. Pistes non explorées et directions futures

### 8.1 Isolation causale de `u`

Expérience recommandée : comparer le feedback réel avec :

- `u` rejoué depuis un autre run ;
- `u` permuté entre nœuds ;
- `u` permuté temporellement ;
- `u` constant au niveau moyen ;
- coupling répulsif fixe ;
- coupling aléatoire de même distribution.

Cette expérience dira si l’effet vient réellement de l’adaptation ou simplement du couplage répulsif moyen.

### 8.2 Théorie du seuil de degré

Construire une approximation mean-field ou pair approximation reliant :

- degré local ;
- distribution de `u` ;
- couplage moyen ;
- stabilité transverse du consensus.

Le seuil `k_harm≈6` ne doit pas remplacer mécaniquement λ₂≈2,31 sans validation hors échantillon.

### 8.3 Classification chimère

Ajouter les diagnostics classiques : strength of incoherence, ordre local de Kuramoto, métastabilité, fraction cohérente/incohérente et durée de vie.

### 8.4 Validation hardware

Avant toute campagne :

- modéliser le canal de couplage électrique réel ;
- inclure bruit, mismatch, dérive et dispersion de fréquences ;
- définir un protocole à trois bras : feedback adaptatif, coupling fixe calibré, contrôle aveugle ;
- publier le modèle compact exact utilisé par SPICE.

### 8.5 Validation logicielle

Ajouter :

- convergence en `dt` ;
- comparaison Euler-Maruyama/Milstein ;
- reproduction des CSV en clone vierge ;
- tests de toutes les expériences du preprint ;
- vérification des chemins, imports et exemples.

---

## 9. Priorisation des actions

### P0 — Bloquant

1. Corriger les documents publics contradictoires.
2. Versionner les outils Guardian.
3. Rendre chaque claim régénérable depuis un clone propre.
4. Unifier version et DOI.
5. Refaire l’ablation centrale dans un environnement verrouillé.

### P1 — Scientifique

6. Isoler feedback adaptatif et coupling répulsif fixe.
7. Refaire les statistiques avec davantage de seeds.
8. Formaliser la chimère.
9. Tester le seuil de degré sur des graphes indépendants.
10. Remplacer les interprétations cognitives par des propriétés dynamiques mesurables.

### P2 — Logiciel

11. Unifier la configuration.
12. Ajouter mode numérique strict.
13. Valider les graphes.
14. Corriger Docker et les exemples.
15. Ajouter lint, typecheck et smoke tests CI.

### P3 — Restructuration

16. Séparer noyau, expériences courantes, explorations et hypothèses réfutées.
17. Générer automatiquement les tables et registres depuis des manifests.
18. Créer une release scientifique réellement reproductible.

### P4 — Exploration

19. Hardware réel et micromagnétisme.
20. Données ou tâches réelles.
21. Intégration LLM/transformer, uniquement après démonstration d’un bénéfice aval.

---

## Conclusion pour discussion d’équipe

Le projet ne doit plus être discuté comme une architecture cognitive générale ni comme une preuve de transition spectrale. La base défendable est :

> **Mem4ristor est un système FHN à couplage de polarité adaptative, capable de résister à la synchronisation dans des régimes spécifiques, avec une différence empirique nette entre doute dynamique et doute gelé.**

La prochaine étape prioritaire n’est pas d’ajouter une nouvelle extension. Elle consiste à démontrer que l’avantage vient bien du **feedback adaptatif** lui-même, et non d’un simple changement de couplage moyen, tout en rendant la chaîne complète de reproduction indépendante des fichiers privés et des artefacts historiques.
