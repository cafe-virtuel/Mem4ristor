# Audit Scientifique du Projet Mem4ristor-v2

**Date :** 24 Avril 2026
**Branche :** `feat/kimi-p419-continuous-entropy`
**Auteur :** Chercheur en systèmes neuromorphiques et dynamiques complexes

Ce rapport présente un audit approfondi du dépôt Mem4ristor-v2, en se concentrant sur la cohérence entre les claims du preprint et le code, l'identification des items oubliés, la proposition de nouvelles pistes de recherche, et une critique externe de type reviewer.

---

## 1. Audit de Cohérence

L'analyse croisée du preprint (`docs/preprint.tex`), du document pivot (`PROJECT_STATUS.md`), et du code source (`src/`, `experiments/`, `tests/`) révèle plusieurs incohérences majeures et des décalages entre l'état de l'art du dépôt et le manuscrit.

### 1.1. Claims du preprint vs Expériences
- **Le rôle des nœuds hérétiques (Claim vacuité) :** Le preprint affirme que les nœuds hérétiques (recevant un stimulus inversé) préviennent l'effondrement du consensus. Cependant, comme documenté dans `PROJECT_STATUS.md` (§3octvicies, FLAW 6) et vérifié dans `experiments/limit02_alpha_sweep.py` et `experiments/ablation_minimality.py`, les expériences "endogènes" utilisent `I_stimulus = 0.0`. Dans `core.py`, l'inversion de polarité `I_eff[heretic_mask] *= -1.0` est une opération nulle (no-op) quand `I_eff = 0`. Ainsi, les expériences endogènes du preprint **ne testent pas** le mécanisme hérétique. Le claim selon lequel les hérétiques sont nécessaires dans ce régime est empiriquement faux dans le code actuel.
- **L'exposant optimal $\gamma^*(m)$ (Claim 4) :** Le preprint affirme avoir identifié un exposant optimal pour la normalisation en loi de puissance sur les réseaux scale-free. Le script `experiments/limit02_alpha_sweep.py` supporte cette affirmation, mais il utilise l'ancienne métrique d'entropie par défaut (`calculate_entropy()`) au lieu de la métrique cognitive stricte (`calculate_cognitive_entropy()`) introduite suite à l'audit KIMI. De plus, ce sweep est réalisé avec `I_stimulus = 0.0`, rendant les hérétiques inactifs.
- **La transition de phase $\lambda_2$ :** Le preprint affirme qu'une transition de phase abrupte se produit autour de $\lambda_2 \approx 2-3$ (pour BA $m \approx 5$). Le script `experiments/p2_edge_betweenness_analysis.py` confirme que $\lambda_2$ est un excellent prédicteur (r=+0.901) de la "dead zone". Cependant, `experiments/p2_finite_size_scaling.py` (détaillé dans `PROJECT_STATUS.md` §3duovigies) montre que la dead zone **disparaît** à plus grande échelle ($N=1600$) avec la normalisation `degree_linear`. Le preprint présente la dead zone comme une limite fondamentale, alors que les expériences récentes suggèrent qu'il s'agit d'un artefact de taille finie ou de normalisation.

### 1.2. Résultats dans PROJECT_STATUS non reflétés dans le preprint
- **Métriques de coordination trajectorielle :** `PROJECT_STATUS.md` (§3novedecies, §3vigies) et `experiments/ablation_coordination.py` introduisent des métriques temporelles cruciales (complexité de Lempel-Ziv, synchronie par paires) pour distinguer la "diversité cognitive structurée" du "désordre aléatoire". Le preprint s'appuie toujours uniquement sur l'entropie spatiale marginale (snapshot entropy), qui a été démontrée comme confondant ces deux états.
- **Bimodalité endogène :** L'expérience `experiments/bimodality_50seeds.py` confirme une bimodalité statistique forte (Hartigan dip p=0.000) dans le régime endogène. Ce résultat fondamental sur la nature multi-attracteur du système est absent du preprint.
- **Le terme de decay de plasticité :** Comme noté dans l'audit KIMI (FLAW 7), l'équation 2 du preprint omet le terme de decay constant $-w_i/\tau_{\text{plast}}$ présent dans `core.py`. Ce terme modifie le point fixe du système.

### 1.3. Couverture des tests
- Le fichier `tests/test_scientific_regression.py` couvre bien les invariants de base (entropie > 0 sur lattice, effondrement sans hérétiques, sauvetage par `degree_linear` sur BA m=3).
- **Lacunes critiques :** Aucun test ne vérifie la transition de phase $\lambda_2$, ni la disparition de la dead zone à grand $N$. Surtout, aucun test ne protège contre le bug `I_stimulus = 0.0` (les tests de nécessité des hérétiques utilisent `I_stimulus = 0.5`, masquant le problème du régime endogène). Les nouvelles métriques de coordination sont testées unitairement (`test_coordination_metrics.py`), mais aucun test d'intégration ne garantit que le modèle complet maintient une faible synchronie et une faible complexité LZ.

---

## 2. Items Oubliés et TODOs Non Trackés

L'exploration du code révèle plusieurs éléments expérimentaux ou inachevés qui semblent flotter sans conclusion claire dans la documentation principale.

### 2.1. Modules Expérimentaux Abandonnés
- **`experimental/mem4ristor_king.py` (The Philosopher King) :** Ce module introduit des concepts fascinants ("Martial Law", "Metacognitive Metabolism") basés sur la frustration et l'ennui. Bien qu'il soit mentionné dans `PROJECT_STATUS.md` comme "EXPERIMENTAL", il n'y a aucune trace d'évaluation de ses performances ou de décision quant à son intégration future. Le code contient un avertissement clair sur des bugs connus (mutation d'état fragile, absence de couplage social en mode standalone) datant de février 2026, suggérant un abandon de fait.
- **`src/mem4ristor/hierarchy.py` (Hierarchical Chimera) :** Ce module tente de construire une architecture V1 -> V4 -> PFC en utilisant le `Mem4ristorKing`. Étant donné que le King est buggé et abandonné, cette hiérarchie est structurellement compromise. Elle n'est pas testée ni mentionnée dans les expériences récentes.
- **`src/mem4ristor/arena.py` (GladiatorMem4ristor) :** Introduit un concept de "Pain Learning" (Predator vs Prey). C'est une déviation majeure de la dynamique de consensus/frustration vers l'apprentissage par renforcement compétitif. Aucune expérience ne l'utilise.
- **`src/mem4ristor/cortex.py` et `symbiosis.py` :** Ces modules implémentent un auto-encodeur (LearnableCortex) et un projecteur créatif (CreativeProjector) pour une phase de "rêve". Bien que testés unitairement (`test_symbiosis_swarm.py`), ils ne sont intégrés dans aucun pipeline d'évaluation scientifique (contrairement au `SensoryFrontend` utilisé dans `demo_applied.py`).

### 2.2. Résultats Négatifs Mal Documentés
- **L'échec de la normalisation spectrale :** `PROJECT_STATUS.md` (§3octies) mentionne que `coupling_norm='spectral'` échoue à briser la dead zone (0/6 wins). Le script `experiments/spectral_normalization_test.py` existe, mais ce résultat négatif (pourtant qualifié de "publiable") n'est pas intégré au preprint, qui se concentre uniquement sur la normalisation par degré.
- **Le bug de l'intégrateur RK45 :** Le FLAW 5 de l'audit KIMI note que `solve_rk45` injecte du bruit à chaque évaluation du RHS, rendant les résultats non reproductibles si $\sigma_v > 0$. Un avertissement a été ajouté dans `core.py`, mais les implications sur les expériences passées qui auraient pu utiliser RK45 avec du bruit ne sont pas discutées.

---

## 3. Nouvelles Pistes de Recherche

À partir de l'état actuel du code et des impasses identifiées (notamment la "dead zone" et la confusion entre diversité et désordre), voici 5 pistes de recherche inédites, classées par faisabilité (Effort × Impact).

### Piste 1 : Résonance Stochastique Dirigée (Impact : Fort, Effort : Faible)
- **Hypothèse :** La "dead zone" sur les réseaux denses (BA $m \ge 5$) n'est pas un puits attracteur absolu, mais un état métastable profond. Un bruit thermique ciblé uniquement sur les nœuds hérétiques (ou les hubs) peut induire une résonance stochastique suffisante pour faire basculer le réseau hors du consensus, sans noyer le signal global.
- **Expérience :** Modifier `core.py` pour appliquer $\sigma_v$ de manière hétérogène (ex: $\sigma_{v,i} \propto \text{deg}(i)$ ou $\sigma_{v,i} > 0$ uniquement si `heretic_mask[i] == True`).
- **Métrique :** Mesurer $H_{\text{stable}}$ et la complexité LZ temporelle en balayant l'amplitude du bruit hétérogène sur BA $m=5$.
- **Pourquoi c'est nouveau :** Les expériences actuelles (`spice_noise_resonance.py`) appliquent un bruit global homogène. Un bruit dirigé exploite la topologie pour maximiser l'impact de la frustration.

### Piste 2 : Dynamique Temporelle du Doute ($\tau_u$) et Bifurcation (Impact : Fort, Effort : Moyen)
- **Hypothèse :** La constante de temps du doute $\tau_u$ contrôle une bifurcation entre un régime de "frustration figée" (où les nœuds se bloquent en opposition) et un régime de "chimère respirante" (où les clusters de consensus se font et se défont dynamiquement).
- **Expérience :** Balayer $\tau_u \in [0.1, 100.0]$ sur un réseau régulier (Lattice) et un réseau scale-free (BA $m=3$).
- **Métrique :** Utiliser `calculate_pairwise_synchrony` et analyser le spectre de Fourier moyen des séries temporelles $v(t)$. On cherche un pic de fréquence caractéristique émergeant à une valeur critique de $\tau_u$.
- **Pourquoi c'est nouveau :** Le projet s'est concentré sur les paramètres de couplage ($D$, normalisation) et la topologie. La dynamique temporelle de la variable de contrôle $u$ est sous-explorée, alors qu'elle est le moteur de l'anti-synchronisation.

### Piste 3 : Couplage Asymétrique et Graphes Dirigés (Impact : Très Fort, Effort : Moyen)
- **Hypothèse :** La "strangulation par les hubs" dans les réseaux scale-free est exacerbée par la symétrie du couplage (le hub influence autant la périphérie que l'inverse). Un réseau dirigé où l'influence (les poids de l'arête) est asymétrique (ex: les hubs écoutent moins qu'ils ne parlent, ou l'inverse) éliminera la dead zone sans nécessiter de normalisation ad-hoc comme `degree_linear`.
- **Expérience :** Générer des graphes BA dirigés. Modifier `Mem4Network` pour accepter des matrices d'adjacence asymétriques. Comparer avec les résultats de `limit02_topology_sweep.py`.
- **Métrique :** $H_{\text{stable}}$ et $\lambda_2$ (calculé sur le Laplacien dirigé).
- **Pourquoi c'est nouveau :** Toutes les topologies testées jusqu'à présent sont non-dirigées. Dans les systèmes neuromorphiques réels, les synapses sont directionnelles.

### Piste 4 : Information Mutuelle Spatio-Temporelle (Impact : Moyen, Effort : Moyen)
- **Hypothèse :** L'entropie marginale $H_{\text{stable}}$ ne capture pas la richesse spatiale. Un réseau avec $H \approx 1.0$ peut être un damier statique ou un système où l'information se propage. L'Information Mutuelle (MI) moyenne entre nœuds voisins vs nœuds distants révélera une longueur de corrélation caractéristique du régime "Mem4ristor".
- **Expérience :** Implémenter une fonction `calculate_spatial_mutual_information(v_history, adjacency_matrix)` dans `metrics.py`. L'évaluer sur les 4 ablations de `ablation_coordination.py`.
- **Métrique :** Décroissance de la MI en fonction de la distance sur le graphe.
- **Pourquoi c'est nouveau :** Cela comble le vide entre l'entropie purement spatiale (snapshot) et la complexité purement temporelle (LZ), en mesurant la structure spatio-temporelle.

### Piste 5 : L'Impact du Paramètre $\delta$ de la Levitating Sigmoid (Impact : Faible, Effort : Faible)
- **Hypothèse :** Le paramètre $\delta = 0.01$ dans $w_i(u_i) = \tanh(\pi(0.5 - u_i)) + \delta$ brise la symétrie parfaite au point de doute maximal ($u=0.5$). Augmenter $\delta$ favorise le consensus, tandis qu'un $\delta$ négatif favorise la divergence. Il existe un $\delta_{crit}$ qui maximise la complexité LZ.
- **Expérience :** Balayer $\delta \in [-0.1, 0.1]$ sur un Lattice 10x10.
- **Métrique :** `calculate_temporal_lz_complexity`.
- **Pourquoi c'est nouveau :** $\delta$ a été introduit comme un "fix" technique (LIMIT-01) pour éviter un couplage nul, mais son impact en tant que paramètre de contrôle de la symétrie sociale n'a jamais été quantifié.

---

## 4. Critique Externe (Perspective Reviewer)

Si je devais évaluer le preprint actuel (`docs/preprint.tex`) pour une conférence ou un journal en physique statistique ou systèmes complexes, voici les 3 arguments majeurs qui motiveraient un **rejet (ou une révision majeure)**, ainsi que les réponses attendues des auteurs.

### Argument 1 : Le mécanisme central (Hérétiques) n'est pas testé dans le régime principal
**Critique :** Le papier présente les "nœuds hérétiques" (fraction $\eta$) comme l'un des piliers empêchant l'effondrement du consensus (Section 2.3). Cependant, l'équation (1) montre que l'influence hérétique est implémentée par une inversion du stimulus externe ($I_{\text{ext},i} = -I_{\text{stimulus}} + I_{\text{coupling},i}$). Or, la majorité des expériences démontrant la diversité (ex: Table 2, sweeps topologiques) sont réalisées dans un régime endogène où $I_{\text{stimulus}} = 0$. Dans ce régime, l'inversion de polarité est mathématiquement nulle. Les auteurs attribuent la diversité à un mécanisme qui est littéralement inactif dans leur code lors de ces simulations.
**Réponse attendue des auteurs :** Les auteurs doivent (1) reconnaître cette erreur méthodologique, (2) requalifier les hérétiques comme un mécanisme de "réponse au forçage" plutôt que de diversité endogène, et (3) inclure les résultats de l'étude d'ablation (`ablation_minimality.py`) qui prouve que sous forçage ($I_{\text{stimulus}} > 0$), les hérétiques sont effectivement cruciaux pour maintenir la diversité. Le papier doit clairement séparer le régime endogène (maintenu par le bruit et la dynamique de $u$) du régime forcé.

### Argument 2 : Utilisation d'une métrique inadéquate (Entropie Marginale)
**Critique :** Le papier s'appuie presque exclusivement sur l'entropie de Shannon marginale ($H_{\text{stable}}$) calculée sur des snapshots spatiaux pour quantifier la "diversité des attracteurs". Comme le montre la littérature sur les systèmes complexes, une entropie spatiale élevée peut simplement indiquer un désordre aléatoire décorrélé (bruit) plutôt qu'une véritable diversité cognitive structurée. Le modèle pourrait simplement agir comme un amplificateur de bruit.
**Réponse attendue des auteurs :** Les auteurs doivent intégrer les métriques trajectorielles déjà développées dans leur dépôt (`metrics.py`), notamment la complexité de Lempel-Ziv temporelle et la synchronie par paires. Ils doivent inclure la figure générée par `experiments/phase_space_coordination.py` démontrant que seul le modèle complet (FULL) se situe dans le quadrant "basse complexité LZ (structuré) / faible synchronie (diversifié)", prouvant ainsi que le système génère une diversité coordonnée et non du bruit aléatoire.

### Argument 3 : La "Dead Zone" présentée comme fondamentale est un artefact de taille finie
**Critique :** L'Abstract et la Section 3.2 affirment l'existence d'une "transition de phase abrupte" vers une "dead zone" pour les réseaux denses (BA $m \ge 5$, $\lambda_2 \approx 2-3$), suggérant une limite fondamentale du modèle face à la redondance des chemins. Cependant, cette conclusion est tirée de simulations sur de très petits réseaux ($N=100$). Dans les réseaux scale-free, les effets de taille finie sont massifs. Sans analyse de *finite-size scaling*, affirmer l'existence d'une transition de phase topologique est prématuré et potentiellement faux.
**Réponse attendue des auteurs :** Les auteurs doivent inclure les résultats de `experiments/p2_finite_size_scaling.py` (qui teste $N=100, 400, 1600$). Si, comme le suggère le `PROJECT_STATUS.md`, la dead zone disparaît à $N=1600$ avec la normalisation `degree_linear`, le narratif du papier doit pivoter. Au lieu de présenter la dead zone comme une limite absolue, le papier doit discuter comment la normalisation par degré modifie l'échelle à laquelle la synchronisation induite par les hubs se produit, et comment $\lambda_2$ caractérise la rigidité spectrale à taille finie.
