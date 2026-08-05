# Analyse Exhaustive et Critique du Dépôt Mem4ristor

## 1. Synthèse Exécutive

Le projet Mem4ristor se présente comme une implémentation computationnelle de dynamiques FitzHugh-Nagumo étendues, visant à étudier les états critiques émergents dans des réseaux neuromorphiques. L'innovation centrale revendiquée est le "Doute Constitutionnel" ($u$), une variable dynamique modulant la polarité du couplage pour empêcher l'effondrement du consensus (la "dead zone").

Cependant, l'analyse approfondie du dépôt révèle un projet qui a évolué par couches successives, accumulant une dette technique et scientifique significative. Bien que le phénomène d'anti-synchronisation (ablation FROZEN_U vs FULL) semble robuste et reproductible, de nombreuses affirmations (claims) publiées reposent sur des bases fragiles, des artefacts méthodologiques ou des scripts non reproductibles. Le dépôt souffre d'une désynchronisation majeure entre sa documentation de surface (README, `RESULTS_INDEX.json`), qui maintient des hypothèses réfutées, et sa documentation interne (`PROJECT_STATUS.md`, audits), qui reconnaît ces failles.

Les problèmes les plus critiques incluent :
- **Une infrastructure de validation hors dépôt** : Le système "Guardian", garantissant la validité des claims, s'exécute localement sur la machine de l'auteur, rendant la reproductibilité impossible pour un tiers.
- **Des artefacts méthodologiques majeurs** : L'effet des "hérétiques" à stimulus nul est un artefact du générateur de nombres aléatoires (RNG), et non un phénomène physique.
- **Des divergences d'implémentation** : L'intégrateur RK45 (`solve_rk45`) implémente un modèle mathématiquement différent de l'intégrateur principal (`step()`), invalidant son usage comme outil de contrôle.
- **Une documentation obsolète** : Le README et d'autres fichiers clés mettent en avant des résultats (comme la transition spectrale $\lambda_2$) qui ont été explicitement réfutés par les mainteneurs eux-mêmes.

## 2. Cartographie du Dépôt

Le dépôt est structuré autour de plusieurs dossiers clés, mais cette structure masque une complexité historique :

- **`src/mem4ristor/`** : Le cœur du projet. Contrairement à ce que suggère une lecture superficielle, `core.py` n'est plus le moteur principal, mais une simple façade de compatibilité. La logique réelle est répartie entre `dynamics.py` (modèle FHN, doute, plasticité), `topology.py` (graphe, couplage, rewiring) et `metrics.py` (calculs d'entropie).
- **`experiments/`** : Contient les scripts de validation. Cependant, une grande partie des scripts producteurs de données (notamment pour les figures du README) se trouve dans `experiments/scratch/`, un dossier ignoré par Git, rendant ces résultats non reproductibles.
- **`tests/`** : Une suite de tests complète, mais dont l'exécution en intégration continue (CI) est cassée en raison de dépendances manquantes (`pyyaml`).
- **`docs/`** : Contient la documentation scientifique, y compris le preprint, l'historique académique et les audits. C'est ici que se trouvent les documents les plus honnêtes (`PROJECT_STATUS.md`, `CLAIMS_REGISTER.md`), qui contredisent souvent la documentation racine.

## 3. Reconstruction Historique des Couches du Projet

Le projet a traversé plusieurs phases conceptuelles, laissant des traces parfois contradictoires :

1. **Phase Initiale (V1-V2)** : Focalisation sur la préservation de la diversité via le doute. Le modèle souffrait d'un effondrement du rapport signal/bruit (SNR) près de $u=0.5$ (la "Dead Zone").
2. **Phase de Stabilisation (V3)** : Introduction de la "Levitating Sigmoid" pour corriger le problème de la Dead Zone. C'est la couche la plus stable du projet.
3. **Phase Topologique (V4)** : Tentative de résolution de "l'étranglement topologique" sur les réseaux scale-free via un recâblage dynamique. Cette approche a échoué et a été remplacée par une normalisation linéaire par degré (`degree_linear`), bien que le code de recâblage subsiste.
4. **Phase Spectrale (V5-V6)** : Hypothèse selon laquelle la valeur de Fiedler ($\lambda_2$) expliquait la transition de phase. Cette hypothèse a été **totalement réfutée** (juillet 2026), mais reste présente dans la documentation de surface.
5. **Phase Actuelle (V6.0.0)** : Recentrage sur le degré de couplage et le champ moyen comme mécanismes explicatifs. Le projet est en phase de correction, mais la dette documentaire reste massive.

## 4. Hypothèses Principales et leur Statut Actuel

| Hypothèse | Statut | Commentaire |
| :--- | :--- | :--- |
| **Le doute ($u$) préserve la diversité (FROZEN_U vs FULL)** | ✅ **Fiable** | Résultat le plus robuste (Cohen's d $\approx$ 9.4). |
| **La transition spectrale ($\lambda_2$) explique la dynamique** | ❌ **Réfutée** | Réfutée en juillet 2026. Le mécanisme réel est lié au degré de couplage. |
| **Les hérétiques brisent le consensus à stimulus nul** | ❌ **Artefact** | L'effet observé est dû à un décalage du RNG, pas à une dynamique physique. |
| **La normalisation `degree_linear` est universelle** | ⚠️ **Partielle** | Fonctionne sur BA m=3, mais échoue sur d'autres topologies (ex: BA m=5). |
| **L'attracteur d'entropie $H \approx 1.94$** | ❌ **Faux** | Valeur transitoire. L'entropie stable est beaucoup plus basse avec les métriques actuelles. |

## 5. Erreurs, Incohérences et Fragilités Identifiées

### 5.1. Artefact Méthodologique sur les Hérétiques
L'affirmation selon laquelle les nœuds "hérétiques" agissent comme des murs structurels à stimulus nul ($I_{stimulus}=0$) est mathématiquement vide. L'équation `I_eff[heretic_mask] *= -1.0` n'a aucun effet si `I_eff` est nul. L'écart observé dans les expériences est uniquement dû à la consommation de tirages aléatoires lors du placement des hérétiques, ce qui décale la séquence de bruit. Cela invalide toutes les expériences A/B sur le ratio d'hérétiques.

### 5.2. Divergence des Intégrateurs
La méthode `solve_rk45` dans `dynamics.py` calcule le Laplacien différemment de la méthode principale `step()` (`adj_matrix @ v - v` au lieu de `adj_matrix @ v - adj_matrix.sum(axis=1) * v`). De plus, elle injecte le bruit de manière incorrecte pour une équation différentielle stochastique. Toute comparaison entre Euler et RK45 dans ce dépôt compare en réalité deux modèles physiques différents.

### 5.3. Infrastructure de Validation Fantôme
Le système "Guardian", garantissant que 20/20 claims sont vérifiés, repose sur des scripts situés dans un dossier `.brain/` non versionné, avec des chemins absolus pointant vers une machine Windows spécifique. Cela rend le projet invérifiable par des pairs.

### 5.4. Scripts Producteurs Non Versionnés
De nombreux résultats publiés (y compris 7 claims sur 18) dépendent de scripts situés dans `experiments/scratch/`, un dossier ignoré par Git. Ces résultats ne peuvent pas être reproduits à partir du dépôt cloné.

## 6. Corrections Proposées

1. **Découplage des RNG** : Dans `dynamics.py`, séparer le générateur aléatoire pour la topologie de celui pour le bruit (`self.rng_topo` vs `self.rng`). Cela corrigera l'artefact des hérétiques.
2. **Correction de l'Intégrateur RK45** : Aligner le calcul du Laplacien dans `solve_rk45` sur celui de `step()` et interdire son utilisation avec un bruit non nul (`sigma_v > 0`).
3. **Versionnement de l'Infrastructure** : Déplacer les scripts du Guardian (`.brain/`) dans un dossier versionné (ex: `tools/guardian/`) et utiliser des chemins relatifs.
4. **Nettoyage Documentaire** : Mettre à jour le README, `RESULTS_INDEX.json` et `CITATION.cff` pour refléter les conclusions de `PROJECT_STATUS.md` (abandon de l'hypothèse spectrale, correction des métadonnées).
5. **Réparation de la CI** : Ajouter `pyyaml` aux dépendances de test et configurer correctement les workflows GitHub Actions.

## 7. Éléments Fiables versus Éléments à Revalider

**Fiable :**
- L'effet du doute ($u$) sur la préservation de la diversité (ablation FROZEN_U vs FULL).
- La correction de la Dead Zone via la "Levitating Sigmoid".
- L'implémentation des métriques d'entropie continue (H_cont).

**À Revalider :**
- Toutes les expériences impliquant des variations du ratio d'hérétiques (en raison de l'artefact RNG).
- La stabilité à long terme sur des réseaux très larges ($N > 2500$).
- L'efficacité de la normalisation `degree_linear` sur des topologies non testées.

## 8. Pistes Non Explorées et Directions Futures

- **Normalisation Adaptative** : Puisque `degree_linear` n'est pas universelle, explorer une normalisation hybride basée sur le coefficient de clustering local.
- **Validation Matérielle** : Les pistes photoniques et spintroniques (mentionnées dans `docs/hardware/`) semblent prometteuses mais nécessitent une validation au-delà de la simple simulation.
- **Analyse de la Redondance des Chemins** : L'échec de la normalisation sur certaines topologies suggère que le problème est lié au transport d'information (redondance des chemins) plutôt qu'à la simple hétérogénéité des degrés.

## 9. Priorisation des Actions Recommandées

1. **Critique (Immédiat)** : Découpler les RNG dans `dynamics.py` pour corriger l'artefact méthodologique majeur.
2. **Critique (Immédiat)** : Versionner l'infrastructure Guardian et les scripts producteurs de `scratch/` pour restaurer la reproductibilité.
3. **Important (Court terme)** : Aligner la documentation de surface (README, `RESULTS_INDEX.json`) sur la réalité scientifique reconnue dans `PROJECT_STATUS.md`.
4. **Important (Court terme)** : Corriger l'intégrateur RK45 et réparer la CI.
5. **Exploratoire (Moyen terme)** : Rejouer toutes les expériences A/B avec les RNG corrigés et investiguer une normalisation de couplage adaptative.
