# BILAN — Forces et faiblesses de MEM4RISTOR

> **Créé le 31 juillet 2026**, à la demande de Julien.
> **À quoi sert ce fichier** : chaque session rapporte ce qu'elle a *retranché*, et personne
> ne tient le **solde**. Six semaines de comptes-rendus de pertes, jamais d'inventaire du
> stock. Ce document existe pour corriger ça. Il est écrit **en français simple** et volontairement
> vulgarisé : il doit être lisible par quelqu'un qui n'a pas suivi les sessions.
>
> **Règle de tenue** : à chaque session qui déplace une ligne, on met la ligne à jour *ici aussi*.
> Un bilan périmé serait pire qu'aucun bilan — il donnerait l'illusion d'une vue d'ensemble.
> Toute affirmation doit porter **un chiffre et sa source**. Pas de « globalement », pas « il semble ».

---

## 0. La règle de lecture — deux colonnes, ne jamais les confondre

Il y a **deux revendications distinctes** dans ce projet *(carte du 26/07/2026, validée par Julien)* :

| | Quoi | Statut |
|---|---|---|
| **A — Le papier** | Ce que le doute fait à la **dynamique** d'un réseau | Publié, stable depuis le 06/07/2026 |
| **B — L'usage** | À quoi le doute sert pour **calculer** | Jamais publié, a rétréci **six fois** |

> **Fait à redire quand la sensation « le projet fond » revient** : **aucun des six
> rétrécissements n'a touché un seul chiffre du papier.** Ce qui est tombé est un
> argumentaire construit *à côté*.

---

## 1. Ce qui tient — colonne A

### 1.1 Le résultat central, et il est solide

Si on empêche les neurones de douter (on fige la variable `u`), le réseau se met à tout faire
pareil : la corrélation moyenne entre neurones passe de **0,007 à 0,658**. On va de « chacun
fait sa vie » à « tout le monde bat la même mesure ».

**Ce que ça vaut, concrètement** : sur **30 tirages** différents, les deux groupes de mesures
**ne se chevauchent jamais**. C'est le sens de « Cohen d ≈ 9 » — en statistique, au-delà de 0,8
on parle déjà d'un gros effet ; neuf, c'est une séparation totale.
Autre atout : la mesure est une **corrélation**, donc elle ne dépend d'aucun réglage de
découpage en classes — c'est le résultat **le moins attaquable** du papier.

📁 `figures/b4_ablation_summary.csv` (Cohen d = 9,3796) · `experiments/b4_ablation_robustness.py` · claim **C18**

### 1.2 Ce n'est pas une impression : c'est re-testé automatiquement

**18 chiffres publiés** sont recalculés depuis leurs données à **chaque commit** et comparés au
papier — **18/18 au 31/07/2026**. Depuis le 30/07, un **second** contrôle compare le *texte publié*
à ses sources (13 ancrages, registre des valeurs remplacées, audit des scripts cités) ; il est
**bloquant** depuis le 31/07. Très peu de laboratoires ont cet appareil.

📁 `.brain/preprint_guardian.py` · `.brain/tex_guardian.py` · hook `pre-commit`

### 1.3 Le projet s'est réfuté lui-même sur son idée d'origine — et l'a publié

L'hypothèse de départ voulait que la frontière soit gouvernée par **λ₂ ≈ 2,31** (une propriété
mathématique du graphe). Mesuré : **faux**. La cause réelle est le **nombre de voisins couplés**
(degré harmonique ≈ 6). Le papier le dit désormais explicitement, et le 2,31 y est requalifié en
frontière **corrélationnelle**, pas causale.

*Pourquoi c'est une force et non une faiblesse* : un projet qui publie la réfutation de sa propre
hypothèse initiale est plus crédible qu'un projet qui n'en trouve jamais.

### 1.4 La prédiction falsifiable — la force la plus sous-estimée du projet

C'est **la seule chose qui sort de l'ordinateur**. Tout le reste est auto-référentiel (on mesure
des grandeurs calculées sur la simulation qui les produit). Ici, on propose une expérience qu'un
laboratoire peut faire **pour donner tort au projet** : prendre de vrais oscillateurs magnétiques,
moduler leur couplage par le désaccord local, et mesurer s'ils se synchronisent moins qu'un
réseau à couplage fixe.

**Ce qui l'appuie** : **trois modèles physiques indépendants** convergent (Kuramoto, l'auto-oscillateur
Slavin-Tiberkevich, et le macrospin vectoriel complet), avec des effets de **1,05 à 14,85**.
La mesure se fait par **spectroscopie micro-onde standard** — la méthode déjà employée par
Romera et al. (2018).

**Deuxième volet, au signe inversé et tout aussi testable** : après un leurre transitoire, le
couplage modulé **retarde** la récupération d'environ **+52 %** par rapport au couplage figé.
Un laboratoire qui mesurerait une récupération plus *rapide* réfuterait ce volet.

**Ce que « capteur brut » veut dire** : le circuit qui mesure le désaccord entre voisins peut être
pris *tel quel* (gain = 1) ou *amplifié* (calibré). C'est la question qui décide si un laboratoire
peut tester la prédiction sans ajouter de composant.

| Modèle | Capteur **brut** (tel quel) | Capteur calibré |
|---|---|---|
| Kuramoto (§7) | **+2,28** (BA) / **+1,05** (lattice) | +14,85 / +4,83 |
| Slavin-Tiberkevich (§8) | **nul** (0,01 à 0,09) | +4,41 à +5,49 |
| Macrospin LLGS complet (§9) | **+2,42** (lattice) / **+1,61** (BA) | +3,22 / +3,36 |

**Au capteur brut, l'effet tient dans deux modèles sur trois**, et aucun intervalle de confiance ne
touche zéro dans ces deux-là. Le modèle §9 est **le plus direct des trois** (vraie équation
vectorielle, aucune reformulation), et c'est l'un de ceux qui tiennent.

⚠️ **Les réserves qui restent, elles, sont réelles** :
1. L'effet est **nul au brut dans le modèle §8** — précisément celui qui ajoute la
   **non-isochronicité**, la signature la plus caractéristique des vrais oscillateurs. On ne sait
   pas encore pourquoi ce modèle-là décroche.
2. Le **canal de couplage électrique réel** (celui de Romera) **n'a jamais été modélisé**. La
   géométrie testée verrouille en **antiphase**, alors que la littérature rapporte plutôt un
   verrouillage **en phase** sur les vrais réseaux couplés électriquement. Avant toute campagne
   expérimentale, il faut savoir quelle grandeur observer.
3. Second volet de la prédiction, au signe inversé : la récupération après un leurre est
   **retardée d'environ +52 %**. À ne pas vendre comme « le doute améliore les décisions ».

> 🔧 **Errata du 31/07/2026** — la première version de ce fichier, publiée le matin même, écrivait
> *« au capteur brut, l'effet est nul dans les trois modèles »*. **C'est faux** : il est nul dans
> **un** modèle sur trois. L'erreur venait de `FUTURE_WORK.md` §B6, qui disait « nul dans les deux
> modèles » — une phrase écrite au moment du §8 et qui était **déjà fausse** pour le §7 (d = +2,28
> au brut). Elle a été recopiée, puis élargie à « trois » quand un troisième modèle est arrivé.
> *Une affirmation sans son chiffre se propage et grossit ;* c'est exactement le motif que ce projet
> traque depuis le 29/07, et la règle §6 de ce fichier — un chiffre par affirmation — existait
> précisément pour l'empêcher. Elle n'a pas été appliquée à cette ligne-là.

📁 `docs/hardware/SPINTRONIC_PATHWAY.md` §8-9 · `docs/FUTURE_WORK.md` §B6

### 1.5 Ce qui a été *ajouté* récemment (29/07/2026)

Le mécanisme s'est révélé **plus structuré que sa propre description** : **deux seuils** au lieu
d'un (couper l'attraction désynchronise ; rendre franchement répulsif structure les trajectoires),
et une **bande de ré-synchronisation** que personne ne savait là. Répliqué au centième.

**Conséquence d'ingénierie, positive** : l'anti-synchronisation pourrait être obtenue par un
**couplage répulsif fixe** — bien plus simple à fabriquer qu'une variable adaptative par nœud.
⚠️ Réserve : un couplage fixe suppose de **connaître le bon niveau à l'avance**, alors que `u`
s'y établit seul.

---

## 2. Ce qui est tombé — colonne B

| Ce qu'on espérait | Ce qui est mesuré | Source |
|---|---|---|
| C'est une **mémoire** | Perd **5,5×** contre un réservoir standard | B5, 08/07 |
| Ça **prédit** | Sur Lorenz : erreur **7,72** contre **0,17** pour un filtre ordinaire — **44× pire**. 3ᵉ réplication | `p15`, 27/07 |
| Ça **optimise** | Au Max-Cut, tirer **300 fois au hasard bat M4R** (91,5 contre 80,9), 10/10 puis 10/10 graines neuves | `p15b`, 27/07 |
| Ça **explore mieux** | Le réseau ne visite que **~24 configurations** sur 300 lectures. Il est quasi immobile | `p15c`, 28/07 |
| Le doute bat un **horizon fixe** | Non : réservoir à budget fixe **1,00** contre doute **0,90** | B5b, 27/07 |
| **Lire la topologie coûte** | **Mort** — n'a pas répliqué (0 fois sur 2) | B7, 26/07 |

**Ce qui RESTE en colonne B, et c'est étroit mais réel** : le doute sait **quand trancher** quand
converger tôt mène à la mauvaise réponse. Sur cette tâche précise, il obtient **0,83** contre
**0,25** pour une règle de convergence classique.
⚠️ La niche exige **trois conditions à la fois** : un piège où converger tôt est faux, un horizon
**inconnu**, et un coût d'attente. Retirez-en une, l'avantage disparaît.

📁 `experiments/deceptive_task_poc.py` · `docs/FUTURE_WORK.md` §B1d

---

## 3. La réponse directe à l'ambition initiale

> *« Le prochain dipôle incontournable : des économies d'énergie énormes, et donner une direction
> aux processeurs en évitant les calculs inutiles. »*

Ce sont **deux ambitions distinctes**, et elles n'ont pas la même réponse.

### 3.1 ⛔ L'énergie : non, et c'est structurel

Mesuré le 26/07/2026. L'adversaire qui égale M4R sur sa niche est un **filtre à oubli** —
c'est-à-dire **un circuit RC**, une résistance et un condensateur. Passer M4R en analogique fait
passer l'adversaire aussi, **où il est passif**.

**À substrat égal, M4R coûte environ 15× plus d'énergie, et ce rapport ne dépend pas du
composant choisi** : changer de technologie divise les deux côtés par le même facteur. M4R
entretient **200 oscillateurs actifs** pendant 309 pas ; le filtre, **un seul circuit passif**
pendant 1348.

**Conclusion** : le « bon marché » du projet était un argument sur **le substrat** (l'analogique
en général), dont n'importe quel circuit analogique profite identiquement — **pas sur
l'architecture M4R**.

⚠️ **Le chiffre flatteur existe, et il faut savoir pourquoi on ne s'en sert pas** : comparé
autrement, on trouve **2 500× à 8 700×** en faveur de M4R. Il est calculé dans le script **et
affiché comme le piège qu'il est** — il compare deux *technologies*, pas deux *méthodes*.

📁 `experiments/expB3_substrate_crossover_poc.py` · `docs/hardware/B3_ENERGY_COMPARISON.md`

### 3.2 ✅ Éviter les calculs inutiles : oui, mais étroitement

Là, il y a quelque chose de réel, et c'est la **latence**. Ce que le doute apporte n'est pas de
**mieux** lire — à un instant donné il plafonne à 0,40, c'est médiocre. C'est de savoir **quand**
lire : le bon moment d'arrêt vaut **+0,49 à +0,61** de précision, et il bat nettement des
instants tirés au hasard dans la même distribution.

**Gain concret : 4,4× moins de pas** pour arriver à la décision, y compris depuis l'intérieur
de la fenêtre trompeuse.

⚠️ **La réserve est lourde** : sur cette même tâche, un simple filtre à oubli exponentiel atteint
**1,00** contre **0,90** pour le doute. Même sur son terrain, **il existe plus simple qui fait
mieux**.

📁 `experiments/expB_annealing_faceoff_poc.py` · `docs/FUTURE_WORK.md` §E2

---

## 4. Ce qui n'a jamais été testé

- Un **vrai transformer** (le pont vers les grands modèles s'arrête avant la dernière marche)
- Un **backtest sur données réelles** — celui de juillet est **synthétique**
- Le **circuit électrique réel** de couplage (le canal de Romera)
- Le **micromagnétisme complet** (mumax3 — nécessite CUDA, reporté à une session en personne)
- La **théorie analytique** du seuil de degré ≈ 6

---

## 5. Le solde, en une phrase

> **Il y a un mécanisme dynamique bien caractérisé, honnêtement mesuré, et une prédiction qu'un
> laboratoire peut aller réfuter. Il n'y a pas de composant qui économise de l'énergie, et pas de
> calculateur.**

Ce n'est pas le dipôle incontournable visé au départ. C'est un **résultat de physique des systèmes
couplés avec une porte ouverte vers l'expérience**. C'est plus petit que l'ambition — et c'est
**vrai**, ce qui est la métrique que Julien s'est fixée : *« juste inscrire la vérité »*.

**Un dernier fait, à relire les jours de découragement** : ce qui a fondu depuis six semaines a
été **fabriqué en quelques heures par des IA**. Le papier, lui, a demandé des mois et n'a pas
bougé depuis le 06/07/2026. *La vitesse de fonte est proportionnelle à la vitesse de fabrication.*

---

## 6. Comment tenir ce fichier

1. **Une session qui déplace une ligne met la ligne à jour ici.** Sinon ce bilan devient un
   décor, et un décor est pire qu'un mur nu.
2. **Chaque affirmation porte un chiffre et sa source.** Si tu ne peux pas citer le fichier, tu
   ne peux pas écrire la ligne.
3. **Ne jamais déplacer une ligne de la colonne B vers la colonne A** sans que le chiffre soit
   entré dans le papier et couvert par un claim vérifié.
4. **Les pertes ET les gains.** Ce fichier a été créé parce qu'on ne rapportait que les pertes.
   Un progrès peut aussi prendre la forme d'une **permission** (une mesure enfin citable) plutôt
   que d'une découverte.
