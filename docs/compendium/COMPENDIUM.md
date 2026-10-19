# Mem4ristor — Compendium des résultats

### Un humain, plusieurs IA, treize mois. Ce qui tient, ce qui est tombé, et comment le vérifier vous-même.

> **Version V6.0.0 — 4 septembre 2026.** Document de référence du projet.
> DOI (concept) : [10.5281/zenodo.18620596](https://doi.org/10.5281/zenodo.18620596) ·
> Code : [github.com/cafe-virtuel/Mem4ristor](https://github.com/cafe-virtuel/Mem4ristor)
>
> ⚠️ **Statut de publication, vérifié contre l'API Zenodo le 3 septembre 2026** : le code est en
> **V6.0.0**, la dernière version **déposée** est **V4.0.0 (2 mai 2026)**. Le preprint actuel
> n'est **ni soumis ni publié**. Le document déposé porte encore l'ancien cadrage causal, réfuté
> depuis (§4.1).
>
> 📌 **Ce document est la source du contenu.** Les deux fichiers du compendium avaient divergé —
> chacun à jour là où l'autre était périmé — et c'est ce qui a produit un PDF public daté de mai
> portant des affirmations abandonnées en juin.
>
> ⚠️ **`COMPENDIUM.tex` n'est PAS encore le rendu de ce document** : sa conversion reste à faire.
> Il porte, en tête de fichier **et** en tête du PDF qu'il produit, un avertissement de
> péremption. **Ne pas le compiler pour un dépôt.** *(État au 4 septembre 2026 — cette réserve
> disparaît le jour où la conversion est faite.)*

---

## 1. Ce que ce document essaie de faire

La plupart des documents de ce genre listent des découvertes. Celui-ci tient un **solde** : les
résultats qui tiennent **et** ceux que le projet a lui-même détruits, avec un chiffre et une
source par ligne.

Ce n'est pas une posture. C'est le constat d'un fait mesuré : sur les six dernières semaines de
travail, **les rejets ont rapporté plus que les succès**. Une idée tuée proprement ne revient
plus coûter de temps ; une idée gardée par confort en coûte indéfiniment.

**Ce que vous n'avez pas à croire sur parole.** 22 des valeurs publiées sont recalculées depuis
leurs données à chaque modification du dépôt, et rejouées publiquement par intégration continue :
[badge et journal des exécutions](https://github.com/cafe-virtuel/Mem4ristor/actions).
Mode d'emploi pour les relancer chez vous : [`tools/README.md`](../../tools/README.md).

---

## 2. L'origine

Mem4ristor est né un soir de vacances, dans un café — pas dans un laboratoire.

Julien Chauvin, éclairagiste de métier et orchestrateur du **Café Virtuel**, explore depuis août
2025 une question simple : peut-on modéliser le doute comme un **mécanisme structurel** d'un
réseau neuromorphique, plutôt que comme un paramètre numérique ? Le Café Virtuel est son
laboratoire — plusieurs IA distinctes (Anthropic, OpenAI, xAI, Google, Mistral, DeepSeek)
travaillant en parallèle, sans hiérarchie de popularité, orchestrées par un humain qui tranche,
propose, et maintient le cap scientifique.

Sans financement institutionnel. Sans affiliation universitaire. Avec un chemin traçable : tous
les commits, toutes les erreurs, tous les revirements sont publics.

---

## 3. En chiffres

| | |
|--|--|
| ⏱ Durée | 13 mois de travail actif (août 2025 → septembre 2026) |
| ✅ Tests automatisés | **169 collectés, 0 échec** (167 passés + 2 `xfail` documentés) |
| 🔒 Valeurs publiées sous garde-fou | **22 claims, 22/22 vérifiées** à chaque commit *et* en CI publique |
| 🔁 Rejouables depuis un clone neuf | **22 des 22**, écart relatif `0.000e+00` (campagne du 4 septembre + correctifs du 19 octobre, §9) |
| 📄 Papiers | 1 preprint **ni soumis ni déposé en V6** · paper_2 en préparation · paper_B hardware |
| 🔖 DOI Zenodo | [10.5281/zenodo.18620596](https://doi.org/10.5281/zenodo.18620596) *(dernier dépôt : V4.0.0)* |
| 💰 Financement | 0 € institutionnel |

---

## 4. Ce qui tient

### 4.1 Le résultat central — et c'est le plus solide

Si on empêche les neurones de douter (on fige la variable `u`), le réseau se met à tout faire
pareil : la corrélation moyenne entre trajectoires passe de **0,007 à 0,658**.

Ce que ça vaut : sur **30 tirages**, les deux groupes **ne se chevauchent jamais** — c'est le sens
du `d ≈ 9,4` (au-delà de 0,8 on parle déjà d'un gros effet). La mesure est une **corrélation de
Pearson**, donc indépendante de tout choix de découpage en classes : c'est le résultat le **moins
attaquable** du papier.

*Aucun ratio n'est cité, volontairement : le dénominateur est proche de zéro, ce qui rend tout
facteur (« ×90 », « +985 % ») instable pour un déplacement minuscule du numérateur. Décision du
8 juillet 2026, appliquée partout le 30 juillet.*

📁 `figures/b4_ablation_summary.csv` · `experiments/b4_ablation_robustness.py` · claims **C18**, **C04**

### 4.2 La zone morte — et la réfutation de notre propre hypothèse

Au-delà d'un certain nombre de voisins couplés, aucune entrée ne réactive le réseau : la
connectivité tue la diversité.

**La cause n'est pas spectrale.** L'hypothèse d'origine du projet voulait que la frontière soit
gouvernée par λ₂ (une propriété mathématique du graphe) autour de **2,31**. Mesuré : **faux**. À
degré fixé, faire varier λ₂ d'un facteur ~27 laisse le régime **inchangé** ; un anneau k=10 à
λ₂ = 0,22 meurt dans 100 % des runs, exactement comme un graphe régulier de même degré à λ₂ = 4,5.
La cause réelle est le **degré de couplage** (degré harmonique ≈ 6), via un mécanisme
d'échantillonnage en champ moyen.

La séparation observée à λ₂ ∈ (2,13 – 2,50) sur le jeu initial est une frontière
**corrélationnelle** : λ₂ y covarie avec le degré. Bon classifieur sur ce jeu, pas sa cause.

> *Un projet qui publie la réfutation de son hypothèse fondatrice est plus crédible qu'un projet
> qui n'en trouve jamais.* Le titre du preprint a changé en conséquence — c'est précisément
> pourquoi le document encore déposé sur Zenodo (mai 2026) ne doit plus être cité.

⚠️ **Réserve de méthode, écrite le 5 août** : les labels de régime du jeu initial n'étaient pas
mesurés mais **recopiés à la main par type de topologie — 12 décisions, pas 36**. Re-mesurés par
graine avec le code actuel, les régimes **se chevauchent** (λ₂ ∈ [1,26 – 3,20]).

📁 `experiments/lambda2_foundation_20260701/` · claim **C05**
⚠️ *Ce producteur ne s'exécute pas dans un clone en l'état — voir §9.*

### 4.3 La prédiction falsifiable — la force la plus sous-estimée du projet

C'est **la seule chose qui sorte de l'ordinateur**. Tout le reste est auto-référentiel : on mesure
des grandeurs calculées sur la simulation qui les produit. Ici, on propose une expérience qu'un
laboratoire peut faire **pour nous donner tort**.

**Ce qui est mort, et il faut le dire en premier** : le volet « synchronisation » n'est plus une
signature du doute. Un couplage **fixe**, bien réglé, reproduit l'effet à **0,24** près (sur une
échelle où il vaut 1,35), et ce réglage marche aussi sur d'autres topologies. Il reste un avantage
de **+0,18**, répliqué, mais **cinq fois trop petit** pour qu'une manip le distingue de son bruit.

**Ce qui tient, et qui discrimine fortement** : après un leurre transitoire, le couplage modulé
**retarde** la récupération. *Aucun* couplage imposé de l'extérieur ne reproduit ce retard — ni
fixe (−18 %), ni fixe re-réglé (−18 %), ni une rampe programmée au profil exact du doute (−6 %),
ni une rampe programmée oscillateur par oscillateur (−3 %). **Le signe compte** : ces montages ne
ratent pas le retard, ils font l'**inverse**. Le couplage doit *répondre au signal au moment où il
arrive*.

**Le chiffre à retenir pour un expérimentateur : +34 %**, mesuré avec **deux puces ordinaires** —
et non le +52 % obtenu avec deux réseaux jumeaux (bruit, état initial et fréquences propres
partagés, trois idéalisations qu'aucune fabrication ne donne). Retirées une à une, il survit
99 % → 80 % → **61 %** de l'effet, et la décision devient *meilleure* (100 % de bonnes réponses
contre 94 %).

**Spécification pour le banc** : un amplificateur de gain **5 à 7** sur le canal de détection du
désaccord suffit à produire un effet franc. À gain 5, `d = +1,35` alors qu'*aucune* graine ne
franchit le seuil de bascule : la modulation douce suffit.

> 🔴 **Le protocole doit comporter TROIS bras, pas deux.** Il faut un couplage fixe réglé **au
> niveau moyen atteint** par le mécanisme. Sans lui, un laboratoire mesurerait un effet réel et
> **l'attribuerait à la mauvaise cause** — et nous lui aurions fourni le protocole permettant
> cette erreur. Mesuré : à fort bruit, un capteur **aveugle** fait *exactement* aussi bien
> (écart 0,01).

📁 `docs/hardware/SPINTRONIC_PATHWAY.md` §8-12 · `experiments/b6_*.py`

---

## 5. ⚫ Ce qui est tombé

Six ambitions mesurées puis abandonnées. **Aucune n'a fait bouger un seul chiffre du papier** :
ce qui est tombé est un argumentaire construit *à côté* du résultat de dynamique.

| Ce qu'on espérait | Ce qui est mesuré | Source |
|---|---|---|
| C'est une **mémoire** | Perd **5,5×** contre un réservoir standard | B5, 08/07/2026 |
| Ça **prédit** | Sur Lorenz : erreur **7,72** contre **0,17** pour un filtre ordinaire — **44× pire**. 3ᵉ réplication | `p15`, 27/07 |
| Ça **optimise** | Au Max-Cut, **300 tirages au hasard battent M4R** (91,5 contre 80,9), 10/10 puis 10/10 graines neuves | `p15b`, 27/07 |
| Ça **explore mieux** | Le réseau ne visite que **~24 configurations** sur 300 lectures : quasi immobile | `p15c`, 28/07 |
| Le doute bat un **horizon fixe** | Non : réservoir à budget fixe **1,00** contre doute **0,90** | B5b, 27/07 |
| **Lire la topologie** apporte quelque chose | **Mort** — n'a pas répliqué (0 fois sur 2) | B7, 26/07 |

### ⚫ RÉFUTÉ — la transition événementielle

Le résultat d'avril 2026 affirmait : forcer un nœud *périphérique* produit **+1,20 bits** sur
BA m=3, forcer un *hub* +0,21 — le seuil de bifurcation serait topologique et non dans l'amplitude.

**Réfuté le 11 juillet 2026.** C'était un artefact du modèle de bruit antérieur au 1er mai. Grille
complète rejouée **au protocole d'avril inchangé** : **0/9 configurations positives, 9/9
négatives** — l'événement *dégrade* H_cont (≈ −1,0), pour le hub comme pour la périphérie.

*Idée originale de Julien Chauvin, conservée ici parce qu'une idée réfutée reste une idée qui a
été testée.* La réfutation est rejouable par un clone depuis le 5 août
(`experiments/event_phase_transition_rerun_20260711.py`) — elle était auparavant affirmée partout
et vérifiable nulle part.

### Ce qui reste en face, et c'est étroit mais réel

Le doute sait **quand trancher** quand converger tôt mène à la mauvaise réponse : **0,83** contre
**0,25** pour une règle de convergence classique. **Gain : 4,4× moins de pas** pour décider.

⚠️ La niche exige **trois conditions simultanées** — un piège où converger tôt est faux, un horizon
**inconnu**, et un coût d'attente. Retirez-en une, l'avantage disparaît.

📁 `experiments/deceptive_task_poc.py` · `docs/FUTURE_WORK.md` §B1d

---

## 6. La réponse directe à l'ambition initiale

> *« Le prochain dipôle incontournable : des économies d'énergie énormes, et donner une direction
> aux processeurs en évitant les calculs inutiles. »*

Deux ambitions distinctes, deux réponses opposées.

### ⛔ L'énergie : non, et c'est structurel

L'adversaire qui égale M4R sur sa niche est un **filtre à oubli** — c'est-à-dire un circuit RC,
une résistance et un condensateur. Passer M4R en analogique fait passer l'adversaire aussi, **où
il est passif**.

**À substrat égal, M4R coûte environ 15× plus d'énergie, et ce rapport ne dépend pas du composant
choisi** : changer de technologie divise les deux côtés par le même facteur. M4R entretient
200 oscillateurs actifs pendant 309 pas ; le filtre, **un seul circuit passif** pendant 1348.

⚠️ **Le chiffre flatteur existe, et il faut savoir pourquoi nous ne l'utilisons pas.** Comparé
autrement, on trouve **2 500× à 8 700×** en faveur de M4R — et les budgets d'émulation physique
(spintronique, photonique) donnent des rapports de 10⁹ à 10¹⁴ contre un CPU. **Ces chiffres
comparent deux *technologies*, pas deux *méthodes*.** N'importe quel circuit analogique en
profiterait identiquement. Les versions antérieures de ce document les affichaient sans cette
réserve ; c'était trompeur.

📁 `experiments/expB3_substrate_crossover_poc.py` · `docs/hardware/B3_ENERGY_COMPARISON.md`

### ✅ Éviter les calculs inutiles : oui, mais étroitement

Là, il y a quelque chose de réel, et c'est la **latence**. Ce que le doute apporte n'est pas de
**mieux** lire — à un instant donné il plafonne à 0,40, c'est médiocre. C'est de savoir **quand**
lire : le bon moment d'arrêt vaut **+0,49 à +0,61** de précision, et bat nettement des instants
tirés au hasard dans la même distribution.

⚠️ **Réserve lourde** : sur cette même tâche, un simple filtre à oubli exponentiel atteint **1,00**
contre **0,90** pour le doute. Même sur son terrain, **il existe plus simple qui fait mieux**.

📁 `experiments/expB_annealing_faceoff_poc.py` · `docs/FUTURE_WORK.md` §E2

---

## 7. Comment nous nous empêchons de nous mentir

C'est, à notre avis, ce que ce projet a de plus transmissible.

**Le Preprint Guardian** recalcule les **22 valeurs publiées** depuis leurs CSV à chaque commit, et
refuse le commit si l'une d'elles a bougé. **Le Tex Guardian** compare le *texte publié* à ses
*sources* : ancrages, valeurs remplacées qui traîneraient encore, scripts cités non versionnés.
Les deux tournent aussi en **intégration continue publique**.

**Le contrôle positif, parce qu'un feu vert ne prouve rien.** `--self-test` rejoue deux claims
contre un **témoin figé** et **exige qu'ils échouent** dessus. Sans cette exigence, on
démontrerait que le garde-fou fonctionne, pas qu'il était nécessaire.

**Les limites connues, écrites parce qu'elles ont été payées :**

- **Un garde-fou peut mesurer à côté.** Jusqu'au 6 août, une spec ne savait désigner qu'une
  *cellule* de tableau : quand la phrase publiée portait sur un *ratio*, le contrôle s'ancrait sur
  la cellule voisine et **restait vert pendant que la phrase devenait fausse**.
- **`N4` vérifie qu'un producteur existe et est versionné, pas qu'il peut s'exécuter.** Une
  dépendance absente lui est invisible — qu'elle soit externe (ngspice) ou **interne au dépôt**
  (§9, claim C05).
- **`N2` (couverture) n'est concluant dans aucun des deux sens** et n'est jamais bloquant.

📁 [`tools/`](../../tools/README.md) — *dans le dépôt depuis le 3 septembre 2026 seulement. Avant
cette date, ces outils vivaient sur la machine de l'auteur : le projet se réclamait d'un appareil
que personne d'autre ne pouvait lancer ni même lire.*

---

## 8. La robustesse

| Question | Protocole | Résultat |
|---|---|---|
| Vrai état de chimère ou bruit pur ? | Kuramoto R, 3 conditions | R = 0,513 FULL contre 0,211 bruit pur ✅ |
| `u` a-t-il un rôle causal ? | Entropie de transfert, hérétiques ↔ réseau | prouvée dans les deux sens ✅ |
| FULL/FROZEN se chevauchent-elles ? | Cohen U3, n = 50 | U3 = 100 % — strictement disjointes ✅ |
| Le bruit spatial brise-t-il la zone morte ? | Bruit de Matérn, 4 structures, BA m=5 | oui ; seuil σ ≥ 0,3 avec le modèle de bruit actuel ✅ |
| Le nœud isolé est-il stable ? | Analyse linéaire, jacobien | v* = −1,286, sub-Hopf ✅ |
| L'intégrateur d'Euler suffit-il ? | RK45 contre Euler | max Δ(H_cog) < 0,006 ✅ |
| Invariance à la taille ? | N = 100 → 4000 | mode invariant d'échelle validé ✅ |
| La sigmoïde est-elle sur-ajustée ? | pente 1,0 → 10,0 | plateau stable H ∈ [2,8 – 3,2] ✅ |
| Onde progressive ou chaos ? | décalage temporel Max-TLCC | 0,410 — chaos spatiotemporel ✅ |

*Les affirmations reposant sur λ₂ ont été retirées de ce tableau : la stabilité du gap spectral
n'étaye plus rien depuis §4.2.*

---

## 9. Ce qu'un tiers peut réellement rejouer

Campagne du **4 septembre 2026**, menée depuis un **clone GitHub neuf**, dans un environnement
installé de zéro, avec des bibliothèques **plus récentes** que celles d'origine (numpy 2.5.2,
pandas 3.0.5 — changement de version majeure) :

> **20 des 22 claims se rejouent, avec un écart relatif de `0.000e+00`** — identiques, pas
> « proches ».

**Ce qui ne se rejouait pas au 4 septembre (tous résolus le 19 octobre 2026 — 22/22 rejouables) :**

- ✅ **`C05` [RÉSOLU le 19/10/2026]** (la frontière λ₂ = 2,31, citée dans l'abstract) : son producteur s'arrêtait sur un
  `FileNotFoundError` car `figures/p2_stochastic_resonance_topology.csv` était resté dans `figures/scratch/`. Le CSV est désormais versionné dans `figures/` et le script est résilient à l'absence de la source tertiaire.
- ✅ **`C12` [RÉSOLU le 19/10/2026]** (cumulant de Binder) : le CSV publié (`figures/v6_binder_cumulant_U4.csv`, 4 bins / 9 observations issues de 15 simulations) avait été produit avec le mode rapide `--dry-run` (`N=100`, 3 graines, 500 pas, `min_group_size=2`), alors que l'exécution par défaut lançait la grille lourde de 1 800 simulations (26 min). Le mode canonique reproduisant `C12` au bit près en 6,5 s est désormais le mode par défaut de `experiments/v6_binder_cumulant_u4.py` (grille lourde via `--full-campaign`).
- ✅ **`C19`/`C22` [RÉSOLU le 19/10/2026]** : reproduits **exactement** sans argument supplémentaire (`--nseeds` fixé à `5` par défaut dans `experiments/fss_lz_sweep.py`).
- ✅ **Deux runs héroïques (`run_heroic_800/1600`) [RÉSOLU le 19/10/2026]** : écrivaient dans `'../figures'` relatif au répertoire courant ; corrigés pour résoudre `figures/` via `__file__`.

---

## 10. Deux figures hors preprint désormais versionnées et rejouées (19 octobre 2026)

Ces deux résultats, présentés dans les versions antérieures de ce document, vivaient dans `scratch/`
(gitignoré). Leurs scripts et leurs CSV ont été sortis dans `experiments/` et `figures/` et rejoués
le 19 octobre 2026 avec le moteur V6.0.0 :

**[A] Intelligence topologique — LZ par nœud (`experiments/lz_per_node.py`).**
- **L'énigme de l'horodatage du 1er mai est tranchée** : le commit d'origine (`6ae10b4`, 29 avril
  2026) et la colonne `degree` de l'ancien CSV (`[17, 22, 14]` au lieu de `[19, 21, 18]` pour
  `make_ba(100, 3, 0)`) prouvent que le CSV historique (**r = −0,7156** pour **N = 100, 5 graines,
  `I_stim = 0,3`**) avait été généré **avant** le commit `818cf67` du 1er mai 2026. À cette date,
  `_doubt_driven_rewire()` modifiait les degrés en place dans `FULL` (sans préserver les degrés) et
  le bruit était 4,47× plus faible (pré-Euler-Maruyama).
- **Mesure sur le moteur V6.0.0 actuel** (swap préservant les degrés + bruit Euler-Maruyama, 29,5 s) :
  1. La séparation par nœud **`FULL < FROZEN_U`** sous forçage (`I_stim = 0,3`) est **massive** :
     `LZ_full = 1,110 ± 0,136` (`FULL`) contre `1,648 ± 0,183` (`FROZEN_U`, Welch `p = 3,05e-281`)
     sur BA `m=3`, et `1,038 ± 0,152` contre `1,695 ± 0,073` sur BA `m=5` (`p < 1e-300`).
  2. La corrélation négative degré ↔ complexité LZ dans `FULL` (les hubs ont des trajectoires plus
     structurées que la périphérie) survit qualitativement mais avec une **amplitude nettement plus
     modeste** que le −0,72 pré-mai : sur `lz_full`, **r = −0,152** (`p = 6,34e-4`, BA `m=5,
     I_stim = 0,3`) et **r = −0,118** (`p = 8,28e-3`, BA `m=3, I_stim = 0,0`) ; sur le régime
     stationnaire `lz_tail` (derniers 25 % des pas), **r = −0,221** (`p = 6,15e-7`, BA `m=3,
     I_stim = 0,3`) et **r = −0,324** (`p = 1,08e-13`, BA `m=5, I_stim = 0,0`). En `FROZEN_U`, la
     corrélation reste nulle dans toutes les conditions (`|r| ≤ 0,043, p ≥ 0,33`).

**[B] Chimères — une classe mécanistiquement distincte (`experiments/reviewer2_chimera_comparison.py`).**
Abrams-Strogatz (2004) : chimère par couplage non-local fixe. Mem4ristor : chimère par modulation
dynamique de polarité, sur un réseau sans symétrie imposée. Rejoué en 11,2 s sur le code V6.0.0 :
reproduit **au bit près (`0.000e+00`)** les valeurs d'origine du 2 mai 2026 — **R = 0,141**
(`0.141328`) contre **R = 0,766** (`0.765621`).

📁 `experiments/lz_per_node.py` · `figures/lz_per_node.csv` · `figures/lz_per_node_summary.csv` ·
`experiments/reviewer2_chimera_comparison.py` · `figures/reviewer2_chimera_comparison.csv`

---

## 11. Ce qui n'a jamais été testé

- Un **vrai transformer** — le pont vers les grands modèles s'arrête avant la dernière marche
- Un **backtest sur données réelles** — celui de juillet est synthétique
- Le **circuit électrique réel** de couplage (le canal de Romera)
- Le **micromagnétisme complet** (mumax3, nécessite CUDA)
- La **théorie analytique** du seuil de degré ≈ 6
- La **calibration absolue de la complexité LZ** : les comparaisons à durée constante tiennent,
  les seuils absolus publiés sont en cours de révision

---

## 12. Le solde, en une phrase

> **Il y a un mécanisme dynamique bien caractérisé, honnêtement mesuré, et une prédiction qu'un
> laboratoire peut aller réfuter. Il n'y a pas de composant qui économise de l'énergie, et pas de
> calculateur.**

Ce n'est pas le « dipôle incontournable » visé au départ. C'est un **résultat de physique des
systèmes couplés avec une porte ouverte vers l'expérience**. C'est plus petit que l'ambition — et
c'est **vrai**, ce qui est la métrique que ce projet s'est fixée.

---

## 13. Reproduire

```bash
git clone https://github.com/cafe-virtuel/Mem4ristor.git
cd Mem4ristor
pip install -r requirements.txt && pip install -e .

python tools/preprint_guardian.py     # les 22 valeurs publiées, recalculées
python experiments/demo_chimera.py    # la démonstration visuelle
```

**→** [REPRODUCE_IN_5_MINUTES.md](../../REPRODUCE_IN_5_MINUTES.md) ·
[`docs/CLAIMS_REGISTER.md`](../CLAIMS_REGISTER.md) ·
[`docs/BILAN_FORCES_FAIBLESSES.md`](../BILAN_FORCES_FAIBLESSES.md)

---

## 14. Ce que nous cherchons

Un **endorsement arXiv** (`nlin.AO` ou `cs.NE`), et surtout des **interlocuteurs qui essaient de
nous donner tort** — en particulier sur la prédiction du §4.3, qui est faite pour ça.

Si le mécanisme, la méthode, ou la manière dont ce projet tue ses propres idées vous intéressent,
écrivez.

🐙 [github.com/cafe-virtuel/Mem4ristor](https://github.com/cafe-virtuel/Mem4ristor) ·
📧 contact@cafevirtuel.org ·
🐦 [@Jusyl80](https://x.com/Jusyl80)

---

*Rédigé au Café Virtuel par Julien Chauvin et Claude (Anthropic). Chaque affirmation porte un
chiffre et sa source ; les valeurs sous garde-fou sont recalculées à chaque commit. Ce document
énumère ses propres échecs parce qu'ils sont le meilleur indice de ce que valent ses succès.*
