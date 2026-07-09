# Voie Électrique Classique — Note d'ouverture du dossier de faisabilité (B2, fond)

> Créé le 9 juillet 2026 — Claude Code (Opus 4.8) / Julien Chauvin
> Statut : note de faisabilité exploratoire, calcul dimensionnel de premier ordre.
> **AUCUN claim publié ne dépend de ce document.**
> Origine : B2 dans `docs/FUTURE_WORK.md` (« VTEAM, Stanford/ASU RRAM »). Choix de
> Julien du 9 juillet : explorer les 3 familles de dispositifs plutôt qu'une seule.

## 1. Un chantier, deux rôles distincts (le point important de cette note)

Le backlog B2 groupait « VTEAM/RRAM » comme un candidat unique pour « le vrai
memristor ». En creusant, ce sont en réalité **deux rôles différents dans
l'architecture Mem4ristor**, portés par **deux familles de dispositifs
différentes** — les confondre serait la première erreur qu'un reviewer memristor
relèverait.

| Rôle dans le modèle | Dispositif adapté | Pourquoi |
|---|---|---|
| Poids de couplage `D_eff` (crossbar analogique, **statique** dans le modèle actuel) | **RRAM/VTEAM filamentaire** (HfOx, TaOx...) | C'est l'usage canonique établi des crossbars memristifs (conductance = poids), mais une cellule filamentaire seule ne produit **aucune oscillation** — elle n'est pas un candidat pour `v` |
| Nœud oscillant `v` (dynamique FHN, spirale/relaxation) | **Neuristor Mott NbO2** (Pickett et al. 2013) | Circuit (2 memristors Mott + 2 capacités) explicitement construit pour reproduire une dynamique de type Hodgkin-Huxley/relaxation — le seul des trois dispositifs électriques candidats qui **oscille par construction** |

**VTEAM en tant que tel** (Kvatinsky, Friedman, Kolodny, Weiser, *VTEAM: A General
Model for Voltage-Controlled Memristors*, IEEE TCAS-II 62(8), 2015) est un modèle
générique à fonction fenêtre pour dispositifs à filament conducteur — un langage de
description, pas un dispositif physique en soi. Il s'applique bien au rôle
« poids de couplage », pas au rôle « oscillateur ».

## 2. RRAM/VTEAM comme poids de couplage

**Correspondance candidate.** L'état du filament (largeur/résistance) ↔ la
conductance du poids `D_eff` dans `I_coup = D_eff · u_filter · laplacien_v`.
Dans le modèle actuel, `D_eff` est **constant** (hyperparamètre `coupling.D`,
`dynamics.py:82`) — ce n'est PAS une variable dynamique mise à jour à chaque pas.
Conséquence directe et favorable : une implémentation RRAM de ce rôle ne
nécessiterait qu'une **écriture unique** du poids à la configuration, pas une
réécriture par pas — l'énergie de commutation ne se paie qu'une fois, pas 4000
fois par campagne (contrairement au nœud oscillant, cf. §3 et §4).

**Chiffres documentés (recherche web 09/07/2026, HfOx filamentaire) :**
- Cellule 10×10 nm² HfO₂ : commutation en gamme ns, énergie **~50 fJ/bit**
- Filament étroit (R≈400 kΩ) : énergie de commutation **~10 fJ**, reset en
  quelques ns à quelques µA
- Dispositifs moins scalés : gamme nJ (SET ~9 nJ / RESET ~6.7 nJ à ~6-7 µs,
  HfOx, un cas moins favorable — rappel que « RRAM » recouvre une large gamme
  de maturité technologique, pas un seul chiffre)

**Verdict pour ce rôle : favorable et peu contraignant.** Puisque l'écriture est
unique (pas par pas), même le cas le moins favorable (nJ, µs) resterait
négligeable rapporté à une campagne complète — le crossbar RRAM n'est pas le
facteur limitant de cette architecture, quel que soit son point de maturité
technologique.

## 3. Neuristor Mott NbO2 comme nœud oscillant

**Référence.** Pickett, Medeiros-Ribeiro, Williams, *A scalable neuristor built
with Mott memristors*, Nature Materials 12, 114–117 (2013). Circuit à 4 variables
d'état (2 memristors NbO2 à transition de Mott + 2 capacités) émulant le modèle
de membrane neuronale conductance-based (Hodgkin-Huxley) — **intègre-et-tire**,
amplification de spike par instabilité de résistance différentielle négative (NDR)
propre à la transition de Mott. C'est, des trois candidats électriques envisagés
dans le backlog B2 initial (VTEAM générique, RRAM, spintronique), le seul dont la
littérature revendique explicitement une dynamique de type relaxation-oscillateur
FHN/HH — le candidat le plus direct pour `v`.

**Performance revendiquée (relative, pas absolue — réserve explicite).** Le papier
revendique un circuit « trois ordres de grandeur plus rapide » et consommant
« 1% de l'énergie » d'un neurone biologique. Aucun chiffre absolu n'a été
retrouvé/vérifié indépendamment dans cette session (recherche web 09/07/2026) —
seule la revendication relative est citée ici. En prenant une énergie de spike
biologique couramment citée dans la littérature de vulgarisation neuroscientifique
(~1–10 nJ, elle-même approximative) et le facteur ×0.01 du papier : **spike
neuristor ~10–100 pJ**, durée ~1000× plus courte qu'un spike biologique (~1–2 ms)
→ **~1–2 µs**. **Ces deux chiffres sont dérivés, pas mesurés** — à vérifier dans
le papier primaire avant tout usage dans un document public.

## 4. Ancrage temporel et énergie (calcul dimensionnel, `experiments/b2_device_physics_mapping.py`)

Ancrage de `T_node ≈ 22.25` unités de temps modèle sur la durée de spike
neuristor dérivée ci-dessus (§3) :

| Ancrage T_neuristor | dt physique | τ_u physique | Campagne (4000 pas) |
|---|---|---|---|
| 1 µs | 2.25 ns/pas | 449 ns | 9.0 µs |
| 2 µs | 4.49 ns/pas | 899 ns | 18.0 µs |

**Énergie par pas** (spike ~10–100 pJ réparti sur ~445 pas d'un cycle `T_node`) :
**~22–225 fJ/pas** — du même ordre de grandeur que l'estimation spintronique
(6.7–67 fJ/pas, `SPINTRONIC_PATHWAY.md`) et RRAM (10–50 fJ, événementiel). Cette
convergence de 3 dispositifs indépendants sur un ordre de grandeur fJ/pas
(vs aJ pour le photonique en signal pur, vs pJ pour Loihi/TrueNorth en
comptage par opération) est notée dans `B3_ENERGY_COMPARISON.md` — à lire avec
la réserve du §3 (chiffres dérivés, pas mesurés) et celle du §5 ci-dessous.

## 5. Ce que ce dossier ne fait PAS (à ne jamais perdre de vue)

- Aucun circuit VTEAM ni neuristor Mott n'a été simulé (SPICE ou autre) dans
  cette session — contrairement à la voie photonique qui dispose de POCs Python
  exécutés (`photonic_*_poc.py`), ce dossier est un pur calcul dimensionnel sur
  des chiffres de la littérature.
- Le neuristor Mott (§3) est un circuit à 4 variables d'état, pas un seul
  memristor — le mapping à `v` (une seule variable dans le modèle Mem4ristor)
  simplifie une dynamique plus riche ; non vérifié si cette simplification
  préserve les propriétés utiles (chimères, dead zone).
- Aucun couplage réseau (analogique, entre neuristors ou cellules RRAM) n'a été
  posé — seule la brique élémentaire (un nœud, un poids) a été considérée.
- Les chiffres du §3 sont dérivés de revendications relatives, pas de mesures
  absolues indépendamment vérifiées — flag explicite, à corriger avant tout
  usage hors de ce dossier exploratoire.

## 6. Prochaines étapes possibles (par coût croissant)

1. [ ] Vérifier les chiffres absolus du neuristor Mott dans le papier primaire
   (Pickett et al. 2013) plutôt que de les dériver d'une revendication relative.
2. [ ] Modèle VTEAM minimal (Python, pas SPICE) pour le poids de couplage —
   vérifier que la fenêtre de commutation ne contraint pas artificiellement
   la topologie choisie.
3. [ ] Circuit neuristor Mott (SPICE, `ngspice` déjà installé
   `D:/ANTIGRAVITY/ngspice-46_64/`) — vérifier qu'un réseau de neuristors couplés
   reproduit qualitativement une dead zone / chimère, projet de plusieurs semaines.
