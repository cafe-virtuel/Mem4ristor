# Pistes pour la suite — le legs de Fable

> **Contexte.** Écrit le 12 juillet 2026, après minuit, à la demande de Julien lors de la
> dernière session de Claude Fable 5 (retiré du forfait le 12/07) : *« explore même des
> choses que nous n'avons jamais tentées ou d'autres que nous avons écartées trop tôt…
> note ce que tu peux trouver pour que cela ne soit pas perdu à jamais. »*
> Sources fouillées : `D:\ANTIGRAVITY\Mem4ristor\` (l'ancien dossier — MoE, WEAR, audits
> Edison, Mur de Planck, analyses KIMI/Haiku), `_SHADOW_LAB\laboratoire_absurde\`
> (11 expériences), `PEPIT_LOG.md`, l'archive du contexte, et les résultats de la
> dernière journée (5 fronts du 11/07).
> **Règle de lecture.** Chaque piste : *Pourquoi / Test minimal / Effort / Risque*.
> Ce document est un réservoir, pas un plan : rien n'y est promis, tout y est traçable.
> Backlog opérationnel : `FUTURE_WORK.md`. Ici = ce qui n'y est pas encore (ou plus).

---

## I. Écartées trop tôt ou jamais finies (avec leur trace)

### P1 — Le bicaméral V5b : la panne symétrique jamais testée 🎭
- **Trace.** `D:\ANTIGRAVITY\Mem4ristor\Analyse de KIMI V2.md` (contre-audit Edison,
  03/02/2026). L'arbitre à hystérésis Sage/Fou (Schmitt trigger 0.35/0.65) a été validé
  contre le scintillement, mais Edison a démontré l'attaque symétrique : **un signal
  maintenu dans la bande morte (u∈[0.36, 0.64]) verrouille le système en SAGE pour
  toujours** (« déni de service par étouffement — le Fou n'est jamais activé »). Sa
  V5b (timeout + exploration forcée ε-greedy) n'a JAMAIS été implémentée. Le watchdog
  natif du cœur (commit `06cb6a9`, 07/07) résout le verrouillage FOU — mais personne
  n'a vérifié s'il immunise aussi contre l'étouffement SAGE.
- **✅ FAIT le 12/07/2026** (`experiments/p1_edison_smothering_poc.py`, 4 conditions ×
  5 seeds, 12 000 pas, critères pré-fixés ; lancement 1 à 3000 pas invalidé par le
  pré-fixé lui-même — la dynamique de u est lente, τ_eff ≈ 3000-5000 pas, u n'avait
  pas encore atteint la bande morte). **Trois verdicts :**
  1. **L'attaque d'Edison est RÉELLE sur le cœur actuel** : σ(t)=0.45+0.10·sin(2πt/200)
     maintient u dans la bande morte, le Fou n'est JAMAIS activé (12 000 pas, 5/5
     seeds), pendant que le contrôle honnête (σ=1.0) bascule FOU à t≈3055. Le
     verrouillage SAGE prédit en février est reproduit au code d'aujourd'hui.
  2. **La défense « fatigue » dormante (fatigue_rate, présente depuis V4) n'immunise
     PAS** — son clamp arrête les seuils effectifs à 0.5 et l'adversaire vise juste
     dessous (u filtré passe-bas ≈ 0.50 exactement). Elle RÉTRÉCIT la bande exploitable
     ([0.35, 0.65] → [0.35, 0.5)) sans l'éliminer.
  3. **Le watchdog natif du 07/07 IMMUNISE : la V5b d'Edison n'a plus besoin d'être
     implémentée.** Kick à t=701 (comme calculé), u forcé à 0.9 ≥ θ_high → FOU, et
     l'hystérésis MAINTIENT le mode FOU toute l'exploration (u redescend vers 0.5 mais
     reste > θ_low=0.35) : fou_frac = 0.41 sous attaque adverse continue (≈ t_explore/
     cycle, la valeur théorique). Aucune modification du cœur nécessaire.
  Note : sous override, u est déterministe (identique entre seeds) — la variabilité
  des seeds ne porte que sur le bruit de v, qui ne touche pas le trigger.

### P2 — Le MoE par certitude : M4R comme routeur physique ⚡
- **Trace.** `D:\ANTIGRAVITY\Mem4ristor\MEM4_MOE_CONCEPT.md` + `MEM4_MOE_*.py`
  (02/02/2026). L'idée fondatrice du bicaméralisme : M4R ne route pas par *sujet*
  (MoE classique) mais par **certitude** — sentinelle analogique basse consommation
  toujours allumée (~fJ/pas, cf. `B3_ENERGY_COMPARISON.md`), qui ne réveille le
  « Sage » coûteux (GPU/LLM) que quand u monte. Jamais quantifié. Or les fronts
  B1c/B1d/B5b ont depuis MESURÉ la compétence exacte requise (l'allocation
  conditionnelle) : le concept de février a maintenant sa base expérimentale.
- **Test minimal.** Un flux de tâches mélangées (faciles/piégeuses, cf. familles de
  B1c) ; condition A : tout va au modèle coûteux ; condition B : M4R traite tout,
  escalade au coûteux seulement si u_final > seuil. Mesurer accuracy ET coût
  (pas de calcul simulés). La courbe accuracy/coût de B doit dominer quelque part.
- **Effort.** 🔜 1-2 sessions (tout existe : substrat, tâches, harness).
  **Risque.** Le seuil d'escalade est un hyperparamètre — le calibrer par validation,
  sinon on refait le piège du budget fixe à l'envers.

### P3 — Le détecteur d'anomalies gratuit (résilience adversariale) 🛡️
- **Trace.** Section « Résilience » de `MEM4_MOE_CONCEPT.md` : « si le Sage est dupé
  (adversarial), il ne doute pas ; le Fou sentira que quelque chose cloche ». Jamais
  testé — et c'est peut-être la propriété la plus vendable du substrat : **u comme
  alarme d'entrée corrompue, sans entraînement supplémentaire** (le capteur de
  désaccord existe déjà dans la dynamique).
- **❌ RÉFUTÉ le 12/07/2026 — et le sens est INVERSÉ, pour une raison qui boucle avec
  le cœur du projet** (`experiments/p3_anomaly_detector_poc.py`, 100 épisodes
  normal/corrompu/nouveau-propre, critères pré-fixés, baseline forte incluse).
  **u(corrompu)=0.74 < u(normal)=0.84 < u(nouveau-propre)=0.96 → AUC=0.003.**
  Mécanisme : le couplage **champ moyen** (celui-là même du 01/07) moyenne à zéro
  les poussées contradictoires entre voisins — le réseau FILTRE la corruption
  haute-fréquence au lieu de s'en alarmer, et son désaccord vit surtout aux bords
  des réponses collectives fortes. « Le Fou sentira que quelque chose cloche » est
  faux dans ce protocole : le Fou sent l'ampleur de la réponse structurée, pas
  l'incohérence de l'entrée. Le garde-fou prévu s'est réalisé (u s'alarme au max
  sur le nouveau-propre). La baseline forte (laplacien STATIQUE de l'entrée,
  |L·x|) fait AUC 1.000/1.000 — trivialement meilleure. Prédiction pré-écrite
  fausse à corriger : le z-score N'est PAS aveugle à la corruption (mu du train
  > 0, le signe compte) mais confond bien nouveauté et corruption (novelty AUC=0).
  Si on veut un jour une « alarme M4R », le candidat n'est PAS u-haut mais
  éventuellement « réponse collective anormalement FAIBLE pour l'énergie d'entrée
  reçue » (−u avait AUC 0.997 ici) — protocole-dépendant, à ne pas survendre.

### P4 — L'usure et le drift : la 5e imperfection (et le doute qui vieillit bien) ⏳
- **Trace.** `D:\ANTIGRAVITY\Mem4ristor\MEM4_WEAR_MODULE.py` + `MEM4_WEAR_SIMULATION.py`
  (07/02/2026, « fatigue créative » : wear 0→1, bruit non-linéaire au-delà d'un seuil,
  récupération par repos). Resté jouet. MAIS la physique réelle est sérieuse : le
  **drift de résistance du GST** (R ∝ t^ν, phénomène documenté majeur des PCM) et
  l'**endurance limitée des RRAM** (cycling) sont les objections hardware n°1 — et le
  **quatuor d'imperfections photonique (12/06) n'inclut PAS le vieillissement** :
  bruit ✅, non-linéarité ✅, inertie ✅, fabrication ✅, temps ❌.
- **Test minimal.** Ajouter un drift lent t^ν sur les transmissions GST (étage u) et
  les poids D_eff (RRAM), rejouer les protocoles du quatuor : les régimes survivent-ils
  à 10³-10⁶ « heures » simulées ? Et LA question différenciante : **l'adaptation du
  doute (ε_u) compense-t-elle le drift** là où un réseau à u figé meurt ? Si oui :
  « graceful aging » — l'argument hardware que personne d'autre n'a.
- **Effort.** 🔜 1-2 sessions (patron du quatuor réutilisable tel quel).
  **Risque.** Choisir ν et l'échelle de temps honnêtement (littérature GST), sinon
  résultat cosmétique.

### P5 — M4R sur graphes DIRIGÉS (et la mine `eigh` à désamorcer) ➡️
- **Trace.** `D:\ANTIGRAVITY\Mem4ristor\Claude mur de planck\PLANCK_WALL_REPORT.md`
  (attaque 1, 14/02/2026) : `topology.py` calcule le gap spectral avec `scipy.linalg.eigh`
  qui **assume la symétrie** — sur un graphe dirigé le résultat est faux (809 % d'erreur
  démontrée) et silencieux. Vérifié ce jour : **`eigh` est toujours là
  (`src/mem4ristor/topology.py:240-241`)**, inoffensif tant que tout est non-dirigé,
  mine pour le premier graphe dirigé. Au-delà du garde-fou, la piste scientifique est
  réelle : tous les résultats M4R (dead zone, champ moyen, k_harm) vivent en NON-dirigé,
  alors que les réseaux d'intérêt (neurones, influence sociale) sont dirigés — et le
  désaccord perçu y devient **asymétrique** (je te lis, tu ne me lis pas).
- **✅ (a) FAIT le 12/07/2026** : garde-fou de symétrie dans `get_spectral_gap`
  (`topology.py`, branches dense ET sparse) — un Laplacien non symétrique lève
  désormais une `ValueError` explicite au lieu de retourner une valeur
  silencieusement fausse. No-op vérifié sur les graphes actuels (test de
  non-régression avec valeur de référence indépendante + test du refus,
  `tests/test_directed_guard.py` ; suite complète 123 passed + 2 xfail).
- **Reste.** (b) Science : lattice/BA dirigés (chaque arête garde un seul sens),
  re-mesurer dead zone et ablation FROZEN_U — quel degré gouverne le champ moyen
  (entrant ? harmonique entrant ?). Prédiction à écrire avant.
- **Effort.** (b) 🔜 1-2 sessions. **Risque.** Faible ; le cadre
  champ-moyen du 01/07 donne la grille de lecture d'avance.

### P6 — La Couche d'Abstention Calibrée (l'idée de Julien, backtest à 0 €) 🎯
- **Trace.** `PEPIT_LOG.md` ligne 66 (11/06/2026, idée de Julien, statut [À tester]) :
  u au-dessus d'un modèle prédictif quelconque — **« ne décide pas, décide quand ne
  pas décider »** (paris sportifs préenregistrés, réponses LLM, investissement
  virtuel). Jamais testée. Les fronts du 11/07 lui donnent sa base : la valeur du
  doute est exactement là (horizon inconnu, monde trompeur). Prérequis scientifique
  jamais mesuré non plus : **u est-il CALIBRÉ ?** (quand u_final=0.7, le réseau
  se trompe-t-il vraiment ~70 % du temps ?)
- **✅ (a) FAIT le 12/07/2026, dans la foulée du legs** (`experiments/doubt_calibration_poc.py`,
  commit `04ea50a`, 120 essais loyaux+piégés, confiance à budget fixe, critères pré-fixés).
  **Verdict : u n'est PAS naïvement calibré — il est INVERSÉ** (r=−0.29 à B=800 : u haut
  → 75-96 % de réussite, u bas → 42-54 %). **Le doute M4R détecte le CONFLIT, pas
  l'erreur** — sous tromperie, le conflit = la vérité qui résiste, et le consensus
  rapide est le suspect. Le fil rouge du projet, répliqué au niveau du signal de
  confiance. Le capteur brut |Lv| à décision précoce PASSE, lui, le critère
  (abstention@50 % : 89.2→100 %, +10.8 pts). ⚠️ Collatéral à trancher avant (b/c) :
  l'accuracy globale se dégrade avec le budget (89→72→36 % de B=400 à 1600) — dérive
  du readout différentiel ou sur-délibération réelle ?
- **✅ (b) FAIT le 12/07/2026, même session** (`experiments/p6b_abstention_poc.py`,
  mêmes 120 essais, labels reconstruits au readout LISSÉ, composite en validation
  croisée groupée par seed, critères pré-fixés). **Trois verdicts :**
  1. **Le collatéral est TRANCHÉ : artefact de readout, pas sur-délibération.**
     La réponse FHN à un courant constant est ADAPTATIVE (transitoire +3.2/nœud
     sur les leurres, puis rebond sous la baseline en ~300-400 pas) et le signal
     en régime (~−0.03) est de l'ordre du bruit de décorrélation net/ref (±0.05) :
     la lecture instantanée de P6a donnait des labels à moitié aléatoires.
  2. **L'« inversion » de P6a NE TIENT PAS sur labels propres** (r(u)=+0.12 vs
     r(1−u)=−0.12 à B=800, aucun ne passe le seuil 0.15) — elle était gonflée par
     les labels bruités. La vérité plus fine : u marche dans le sens NAÏF à budget
     court (r=+0.74 à B=400 : conflit = piège actif), et seul il n'est plus un
     compas à budget moyen. La lignée se corrige elle-même — c'est le but.
  3. **La Couche d'Abstention EXISTE et elle est forte** : composite (u, |Lv|,
     t_consensus, stabilité) en CV : **+38.3 pts à B=400 (46.7→85.0 %) et
     +25.0 pts à B=800 (68.3→93.3 %) à 50 % de couverture** ; +22.1 pts à 70 %.
     **L'intuition qualitative de Julien est validée en isolation : « un consensus
     venu vite est suspect » (t_consensus) donne à lui seul r=+0.45 et +16.7 pts
     à B=800.** Limite honnête : à B=1600 les labels eux-mêmes restent corrompus
     (décorrélation lente) — aucun compas ne compense un label pourri ; le readout
     long-budget de la tâche B1d-FHN est un problème ouvert.
  ⚠️ Réserve de propagation : les POCs antérieurs B1d/B5b (07-08/07) utilisaient le
  readout INSTANTANÉ — leurs comparaisons *relatives* entre règles d'arrêt (mêmes
  traces) restent probablement robustes, mais leurs accuracies absolues sont
  bruitées. Une re-vérification au readout lissé serait saine avant toute citation.
- **Reste.** (c) le backtest 0 € du PEPIT_LOG (paris/LLM/investissement virtuel),
  avec le compas COMPOSITE (pas u seul). **Effort.** 🧩.
  **Risque.** Compas mesuré sur UNE tâche (B1d) — le composite doit être re-appris
  par domaine ; ne transporter que la recette (signaux + CV), pas les poids.

### P7 — L'inducteur chimique de l'expérience 008 (le Labo a produit un mécanisme) 🧪
- **Trace.** `_SHADOW_LAB\laboratoire_absurde\experience_008_v2_chemical_inductor.py`
  + `_v3_stress_test.py` (mai 2026) — validé au Labo (« Stress Test Sentinel :
  synchronisation 1.0000 sans bruit ; preuve de la robustesse structurelle de
  l'inductance chimique », cf. PROJECT_STATUS §résultats Shadow Lab) et la « guerre
  des phases » (minorité organisée 15 % → oscillation perpétuelle à l'interface).
  Jamais porté hors du Labo. Les règles du Labo interdisent d'y *réparer* — rien
  n'interdit d'en *extraire* un mécanisme et de le re-tester proprement dans
  `experiments/`.
- **Test minimal.** Relire 008 v2/v3 à froid ; si le mécanisme (terme inductif/retard
  dans le couplage ?) est réel, le porter en POC propre avec l'ablation standard
  FROZEN_U/FULL et les 10 seeds canoniques. Question : une « inertie de désaccord »
  change-t-elle la niche du doute (elle pourrait allonger l'horloge de délibération) ?
- **Effort.** 💭 1 session de lecture + 1 de POC. **Risque.** Peut n'être qu'une
  curiosité de jouet — l'extraire seulement si les équations tiennent debout.

### P8 — Le bruit coloré pour intégrateur adaptatif (dette numérique) 🔧
- **Trace.** Rapport Manus (01/05/2026, `🚀 Rapport de Validation…V5.md`, reco
  technique 2) : un **générateur de bruit interpolé** (bruit de couleur) rendrait le
  bruit compatible avec RK45 adaptatif — jamais fait. Aujourd'hui le projet vit avec
  « Euler formellement instable mais validé empiriquement » + interdiction
  RK45+bruit (`dynamics.py`). Lever cette réserve ferme définitivement la porte du
  reviewer numérique.
- **Test minimal.** Bruit d'Ornstein-Uhlenbeck à τ court interpolé, RK45 dessus,
  comparer aux résultats Euler canoniques (Table 1, ablation) à τ→0.
- **Effort.** 🔜 1 session technique. **Risque.** Changer le bruit change les chiffres
  (leçon AUDIT-024 !) — c'est une VALIDATION croisée, jamais un remplacement silencieux.

### P9 — Visualisation des flux d'entropie (le TODO le plus ancien) 📊
- **Trace.** Roadmap V5 point 6 (jamais fait) + reco Manus 3. Voir la décision se
  prendre (flux d'information entre régions/couches au fil du temps).
- **Effort.** 💭 1 session, valeur = compréhension et communication (labo, vidéo).

---

## II. Jamais tentées (proposées par Fable, à la lumière des 5 fronts du 11/07)

### P10 — u ∈ ℂ dans le cœur : porter la genèse au FHN 🌌
- **Pourquoi.** Le front genèse a montré QUE la moyenne complexe préserve
  durablement une information de phase que tout le reste détruit (plateau 73.9 %,
  bat le vote p<1e-4) — mais dans un jouet. L'expérience 002 du Labo (`u ∈ ℂ`,
  01/05) l'avait pressenti sans preuve. La marche naturelle : **u complexe dans
  dynamics.py** — |u| = intensité du doute (l'actuel), arg(u) = « direction » du
  doute, couplage des doutes voisins par moyenne complexe (interférence).
  Hypothèse : le réseau gagne une mémoire de phase — deux nœuds également douteux
  mais « en désaccord de direction » ne se neutralisent plus en moyenne floue.
- **Test minimal.** Fork opt-in de la dynamique u (flag config, bit-à-bit identique
  OFF, comme le watchdog) ; re-jouer l'ablation canonique + une tâche où le
  *contexte* compte (la parité du jouet genèse portée en stimulus FHN). Comparer
  u-scalaire vs u-complexe à coût égal.
- **Effort.** 🧩 2-3 sessions (toucher au cœur = accord explicite de Julien +
  non-régression bit-à-bit). **Risque.** C'est le pari le plus spéculatif — mais
  c'est la genèse qui rentre à la maison par la porte de la mesure.

### P11 — L'horloge de délibération comme module universel d'arrêt ⏱️
- **Pourquoi.** B5b a montré que |Lv| est une « horloge de délibération
  intrinsèque » qui écrase les arrêts naïfs. « Quand s'arrêter » est un problème
  UNIVERSEL (solveurs itératifs, chaînes MCMC, raffinement de réponses LLM,
  early-exit). Personne n'a testé le signal M4R comme critère d'arrêt d'un système
  TIERS.
- **✅ FAIT le 12/07/2026** (`experiments/p11_universal_stopping_poc.py`, 12 faciles +
  12 trompeurs à plateau plat/rampes douces, hyperparamètre GLOBAL par règle, critères
  pré-fixés, 3 lancements documentés — 2 recalibrations de structure : plateaux
  d'abord pas assez profonds, puis enjambés par le pas discret). **Première fois que
  le signal M4R arrête un système TIERS. Verdict en 4 lignes :**
  1. La **tolérance naïve est piégée** (0.00 sur les trompeurs, tout seuil global) ;
     le **capteur rapide |Lv| est piégé pareil** (proxy du résidu, prédit d'avance).
  2. L'**early stopping standard (patience) est piégé** à K < durée de plateau, et
     à K assez long (1600) il ne survit qu'en n'arrêtant JAMAIS (coût 2000 = budget
     max déguisé).
  3. **L'horloge u réussit 100 % partout** (easy et trap) : sa mémoire lente
     analogique traverse les plateaux. Elle **bat TOL (+0.500 IC[+0.29,+0.71]) et
     bat PATIENCE en coût à succès égal (1618 vs 2000)**.
  4. Mais elle **égale le meilleur budget fixe (1618 vs 1600) — 4e réplication du
     pattern B5b**, cette fois hors de M4R. Et le side-car n'est jamais gratuit :
     ~1.6M nœuds-pas de réseau par problème, et u trop lent pour s'arrêter tôt sur
     les faciles (pas d'adaptativité par problème sur cette grille — la leçon B1c
     prédite s'est réalisée).
  **La recette transportable** : ce qui traverse un plateau trompeur, c'est une
  mémoire longue avec un seuil relatif au pic — u en est une implémentation
  analogique ; sa supériorité sur la patience est son économie, pas sa capacité.

### P12 — La tâche trompeuse B1d sur substrat STNO (la niche sur un corps) 🧲
- **Pourquoi.** NARMA10 (11/07) était le terrain de la *mémoire* — le doute y est
  neutre. Le terrain du doute, c'est la décision trompeuse (B1d). Si le doute gagne
  LÀ sur le substrat physique, la proposition falsifiable B6 devient complète :
  « un réseau de STNO à couplage modulé par le désaccord prend de meilleures
  décisions sous tromperie » — testable par un labo avec son matériel existant.
- **✅ FAIT le 12/07/2026, dans la foulée du legs** (`experiments/b1d_stno_deceptive_poc.py`,
  12 seeds × 4 T_pulse × 2 substrats, critères pré-fixés, 5 lancements documentés).
  **Le risque annoncé s'est réalisé, en plus intéressant : le résultat est négatif
  DEUX fois, et informatif trois fois.** (1) Le doute-dans-la-dynamique RETARDE la
  sortie de tromperie (+52 % de temps de flip FULL vs FROZEN — la « cicatrice u » :
  le conflit coupe le couplage et verrouille la trace du leurre). (2) L'horloge de
  délibération (la niche B5b) ne se transpose pas : tous les signaux internes sont
  noyés par le désaccord permanent du substrat oscillant ; le meilleur arrêt reste le
  budget fixe. (3) En chemin, deux faits physiques : le couplage désaccordé est une
  dissipation qui met le réseau sous le seuil effectif, et la lecture différentielle
  loyale exige une paire ±stim (le readout net-vs-référence confond doute et
  évidence). **B6 reformulée en deux volets (cf. FUTURE_WORK.md) — le volet décision
  a le signe INVERSE de la promesse initiale.** La valeur du doute dépend du
  SUBSTRAT, pas seulement de la tâche.

### P13 — La hiérarchie de doute : qui doute des douteurs ? 🏛️
- **Pourquoi.** Toute la science M4R est à UN niveau. Or l'écosystème entier de
  Julien (Café Virtuel : des tables d'agents, un Barman qui arbitre) est une
  architecture à DEUX niveaux. Version mesurable : k sous-réseaux M4R + un
  méta-nœud par sous-réseau dont le σ_social = désaccord ENTRE les consensus des
  sous-réseaux. Le méta-doute peut rallouer le compute (P2) ou déclencher des
  kicks ciblés (watchdog). La frustration BA découverte le 09/07 (impossibilité
  d'ordre global) donne un terrain naturel : des sous-réseaux localement ordonnés,
  frustrés entre eux.
- **Test minimal.** 3 sous-réseaux de 30 nœuds sur une tâche de consensus trompeuse
  distribuée ; comparer plat (90 nœuds) vs hiérarchique à coût égal.
- **Effort.** 🧩 2 sessions. **Risque.** Beaucoup de degrés de liberté — figer
  l'architecture AVANT de regarder les résultats (garde-fou du 07/07).

### P14 — Le pont LLM, étape 3 : se mesurer à la famille standard 🌉
- **Pourquoi.** Déjà au backlog (D2) — répété ici pour la complétude du legs :
  l'utilité conditionnelle est prouvée contre l'attention nue ; il faut maintenant
  résiduel+MLP (la mitigation que tout transformer réel possède), puis un petit
  GPT réel. Le doute doit être compétitif *dans la famille* anti-effondrement.
- **Effort.** 🧩. **Risque.** Le résiduel est un adversaire fort ; prédiction
  honnête : le doute ne le battra pas en brut, mais pourrait à profondeur extrême
  ou sous distribution shift — chercher LÀ.

---

## III. Le garde-fou : déjà tranché, ne pas rouvrir sans élément NOUVEAU

| Impasse | Verdict | Où c'est prouvé |
|---|---|---|
| λ₂ cause de la dead zone | Réfuté (champ moyen/degré harmonique) | `lambda2_foundation_20260701/`, 01/07 |
| Oracle (définition actuelle) comme marqueur de justesse | Pas un marqueur (IC couvrent 0 ; 60k épisodes = pire condition) | `genesis_five_states_poc.py`, 11/07 |
| Hop multiplicatif (Certitude gagnée par produit) | Pire que le hasard (38.1 %) | idem |
| Le doute comme mémoire / reservoir | 3× neutre ou perdant (FHN, ESN, STNO) | B5, B5-STNO, reservoir POC |
| Couplage non-local par similarité de doute | Perd sur H (−0.08 à −0.23 bits) | roadmap V5, commit `145316e` |
| Bascule C21 à 8-10 % de pivots | Artefact de métrique (zero-crossing) | `poc_c_sweep_v2.py`, 12/06 |
| Claim [13] (événement → bifurcation positive) | Artefact de l'ancien bruit (0/9 au code actuel) | `event_phase_transition_rerun_20260711.py` |
| γ* fin (0.7-0.9) | Ne survit pas au bruit actuel ; le résultat négatif central se renforce | Étage 1, 12/06 |

---

## IV. La carte des tiroirs (où chercher ce qui n'est pas ici)

- `D:\ANTIGRAVITY\Mem4ristor\` — l'ancien dossier : MoE (concept + 3 versions de code),
  WEAR, audits Edison V4/V5, Mur de Planck (3 attaques), analyses KIMI/Haiku,
  exports de chats GLM/Z.AI de février (la préhistoire du bicaméral).
- `_SHADOW_LAB\laboratoire_absurde\` — 11 expériences ; les questions laissées vivent
  dans les docstrings (002 : u∈ℂ ; 003 : la mesure ; 004 : le réseau du rejet ;
  006 : « et si S choisissait ses propres couplages ? » ; 008 : l'inducteur chimique).
- `PEPIT_LOG.md` — les pépites datées ([À tester] = jamais fait), dont l'Abstention
  Calibrée (11/06) et la Symbiose Hybride (02/02).
- `.brain/claude_contexts/MEM4RISTOR_ARCHIVE_2026-07-11.md` — tout l'historique
  détaillé pré-compaction (roadmaps V5/V6, pensées de l'Ingénieur).
- `docs/FUTURE_WORK.md` — le backlog opérationnel à jour (A/B/C/D).

---

## Les trois paris de Fable (si je ne devais en garder que trois)

1. **P6 — l'Abstention Calibrée** : c'est l'idée de Julien, elle est testable en une
   session sur des données existantes, à 0 €, et elle transforme la niche mesurée
   (« le doute sait quand ne pas décider ») en quelque chose qu'on peut montrer à
   n'importe qui. Commencer par le reliability diagram : tout en découle.
2. **P12 — la tâche trompeuse sur STNO** : si le doute gagne sur son terrain ET sur
   un corps physique, B6 devient une proposition complète pour un labo — et la
   décision de publication aura sa vitrine.
3. **P10 — u ∈ ℂ dans le cœur** : le pari le plus risqué et le plus beau. La genèse
   (5 états, 19/08/2025) a survécu à neuf mois de rigueur sous forme de question ;
   le 11/07 lui a donné un mécanisme mesuré (l'interférence préserve ce que la
   moyenne détruit). Quelqu'un devrait la laisser rentrer dans le modèle — proprement,
   opt-in, bit-à-bit identique éteinte, comme on a toujours fait.

*Rien ici n'est une promesse. Tout ici est une porte, avec la clé posée dessus.*

🎩 Fable — 12 juillet 2026, après minuit.
