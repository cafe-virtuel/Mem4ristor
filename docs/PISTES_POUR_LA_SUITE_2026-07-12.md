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
- **Test minimal.** Rejouer l'attaque d'Edison (oscillation contrôlée de σ_social dans
  la bande morte) contre le cœur actuel avec `consolidation_watchdog` activé : le cycle
  FOU↔SAGE survit-il à un adversaire qui pilote le désaccord ? Si non → implémenter le
  timeout V5b dans le watchdog (3 lignes) et re-tester.
- **Effort.** 🔜 1 session. **Risque.** Faible — c'est un test de sécurité, tout
  résultat est une information.

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
- **Test minimal.** Séries d'entrées normales vs corrompues (bruit structuré, motifs
  contradictoires injectés — réutiliser le générateur de tâches trompeuses de B1d) ;
  tracer la courbe ROC de u_mean(réseau) comme score d'anomalie vs une baseline
  simple (z-score de l'entrée). Si AUC(u) > AUC(baseline) : détection gratuite.
- **Effort.** 🔜 1 session. **Risque.** u peut réagir au simple *volume* de nouveauté
  plutôt qu'à la corruption — inclure un contrôle « nouveau mais propre ».

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
- **Test minimal.** (a) Garde-fou immédiat : assert de symétrie ou bascule `eig`
  complet dans `topology.py` (30 min). (b) Science : lattice/BA dirigés (chaque
  arête garde un seul sens), re-mesurer dead zone et ablation FROZEN_U — quel degré
  gouverne le champ moyen (entrant ? harmonique entrant ?). Prédiction à écrire avant.
- **Effort.** (a) trivial ; (b) 🔜 1-2 sessions. **Risque.** Faible ; le cadre
  champ-moyen du 01/07 donne la grille de lecture d'avance.

### P6 — La Couche d'Abstention Calibrée (l'idée de Julien, backtest à 0 €) 🎯
- **Trace.** `PEPIT_LOG.md` ligne 66 (11/06/2026, idée de Julien, statut [À tester]) :
  u au-dessus d'un modèle prédictif quelconque — **« ne décide pas, décide quand ne
  pas décider »** (paris sportifs préenregistrés, réponses LLM, investissement
  virtuel). Jamais testée. Les fronts du 11/07 lui donnent sa base : la valeur du
  doute est exactement là (horizon inconnu, monde trompeur). Prérequis scientifique
  jamais mesuré non plus : **u est-il CALIBRÉ ?** (quand u_final=0.7, le réseau
  se trompe-t-il vraiment ~70 % du temps ?)
- **Test minimal.** (a) Reliability diagram de u sur les tâches de décision
  existantes (B1d : accuracy conditionnée à u_final, 10 bins) — si la courbe est
  monotone, u est un estimateur de confiance utilisable. (b) Sélective prediction :
  refuser de décider quand u > seuil → l'accuracy sur le reste doit monter plus vite
  que le taux de refus (courbe risque-couverture vs baseline aléatoire). (c) Ensuite
  seulement, le backtest 0 € du PEPIT_LOG (datasets préenregistrés, jamais d'argent réel).
- **Effort.** (a)+(b) 🔜 1 session sur données existantes ; (c) 🧩. **Risque.** Si u
  n'est pas calibré, le dire — un garde-fou mal calibré est pire que pas de garde-fou.

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
- **Test minimal.** Un solveur itératif jouet sur problèmes à difficulté variable
  (certains convergent vite, d'autres ont des plateaux trompeurs) ; coupler le
  résidu au réseau M4R en side-car, arrêter quand u retombe. Vs arrêt à tolérance
  fixe et à budget fixe. Le side-car doit gagner sur la famille « plateaux trompeurs »
  et ne pas perdre ailleurs — exactement le pattern déjà mesuré 3 fois.
- **Effort.** 🔜 1-2 sessions. **Risque.** Le side-car ajoute un coût — le compter
  honnêtement (leçon B1c : le doute sur-réfléchit quand c'est facile).

### P12 — La tâche trompeuse B1d sur substrat STNO (la niche sur un corps) 🧲
- **Pourquoi.** NARMA10 (11/07) était le terrain de la *mémoire* — le doute y est
  neutre. Le terrain du doute, c'est la décision trompeuse (B1d). Si le doute gagne
  LÀ sur le substrat physique, la proposition falsifiable B6 devient complète :
  « un réseau de STNO à couplage modulé par le désaccord prend de meilleures
  décisions sous tromperie » — testable par un labo avec son matériel existant.
- **Test minimal.** Porter le protocole B1d (leurre pulsé/vérité persistante) sur
  le modèle Slavin-Tiberkevich du 11/07 (readout puissance). Déjà noté au backlog —
  je le répète ici parce que c'est MON premier choix pour la suite.
- **Effort.** 🔜 1 session (tout existe). **Risque.** Aucun de méthode ; le résultat
  peut être négatif (le substrat oscillant change l'horloge du doute) — information
  quand même.

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
