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
- **🟡 FAIT AUX TROIS QUARTS le 13/07/2026** (`experiments/p2_moe_certainty_router_poc.py`,
  tâche B1d canonique + compas COMPOSITE de P6b réutilisés tels quels, split
  CALIB=18 seeds/HOLDOUT=6 seeds pour éviter exactement le piège annoncé, sweep
  rho=coût_coûteux/B_cheap ∈ {2..100}, critère pré-fixé : domine ALWAYS-CHEAP en
  accuracy ET ALL-EXPENSIVE en coût à accuracy≥90 % sur HOLDOUT jamais vu au choix
  du seuil). **Le mécanisme est réel et stable, mais plafonne sous la barre
  auto-imposée.** Sur le split original : HOLDOUT acc=86.7 % (cible 90 %), domine
  ALWAYS-CHEAP de +46.7 pts (86.7 vs 40.0 %) et bat ALL-EXPENSIVE en coût à tout
  ratio ≥2. **Contrôle de robustesse (9 splits aléatoires indépendants)** :
  accuracy HOLDOUT stable à 87.0 %±2.9 pts, bat ALWAYS-CHEAP dans 9/9 cas, mais
  n'atteint la cible 90 % que dans 2/9 — un plafond REPRODUCTIBLE, pas un artefact
  d'échantillonnage. Le routage par certitude fonctionne (le signal composite de
  P6b transporte hors de son harness d'origine) ; le critère à 90 % était sévère
  et n'est pas atteint tel quel.

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
- **❌ RÉFUTÉ le 13/07/2026** (`experiments/p4_wear_drift_poc.py`, extension du
  quatuor photonique 12/06 avec drift multiplicatif (t/t_ref)^(-ν) sur la
  transmission GST ET sur D_eff (proxy RRAM), ν∈{0.05, 0.10} plage littérature
  PCM (Ielmini & Lacaita 2008), elapsed_hours 1→10⁶, ablation FULL/FROZEN_U
  standard du projet, critère pré-fixé : FULL dévie plus tard que FROZEN_U à
  m=3). **PAS DE VIEILLISSEMENT GRACIEUX — c'est l'inverse qui est vrai.** À
  m=3 (fonctionnel), FULL dévie PLUS TÔT que FROZEN_U aux deux ν testés
  (ν=0.05 : FULL h=100 vs FROZEN_U h=10 000 ; ν=0.10 : FULL h=10 vs FROZEN_U
  h=100). Mécanisme identifié : le doute adaptatif RÉAGIT au drift — H_cont
  dérive de façon monotone et croissante avec les heures écoulées (+0.24 à
  +0.62 bits à h=10⁶) — alors qu'un u figé maintient un filtre de couplage
  constant et voit sa dérive de H_cont rester contenue plus longtemps, avant
  d'être finalement rattrapé par la désynchronisation (dsync jusqu'à −0.65 à
  h=10⁶). Nuance à ne pas survendre : à m=5 (dead zone) et ν=0.10, le sens
  s'inverse (FULL dévie à h=100, FROZEN_U à h=10) — hors du critère pré-fixé
  (qui ciblait m=3). Résultat négatif honnête : l'adaptation du doute
  n'amortit pas le vieillissement lent en régime fonctionnel, elle
  l'AMPLIFIE.

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
- **✅ (b) FAIT le 13/07/2026** (`experiments/p5b_directed_graphs_poc.py`,
  conversion lattice/BA en dirigés — convention vérifiée dans le code même
  (`_rebuild_laplacian`) : A[i,j]=1 signifie "i écoute j" —, test contrastif
  HUBS_LISTEN vs HUBS_BROADCAST à topologie fixée + corrélation globale
  k_harm_in/k_harm_out vs H_cont sur 150 runs dirigés, ablation FROZEN_U/FULL,
  vérification du garde-fou P5a). **Prédiction confirmée sur le fond, réfutée
  sur le mécanisme anticipé.** Corrélation globale : H_cont vs k_harm_in
  rho=−0.884 (p=1e-50) contre k_harm_out rho=−0.168 (p=0.04) — **le degré
  ENTRANT domine massivement**, exactement comme prédit (seule l'équation de
  couplage d'un nœud dépend de qui IL écoute). Mais le test contrastif
  directionnel est RÉFUTÉ : HUBS_LISTEN a un H_cont plus HAUT (4.204) que
  HUBS_BROADCAST (3.790) à m=3 — l'inverse de l'intuition "les hubs qui
  écoutent beaucoup moyennent davantage". Mécanisme réel découvert a
  posteriori : HUBS_LISTEN prive la majorité des nœuds périphériques de leur
  droit d'écoute (41 % de nœuds totalement SOURDS, in-degree=0, contre 1.2 %
  sous HUBS_BROADCAST) — ces nœuds sourds deviennent des oscillateurs FHN NON
  COUPLÉS, ce qui AUGMENTE la diversité globale malgré la forte intégration
  des quelques hubs restants ; le k_harm_in bas de HUBS_LISTEN (2.07 vs 2.78)
  reflète cette population majoritairement sourde et reste cohérent avec la
  corrélation globale. Garde-fou P5a vérifié OK (ValueError levée comme
  prévu). Ablation FROZEN_U/FULL : **le résultat central SURVIT en dirigé**
  (sync FROZEN_U ≫ FULL à m=3 : 0.105 vs 0.024 ; à m=5 : 0.274 vs 0.015).

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
- **✅ (c) FAIT le 13/07/2026** (`experiments/p6c_backtest_poc.py`, domaine
  « investissement virtuel » synthétique aux statistiques GENUINEMENT différentes
  de B1d — paramètres de marché (forces, effectifs, durée du faux breakout) TIRÉS
  par essai plutôt que fixes —, 60 épisodes, même critère de succès que P6b :
  r_pb>0.15 ET gain@50 %>+3 pts). **La recette transporte, largement.**
  ALWAYS-TRADE=63.3 % ; COMPOSITE_CV ré-appris ici : **+30.0 pts @50 % couverture
  (63.3→93.3 %), r_pb=+0.463** — le meilleur résultat d'abstention du projet à ce
  jour, au-delà même de P6b. Le signal individuel gagnant de B1d (conf_u_inv)
  transporté TEL QUEL fonctionne aussi ici (+13.3 pts) mais nettement moins bien
  que le composite ré-appris — la recette (signaux + CV) ajoute une vraie valeur
  au-delà d'un signal figé, même quand les deux « marchent ».

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
- **❌ PARITÉ (négatif honnête) le 13/07/2026** (`experiments/p7_chemical_inductor_poc.py`,
  mécanisme extrait de 008 — un filtre passe-bas dL/dt=β(signal−L), débarrassé du
  décor majorité/minorité — appliqué EN POST-TRAITEMENT au signal side-car M4R_SIG
  du harness P11 (aucune modification du cœur), β∈{0.01, 0.03, 0.1, 0.3}, mêmes
  24 problèmes/protocole que P11). **M4R_L = M4R_SIG exactement** (succès trap
  0.00 pour les deux, TOUS les β testés) : l'inertie appliquée au signal RAPIDE
  (|Lv| lissé) ne récupère PAS la robustesse du signal LENT natif (M4R_U : succès
  trap 1.00). L'« inductance chimique » de 008 ne substitue pas à l'intégration
  temporelle propre de u (ε_u adaptatif, τ_u) — un simple filtre passe-bas externe
  n'équivaut pas au mécanisme natif. Résultat négatif honnête, mécanisme du Labo
  n'apporte rien ici tel quel.

### P8 — Le bruit coloré pour intégrateur adaptatif (dette numérique) 🔧
- **Trace.** Rapport Manus (01/05/2026, `🚀 Rapport de Validation…V5.md`, reco
  technique 2) : un **générateur de bruit interpolé** (bruit de couleur) rendrait le
  bruit compatible avec RK45 adaptatif — jamais fait. Aujourd'hui le projet vit avec
  « Euler formellement instable mais validé empiriquement » + interdiction
  RK45+bruit (`dynamics.py`). Lever cette réserve ferme définitivement la porte du
  reviewer numérique.
- **🟡 FAIT AUX TROIS QUARTS le 12/07/2026** (`experiments/p8_colored_noise_rk45_poc.py`,
  bruit OU exact interpolé calibré à même densité spectrale S(0)=σ_v², RHS répliqué
  depuis `solve_rk45` du cœur avec **gate de fidélité 6×10⁻¹⁵**, τ ∈ {0.4, 0.1,
  0.025, 0.00625}, 4 seeds, FULL/FROZEN, critères pré-fixés ; périmètre = la
  dynamique réduite que le cœur lui-même déclare intégrable par RK45, PAS le
  pipeline step() complet ; aucun chiffre canonique touché). **Verdict :**
  1. **H_cont converge exactement** : RK45+OU(τ=0.00625) = 1.6214 vs Euler+blanc
     1.6223 en FULL (dH=−0.0009, bien dans 2sd) ; idem FROZEN. L'observable de la
     famille Table 1 ne dépend ni de la couleur du bruit ni de l'intégrateur.
  2. **La sync du réseau VIVANT converge monotonement** (0.121→0.093→0.079→0.077
     vs 0.072) mais garde un résidu **+6.6 %** au τ le plus fin — le critère strict
     (2sd inter-seeds = 0.0018, très serré) n'est PAS atteint. Bruit additif ⇒
     Itô=Stratonovich, la convergence exacte est attendue théoriquement ; la
     fermeture demanderait τ ≤ 0.0016 (coût ×4 par étape) ou une analyse d'ordre.
  3. **L'ablation centrale survit dans les 18 configurations** (sync FROZEN >
     FULL partout) — le résultat central est robuste à tout ce qui a été varié.
  4. **Fait quantifié en passant** : Euler à dt=0.05 sur un bruit τ<dt est
     ALIASÉ (H_cont 2.25 au lieu de 1.62 !) — un intégrateur à pas fixe déforme
     un bruit plus rapide que son pas ; c'est exactement l'artefact que la
     réserve craignait, et il vit du côté « bruit trop rapide », pas du côté
     des paramètres canoniques (τ_blanc effectif = dt).
  Sensibilité réelle découverte : le réseau vivant (FULL) est sensible à la
  couleur du bruit (τ=0.4 : sync +65 %) — le gelé non. À retenir pour le hardware
  (le bruit physique n'est jamais blanc).
- **Reste.** Fermer le résidu sync-FULL (τ plus fin ou ordre de convergence) ;
  étendre au pipeline step() complet (hysteresis/plasticité) si la Table 1 doit
  un jour être certifiée RK45. **Effort.** 🧩.
- **🟡 MARCHE FAITE le 13/07/2026** (`experiments/p8_closure_poc.py`, un 5e point
  τ=0.0025 (RK45_OU, 4 seeds × 2 ablations, ~2,5× plus fin que le précédent) +
  analyse d'ordre log-log sur les 5 points désormais disponibles). **FROZEN_U
  ferme** (résidu +0.0003 < tolérance 0.0246). **FULL toujours HORS tolérance**
  (résidu +0.0028 vs 2sd=0.0018) mais la régression log-log est PROPRE sur les
  5 points (ordre ajusté p=0.56, R²=0.976 — la "décélération" apparente vue à 4
  points au 12/07 était du bruit d'échantillonnage, pas un plancher). Extrapolation
  à τ=1e-5 : résidu prédit ~0.00012, bien sous tolérance. **Verdict : pas encore
  fermé, mais l'analyse d'ordre confirme que c'est une question de budget de
  calcul (τ≤0.0016 comme prévu, coût ×4/pas), pas un biais structurel entre le
  RHS réduit et Euler+blanc.** Reste : pousser à τ≤0.0016 (coûteux) ou accepter
  la fermeture par extrapolation documentée ; étendre au pipeline step() complet
  reste non fait.

### P9 — Visualisation des flux d'entropie (le TODO le plus ancien) 📊
- **Trace.** Roadmap V5 point 6 (jamais fait) + reco Manus 3. Voir la décision se
  prendre (flux d'information entre régions/couches au fil du temps).
- **Effort.** 💭 1 session, valeur = compréhension et communication (labo, vidéo).
- **✅ FAIT le 13/07/2026** (`experiments/p9_entropy_flow_viz_poc.py`, réutilise
  `calculate_transfer_entropy` déjà établie du projet — défenses Reviewer2, « TE
  causalité de u » — pour visualiser un FLUX SPATIAL au cours du temps plutôt
  qu'un chiffre unique de causalité globale). Stimulus injecté dans un quadrant
  Q1 d'un lattice 10×10 ; kymographes spatiaux + courbes TE(Q1→Qi) glissantes +
  graphe de flux à 3 instants. Vérification de cohérence physique (pas un
  claim) : les voisins directs (Q2, Q3) reçoivent plus de flux tôt que la
  diagonale (Q4) — cohérent. Valeur = communication/compréhension (labo,
  vidéo), la voie qui manquait depuis la Roadmap V5.

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
- **✅ V1 AU CŒUR le 12/07/2026, avec l'accord explicite de Julien** (commit du cœur :
  `_step_complex_doubt` dans dynamics.py, opt-in `complex_doubt.enabled=False` par
  défaut, **bit-à-bit identique éteint** — 5 tests dédiés dont l'interférence
  destructive unitaire, suite 128 passed + 2 xfail, Guardian 14/14 après le commit).
  Contenu V1 : cible locale SIGNÉE (k_u·Lv au lieu de k_u·|Lv|), interférence
  sociale (γ_int·(moyenne complexe des u_c voisins − u_c)), porte ω_u posée
  (rotation, défaut 0 → phases {0, π}). POC : `experiments/p10_complex_doubt_poc.py`.
  **Deux verdicts francs :**
  1. **La capacité nouvelle EXISTE — mémoire directionnelle dans la phase** :
     après un pulse signé, Re(u_c) décode la direction à 100 % (D≤200 pas), puis
     0.85-0.95 de D=900 à 2400 pas (~12 τ_u) — pendant que le u scalaire est au
     hasard dès D=400 (0.53) et que v oscille et s'estompe (0.72). Fenêtre aveugle
     D=400-600 (le rebond adaptatif FHN inverse transitoirement la cible — la
     dynamique du rebond est à caractériser). Convention de signe documentée :
     b est gravé en −Re(u_c[stim]) (le désaccord laplacien d'un nœud au-dessus
     du consensus est négatif).
  2. **Le coût est réel et EXPLIQUÉ — critère d'ablation ÉCHOUÉ tel quel** :
     sync(COMPLEX)=0.35 > 0.5·sync(FROZEN)=0.28 (scalaire : 0.25). Cause double,
     mesurée par ablation γ_int=0 (0.32) : l'interférence ET la cible signée.
     Racine mathématique : par Jensen, |filtré(signé)| ≤ filtré(|signé|) — le
     doute complexe est structurellement plus PETIT que le scalaire → couplage
     plus attractif. **La mémoire directionnelle se paie en anti-synchronisation.**
- **Reste (la marche suivante).** ω_u > 0 + lecture globale de phase → la parité
  multiplicative du jouet genèse (une dynamique relaxante à phases {0, π} ne peut
  pas la porter, documenté dans le POC) ; caractériser le rebond D=400-600 ;
  explorer le compromis γ_int (mémoire vs anti-sync). **Effort.** 🧩 1-2 sessions.
- **✅ MARCHE FAITE le 13/07/2026** (`experiments/p10_next_steps_poc.py`, coeur
  déjà accordé le 12/07 — γ_int et ω_u sont déjà des clés de config exposées,
  AUCUNE nouvelle modification de dynamics.py). **Trois résultats :**
  1. **Compromis γ_int : le défaut (0.15) n'était PAS optimal.** Sweep
     γ_int∈{0, 0.05, 0.15, 0.3, 0.5, 1.0} à D=1200 fixe : **γ_int=0 (interférence
     éteinte, cible signée seule) donne la MEILLEURE mémoire (0.88, égale au
     défaut) ET le MEILLEUR ratio d'anti-sync (0.579 vs 0.633 au défaut)** —
     cohérent avec le diagnostic COMPLEX_NOINT de V1 (le surplus de sync venait
     de l'interférence). À γ_int=1.0, tout s'effondre (mémoire 0.58, ratio→1.0,
     plus d'anti-sync du tout).
  2. **Le rebond D=400-600 EXPLIQUÉ** : trace fine autour du pulse — v change de
     signe dès t=243 (rebond adaptatif massif, creux à ~−0.6), traverse zéro à
     nouveau vers t~780 ; **Re(u_c) reste POSITIF tout du long de la fenêtre
     aveugle** (+0.177 à t=600, +0.066 à t=800) — la mémoire de phase ne suit
     PAS le rebond de v, elle en est mécaniquement indépendante. C'est
     exactement pourquoi PHASE_UC survit là où V_STATE échoue.
  3. **ω_u global (test PARTIEL, périmètre explicitement limité)** : sweep
     ω_u∈{0, 0.005, 0.02, 0.05, 0.1} sur la mémoire à 4 délais — **aucune
     amélioration** (les valeurs élevées dégradent fortement, ex. D=600 chute
     à 0.04-0.12). Confirme l'attente de la piste : une rotation GLOBALE
     (même vitesse pour tous les nœuds) ne porte pas plus d'info qu'un doute
     réel à phases {0,π}. **La parité multiplicative complète resterait hors
     de portée sans rotation PAR GROUPE — une extension du cœur au-delà du
     fork actuel, donc un NOUVEL accord explicite de Julien serait requis
     avant de la tenter.**
- **ROTATION PAR GROUPE FAITE le 13/07/2026, accord explicite de Julien
  (« P10 svp »)** — extension du coeur, une seule ligne (`_step_complex_doubt` :
  `omega_u = np.asarray(cd.get('omega_u', 0.0), dtype=float)` au lieu de
  `float(...)`). Un scalaire garde le comportement V1 **bit-a-bit identique**
  (`test_omega_scalar_equals_uniform_array`) ; un array de taille N permet une
  vitesse de rotation PAR NOEUD/GROUPE, teste en isolation deterministe
  (`test_omega_per_group_diverges`, sigma_v=0 pour couper le bruit stochastique
  qui masquait sinon l'effet). Suite 130 passed + 2 xfail, Guardian 14/14
  apres commit. POC : `experiments/p10_group_rotation_poc.py`. Protocole :
  2 groupes de 30 noeuds pulses SIMULTANEMENT (bits b_A, b_B independants),
  groupe A ancre sur le canal V1 (omega_A=0), groupe B recoit
  omega_B in {0, 0.02, 0.05} et un readout DE-ROTATE (`Re(u_c . e^{-i omega_B T})`).
  **Trois resultats, aucun arrondi :**
  1. **Crosstalk REEL, NON repare par la separation de frequence.** A seul :
     accuracy 1.000 (D=1200). A+B simultanes, meme canal (omega_B=0) : 0.812
     (-0.188). Separer les frequences ne repare PAS ce crosstalk dans la
     plage testee -- omega_B=0.02 -> 0.771, omega_B=0.05 -> 0.792, toujours
     sous le canal partage. L'hypothese « la frequence protege du melange »
     est REFUTEE ici : l'interference sociale gamma_int melange les voisins
     par la TOPOLOGIE, pas par la frequence -- deux groupes adjacents sur le
     lattice se perturbent quelle que soit leur vitesse de rotation respective.
  2. **Decode B non monotone en omega_B** : 0.833 (omega_B=0, readout brut)
     -> 0.938 (omega_B=0.02, meilleur point) -> 0.750 (omega_B=0.05, degrade).
     Pas de regle simple « plus de rotation = mieux decode » -- coherent avec
     une dynamique qui n'est PAS un rotor libre (le rappel vers la cible reelle
     rivalise avec la rotation, cf. le rebond adaptatif documente au-dessus).
  3. **Parite globale (le test genese) : AUCUN gain, et ce n'est pas un
     hasard -- c'est structurel.** A chaque omega_B teste, `parite_separee` et
     `parite_globale` sont IDENTIQUES (0.812/0.812 au meilleur point). Raison
     mathematique trouvee apres coup : quand chaque groupe est d'abord reduit
     a UN SEUL angle agrege (moyenne sur 30 noeuds), `signe(cos theta_A) . signe(cos theta_B)`
     et `signe(cos(theta_A+theta_B))` sont une IDENTITE des que theta_A, theta_B
     sont proches de {0, pi} -- pas une question empirique. Le gain de la genese
     (11/07, moyenne complexe bat le vote +5.5 pts) venait de combiner N=5
     unites individuellement bruitees PAR INTERFERENCE AVANT tout seuillage, pas
     de comparer deux angles deja agreges. **Ce test-ci n'a donc pas isole le
     bon mecanisme** -- la vraie marche suivante serait un decode PAR NOEUD (30
     votes individuels par groupe) comparant majorite-de-signes vs
     interference-complexe-avant-signe, la structure exacte de la genese,
     jamais tentee sur le reseau physique. Non lance cette session (nouveau
     perimetre, a proposer).
  **Bilan honnete de la marche.** Le coeur fait ce qu'on lui demande (rotation
  par groupe verifiee, mecaniquement saine, tests + Guardian intacts). Les
  DEUX hypotheses pratiques testees (crosstalk repare par frequence ; parite
  mieux lue globalement) sont refutees ou non concluantes SUR CE PROTOCOLE
  PRECIS -- et la raison de l'echec du point 3 est elle-meme un resultat :
  agreger avant de comparer detruit exactement le mecanisme qu'on voulait
  tester.
- **✅ VOTE vs INTERFERENCE FAIT le 13/07/2026, meme jour** (`experiments/p10_vote_vs_interference_poc.py`,
  accord de Julien : « oui decodage par noeuds c'est interessant »). Reprend
  le vrai mecanisme relu dans `genesis_five_states_poc.py` (pas de memoire) :
  R2 (global) = signe(cos(SOMME des phases dominantes)) = produit des
  phaseurs, la definition meme de la parite ; le "vote" de la genese est la
  majorite des N=5 BITS INDIVIDUELS, structurellement plafonnee (68.75% a
  N=5, fait mathematique). Notre reseau n'a pas 5 unites-un-bit-chacune ; il
  a 30 noeuds redondants portant chacun une estimation bruitee du MEME bit --
  question differente mais legitime : moyenner-puis-seuiller (interference)
  bat-il seuiller-chaque-noeud-puis-voter (vote) ? Premiere fois que cette
  comparaison est reellement isolee (le POC precedent agregeait deja des
  DEUX cotes de la comparaison, ce qui l'annulait par construction).
  *(Un bug de convention de signe a d'abord donne 10% d'accuracy en Partie 1 --
  `mask - idle` au lieu du `idle - mask` etabli par V1 ; corrige, verifie
  contre le POC precedent avant de conclure quoi que ce soit.)*
  **Deux resultats, une explication qui tient :**
  1. **Solo (1 bit, 1 groupe, 40 problemes, D=1200) : AUCUNE difference.**
     VOTE=0.925±0.042, INTERFERENCE=0.900±0.048 (delta −0.025, sous le
     bruit). En regime propre (pas de crosstalk), les deux methodes
     saturent deja pres du plafond -- rien pour les departager.
  2. **Parite (2 groupes simultanes, 48 problemes, omega_B=0) : INTERFERENCE
     bat VOTE nettement.** VOTE=0.542±0.073 (a peine au-dessus du hasard),
     INTERFERENCE=0.812±0.057 (delta +0.271, tres au-dela de 2 SE). Retro-
     calcul : l'accuracy PAR GROUPE sous VOTE en regime crosstalk tombe a
     ~0.645 (contre 0.812 pour INTERFERENCE au meme point, cf. marche
     precedente) -- le seuillage dur PAR NOEUD jette de l'info (magnitude,
     confiance) que le crosstalk a deja abimee, et cette perte se compose
     MULTIPLICATIVEMENT une fois qu'on prend le produit de deux decisions
     dures. L'interference (moyenne complexe, un seul seuillage final)
     retarde la decision et absorbe une partie du bruit avant de trancher.
  **Bilan.** Le mecanisme de la genese (combiner avant de seuiller) SE
  TRANSFERE au reseau physique, mais **conditionnellement au regime** --
  invisible en decode propre (Partie 1, comme B1c/B1d l'avaient deja montre
  pour le doute scalaire), decisif des que le signal est degrade par le
  crosstalk (Partie 2). Encore une instance du fil rouge du projet : un
  mecanisme n'aide pas dans l'absolu, il aide **quand les choses se
  compliquent**.
- **ROTATION PAR GROUPE / GAMMA_INT REPARE LE CROSSTALK, FAIT le 13/07/2026,
  meme jour** (`experiments/p10_gamma_int_crosstalk_poc.py`, accord de
  Julien : « oui continuons svp »). Relie deux fils de la meme journee : ce
  matin, `p10_next_steps_poc.py` avait trouve gamma_int=0 (interference
  sociale coupee) MEILLEUR que le defaut 0.15 pour la memoire a UN SEUL
  pulse (memoire egale, anti-sync meilleur). Le crosstalk mesure cet
  apres-midi passe-t-il PAR ce meme canal ? Sweep gamma_int in
  {0, 0.05, 0.15 (defaut), 0.3, 0.5}, D=1200, omega_B=0, readouts
  INTERFERENCE et VOTE en parallele. **Reponse nette : OUI.**

  | gamma_int | solo A (int/vote) | A+B actif (int/vote) | crosstalk (int/vote) |
  |---|---|---|---|
  | **0.0**   | 0.925 / 0.925 | 0.958 / 0.896 | **+0.033 / -0.029** (nul) |
  | 0.05      | 0.950 / 0.975 | 0.854 / 0.771 | -0.096 / -0.204 |
  | 0.15 (def)| 0.900 / 0.925 | 0.812 / 0.729 | -0.088 / -0.196 |
  | 0.3       | 0.825 / 0.900 | 0.750 / 0.771 | -0.075 / -0.129 |
  | 0.5       | 0.850 / 0.800 | 0.750 / 0.688 | -0.100 / -0.113 |

  - **gamma_int=0 est le SEUL point ou A+B actif n'est pas degrade par
    rapport a A seul** (les deux readouts s'accordent : crosstalk quasi nul,
    la ou TOUT gamma_int>0 teste montre une degradation systematique et
    reproductible ~-0.08 a -0.20 pts). Confirme mecaniquement l'hypothese :
    gamma_int (moyenne complexe des u_c VOISINS, y compris inter-groupes
    sur le lattice) est bien le canal par lequel un groupe actif perturbe
    son voisin -- pas le couplage spatial direct v/w, qui reste identique
    a tout gamma_int.
  - **Troisieme confirmation independante que gamma_int=0 domine le defaut
    0.15 pour un usage memoire** (apres la memoire seule et l'anti-sync
    ce matin) : ici, en plus d'annuler le crosstalk, la parite VOTE bondit
    de 0.542 (defaut) a 0.729 (gamma=0) -- la lecture la plus fragile est
    celle qui beneficie le plus de couper le canal qui la corrompait.
    Parite INTERFERENCE : 0.812 -> 0.854, amelioration reelle mais plus
    modeste (le readout deja robuste avait moins a gagner).
  - **Reserve.** gamma_int=0 desactive ENTIEREMENT le canal d'interference
    sociale de P10 (le "canal nouveau" introduit le 12/07) -- ce n'est pas
    un reglage fin, c'est l'eteindre. Cela n'invalide pas P10 V1 (la memoire
    directionnelle de base ne depend pas de gamma_int), mais suggere que
    le couplage social entre doutes voisins, tel qu'implemente, est un
    COUT net pour toute tache de memoire/multiplexage testee a ce jour
    (single-pulse ce matin, crosstalk cet apres-midi) -- sa valeur reste a
    trouver ailleurs (peut-etre la synchronisation/consensus, jamais
    testee comme benefice direct de gamma_int>0).
  **Marche suivante possible (non tentee) :** re-tester la separation de
  frequence (omega_B>0) A gamma_int=0 -- p10_group_rotation_poc.py l'avait
  refutee au defaut 0.15, jamais essayee sur un canal deja assaini.
- **"LA FLEMME" (intuition de Julien) TESTEE le 13/07/2026, meme jour**
  (`experiments/p10_flemme_frontiere_poc.py`, accord : « on continue » +
  caracterisation textuelle « ce gamma_int c'est la flemme en quelque
  sorte »). Lecture proposee : au lieu de calculer honnetement sa propre
  cible locale (k_u . laplacian_v, le desaccord REEL avec SES voisins v),
  un noeud a gamma_int>0 se cale PARTIELLEMENT sur la moyenne de ses
  voisins u_c, quelle que soit la pertinence de ce qu'ils portent --
  "flemme" au sens propre : copier plutot que calculer.
  **Prediction posee AVANT de lancer** : si la flemme est un effet de
  PROXIMITE, le cout de gamma_int devrait etre concentre aux noeuds A LA
  FRONTIERE directe entre A et B (voisins lattice en desaccord de bit),
  et nul aux noeuds INTERIEURS (les 4 voisins lattice tous dans le meme
  groupe/idle -- copier un voisin d'accord ne coute rien).
  **REFUTEE TELLE QUELLE.** Lecture par noeud (2880 noeuds-problemes,
  gamma_int in {0.15, 0.0}) : accuracy INTERIEUR et FRONTIERE quasi
  identiques a chaque gamma_int (0.653 vs 0.658 au defaut ; 0.632 vs 0.670
  a gamma=0 -- ecarts dans le bruit, ±0.02 SE, aucune tendance coherente).
  Le cout de gamma_int N'EST PAS un effet de premier voisin.
  **Explication qui tient** : gamma_int s'applique a CHAQUE pas pendant
  1400 pas (B_PULSE+DELAY) ; sur un lattice periodique de diametre ~10,
  la contamination a largement le temps de DIFFUSER a tout le reseau par
  sauts successifs -- un noeud sans aucun voisin direct dans l'autre
  groupe est quand meme touche des que SON voisin a ete lui-meme influence
  par un voisin plus loin. "La flemme" de Julien est confirmee au niveau
  MECANISME (couper gamma_int repare le crosstalk, section precedente) et
  au niveau GROUPE (le vote/l'interference du groupe entier en patissent),
  mais elle n'est pas contagieuse par CONTACT -- elle est contagieuse par
  DIFFUSION GLOBALE, le temps d'exposition (nombre de pas) comptant plus
  que la distance topologique au groupe adverse. Cout par noeud (~0.65)
  nettement sous l'accuracy de GROUPE (0.81-0.96 en interference, 0.73-0.90
  en vote) -- attendu, un seul noeud est un estimateur bien plus bruite
  que 30 combines (cf. la marche vote-vs-interference).
- **SAGESSE DES FOULES (le test le plus favorable a gamma_int) FAIT le
  13/07/2026, meme jour** (`experiments/p10_sagesse_foules_poc.py`).
  Verdict de Julien apres 3 tests defavorables : « la flemme ne rapporte
  rien et detruit tout -- comme une personne qui aurait la flemme d'aller
  bosser : pas de travail, pas d'argent, pas d'argent, vie difficile ».
  Avant de valider cette conclusion SANS reserve, un point de methode :
  les 3 tests precedents opposaient tous des voisins en DESACCORD (bits
  opposes) -- copier un voisin qui pense le contraire coute, forcement.
  Le cas jamais teste, le plus favorable possible a gamma_int : des voisins
  qui PARTAGENT le meme bit vrai mais avec une confiance individuelle
  inegale (sagesse des foules -- un noeud a signal faible/ambigu qui
  capte la moyenne de voisins confiants ET D'ACCORD ne devrait rien
  perdre a le faire). Protocole : 1 groupe de 30 noeuds, bit b_a PARTAGE
  par tous, 15 noeuds a stimulus FORT (echelle 1.0), 15 a stimulus FAIBLE
  (echelle 0.15, quasi noye dans le bruit sigma_v=0.05), interleaves sur le
  lattice. Sweep gamma_int in {0, 0.05, 0.15, 0.3, 0.5}, D=1200, 40
  problemes/gamma_int (20 seeds x 2 signes), lecture par noeud (metrique
  primaire pre-enregistree) ET par sous-groupe (vote/interference).
  **Verdict, metrique primaire (noeud faible individuel) : AUCUN gain net.**
  gamma_int=0 -> 0.537 ; meilleur point (gamma_int=0.15) -> 0.583
  (+0.047, largement sous le bruit d'echantillonnage a n=40, SE~0.08).
  Le noeud fort ne bouge pas non plus de facon coherente (0.590-0.643,
  aucune tendance). **Nuance honnete (metrique secondaire, non pre-
  enregistree, a ne pas survendre)** : le VOTE du sous-groupe faible
  montre un pic suggestif a gamma_int=0.15 (0.575 -> 0.675 -> 0.725 puis
  redescend a 0.625/0.500 aux gamma_int plus eleves) -- mais ce pic NE SE
  REPLIQUE PAS sur l'INTERFERENCE du meme sous-groupe (0.450/0.350/0.550/
  0.550/0.575, aucun motif clair, souvent proche ou sous 0.5). Deux
  lectures du meme phenomene qui ne s'accordent pas = signal probablement
  bruite, pas confirme. **Conclusion : sur les 4 tests du 13/07 (memoire,
  crosstalk, diffusion, sagesse des foules -- y compris le cas le PLUS
  favorable a gamma_int qu'on pouvait construire), gamma_int>0 n'a JAMAIS
  rapporte de gain net et confirme.** L'intuition de Julien tient : ce
  canal, tel qu'implemente, est un cout sans contrepartie mesuree a ce
  jour. Sa valeur (si elle existe) reste a chercher hors du registre
  memoire/decodage -- peut-etre la synchronisation/consensus pur (jamais
  teste comme benefice), domaine que P10 cherche justement a REDUIRE, ce
  qui expliquerait pourquoi gamma_int n'a jamais eu l'occasion de payer
  dans un projet dont l'objectif est l'anti-synchronisation.
- **BATTERIE LARGE (« pousse au maximum, meme sur des criteres hors de
  propos ») FAITE le 13/07/2026, meme jour**
  (`experiments/p10_gamma_int_batterie_large_poc.py`). 4 criteres
  GENUINEMENT DIFFERENTS des precedents, pas une repetition plus profonde :
  1. **Rejet de menteurs** (5/30 noeuds recoivent le stimulus INVERSE, la
     vraie sagesse des foules classique -- different du "signal faible"
     deja teste). Vote : decline MONOTONE avec gamma_int (0.933 -> 0.733).
     Interference : pic NON-monotone a gamma_int=0.3 (0.733 -> 0.833,
     +0.100) -- **seul hint positif de toute la journee**, mais n=30/point,
     SE~0.07, gap ~1.4 SE, ET contredit par le VOTE sur les MEMES donnees
     (qui decline). Signal ambigu, pas confirme.
  2. **Spread final sous stimulus PARTAGE** (le terrain natif litteral de
     gamma_int -- reduire la dispersion aurait du etre son point fort par
     construction). Bug de protocole trouve et corrige en route (l'etat
     initial demarrait a u=0.05 IDENTIQUE pour tous les noeuds -- le
     "consensus" etait deja atteint avant toute dynamique ; corrige par un
     etat initial u_c RANDOMISE). Decouverte apres correction : le spread
     ne DECROIT jamais vers un seuil bas (il CROIT depuis l'init, 0.29 ->
     jusqu'a 0.75 selon le point du temps) -- la metrique "temps jusqu'au
     consensus" ne s'applique pas, remplacee par le spread a horizon fixe
     (1500 pas). **Resultat : spread croit LEGEREMENT avec gamma_int**
     (0.360 a 0 -> 0.319 a 0.05 -> 0.349 -> 0.377 -> 0.399 a 0.5) --
     **meme sur son PROPRE terrain, gamma_int n'homogeneise pas mieux, il
     homogeneise plutot MOINS** au-dela d'un leger creux a 0.05.
  3. **Bruit environnemental amplifie** (sigma_v x4=0.20, x8=0.40 -- le
     regime ou la moyenne statistique a le plus de marge theorique pour
     aider). AUCUN gain a AUCUN niveau de bruit : gamma_int=0 reste le
     meilleur point a sigma_v=0.05 (1.000), 0.20 (0.667) ET 0.40 (0.600),
     et la degradation avec gamma_int croissant reste presente a chaque
     niveau (ex. sigma_v=0.40 : 0.600 -> 0.433 -> 0.267 -> 0.200 -> 0.233).
     Plus de bruit ambiant ne donne PAS plus de raisons a gamma_int d'aider.
  4. **Topologie BA scale-free (m=3)** au lieu du lattice (le maillon
     faible connu du projet depuis B1/B4/B5). AUCUN gain net : gamma_int=0
     (0.875) egale le meilleur point (gamma_int=0.3, aussi 0.875), avec un
     creux entre les deux (0.708 a 0.15) -- meme motif "pas d'amelioration,
     parfois un creux" que sur lattice.
  **BILAN DE LA JOURNEE SUR GAMMA_INT (8 tests, 4 ce matin/debut
  d'apres-midi + 4 ici) : sur AUCUN des 8 criteres testes -- memoire,
  anti-sync, crosstalk de groupe, diffusion locale, sagesse des foules par
  signal faible, rejet de menteurs, homogeneisation sous stimulus partage,
  bruit amplifie, topologie BA -- gamma_int>0 n'a produit un gain
  CONFIRME et REPLIQUE.** Le seul hint (menteurs, lecture interference)
  est contredit par l'autre lecture (vote) des memes donnees. L'intuition
  de Julien (« la flemme ne rapporte rien ») est la lecture la plus
  defendable de la journee entiere. Question qui reste GENUINEMENT
  ouverte (non testee, hors de portee du temps de la session) : un
  benefice pourrait exister sur une tache de VRAI consensus distribue
  (ou l'objectif EST de converger vite vers une reponse partagee, contexte
  differents des taches de memoire/decodage/anti-sync qui structurent
  tout Mem4ristor) -- mais rien dans ce projet n'est construit pour
  valoriser ce cas, ce qui pourrait expliquer honnetement pourquoi
  gamma_int n'a jamais eu l'occasion de payer ici.
- **LE VRAI TEST DE CONSENSUS (Condorcet) FAIT le 13/07/2026, meme jour**
  (`experiments/p10_vrai_consensus_poc.py`, demande de Julien : « peux-tu
  tester cette question justement ? » -- la question laissee ouverte par
  la batterie large). Corrige le defaut de la Partie 2 precedente (signal
  HOMOGENE, aucun desaccord initial a reconcilier) par le vrai protocole
  du theoreme du jury de Condorcet : chaque noeud d'un groupe de 30 tire
  INDEPENDAMMENT son propre signe (Bernoulli p_correct pour le bon signe,
  erreur de MESURE individuelle -- pas des menteurs deliberes). Sweep
  gamma_int in {0, 0.05, 0.15, 0.3, 0.5} x p_correct in {0.6, 0.7, 0.8},
  D=1200, 40 problemes/point.
  **PREMIER RESULTAT POSITIF NET DE LA JOURNEE :**
  | p_correct | gamma_int=0 (interference) | meilleur point | gain |
  |---|---|---|---|
  | 0.6 (bruite) | 0.550 | **0.600** (gamma=0.15, vote: **0.725**, +0.175) | +0.050 (int) / **+0.175 (vote)** |
  | 0.7 (bruite) | 0.625 | **0.750** (gamma=0.3, vote: **0.775**, +0.125) | **+0.125 (int)** / +0.125 (vote) |
  | 0.8 (fiable) | 0.775 | 0.775 (gamma=0.0) | +0.000 (aucun gain) |
  - **gamma_int MODERE (0.15-0.3) aide REELLEMENT le consensus de groupe,
    mais SEULEMENT quand les individus sont BRUITES** (p_correct=0.6-0.7,
    30-40% d'erreur individuelle) -- gain +0.125 a +0.175 pts, coherent
    sur les DEUX lectures (vote ET interference) et sur les DEUX niveaux
    de bruit testes. **Quand les individus sont deja fiables (p_correct=0.8),
    le gain disparait totalement** (0.000) -- rien a corriger, gamma_int
    n'apporte rien de plus que le vote brut.
  - **A gamma_int=0.5 (trop de conformite), l'effondrement est SYSTEMATIQUE
    a TOUS les niveaux de bruit** (vote : 0.475/0.525/0.725, tous EN
    DESSOUS de leur propre gamma_int=0 -- meme a p_correct=0.8 ou gamma_int
    plus modeste ne faisait aucun mal). La courbe est un **sweet spot en
    cloche inversee**, pas une droite : trop peu ne corrige rien, trop
    detruit le signal correct de la majorite elle-meme.
  - **Minorite initialement fausse (par tirage) "corrigee" ?** Petit effet
    positif et REMARQUABLEMENT constant a gamma_int=0.05 sur les 3 niveaux
    de bruit (+0.075 partout, 0.500->0.575, 0.500->0.575, 0.525->0.600) --
    mais s'effondre nettement SOUS le hasard a gamma_int=0.5 (0.325, 0.250,
    0.350) -- meme mecanisme de sweet-spot, a une echelle plus fine.
  **VERDICT DE LA QUESTION OUVERTE : OUI, gamma_int a une vraie niche.**
  Elle n'est ni dans la memoire, ni dans l'anti-crosstalk, ni dans
  l'homogeneisation sous signal partage propre -- elle est dans la
  CORRECTION D'ERREUR DE MESURE INDIVIDUELLE INDEPENDANTE quand le bruit
  individuel est substantiel (30-40%) et le gamma_int reste MODERE.
  Aucun des 8 tests precedents de la journee n'avait ce bruit-la (signal
  partage PROPRE, ou conflit de groupes, ou signal nul) -- c'est
  exactement pourquoi gamma_int n'avait jamais eu sa chance avant ce test.
  **Le fil rouge du projet, une fois de plus** : la valeur d'un mecanisme
  est conditionnelle -- gamma_int perd partout ou il n'y a rien a moyenner
  honnetement (conflit, signal propre, silence), et gagne exactement la
  ou le theoreme de Condorcet dit qu'il devrait gagner (bruit individuel
  independant sur un signal partage). Reserve : n=40/point, gains a la
  limite de 2 SE, a repliquer avec plus de seeds avant de citer hors
  contexte exploratoire.
- **REPLICATION FAITE le 13/07/2026, meme jour -- LE RESULTAT NE TIENT
  PAS.** (`experiments/p10_condorcet_replication_poc.py`, demande de
  Julien : « 1 et une fois trouve on passera a 2 » -- ici la piste 2,
  repliquer avant de generaliser). Meme protocole EXACT, mais 40 seeds
  (n=80/point au lieu de 40) sur une plage de graines DISJOINTE (100-139
  vs 0-19 du test original) -- une vraie replication independante, pas un
  re-run. Restreint aux 2 points qui comptaient (p_correct=0.6/0.7) x
  3 gamma_int (0, 0.15, 0.3).
  **VERDICT : 0/8 comparaisons confirmees.** Les 4 gains rapportes par le
  premier test (vote +0.175 et +0.125 a p_correct=0.6/0.7 gamma=0.15 ;
  interference +0.125 a p_correct=0.7 gamma=0.3) s'effondrent TOUS a des
  IC bootstrap couvrant 0 sur ce nouvel echantillon (deltas +0.075, -0.062,
  +0.050, +0.025 -- direction meme pas coherente d'un test a l'autre). PIRE :
  un point part significativement dans le sens INVERSE -- p_correct=0.7,
  gamma_int=0.3, VOTE : delta=-0.163 CI[-0.300,-0.037], gamma_int=0.3 est
  ici SIGNIFICATIVEMENT PIRE que gamma_int=0 (l'oppose exact de ce que le
  premier test semblait montrer a ce meme point).
  **Le "premier resultat positif de la journee" etait du BRUIT
  d'echantillonnage a n=40, pas un effet reel.** C'est exactement ce qu'un
  gate de replication est cense attraper (meme pratique que la genese du
  11/07 : "la tendance d'hier etait du bruit"). **Verdict final, corrige :
  sur les 9 tests de gamma_int menes le 13/07 (memoire, anti-sync,
  crosstalk, diffusion, sagesse des foules, menteurs, homogeneisation,
  bruit amplifie, topologie BA, ET desormais Condorcet), AUCUN gain net
  pour gamma_int n'a survecu a une replication independante.** L'intuition
  de Julien (« la flemme ne rapporte rien ») ressort de la journee ENCORE
  PLUS solide qu'avant ce test -- pas malgre la replication, GRACE a elle :
  le seul contre-exemple apparent s'est demonte lui-meme des qu'on lui a
  donne une vraie chance de tenir. La piste #2 initialement prevue
  (tester Condorcet sur BA/ER) est donc ABANDONNEE -- generaliser un
  resultat qui ne replique meme pas sur la MEME topologie n'aurait aucun
  sens.

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
- **✅ WARM START CONTINU FAIT le 13/07/2026** (`experiments/p11_warm_start_poc.py`,
  idée de Julien après une longue discussion honnête sur ce que M4R fait vraiment :
  « imaginons que dès le départ M4R donne la direction... warm start continu,
  j'aurais dû te donner ça en premier »). Différent de P11 (QUAND arrêter un
  solveur qui démarre toujours au même point connu) : ici la cible du solveur est
  **ambiguë en signe** (b∈{-1,+1} inconnu a priori) — le vrai test d'un démarrage
  informé. Réutilise le grad/h EXACT de P11 (même dérivation h_min/w_flat), cible
  et plateau mis en miroir par b. BLIND (x0=0, aucune info) doit toujours traverser
  le plateau. WARM (M4R) : lecture bon marché (30 nœuds, pulse signé, readout
  interférence déjà établi aujourd'hui) → devine b, démarre AU-DELÀ du plateau côté
  deviné — si juste, l'évite entièrement ; si faux, pénalité honnête (démarre du
  MAUVAIS côté, doit tout traverser en plus).
  **Résultat, répliqué sur 2 plages de graines disjointes (0-29 et 100-129,
  intégré dès le premier lancement — leçon du Condorcet du même jour)** :
  lecture M4R accuracy=1.00 sur les deux plages ; économie **+1458 et +1507
  itérations** (~97 % de moins que BLIND ~1500-1550) — **identique aux deux
  graines, pas un artefact isolé.**
  **Deux vérifications avant de croire ce chiffre** (le 100 % d'accuracy sentait
  le cas trop favorable — aucune des 60 estimations n'était fausse, la pénalité
  n'avait jamais été exercée) : (1) **pénalité vérifiée manuellement** — une
  mauvaise estimation forcée coûte bien PLUS que BLIND (ex. seed 0 : blind=1369,
  warm_faux=1380) — pas de repas gratuit si M4R se trompe, le mécanisme est honnête.
  (2) **lecture volontairement affaiblie** (T_READ 300→30, B_E 0.8→0.3, un cas
  bien plus dur et réaliste) : accuracy tombe à **0.875** (12,5 % d'erreurs
  réelles) — et l'économie **tient quand même** : blind=1497, warm (mélange
  vrai/faux)=265, soit **+1232 (−82 %)**. Le petit coût des erreurs occasionnelles
  est largement dominé par le gain massif des bonnes estimations.
  **Verdict initial : le warm start continu fonctionne**, vérifié dans les deux
  sens (le cas où ça marche ET le cas où ça coûte), sur cette classe de problème
  (solveur itératif piégé par un plateau trompeur symétrique en signe).
  **⚠️ DEUX CONTRE-ÉPREUVES FAITES le 13/07/2026, même jour** (`experiments/p11_warm_start_stress_poc.py`,
  demande explicite de Julien : « bien sûr que l'on pousse au max de ce qui peut
  être testé et on fait aussi tout pour le prendre en défaut »). **Le « +97 % »
  pris seul était trompeur — corrigé ici.**
  1. **Le hasard fait DÉJÀ la moitié du travail.** Une devinette PILE OU FACE
     (aucune information de M4R) économise déjà **+49 %** (+718 itérations) —
     parce que le piège est très asymétrique : bien deviner est quasi gratuit
     (~45 pas), mal deviner coûte à peine plus cher que ne rien faire (+11 à
     +37 pas sur ~1000-1700). **La vraie contribution de M4R, au-delà du
     hasard, est de +712 itérations supplémentaires** (de 49 % à 97 %),
     portée par son excès de justesse de **+48,3 points** sur le hasard
     (accuracy M4R=1.00 vs hasard=0.517). Réel et substantiel, mais la moitié
     du chiffre brut vient de la structure du piège, pas de M4R — à toujours
     présenter comme (part structure + part M4R), jamais comme un chiffre unique.
  2. **Le warm start dégrade GRACIEUSEMENT, ne s'effondre pas, si la position
     du piège est moins prévisible.** X_WARM=1.5 avait été choisi pour être
     TOUJOURS au-delà du piège dans la plage testée (x_p_mag∈[0.9,1.3]) — un
     choix qui suppose de connaître cette plage à l'avance. En élargissant à
     [0.9,1.9] (le warm start atterrit alors DANS le piège 15/60 fois,
     vérifié explicitement), l'économie descend à **+46 %** (au lieu de
     +97 %) mais reste largement positive et substantielle — pas
     d'effondrement, dégradation proportionnée au nombre de cas où
     l'hypothèse de distance sûre est violée.
  **Bilan honnête complet** : le mécanisme est réel, mais le narratif correct
  n'est PAS « M4R économise 97 % » — c'est « sur ce type de piège asymétrique,
  N'IMPORTE QUELLE heuristique de direction économise déjà ~49 %, et la
  justesse mesurée de M4R (bien au-dessus du hasard) ajoute ~48 points
  supplémentaires, à condition que l'hypothèse de distance sûre du warm start
  tienne — et si elle ne tient qu'à moitié, le gain se dégrade proportionnellement
  sans disparaître ». Différence qualitative avec P11 (qui égalait seulement
  le meilleur budget fixe) : ici, même après correction, le gain net attribuable
  à M4R reste large et clair. **Réserve inchangée** : un seul type de piège
  testé ; coût de la lecture M4R non converti en unité comparable aux
  itérations solveur.
- **✅ LE DERNIER MAILLON — RAFFINEMENT ITÉRATIF — TESTÉ le 13/07/2026, même
  jour** (`experiments/p11_refinement_scar_poc.py`, Julien : « ah oui c'est
  vrai je l'ai oublié celui-là »). Seul maillon jamais testé de la chaîne
  complète décrite par Julien (direction → solveur → veille → retour du
  calcul → **raffinement** → direction plus précise → solveur). **Motif de
  méfiance posé AVANT de tester** : P12 (12/07, substrat STNO) avait déjà
  trouvé une **cicatrice** — un conflit antérieur retarde la sortie de
  tromperie (+52 % de temps de flip). Le raffinement pourrait donc être
  compromis par cette même inertie.
  **Test le plus dur** : simule une PREMIÈRE lecture FAUSSE de M4R (T1=200
  pas de stimulus faible dans le mauvais sens), PUIS une preuve corrective
  (T2 pas de stimulus fort dans le bon sens, réseau continu) — compare
  contre un réseau FRAIS qui reçoit la MÊME preuve corrective sans avoir
  eu de fausse impression à désapprendre.
  | T2 (pas de correction) | accuracy PRIME_FAUX | accuracy FRAIS | coût cicatrice |
  |---|---|---|---|
  | 50  | 0.067 | 0.700 | **+0.633** (quasi bloqué) |
  | 150 | 0.667 | 0.933 | +0.267 (encore réel) |
  | 300 | 1.000 | 1.000 | **0.000** (refermée) |
  | 600 | 1.000 | 1.000 | 0.000 |
  **La cicatrice de P12 EXISTE AUSSI ici, sur FHN+lattice (pas seulement
  STNO)** — 3e contexte où elle apparaît, ce qui la solidifie comme
  propriété générale de M4R plutôt qu'un artefact de substrat. Une première
  impression fausse résiste réellement à la correction (à T2=50, le réseau
  primé-faux reste trompé 93 % du temps contre 30 % pour un réseau frais).
  **MAIS elle se referme COMPLÈTEMENT avec assez de temps correctif**
  (quelque part entre T2=150 et 300 ici) — accuracy identique au réseau
  frais, aucun résidu permanent.
  **Verdict du maillon manquant : le raffinement FONCTIONNE, mais n'est ni
  instantané ni gratuit.** Une architecture qui l'utiliserait devrait
  budgétiser un temps de correction suffisant après une mauvaise direction
  initiale — pas juste "renvoyer un signal plus fort et repartir tout de
  suite". **Chaîne complète de Julien, bilan final** : direction (✅ P10),
  solveur+veille (✅ P11 original + warm start), raffinement (✅ ici, avec
  coût temporel caractérisé) — les trois maillons sont maintenant testés,
  aucun n'est gratuit, aucun n'est cassé.
- **⚠️ COMPARÉ À UNE SOLUTION TRADITIONNELLE le 13/07/2026, même jour**
  (`experiments/p11_refinement_vs_traditional_poc.py`, Julien : « il finit
  par se stopper lui-même mais il a coûté en compute avant cela — est-il
  possible de faire la même expérience avec les solutions traditionnelles
  pour voir ce qu'il coûte par rapport à eux ? »). Candidat traditionnel le
  plus simple et le moins généreux : un **accumulateur naïf** — une
  moyenne mobile exponentielle (EMA) du signal brut reçu, AUCUN réseau,
  AUCUN doute. Même séquence exacte (T1=200 pas faux, puis T2 pas
  correctifs). Sweep de la constante de temps tau_ema∈{5,10,20,50} (la
  plage encadre tau_u=10 du cœur M4R, pour ne favoriser personne).
  **Résultat, net et défavorable à M4R sur CE point précis** :
  | | correction complète (accuracy≥0.99) |
  |---|---|
  | EMA tau=5   | T2≥30 |
  | EMA tau=10  | T2≥50 |
  | EMA tau=20  | T2≥100 |
  | EMA tau=50  | T2≥100 |
  | **M4R (FHN+lattice+doute)** | **entre T2=150 et T2=300** |
  **L'accumulateur naïf corrige PLUS VITE à TOUS les réglages testés** —
  même le plus lent (tau=50, 5× la constante de temps propre de M4R)
  corrige à T2=100, quand M4R a besoin d'au moins 150. **Pour CE sous-
  problème précis (suivre rapidement « quelle est la meilleure estimation
  actuelle de la direction »), la dynamique FHN+doute de M4R coûte PLUS
  cher qu'une simple moyenne mobile — pas moins.**
  **Ce que ça implique pour l'architecture de Julien** : ce résultat ne
  contredit PAS le warm start (validé séparément, avec sa propre valeur
  ajoutée mesurée au-delà du hasard) ni la veille P11 (M4R bat la patience
  sur un solveur tiers) — mais il précise où NE PAS utiliser M4R dans une
  chaîne complète. Si le maillon « suivre/corriger rapidement une
  estimation » doit être RAPIDE, un accumulateur simple le fait mieux ;
  M4R garde sa valeur propre ailleurs (la lecture initiale au-dessus du
  hasard, la surveillance d'un solveur tiers). **Cohérent avec la
  conclusion de Julien du même jour** (« il ne peut pas le faire tout
  seul, c'est l'un des composants pas le seul ») — précisée encore : même
  DANS la chaîne, certains maillons sont mieux servis par une solution
  traditionnelle que par M4R lui-même. Réserve : comparaison sur CE
  readout scalaire seul (direction binaire), pas sur la tâche complète
  (crosstalk, multiplexage) où M4R a montré d'autres propriétés
  aujourd'hui.

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
- **❌ RÉSULTAT NÉGATIF HONNÊTE le 13/07/2026** (`experiments/p13_doubt_hierarchy_poc.py`,
  architecture figée avant résultats : 3 sous-réseaux BA(N=30) + méta-désaccord
  injecté via `sigma_social_override` — hook d'ablation déjà existant, aucune
  modification du cœur —, comparé à FLAT (BA N=90 poolé) et HIER_VOTE (3
  sous-réseaux sans méta-couplage), à coût égal, 20 essais avec hétérogénéité de
  tromperie locale entre sous-réseaux). **Les trois conditions obtiennent
  EXACTEMENT la même accuracy (0.85) et le MÊME résultat trial par trial.**
  Contrôle : le terme méta a un effet numérique réel et non nul sur le signal de
  décision (jusqu'à ±0.13) mais jamais assez fort pour inverser un signe de
  décision sur les 20 essais testés à γ_meta=0.5. « Qui doute des douteurs »
  reste une question ouverte — le premier essai discipliné ne tranche pas ; le
  risque annoncé (trop de degrés de liberté) ne s'est pas concrétisé en biais,
  juste en absence de signal détectable à ce réglage.

### P14 — Le pont LLM, étape 3 : se mesurer à la famille standard 🌉
- **Pourquoi.** Déjà au backlog (D2) — répété ici pour la complétude du legs :
  l'utilité conditionnelle est prouvée contre l'attention nue ; il faut maintenant
  résiduel+MLP (la mitigation que tout transformer réel possède), puis un petit
  GPT réel. Le doute doit être compétitif *dans la famille* anti-effondrement.
- **Effort.** 🧩. **Risque.** Le résiduel est un adversaire fort ; prédiction
  honnête : le doute ne le battra pas en brut, mais pourrait à profondeur extrême
  ou sous distribution shift — chercher LÀ.
- **✅ FAIT le 13/07/2026 — le résultat le plus riche des 5 pistes du jour**
  (`experiments/p14_llm_bridge_residual_poc.py`, condition RESIDUAL_MLP ajoutée
  au harness EXACT du 11/07 : attention+résidu PUIS bloc MLP par token avec sa
  propre connexion résiduelle, poids fixes par seed non entraînés — cohérent avec
  le reste du jouet où rien n'est appris sauf le readout). **Les trois terrains
  prédits, trois réponses différentes :**
  1. **Standard (profondeur choisie par validation) : PARITÉ** (+0.2 pts, ns) —
     ni gagne ni perd contre le vrai adversaire (mieux que la prédiction
     « perd probablement »).
  2. **Profondeur extrême (L=120, 3× le max standard) : DOUBT GAGNE nettement**
     (+12.1 pts, IC[+11.3,+13.0]) — RESIDUAL_MLP finit par s'effondrer comme
     ATTRACTIVE (identité : 86 %→50 %, quasi-hasard) alors que DOUBT stabilise
     (86 %, proche du plafond NO_UPDATE=85 %). La mitigation standard échoue
     elle aussi à très grande profondeur ; le doute non.
  3. **Distribution shift (bruit de test 3.5 vs entraînement 2.0, mêmes poids de
     lecture) : RESIDUAL_MLP légèrement devant** (−0.5 pts, SIGNIF mais petit
     effet) — la seule prédiction qui ne s'est PAS confirmée.
  **Bonus non prédit** : à L=40 FIXE (sans early-stop), DOUBT écrase déjà
  RESIDUAL_MLP (+10.5 pts groupe / +35.0 pts identité) — la mitigation standard
  décroche bien avant la profondeur extrême dès qu'aucun early-stop ne la
  secourt. Le fil rouge du projet se réplique une fois de plus, avec la
  précision la plus fine à ce jour : conditionnel à la profondeur, et étendu
  maintenant au véritable adversaire (pas seulement l'attention nue).
  **Reste** : « un petit GPT réel » (entraîné, pas seulement structurel) — non
  fait cette session, effort dédié à part entière.

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
