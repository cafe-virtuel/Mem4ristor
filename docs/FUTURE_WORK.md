# Mem4ristor — Travaux futurs (backlog priorisé)

> **But.** Ne rien perdre des pistes ouvertes. Chaque entrée est autonome :
> *Pourquoi / Comment / Effort / Statut*, lisible à froid par un futur agent.
> **Origine.** Audit externe simulé « posture Grollier » du 2026-07-06
> ([docs/audit_externe_grollier_2026-07-06.md](audit_externe_grollier_2026-07-06.md))
> + mandat de réfutation λ₂ du 2026-07-01 (`experiments/lambda2_foundation_20260701/SYNTHESE.md`).
> **Mise à jour.** 2026-07-06.

Légende statut : ✅ fait · 🔜 prêt à démarrer · 🧩 projet (plusieurs jours/semaines) · 💭 exploratoire.

---

## Priorité recommandée (si on ne fait qu'une chose à la fois)

1. **B1 — une tâche computationnelle** (reservoir computing). Transforme « le système
   maintient la diversité » en « le doute améliore une performance mesurable ». Plus gros
   gain de crédibilité par heure investie.
2. **A2 — remonter FROZEN_U** comme résultat principal du preprint (déjà mesuré, robuste).
3. **A3 — refaire la régression** de régime avec de vraies simulations (retire la faille
   méthodologique centrale).
4. **A4 — corriger le protocole cold-start** (incohérence code/texte, rapide).
5. Le reste (B2 memristor réel, B3 énergie, B6 prédiction falsifiable) = projets de fond.

---

## A. Cohérence & honnêteté du preprint (issu de la revue)

### A1 — Reformuler λ₂ → degré de couplage (champ moyen) ✅ FAIT (2026-07-06)
Commits `ef5f53c` (preuves) + `eb862f2` (preprint). Titre, abstract, §4.5/4.6/4.7,
Discussion, Conclusion, Limitations réécrits ; le « 2.31 » requalifié en frontière
corrélationnelle. PDF 25 p, 0 undefined ref, Guardian 13/13. **Reste lié : A3.**

### A2 — Faire de FROZEN_U le résultat principal 🔜
- **Pourquoi.** L'ablation « geler u → synchronie ×24 (BA m=3) / ×90 (lattice) » est mesurée
  sur la corrélation de Pearson (indépendante du binning) : c'est le résultat le plus
  robuste et le moins attaquable du papier. Il est aujourd'hui enterré au milieu, tandis
  que le résultat le plus fragile était en titre.
- **Comment.** Remonter l'ablation (Table 2 + phase-space) en tête des Results ; réorienter
  l'abstract/intro pour que « le doute est le moteur d'anti-synchronisation » soit le
  message n°1, la frontière de degré étant la *limite* de ce mécanisme.
- **Effort.** Rédaction, ~1 session. Aucune nouvelle simulation.

### A3 — Régression de régime sur de vraies simulations 🔜
- **Pourquoi.** `p2_edge_betweenness_analysis.py` ne simule pas : il lit un dict `REGIME`
  codé en dur, labellisé par *type* de topologie (12 décisions dupliquées ×3, pas 36
  observations). La « séparation complète » est quasi-tautologique.
- **Comment.** Pour chaque (topologie, seed) : simuler, mesurer **H_cont** (continu, pas
  H_cog), étiqueter *par mesure*. Puis régresser le régime sur **k_harm ET λ₂ côte à côte**.
  Le script `experiments/lambda2_foundation_20260701/bouclage_regime_vs_predicteurs.py` le
  fait déjà (k_harm 2/70 erreurs vs λ₂ 15/70) — le porter proprement dans le papier avec
  figure. Transforme la faille en force.
- **Effort.** ~1-2 sessions (le gros du code existe).

### A4 — Corriger le protocole cold-start 🔜
- **Pourquoi.** Le texte revendique « v=w=0, la diversité *émerge* » mais
  `verify_table1_preprint.py` n'appelle pas `cold_start=True` → init aléatoire
  v∈[-1.5,1.5] (vérifié : H=4.03 non-cold vs 4.27 cold ; la valeur annoncée 4.06 correspond
  au non-cold). Contradiction visible par tout reviewer qui relance le script.
- **Comment.** Option 1 (propre) : passer `cold_start=True`, régénérer le Tableau 1, mettre
  à jour les chiffres + le CSV + claims_mapping. Option 2 (minimal) : décrire l'init réelle
  dans le texte et retirer l'argument « not from favorable initialization ».
- **Effort.** ½ session (option 1 : relancer + Guardian).

### A5 — Bannir H_cog des résultats primaires 🔜
- **Pourquoi.** H_cog (5 bins) est un artefact reconnu (« valeurs à ne pas citer ») et
  pourtant il sous-tend toute la cartographie de la dead zone. À réserver strictement au
  pont SPICE (où l'échelle de tension le justifie).
- **Comment.** Repasser chaque table/figure de régime sur H_cont ; vérifier que la frontière
  (k_harm≈6) tient avec la métrique continue (voir C1 ci-dessous — risque : la valeur
  numérique bouge).
- **Effort.** ~1 session (re-runs + vérif que les conclusions tiennent).

---

## B. Crédibilité « Grollier » (manques structurels)

### B1 — Une tâche computationnelle ⭐ PREMIERS RÉSULTATS (7 juillet 2026)
- **Pourquoi.** « Maintenir la diversité » ne dit pas *pour calculer quoi*.
- **Fait — 3 POCs committés, 5 seeds chacun :**
  - `experiments/reservoir_narma10_poc.py` (`6e9055e`) : le doute **ne bat pas le découplé**
    (D=0 gagne). Le couplage est un handicap quand la tâche n'exige pas d'intégration inter-nœuds.
  - `experiments/bicameral_rhythm_poc.py` (`dfb01d4`) : en pilotant un cycle FOU→SAGE **de
    l'extérieur** (la 2ᵉ chambre que le modèle n'a pas), les solutions deviennent plus
    **cohérentes** (0.225 vs 0.11 hasard). Qualité, pas quantité.
  - `experiments/bicameral_multimodal_poc.py` (`7ed080f`) : le doute explore en restant
    **valide à 95 %** (vs 35 % hasard) par **marche structurée** (dist. consécutive 0.21 vs 0.35).
- **Caractérisation (résultat honnête).** Le doute **n'est pas** un générateur de diversité
  brute (le bruit thermique fait mieux en nombre). C'est un **explorateur discipliné** : il
  visite plusieurs solutions **valides** (respectant les contraintes) par une marche continue,
  sans les casser. Couverture modeste (~2.8 solutions distinctes, pas « infinie »).
  → Réponse **nuancée** à la vision « explorer une infinité de raisonnements en gardant chacun ».
- **Réserve.** Contraste FULL/FROZEN réel mais sur fond de perf. absolue faible ; seed 42 /
  lattice / petites tailles. Caractérisation solide, chiffres à consolider (multi-seed/topo).
- **Suite.** Enrichir la contrainte (multi-modalité plus riche) + régler le rythme
  (T_FOU/T_SAGE) pour voir jusqu'où la couverture monte. Puis **B1b** (watchdog natif).

### B1b — Watchdog de consolidation dans `dynamics.py` (le chaînon manquant) ✅ FAIT + VALIDÉ (2026-07-07)
- **Pourquoi.** Diagnostic **mesuré** (calibrations 7/07) : le modèle se **verrouille en mode
  FOU** — `u` sature >0.5, les seuils de retour SAGE sont bornés à 0.5 (`dynamics.py:134`),
  donc **~0 bascule FOU→SAGE**. La chambre « consolidation » est structurellement inaccessible.
  C'est la panne **symétrique** de celle qu'Edison avait trouvée (verrouillage SAGE ; sa V5b
  jamais implémentée). Les POCs bicaméraux la contournent en pilotant `u` de l'extérieur.
- **Fait.** Watchdog opt-in ajouté au cœur (`dynamics.py:363`, commit `06cb6a9`) : cycle natif
  FOU→SAGE avec **KICK** `u=0.9` en début d'exploration (le doute ne remonte pas seul depuis un
  consensus). Désactivé par défaut, **bit-à-bit identique OFF** (`tests/test_consolidation_watchdog.py`).
- **Validé.** `experiments/watchdog_multimodal_poc.py` (5 seeds, problème multi-modal, 4e condition
  WATCHDOG + contrôle BICAMERAL_KICK). Résultats :
  | Condition | Validité | Couverture (sol. distinctes) |
  |---|---|---|
  | WATCHDOG (natif + kick) | **0.97** | **6.0** |
  | BICAMERAL_KICK (externe + kick) | 1.00 | 6.4 |
  | BICAMERAL (externe, bruit-driven) | 0.95 | 2.8 |
  | HASARD | 0.35 | 1.0 |
  | ATTRACTIF | 0.48 | 3.0 |
  - **(1) Utile** : le cycle natif tient la validité au niveau du pilotage externe et **écrase le
    hasard** (0.97 vs 0.35). La question « son utilité reste à prouver » est tranchée : **oui**.
  - **(2) La couverture ×2 vient du KICK, pas de la « nativité »** : le contrôle BICAMERAL_KICK
    (externe+kick, 6.4) ≈ WATCHDOG (natif+kick, 6.0), écart 6 % (bruit des seeds), les deux
    au-dessus du BICAMERAL bruit-driven (2.8). **Le vrai apport du watchdog = internaliser le
    kick dans le cœur, fidèlement** (plus besoin de piloter `u` dehors), pas un mécanisme émergent.
  - **Réserve.** Couverture modeste (~6, pas « infinie ») ; seeds 0-4, lattice 10×10, E=1.0.
- **Rythme (2026-07-07).** Sweep T_FOU×T_SAGE (`experiments/watchdog_rhythm_sweep.py`, 5 seeds)
  puis raffinement (`experiments/watchdog_rhythm_refine.py`, 8 seeds, barres d'erreur). Résultat :
  - **La couverture n'a pas de pic gaussien : c'est une FALAISE.** Elle croît avec T_SAGE (plus
    on consolide, plus on décide de solutions distinctes) puis **s'effondre d'un coup** quand la
    validité chute (0.97→0.20). Ce seuil = la **dead zone temporelle** : consolider trop longtemps
    (`u=0.05` maintenu) synchronise le réseau et tue la structure. C'est le compromis `u` calibré
    du preprint, transposé du spatial (degré) au **temporel** (durée de consolidation).
  - **Couplage T_FOU↔T_SAGE** (invisible au maillage grossier) : la falaise arrive **plus tôt
    quand T_FOU est plus long** (T_FOU=500 → seuil T_SAGE~400 ; T_FOU=300 → ~450). Explorer et
    consolider tirent en sens inverse.
  - **Point de fonctionnement recommandé : T_FOU=500, T_SAGE=400** (couverture 7.1±1.5, validité
    0.99, plateau robuste 350-450). Le max absolu de couverture (7.8) est au bord instable
    (validité 0.89) — pas rentable.
  - **Couverture plafonne à ~7-8, jamais près de N_CYCLES=12** : ce n'est **pas l'exploration**
    qui borne, c'est la falaise. **Aucun rythme ne débloque « une infinité de raisonnements »** :
    fenêtre étroite avant la mort par sur-consolidation.
  - **Réserve.** σ≈1.5 sur la couverture (8 seeds) : le message robuste est la falaise + son
    couplage à T_FOU, pas les valeurs exactes. Grille lattice 10×10, E=1.0.

### B2 — Un vrai memristor 🧩
- **Pourquoi.** Le projet s'appelle Mem4ristor mais le modèle est un FHN abstrait ; le SPICE
  utilise des *behavioral sources*, pas un modèle de dispositif. Grollier demandera où est
  la variable d'état physique et à quoi correspond `u`.
- **Comment.** Choisir un modèle de dispositif (VTEAM, Stanford/ASU RRAM, GST/PCM, ou
  spintronique) ; établir la correspondance `u` ↔ grandeur physique (lacunes d'oxygène,
  phase, aimantation) avec constantes de temps réelles ; réécrire au moins un étage SPICE
  avec ce modèle. Lien avec la voie photonique déjà explorée (`docs/hardware/PHOTONIC_PATHWAY.md`).
- **Effort.** 🧩 plusieurs semaines.

### B3 — Métriques d'énergie / vitesse / surface 🧩
- **Pourquoi.** En neuromorphique la question est toujours pJ/opération, TOPS/W, latence.
  Le papier n'a aucune unité physique (dt=0.05 sans dimension).
- **Comment.** Ancrer dt et les tensions dans une échelle physique (via B2) ; estimer un
  ordre de grandeur énergie/opération ; comparer à un point de référence CMOS/mémristif.
- **Effort.** 🧩 dépend de B2.

### B4 — Robustesse statistique 🔜
- **Pourquoi.** Résultat central sur 3 seeds, Tableau 1 sur N≤625, N≤2500 max. La « complete
  separation » esquive l'intervalle de confiance au lieu de le fournir.
- **Comment.** ≥20-30 seeds sur les résultats-clés ; finite-size scaling jusqu'à des N
  pertinents ; remplacer la séparation complète par un IC honnête (bootstrap sur des labels
  *mesurés*, pas assignés — dépend de A3).
- **Effort.** ~1-2 sessions de calcul (long mais mécanique).

### B5 — Comparaison à l'état de l'art réel 🧩
- **Pourquoi.** Le benchmark actuel bat Kuramoto/Voter/Consensus (modèles-jouets).
  « Mieux que Kuramoto » n'impressionne personne.
- **Comment.** Comparer sur une tâche standard (via B1) à un echo state network, et surtout
  aux **oscillateurs spintroniques couplés** (domaine de Grollier). Positionner : avantage
  réel ? parité ? niche ?
- **Effort.** 🧩 dépend de B1.

### B6 — Prédiction falsifiable / signature expérimentale 💭
- **Pourquoi.** Tout est auto-référentiel (H, sync, MI calculés sur le même v(t) simulé).
  Manque une prédiction qu'un manip pourrait réfuter.
- **Comment.** Identifier une signature du doute mesurable sur un dispositif réel, distincte
  d'un système sans doute (ex. réponse spectrale, hystérésis, statistique de commutation).
- **Effort.** 💭 réflexion + éventuelle campagne SPICE ciblée.

### B7 — Reproductibilité end-to-end des figures 🔜
- **Pourquoi.** AUDIT-024 a montré que deux générations de code coexistaient sans détection.
  Les tests unitaires (118) + Guardian (13 chiffres) ne garantissent pas que chaque figure
  se régénère depuis zéro.
- **Comment.** Un script one-command, seed fixe, par figure/table du papier, régénérant tout
  depuis zéro ; idéalement vérifié en CI. Étendre le Guardian à la génération, pas seulement
  à la vérification.
- **Effort.** ~1-2 sessions.

---

## C. Prolongements scientifiques (de SYNTHESE.md, mandat λ₂)

### C1 — Dériver k_harm,crit analytiquement 💭
- **Pourquoi.** La valeur « k_harm≈6 » est empirique et dépend de la métrique de régime
  (H_cog, artefact de binning). Le *mécanisme* (champ moyen) est blindé ; la *valeur* l'est moins.
- **Comment.** Fokker-Planck de champ moyen : dériver le seuil depuis σ_v, v* et la géométrie
  des bins — **mais sur une métrique continue**, sinon on refonde un artefact (piège identique
  au 2.31). Croiser avec la re-mesure H_cont de A5.
- **Effort.** 💭 théorique, incertain.

### C2 — Rôle fin de l'hétérogénéité de degré 💭
- **Pourquoi.** k_harm est dominé par les nœuds de bas degré (Jensen) : les scale-free
  survivent à ⟨k⟩ plus élevé via leur périphérie. Contour du mécanisme, moins solide que le cœur.
- **Comment.** Isoler la contribution de la queue de distribution des degrés ; tester des
  familles à hétérogénéité contrôlée.
- **Effort.** 💭 ~1 session exploratoire.

---

## Dépendances rapides
- A3 alimente A5 (mesure H_cont per-seed) et B4 (IC sur labels mesurés).
- B1 débloque B3 (énergie/tâche) et B5 (comparaison SOTA).
- B2 débloque B3 et B6 (physique du dispositif).
- C1 dépend de A5 (métrique continue).
