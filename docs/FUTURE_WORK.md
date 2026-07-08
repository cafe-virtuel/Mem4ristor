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
2. ~~**A2 — remonter FROZEN_U** comme résultat principal du preprint~~ ✅ FAIT (2026-07-08).
3. ~~**A3 — refaire la régression** de régime avec de vraies simulations~~ ✅ FAIT (2026-07-08).
4. ~~**A4 — corriger le protocole cold-start**~~ ✅ FAIT (2026-07-08).
5. ~~A5 — bannir H_cog des résultats primaires~~ ✅ FAIT (2026-07-08, transparence + rétrograder).
6. Le reste (B2 memristor réel, B3 énergie, B6 prédiction falsifiable) = projets de fond.

---

## A. Cohérence & honnêteté du preprint (issu de la revue)

### A1 — Reformuler λ₂ → degré de couplage (champ moyen) ✅ FAIT (2026-07-06)
Commits `ef5f53c` (preuves) + `eb862f2` (preprint). Titre, abstract, §4.5/4.6/4.7,
Discussion, Conclusion, Limitations réécrits ; le « 2.31 » requalifié en frontière
corrélationnelle. PDF 25 p, 0 undefined ref, Guardian 13/13. **Reste lié : A3.**

### A2 — Faire de FROZEN_U le résultat principal ✅ FAIT (2026-07-08)
- **Pourquoi.** L'ablation « geler u → synchronie ×24 (BA m=3) / ×90 (lattice) » est mesurée
  sur la corrélation de Pearson (indépendante du binning) : c'est le résultat le plus
  robuste et le moins attaquable du papier. Il était enterré au milieu, tandis que le
  résultat le plus fragile (λ₂) était en avant.
- **Fait** (commit `cea081a`, recadrage éditorial, 0 nouvelle simu, 0 chiffre modifié ;
  choix Julien : garder le titre, recadrer sans déplacer). (a) Abstract mène avec l'ablation
  FROZEN_U comme résultat central, explicitement « least parameter-sensitive » (Pearson,
  binning-independent) ; la frontière de degré/λ₂ devient « the limit ». (b) Contribution (2)
  de l'intro = identification du doute comme mécanisme **primaire** d'anti-synchro. (c)
  Paragraphe d'ouverture des Results désignant l'ablation comme résultat causal central
  (label `sec:ablation` ajouté). PDF 26p, Guardian 13/13.

### A3 — Régression de régime sur de vraies simulations ✅ FAIT (2026-07-08)
- **Pourquoi.** `p2_edge_betweenness_analysis.py` ne simule pas : il lit un dict `REGIME`
  codé en dur, labellisé par *type* de topologie (12 décisions dupliquées ×3, pas 36
  observations). La « séparation complète » est quasi-tautologique.
- **Fait.** `experiments/a3_regime_regression_hcont.py` (commit `0ca04b1`) : 14 topologies
  × 5 seeds = 70 vraies simulations, régime étiqueté **par mesure en H_cont** (100 bins),
  pas H_cog. Régression continue : Spearman ρ = **−0.83** (k_harm) / **−0.78** (k_mean) /
  **−0.52** (λ₂) ; n=70, p<1e-6. Le contrôle H_cog reproduit l'ordre (4/70 vs 8/70 vs 17/70).
  Figure `fig:regime_degree` remplace `fig:fiedler` (H_cog, n=30, labels par type) dans le
  preprint ; caption tautologique résiduelle de `tab:ba_m_sweep` requalifiée.
- **Deux nuances gravées (honnêteté).** (1) En H_cont le régime est un **déclin continu**
  (~3.9 → ~2.6 bits), pas un effondrement : le « dead zone » est en partie un artefact du
  H_cog 5 bins, aucun seuil binaire net en continu (5/70 sous le plus grand gap). (2) k_harm
  ≈ k_mean en H_cont (−0.83 vs −0.78) : la donnée identifie le **degré de couplage** (pas λ₂),
  mais ne distingue pas nettement k_harm de k_mean (net seulement en H_cog).
- **Retombée.** Amorce A5 (H_cont adopté dans la figure/régression principale de régime).

### A4 — Corriger le protocole cold-start ✅ FAIT (2026-07-08, Option 1 + nuance L109)
- **Pourquoi.** Le texte revendique « v=w=0, la diversité *émerge* » mais
  `verify_table1_preprint.py` n'appelait pas `cold_start=True` → init aléatoire. Contradiction
  visible par tout reviewer qui relance le script.
- **Fait** (commit `4e507b8`, Option 1). `cold_start=True` ajouté ; le script **écrit
  désormais** `figures/p2_table1_lattice.csv` (repro : la table sort d'un seul run). Mesure :
  à I_stim=0.5 le cold-start change **peu** (u sature ~0.99, l'état oublie l'init) : 4×4
  3.22→3.21, 10×10 4.06→**4.09**, 25×25 4.28→**4.40** (seul le 25×25 monte, +0.12). La
  conclusion de Table 1 est donc **robuste à l'init**, et l'argument « émergence depuis v=w=0 »
  est maintenant littéralement vrai. Table 1 + abstract + benchmark + conclusion à jour
  (4.06±0.08 → 4.09±**0.19**, la std du 10×10 s'élargit en cold — rapporté). L109 nuancée
  (« Unless otherwise noted »). Guardian C02=3.205 / C03=4.404 régénérés, 13/13.
- **Note connexe.** La revendication L109 « All simulations v=w=0 » était globale ; 31 scripts
  sont déjà cold, `verify_table1` était l'exception. Une passe systématique cold-start sur
  *tous* les résultats secondaires reste possible (lié B7 repro end-to-end).

### A5 — Bannir H_cog des résultats primaires ✅ FAIT (2026-07-08, résultat nuancé)
- **Pourquoi.** H_cog (5 bins) est un artefact reconnu, pourtant il sous-tend la
  cartographie de la dead zone.
- **Fait** (commit `2e71aab`, choix Julien : transparence + rétrograder + documenter).
  Re-mesure fonctionnelle (2 scripts, cold start, 10 seeds) : `limit02_alpha_sweep.py`
  (réécrit, sync+LZ) + `limit02_regime_map_functional.py` (nouveau).
- **Découverte centrale (résultat négatif méthodo).** AUCUNE métrique fonctionnelle continue
  ne remplace proprement H_cog en régime **endogène** : (a) la **synchronie** ne montre aucune
  dead zone (r̄≈0 partout, pic 0.13 vs 0.75 driven) → la dead zone endogène **n'est pas un
  consensus temporel** mais un effondrement spatial sur un point fixe commun ; (b) **H_cont**
  récompense le quasi-découplage à faible couplage (à m=2, H_cont max à γ=0, H_cog max à γ=1).
  La frontière multi-états est **intrinsèquement discrète** → H_cog gardé comme indicateur
  **relatif** (valeurs absolues non citées), preuve robuste = `fig:regime_degree` (A3) +
  ablation synchronie driven (A2).
- **Effet cold-start (lien A4).** La dead zone endogène se décale de m≥3 (artefact non-cold)
  à **m≥6** ; déclin **graduel** (cohérent A3/Binder crossover).
- **Livré.** tab:alpha_sweep + tab:ba_m_sweep : colonnes fonctionnelles (H_cont, sync) à côté
  de H_cog, valeurs cold-start ; note méthodo « why H_cog here » gravée ; explication du
  résultat négatif reformulée (champ-moyen d'échantillonnage, plus la redondance de chemins).
  **Lien C1** : la re-mesure H_cont confirme le déclin continu ; la valeur k_harm reste liée
  à la définition multi-états (H_cog), comme prévu.

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

### B1c — Le doute comme allocateur de compute (flux d'entrées) ⚠️ RÉSULTAT MITIGÉ (2026-07-07)
- **Pourquoi.** La vision « explorer tant que le doute persiste » = un flux de problèmes, le
  doute allouant le compute à chacun (adaptive computation time piloté par le doute). Plus
  fidèle à la vision que le watchdog interne (B1b).
- **Fait.** `experiments/doubt_compute_allocation_poc.py` (6 seeds, K=12 : 3 familles ÉVIDENCE /
  CONTRADICTION / TOPOLOGIE). Substrat = tâche de **décision/consensus** (le réseau tranche un
  signe global d'après l'évidence nette). Readout **différentiel** (run de référence `stim=0` au
  même seed → annule le biais du point fixe négatif `v*≈−1.29` ET le bruit). 3 conditions à
  budget total égal : **DOUTE** (arrêt quand `sigma_social=|L·v|` chute sous 30 % du pic),
  **UNIFORME** (budget fixe/problème), **CONVERGENCE** (contrôle honnête : arrêt quand la variable
  de décision cesse de bouger — n'utilise pas le doute).
- **Résultats (budget serré 0.75×) :**
  | Condition | Réussite | Coût moyen (pas/pb) |
  |---|---|---|
  | DOUTE | 0.92 | 378 |
  | **CONVERGENCE** | **0.94** | **107** |
  | UNIFORME | 0.47 | — |
  - **(1) L'allocation adaptative écrase l'uniforme** (0.92-0.94 vs 0.31-0.74) : répartir le
    compute selon un critère d'arrêt, ça paie. La thèse ACT tient.
  - **(2) Le DOUTE ne bat PAS le contrôle trivial** : la convergence-de-décision est aussi
    précise (0.94 vs 0.92) et **3,5× moins chère** (107 vs 378 pas). Le doute sur-réfléchit.
  - **(3) Mécanisme** : le désaccord local `|L·v|` **ne retombe jamais sur les topologies sparse/BA**
    (25 % saturent au budget max, c_doubt=717 vs c_conv=109) → le doute s'y accroche et **affame
    le budget** du reste du flux. La convergence est robuste à la topologie (~90-123 pas partout).
  - **Allocation vs difficulté** : corr(compute, oracle)≈0 pour les deux — mais proxy oracle
    faible (dominé par des flips de bruit tardifs) → **inconcluant**, à ne pas surinterpréter.
- **Fil rouge** : 3e confirmation (après reservoir NARMA10 D=0, et B1b kick=watchdog) que **la
  valeur est dans la partie ADAPTATIVE, pas dans le DOUTE en soi**. Le doute reste un explorateur
  discipliné, pas un mécanisme magique.
- **RÉSERVE MAJEURE (bridge B1d)** : cette tâche est *piégée contre le doute* — « se stabiliser =
  avoir juste », donc un critère de convergence ne peut structurellement pas se tromper. Le doute
  est censé briller quand **se stabiliser tôt = se tromper** (optimum local trompeur). Non testé.

### B1d — Tâche TROMPEUSE : le doute gagne ✅ FAIT (2026-07-07)
- **Pourquoi.** B1c montre que sur une tâche où convergence=correction, le doute ne peut pas
  gagner. Le doute ne peut ajouter de la valeur que si **converger tôt mène à la mauvaise réponse**.
- **Fait.** `experiments/deceptive_task_poc.py` (12 seeds). Piège **pulsé** : leurre NOMBREUX (26
  nœuds) + fort, signe −D\*, **retiré après T_pulse** (domine la moyenne globale TÔT → faux) ;
  vérité PERSISTANTE (14 nœuds), signe +D\* (seule active après le pulse → gagne TARD). Readout
  différentiel (B1c). DOUTE (`sigma_social` chute <30 %) vs CONVERGENCE (décision stabilisée).
  *(NB : 1ʳᵉ calibration ratée — vérité nombreuse dominait la moyenne dès le début, tâche « juste
  tôt » et non trompeuse ; corrigé en rendant le leurre nombreux+pulsé. Diagnostic gravé.)*
- **Résultats (T_pulse ≥ 350) :**
  | | acc DOUTE | acc CONVERGENCE | arrêt |
  |---|---|---|---|
  | Tâche trompeuse | **0.83** | **0.25** | doute ~380 (après pulse) / conv ~205 (dans le leurre) |
  - **Le DOUTE bat la convergence de +0.58.** La convergence s'arrête sur le **plateau du leurre**
    (~205 pas, faux) ; le doute voit la **tension locale** du bras-de-fer persister et tient
    jusqu'**après la fin du pulse** (~380), quand la vérité reprend → juste.
  - **Fenêtre nécessaire** : à T_pulse=150 (piège trop court) les deux échouent (0.25) — le doute
    ne gagne que si le leurre dure assez pour que la convergence s'y engage.
- **Fil rouge COMPLET (la caractérisation la plus défendable du projet sur le doute)** :
  - B1c (se stabiliser = juste) → doute **≤** convergence (sur-réfléchit).
  - B1d (se stabiliser tôt = faux) → doute **>** convergence (+0.58, refuse le faux consensus).
  - **La valeur du doute est CONDITIONNELLE : elle paie exactement quand converger tôt est un
    piège.** Ni gadget, ni magie — un mécanisme dont on connaît le domaine d'utilité.
- **Réserves.** Plafond `acc_FIN=0.75` (tâche pas parfaitement soluble) → comparaison relative ;
  « flip moy ~2200 » = métrique stricte gonflée par des flips de bruit tardifs, la vraie transition
  est à la fin du pulse (~350) ; 12 échantillons (0.83=10/12) → direction robuste, valeurs à N modeste.

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
