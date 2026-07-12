# Mem4ristor — Travaux futurs (backlog priorisé)

> **But.** Ne rien perdre des pistes ouvertes. Chaque entrée est autonome :
> *Pourquoi / Comment / Effort / Statut*, lisible à froid par un futur agent.
> **Origine.** Audit externe simulé « posture d'une chercheuse de référence en neuromorphique » du 2026-07-06
> ([docs/audit_externe_neuromorphique_2026-07-06.md](audit_externe_neuromorphique_2026-07-06.md))
> + mandat de réfutation λ₂ du 2026-07-01 (`experiments/lambda2_foundation_20260701/SYNTHESE.md`).
> **Mise à jour.** 2026-07-11 (genèse 5 états consolidée D1, pont LLM tâche aval D2,
> [13] révisé au code actuel, B5-STNO NARMA10 fait — voir sections correspondantes).
> **Réservoir d'idées complémentaire** (écartées trop tôt, jamais tentées, garde-fou des
> impasses) : [PISTES_POUR_LA_SUITE_2026-07-12.md](PISTES_POUR_LA_SUITE_2026-07-12.md)
> — le legs de Fable, 14 pistes sourcées (bicaméral V5b, MoE par certitude, usure/drift,
> graphes dirigés, abstention calibrée, u∈ℂ dans le cœur…).


Légende statut : ✅ fait · 🔜 prêt à démarrer · 🧩 projet (plusieurs jours/semaines) · 💭 exploratoire.

---

## Priorité recommandée (si on ne fait qu'une chose à la fois)

0. **09/07/2026 — Fond du Volet B cadré (B2/B3/B5/B6)**, choix de Julien : « tout explorer »
   plutôt qu'un seul dispositif. 3 dossiers de correspondance physique (photonique/
   spintronique/électrique), énergie comparée, positionnement spintronique qualitatif,
   proposition falsifiable concrète (réseau STNO couplé, cf. B6). **Aucune simulation
   physique réelle (LLG/SPICE) — reste un projet de fond.** Aucun fichier public/preprint
   touché, cœur non modifié, tests 118+2xfail OK, Guardian 14/14.
1. ~~**B1 — une tâche computationnelle**~~ ✅ CONSOLIDÉ (2026-07-08). La caractérisation
   (doute = explorateur discipliné à **valeur conditionnelle**) est désormais robuste aux seeds
   et à **3 topologies** (lattice / BA scale-free / ER) avec IC bootstrap. Voir bandeau section B.
   Reste ouvert : B5 (comparaison SOTA), tâche à perf. absolue plus élevée.
2. ~~**A2 — remonter FROZEN_U** comme résultat principal du preprint~~ ✅ FAIT (2026-07-08).
3. ~~**A3 — refaire la régression** de régime avec de vraies simulations~~ ✅ FAIT (2026-07-08).
4. ~~**A4 — corriger le protocole cold-start**~~ ✅ FAIT (2026-07-08).
5. ~~A5 — bannir H_cog des résultats primaires~~ ✅ FAIT (2026-07-08, transparence + rétrograder).
6. ~~Le reste (B2 memristor réel, B3 énergie, B6 prédiction falsifiable) = projets de fond.~~
   🟡 **Cadrage fait (09/07/2026)** : 3 dossiers de correspondance physique (B2),
   comparaison d'énergie (B3), positionnement spintronique qualitatif (B5), proposition
   falsifiable concrète (B6). **Aucune simulation physique réelle (LLG/SPICE) — reste
   un projet de fond de plusieurs semaines.** Voir section B ci-dessous.

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

## B. Crédibilité « chercheuse de référence en neuromorphique » (manques structurels)

> **✅ CONSOLIDATION B1 (2026-07-08, Claude Opus 4.8).** Les POCs B1/B1b/B1c/B1d (seed unique
> ou ≤12 seeds, lattice seul) ont été rejoués sur **30/18 seeds × 3 topologies** (LATTICE régulier,
> BA m=3 scale-free, ER aléatoire) avec **IC bootstrap**. Scripts :
> `experiments/b1{d,b,c}_*_consolidation.py` (+ CSV/PNG), capstone
> `experiments/b1_conditional_synthesis.py` → `docs/b1_conditional_synthesis.md` +
> `figures/b1_conditional_synthesis.png`. Cœur non touché, Guardian 13/13.
>
> | Topologie | Tâche LOYALE (doute−conv) | Tâche TROMPEUSE (doute−conv) | Watchdog (valid−hasard) |
> |---|---|---|---|
> | LATTICE | −0.06 [−0.09,−0.02] | +0.67 [+0.43,+0.87] | +0.73 [+0.67,+0.79] |
> | BA_m3 | −0.48 [−0.56,−0.39] | +0.35 [+0.02,+0.65] | +0.15 [+0.02,+0.29] |
> | ER_p06 | −0.25 [−0.30,−0.21] | +0.63 [+0.40,+0.83] | +0.74 [+0.69,+0.79] |
>
> **Verrouillé.** La **valeur du doute est conditionnelle** (≤0 sur tâche loyale, >0 sur tâche
> trompeuse), robuste aux seeds et à la topologie. Le watchdog natif est utile partout.
> **Découverte transversale non prévue :** **BA scale-free est le cas le plus faible des trois
> expériences** — sur tâche loyale le doute y devient même pire que l'uniforme (sur BA, `|L·v|`
> ne retombe jamais → saturation, famine de budget). Ce n'est pas la densité (ER ≈ lattice à
> ⟨k⟩ égal) mais **l'hétérogénéité de degré / les hubs** — même variable que la reformulation
> λ₂→degré du preprint. Deux fausses alertes de petit échantillon levées par les seeds
> (watchdog « inutile sur BA » à 2 seeds ; contrôle négatif T_pulse=150 impur sur BA/ER).

### B1 — Une tâche computationnelle ✅ CONSOLIDÉ (30 seeds × 3 topos, 8 juillet 2026)
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

### B2 — Un vrai memristor 🟡 3 dossiers de correspondance ouverts (09/07/2026), simulation physique restante
- **Pourquoi.** Le projet s'appelle Mem4ristor mais le modèle est un FHN abstrait ; le SPICE
  utilise des *behavioral sources*, pas un modèle de dispositif. une chercheuse de référence en neuromorphique demandera où est
  la variable d'état physique et à quoi correspond `u`.
- **Comment.** Choisir un modèle de dispositif (VTEAM, Stanford/ASU RRAM, GST/PCM, ou
  spintronique) ; établir la correspondance `u` ↔ grandeur physique (lacunes d'oxygène,
  phase, aimantation) avec constantes de temps réelles ; réécrire au moins un étage SPICE
  avec ce modèle. Lien avec la voie photonique déjà explorée (`docs/hardware/PHOTONIC_PATHWAY.md`).
- **✅ Fait (09/07/2026, choix Julien : « tout explorer », pas un seul dispositif).**
  Calcul dimensionnel reproductible (`experiments/b2_device_physics_mapping.py`
  → `figures/b2_device_physics_mapping.csv`), ancré sur la pulsation propre mesurée
  du nœud FHN isolé (`reviewer2_linear_stability.py`, λ=−0.0473±0.2824i → T_node≈22.25
  unités modèle). **Découverte structurante : les 3 familles ne se substituent pas
  l'une à l'autre — chacune correspond à un RÔLE différent dans l'architecture**, pas
  à une réimplémentation complète du modèle :
  - **Photonique (GST)** → rôle `u` (doute, lent, multi-niveau). §5 ajouté à
    `docs/hardware/PHOTONIC_PATHWAY.md` (le dossier le plus avancé, quatuor
    d'imperfections déjà validé le 12/06). Ancrage 100–200 ns (littérature GST
    vérifiée), énergie de signal 1.28 aJ/pas (plancher théorique, hors overhead).
  - **Spintronique (STNO à vortex)** → rôle `v` (oscillateur). Nouveau dossier
    `docs/hardware/SPINTRONIC_PATHWAY.md`. Candidat le plus **direct** (un STNO
    EST un oscillateur auto-entretenu, contrairement au nœud FHN isolé qui est une
    spirale stable sous Hopf) et le plus **rapide** des trois (dt physique en
    picosecondes).
  - **Électrique** → **deux rôles distincts, deux dispositifs** (nouveau dossier
    `docs/hardware/ELECTRICAL_PATHWAY.md`) : RRAM/VTEAM filamentaire pour le poids
    de couplage `D_eff` (statique dans le modèle actuel → énergie payée UNE FOIS,
    pas par pas, ~10-50 fJ/écriture cas optimiste) ; neuristor Mott NbO2
    (Pickett et al. 2013, Nature Materials) pour l'oscillateur `v` (seul candidat
    électrique qui oscille par construction).
- **✅ Modèle STNO macrospin minimal FAIT (09/07/2026, suite immédiate)** —
  `experiments/b2_stno_phase_coupling_poc.py`, voir `SPINTRONIC_PATHWAY.md` §7.
  Réduction phase-oscillateur (Kuramoto/Slavin-Tiberkevich, le niveau d'abstraction
  standard de la littérature STNO pour la synchronisation de réseau) portant le
  mécanisme `u`/`u_filter` **à l'identique** (mêmes constantes que `dynamics.py`).
  **Le mécanisme se porte** : le doute réduit la synchronisation de Kuramoto sur ce
  substrat totalement différent (Cohen d 1.05–2.28 « tel quel » ; 4.83–14.85 une fois
  le capteur de désaccord calibré pour laisser `u` franchir son seuil de bascule à
  0.5 — un paramètre de capteur ajouté, pas retouché sur le mécanisme lui-même).
  Ordre BA>lattice répliqué (cohérent B1/B4, non cherché). Réserve : test de
  portabilité mathématique, pas une simulation LLG ni une validation physique.
- **✅ Généralisation amplitude+phase FAIT (09/07/2026, suite immédiate — Julien : « je
  veux voir ce que ça donne »).** `experiments/b2_stno_amplitude_phase_poc.py`, voir
  `SPINTRONIC_PATHWAY.md` §8. Le Kuramoto pur ci-dessus est le cas limite **isochrone**
  d'un modèle plus fidèle (Slavin-Tiberkevich, dérivé de LLGS) : amplitude ET phase
  dynamiques, **décalage de fréquence non-linéaire** (non-isochronicité, la signature
  physique la plus caractéristique des STNO, absente du Kuramoto pur). **Résultat :
  au capteur brut (gain=1), l'effet devient NUL** (Cohen d 0.01–0.09, plus net que le
  Kuramoto pur) ; **une fois calibré (gain=10), le mécanisme est robuste sur toute la
  plage de non-isochronicité testée** (Cohen d 4.41–5.49 BA m=3, 1.79–2.60 lattice,
  N_nonlin 0→10, aucun effondrement). Vérification indépendante rassurante : `R_FROZEN_U`
  diminue avec la non-isochronicité, cohérent avec la littérature (élargissement de raie).
  Calibration numérique documentée (Euler diverge à dt=0.01/gain=10/N_nonlin≥10, confirmé
  non-physique, corrigé à dt=0.005 — même esprit que le stiffness proof Euler du 1er mai).
- **✅ Macrospin LLGS complet FAIT (09/07/2026, palier choisi par Julien après vérif
  matériel — RTX 3070 8Go/32Go RAM, largement suffisant, pas besoin de GPU ici).**
  `experiments/b2_stno_macrospin_llgs_poc.py`, voir `SPINTRONIC_PATHWAY.md` §9. Vrai
  vecteur d'aimantation 3D, équation de Landau-Lifshitz-Gilbert-Slonczewski explicite
  (pas une réduction phénoménologique comme §7/§8). **Découverte de calibration** : cette
  géométrie de couplage verrouille en **antiphase** (pas en phase) — phénomène réel pour
  les oscillateurs gyrotropes couplés — nécessite de mesurer le 2e harmonique de Kuramoto
  (`R2`), pas `R`. **Découverte topologique non cherchée** : lattice (bipartite) atteint
  un vrai ordre antiphase (R2=0.83 en FROZEN_U) ; **BA m=3 (non bipartite) est FRUSTRÉ**
  (R2~0.15-0.18 dans toutes les conditions) — 3e mécanisme indépendant où BA se comporte
  différemment de lattice. **Résultat sur lattice (où un ordre existe) : le doute réduit
  R2 nettement, dès le capteur brut** (Cohen d=2.42, IC[+0.31,+0.65]), renforcé calibré
  (Cohen d=3.22) — première fois sur les 3 modèles que l'effet brut est déjà fort. Sur BA
  frustré, effet statistiquement réel mais petit en absolu (Cohen d 1.6-3.4, diff
  +0.04 à +0.07 seulement). Vérifications physiques préalables cohérentes (cône de
  précession stable et continûment ajustable par β ; non-isochronicité émergeant
  naturellement de l'anisotropie H_k, sans terme ajouté — valide le §8 a posteriori).
- **Reste (🧩 projet de fond, plusieurs semaines).** Aucune résolution spatiale de la
  texture de vortex (micromagnétisme complet type mumax3, ou modèle de Thiele) ni SPICE
  VTEAM/neuristor n'a été faite — décision explicite de Julien de s'arrêter à ce palier
  pour cette session. Le canal de couplage électrique réel (courant partagé, Romera et al.
  2018) n'a pas été modélisé explicitement, seulement un champ de couplage générique qui
  favorise l'antiphase — les vrais réseaux STNO électriquement couplés sont rapportés
  comme favorisant plutôt le verrouillage en phase, à garder en tête. Le rôle physique de
  `u` (quel circuit lit le désaccord et pilote une variable lente) reste non résolu — les
  3 tests montrent que le mécanisme mathématique se porte, pas qu'un dispositif réel peut le lire.

### B3 — Métriques d'énergie / vitesse / surface 🟡 cadré (09/07/2026), pas clos
- **Pourquoi.** En neuromorphique la question est toujours pJ/opération, TOPS/W, latence.
  Le papier n'a aucune unité physique (dt=0.05 sans dimension).
- **Comment.** Ancrer dt et les tensions dans une échelle physique (via B2) ; estimer un
  ordre de grandeur énergie/opération ; comparer à un point de référence CMOS/mémristif.
- **✅ Fait (09/07/2026).** `docs/hardware/B3_ENERGY_COMPARISON.md` : tableau des 3
  familles + référence CMOS (Loihi ~24 pJ/op, TrueNorth ~26 pJ/événement, vérifiés
  par recherche web). **Résultat qualitatif honnête** : les 3 dispositifs dynamiques
  convergent vers un ordre de grandeur fJ/pas (3-4 ordres sous Loihi/TrueNorth, mais
  échelles de comptage différentes — pas une victoire directe) ; le RRAM en rôle de
  poids statique est structurellement le moins coûteux (énergie payée une fois).
  **Réserve dominante** : aucune énergie « système complet » (interconnexion, overhead
  laser/détecteur photonique, etc.) n'a été calculée — B3 reste un cadrage d'ordres de
  grandeur, pas une preuve de faisabilité énergétique bout en bout.
- **Reste.** Choisir UNE architecture hybride précise (quel dispositif pour quel rôle,
  N nœuds, overhead d'interconnexion) et la chiffrer bout en bout — projet de
  plusieurs semaines.

### B4 — Robustesse statistique ✅ FAIT (2026-07-08) — ablation centrale + Table 1 + FSS
- **Pourquoi.** Résultat central sur peu de seeds, Tableau 1 sur N≤625. La « complete
  separation » esquive l'intervalle de confiance au lieu de le fournir.
- **✅ Fait — ablation centrale FROZEN_U** (`experiments/b4_ablation_robustness.py`, commit `53736fe`).
  30 seeds (10 canoniques + 20 nouveaux) × 2 topos (LATTICE, BA m=3), réutilise
  `p2_sigma_social_ablation.run_one`. IC bootstrap :
  - BA m=3 : FULL 0.0088 → FROZEN 0.688, **diff +0.679 CI[+0.653,+0.702], Cohen d=9.4, séparation COMPLÈTE**.
  - LATTICE : FULL 0.0120 → FROZEN 0.525, **diff +0.513 CI[+0.474,+0.551], Cohen d=4.7, séparation COMPLÈTE**.
  - **Reproductibilité** : sur les 10 seeds canoniques, BA m=3 reproduit EXACTEMENT le CSV committé
    (0.0072 → 0.6582, 91×) — le « 0.007→0.658 ~90-fold » de l'abstract est bien le chiffre **BA m=3**.
  - **⚠️ Findings de cadrage (pour le preprint)** : le **ratio** FROZEN/FULL est une statistique
    **fragile** (dénominateur FULL ≈ 0, un seed BA donne FULL<0 → ratio ~1e9). Le résultat honnête
    est un **SAUT** : différence + séparation complète + Cohen d, pas un « ~90-fold ».
    **Recommandation** (décision Julien) : dans l'abstract/`sec:ablation`, mener avec
    « rises from 0.007 to 0.658 (complete separation over 30 seeds, Cohen d≈9) » et rétrograder
    le « ~90-fold » (ou le retirer). H_cont s'effondre aussi (diff +0.68 à +0.98 CI).
- **✅ Fait — Table 1 (diversité H_cont) + finite-size scaling** (`experiments/b4_table1_robustness.py`,
  commit `182eab5`). 30 seeds × 7 tailles (N = 16…900), IC bootstrap, mesure identique à la Table 1
  canonique (cold_start, I_stim=0.5, 3000 steps). Supplément (n'écrase pas C02/C03).
  - **Reproductibilité** : 10×10 30 seeds = 4.095 CI[4.044,4.146] vs canonique 4.086 (preprint ~4.09) ;
    4×4 = 3.207 vs C02 3.205 → Table 1 confirmée à 30 seeds.
  - **FSS** : H_cont croît (+0.187 bits/octave) puis **sature** (queue +0.062, ~3× plus lent),
    plateau ~4.38 bits, **jamais d'effondrement** (min 3.21 ≫ 0 ; plafond binning 6.64). La diversité
    n'est PAS un artefact de taille finie ; l'IC se resserre avec N (std 0.118 → 0.047).
- **Reste (optionnel).** IC sur labels *mesurés* de régime (A3 fait le fournit déjà) ; N > 900 si un
  reviewer l'exige (plateau déjà visible). B4 considéré **clos** pour les résultats-clés.

### B5 — Comparaison à l'état de l'art réel 🟡 ESN FAIT (2026-07-08), reste spintronique
- **Pourquoi.** Le benchmark actuel bat Kuramoto/Voter/Consensus (modèles-jouets).
  « Mieux que Kuramoto » n'impressionne personne.
- **✅ Fait — vs Echo State Network sur NARMA10** (`experiments/b5_esn_comparison.py`, commit `3df5cfc`).
  Comparaison **loyale** (même tâche/split/readout/N=100, chaque modèle avec son balayage
  d'hyperparamètres), 8 seeds, IC bootstrap. **Résultat honnête et net :**
  - ESN = **0.351 ± 0.026** NRMSE (reservoir utile, < 1.0) ; Mem4ristor FULL = **1.942 ± 0.302**
    (> 1.0, pire que prédire la moyenne) ; écart FULL−ESN = **+1.59 CI[+1.36,+1.81]** → **ESN ~5.5× meilleur**.
  - **Positionnement** : Mem4ristor n'est **PAS** un reservoir NARMA10 compétitif. NARMA10 récompense
    la **mémoire**, pas la diversité → c'est le **pendant SOTA de B1c/B1d** (tâche loyale : le doute
    n'aide pas). La contribution du projet est le **mécanisme du doute** (anti-synchro, diversité
    maintenue), pas la performance mémoire brute. On sait désormais sur quelle tâche **ne pas** le vendre.
  - **Question ouverte RÉPONDUE (honnête, nuancée)** — `experiments/b5b_deceptive_exploration.py`,
    commit `00094d4`. Décision **en ligne** trompeuse (converger tôt = se tromper), doute natif vs
    ESN de référence, 15 seeds. **(1)** Le doute (0.87) **écrase** le meilleur arrêt *naïf* de l'ESN
    (0.00, +0.87 CI[+0.67,+1.00]) : l'ESN se fige instantanément sur le leurre (arrêt au plancher
    t=31-81), le doute `|Lv|` tient jusqu'après le pulse → **horloge de délibération intrinsèque**.
    **(2)** Mais le doute (0.87) **égale** l'ESN à *meilleur budget fixe* (B=800 > durée du leurre,
    0.93 ; −0.07 CI[−0.27,+0.13]). **Niche réelle mais étroite** : le doute bat les arrêts naïfs
    sans rien régler, mais pas un horizon fixe optimal quand l'horizon est **borné** et attendre est
    **gratuit**. Sa valeur décisive exige un **horizon inconnu/non-borné** OU un **coût d'attente**
    (cohérent B1c : le doute paie quand le budget est rare). Le cadrage « explorateur, pas mémoire »
    est validé au niveau des règles d'arrêt, à cette condition près.
- **✅ Comparaison de PERFORMANCE spintronique FAITE (11/07/2026)** —
  `experiments/b5_stno_narma10_poc.py` (commit `9abd12c`) : NARMA10 sur un réseau de 100
  oscillateurs Slavin-Tiberkevich (harness/tâche/seeds STRICTEMENT ceux de la comparaison
  ESN du 08/07 ; entrée = modulation du gain par courant STT ; lecture = puissance moyenne
  par symbole ; doute identique à dynamics.py, gain calibré ; dt=0.005 vérifié en pré-vol ;
  fairness : chaque condition choisit iscale/K_SUB/N_nonlin par seed). **Hiérarchie mesurée :
  ESN 0.362 < STNO_DECOUPLE 0.831 < STNO_FROZEN 0.920 ≈ STNO_FULL 0.926 << M4R-FHN 1.811.**
  (1) Doute **NEUTRE** (+0.006 IC[−0.017,+0.026]) — il maintient pourtant un rang effectif
  bien plus haut (~78 vs ~52) : la diversité ne se convertit pas en mémoire. (2) Le
  **découplé gagne** (+0.095 IC[+0.062,+0.139]) — réplication du pattern FHN du 07/07 sur
  un 2e substrat. (3) **Le substrat STNO divise l'erreur du FHN par 2 et passe SOUS
  NRMSE=1.0 (reservoir UTILE dans l'absolu)** — la physique du substrat compte plus que le
  mécanisme sur cette tâche. (4) L'ESN reste devant (+0.56). Contexte littérature :
  Torrejon 2017 / Romera 2018 (protocoles différents, single-node time-multiplexé — assumé
  dans la docstring). ~~Reste éventuel : la tâche trompeuse B1d sur ce substrat (le terrain
  du doute), 💭 1 session.~~
- **✅ Tâche trompeuse B1d sur substrat STNO FAITE (12/07/2026, piste P12 du legs)** —
  `experiments/b1d_stno_deceptive_poc.py` (12 seeds × 4 T_pulse × 2 substrats, règles
  d'arrêt à hyperparamètre GLOBAL, critères pré-fixés, 5 lancements documentés dans la
  docstring — 2 recalibrations de STRUCTURE avant de comparer quoi que ce soit). **Trois
  faits physiques, aucun n'est celui qu'on espérait :**
  1. **Le piège B1d ne se lit pas naïvement sur ce substrat** : le couplage entre
     oscillateurs désaccordés est une dissipation (~K·u_filter≈0.27 comparable au gain net
     0.2) → le réseau couplé vit sous le seuil effectif, jamais à l'équilibre en 6000 pas ;
     et la lecture différentielle contre un réseau de référence sans stimulus confond
     « doute monté » avec « évidence positive » (cicatrice u). Réparé loyalement par un
     readout en **paire différentielle** (+stim/−stim, même bruit) : 100 % de bascule sur
     FROZEN, tâche loyale.
  2. **La cicatrice u RETARDE la sortie de tromperie** : flip FULL = 5275 pas vs FROZEN =
     3467 (+52 % ; +~2500 pas ≈ 1.25 τ_u aux pulses longs ; à T_pulse=4500, 2/12 problèmes
     ne basculent plus dans le budget 9000). Le conflit fait monter u → couplage coupé →
     la trace du leurre se verrouille au lieu de s'effacer. **Sur STNO, le doute-dans-la-
     dynamique est un handicap pour la décision trompeuse** (avec le capteur calibré
     gain=10 du 09/07 ; dépendance au gain non balayée).
  3. **L'horloge de délibération de B5b ne se transpose PAS** : |S| aveugle (désaccord de
     phase permanent), |L·p| fond permanent (chute de 12 % seulement), u aveugle (piloté
     par |S|), et même le désaccord d'évidence entre les bras de la paire (lissé court ou
     long) ne bat pas le budget fixe (DOUBT_PAIRL s'arrête réellement sur FROZEN aux longs
     pulses, 7631 pas en moyenne, mais FIXED global fait 6600 à accuracy égale 1.00).
     **Le pattern conditionnel gagne une dimension : la valeur du doute dépend du SUBSTRAT,
     pas seulement de la tâche** — sur FHN (contractant, le désaccord retombe à la
     résolution) l'horloge est gratuite ; sur STNO (oscillant, bruité, désaccord permanent)
     elle est noyée et le meilleur arrêt reste le budget fixe.

### B6 — Prédiction falsifiable / signature expérimentale 🟢 appuyée par un résultat en silico (09/07/2026)
- **Pourquoi.** Tout est auto-référentiel (H, sync, MI calculés sur le même v(t) simulé).
  Manque une prédiction qu'un manip pourrait réfuter.
- **Comment.** Identifier une signature du doute mesurable sur un dispositif réel, distincte
  d'un système sans doute (ex. réponse spectrale, hystérésis, statistique de commutation).
- **✅ Proposition concrète (09/07/2026), maintenant appuyée par un test en silico (même
  jour).** S'appuyer sur le résultat le plus robuste et le mieux quantifié du projet —
  l'ablation FROZEN_U (Cohen d≈9 sur 30 seeds, B4, 8 juillet) — plutôt que sur une
  signature énergétique (pas d'équivalent expérimental évident, cf.
  `docs/hardware/B3_ENERGY_COMPARISON.md` §5). **Prédiction falsifiable proposée** : un
  petit réseau de STNO physiques couplés, avec un gain de couplage modulé par le
  désaccord local (`u`, polarité inversée au-delà du seuil), devrait montrer une
  synchronisation **significativement plus faible** qu'un réseau identique à couplage
  fixe (contrôle FROZEN_U), mesurable par spectroscopie micro-onde standard (méthode
  déjà utilisée par Romera et al. 2018). **Le modèle STNO macrospin minimal prévu comme
  prérequis a été construit et testé le jour même** (`b2_stno_phase_coupling_poc.py`,
  réduction phase-oscillateur, pas LLG) : le mécanisme réduit bien la synchronisation
  sur ce substrat (Cohen d 1.05–14.85 selon calibration du capteur de désaccord) — la
  prédiction falsifiable n'est donc plus une simple analogie, elle est appuyée par un
  résultat en silico reproductible. Reste falsifiable de la même façon : un effet nul
  ou de signe opposé sur un vrai dispositif réfuterait le transfert au substrat physique.
- **✅ Renforcé le jour même par la généralisation amplitude+phase** (`b2_stno_amplitude_phase_poc.py`,
  cf. B2 et `SPINTRONIC_PATHWAY.md` §8) : le mécanisme reste robuste (Cohen d 1.79–5.49)
  quand on ajoute la non-isochronicité — la signature physique la plus caractéristique
  des vrais STNO, absente du premier test. La prédiction falsifiable repose maintenant
  sur 2 modèles convergents (Kuramoto pur et auto-oscillateur Slavin-Tiberkevich), pas
  un seul. Réserve inchangée : au capteur brut (non calibré), l'effet est nul dans les
  deux modèles — la prédiction telle que formulée suppose implicitement qu'un vrai
  circuit de détection de désaccord aurait un gain suffisant, hypothèse non vérifiée.
- **✅ Confirmé le jour même par le macrospin LLGS complet** (`b2_stno_macrospin_llgs_poc.py`,
  cf. B2 et `SPINTRONIC_PATHWAY.md` §9) : la prédiction repose maintenant sur **3 modèles
  convergents** (Kuramoto, Slavin-Tiberkevich, LLGS vectoriel complet), le dernier étant le
  plus direct (pas de réduction phénoménologique). **Nuance importante découverte ici** :
  la géométrie de couplage testée verrouille en **antiphase**, pas en phase — si la
  prédiction B6 est un jour testée sur un vrai réseau STNO couplé électriquement (le canal
  réel de Romera et al. 2018), il faudra vérifier quel type de verrouillage ce canal
  favorise avant d'attendre une réduction de `R` plutôt que de `R2`. La prédiction reste
  falsifiable, mais sa formulation exacte (quel paramètre d'ordre observer) dépend du canal
  de couplage physique choisi — à préciser avant toute campagne expérimentale réelle.
- **⚠️ Nuancé le 12/07/2026 par P12 (tâche trompeuse B1d sur STNO, cf. B5)** : au niveau
  de la DÉCISION (pas de la synchronisation), le couplage modulé par le désaccord
  **retarde** la récupération post-leurre (+52 % de temps de flip vs couplage figé) au
  lieu de l'améliorer. La prédiction falsifiable B6 gagne donc un **second volet,
  au signe inversé et tout aussi testable** : « dans un réseau de STNO à couplage
  modulé par le désaccord, la synchronisation est réduite (volet 1, confirmé sur 3
  modèles) ET la récupération après un leurre transitoire est retardée d'environ τ_u
  par rapport au même réseau à couplage fixe (volet 2, P12) ». Un labo qui mesurerait
  une récupération plus RAPIDE réfuterait le volet 2. Ne pas vendre B6 comme « le doute
  améliore les décisions du dispositif » — ce n'est pas ce que la simulation dit.
- **Reste (🧩).** Non testé en circuit réel ni en micromagnétisme spatial complet
  (texture de vortex résolue, mumax3, ou modèle de Thiele) — palier explicitement
  reporté par Julien à une décision séparée (nécessite d'installer mumax3/CUDA, campagnes
  de plusieurs heures). Le canal de couplage électrique réel n'a pas été modélisé.

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

## D. Fils exploratoires hors preprint

### D1 — Genèse 5 états (ψ∈ℂ⁴ + Oracle) ✅ CONSOLIDÉ statistiquement (11/07/2026), piste requalifiée
- **Pourquoi.** Mem4ristor est né (Session 1 du Café, 19/08/2025) comme 5 états cognitifs ;
  9 mois de rigueur l'ont réduit au scalaire u. Premier jouet le 10/07 (Labo de l'Absurde,
  60 essais sans IC) : tendance apparente « moins linéaire = mieux ».
- **✅ Fait (11/07/2026)** — `experiments/genesis_five_states_poc.py` (commit `cb36b4a`),
  1000 essais, IC Wilson + bootstrap apparié, **gate de réplication exacte** des seeds du 10/07.
  **La tendance du 10/07 était du bruit** (lecture locale ~50-52 % partout) ; le hop
  multiplicatif fait pire que le hasard (38.1 %) ; l'Oracle actuel n'est pas un marqueur de
  réussite. **MAIS : l'interférence (moyenne complexe) préserve l'info de parité sur un
  plateau stable 73.9 % (t=20→150) et, lue globalement (produit des phases dominantes),
  bat le vote : +5.5 pts IC[+3.5,+7.5] p<1e-4.** Le goulot du 10/07 = la lecture locale
  (−21.4 pts), pas le réseau.
- **Prochaines marches (si la piste revient).** Readout local *appris* (le plateau est-il
  lisible localement ?) ; N>5 ; tâche où la phase compte sans prior de parité dans le readout.
  Réserve : le readout global encode un prior de tâche — le mérite démontré est la
  *préservation*, pas le calcul spontané.
- **Effort.** 💭 1 session par marche.

### D2 — Pont M4R↔LLM (anti-effondrement de rang) ✅ TÂCHE AVAL FAITE (11/07/2026), niche conditionnelle confirmée
- **Pourquoi.** Idée de Julien (08/07) : le couplage modulé par le doute contre
  l'oversmoothing/rank collapse. POC rang concluant le 08/07 (`llm_doubt_rank_poc.py`),
  réserve explicite : utilité aval non prouvée.
- **✅ Fait (11/07/2026)** — `experiments/llm_doubt_downstream_poc.py` (commit `d119604`),
  tâche double loyale (débruitage de groupe = exige le mélange ; identité individuelle =
  punit la fusion), contrôles de loyauté, 10 seeds, profondeur d'arrêt par validation pour
  TOUTES les conditions. **(1)** Avec early-stop réglé : l'attention pure à l=1-2 bat le
  doute de 0.8 pts — le budget fixe reste devant (pattern B5b). **(2)** Sans réglage
  (L=40 fixe, régime réel d'un transformer) : **le doute domine — groupe +4.6, identité
  +33.7 pts** ; fenêtre fragile (85→52 %) vs plateau stable (~85 %). **Utilité aval réelle
  mais conditionnelle = 3e réplication du positionnement B1d/B5b** (le doute paie quand
  l'horizon/la profondeur ne peut pas être réglé d'avance).
- **Prochaines marches.** Se mesurer à une mitigation *standard* (résiduel+MLP) sur la même
  tâche ; puis un petit transformer réel. Le doute doit être compétitif *dans la famille*
  anti-effondrement, pas seulement meilleur que la pathologie nue.
- **Effort.** 💭→🧩 selon la marche.

### D3 — Couche d'Abstention Calibrée (idée de Julien, PEPIT 11/06) ✅ (a)+(b) FAITS (12/07/2026), compas composite validé
- **Pourquoi.** « Ne décide pas, décide quand ne pas décider » : u au-dessus d'un modèle
  prédictif quelconque. Prérequis jamais mesuré : u est-il calibré ?
- **✅ (a) Calibration (12/07)** — `experiments/doubt_calibration_poc.py` (commit `04ea50a`) :
  u n'est pas naïvement calibré ; verdict initial « inversé » (r=−0.29 à B=800).
- **✅ (b) Abstention (12/07, même session)** — `experiments/p6b_abstention_poc.py` :
  le collatéral de (a) est tranché — **artefact de readout** (réponse FHN adaptative :
  transitoire fort puis rebond sous baseline ; signal en régime ~−0.03 vs décorrélation
  net/ref ±0.05 → labels instantanés à moitié aléatoires). Labels reconstruits au readout
  LISSÉ (W=200) : **l'« inversion » de (a) ne tient pas** (r(u)=+0.12 à B=800) — u seul
  n'est pas un compas ; il marche dans le sens naïf à budget court (r=+0.74 à B=400).
  **La Couche d'Abstention, elle, existe : composite (u, |Lv|, t_consensus, stabilité)
  en validation croisée groupée par seed : +38.3 pts à B=400 (46.7→85.0 %) et +25.0 pts
  à B=800 (68.3→93.3 %) à 50 % de couverture.** L'intuition de Julien « un consensus venu
  vite est suspect » validée en isolation (t_consensus : r=+0.45, +16.7 pts à B=800).
  Limite : à B=1600 les labels restent corrompus (décorrélation lente) — readout
  long-budget = problème ouvert. ⚠️ Réserve de propagation : B1d/B5b (07-08/07) utilisaient
  le readout instantané ; comparaisons relatives probablement robustes, accuracies absolues
  bruitées — re-vérification au readout lissé saine avant citation.
- **Reste.** (c) le backtest 0 € (paris préenregistrés / réponses LLM / investissement
  virtuel) avec la RECETTE composite (signaux + CV), pas les poids. **Effort.** 🧩.

---

## Dépendances rapides
- A3 alimente A5 (mesure H_cont per-seed) et B4 (IC sur labels mesurés).
- B1 débloque B3 (énergie/tâche) et B5 (comparaison SOTA).
- B2 débloque B3 et B6 (physique du dispositif). **09/07/2026** : B2 a livré 3 dossiers
  de correspondance (photonique/spintronique/électrique), débloquant un premier B3
  (cadré) et un premier B6 (proposition falsifiable) — mais la simulation physique
  réelle (LLG, SPICE) reste à faire avant que B2 soit clos.
- C1 dépend de A5 (métrique continue).
