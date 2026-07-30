# Mem4ristor — Travaux futurs (backlog priorisé)

> **But.** Ne rien perdre des pistes ouvertes. Chaque entrée est autonome :
> *Pourquoi / Comment / Effort / Statut*, lisible à froid par un futur agent.
> **Origine.** Audit externe simulé « posture d'une chercheuse de référence en neuromorphique » du 2026-07-06
> ([docs/audit_externe_neuromorphique_2026-07-06.md](audit_externe_neuromorphique_2026-07-06.md))
> + mandat de réfutation λ₂ du 2026-07-01 (`experiments/lambda2_foundation_20260701/SYNTHESE.md`).
> **Mise à jour.** 2026-07-26, soir (question du `std(v)` **soldée** — voir E5 : le signal
> d'arrêt ne tire aucun bénéfice de la topologie, et l'effet « la lire coûte » entrevu en
> chemin est mort à son gate de réplication).
> Précédente : 2026-07-26, jour (audit externe « DeepMind » SOLDÉ en entier — voir section E ;
> cadrage « bon marché » corrigé ; B3 complété par le pont opérations↔énergie).
> Précédente : 2026-07-11 (genèse 5 états consolidée D1, pont LLM tâche aval D2,
> [13] révisé au code actuel, B5-STNO NARMA10 fait — voir sections correspondantes).
> **Réservoir d'idées complémentaire** (écartées trop tôt, jamais tentées, garde-fou des
> impasses) : [PISTES_POUR_LA_SUITE_2026-07-12.md](PISTES_POUR_LA_SUITE_2026-07-12.md)
> — le legs de Fable, 14 pistes sourcées (bicaméral V5b, MoE par certitude, usure/drift,
> graphes dirigés, abstention calibrée, u∈ℂ dans le cœur…).


Légende statut : ✅ fait · 🔜 prêt à démarrer · 🧩 projet (plusieurs jours/semaines) · 💭 exploratoire.

---

## Priorité recommandée (si on ne fait qu'une chose à la fois)

**26/07/2026 (soir) — CE QUI RESTE VRAIMENT.** Les trois critiques de l'audit externe
« DeepMind » sont soldées (section E), **et la première des deux questions nouvelles
également** (E5, le `std(v)`). Le backlog de tête est vide : tout ce qui reste ci-dessous
est soit un **projet de fond de plusieurs semaines** (B2 SPICE, couplage électrique réel,
micromagnétisme mumax3 — ce dernier réservé par Julien à une session **en personne**),
soit de la **théorie** (C1 k_harm analytique, C2 hétérogénéité), soit :
  - ✅ ~~pourquoi `std(v)` égale `mean(|L·v|)`~~ **RÉPONDU le 26/07 au soir — voir E5.**
    Le signal d'arrêt ne tire **aucun** bénéfice de la topologie : sur sept mesures
    indépendantes, l'écart est toujours ≤ 0 et jamais une fois en faveur du désaccord
    local. **5ᵉ rétrécissement de la revendication, et le mieux tenu des cinq.**
  - le **mécanisme par nœud** de la cascade topologique (deux explications proposées et
    réfutées le jour même : ni l'amplitude du champ reçu, ni la variance de la cible
    locale — le degré garde ρ partiel −0.37 après contrôle). **Seule question nouvelle
    encore ouverte.** Effort : 1 session.
  - la **dette de méthode B5b** (E4) : ~1 h, à faire avant toute citation au chiffre.

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
- **✅ ÉCONOMIE D'ÉCHELLE TESTÉE (13/07/2026)** — `experiments/b1d_scale_economy_poc.py`. Après
  une discussion honnête avec Julien sur ce que M4R fait réellement (« meilleur en recherche pure
  sans contrainte de temps, pire que tout sous contrainte » — confirmé par la journée entière de
  tests gamma_int et par B5/context-reinjection), question posée sur un AUTRE axe que la
  performance brute : **l'avantage du doute (B1d) survit-il avec beaucoup MOINS de nœuds** ? En
  matériel réel, N = nombre de dispositifs physiques — si le gain tient à N petit, c'est un
  argument d'économie matérielle plausible ; s'il s'effondre, ça borne la réponse.
  Protocole IDENTIQUE à B1d (même piège pulsé, mêmes seuils), seul SIDE varie (N=9/25/49/100,
  N_DISTRACT/N_TRUE gardés aux MÊMES proportions que l'original). **Résultat : le gain SURVIT à
  toutes les tailles testées** — N=100 : +0.58 ; N=49 : +0.75 ; **N=25 : +0.75** ; N=9 : +0.58
  (mais `acc_FIN`=0.58 à N=9, sous le seuil 0.6 de régime trompeur fiable — réserve honnête, la
  tâche elle-même devient moins résoluble à cette taille extrême, à distinguer du mécanisme
  doute-vs-convergence qui reste positif). **Réplication FAITE avec 20 graines DISJOINTES**
  (200-219, avant de conclure quoi que ce soit — la leçon du Condorcet du même jour) :
  N=25 → gain +0.400 (vs +0.75 initial, magnitude plus modeste mais direction et signe
  confirmés) ; N=100 → gain +0.500 (vs +0.58, très proche). **Contrairement au résultat
  Condorcet (mort à la réplication le même jour), celui-ci tient : direction et ordre de
  grandeur confirmés sur graines indépendantes.** Verdict : **une réduction de N par 4
  (100→25) ne coûte quasiment rien à l'avantage du doute** — le mécanisme ne dépend pas d'une
  grande population de nœuds, ce qui soutient (sans le prouver au niveau matériel réel, cf. B3)
  l'hypothèse d'une économie de composants sur la niche déjà établie (décision sous incertitude
  à horizon inconnu), distincte de la question de performance brute (déjà tranchée : non,
  cf. B5).

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
- **✅ Complété (26/07/2026) — le pont opérations↔énergie, et il ne va pas dans le sens
  espéré.** `experiments/expB3_substrate_crossover_poc.py`. La question posée : le cadrage
  « composant d'orientation bon marché » (13/07) redevient-il vrai sur un substrat où un
  pas coûte peu ? **Non, et la raison est structurelle.** Le fait décisif est que
  l'adversaire qui égale M4R sur sa niche — un filtre à oubli exponentiel — **est un
  circuit RC** : passer M4R en analogique y fait passer l'adversaire aussi, où il est
  passif. À substrat égal (agrégation par Kirchhoff, le régime physiquement honnête),
  **M4R coûte ~15× plus d'énergie, et le rapport ne dépend PAS du dispositif** — changer de
  substrat divise les deux côtés par le même facteur. M4R entretient 200 oscillateurs
  ACTIFS pendant 309 pas, le filtre un seul RC PASSIF pendant 1348.
  ⚠️ La comparaison flatteuse (M4R sur STNO contre un filtre compté en opérations Loihi)
  donne 2 500× à 8 700× en faveur de M4R : elle est calculée dans le script **et affichée
  comme le piège qu'elle est**, car elle compare deux substrats et non deux méthodes —
  exactement ce contre quoi la section 3 de `B3_ENERGY_COMPARISON.md` met en garde.
  **Conclusion à retenir** : le « bon marché » du projet était un argument sur le
  **substrat**, dont tout dispositif analogique bénéficierait identiquement, pas sur
  l'architecture. Ce qui appartient en propre à M4R est la **latence** (voir E2).
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
  - **✅ L'ANALOGIE LLM TESTÉE (13/07/2026)** — `experiments/b5_context_reinjection_poc.py`.
    Question de Julien : un LLM n'a pas de mémoire entre deux requêtes non plus, c'est la
    **reprise du contexte** (le prompt re-injecte l'historique) qui fait le travail — la même
    analogie s'applique-t-elle à M4R ? **Nuance posée avant de tester** : un LLM redémarre à
    ZÉRO entre appels (mémoire purement externe) ; M4R a déjà un état interne CONTINU (v, w,
    u, u_c ne sont jamais réinitialisés) — son problème n'est pas « pas de mémoire du tout »,
    c'est « l'état persistant n'encode pas ce qu'il faut pour NARMA10 ». Mais la SOLUTION de
    l'analogie (réinjecter explicitement l'historique brut plutôt que compter sur la mémoire
    interne) reste testable telle quelle : NARMA10 dépend explicitement de u[t−9] et u[t], donc
    on ajoute cette fenêtre brute de 10 valeurs au readout — EXACTEMENT ce qu'un contexte de
    LLM fournirait — de M4R ET de l'ESN (même augmentation aux deux, pour isoler ce qui est
    spécifique à M4R), + un contrôle « contexte SEUL, aucun réservoir ». Aucune nouvelle
    simulation réseau (le contexte s'ajoute au niveau du readout, pas de la dynamique).
    **Trois résultats, aucun arrondi :**
    1. **M4R ne gagne RIEN de significatif** (1.942→1.880, CI[−0.156,+0.253] couvre 0) — donner
       à M4R le même historique explicite qu'un contexte de LLM ne répare PAS son désavantage.
    2. **L'ESN, lui, gagne un peu** (0.351→0.335, CI[+0.011,+0.023] exclut 0, ~5%) — le contexte
       aide un modèle qui sait DÉJÀ exploiter la mémoire ; ce n'est pas un effet spécifique à la
       faiblesse de M4R, c'est une aide générale marginale.
    3. **L'écart M4R-ESN NE se referme PAS** (+1.59 sans contexte → +1.55 avec, quasi inchangé)
       — **même avec le MÊME historique explicite que l'ESN, M4R reste ~5,5× pire.** Le
       problème n'est donc PAS (ou pas seulement) la mémoire au sens LLM du terme.
    **Découverte annexe, la plus parlante** : une régression linéaire sur le CONTEXTE SEUL
    (aucun réservoir, juste u[t−9..t] brut) fait NRMSE=**0.829** — **MEILLEUR que M4R avec OU
    sans contexte** (1.88-1.94) ! Le réservoir FHN+doute de M4R fait donc **pire que ne rien
    faire** sur cette tâche précise : sa propre dynamique (l'exploration/anti-synchronisation)
    ne préserve pas l'information brute, elle la **brouille** activement au-delà de ce qu'une
    lecture linéaire directe des entrées récupérerait seule. **Verdict de la question de
    Julien : l'analogie LLM ne sauve PAS M4R ici** — pas parce que la reprise de contexte ne
    marche jamais (elle aide l'ESN), mais parce que le déficit de M4R sur NARMA10 est un
    problème de TRAITEMENT (la dynamique du doute transforme l'info dans un sens défavorable
    à cette tâche précise), pas seulement d'ABSENCE de mémoire externe à réinjecter. Cohérent
    avec le cadrage établi (M4R = explorateur, pas mémoire) — mais precise maintenant *pourquoi*
    au niveau mécanique, pas seulement au niveau du score final.
  - **✅ QUI EST RESPONSABLE ? TRANCHÉ (13/07/2026, même jour)** —
    `experiments/b5_context_conditions_poc.py`. Le contexte seul (0.829) bat M4R FULL —
    est-ce le DOUTE qui abîme l'info, ou l'architecture FHN+lattice elle-même ? Test des
    3 conditions déjà définies par B1 (`reservoir_narma10_poc.py`) contre le contexte seul :
    **les TROIS battent — perdent, plutôt — contre le contexte seul, sans exception.**
    FULL = 1.942 (delta +1.113 CI[+0.90,+1.31]) ; **FROZEN_U = 2.273, le PIRE des trois**
    (delta +1.445 CI[+1.03,+1.90] — geler le doute n'aide PAS, ça aggrave même légèrement) ;
    **DECOUPLE (nœuds FHN isolés, AUCUN couplage, AUCUN doute) = 1.697** (delta +0.868
    CI[+0.61,+1.11]) — **le meilleur des trois, mais toujours nettement pire que le
    contexte seul.** **Verdict : le problème n'est ni le doute (FROZEN_U ne fait pas
    mieux, il fait pire) ni le couplage spatial (DECOUPLE reste perdant même sans lui)
    — il remonte au NŒUD FHN INDIVIDUEL lui-même.** Un seul neurone FHN isolé, piloté par
    le signal brut, produit un état v(t) qu'un readout linéaire exploite MOINS BIEN que
    les entrées brutes elles-mêmes. Hypothèse mécanique à vérifier plus tard (hors
    session) : la relaxation rapide/lente (type spike) du FHN pourrait saturer/compresser
    l'amplitude plutôt que préserver un mélange non-linéaire graduel exploitable, contrairement
    au tanh lisse d'un ESN. Le doute et le couplage ne sont donc pas la cause du désavantage
    NARMA10 de M4R — ils s'ajoutent à un déficit déjà présent dans le nœud de base.
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
- **✅ (c) Backtest 0 € FAIT (13/07/2026)** — `experiments/scratch/p6c_backtest_poc.py`
  (cette entrée disait encore « Reste » jusqu'au 28/07 : oubli de mise à jour, le
  travail était fait depuis le 13/07, commit `e5d3ed9`). Domaine « investissement
  virtuel » synthétique aux statistiques **génuinement** différentes de B1d (forces,
  effectifs et durée du faux breakout tirés par essai au lieu d'être fixes), 60
  épisodes, critère de succès identique à P6b (r_pb > 0.15 **ET** gain@50 % > +3 pts).
  **La recette transporte, largement** : ALWAYS-TRADE = 63.3 % ; COMPOSITE_CV ré-appris
  sur le nouveau domaine = **+30.0 pts @ 50 % de couverture (63.3 → 93.3 %), r_pb =
  +0.463** — le meilleur résultat d'abstention du projet à ce jour, au-delà de P6b.
  **Ce qui est démontré, précisément** : c'est la **recette** (les signaux + la CV
  groupée) qui transporte, pas les poids — le signal individuel gagnant de B1d
  (`conf_u_inv`) transporté tel quel fonctionne aussi (+13.3 pts) mais **nettement
  moins bien** que le composite ré-appris. **Réserve** : domaine **synthétique**, pas
  un vrai backtest sur données de marché ni sur réponses de LLM ; le PEPIT_LOG parlait
  de paris préenregistrés / réponses LLM / investissement virtuel — seul le troisième
  est couvert, et en simulation. **Reste (🧩) :** un domaine à données réelles.

---

## E. Audit externe « Google DeepMind » (Gemini, 17/07/2026) — SOLDÉ EN ENTIER

> Les trois critiques de cet audit ont été traitées. Elles sont consignées ici parce que
> deux d'entre elles ont produit les résultats les plus précis du projet sur sa propre
> niche, et parce que la troisième a failli être racontée de travers.

### E1 — « Friston / énergie libre » ✅ TRAITÉE (17/07/2026)
Un draft interne (`docs/draft_friston_free_energy.md`) ressuscitait une thèse déjà rejetée
deux fois par les audits internes, en portant une étiquette « validée par la Red Team »
fausse. **Retiré** avec bandeau explicite (marqué, pas caché), commit `5209d0b`.

### E2 — « Quelle classe computationnelle bat une exploration purement aléatoire ? » ✅ FAIT (26/07/2026)
`experiments/expB_annealing_faceoff_poc.py` (+ B2/B3/B4/B5/B6-bis). Harness B1d/B5b exact,
40 graines, hyperparamètres des adversaires choisis sur graines d'entraînement et mesurés
sur graines **disjointes**.
- **Les adversaires cités sont battus** : recuit simulé 0.45 (signal) / 0.50 (flux brut)
  contre 0.90 ; exploration stochastique pure 0.38 / **0.00**. L'exploration aléatoire ne
  bat rien ici — c'était littéralement la question.
- **Mais un filtre à oubli exponentiel atteint 1.00**, et **ce n'est pas son exploration** :
  privé de son ε, il fait aussi bien. Ce qui bat le doute est un **passe-bas**, et la
  différence avec l'intégrateur pur (0.00) est l'**oubli**, pas l'aléa.
- **Ce que le doute apporte, isolé** (`expB5-bis`, `expB6-bis`) : il ne lit pas mieux — à
  instant fixe le réseau est à 0.40 — il sait **quand** lire. L'arrêt vaut **+0.49 à +0.61**,
  et bat de loin des instants de même distribution tirés sans lien au run (0.46-0.57).
  **Latence : 4.4× moins de pas**, y compris depuis l'intérieur de la fenêtre trompeuse.
- ~~**Reste ouvert**~~ ✅ **SOLDÉ le 26/07 au soir → voir E5.** ⚠️ La formulation
  « égale **voire dépasse** » écrite ici et dans le commit `a4b4e76` était **plus forte que
  les chiffres** : l'intervalle touche zéro (−0.07, IC [−0.17, **+0.00**]). Ce qui est
  établi, c'est que `std(v)` **n'est pas moins bon**, pas que le désaccord local soit moins
  bon. Corrigé, pas réécrit.

> ⚠️ **LE PENDANT NÉGATIF, MESURÉ LE 27/07 AU SOIR — la réponse à E2 est CONDITIONNELLE À LA
> TÂCHE, pas un verdict général.** Sur la niche établie (décision trompeuse en ligne, harness
> B1d/B5b), l'exploration purement aléatoire ne bat rien : **0.00**. Sur le **Max-Cut**
> (`p15_maxcut_ising_poc.py`, tâche hors niche), c'est l'inverse et l'écart est franc : à
> budget d'échantillons **égal** — M4R lit `sign(v)` 300 fois sur 3000 pas, on tire 300
> configurations de spins uniformes — le **meilleur de 300 tirages aléatoires bat M4R sur
> 10/10 graines puis 10/10 graines neuves** (aléatoire 91.50 / 89.10 ; FROZEN_U 82.10 / 80.50 ;
> FULL 80.90 / 79.80). Critère ≥ 8/10 posé avant mesure ; diagnostic et gate de fidélité dans
> `experiments/p15b_maxcut_identity_diagnosis.py`. **Ne jamais citer E2 comme « le doute bat
> l'exploration aléatoire » sans nommer la tâche.**
> ~~Reste **non tranché**~~ ✅ **TRANCHÉ le 28/07/2026** — `experiments/p15c_maxcut_identity_mechanism.py`
> (gate de fidélité G0 passé, critères posés avant chaque volet, gate de réplication sur les
> graines **20-39** jamais touchées). Pourquoi 10 graines sur 20 donnaient une coupe **et** une
> énergie strictement identiques entre FULL et FROZEN_U — **quatre réponses, dont deux défauts
> d'énoncé, et aucune ne concerne `u`** :
> 1. **L'énoncé comptait deux fois la même chose.** `cut = 0.25·Σ|J| + 0.5·E` exactement
>    (bijection affine lisible dans `compute_cut_and_energy`) : résidu **nul sur 20/20**, les
>    deux tests d'identité coïncident sur 20/20. Le « **ET** énergie » — ce qui rendait
>    l'observation frappante — ne portait aucune information. Établi **sans simulation**.
> 2. **Ce n'est pas une coïncidence de valeur, c'est le même état** : `best_s(FULL)` et
>    `best_s(FROZEN_U)` sont **le même vecteur de spins** sur 9 graines identiques sur 10,
>    et **11 sur 11** en réplication. La question « pourquoi la même coupe » se dissout.
> 3. **La piste laissée au froid est morte** (7ᵉ explication rejetée à son propre critère) :
>    deux séries de 300 tirages uniformes **indépendantes** collisionnent à **0.10** (0.15 en
>    réplication) contre **0.50** observé — facteur 5.
> 4. **Le mécanisme, répliqué** : le record est battu dans un état de signes que les **deux
>    conditions occupent au même instant** — 0.90 puis **1.00** chez les graines identiques
>    contre 0.30 puis **0.22** chez les autres (critère posé avant : ≥ 0.70 et écart ≥ 0.30).
>    Quand elles diffèrent, c'est un désaccord d'**exploration**, jamais de sélection (0 cas
>    sur 10 où les deux optima sont vus des deux côtés).
>
> **⚠️ LE FAIT DUR, NON CHERCHÉ, ET LE PLUS UTILE DES QUATRE.** Sur **300 lectures**, M4R ne
> visite que **~24 états de signes distincts en FULL et ~40 en FROZEN_U** (25 / 45 en
> réplication), pour un **n_eff = 5** — il paie 300 lectures l'équivalent de 5 tirages
> indépendants, **facteur 60**. Le réseau est quasi immobile : l'identité n'a plus rien
> d'étonnant, c'est l'inverse qui demanderait une explication. Trois conséquences :
> - **Le doute explore MOINS que le doute gelé** (24 contre 40 états) et il fige la dispersion
>   de la coupe là où `FROZEN_U` la garde : sd early→late **3.01 → 0.47 en FULL sur 20/20
>   graines**, contre 3.01 → 2.01 en FROZEN_U sur **8/20** seulement. Sur cette tâche, `u` est
>   un frein à l'exploration. **3ᵉ contradiction de la phrase « grâce au doute »** de l'en-tête
>   d'origine, après l'ablation du 27/07 au matin et le hasard du 27/07 au soir.
> - **R2 et R5 (27/07) avaient le même angle mort** : ils testaient la **moyenne** de la coupe,
>   qui ne bouge pas (72.9 → 73.8). C'est la **variance** qui s'effondre. Un critère posé sur la
>   mauvaise statistique ne se rattrape pas en repêchant son seuil.
> - Le transitoire lui-même **n'est pas** un tirage uniforme (sd 3.01 contre **6.03** mesurés sur
>   le même graphe) : M4R est déjà deux fois moins dispersé que le hasard **avant** de se figer,
>   ce qui suffit à expliquer sa défaite contre best-of-300.
>
> **✅ SUITE IMMÉDIATE (28/07, `experiments/p16_state_mobility_on_niche.py`) — le fait dur
> a été porté SUR LA NICHE, et l'attribution au doute est tombée.** Harnais B1d exact
> (gate de fidélité : `acc_final` 0.7500 et `acc_conv` 0.2500 reproduits au chiffre près
> sur les graines du POC), **quatre** bras au lieu de deux, gate de réplication sur les
> graines 20-39.
> - **Le fait se réplique, massivement** : 20/20 puis 20/20 graines disjointes. Sur la
>   niche, le doute visite **35 états distincts contre 75** pour `FROZEN_U(0.5)`.
>   Ce n'était donc pas un accident du Max-Cut.
> - **⚠️ Mais l'attribution au doute est FAUSSE, et dans le sens inverse de celui qu'on
>   pouvait craindre.** À `u` **figé à 0.95** — le niveau même qu'atteint le doute
>   (`u_end ≈ 0.88`) — le réseau visite **encore moins** d'états : **15.7 contre 35.0**,
>   20/20 puis 20/20. **L'adaptativité du doute MODÈRE l'immobilisation, elle ne la
>   produit pas.**
> - **Le signe tranche, la force non** : à force de couplage égale, le synchronisant
>   (`u=0.05`) visite **75.5** états contre **15.7** pour l'anti-synchronisant — soit le
>   même ordre que le couplage **nul** (75.3). **Ce qui immobilise est
>   l'anti-synchronisation**, pas l'intensité du couplage. C'est un figement et non un
>   cycle (Hamming consécutive 0.18 / 0.29 / 0.51 / 0.56 nœuds sur 100).
> - **Formulation défendable** : « le doute explore moins » est **vrai contre
>   `FROZEN_U(0.5)` et FAUX contre `FROZEN_U(0.95)` ** — ne jamais l'écrire sans nommer le
>   comparateur. Le comparateur historique `u=0.5` est un réseau **quasi découplé**
>   (`u_filter ≈ 0`), pas un réseau « sans doute ». La caractérisation **« explorateur
>   discipliné » (08/07) n'est pas renversée** : elle gagne une précision — la discipline
>   se paie en mobilité.
> - **🛡️ Le garde-fou colonne A a été posé, il est tombé, et la vérification a montré
>   qu'il n'y avait pas d'alerte.** Sur `H_cont` — l'observable de Table 1 — FULL 3.609 <
>   FROZ_05 3.828. **Avant d'écrire quoi que ce soit** : (a) `b4_ablation_robustness.csv`,
>   dans le régime du preprint, donne **déjà le même ordre** (`full_hcont` 3.645/3.674
>   contre `frozen_hcont` 4.327/4.653) ; (b) la légende de `tab:ablations` dans
>   `preprint.tex` l'écrit noir sur blanc — l'entropie instantanée n'y est pas rapportée
>   car elle *« donne des résultats directionnellement incorrects pour cette
>   comparaison »*. Le preprint ne revendique **rien** sur `H` entre FULL et FROZEN : sa
>   revendication est la **synchronie** (0.002 contre 0.697) et la complexité LZ.
>   *(valeurs mises à jour le 30/07 par la passe soustractive : ce paragraphe, écrit le
>   28/07, citait encore `0.031 contre 0.751` — les valeurs du CSV du 26/04, rendues
>   périmées par la régénération de `tab:ablations` du 29/07. L'argument est inchangé.)*
>   **La colonne A n'est pas approchée, et elle avait anticipé le point avant nous.**
>   *Leçon : un garde-fou qui tombe n'est pas une alerte tant qu'on n'a pas vérifié ce que
>   la cible affirme réellement.*
>
> **Portée.** Tâche **hors niche**, où M4R perd déjà contre le tirage aléatoire à budget égal.
> C'est l'explication d'une identité de **mesure**, pas une propriété de `u` — exactement ce qui
> était annoncé avant d'ouvrir la question. Le **−1.20 FULL−FROZEN_U reste à ne pas citer**, et
> pour une raison de plus : c'est l'écart entre deux maximums pris sur ~24 et ~40 états, souvent
> le même.

### E3 — « L'entrée en dead zone sur BA est-elle une cascade initiée par les hubs ? » ✅ FAIT (26/07/2026)
`experiments/expA_ba_cascade_poc.py` (160 runs) + `expA_mechanism_drive_poc.py`.
- **Oui, le phénomène existe** : ρ(degré, temps de verrouillage) = −0.56, **10/10 graines
  négatives en primaire ET sur graines disjointes** ; les hubs se verrouillent ~490 pas
  avant la périphérie ; le sous-graphe vivant éclate en 39.9 composantes contre 7.9 sous
  retrait aléatoire du même nombre de nœuds.
- **Mais l'attribution de l'audit est fausse sur trois points** : l'effet disparaît sous
  normalisation par degré (c'est une propriété du **protocole de couplage**, pas du
  réseau) ; il n'est **pas spécifique au scale-free** (ER à ⟨k⟩ comparable : −0.61) ; il
  n'est pas un artefact d'indice (relabeling aléatoire : inchangé) ni de binning (tient
  sous v*).
- **Mécanisme par nœud : OUVERT.** Deux explications proposées, deux réfutées le jour même —
  ni l'amplitude du champ reçu (partiel deg|drive = −0.37, le champ n'absorbe pas le degré),
  ni la variance temporelle de la cible locale (ρ ≈ +0.03), car l'argument d'échantillonnage
  du 01/07 est **cinématique** et suppose des voisins indépendants, ce que la dynamique
  couplée viole. Effort : 1 session.

### E4 — PIÈGE DE MÉTHODE À NE PAS REJOUER (découvert et corrigé le 26/07/2026)
Régler un adversaire « à son maximum par **oracle par run** » (le meilleur de N combinaisons
d'hyperparamètres, choisi run par run en connaissant la bonne réponse) **fabrique du
résultat** : sur un signal sans aucune information (bruit pur), cette procédure rend
**0.935 d'accuracy au lieu de 0.500**, parce qu'avec 18 combinaisons et une décision binaire
il s'en trouve presque toujours une qui tombe juste. Le biais joue **contre** le composant
qui n'ajuste rien.
- **Où c'est corrigé** : `expB_annealing_faceoff_poc.py` et `expB2_wiring_budget_poc.py`
  sélectionnent désormais sur graines d'entraînement et mesurent sur graines disjointes ; le
  contrôle « bruit pur » est rejoué et imprimé à chaque exécution.
- ~~⚠️ **Où le piège subsiste**~~ ✅ **DETTE PAYÉE le 27/07/2026 au soir**
  (`experiments/b5b_bis_deceptive_traintest.py`, harness B5b identique, 20 graines TRAIN +
  20 graines TEST **disjointes**, tout choix — hyperparamètres, règle d'arrêt, budget — fait
  sur TRAIN et figé). Les **trois** endroits où le piège était présent dans
  `b5b_deceptive_exploration.py` sont corrigés : `esn_best_by_oracle` (combo choisi run par
  run), `np.maximum(acc[DROP], acc[CONV])` (meilleure règle prise graine par graine) et
  `best_fixed` (budget choisi sur les données de mesure).
  - **Le piège était RÉEL DANS LE CODE mais SANS EFFET sur les chiffres.** Contrôle par
    permutation d'étiquettes sur les vraies trajectoires : les 6 combinaisons (ρ, fuite)
    rendent des décisions **identiques dans 100 % des problèmes**, sur les trois canaux que
    la sélection regarde (décision finale, décision à l'arrêt DROP, à l'arrêt CONV). Sans
    divergence entre combinaisons, un oracle par run n'a rien à trier : oracle-par-run sous
    permutation **0.55** au lieu du 0.935 constaté ailleurs dans le projet. Le critère de
    pouvoir posé avant mesure (≥ 0.65) n'est **pas** atteint — rapporté tel quel, sans en
    faire une victoire. La conclusion de B5b n'était donc pas seulement conservative : elle
    n'était pas biaisée du tout par ce mécanisme.
  - **Résultat sur graines TEST** : doute natif **0.90** ; M4R convergence 0.40 ; M4R meilleur
    budget fixe 0.80 ; ESN meilleure règle naïve (DROP) **0.00** ; ESN meilleur budget fixe
    (B=800) **1.00**.
  - **V2, le seul test qui engage quelque chose** : −0.10, IC [−0.25, **+0.00**]. **Non
    concluant au sens strict** (l'IC touche zéro) et l'écart ponctuel va dans le sens de
    l'ESN. Ce qui est **établi** : le doute n'est **pas meilleur** qu'un horizon fixe bien
    choisi. Ce qui ne l'est **pas** : qu'il soit pire. **La conclusion du 08/07 tient, et elle
    est maintenant citable.** Noter que B=800 dépasse le leurre le plus long (700) : un budget
    fixe supérieur à l'horizon **maximal** suffit — ce qu'un horizon inconnu interdit de choisir.
  - ⚠️ **V1 (+0.90, IC [+0.75, +1.00]) est à ne pas survendre** : l'arrêt naïf de l'ESN est au
    **plancher (0.00)**, son signal se déclenche pendant le leurre. Un adversaire qui répond
    systématiquement faux n'est pas une baseline, c'est une baseline en panne. Le script
    imprime cet avertissement automatiquement dès que l'adversaire tombe sous 0.10.
  - 🔍 **Défaut d'instrument rencontré et conservé** : le premier contrôle écrit ce soir-là
    (flux nul à stimulus constant) rendait **0.500** pour la procédure corrigée — le bon
    chiffre — mais **0.550** pour le témoin censé montrer le piège, parce que ce flux
    dégénéré ne produit aucune divergence entre combinaisons. Il ne testait rien. **Un
    contrôle qui donne le bon chiffre pour la mauvaise raison est un contrôle en panne, pas
    un feu vert** ; il est laissé dans le script, marqué invalide.

### E5 — « Pourquoi un simple `std(v)` égale-t-il le désaccord local ? » ✅ FAIT (26/07/2026, soir)

Question née de E2, traitée en quatre volets (`expB7_*`, commits `cb3dc1c`, `7c6c30c`,
`a866eac`). **Réponse : le signal d'arrêt ne tire aucun bénéfice de la topologie.**

- **Volet 1 — le diagnostic** (`expB7_spatial_structure_diagnostic_poc.py`). Les deux
  signaux sont **une seule lecture** : r = 0.981 IC[0.976, 0.985] sur les signaux
  **normalisés par leur pic** — c'est-à-dire sur exactement ce que voit la règle d'arrêt.
  Et la tâche n'a **aucune structure spatiale** à exploiter : Moran's I du champ de
  stimulus = **−0.010** IC[−0.039, +0.021] (lu dans le code de la tâche, pas deviné :
  `deceptive_task_poc.make_deceptive` disperse les 26 leurres et 14 vérités par
  `rng.choice` sur le tore, donc être voisin ne dit rien de ce qu'un nœud mesure). Le
  réseau n'en fabrique pas non plus : Moran's I de `v` ≤ **0.022** à tous les instants
  sondés. ⚠️ Une explication « d'échelle » a été proposée puis **réfutée** : les deux
  signaux déclenchent à 39 pas d'écart, mais retarder le natif du décalage exact — estimé
  sur graines d'entraînement à partir des seuls **instants d'arrêt**, jamais de la justesse
  — ne lui fait rien gagner (0.90 → 0.90).
- **Volet 2 — le test décisif** (`expB7_contiguous_sources_poc.py`). Même tâche, mêmes
  effectifs, mêmes amplitudes, même graphe : seules les **positions** changent (sources
  groupées en patchs, Moran's I du stimulus **+0.382**, porté par le champ `v` à +0.125).
  **Rien ne change** : interaction (LAP−STD)_structuré − (LAP−STD)_dispersé = **+0.02**
  IC[−0.08, +0.12]. Le contrôle est **structurel** : permuter les étiquettes de nœuds du
  monde groupé redonne exactement le monde dispersé, qui en est donc le null de permutation.
  ⚠️ Premier essai **échoué à son propre gate** et conservé : un bloc unique par rôle
  (Moran's I +0.675) ne durcit pas la tâche, il la **casse** (acc_final 0.50 = hasard, contre
  0.78) — concentrées, les sources de vérité saturent leur région et ne remontent plus dans
  le readout global `mean(v)`. **Fait utile au passage : structure spatiale et solvabilité
  sont en tension sur ce readout.**
- **Volet 3 — la contre-épreuve du plafond** (`expB7_ceiling_control_poc.py`). `std(v)`
  marquait 1.00 : une baseline saturée ne peut pas révéler une supériorité du natif. Tâche
  durcie sur ses deux leviers, **aucun niveau écarté après coup**. Deux faits de méthode :
  (a) le plafond ne bloque que la détection d'une **supériorité** — une infériorité reste
  mesurable, donc le critère de lecture est le **signe** de l'écart ; (b) **allonger le
  leurre ne durcit rien** du point de vue de l'arrêt (les arrêts tombent vers 270-310 pas,
  bien avant sa fin — T=1200 reproduit T=700 à l'identique). Seule la **force** du leurre
  est un levier.
- **Volet 4 — le gate de réplication** (`expB7_replication_poc.py`), obligatoire ici parce
  que les récits topologiques ont déjà cassé deux fois ([13] révisé, P3 réfuté). Le volet 3
  avait fait apparaître un −0.30 IC[−0.45, −0.17] suggérant que **lire la topologie coûte**.
  Rejoué sur **deux groupes de graines jamais touchées et disjoints** (0-59 étant consommées),
  chacun avec son propre réglage de seuil : **0/2 confirment** (+0.00 IC[−0.07, +0.07] et
  −0.03 IC[−0.12, +0.05]). **L'effet est mort** — même motif que le Condorcet du 13/07.

**Ce qui survit, et c'est solide.** Sur **sept mesures indépendantes** (3 niveaux lisibles
au volet 3, 4 cellules au volet 4, graines disjointes, seuils réglés séparément), l'écart
`mean(|L·v|) − std(v)` est **toujours ≤ 0 et pas une seule fois** en faveur du désaccord
local. L'amplitude est instable ; **le signe ne l'est pas**. Formulation défendable :

> Le signal d'arrêt ne tire **aucun bénéfice de la topologie**. Il exige de l'information
> **spatiale** — un signal purement temporel s'effondre à 0.03 — mais une simple dispersion
> globale, aveugle à qui est voisin de qui, fait aussi bien **partout où on a regardé** :
> sur une tâche sans structure spatiale comme sur une tâche qui en a, à difficulté normale
> comme durcie.

Sans le « et la lire coûte » : cette partie-là était du bruit.

---

### E6 — « L'ablation FROZEN_U mesure-t-elle l'adaptativité de `u`, ou son niveau ? » ✅ FAIT (29/07/2026)

Née de la relecture du 28/07 (« nommer le comparateur »). Deux scripts :
`p17_frozen_u_level_vs_dynamics.py`, `p18_polarity_threshold_both_protocols.py`.

**⚠️ Correction d'une note du 28/07.** Il existe **deux comparateurs distincts nommés
`FROZEN_U`** dans le projet, à deux valeurs opposées du filtre
`u_filter = tanh(π(0,5−u)) + 0,01` :
- **colonne A / preprint** — `u` gelé à `sigma_baseline = 0,05` → filtre **+0,90**, couplage
  **synchronisant fort** (`ablation_coordination.py:112`, `p2_sigma_social_ablation.py`) ;
- **colonne B** — `u` gelé à **0,50** → filtre **+0,01**, réseau **quasi découplé**
  (`p15*`, bras `FROZ_05` de P16).

La note laissée dans SYNAPSE le 28/07 (« toute comparaison FULL/FROZEN_U du projet hérite de
l'ambiguïté u=0,5 = sans lien ») est **vraie de la colonne B et fausse du preprint**.

**RÉSULTAT 1 (P17, régime forcé).** Un `u` **figé** à 0,997 — constante, aucune adaptativité —
désynchronise **autant** que le doute adaptatif : synchronie **+0,0030 contre +0,0023**,
**10/10 puis 10/10** sur graines neuves ; LZ 1,109 contre 1,096. Un bras supplémentaire fige
le **profil par nœud** de FULL (même moyenne, hétérogénéité conservée) : écart 0,004 → **ce
n'est pas non plus l'hétérogénéité spatiale de `u`**. Contrôle d'instrument passé
(`max|u_fin − u_init| = 0` exactement, 112 runs figés).
→ *ce que l'ablation du preprint isole est la **polarité** du couplage, pas la **dynamique**
de `u`.* La Discussion §5.1 du preprint écrit déjà la version correcte (« when a node's doubt
exceeds u = 0,5, its coupling flips from attractive to repulsive ») ; l'abstract, la
contribution (2) et l'étiquette de `tab:ablations` (« no doubt dynamics ») sont **plus fortes
que ce que la mesure établit**.

**RÉSULTAT 2 (P18, les deux protocoles, balayage de `u` figé).** Ma prédiction — « la
frontière est au changement de signe, u = 0,5 » — est **REJETÉE**, et c'est le plus instructif :
- **régime forcé** : la chute se produit entre `u=0,30` (sync 0,42) et `u=0,45` (sync 0,087),
  donc **pendant que le couplage est encore attractif** (filtre +0,166). Ce n'est **pas** le
  signe qui désynchronise, c'est **l'absence d'attraction forte** ;
- **régime endogène** : **aucune frontière** — les 8 conditions, de filtre +0,898 à −0,762,
  donnent toutes une synchronie ≈ 0 et le quadrant structuré, **10/10 puis 10/10**. *Sans
  stimulus, le couplage ne joue aucun rôle anti-synchronisation.* (Le CSV périmé du 26/04 le
  disait déjà : endogène FROZEN_U 0,006 contre FULL 0,199.) ;
- **deux effets à deux seuils** : la synchronie décroche vers filtre ≈ +0,17, mais la
  complexité **LZ** ne décroche qu'au-delà de |filtre| ≈ 0,55 (1,55 → 1,11). Désynchroniser
  et structurer temporellement ne sont **pas** le même seuil du même paramètre.
- ⚠️ **Critère E3 « accepté » mais qui n'établit pas ce qu'il visait** : le répulsif faible
  (u=0,55) passe 10/10 — et son **miroir attractif** (u=0,45, même intensité) aussi. Critère
  mal posé (suffisance sans exclusivité) ; c'est le contrôle miroir qui l'a rattrapé.
- ⚠️ **Tension à nommer avec P16 (28/07)** : « le signe tranche, la force non » y était mesuré
  sur la **mobilité d'états** de la niche B1d ; ici, sur la **synchronie de Pearson** en BA
  m=3, c'est **l'inverse**. Régimes et observables différents — pas une contradiction, mais
  ne jamais citer l'une des deux phrases sans son régime.

**RÉSULTAT 3 — UN TROU DE COUVERTURE, réparé à moitié.** Le gate de fidélité bit-à-bit a
échoué et **n'a pas été repêché**. Cause établie avant la mesure : le CSV de `tab:ablations`
(`figures/scratch/ablation_coordination.csv`) date du **26/04/2026**, soit **avant le fix de
bruit Euler-Maruyama du 01/05** (AUDIT-024, `818cf67`) ; `figures/scratch/` est **gitignoré**
donc il n'est pas versionné ; et **aucun des 14 claims du Guardian ne couvre `tab:ablations`**.
Régénéré avec le code actuel : les **synchronies tiennent** (FULL +0,0023 vs 0,031 ± 0,034 ;
FROZEN +0,697 vs 0,751 ± 0,060) mais les **LZ sont hors des écarts publiés** (1,096 vs
1,069 ± 0,016 ; 1,603 vs 1,635 ± 0,006). Les mesures faites avec le code actuel concordent
entre elles (b4 sur BA m=3 : 0,0088 / 0,6875) ; c'est le CSV du 26/04 qui est l'intrus.
→ **OUVERT** : régénérer `tab:ablations` + lui donner un claim Guardian (précédent : C04 et
C08 le 12/06, « Option A » validée par Julien). Décision de Julien, non prise à ce jour.

**Réparé au passage** : `experiments/scratch/p2_sigma_social_ablation.py` — **non versionné**
alors qu'il produit les CSV canoniques de **deux** claims (C04, C13) — a été sorti de
`scratch/` vers `experiments/`. `b4_ablation_robustness.py` ne s'exécutait plus depuis
(ImportError) ; il rejoue désormais ses deux CSV **bit à bit** (Cohen d 9,38 BA / 4,72
lattice, séparation complète, 30 graines × 2 topologies).

**RÉSULTAT 4 (P19, `p19_threshold_vs_stimulus.py`)** — les deux questions ci-dessus, traitées
par un balayage croisé **7 intensités de stimulus × 11 valeurs de couplage figé** (990 runs).

- **F2 ACCEPTÉE, 6 régimes sur 6.** Les mi-transitions des deux observables ne coïncident
  **jamais** : synchronie entre filtre **+0,18 et +0,51**, LZ entre **−0,25 et −0,80**, écart
  0,48 à 1,26. Cas le plus net, à `I_stim = 0,35` : le couplage passe de +0,90 à −0,15, la
  synchronie s'effondre de **0,837 à 0,068** et la LZ **ne bouge pas** (1,639 → 1,655) — elle
  ne chute qu'à filtre −0,43 (→ 1,20).
  → **couper l'attraction désynchronise ; rendre franchement répulsif structure les
  trajectoires. Deux effets, deux seuils, séparés par une large zone où l'un agit sans
  l'autre.** `tab:ablations` les rapporte côte à côte comme une seule ablation parce qu'à
  `u = 0,05` on franchit les deux d'un coup.
- **F1 REJETÉE, et le critère était MAL POSÉ** (à consigner comme tel) : le seuil de synchronie
  bouge énormément (étendue **1,18**) mais **non monotonement** en `I_stim` (+0,37 ; −0,04 ;
  +0,05 ; +0,27 ; +0,31 ; −0,81). Ni « propriété du forçage » (L1), ni « propriété du réseau »
  (L2). **Mon critère présupposait une transition unique et monotone** — la cause du rejet est
  le fait ci-dessous. Deuxième critère mal posé de la journée après E3 ; dans les deux cas ce
  sont les contrôles écrits à côté qui ont rattrapé.
- **F3 ACCEPTÉE** : à `I_stim = 0`, l'étendue de la synchronie sur les 11 valeurs de couplage
  vaut **0,0047**. Sans forçage commun, le couplage ne produit aucune transition. Contrôle
  d'instrument passé sur les 990 runs (`u` immobile au bit près).

**⚠️ FAIT NON CHERCHÉ, RÉPLIQUÉ, NON EXPLIQUÉ — la courbe n'est pas monotone.** À fort
stimulus, en allant vers le répulsif, la synchronie descend, **remonte**, puis redescend :

| `u` figé | 0,50 | 0,55 | **0,65** | **0,80** | 0,95 |
|---|---|---|---|---|---|
| filtre | +0,01 | −0,15 | **−0,43** | **−0,73** | −0,88 |
| synchronie (`I_stim` = 1,0) | 0,284 | 0,163 | **0,310** | **0,282** | 0,047 |
| réplication, graines 3041-3050 | 0,277 | 0,162 | **0,308** | **0,276** | 0,078 |

Présent aussi à `I_stim = 0,75`. Répliqué au centième. **Hypothèse à tester, pas un
résultat** : état à **clusters en anti-phase déséquilibrés** — une répulsion modérée
scinderait le réseau, et des groupes de tailles inégales remonteraient la corrélation moyenne
de Pearson. Ce serait la signature « chimère » dont parle le preprint.

**RÉSULTAT 5 (P20, `p20_resync_band_cluster_structure.py`)** — la bande ouverte par P19,
instruite. Partition par le **signe du premier vecteur propre** de la matrice de corrélation
(méthode non supervisée, écrite avant), 8 cellules × 20 graines, réplication sur 3051-3060.

| cellule (`I_stim` = 1,0) | filtre | synchronie | cohésion **intra**-camp | opposition **inter**-camps | petit camp |
|---|---|---|---|---|---|
| u=0,05 attractif fort | +0,90 | 0,855 | +0,855 | *(un seul camp)* | 0 |
| u=0,55 — le creux | −0,15 | 0,162 | +0,611 | −0,393 | 34 |
| u=0,65 — la bande | −0,43 | **0,310** | +0,696 | −0,343 | 24 |
| u=0,80 — la bande | −0,73 | 0,286 | +0,675 | −0,337 | 26 |
| u=0,95 | −0,88 | 0,048 | **+0,213** | −0,139 | 38 |

- **C1 ACCEPTÉE (10/10 × 4, canoniques et réplication)** : la bande est bien un état à **deux
  camps en anti-phase**.
- **C2 REJETÉE (0/10)** : mon explication — « le déséquilibre des tailles fait remonter la
  moyenne » — est **fausse**. Test contrefactuel (mêmes corrélations, camps égalisés à 50/50) :
  la synchronie resterait à **+0,17**, pas à 0. La cause est que la **cohésion interne (+0,70)
  vaut deux fois l'opposition externe (−0,34)**.
- **C3 REJETÉE** : le creux a **la même structure** que la bande. La présence de camps ne
  distingue rien.
- **Ce qui reste, et qui est plus intéressant que l'hypothèse morte** : le réseau **se scinde
  en deux camps dès que l'attraction faiblit**, et le reste dans tout le régime répulsif ; un
  seul camp uniquement en attraction forte. Ce qui varie le long de l'axe n'est pas *s'il y a*
  des camps mais **leurs paramètres**, et ils ne varient pas dans le même sens : de u=0,55 à
  0,65 les camps deviennent plus cohésifs, moins opposés et plus déséquilibrés (les trois
  poussent la moyenne vers le haut → la bosse) ; à u=0,95 ils se **dissolvent** (cohésion
  +0,70 → +0,21). *Lecture post-hoc, marquée comme telle* : le déséquilibre pèse pour environ
  la moitié de la remontée (+0,171 contre +0,104 au contrefactuel égalisé), l'asymétrie
  cohésion/opposition pour l'autre moitié — aucun critère préalable ne le valide.

**RÉSULTAT 6 (P20b, `p20b_full_camps_or_independent.py`) — un soupçon sur l'observable
centrale, posé puis LEVÉ par la mesure.** P20 avait laissé un trou : la synchronie moyenne de
Pearson **ne distingue pas** « nœuds indépendants » de « deux camps qui se compensent », et
FULL n'avait pas été mesuré. Or le preprint écrit *« r̄ ≈ 0 : independent »*. Deux hypothèses
**mutuellement exclusives** écrites avant la mesure :

| cellule | synchronie | cohésion intra | opposition inter | petit camp |
|---|---|---|---|---|
| **FULL** (`I_stim` = 0,5, régime de Table 1) | +0,002 | **+0,115** | −0,110 | 47 |
| u=0,05 (l'ablation) | +0,698 | +0,726 | −0,321 | **1,4** |
| u=0,95 figé | +0,000 | +0,113 | −0,114 | 45 |

**D2 ACCEPTÉE, 10/10 puis 10/10** (et D1 rejetée 0/10) : dans FULL les corrélations sont
quasi nulles **des deux côtés** de la partition — ce n'est pas une structure de camps, c'est du
bruit qu'un algorithme coupe en deux parce qu'on le lui demande. **Les nœuds sont réellement
décorrélés ; la lecture du preprint est justifiée.** Contrôle qui boucle : l'ablation à u=0,05
donne un **camp unique** (le second ne contient que 1,4 nœud).
→ troisième soupçon de la journée porté sur la colonne A, vérifié, et **levé**. Même motif que
le garde-fou du 28/07 : *un signal d'alarme est une question, pas une conclusion.*

*Observation non mesurée, à consigner* : `u` monte de 0,05 à 0,997, donc **le doute traverse la
bande de ré-synchronisation** pendant sa montée (filtre −0,43 à −0,73). Il ne s'y arrête pas —
mais personne n'a jamais regardé ce que fait le réseau pendant cette traversée.

**RÉSULTAT 7 (P21, `p21_doubt_traversal_of_the_band.py`) — le doute ne traverse pas la bande,
il la franchit trop vite pour l'habiter.** `u` monte de 0,05 à 0,997 : il passe donc
*à travers* la bande. C'était le seul endroit où la **dynamique** de `u` pouvait faire quelque
chose qu'un `u` figé ne fait pas (un bras figé n'a pas de trajectoire). 3 vitesses
(`tau_u` ∈ {2, 10, 50}), 9000 pas, `I_stim` = 1,0 (la bande n'existe qu'à fort stimulus),
30 blocs de 300 pas, réplication sur 3071-3080.

| pas | `u` | filtre | synchronie |
|---|---|---|---|
| 0–300 | 0,486 | **+0,054** | +0,211 |
| 300–600 | 0,913 | **−0,851** | +0,159 |
| 600–900 | 0,975 | −0,894 | +0,096 |
| stationnaire | 0,999 | −0,907 | ≈ +0,04 |

Entre les deux premiers blocs le filtre passe de +0,05 à −0,85 : **toute la bande est franchie
à l'intérieur d'un seul bloc de 300 pas**, et cela reste vrai à la vitesse la plus lente
testée. **La bande est un régime stationnaire qui n'existe que si l'on fige `u`.**

- **T1 ACCEPTÉE (10/10 puis 10/10) mais T2 REJETÉE (0/10)** : les remontées transitoires de
  synchronie existent — elles ne sont **pas** la bande, ce sont des fluctuations du régime
  stationnaire (filtre −0,907). T1 seule aurait pu être présentée comme « la traversée laisse
  une signature » ; c'est T2, écrite à côté, qui l'interdit. **Troisième fois de la journée**
  (après E3 et F1) que c'est le contrôle adjacent qui rattrape.
- **T3 REJETÉE (6/10 puis 5/10)** : traverser vite ou lentement ne change pas l'état final.
  Conforme à la **présomption négative écrite avant la mesure**.
- **T4 (contrôle d'instrument) PASSÉE, 10/10, et elle tranche une inversion du preprint** :
  `tau_u` = 2 → `u` franchit 0,5 en **35 pas** ; `tau_u` = 10 → 137 ; `tau_u` = 50 → **896**.
  Donc **`tau_u` grand = LENT**, conforme à `dynamics.py:338` (`du = ε_eff·(…−u)/tau_u`). Or la
  Discussion du preprint (« Doubt time-scale and criticality ») écrit *« for τ_u < 10, doubt
  dynamics are too slow »* et *« for τ_u > 50, fast doubt dynamics »* : **le vocabulaire y est
  inversé**. Aucun chiffre n'en dépend (le balayage reste ce qu'il est), seule son
  interprétation est retournée.

**CE QUE LA JOURNÉE ÉTABLIT, PAR CONVERGENCE.** Cinq expériences, cinq chemins indépendants,
la même réponse : **seul le NIVEAU atteint par `u` compte pour l'anti-synchronisation.** Le
saut direct l'égale (P17) ; la carte des seuils est une affaire d'intensité de couplage
(P18, P19) ; l'état stationnaire est identique (P20b) ; le chemin pour y arriver n'y change
rien (P21).
→ **Conséquence d'ingénierie, positive** : l'anti-synchronisation pourrait être obtenue par un
**couplage répulsif fixe**, bien plus simple à fabriquer qu'une variable d'état adaptative par
nœud.
→ ⚠️ **Réserve, à ne pas omettre** : un couplage fixe suppose de connaître le bon niveau à
l'avance, alors que `u` s'y établit **seul** ; et sur la niche B1d ce qui sert n'est pas
l'anti-synchronisation mais *savoir quand trancher*, où l'adaptativité garde son rôle. Ce qui
tombe ici est **borné à l'anti-synchronisation**.

**Reste ouvert** : (a) `tab:ablations` à régénérer + à couvrir par un claim Guardian —
décision de Julien, non prise à ce jour ; (b) les trois formulations « doubt **dynamics** »
(abstract, contribution 2, étiquette de Table 1) plus fortes que ce que l'ablation établit,
alors que la Discussion §5.1 dit déjà juste — proposition de réécriture non rédigée, décision
de Julien ; (c) l'inversion de vocabulaire sur `tau_u` dans la Discussion (T4) — correction de
texte, décision de Julien ; (d) pourquoi la cohésion intra-camp l'emporte sur l'opposition
inter-camps dans la bande ; (e) ⚠️ **point de méthode** : les corrélations intra/inter dépendent
de la **longueur de fenêtre** (P20b sur 750 pas donne +0,218 là où P21 sur 300 pas donne
+0,35 dans le même régime). Ne jamais comparer deux mesures de structure sans vérifier la
fenêtre.

---

## Dépendances rapides
- A3 alimente A5 (mesure H_cont per-seed) et B4 (IC sur labels mesurés).
- B1 débloque B3 (énergie/tâche) et B5 (comparaison SOTA).
- B2 débloque B3 et B6 (physique du dispositif). **09/07/2026** : B2 a livré 3 dossiers
  de correspondance (photonique/spintronique/électrique), débloquant un premier B3
  (cadré) et un premier B6 (proposition falsifiable) — mais la simulation physique
  réelle (LLG, SPICE) reste à faire avant que B2 soit clos.
- C1 dépend de A5 (métrique continue).
