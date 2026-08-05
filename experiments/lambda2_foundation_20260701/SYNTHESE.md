# Mandat non-livrable — Fonder λ₂_crit ≈ 2.31 (ou prouver l'impossibilité)

**Session** : 1er juillet 2026 · Claude Code (Opus 4.8), L'Ingénieur · avec Julien
**Statut** : ABOUTI — branche « prouver qu'on ne peut pas » + mécanisme réel trouvé
**Nature** : recherche (compréhension), pas livrable. Aucun fichier public/preprint touché.

---

## Question de départ

Le preprint (`docs/papers/preprint/preprint.tex`, §`sec:lambda2`) affirme :
« λ₂_crit ∈ (2.13, 2.50), midpoint 2.31, séparation complète sur 36 configs ;
topology alone determines cognitive capacity via algebraic connectivity ».
Mandat : **comprendre pourquoi ~2.3**, ou prouver que c'est impossible.

## Réponse en une phrase

**On ne peut pas fonder λ₂_crit sur λ₂ : λ₂ n'est pas le mécanisme.** La mort
cognitive (« dead zone ») est un effet de **champ moyen par échantillonnage**,
gouverné par le **degré harmonique** `k_harm = 1/⟨1/deg⟩` (frontière ≈ 6),
indépendamment de la structure spectrale.

---

## Fil de la démonstration (scripts dans ce dossier)

### 1. `diag_fixedpoint_vs_orbit.py` — point fixe, pas orbite
- En déterministe (σ_v=0), TOUTE topologie converge vers le point fixe synchrone
  `v_i = −1.294 ∀i` (= point fixe du nœud isolé `v*≈−1.286`). temporal_std≈0.
  → **Floquet définitivement écarté** (on linéarise autour d'un point fixe).
- SURPRISE : sans bruit, aucune diversité pour aucune topologie. **La diversité
  du régime fonctionnel est entièrement entretenue par le bruit.**
- La dead zone se fait par **translation** de la distribution de v sous le bord
  de bin −1.2 (H_cog=0 = artefact de binning, déjà connu D1/preprint:332), pas
  par effondrement de largeur.

### 2. `remesure_regime_par_seed.py` — la séparation complète s'effondre
- Découverte annexe : `p2_edge_betweenness_analysis.py` NE SIMULE PAS ; les
  labels de régime y sont **recopiés à la main** (dict `REGIME`) par TYPE de
  topologie (12 décisions, pas 36), figés avant AUDIT-024. Le « n=36 » est gonflé.
- Re-mesure dynamique PAR SEED (code actuel, endogène, best-of-two-norms H_cog) :
  **chevauchement λ₂ ∈ [1.26, 3.20]**, pas de séparation nette. Inversion :
  ER p=0.08 (λ₂=1.67) meurt, BA m=4 (λ₂=2.22) vit. → λ₂ n'est pas universel.

### 3. `crucial_kfixe_lambda2_variable.py` — le crucial experiment
- Watts-Strogatz découple ⟨k⟩ et λ₂ : à degré FIXE, le recâblage p balaie λ₂ en
  gardant ⟨k⟩ exactement constant.
- À CHAQUE k fixe, `dead_frac` est PLAT quand λ₂ varie. Cas juge : anneau k=10
  (λ₂=0.22, ≪ 2.31) mort à 100 %, comme random-regular k=10 (λ₂=4.5). → **λ₂
  non-causal.**

### 4. `consolidation_invariance_N.py` — le clou (invariance en taille)
- Pour un anneau, λ₂ ~ 1/N² s'effondre avec N ; ⟨k⟩ reste constant.
- k=8 anneau : λ₂ = 0.118 → 0.030 → 0.0074 (N=100→200→400), `dead_frac`=1.0
  partout. À N=400, λ₂ est **~300× sous le seuil 2.31** et le réseau est mort.
- k=4 contrôle : fonctionnel partout (λ₂ variant ×500). → **Une grandeur qui
  tend vers 0 ne peut pas causer un effet constant. Clos.**

### 5. `pourquoi_champ_moyen_cinematique.py` — le mécanisme
- Le couplage tire chaque nœud vers `cible_i = ⟨v des voisins⟩` = moyenne de
  deg_i échantillons. Donc `Var_i(cible) ≈ V·⟨1/deg⟩` (loi des grands nombres).
- Test cinématique (sans dynamique) : `Var(cibles)` vs `⟨1/deg⟩` → **r=0.9998**,
  pente = V. vs λ₂ → r=−0.50 (fantôme de covariance). Anneau vs rr même k :
  λ₂ diffère ×100, `Var(cibles)` identique. → mécanisme fixé par le degré.
- Réconcilie l'hétérogénéité : `⟨1/deg⟩` est dominé par les nœuds de bas degré
  (Jensen : `⟨1/deg⟩ ≥ 1/⟨k⟩`). BA garde des cibles bruitées via sa nuée
  périphérique malgré ses hubs → survit à ⟨k⟩ plus élevé qu'un régulier.

### 6. `bouclage_regime_vs_predicteurs.py` — le régime suit le degré harmonique
- Qualité de séparation dead/live (erreurs sur 70 instances, N=200) :
  - λ₂ : **15/70** (le pire)
  - ⟨k⟩ : 3/70 · degré médian : 6/70
  - **⟨1/deg⟩ (degré harmonique) : 2/70** (le meilleur, comme prédit)
- Frontière : **k_harm ≈ 6** unifie BA / ER / random-regular / anneau sur une
  seule courbe, là où λ₂ les éparpille.

---

## Mécanisme complet (le « pourquoi »)

1. Couplage = attraction vers la moyenne des voisins (cible locale du champ).
2. Cette cible est une moyenne de `k` échantillons → sa dispersion inter-nœuds
   suit `⟨1/deg⟩` (∝ 1/k pour un régulier).
3. Degré élevé → toutes les cibles ≈ moyenne globale → attraction cohérente vers
   UN point → consensus (mort cognitive). Degré faible → cibles bruitées et
   distinctes → diversité maintenue (par le bruit thermique).
4. Seul le NOMBRE de voisins compte (échantillonnage), pas leur arrangement
   (λ₂). D'où l'invariance à λ₂ et l'unification par `k_harm`.

## Reformulation proposée du preprint (décision Julien — OK donnée en session)

Remplacer « spectral regime boundary at λ₂_crit=2.31 » par :
« La diversité cognitive s'effondre au-dessus d'un degré de couplage critique
(degré harmonique k_harm ≈ 6), via un mécanisme de champ moyen par
échantillonnage ; la connectivité algébrique λ₂ n'a pas de rôle causal (elle
covarie avec le degré dans les familles BA/ER du jeu de données initial). »
→ Sections concernées : abstract, titre (« spectral »), §sec:lambda2,
§sec:topo_regimes. Le phénomène (dead zone) survit ; seule la cause est renommée.

## Réserves d'honnêteté (avant de graver)

- Métrique de régime = H_cog (5 bins), elle-même un artefact de binning ; la
  valeur exacte de la frontière (k_harm≈6) dépend un peu de la métrique et des
  paramètres. **Le mécanisme (champ-moyen, λ₂ non-causal) est solide ; la valeur
  numérique de la frontière l'est moins.**
- N ≤ 400, 5-8 seeds. Le point central est blindé (invariance N + crucial exp) ;
  les contours (frontière exacte, rôle fin de l'hétérogénéité) méritent plus.
- Pistes ouvertes : dériver analytiquement k_harm,crit depuis σ_v, v* et la
  géométrie des bins ; refaire avec H_cont (métrique continue) pour voir si la
  frontière se déplace.
