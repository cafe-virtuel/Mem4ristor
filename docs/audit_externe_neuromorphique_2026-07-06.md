# Audit externe simulé — posture « chercheuse de référence en neuromorphique » (2026-07-06)

> **Nature du document.** Revue critique conduite par Claude Code (L'Ingénieur) à la
> demande de Julien Chauvin, en adoptant la posture d'une chercheuse exigeante du
> calcul neuromorphique découvrant le projet pour la première fois. Chaque grief
> ci-dessous a été **vérifié dans le code, les CSV, ou par ré-exécution** durant la
> session — rien n'est repris de mémoire sans contrôle. Ce fichier est une note
> d'audit interne, pas un document public de soumission.

---

## Périmètre vérifié
`preprint.tex` (787 l.), `dynamics.py`, `topology.py`, `core.py`,
`p2_edge_betweenness_analysis.py`, `lambda2_crit_regression.py`,
`verify_table1_preprint.py`, les CSV `p2_edge_betweenness.csv`,
`crucial_kfixe_lambda2_variable.csv`, `consolidation_invariance_N.csv`, et un
contrôle d'initialisation exécuté indépendamment.

---

## 1. [BLOQUANT] Le résultat-titre (λ₂_crit = 2.31, causal) est contredit par les fichiers du dépôt

L'abstract, le titre, la contribution (3), la §4.6, la §4.7 et la Discussion affirment
que la connectivité algébrique λ₂ **détermine causalement** la capacité cognitive, avec
un seuil à 2.31.

**Preuve du contraire, dans le dépôt même :**
- `consolidation_invariance_N.csv` : anneau k=8 mort (`dead=1`) à N=100/200/400 pendant
  que λ₂ chute de 0.118 → 0.0074 (~300× **sous** le seuil 2.31). Anneau k=4 vivant partout
  au même λ₂ (~0.01). À λ₂ égal, régimes opposés selon le degré. Une grandeur → 0 ne peut
  causer un effet constant.
- `crucial_kfixe_lambda2_variable.csv` : à degré fixe, balayer λ₂ (facteur ~27 via
  Watts-Strogatz) ne change pas le régime.

**Conclusion :** le mécanisme est de **champ moyen, gouverné par le degré** (degré
harmonique k_harm = 1/⟨1/deg⟩, frontière empirique ≈ 6), pas spectral. λ₂ « marchait »
uniquement parce qu'il covarie avec le degré dans les familles BA/ER du jeu de données.
Publier une causalité démentie par un CSV du même dépôt = rejet assuré.

## 2. [MAJEUR] La « régression sur 36 configurations » ne simule rien

`p2_edge_betweenness_analysis.py` calcule des métriques de graphe et lit un dictionnaire
`REGIME` **codé en dur** (lignes 95–108), assigné **par type de topologie** — 12 décisions
humaines. Le manuscrit (§4.6, l.339) les décrit comme « labelled empirically per seed »
et en tire « complete linear separation sur 36 configurations ». Faux : 12 décisions
dupliquées ×3. Preuve : les 3 lignes « Lattice » de `p2_edge_betweenness.csv` sont
**bit-à-bit identiques** (λ₂ = 0.0978869674096891). La séparation est quasi-tautologique
(on décrète BA m≥5 = dead, et BA m≥5 a un grand λ₂).

## 3. [MAJEUR] Le « 2.31 » repose sur deux points, dont un aberrant

Le gap (2.13, 2.50) est fixé par BA m=4 seed 2 (λ₂=2.126, fonctionnel) et BA m=5 seed 2
(λ₂=2.503, mort). Les deux autres seeds de BA m=4 sont à λ₂≈0.94 (variance intra-topologie
×2.3). Le point-frontière haut est un seed atypique. Un IC honnête engloutirait le 2.31 ;
le bootstrap de `lambda2_crit_regression.py` le masque en rééchantillonnant des labels
déjà décidés.

## 4. [MAJEUR] Le Tableau 1 ne suit pas le Cold Start revendiqué

Le texte (l.109, 193, 205) : « All simulations use homogeneous initial conditions
(v=w=0)… Diversity must emerge, not from favorable initialization. »
`verify_table1_preprint.py` instancie `Mem4Network(...)` **sans `cold_start=True`** →
défaut `cold_start=False` (topology.py:12) → v ∼ U(−1.5, 1.5) (dynamics.py:100).

**Contrôle exécuté (N=100, I=0.5, 3 seeds) :**
```
cold_start=False (DÉFAUT = Tableau 1 réel) : v initial std=0.839  → H_stable = 4.03 ± 0.07
cold_start=True  (revendiqué par le texte) : v initial std=0.000  → H_stable = 4.27 ± 0.17
```
La valeur annoncée 4.06 ± 0.08 correspond au **non**-cold-start ; le vrai cold start donne
4.27 (hors barre d'erreur). La diversité survit (crédit au modèle), mais le protocole
exécuté ≠ protocole décrit, sur un point que le papier met en avant.

## 5. [DE FOND] Métrique et provenance

- **H_cog (5 bins) est un artefact** (reconnu en Limitations : valeurs « not to be cited »)
  et pourtant il sous-tend toute la cartographie de la dead zone et le 2.31. Le résultat
  central doit reposer sur H_cont ou la synchronie.
- **Deux générations de code cohabitent** (changement de bruit Euler-Maruyama ~×4.5, 1ᵉʳ mai) :
  certains chiffres proviennent d'un code que le dépôt public ne reproduit plus. Rapiécé,
  mais signal d'alarme pour un reviewer soucieux de reproductibilité.

---

## Ce qui est réellement solide (et sous-vendu)

1. **Le mécanisme existe dans le code** : `w_i(u_i) = tanh(π(0.5 − u_i)) + δ` implémenté tel
   quel (dynamics.py:199). Polarité état-dépendante réelle. Nouveauté authentique vs
   Provata 2026 (signe vs magnitude).
2. **Le vrai résultat, c'est l'ablation FROZEN_U** : geler u fait exploser la synchronie
   (×24 sur BA m=3, ×90 sur lattice), mesurée sur la **corrélation de Pearson**
   (indépendante du binning). Robuste, difficile à attaquer — et enterré au milieu du papier.
3. **L'infrastructure d'honnêteté** (résultats négatifs documentés, auto-réfutation) est
   meilleure que celle de beaucoup de labos. C'est ce qui rend le point 1 réparable.

---

## Pistes (par priorité)
1. Reformuler titre + abstract : « spectral boundary at algebraic connectivity » →
   frontière de **degré de couplage** via champ moyen. Le phénomène survit, la cause change.
2. Remonter FROZEN_U comme résultat principal.
3. Refaire la « régression » avec de vraies simulations : mesurer H_cont par (topologie, seed),
   étiqueter par mesure, régresser régime sur k_harm **et** λ₂ côte à côte
   (`bouclage_regime_vs_predicteurs.py` : 2/70 erreurs pour k_harm vs 15/70 pour λ₂).
4. Corriger l'initialisation (passer cold_start=True et régénérer, ou décrire l'init réelle).
5. Bannir H_cog des résultats primaires (réserver au pont SPICE).
6. Dériver k_harm,crit analytiquement — mais sur métrique continue, sinon on refonde un
   artefact.

## Verdict
**Non soumissible en l'état** : résultat-titre contredit par ses propres fichiers, preuve
statistique centrale sans simulation, tableau principal hors protocole. Mais la science
sous-jacente est plus solide que sa mise en scène : mécanisme réel et nouveau, ablation
FROZEN_U probante, réfutation déjà produite en interne (`experiments/lambda2_foundation_20260701/`).
Le chemin vers un papier honnête est court — il reste le courage éditorial de renommer la
cause dans le document public.

---
*Claude Code (L'Ingénieur) — audit simulé, posture externe. Toutes les assertions vérifiées
en session sur le code/CSV/exécution.*
