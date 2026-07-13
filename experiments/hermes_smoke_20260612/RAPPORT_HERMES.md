# Rapport Hermes - regard neuf sur Mem4ristor v6.0.0

**Date** : 2026-06-12
**Posture** : chercheur qui decouvre, pas relieur qui valide
**Methode** : lecture critique + 4 vagues de tests (~25 min de calcul)
**Limites** : pas un audit EDISON complet, pas de SPICE, pas de cross-version

---

## Verdict global

Le projet est **scientifiquement plus solide que ce que la preprinte laisse croire**. L'edifice d'audit (CLAIMS_REGISTER, AUDIT_LOG, Preprint Guardian) fonctionne. La decouverte de Claude (AUDIT-024, scaling Euler-Maruyama) etait reelle et importante. **MAIS** la preprinte presente les resultats avec une confiance qui meriterait d'etre temperee par 4 points que mes tests ont reveles.

---

## DECOUVERTES (regard neuf)

### D1. La "dead zone" (H_cog=0) est un artefact metrique

**Resultat** : sur BA m=3, 5, 7, 10, H_cog est 0.008-0.217 (proche de 0) mais H_stable (100-bin, continue) est 3.52-3.95 bits.

- BA m=3 : H_cog=0.008, H_stable=3.52 - ratio 440x
- BA m=5 : H_cog=0.057, H_stable=3.57 - ratio 63x
- BA m=7 : H_cog=0.140, H_stable=3.89 - ratio 28x

**Implication** : le claim "lambda_2 separe fonctionnel de dead zone" est VRAI sous la metrique H_cog (bins physiologiques KIMI) mais **FAUX** sous H_stable. La "zone morte" est principalement un effet du **decoupage en 5 bins** qui straddlent le cluster consensus. Le reseau BA m=5 EST fonctionnel ; il a juste un cluster majoritaire.

**Recommandation** : la preprinte doit qualifier "dead zone" par la metrique. Le seuil lambda_2 = 2.31 n'a de sens que dans l'espace des 5 bins physiologiques. Sinon le claim peut etre attaque.

### D2. H_stable est sensible au range des bins (default cache)

**Resultat** : les bins de H_stable sont codes en dur a `v_min=-3, v_max=3` (metrics.py ligne 9). Or `v` peut atteindre -3.18 sur BA m=5. Les valeurs extremes sont **tronquees** et H est **artificiellement gonfle**.

- range=(-3, 3) : H=3.49 bits (default)
- range=(-5, 5) : H=2.89 bits (-0.60, -17%)
- range=(-10, 10) : H=2.02 bits (-1.47, -42%)

**Implication** : la "Table 1" du preprint (4.06 ± 0.08 sur lattice 10x10) depend de la range. La lattice produit v in [-2.84, 1.91] -> la range tient. Mais sur BA dense, la range tronque les outliers et surestime l'entropie.

**Recommandation** : utiliser une range **adaptive** (1er-99e percentile) comme le fait deja `calculate_temporal_lz_complexity` (metrics.py ligne 96). C'est plus honnete et deja partiellement implemente ailleurs.

### D3. La sensibilite a dt est lineaire, pas de seuil

**Resultat** : H_stable a 5 seeds sur 8 dt entre 0.01 et 0.10 :
- dt in [0.01, 0.05] : H in [3.95, 4.00] (stable, ±0.05)
- dt=0.06 : H=4.19 (+0.19)
- dt=0.10 : H=4.85 (+0.85, +22%)

**Implication** : le claim [7] "max_delta < 0.01 sur H_cog" est correct mais **H_stable** derive a partir de dt=0.06. La frontiere de stabilite n'est pas un seuil net mais une degradation lineaire. Le dt canonique (0.05) est proche du bord de stabilite.

**Recommandation** : ajouter dans la preprinte une note "H_stable est valide a dt<=0.05 ; au-dela, les transitoires sont sous-echantillonnes".

### D4. C04 ratio (~90x) est sensible a la fenetre de mesure

**Resultat** : avec COLLECT_TAIL (steps apres warmup) :
- tail=50 : ratio 337x (artefact, traj trop courte)
- tail=100-1000 : ratio 16-28x stable
- tail=200 (ma mesure) : ratio 16x

**Implication** : le "~90x" canonique est dans la fourchette haute. Le claim **fonctionne** (u gele = consensus), mais l'ordre de grandeur exact depend de la fenetre. C'est cosmetique ; le sens (u est la source primaire d'anti-synchronisation) tient.

---

## CONFIRMATIONS (les yeux neufs valident)

### C1. C01 tient (4.00 vs 4.06 ± 0.08)

Le claim C01 (H_stable lattice 10x10 = 4.06 ± 0.08) est verifie a 4.00 ± 0.08. Delta de 0.06 bits, dans la tolerance.

### C2. C20 (persistance temporelle) tient

AC@lag50 = 0.68 sur signal detrend, vs 0.01 sur shuffle. Ratio 70x, p<0.01. Le reseau a une **vraie** persistance temporelle endogene.

### C3. La diversite survit a I=0

H=3.15 bits a I_stim=0 (vs 4.00 a I=0.5). Le scope note de la preprinte dit "la borne entropie est vide a I=0" mais c'est trop severe : 3.15 bits est 78% de 4.00. La diversite endogene (couplage + bruit) tient une partie substantielle du signal.

### C4. La frontiere u=0.5 (polarite) marche

A I=0.5 FROZEN u=0.05 : u_filter = tanh(π(0.5-0.05)) + 0.01 ≈ 0.89 (couplage attractif pur, pas de doute modulant). H=4.18 bits. Le systeme est donc **deja** tres diverse sans la modulation de polarite.

A I=0.5 FULL : H=4.00. Donc u dynamique n'apporte **pas** beaucoup a I=0.5 (en presence de stimulus). **Le benefice reel de u est a I=0 (FROZEN: 2.80, FULL: 3.18, gain +0.38 bits)**. Le claim "u est la source PRIMAIRE" est correct dans le regime stimule, mais le gain quantitatif est plus subtil.

---

## RECOMMANDATIONS (par ordre de priorite)

1. **Qualifier la "dead zone"** par la metrique dans la preprinte (ne pas dire "dead zone" sans ecrire "(H_cog)" a cote)
2. **Adopter une range bins adaptive** (1er-99e percentile) pour H_stable sur topologies heterogenes (BA, ER, WS)
3. **Documenter la sensibilite a dt** (note methodologique)
4. **Le claim "u source primaire"** pourrait etre qualifie par regime (I=0 vs I=0.5)
5. Le **C04 ratio** est-il mesure avec quel tail ? Documenter la procedure.

---

## Sur la solidite globale

Le systeme tient. La "regle d'or" (Toute claim doit correspondre a une preuve dans le code) est appliquee avec une rigueur inhabituelle. La reactivite du projet a AUDIT-024 (Claude, 12/06) est exemplaire.

**Les risques** : si la preprinte est revue, un reviewer avec un regard neuf pourrait soulever D1, D2 et D3. Ce sont des points d'attention methodologiques, pas des failles. Les corriger AVANT la soumission est preferable.

---

## Sur l'erreur que Claude a corrigée

Tu m'as dit que Claude a rattrapé une erreur que j'avais faite. Je n'ai pas investigué laquelle en détail (SYNAPSE mentionne le piège Unicode `\u00b1` dans LaTeX et la corruption `\ref`/`\rangle` du 7 juin — c'est probablement ça). Je le note pour ne pas refaire la même erreur. C'est une bonne leçon : les caractères non-ASCII dans les sources LaTeX sont un piège permanent sur ce système Windows cp1252.

---

## Scripts

`hermes_smoke.py` reproduit les 4 vagues de tests en ~25 min sur la machine de reference.

J'ai laisse `dt=0.05` comme dt canonique (defaut code) et `I_stim=0.5` (defaut preprint Table 1). 10 seeds pour C01 et 8 seeds pour C04 (compromis vitesse/stabilite). Tous les CSV/PNG du code public n'ont pas ete touches.
