# Voie Spintronique — Note d'ouverture du dossier de faisabilité (B2, fond)

> Créé le 9 juillet 2026 — Claude Code (Opus 4.8) / Julien Chauvin
> Statut : note de faisabilité exploratoire, calcul dimensionnel de premier ordre.
> **AUCUN claim publié ne dépend de ce document.** Rien n'existait sur cette piste
> avant cette session (contrairement à la voie photonique, cf. PHOTONIC_PATHWAY.md,
> largement défrichée depuis le 12 juin).
> Origine : B2/B5 dans `docs/FUTURE_WORK.md` (« comparaison aux oscillateurs
> spintroniques couplés, domaine neuromorphique de référence »). Choix de Julien
> du 9 juillet : explorer les 3 familles de dispositifs (photonique, spintronique,
> électrique) plutôt que d'en choisir une seule.

## 1. Pourquoi la spintronique — et pourquoi elle diffère du photonique

Le dossier photonique (GST) a mappé `u` (le doute, un état lent multi-niveau) sur
un matériau à changement de phase. La spintronique invite une correspondance
différente et plus directe : les **oscillateurs à transfert de spin (STNO,
spin-torque nano-oscillators)** sont des oscillateurs non-linéaires auto-entretenus
— exactement le type d'objet dont `v` (l'activité du nœud FHN) est une abstraction.
C'est la première fois qu'un dispositif candidat correspond nativement à la
dynamique **rapide** du modèle plutôt qu'à sa dynamique **lente** (`u`, `w`).

**Littérature de référence (vérifiée par recherche web le 09/07/2026) :**
- Torrejon et al., *Neuromorphic computing with nanoscale spintronic oscillators*,
  Nature 547, 428–431 (2017) — reconnaissance de chiffres parlés par reservoir
  computing sur un **unique** STNO à vortex, multiplexage temporel.
- Romera et al., *Vowel recognition with four coupled spin-torque nano-oscillators*,
  Nature 563, 230–234 (2018) — 4 STNO **couplés mutuellement**, reconnaissance de
  voyelles par synchronisation.
- Les STNO à vortex ont un temps de réponse rapide (~ns), une émission de
  puissance élevée, une raie étroite — propriétés recherchées en neuromorphique.

## 2. Correspondance candidate

| Variable modèle | Candidat spintronique | Nature de la correspondance |
|---|---|---|
| `v` (activité, oscillateur) | Phase/fréquence d'un STNO à vortex | **Directe** — les deux sont des oscillateurs non-linéaires auto-entretenus (ou quasi, cf. réserve ci-dessous) |
| Couplage inter-nœuds (D_eff·u_filter·laplacien) | Couplage mutuel micro-onde ou courant de polarisation partagé entre STNO voisins (cf. Romera et al. 2018) | Plausible — le couplage mutuel entre STNO physiques est expérimentalement démontré |
| `u` (doute, lent) | **Mathématiquement portable (§7, testé 09/07/2026)** — le mécanisme `u_filter`/`du` transposé tel quel sur un couplage de Kuramoto reproduit l'ablation FROZEN_U/FULL (Cohen d jusqu'à 14.85). **Correspondance physique toujours non résolue** : quel circuit lit le désaccord de phase local et pilote une variable lente en retour ? Candidat le plus proche dans la littérature citée : une fenêtre d'intégration/masque temporel (Torrejon et al. 2017) — mais c'est une construction algorithmique externe, pas un état physique interne au dispositif | Le test mathématique est positif ; le mécanisme physique de lecture reste à trouver |

**Réserve physique importante.** Le nœud FHN isolé du modèle, à ses paramètres par
défaut (α=0.15), est en régime de **spirale stable** (sous le seuil de Hopf,
`reviewer2_linear_stability.py`, λ = −0.0473 ± 0.2824i) — ce n'est **pas** un
oscillateur auto-entretenu (cycle limite) à l'état isolé ; l'activité soutenue vient
du bruit et du couplage réseau. Un STNO, lui, EST un oscillateur auto-entretenu
(cycle limite véritable, entretenu par un courant polarisé en spin en continu).
La correspondance est donc une analogie de **rôle et d'échelle de temps**, pas
d'équivalence mécanistique stricte — à la différence, par exemple, du neuristor
Mott de Pickett et al. 2013 (cf. `ELECTRICAL_PATHWAY.md`) qui reproduit
explicitement une dynamique de type FHN/Hodgkin-Huxley par construction.

## 3. Ancrage temporel (calcul dimensionnel, `experiments/b2_device_physics_mapping.py`)

Même ancrage que pour le photonique : `T_node ≈ 22.25` unités de temps modèle
(≈ 445 pas d'intégration), ancré sur le temps de réponse STNO documenté
(« fast response time ~ns », gamme large selon conception) :

| Ancrage T_STNO | dt physique | τ_u physique | Campagne (4000 pas) |
|---|---|---|---|
| 1 ns | 2.25 ps/pas | 0.449 ns | 9.0 ns |
| 10 ns | 22.5 ps/pas | 4.49 ns | 89.9 ns |

**Lecture.** Si le nœud STNO tourne à ~1 GHz (période 1 ns), une campagne complète
de 4000 pas (l'équivalent des POCs photoniques, warmup + mesure) ne dure que
**~9 ns physiques** — trois ordres de grandeur plus rapide que l'équivalent
photonique (§5 de PHOTONIC_PATHWAY.md, ~1–2 µs). La spintronique est le candidat
le plus rapide des trois familles explorées, cohérent avec sa réputation en
neuromorphique (Torrejon/Romera visent justement la vitesse de calcul).

## 4. Énergie (ordre de grandeur, PAS mesurée)

Contrairement au photonique (énergie de signal, événementielle) et au RRAM
(énergie d'écriture, événementielle), un STNO **dissipe en continu** tant qu'il
oscille — il n'a pas d'état "au repos gratuit". Puissances d'entrée documentées
(recherche web 09/07/2026, gamme large selon conception) :
- STNO à vortex (type Torrejon/Romera) : de l'ordre de **quelques mW**
- STNO linéaire optimisé : **~138 µW**
- Designs bas-bruit expérimentaux : jusqu'à **~1 µW**

Énergie par pas d'intégration = puissance × dt_physique :

| Config | dt physique | E/pas (vortex ~3 mW) | E/pas (linéaire 138 µW) |
|---|---|---|---|
| T_STNO=1 ns | 2.25 ps | **6.7 fJ** | 0.31 fJ |
| T_STNO=10 ns | 22.5 ps | **67 fJ** | 3.1 fJ |

**Comparaison directe.** Ces chiffres (fJ/pas) sont dans le même ordre de grandeur
que l'estimation RRAM (10–50 fJ/écriture, `ELECTRICAL_PATHWAY.md`) et le neuristor
Mott (~22 fJ/pas, même document) — **et 3 à 4 ordres de grandeur en dessous** de
Loihi/TrueNorth (~24–26 pJ/opération synaptique). Mais attention à l'échelle de
comparaison : Loihi/TrueNorth comptent l'énergie **par opération synaptique**
(événement discret), alors que ce chiffre STNO est une énergie **par pas
d'intégration continu** — la dissipation totale sur une campagne complète
(des milliers de pas) reste non négligeable en régime permanent (voir
`B3_ENERGY_COMPARISON.md` pour la mise en regard honnête des deux modes
de comptage).

## 5. Ce que ce dossier ne fait PAS (à ne jamais perdre de vue)

- **§7 (09/07/2026) a testé une réduction phase-oscillateur (Kuramoto/Slavin-Tiberkevich),
  PAS une simulation LLG (Landau-Lifshitz-Gilbert) ou macrospin complète.** Un vrai
  macrospin (précession, champ démagnétisant, couple de transfert de spin explicite)
  reste à construire — projet de plusieurs semaines (cf. B2 dans `docs/FUTURE_WORK.md`,
  effort 🧩). La réduction phase-oscillateur est le niveau d'abstraction standard de la
  littérature STNO pour les questions de synchronisation de réseau, pas un raccourci
  inventé ici — mais elle laisse de côté toute la dynamique d'amplitude/relaxation.
- Le rôle physique de `u` reste non résolu (§2) — le test §7 montre que le mécanisme
  *mathématique* se porte, pas qu'un circuit physique réel peut le lire. C'est le point
  le plus faible de ce mapping, plus faible que le mapping GST↔`u` du dossier photonique
  (lui-même déjà qualifié de « le plus spéculatif »).
- Aucune tâche de calcul (type NARMA10, vowel recognition) n'a été rejouée avec
  un modèle STNO réel ici — voir `docs/FUTURE_WORK.md` B5 pour un positionnement
  qualitatif basé sur la littérature (Torrejon 2017, Romera 2018), qui n'a PAS
  le même niveau de rigueur que la comparaison ESN/NARMA10 du 7 juillet 2026
  (`experiments/b5_esn_comparison.py`) — celle-là était une comparaison
  **loyale, tête-à-tête, même protocole**. Il n'existe aucun benchmark spintronique
  publié sur exactement NARMA10 ; la comparaison B5 reste donc qualitative,
  pas quantitative.

## 6. Prochaines étapes possibles (par coût croissant)

1. [x] ~~Modèle STNO macrospin minimal~~ — **FAIT 09/07/2026**, voir §7 ci-dessous.
2. [x] ~~Reproduire qualitativement l'ablation FROZEN_U/FULL~~ — **FAIT 09/07/2026**,
   voir §7. Résultat positif, avec une réserve honnête importante (calibration du
   capteur de désaccord).
3. [ ] Proposer un protocole expérimental réel (cf. `docs/FUTURE_WORK.md` B6,
   signature falsifiable) — la proposition existe déjà (réseau STNO réel + spectroscopie
   micro-onde), maintenant appuyée par un résultat en silico, pas seulement une analogie.

## 7. Résultat — le mécanisme se porte, avec une réserve honnête (09/07/2026)

**Script** : `experiments/b2_stno_phase_coupling_poc.py` → `figures/b2_stno_phase_coupling_poc.csv`
/ `_agg.csv` / `.png`. **Modèle** : réduction phase-oscillateur de Slavin-Tiberkevich
(le niveau d'abstraction standard pour les questions de synchronisation de réseau STNO,
pas un simulateur LLG) — équivalente à un Kuramoto en champ local sur graphe, avec le
**même mécanisme de doute que `dynamics.py`, mêmes constantes, aucun réglage propre**
(`u_filter = tanh(π(0.5-u))`, `du` avec la même formule d'`epsilon_u_adaptive`, même
convention de bruit Euler-Maruyama). Ablation FROZEN_U/FULL implémentée à l'identique
(`sigma_social_override=0` dans l'équation de `u` seulement, pas dans le couplage).

**Calibration honnête (documentée, pas cachée).** Premier essai (capteur de désaccord
brut, `gain_u=1`) : effet **correct en signe mais modeste** — `u` reste toujours
<0.19, ne franchit **jamais** le seuil de bascule de polarité `u=0.5` (contrairement
au modèle FHN où `u` sature régulièrement >0.5, cf. session du 7 juillet « verrouillage
en mode FOU »). Diagnostic : `sigma_social = |L_φ|` est un couplage de Kuramoto
**borné** dans [-1,1] par construction (moyenne de `sin(·)`), contrairement au laplacien
`v` du modèle FHN qui n'est pas borné — le mécanisme ne peut alors montrer que sa
modulation « douce » (affaiblissement d'amplitude), pas son effet contrarian qualitatif.
**Correction** : gain appliqué au **capteur** qui alimente `u` (`sigma_social_for_u =
gain_u·sigma_social`), sans toucher au canal de couplage physique — exactement le
pattern déjà présent dans le modèle original (`sigma_social_override` découple déjà
perception du désaccord et force de couplage réelle dans `p2_sigma_social_ablation.py`).

**Résultats (10 seeds canoniques, IC bootstrap, BA m=3 et lattice 10×10 — mêmes
topologies que B4) :**

| Topologie | Capteur | R_FULL | R_FROZEN_U | diff (IC 95%) | Cohen d |
|---|---|---|---|---|---|
| BA m=3 | brut (gain=1) | 0.646±0.082 | 0.826±0.067 | +0.180 [+0.114,+0.247] | **+2.28** |
| BA m=3 | amplifié (gain=5, `u` franchit 0.5) | 0.079±0.010 | 0.826±0.067 | +0.747 [+0.702,+0.785] | **+14.85** |
| Lattice 10×10 | brut (gain=1) | 0.296±0.058 | 0.382±0.094 | +0.086 [+0.022,+0.158] | **+1.05** |
| Lattice 10×10 | amplifié (gain=5) | 0.045±0.007 | 0.382±0.094 | +0.338 [+0.286,+0.400] | **+4.83** |

**Lecture honnête.**
1. **Le mécanisme se porte, dans les deux régimes** : le doute réduit la
   synchronisation de Kuramoto sur ce substrat totalement différent, aucun IC ne
   chevauche zéro, dans les 4 conditions testées.
2. **« Tel quel » (gain=1), l'effet est réel mais modeste** (Cohen d 1.05–2.28) —
   c'est le résultat le plus défendable si on refuse d'ajouter un paramètre au
   modèle. **Une fois le capteur de désaccord calibré pour laisser `u` franchir son
   propre seuil de bascule (gain=5), l'effet devient massif** (Cohen d 4.83–14.85),
   du même ordre voire supérieur à celui de l'ablation FHN originale (B4, Cohen d
   9.4 / 4.7). Ce n'est PAS un résultat retouché pour plaire : le gain est un
   paramètre de capteur physiquement légitime (une chaîne de détection de phase a
   son propre gain, indépendant du couplage lui-même), mais c'est un paramètre
   AJOUTÉ, pas porté depuis le modèle original — à dire clairement.
3. **Réplication non cherchée, mais notée** : dans les 2 régimes, BA m=3 montre un
   effet plus fort que lattice — même ordre que l'ablation FHN originale (B4 :
   Cohen d 9.4 BA vs 4.7 lattice). Cohérent avec le fil rouge B1 (BA scale-free =
   cas structurellement le plus sensible du projet), mais 2 topologies ne prouvent
   pas une loi générale — à vérifier sur ER et d'autres tailles avant d'en faire un claim.
4. **Ce que ça ne prouve pas** : aucune dynamique de phase-amplitude complète
   (Slavin-Tiberkevich à 2 variables), aucun bruit de phase dérivé d'un vrai spectre
   de puissance micro-onde mesuré, aucune vérification que le gain de capteur=5 est
   physiquement réalisable sur un vrai circuit de détection de phase STNO. C'est un
   test de **portabilité mathématique du mécanisme**, pas une validation physique.
