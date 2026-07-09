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
3. [x] ~~Généraliser à un modèle amplitude+phase avec non-isochronicité~~ — **FAIT
   09/07/2026**, voir §8. Julien : « tu m'as mis l'eau à la bouche, je veux voir ce
   que ça donne ». Résultat : le mécanisme survit, robuste à la non-isochronicité.
4. [ ] Proposer un protocole expérimental réel (cf. `docs/FUTURE_WORK.md` B6,
   signature falsifiable) — la proposition existe déjà (réseau STNO réel + spectroscopie
   micro-onde), maintenant appuyée par 2 résultats en silico convergents (§7 et §8).
5. [ ] Vraie simulation LLG/macrospin complète (précession, champ démagnétisant, couple
   de transfert de spin explicite) ou modèle de Thiele pour la dynamique du cœur de
   vortex — le palier suivant, encore non franchi (§8 reste une réduction de type
   auto-oscillateur, pas une résolution spatiale de la texture magnétique).

## 8. Généralisation amplitude+phase (Slavin-Tiberkevich) — le mécanisme survit (09/07/2026)

**Pourquoi ce palier.** Le §7 utilisait un Kuramoto pur (phase seule, amplitude figée à 1)
— le cas limite **isochrone** d'un modèle plus complet et plus fidèle à la littérature
STNO : l'**oscillateur auto-entretenu non-linéaire de Slavin & Tiberkevich** (IEEE Trans.
Magn. 2009), qui dérive formellement de l'équation LLGS complète et qui EST le modèle que
le domaine utilise pour les questions de synchronisation de réseau. Sa signature physique
centrale, absente du §7 : le **décalage de fréquence non-linéaire** (non-isochronicité)
`ω(p) = ω0 + N·p` où `p=|a|²` est la puissance d'oscillation — précisément ce que
Slavin-Tiberkevich identifient comme LA différence qualitative entre un STNO et un
oscillateur conventionnel. Tester sans ce terme, c'est tester un cas particulier, pas le
régime STNO réel.

**Script** : `experiments/b2_stno_amplitude_phase_poc.py` → `figures/b2_stno_amplitude_phase_poc.csv`
/ `_agg.csv` / `.png`. Amplitude complexe `a_i` par nœud, `da_i/dt = [croissance/saturation
+ i·ω(p_i)]·a_i + K·u_filter_i·S_i + bruit`, `S_i` = couplage complexe (généralise `sin(Δφ)`
du §7, porte à la fois la partie réactive et dissipative). Mécanisme du doute **identique**
à `dynamics.py`, aucun réglage propre.

**Calibration numérique documentée (comme le stiffness proof Euler du 1er mai)** : à
dt=0.01, gain de capteur=10 et non-isochronicité≥10 font **diverger** l'intégration Euler
explicite (overflow) — confirmé non-physique par test à dt décroissant (dt≤0.005 reste fini
et converge). **Correction : dt=0.005 pour toute la campagne.**

**Résultats (10 seeds, IC bootstrap, BA m=3 et lattice 10×10, N_nonlin ∈ {0, 3, 10}) :**

| Topologie | N_nonlin | Capteur | R_FULL | R_FROZEN_U | Cohen d |
|---|---|---|---|---|---|
| BA m=3 | 0 | brut (gain=1) | 0.613±0.092 | 0.620±0.090 | **+0.08** (nul) |
| BA m=3 | 0 | calibré (gain=10) | 0.258±0.055 | 0.620±0.090 | **+4.59** |
| BA m=3 | 3 | calibré (gain=10) | 0.230±0.032 | 0.481±0.052 | **+5.49** |
| BA m=3 | 10 | calibré (gain=10) | 0.135±0.015 | 0.270±0.039 | **+4.41** |
| Lattice | 0 | calibré (gain=10) | 0.166±0.039 | 0.342±0.084 | **+2.55** |
| Lattice | 3 | calibré (gain=10) | 0.149±0.034 | 0.272±0.086 | **+1.79** |
| Lattice | 10 | calibré (gain=10) | 0.111±0.017 | 0.176±0.029 | **+2.60** |

**Lecture honnête.**
1. **Au capteur brut (gain=1), l'effet est cette fois NUL** (Cohen d 0.01–0.09, IC
   chevauchant toujours zéro) — plus net que le §7 (qui montrait un effet modeste mais
   réel à gain=1). `u` reste collé à ~0.06 (quasi identique au FROZEN_U figé à 0.05) :
   dans ce modèle plus riche, le couplage complexe near-synchronisé laisse encore moins
   de désaccord perceptible au capteur brut. Résultat assumé tel quel, pas arrondi.
2. **Une fois le capteur calibré (gain=10, `u` franchit 0.5), le mécanisme est robuste
   à la non-isochronicité** : Cohen d reste dans [4.41, 5.49] sur BA m=3 et [1.79, 2.60]
   sur lattice à travers TOUTE la plage testée (N_nonlin 0→10) — **aucun effondrement**
   à mesure que le paramètre le plus caractéristique des STNO augmente. C'est le test de
   robustesse que l'honnêteté scientifique imposait avant de faire confiance au résultat
   du §7 (qui n'avait testé que N_nonlin=0).
3. **Vérification physique indépendante (bon signe)** : `R_FROZEN_U` diminue lui-même
   avec `N_nonlin` (0.62→0.48→0.27 sur BA ; 0.34→0.27→0.18 sur lattice), cohérent avec la
   littérature (la non-isochronicité élargit la raie spectrale / réduit la cohérence d'une
   population d'oscillateurs) — le modèle se comporte comme la physique le prédit
   indépendamment du mécanisme du doute, ce qui renforce la confiance dans le reste.
4. **Ce que ça ne prouve toujours pas** : aucune résolution spatiale de la texture de
   vortex (modèle de Thiele ou LLGS complet), valeur de non-isochronicité pour un vrai
   STNO à vortex non trouvée par recherche web (testée sur une plage, pas une valeur
   affirmée), et le gain de capteur=10 reste un paramètre ajouté à interpréter
   physiquement (§7), pas mesuré sur un vrai circuit.

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
