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
4. [x] ~~Vraie simulation macrospin LLGS (vecteur d'aimantation 3D, Slonczewski)~~ —
   **FAIT 09/07/2026**, voir §9. Vérif matériel faite (RTX 3070 8Go, largement
   suffisant, pas de GPU nécessaire à ce palier) ; micromagnétisme spatial complet
   (mumax3) explicitement reporté à une décision séparée (Julien a choisi ce palier-ci).
5. [ ] Proposer un protocole expérimental réel (cf. `docs/FUTURE_WORK.md` B6,
   signature falsifiable) — la proposition existe déjà (réseau STNO réel + spectroscopie
   micro-onde), maintenant appuyée par 3 résultats en silico convergents (§7, §8, §9).
6. [ ] Micromagnétisme spatial complet (mumax3, texture de vortex résolue) ou modèle
   de Thiele — le palier suivant, non franchi. Nécessite d'installer mumax3/CUDA,
   projet à part (heures de calcul, pas une suite de session).

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

## 9. Macrospin LLGS complet — le vrai vecteur d'aimantation (09/07/2026)

**Pourquoi ce palier.** Julien, après avoir vu le §8 : « tu m'as mis l'eau à la bouche, je
veux voir ce que ça donne ». Vérification matérielle faite avant de s'engager (pas de
supposition) : Ryzen 7 5800H, 32 Go RAM, **RTX 3070 Laptop 8 Go VRAM** — largement
suffisant pour ce palier (pur Python/numpy, pas de GPU nécessaire ici). Le micromagnétisme
spatial complet (texture de vortex résolue, à la mumax3) aurait justifié le GPU mais
demande d'installer un nouvel outil et des campagnes de plusieurs heures — **écarté
explicitement par Julien** au profit de ce palier-ci, plus proportionné à une session.

**Script** : `experiments/b2_stno_macrospin_llgs_poc.py` → `figures/b2_stno_macrospin_llgs_poc.csv`
/ `_agg.csv` / `.png`. Contrairement aux §7-8 (réductions phénoménologiques), `m_i` est ici
un **vrai vecteur unité 3D** intégré par l'équation de **Landau-Lifshitz-Gilbert-Slonczewski**
explicite (précession + amortissement de Gilbert + couple de spin-transfert de Slonczewski,
terme "field-like" plus petit omis — simplification assumée). Mécanisme du doute identique.

**Vérifications physiques préalables (avant le réseau, sanity checks)** :
- Isolé, un tilt initial converge vers un **cône de précession stable** (pas un point fixe,
  pas un renversement) dès que β dépasse un seuil ; angle continûment ajustable par β
  (0.6° à β=0.005, ~20° à β=0.025, 133°/renversement à β=0.04) — comportement STT
  qualitativement correct, régime « oscillateur » bien distinct du régime « commutation ».
- La fréquence de précession dépend de `H_k·m_z` (1.00 à H_k=0, 1.26 à H_k=0.3, 1.56 à
  H_k=0.6) : **la non-isochronicité émerge naturellement de l'anisotropie**, sans terme
  ajouté à la main — bonne nouvelle de cohérence avec le paramètre `N` phénoménologique du §8.

**Découverte de calibration (documentée, pas cachée)** : un test minimal à 2 macrospins
couplés montre que **cette géométrie de couplage (champ effectif vectoriel, couple
gyroscopique) verrouille en ANTIPHASE** (`Δφ→π`), pas en phase comme les §7-8 — un phénomène
réel et documenté pour les oscillateurs gyrotropes couplés (le canal de couplage — dipolaire
vs électrique — détermine le signe effectif du verrouillage dans la vraie littérature STNO).
Conséquence : le paramètre d'ordre de Kuramoto standard `R` reste au plancher même à
verrouillage parfait — il faut mesurer le **2e harmonique** `R2 = |mean(exp(2iφ))|`,
standard pour détecter un état à 2 clusters (antiphase), vérifié sans ambiguïté sur 2
oscillateurs (`Δφ=π → R2=1, R=0`).

**Découverte topologique (non cherchée, notée)** : sur **lattice** (graphe **bipartite**,
compatible avec un damier antiphase globalement cohérent), `R2` atteint 0.83 en FROZEN_U —
un vrai ordre existe. Sur **BA m=3** (graphe **non bipartite**, cycles impairs), le
couplage antiphase est **FRUSTRÉ** — `R2` reste bas (~0.15–0.18) dans TOUTES les conditions,
doute ou pas. **3e mécanisme indépendant** (après B1 et le mapping u↔GST/spintronique du §7-8)
où BA scale-free se comporte différemment de lattice — cette fois pas comme « cas le plus
sensible », mais comme « cas où aucun ordre global n'émerge du tout » sous cette géométrie
de couplage précise.

**Résultats (10 seeds, IC bootstrap) :**

| Topologie | Capteur | R2_FULL | R2_FROZEN_U | Cohen d |
|---|---|---|---|---|
| Lattice 10×10 | brut (gain=1) | 0.316±0.090 | 0.832±0.272 | **+2.42** |
| Lattice 10×10 | calibré (gain=3) | 0.175±0.035 | 0.832±0.272 | **+3.22** |
| BA m=3 (frustré) | brut (gain=1) | 0.147±0.020 | 0.185±0.024 | **+1.61** |
| BA m=3 (frustré) | calibré (gain=3) | 0.118±0.011 | 0.185±0.024 | **+3.36** |

**Lecture honnête.**
1. **Sur lattice (où un ordre réel existe), le mécanisme est net dès le capteur brut**
   (Cohen d=2.42, IC[+0.314,+0.652] — recalculé le 31/07, cf. la note du §7 sur `hash()`)
   et se renforce une fois calibré (Cohen d=3.22) — la
   première fois sur les 3 modèles testés que l'effet brut (non calibré) est déjà fort,
   pas seulement « correct en signe mais modeste ».
2. **Sur BA (frustré), un effet statistiquement réel mais de faible amplitude absolue
   persiste** (diff +0.037 à +0.066, IC ne chevauchant jamais zéro, Cohen d 1.61–3.36 —
   grand en unités d'écart-type parce que la variance résiduelle est petite, pas parce que
   l'effet est spectaculaire en absolu). Lecture correcte : le doute réduit encore un peu
   un système déjà proche du plancher de désordre, il ne « sauve » pas la frustration ni
   ne la change qualitativement.
3. **Ce résultat est le plus direct des trois** (§7, §8, §9) : aucune reformulation
   phénoménologique, la vraie équation vectorielle. Le prix : la géométrie de couplage
   simplifiée choisie ici (champ effectif « diffusif ») privilégie l'antiphase, alors que
   les vrais réseaux STNO couplés électriquement (Romera et al. 2018) sont rapportés en
   littérature comme favorisant plutôt le verrouillage en phase — **canal de couplage
   différent, pas un artefact**, mais à garder en tête avant de généraliser.
4. **Ce que ça ne prouve toujours pas** : aucune résolution spatiale de la texture de
   vortex (le prochain palier serait le micromagnétisme complet, mumax3, ou le modèle de
   Thiele) ; le canal de couplage électrique réel (courant de polarisation partagé) n'a
   pas été modélisé explicitement, seulement un champ de couplage générique.

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
| BA m=3 | brut (gain=1) | 0.646±0.082 | 0.826±0.067 | +0.180 [+0.114,+0.245] | **+2.28** |
| BA m=3 | amplifié (gain=5, `u` franchit 0.5) | 0.079±0.010 | 0.826±0.067 | +0.747 [+0.701,+0.784] | **+14.85** |
| Lattice 10×10 | brut (gain=1) | 0.296±0.058 | 0.382±0.094 | +0.086 [+0.021,+0.157] | **+1.05** |
| Lattice 10×10 | amplifié (gain=5) | 0.045±0.007 | 0.382±0.094 | +0.338 [+0.284,+0.400] | **+4.83** |

> 🔧 **IC recalculés le 31/07/2026, et pourquoi ils ont changé.** Le script tirait sa graine de
> bootstrap par `hash((topologie, gain))` — or `hash()` d'une **chaîne** est randomisé à chaque
> démarrage de l'interpréteur Python (`PYTHONHASHSEED`, protection anti-collision depuis 3.3).
> Ce `seed=` avait donc toutes les apparences d'une graine fixée **sans en être une** : chaque
> exécution rendait un intervalle différent. Mesuré : 60 tirages sur les **mêmes** données
> donnaient **six** valeurs distinctes à la 3ᵉ décimale, sur chaque borne. Les valeurs
> antérieures (`[+0.114,+0.247]`, `[+0.702,+0.785]`, `[+0.022,+0.158]`, `[+0.286,+0.400]`)
> venaient d'un tirage du 09/07 **définitivement irrécupérable**.
> Corrigé (`zlib.crc32`, déterministe) et `n_boot` porté de 5 000 à 50 000, ce qui ramène
> l'erreur de Monte-Carlo de ~0.0013 à ~0.0004 par borne — les bornes à 3 décimales ne
> dépendent plus du tirage. **Le défaut ne touchait QUE les IC** : `R_FULL`, `R_FROZEN`, les
> écarts-types et les Cohen *d* se rejouent **au bit près**, vérifié le 31/07. Même correction
> appliquée aux §8 et §9, qui portaient le même défaut. *Motif déjà rencontré le matin même
> avec `tab:benchmarks` : un chiffre exact, produit autrement qu'il n'y paraît.*

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

---

## §10 — De quel gain de capteur un laboratoire a-t-il besoin ? (2026-07-31)

`experiments/b6_sensor_gain_threshold.py` · `figures/b6_sensor_gain_threshold*.csv` (**versionnés**)

Les §7-§9 laissaient une réserve identique et **floue** : *« une fois le capteur calibré »*,
sans jamais dire **combien** il faut. Un expérimentateur ne peut rien faire d'une telle
phrase. Ce volet la remplace par un nombre. **Gate de fidélité G1 passé au chiffre près**
sur quatre références du CSV du 09/07 (`u_mean` 0.0608 / 0.5034 / 0.0607 / 0.5252).

**1. La non-isochronicité était accusée à tort, et la preuve dormait dans le CSV du 09/07.**
À `n_nonlin = 0` — modèle **isochrone**, elle est absente — l'effet est **déjà nul** au
capteur brut (diff +0.0076 BA / +0.0036 lattice, les deux IC chevauchant zéro). Re-mesuré
ici : `u_mean` au gain 1 vaut **0.0608** (n_nonlin=0) contre **0.0599** (n_nonlin=10),
**étendue 0.0010**. La non-isochronicité n'agit pas sur le capteur.

**2. La cause est une échelle de capteur, lue dans le code et non devinée.**
`u ≈ gain·|S| + 0.05` à l'équilibre, et la bascule de polarité exige `u > 0.5` : il faut donc
`|S| > 0.45/gain`. Ce modèle produit `|S| ≈ 0.011` — **facteur 41 manquant**. Ici `S` est une
moyenne **vectorielle complexe** de `(a_j − a_i)`, dont les contributions s'annulent entre
elles ; le §7 moyennait des sinus d'ordre 1. **Problème d'unité de mesure, pas de physique
d'oscillateur** — ce qui change entièrement la portée de la réserve.

**3. La spécification.**

| Ce qu'on veut obtenir | Gain nécessaire |
|---|---|
| Effet franc (Cohen d ≈ 1,3 à 1,9) | **5 à 7** |
| Bascule complète de polarité (≥6/10 graines) | **7** (isochrone) à **10** (non-isochronicité maximale) |

Le seuil de bascule est **identique sur les deux topologies** — réseau irrégulier ou grille
régulière, même réponse. La non-isochronicité **déplace** ce seuil (7 → 10) sans jamais
toucher au capteur : elle agit sur la dynamique, pas sur la mesure.

**4. Le fait non prédit, et le plus utile pour une campagne réelle.**
L'effet est franc **avant toute bascule** : BA, `n_nonlin=0`, gain 5 → **Cohen d = +1.35 avec
0/10 graines** au-dessus de `u = 0.5` ; à `n_nonlin=10`, gain 7 → d = +1.92 avec **1/10**.
**Le mécanisme ne requiert donc pas l'inversion de polarité du couplage** : la modulation
*douce* de son amplitude suffit. Pour un expérimentateur, la cible n'est pas « faire basculer
`u` » mais « obtenir un effet détectable » — et cela demande **moins** de gain que ce que les
§7-§9 laissaient croire.

**5. Un critère à moi a échoué, et il est conservé affiché.**
J'avais prédit une transition **abrupte** par emballement — la boucle `gain ↑ → |S| ↑ → u ↑`
est réelle (`|S|` passe de 0.011 à 0.045), donc l'emballement était plausible. La montée est
en fait **régulière**. C'est l'échec de cette prédiction qui a fait apparaître le point 4 : un
seuil net aurait masqué la croissance continue de l'effet.

**6. Ce que ça ne prouve toujours pas.**
Ce modèle n'a **aucun bruit sur le capteur**. Un amplificateur de gain 7 amplifie aussi le
bruit, et rien ici ne dit que l'effet y survit — **c'est la question suivante, et elle est
mesurable**. Un gain n'est pas gratuit en surface ni en consommation : ce volet chiffre une
**exigence**, pas un coût. Et le canal de couplage électrique réel (Romera) n'est toujours pas
modélisé.

> ~~⚠️ **Dette signalée le 31/07, non traitée** : les CSV des §7, §8 et §9 vivent dans
> `figures/scratch/`, **gitignoré**.~~ ✅ **SOLDÉE le soir même.** Les trois scripts écrivaient
> déjà dans `figures/` — personne ne les avait relancés depuis le rangement du 14/07.
> Régénérés et **versionnés**. La régénération a doublé de test de reproductibilité :
> **toutes** les colonnes hors IC sont identiques (20 lignes, 3 fichiers, **0 différence**
> vérifiée mécaniquement). Ce n'était pas une dette de *valeur*, seulement de versionnement —
> à l'exception des IC, cf. la note du §7.

---

## §11 — Le bruit du capteur aide, et c'est une mauvaise nouvelle (2026-07-31)

`experiments/b6_sensor_noise.py` · `figures/b6_sensor_noise*.csv` (**versionnés**)

Question posée par Julien : *« je pensais que le bruit était même bénéfique pour M4R ? »*
Vérifié d'abord : **oui pour le bruit de dynamique** — la variabilité de fabrication augmente
l'entropie de +0,05 à +0,75 bits (`PHOTONIC_PATHWAY` §4quater). Le bruit du **capteur** est
autre chose : du bruit sur une **information**, pas sur le mouvement. Jamais testé avant.

**1. Son intuition est vérifiée, et par le mécanisme prévu à l'avance.**
Le capteur mesure `|S|`, une **valeur absolue** : un bruit symétrique ajouté avant le module
**augmente** systématiquement la mesure (rectification, `E[|S+ε|] > |E[S]|`). Le désaccord
capté passe de 0,15 à 1,02.

| Bruit du capteur, à **gain = 1** (aucun amplificateur) | Cohen d |
|---|---|
| 0 (capteur propre) | +0,08 |
| 0,4 | +0,68 |
| **0,8** | **+3,22** |

**Le bruit remplace l'amplificateur** — et fait mieux qu'un ampli de gain 7 sans bruit (+2,75).

**2. Mais il apporte du NIVEAU, pas de l'INFORMATION.** Deux contrôles écrits *contre* cette
envie, et ils convergent :
- un `u` **figé au niveau atteint** fait aussi bien 4 fois sur 6, et **mieux** dans les deux
  exceptions (+7,98 contre +5,78 ; +3,73 contre +3,22) — le bruit n'apporte **jamais** un
  avantage sur le niveau ;
- à fort bruit, un capteur **aveugle**, qui ne mesure que du bruit et rien du réseau, fait
  **exactement** aussi bien : **+7,91 contre +7,93, écart 0,01**.

La fenêtre où l'information compte encore est **étroite** : σ ∈ [0,05 ; 0,20] à gain 5, où le
capteur informé bat nettement l'aveugle (+2,68 contre +0,98). Au-delà, le dispositif marche
**sans rien mesurer**.

**3. Conséquence pour le protocole expérimental, et elle coupe dans les deux sens.**

✅ **La réserve « le bruit du capteur pourrait tuer l'effet » est levée — c'est l'inverse.** Un
laboratoire n'a besoin ni d'une chaîne de détection propre, ni forcément d'un amplificateur.

🔴 **Mais le pouvoir discriminant de la prédiction est en jeu.** Si un capteur bruité produit
le même effet qu'un capteur informé, l'expérience telle que formulée **ne teste plus le
mécanisme du doute** : elle teste « un couplage répulsif en moyenne ». Le contrôle actuel —
couplage figé à sa valeur **initiale** — ne distingue pas les deux.

> ➡️ **Le protocole B6 doit comporter TROIS bras, pas deux :**
> 1. couplage modulé par le désaccord (le mécanisme) ;
> 2. couplage fixe à la valeur **initiale** (contrôle historique) ;
> 3. **couplage fixe réglé au NIVEAU MOYEN ATTEINT** par le mécanisme ← **le bras manquant**.
>
> C'est le contrôle `FROZEN_U(0,95)` que Julien avait fait ajouter le 28/07 sur la niche,
> transposé au dispositif physique. **Sans lui, un laboratoire mesurerait un effet réel et
> l'attribuerait à la mauvaise cause** — et nous lui aurions fourni le protocole qui permet
> cette erreur.
>
> ⚠️ **PÉRIMÉ LE SOIR MÊME, conservé pour la trace — lire le §12.** Le bras 3 a été construit
> quelques heures plus tard, et il a donné **l'inverse** de ce qui est écrit ci-dessus. (a) Il
> en faut **cinq**, pas trois. (b) Le bras 3 tel que formulé ici — « au niveau moyen atteint »,
> c'est-à-dire au niveau de `u` — **n'est pas réalisable** : `u` est une variable interne. (c)
> Surtout : ajouter ce bras **ne sauve pas le volet 1**, il l'enterre — un couplage fixe y
> reproduit l'effet à 0,24 de Cohen *d* près, et son réglage transfère d'une condition à
> l'autre. **C'est le volet 2 (retard de récupération) qui porte le pouvoir discriminant.**

**4. Ce que ça ne dit pas.** Un gain n'est gratuit ni en surface ni en consommation : ce volet
chiffre une **exigence**, pas un coût. Et le canal de couplage électrique réel (Romera) n'est
toujours pas modélisé.

> ➡️ **Le troisième bras a été construit le soir même. Il a donné l'inverse de ce qu'on
> attendait : ce n'est pas le volet 1 qu'il sauve, c'est le volet 2. Voir §12.**

---

## §12 — Le protocole complet : sur QUELLE observable la prédiction discrimine (2026-07-31, soir)

`experiments/b6_third_arm.py` · `b6_third_arm_transient.py` · `b6_fourth_arm_profile.py` ·
`b6_fifth_arm_per_node.py` → CSV **versionnés** dans `figures/`.

Le §11 laissait une consigne : ajouter un troisième bras, *« couplage fixe réglé au niveau
moyen atteint »*. En le construisant, deux choses sont apparues — l'une avant toute mesure,
l'autre contre toute attente.

### 1. Le bras demandé n'était pas exécutable, et le corriger a changé la question

`u` est une **variable interne** : aucun laboratoire ne peut la lire ni la régler. Ce qu'un
dispositif câble, c'est le **couplage**, `u_filter = tanh(π(0,5 − u)) + 0,01`. Comme `tanh`
est non linéaire, `⟨tanh(π(0,5 − u))⟩ ≠ tanh(π(0,5 − ⟨u⟩))` (Jensen), et `u` varie à la fois
dans le temps et par nœud. Le bras 3 se dédouble donc :

| | réglage | réalisable sur un dispositif ? |
|---|---|---|
| **3a** | `u` figé à ⟨u⟩ | ❌ non — `u` n'est pas accessible |
| **3b** | couplage figé à ⟨u_filter⟩ | ✅ **oui — c'est celui du protocole** |

L'écart de Jensen mesuré vaut **−0,017** en régime stationnaire (§7-§10) mais **+0,208** sous
tâche trompeuse, parce que `u` y monte à 0,65 et **traverse 0,5**, la zone où `tanh` bascule
de signe. La distinction est donc négligeable dans un cas et forte dans l'autre — raison de
plus pour spécifier **3b**, le seul des deux qui existe physiquement.

### 2. 🔴 Le volet 1 (synchronisation stationnaire) NE teste PAS le mécanisme du doute

Modèle §8, gain 5, `n_nonlin = 0`, Cohen *d* contre le bras 2. Gate de fidélité **4/4** contre
le CSV du 09/07.

| | d(B1) *doute* | d(3a) | d(3b) *câblable* |
|---|---|---|---|
| BA m=3 | +1,35 | +0,93 | **+1,11** |
| Lattice 10×10 | +0,67 | +0,46 | **+0,54** |

Un couplage fixe reproduit l'effet à **0,24** de Cohen *d* près. Et son réglage **transfère** :
calé sur BA il fonctionne sur lattice (**97 %**, **103 %**), calé à σ_ω = 0,15 il fonctionne à
0,30 et 0,075 (**167 %**, **145 %** — donc *mieux* que le réglage local). L'argument
d'auto-calibration ne tient pas ici.

Ce qui subsiste, **répliqué sur graines 3081-3090 jamais utilisées** : le doute garde un
avantage dans 3 des 4 conditions distinctes, d'ampleur moyenne **+0,18** de Cohen *d*. C'est
réel et c'est **cinq fois sous le seuil de 1,0** fixé avant mesure pour parler de
discrimination — donc trop petit pour qu'une manip le sépare de son bruit expérimental.
**Ne jamais citer ce +0,18 sans cette phrase.**

> ⚠️ **Réserve née de cette réplication, et elle porte au-delà de ce volet.** Le Cohen *d* de
> ce dispositif est **instable d'un jeu de graines à l'autre** : à σ_ω = 0,075 il passe de
> **+1,47 à +0,55**, sur lattice de **+0,67 à +0,35**. Un facteur 2 à 3 entre deux jeux de dix
> graines. Cela vaut aussi pour les valeurs des §7-§10, y compris le **+1,35** du §10 : elles
> sont exactes et elles sont peu stables. Toute campagne réelle doit prévoir bien plus de
> dix répétitions.

### 3. 🟢 Le volet 2 (retard de récupération) discrimine, contre TOUTE boucle ouverte

Harnais `b1d_stno_deceptive_poc.py` (12/07), lattice 10×10, 12 graines. Gate de fidélité
**8/8 au dixième de pas** contre le CSV du 12/07 — les quatre bras ajoutés partagent donc
exactement la dynamique qui a produit le résultat publié.

Référence : temps de basculement moyen **5274,7 pas** pour le bras 1 contre **3466,7** pour le
bras 2 — le « +52 % » du 12/07, reproduit ici au dixième de pas.

| bras | quel couplage | part du retard reproduite |
|---|---|---|
| **B1** | modulé par le désaccord (**boucle fermée**) | *référence, 100 %* |
| **B2** | fixe à la valeur initiale | *référence, 0 %* |
| **B3a** | fixe au niveau de `u` atteint *(non réalisable)* | **−25 %** |
| **B3b** | fixe au niveau du **couplage** atteint *(câblable)* | **−18 %** |
| **B3c** | fixe, **re-réglé pour chaque T_pulse** | **−18 %** |
| **B4a** | **programmé dans le temps**, profil exact du bras 1 | **−6 %** |
| **B4b** | même profil, calé sur un autre T_pulse | **−8 %** |
| **B5b** | programmé **par nœud**, profil exact du bras 1 | **−3 %** |
| **B5a** | programmé par nœud **et par copie** | **+100 %** *(gate d'instrument)* |

> ⚠️ **Toutes ces fractions sont calculées de la même façon** — moyenne des fractions par
> `T_pulse` sur {1500, 3000, 4500} — et c'est une correction : le premier jet de ce tableau
> mélangeait deux méthodes (ratio des moyennes globales pour les bras 3, moyenne des fractions
> pour les bras 4 et 5), ce qui donnait −21 % au lieu de −18 %. `T_pulse = 500` est **exclu
> partout** : le retard y vaut 2,8 pas et toute fraction y explose. *Deux chiffres présentés
> côte à côte doivent sortir du même calcul, même quand l'écart est petit.*

Le signe compte : les bras en boucle ouverte ne *manquent* pas le retard, ils vont **dans le
sens inverse** — un couplage fixe au niveau du doute *accélère* la récupération que le doute
*ralentit*.

**B5a est un gate, pas un résultat** : rejouer le couplage enregistré nœud par nœud *et* copie
par copie redonne le bras 1 **exactement** (`frac = 1,000`). Il établit que `u` n'agit que par
`u_filter` — aucun canal caché — et **valide rétroactivement l'échec des bras 3 et 4**, qui
sans lui aurait pu passer pour un défaut d'implémentation.

### 4. ⚠️ La « cicatrice de doute » du 12/07 est mal décrite — correction

Le 12/07 décrivait le mécanisme comme *« le conflit fait monter `u` durablement »*, donc comme
une **hystérésis moyenne**. C'est faux. Le profil moyen `⟨u_filter(t)⟩` va bien de **+0,90 à
−0,88**, et l'imposer ne produit **aucun** retard (−6 %). Le profil par nœud non plus (−3 %).

Par élimination, la seule différence entre B5a (reproduit tout) et B5b (ne reproduit rien) est
que le couplage de chaque copie **répond au signal que cette copie reçoit**. **Le mécanisme est
irréductiblement en boucle fermée** : le reproduire en boucle ouverte exigerait de connaître le
signal avant de l'avoir reçu.

### 5. Le protocole, tel qu'un laboratoire doit l'exécuter

**Observable : PAS la synchronisation stationnaire** (§2 ci-dessus : elle ne discrimine pas),
mais le **temps de récupération après un leurre transitoire** — leurre nombreux et fort retiré
après `T_pulse`, vérité moins nombreuse et plus faible maintenue.

**Ordre d'opérations — il n'est pas libre** : le bras 1 doit tourner **en premier**, car c'est
lui qui livre le niveau et le profil de couplage sur lesquels les bras 3, 4 et 5 se règlent.

| | bras | ce qu'il élimine s'il échoue à reproduire |
|---|---|---|
| 1 | couplage modulé par le désaccord mesuré | *le mécanisme lui-même* |
| 2 | couplage fixe à sa valeur initiale | *contrôle historique* |
| 3 | couplage fixe au **niveau moyen** du bras 1 | « c'est juste un couplage plus répulsif » |
| 4 | couplage **programmé dans le temps** (profil du bras 1) | « il suffit d'une rampe » |
| 5 | couplage programmé **par nœud** *(si le dispositif le permet)* | « il suffit d'une rampe par oscillateur » |

**Critères de falsification, dans les deux sens :**
- si le **bras 1 ne retarde pas** la récupération par rapport au bras 2 → le mécanisme est
  **réfuté** ;
- si **l'un des bras 3, 4 ou 5 reproduit le retard** → l'effet observé existe mais **n'est pas
  le mécanisme du doute**, c'est un effet de couplage. En simulation ils rendent −21 %, −6 % et
  −3 % : un bras qui rendrait une fraction franchement positive contredirait ce dossier.

**Gain de capteur requis** : 5 à 7 (§10). **Chaîne de détection propre : non nécessaire** — le
bruit du capteur aide par rectification (§11).

### 6. Ce que ce volet ne prouve pas

- 🔴 **La lecture est DIFFERENTIELLE**, et c'est la réserve principale. Le dispositif de mesure
  du 12/07 compare deux copies jumelles recevant `+stim` et `−stim`. L'effet isolé au §4 est
  précisément porté par **l'écart entre ces deux bras**. Il pourrait donc tenir au **protocole
  de lecture** autant qu'au mécanisme. Un vrai dispositif n'a pas deux copies. **Refaire la
  mesure sur une lecture non différentielle avant d'en faire un argument autonome — non fait.**
- Le retard est un **coût**, pas un gain : le doute décide *plus lentement*. C'est ce qui en
  fait une bonne signature falsifiable, mais **B6 ne doit jamais être vendue comme « le doute
  améliore les décisions du dispositif »** (déjà écrit dans `FUTURE_WORK` B6, 12/07).
- À `T_pulse = 500`, le retard vaut **2,8 pas** : aucune fraction n'y est interprétable (le
  dénominateur est quasi nul). Le leurre y est trop court pour laisser une trace. Les chiffres
  ci-dessus portent sur `T_pulse ≥ 1500`.
- Aucun circuit réel, aucun micromagnétisme spatial résolu, et le canal de couplage électrique
  de Romera *et al.* 2018 n'est toujours pas modélisé.
