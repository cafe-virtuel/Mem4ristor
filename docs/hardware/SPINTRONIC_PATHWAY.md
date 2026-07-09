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
| `u` (doute, lent) | **Non résolu ici.** Candidat le plus proche : une fenêtre d'intégration/masque temporel comme celle utilisée en reservoir computing à un seul STNO (Torrejon et al. 2017) — mais c'est une construction algorithmique externe dans la littérature citée, pas un état physique interne au dispositif | Spéculatif, plus faible que le mapping `v` |

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

- Aucune simulation LLG (Landau-Lifshitz-Gilbert) ou macrospin d'un réseau de STNO
  couplés n'a été réalisée. Construire un tel simulateur — et vérifier qu'il
  reproduit qualitativement le mécanisme du doute (polarité de couplage
  dépendante de l'état) — est un projet de plusieurs semaines (cf. B2 dans
  `docs/FUTURE_WORK.md`, effort 🧩).
- Le rôle de `u` reste non résolu physiquement (§2) — c'est le point le plus
  faible de ce mapping, plus faible que le mapping GST↔`u` du dossier photonique
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

1. [ ] Modèle STNO macrospin minimal (équation de Slonczewski simplifiée) — vérifier
   qu'un couplage mutuel dépendant du désaccord de phase peut réellement inverser
   de signe (condition nécessaire pour porter le mécanisme du doute).
2. [ ] Reproduire qualitativement l'ablation FROZEN_U/FULL (Cohen d≈9, B4) sur un
   réseau simulé de STNO couplés — test de portabilité du résultat le plus robuste
   du projet à un substrat physique différent.
3. [ ] Si (1) et (2) tiennent : proposer un protocole expérimental réel
   (cf. `docs/FUTURE_WORK.md` B6, signature falsifiable).
