# B3 — Comparaison énergie/vitesse entre les 3 familles de dispositifs

> Créé le 9 juillet 2026 — Claude Code (Opus 4.8) / Julien Chauvin
> Statut : synthèse de calculs dimensionnels (pas de mesure physique).
> Dépend de B2 (`PHOTONIC_PATHWAY.md` §5, `SPINTRONIC_PATHWAY.md`,
> `ELECTRICAL_PATHWAY.md`) — script source unique :
> `experiments/b2_device_physics_mapping.py` → `figures/b2_device_physics_mapping.csv`.
> **AUCUN claim publié ne dépend de ce document.**

## 1. Le problème que B3 pose (rappel `docs/FUTURE_WORK.md`)

« Le papier n'a aucune unité physique (dt=0.05 sans dimension) ». Ancrer dt
suppose de choisir UN dispositif — mais B2 en a exploré trois, avec des rôles
différents (voir tableau). Ce document compare les trois de façon honnête,
sans en déclarer un vainqueur artificiel : **ils ne mesurent pas la même chose**.

## 2. Tableau comparatif (ordres de grandeur, pas des mesures)

| Famille | Rôle dans le modèle | T_node ancré | dt physique | Mode de comptage énergie | Énergie |
|---|---|---|---|---|---|
| Photonique (GST) | `u` (doute, lent) | 100–200 ns | 225–449 ps/pas | Signal seul (Λ=10 photons), **hors overhead** | 1.28 aJ/pas (**plancher théorique, pas système**) |
| Spintronique (STNO vortex) | `v` (oscillateur) | 1–10 ns | 2.25–22.5 ps/pas | Puissance continue × dt | 6.7–67 fJ/pas (vortex ~3 mW) |
| Électrique (neuristor Mott NbO2) | `v` (oscillateur, alternative) | 1000–2000 ns | 2.25–4.49 ns/pas | Spike (dérivé, non mesuré) réparti sur le cycle | 22–225 fJ/pas |
| Électrique (RRAM/VTEAM) | Poids de couplage (statique) | — (pas de dynamique propre) | — | Écriture **unique**, pas par pas | 10–50 fJ/écriture (cas optimiste scalé) à qq nJ (cas moins mature) |
| **Référence CMOS/neuromorphique** | — | — | — | Par opération synaptique (discrète) | Loihi ~24 pJ/op ; TrueNorth ~26 pJ/événement |

## 3. Ce que le tableau dit — et ce qu'il ne dit PAS

**Ce qu'il dit, avec confiance modérée (calcul reproductible, littérature vérifiée) :**
- Les trois familles de dispositifs dynamiques (photonique-signal excepté, cf.
  réserve ci-dessous) convergent vers un ordre de grandeur **fJ/pas**, trois à
  quatre ordres de grandeur **sous** la référence CMOS/neuromorphique
  (pJ/opération). C'est cohérent — un « pas » du modèle est une fraction d'un
  cycle d'activité, alors qu'une « opération synaptique » Loihi/TrueNorth est un
  événement plus intégrateur ; il ne faut pas lire ceci comme « Mem4ristor bat
  Loihi de 4 ordres de grandeur », mais comme un signe que les échelles physiques
  choisies ne sont pas absurdes.
- La spintronique est la plus rapide (dt physique en picosecondes), le photonique
  intermédiaire (dt en centaines de picosecondes), l'électrique-Mott le plus lent
  des trois candidats dynamiques (dt en nanosecondes) — mais tous restent dans
  une fenêtre de 3 ordres de grandeur les uns des autres, pas 10.
- Le RRAM en rôle de poids statique est structurellement le moins coûteux des
  trois éléments électriques/photoniques : il ne paie l'énergie de commutation
  qu'une fois, pas à chaque pas.

**Ce qu'il ne dit PAS — réserves qui priment sur le tableau :**
- Le chiffre photonique (aJ) **n'est pas comparable tel quel** aux autres : c'est
  une énergie de signal pure, alors que les trois autres incluent (approximativement)
  la dissipation du dispositif actif lui-même. Une comparaison honnête demanderait
  d'ajouter les pertes d'insertion/détection/laser au photonique — non fait ici,
  cf. `PHOTONIC_PATHWAY.md` §5.
- Le chiffre neuristor Mott est **dérivé d'une revendication relative** (Pickett
  et al. 2013, « 1% de l'énergie biologique »), pas mesuré indépendamment —
  cf. `ELECTRICAL_PATHWAY.md` §3, réserve explicite.
- Aucun de ces trois dispositifs n'a été simulé en réseau couplé reproduisant le
  mécanisme du doute — la comparaison énergie/vitesse porte sur **la brique
  élémentaire** (un nœud ou un poids), pas sur un système Mem4ristor complet
  physiquement réalisé. B3 reste donc un exercice de cadrage, pas une preuve de
  faisabilité énergétique du système.
- « dt physique » diffère selon la famille (RRAM n'en a pas, car statique) —
  le tableau ne doit pas être lu comme « on choisit LA bonne famille » mais comme
  « chaque famille joue un rôle plausible, à des échelles de temps compatibles
  entre elles à un facteur ~1000 près (ps à ns), ce qui reste physiquement
  raisonnable pour un système hybride multi-dispositifs ».

## 4. Verdict honnête

**B3 est cadré, pas clos.** Les trois familles donnent des ordres de grandeur
mutuellement compatibles (aucune n'est disqualifiée par un mismatch de 6+ ordres
de grandeur), et individuellement compatibles avec des systèmes réels documentés
dans la littérature (Feldmann 2019, Torrejon/Romera 2017/2018, Pickett 2013).
Mais aucune énergie « système complet » n'a été calculée — cela demanderait de
choisir UNE architecture hybride précise (quel dispositif pour quel rôle, combien
de nœuds, quel overhead d'interconnexion) et de la chiffrer bout en bout, ce qui
est un projet de plusieurs semaines (cohérent avec le statut 🧩 de B2/B3 dans
`docs/FUTURE_WORK.md`).

## 5. Lien avec B6 (signature falsifiable)

Le résultat le plus robuste et le mieux quantifié du projet (ablation FROZEN_U,
Cohen d≈9, B4 du 8 juillet 2026) porte sur le **couplage**, pas sur l'énergie —
c'est pour cette raison que la proposition B6 (voir `docs/FUTURE_WORK.md` section
B6) s'appuie sur un réseau de STNO couplés plutôt que sur une mesure d'énergie :
la synchronisation est directement mesurable sur un dispositif spintronique réel
(spectroscopie micro-onde, cf. Romera et al. 2018), alors qu'une signature
énergétique du doute n'a pas d'équivalent expérimental évident à ce stade.
