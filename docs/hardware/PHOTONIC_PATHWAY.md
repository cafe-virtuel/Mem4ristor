# Voie Photonique — Note d'ouverture du dossier de faisabilité (Vague 2)

> Créé le 12 juin 2026 — Claude Code (Fable) / Julien Chauvin
> Statut : note de faisabilité exploratoire. AUCUN claim publié ne dépend de ce document.
> Premier résultat : `experiments/photonic_transduction_poc.py` (10 seeds, code actuel).

## 1. L'idée (piste V6, notée le 05/05/2026)

Remplacer le stimulus électrique `I_stimulus` par une stimulation lumineuse :
une fibre (ou un guide d'onde) par nœud, un matériau photosensible en réception.

**Avantages attendus** : isolation galvanique (pas de diaphonie électrique),
parallélisme naturel (1 canal = 1 nœud), vitesse, et compatibilité avec les
plateformes photoniques neuromorphiques existantes.

## 2. Ancrage dans la littérature (vérifié)

- **GST (Ge₂Sb₂Te₅)** : matériau à changement de phase. Neurones photoniques à
  spikes tout-optiques démontrés expérimentalement sur guides d'onde :
  Feldmann, Youngblood, Wright, Bhaskaran, Pernice, *All-optical spiking
  neurosynaptic networks with self-learning capabilities*, **Nature 569,
  208–214 (2019)**. Le comportement memristif optique du GST est établi.
- **VO₂** : transition isolant-métal déclenchable optiquement/thermiquement,
  oscillateurs neuromorphiques étudiés.
- **WO₃** : photochrome/électrochrome, plus lent, stable.

Correspondance conceptuelle à explorer : la variable de doute `u` comme état
de transmittance d'une cellule GST (nœud « certain » = transparent au voisinage,
nœud « douteux » = modulant). Non testé — idée directrice pour le dossier.

## 3. Premier résultat : tolérance au bruit de photons (12/06/2026)

**Question** : le régime fonctionnel et la dead zone survivent-ils quand
`I_stimulus` est délivré avec le bruit de grenaille (Poisson), dont l'amplitude
relative est imposée par la physique (σ/I = 1/√Λ, Λ = photons/nœud/pas) ?

**Protocole** : BA m=3 (fonctionnel) et m=5 (dead zone), N=100, I_nominal=0.5,
heretic_ratio=0.15 (inversion au niveau de la photodétection), 10 seeds canoniques,
Λ ∈ {3, 10, 30, 100, 300, 1000} + contrôle déterministe.
Script : `experiments/photonic_transduction_poc.py`.
Données : `figures/photonic_transduction_poc*.csv`, figure `.png`.

**Résultats** :

| Régime | Verdict |
|---|---|
| Fonctionnel (m=3) | Signatures intactes sur TOUTE la plage, jusqu'à Λ=3 (bruit 58 %) |
| Dead zone (m=5) | H_cog = 0.0000 partout — immune au bruit de photons (cohérent claim [12]) |

**Chiffre de dimensionnement : Λ_min ≈ 10 photons/nœud/pas** (avec marge ; le
régime qualitatif tient même à Λ=3, seule la dispersion de H_cont s'élargit).

**Interprétation** : le modèle opère déjà avec un bruit intrinsèque gaussien
σ_eff ≈ 0.22 (convention Euler-Maruyama, σ_v=0.05, dt=0.05) ; le bruit de
grenaille multiplicatif sur le stimulus (σ ≈ 0.16 à Λ=10) reste dans le même
ordre de grandeur et se fond dans le régime bruité existant. La robustesse au
bruit n'est pas un accident : elle prolonge le claim [12] (types de bruit
non-discriminants, seule l'amplitude compte).

## 4. Conversion en puissance physique (à faire — dépend de dt physique)

Le pas de temps dt du modèle n'a pas d'échelle physique fixée. Le jour où elle
l'est (par la constante de temps du matériau choisi) :

```
P_par_nœud = Λ × E_photon / Δt_physique        E_photon(1550 nm) ≈ 1.28×10⁻¹⁹ J
```

Exemple indicatif : Λ=10, Δt=1 µs → P ≈ 1.3 fW/nœud (dérisoire — le budget
réel sera dominé par les pertes d'insertion et la photodétection, pas par le
signal lui-même).

## 4bis. Deuxième résultat : couplage optique et TOUT-OPTIQUE (12/06/2026)

**Question** : si le canal de couplage inter-nœuds (le terme laplacien — là où agit
la polarité modulée par u) devient optique, les régimes survivent-ils ? Subtilité :
σ_social = |laplacien| pilote la dynamique de u → le bruit contamine aussi la
*perception* du désaccord local. Signe du couplage préservé par détection
différentielle (deux rails / détection cohérente).

**Protocole** : mêmes conditions que §3, trois canaux : COUPLING seul (stimulus
électrique parfait), FULL-OPTICAL (stimulus + couplage au même Λ), contrôle
déterministe (vérifié **identique bit à bit** au code de référence — la sous-classe
`PhotonicCouplingNet` ne modifie que le canal optique).
Script : `experiments/photonic_coupling_poc.py` (260 runs, 10 seeds).

**Résultats** :

| Canal optique | Verdict |
|---|---|
| Couplage seul | Intact sur TOUTE la plage, y compris Λ=3 (58 % de bruit) — plus tolérant que le stimulus |
| Tout-optique | Intact dès **Λ=10** ; à Λ=3, léger étalement H_cont (+0.17–0.20) sans changement de régime |
| Dead zone | H_cog = 0.0000 sur les 12 conditions — imperturbable |

**Conclusion d'étape** : un Mem4ristor **entièrement photonique** (stimulus +
couplage) est compatible avec un budget de **~10 photons/nœud/pas** sur chaque
canal. Le mécanisme central (polarité par u) est même PLUS tolérant au shot noise
que l'entrée. Interprétation : le doute u est un filtre passe-bas naturel
(τ_u = 10 pas) — il moyenne le grain photonique avant qu'il n'atteigne la décision
de polarité.

## 4ter. Troisième résultat : chaîne de transduction GST réaliste (12/06/2026)

**Question** : un matériau réel — qui sature (réponse en S) et qui a une inertie
(constante de temps τ_mat) — préserve-t-il les régimes ? Chaîne complète testée :
photons (Poisson Λ=10) → saturation T(P)=P(1+s)/(1+sP) → passe-bas 1er ordre (τ_mat)
→ stimulus. Script : `experiments/photonic_gst_transduction_poc.py` (380 runs, 10 seeds).

**Résultat : 36/36 conditions OK.** Saturation jusqu'à s=3 (compression forte) ×
inertie jusqu'à τ_mat=100 pas : aucun changement de régime (fonctionnel intact,
dead zone H_cog=0.0000 partout).

**⚠️ Lecture honnête du τ_mat=100** : dans ce protocole le stimulus nominal est
STATIONNAIRE — l'inertie ne retarde rien, elle filtre le bruit (bénéfique). La
spécification est donc : *aucune contrainte de bande passante matériau pour un
stimulus stationnaire*. Pour des stimuli DYNAMIQUES (événements type claim [13],
drive variable type POC C), τ_mat redevient critique — c'est l'étape 2bis ci-dessous.

**Conséquence pour la conversion physique** : en régime stationnaire, le choix de
dt_physique n'est PAS contraint par le matériau de transduction — il le sera par
la dynamique du COUPLAGE (la boucle réseau), pas par l'étage d'entrée. Les matériaux
photochromes « lents » (WO₃) redeviennent candidats pour l'étage stimulus.

**Note de modèle** : T(P) est un absorbeur saturable générique, pas une courbe GST
mesurée — toute courbe matériau réelle se branche à la place, protocole inchangé.

## 5. Prochaines étapes du dossier (par coût croissant)

1. [x] ~~Bruit de photons sur le COUPLAGE~~ — **FAIT 12/06/2026** (§4bis) :
   favorable, tout-optique viable à Λ≈10.
2. [x] ~~Transduction non-linéaire réaliste (saturation + τ_mat)~~ — **FAIT
   12/06/2026** (§4ter) : 36/36 OK en stationnaire ; spec bande passante = aucune
   pour stimulus stationnaire.
2bis. [x] ~~Stimuli DYNAMIQUES à travers la chaîne GST~~ — **FAIT 12/06/2026**
   (`experiments/photonic_event_poc.py`, 80 runs) : l'effet réseau d'un événement
   (protocole [13] : I=1.5, 150 pas, nœud périphérique, BA m=3) est transmis à
   ~100 % sur toute la plage τ_mat ∈ [0, 300], y compris quand l'amplitude
   effective tombe à 0.47 (τ=300). La mesure i_eff_max suit exactement la
   prédiction analytique 1−exp(−T/τ) (mécanique validée). **Spécification de
   bande passante : τ_mat ≤ T_event × marge, où la contrainte réelle est
   I_event·(1−exp(−T/τ)) ≥ seuil topologique (~0.5 observé)** — avec I=1.5,
   même τ=2·T_event passe. ⚠️ DÉCOUVERTE COLLATÉRALE MAJEURE : la référence
   électrique re-mesurée avec le code actuel donne dH=−0.76 (le claim [13]
   d'avril donnait +1.20) — **signe inversé par le changement de bruit
   (AUDIT-024)**, claim [13] marqué À RE-VÉRIFIER dans PROJECT_STATUS. Le
   verdict photonique (transmission de l'effet) reste valide : il compare la
   chaîne optique à la référence électrique DU MÊME code.
3. [ ] Hérétiques optiques : inversion par interféromètre (Mach-Zehnder) vs
   canal séparé — coût en composants.
4. [x] ~~Variabilité de fabrication optique~~ — **FAIT 12/06/2026** (§4quater).
5. [x] ~~Mapping `u` ↔ transmittance GST~~ — **FAIT 09/07/2026** (§5 ci-dessous) :
   correspondance candidate + ancrage dt physique, calcul dimensionnel seul (pas
   de circuit simulé).

## 4quater. Quatrième résultat : variabilité de fabrication (12/06/2026)

**Question** : des pertes d'insertion statiques différentes par nœud (t_i ~ N(1, σ_fab),
sur les DEUX canaux du système tout-optique Λ=10) cassent-elles les régimes ?
Script : `experiments/photonic_fabrication_poc.py` (140 runs, 10 « puces »).

**Résultat — aucun mode de défaillance catastrophique** : jamais de synchronisation
(sync ≤ 0.012 partout), jamais de mort cognitive. L'effet dominant de la mauvaise
fabrication est une **hausse de H_cont** (+0.05 à +0.75 bits) : les gains hétérogènes
agissent comme des quasi-hérétiques structurels — de la diversité « gratuite ».
Cohérent avec la philosophie du modèle (l'hétérogénéité est une ressource) et avec
l'étude de tolérance SPICE (dead zone immune au mismatch électrique).

**Spécification en trois zones** :
| σ_fab | Verdict |
|---|---|
| ≤ 5 % | Valeurs nominales reproduites (zone verte) |
| ≤ 20-30 % | Régimes qualitatifs préservés, point de fonctionnement décalé (recalibrage requis) |
| ≥ 30 % | La dead zone commence à s'éroder (H_cog : 0 → 0.012 à 0.3, → 0.097 à 0.5) |

La photonique intégrée moderne tient les pertes d'insertion à quelques % de
dispersion — très confortablement en zone verte.

## Bilan du quatuor des imperfections physiques (12/06/2026)

| Imperfection | Nature | Verdict | Spec |
|---|---|---|---|
| Bruit quantique (shot noise) | temporel aléatoire | ✅ | Λ ≥ 10 photons/nœud/pas/canal |
| Non-linéarité matériau | statique déterministe | ✅ | saturation s ≤ 3 |
| Inertie matériau | temporel déterministe | ✅ | I_event·(1−exp(−T/τ)) ≥ ~0.5 ; aucune contrainte en stationnaire |
| Pertes de fabrication | statique aléatoire | ✅ | σ_fab ≤ 5 % nominal, ≤ 20 % qualitatif |

**Conclusion du dossier à ce stade : aucune des quatre familles d'imperfections
physiques d'un système photonique réel ne détruit les régimes de Mem4ristor aux
tolérances industrielles courantes.** Le système est nativement compatible avec
une implémentation photonique imparfaite — par conception, pas par chance : le
doute calibré (filtre passe-bas temporel) et les hérétiques (hétérogénéité assumée)
sont précisément les mécanismes qui absorbent ces imperfections.

## 5. Mapping `u` ↔ transmittance GST + ancrage dt physique (09/07/2026)

> **Nature de ce travail : calcul dimensionnel de premier ordre, reproductible
> (`experiments/b2_device_physics_mapping.py` → `figures/b2_device_physics_mapping.csv`),
> PAS une simulation de circuit GST.** Ne prouve aucune compatibilité physique
> réelle — montre que les ordres de grandeur ne sont pas absurdes. Voir aussi
> `docs/hardware/SPINTRONIC_PATHWAY.md` et `docs/hardware/ELECTRICAL_PATHWAY.md`
> (même exercice pour les deux autres familles de dispositifs, B2 "tout explorer").

**Correspondance candidate.** `u` (variable de doute, [0,1]) ↔ **fraction amorphe**
d'une cellule GST : `u→0` = cristallin (transmittance haute, "certain"), `u→1` =
amorphe (transmittance basse, "douteux"). Les états multi-niveaux gradués (fraction
amorphe intermédiaire, transmittance stable) sont établis expérimentalement pour le
stockage optique GST (au-delà de Feldmann et al. 2019, cf. littérature sur la
mémoire optique multi-niveau GST). Le seuil de polarité `u=0.5` correspondrait à un
état mi-cristallin — plausible mais **non testé en circuit**.

**Mécanisme de relaxation (spéculatif, non construit).** La dynamique de `u`
(relaxation vers `sigma_baseline` + excursion pilotée par `sigma_social`) demanderait
physiquement qu'un désaccord local perçu pilote une amorphisation graduelle
(via des pulses partiels), et qu'une dérive/recuit thermique lent ramène vers l'état
cristallin de repos. Aucun circuit de ce type n'a été construit ni simulé ici —
c'est l'idée directrice du dossier, restée « la plus spéculative » (cf. version
précédente de cette section), et elle le reste après ce calcul : seul l'ancrage
temporel/énergie a avancé, pas le mécanisme d'écriture.

**Ancrage dt physique.** Le nœud FHN isolé a une pulsation propre mesurée
(`experiments/reviewer2_linear_stability.py`, 1er mai 2026, α=0.15) : spirale stable
λ = −0.0473 ± 0.2824i → période propre `T_node = 2π/Im(λ) ≈ 22.25` unités de temps
modèle (≈ 445 pas d'intégration à dt=0.05). En ancrant `T_node` sur le temps de
réponse d'amorphisation/cristallisation GST documenté (**~100–200 ns**, pulses UV
nanoseconde, cristallisation pleine >180 ns — Structural Transitions in Ge2Sb2Te5
..., PMC7254329) :

| Ancrage | dt physique | τ_u physique | Campagne (4000 pas) |
|---|---|---|---|
| T_GST = 100 ns | 225 ps/pas | 44.9 ns | 0.90 µs |
| T_GST = 200 ns | 449 ps/pas | 89.9 ns | 1.80 µs |

**Énergie (signal seul, hors overhead).** Au budget nominal Λ=10 photons/nœud/pas
(§3), l'énergie de signal par pas est **indépendante de dt** : `10 × E_photon(1550nm)
≈ 1.28×10⁻¹⁸ J/nœud/pas`. Rapportée à dt physique ci-dessus : puissance **≈ 2.8–5.7
nW/nœud**. **Réserve non négociable** : ceci ignore les pertes d'insertion, le
rendement du photodétecteur et le wall-plug du laser — le budget réel d'un système
photonique complet se compte en pJ–nJ par événement (cf. Feldmann et al. 2019 et la
littérature sur les accélérateurs photoniques), pas en aJ. Le chiffre aJ n'est valide
que comme *plancher théorique du signal*, pas comme consommation système. Voir
`docs/hardware/B3_ENERGY_COMPARISON.md` pour la mise en regard avec les deux autres
familles et la référence CMOS (Loihi/TrueNorth).
