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

## 5. Prochaines étapes du dossier (par coût croissant)

1. [x] ~~Bruit de photons sur le COUPLAGE~~ — **FAIT 12/06/2026** (§4bis) :
   favorable, tout-optique viable à Λ≈10.
2. [ ] Transduction non-linéaire réaliste (réponse en S du GST, saturation,
   constante de temps du matériau → dt physique).
3. [ ] Hérétiques optiques : inversion par interféromètre (Mach-Zehnder) vs
   canal séparé — coût en composants.
4. [ ] Variabilité de fabrication optique (pertes d'insertion par nœud) —
   l'équivalent photonique de l'étude de tolérance SPICE existante.
5. [ ] Mapping `u` ↔ transmittance GST (l'idée directrice, la plus spéculative).
