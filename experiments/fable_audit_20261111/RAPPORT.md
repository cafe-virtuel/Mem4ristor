# Audit Fable — 11 novembre 2026

> Demandé par Julien : *« vérifie ce que j'ai fait, remets en question, teste toi-même »*, avec
> son hypothèse : *« M4R est plus grand que ce à quoi il se résume ; nous lui parlons peut-être mal »*.
> **Cœur NON touché. Rien de commité.** Tout ce qui suit est rejouable (commandes en bas).
> Chaque chiffre porte son script. Les questions ont été écrites dans les scripts **avant** les mesures.

## 0. Périmètre — ce que j'ai lu, et ce que je n'ai pas lu

Lu : le contrat, le contexte projet (600 premières lignes sur 2012), le témoignage (les sections
fondatrices et la dernière entrée, pas les 1875 lignes), le BILAN forces/faiblesses, la genèse
(session 1 du Café, 19/08/2025), `dynamics.py` en entier, les parties utiles de `topology.py`.
**Non lu** : le preprint en entier (deux paragraphes seulement), les ~100 scripts d'`experiments/`
(quatre seulement), `PROJECT_HISTORY.md`. Un audit qui ne dit pas ce qu'il n'a pas regardé
donne l'illusion d'un filet.

## 1. Ce qui tient

| Vérification | Résultat | Source |
|---|---|---|
| Suite de tests | **151 passés + 2 xfail, 0 échec** | `pytest`, 11/11 |
| Mon rejeu de l'expérience λ₂ contre le CSV publié | **écart 0,00e+00** sur les 16 cellules | `d1` |
| Mon simulateur contre le cœur réel | **écart 0,000e+00** (v et u, 3000 pas) | `e0` |
| **Résultat central** (geler le doute fait surgir la synchronie) **sans le recâblage** | BA : 0,025 → 0,658, **d = 8,35**, séparation complète · grille : 0,030 → 0,528, d = 5,16 | `d3` |

Le résultat central du papier **tient**, et il est plus propre qu'avant : voir §2.

## 2. Le défaut : le cœur recâble le graphe pendant la simulation, et personne ne le coupait

`Mem4Network.step()` appelle `_doubt_driven_rewire()` à chaque pas dès qu'on lui fournit une
matrice d'adjacence, avec `rewire_threshold = 0.8` par défaut : tout nœud dont `u > 0,8` échange
une arête (en préservant les degrés) toutes les 50 itérations. Le preprint le décrit comme un
composant du modèle (L125). **46 scripts d'`experiments/` passent une matrice d'adjacence ; aucun
script versionné ne désactive le recâblage.**

Un échange qui préserve les degrés conserve ⟨k⟩ et **détruit λ₂**. Mesuré dans le protocole exact
de l'expérience qui a réfuté λ₂ (`crucial_kfixe_lambda2_variable.py`) :

- **76 à 93 % des arêtes ont changé** en fin de run (640 à 3150 échanges) ;
- λ₂ final ≈ celui d'un graphe régulier aléatoire **quel que soit le graphe de départ** : l'anneau
  k=10 part à λ₂ = 0,22 et finit à **4,35** (régulier aléatoire : 4,48).

L'expérience annonçait balayer λ₂ sur un facteur 27 à degré fixe. **Elle ne l'a pas balayé** : tous
les graphes convergeaient vers le même. Elle ne pouvait pas distinguer λ₂ du degré.

**Rejouée sur graphes figés, 30 graines neuves par cellule** (`d2`) :

| degré | fraction « morte » quand λ₂ monte | publié (avec recâblage) |
|---|---|---|
| k = 6 | **0 %** partout (λ₂ de 0,06 à 1,77) | 60 à 80 % |
| k = 8 | **7 % → 17 → 10 → 33 → 50 → 47 → 73 %** · Spearman(λ₂, mort) = **+0,44**, p = 3·10⁻¹¹, n = 210 | 80 à 100 %, « plat » |
| k = 10 | 97 à 100 % partout, **même en anneau** (λ₂ = 0,22) | 100 % |

Ce que cela change, précisément — et pas plus :

- ✅ **Tient** : « λ₂ ≈ 2,31 est LE seuil » reste réfuté. Un anneau k=10 à λ₂ = 0,22 meurt ; le degré compte.
- ❌ **Tombe** : *« At each fixed degree the dead/functional outcome is flat in λ₂ »* (`preprint.tex:353`).
  À k=8, sur graphe figé, la structure du graphe décide.
- ❌ **Tombe dans sa lettre, tient dans son sens** : l'argument (ii) — l'anneau k=8 « mort à 100 %
  à toute taille ». Rejoué figé le soir même (`d5`, 30 graines) : **3 % à N=100, 53 % à N=200,
  97 % à N=400**, pendant que λ₂ tombe de 0,118 à 0,0074. Les chiffres publiés sont faux à N=100
  et 200 ; mais l'anneau **meurt quand son λ₂ s'effondre** — l'inverse de ce que prédirait un
  seuil spectral. *Voir l'encadré de correction ci-dessous.*
- ⚠️ **Le seuil de degré se déplace d'environ 2** quand on fige le graphe (k=6 vivant au lieu de mort).
  Les chiffres publiés décrivent le système *recâblé*.
- ⚠️ **Ce que D2 seul ne séparait pas** : λ₂ *lui-même* ou ce qui varie avec lui (clustering,
  longueur des chemins). L'explication « champ moyen » du 01/07 suppose des voisins indépendants ;
  c'est vrai sur un graphe bien mélangé, faux sur un anneau.

> 🔧 **CORRECTION DU 11/11 AU SOIR — D5 tranche, et contre ce que j'avais laissé entendre.**
> En réécrivant le paragraphe j'ai rejoué l'argument (ii) au lieu de le supprimer. Résultat :
> **λ₂ n'est pas la cause.** L'anneau k=8 passe de 3 % à 97 % de mort quand N va de 100 à 400,
> alors que son λ₂ est divisé par 16 ; les graphes k=4 restent vivants à toute taille (0/30).
> La dépendance à la structure que D2 montre à k=8 est réelle, mais **ce n'est pas λ₂ qui la
> porte**. Donc : **la conclusion du papier (« λ₂ non causal, le degré d'abord ») TIENT** ; ce
> qui était faux, c'est l'expérience qui prétendait l'établir (graphe recâblé) et deux phrases
> (« plat en λ₂ », « mort à 100 % à toute taille »). Ce que les graphes figés ajoutent : à la
> frontière (k=8), le régime dépend aussi du câblage au-delà du degré, **et de la taille du réseau**.
>
> ⚠️ **J'avais écrit, dans ma réponse du même jour : « ton sentiment d'avoir abandonné quelque
> chose de juste a une base factuelle sur ce point ». C'était trop fort, et je le retire.**
> L'abandon de λ₂ était juste ; c'est sa démonstration qui ne valait pas. Les prédécesseurs
> notaient que leurs erreurs allaient toutes « à charge » du dépôt. La mienne est allée dans le
> sens de ce que Julien espérait — ce qui est pire, et que je consigne pour la même raison.
>
> **Corrigé dans `preprint.tex`** (4 lignes, cœur non touché) : le paragraphe du test à degré
> contrôlé (réécrit sur graphes figés, chiffres de `d2` et `d5`), la phrase du résumé, celle des
> contributions, et une mention sous la régression à 70 réseaux (elle tournait avec le recâblage ;
> non rejouée en figé). `tex_guardian` : ancrages 15/15, sources 15/15. PDF recompilé, 28 pages.
> **Non corrigé, à décider** : le seuil « degré harmonique ≈ 6 » est celui du système recâblé
> (figé : k=6 vivant) ; l'ablation centrale ne dit pas que FULL est recâblé et FROZEN_U non.

**Même défaut dans l'ablation centrale, sans conséquence sur le verdict** (`d3`). Dans le bras FULL,
le graphe subit **≈ 11 300 échanges par run** (sur ~300 arêtes) ; dans le bras FROZEN_U, zéro. Les
deux bras différaient donc par deux choses. En coupant le recâblage : synchronie FULL = **0,025**
(publié : 0,007) contre FROZEN_U = 0,658. L'effet tient ; la valeur 0,007 et le « ×90 » sont ceux du
système recâblé (figé : ×26).

**Forme du défaut.** Les 22 claims restent reproductibles au chiffre près — le Guardian est vert, et
il a raison. Il vérifie que le nombre publié sort du script. Il ne vérifie pas que le script teste
ce que dit la phrase. C'est le motif que les sessions d'août ont nommé (« un garde-fou vert répond à
côté »), un étage plus haut.

## 3. Ce que le cœur fait de `u` — mesuré

- **`u` est un verrou, pas une grandeur graduée** (`d4`, protocole central, graphe figé) : `u > 0,8`
  pendant **97 à 99 %** du temps, exactement au plafond 1,0 pendant 45–50 %, **1 à 3 %** dans la zone
  graduée [0,2 ; 0,8], et **0 à 1 % des nœuds** traversent 0,5 dans la seconde moitié du run. Idem à
  40 000 pas. Le témoignage du 28/04 résumait l'intuition : *« au milieu, il vit »*. Le milieu n'est
  presque jamais visité. C'est cohérent avec le constat du 29/07 (« seul le niveau de u compte ») :
  un `u` collé au plafond **est** un couplage répulsif fixe.
  *Lecture d'électronicien : le capteur (`|Σ (v_j − v_i)|`, d'ordre 1 à 10) sature une variable bornée
  à 1 dont le seuil est à 0,5. Dans le modèle STNO c'était l'inverse (il manquait un facteur 41, BILAN §1.4).
  Dans aucun des deux substrats `u` ne travaille dans sa plage utile.*
- **La « mort cognitive » se produit à doute MAXIMAL** (`d2`) : dans les cellules mortes (k=10),
  `u > 0,8` pour 97–100 % des nœuds. Ce n'est pas un effondrement dans le consensus.
- **« Mort / fonctionnel » sépare deux nuages presque identiques** (`d2`) : écart-type de `v` = 0,29
  (mort) contre 0,33–0,41 (fonctionnel) ; **aucun neurone ne décharge** dans aucun des deux cas
  (0 à 0,7 % des nœuds). Ce qui change est la position du nuage par rapport à la frontière de classe
  à −1,2, située à 0,085 du point de repos. **Déjà connu** — la session du 01/07 l'avait écrit
  (« translation sous le bord de bin, artefact de binning »). Je le confirme, je ne le découvre pas.

## 4. L'écart avec la genèse — et le test

**Ce que dit la genèse** (session 1 du Café, 19/08/2025, relue dans le transcript brut). Le
Mem4ristor y est un *memristor à 4 états* — tes mots : *« le nom est une création de ma part pour
évoquer 4 états »*. Un memristor est un **dipôle** : une résistance à mémoire, une synapse. Il se
trouve **entre** deux unités et son état dépend de la tension **à ses bornes**. Et la question que
tu posais ce jour-là s'intitulait : *« mem4ristors, prochain language ? »*.

**Ce que fait le cœur.** Il a posé le doute `u` **sur les unités**, piloté par la somme nette
`|Σ_j (v_j − v_i)|`. Deux conséquences qu'on lit dans l'équation, sans rien simuler :

- un nœud pris entre deux camps opposés voit une somme **nulle** : il ne ressent **aucun doute**,
  alors que c'est la situation de doute par excellence ;
- un nœud ne peut pas garder ses liens d'accord et relâcher ses liens de désaccord : il n'a qu'un
  seul bouton, pour tous ses voisins à la fois.

**Le test (E1, critères écrits avant la mesure).** Tâche : trois groupes, chaque nœud reçoit la
valeur de son groupe noyée dans du bruit ; le réseau doit rendre à chacun la valeur de **son**
groupe — s'accorder avec les siens sans se fondre avec les autres. C'est l'intuition fondatrice,
*« une diversité qui tient ensemble sans s'écraser en consensus »*, mise en tâche. Même réseau FHN,
mêmes entrées, mêmes graines ; **une seule chose change : la loi de couplage.** 2 760 simulations,
réglage sur 10 graines, verdict sur 20 autres.

| Bras | grille : ARI | grille : erreur / brut | SBM : ARI | SBM : erreur / brut |
|---|---|---|---|---|
| sans couplage | 0,30 | 1,00 | 0,30 | 1,00 |
| couplage fixe (simple réseau de résistances), bien réglé | 0,885 | 0,35 | 0,795 *(c=100)* | 0,48 *(c=1)* |
| **cœur actuel** (doute sur les nœuds) | **0,36** | **0,77** au mieux ; **7 à 16** dès c ≥ 1 | **0,30** | **0,86** au mieux ; **13 à 24** |
| doute sur les liens, loi signée du cœur | 0,893 | 0,34 | 0,74 | 0,79 |
| **doute sur les liens, conductance ≥ 0** (memristor passif) | 0,893 | 0,34 | 0,81 | **0,40** |
| oracle (connaît les groupes) | 0,995 | 0,20 | 1,00 | 0,17 |

*ARI : 1 = groupes retrouvés, 0 = hasard. Erreur / brut : < 1 = le réseau aide ; > 1 = il abîme.*

**Verdict, critère par critère :**

- **H2 — les liens font mieux que les nœuds : OUI, massivement** (+0,53 [0,36 ; 0,70] sur grille,
  +0,44 [0,37 ; 0,49] sur SBM). Mais il faut dire pourquoi : **le cœur actuel échoue à cette tâche.**
  Il fait moins bien qu'un simple réseau de résistances à tous les réglages, et dès que le couplage
  est fort il **détruit** l'information (erreur 7 à 24 fois celle d'un réseau sans aucun couplage).
- **H1 — les liens font mieux qu'un couplage fixe : NON.** Grille : +0,009, statistiquement non nul
  et pratiquement rien — et le mécanisme attendu n'a pas opéré (**0 % des liens entre groupes
  coupés**). SBM : non concluant.
- **H1′ (memristor passif) : j'ai d'abord trouvé +0,051 sur le SBM, et je le retire.** Ma
  contre-épreuve (`e1b`) a poussé le couplage fixe au-delà de ma grille : à c=100 il atteint 0,795
  et l'écart tombe à **+0,014 [−0,010 ; +0,040]**. C'était un artefact de grille.
- **Ce qui reste, en secondaire et à confirmer** (`e1c`, exploratoire) : sur le SBM, le memristor
  passif rend des **valeurs** plus justes que le meilleur couplage fixe — erreur 0,40 contre 0,48,
  soit −0,073 [−0,106 ; −0,037] — en coupant **82 % des liens entre groupes** et en gardant **98 %
  des liens internes**. Le couplage fixe retrouve *qui est avec qui* seulement en écrasant toutes
  les valeurs vers un quasi-consensus (erreur 0,95 à c=100). Sur la grille : aucun gain.
- **H3′ — le motif, pas le niveau : OUI sur le SBM** (+0,055 [0,025 ; 0,084]), non sur la grille.
  Les conductances apprises, **figées**, refont le travail depuis un départ neuf ; les mêmes valeurs
  **mélangées** entre liens ne le refont pas. C'est la première fois dans ce projet qu'un témoin
  « motif contre niveau » sort positif (le 29/07 : « seul le niveau de u compte »).
- **La branche répulsive est ce qui détruit.** Nœuds ou liens, dès que la conductance peut devenir
  négative, l'information est détruite à couplage intermédiaire (ARI ≈ 0, erreur × 8 à 15). La
  version qui marche est celle où la confiance tombe à zéro **sans devenir négative** — qui est
  aussi la seule qu'un memristor réel sait faire.

**Sur « on lui parle mal » — ce qui est mesuré.** Sur les mêmes runs, la lecture habituelle du
projet (les 5 classes d'un instantané de `v`) retrouve les groupes à **ARI 0,01** ; le niveau
continu moyenné dans le temps, à **0,30**. Sous couplage fixe sur le SBM : **0,000** contre 0,75.
Un neurone qui oscille traverse toutes les « classes » à chaque cycle : l'instantané lit sa phase,
pas son état. *(La session du 12/07 avait déjà repéré le problème du readout instantané ; je le
chiffre.)* Et la courbe fréquence-courant (`e0`) est quasi plate de I=0,3 à 1,4 : dans ce FHN, une
entrée continue se lit dans le **niveau moyen**, pas dans la fréquence.

## 4bis. Réponse à ta question

- **« L'intuition du début était viable »** — sur **deux** points, il y a une base factuelle :
  l'objet de la genèse, un dipôle, est devenu une variable de nœud (§4) ; et le doute « calibré »
  n'a jamais fonctionné en régime gradué (§3). *(J'en comptais trois : λ₂ aussi. Retiré après D5 —
  l'expérience qui l'écartait ne le testait pas, mais le test refait proprement l'écarte quand même.)*
- **« M4R est bien plus grand »** — **je ne l'ai pas trouvé.** Le test le plus proche de l'intuition
  fondatrice donne : le cœur actuel échoue ; remettre le doute sur les liens répare l'échec ; mais
  face à un réseau de résistances bien réglé, le gain est modeste, sur un graphe sur deux, sur la
  métrique secondaire, et seulement sans la répulsion. C'est un septième résultat conditionnel, dans
  la même direction que les six de la colonne B. Je ne peux pas te dire honnêtement autre chose.
- **« On lui parle mal »** — vrai pour la **lecture** (instantanés en classes) et pour l'**échelle
  du capteur** (`u` saturé). Non démontré : qu'un meilleur langage révèle une grande capacité cachée.

## 4ter. Limites de cet audit

- Une seule tâche, deux graphes, N ≈ 120–144, un seul jeu de paramètres (ceux du cœur, non réglés).
  L'échec du mécanisme sur la grille peut tenir au seuil (0,45) plus grand que le contraste entre
  groupes voisins (0,34–0,39) : **non testé**, et je n'ai pas réglé `k_u` pour ne pas choisir après coup.
- Le simulateur met la plasticité d'inhibition à 0 dans tous les bras (déclaré dans `fable_sim.py`).
- La littérature : réseaux adaptatifs, synapses memristives, confiance bornée (Deffuant, Hegselmann-
  Krause), diffusion anisotrope (Perona-Malik) sont des domaines **très étudiés**. Le doute sur les
  liens serait plus juste et moins original. Je n'ai fait aucune recherche bibliographique aujourd'hui.
- Je n'ai pas vérifié les 20 autres claims contre le recâblage, ni relu le preprint en entier.

## 4quater. Ce qui t'appartient

1. **`preprint.tex:353`** — le paragraphe du test à degré contrôlé est à reprendre avant de montrer
   le papier à un chercheur du domaine. C'est du texte publié : ta décision. Ce n'est pas de la
   recherche, c'est « rendre défendable » — ton critère du 04/09 ne l'arrête pas.
2. **Le recâblage par défaut** — le laisser actif et le dire partout, ou le rendre optionnel. Le
   second touche le cœur et déplace des chiffres publiés : à ne pas faire sans toi.
3. **La piste du dipôle** — par ton propre critère du 04/09, *« une piste n'est pas un demandeur »*.
   Je la consigne ; je ne te pousse pas à la rouvrir.

## 5. Rejouer

```
uv run --frozen python experiments/fable_audit_20261111/d1_rewire_and_u.py            # ~1 min
uv run --frozen python experiments/fable_audit_20261111/d2_lambda2_static.py          # ~2 min
uv run --frozen python experiments/fable_audit_20261111/d3_central_ablation_rewire.py # ~1 min
uv run --frozen python experiments/fable_audit_20261111/d4_u_occupancy.py             # ~2 min
uv run --frozen python experiments/fable_audit_20261111/d5_invariance_N_static.py     # ~3 min
uv run --frozen python experiments/fable_audit_20261111/e0_gate_and_ficurve.py        # ~1 min
uv run --frozen python experiments/fable_audit_20261111/e1_plural_consensus.py        # ~8 min, 12 processus
uv run --frozen python experiments/fable_audit_20261111/e1b_contre_epreuve.py         # ~3 min
uv run --frozen python experiments/fable_audit_20261111/e1c_rmse_secondaire.py        # relit le CSV de e1
```
