# Audit externe — Mem4ristor V6.0.0

**Date :** 2026-08-02
**Auditeur :** Claude (Opus 5), regard neuf, aucune connaissance préalable du projet
**Périmètre :** `src/`, `tests/`, `.github/`, documentation racine + `docs/`, hygiène du dépôt
**Méthode :** lecture intégrale des 2 646 lignes de `src/mem4ristor/`, exécution de la suite de
tests, et **vérification expérimentale** de chaque constat marqué ⚙️ (script rejouable donné).

---

## Comment lire ce rapport

| Marque | Sens |
|---|---|
| 🔴 **Bloquant** | Fausse un résultat publié, ou rend le dépôt non reproductible par un tiers. À traiter avant toute soumission arXiv. |
| 🟠 **Important** | Piège réel : ne casse rien aujourd'hui, mais produira un faux résultat silencieux tôt ou tard. |
| 🟡 **Cosmétique** | Confusion, dette, incohérence de surface. |
| ⚙️ | Constat **vérifié par exécution**, pas par lecture. La commande est donnée. |

**Note liminaire, et elle est sincère :** `docs/limitations.md` et `PROJECT_STATUS.md` sont d'une
honnêteté rare. Plusieurs des problèmes ci-dessous y sont **déjà identifiés par vous** — je les
reprends quand même, parce que le constat est posé dans un fichier interne pendant que le README,
le CITATION.cff et le `RESULTS_INDEX.json` continuent d'affirmer le contraire à la porte d'entrée.
L'écart n'est pas dans votre lucidité ; il est dans la propagation.

---

# 🔴 BLOQUANT

## B1 — L'infrastructure de vérification des claims est hors du dépôt

`PROJECT_STATUS.md` §0 met en avant, trois fois, la garantie centrale du projet :
« **Guardian : 20/20 claims vérifiées automatiquement à chaque commit** », « Tex Guardian
**bloquant** », « Contrôle N4 : 12/12 ».

Or :

```
$ git log --all -- .brain     → (vide : jamais versionné)
$ ls .brain                   → No such file or directory
$ head .git/hooks/pre-commit
    GUARDIAN_SCRIPT="D:/ANTIGRAVITY/.brain/preprint_guardian.py"
    PYTHON_BIN="C:/Users/julch/AppData/Local/Programs/Python/Python313/python.exe"
```

Le Guardian vit **en dehors du dépôt**, à un chemin absolu Windows, sur une seule machine. Les
hooks git ne sont jamais clonés. Huit fichiers versionnés référencent `.brain/`, dont
`experiments/verify_table1_preprint.py` (« la synchronie publiée venait de `.brain/gen_c07_csv.py` »).

Conséquences :

1. **Aucun tiers ne peut vérifier le chiffre « 20/20 ».** Pour un relecteur arXiv, c'est une
   affirmation invérifiable présentée comme une garantie mécanique.
2. **Point de défaillance unique.** Une panne de disque emporte tout l'appareil de contrôle
   qualité du projet — y compris les générateurs de données publiées.
3. Le hook ne se déclenche que sur votre poste : sur toute autre machine, les commits passent
   sans contrôle, en silence.

**Correctif**
- Déplacer `preprint_guardian.py`, `claims_mapping.json`, `gen_c07_csv.py` dans `tools/guardian/`
  **dans le dépôt**.
- Remplacer les chemins absolus par des chemins relatifs à la racine git
  (`git rev-parse --show-toplevel`) et `PYTHON_BIN` par `python3`.
- Versionner le hook dans `tools/hooks/pre-commit` + une ligne d'installation
  (`git config core.hooksPath tools/hooks`) dans `CONTRIBUTING.md`.
- Ajouter un job GitHub Actions qui lance le Guardian : c'est ce qui rend le « 20/20 » **public**.

---

## B2 — ⚙️ L'exemple phare du README ne reproduit pas le chiffre qu'il annonce

`README.md` (bloc « Scale-Free Networks ») :

```python
net = Mem4Network(adjacency_matrix=adj, heretic_ratio=0.15, coupling_norm='degree_linear', seed=42)
for step in range(3000): net.step(I_stimulus=0.0)
print(f"Scale-Free Entropy: {net.calculate_entropy():.4f}")
# Expected: H_stable ~ 0.83 (86% recovery of lattice performance)
```

Exécution littérale du bloc :

```
H (calculate_entropy, défaut = continue 100 bins) = 3.1809
H_cog (bins ±0.4/±1.2)                           = 0.0000
```

Ni l'un ni l'autre ne vaut 0,83. Deux causes se cumulent :

1. **Ambiguïté du symbole `H`.** Le 0,828 de `docs/limitations.md` (LIMIT-02) vient de
   l'entropie **cognitive** (5 bins, plafond log₂5 ≈ 2,32). Mais `calculate_entropy()` renvoie
   par défaut l'entropie **continue** (100 bins, plafond log₂N ≈ 6,64). Deux métriques, un seul
   nom, un facteur ~4 d'écart. `docs/limitations.md` ne précise jamais laquelle est tabulée.
2. **H_cog = 0,0000 exactement** signifie que les 100 nœuds sont dans le même bin : sur cette
   graine, `degree_linear` ne restaure **pas** la diversité. Le tableau LIMIT-02 annonce
   0,828 ± 0,069 sur 5 graines — le protocole exact (nombre de pas, `I_stim`, graines) n'est
   nulle part, donc l'écart est indécidable en l'état.

**Correctif**
- Renommer sans ambiguïté : `H_cog` vs `H_cont` **partout** (docs, figures, CSV, code).
- Publier le protocole complet de LIMIT-02 (graines, steps, `I_stim`, métrique) et le script.
- Corriger le bloc du README pour qu'il affiche la valeur qu'il produit réellement, ou changer
  l'appel en `net.calculate_entropy(use_cognitive_bins=True)` si c'est bien H_cog qui est visé.

---

## B3 — ⚙️ Les hérétiques sont un no-op exact à `I_stimulus=0` — et l'écart observé est un artefact RNG

`docs/limitations.md` le dit déjà (« 🚨 VACUOUS — AUDIT 2026-04-22 »). Je confirme et j'ajoute
un second effet que la note ne mentionne pas :

```
héretiques 15 % vs 0 %, I_stim=0, 300 pas :
    avec bruit  sigma_v=0.05 : max|Δv| = 1.641e-01
    SANS bruit  sigma_v=0.0  : max|Δv| = 0.000e+00      ← no-op exact
```

Sans bruit, l'écart est **rigoureusement nul** : `I_eff[heretic_mask] *= -1.0`
(`dynamics.py:299`) est sans effet quand `I_eff = 0`. L'écart de 0,164 observé avec bruit **n'est
pas physique** : le placement des hérétiques consomme 15 tirages `self.rng.randint`
(`dynamics.py:145`), ce qui **décale le flux du RNG** et donc toute la séquence de bruit `eta`.

C'est plus grave que « vacuous » :

> **Toute expérience A/B qui fait varier `heretic_ratio` compare deux réalisations de bruit
> différentes, pas deux physiques différentes.** L'effet mesuré est confondu avec un changement
> de graine effective.

C'est un problème de **plan d'expérience**, pas de modélisation, et il contamine potentiellement
tout balayage de `heretic_ratio` du projet.

En parallèle, `REPRODUCE_IN_5_MINUTES.md` — la porte d'entrée mise en avant en gros dans le
README — affirme toujours : « heretic nodes act as causal structural walls… shatters the
dead-zone consensus ». Si la démo tourne à `I_stim=0`, cette phrase décrit un mécanisme
mathématiquement inactif.

**Correctif**
1. **Découpler les RNG** — le plus urgent, 3 lignes :
   ```python
   # dynamics.py, _initialize_params
   self.rng_topo = np.random.RandomState(seed + 10_000)   # placement hérétiques, rewiring
   self.rng      = np.random.RandomState(seed)            # bruit uniquement
   ```
   Le bruit devient invariant au `heretic_ratio` ; les A/B redeviennent interprétables.
2. Rejouer les balayages `heretic_ratio` publiés avec les RNG découplés, et vérifier lesquels
   survivent.
3. Aligner `REPRODUCE_IN_5_MINUTES.md` sur `limitations.md`, ou faire tourner la démo à
   `I_stim ≠ 0` où le mécanisme existe réellement.

---

## B4 — ⚙️ `solve_rk45()` intègre un modèle différent de `step()`

`dynamics.py:521-568`. Cette méthode est présentée comme un intégrateur de contrôle. Elle
diverge de `step()` sur quatre points, dont un franchement faux :

| # | Ligne | Problème |
|---|---|---|
| 1 | `dynamics.py:540` | `laplacian_v = adj_matrix @ v - v`. Or `Mem4Network.step` calcule `-(L @ v) = A@v − deg·v` (`topology.py:315`). Sur un lattice degré 4, **ce n'est pas le même opérateur**. |
| 2 | `dynamics.py:548` | `self.rng.normal(...)` est appelé **dans la fonction dérivée**, réévaluée un nombre non déterministe de fois par `solve_ivp` (pas rejetés inclus), et **sans mise à l'échelle en √dt**. Le bruit n'est ni un bruit d'Itô ni reproductible. |
| 3 | `dynamics.py:552` | `(u > 0.5)` en dur : ignore l'hystérésis, pourtant `enabled=True` par défaut. |
| 4 | — | ART, watchdog, couplage non-local, compartiments, plasticité métacognitive, hérétiques dynamiques, doute complexe : **aucun** n'existe dans ce chemin. |

Vérification de l'écart d'opérateur (lattice 4×4, `v = 0..15`) :

```
max |(A@v − deg·v) − (A@v − v)| = 45.000
```

Le risque : utiliser RK45 pour « valider » les résultats Euler donnerait une comparaison entre
deux modèles distincts, et un désaccord serait attribué à l'intégrateur.

**Correctif**
- Corriger la ligne 540 en `adj_matrix @ v - adj_matrix.sum(axis=1) * v`.
- Sortir le bruit de la fonction dérivée : imposer `sigma_v = 0` (erreur, pas warning) et
  documenter RK45 comme **strictement déterministe**.
- Ajouter en tête de la docstring : « ne supporte aucune extension V5/V6 ». Idéalement,
  factoriser un `_drift(v, w, u, lap)` unique appelé par les deux chemins — c'est la seule
  façon durable d'empêcher la re-divergence.

---

## B5 — ⚙️ CI cassée, alors que le badge « Tests » est affiché dans le README

`.github/workflows/test.yml` installe `requirements.txt` puis `pytest`. Or `requirements.txt`
**ne contient pas PyYAML**, tandis que `dynamics.py:3` fait `import yaml` au niveau module.

```
$ python3 -m venv v && ./v/bin/pip install -r requirements.txt
$ ./v/bin/python -c "import yaml"
ModuleNotFoundError: No module named 'yaml'
```

Aucun test ne peut donc s'importer en CI. Le badge du README pointe vers un workflow qui, s'il
tourne, échoue à la collecte.

Divergences associées :

| | `requirements.txt` | `pyproject.toml` |
|---|---|---|
| pyyaml | **absent** (requis par le code) | présent |
| pandas, tqdm | présents | **absents** |
| numpy | ≥1.24 | ≥1.20 |
| scipy | ≥1.10 | ≥1.7 |

Le workflow n'installe jamais le paquet (`pip install -e .`) et teste Python 3.9-3.11, alors que
votre `.venv` locale est en 3.13 — la version que vous utilisez réellement n'est pas testée.

**Correctif**
- Supprimer `requirements.txt` et faire de `pyproject.toml` la source unique
  (`pip install -e ".[dev]"`), ou a minima y ajouter `pyyaml`.
- Workflow : `pip install -e .` + `pip install pytest`, matrice `["3.10","3.11","3.12","3.13"]`,
  `actions/checkout@v4` / `setup-python@v5`.

---

## B6 — La reproductibilité annoncée passe par `experiments/scratch/`, qui est gitignoré

```
experiments/scratch/  → 178 fichiers sur disque, 0 versionné
figures/scratch/      → 294 fichiers sur disque, 0 versionné
```

`README.md` référence `experiments/scratch/…` comme producteur pour, entre autres :
`p2_tau_u_bifurcation`, `p2_finite_size_scaling`, `p2_delta_sweep`,
`p2_doubt_community_detection`, `spice_mismatch_50seeds`, `fiedler_phase_diagram`,
`run_heroic_1600`. **5 des 8 figures du tableau « Figures » du README** ont leur producteur dans
`scratch/`.

`PROJECT_STATUS.md` a identifié le problème le 31/07 (« 155 scripts de `scratch/` cités dans la
doc versionnée, dont 19 engageant la reproductibilité et 7 portant un claim ») et a versionné
**7** scripts. Il en reste 178 dans `scratch/`, et le README pointe toujours vers eux.

Deux figures annoncées n'existent même pas :

```
figures/spice_art_kirchhoff.png   disque:N  git:N
figures/p2_art_benchmark.png      disque:N  git:N
```

**Correctif**
- Passer les scripts producteurs de `scratch/` vers `experiments/` et les versionner ; ne garder
  dans `scratch/` que ce qu'aucun document versionné ne cite.
- Ajouter au Guardian un contrôle N5 : *« tout chemin `experiments/…` ou `figures/…` cité dans un
  fichier versionné doit être versionné et exister »*. Les 10 références mortes du README
  détectées le 31/07 seraient tombées automatiquement.

---

## B7 — `RESULTS_INDEX.json` présente comme « résultat central du preprint » un cadrage que vous avez réfuté

`RESULTS_INDEX.json`, EXP_001 :

> `"key_result": "λ₂_crit = 2.31 (midpoint séparation complète). Accuracy 100% sur n=36."`
> `"significance": "La topologie seule détermine la capacité cognitive — résultat central du preprint."`

`PROJECT_STATUS.md` §0 :

> « l'ancien cadrage « transition spectrale » (λ₂_crit) a été **réfuté puis remplacé** par un
> cadrage degré de couplage / champ moyen »

Deux fichiers versionnés à la racine, lus par des tiers, disent l'inverse l'un de l'autre sur le
résultat principal. `RESULTS_INDEX.json` porte par ailleurs `"version": "V4.0.0"` et
`"date": "2026-05-02"` — figé depuis trois mois.

**Correctif :** régénérer `RESULTS_INDEX.json` depuis `docs/CLAIMS_REGISTER.md` (ou le supprimer
si le registre le remplace) et l'ajouter au périmètre du Tex Guardian.

---

## B8 — Métadonnées de citation incohérentes : DOI, version et titre divergents

| Source | Version | DOI | Titre |
|---|---|---|---|
| `README.md` (bibtex + badge) | V6 | `10.5281/zenodo.19700749` | « Mem4ristor V6: Spatiotemporal Chaos… » |
| `CITATION.cff` | **4.0.0** | **`10.5281/zenodo.18620596`** | « Mem4ristor **V4**: … » |
| `RESULTS_INDEX.json` | V4.0.0 | `10.5281/zenodo.18620596` | — |
| `VERSION` / `pyproject.toml` | V6.0.0 / 6.0.0 | — | — |
| `src/mem4ristor/__init__.py:54` | **`__version__ = "4.0.0"`** | — | — |
| `src/mem4ristor/config.py:78` | *« v3.1.0 configuration »* | — | — |
| `src/mem4ristor/config.yaml:1` | *« V4.0.0 »* | — | — |
| `dynamics.py:13` | *« v3.0.0 »* | — | — |

`CITATION.cff` est le fichier que **GitHub et Zenodo lisent** pour le bouton « Cite this
repository ». Aujourd'hui, toute citation automatique renvoie au **mauvais enregistrement Zenodo
et au mauvais titre**. Pour une soumission arXiv, c'est le premier truc qu'un relecteur clique.

**Correctif :** une source unique. `pyproject.toml` porte la version ;
`__init__.py` fait `__version__ = importlib.metadata.version("mem4ristor")` ; `CITATION.cff`,
`VERSION`, `README` et `config.yaml` sont alignés à la main **une fois** et vérifiés par le
Guardian à chaque commit.

---

# 🟠 IMPORTANT

## I1 — ⚙️ Modifier `cfg['coupling']['D']` après construction n'a aucun effet

`dynamics.py:151` fige `self.D_eff = D / sqrt(N)` dans `_initialize_params`, appelé une seule
fois à la construction. `Mem4Network.__init__` (`topology.py:55-58`) instancie le modèle **sans
possibilité d'injecter une config**. La seule voie d'accès à D est donc de muter
`net.model.cfg` après coup… ce qui ne fait rien :

```
net.model.cfg['coupling']['D'] = 10.0  (vs 0.15) → max|Δv| après 200 pas = 0.000e+00
```

Pire, le piège est **asymétrique** : avec `coupling_norm != 'uniform'`, `topology.step:317` relit
bien `D` depuis `cfg`… mais D se simplifie dans `scale_factors`, donc le seul canal réel reste
`D_eff`, figé. Un balayage de D écrit de cette façon produit une courbe plate qu'on
interprétera comme « le modèle est insensible à D ».

**Correctif**
- Faire de `D_eff` une `@property` calculée à la volée depuis `cfg`.
- Ajouter `config: Optional[dict] = None` à `Mem4Network.__init__` et le transmettre à
  `Mem4ristorV3(config=...)`.
- Test de non-régression : `D=0.3` doit produire une trajectoire différente de `D=0.15`.

---

## I2 — ⚙️ Deux sources de configuration divergentes, selon qu'on passe un dict ou non

`dynamics.py:38-47` : si `config` est fourni, **`config.yaml` n'est jamais lu**. Or les deux
sources ne contiennent pas les mêmes sections.

```
Mem4ristorV3()                  → 11 sections
Mem4ristorV3(config={'dynamics': {...}})  →  7 sections
perdues en silence : compartments, metacognitive, nonlocal_coupling, topological_regulation
```

Autrement dit : passer un dict pour changer `dt` **désactive silencieusement ART, les
compartiments, la plasticité métacognitive et le couplage non-local**. Aucun warning.

Même piège via `config.py` :

```
config.yaml sections = [compartments, coupling, doubt, dynamics, metacognitive,
                        noise, nonlocal_coupling, topological_regulation]
Mem4Config.from_yaml() garde = [coupling, doubt, dynamics, noise]
PERDUES = [compartments, metacognitive, nonlocal_coupling, topological_regulation]
```

`Mem4Config` se documente pourtant comme « **Complete** Mem4ristor configuration » et « Fully
backward-compatible ». Elle ne peut exprimer que la moitié du modèle.

**Correctif**
- Toujours charger `config.yaml` comme base, **puis** fusionner le dict utilisateur par-dessus.
- Aligner `default_cfg` (code) et `config.yaml` sur les mêmes clés, ou supprimer l'un des deux.
- Compléter `Mem4Config` avec les 7 sections manquantes, et lever un warning sur toute clé
  inconnue au lieu de la jeter.

---

## I3 — ⚙️ Le placement « uniforme » des hérétiques laisse un bord du réseau systématiquement vide

`dynamics.py:139-147` : `step = max(int(1.0/hr), 1)` puis arrêt dès que
`len(heretic_ids) == int(N*hr)`.

```
N=100, hr=0.15 → 15 hérétiques, indice max = 88/99   (nœuds 89-99 : jamais hérétiques)
N=400, hr=0.15 → 60 hérétiques, indice max = 356/399 (nœuds 357-399 : jamais hérétiques)
```

`int(1/0.15) = 6` produit ⌈N/6⌉ = 17 blocs pour 15 hérétiques : les 2 derniers blocs sont
toujours ignorés. **~11 % du réseau est structurellement exclu**, toujours du même côté. Sur un
lattice 2D indexé en ligne, cela veut dire les **dernières rangées**, ce qui est exactement le
type de biais spatial qui fabrique un faux motif de chimère.

**Correctif** — découper `[0, N)` en exactement `n_her` blocs et tirer un nœud dans chacun :

```python
n_her = int(self.N * hr)
edges = np.linspace(0, self.N, n_her + 1).astype(int)   # n_her blocs couvrant TOUT [0, N)
heretic_ids = [self.rng.randint(lo, hi)
               for lo, hi in zip(edges[:-1], edges[1:]) if hi > lo]
```

Ajouter un test : `heretic_mask` doit contenir au moins un `True` dans chaque quartile d'indices.

---

## I4 — ⚙️ Un NaN traverse `step()` sans déclencher aucune alarme

Politique contradictoire dans `dynamics.py` :

- lignes 193-195 : NaN/Inf sur `v`/`w`/`u` sont **réparés en silence** en début de pas ;
- lignes 350-351 : `|v| > 100` lève un `OverflowError` bruyant (« le filet de sécurité est retiré »).

Mais `np.abs(nan) > 100` vaut `False`. Donc un NaN **échappe au garde-fou**, franchit la fin du
pas, et se fait remettre à 0 au pas suivant :

```
v[0] = NaN → step() → aucune erreur → v[0] = -0.030
```

La simulation continue depuis un état corrompu, et rien dans la trajectoire ne le signale. Le
commentaire ligne 191 renvoie à `health_check()` — mais cette méthode n'existe que sur
`Mem4Network` (`topology.py:347`), pas sur le modèle, et il faut penser à l'appeler.

**Correctif**
```python
if not np.all(np.isfinite(self.v)) or not np.all(np.isfinite(self.w)):
    raise FloatingPointError("État non fini détecté — la simulation est invalide.")
```
Une seule politique : soit on répare partout et on **compte** les réparations dans un compteur
exposé, soit on lève partout. Le mélange actuel est le pire des deux.

---

## I5 — La fenêtre de l'entropie continue est figée à [-3, 3] : l'explosion y est invisible

`metrics.py:9` : `calculate_continuous_entropy(v, bins=100, v_min=-3.0, v_max=3.0)`.
`np.histogram` **ignore silencieusement** les valeurs hors plage, et `total = np.sum(counts)`
renormalise sur le seul sous-ensemble restant.

```
H(v)                              = 4.2770
H(v augmenté de 2 nœuds à ±50)    = 4.2770   ← identique
```

Un réseau qui diverge peut donc afficher une entropie parfaitement saine.

Second point, méthodologique : avec `bins=100` et `N=100` nœuds, l'estimateur est
**massivement biaisé** — chaque nœud tombe dans son propre bin et H sature à log₂(N), pas à
l'entropie différentielle. La quantité mesurée est essentiellement « combien de bins sont
occupés ». Sans correction de biais (Miller-Madow) ni baseline sur données permutées, une
comparaison de H entre deux conditions de N différents n'a pas de sens.

**Correctif**
- Compter les valeurs hors fenêtre et lever/avertir si > 0.
- Documenter H_cont comme un **indice relatif à N et bins fixés**, jamais comparé entre N différents.
- Ajouter une baseline permutée à côté de chaque valeur d'entropie publiée.

---

## I6 — `_lz76_phrases` n'est pas l'algorithme LZ76

`metrics.py:54-74`. L'implémentation ajoute chaque nouvelle phrase à un **ensemble** et teste
l'appartenance à cet ensemble. C'est **LZ78**. LZ76 (Lempel-Ziv 1976) teste si le candidat est
une **sous-chaîne du préfixe déjà analysé** — ce n'est pas la même complexité, ni la même
constante de normalisation.

Or la normalisation appliquée ligne 113 (`c · log₂T / T`) est celle de **LZ76**. Le résultat
n'est donc calibré ni pour l'un ni pour l'autre : « ≈ 1 pour une séquence aléatoire » n'est pas
garanti.

Le nom circule dans les documents publiés (`RESULTS_INDEX.json` EXP_003 : « LZ76 par nœud,
r = −0.716, p = 1.29e-79 »).

**Correctif :** soit renommer en `lz78` partout (docs, figures, tableaux du preprint), soit
implémenter le vrai LZ76 et **rejouer** EXP_003. Dans les deux cas, vérifier empiriquement que la
valeur normalisée tend vers 1 sur une séquence i.i.d. — c'est un test de 5 lignes.

---

## I7 — Le module `benchmarks/engine.py` rendrait toute comparaison structurellement biaisée

`benchmarks/engine.py:7-13`, commentaire : *« Threshold mapping to align with Mem4ristor bins »*.
Les seuils ne s'alignent pas :

| | seuils | états atteignables |
|---|---|---|
| `BenchmarkModel.get_states` | ±0.5 | **3** (`{1,3,5}` — jamais 2 ni 4) |
| `metrics.get_cognitive_states` | ±0.4 / ±1.2 | **5** |

Plafond d'entropie : log₂3 = 1,58 bit pour les baselines, log₂5 = 2,32 bit pour Mem4ristor.
**Toute comparaison « Mem4ristor a plus d'entropie que Kuramoto/Voter/Consensus » serait gagnée
d'avance par construction.** S'y ajoute que Kuramoto sort `v = cos θ ∈ [-1,1]` et Voter
`v ∈ {−1,+1}`, bornés, alors que Mem4ristor ne l'est pas — et que leurs `dt` valent 0,1 contre
0,05 (comparaison « par pas » = temps physiques différents).

Vérification :

```
Kuramoto  get_states={3}  | get_cognitive_states={3,4}
Voter     get_states={5}  | get_cognitive_states={4}
Consensus get_states={3}  | get_cognitive_states={3}
Mem4ristor                | get_cognitive_states={1,2}
```

Bonne nouvelle : ce module est **du code mort**. Aucun script courant ne l'importe (seuls
`archives/old_reproduction/` le font, avec leur propre copie). Aucun résultat publié n'en dépend.

**Correctif :** supprimer `src/mem4ristor/benchmarks/` du paquet distribué (il n'a même pas
d'`__init__.py`), ou le réparer avant tout usage — même discrétisation, même `dt`, même horizon
temporel physique. Le laisser tel quel, importable depuis le paquet officiel, c'est un piège
armé pour la prochaine session.

`VoterModel.step` (ligne 43) porte au passage un bug distinct : `p_flip = 0.5 + 0.1*I_stim`
module la *fréquence* de copie, pas sa *direction* — le commentaire « Stimulus bias » décrit un
biais qui n'existe pas. Et la copie se fait depuis un nœud **uniformément aléatoire**, pas depuis
un voisin, malgré le commentaire.

---

## I8 — `Mem4ristorV2 = Mem4ristorV3`

`dynamics.py:583`. Tout script historique qui importe `Mem4ristorV2` — y compris
`symbiosis.py:3` et les benchmarks archivés — exécute en réalité **V3 avec les extensions V4/V5**.
Un « test de non-régression V2 vs V3 » comparerait donc la classe à elle-même.

**Correctif :** émettre un `DeprecationWarning` à l'import de l'alias, ou le supprimer et
corriger les 3-4 appelants. Un alias silencieux entre deux versions d'un modèle scientifique est
exactement le genre de chose qui rend un résultat inexplicable trois mois plus tard.

---

## I9 — Terme de répulsion inter-compartiments non borné

`dynamics.py:275-280`, mode `'full'` :

```python
I_comp[mask_k] -= gamma * (v_mean_other - self.v[mask_k])
```

soit `+γ·(vᵢ − v̄_autres)` : une force **proportionnelle à l'écart et de même signe que lui**.
C'est une rétroaction positive : plus le groupe s'éloigne, plus il est poussé. Sans saturation,
la divergence est exponentielle et se termine sur l'`OverflowError` de la ligne 350.

Par ailleurs `mode` n'est jamais validé (ligne 275, `if mode == 'full'`) : une faute de frappe
dans `config.yaml` (`'Full'`, `'ful'`) retombe **en silence** sur `attraction`, et l'expérience
« mode full » mesure en réalité le mode attraction.

**Correctif :** borner la répulsion (`np.tanh` ou clip sur `I_comp`) et valider `mode ∈
{'attraction','full'}` dans `_validate_config` avec une `ValueError`.

---

## I10 — Défaut de `alpha_meta` incohérent avec la recommandation documentée

`dynamics.py:321` : `alpha_meta = meta.get('alpha_meta', 0.5)`.
`config.yaml` : `alpha_meta: -0.5` avec 8 lignes expliquant que **le signe négatif est la bonne
direction** (gain +0,79 bit vs +0,28).

Comme `config.yaml` n'est pas lu quand on passe un dict (cf. I2), le repli du code est
`+0.5` — le signe que vos propres mesures désignent comme le moins bon.

**Correctif :** `meta.get('alpha_meta', -0.5)`, et un test verrouillant le défaut.

---

## I11 — `health_check` décrit un mécanisme de clipping qui n'existe pas

`topology.py:387-392` :

> `'{n}/{N} nœuds ont |v| > 50 — explosion possible, clipping silencieux actif'`

Il n'y a **aucun clipping de `v`** dans `dynamics.py` : il y a un `OverflowError` à ±100
(ligne 350). Le message décrit un comportement d'une version antérieure. Un utilisateur qui le
lit conclura que ses valeurs ont été tronquées alors qu'elles ne l'ont pas été.

**Correctif :** reformuler en « approche du seuil de divergence (OverflowError à ±100) ».

---

# 🟡 COSMÉTIQUE

## C1 — La carte du dépôt dans le README est fausse sur 6 modules sur 13

| README dit | Réalité |
|---|---|
| `core.py` = « facade Mem4Network, expose `step()` » | 25 lignes de ré-export. `Mem4Network` est dans `topology.py`. |
| `topology.py` = « générateurs BA / lattice / ER » | Ils sont dans `graph_utils.py`. `topology.py` contient `Mem4Network`. |
| `graph_utils.py` = « helpers NetworkX, community detection, NMI » | Aucun import networkx, aucune détection de communauté, aucun NMI. |
| `inception.py` = « cold-start protocol » | `DreamVisualizer` (décodage par pseudo-inverse). Le cold start est un kwarg de `Mem4Network`. |
| `cortex.py` = « hierarchical coupling » | MLP auto-encodeur à une couche cachée. |
| `benchmarks/engine.py` = « harness throughput/reproductibilité » | Modèles Kuramoto / Voter / Consensus. |
| `dynamics.py` = « … sparse CSR backend » | Le backend sparse est dans `topology.py`. |
| section « Experimental modules — `experimental/` » | **Ce dossier n'existe pas.** Les fichiers sont dans `examples/`. |

## C2 — La section « Configuration » du README décrit des clés inexistantes

> « **ART (V6):** `art_enabled`, `art_threshold`, `art_window` »

Aucune de ces trois clés n'existe. Les vraies sont
`topological_regulation.{enabled, u_min, rigid_threshold, mode, alpha_art_soft, alpha_art_hard}`.
`art_window` n'apparaît nulle part dans le dépôt.

Idem : `coupling_norm` est listé comme réglable dans `config.yaml`, alors que c'est un argument
du constructeur `Mem4Network`.

## C3 — « ART » désigne deux choses différentes

- `README.md` (Key Scientific Features) : « **ART (Adaptive Reset Threshold)** — per-node reset
  threshold modulated by doubt history ».
- `config.yaml` + `dynamics.py:355` : « ART — **Autorégulation Topologique** — Kirchhoff passif ».

Ce ne sont pas le même mécanisme. Le second est celui qui est implémenté. Un relecteur qui lit
le README puis le code cherchera un seuil de reset qui n'existe pas.

## C4 — Chemins `docs/` faux dans le README

```
docs/preprint.tex        MANQUANT   → docs/papers/preprint/preprint.tex
docs/preprint.pdf        MANQUANT   → docs/papers/preprint/preprint.pdf
docs/paper_B/paper_B.tex MANQUANT   → docs/papers/paper_B/paper_B.tex
docs/paper_2/paper_2.tex MANQUANT   → docs/papers/paper_2/paper_2.tex
```

`limitations.md` apparaît par ailleurs deux fois dans deux tableaux différents du README.

## C5 — Garde de type du stimulus à la logique morte

`dynamics.py:186-189` :

```python
if isinstance(I_stimulus, (dict, set, list, tuple, str)):
    if not isinstance(I_stimulus, (list, tuple)):
        raise TypeError(...)
```

La condition interne annule `list` et `tuple` de la condition externe. Le bloc équivaut
exactement à `if isinstance(I_stimulus, (dict, set, str)): raise`. Trois types listés pour rien.

## C6 — L'optimisation annoncée dans `calculate_pairwise_synchrony` n'en est pas une

`metrics.py:131-152`. Le commentaire annonce « Subsample pairs for large N (avoid O(N²) cost) »
et plafonne à `MAX_PAIRS = 2000` — puis ligne 152 :

```python
corrs = np.einsum('ti,tj->ij', z, z)[i_idx, j_idx] / T
```

L'einsum matérialise la matrice **N×N complète** avant d'en extraire 2 000 éléments. Le coût
reste O(T·N²) en temps et O(N²) en mémoire. À N=10 000, c'est 800 Mo pour 2 000 corrélations.

**Correctif :** `corrs = np.einsum('ti,ti->i', z[:, i_idx], z[:, j_idx]) / T`.

## C7 — `print()` de débogage dans une fonction de métrique

`metrics.py:235`, au milieu de `calculate_transfer_entropy` :

```python
print(f"H_Yp={...}, H_Ycp={...}, H_YpXp={...}, H_3d={...}, TE={...}")
```

À supprimer ou passer en `logging.debug`.

Deux remarques dans la même fonction :
- `max(0.0, TE)` (ligne 236) **biaise l'estimateur vers le positif** : les estimations négatives
  (bruit d'échantillonnage) sont écrasées à 0, jamais compensées. Publier une TE sans baseline
  sur données permutées surestime systématiquement le transfert.
- `density=True` (lignes 213-231) ne renvoie des probabilités que parce que la largeur de bin
  vaut exactement 1,0 ici. Le code est juste **par coïncidence** ; il cassera silencieusement si
  la plage change. Utiliser `density=False` et diviser par le total.

## C8 — Optimisation de rewiring annulée au pas suivant

`topology.py:75-89` calcule un swap laplacien incrémental (« O(1) speedup » vanté dans
l'en-tête de `core.py`) puis pose `_weights_dirty = True` (ligne 223) — ce qui déclenche un
`_rebuild_laplacian()` **complet** au pas suivant (lignes 309-311). Le travail incrémental est
systématiquement jeté. `_update_laplacian_incremental` (lignes 71-73) est par ailleurs un `pass`
marqué OBSOLETE.

Dans la même boucle, `rows, cols = adj_lil.nonzero()` (ligne 183) est recalculé **à chaque nœud
éligible**, en O(E) — c'est le vrai coût, et il n'est pas optimisé.

## C9 — Hygiène du dépôt

| Constat | Détail |
|---|---|
| **11 Go** dans `experiments/spice/results/` | 375 `.cir` + 374 `.dat`. Correctement gitignorés, mais aucun script de purge. C'est 99 % du poids du dossier sur votre disque. |
| `.gitignore` vs suivi réel | `results/`, `archives/`, `sessions/` sont ignorés **mais 36 / 38 / 4 fichiers y sont déjà suivis**. `.gitignore` n'annule pas le suivi : tout **nouveau** résultat sera silencieusement absent des commits. État mi-dedans mi-dehors. |
| Arbre de travail sale | ~30 fichiers modifiés non commités, dont `PROJECT_STATUS.md`, `RESULTS_INDEX.json`, `CITATION.cff` et 6 `.tex`. |
| `.gitignore` redondant | Le bloc LaTeX (`*.aux`, `*.log`, `*.out`, `*.toc`, `*.fls`, `*.fdb_latexmk`) est présent **deux fois**. |
| `uv.lock` | 384 Ko présent sur disque, gitignoré, alors que `pyproject.toml` déclare setuptools. Outillage mixte non documenté. |
| Chemins absolus Windows versionnés | `experiments/spice_art_kirchhoff.py` contient `D:/ANTIGRAVITY/…` — il ne tourne sur aucune autre machine. 10 fichiers versionnés au total portent `D:/ANTIGRAVITY` ou `C:/Users/julch`. |
| `benchmarks/` sans `__init__.py` | Fonctionne via les paquets-espaces de noms, mais l'auto-découverte setuptools peut l'omettre du wheel. |
| `graph_utils.make_directed` non exporté | Absent de `__init__.py`/`__all__` alors que ses 3 voisines y sont. |
| `try/except ImportError` mort | `__init__.py:38-47` protège contre une absence de matplotlib que `viz.py` ne déclenche jamais (import paresseux via `_require_mpl()`). Vérifié : `from mem4ristor import *` fonctionne sans matplotlib. |
| Doublons de nom | `FOLDER_SUMMARY.md` ×3, `FINAL_SCIENTIFIC_REPORT_V23.md` ×2. |

---

# État de la suite de tests

**Résultat : ~130 tests, 100 % au vert** (2 `xfail` documentés dans `test_adversarial.py`).
La suite est sérieuse — `test_u_clamp_invariant.py`, `test_complex_doubt.py` et
`test_consolidation_watchdog.py` verrouillent de vrais invariants bit-à-bit, ce qui est rare et
mérite d'être dit.

Ce qui manque, et que les constats ci-dessus rendent nécessaire :

| Test à ajouter | Verrouille |
|---|---|
| `D=0.3` ≠ `D=0.15` sur 200 pas | I1 (paramètre figé) |
| `Mem4ristorV3(config={...})` conserve les 11 sections | I2 |
| `heretic_mask` a ≥1 vrai dans chaque quartile d'indices | I3 |
| `v[0]=NaN` → exception | I4 |
| `sigma_v=0` ⇒ trajectoire identique pour `hr=0` et `hr=0.15` | B3 (découplage RNG) |
| `solve_rk45` ≈ `step()` à `sigma_v=0`, `dt→0` | B4 |
| `lz_complexity(séquence i.i.d.)` ≈ 1 | I6 |
| `__version__` == `pyproject.version` == `CITATION.cff.version` | B8 |
| `import yaml` dans un env issu du seul `requirements.txt` | B5 |

---

# Plan d'action proposé

### Avant toute soumission arXiv
1. **B8** — aligner `CITATION.cff` sur V6 et le bon DOI *(15 min, impact maximal)*
2. **B5** — ajouter `pyyaml`, réparer la CI *(15 min)*
3. **B7** — régénérer ou supprimer `RESULTS_INDEX.json` *(30 min)*
4. **B1** — versionner `.brain/` dans `tools/guardian/` *(2 h)*
5. **B6** — sortir de `scratch/` les producteurs cités, contrôle N5 au Guardian *(3 h)*
6. **B2** — désambiguïser `H_cog` / `H_cont`, republier le protocole LIMIT-02 *(3 h)*

### Avant la prochaine campagne expérimentale
7. **B3** — découpler les RNG topologie/bruit, puis **rejouer les balayages `heretic_ratio`**
8. **I1 + I2** — config unifiée, `D_eff` en property
9. **I3** — corriger le placement des hérétiques
10. **I4** — politique NaN unique

### Dette technique
11. **B4** — factoriser un `_drift()` commun à `step()` et `solve_rk45()`
12. **I6** — trancher LZ76 vs LZ78 et rejouer EXP_003 si besoin
13. **I7** — supprimer `benchmarks/` du paquet
14. **C1-C4** — réécrire la carte du dépôt du README depuis le code réel

---

## Ce qui est solide, et qu'il ne faut pas casser en corrigeant

- `docs/limitations.md` : une table de vérité qui note ses propres claims « FALSE » et
  « VACUOUS ». C'est le contraire de ce qu'on trouve d'habitude, et ça vaut plus qu'un résultat.
- Les tests d'invariance bit-à-bit (`test_complex_doubt`, `test_consolidation_watchdog`,
  `test_u_clamp_invariant`) : la bonne façon de rendre une extension opt-in vérifiable.
- Le garde-fou de symétrie dans `get_spectral_gap` (`topology.py:232-243`), avec le rapport
  d'erreur de 809 % en commentaire : refuser de répondre plutôt que mentir en silence.
- `_validate_config` refusant les configs où `u_clamp` écraserait le watchdog
  (`dynamics.py:104-118`) : anticipation d'un mode de défaillance silencieux.
- La discipline des commentaires `@DOUBT` et des dates d'accord explicite dans le code.

Le projet n'a pas un problème de rigueur. Il a un problème de **propagation** : ce qui est su
en profondeur (`limitations.md`, `PROJECT_STATUS.md`) n'atteint pas la surface (README,
`CITATION.cff`, `RESULTS_INDEX.json`) — et c'est la surface que lit un relecteur arXiv.

---

*Rapport généré le 2026-08-02.*

**Scripts de vérification laissés à la racine** — tout constat marqué ⚙️ est rejouable :

```bash
python _audit_check.py    # B2 (chiffre du README), B3 (hérétiques), I1 (D figé), I3 (placement)
python _audit_check2.py   # B3 (artefact RNG), B4 (opérateur RK45), I5 (fenêtre entropie), I7 (benchmarks)
python _audit_check3.py   # I2 (config perdue), B4 (opérateur), I4 (NaN silencieux)
```

À supprimer une fois les constats confirmés de votre côté — ils ne font partie ni du paquet ni
des tests.

---
---

# 📝 Note du relecteur — 2026-08-02, même jour

> **Le texte ci-dessus n'a pas été modifié d'une ligne.** Cette note est ajoutée à la suite,
> selon la règle de la maison : on annote un rapport, on ne l'édite pas. Elle existe surtout
> parce que **le plan d'action ci-dessus contient un correctif qu'il ne faut pas appliquer**.

## Les scripts n'ont PAS été supprimés, et ils ont déménagé

Ils vivent désormais à côté de ce rapport, dans `docs/audits/2026-08-02/`. Leur résolution de
`src/` a été corrigée en conséquence — **déplacer un script sans corriger sa racine le casse**,
défaut déjà payé deux fois dans ce dépôt (29/07 et 31/07). Rejoués depuis leur nouvel
emplacement : **valeurs identiques** à celles du rapport.

```bash
python docs/audits/2026-08-02/_audit_check.py
python docs/audits/2026-08-02/_audit_check2.py
python docs/audits/2026-08-02/_audit_check3.py
```

Ils sont conservés, et non supprimés, parce qu'ici un constat sans script rejouable n'a pas de
statut. C'est la règle qui a produit ce dépôt ; elle vaut aussi pour ce qui l'accuse.

## 🔴 B8 — NE PAS appliquer le correctif proposé : il aggrave

Vérifié contre l'**API Zenodo** (et non par lecture) le 02/08. Le concept
`10.5281/zenodo.18620596` compte **8 versions** ; la plus récente est **V4.0.0 du 2026-05-02**
(DOI de version `…19986042`). **Ni V5 ni V6 ne sont déposées.**

- Le fichier faux était le **README** : son badge annonçait « V6 » avec `…19700749`, qui est la
  **v3.2.0 du 22 avril**, sous un autre titre.
- `CITATION.cff`, désigné ici comme le pire du lot, portait le **concept DOI** — la bonne
  pratique — et la **bonne version**. Seuls son titre et sa date étaient faux.
- « Aligner `CITATION.cff` sur V6 et le bon DOI *(15 min, impact maximal)* » aurait donc
  remplacé un identifiant correct par un DOI obsolète, et fabriqué une référence vers un dépôt
  **qui n'existe pas**.

Corrigé autrement au commit `42700eb`, avec `tests/test_version_consistency.py` en garde-fou.

## 🟠 Trois portées à corriger

- **B3** — l'artefact RNG est réel, mais **ne touche aucun résultat publié** :
  `experiments/ablation_coordination.py:108` construit *tous* les bras avec
  `heretic_ratio=0.15` et neutralise le masque après coup. Les tirages sont donc identiques
  entre FULL et NO_HERETIC, et `tab:ablations` est indemne. Le risque reste entier pour tout
  script futur qui ferait varier `hr` à la construction.
- **B4** — l'écart mesuré RK45/Euler (0,104) ne prouve rien : deux intégrateurs différents
  divergent de toute façon. Le constat tient sur l'écart **analytique** d'opérateur (45,0).
  À noter aussi : `dynamics.py:530` émet **déjà** un `RuntimeWarning` sur RK45 + bruit.
- **C9** — « ~30 fichiers modifiés non commités » était vrai à 14 h et **périmé le soir même** :
  le dépôt était synchro 0/0 après clôture.

## 🔴 Ce que l'audit a manqué, et qui est plus grave que ce qu'il liste

**I6 s'arrête à la nomenclature.** Le test de 5 lignes qu'il propose sans l'exécuter a été fait.
`calculate_temporal_lz_complexity` rend, à T=300, `n_bins=5` :

| séquence de référence | `C_LZ` |
|---|---|
| bruit blanc i.i.d. | **2,70** |
| marche aléatoire | 1,19 |
| **sinusoïde pure** | **2,00** |
| constante | 0,00 |

La métrique classe donc une **oscillation périodique** (2,00) comme presque aussi désordonnée
que du bruit blanc, et bien plus qu'une marche aléatoire — signature directe du parsing LZ78
(sur `'0101…'` : 34 phrases contre 3 pour le vrai LZ76). Sur un papier consacré à des réseaux
d'oscillateurs, c'est l'angle mort le plus coûteux possible.

Second effet : la référence **dépend de T** (marche aléatoire : 1,64 à T=100 · 1,25 à T=300 ·
**0,65 à T=2000**). Or `tab:lz_regime` tourne à T=2000 et pose le seuil absolu
*« structured regime : $C_{LZ} < 0.85$ »* — seuil qu'une **marche aléatoire franchit**.

Ce qui n'est **pas** remis en cause : les comparaisons entre conditions à T constant
(FULL 1,096 < FROZEN_U 1,602), donc la transition à m=6. Ce qui tombe : les **ancres verbales
absolues** publiées (`≈ 1 : random walk`, `< 0.85 : structured`) et la docstring qui annonce un
retour dans `(0, 1]`.

## 🟠 B7 est plus profond que décrit — mesuré sur les 19 entrées

Ce n'est pas « une entrée qui contredit `PROJECT_STATUS` » :

```
scripts cités NON versionnés  : 16/19
entrées ayant perdu un output : 13/19
```

Seules EXP_001, EXP_002 et EXP_010 ont encore leur producteur au chemin annoncé et versionné.
Un soupçon a par ailleurs été **levé** : EXP_007 (Transfer Entropy) n'est *pas* vacuous — son
script force `I_stimulus=0.5`, précaution explicite dans le code. En revanche la TE est
**absente de tous les papiers**, alors que l'entrée déclare `"paper": "preprint.tex"`.

---

*Vérifications rejouables ; aucun constat de cette note n'a été obtenu par lecture seule, sauf
mention contraire. — 🎩 Claude (Anthropic — L'Ingénieur, Opus 5), 2026-08-02*
