# PROJECT STATUS — Mem4ristor V6.0.0 (preprint reformulé, non soumis)
**Dernière mise à jour : 2026-07-26, soir (Claude Code/Opus 5 — le signal d'arrêt ne tire aucun bénéfice de la topologie : voir §0 et §3bis)**
**Auteur : Julien Chauvin (Barman / Orchestrateur) & Antigravity (Orchestrateur 2026)**
**Contexte : Café Virtuel — Laboratoire d'Émergence Cognitive**

> Ce fichier est le point d'entrée pour quiconque (humain ou IA) travaille sur ce projet.
> Lisez-le en premier. Pour l'historique complet des sessions et investigations :
> → **[PROJECT_HISTORY.md](PROJECT_HISTORY.md)**

---

## 0. ÉTAT ACTUEL EN UNE MINUTE (2026-07-26)

- **Version** : V6.0.0. Le preprint a été **reformulé** (titre/abstract/résultats) : l'ancien
  cadrage « transition spectrale » (λ₂_crit) a été **réfuté puis remplacé** par un cadrage
  **degré de couplage / champ moyen** (voir §3). Le phénomène (dead zone, anti-synchronisation)
  survit ; seule son explication causale a changé.
- **Résultat central actuel** : l'ablation **FROZEN_U vs FULL** (geler `u` détruit
  l'anti-synchronisation) — séparation complète, Cohen d ≈ 9.4 (BA m=3) / 4.7 (lattice),
  30 seeds. C'est le résultat le plus robuste et le moins attaquable du papier (corrélation
  de Pearson, indépendant du binning).
- **Guardian** : 14/14 claims vérifiées automatiquement à chaque commit
  (`.brain/claims_mapping.json` + `.brain/preprint_guardian.py`, hook pre-commit).
- **Git** : branche `main`, **synchronisée avec `origin/main`**. La publication a eu lieu
  vers le ~14 juillet 2026 : tout l'historique est sur `github.com/cafe-virtuel/Mem4ristor`.
  Le verrou de publication, le plus ancien du projet, est levé — il ne reste éventuellement
  qu'une soumission arXiv/HAL. Cœur (`dynamics.py`) stable depuis plusieurs semaines.
- **La niche du doute a été resserrée le 26/07** (expériences A/B, §3bis). La formule
  « composant d'orientation bon marché » (13/07) est **corrigée** : elle était vraie en
  nombre de pas et fausse en énergie. Ce qui tient, mesuré sur graines réservées : le doute
  **décide ~4.4× plus tôt**, y compris depuis l'intérieur de la fenêtre trompeuse, au prix
  de 10-20 points de justesse. Voir §3.
- **Resserrée une 5ᵉ fois le 26/07 au soir** (expériences B7, §3bis) : le signal d'arrêt
  exige de l'information **spatiale**, mais **ne tire aucun bénéfice de la topologie** —
  une simple dispersion globale `std(v)`, aveugle à qui est voisin de qui, fait aussi bien
  partout où on a regardé, y compris sur une tâche construite pour que la topologie porte
  de l'information. Sept mesures indépendantes, écart toujours ≤ 0, jamais en faveur du
  désaccord local. C'est le rétrécissement **le mieux tenu des cinq** : il a survécu à sa
  propre contre-épreuve et à son gate de réplication.
- ⚠️ **Ce fichier a déjà dérivé deux fois, et la deuxième a coûté cher.** Périmé du 12 juin
  au 9 juillet (5 semaines) ; puis du 9 au 26 juillet, période pendant laquelle il a
  continué d'annoncer « 51 commits non poussés / publication en attente » **alors que la
  publication était faite depuis le 14**. Un audit externe du dépôt (juillet) a repris cette
  phrase telle quelle et en a tiré « publication BLOQUÉE » : l'erreur ne venait pas de
  l'auditeur, **elle venait d'ici**. Ce fichier est lu par des tiers et par des IA :
  **le mettre à jour à chaque clôture de session**, même par 2-3 lignes dans §3bis.

---

## 1. QU'EST-CE QUE CE PROJET ?

Mem4ristor est une implémentation computationnelle de dynamiques FitzHugh-Nagumo étendues, conçue pour étudier les états critiques émergents dans des réseaux neuromorphiques. L'innovation centrale est le **Doute Constitutionnel (u)** : une variable dynamique qui module la polarité du couplage entre les neurones, créant une frustration adaptative qui empêche l'effondrement du consensus.

Le projet est né au sein du **Café Virtuel**, un laboratoire de collaboration entre un humain et plusieurs IA (Anthropic, OpenAI, xAI, Google, Mistral, DeepSeek). L'historique complet est dans le dépôt Café Virtuel : https://github.com/cafe-virtuel/

Publication : DOI 10.5281/zenodo.19986042 (V4.0.0 — dernière release Zenodo ; le code a
évolué depuis, la prochaine release Zenodo portera les valeurs V6.0.0 reformulées).
Preprint actuel (non soumis, non publié sur Zenodo) : `docs/papers/preprint/preprint.tex`
→ `preprint.pdf` (27 pages, Guardian **18/18**, Tex Guardian 11/11 ancres · 0 valeur morte · 12/12 sources).

---

## 2. ARCHITECTURE DU CODE

> **⚠️ Correction 09/07/2026** : cette section décrivait encore `core.py` comme le
> « moteur V3 canonique » avec 6 modes de coupling_norm. C'est FAUX depuis la
> refactorisation qui a séparé le moteur en 3 modules (`dynamics.py`/`topology.py`/
> `metrics.py`) — `core.py` n'est plus qu'une **façade de compatibilité de 25 lignes**
> qui ré-exporte ces modules (déjà noté comme désynchronisation dans l'ancien §10.4
> de ce fichier, jamais corrigé jusqu'ici). Table ci-dessous mise à jour sur la base
> de la structure réelle de `src/mem4ristor/`.

### Noyau stable (PRODUCTION-READY)

| Fichier | Rôle | État |
|:--------|:-----|:-----|
| `src/mem4ristor/dynamics.py` | **Le moteur réel** : `Mem4ristorV3` (FHN + doute `u` + plasticité + hystérésis + watchdog de consolidation opt-in) | STABLE |
| `src/mem4ristor/topology.py` | `Mem4Network` : graphe, couplage (laplacien, normalisation par degré), rewiring doute-piloté, **sparse CSR** auto-détecté pour N > 1000 | STABLE |
| `src/mem4ristor/metrics.py` | `calculate_cognitive_entropy` (H_cog, 5 bins — indicateur relatif legacy), `calculate_continuous_entropy` (H_cont, 100 bins — métrique primaire actuelle), synchronie, MI spatiale | STABLE |
| `src/mem4ristor/graph_utils.py` | Source unique de vérité pour `make_ba()`, `make_er()`, `make_lattice_adj()` (tous les scripts `experiments/p2_*` importent d'ici, pas de réimplémentation locale) | STABLE |
| `src/mem4ristor/core.py` | **Façade de compatibilité (25 lignes)**, ré-exporte `dynamics.py`/`topology.py`/`metrics.py` pour les scripts historiques qui font `from mem4ristor.core import ...` | STABLE (facade) |
| `src/mem4ristor/config.py` + `config.yaml` | Paramètres par défaut | STABLE |
| `src/mem4ristor/__init__.py` | Exporte Mem4ristorV3, Mem4ristorV2 (alias), Mem4Network | STABLE |
| `src/mem4ristor/symbiosis.py` | CreativeProjector (Phase 4) + SymbioticSwarm | STABLE |
| `src/mem4ristor/cortex.py` | LearnableCortex (MLP autoencoder pour consolidation mémoire) | STABLE |
| `src/mem4ristor/sensory.py` | SensoryFrontend (convolution + projection pour entrées visuelles) | STABLE (lent, voir §5) |
| `src/mem4ristor/sonification.py` | Conversion des trajectoires en son (exploratoire, 1er mai) | STABLE |
| `src/mem4ristor/viz.py` | Visualisation : entropy trace, doubt map, phase portrait, state distribution, dashboard | STABLE |

**Le mécanisme du « spectral » n'existe plus en tant que cadrage causal** : le
`coupling_norm='spectral'` (eigenvector centrality) reste implémenté dans `topology.py`
mais l'ancien claim « λ₂_crit=2.31 explique la dead zone » a été **réfuté** (mandat du
1er juillet 2026, voir §3bis) — le mécanisme réel est le **degré de couplage / champ
moyen**. Ne pas présenter `spectral` comme le mode « validé par la théorie » dans un
nouveau document — c'est l'inverse qui a été montré.

### Modules expérimentaux (NON PRODUCTION)

| Fichier | Rôle | État |
|:--------|:-----|:-----|
| `experimental/mem4ristor_king.py` | "Philosopher King" : loi martiale, métacognition | EXPERIMENTAL |

### Dossiers hardware (exploratoires, aucun claim publié n'en dépend)

| Dossier | Contenu | État |
|:--------|:--------|:-----|
| `docs/hardware/PHOTONIC_PATHWAY.md` | Voie GST/photonique — quatuor d'imperfections physiques testé (bruit quantique, non-linéarité, inertie, fabrication), mapping `u`↔transmittance | Le plus avancé des 3 |
| `docs/hardware/SPINTRONIC_PATHWAY.md` | Voie STNO — 3 POCs en escalade (Kuramoto → Slavin-Tiberkevich → macrospin LLGS complet), mécanisme du doute testé et robuste | Ajouté 09/07/2026 |
| `docs/hardware/ELECTRICAL_PATHWAY.md` | RRAM/VTEAM (poids de couplage) + neuristor Mott NbO₂ (oscillateur) | Ajouté 09/07/2026 |
| `docs/hardware/B3_ENERGY_COMPARISON.md` | Comparaison d'énergie/vitesse entre les 3 familles + référence CMOS (Loihi/TrueNorth) | Ajouté 09/07/2026 |

---

## 3. ÉTAT DES CLAIMS SCIENTIFIQUES

**Sources de vérité actuelles** (pas `docs/limitations.md`, périmé) :
- **`.brain/claims_mapping.json`** + **`.brain/preprint_guardian.py`** — vérification
  **automatisée** à chaque commit (hook pre-commit), **18 claims**, **18/18 OK** au
  31 juillet 2026.
- **`.brain/tex_guardian.py`** (créé le 30/07) — second garde-fou au même hook :
  il compare le **texte publié** à ses **données** (ancrages, registre des valeurs mortes,
  audit des sources citées). **13/13 ancres, 0 valeur morte, 12/12 sources** au 31/07.
- **`docs/CLAIMS_REGISTER.md`** — registre narratif détaillé (valeur, script, seeds,
  statut) pour chaque claim + claims secondaires/exploratoires (S01-S09).

> ✅ **Collisions d'étiquettes RÉSOLUES le 31/07/2026** (accord explicite de Julien).
> Les deux fichiers ci-dessus utilisaient la même étiquette pour des claims différents,
> et c'était le cas **deux fois**. Aucun chiffre vérifié n'a changé : seules les étiquettes.
>
> - **`C13`** (connue depuis le 09/07) — `claims_mapping.json` : « Cohen d ablation
>   FROZEN_U vs FULL, BA m=3, 30 seeds » ; `CLAIMS_REGISTER.md` : « LZ76 regime
>   classification (adaptive D(u)) », un claim plus ancien. → le claim du Guardian est
>   renommé **`C18`**, étiquette libre des deux côtés (le registre va jusqu'à C21).
>   ⚠️ **Le diagnostic du 30/07 était faux sur un point, et il faut le savoir** : N2 du
>   `tex_guardian` signalait ce claim comme « ne protégeant rien de publié ». Il protège
>   en réalité le **résultat central du papier** — « Cohen's *d* ≈ 9 » est publié à
>   **trois** endroits (contributions, `sec:ablation`, Conclusion). N2 ne le voit pas
>   parce que le texte l'écrit en **entier** et que le détecteur ignore les entiers
>   **par conception** (sinon : bruit des tailles de réseau, numéros de section, comptes
>   de graines). Le signalement de `C18` est donc **structurel et permanent, pas une
>   dette** — et le retirer aurait désarmé le garde-fou du résultat central.
> - **`C07`** (trouvée le 30/07) — le Guardian appelait `C07` ce que le registre et
>   `AUDIT_LOG.md:1284` (12/06) appellent **`C07b`** (lattice 10×10, alors publié à
>   0.0197 ± 0.0142 `[valeur morte]` — voir la correction de protocole ci-dessous).
>   → clé renommée **`C07b`**. Le vrai `C07` du registre (BA m=3 FORCED = 0.031 `[valeur morte]`)
>   porte un chiffre **périmé** depuis la régénération du 29/07 : il est marqué **remplacé par `C14`**
>   (0.002312, CSV versionné, reproductible bit à bit). ⚠️ Dette laissée ouverte :
>   `figures/p2_table1_sync.csv` (source de C07b) n'est régénérable par **aucun script du
>   dépôt** — son producteur vit dans le dépôt pilote et vise un répertoire disparu.
> - **`C01`** — corrigé le même jour, défaut différent et réel : le claim vérifiait
>   `p2_delta_sweep.csv` (δ=0 → 4.157), **valeur publiée nulle part**. Repointé sur
>   `p2_table1_lattice.csv` (lattice_10x10 = 4.086280), soit le **4.09 ± 0.19** publié
>   dans `tab:scaling` et `tab:benchmarks`, et ancré aux deux. Le registre portait encore
>   `4.06 ± 0.08`, valeur d'avant la régénération du 08/07 : **même défaut que C02 et C03,
>   corrigés le 30/07 — C01 avait été sauté.** La correction avait été appliquée aux lignes
>   trouvées, pas à l'affirmation ; c'est le motif du « ~90-fold », une fois de plus.
>
> 🔴 **Et une correction de VALEUR PUBLIÉE, trouvée le même jour en payant la dette du CSV**
> (décision de Julien). `tab:benchmarks` publiait une synchronie de `0.0197` `[valeur morte]`
> produite **sans cold start**, alors que le papier revendique le Cold Start Protocol et affirmait en **trois
> endroits** (L182, L195, L515) que les deux colonnes de cette ligne viennent d'un seul
> protocole. Le CSV n'était pas faux : il se rejouait **au bit près**. C'est le **protocole**
> qui ne correspondait pas à ce que le texte annonçait. Mesuré : cold start → **0.036 ± 0.023**,
> soit **+0.0162, au-delà de l'écart-type publié**. `verify_table1_preprint.py` écrit désormais
> les **deux** CSV dans le **même run** — la phrase publiée est devenue vraie au lieu d'être
> corrigée — et la dette de régénération est fermée du même geste.
> ⚠️ **La correction affaiblit le chiffre** (synchronie plus haute), comme Floquet le 30/07 :
> l'écart aux baselines passe d'un facteur ~51 à ~28. Les baselines étant à ≈ 1.0, la
> conclusion du tableau est inchangée.
> 💡 **Ce que ça apprend sur nos garde-fous, et il faut le garder** : l'ancre du `tex_guardian`
> **tenait parfaitement** avant cette correction — `0.0197` était bien la valeur exacte de son
> CSV. *Une ancre qui tient prouve qu'un nombre publié vient de son CSV ; elle ne prouve rien
> sur le protocole qui a produit ce CSV.* C'est un étage au-dessus de ce que l'outil sait voir.

### Résultats scientifiques actuels (résumé, voir CLAIMS_REGISTER.md pour le détail)

- **Résultat central : ablation FROZEN_U vs FULL** — geler `u` détruit
  l'anti-synchronisation. Séparation complète, **Cohen d = 9.4 (BA m=3) / 4.7 (lattice)**,
  30 seeds, IC bootstrap (**C18** dans `claims_mapping.json`, ex-C13). Mesure indépendante du binning
  (corrélation de Pearson) — c'est le résultat le moins attaquable du papier.
- **Le mécanisme causal n'est PAS spectral** : λ₂_crit≈2.31 a été **réfuté** comme
  mécanisme (mandat du 1er juillet 2026) — c'est un artefact de corrélation avec le
  **degré de couplage / champ moyen** (k_harm≈6), pas la connectivité algébrique. Preprint
  reformulé en conséquence (titre, abstract, §4.5-4.7). Le phénomène (dead zone) survit,
  seule la cause a changé.
- **Table 1 (diversité H_cont) + finite-size scaling** : robuste à 30 seeds, sature à
  ~4.38 bits (jamais d'effondrement de taille finie), std se resserre avec N.
- **Comparaison SOTA** : bat Kuramoto/Voter/Consensus (faible barre) ; perd contre un
  **Echo State Network réel sur NARMA10** (~5.5× moins bon — Mem4ristor n'est **pas** une
  mémoire, c'est un explorateur/anti-synchroniseur — positionnement assumé).
- **Sur sa propre niche (décision sous convergence-piège, horizon inconnu), face aux
  adversaires cités par l'audit externe** (26/07, graines d'entraînement et de mesure
  disjointes) : le **recuit simulé est battu** (0.45 vs 0.90) et l'**exploration purement
  aléatoire échoue** (0.00 sur le flux brut). Mais un **filtre à oubli exponentiel** — un
  circuit RC, un paramètre, aucune exploration — atteint 1.00 contre 0.90, en payant ~4×
  plus de pas. Ce qui bat le doute n'est donc pas une classe d'exploration, c'est un
  passe-bas.
- **Ce que le doute apporte, précisément** : il ne lit pas mieux, il sait **quand** lire.
  À un instant fixe dans la fenêtre trompeuse le réseau est à 0.40 de justesse ; à
  l'instant que son signal d'arrêt choisit, 0.90-1.00 (contre 0.46-0.57 pour des instants
  de même distribution tirés sans lien au run). L'arrêt vaut **+0.49 à +0.61**.
  ⚠️ Nuance à ne pas omettre en citant : ce signal d'arrêt exige de l'information
  **spatiale**, mais **pas la topologie** — un simple `std(v)`, aveugle à qui est voisin de
  qui, égale le désaccord local `mean(|L·v|)` du modèle (−0.07, IC [−0.17, +0.00]).
  *Correction du 26/07 au soir : ce paragraphe écrivait « égale, voire dépasse ». C'était*
  *plus fort que les chiffres — l'intervalle touche zéro. Ce qui est établi, c'est que*
  *`std(v)` n'est pas moins bon, pas que le désaccord local soit moins bon.*
- **Le signal d'arrêt ne tire AUCUN bénéfice de la topologie** (26/07 soir, expériences
  B7 ; détail dans `docs/FUTURE_WORK.md` §E5). Trois faits, tous sur graines réservées :
  (1) sur la tâche d'origine les deux signaux sont **une seule lecture** (r = 0.981) parce
  qu'elle n'a **aucune structure spatiale** (Moran's I du stimulus = −0.010) et que le
  réseau n'en fabrique pas (Moran's I de `v` ≤ 0.022) ;
  (2) construire une variante où la structure existe bel et bien (Moran's I +0.382, portée
  par le champ `v`) **ne change rien** — interaction +0.02, IC [−0.08, +0.12] ;
  (3) sur **sept mesures indépendantes** (deux niveaux de difficulté, deux mondes, groupes
  de graines disjoints, seuils réglés séparément), l'écart est **toujours ≤ 0 et jamais une
  seule fois** en faveur du désaccord local — l'amplitude est instable, le signe ne l'est pas.
  ⚠️ Ce qu'il ne faut **pas** citer : « et lire la topologie coûte ». Un effet net
  (−0.30, IC [−0.45, −0.17]) est apparu sur la tâche durcie puis **n'a pas répliqué** sur
  deux groupes de graines jamais touchées (0/2). Il est consigné comme mort.
- **Coût réel** : à substrat égal, M4R demande ~15× plus d'énergie qu'un RC équivalent, et
  6.7× à 20× plus d'opérations. Le « bon marché » du projet était un argument sur le
  **substrat** (dont tout dispositif analogique bénéficierait), pas sur l'architecture.
- **H_cog (5 bins) est un indicateur legacy relatif**, pas une métrique primaire — les
  valeurs absolues ne doivent pas être citées (voir A5, reformulation du 08/07).
- Environ 30 claims antérieurs (avril-mai 2026, incluant tous les items LIMIT-01 à
  LIMIT-05 et [1] à [20]) sont **archivés avec leur détail complet** dans
  `PROJECT_HISTORY.md` §13 — retirés d'ici pour lisibilité, pas supprimés.

---

## 3bis. SESSIONS RÉCENTES (12 juin → 26 juillet 2026, condensé)

> Résumé volontairement condensé — chaque ligne pointe vers l'entrée complète dans
> `ARCHIVES_INDEX.md` (racine `D:\ANTIGRAVITY`) et/ou `.brain/claude_contexts/MEM4RISTOR.md`
> (contexte privé de travail) pour le détail narratif complet. Ordre chronologique.

- **12/06 (soir)** — Triage de l'audit « regard neuf » d'Hermès : aucune des 4 découvertes
  ne touchait un chiffre publié, décision de Julien de laisser tel quel.
- **12/06** — **AUDIT-024, découverte majeure** : le commit `818cf67` (1er mai) avait changé
  le scaling du bruit (Euler-Maruyama ×4.47) sans que personne ne s'en aperçoive — les CSV
  pré-mai n'étaient plus reproductibles avec le code public. Option A appliquée le jour même
  (CSV régénérés, preprint corrigé, tous les effets qualitatifs survivent et s'amplifient).
  Puis quatuor d'imperfections physiques photonique complet (bruit quantique, non-linéarité,
  inertie, fabrication) — aucune ne détruit les régimes aux tolérances industrielles.
- **01/07** — **Mandat non-livrable : λ₂_crit=2.31 RÉFUTÉ.** La dead zone n'est pas causée
  par la connectivité algébrique mais par un effet de champ moyen gouverné par le degré
  harmonique (k_harm≈6). Décision de Julien : reformuler le preprint (fait le 06/07).
- **07/07 (matin)** — Audit externe simulé + **preprint reformulé** (spectral → degré/champ
  moyen). 3 POCs caractérisent le doute comme explorateur discipliné (pas générateur de
  diversité brute). Watchdog de consolidation ajouté au cœur (opt-in, résout le verrouillage
  en mode FOU).
- **07/07 (soir)** — Watchdog natif validé. Le doute comme allocateur de compute :
  **conditionnel** — perd contre la convergence triviale sur tâche loyale (B1c), **gagne**
  sur tâche trompeuse où converger tôt = se tromper (B1d, +0.58).
- **08/07 (matin)** — **Volet A clos** (A2-A5) : FROZEN_U remonté en résultat central du
  preprint ; régression de régime sur 70 vraies simulations (le degré prédit le régime,
  pas λ₂) ; cold-start corrigé (diversité robuste à l'init) ; H_cog rétrogradé (aucune
  métrique fonctionnelle continue ne le remplace en endogène).
- **08/07 (soir)** — **Volet B** : B1 consolidé (30 seeds × 3 topos) ; ablation centrale
  refaite avec IC (Cohen d 9.4/4.7, résultat central actuel) ; Table 1 + FSS (30 seeds ×
  7 tailles, sature ~4.38 bits) ; comparaison **Echo State Network** réelle sur NARMA10
  (Mem4ristor perd ~5.5×, n'est pas une mémoire) ; POC pont M4R↔LLM (le doute maintient le
  rang d'attention contre l'oversmoothing — mécanisme prouvé, utilité aval pas encore).
- **09/07** — **Volet B, le fond** (B2/B3/B5/B6) : 3 dossiers de correspondance physique
  (photonique→u, spintronique→v, électrique→2 rôles) ; comparaison d'énergie cadrée ; 3 POCs
  spintroniques en escalade (Kuramoto → Slavin-Tiberkevich → **macrospin LLGS complet**,
  vrai vecteur d'aimantation 3D) — découverte que cette géométrie de couplage verrouille en
  antiphase et que BA scale-free est frustré (3e mécanisme indépendant où BA diffère de
  lattice). Rattrapage de ce fichier (vous le lisez).
- **11-13/07** — Genèse à 5 états consolidée (la tendance de la veille était du bruit ;
  l'interférence complexe, elle, préserve la parité et bat le vote) ; **le legs de 14 pistes
  exécuté en entier** ; `u ∈ ℂ` entre dans le cœur (opt-in, bit-à-bit identique OFF) ;
  γ_int passé au crible sur 9 critères, **aucun gain net** (le seul positif, Condorcet,
  s'effondre à la réplication) ; architecture M4R↔solveur (P11) et **son bilan matériel
  fermé** : le « −96 % » n'était qu'un décompte d'itérations, il devient +4205 % en temps
  réel — leçon reprise le 26/07.
- **~14/07** — **PUBLICATION.** Julien pousse tout l'historique vers
  `github.com/cafe-virtuel/Mem4ristor`. Le plus vieux verrou du projet est levé.
- **17/07** — Audit externe (Gemini jouant « DeepMind ») traité : l'inquiétude « les
  expériences invalident le preprint » **mesurée et réfutée** (preprint inchangé, Guardian
  14/14, l'expérience accusatrice le *confirme*). Un draft interne ressuscitant une thèse
  déjà rejetée deux fois (`draft_friston_free_energy.md`) est **retiré** avec bandeau.
- **26/07 (matin)** — Hygiène du dépôt pilote ; avis sur une analyse « IA GitHub » flatteuse
  travaillant sur un état périmé ; une **vraie imprécision** trouvée à côté et corrigée dans
  le preprint (scope note I_stim, commit `71afa83`).
- **26/07 (jour)** — **Les deux dernières critiques de l'audit soldées, en 7 commits.**
  *Expérience A* : la cascade d'entrée en dead zone par les hubs **existe** (ρ = −0.56,
  répliquée sur graines disjointes) et fragmente bien au-delà d'une percolation aléatoire —
  mais c'est une propriété du **protocole de couplage**, absente sous normalisation par
  degré, et **non spécifique au scale-free** (ER donne −0.61). Son mécanisme par nœud reste
  **ouvert** : deux explications ont été proposées puis réfutées par leurs propres tests.
  *Expérience B* : recuit simulé et exploration aléatoire **battus** ; un filtre à oubli les
  remplace avantageusement (voir §3). *Suites B2-B6* : le « bon marché » corrigé (~15× plus
  cher à substrat égal), la **latence** identifiée comme la seule grandeur qui survit, et son
  mécanisme isolé — le doute sait *quand* trancher, via un signal spatial mais non
  topologique. **Un défaut de méthode découvert et corrigé le jour même** : régler un
  adversaire « à son maximum par oracle par run » fabriquait 0.935 d'accuracy sur du bruit
  pur ; toutes les mesures ont été refaites avec sélection sur graines d'entraînement et
  mesure sur graines disjointes. Le commit fautif et sa correction sont tous deux publics.

- **26/07 (soir)** — **La question laissée ouverte par B6-bis est soldée, en 4 volets**
  (`expB7_*`, commits `cb3dc1c`, `7c6c30c`, `a866eac`). *Pourquoi un simple `std(v)`
  égale-t-il le désaccord local ?* Parce que, sur cette tâche, il n'y a rien à exploiter :
  les deux signaux sont une seule lecture (r = 0.981) et le champ n'a aucune structure
  spatiale (Moran's I = −0.010). Une variante construite pour en avoir (Moran's I +0.382)
  **ne change rien** (interaction +0.02, IC [−0.08, +0.12]). Sur sept mesures indépendantes,
  l'écart est toujours ≤ 0 et jamais en faveur du désaccord local → **le signal d'arrêt ne
  tire aucun bénéfice de la topologie** (voir §3). **Deux garde-fous ont fonctionné et sont
  publics** : un premier essai a **échoué à son propre gate** (grouper les sources en un
  bloc unique ne durcit pas la tâche, elle la casse — acc_final 0.50) et n'a pas été lu ;
  et un effet net apparu ensuite (« lire la topologie coûte », −0.30) **est mort à son gate
  de réplication** sur graines jamais touchées (0/2), comme le Condorcet du 13/07. Trois
  hypothèses mécanistes de plus proposées puis réfutées par leurs propres tests.

**Total sur la période** : ~60 commits substantiels sur `main`, cœur (`dynamics.py`) jamais
modifié de façon non rétrocompatible (les deux extensions du 07/07 et du 12/07 sont opt-in
et bit-à-bit identiques désactivées), Guardian toujours ≥13/13 (13→14/14 le 08/07).
**Tout est poussé** vers `cafe-virtuel/Mem4ristor` depuis le ~14/07 ; la branche `main` est
synchronisée avec `origin/main`.

---

## 4. TESTS — CE QUI PASSE, CE QUI NE PASSE PAS

### Tests fonctionnels (devraient passer)

| Fichier | Couvre | Notes |
|:--------|:-------|:------|
| `tests/test_kernel.py` | Noyau V3, sigmoid, plasticité | 7.9 KB |
| `tests/test_robustness.py` | NaN, overflow, edge cases | 9.0 KB |
| `tests/test_fuzzing.py` | Fuzzing aléatoire (200 iter) | 3.1 KB |
| `tests/test_adversarial.py` | Attaques adverses | 2.9 KB |
| `tests/test_manus_v2.py` | Assertions V3 audit | 3.7 KB |
| `tests/test_symbiosis_*.py` | CreativeProjector, Swarm | 3.5 KB |
| `tests/test_v4_extensions.py` | Meta-doubt, rewiring | 16.9 KB |
| `tests/test_v5_hysteresis.py` | V5 hysteresis : configuration, latching, watchdog fatigue | **3 tests ACTIFS (activés 2026-03-22)** |

### Tests xfail

**2 xfail** (vérifié en relançant la suite le 2026-07-26) :
`tests/test_adversarial.py::test_snr_significance_breakdown` (effondrement du SNR en régime
de bruit fort) et `::test_euler_drift_torture` (instabilité numérique à dt > 0.1).
⚠️ *Cette section affirmait « Aucun » tout en étant contredite par l'arborescence du §5
(« 118 passed + 2 xfail ») — les deux étaient faux. Corrigé le 26/07 au soir.*

### Compte réel de la suite

**130 passed + 2 xfail**, mesuré le 2026-07-26 :
`python -m pytest --tb=no -q --ignore=tests/results` (l'option `--ignore` est obligatoire :
`tests/results` contient des dumps UTF-16 qui cassent la collecte).

### Tests de régression scientifique (2026-04-10) ★

| Fichier | Couvre | Notes |
|:--------|:-------|:------|
| `tests/test_scientific_regression.py` | 6 propriétés fondamentales du modèle | **NOUVEAU** — 6/6 PASS |

Propriétés testées :
1. Entropie > 0 avec heretics (Cold Start)
2. Collapse sans heretics
3. Doute clamped [0, 1]
4. Noise nécessaire sous Cold Start
5. Scaling entropie invariant
6. Levitating Sigmoid correcte

---

## 5. ARBORESCENCE DU PROJET

> **⚠️ Non revérifiée exhaustivement le 09/07/2026** — voir §2 pour la structure
> `src/mem4ristor/` (corrigée). Le reste de cette arborescence date d'avril-mai 2026 et
> n'a pas été audité fichier par fichier dans ce rattrapage ; corrections connues
> appliquées ci-dessous (VERSION, chemin du preprint), le reste à vérifier si besoin.

```
mem4ristor-v2-main/
├── src/mem4ristor/           # PACKAGE PRINCIPAL — voir §2 pour le détail à jour
│   ├── dynamics.py            # Le moteur réel (FHN + doute + plasticité + hystérésis)
│   ├── topology.py             # Mem4Network (graphe, couplage, sparse CSR)
│   ├── metrics.py               # H_cog, H_cont, synchronie, MI
│   ├── graph_utils.py            # make_ba/make_er/make_lattice_adj
│   ├── core.py                    # Façade de compatibilité (25 lignes)
│   ├── config.yaml                 # Paramètres par défaut
│   ├── viz.py                       # Visualisation (entropy, doubt map, phase, dashboard)
│   ├── symbiosis.py                  # CreativeProjector + SymbioticSwarm
│   ├── cortex.py                      # LearnableCortex (MLP autoencoder)
│   ├── sensory.py                      # SensoryFrontend
│   └── ... (voir §2)
│
├── experimental/              # MODULES NON PRODUCTION
│   └── mem4ristor_king.py     # "Philosopher King"
│
├── tests/                     # SUITE DE TESTS (130 passed + 2 xfail, mesuré le 26/07/2026 — voir §4)
│
├── experiments/               # SCRIPTS D'EXPÉRIENCE (b1_*, b2_*, b4_*, b5_*, p2_*, ...)
│
├── docs/
│   ├── papers/preprint/preprint.tex   # Preprint actuel (reformulé, 27 pages, Guardian 18/18)
│   ├── papers/preprint/preprint.pdf
│   ├── CLAIMS_REGISTER.md              # Registre narratif des claims (voir §3)
│   └── hardware/                        # Dossiers de correspondance physique (voir §2)
│
├── .brain/ (racine D:\ANTIGRAVITY)
│   ├── claims_mapping.json    # Source de vérité AUTOMATISÉE du Guardian (voir §3)
│   └── preprint_guardian.py   # Script de vérification, hook pre-commit
│
├── failures/                  # ÉCHECS DOCUMENTÉS
│
├── PROJECT_STATUS.md          # CE FICHIER
├── PROJECT_HISTORY.md         # Historique détaillé + archive des claims antérieurs (§13)
├── VERSION                    # V6.0.0
└── ...
```

---

## 6. COMMENT DÉMARRER

```bash
git clone https://github.com/cafe-virtuel/Mem4ristor.git
cd Mem4ristor
pip install -e .
pytest tests/
```

Quick test :
```python
from mem4ristor.core import Mem4Network
net = Mem4Network(size=10, heretic_ratio=0.15, seed=42)
for step in range(1000):
    net.step(I_stimulus=0.5)
print(f"Entropy: {net.calculate_entropy():.4f}")
```

Demo complète (2026-03-22) :
```bash
python examples/demo_applied.py
# Produit 5 PNG : sensory_pipeline.png, hysteresis_comparison.png,
#   scalefree_sparse.png, phase_diversity.png, + dashboard
```

---

## 7. CONTRIBUTEURS IA (traçabilité)

| Modèle | Rôle principal |
|:-------|:---------------|
| Claude 3.5 Sonnet (Anthropic) | Conceptualisation initiale, drafting LaTeX, core logic v2.0 |
| Grok-2 (xAI) | Tests adverses, hypothèse 15%, philosophie du doute |
| Gemini 1.5 Pro (Google) | Refactoring, sweeps de sensibilité, visualisation |
| Claude Opus 4.6 (Anthropic) | Audit V3, migration imports, investigation LIMIT-02/04/05 (2026-03-21). V5 hysteresis, sparse CSR, demo appliquée (2026-03-22) |
| Antigravity / Gemini 2.5 Pro | Consolidation v3.2.0, tests régression, BA m/α sweep, restructuration preprint, réponse critique externe (2026-04-10) |
| Claude Opus 4.7 (Anthropic) | Fix bugs P1 (symbiosis swarm, V4 entropy), validation SPICE/Python sub-1% RMS, figure phase diagram λ₂ vs H_stable, normalisation spectrale (résultat négatif publiable) (2026-04-19) |
| Antigravity (Gemini 3 Flash) | **Run Héroïque N=1600**, validation Binder $U_4$, résolution audit Mistral (19/20), Expérience 008 "Guerre des Phases" (2026-05-16) |
| **Aria, Flux, Sentinel** | **Système MAS 2026** : Veille scientifique, optimisation Chemical Inductor, Stress Test V3 (Synchronisation 100%) (2026-05-16) |
| Claude Code (Fable 5, Anthropic) | Vague 1 preprint (polissage, réparation Guardian), quatuor d'imperfections photonique, AUDIT-024 (détection + résolution), triage audit Hermès (11-12/06/2026) |
| Claude Code (Opus 4.8, Anthropic) | Mandat λ₂ (réfutation), reformulation preprint (spectral→degré), Volet A (A2-A5), Volet B (ablation IC/FSS, comparaison ESN, pont LLM), Volet B fond (dossiers hardware B2, escalade STNO Kuramoto→LLGS macrospin) (01/07-09/07/2026) |
| Claude Code (Opus 5, Anthropic) | Audit « DeepMind » soldé (Expériences A et B), cadrage « bon marché » corrigé (bilan câblage/énergie), latence B4, mécanisme d'arrêt B5-bis/B6-bis, détection et correction du piège de l'oracle par run ; puis B7 — le signal d'arrêt ne tire aucun bénéfice de la topologie (26/07/2026) |

Niveau de transparence : **Radical** — transcripts complets dans le dépôt Café Virtuel.

---


## 8. RÈGLE D'OR

> **Toute claim doit correspondre à une preuve dans le code.**
> Zéro valeur numérique publiée sans script de vérification reproductible listé dans
> `docs/CLAIMS_REGISTER.md` et vérifiable par `.brain/claims_mapping.json` (Guardian).
> Si une claim est marquée FAUX ou RÉFUTÉE, elle doit être qualifiée de
> «phénoménologique» ou «spéculative» dans le preprint, jamais silencieusement retirée.
> Les échecs sont conservés (`failures/`, `PROJECT_HISTORY.md` §13). Rien n'est effacé.

---

## 🚀 Roadmap V5 (Cognitive Innovations) — État au 2026-05-05

Suivant l'Audit Professionnel ManusAI (2026-05-01), les trois axes V5 ont été explorés sur la branche `main` (commits 695a403, 145316e, dbf40b4).

### 1. Metacognitive Plasticity ($u \to \epsilon$) — ✅ ARRETE (2026-05-31)
- **Anciens résultats (2026-05-05)**: alpha=-0.5 → H=4.82 bits (+0.79 vs V4 pur 4.03)
- **Resultat FINAL (2026-05-31, 12 seeds + AUDIT-017)**: alpha=-4.0 beneficiaire sur m>=7 (LZ baisse, structure) mais CHAOTICgenic sur m<=5 (LZ monte a 0.86-0.94). Gains vs V4: m=3 (LZ=0.942 CHAOTIC — non-structurel), m=5 (LZ=0.862 CHAOTIC — non-structurel), m=7 (LZ=0.560 FUNCTIONAL, +1.47 bits), m=10 (LZ=0.510 FUNCTIONAL, +0.84 bits).
- **Mecanisme revise (AUDIT-017)**: alpha=-4.0 gele quasi-totalement la plasticite w. L'effet est regime-dependent:
  - m<=5 (sparse): gele dans le chaos — LZ passe de 0.79->0.94. Gain H_cont = gain chaos.
  - m>=7 (dense): gele dans la structure — LZ passe de 0.77->0.51. Gain legitime.
  - Le SWEET SPOT D(u)+alpha=-4.0 est optimal POUR LES TOPOLOGIES DENSES (m>=7), pas universellement.
- **ARRET (AUDIT-018)**: Section Binder du preprint INVALIDEE. Chemin B (correlation LZ-H_stable) ABANDONNE. Manuscrit重构 autour de la depression H_stable (crossover progressif) et de la baisse LZ — pas d'une transition de phase thermodynamique. Voir AUDIT-017 + AUDIT-018.

### 2. Non-Local Topological Coupling (Doubt Similarity) — ❌ ÉCHEC
- **Résultat** : u_spread augmente (+0.01 à +0.03) mais H baisse (-0.08 à -0.23 bits). La synchronisation inter-nœuds similaires étouffe la chimère. Commit 145316e.
- **Conclusion** : V4 pur reste optimal sur cet axe.

### 3. Dynamic Compartmentalization (Sub-Personalities) — ✅ IMPLÉMENTÉ + CLARIFIÉ
- **Ancien résultat (2026-05-05)**: K=3 full gamma=0.10 → H=4.18 bits (+0.15 vs V4 pur)
- **Résultat FINAL (2026-05-31)**: la compartimentalisation est NEUTRE ou NOCIVE en combinaison avec D(u)+alpha=-4.0.
  Sur BA m>=7 (dead zone): K=3 reduit H de -0.30 bits vs D(u)+alpha=-4.0 seul.
  La separation en sous-groupes n'apporte rien quand la plasticite w est geleed.
- **Conclusion**: K=3 etait beneficial en isolation (V4+compart) mais pas en combination avec D(u)+meta.
  K=2 ou K=10 sont a peu pres equivalents a pas de compartiments (within noise).

### 4. Prochaine étape V5 — ✅ RÉSOLUE (2026-05-31)
- **PRIORITÉ initiale**: combiner Metacognitif (alpha=-0.5) + Compartimentalisation (K=3 full)
- **Résultat FINAL**: D(u)=0.50*u + alpha_meta=-4.0 SANS compartimentalisation = OPTIMAL
  Gains vs V4: m=3 (+1.50), m=5 (+1.90), m=7 (+1.47), m=10 (+0.89)
- **Piste V6**: Memristors Photoniques — I_stimulus via fibre optique (matériaux : GST, VO2, WO3)

### 5. Advanced Visualization
- **Concept** : 3D mapping of entropy flows between Hierarchical layers (V1 → V4 → PFC). Pending.



---

## 9. FIGURES DE RÉFÉRENCE — INVENTAIRE INTERNE (02 mai 2026)

Ces figures existent dans figures/ mais ne sont pas citées dans les documents publiés.
Elles sont documentées ici pour traçabilité et pour répondre aux reviewers si nécessaire.

### Résultats scientifiques confirmés (non publiés)

| Fichier | Expérience | Résultat | Statut |
|---------|-----------|---------|--------|
| event_phase_transition*.csv + _rerun_20260711* | [13] Transition de phase événementielle | ⚠️ **RÉVISÉ 11/07/2026** : le « dH=+1.20, périphérique > hub » d'avril NE se reproduit PAS au code actuel (AUDIT-024, bruit ×4.47) — grille complète rejouée : 0/9 configs du claim positives, l'événement DÉGRADE H_cont (≈−1.0) sur BA m=3, hub comme périphérique ; effet ~nul sur la dead zone m=5. Claim d'avril = artefact de l'ancien bruit. CSV d'avril préservés. | RÉVISÉ (rerun 2026-07-11, `event_phase_transition_rerun_20260711.py`) |
| matern_noise.png / .csv / _summary.csv | [12] Bruit Matern spatialement corrélé | Tous types de bruit (i.i.d., Matern exp/Gauss) brisent la dead zone à eta=0.1+. Amplitude seule compte. | Confirmé, commit a8d6ba4 |
| v4_dynamic_heretics_emergence.csv | V4 Dynamic Heretics | Loi linéaire t_first = steps_required + 130 * u_threshold (R²~1). Cascade quasi-inévitable pour u_thr <= 0.8. | Confirmé, commit 88b9983 |
| exp_008_guerre_phases.png | Expérience 008 | La minorité organisée (15%) crée une oscillation perpétuelle à l'interface. | Confirmé (Shadow Lab) |
| exp_008_v3_sentinel_proof.png | Expérience 008 V3 | **Stress Test Sentinel** : Synchronisation 1.0000 sans bruit. Preuve de la robustesse structurelle de l'inductance chimique. | VALIDÉ (Mai 2026) |

### Analyses de robustesse et défenses reviewer

| Fichier | Analyse | Note |
|---------|---------|------|
| floquet_multipliers.png | Multiplicateurs de Floquet | ⚠️ **N'est plus dans le preprint.** La section Floquet est un claim **supprimé le 06/05/2026** (`CLAIMS_REGISTER.md`, motif : le Jacobien donne \|μ\|≈0.20, contradictoire). La figure et cette ligne ont survécu à la suppression ; la dernière phrase qui s'y référait encore dans le `.tex` a été retirée le 30/07. Conservée comme archive, **à ne pas citer**. |
| lyapunov_analysis.png | Exposant de Lyapunov max vs couplage D (Section 5.2 preprint) | Ajouté figure preprint.tex 02 mai 2026 |
| phase_diagram_corrected.png | Diagramme de phase corrigé (post-audit KIMI) | Remplacé par fiedler_phase_diagram.png comme référence principale |
| phase_diagram_v4_extensions.png | Extensions V4 (dynamic heretics) | Exploratoire, données dans figures/v4_parametric_sweep.csv |
| flop_benchmark.png | Benchmark performances (FLOP/s vs N) | Interne — pas dans les papers |

### Données hardware SPICE

|| Fichier | Contenu | Note |
|---------|---------|------|
|| p420_hfo2_memristor.csv | Caractérisation SPICE HfO₂ — 50 seeds Monte Carlo | Référencé dans Paper B (in preparation) |

---

## 10. AUDIT HERMES M3 — 2026-06-01 (sessions 010 + 011) — **RÉSOLU, archivé**

> Cette section listait 3 corrections obligatoires + 2 recommandées identifiées début
> juin. **Toutes sont résolues depuis** (vérifié le 09/07/2026 par recoupement avec
> `docs/CLAIMS_REGISTER.md` actuel) :
> - **C07** (topologie erronée) → résolu : le registre distingue maintenant C07 (BA m=3)
>   et C07b (lattice, alors 0.0197±0.0142 `[valeur morte]`) séparément. **Complété le 31/07** : le Guardian, lui,
>   appelait encore « C07 » le claim lattice — clé renommée `C07b` ; et le C07 du registre
>   est désormais **remplacé par C14** (sa valeur 0.031 `[valeur morte]` l'est depuis le 29/07).
> - **C12** (Binder U4 plat) → résolu : le claim est marqué `~~C12~~ INFIRMÉE` dans le
>   registre, section Binder retirée du preprint.
> - **C05** (valeur canonique floue λ₂) → **dépassé plutôt que résolu** : λ₂ n'est plus le
>   mécanisme causal du tout depuis le mandat du 1er juillet (voir §3). La question
>   « EBC vs combined » ne se pose plus dans les mêmes termes.
> - **C08** (seeds/topologie) → résolu : C08 (lattice) et C08b (BA m=3) séparés et
>   régénérés à 10 seeds chacun (AUDIT-024, 12/06).
> - **item structurel « core.py = 25 lignes, désync avec §2 »** → résolu dans ce
>   rattrapage du 09/07 (voir §2 ci-dessus).
>
> Détail complet de l'audit original conservé dans `PROJECT_HISTORY.md` §13 (règle
> d'or : rien n'est effacé). L'inventaire structurel (working tree pollué, 74 scripts
> sans CSV/PNG, 3 papiers LaTeX en parallèle) n'a pas été revérifié le 09/07 — à
> auditer à nouveau si quelqu'un veut faire du ménage de dépôt.

---

**Statut publication** : le code et l'historique **sont publics depuis le ~14/07** (voir §0)
— ne pas lire ce paragraphe comme « publication bloquée », c'est l'erreur qu'un audit externe
a tirée de ce fichier en juillet. Ce qui reste ouvert est la seule **soumission arXiv/HAL**.
Le preprint est reformulé et cohérent avec le code (Guardian **18/18**, et depuis le
30/07 le **Tex Guardian** vérifie aussi le texte publié contre les CSV) ; il n'y a plus de
correction technique bloquante depuis juin. ⚠️ **Nuance ajoutée le 30/07** : « cohérent avec
le code » n'a longtemps désigné que les **CSV**. La première passe systématique du texte
(801 lignes) a relevé **17 défauts, tous dans le `.tex` et aucun dans les données** — dont un
claim supprimé le 06/05 encore publié. Corrigés depuis ; voir le transcript du 30/07. **Position de Julien au 26/07/2026** : c'est
trop tôt pour arXiv tant que le travail n'a pas été confronté à de vrais chercheurs — un
endorsement (nlin.AO / cs.NE) ne se demande pas dans le vide, il se demande à quelqu'un qui
a lu. Ce n'est pas de l'attentisme, c'est un ordre d'opérations.