# SESSION 010 — Cartographie Structurelle TEST_HERMES

**Date** : 2026-06-01
**Agent** : Hermes (M3)
**Type** : Audit structurel a froid (Phase 1/3 de l'audit complet)
**Mandat** : Julien Chauvin — "auditer totalement le dossier TEST_HERMES, peut-etre quelque chose oublie ou nouvelle approche"

---

## 1. Verdict global

Le dossier TEST_HERMES est **structurellement coherent** mais porte **3 charges lourdes** qui meritent attention immediate :

1. **Working tree pollue** : 80+ fichiers "deleted" par git mais presents sur disque (deplaces dans `archives/`), 90+ untracked. Etat confus entre ce qui est canonique et ce qui est archive.
2. **74 scripts sur ~120 sans CSV/PNG produit identifiable** dans `figures/`. Soit non executes, soit outputs sous d'autres conventions de nommage, soit deplaces.
3. **3 papiers LaTeX en parallele** (preprint V6, paper_2, paper_B) — coherence inter-papiers non verifiee depuis le degonflage du 31 mai.

**Bilan** : le projet est plus profond que ce que PROJECT_STATUS.md raconte, et c'est une bonne nouvelle pour la chasse aux approches nouvelles.

---

## 2. Inventaire quantitatif

### 2.1 Tailles

| Repertoire     | Taille  | Commentaire                                  |
|----------------|---------|----------------------------------------------|
| experiments/   | 11 GB   | Dont 11 GB dans `spice/results/` (gitignore) |
| figures/       | 13 MB   | 150 PNG + 108 CSV                            |
| archives/      | 8.1 MB  | Audits, notes, backups                       |
| docs/          | 7.7 MB  | 3 papiers LaTeX + compendium                 |
| results/       | 4.6 MB  | WORK_LOG + plots historiques                 |
| notebooks/     | 2.2 MB  | notebooks/edison uniquement                  |
| tests/         | 1.5 MB  | 27 fichiers .py dont exploits/               |
| examples/      | 1.4 MB  | demos + sonification                         |
| src/           | 579 KB  | Code canonique (core/dynamics/topology)      |
| sessions/      | 72 KB   | 11 fichiers HANDOVER/RAPPORT                 |

### 2.2 Inventaire code

- **2992 fichiers .py au total** (incluant __pycache__ parasites — reel ~150 .py source)
- **Code canonique (src/)** : 13 fichiers, 1228 lignes
  - `core.py` = 25 lignes (façade), routing vers dynamics/topology/metrics
  - `dynamics.py` = 434 lignes
  - `topology.py` = 452 lignes
  - `metrics.py` = 317 lignes
  - Reste = < 50 lignes chacun (sensory, cortex, sonification, etc.)
- **Tests** : 27 fichiers (incluant 11 dans `exploits/` — anciennes attaques record)

### 2.3 Inventaire branches

```
* main                                     (en retard 2 commits sur origin)
  feat/kimi-p419-continuous-entropy        (5 commits, jamais merge)
  feat/spice-mismatch-sweep-escape         (4 commits, jamais merge)
  feat/v4-dynamic-heretics                 (4 commits, jamais merge)
  research/v5-cognitive-exploration        (10 commits, AUCUN unique vs main — DEJA MERGE)
```

**Conclusion branches** : aucune branche locale n'a de commits ahead de main. Toutes les branches distantes sont soit obsoletes (kimi, v4-dynamic-heretics, spice) soit completement absorbees (v5-cognitive). 

Les branches feat/* ne contiennent rien de nouveau par rapport a main — ce sont des archives historiques qui devraient etre preservees en tags, pas maintenues actives.

---

## 3. Top 10 red flags structurels

### Red flag #1 : Working tree incoherent (CRITIQUE)

80+ fichiers "deleted" par git mais presents sur disque. Pattern repete :
- `docs/paper_2/paper_2.{tex,pdf,aux,log,out}` → restaure depuis archives
- `docs/paper_B/{...}` → restaure depuis archives
- `docs/preprint.pdf` + `docs/preprint.tex` → restaure, mais aussi une copie a `docs/papers/preprint/`
- `experimental/arena.py`, `hierarchy.py`, `mem4ristor_king.py`, `demo_sonification.py` → deplaces dans `examples/`
- `failures/README.md` → archive
- `hw_models/mem4ristor_v26.va` → deplace dans `src/mem4ristor/hw_models/`
- `logs/*.txt` → archive dans `archives/failures_logs/`
- `spice/mem4ristor_coupled_3x3.cir` → deplace dans `experiments/spice/`
- `notebooks/edison_review_v310.ipynb` → deplace dans `notebooks/edison/`

**Risque** : confusion sur ce qui est canonique. Un nouvel agent peut lire l'ancienne version au lieu de la nouvelle.

### Red flag #2 : SPICE 11 GB non audite (MAJEUR)

`experiments/spice/results/` contient 374 fichiers `.dat` + 376 fichiers `.cir` (10 GB+), tous gitignores. 172 d'entre eux sont des simulations `19ter_bisect_*` de 30 MB chacune.

**Question ouverte** : ces donnees sont-elles les traces brutes qui ont produit les figures du preprint, ou des caches regenerationnables ?

**Risque** : si elles sont dans le gitignore, c'est qu'elles sont regenerables — donc 11 GB de cache sur disque pour rien. Sinon, c'est un risque de perte en cas de crash disque.

### Red flag #3 : 3 papiers LaTeX, 1 table de verite (MAJEUR)

- `docs/papers/preprint/preprint.tex` (21 pages, V6.0.0, EDISON-clean)
- `docs/papers/paper_2/paper_2.tex` (12 pages ?, "Doubt Variable as Anti-Synchronization Filter")
- `docs/papers/paper_B/paper_B.tex` (Paper B hardware, escape par bruit Johnson-Nyquist)

`docs/CLAIMS_REGISTER.md` = la table de verite unique. Mais les 3 papiers peuvent deriver en parallele. EDISON 31 mai a nettoye preprint, mais paper_2 et paper_B n'ont pas ete re-audites.

### Red flag #4 : `core.py` = 25 lignes = façade vide (MINEUR mais signifiant)

Le code canonique est decompose en 3 modules (dynamics 434L, topology 452L, metrics 317L), et `core.py` n'est plus qu'un import-router. C'est une bonne architecture, mais **PROJECT_STATUS.md pretend toujours que `core.py` = "Moteur V3 canonique (Mem4ristorV3 + Mem4Network)"** — c'est desormais faux au sens litteral.

### Red flag #5 : SESSION_001, SESSION_007, SESSION_010 absents (MINEUR)

`sessions/` contient 002, 003, 004, 005, 006, 008, 009, AUDIT_011, EDISON_REVIEW, plus 1 HANDOVER.md isole. Manquent : 001, 007, 010 (= celle-ci). Sequence documentee est **non-continue**.

### Red flag #6 : DZ1, fss_lambda2_sweep mentionnes mais absents (POTENTIEL)

`PROJECT_STATUS.md` mentionne `fss_lambda2_sweep.py` dans MEM4RISTOR research-mode. Or le script **existe** dans `experiments/` (3 versions : original, v2, extended). Mais aucune trace de DZ1 dans la liste des scripts (DZ1 = hardware sweep selon nomenclature). `spice/DZ1_hardware_sweep.py` pourrait-etre manquant, voir.

### Red flag #7 : `preprint.tex` Edits EDISON post-AUDIT-019 — recompile ou pas ?

L'AUDIT-019 du 1er juin a fait 16 patches sur preprint.tex. Le git working tree montre 2 commits ahead de origin :
- `cc0227a fix: correct demo_applied.py path — examples/, not experimental/`
- `76dc9ce docs: align README and metadata with V6.0.0 reality`

Mais l'edit EDISON du 31 mai / 1er juin n'est pas dans ces commits. Le compile final du 1er juin a-t-il ete commit ? Risque de drift.

### Red flag #8 : `paper_2` contient "Topological Diversity Boundaries" — pas dans preprint

Le titre de paper_2 parle de "Topological Diversity Boundaries and Their Adaptive Resolution". C'est un angle different du preprint (V6 = "spectral dead zone and chimera states"). Risque de chevauchement editorial entre paper_2 et preprint §3.4 sur D(u) adaptive protocol.

### Red flag #9 : `docs/AVANCEE_A_REPRENDRE.md` + `docs/TEST_HERMES.md` + `docs/TESTS_GUIDE.md` (MINEUR)

3 fichiers de meta-documentation au statut flou. Sont-ils a jour ? Sont-ils suivis ?

### Red flag #10 : Branche `v5-cognitive-exploration` deja mergee (INFO)

Le diff entre main et la branche est **vide** — tous les commits uniques sont deja sur main. Cette branche peut etre supprimee sans risque (ou archivee en tag `v5-cognitive-final`).

---

## 4. Pistes pour Phase 2 (audit scientifique) et Phase 3 (chasse aux approches)

**Hypotheses pour la suite (a verifier en Phase 2)** :

A. `experiments/campaign_j_binder_lz.py` (mai 31, 16 KB) — script recent avec CSV `campaign_j_agg.csv` deja dans figures/. Probablement le coeur de la campagne J, lien direct avec AUDIT-017 (Binder U4 abandonne).

B. `experiments/fss_lz_sweep.py` (mai 30, 20 KB) — sweep LZ76 le plus recent. Probablement le "regime map" qui remplace la Binder claim.

C. `experiments/p2_v5_final_best.py` (mai 31, ~12 KB) — protocole V5 optimal (D(u)=0.50*u + alpha_meta=-4.0). Le "BEST" du titre est un signal.

D. `experiments/spice_art_kirchhoff.py` (mai 15, 24 KB) — seul script qui a produit un resultat SPICE/Python avec accord parfait (claim C11, AUDIT-005). Le plus stable.

E. Branche `feat/kimi-p419-continuous-entropy` — 10 commits, 25-26 avril. Contenait une grosse passe de corrections metriques et paper_2 work. A voir si certains fixes metriques n'ont pas ete reportes dans main (la KIMI est dans core.py facade, donc oui probable).

F. `experimental/mem4ristor_king.py` deplace dans `examples/mem4ristor_king.py`. Le "Philosopher King" — concept V5 archive, jamais exploite dans la version finale.

G. `notebooks/edison/edison_review_v310.ipynb` — seul notebook. Contient-il les chiffres EDISON 2026-06-01 ou une version anterieure ?

**Hypothese forte** : il y a probablement un Paper C (ou une section orpheline) cache dans `paper_2` qui meriterait d'etre isolee comme troisieme publication distincte.

---

## 5. Recommandations immediates (sans risque)

1. **Commit la cartographie** dans `sessions/SESSION_010_CARTOGRAPHIE_HERMES.md` (ce fichier, fait).
2. **Faire un `git status` clean** : valider que les 2 commits ahead de origin sont effectivement desirables (cc0227a + 76dc9ce) et les pusher. Sinon, les rebaser.
3. **Documenter la migration `docs/preprint.pdf` → `docs/papers/preprint/preprint.pdf`** dans un fichier `docs/MIGRATION_LOG.md` pour tracer les renommages anciens.
4. **Verifier si `experiments/spice/results/` (11 GB) est regenerable** — si oui, le sortir du working tree (deplacer dans un volume externe), si non, le backuper.
5. **Archeologie rapide de `docs/paper_2/` vs `docs/papers/paper_2/`** — sont-ce les memes fichiers ? Si doublons, n'en garder qu'un.

---

## 6. Prochaines etapes

- **Phase 2** (audit scientifique froid) : verifier que les claims 12-19 du preprint reposent sur des scripts qui fonctionnent et des CSV qui existent. EDISON-stop-rule sur chaque.
- **Phase 3** (chasse aux approches nouvelles) : sortir les scripts qui n'ont pas ete publies (campaign_j, fss_lz_sweep, p2_v5_final_best) et voir s'ils contiennent un resultat qui merite section dans preprint V6 ou un paper dedie.

---

*Hermes — 2026-06-01, session 010, modele M3*
