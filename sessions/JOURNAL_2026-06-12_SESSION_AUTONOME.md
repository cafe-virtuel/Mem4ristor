# 📓 Journal de bord — Session autonome du 12 juin 2026

> **Agent** : Claude Code (Fable 5) — L'Ingénieur
> **Mandat de Julien** : « faire aboutir mem4ristor, le tester, l'améliorer si possible,
> l'amener aux frontières du réel » — en autonomie, avec journal pour transmission.
> **Règles auto-imposées** : pas de push GitHub (décision Julien), pas de modification
> structurelle, tout chiffre nouveau = script reproductible, tout doute = `// @DOUBT`.

Ce journal est écrit pour qu'un autre agent (ou Julien) puisse reprendre chaque fil.
Chaque entrée : QUOI / POURQUOI / COMMENT REPRODUIRE / RÉSULTAT.

---

## Plan de session (établi à l'ouverture)

Priorités tirées de SYNAPSE (11 juin) + contexte MEM4RISTOR.md :

1. **Vérifier l'état** : suite de tests complète (référence : 118 passed, 2 xfail).
2. **Trancher C04** : écart sync_FULL 0.0072 (re-run Hermès 5 seeds) vs 0.0673
   (CSV commité + preprint, 10 seeds). Re-run documenté DANS TEST_HERMES. Bloquant soumission.
3. **Reformuler C20** sur AC@lag50 uniquement (LZ et fréquence invalidés par contre-expertise 3 juin).
4. **Relancer POC C** avec FFT + LZ_state (remplacer le classifieur binaire par f_v brute).
5. **POC #5** : relancer sur 5 seeds (actuellement n=1, aucune valeur statistique).
6. Si temps : finitions preprint marquées « décision Julien » → PRÉPARÉES mais pas appliquées
   sans accord (H_cog → appendice, adoucir « Paper B »), sauf si trivialement réversibles.

---

## Entrées

### [Ouverture] État initial
- Branche `main`, HEAD `7893e3e` (local, ahead 1, non poussé — on n'y touche pas).
- 164 fichiers untracked/modifiés d'anciennes sessions — NON TOUCHÉS (à ranger dans une session dédiée).
- Suite pytest lancée en arrière-plan, résultat à venir.

### [1] Suite de tests — VERTE ✅
- **QUOI** : pytest complet sur `tests/`.
- **PIÈGE pour le suivant** : (a) le `python` du PATH = venv Hermès SANS pytest ; le bon
  interpréteur est `C:\Users\julch\AppData\Local\Programs\Python\Python313\python.exe`.
  (b) pytest crashe en collectant `tests/results/*.txt` (dumps UTF-16 non commités) →
  ajouter `--ignore=tests\results`.
- **COMMANDE** : `python -m pytest tests --ignore=tests\results -q`
- **RÉSULTAT** : 120 collectés = 118 passed + 2 xfail connus (SNR, Euler drift). Conforme référence.

### [2] C04 TRANCHÉ — découverte majeure : le bruit a changé le 1er mai 🚨
- **QUOI** : question ouverte bloquante (0.0072 vs 0.0673 pour sync_FULL).
- **MÉTHODE** : script `experiments/c04_rerun_20260612.py` (réutilise `run_one()` du script
  original sans le modifier, écrit `figures/c04_rerun_20260612.csv` — n'écrase RIEN).
  Puis bisection par `git worktree` sur 5 commits (~15 s par run).
- **RÉSULTAT** :
  - Code actuel (HEAD 7893e3e) : sync_FULL = **0.0072**, FROZEN = 0.6513 → reproduit Hermès, PAS le preprint.
  - Code du 25 avril (0fdeee0) : sync_FULL = **0.0673**, FROZEN = 0.7302 → reproduit le preprint exactement.
  - Bisection : 0fdeee0 OK → 88b9983 OK → **818cf67 PREMIER FAUTIF** → tous BAD ensuite.
- **CAUSE RACINE** : `818cf67` (1er mai, « Hostile Audit Defense », Antigravity) a introduit
  le scaling Euler-Maruyama dans `dynamics.py` : `eta = N(0,σ_v)/sqrt(dt)`. Avec dt=0.05,
  bruit effectif ×4.47. Mathématiquement CORRECT (Itô), mais change la dynamique :
  le régime FULL est plus décorrélé (sync 0.067→0.007, H_cog 0.018→0.180).
- **IMPLICATION GRAVE** : le preprint mélange deux générations de code :
  - C01–C03, C07, C09, C10 (vérifiés 6–11 mai) = nouveau bruit → cohérents avec HEAD. ✅
  - C04 (CSV du 25 avril) et C08 (CSV du 24 avril) = ANCIEN bruit → NON reproductibles avec HEAD. ❌
  - Le « re-run parasite » C08 du 10 juin (MI 0.634 vs 0.870) était probablement le code
    actuel qui disait la vérité — la restauration du CSV commité a restauré une valeur
    d'un code disparu. À re-vérifier (entrée suivante).
- **NOTE** : le claim qualitatif C04 SURVIT et devient plus fort : FROZEN/FULL passe de
  ~11× (+985%) à ~90× avec le code actuel. C'est la valeur chiffrée qui est fausse, pas l'effet.
- Worktrees de bisection : `C:\Temp\c04_b*` — à nettoyer en clôture (`git worktree prune`).

### [3] Inventaire de l'ampleur — quels claims reposent sur l'ancien bruit ?
Dates git des CSV mappés par le Guardian :
| CSV | Dernier commit | Génération de code |
|---|---|---|
| p2_delta_sweep.csv (C01) | 24/04 | ⚠️ ANCIEN bruit |
| p2_sigma_social_ablation.csv (C04) | 26/04 | ⚠️ ANCIEN bruit |
| lambda2_crit_regression.csv (C05) | 27/04 | ⚠️ méta-analyse sur CSV d'avril |
| p2_spatial_mutual_information.csv (C08) | 24/04 | ⚠️ ANCIEN bruit |
| reviewer2_linear_stability (C06) | 02/05 | ✅ nouveau (et déterministe) |
| p2_art_benchmark (C09), p2_v5_combination (C10) | 11/05 | ✅ nouveau |
| spice_art_kirchhoff (C11) | 15/05 | ✅ nouveau |
| v6_binder_cumulant_U4 (C12) | 22/05 | ✅ nouveau |
| p2_table1_lattice.csv, p2_table1_sync.csv (C02/C03/C07b) | **NON COMMITÉS** (disque : 27/05) | ✅ nouveau, mais hors source unique ! |

→ **Faille séparée découverte** : les CSV Table 1 vérifiés par le Guardian ne sont PAS dans git.
À commettre (décision facile, je le signale dans la synthèse).

### [4] C08 re-vérifié avec le code actuel (HEAD, worktree jetable)
- **COMMANDE** : `python experiments/p2_spatial_mutual_information.py` dans worktree à 7893e3e.
- **RÉSULTAT** : lattice d=1 : FULL=0.5997, FROZEN_U=1.7012 → ratio **2.84×** (publié : 2.25×).
  BA m=3 : FULL=0.4852, FROZEN=1.6361 → ratio **3.37×** (publié : 1.84×).
- Le « re-run parasite » du 10 juin (FULL=0.634) était bien le code actuel : il disait la vérité.
  La restauration du CSV commité le 10 juin a réinstallé une valeur d'un code disparu.
- Effet qualitatif (FROZEN ≫ FULL) : SURVIT et s'amplifie. Valeurs publiées : non reproductibles.

### [5] C01 re-vérifié + C05 hors de cause + Dead Zone testée
- **C01** (delta sweep, worktree HEAD, 44.7 s) : H_cont(lattice, δ=0) = 4.157 vs 4.06±0.08 publié.
  Dérive +0.1 bit, à la limite de tolérance. Preuve : `figures/c01_rerun_20260612_HEAD.csv`.
- **C05** (λ₂_crit=2.31) : le re-run EBC à HEAD produit un CSV STRICTEMENT IDENTIQUE au commité
  (diff vide, 2,4 s — pas de simulation, métriques de graphe pures). C05 INTACT.
- **Dead Zone** (`experiments/deadzone_check_20260612.py`, BA m=5 vs m=3, 2 worktrees) :
  H_cog(m=5) = 0.000 dans les DEUX codes ; chute H_cont m=3→m=5 préservée (−0.49 nouveau,
  −0.60 ancien). **La dead zone SURVIT au changement de bruit.** Le claim central tient.
- Crainte écartée : le bruit effectif ~0.22 ne brise PAS la dead zone (le claim [12] « η≥0.1
  brise la dead zone » portait sur du bruit ajouté η explicite, autre canal).

### [6] Documentation propagée
- **AUDIT-024** écrit dans `AUDIT_LOG.md` (méthode, tableau d'ampleur, décision requise A/B).
- **CLAIMS_REGISTER.md** : bandeau AUDIT-024 + statuts C01/C04/C08/C08b mis à jour ;
  correction au passage : C04 indiquait « 10 seeds », le script en a toujours eu 5.
- **C20 REFORMULÉ** (tâche en attente depuis le 03/06) : claim restreint à la persistance
  temporelle AC@lag50 = +0.57 à +0.74 (script `poc1_absence_v2.py`, code actuel, sain).
  Retirés : fréquence f~0.01 (réel 0.002) et LZ sur v_mean. « Intrinsic oscillator » abandonné.
  Propagé dans PROJECT_STATUS.md [20].
- **claims_mapping.json** (hors repo) : note C04 passée de « QUESTION OUVERTE » à « RÉSOLU »,
  expected reste 0.0673 (= CSV commité) tant que Julien n'a pas tranché la régénération.
- **Guardian relancé après tout ça : 12/12 OK** (aucun CSV commité modifié).

### [7] POC v2 lancés (réponses à la contre-expertise du 03/06)
- `experiments/poc_c_sweep_v2.py` : FFT au lieu de zero-crossing, drive_power_frac
  (métrique CONTINUE d'entraînement) au lieu du classifieur binaire, LZ_state (T,N).
  Grille identique v1 (300 runs). Sorties : `figures/poc_c_sweep_v2_*`.
- `experiments/poc5_bruit_v2.py` : POC #5 sur 5 seeds (était n=1). Sorties : `figures/poc5_bruit_v2_*`.

### [8] Résultats POC v2
- **POC #5 v2 (5 seeds)** : verdict PASSIVE consolidé — cc_drive moyen +0.668 (4/5 réalisations
  de bruit transmettent fortement 0.64–0.92, 1/5 à ~0.04). Fait notable : cc_drive dépend
  UNIQUEMENT de la réalisation du bruit, quasi identique à travers m et D — cette
  insensibilité à la topologie est en soi la meilleure preuve de transmission passive.
- **POC C v2 — C21 RÉVISÉ 🔬** : avec la FFT, AUCUNE bascule ≤10% pivots. f_fft reste
  endogène (~0.002) dans 59/60 conditions. drive_power_frac croît CONTINÛMENT avec
  n_pivots et D (max 0.343 à m=6 D=0.5 10%). La « bascule à 8%/10% » (C21 v1) était un
  artefact zero-crossing + classifieur binaire — la contre-expertise du 03/06 (PROBLÈME
  MAJEUR 1) avait vu juste. CLAIMS_REGISTER C21 mis à jour. lz_state décroît légèrement
  avec n_pivots à D>0 (1.73→1.63) : le drive impose un peu de structure.

### [9] Fix iso-comportemental : NaN silencieux à D=0 (topology.py:299)
- **QUOI** : à D=0 avec coupling_norm≠'uniform', `scale_factors = 0/0 = NaN`
  (RuntimeWarning à chaque step) ; les NaN étaient lessivés en 0 par `nan_to_num` dans
  dynamics.py:181 — σ_social=0 à D=0 PAR ACCIDENT, pas par décision.
- **FIX** : branche explicite `D > 0` dans topology.py ; à D=0, `l_v = zeros` (comportement
  strictement identique, rendu intentionnel). Marqué `@DOUBT` : la limite algébrique D→0
  de scale_factors est `node_weights·√N ≠ 0` — si l'on voulait que σ_social survive à D=0,
  c'est un choix de MODÉLISATION à trancher (Julien/Architecte).
- **VÉRIFICATION** : état (v,u,w) identique bit à bit après 200 steps (test A/B,
  `C:/Temp/d0_before.npy`) ; plus aucun warning ; pytest 118+2 inchangé (exit 0).

---

## 📋 SYNTHÈSE & MÉMO DE DÉCISION (pour Julien)

### Ce qui est tranché
1. **C04 résolu** : ce n'était ni les seeds ni le repo — le commit `818cf67` (1er mai) a
   changé le modèle de bruit (Euler-Maruyama, ×4.47 effectif). Bisection git reproductible.
2. **C21 révisé** : pas de bascule à 8-10% ; entraînement graduel continu (artefact de métrique).
3. **C20 reformulé** : persistance temporelle AC@lag50 seule (comme demandé le 03/06).
4. **POC #5** : PASSIVE confirmé sur 5 seeds.
5. **Dead zone** : SURVIT au changement de bruit (H_cog=0 à m=5 dans les deux codes). ✅
6. **λ₂_crit=2.31 (C05)** : INTACT (purement topologique).

### ⚖️ DÉCISION REQUISE — incohérence de générations dans le preprint
Le preprint cite des valeurs de DEUX codes différents. Un lecteur qui clone le repo
public et relance les scripts obtiendra :
| Valeur preprint | Obtenue aujourd'hui (HEAD) |
|---|---|
| sync FULL 0.067 / FROZEN 0.730 (+985%) | 0.0072 / 0.6513 (~×90) |
| MI ratio lattice 2.25× | 2.84× |
| MI ratio BA m=3 1.84× | 3.37× |
| H_stable lattice δ=0 : 4.06±0.08 | 4.157 |

**Option A (recommandée)** : régénérer les CSV pré-mai avec le code actuel et corriger les
valeurs du preprint. Tous les effets se RENFORCENT — c'est une correction favorable.
Coût : ~1 session (re-runs faits, il reste à écraser les CSV canoniques + éditer le preprint
+ mettre à jour claims_mapping + recompiler + Guardian).
**Option B** : flag config « bruit legacy » pour reproduire les anciennes valeurs — déconseillé
(l'ancien scaling était non-standard, et le fix venait d'une défense Reviewer 2).

### Actions faciles restantes (pas faites, par prudence)
- [ ] Committer `figures/p2_table1_lattice.csv` + `p2_table1_sync.csv` (C02/C03/C07b vérifiés
  par le Guardian mais hors git ! Générés 27/05 avec le code actuel, sains).
- [ ] Finitions Grok marquées « décision Julien » : H_cog table → appendice ; adoucir « Paper B ».
- [ ] Push des commits locaux (décision publication).
- [ ] Trancher le @DOUBT de topology.py : à D=0, σ_social=0 (actuel) ou node_weights·√N (limite algébrique) ?

### Fichiers de cette session
- Scripts : `experiments/c04_rerun_20260612.py`, `deadzone_check_20260612.py`,
  `poc_c_sweep_v2.py`, `poc5_bruit_v2.py`
- Preuves : `figures/c01_rerun_20260612_HEAD.csv`, `c04_rerun_20260612.csv`,
  `c08_rerun_20260612_HEAD.csv`, `poc_c_sweep_v2_*.csv/png`, `poc5_bruit_v2_*.csv`
- Docs : `AUDIT_LOG.md` (AUDIT-024), `docs/CLAIMS_REGISTER.md` (C01/C04/C08/C20/C21),
  `PROJECT_STATUS.md` ([20]), ce journal
- Code : `src/mem4ristor/topology.py` (fix iso-comportemental D=0)
- Hors repo : `D:/ANTIGRAVITY/.brain/claims_mapping.json` (note C04 résolue)
