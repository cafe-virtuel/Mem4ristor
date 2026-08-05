# CONTEXT — Mem4ristor V6.0.0 (code) · V4.0.0 (dernière version déposée)

> Point d'entrée universel. Lisible par n'importe quel LLM ou humain en < 2 minutes.
> Pour l'état technique détaillé : PROJECT_STATUS.md
> Pour l'historique des sessions et investigations : PROJECT_HISTORY.md

---

## Le modèle en 3 phrases

Mem4ristor simule des réseaux de neurones FitzHugh-Nagumo (FHN) où chaque nœud possède une variable de **doute u ∈ [0,1]** qui module dynamiquement la polarité de son couplage avec ses voisins. Quand u ≈ 0 ou u ≈ 1, le nœud est "hérétique" — il pousse activement contre le consensus, empêchant l'effondrement synchrone. La connectivité du réseau détermine si ce mécanisme peut fonctionner : au-delà d'un certain **degré de couplage** (degré harmonique k_harm ≈ 6), le réseau entre dans une **dead zone** où aucune entrée ne réactive la diversité. Le phénomène est réel ; sa cause n'est **pas** spectrale — à degré fixé, faire varier λ₂ d'un facteur ~27 ne change pas le régime. La séparation en λ₂ ≈ 2.31 observée sur le jeu initial (BA/ER) est une frontière **corrélationnelle** : λ₂ y covarie avec le degré.

---

## Glossaire (8 termes essentiels)

| Terme | Définition |
|-------|------------|
| **u** | Variable de doute constitutionnel ∈ [0,1]. Module la polarité du couplage. Le mécanisme central du modèle. |
| **Hérétique** | Nœud avec u ≈ 0 ou u ≈ 1. S'oppose activement au consensus local. Ratio typique : 15%. |
| **Dead zone** | Régime BA m≥5 où H_cog ≈ 0 quelle que soit l'entrée. Gouvernée par le **degré de couplage** (k_harm ≈ 6) via un mécanisme d'échantillonnage en champ moyen — **pas** par λ₂. L'ancien nom « dead zone *spectrale* » est abandonné : il nommait une cause réfutée. |
| **FULL** | Configuration normale — u dynamique actif. |
| **FROZEN_U** | Ablation — u gelé à sa valeur initiale. Sert de baseline. Corrélation moyenne entre trajectoires : 0.007 (FULL) → 0.658 (FROZEN_U), séparation complète sur 30 graines. |
| **H_cog** | Entropie cognitive (5 bins) — mesure la diversité des états. H_cog > 0 = réseau fonctionnel. |
| **H_cont** | Entropie continue (100 bins) — mesure plus fine, utile pour comparaisons cross-conditions. |
| **Levitating Sigmoid** | `w(u) = tanh(π(0.5−u)) + δ`. Remplace la fonction linéaire (1−2u) — élimine la singularité à u=0.5. |

---

## 5 findings historiques — 4 tiennent, 1 est réfuté

> Les numéros d'origine sont conservés : on ne renumérote pas une liste dont un membre est mort,
> sinon les renvois anciens pointent vers le mauvais résultat.

| # | Finding | Chiffre clé | Script |
|---|---------|-------------|--------|
| 1 | **Dead zone** — un seuil de **degré de couplage** sépare réseaux fonctionnels et morts (k_harm ≈ 6) ; la séparation en λ₂ ≈ 2.31 du jeu initial est **corrélationnelle**, pas causale | à degré fixé, λ₂ varie ×27 sans changer le régime | `experiments/lambda2_foundation_20260701/` (versionné le 05/08) |
| 2 | **Intelligence topologique** — dans FULL, les hubs ont des trajectoires plus structurées (u couple complexité ↔ connectivité) | r=−0.716, p=1.29e-79 (BA m=5) | `lz_per_node.py` |
| 3 | ⚫ **RÉFUTÉ — Transition événementielle** (voir la note sous ce tableau) | 0/9 configurations positives au rejeu | `experiments/event_phase_transition_rerun_20260711.py` (versionné le 05/08) |
| 4 | **Chimère — classe distincte** — R=0.141 (Mem4ristor) vs R=0.766 (Abrams-Strogatz 2004) | Deux mécanismes séparables | `reviewer2_chimera_comparison.py` |
| 5 | **u = filtre anti-synchronisation** — bloquer u transforme le réseau en meute synchronisée | corrélation 0.007 → 0.658, séparation complète (Cohen d ≈ 9, 30 graines) | `p2_sigma_social_ablation.py` |

> ⚫ **Finding 3 — RÉFUTÉ le 11/07/2026.** Le résultat d'avril — *« forcer un nœud périphérique
> produit +1.20 bits, forcer un hub +0.21 »* — était un artefact du bruit antérieur au 1er mai
> (AUDIT-024, Euler-Maruyama ×4.47). Grille complète rejouée au protocole d'avril **inchangé**
> (`experiments/event_phase_transition_rerun_20260711.py`) : **0/9 configurations du claim
> positives, 9/9 négatives** — l'événement *dégrade* H_cont (≈ −1.0) sur BA m=3, pour le hub
> comme pour la périphérie ; effet ~nul sur la dead zone m=5. **Le preprint ne cite pas ce
> résultat ; la soumission n'est pas affectée.** *Idée originale de Julien Chauvin — conservée
> ici parce qu'une idée réfutée reste une idée qui a été testée.*
>
> ✅ **Rejouable par un clone depuis le 05/08/2026.** Le script du rejeu, le protocole d'avril
> qu'il importe (`experiments/event_phase_transition.py`) et les CSV de référence des deux
> campagnes sont versionnés. Ils vivaient dans `experiments/scratch/` et `figures/scratch/`,
> gitignorés : la réfutation était affirmée partout et vérifiable nulle part.

---

## Structure des dossiers

| Dossier | Rôle | Navigation |
|---------|------|------------|
| `src/mem4ristor/` | Package principal — core, dynamics, metrics, topology, config | [FOLDER_SUMMARY.md](src/mem4ristor/FOLDER_SUMMARY.md) |
| `experiments/` | 69 scripts d'expérience classés par tier (1=finding, 2=validation, 3=robustesse) | [FOLDER_SUMMARY.md](experiments/FOLDER_SUMMARY.md) |
| `docs/` | preprint.tex (Paper 1), paper_2/, results_compendium/ | [FOLDER_SUMMARY.md](docs/FOLDER_SUMMARY.md) |
| `figures/` | Outputs CSV + PNG générés par experiments/ | [FOLDER_SUMMARY.md](figures/FOLDER_SUMMARY.md) |
| `tests/` | 153 tests — exit 0, 2 `xfail` documentés | — |
| `archives/` | Code historique — ne pas modifier | [FOLDER_SUMMARY.md](archives/FOLDER_SUMMARY.md) |

---

## État actuel (2026-08-02)

> Ce bloc datait du 2026-05-02 et était faux sur six lignes (version, branche, tests, DOI,
> statut de publication, jalon). Corrigé le 02/08/2026 — audit externe B8, valeurs revérifiées
> le jour même. **Source de vérité en cas de doute : `PROJECT_STATUS.md`.**

- **Version du code** : V6.0.0 (`VERSION`, `pyproject.toml`)
- **Dernière version déposée sur Zenodo** : **V4.0.0** (2026-05-02, DOI de version
  `10.5281/zenodo.19986042`). **V5 et V6 ne sont pas déposées.**
- **Branche active** : `main`
- **Tests** : 153 collectés, `pytest` exit 0 (2 `xfail` documentés dans `test_adversarial.py`)
- **DOI** : 10.5281/zenodo.18620596 — *concept DOI*, résout toujours vers la dernière version déposée
- **GitHub** : https://github.com/cafe-virtuel/Mem4ristor
- **Papiers** : preprint **non soumis et non publié sur Zenodo**
  (`docs/papers/preprint/preprint.tex` → 27 p) · paper_2 en préparation · paper_B (hardware) en préparation
- **Prochain jalon** : voir `PROJECT_STATUS.md` §0

---

## Origine

Né au **Café Virtuel** — méthode de collaboration humain + IAs (Anthropic, OpenAI, xAI, Google, Mistral, DeepSeek). Orchestré par Julien Chauvin (non-chercheur). Citation fondatrice (Grok, 19/08/2025) : *"Ce soir, nous avons prouvé que 5 IA + 1 barman > somme des parties."*
