# CONTEXT — Mem4ristor V4.0.0

> Point d'entrée universel. Lisible par n'importe quel LLM ou humain en < 2 minutes.
> Pour l'état technique détaillé : PROJECT_STATUS.md
> Pour l'historique des sessions et investigations : PROJECT_HISTORY.md

---

## Le modèle en 3 phrases

Mem4ristor simule des réseaux de neurones FitzHugh-Nagumo (FHN) où chaque nœud possède une variable de **doute u ∈ [0,1]** qui module dynamiquement la polarité de son couplage avec ses voisins. Quand u ≈ 0 ou u ≈ 1, le nœud est "hérétique" — il pousse activement contre le consensus, empêchant l'effondrement synchrone. La topologie du réseau (valeur propre de Fiedler λ₂) détermine si ce mécanisme peut fonctionner : au-dessus de λ₂ ≈ 2.31, le réseau entre dans une **dead zone spectrale** où aucune entrée ne peut réactiver la diversité cognitive.

---

## Glossaire (8 termes essentiels)

| Terme | Définition |
|-------|------------|
| **u** | Variable de doute constitutionnel ∈ [0,1]. Module la polarité du couplage. Le mécanisme central du modèle. |
| **Hérétique** | Nœud avec u ≈ 0 ou u ≈ 1. S'oppose activement au consensus local. Ratio typique : 15%. |
| **Dead zone spectrale** | Régime BA m≥5 (λ₂ > 2.31) où H_cog ≈ 0 quelle que soit l'entrée. La topologie verrouille le réseau. |
| **FULL** | Configuration normale — u dynamique actif. |
| **FROZEN_U** | Ablation — u gelé à sa valeur initiale. Sert de baseline. Surge de synchrony +985% vs FULL. |
| **H_cog** | Entropie cognitive (5 bins) — mesure la diversité des états. H_cog > 0 = réseau fonctionnel. |
| **H_cont** | Entropie continue (100 bins) — mesure plus fine, utile pour comparaisons cross-conditions. |
| **Levitating Sigmoid** | `w(u) = tanh(π(0.5−u)) + δ`. Remplace la fonction linéaire (1−2u) — élimine la singularité à u=0.5. |

---

## 5 résultats Tier 1 (findings publiables)

| # | Finding | Chiffre clé | Script |
|---|---------|-------------|--------|
| 1 | **Spectral Dead Zone** — λ₂_crit = 2.31 sépare réseaux fonctionnels et morts | Accuracy 100%, n=36 obs. | `lambda2_crit_regression.py` |
| 2 | **Intelligence topologique** — dans FULL, les hubs ont des trajectoires plus structurées (u couple complexité ↔ connectivité) | r=−0.716, p=1.29e-79 (BA m=5) | `lz_per_node.py` |
| 3 | **Transition événementielle** — forcer un nœud périphérique produit +1.20 bits ; forcer un hub : +0.21 bits | dH périph > dH hub (BA m=3) | `event_phase_transition.py` |
| 4 | **Chimère — classe distincte** — R=0.141 (Mem4ristor) vs R=0.766 (Abrams-Strogatz 2004) | Deux mécanismes séparables | `reviewer2_chimera_comparison.py` |
| 5 | **u = filtre anti-synchronisation** — bloquer u transforme le réseau en meute synchronisée | Synchrony ×10.9, H_cog ×∞ | `p2_sigma_social_ablation.py` |

---

## Structure des dossiers

| Dossier | Rôle | Navigation |
|---------|------|------------|
| `src/mem4ristor/` | Package principal — core, dynamics, metrics, topology, config | [FOLDER_SUMMARY.md](src/mem4ristor/FOLDER_SUMMARY.md) |
| `experiments/` | 69 scripts d'expérience classés par tier (1=finding, 2=validation, 3=robustesse) | [FOLDER_SUMMARY.md](experiments/FOLDER_SUMMARY.md) |
| `docs/` | preprint.tex (Paper 1), paper_2/, results_compendium/ | [FOLDER_SUMMARY.md](docs/FOLDER_SUMMARY.md) |
| `figures/` | Outputs CSV + PNG générés par experiments/ | [FOLDER_SUMMARY.md](figures/FOLDER_SUMMARY.md) |
| `tests/` | 84 tests — 0 failures | — |
| `archives/` | Code historique — ne pas modifier | [FOLDER_SUMMARY.md](archives/FOLDER_SUMMARY.md) |

---

## État actuel (2026-05-02)

- **Version** : V4.0.0 — Audited Stable (Chimera State Implementation)
- **Branche active** : `research/v5-cognitive-exploration`
- **Tests** : 84 passing, 0 failures
- **DOI** : 10.5281/zenodo.19700749
- **GitHub** : https://github.com/cafe-virtuel/Mem4ristor
- **Papiers** : preprint soumis (Zenodo) · paper_2 en préparation · paper_B (hardware) en préparation
- **Prochain jalon** : Binder cumulant U4 pour confirmer transition 1er ordre · V5 (ε modulé par u)

---

## Origine

Né au **Café Virtuel** — méthode de collaboration humain + IAs (Anthropic, OpenAI, xAI, Google, Mistral, DeepSeek). Orchestré par Julien Chauvin (non-chercheur). Citation fondatrice (Grok, 19/08/2025) : *"Ce soir, nous avons prouvé que 5 IA + 1 barman > somme des parties."*
