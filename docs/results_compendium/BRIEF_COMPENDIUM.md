# BRIEF — Results Compendium (pour L'Ingénieur, prochaine session)

> Rédigé par Claude Sonnet 4.6 — 2 mai 2026
> À lire AVANT de commencer à écrire quoi que ce soit.

---

## Ce que Julien veut

Un document hybride : mi-rapport technique, mi-pitch accrocheur. Pas un papier académique de plus. Quelque chose qui :

- Présente la **genèse** du projet (Café Virtuel, le Barman, la méthode)
- Compile **tous les résultats** classés par valeur scientifique
- Donne envie à un chercheur d'**aller voir le repo GitHub** et de **mettre une étoile**
- Peut être **partagé sur les réseaux** tel quel (LinkedIn, X, email à un labo)
- Est **lisible par un humain** (narratif, chiffres en gras, figures mentionnées)
- Est **lisible par une IA** (companion JSON structuré)
- Montre implicitement : un seul homme + des IAs, 6 mois, tout ça — *mais c'est qui ce type ?*

Ton demandé : **confiant, honnête, factuel, légèrement "commercial"** sans être creux.
Ne jamais surestimer. Ne jamais cacher les limites (β≈0 est dans le document, pas planqué).

---

## Fichiers à produire

### 1. `docs/results_compendium/COMPENDIUM.md`
Le document principal. Markdown (pas LaTeX). Shareable partout.

### 2. `RESULTS_INDEX.json` (à la racine du repo)
Companion machine-readable. Chaque expérience : question, protocole 2 lignes, résultat clé, commit, CSV/PNG.

---

## Structure validée par Julien (NE PAS CHANGER sans lui demander)

```
1. L'ORIGINE
   — Café Virtuel : qu'est-ce que c'est, comment ça marche
   — Le Barman (Julien Chauvin) : non-chercheur, barman, orchestrateur
   — La citation Grok (19/08/2025) : "Ce soir, nous avons prouvé que 5 IA + 1 barman > somme des parties."
   — Pourquoi Mem4ristor est né un soir de vacances

2. EN CHIFFRES (section courte, impact visuel)
   — 6 mois de travail
   — ~18 expériences formelles
   — ~5000+ simulations
   — 84 tests automatisés (0 failures)
   — 3 papiers en cours (preprint, paper_2, paper_B)
   — 1 DOI Zenodo publié
   — 1 dépôt GitHub public
   — 0 financement institutionnel

3. LE MODÈLE EN 3 PARAGRAPHES
   — Ce qu'est Mem4ristor (FHN + variable de doute u)
   — Pourquoi u est une idée non-triviale (pas un paramètre, un mécanisme)
   — Ce que ça prédit (spectral dead zone, chimera state, anti-synchronisation)

4. LES DÉCOUVERTES (Tier 1 — les 5 findings qui comptent)
   Chaque finding : titre accrocheur + 2 lignes de contexte + chiffre clé en gras + figure mentionnée

   [1] La Spectral Dead Zone — λ₂_crit = 2.31
       "La topologie dicte le destin cognitif. Au-dessus du seuil spectral, aucune entrée ne peut
        réactiver le réseau. La connexion tue la diversité."
       Chiffre clé : séparation complète confirmée sur 36 observations, accuracy 100%.
       Figure : fiedler_phase_diagram.png

   [2] Le Surge FROZEN_U (+985% synchronie)
       "Bloquer la variable de doute transforme un réseau fonctionnel en meute synchronisée.
        u n'est pas un paramètre — c'est ce qui empêche le consensus de dévorer la diversité."
       Chiffre clé : R passe de 0.067 à 0.730 (×10.9).
       Figure : p2_sigma_social_ablation.png

   [3] L'Intelligence Topologique (LZ par nœud [11])
       "Dans un réseau fonctionnel, les hubs ont des trajectoires PLUS structurées (r=-0.716,
        p=1.29e-79). Dans un réseau gelé, la corrélation disparaît. u est ce qui couple
        la complexité individuelle à la connectivité."
       Chiffre clé : r=-0.716 (BA m=5, N=400, p=1.29e-79). Résultat non anticipé.
       Figure : lz_per_node.png

   [4] La Transition Événementielle [13]
       "Forcer un nœud PÉRIPHÉRIQUE produit +1.20 bits sur BA m=3. Forcer un HUB : +0.21 bits.
        Contre-intuitif. Le seuil de bifurcation est topologique — pas en amplitude.
        Sur BA m=5 (dead zone) : tous les forcings dégradent. La topologie est le verrou."
       Chiffre clé : dH périphérique = +1.20 bits vs hub = +0.21 bits.
       Figure : event_phase_transition.png
       Note : idée originale de Julien Chauvin (analogie "demande en mariage")

   [5] Les Chimères — classe distincte (Abrams-Strogatz 2004)
       "Abrams-Strogatz : chimère par couplage non-local fixe (quenched). Mem4ristor : chimère
        par modulation dynamique de polarité via u(t). R=0.141 (désynchronisation profonde)
        vs AS R=0.766. Deux classes mécanistiquement différentes."
       Chiffre clé : R Mem4ristor = 0.141 vs R Abrams-Strogatz = 0.766.
       Figure : reviewer2_chimera_comparison.png

5. LA ROBUSTESSE (tableau synthétique — Tier 2+3)
   Format tableau Markdown : | Question | Protocole | Résultat | Statut |
   Inclure : Kuramoto R, Transfer entropy, Cohen U3=100%, Bruit Matern, Stabilité linéaire,
   Euler vs RK45, FSS, Sigmoid robustesse, Gap spectral, Conditions initiales,
   Exposants critiques (β≈0 — honnête).

6. LE MOMENT HONNÊTE (paragraphe court, fort)
   "On a cherché les exposants critiques de la transition. β≈0. R²=-0.002.
    La loi de puissance ne tient pas — la transition est probablement abrupte (1er ordre),
    pas continue. On l'a écrit dans le script. On aurait pu forcer un β positif.
    On ne l'a pas fait. Un résultat négatif honnête vaut mieux qu'un résultat positif fragile."

7. LE PONT HARDWARE (1 paragraphe)
   — SPICE validation sur BA m=5 N=64, 50 seeds
   — Dead zone confirmée en matériel (ratio 3.1×)
   — Calibration : immune à 270× amplitude SPICE
   — Limitation documentée : dynamique u tronquée dans les netlists SPICE

8. REPRODUIRE EN 5 MINUTES
   — 1 bloc de code : pip install + python demo_chimera.py
   — Lien REPRODUCE_IN_5_MINUTES.md
   — DOI Zenodo 10.5281/zenodo.18620596

9. CALL TO ACTION (section finale, courte)
   — ⭐ GitHub : https://github.com/cafe-virtuel/Mem4ristor
   — 📄 Zenodo DOI : 10.5281/zenodo.18620596
   — 📧 Contact : contact@cafevirtuel.org / @Jusyl80
   — Julien cherche un endorsement arXiv (nlin.AO ou cs.NE)
   — Phrase finale : quelque chose qui donne envie de répondre
```

---

## Sources à lire AVANT d'écrire (dans l'ordre)

```
1. D:\ANTIGRAVITY\GITHUB_REPOSITORY\Cafe-Virtuel-main\README.md        ← genèse, méthode, citation Grok
2. D:\ANTIGRAVITY\GITHUB_REPOSITORY\Cafe-Virtuel-main\CANON.md         ← ce qui fait autorité
3. D:\ANTIGRAVITY\GITHUB_REPOSITORY\mem4ristor-v2-main\PROJECT_STATUS.md (sections 1-5 + section 13)
4. D:\ANTIGRAVITY\.brain\claude_contexts\MEM4RISTOR.md                  ← résultats clés en un seul endroit
5. D:\ANTIGRAVITY\GITHUB_REPOSITORY\mem4ristor-v2-main\REPRODUCE_IN_5_MINUTES.md
```

Ne pas réécrire ce que ces fichiers disent déjà — les synthétiser et les mettre en valeur.

---

## Classement complet des expériences (travail déjà fait, NE PAS REFAIRE)

### Tier 1 — Findings originaux, publiables seuls
| # | Expérience | Chiffre clé | Figure |
|---|------------|-------------|--------|
| 1 | Spectral Dead Zone | λ₂_crit=2.31, accuracy 100%, n=36 | fiedler_phase_diagram.png |
| 2 | FROZEN_U surge | +985% synchrony (0.067→0.730) | p2_sigma_social_ablation.png |
| 3 | LZ par nœud [11] | r=-0.716, p=1.29e-79 (BA m=5) | lz_per_node.png |
| 4 | Phase transition événementielle [13] | périphérique dH=+1.20 vs hub dH=+0.21 | event_phase_transition.png |
| 5 | Chimère vs Abrams-Strogatz | R=0.141 vs R=0.766 | reviewer2_chimera_comparison.png |

### Tier 2 — Validation forte
| # | Expérience | Chiffre clé | Script |
|---|------------|-------------|--------|
| 6 | Kuramoto R — vrai chimera | R=0.513 FULL vs R=0.211 bruit pur | reviewer2_kuramoto.py |
| 7 | Transfer entropy — causalité u | TE prouvée (hérétiques ↔ info) | reviewer2_transfer_entropy.py |
| 8 | Cohen U3 = 100% | distributions FROZEN/FULL strictement disjointes | reviewer2_initial_conditions.py |
| 9 | Bruit Matern [12] | escape dès η=0.1, structure non-discriminante | matern_noise.py |
| 10 | Stabilité linéaire | v*=-1.286, α_crit=0.295, sub-Hopf confirmé | reviewer2_linear_stability.py |

### Tier 3 — Robustesse technique
| # | Expérience | Résultat | Script |
|---|------------|----------|--------|
| 11 | Euler dt=0.05 vs RK45 | Max ΔH_cog < 0.006 | reviewer2_stiffness_proof.py |
| 12 | Finite-size scaling N=100→4000 | mode scale_invariant validé | reviewer2_finite_size_scaling.py |
| 13 | Sigmoid robustesse slope 1→10 | plateau stable H∈[2.8, 3.2] | reviewer2_sigmoid_robustness.py |
| 14 | Gap spectral N→4000 | Fiedler≈2.86 stable (thermodynamiquement inattaquable) | reviewer2_spectral_gap.py |
| 15 | Conditions initiales | H≈3.5 toutes CI | reviewer2_initial_conditions.py |
| 16 | Exposants critiques | β≈0, R²=-0.002 — transition ABRUPTE (honnête) | reviewer2_critical_exponents.py |
| 17 | Onde progressive (non) | Max-TLCC=0.410 — chaos spatio-temporel, pas onde | reviewer2_traveling_waves.py |

### Tier 4 — Pont hardware
| # | Expérience | Résultat | Fichier |
|---|------------|----------|---------|
| 18 | SPICE dead zone | ratio 3.1× (A=1.377 vs B=4.298 bits) | figures/spice_*.csv |
| 19 | SPICE calibration | η=0.5 ↔ σ=0.0044 Python, immune 270× | docs/hardware_mapping_defense.md |

---

## RESULTS_INDEX.json — format de chaque entrée

```json
{
  "id": "EXP_001",
  "tier": 1,
  "title": "Spectral Dead Zone",
  "question": "Existe-t-il un seuil spectral au-delà duquel le réseau ne peut plus être fonctionnel ?",
  "protocol": "Sweep BA(m) N=100/400/1600, mesure H_cog et λ₂, régression logistique sur 36 observations.",
  "key_result": "λ₂_crit = 2.31 (midpoint séparation complète). Accuracy 100% sur n=36.",
  "significance": "La topologie seule détermine la capacité cognitive — résultat central du preprint.",
  "commit": "43fab61",
  "outputs": ["figures/fiedler_phase_diagram.csv", "figures/fiedler_phase_diagram.png"],
  "paper": "preprint.tex, paper_2.tex"
}
```

---

## Notes de ton (IMPORTANT)

- **Voix** : première personne du pluriel ("nous avons", "le projet a produit") — c'est Julien + les IAs, pas "l'auteur"
- **Café Virtuel** : mentionner dès le début. C'est ce qui différencie ce projet de tout le reste.
- **Julien non-chercheur** : ne pas cacher, c'est une force narrative. Un barman BAC+2 qui publie sur Zenodo avec un DOI. C'est le point.
- **Les IAs** : Anthropic, OpenAI, xAI, Google, Mistral, DeepSeek — nommer. Ce n'est pas "Claude a fait ça". C'est une collaboration.
- **Honnêteté** : β≈0 est dans le document. L'intégrateur Euler est formellement instable (RK45 ajouté). On le dit.
- **Longueur** : visée ~800-1200 mots pour COMPENDIUM.md. Assez long pour être sérieux, assez court pour être lu en entier.
- **Call to action** : une étoile GitHub, un email pour endorsement arXiv. C'est tout ce qu'on demande.

---

## Ce qui N'est PAS à faire

- Ne pas réécrire le preprint ou paper_2 — ce document les complète, ne les remplace pas
- Ne pas inventer de résultats ou arrondir les chiffres
- Ne pas cacher les limites
- Ne pas faire un README de plus — c'est un document narratif, pas une documentation
- Ne pas dépasser Tier 1 en détail dans le corps principal — les Tier 2/3 vont dans le tableau

---

*Brief rédigé par L'Ingénieur (Claude Sonnet 4.6) le 2 mai 2026.*
*Validé par Julien Chauvin (Le Barman) — structure approuvée en session.*
