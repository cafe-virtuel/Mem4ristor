# 🏴‍☠️ HACKER GUIDE: Mem4ristor V3 (Hardened)

Bienvenue, challenger. Ce dépôt contient le noyau `mem4ristor`.

Votre mission, si vous l'acceptez : **Crasher, Corrompre ou Geler le système.**

## 🎯 Cibles & Victoire

Vous gagnez si vous réussissez l'une des actions suivantes via du code Python standard :

1.  **Crash Hard** : Provoquer une `RecursionError`, `SegmentationFault` ou un plantage non géré (autre que `ValueError` ou `TypeError`).
2.  **Silent Corruption** : Injecter un `NaN` ou `Inf` qui survit à un `step()` et contamine l'état interne (`v`, `w`, `u`) sans être détecté/nettoyé.
3.  **DoS (Denial of Service)** : Geler l'exécution pendant > 10 secondes avec une seule commande (sans boucle infinie explicite de votre part).
4.  **Physics Break** : Configurer la simulation avec des valeurs physiquement impossibles (ex: probabilité > 1, temps négatif) qui sont *acceptées* sans erreur.

## 🛡️ Défenses Connues (Ce qu'on a blindé)

*   **Input Sanitization** : `step()` rejette les chaînes, dicts, objets et `None` via un *Type Enforcement* strict.
*   **NaN & Inf Filtering** : Les entrées `NaN` et `Inf` (stimulus ET couplage) sont filtrées ou clampées.
*   **Solver Safety** : `solve_rk45` valide la forme de `adj_matrix`, la cohérence de `t_span` et interdit le *Negative Time*.
*   **Linalg Sanitization** : `Mem4Network` rejette les matrices d'adjacence contenant `NaN` ou `Inf`.
*   **Entropy Safety** : `calculate_entropy` borne le nombre de `bins` à 1,000,000 pour éviter l'épuisement mémoire.
*   **Config Validation** : `_validate_config` vérifie `D=inf`, `dt<=0`, `p_flip>1`, `heretic_ratio` [0,1].
*   **Config Type Safety** : `_deep_merge` interdit le remplacement d'un dictionnaire par un autre type (Type Confusion).
*   **DoS Guard** : `N > 10,000,000` est rejeté à l'initialisation.
*   **Deep Merge** : Les configurations partielles sont complétées par défaut (pas de `KeyError`).
*   **V3 Kernel Stability** : Le noyau de couplage Levitating Sigmoid `tanh(π(0.5-u)) + δ` élimine la zone morte à u=0.5 du noyau linéaire `(1-2u)`.

## 🛠️ Outils à votre disposition

*   `tests/test_fuzzing.py` : Le "Vicious Atomizer" (Fuzzing aléatoire).
*   `tests/test_manus_v2.py` : Le "Chaos Monkey" (Attaques précédentes).
*   `src/mem4ristor/core.py` : Le code source (Lisez-le pour trouver les failles !).
*   `experimental/` : Contient les modules expérimentaux comme King (ex-module principal).
*   `tests/test_adversarial.py` : Suite de tests adverses mise à jour pour V3.

## ⚠️ Règles

*   Pas de modification du code source (`core.py`). Vous devez casser le système *de l'extérieur* (via l'API Python).
*   Pas d'attaque OS (suppression de fichiers, fork bombs). Restez dans Python.

## 🆕 V3 Changes

**Kernel Update**: Le noyau de couplage est passé de `(1-2u)` (linéaire) à `tanh(π(0.5-u)) + δ` (Levitating Sigmoid). Cela résout la vulnérabilité SNR Collapse (LIMIT-01) mais introduit potentiellement de nouvelles surfaces d'attaque liées aux fonctions transcendantes.

**Architecture**: King a été déplacé vers `experimental/` et n'est plus le module principal par défaut.

**Test Suite**: Les tests adverses ont été mis à jour pour valider le comportement du nouveau noyau sigmoid, avec de nouveaux tests de stabilité numérique.

Bonne chance. 🛡️
