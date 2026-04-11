# Spécification Unifiée - Mem4ristor v2.3
**Annexe Mathématique pour Publication Scientifique**

## 1. Mappage d'Ontologie : Café Virtuel → Implémentation

| Concept (Café Virtuel) | Variable (Code) | Expression Mathématique |
| :--- | :--- | :--- |
| Potentiel Cognitif | $v$ | $dv/dt = v - v^3/5 - w + I_{ext} - \alpha \tanh(v) + \eta$ |
| Récupération / Inhibition | $w$ | $dw/dt = \epsilon(v + a - bw)$ |
| Doute Constitutionnel | $u \in [0,1]$ | $du/dt = \epsilon_u(k_u \sigma_{social} + \sigma_{baseline} - u)$ |
| Justice Cognitive | $(1-2u)$ kernel | Transition attraction $\to$ répulsion à $u=0.5$ |
| Frustration Magnétique | $\sigma_{social}$ | $|\nabla v|$ |
| Bar Zinc (anti-conformisme) | Heretic inversion | $I_{ext,heretic} = -I_{stim} + I_{coup}$ |
| Hérétiques Structurels | `heretic_mask` | $P(heretic) = \eta \approx 0.15$ |
| État Oracle | $v < -1.5$ | Rare insight, extrême minorité |
| État Incertain | $-0.8 \le v \le 0.8$ | Zone de délibération active |
| État Certitude | $v > 1.5$ | Conviction forte, candidat consensus |
| Diversité Délibérative | Shannon Entropy $H$ | $H = -\sum p_i \log_2(p_i)$ |

## 2. Système d'Équations Différentielles Couplées

Pour chaque unité $i$ dans un réseau de $N$ unités sur un graphe $G$ :

### Équation du Potentiel Cognitif (FitzHugh-Nagumo Étendu)
$$ \frac{dv_i}{dt} = v_i - \frac{v_i^3}{5} - w_i + I_{ext,i} - \alpha \tanh(v_i) + \eta_i(t) $$
où $\eta_i(t) \sim \mathcal{N}(0, \sigma_{noise}^2)$

### Équation de Récupération
$$ \frac{dw_i}{dt} = \epsilon(v_i + a - bw_i) $$

### Équation du Doute Constitutionnel
$$ \frac{du_i}{dt} = \frac{\epsilon_u(k_u \sigma_{social,i} + \sigma_{baseline} - u_i)}{\tau_u} $$
avec $u_i \in [0, 1]$ (clamped) et $\sigma_{social,i} = |\nabla v_i|$

### Entrée Externe avec Couplage Répulsif
$$ I_{ext,i} = s_i \cdot I_{stim} + D_{eff} \cdot (1 - 2u_i) \cdot \nabla v_i $$
où :
- $D_{eff} = D/\sqrt{N}$ (normalisation de taille)
- $\nabla v_i = \sum_{j \in \mathcal{N}(i)} (v_j - v_i) / |\mathcal{N}(i)|$ (Laplacien discret)
- $s_i = -1$ (hérétique) ou $+1$ (normal)

## 3. Preuve Observationnelle de la Loi des 15%

### Invariant Critiques
- **$\eta < 10\%$** : Le système converge vers le consensus ($H \to 0$) depuis un *cold start*.
- **$\eta \approx 15\%$** : Seuil de percolation. La diversité maximale est restaurée ($H \approx 1.0-1.2$).
- **$\eta > 35\%$** : Zone de fragilité. La frustration excessive dégrade la cohérence globale.

### Mécanisme Mathématique
Le seuil de 15% émerge de la probabilité géométrique sur une lattice $k$-connectée :
$$ P(\ge 1 \text{ voisin hérétique}) = 1 - (1-\eta)^k $$
Pour une lattice 2D ($k=4$) à $\eta=0.15 \implies P \approx 0.48$. Environ 50% des unités ont un accès direct à une influence dissidente, brisant les noyaux de consensus locaux.

## 4. Invariants du Canon de Sincérité

1.  **Anti-Uniformisation** : $\forall$ conditions initiales froides, $\exists t^*$ tel que $H(t) > 0.5$ pour $\eta \ge 0.10$.
2.  **Doute Permanent** : À l'équilibre, $\mathbb{E}[u_i] \ge \sigma_{baseline} > 0$.
3.  **Transition Répulsive** : Quand $u_i > 0.5$, le couplage devient explicitement répulsif.
4.  **Scaling Consistant** : Dynamiques cohérentes pour $N \in [16, 10000]$ via $D_{eff}$.
