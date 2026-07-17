> # ⛔ DRAFT RETIRÉ — NE PAS INTÉGRER AU PREPRINT / WITHDRAWN DRAFT — DO NOT INTEGRATE
> **Date de retrait : 2026-07-17.**
>
> Ce brouillon est conservé pour la traçabilité, **pas comme position du projet**. Il
> contredit la science corrigée du preprint sur trois points, et sa mention
> « Validée par la Red Team » (ci-dessous) est **factuellement fausse** :
>
> 1. **Binder / ordre de la transition.** Ce texte affirme une « continuous (second-order)
>    phase transition ». Le preprint (§ Binder, `preprint.tex`) et le COMPENDIUM corrigé
>    (commit `7057ac8`, 2026-07-14) concluent l'**inverse** : U₄ plat ≈ 2/3 → **crossover
>    lisse, PAS une transition thermodynamique**. C'est aussi la lignée du β≈0 (mai 2026).
> 2. **λ₂ causal.** Ce texte réintroduit λ₂_crit ≈ 2.31 comme seuil causal gouverné par une
>    « Kirchhoff ART ». La réfutation du 2026-07-01 (`experiments/scratch/lambda2_foundation_20260701/`)
>    et la reformulation du 2026-07-06 ont établi que le mécanisme est le **champ moyen /
>    degré harmonique (k_harm≈6)** ; 2.31 est une **frontière corrélationnelle**, pas causale.
> 3. **ART présentée comme gouverneur du régime.** L'ART (Kirchhoff) est **opt-in et
>    `enabled: false` par défaut** (`src/mem4ristor/config.yaml`) ; les mesures canoniques de
>    la dead zone tournent **sans elle**. Elle ne peut pas être le « homeostatic governor »
>    du régime fonctionnel.
>
> **La vraie Red Team avait déjà rejeté cette thèse** : AUDIT-020 (Hermès/EDISON, 2026-06-01,
> `AUDIT_LOG.md`) — « CONFIRMED... The mechanism is feedback control / homeostatic
> regulation », recommandation « retirer Active Inference, dégrader §6.3 ». Le preprint a
> suivi cette recommandation (§6.3 retirée ; Friston reste en bibliographie seulement).
>
> **Si Friston doit un jour être évoqué :** paragraphe de Discussion *explicitement
> analogique* (« la boucle u ressemble à de l'inférence active au sens faible »), SANS λ₂
> causal, SANS second ordre, avec le mécanisme dans le bon sens — OU une dérivation formelle
> complète (u̇ ∝ −∂F/∂u pour une énergie libre variationnelle explicite), qui n'existe pas
> à ce jour.
>
> ---
> *Le texte original du brouillon suit, inchangé, à titre d'archive.*

---

# Draft V2 : Thermodynamic Viability and Active Inference (Friston Free Energy)

## [PROPOSITION D'INTÉGRATION - ⚠️ étiquette d'origine ERRONÉE, voir bandeau de retrait ci-dessus]

*Ce texte était destiné à être intégré dans la Section 5 (Discussion) du preprint. Retiré le 2026-07-17 — voir bandeau.*

---

### Thermodynamic Viability and Active Inference

A fundamental question regarding the observed "Spectral Dead Zone" is whether the collapse of network dynamics represents a mere computational artifact or a physical, thermodynamic reality of the coupled system. Our results suggest the latter. By interpreting the network through the lens of Karl Friston’s Free Energy Principle (FEP) and Active Inference, the parameter $u_i(t)$ (constitutional doubt) can be formally understood as an active mechanism for surprise minimization.

In the physical interpretation of the FEP framework, self-organizing systems maintain their structural and functional integrity (their "viability") by continually minimizing their variational free energy, which serves as an upper bound on sensory surprise. In the Mem4ristor network, the absolute value of the local Laplacian $|\mathcal{L}_v|_i$ acts as the sensory prediction error, which under the FEP drives the minimization of surprise for node $i$ relative to its topological neighborhood.

The dynamics of $u_i$ (Eq. 4) are explicitly driven by this prediction error:
$$ \tau_u \dot{u}_i = \epsilon_u(t) \left( k_u |\mathcal{L}_v|_i + \sigma_{baseline} - u_i \right) $$

When a node experiences high conflict with its neighbors (high $|\mathcal{L}_v|_i$), its prediction error increases, driving $u_i \to 1$. This increase in $u_i$ locally reduces the coupling strength $D_{eff}$, effectively decoupling the node from the source of the conflict. This dynamic can be formally interpreted as a form of Active Inference: the node alters its effective coupling with the environment—the network topology—to minimize future prediction errors, without invoking an explicit generative model.

It is crucial to note that this active inference mechanism is not imposed externally, but emerges from the topological self-regulation of the network. As established in Section [X] (Kirchhoff ART), the effective algebraic connectivity $\lambda_2$ is not a fixed parameter but a dynamical variable constrained by the Kirchhoff flow. In the functional regime, the ART acts as a homeostatic governor: it maintains $\lambda_2$ below the critical threshold by locally adjusting the weights of conflicted edges, thereby preserving the network's capacity for localized surprise minimization. 

Yet, the FEP also implies that this homeostasis has physical limits: active inference is only viable within a bounded region of the state space. When the density of topological constraints overwhelms the dissipative capacity of the local $u_i$-mediated decoupling (i.e., when $\lambda_2$ exceeds the numerically observed critical threshold $\lambda_{2,crit} \approx 2.31$), the pervasive, non-local influences override the local capacity for active inference. The prediction error $|\mathcal{L}_v|$ becomes uniformly high across the network, forcing $u_i \to 1$ globally.

We term this global saturation a **thermodynamic collapse**: the network undergoes a transition into a state of maximal rigidity where the continuous entropy production drops to a residual baseline ($H \to H_{\text{res}} \approx 2.4$), analogous to a glass-like arrest of its functional degrees of freedom. This is rigorously supported by the Binder cumulant analysis in Section [X], which confirms the emergence of a sharp finite-size transition, with the $U_4$ minimum converging toward $\lambda_{2,\text{crit}}$ as $N \to \infty$, characteristic of a continuous (second-order) phase transition into a frozen, non-dissipative state.

Thus, the transition into the Dead Zone is not a failure of the model, but a rigorous demonstration of a physical limit. The FEP thus provides the inferential interpretation of a phenomenon whose topological necessity was proven by the ART equations. The conclusion is physical: a hyper-connected network stripped of its capacity for localized active inference inevitably collapses into a thermodynamically unviable state.
