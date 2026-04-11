# Session 5: Benchmarking Mem4Ristor v2.0

## 🎯 Mission
Prove that Mem4Ristor v2.0 out-performs state-of-the-art models in preserving cognitive diversity and resisting uniformization.

## 🔬 Comparison Models & Formulas

### 1. Kuramoto Model (Classical Synchronization)
**Axiom:** Optimizes phase synchronization.
**Formula:** 
$$\dot{\theta}_i = \omega_i + \frac{K}{N} \sum_{j \in \mathcal{N}(i)} \sin(\theta_j - \theta_i)$$
**Blind Spot:** No mechanism for intentional diversity preservation; synchronization is ineluctable for $K > K_c$.

### 2. Mirollo-Strogatz Model (Pulse-Coupled Fireflies)
**Axiom:** Decentered temporal synchronization.
**Formula:**
$$x_i(t+\Delta t) = f(x_i(t)) + \epsilon \text{ if neighbor flashes, where } f(x) \text{ is a concave drive.}$$
$$\text{If } x_i \geq 1, \text{ flash and reset to } 0.$$
**Blind Spot:** Incapable of stable polyphony; information in phase offset is treated as error.

### 3. Distributed Averaging (Consensus Algorithms)
**Axiom:** Network-wide convergence to a single value.
**Formula:**
$$x_i(t+1) = \sum_{j \in \mathcal{N}(i)} W_{ij} x_j(t), \text{ where } \sum_j W_{ij} = 1$$
**Blind Spot:** Entirely erases nuance; the final state is the arithmetic mean, losing all minority input.

### 4. Voter Model (Social Imitation)
**Axiom:** Mimicry of neighboring states.
**Update Rule:**
$$x_i(t+1) = x_j(t) \text{ where } j \text{ is a random neighbor.}$$
**Blind Spot:** Final diversity is purely a topological artifact, not a functional feature.

## 📊 Canonical Metrics for Benchmark v2.0
1. **Dynamic Entropy (H(t))**: $H = -\sum p_i \log_2(p_i)$ over 1000 steps.
2. **Collapse Time (τ₈₀)**: Steps until >80% consensus is reached.
3. **Resilience**: Measure of diversity recovery after a biased stimulus shock ($I_{stim} = 1.0$).
4. **Minority Survival Rate**: Persistence of heretic units ($15\%$) against majority pressure.
5. **Effective Scaling Cost**: Computational cost per unit across sizes $N=\{10, 25, 50\}$.
