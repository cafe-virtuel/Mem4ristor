* Mem4ristor v2.6 - Behavioral Model Validation
* Simulation d'une cellule unique avec stimulation

* --- Paramètres (FHN) ---
.param pa=0.7
.param pb=0.8
.param peps=0.08
.param palpha=0.15
.param pdiv=5.0

* --- Paramètres (Doute/Stimulus) ---
.param istim=0.5
* Pour ce test unitaire, on fixe le couplage a zero (cellule isolee) pour verifier la dynamique interne
.param icoup=0.0

* --- Circuit Comportemental (Analog Computer) ---

* 1. Variable V (Potentiel Cognitif)
* Integrateur: I = C * dV/dt => I_source = dV/dt
* Equation: dV/dt = V - V^3/5 - W + I_total - alpha*tanh(V)
* On utilise une source de courant B (Non-linear source) chargeant un condensateur de 1.0 Farad.
C_v v_node 0 1.0
B_dv v_node 0 I = V(v_node) - (pow(V(v_node),3)/{pdiv}) - V(w_node) + {istim} - {palpha}*tanh(V(v_node))

* 2. Variable W (Recuperation)
* Equation: dW/dt = epsilon * (V + a - b*W)
C_w w_node 0 1.0
B_dw w_node 0 I = {peps} * (V(v_node) + {pa} - {pb}*V(w_node))

* --- Initialisation ---
.ic V(v_node)=-1.0 V(w_node)=0.0

* --- Simulation ---
* Transient analysis: step=0.1s, stop=100s, start=0s
.tran 0.1 200 uic

* --- Commandes de controle ---
.control
run
* Export des resultats en ASCII pour analyse Python
print V(v_node) V(w_node) > results/spice_output.txt
.endc

.end
