* MEM4RISTOR 1T1R ELEMENTARY CELL v2.0.4
* Canonical Hardware Model for Physical Justice Cognitive

.param VDD=1.0
.param R_INIT=1000

* Node 1: Stimulus Input
* Node 2: Intermediate (Transistor Drain to Memristor Top)
* Node 3: Ground
* Node Gate: Control for Doubt (u)

VS1 1 0 PULSE(0 {VDD} 1n 1n 1n 10n 20n)

* RTN Bruit Aléatoire (Allié de la Diversité)
.param freq_rtn=100Meg
.param amp_rtn=0.05
Vnoise 4 0 TRNOISE({amp_rtn} 1n 0 0) 

* The Transistor (Control Gate modulated by Doubt + Noise)
M1 2 Gate 0 0 NMOS_MODEL W=1u L=0.18u

* The Memristor (Modeled as a Behavioral Resistor for simplified SPICE)
* The resistance R is a function of the internal state (phi/v)
Rmem 1 2 R=R_INIT

* Doubt Integration (Simplified RC for du/dt)
Rdoubt Stim 4 100k
Cdoubt 4 3 10p

.model NMOS_MODEL NMOS (LEVEL=54 VGEXP=1.2)
.tran 1n 100n

.control
  run
  plot v(1) v(2)
.endc
.end
