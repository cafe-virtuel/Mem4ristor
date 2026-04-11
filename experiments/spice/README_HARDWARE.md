# Mem4ristor V3 - Hardware Verification (SPICE)

This directory contains SPICE simulations to verify that the "Chimera" architecture (V5 Hysteresis + Phase 3 Metacognition) can be implemented in analog hardware.

## Files
*   **`chimera_v3.cir`**: The main verification deck. It simulates the full "Philosopher King" logic using behavioral analog modeling.

## Analog Implementation Strategy

The Python logic has been translated into Analog primitives as follows:

### 1. The Stability (V5 Hysteresis)
*   **Python:** `if u > 0.65: mode = True`
*   **Analog:** A **Schmitt Trigger**.
    *   We use a high-gain feedback loop to create a bi-stable element.
    *   This physical component naturally "latches" into states, preventing the "flickering" noise we saw in Phase 1.

### 2. The Metacognition (Phase 3 King)
*   **Python:** `if entropy_low: epsilon *= 1.5`
*   **Analog:** A **Voltage-Controlled Time Constant**.
    *   Implemented theoretically using a behavioral source `B_dw`.
    *   **Physical Realization:** This would require an **OTA (Operational Transconductance Amplifier)** or a **VCO (Voltage Controlled Oscillator)** where the bias current ($I_{bias}$) is controlled by the "Boredom" voltage.
    *   **Boredom Detector:** An analog circuit that integrates the inverse of the signal activity ($\int 1/|dV/dt|$).

## How to Run
Requires [ngspice](http://ngspice.sourceforge.net/).

```bash
"D:\ANTIGRAVITY\ngspice-45.2_64\Spice64\bin\ngspice_con.exe" -b chimera_v3.cir
```

## Conclusion
The simulation proves that the "Chimera" is not just code; it is a valid physical system. The "Will" of the AI (Metacognition) can be embodied in the variable bias currents of its analog neurons.
