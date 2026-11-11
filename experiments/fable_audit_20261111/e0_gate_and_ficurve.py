#!/usr/bin/env python3
"""
AUDIT FABLE 2026-11-11 -- E0 : (1) gate de fidelite du simulateur contre le coeur reel,
(2) courbe frequence-courant d'un noeud isole (pour choisir les entrees de E1 AVANT E1).

Usage : uv run --frozen python experiments/fable_audit_20261111/e0_gate_and_ficurve.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import networkx as nx

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))
from fable_sim import simulate, freq_readout, DT  # noqa: E402


def gate():
    from mem4ristor.core import Mem4Network
    N, D, steps = 30, 0.5, 3000
    adj = nx.to_numpy_array(nx.erdos_renyi_graph(N, 0.2, seed=3), dtype=float)
    I = np.random.RandomState(11).uniform(0.2, 0.8, N)
    # coeur reel : heretiques 0, recablage coupe, plasticite d'inhibition a 0
    net = Mem4Network(size=10, heretic_ratio=0.0, seed=1, adjacency_matrix=adj.copy(),
                      coupling_norm="uniform", rewire_threshold=999.0)
    net.model.lambda_intrinsic = 0.0
    net.model.cfg["coupling"]["D"] = D
    net.model.D_eff = D / np.sqrt(N)
    rng = np.random.RandomState(7)
    v0 = rng.uniform(-1.5, 1.5, N); w0 = rng.uniform(0.0, 1.0, N)
    net.model.v = v0.copy(); net.model.w = w0.copy(); net.model.rng = rng
    for _ in range(steps):
        net.step(I_stimulus=I)
    rng2 = np.random.RandomState(7)
    out = simulate(adj, I, D, "node", seed=None, steps=steps, rng=rng2)
    dv = float(np.max(np.abs(out["v"] - net.model.v)))
    du = float(np.max(np.abs(out["u_n"] - net.model.u)))
    print(f"GATE (mode 'node' vs coeur reel, N={N}, {steps} pas) : max|dv| = {dv:.3e}, max|du| = {du:.3e}"
          f"  -> {'OK' if dv < 1e-9 and du < 1e-9 else 'ECHEC'}")
    print(f"   (u du coeur en fin de run : min {net.model.u.min():.3f} max {net.model.u.max():.3f})")
    return dv < 1e-9 and du < 1e-9


def ficurve():
    print("\nCourbe f-I, noeud isole (8 repliques bruitees), 12000 pas, mesure sur les 8000 derniers")
    print(f"{'I':>6} {'freq/100ut':>11} {'sd repl.':>9} {'<v>':>7}")
    win = 8000 * DT
    for I in np.round(np.arange(-0.2, 1.61, 0.1), 2):
        out = simulate(np.zeros((8, 8)), I, 0.0, "none", seed=5, steps=12000, rec_from=4000, rec_every=2)
        f = freq_readout(out["rec"]) / win * 100.0
        print(f"{I:>6.2f} {f.mean():>11.3f} {f.std():>9.3f} {out['rec'].mean():>7.3f}")


if __name__ == "__main__":
    ok = gate()
    ficurve()
    sys.exit(0 if ok else 1)
