# -*- coding: utf-8 -*-
"""
Crosscheck Claude (L'Ingenieur) — 2026-05-30
============================================
Question : la decouverte d'Hermes (I=0.1 maximise H_cont) correspond-elle a une
vraie difference de DECORRELATION (synchronie, binning-independante), ou est-ce
un effet H_cont (etalement du nuage) qui n'apparait pas sur la metrique solide ?

On mesure, cote a cote, pour chaque I_stimulus :
  - H_cont         (entropie continue 100 bins — la metrique-nuage)
  - synchronie     (correlation de Pearson moyenne de v(t) — binning-independant)
  - %etat 5        (fraction des noeuds en consensus fort, bins KIMI)
  - n_etats_actifs (nb d'etats >5% — diversite discrete)
  - u_moyen

Lattice N=100, cold_start, heretic 0.15. Leger pour ne pas gener le test en cours.
"""

import sys, os
import numpy as np

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, SRC)

from mem4ristor.topology import Mem4Network
from mem4ristor.metrics import (
    calculate_continuous_entropy,
    calculate_pairwise_synchrony,
    get_cognitive_states,
)

I_VALUES = [0.0, 0.1, 0.5, 1.0]
SEEDS = [42, 123, 777, 7, 99]
WARMUP = 300
MEASURE = 700
SAMPLE_EVERY = 4
SIGMA_V = 0.15   # on colle au regime du record d'Hermes (etait par defaut 0.05)

def run_one(I, seed):
    net = Mem4Network(size=10, heretic_ratio=0.15, seed=seed, cold_start=True)
    try:
        net.model.cfg['noise']['sigma_v'] = SIGMA_V
    except Exception as e:
        print(f"[warn] impossible de fixer sigma_v: {e}")
    for _ in range(WARMUP):
        net.step(I_stimulus=I)
    v_hist = []
    h_samples = []
    for t in range(MEASURE):
        net.step(I_stimulus=I)
        v_hist.append(net.v.copy())
        if t % SAMPLE_EVERY == 0:
            h_samples.append(calculate_continuous_entropy(net.v, bins=100))
    v_hist = np.array(v_hist)                      # (T, N)
    h_cont = float(np.mean(h_samples))
    sync = calculate_pairwise_synchrony(v_hist)    # binning-independant
    # distribution d'etats sur le dernier snapshot
    states = get_cognitive_states(net.v)
    frac = np.array([np.mean(states == s) for s in (1, 2, 3, 4, 5)])
    pct_state5 = float(frac[4]) * 100
    n_active = int(np.sum(frac > 0.05))
    u_mean = float(net.model.u.mean())
    return h_cont, sync, pct_state5, n_active, u_mean

print("\n" + "=" * 78)
print("  CROSSCHECK : H_cont (nuage) vs Synchronie (decorrelation) vs Etats")
print("  sigma_v = {}".format(SIGMA_V))
print("  Lattice N=100, cold_start, heretic=0.15, {} seeds".format(len(SEEDS)))
print("=" * 78)
print(f"\n{'I_stim':>7} | {'H_cont':>8} | {'synchro':>8} | {'%etat5':>7} | "
      f"{'n_etats':>7} | {'u_moy':>6}")
print("-" * 78)

rows = []
for I in I_VALUES:
    res = np.array([run_one(I, s) for s in SEEDS])
    m = res.mean(axis=0)
    rows.append((I, *m))
    print(f"{I:>7.2f} | {m[0]:>8.3f} | {m[1]:>8.3f} | {m[2]:>6.1f}% | "
          f"{m[3]:>7.1f} | {m[4]:>6.3f}")

print("-" * 78)
print("\nLecture :")
print("  - Si synchro MONTE de I=0.1 a I=1.0  -> I=0.1 est aussi meilleur sur la")
print("    metrique solide : decouverte d'Hermes ROBUSTE.")
print("  - Si synchro reste PLATE alors que H_cont chute -> l'avantage I=0.1 est")
print("    un effet H_cont/etats, INVISIBLE a la decorrelation (a relativiser).")
print("=" * 78 + "\n")
