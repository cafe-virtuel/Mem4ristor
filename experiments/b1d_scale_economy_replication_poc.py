#!/usr/bin/env python3
"""
B1d -- REPLICATION de l'economie d'echelle (graines disjointes).
=============================================================================
Cree : 2026-07-13 (Claude Sonnet 5, L'Ingenieur). Suite immediate de
`b1d_scale_economy_poc.py`, qui a trouve que le gain du doute (B1d) survit
a une reduction de N par 4 (100->25) et meme par 11 (100->9, avec reserve).
Avant de conclure quoi que ce soit -- lecon du meme jour (le resultat
Condorcet mort a la replication) -- verification sur une plage de graines
totalement DISJOINTE (200-219, 20 seeds) des deux points qui comptent le
plus (N=25 et N=100, T_pulse=350). Reutilise les fonctions de
b1d_scale_economy_poc.py a l'identique, aucun nouveau protocole.

Statut : exploratoire, hors preprint, aucune modification de dynamics.py.
Guardian doit rester 14/14. Sorties : figures/b1d_scale_economy_replication_poc.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
import b1d_scale_economy_poc as p  # noqa: E402

FIG = ROOT / "figures"
SEEDS_REP = list(range(200, 220))  # disjoint de 0..11 (test original)
SIDES_T_PULSE = [(5, 350), (10, 350)]


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    print("=== REPLICATION economie d'echelle (20 graines disjointes, 200-219) ===\n")
    rows = []
    for side, t_pulse in SIDES_T_PULSE:
        n, n_distract, n_true = p.counts_for_side(side)
        acc_d, acc_c = [], []
        for seed in SEEDS_REP:
            rng = np.random.RandomState(3000 + seed)
            adj, stim_on, stim_off, dstar = p.make_deceptive(rng, n, n_distract, n_true)
            sig, dec, d_var = p.simulate(side, n, adj, stim_on, stim_off, seed * 10 + 1, t_pulse)
            cd = p.stop_doubt(sig)
            cc = p.stop_conv(d_var)
            acc_d.append(int(p.dec_at(dec, cd) == dstar))
            acc_c.append(int(p.dec_at(dec, cc) == dstar))
        ad, ac = float(np.mean(acc_d)), float(np.mean(acc_c))
        gain = ad - ac
        rows.append((side, n, t_pulse, ad, ac, gain, len(SEEDS_REP)))
        print(f"SIDE={side} N={n} T_pulse={t_pulse}: acc_DOUTE={ad:.3f} acc_CONV={ac:.3f} "
              f"gain={gain:+.3f} (n={len(SEEDS_REP)}, seeds 200-219)")

    print("\n=== COMPARAISON AVEC LE TEST ORIGINAL (seeds 0-11) ===")
    original = {5: 0.75, 10: 0.58}  # gains rapportes par b1d_scale_economy_poc.py a T_pulse=350
    for side, n, t_pulse, ad, ac, gain, nseeds in rows:
        orig = original[side]
        print(f"  N={n}: original(n=12)={orig:+.3f} -> replication(n=20, seeds disjointes)={gain:+.3f} "
              f"-- {'MEME DIRECTION, tient' if gain > 0.10 else 'NE TIENT PAS'}")

    with (FIG / "b1d_scale_economy_replication_poc.csv").open("w", encoding="utf-8") as f:
        f.write("side,n,t_pulse,acc_doute,acc_conv,gain,n_seeds\n")
        for r in rows:
            f.write(",".join(f"{x:.6f}" if isinstance(x, float) else str(x) for x in r) + "\n")
    print(f"\n[csv] {FIG / 'b1d_scale_economy_replication_poc.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
