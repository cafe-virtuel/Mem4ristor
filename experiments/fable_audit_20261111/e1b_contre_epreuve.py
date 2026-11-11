#!/usr/bin/env python3
"""
AUDIT FABLE 2026-11-11 -- E1b : contre-epreuve de E1, a charge contre mon propre resultat.

E1 donne, sur le SBM, edge_noflip (c=3) > fixed (c=10) de +0.051 d'ARI. Mais le c optimal du
bras 'fixed' est AU BORD de la grille (son profil monte encore a c=10) : la comparaison
"chaque bras a son meilleur c" est donc biaisee CONTRE le couplage fixe. Monter c au pas du
coeur (dt=0.05) rendrait Euler instable ; on refait donc tout a dt=0.01 (5x plus de pas).

Question, ecrite AVANT la mesure : un couplage fixe pousse a c=30 puis c=100 rattrape-t-il
edge_noflip (c=3) ?  Si oui, l'avantage de E1 etait un artefact de grille, et je le retire.
Controle d'integration : fixed c=10 et edge_noflip c=3 refaits a dt=0.01 doivent redonner
les valeurs de E1 (0.753 et 0.805) a peu pres.

Memes 20 graines de test que E1 (10..29), memes graphes, memes entrees.
Usage : uv run --frozen python experiments/fable_audit_20261111/e1b_contre_epreuve.py
"""
from __future__ import annotations
import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
FACT = 5
ARMS = [("fixed", 10.0), ("fixed", 30.0), ("fixed", 100.0), ("edge_noflip", 3.0), ("edge_noflip", 10.0)]


def job(args):
    seed, mode, c = args
    import fable_sim as fs
    import e1_plural_consensus as e1
    fs.DT = 0.05 / FACT
    adj, groups = e1.build("sbm", seed)
    N = adj.shape[0]
    I = e1.MU[groups] + np.random.RandomState(5000 + seed).normal(0, e1.SIGMA_IN, N)
    D = c / (adj.sum() / N) * np.sqrt(N)
    out = fs.simulate(adj, I, D, mode, seed, steps=e1.STEPS * FACT, rec_from=e1.REC_FROM * FACT,
                      rec_every=e1.REC_EVERY * FACT)
    ideal = np.array([-0.524, -0.132, 0.207])           # valeurs de E1 (meme protocole)
    r = e1.read(out, groups, ideal)
    r.update(seed=seed, mode=mode, c=c)
    return r


def main():
    import e1_plural_consensus as e1
    jobs = [(s, m, c) for (m, c) in ARMS for s in e1.TEST]
    with Pool(12) as pool:
        rows = pool.map(job, jobs, chunksize=1)
    raw = 0.305
    print("E1b -- SBM, graines de test 10..29, dt = 0.01")
    print(f"{'bras':>14} {'c':>6} | {'ARI_mean':>8} {'IC95':>16} | {'RMSE/brut':>9} | {'cut_in':>6} {'cut_out':>7}")
    get = lambda m, c, k: np.array([r[k] for r in rows if r["mode"] == m and r["c"] == c])
    for (m, c) in ARMS:
        a = get(m, c, "ari_mean"); mu, lo, hi = e1.boot_ci(a)
        print(f"{m:>14} {c:>6} | {mu:>8.3f} [{lo:.3f},{hi:.3f}]   | {np.mean(get(m, c, 'rmse')) / raw:>9.3f} | "
              f"{np.nanmean(get(m, c, 'cut_in')):>6.2f} {np.nanmean(get(m, c, 'cut_out')):>7.2f}")
    best_fixed = max([10.0, 30.0, 100.0], key=lambda c: get("fixed", c, "ari_mean").mean())
    for cn in (3.0, 10.0):
        d = get("edge_noflip", cn, "ari_mean") - get("fixed", best_fixed, "ari_mean")
        mu, lo, hi = e1.boot_ci(d)
        print(f"edge_noflip(c={cn}) - fixed(c={best_fixed}, son meilleur) : {mu:+.3f} IC95 [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    main()
