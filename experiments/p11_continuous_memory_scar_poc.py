#!/usr/bin/env python3
"""
P11 -- OPTION B : departager memoire vs cicatrice sur le SIGNAL CONTINU
(Re(u_c)), pas sur la decision binaire bruitee -- suite de
p11_multiround_chain_poc.py (meme jour), dont les IC couvraient 0 a n=40
chaines sur l'accuracy binaire.
=============================================================================
Cree : 2026-07-13 (Claude Sonnet 5, L'Ingenieur). Julien : "on lance B".
Meme methode que p10_complex_doubt_poc.py (12/07) qui avait detecte la
memoire directionnelle de Re(u_c) avec PEU de repetitions parce qu'elle
lisait un signal continu, pas un guess seuille -- ici on applique la meme
idee au chainage multi-tours pour trancher la question laissee ouverte.

DESIGN, deux sequences de signes CONTROLEES (au lieu du signe aleatoire par
tour du POC precedent, qui melangeait tous les types de transitions et
diluait le signal) :
  - SAME : b_k identique a tous les tours (b1 fixe par chaine, meme signe
    repete K fois) -- AUCUN conflit entre tours ; terrain le plus favorable
    a une signature de MEMOIRE (le signal devrait se renforcer si u_c
    persiste et est correctement oriente).
  - FLIP : b_k alterne a chaque tour (+1,-1,+1,-1) -- conflit MAXIMAL entre
    tours consecutifs ; terrain le plus favorable a une signature de
    CICATRICE (une trace u_c du tour precedent resisterait au changement).
Pour chaque sequence, FRESH (reseau reconstruit) vs PERSISTENT (etat
conserve) -- 4 conditions, meme groupe physique (mask) dans les 4, seule la
persistance de l'etat differe (meme isolation que le POC precedent).

METRIQUE. Pas un guess +1/-1 seuille -- le signal CONTINU signe :
  signed_final = diff_final * b_k_vrai
  ou diff_final = Re(u_c[idle]).mean() - Re(u_c[mask]).mean() a la fin de
  la fenetre de lecture (T_READ=30 pas). signed_final > 0 et grand = le
  reseau est fortement et correctement oriente ; proche de 0 ou negatif =
  desoriente ou trompe par une trace anterieure.

CRITERES PRE-FIXES (bootstrap apparie par chaine, sur les tours 2..K -- le
tour 1 n'a par construction aucune trace anterieure a tester) :
  1. CICATRICE : signed_final(FLIP, PERSISTENT) < signed_final(FLIP, FRESH) ?
     (une trace anterieure resiste au changement de signe)
  2. MEMOIRE   : signed_final(SAME, PERSISTENT) > signed_final(SAME, FRESH) ?
     (une trace anterieure correcte renforce l'orientation)

Statut : exploratoire, hors preprint, aucune modification de dynamics.py.
Guardian doit rester 14/14. Sorties : figures/p11_continuous_memory_scar_poc.csv + .png
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from mem4ristor.topology import Mem4Network       # noqa: E402
from mem4ristor.graph_utils import make_lattice_adj  # noqa: E402
import p11_warm_start_poc as pw       # noqa: E402
import p11_coupled_pipeline_poc as pc  # noqa: E402 -- met pw.T_READ=30, pw.B_E=0.3

FIG = ROOT / "figures"
K = 4
N_CHAINS = 40


def make_b_sequence(chain_seed, mode):
    rng = np.random.RandomState(920_000 + chain_seed)
    b1 = 1 if rng.random() < 0.5 else -1
    if mode == "same":
        return [b1] * K
    return [b1 * ((-1) ** k) for k in range(K)]


def read_fresh(chain_seed, mask, idle, b, k):
    net = Mem4Network(size=pw.SIDE, heretic_ratio=0.0, seed=chain_seed * 10 + 1 + k,
                       adjacency_matrix=make_lattice_adj(pw.SIDE, periodic=True))
    net.model.cfg['complex_doubt']['enabled'] = True
    stim = np.zeros(pw.N)
    stim[mask] = b * pw.B_E
    for _ in range(pw.T_READ):
        net.step(I_stimulus=stim)
    diff = float(np.real(net.model.u_c[idle].mean() - net.model.u_c[mask].mean()))
    return diff


class PersistentReader:
    def __init__(self, chain_seed, mask):
        self.mask = mask
        self.net = Mem4Network(size=pw.SIDE, heretic_ratio=0.0, seed=chain_seed * 10 + 1,
                                adjacency_matrix=make_lattice_adj(pw.SIDE, periodic=True))
        self.net.model.cfg['complex_doubt']['enabled'] = True

    def read(self, b, idle):
        stim = np.zeros(pw.N)
        stim[self.mask] = b * pw.B_E
        for _ in range(pw.T_READ):
            self.net.step(I_stimulus=stim)
        return float(np.real(self.net.model.u_c[idle].mean() - self.net.model.u_c[self.mask].mean()))


def run_chain(chain_seed, bmode, readmode):
    mask, idle = pw.build_group(chain_seed)  # meme groupe physique dans les 4 conditions
    b_seq = make_b_sequence(chain_seed, bmode)
    persistent = PersistentReader(chain_seed, mask) if readmode == "persistent" else None

    signed_finals = []
    for k in range(K):
        b_k = b_seq[k]
        if readmode == "fresh":
            diff = read_fresh(chain_seed, mask, idle, b_k, k)
        else:
            diff = persistent.read(b_k, idle)
        signed_finals.append(diff * b_k)
    return signed_finals


def boot_ci_paired(a, b, n_boot=10000, seed=20260713):
    rng = np.random.RandomState(seed)
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    m = np.empty(n_boot)
    for i in range(n_boot):
        m[i] = d[rng.randint(0, n, n)].mean()
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"=== P11 OPTION B : memoire vs cicatrice sur le signal CONTINU (Re(u_c)) ===\n")

    chain_seeds = list(range(N_CHAINS))
    results = {}
    for bmode in ["same", "flip"]:
        for readmode in ["fresh", "persistent"]:
            tlab = time.time()
            mat = np.empty((K, N_CHAINS))
            for i, cs in enumerate(chain_seeds):
                sf = run_chain(cs, bmode, readmode)
                mat[:, i] = sf
            results[(bmode, readmode)] = mat
            print(f"[{bmode:<5}/{readmode:<11}] {N_CHAINS} chaines x {K} tours, "
                  f"{time.time()-tlab:.1f}s -- signed_final moyen par tour : "
                  + " ".join(f"{mat[k].mean():+.3f}" for k in range(K)))

    print(f"\n{'condition':<20}" + "".join(f"{'tour '+str(k+1):>10}" for k in range(K)))
    print("-" * (20 + 10 * K))
    for (bmode, readmode), mat in results.items():
        label = f"{bmode}/{readmode}"
        print(f"{label:<20}" + "".join(f"{mat[k].mean():>10.3f}" for k in range(K)))

    print("\n=== VERDICT (criteres pre-fixes, bootstrap apparie par chaine, tours 2..K) ===")

    # 1. CICATRICE : FLIP/PERSISTENT vs FLIP/FRESH, moyenne des tours 2..K par chaine
    flip_pers = results[("flip", "persistent")][1:, :].mean(axis=0)
    flip_fresh = results[("flip", "fresh")][1:, :].mean(axis=0)
    d1, lo1, hi1 = boot_ci_paired(flip_pers, flip_fresh)
    if hi1 < 0:
        tag1 = "CICATRICE CONFIRMEE (PERSISTENT significativement PLUS FAIBLE que FRESH sous conflit)"
    elif lo1 > 0:
        tag1 = "INVERSE : PERSISTENT plus fort que FRESH meme sous conflit (pas de cicatrice)"
    else:
        tag1 = "IC couvre 0 -- pas de cicatrice confirmee"
    print(f"  1. CICATRICE (FLIP) : PERSISTENT={flip_pers.mean():.3f}  FRESH={flip_fresh.mean():.3f}  "
          f"delta={d1:+.3f} CI[{lo1:+.3f},{hi1:+.3f}]")
    print(f"     -> {tag1}")

    # 2. MEMOIRE : SAME/PERSISTENT vs SAME/FRESH, moyenne des tours 2..K par chaine
    same_pers = results[("same", "persistent")][1:, :].mean(axis=0)
    same_fresh = results[("same", "fresh")][1:, :].mean(axis=0)
    d2, lo2, hi2 = boot_ci_paired(same_pers, same_fresh)
    if lo2 > 0:
        tag2 = "MEMOIRE CONFIRMEE (PERSISTENT significativement PLUS FORT que FRESH en renfort)"
    elif hi2 < 0:
        tag2 = "INVERSE : PERSISTENT plus faible que FRESH meme en renfort (pas de memoire utile)"
    else:
        tag2 = "IC couvre 0 -- pas de memoire confirmee"
    print(f"  2. MEMOIRE (SAME)   : PERSISTENT={same_pers.mean():.3f}  FRESH={same_fresh.mean():.3f}  "
          f"delta={d2:+.3f} CI[{lo2:+.3f},{hi2:+.3f}]")
    print(f"     -> {tag2}")

    # 3. bonus : l'effet du CONFLIT lui-meme, independant de la memoire (FLIP vs SAME, en FRESH
    #    -- controle : le conflit degrade-t-il meme sans aucune memoire possible ?)
    d3, lo3, hi3 = boot_ci_paired(results[("same", "fresh")][1:, :].mean(axis=0),
                                   results[("flip", "fresh")][1:, :].mean(axis=0))
    print(f"  3. Controle -- cout du conflit SANS memoire (FRESH seul) : SAME-FLIP delta={d3:+.3f} "
          f"CI[{lo3:+.3f},{hi3:+.3f}] (verifie que FLIP est bien un piege plus dur en soi, "
          f"pas seulement via la memoire)")

    with (FIG / "p11_continuous_memory_scar_poc.csv").open("w", encoding="utf-8") as f:
        f.write("bmode,readmode,chain_seed,round,signed_final\n")
        for (bmode, readmode), mat in results.items():
            for k in range(K):
                for i, cs in enumerate(chain_seeds):
                    f.write(f"{bmode},{readmode},{cs},{k+1},{mat[k, i]:.6f}\n")
    print(f"\n[csv] {FIG / 'p11_continuous_memory_scar_poc.csv'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5.2))
        rounds = np.arange(1, K + 1)
        styles = {("same", "fresh"): ("#1f77b4", "--", "o"),
                  ("same", "persistent"): ("#1f77b4", "-", "s"),
                  ("flip", "fresh"): ("#d62728", "--", "o"),
                  ("flip", "persistent"): ("#d62728", "-", "s")}
        for key, mat in results.items():
            c, ls, mk = styles[key]
            means = mat.mean(axis=1)
            sems = mat.std(axis=1) / np.sqrt(N_CHAINS)
            ax.errorbar(rounds, means, yerr=sems, color=c, linestyle=ls, marker=mk,
                        label=f"{key[0]}/{key[1]}", capsize=3)
        ax.axhline(0, c="k", lw=0.8)
        ax.set_xlabel("tour")
        ax.set_ylabel("signed_final = diff x b_vrai (oriente correctement si >0)")
        ax.set_title("Memoire (SAME, bleu) vs cicatrice (FLIP, rouge) -- signal continu")
        ax.set_xticks(rounds)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIG / "p11_continuous_memory_scar_poc.png", dpi=140)
        print(f"[png] {FIG / 'p11_continuous_memory_scar_poc.png'}")
    except Exception as e:
        print(f"[png] skipped: {e}")

    print(f"\nWall time: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
