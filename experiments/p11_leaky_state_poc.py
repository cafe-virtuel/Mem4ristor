#!/usr/bin/env python3
"""
P11 -- L'ETAT FUYANT (leaky) : troisieme tentative, apres deux echecs de
detecteur de conflit (p11_adaptive_watchdog_poc.py et
p11_trajectory_watchdog_poc.py, meme jour).
=============================================================================
Cree : 2026-07-13 (Claude Sonnet 5, L'Ingenieur). Julien : "tentons l'etat
fuyant et si nous n'apprenons rien de ces erreurs nous arreterons de
chercher". Les deux detecteurs precedents echouaient a DETECTER le
conflit ; celui-ci ne cherche plus a detecter -- il laisse l'etat
s'estomper naturellement entre les tours, sans jamais decider explicitement
garder/reset.

MECANISME, aucune modification du coeur : entre le tour k et le tour k+1,
le reseau tourne N_GAP pas SANS stimulus (relaxation libre selon sa PROPRE
dynamique FHN+doute, pas un forcage artificiel de u_c). A N_GAP=0, on
retrouve exactement PERSISTENT (memoire pleine, cicatrice pleine) --
verification de continuite. A N_GAP grand, l'etat devrait s'approcher d'un
FRESH de facto (tout oublie avant le tour suivant).

PREDICTION PRE-FIXEE : il devrait exister un N_GAP intermediaire ou le
compromis est favorable -- suffisamment de relaxation pour attenuer la
cicatrice (FLIP) sans effacer completement le gain de memoire (SAME), un
sweet spot ou MIXED (le regime realiste) bat A LA FOIS FRESH pur ET
PERSISTENT pur (N_GAP=0). Prediction alternative honnete : si le
mecanisme de cicatrice est un verrou plutot qu'une accumulation graduelle
(ce que le v2, meme jour, a trouve -- verrouillage rapide, pas de lutte
visible avant 150-300 pas), la decroissance pourrait n'avoir aucun effet
utile avant un N_GAP deja assez grand pour avoir aussi efface la memoire
-- pas de sweet spot, juste un slider entre les deux extremes sans jamais
les depasser.

Statut : exploratoire, hors preprint, aucune modification de dynamics.py.
Guardian doit rester 14/14. Sorties : figures/p11_leaky_state_poc.csv + .png
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
from p11_continuous_memory_scar_poc import read_fresh  # noqa: E402
from p11_adaptive_watchdog_poc import make_b_sequence  # noqa: E402

FIG = ROOT / "figures"
K = 4
N_CHAINS = 40
N_GAP_SWEEP = [0, 10, 30, 60, 120, 250]


class LeakyReader:
    """Relaxation libre (sans stimulus) de N_GAP pas entre les tours --
    aucune decision explicite garder/reset, juste le temps qui passe."""

    def __init__(self, chain_seed, mask, n_gap):
        self.mask = mask
        self.n_gap = n_gap
        self.first_read = True
        self.net = Mem4Network(size=pw.SIDE, heretic_ratio=0.0, seed=chain_seed * 10 + 1,
                                adjacency_matrix=make_lattice_adj(pw.SIDE, periodic=True))
        self.net.model.cfg['complex_doubt']['enabled'] = True

    def read(self, b, idle):
        if not self.first_read and self.n_gap > 0:
            zero_stim = np.zeros(pw.N)
            for _ in range(self.n_gap):
                self.net.step(I_stimulus=zero_stim)
        self.first_read = False
        stim = np.zeros(pw.N)
        stim[self.mask] = b * pw.B_E
        for _ in range(pw.T_READ):
            self.net.step(I_stimulus=stim)
        return float(np.real(self.net.model.u_c[idle].mean() - self.net.model.u_c[self.mask].mean()))


def run_chain_leaky(chain_seed, mode, n_gap):
    mask, idle = pw.build_group(chain_seed)
    b_seq = make_b_sequence(chain_seed, mode)
    reader = LeakyReader(chain_seed, mask, n_gap)
    signed_finals = []
    for k in range(K):
        b_k = b_seq[k]
        diff = reader.read(b_k, idle)
        signed_finals.append(diff * b_k)
    return signed_finals


def run_chain_fresh(chain_seed, mode):
    mask, idle = pw.build_group(chain_seed)
    b_seq = make_b_sequence(chain_seed, mode)
    signed_finals = []
    for k in range(K):
        b_k = b_seq[k]
        diff = read_fresh(chain_seed, mask, idle, b_k, k)
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
    print("=== P11 -- L'ETAT FUYANT : troisieme tentative ===\n")

    chain_seeds = list(range(N_CHAINS))
    fresh_results = {}
    for mode in ["same", "flip", "mixed"]:
        mat = np.array([run_chain_fresh(cs, mode) for cs in chain_seeds]).T
        fresh_results[mode] = mat
        print(f"[fresh  /{mode:<6}] tours 2..K moyenne = {mat[1:].mean():+.4f}")

    leaky_results = {}
    for mode in ["same", "flip", "mixed"]:
        for n_gap in N_GAP_SWEEP:
            tlab = time.time()
            mat = np.array([run_chain_leaky(cs, mode, n_gap) for cs in chain_seeds]).T
            leaky_results[(mode, n_gap)] = mat
            print(f"[leaky  /{mode:<6}/gap={n_gap:<4}] {time.time()-tlab:.1f}s -- "
                  f"tours 2..K moyenne = {mat[1:].mean():+.4f}")

    print(f"\n{'mode':<8}{'N_GAP':>8}" + "".join(f"{'tour '+str(k+1):>10}" for k in range(K))
          + f"{'moy. 2-4':>12}")
    print("-" * (8 + 8 + 10 * K + 12))
    for mode in ["same", "flip", "mixed"]:
        for n_gap in N_GAP_SWEEP:
            mat = leaky_results[(mode, n_gap)]
            print(f"{mode:<8}{n_gap:>8}" + "".join(f"{mat[k].mean():>10.4f}" for k in range(K))
                  + f"{mat[1:].mean():>12.4f}")
        print(f"{mode:<8}{'FRESH':>8}" + "".join(f"{fresh_results[mode][k].mean():>10.4f}" for k in range(K))
              + f"{fresh_results[mode][1:].mean():>12.4f}")

    print("\n=== VERDICT (sweet spot sur MIXED, bootstrap apparie par chaine, tours 2..K) ===")
    mode = "mixed"
    fresh_mix = fresh_results[mode][1:, :].mean(axis=0)
    pers_mix = leaky_results[(mode, 0)][1:, :].mean(axis=0)  # N_GAP=0 == PERSISTENT pur
    best_gap, best_val = None, -np.inf
    for n_gap in N_GAP_SWEEP:
        leaky_mix = leaky_results[(mode, n_gap)][1:, :].mean(axis=0)
        if leaky_mix.mean() > best_val:
            best_val = leaky_mix.mean()
            best_gap = n_gap
    best_mix = leaky_results[(mode, best_gap)][1:, :].mean(axis=0)
    d1, lo1, hi1 = boot_ci_paired(best_mix, fresh_mix)
    d2, lo2, hi2 = boot_ci_paired(best_mix, pers_mix)
    print(f"  Meilleur N_GAP sur MIXED : {best_gap} (signed_final moyen = {best_val:+.4f})")
    print(f"  MEILLEUR_LEAKY vs FRESH      : delta={d1:+.4f} CI[{lo1:+.4f},{hi1:+.4f}] -> "
          f"{'bat fresh' if lo1 > 0 else ('perd contre fresh' if hi1 < 0 else 'parite')}")
    print(f"  MEILLEUR_LEAKY vs PERSISTENT : delta={d2:+.4f} CI[{lo2:+.4f},{hi2:+.4f}] -> "
          f"{'bat persistent' if lo2 > 0 else ('perd contre persistent' if hi2 < 0 else 'parite')}")
    if lo1 > 0 and lo2 > 0:
        print("  -> SWEET SPOT CONFIRME : l'etat fuyant bat les deux extremes fixes sur le "
              "regime realiste.")
    else:
        print("  -> PAS DE SWEET SPOT CONFIRME : l'etat fuyant reste, au mieux, un curseur "
              "entre les deux extremes -- pas une strategie superieure aux deux.")

    print("\n  Trajectoires SAME et FLIP (le compromis memoire/cicatrice tient-il sur toute "
          "la plage de N_GAP, ou casse-t-il d'un cote avant l'autre ?) :")
    for mode2 in ["same", "flip"]:
        vals = [leaky_results[(mode2, g)][1:, :].mean() for g in N_GAP_SWEEP]
        print(f"     {mode2:<6} : " + " ".join(f"gap={g}:{v:+.3f}" for g, v in zip(N_GAP_SWEEP, vals)))

    with (FIG / "p11_leaky_state_poc.csv").open("w", encoding="utf-8") as f:
        f.write("mode,n_gap,chain_seed,round,signed_final\n")
        for mode in ["same", "flip", "mixed"]:
            for k in range(K):
                for i, cs in enumerate(chain_seeds):
                    f.write(f"{mode},fresh,{cs},{k+1},{fresh_results[mode][k, i]:.6f}\n")
            for n_gap in N_GAP_SWEEP:
                mat = leaky_results[(mode, n_gap)]
                for k in range(K):
                    for i, cs in enumerate(chain_seeds):
                        f.write(f"{mode},{n_gap},{cs},{k+1},{mat[k, i]:.6f}\n")
    print(f"\n[csv] {FIG / 'p11_leaky_state_poc.csv'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5.2))
        colors = {"same": "#1f77b4", "flip": "#d62728", "mixed": "#2ca02c"}
        for mode in ["same", "flip", "mixed"]:
            vals = [leaky_results[(mode, g)][1:, :].mean() for g in N_GAP_SWEEP]
            ax.plot(N_GAP_SWEEP, vals, "o-", color=colors[mode], label=f"{mode} (leaky)")
            ax.axhline(fresh_results[mode][1:, :].mean(), ls="--", color=colors[mode], alpha=0.5,
                       label=f"{mode} (fresh, reference)")
        ax.axhline(0, c="k", lw=0.8)
        ax.set_xlabel("N_GAP (pas de relaxation libre entre les tours)")
        ax.set_ylabel("signed_final moyen (tours 2-4)")
        ax.set_title("Etat fuyant : le compromis memoire/cicatrice en fonction du gap")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIG / "p11_leaky_state_poc.png", dpi=140)
        print(f"[png] {FIG / 'p11_leaky_state_poc.png'}")
    except Exception as e:
        print(f"[png] skipped: {e}")

    print(f"\nWall time: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
