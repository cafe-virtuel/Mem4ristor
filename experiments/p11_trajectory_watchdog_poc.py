#!/usr/bin/env python3
"""
P11 -- LE WATCHDOG PAR TRAJECTOIRE : reparer l'angle mort du premier
watchdog (p11_adaptive_watchdog_poc.py, meme jour).
=============================================================================
Cree : 2026-07-13 (Claude Sonnet 5, L'Ingenieur). Le premier detecteur
(coherence_k = diff_final * guess_precedent) etait aveugle a la cicatrice :
une cicatrice, PAR DEFINITION, fait que le reseau RESISTE au nouveau
stimulus et reste aligne sur l'ancien guess -- exactement le signal que le
detecteur lisait comme "tout va bien, je persiste". Confirmation legitime
et resistance de cicatrice produisaient la MEME valeur finale.

LA CORRECTION : ne plus regarder la valeur FINALE de la lecture, regarder
sa TRAJECTOIRE pendant les T_READ=30 pas. Une confirmation legitime ou une
orientation nouvelle SANS conflit devrait converger sans jamais changer de
signe (le stimulus pousse dans une direction, le signal s'y installe).
Une VRAIE lutte interne (l'ancien etat tire d'un cote, le nouveau stimulus
de l'autre) laisse une signature qu'un simple point final ne peut pas voir :
le signal peut TRAVERSER ZERO en cours de lecture, signe qu'une des deux
forces est en train de l'emporter sur l'autre au lieu d'une convergence
propre. Critere SANS SEUIL DE MAGNITUDE (pas de reglage fin sur les
donnees) : un changement de signe reel dans la trajectoire (apres une
courte periode de mise en route de 5 pas, pour ignorer le seul zero
degenere du tout debut) -> reset avant le tour suivant. Sinon -> garde
l'etat.

MEME PROTOCOLE que le watchdog v1 : 3 regimes de signes (SAME, FLIP,
MIXED) x 3 strategies (FRESH, PERSISTENT, ADAPTIVE_V2), 40 chaines, K=4
tours, aucune fuite du vrai signe b_k dans la decision de reset (le
critere ne depend que de la trajectoire observee par le systeme lui-meme).

PREDICTIONS PRE-FIXEES (avant de lancer, revisees a la lumiere de l'echec
du v1) :
  1. SAME : ADAPTIVE_V2 ~= PERSISTENT (pas de conflit, pas de traversee de
     zero attendue -- comme v1).
  2. FLIP : cette fois, ADAPTIVE_V2 doit REELLEMENT se rapprocher de FRESH
     (le point que v1 ratait) -- le taux de detection de conflit doit etre
     nettement PLUS ELEVE qu'en SAME.
  3. MIXED : ADAPTIVE_V2 doit battre ou au moins egaler les deux extremes
     fixes -- le vrai test.

Statut : exploratoire, hors preprint, aucune modification de dynamics.py.
Guardian doit rester 14/14. Sorties : figures/p11_trajectory_watchdog_poc.csv + .png
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
from p11_continuous_memory_scar_poc import read_fresh, PersistentReader  # noqa: E402
from p11_adaptive_watchdog_poc import round_b, make_b_sequence  # noqa: E402

FIG = ROOT / "figures"
K = 4
N_CHAINS = 40
WARMUP_STEPS = 5  # ignore les tout premiers pas (transitoire de demarrage), fixe a priori


class TrajectoryWatchdogReader:
    """Reset si la trajectoire de diff(t) traverse zero apres le
    demarrage -- signature d'une lutte interne, pas juste un point final."""

    def __init__(self, chain_seed, mask):
        self.chain_seed = chain_seed
        self.mask = mask
        self.build_idx = 0
        self.n_resets = 0
        self._build_net()

    def _build_net(self):
        self.net = Mem4Network(size=pw.SIDE, heretic_ratio=0.0,
                                seed=self.chain_seed * 10 + 1 + self.build_idx,
                                adjacency_matrix=make_lattice_adj(pw.SIDE, periodic=True))
        self.net.model.cfg['complex_doubt']['enabled'] = True

    def read(self, b, idle):
        stim = np.zeros(pw.N)
        stim[self.mask] = b * pw.B_E
        trace = np.empty(pw.T_READ)
        for t in range(pw.T_READ):
            self.net.step(I_stimulus=stim)
            trace[t] = float(np.real(self.net.model.u_c[idle].mean() - self.net.model.u_c[self.mask].mean()))
        diff = float(trace[-1])
        post_warmup = trace[WARMUP_STEPS:]
        signs = np.sign(post_warmup)
        signs = signs[signs != 0]
        conflict_detected = len(signs) > 1 and bool(np.any(signs[1:] != signs[:-1]))
        if conflict_detected:
            self.build_idx += 1
            self._build_net()
            self.n_resets += 1
        return diff


def run_chain(chain_seed, mode, strategy):
    mask, idle = pw.build_group(chain_seed)
    b_seq = make_b_sequence(chain_seed, mode)
    if strategy == "persistent":
        reader = PersistentReader(chain_seed, mask)
    elif strategy == "adaptive_v2":
        reader = TrajectoryWatchdogReader(chain_seed, mask)
    signed_finals = []
    for k in range(K):
        b_k = b_seq[k]
        if strategy == "fresh":
            diff = read_fresh(chain_seed, mask, idle, b_k, k)
        else:
            diff = reader.read(b_k, idle)
        signed_finals.append(diff * b_k)
    n_resets = reader.n_resets if strategy == "adaptive_v2" else None
    return signed_finals, n_resets


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
    print("=== P11 -- WATCHDOG PAR TRAJECTOIRE : reparer l'angle mort du v1 ===\n")

    chain_seeds = list(range(N_CHAINS))
    results = {}
    resets = {}
    for mode in ["same", "flip", "mixed"]:
        for strategy in ["fresh", "persistent", "adaptive_v2"]:
            tlab = time.time()
            mat = np.empty((K, N_CHAINS))
            n_resets_list = []
            for i, cs in enumerate(chain_seeds):
                sf, nr = run_chain(cs, mode, strategy)
                mat[:, i] = sf
                if nr is not None:
                    n_resets_list.append(nr)
            results[(mode, strategy)] = mat
            resets[(mode, strategy)] = np.mean(n_resets_list) if n_resets_list else None
            extra = (f"  (resets moyens/chaine = {resets[(mode, strategy)]:.2f}/{K-1})"
                     if resets[(mode, strategy)] is not None else "")
            print(f"[{mode:<6}/{strategy:<12}] {time.time()-tlab:.1f}s -- "
                  f"tours 2..K moyenne = {mat[1:].mean():+.3f}" + extra)

    print(f"\n{'mode/strategie':<24}" + "".join(f"{'tour '+str(k+1):>10}" for k in range(K))
          + f"{'moy. tours 2-4':>16}")
    print("-" * (24 + 10 * K + 16))
    for (mode, strategy), mat in results.items():
        label = f"{mode}/{strategy}"
        print(f"{label:<24}" + "".join(f"{mat[k].mean():>10.3f}" for k in range(K))
              + f"{mat[1:].mean():>16.3f}")

    print("\n=== VERDICT (predictions pre-fixees, bootstrap apparie par chaine, tours 2..K) ===")

    def cmp(mode, s1, s2, label):
        a = results[(mode, s1)][1:, :].mean(axis=0)
        b = results[(mode, s2)][1:, :].mean(axis=0)
        d, lo, hi = boot_ci_paired(a, b)
        if lo > 0:
            tag = f"{s1} BAT {s2}"
        elif hi < 0:
            tag = f"{s2} BAT {s1}"
        else:
            tag = "parite (IC couvre 0)"
        print(f"  {label:<48} delta={d:+.4f} CI[{lo:+.4f},{hi:+.4f}]  -> {tag}")

    print("  1. SAME (ADAPTIVE_V2 doit approcher PERSISTENT) :")
    cmp("same", "adaptive_v2", "persistent", "     ADAPTIVE_V2 vs PERSISTENT")
    cmp("same", "adaptive_v2", "fresh", "     ADAPTIVE_V2 vs FRESH")

    print("  2. FLIP (ADAPTIVE_V2 doit se rapprocher de FRESH -- le point rate par v1) :")
    cmp("flip", "adaptive_v2", "persistent", "     ADAPTIVE_V2 vs PERSISTENT")
    cmp("flip", "adaptive_v2", "fresh", "     ADAPTIVE_V2 vs FRESH")

    print("  3. MIXED -- LE VRAI TEST :")
    cmp("mixed", "adaptive_v2", "fresh", "     ADAPTIVE_V2 vs FRESH")
    cmp("mixed", "adaptive_v2", "persistent", "     ADAPTIVE_V2 vs PERSISTENT")

    print("\n  Taux de detection de conflit (resets moyens/chaine, sur 3 occasions, tours 2-4) :")
    for mode in ["same", "flip", "mixed"]:
        r = resets[(mode, "adaptive_v2")]
        print(f"     {mode:<6} : {r:.2f}/{K-1}")
    same_rate = resets[("same", "adaptive_v2")]
    flip_rate = resets[("flip", "adaptive_v2")]
    print(f"  -> {'v1 corrige : FLIP detecte nettement plus que SAME' if flip_rate > same_rate * 1.5 else 'toujours pas de separation nette FLIP/SAME -- angle mort probablement pas repare'}")

    with (FIG / "p11_trajectory_watchdog_poc.csv").open("w", encoding="utf-8") as f:
        f.write("mode,strategy,chain_seed,round,signed_final\n")
        for (mode, strategy), mat in results.items():
            for k in range(K):
                for i, cs in enumerate(chain_seeds):
                    f.write(f"{mode},{strategy},{cs},{k+1},{mat[k, i]:.6f}\n")
    print(f"\n[csv] {FIG / 'p11_trajectory_watchdog_poc.csv'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
        rounds = np.arange(1, K + 1)
        colors = {"fresh": "#1f77b4", "persistent": "#d62728", "adaptive_v2": "#2ca02c"}
        for ax, mode in zip(axes, ["same", "flip", "mixed"]):
            for strategy in ["fresh", "persistent", "adaptive_v2"]:
                mat = results[(mode, strategy)]
                means = mat.mean(axis=1)
                sems = mat.std(axis=1) / np.sqrt(N_CHAINS)
                ax.errorbar(rounds, means, yerr=sems, color=colors[strategy], marker="o",
                            label=strategy, capsize=3)
            ax.axhline(0, c="k", lw=0.8)
            ax.set_xlabel("tour")
            ax.set_title(mode.upper())
            ax.set_xticks(rounds)
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("signed_final")
        axes[0].legend(fontsize=8)
        fig.suptitle("Watchdog par trajectoire (v2) : repare-t-il l'angle mort du v1 ?", fontsize=11)
        plt.tight_layout()
        plt.savefig(FIG / "p11_trajectory_watchdog_poc.png", dpi=140)
        print(f"[png] {FIG / 'p11_trajectory_watchdog_poc.png'}")
    except Exception as e:
        print(f"[png] skipped: {e}")

    print(f"\nWall time: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
