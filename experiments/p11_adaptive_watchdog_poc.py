#!/usr/bin/env python3
"""
P11 -- LE WATCHDOG SAGE/FOU ADAPTATIF : peut-on avoir la memoire ET eviter
la cicatrice, sans connaitre le regime a l'avance ?
=============================================================================
Cree : 2026-07-13 (Claude Sonnet 5, L'Ingenieur). Suite de
p11_continuous_memory_scar_poc.py (meme jour), qui a confirme que memoire et
cicatrice sont le MEME mecanisme (persistance de u_c), avec un signe oppose
selon que le monde reste coherent (SAME, memoire gagne) ou change (FLIP,
cicatrice coute). Limite pratique : PERSISTENT vs FRESH est un choix FIGE
A L'AVANCE -- en deploiement reel, on ne connait jamais le regime avant coup.

IMPORTANT -- precision sur le nom. Le `consolidation_watchdog` deja present
dans dynamics.py (07/07) est un cycle TEMPOREL FIXE (bascule FOU<->SAGE sur
une horloge t_explore/t_consolidate), PAS adaptatif au contenu du signal --
il ne fait pas ce dont on a besoin ici et n'est PAS reutilise tel quel
(coeur non touche). Ce POC construit un NOUVEAU detecteur, au niveau du
harness experimental, inspire de la MEME philosophie bicamerale
(SAGE=consolider/persister, FOU=explorer/reinitialiser) mais pilote par le
signal lui-meme plutot que par une horloge.

DESIGN, causal, AUCUNE fuite du signe vrai b_k dans la decision (seul ce
que le systeme peut observer lui-meme est utilise) :
  A la fin de chaque tour k>=2, on compare le nouveau diff_k au GUESS
  (pas le vrai signe) du tour precedent : coherence_k = diff_k * guess_{k-1}.
  Si coherence_k <= 0 (le nouveau signal contredit ou ne confirme pas la
  direction precedente) -> RESET du reseau avant le tour k+1 (mode FOU).
  Sinon -> garde l'etat (mode SAGE, comme PERSISTENT).
  Seuil a 0, sans reglage -- aucun hyperparametre ajuste sur les donnees.

TROIS REGIMES DE SIGNES tires par chaine (b_k) :
  - SAME  : signe identique repete (terrain de la memoire pure)
  - FLIP  : signe invers a chaque tour (terrain de la cicatrice pure)
  - MIXED : signe tire independamment a chaque tour (le regime REALISTE,
    ou on ne sait jamais a l'avance si le monde va rester coherent ou
    changer -- le vrai test de valeur de l'adaptativite).

PREDICTIONS PRE-FIXEES (avant de lancer) :
  1. Sur SAME : ADAPTIVE ~= PERSISTENT (jamais de conflit detecte des le
     2e tour, jamais de reset).
  2. Sur FLIP : ADAPTIVE entre PERSISTENT et FRESH, plus proche de FRESH
     (le conflit n'est detectable qu'APRES l'avoir subi une fois -- un
     reset reactif, pas preventif).
  3. Sur MIXED (le vrai test) : ADAPTIVE bat A LA FOIS FRESH pur ET
     PERSISTENT pur -- c'est la seule condition ou l'adaptativite peut
     demontrer une valeur que les deux extremes fixes n'ont pas.

Statut : exploratoire, hors preprint, aucune modification de dynamics.py.
Guardian doit rester 14/14. Sorties : figures/p11_adaptive_watchdog_poc.csv + .png
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

FIG = ROOT / "figures"
K = 4
N_CHAINS = 40


def round_b(chain_seed, k):
    rng = np.random.RandomState(910_000 + chain_seed * 100 + k)
    return 1 if rng.random() < 0.5 else -1


def make_b_sequence(chain_seed, mode):
    rng = np.random.RandomState(920_000 + chain_seed)
    b1 = 1 if rng.random() < 0.5 else -1
    if mode == "same":
        return [b1] * K
    if mode == "flip":
        return [b1 * ((-1) ** k) for k in range(K)]
    return [round_b(chain_seed, k) for k in range(K)]  # mixed : tirage independant


class WatchdogReader:
    """SAGE (persiste) tant que le nouveau signal confirme le guess
    precedent ; bascule FOU (reset) des que coherence_k <= 0. Ne voit
    JAMAIS le vrai signe b_k, seulement ses propres guess passes."""

    def __init__(self, chain_seed, mask):
        self.chain_seed = chain_seed
        self.mask = mask
        self.build_idx = 0
        self.prev_guess = None
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
        for _ in range(pw.T_READ):
            self.net.step(I_stimulus=stim)
        diff = float(np.real(self.net.model.u_c[idle].mean() - self.net.model.u_c[self.mask].mean()))
        guess = 1 if diff >= 0 else -1
        if self.prev_guess is not None and diff * self.prev_guess <= 0:
            self.build_idx += 1
            self._build_net()
            self.n_resets += 1
        self.prev_guess = guess
        return diff


def run_chain(chain_seed, mode, strategy):
    mask, idle = pw.build_group(chain_seed)
    b_seq = make_b_sequence(chain_seed, mode)
    if strategy == "persistent":
        reader = PersistentReader(chain_seed, mask)
    elif strategy == "adaptive":
        reader = WatchdogReader(chain_seed, mask)
    signed_finals = []
    for k in range(K):
        b_k = b_seq[k]
        if strategy == "fresh":
            diff = read_fresh(chain_seed, mask, idle, b_k, k)
        else:
            diff = reader.read(b_k, idle)
        signed_finals.append(diff * b_k)
    n_resets = reader.n_resets if strategy == "adaptive" else None
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
    print("=== P11 -- WATCHDOG SAGE/FOU ADAPTATIF : la memoire sans la cicatrice ? ===\n")

    chain_seeds = list(range(N_CHAINS))
    results = {}
    resets = {}
    for mode in ["same", "flip", "mixed"]:
        for strategy in ["fresh", "persistent", "adaptive"]:
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
            print(f"[{mode:<6}/{strategy:<11}] {time.time()-tlab:.1f}s -- "
                  f"tours 2..K moyenne = {mat[1:].mean():+.3f}" +
                  (f"  (resets moyens/chaine = {resets[(mode, strategy)]:.2f}/{K-1})"
                   if resets[(mode, strategy)] is not None else ""))

    print(f"\n{'mode/strategie':<22}" + "".join(f"{'tour '+str(k+1):>10}" for k in range(K))
          + f"{'moy. tours 2-4':>16}")
    print("-" * (22 + 10 * K + 16))
    for (mode, strategy), mat in results.items():
        label = f"{mode}/{strategy}"
        print(f"{label:<22}" + "".join(f"{mat[k].mean():>10.3f}" for k in range(K))
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
        print(f"  {label:<45} delta={d:+.4f} CI[{lo:+.4f},{hi:+.4f}]  -> {tag}")
        return d, lo, hi

    print("  1. SAME (ADAPTIVE doit approcher PERSISTENT) :")
    cmp("same", "adaptive", "persistent", "     ADAPTIVE vs PERSISTENT")
    cmp("same", "adaptive", "fresh", "     ADAPTIVE vs FRESH")

    print("  2. FLIP (ADAPTIVE doit etre entre les deux, proche de FRESH) :")
    cmp("flip", "adaptive", "persistent", "     ADAPTIVE vs PERSISTENT")
    cmp("flip", "adaptive", "fresh", "     ADAPTIVE vs FRESH")

    print("  3. MIXED -- LE VRAI TEST (ADAPTIVE doit battre les DEUX extremes fixes) :")
    cmp("mixed", "adaptive", "fresh", "     ADAPTIVE vs FRESH")
    cmp("mixed", "adaptive", "persistent", "     ADAPTIVE vs PERSISTENT")

    print("\n  Resets moyens par chaine (sur 3 occasions possibles, tours 2-4) :")
    for mode in ["same", "flip", "mixed"]:
        r = resets[(mode, "adaptive")]
        print(f"     {mode:<6} : {r:.2f}/{K-1}")

    with (FIG / "p11_adaptive_watchdog_poc.csv").open("w", encoding="utf-8") as f:
        f.write("mode,strategy,chain_seed,round,signed_final\n")
        for (mode, strategy), mat in results.items():
            for k in range(K):
                for i, cs in enumerate(chain_seeds):
                    f.write(f"{mode},{strategy},{cs},{k+1},{mat[k, i]:.6f}\n")
    print(f"\n[csv] {FIG / 'p11_adaptive_watchdog_poc.csv'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
        rounds = np.arange(1, K + 1)
        colors = {"fresh": "#1f77b4", "persistent": "#d62728", "adaptive": "#2ca02c"}
        for ax, mode in zip(axes, ["same", "flip", "mixed"]):
            for strategy in ["fresh", "persistent", "adaptive"]:
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
        fig.suptitle("Watchdog adaptatif : la memoire sans la cicatrice ?", fontsize=11)
        plt.tight_layout()
        plt.savefig(FIG / "p11_adaptive_watchdog_poc.png", dpi=140)
        print(f"[png] {FIG / 'p11_adaptive_watchdog_poc.png'}")
    except Exception as e:
        print(f"[png] skipped: {e}")

    print(f"\nWall time: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
