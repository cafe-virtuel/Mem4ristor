#!/usr/bin/env python3
"""
P11 -- LE WATCHDOG V3 (ADAPTIVE_V3) : validation de la 4eme voie du watchdog
=============================================================================
Cree : 2026-07-13 (Antigravity).
Ce script implemente le watchdog ADAPTIVE_V3 base sur la signature de changement
de la difference de tension late-early :
    f2 = (mean(v_df[1:]) - v_df[0]) * (-guess_prev)
Avec un seuil fixe a -0.043.
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


class VoltageWatchdogReader:
    """Watchdog ADAPTIVE_V3 utilisant la signature de tension late-early
    pour detecter les conflits subjectifs sans connaitre le signe b_k."""

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
        # difference de tension initiale (avant premier pas de ce tour)
        v_df_0 = float(self.net.model.v[idle].mean() - self.net.model.v[self.mask].mean())

        stim = np.zeros(pw.N)
        stim[self.mask] = b * pw.B_E
        
        v_df_trace = []
        for _ in range(pw.T_READ):
            self.net.step(I_stimulus=stim)
            v_df_t = float(self.net.model.v[idle].mean() - self.net.model.v[self.mask].mean())
            v_df_trace.append(v_df_t)
            
        diff = float(np.real(self.net.model.u_c[idle].mean() - self.net.model.u_c[self.mask].mean()))
        guess = 1 if diff >= 0 else -1
        
        if self.prev_guess is not None:
            f2 = (np.mean(v_df_trace) - v_df_0) * (-self.prev_guess)
            # Seuil fixe a -0.043
            if f2 < -0.043:
                self.build_idx += 1
                self._build_net()
                self.n_resets += 1
                guess = None  # Reset de la memoire de guess apres reconstruction

        self.prev_guess = guess
        return diff


def run_chain(chain_seed, mode, strategy):
    mask, idle = pw.build_group(chain_seed)
    b_seq = make_b_sequence(chain_seed, mode)
    if strategy == "persistent":
        reader = PersistentReader(chain_seed, mask)
    elif strategy == "adaptive_v3":
        reader = VoltageWatchdogReader(chain_seed, mask)
    signed_finals = []
    for k in range(K):
        b_k = b_seq[k]
        if strategy == "fresh":
            diff = read_fresh(chain_seed, mask, idle, b_k, k)
        else:
            diff = reader.read(b_k, idle)
        signed_finals.append(diff * b_k)
    n_resets = reader.n_resets if strategy == "adaptive_v3" else None
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
    print("=== P11 -- WATCHDOG V3 (ADAPTIVE_V3) : validation de la 4eme voie ===\n")

    chain_seeds = list(range(N_CHAINS))
    results = {}
    resets = {}
    for mode in ["same", "flip", "mixed"]:
        for strategy in ["fresh", "persistent", "adaptive_v3"]:
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

    print("  1. SAME (ADAPTIVE_V3 doit approcher PERSISTENT) :")
    cmp("same", "adaptive_v3", "persistent", "     ADAPTIVE_V3 vs PERSISTENT")
    cmp("same", "adaptive_v3", "fresh", "     ADAPTIVE_V3 vs FRESH")

    print("  2. FLIP (ADAPTIVE_V3 doit se rapprocher de FRESH) :")
    cmp("flip", "adaptive_v3", "persistent", "     ADAPTIVE_V3 vs PERSISTENT")
    cmp("flip", "adaptive_v3", "fresh", "     ADAPTIVE_V3 vs FRESH")

    print("  3. MIXED -- LE VRAI TEST :")
    cmp("mixed", "adaptive_v3", "fresh", "     ADAPTIVE_V3 vs FRESH")
    cmp("mixed", "adaptive_v3", "persistent", "     ADAPTIVE_V3 vs PERSISTENT")

    print("\n  Taux de detection de conflit (resets moyens/chaine, sur 3 occasions, tours 2-4) :")
    for mode in ["same", "flip", "mixed"]:
        r = resets[(mode, "adaptive_v3")]
        print(f"     {mode:<6} : {r:.2f}/{K-1}")

    with (FIG / "p11_watchdog_v3_poc.csv").open("w", encoding="utf-8") as f:
        f.write("mode,strategy,chain_seed,round,signed_final\n")
        for (mode, strategy), mat in results.items():
            for k in range(K):
                for i, cs in enumerate(chain_seeds):
                    f.write(f"{mode},{strategy},{cs},{k+1},{mat[k, i]:.6f}\n")
    print(f"\n[csv] {FIG / 'p11_watchdog_v3_poc.csv'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
        rounds = np.arange(1, K + 1)
        colors = {"fresh": "#1f77b4", "persistent": "#d62728", "adaptive_v3": "#2ca02c"}
        for ax, mode in zip(axes, ["same", "flip", "mixed"]):
            for strategy in ["fresh", "persistent", "adaptive_v3"]:
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
        fig.suptitle("Watchdog ADAPTIVE_V3 (Tension) : memoire vs cicatrice", fontsize=11)
        plt.tight_layout()
        plt.savefig(FIG / "p11_watchdog_v3_poc.png", dpi=140)
        print(f"[png] {FIG / 'p11_watchdog_v3_poc.png'}")
    except Exception as e:
        print(f"[png] skipped: {e}")

    print(f"\nWall time: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
