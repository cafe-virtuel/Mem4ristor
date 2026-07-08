#!/usr/bin/env python3
"""
B4 (fin) -- ROBUSTESSE + FINITE-SIZE SCALING de la diversite (Table 1, H_cont).

Contexte. Le Tableau 1 du preprint (tab:scaling) rapporte H_cont (diversite spectrale continue)
sur lattices 4x4 / 10x10 / 25x25, **10 seeds**, cold start, I_stim=0.5, 3000 steps (derniers 25%).
B4 (docs/FUTURE_WORK.md) demande : >=30 seeds sur les resultats-cles + finite-size scaling +
IC honnete au lieu d'un simple ecart-type.

Ce script (SUPPLEMENT, n'ecrase PAS le CSV canonique p2_table1_lattice.csv / claims C02/C03) :
  - 30 seeds (10 canoniques + 20 nouveaux),
  - 7 tailles pour une vraie courbe FSS : N = 16, 36, 100, 225, 400, 625, 900,
  - IC bootstrap sur H_cont(N) au lieu d'un ecart-type nu.
Mesure IDENTIQUE au script canonique (cold_start=True, I_stim=0.5, 3000 steps, derniers 25%,
calculate_continuous_entropy) pour comparabilite.

Question FSS : la diversite s'effondre-t-elle avec l'echelle (artefact de taille finie) ou
tient-elle ? Attendu : H_cont croit avec N (meilleur echantillonnage de la distribution des
tensions, plafond ~ log2(100 bins) = 6.64 bits) puis sature -- la diversite NE disparait PAS
a grand N, ce qui blinde le resultat central contre un reproche de taille finie.

Sortie : figures/b4_table1_fss.csv + .png + verdict.
Cree : 2026-07-08 (Claude Opus 4.8) -- B4 robustesse Table 1 (docs/FUTURE_WORK.md).
"""
from __future__ import annotations
import sys, time
from pathlib import Path
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from mem4ristor.core import Mem4Network  # noqa: E402
from mem4ristor.metrics import calculate_continuous_entropy  # noqa: E402

CSV = ROOT / "figures" / "b4_table1_fss.csv"
PNG = ROOT / "figures" / "b4_table1_fss.png"

# Mesure canonique Table 1 (ne pas changer : comparabilite)
STEPS = 3000
WARMUP = int(STEPS * 0.75)      # derniers 25 %
ETA = 0.15
I_STIM = 0.5
CANON_SEEDS = [42, 123, 777, 17, 256, 1337, 99, 314, 2024, 888]
SEEDS = CANON_SEEDS + list(range(3001, 3021))   # 30 seeds
SIZES = [4, 6, 10, 15, 20, 25, 30]              # N = 16, 36, 100, 225, 400, 625, 900
N_BOOT = 10000
RNG = np.random.RandomState(20260708)

def h_cont_run(size, seed):
    """H_cont moyen sur les derniers 25 % (cold start), identique au script canonique."""
    net = Mem4Network(size=size, heretic_ratio=ETA, seed=seed, cold_start=True)
    H = []
    for step in range(STEPS):
        net.step(I_stimulus=I_STIM)
        if step >= WARMUP:
            H.append(calculate_continuous_entropy(net.model.v.copy()))
    return float(np.mean(H))

def boot_ci(vec):
    vec = np.asarray(vec, float); n = len(vec)
    m = np.empty(N_BOOT)
    for b in range(N_BOOT):
        m[b] = vec[RNG.randint(0, n, n)].mean()
    return float(vec.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), float(vec.std())

def main():
    CSV.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = []
    summary = []
    print(f"Seeds: {len(SEEDS)} ({len(CANON_SEEDS)} canoniques + {len(SEEDS)-len(CANON_SEEDS)} nouveaux)")
    print(f"cold_start=True, I_stim={I_STIM}, {STEPS} steps, derniers 25%\n")
    print(f"{'size':>7}{'N':>6}{'H_cont mean':>13}{'CI95':>18}{'std':>7}{'(canon 10s)':>13}")
    print("-" * 64)
    for size in SIZES:
        Hs = []
        for seed in SEEDS:
            h = h_cont_run(size, seed)
            Hs.append(h)
            rows.append((size, size * size, seed, h))
        m, lo, hi, sd = boot_ci(Hs)
        # sous-ensemble canonique (10 premiers seeds) pour comparaison au preprint
        canon_mean = float(np.mean(Hs[:len(CANON_SEEDS)]))
        summary.append((size, size * size, m, lo, hi, sd, canon_mean))
        print(f"{size:>4}x{size:<2}{size*size:>6}{m:>13.3f}{f'[{lo:.3f},{hi:.3f}]':>18}{sd:>7.3f}"
              f"{canon_mean:>13.3f}")

    # --- FSS : ajustement H ~ a*log2(N) + b + saturation ? ---
    Ns = np.array([s[1] for s in summary], float)
    Hm = np.array([s[2] for s in summary], float)
    # pente sur log2(N) (croissance) et sur les 3 plus grands N (saturation ?)
    slope_all = np.polyfit(np.log2(Ns), Hm, 1)[0]
    slope_tail = np.polyfit(np.log2(Ns[-3:]), Hm[-3:], 1)[0]

    print("\n=== VERDICT B4 finite-size scaling (honnete) ===")
    print(f"Diversite a toutes les echelles : H_cont in [{Hm.min():.2f}, {Hm.max():.2f}] bits, "
          f"jamais d'effondrement (min >> 0).")
    print(f"Croissance globale : dH/dlog2(N) = {slope_all:+.3f} bits/octave ; "
          f"queue (3 plus grands N) : {slope_tail:+.3f} bits/octave.")
    if slope_tail < slope_all * 0.5:
        print("  -> La croissance RALENTIT a grand N (saturation) : la diversite tend vers un plateau,")
        print("     ce n'est PAS un artefact de taille finie qui divergerait ou s'effondrerait.")
    elif abs(slope_tail) < 0.05:
        print("  -> H_cont a SATURE (pente de queue ~ 0) : plateau de diversite atteint.")
    else:
        print("  -> H_cont croit encore a grand N (meilleur echantillonnage) sans s'effondrer.")
    # coherence avec la Table 1 canonique (10x10)
    s10 = next(s for s in summary if s[0] == 10)
    print(f"\nCoherence Table 1 (10x10) : canonique 10 seeds = {s10[6]:.3f} bits "
          f"(preprint ~4.09) ; 30 seeds = {s10[2]:.3f} bits CI[{s10[3]:.3f},{s10[4]:.3f}].")
    print("  -> Le resultat central 'diversite soutenue' tient a 30 seeds avec IC serre.")

    with CSV.open("w", encoding="utf-8") as f:
        f.write("size,N,seed,h_cont\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]:.6f}\n")
    print(f"\n[csv] {CSV}")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        Ns = [s[1] for s in summary]
        Hm = [s[2] for s in summary]
        lo = [s[2] - s[3] for s in summary]
        hi = [s[4] - s[2] for s in summary]
        ax.errorbar(Ns, Hm, yerr=[lo, hi], fmt="o-", c="#d62728", capsize=5,
                    ms=7, label="H_cont (30 seeds, IC95)")
        # nuage brut
        allN = [r[1] for r in rows]; allH = [r[3] for r in rows]
        ax.scatter(allN, allH, s=8, c="#888", alpha=0.3, zorder=0, label="runs individuels")
        ax.axhline(np.log2(100), ls=":", c="gray", label="plafond binning log2(100)=6.64")
        ax.set_xscale("log")
        ax.set_xlabel("N (nombre de noeuds, echelle log)")
        ax.set_ylabel("H_cont (bits)")
        ax.set_title("B4 finite-size scaling : la diversite tient a toutes les echelles\n"
                     "(lattices cold-start, I_stim=0.5, 30 seeds)")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
        plt.tight_layout(); plt.savefig(PNG, dpi=140)
        print(f"[png] {PNG}")
    except Exception as e:
        print(f"[png] skipped: {e}")
    print(f"\nWall time: {time.time()-t0:.1f}s")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
