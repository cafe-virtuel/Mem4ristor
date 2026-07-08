#!/usr/bin/env python3
"""
B5 -- COMPARAISON A L'ETAT DE L'ART REEL : Mem4ristor vs Echo State Network sur NARMA10.

Contexte. Le benchmark du preprint bat des modeles-jouets (Kuramoto / Voter / Consensus).
B5 (docs/FUTURE_WORK.md) demande la comparaison qui compte : un vrai reservoir de reference.
Le POC reservoir_narma10_poc.py (B1) comparait FULL/FROZEN_U/DECOUPLE de Mem4ristor ENTRE EUX
(NRMSE > 1.0, note honnete : "mauvais reservoir dans l'absolu"). Ici on ajoute la BASELINE
CANONIQUE du reservoir computing -- l'Echo State Network (ESN, Jaeger 2001) -- sur la MEME tache.

Comparaison LOYALE :
  - Meme tache NARMA10 (rc.make_narma10), meme split washout/train/test, meme readout ridge,
    meme taille N=100, memes seeds.
  - Chaque modele a SON propre balayage d'hyperparametres (Mem4ristor : input_scale ;
    ESN : rayon spectral x input_scale x fuite), meilleur NRMSE retenu -> aucun handicape.
  - IC bootstrap (apparie par seed) sur l'ecart NRMSE(Mem4ristor FULL) - NRMSE(ESN).

But : positionner honnetement. Avantage reel ? parite ? niche ? desavantage ? La reponse
attendue (vu B1) est que Mem4ristor n'est PAS un bon reservoir NARMA10 -- sa valeur est le
mecanisme du doute (diversite maintenue), pas la performance memoire brute. On le mesure et
on le dit tel quel : savoir sur quoi on N'EST PAS competitif est une information de cadrage.

Sortie : figures/b5_esn_comparison.csv + .png + verdict.
Cree : 2026-07-08 (Claude Opus 4.8) -- B5 comparaison SOTA (docs/FUTURE_WORK.md).
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
sys.path.insert(0, str(ROOT / "experiments"))
import reservoir_narma10_poc as rc  # noqa: E402  (make_narma10, run_reservoir, ridge_nrmse, constants)

CSV = ROOT / "figures" / "b5_esn_comparison.csv"
PNG = ROOT / "figures" / "b5_esn_comparison.png"

N = rc.SIZE * rc.SIZE               # 100, meme taille que le reservoir Mem4ristor
SEEDS = list(range(8))
N_BOOT = 10000
RNG_BOOT = np.random.RandomState(20260708)

# Balayage ESN (standard). Grilles comparables en cardinalite au balayage Mem4ristor (input_scale).
ESN_RHO = [0.8, 0.95, 1.1]          # rayon spectral (bord du chaos ~1)
ESN_ISCALE = [0.1, 0.5, 1.0]        # echelle d'entree
ESN_LEAK = [0.3, 1.0]               # taux de fuite (leaky integrator)
ESN_DENSITY = 0.1                   # densite de la matrice recurrente

def make_esn(seed, rho, density=ESN_DENSITY):
    """Matrice recurrente sparse aleatoire, mise a l'echelle au rayon spectral rho."""
    rng = np.random.RandomState(5000 + seed)
    W = rng.uniform(-1.0, 1.0, (N, N))
    mask = rng.uniform(0.0, 1.0, (N, N)) < density
    W = W * mask
    eig = np.max(np.abs(np.linalg.eigvals(W)))
    if eig > 1e-9:
        W = W * (rho / eig)
    w_in = rng.uniform(-1.0, 1.0, N)
    return W, w_in

def run_esn(u_in, W, w_in, iscale, leak):
    """Etats d'un ESN leaky-tanh standard : x <- (1-a)x + a*tanh(W_in u + W x)."""
    x = np.zeros(N)
    states = np.zeros((len(u_in), N))
    for t, ui in enumerate(u_in):
        pre = w_in * (iscale * ui) + W @ x
        x = (1.0 - leak) * x + leak * np.tanh(pre)
        states[t] = x
    return states

def boot_ci_paired(a, b):
    d = np.asarray(a, float) - np.asarray(b, float); n = len(d)
    m = np.empty(N_BOOT)
    for k in range(N_BOOT):
        m[k] = d[RNG_BOOT.randint(0, n, n)].mean()
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

def main():
    CSV.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_steps = rc.T_WASH + rc.T_TRAIN + rc.T_TEST
    rows = []
    m4_full, m4_dec, esn_best = [], [], []

    print(f"{'seed':>5}{'M4R_FULL':>10}{'M4R_DEC':>10}{'ESN':>10}{'ESN(rho,is,leak)':>20}")
    print("-" * 55)
    for seed in SEEDS:
        u_in, y = rc.make_narma10(n_steps, seed=seed)
        rng_mask = np.random.default_rng(1000 + seed)
        w_in_base = rng_mask.uniform(-1.0, 1.0, N)

        # --- Mem4ristor FULL et DECOUPLE : meilleur input_scale (comme le POC) ---
        def best_m4(cond):
            best = np.inf
            for scale in rc.INPUT_SCALES:
                states = rc.run_reservoir(u_in, w_in_base * scale, cond, seed)
                nrmse, _ = rc.ridge_nrmse(states, y)
                best = min(best, nrmse)
            return best
        f_nrmse = best_m4("FULL")
        d_nrmse = best_m4("DECOUPLE")

        # --- ESN : meilleur sur (rho, input_scale, leak) ---
        e_best, e_arg = np.inf, None
        for rho in ESN_RHO:
            W, w_in = make_esn(seed, rho)
            for iscale in ESN_ISCALE:
                for leak in ESN_LEAK:
                    states = run_esn(u_in, W, w_in, iscale, leak)
                    nrmse, _ = rc.ridge_nrmse(states, y)
                    rows.append((seed, "ESN", rho, iscale, leak, nrmse))
                    if nrmse < e_best:
                        e_best, e_arg = nrmse, (rho, iscale, leak)
        rows.append((seed, "M4R_FULL", -1, -1, -1, f_nrmse))
        rows.append((seed, "M4R_DECOUPLE", -1, -1, -1, d_nrmse))
        m4_full.append(f_nrmse); m4_dec.append(d_nrmse); esn_best.append(e_best)
        print(f"{seed:>5}{f_nrmse:>10.4f}{d_nrmse:>10.4f}{e_best:>10.4f}"
              f"{str(e_arg):>20}")

    m4 = np.array(m4_full); md = np.array(m4_dec); es = np.array(esn_best)
    print("\n=== RESUME (NRMSE NARMA10, plus bas = mieux ; meilleur hyperparam par modele) ===")
    print(f"  Mem4ristor FULL     : {m4.mean():.4f} +/- {m4.std():.4f}")
    print(f"  Mem4ristor DECOUPLE : {md.mean():.4f} +/- {md.std():.4f}")
    print(f"  Echo State Network  : {es.mean():.4f} +/- {es.std():.4f}")

    dm, dlo, dhi = boot_ci_paired(m4, es)      # M4R - ESN (positif = ESN meilleur)
    print("\n=== VERDICT B5 (honnete) ===")
    print(f"Ecart NRMSE (Mem4ristor FULL - ESN) = {dm:+.4f} CI[{dlo:+.4f},{dhi:+.4f}]")
    if dlo > 0:
        factor = m4.mean() / max(es.mean(), 1e-9)
        print(f"  -> L'ESN BAT Mem4ristor sur NARMA10 (~{factor:.1f}x meilleur NRMSE), IC excluant 0.")
        print("     POSITIONNEMENT HONNETE : Mem4ristor n'est PAS un reservoir NARMA10 competitif.")
        print("     Sa contribution est le MECANISME du doute (diversite maintenue, anti-synchro),")
        print("     pas la performance memoire brute. C'est une information de cadrage, pas un echec")
        print("     cache : on sait desormais sur quelle tache ne pas le vendre.")
    elif dhi < 0:
        print("  -> Mem4ristor BAT l'ESN (IC excluant 0) : avantage reel sur NARMA10.")
    else:
        print("  -> PARITE statistique (IC couvre 0) : ni avantage ni desavantage net.")
    if es.mean() < 1.0 <= m4.mean():
        print(f"  [absolu] ESN < 1.0 (reservoir utile) ; Mem4ristor > 1.0 (pire que predire la moyenne)")
        print("           -> l'ecart n'est pas cosmetique : NARMA10 exige une memoire que le lattice")
        print("              FHN couple ne fournit pas. Tache defavorable a l'architecture, assume.")

    with CSV.open("w", encoding="utf-8") as f:
        f.write("seed,model,rho,input_scale,leak,nrmse\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]:.6f}\n")
    print(f"\n[csv] {CSV}")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
        labels = ["ESN\n(SOTA)", "Mem4ristor\nFULL", "Mem4ristor\nDECOUPLE"]
        means = [es.mean(), m4.mean(), md.mean()]
        stds = [es.std(), m4.std(), md.std()]
        colors = ["#1f77b4", "#2ca02c", "#7f7f7f"]
        ax1.bar(labels, means, yerr=stds, color=colors, edgecolor="k", capsize=5)
        ax1.axhline(1.0, ls=":", c="red", label="NRMSE=1 (predire la moyenne)")
        ax1.set_ylabel("NARMA10 NRMSE (plus bas = mieux)")
        ax1.set_title(f"Mem4ristor vs Echo State Network (N={N}, {len(SEEDS)} seeds)")
        ax1.legend(fontsize=8); ax1.grid(axis="y", alpha=0.3)
        # nuage apparie par seed
        x = np.arange(len(SEEDS))
        ax2.plot(x, es, "o-", c="#1f77b4", label="ESN")
        ax2.plot(x, m4, "s-", c="#2ca02c", label="Mem4ristor FULL")
        ax2.axhline(1.0, ls=":", c="red")
        ax2.set_xlabel("seed"); ax2.set_ylabel("NRMSE")
        ax2.set_title("Apparie par seed"); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
        fig.suptitle("B5 : positionnement vs l'etat de l'art reservoir (NARMA10)", fontsize=11)
        plt.tight_layout(); plt.savefig(PNG, dpi=140)
        print(f"[png] {PNG}")
    except Exception as e:
        print(f"[png] skipped: {e}")
    print(f"\nWall time: {time.time()-t0:.1f}s")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
