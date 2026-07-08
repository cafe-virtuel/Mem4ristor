#!/usr/bin/env python3
"""
POC PONT M4R <-> LLM : le doute contre l'effondrement de rang (oversmoothing) en profondeur.

Idee (Julien, 2026-07-08). Dans un transformer profond, l'attention pure fait CONVERGER toutes
les representations de tokens vers la meme direction ("rank collapse" / oversmoothing) : les tokens
se ressemblent, et le canal d'information entre couches se retrecit (le "goulot d'etranglement" que
Julien pointe). Ces deux formulations sont UNE seule grandeur : le RANG EFFECTIF des representations
en fonction de la profondeur.

C'est la MORT COGNITIVE de M4R transposee. Le resultat central du preprint -- le doute empeche la
synchronisation et maintient la diversite -- devrait, comme principe, MAINTENIR le rang la ou
l'attention pure l'effondre. On le teste comme une ablation, exactement comme FROZEN_U vs FULL :

  ATTRACTIVE (analogue FROZEN_U) : couplage toujours attractif -> chaque token se deplace vers le
      contexte attendu = moyennage = effondrement. C'est l'attention pure (residuelle).
  DOUBT (analogue FULL)          : couplage a POLARITE modulee par le doute u, kernel M4R
      f(u)=tanh(pi(0.5-u)) : attractif si doute bas, REPULSIF si doute haut. Le doute d'un token
      monte avec la "pression de conformite" (a quel point il est tire vers le consensus) -> quand
      un token va perdre son identite, il se met a REPOUSSER -> maintient la distinction.

GARDE-FOU d'honnetete (grave 2026-07-07) : hyperparametres FIXES ci-dessous AVANT de regarder les
resultats ; on rapporte le rang quel qu'il soit ; le rang eleve = distinction MAINTENUE (pas une
preuve d'utilite calculatoire aval -- ce POC teste le MECANISME, pas une tache LLM reelle).

Metriques par profondeur : rang effectif (participation ratio des valeurs singulieres) + similarite
cosinus moyenne entre tokens (indicateur direct d'effondrement). IC bootstrap sur le rang final.

Sortie : figures/llm_doubt_rank_poc.csv + .png + verdict.
Cree : 2026-07-08 (Claude Opus 4.8) -- pont M4R/LLM (idee de Julien, option 2).
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
CSV = ROOT / "figures" / "llm_doubt_rank_poc.csv"
PNG = ROOT / "figures" / "llm_doubt_rank_poc.png"

# --- Hyperparametres FIXES avant de regarder les resultats (honnetete) ---
T = 64            # nombre de tokens
D = 48            # dimension de representation
L = 40            # profondeur (couches)
SEEDS = list(range(10))
EPS = 0.5         # pas de mise a jour (identique aux 2 conditions)
DOUBT_RATE = 0.3  # vitesse de relaxation du doute vers la pression de conformite
N_BOOT = 10000
RNG_BOOT = np.random.RandomState(20260708)

def softmax_rows(M):
    M = M - M.max(axis=1, keepdims=True)
    E = np.exp(M)
    return E / (E.sum(axis=1, keepdims=True) + 1e-12)

def row_normalize(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

def eff_rank(X):
    """Rang effectif = participation ratio des valeurs singulieres (centrees).
    Robuste a l'effondrement : si aucune direction n'a de spread significatif (tous les tokens
    identiques), le rang effectif est 1 (rang-1), pas 0. Seuil RELATIF a sigma_max pour ignorer
    le bruit numerique (sinon le participation ratio d'un spectre de bruit ~egal remonte a tort)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(Xc, compute_uv=False)
    smax = sv.max() if sv.size else 0.0
    if smax < 1e-9:                       # effondrement complet : plus aucun spread reel
        return 1.0
    sv = sv[sv > 1e-9 * smax]             # jette les singulieres = bruit numerique
    return float((sv.sum() ** 2) / np.sum(sv ** 2))

def mean_cos(X):
    Xn = row_normalize(X)
    S = Xn @ Xn.T
    iu = np.triu_indices(T, k=1)
    return float(np.mean(S[iu]))

def run(condition, seed):
    rng = np.random.RandomState(seed)
    X = row_normalize(rng.standard_normal((T, D)))
    u = np.zeros(T)                          # doute initial bas (comporte comme l'attention au debut)
    ranks = [eff_rank(X)]; coss = [mean_cos(X)]
    for _ in range(L):
        A = softmax_rows(X @ X.T / np.sqrt(D))     # attention (qui attend qui)
        ctx = A @ X                                 # contexte attendu par token
        delta = ctx - X                             # pression : tire chaque token vers son contexte
        if condition == "ATTRACTIVE":
            X = X + EPS * delta                      # moyennage pur -> effondrement (analogue FROZEN_U)
        else:  # DOUBT (analogue FULL)
            pressure = np.linalg.norm(delta, axis=1)
            p_norm = np.tanh(pressure / (np.median(pressure) + 1e-9))   # -> ~[0,1)
            u = u + DOUBT_RATE * (p_norm - u)        # le doute monte avec la pression de conformite
            f = np.tanh(np.pi * (0.5 - u))           # kernel M4R : + attractif / - repulsif
            X = X + EPS * f[:, None] * delta
        X = row_normalize(X)                         # normalisation type layernorm (identique aux 2)
        ranks.append(eff_rank(X)); coss.append(mean_cos(X))
    return np.array(ranks), np.array(coss)

def boot_ci(vec):
    vec = np.asarray(vec, float); n = len(vec)
    m = np.empty(N_BOOT)
    for b in range(N_BOOT):
        m[b] = vec[RNG_BOOT.randint(0, n, n)].mean()
    return float(vec.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

def main():
    CSV.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    conds = ["ATTRACTIVE", "DOUBT"]
    curves = {c: {"rank": [], "cos": []} for c in conds}
    for seed in SEEDS:
        for c in conds:
            r, co = run(c, seed)
            curves[c]["rank"].append(r); curves[c]["cos"].append(co)
    rank_mean = {c: np.mean(curves[c]["rank"], axis=0) for c in conds}
    cos_mean = {c: np.mean(curves[c]["cos"], axis=0) for c in conds}

    print(f"POC pont M4R/LLM -- T={T} tokens, D={D}, L={L} couches, {len(SEEDS)} seeds")
    print(f"Rang effectif (max possible = {min(T,D)})\n")
    print(f"{'couche':>7}{'ATTRACTIVE rank':>18}{'DOUBT rank':>14}{'ATTR cos':>11}{'DOUBT cos':>11}")
    print("-" * 61)
    for l in [0, 1, 2, 5, 10, 20, L]:
        print(f"{l:>7}{rank_mean['ATTRACTIVE'][l]:>18.2f}{rank_mean['DOUBT'][l]:>14.2f}"
              f"{cos_mean['ATTRACTIVE'][l]:>11.3f}{cos_mean['DOUBT'][l]:>11.3f}")

    # rang final par seed -> IC
    fa = [curves["ATTRACTIVE"]["rank"][s][-1] for s in range(len(SEEDS))]
    fd = [curves["DOUBT"]["rank"][s][-1] for s in range(len(SEEDS))]
    ma, la, ha = boot_ci(fa)
    md, ld, hd = boot_ci(fd)
    diff = np.array(fd) - np.array(fa)
    dm, dlo, dhi = boot_ci(diff)

    print("\n=== VERDICT (honnete) ===")
    print(f"Rang effectif final (couche {L}) : ATTRACTIVE={ma:.2f} [{la:.2f},{ha:.2f}]  "
          f"DOUBT={md:.2f} [{ld:.2f},{hd:.2f}]")
    print(f"Ecart DOUBT - ATTRACTIVE = {dm:+.2f} CI[{dlo:+.2f},{dhi:+.2f}]")
    if dlo > 0 and ma < 3.0:
        print("  -> LE PRINCIPE TRANSFERE : l'attention pure s'effondre (rang -> quasi 1, tokens")
        print("     colineaires), le couplage module par le doute MAINTIENT le rang. C'est FROZEN_U")
        print("     vs FULL transpose a l'espace des representations : le doute garde le goulot")
        print("     inter-couches OUVERT (diversite representationnelle preservee).")
        print("  [reserve] rang eleve = distinction maintenue, PAS une preuve d'utilite sur une vraie")
        print("            tache LLM. Prochain pas : brancher un readout/tache pour tester le AVAL.")
    elif dhi < 0:
        print("  -> Le doute n'aide pas ici (rang <= attention pure) : resultat negatif, assume.")
    else:
        print("  -> Pas d'ecart net (IC couvre 0) : le principe ne transfere pas proprement sur ce jouet.")

    with CSV.open("w", encoding="utf-8") as f:
        f.write("layer,attractive_rank,doubt_rank,attractive_cos,doubt_cos\n")
        for l in range(L + 1):
            f.write(f"{l},{rank_mean['ATTRACTIVE'][l]:.4f},{rank_mean['DOUBT'][l]:.4f},"
                    f"{cos_mean['ATTRACTIVE'][l]:.4f},{cos_mean['DOUBT'][l]:.4f}\n")
    print(f"\n[csv] {CSV}")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
        xs = np.arange(L + 1)
        for c, col in [("ATTRACTIVE", "#1f77b4"), ("DOUBT", "#d62728")]:
            rk = np.array(curves[c]["rank"])
            ax1.plot(xs, rk.mean(0), color=col, label=c, lw=2)
            ax1.fill_between(xs, rk.mean(0) - rk.std(0), rk.mean(0) + rk.std(0), color=col, alpha=0.15)
        ax1.axhline(1.0, ls=":", c="gray", label="effondrement (rang 1)")
        ax1.set_xlabel("profondeur (couche)"); ax1.set_ylabel("rang effectif")
        ax1.set_title("Le goulot inter-couches : rang qui survit en profondeur")
        ax1.legend(); ax1.grid(alpha=0.3)
        for c, col in [("ATTRACTIVE", "#1f77b4"), ("DOUBT", "#d62728")]:
            ax2.plot(xs, np.mean(curves[c]["cos"], 0), color=col, label=c, lw=2)
        ax2.set_xlabel("profondeur (couche)"); ax2.set_ylabel("similarite cosinus moyenne")
        ax2.set_title("Effondrement direct : tokens qui se ressemblent")
        ax2.legend(); ax2.grid(alpha=0.3)
        fig.suptitle("Pont M4R -> LLM : le doute (FULL) contre l'effondrement de rang de "
                     "l'attention pure (FROZEN_U)", fontsize=11)
        plt.tight_layout(); plt.savefig(PNG, dpi=140)
        print(f"[png] {PNG}")
    except Exception as e:
        print(f"[png] skipped: {e}")
    print(f"\nWall time: {time.time()-t0:.1f}s")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
