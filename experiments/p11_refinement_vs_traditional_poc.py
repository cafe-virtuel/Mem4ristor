#!/usr/bin/env python3
"""
P11 -- LE COUT DE LA CICATRICE, COMPARE A UNE SOLUTION TRADITIONNELLE.
=============================================================================
Cree : 2026-07-13 (Claude Sonnet 5, L'Ingenieur). Suite immediate de
`p11_refinement_scar_poc.py`. Julien : « il finit par se stopper lui-meme
mais il a coute en compute avant cela -- est-il possible de faire la meme
experience avec les solutions traditionnelles pour voir ce qu'il coute par
rapport a eux ? »

Le candidat "traditionnel" le plus honnete et le moins genereux envers M4R :
un ACCUMULATEUR NAIF -- une moyenne mobile exponentielle (EMA) du signal brut
recu, aucun reseau, aucun doute, aucune dynamique FHN. Le readout le plus
simple possible pour suivre "quelle est la meilleure estimation actuelle
de la direction". Meme sequence de stimulus EXACTE que le test precedent
(T1=200 pas de signal FAIBLE dans le MAUVAIS sens, puis T2 pas de signal
FORT dans le BON sens) -- seul le "lecteur" change : M4R (FHN+lattice+doute)
vs EMA (une seule variable scalaire, aucun reseau).

Sweep de tau_ema (constante de temps de l'accumulateur, l'equivalent de
tau_u=10 pour M4R) : {5, 10, 20, 50} -- pour ne pas favoriser un choix
arbitraire, montre la plage.

Question posee : a combien de pas T2 l'EMA corrige-t-il, compare aux
T2 in {50, 150, 300, 600} deja mesures pour M4R (scar ferme entre 150
et 300) ?

Statut : exploratoire, hors preprint, aucune modification de dynamics.py.
Guardian doit rester 14/14. Sorties : figures/p11_refinement_vs_traditional_poc.csv + .png
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
import p11_refinement_scar_poc as scar  # noqa: E402

FIG = ROOT / "figures"
T2_VALUES = [10, 30, 50, 100, 150, 300, 600]
TAU_EMA_VALUES = [5.0, 10.0, 20.0, 50.0]
DT = 0.05  # meme pas de temps que le coeur M4R (Mem4Network.dt)
SEEDS = list(range(30))


def run_ema(seed, b_true, t2, tau_ema):
    """Moyenne mobile exponentielle du signal brut recu par le groupe --
    aucun reseau, aucun doute. alpha derive de tau_ema comme dans les EDS
    du coeur (alpha = dt/tau)."""
    alpha = DT / tau_ema
    ema = 0.0
    stim_wrong = -b_true * scar.B_E_WEAK
    stim_correct = b_true * scar.B_E_STRONG
    for _ in range(scar.T1):
        ema += alpha * (stim_wrong - ema)
    for _ in range(t2):
        ema += alpha * (stim_correct - ema)
    return 1 if ema >= 0 else -1


def sweep_ema():
    print("=== ACCUMULATEUR NAIF (EMA du signal brut) : meme sequence, aucun reseau ===\n")
    rows = []
    for tau_ema in TAU_EMA_VALUES:
        for t2 in T2_VALUES:
            correct = []
            for seed in SEEDS:
                b_true = 1 if (seed % 2 == 0) else -1
                g = run_ema(seed, b_true, t2, tau_ema)
                correct.append(int(g == b_true))
            acc = float(np.mean(correct))
            rows.append((tau_ema, t2, acc))
        accs = [r[2] for r in rows if r[0] == tau_ema]
        t2_full = next((t2 for t2, a in zip(T2_VALUES, accs) if a >= 0.99), None)
        suffix = f"[correction complete a T2={t2_full}]" if t2_full is not None else "[jamais 100%]"
        line = f"  tau_ema={tau_ema:<5} accuracy par T2 : "
        line += " ".join(f"T2={t2}:{a:.2f}" for t2, a in zip(T2_VALUES, accs))
        line += "   " + suffix
        print(line)
    return rows


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = sweep_ema()

    print("\n=== COMPARAISON AVEC M4R (deja mesure, p11_refinement_scar_poc.py) ===")
    print("  M4R (FHN+lattice+doute) : accuracy T2=50->0.067, T2=150->0.667, T2=300->1.000")
    print("  -> correction complete quelque part entre T2=150 et T2=300.\n")
    for tau_ema in TAU_EMA_VALUES:
        sub = [(t2, a) for tau, t2, a in rows if tau == tau_ema]
        full_at = next((t2 for t2, a in sub if a >= 0.99), None)
        acc_at_50 = next(a for t2, a in sub if t2 == 50)
        acc_at_150 = next(a for t2, a in sub if t2 == 150)
        acc_at_300 = next(a for t2, a in sub if t2 == 300)
        print(f"  EMA tau={tau_ema:<5}: T2=50->{acc_at_50:.2f}  T2=150->{acc_at_150:.2f}  "
              f"T2=300->{acc_at_300:.2f}  [complet a T2={full_at}]")

    print("\n=== VERDICT ===")
    full_by_tau = []
    for tau_ema in TAU_EMA_VALUES:
        sub = [(t2, a) for tau, t2, a in rows if tau == tau_ema]
        full_at = next((t2 for t2, a in sub if a >= 0.99), 9999)
        full_by_tau.append(full_at)
    best_ema_full = min(full_by_tau)
    print(f"  Meilleur EMA (parmi les tau testes) corrige completement a T2>={best_ema_full}.")
    print("  M4R corrige completement entre T2=150 et T2=300.")
    if best_ema_full < 150:
        print("  -> L'accumulateur NAIF corrige PLUS VITE que M4R sur cette tache precise -- "
              "la dynamique FHN+doute de M4R COUTE PLUS CHER en temps de correction qu'une "
              "simple moyenne mobile, pour ce readout de direction seul.")
    elif best_ema_full > 300:
        print("  -> M4R corrige PLUS VITE que le meilleur accumulateur naif teste -- "
              "la dynamique du reseau apporte une valeur reelle au-dela d'une simple moyenne.")
    else:
        print("  -> Les deux sont dans le MEME ORDRE DE GRANDEUR -- M4R ne coute ni plus ni "
              "moins qu'une solution triviale pour ce readout de direction seul. Sa valeur "
              "ajoutee, si elle existe, serait ailleurs (le warm start du solveur, la "
              "robustesse au crosstalk, etc.), pas dans la vitesse de correction elle-meme.")

    with (FIG / "p11_refinement_vs_traditional_poc.csv").open("w", encoding="utf-8") as f:
        f.write("tau_ema,t2,accuracy\n")
        for r in rows:
            f.write(",".join(f"{x:.6f}" if isinstance(x, float) else str(x) for x in r) + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.5, 5))
        for tau_ema in TAU_EMA_VALUES:
            sub = [(t2, a) for tau, t2, a in rows if tau == tau_ema]
            ax.plot([t2 for t2, a in sub], [a for t2, a in sub], "o-", label=f"EMA tau={tau_ema}")
        m4r_t2 = [50, 150, 300, 600]
        m4r_acc = [0.067, 0.667, 1.000, 1.000]
        ax.plot(m4r_t2, m4r_acc, "s--", color="k", linewidth=2, label="M4R (FHN+lattice+doute)")
        ax.set_xlabel("T2 (pas de correction)"); ax.set_ylabel("accuracy finale")
        ax.set_ylim(0, 1.05)
        ax.set_title("Cout de la cicatrice : M4R vs accumulateur naif")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIG / "p11_refinement_vs_traditional_poc.png", dpi=140)
        print(f"\n[png] {FIG / 'p11_refinement_vs_traditional_poc.png'}")
    except Exception as e:
        print(f"[png] skipped: {e}")

    print(f"\nWall time: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
