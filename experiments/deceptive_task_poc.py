#!/usr/bin/env python3
"""
POC B1d -- TACHE TROMPEUSE : le seul test loyal du doute (suite de B1c).

B1c a montre que sur une tache ou "se stabiliser = avoir juste", un critere de CONVERGENCE
trivial bat le doute (aussi precis, moins cher). Le doute ne peut ajouter de la valeur que si
CONVERGER TOT MENE A LA MAUVAISE REPONSE.

Construction du leurre (compromis vitesse/justesse) :
  - LEURRE  : peu de capteurs (n_d), signe -D*, E FORT  -> transitoire rapide vers le FAUX.
  - VERITE  : beaucoup de capteurs (n_t), signe +D*, E FAIBLE, mais total (n_t*E_t) > total
              leurre (n_d*E_d) -> regime PERMANENT vers le JUSTE.
  => la decision globale bascule FAUX -> JUSTE au cours du temps (tardivement).

Hypothese : le DOUTE = desaccord LOCAL |L v|. La verite faible mais nombreuse entretient une
tension locale meme quand la decision GLOBALE semble tranchee sur le leurre. Donc :
  - CONVERGENCE (lit le global) s'arrete tot, satisfaite et FAUSSE.
  - DOUTE (voit la tension locale) reste engage jusqu'au basculement -> JUSTE.

On mesure, sur des problemes trompeurs (6 seeds x plusieurs forces de leurre) :
  - taux de bonne reponse a l'arret, DOUTE vs CONVERGENCE (+ UNIFORME budget large en reference).
  - instant d'arret vs instant de BASCULEMENT (flip faux->juste).

Readout differentiel (run de reference stim=0 au meme seed -> annule v*<0 et le bruit), comme B1c.
Sortie : figures/deceptive_task_poc.csv + .png + verdict.
Cree : 2026-07-07 (Claude Opus 4.8) -- piste 2, tache trompeuse.
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
from mem4ristor.topology import Mem4Network  # noqa: E402
from mem4ristor.graph_utils import make_lattice_adj  # noqa: E402

CSV = ROOT / "figures" / "deceptive_task_poc.csv"
PNG = ROOT / "figures" / "deceptive_task_poc.png"

SIDE, N = 10, 100
MAX_BUDGET = 3000
WARMUP = 30
DOUBT_DROP = 0.30           # DOUTE : arret quand sigma < 30% du pic initial
CONV_W = 50
CONV_THR = 0.02             # CONVERGENCE : la variable de decision a cesse de bouger
SEEDS = list(range(12))

# Piege PULSE : leurre NOMBREUX + fort, mais retire apres T_pulse (domine la moyenne TOT).
# Verite PERSISTANTE, moins nombreuse (seule active apres le pulse -> gagne TARD).
N_DISTRACT = 26
N_TRUE = 14
E_TRUE = 0.6
E_DISTRACT = 1.0
T_PULSE_LEVELS = [150, 350, 700, 1200]     # duree du leurre : plus long -> convergence piegee

def make_deceptive(rng):
    adj = make_lattice_adj(SIDE, periodic=True)
    dstar = rng.choice([-1, 1])
    nodes = rng.choice(N, size=N_DISTRACT + N_TRUE, replace=False)
    d_nodes, t_nodes = nodes[:N_DISTRACT], nodes[N_DISTRACT:]
    stim_on = np.zeros(N)                                # phase leurre : leurre + verite
    stim_on[d_nodes] = -dstar * E_DISTRACT              # leurre nombreux, signe oppose
    stim_on[t_nodes] = +dstar * E_TRUE                  # verite persistante, signe correct
    stim_off = np.zeros(N)                               # apres le pulse : verite seule
    stim_off[t_nodes] = +dstar * E_TRUE
    return adj, stim_on, stim_off, dstar

def simulate(adj, stim_on, stim_off, seed, t_pulse):
    net = Mem4Network(size=SIDE, heretic_ratio=0.0, seed=seed, adjacency_matrix=adj)
    ref = Mem4Network(size=SIDE, heretic_ratio=0.0, seed=seed, adjacency_matrix=adj)
    L = net.L; zero = np.zeros(N)
    sig = np.empty(MAX_BUDGET); d_var = np.empty(MAX_BUDGET); dec = np.empty(MAX_BUDGET, dtype=int)
    for t in range(MAX_BUDGET):
        stim = stim_on if t < t_pulse else stim_off
        net.step(I_stimulus=stim); ref.step(I_stimulus=zero)
        v = net.model.v
        sig[t] = float(np.mean(np.abs(L @ v)))
        d = float(np.mean(v) - np.mean(ref.model.v))
        d_var[t] = d; dec[t] = 1 if d >= 0 else -1
    return sig, dec, d_var

def stop_doubt(sig):
    peak = float(np.max(sig[:WARMUP + 20])); thr = DOUBT_DROP * peak
    for t in range(WARMUP, len(sig)):
        if sig[t] < thr:
            return t + 1
    return len(sig)

def stop_conv(d_var):
    for t in range(WARMUP + CONV_W, len(d_var)):
        if abs(d_var[t] - d_var[t - CONV_W]) < CONV_THR:
            return t + 1
    return len(d_var)

def flip_time(dec, dstar):
    """Premier t apres lequel la decision reste == dstar jusqu'a la fin (basculement definitif)."""
    correct = (dec == dstar)
    for t in range(len(dec)):
        if np.all(correct[t:]):
            return t + 1
    return MAX_BUDGET + 1     # ne bascule jamais

def dec_at(dec, t):
    return int(dec[min(int(t), MAX_BUDGET) - 1])

def main():
    CSV.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = []
    print(f"{'T_pulse':>8}{'flip moy':>10}{'%bascule':>10}"
          f"{'c_doubt':>9}{'c_conv':>9}{'acc_DOUTE':>11}{'acc_CONV':>10}{'acc_FIN':>9}")
    print("-" * 75)
    summ = {}
    for t_pulse in T_PULSE_LEVELS:
        flips, cds, ccs = [], [], []
        acc_d, acc_c, acc_fin = [], [], []
        for seed in SEEDS:
            rng = np.random.RandomState(3000 + seed)
            adj, stim_on, stim_off, dstar = make_deceptive(rng)
            sig, dec, d_var = simulate(adj, stim_on, stim_off, seed * 10 + 1, t_pulse)
            ft = flip_time(dec, dstar)
            cd = stop_doubt(sig); cc = stop_conv(d_var)
            flips.append(ft); cds.append(cd); ccs.append(cc)
            a_d = int(dec_at(dec, cd) == dstar)
            a_c = int(dec_at(dec, cc) == dstar)
            a_f = int(dec[-1] == dstar)          # reponse en budget illimite (reference)
            acc_d.append(a_d); acc_c.append(a_c); acc_fin.append(a_f)
            rows.append((t_pulse, seed, dstar, ft, cd, cc, a_d, a_c, a_f))
        pct_flip = 100.0 * np.mean([f <= MAX_BUDGET for f in flips])
        md, mc = np.mean(cds), np.mean(ccs)
        ad, ac, af = np.mean(acc_d), np.mean(acc_c), np.mean(acc_fin)
        summ[t_pulse] = (np.mean(flips), pct_flip, md, mc, ad, ac, af)
        print(f"{t_pulse:>8}{np.mean(flips):>10.0f}{pct_flip:>9.0f}%"
              f"{md:>9.0f}{mc:>9.0f}{ad:>11.2f}{ac:>10.2f}{af:>9.2f}")

    print("\n=== VERDICT B1d (honnete) ===")
    print("On ne juge que le REGIME TROMPEUR : problemes qui basculent faux->juste (%bascule eleve),")
    print("ou la reponse finale (acc_FIN) est juste mais la decision precoce est fausse.")
    best = None
    for t_pulse, (mf, pf, md, mc, ad, ac, af) in summ.items():
        # regime trompeur exploitable : la verite finit par gagner (af haut) ET le basculement
        # est tardif (le leurre trompe d'abord). L'ecart acc_DOUTE - acc_CONV y est la mesure clef.
        if af >= 0.6 and mf > 100:
            gain = ad - ac
            if best is None or gain > best[1]:
                best = (t_pulse, gain)
            print(f"  T_pulse={t_pulse}: acc_DOUTE={ad:.2f} vs acc_CONV={ac:.2f} "
                  f"(gain doute {gain:+.2f}) ; flip~{mf:.0f}, c_conv~{mc:.0f}, c_doubt~{md:.0f}")
    if best is None:
        print("  -> aucun regime trompeur exploitable dans la grille (le leurre ne trompe pas, ou")
        print("     la verite ne gagne jamais). Elargir T_pulse / E_TRUE / N_TRUE.")
    else:
        t_pulse, gain = best
        if gain > 0.15:
            print(f"\n  -> LE DOUTE GAGNE : a T_pulse={t_pulse}, il tranche juste la ou la convergence")
            print(f"     s'arrete faux (gain {gain:+.2f}). 'Explorer tant que le doute persiste' a")
            print("     enfin un sens : la tension locale survit au faux consensus global.")
        elif gain > 0.03:
            print(f"\n  -> LE DOUTE AIDE un peu (gain {gain:+.2f} a E_leurre={e_d}), mais modeste.")
        else:
            print(f"\n  -> MEME EN TROMPEUR, le doute ne bat pas la convergence (gain {gain:+.2f}) :")
            print("     soit le doute se fait aussi pieger, soit le flip precede les deux arrets.")

    with CSV.open("w", encoding="utf-8") as f:
        f.write("t_pulse,seed,dstar,flip_time,c_doubt,c_conv,acc_doubt,acc_conv,acc_final\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print(f"\n[csv] {CSV}")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
        levels = T_PULSE_LEVELS
        axes[0].plot(levels, [summ[e][4] for e in levels], "o-", c="#d62728", label="DOUTE")
        axes[0].plot(levels, [summ[e][5] for e in levels], "s-", c="#1f77b4", label="CONVERGENCE")
        axes[0].plot(levels, [summ[e][6] for e in levels], "^--", c="#2ca02c", label="budget illimite")
        axes[0].set_xlabel("Duree du leurre T_pulse"); axes[0].set_ylabel("Taux de bonne reponse")
        axes[0].set_title("Justesse a l'arret vs duree du leurre"); axes[0].grid(alpha=0.3)
        axes[0].legend(); axes[0].set_ylim(-0.05, 1.05)
        axes[1].plot(levels, [summ[e][0] for e in levels], "o-", c="k", label="flip faux->juste")
        axes[1].plot(levels, [summ[e][2] for e in levels], "o-", c="#d62728", label="arret DOUTE")
        axes[1].plot(levels, [summ[e][3] for e in levels], "s-", c="#1f77b4", label="arret CONVERGENCE")
        axes[1].set_xlabel("Duree du leurre T_pulse"); axes[1].set_ylabel("Pas")
        axes[1].set_title("Instant d'arret vs basculement"); axes[1].grid(alpha=0.3); axes[1].legend()
        fig.suptitle("B1d tache trompeuse : le doute tient-il quand converger tot = se tromper ?", fontsize=11)
        plt.tight_layout(); plt.savefig(PNG, dpi=140)
        print(f"[png] {PNG}")
    except Exception as e:
        print(f"[png] skipped: {e}")
    print(f"\nWall time: {time.time()-t0:.1f}s")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
