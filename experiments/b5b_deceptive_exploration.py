#!/usr/bin/env python3
"""
B5b -- LA NICHE : le doute vs un Echo State Network sur une tache TROMPEUSE en ligne.

Motivation (Julien, 2026-07-08). Mem4ristor n'est pas une memoire (B5 : l'ESN le bat ~5.5x sur
NARMA10). C'est un EXPLORATEUR : le doute sert a NE PAS se fier a la premiere apparence ni a
accorder sa confiance trop tot. Ce cadrage se teste dans le regime EN LIGNE / DECISION (quand
s'engager ?), PAS en prediction supervisee (ou l'entrainement du readout annule la tromperie).

Tache (fixee AVANT de regarder les resultats -- garde-fou d'honnetete du 2026-07-07) :
  flux TROMPEUR de deceptive_task_poc.py -- un leurre NOMBREUX+PULSE domine la moyenne globale
  TOT (pousse a la fausse decision), une verite PERSISTANTE gagne TARD. Converger tot = se tromper.
  Lecture differentielle (run de reference stim=0, meme seed) -> decision = signe(mean_v - mean_ref).

Comparaison LOYALE, deux reservoirs sur le MEME flux, memes seeds, MEMES regles d'arret :
  - Mem4ristor (FHN + doute) : decision = signe du consensus differentiel ; signal d'arret NATIF
    = le doute sigma_social=|L v| qui retombe. Regles : DOUTE (natif) et CONVERGENCE (generique).
  - Echo State Network (Jaeger 2001) : meme flux en entree par noeud, lecture differentielle
    identique ; PAS de doute natif -> on lui donne son MEILLEUR signal d'arret generique :
    CONVERGENCE (decision stabilisee) ET DROP (son propre changement d'etat |x_t - x_{t-1}| qui
    retombe -- l'analogue le plus proche du doute, pour ne pas le desavantager). Hyperparametres
    (rho, fuite) choisis par la MEILLEURE precision ORACLE (budget illimite) -> l'ESN a son max.

Question : le doute NATIF de Mem4ristor resiste-t-il au leurre mieux qu'un ESN de reference muni
de son meilleur arret generique ? Reponse rapportee telle quelle (positive ou negative).

Metrique : precision de decision A L'ARRET (== dstar), sur le regime trompeur. Oracle (budget
illimite) rapporte aussi pour separer "peut-il finir juste" de "s'engage-t-il trop tot".

Sortie : figures/b5b_deceptive_exploration.csv + .png + verdict.
Cree : 2026-07-08 (Claude Opus 4.8) -- B5b niche exploration (docs/FUTURE_WORK.md).
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
import deceptive_task_poc as dp  # noqa: E402  (make_deceptive, simulate, stop_doubt/conv, dec_at, consts)

dp.MAX_BUDGET = 2000               # t_pulse <= 700, marge suffisante ; accelere
CSV = ROOT / "figures" / "b5b_deceptive_exploration.csv"
PNG = ROOT / "figures" / "b5b_deceptive_exploration.png"

N = dp.N
SEEDS = list(range(15))
T_PULSE_LEVELS = [350, 700]        # regime trompeur (cf. B1d : 150 trop court)
ESN_RHO = [0.9, 1.0, 1.1]
ESN_LEAK = [0.3, 1.0]
ESN_DENSITY = 0.1
N_BOOT = 10000
RNG_BOOT = np.random.RandomState(20260708)

# ---- ESN standard -----------------------------------------------------------
def make_esn(seed, rho):
    rng = np.random.RandomState(7000 + seed)
    W = rng.uniform(-1.0, 1.0, (N, N)) * (rng.uniform(0, 1, (N, N)) < ESN_DENSITY)
    eig = np.max(np.abs(np.linalg.eigvals(W)))
    if eig > 1e-9:
        W *= rho / eig
    return W

def run_esn_decision(W, stim_on, stim_off, t_pulse, leak):
    """ESN pilote par le MEME stimulus par noeud (leurre puis verite), + run de reference stim=0.
    Retourne : d_var (decision differentielle continue), dec (signe), change (|x_t-x_{t-1}| moyen,
    l'analogue du doute pour l'arret DROP)."""
    x = np.zeros(N); xr = np.zeros(N)
    T = dp.MAX_BUDGET
    d_var = np.empty(T); dec = np.empty(T, dtype=int); change = np.empty(T)
    x_prev = x.copy()
    for t in range(T):
        stim = stim_on if t < t_pulse else stim_off
        x = (1.0 - leak) * x + leak * np.tanh(W @ x + stim)
        xr = (1.0 - leak) * xr + leak * np.tanh(W @ xr)          # reference stim=0
        d = float(x.mean() - xr.mean())
        d_var[t] = d; dec[t] = 1 if d >= 0 else -1
        change[t] = float(np.mean(np.abs(x - x_prev)))
        x_prev = x.copy()
    return d_var, dec, change

def esn_best_by_oracle(stim_on, stim_off, t_pulse, seed, dstar):
    """Choisit (rho, leak) qui MAXIMISE la precision oracle (dec[-1]) -> l'ESN a son meilleur reglage.
    Retourne ses accs (DROP, convergence, oracle) + la trajectoire dec du meilleur combo (pour
    evaluer une politique a BUDGET FIXE : baseline non-adaptative la plus forte)."""
    best = None
    for rho in ESN_RHO:
        W = make_esn(seed, rho)
        for leak in ESN_LEAK:
            d_var, dec, change = run_esn_decision(W, stim_on, stim_off, t_pulse, leak)
            oracle = int(dec[-1] == dstar)
            c_conv = dp.stop_conv(d_var)
            c_drop = dp.stop_doubt(change)            # DROP : changement d'etat qui retombe
            a_conv = int(dp.dec_at(dec, c_conv) == dstar)
            a_drop = int(dp.dec_at(dec, c_drop) == dstar)
            score = (oracle, a_drop + a_conv)         # priorite a l'oracle, puis aux arrets
            if best is None or score > best[0]:
                best = (score, dict(oracle=oracle, a_conv=a_conv, a_drop=a_drop,
                                    rho=rho, leak=leak, dec=dec))
    return best[1]

def boot_ci_paired(a, b):
    d = np.asarray(a, float) - np.asarray(b, float); n = len(d)
    m = np.empty(N_BOOT)
    for k in range(N_BOOT):
        m[k] = d[RNG_BOOT.randint(0, n, n)].mean()
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

def main():
    CSV.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = []
    # accumulateurs (moyennes par seed sur les niveaux trompeurs)
    acc = {k: [] for k in ["M4R_DOUBT", "M4R_CONV", "M4R_ORACLE",
                           "ESN_DROP", "ESN_CONV", "ESN_ORACLE"]}
    # baseline BUDGET FIXE : acc[modele][B] = liste (une par (seed,tpulse)) de 0/1 a l'arret force a B.
    BUDGET_GRID = list(range(100, dp.MAX_BUDGET, 100))
    fixed = {"ESN": {B: [] for B in BUDGET_GRID}, "M4R": {B: [] for B in BUDGET_GRID}}
    print(f"{'seed':>5}  {'M4R_doubt':>10}{'M4R_conv':>9}{'M4R_or':>8}   "
          f"{'ESN_drop':>9}{'ESN_conv':>9}{'ESN_or':>8}")
    print("-" * 70)
    for seed in SEEDS:
        per = {k: [] for k in acc}
        for t_pulse in T_PULSE_LEVELS:
            rng = np.random.RandomState(3000 + seed)
            adj, stim_on, stim_off, dstar = dp.make_deceptive(rng)   # lattice, flux trompeur
            # --- Mem4ristor (doute natif) ---
            sig, dec, d_var = dp.simulate(adj, stim_on, stim_off, seed * 10 + 1, t_pulse)
            cd = dp.stop_doubt(sig); cc = dp.stop_conv(d_var)
            per["M4R_DOUBT"].append(int(dp.dec_at(dec, cd) == dstar))
            per["M4R_CONV"].append(int(dp.dec_at(dec, cc) == dstar))
            per["M4R_ORACLE"].append(int(dec[-1] == dstar))
            # --- ESN (meilleur reglage oracle) ---
            e = esn_best_by_oracle(stim_on, stim_off, t_pulse, seed, dstar)
            per["ESN_DROP"].append(e["a_drop"])
            per["ESN_CONV"].append(e["a_conv"])
            per["ESN_ORACLE"].append(e["oracle"])
            # --- baseline BUDGET FIXE (politique non-adaptative) : dec forcee a chaque B ---
            for B in BUDGET_GRID:
                fixed["ESN"][B].append(int(dp.dec_at(e["dec"], B) == dstar))
                fixed["M4R"][B].append(int(dp.dec_at(dec, B) == dstar))
            rows.append((seed, t_pulse, dstar, per["M4R_DOUBT"][-1], per["M4R_CONV"][-1],
                         per["M4R_ORACLE"][-1], per["ESN_DROP"][-1], per["ESN_CONV"][-1],
                         per["ESN_ORACLE"][-1], e["rho"], e["leak"]))
        for k in acc:
            acc[k].append(float(np.mean(per[k])))
        print(f"{seed:>5}  {np.mean(per['M4R_DOUBT']):>10.2f}{np.mean(per['M4R_CONV']):>9.2f}"
              f"{np.mean(per['M4R_ORACLE']):>8.2f}   {np.mean(per['ESN_DROP']):>9.2f}"
              f"{np.mean(per['ESN_CONV']):>9.2f}{np.mean(per['ESN_ORACLE']):>8.2f}")

    m = {k: float(np.mean(v)) for k, v in acc.items()}
    print("\n=== RESUME (precision de decision a l'arret, regime trompeur, 15 seeds) ===")
    print(f"  Mem4ristor  DOUTE(natif)={m['M4R_DOUBT']:.2f}  CONV={m['M4R_CONV']:.2f}  "
          f"ORACLE={m['M4R_ORACLE']:.2f}")
    print(f"  ESN         DROP(analogue)={m['ESN_DROP']:.2f}  CONV={m['ESN_CONV']:.2f}  "
          f"ORACLE={m['ESN_ORACLE']:.2f}")

    # --- baseline BUDGET FIXE : meilleur budget unique par modele (politique non-adaptative) ---
    def best_fixed(model):
        best_B, best_acc = None, -1.0
        for B in BUDGET_GRID:
            a = float(np.mean(fixed[model][B]))
            if a > best_acc:
                best_acc, best_B = a, B
        return best_B, best_acc
    esn_fx_B, esn_fx_acc = best_fixed("ESN")
    m4r_fx_B, m4r_fx_acc = best_fixed("M4R")

    print("\n=== VERDICT B5b (honnete) ===")
    # (1) doute vs arret NAIF (self-terminant) de l'ESN
    esn_best_stop = np.maximum(acc["ESN_DROP"], acc["ESN_CONV"])
    dm, dlo, dhi = boot_ci_paired(acc["M4R_DOUBT"], esn_best_stop)
    print(f"(1) doute natif vs meilleur arret NAIF ESN : {dm:+.2f} CI[{dlo:+.2f},{dhi:+.2f}]")
    print(f"    -> {'le doute bat' if dlo>0 else 'pas d ecart net vs'} un ESN a arret self-terminant "
          f"(l ESN se stabilise instantanement sur le leurre : arret au plancher).")
    # (2) LE TEST QUI COMPTE : doute (adaptatif) vs MEILLEUR BUDGET FIXE de l'ESN (non-adaptatif fort)
    esn_fixed_per_seed = []
    # reconstruire l'acc du meilleur B par seed (moyenne sur t_pulse) pour l'IC apparie
    per_seed_map = {}
    # fixed[...][B] est indexe par (seed,tpulse) dans l'ordre d'insertion -> regrouper par seed
    idx = 0
    npairs = len(T_PULSE_LEVELS)
    for si in range(len(SEEDS)):
        vals = [fixed["ESN"][esn_fx_B][si * npairs + j] for j in range(npairs)]
        esn_fixed_per_seed.append(float(np.mean(vals)))
    d2m, d2lo, d2hi = boot_ci_paired(acc["M4R_DOUBT"], esn_fixed_per_seed)
    print(f"(2) doute natif (adaptatif) vs ESN MEILLEUR BUDGET FIXE (B={esn_fx_B}, acc={esn_fx_acc:.2f}) : "
          f"{d2m:+.2f} CI[{d2lo:+.2f},{d2hi:+.2f}]")
    if d2lo > 0:
        print("    -> LA NICHE TIENT : le doute bat meme le meilleur ESN NON-ADAPTATIF. La valeur")
        print("       est l ADAPTIVITE a un horizon de tromperie inconnu (leurre 350 ET 700).")
    elif d2hi < 0:
        print("    -> HONNETE : un ESN a budget fixe bien choisi (> duree du leurre) BAT le doute.")
        print("       La niche se reduit : le doute bat les arrets naifs, pas un horizon fixe optimal.")
        print("       Sa valeur ne ressort que si l horizon est inconnu/non-borne OU si attendre COUTE")
        print("       (cf. B1c : le doute paie quand le budget est rare). Resultat rapporte tel quel.")
    else:
        print("    -> PARITE avec le meilleur budget fixe (IC couvre 0) : pas d avantage adaptatif net")
        print("       sur cette grille bornee. La valeur du doute reste conditionnee au cout d attente.")
    print(f"  [oracle] M4R={m['M4R_ORACLE']:.2f}  ESN={m['ESN_ORACLE']:.2f} "
          f"| M4R meilleur budget fixe : B={m4r_fx_B}, acc={m4r_fx_acc:.2f}")

    with CSV.open("w", encoding="utf-8") as f:
        f.write("seed,t_pulse,dstar,m4r_doubt,m4r_conv,m4r_oracle,esn_drop,esn_conv,esn_oracle,esn_rho,esn_leak\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print(f"\n[csv] {CSV}")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10.5, 5))
        labels = ["M4R\nDOUTE\n(natif,\nadaptatif)", "M4R\nCONV", "ESN\nDROP", "ESN\nCONV",
                  f"ESN\nBUDGET FIXE\n(B={esn_fx_B})", "ESN\noracle"]
        means = [m["M4R_DOUBT"], m["M4R_CONV"], m["ESN_DROP"], m["ESN_CONV"],
                 esn_fx_acc, m["ESN_ORACLE"]]
        colors = ["#d62728", "#ff9896", "#1f77b4", "#aec7e8", "#2ca02c", "#c7c7c7"]
        ax.bar(labels, means, color=colors, edgecolor="k")
        ax.axhline(0.5, ls=":", c="gray", label="hasard (0.5)")
        ax.set_ylabel("Precision de decision a l'arret")
        ax.set_title("B5b : le doute vs l'ESN sur flux TROMPEUR (converger tot = se tromper)\n"
                     "le doute bat les arrets NAIFs de l'ESN, mais EGALE son meilleur budget fixe "
                     "(horizon borne)")
        ax.set_ylim(0, 1.05); ax.legend(); ax.grid(axis="y", alpha=0.3)
        plt.tight_layout(); plt.savefig(PNG, dpi=140)
        print(f"[png] {PNG}")
    except Exception as e:
        print(f"[png] skipped: {e}")
    print(f"\nWall time: {time.time()-t0:.1f}s")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
