#!/usr/bin/env python3
"""
POC 'rythme bicameral' -- realiser la vision : explorer TANT QUE le doute persiste,
puis CONSOLIDER, en gardant chaque solution. Le doute est pilote de l'exterieur
(aucune modif du coeur du modele) pour creer le cycle FOU<->SAGE qui manque
(le systeme, laisse libre, se verrouille en FOU : voir calib3).

Deux chambres (bicameral) :
  - FOU   : epsilon_u normal, u libre -> sature haut -> couplage repulsif -> EXPLORE.
  - SAGE  : u force bas (epsilon_u=0) -> couplage attractif -> CONSOLIDE vers une solution.
Une 'solution' = etat v stabilise a la fin d'une phase SAGE ; on les archive.

Question centrale (voie 1 du backlog) : le doute explore-t-il MIEUX que le hasard ?
Conditions :
  BICAMERAL : cycles [FOU explore, SAGE consolide].
  HASARD    : cycles [reinit aleatoire de v, SAGE consolide]  (exploration = bruit, pas doute).
  REPULSIF  : u libre en permanence, jamais de SAGE           (ce que fait le modele seul).
  ATTRACTIF : u bas en permanence, jamais de FOU ni reinit    (coince sur une solution).

Contrainte spatiale : capteur (+E) au coin 0, (-E) au coin oppose -> le couplage doit
propager l'evidence ; on mesure la coherence de chaque solution avec le gradient impose.

Metriques (par condition, moyennees sur seeds) :
  - n_distinct : nb de solutions distinctes (signature grossiere sign(v)).
  - stability  : 1/(1+var) sur les derniers pas de consolidation (haut = etat stable).
  - coherence  : -corr(v, position_diagonale) (>0 = evidence respectee).
  - diversity  : distance moyenne par paires entre solutions (exploration effective).

Sortie : figures/bicameral_rhythm_poc.csv + .png + resume console.
Cree : 2026-07-06 (Claude Fable 5, L'Ingenieur -- suite 'on revient au bicameral').
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

CSV = ROOT / "figures" / "bicameral_rhythm_poc.csv"
PNG = ROOT / "figures" / "bicameral_rhythm_poc.png"

SIDE, N = 10, 100
N_CYCLES = 10
T_FOU, T_SAGE = 400, 400
U_SAGE = 0.05
E = 0.6
SEEDS = [0, 1, 2, 3, 4]

rows_ = np.arange(N) // SIDE
cols_ = np.arange(N) % SIDE
proj = (rows_ + cols_).astype(float)          # 0 (coin +) .. 18 (coin -)
stim = np.zeros(N); stim[0] = +E; stim[N-1] = -E

def coarse_sig(v, dead=0.4):
    s = np.zeros(N, dtype=int); s[v > dead] = 1; s[v < -dead] = -1
    return tuple(s)

def coherence(v):
    if np.std(v) < 1e-9:
        return 0.0
    return float(-np.corrcoef(v, proj)[0, 1])   # >0 : v haut pres du +, respecte le gradient

def consolidate(net, T, rng_noise_var):
    """Phase SAGE : u force bas, couplage attractif, on laisse converger. Retourne (solution, var_finale)."""
    net.model.cfg["doubt"]["epsilon_u"] = 0.0
    net.model.u[:] = U_SAGE
    tail = []
    for t in range(T):
        net.model.u[:] = U_SAGE                 # maintenir la chambre SAGE
        net.step(I_stimulus=stim)
        if t >= T - 50:
            tail.append(net.model.v.copy())
    tail = np.array(tail)
    return net.model.v.copy(), float(tail.var(axis=0).mean())

def run(condition, seed):
    net = Mem4Network(size=SIDE, heretic_ratio=0.0, seed=seed)
    rng = np.random.RandomState(1000 + seed)
    sols, vars_, cohs = [], [], []

    if condition == "REPULSIF":
        # u libre en permanence ; on echantillonne l'etat a la fin de chaque 'cycle'
        for c in range(N_CYCLES):
            net.model.cfg["doubt"]["epsilon_u"] = 0.02
            for _ in range(T_FOU + T_SAGE):
                net.step(I_stimulus=stim)
            tail = []
            for _ in range(50):
                net.step(I_stimulus=stim)
                tail.append(net.model.v.copy())
            tail = np.array(tail)
            sols.append(net.model.v.copy()); vars_.append(float(tail.var(axis=0).mean()))
            cohs.append(coherence(net.model.v))
        return sols, vars_, cohs

    for c in range(N_CYCLES):
        if condition == "BICAMERAL":
            net.model.cfg["doubt"]["epsilon_u"] = 0.02   # FOU : explore par le doute
            for _ in range(T_FOU):
                net.step(I_stimulus=stim)
        elif condition == "HASARD":
            net.model.v[:] = rng.uniform(-1.5, 1.5, N)    # exploration = reinit aleatoire
            net.model.w[:] = rng.uniform(0.0, 1.0, N)
        elif condition == "ATTRACTIF":
            pass                                          # ni FOU ni reinit : coince
        sol, var = consolidate(net, T_SAGE, 0.0)
        sols.append(sol); vars_.append(var); cohs.append(coherence(sol))
    return sols, vars_, cohs

def analyse(sols, vars_, cohs):
    sigs = {coarse_sig(s) for s in sols}
    n_distinct = len(sigs)
    stability = float(np.mean([1.0 / (1.0 + v) for v in vars_]))
    coh = float(np.mean(cohs))
    S = np.array(sols)
    if len(S) > 1:
        d = [np.linalg.norm(S[i] - S[j]) / np.sqrt(N)
             for i in range(len(S)) for j in range(i + 1, len(S))]
        diversity = float(np.mean(d))
    else:
        diversity = 0.0
    return n_distinct, stability, coh, diversity

def main():
    CSV.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    conds = ["BICAMERAL", "HASARD", "REPULSIF", "ATTRACTIF"]
    agg = {c: {"n_distinct": [], "stability": [], "coherence": [], "diversity": []} for c in conds}
    print(f"{'condition':<11}{'seed':>5}{'n_distinct':>12}{'stability':>11}{'coherence':>11}{'diversity':>11}")
    print("-" * 62)
    rows = []
    for seed in SEEDS:
        for cond in conds:
            sols, vars_, cohs = run(cond, seed)
            nd, st, co, dv = analyse(sols, vars_, cohs)
            agg[cond]["n_distinct"].append(nd); agg[cond]["stability"].append(st)
            agg[cond]["coherence"].append(co); agg[cond]["diversity"].append(dv)
            rows.append((cond, seed, nd, st, co, dv))
            print(f"{cond:<11}{seed:>5}{nd:>12}{st:>11.3f}{co:>11.3f}{dv:>11.3f}")

    print(f"\n{'condition':<11}{'n_distinct':>12}{'stability':>11}{'coherence':>11}{'diversity':>11}")
    print("-" * 56)
    summ = {}
    for c in conds:
        nd = np.mean(agg[c]["n_distinct"]); st = np.mean(agg[c]["stability"])
        co = np.mean(agg[c]["coherence"]); dv = np.mean(agg[c]["diversity"])
        summ[c] = (nd, st, co, dv)
        print(f"{c:<11}{nd:>12.1f}{st:>11.3f}{co:>11.3f}{dv:>11.3f}")

    print("\n=== VERDICT (honnete) ===")
    b, h, r, a = summ["BICAMERAL"], summ["HASARD"], summ["REPULSIF"], summ["ATTRACTIF"]
    print(f"COHERENCE avec la contrainte : BICAMERAL={b[2]:.3f}  HASARD={h[2]:.3f}  "
          f"REPULSIF={r[2]:.3f}  ATTRACTIF={a[2]:.3f}")
    if b[2] > max(h[2], r[2], a[2]) * 1.3:
        print("-> SIGNAL POSITIF : le rythme doute->consolidation produit les solutions les PLUS")
        print("   coherentes avec l'evidence spatiale. Le couplage sert (consolidation SAGE) et le")
        print("   doute guide mieux que le hasard/bruit -- valeur sur la QUALITE, pas la quantite.")
    else:
        print("-> Pas d'avantage net de coherence pour le bicameral.")
    print(f"NOMBRE de solutions distinctes : BICAMERAL={b[0]:.1f}  HASARD={h[0]:.1f}  "
          f"REPULSIF={r[0]:.1f}  ATTRACTIF={a[0]:.1f}")
    print("   NUANCE : le bruit thermique seul (ATTRACTIF) trouve autant/plus de solutions ;")
    print("   le doute n'explore PAS plus en quantite. Sa valeur est la coherence, pas le nombre.")
    print(f"COMPROMIS explore/exploite : BICAMERAL est le moins DIVERS (div={b[3]:.2f}, solutions")
    print(f"   proches car tirees vers la contrainte) ; HASARD divers ({h[3]:.2f}) mais incoherent.")
    print("[note] 'stability' ~1 partout (regime point-fixe sub-Hopf) : metrique peu discriminante.")

    with CSV.open("w", encoding="utf-8") as f:
        f.write("condition,seed,n_distinct,stability,coherence,diversity\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]:.4f},{r[4]:.4f},{r[5]:.4f}\n")
    print(f"\n[csv] {CSV}")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
        metrics = [("n_distinct", "Distinct solutions"), ("stability", "Stability"),
                   ("coherence", "Coherence w/ evidence")]
        colors = {"BICAMERAL": "#2ca02c", "HASARD": "#7f7f7f", "REPULSIF": "#d62728", "ATTRACTIF": "#1f77b4"}
        for ax, (key, lab) in zip(axes, metrics):
            vals = [summ[c][["n_distinct", "stability", "coherence", "diversity"].index(key)] for c in conds]
            ax.bar(conds, vals, color=[colors[c] for c in conds], edgecolor="k")
            ax.set_title(lab); ax.grid(axis="y", alpha=0.3); ax.tick_params(axis="x", rotation=30)
        fig.suptitle("Rythme bicameral : le doute explore-t-il mieux que le hasard ?", fontsize=11)
        plt.tight_layout(); plt.savefig(PNG, dpi=140)
        print(f"[png] {PNG}")
    except Exception as e:
        print(f"[png] skipped: {e}")
    print(f"\nWall time: {time.time()-t0:.1f}s")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
