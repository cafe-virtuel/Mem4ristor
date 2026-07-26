#!/usr/bin/env python3
"""
EXPERIENCE A (suite) -- le degre est-il la CAUSE, ou seulement un PROXY ?

`expA_ba_cascade_poc.py` a etabli, replique sur graines disjointes :
  - couplage `uniform`      : rho(degre, temps de verrouillage) ~ -0.55, les hubs
                              se verrouillent ~300-500 pas avant la peripherie,
                              et le sous-graphe vivant fragmente bien au-dela de
                              ce qu'un retrait ALEATOIRE du meme nombre de noeuds
                              produirait.
  - couplage `degree_linear`: l'effet DISPARAIT et s'inverse faiblement (+0.07).

Ce script teste l'interpretation que ce contraste suggere, AVANT de la raconter :
  le gouverneur ne serait pas le degre en tant que propriete topologique, mais
  l'INTENSITE DU CHAMP DE COUPLAGE effectivement recue par le noeud. Sous
  `uniform` cette intensite croit avec le degre (le noeud recoit la somme de ses
  voisins) ; sous `degree_linear` elle est egalisee par construction (division
  par k). Le degre ne serait donc predictif que lorsque la normalisation le
  laisse passer.

PREDICTION FALSIFIABLE, posee avant la mesure :
  si l'interpretation est juste, rho(drive, t_mort) doit etre nettement NEGATIF
  dans les DEUX normalisations, la ou rho(degre, t_mort) ne l'est que sous
  `uniform`. Si rho(drive, t_mort) s'effondre lui aussi sous `degree_linear`,
  l'interpretation est FAUSSE et le compte rendu doit le dire.

`drive_i` = moyenne de |l_v_i| sur une fenetre PRECOCE (avant les premieres
morts), ou l_v est exactement la grandeur que le coeur injecte dans I_coup et
dans la dynamique du doute (sigma_social) -- recalculee ici a l'identique depuis
L et les scale_factors, sans toucher au coeur.

RESULTAT (2026-07-26) : L'INTERPRETATION CI-DESSUS EST REFUTEE. Le signe de
rho(drive, t_mort) tient bien sur les six lignes, mais les tests symetriques la
demontent : le partiel deg|drive reste a -0.26/-0.38 sous `uniform` (le champ
recu n'absorbe pas le degre), et la traduction par-noeud du mecanisme du 01/07
-- la variance temporelle de la cible locale -- ne predit rien du tout
(rho(tvar, t_mort) ~ +0.1 ; rho(deg, tvar) ~ -0.1), parce que cet argument
suppose des voisins INDEPENDANTS alors que la dynamique les correle.
Le phenomene mesure par `expA_ba_cascade_poc.py` tient ; son mecanisme au
niveau du noeud reste une question OUVERTE. Voir le verdict imprime en fin de run.

Cree : 2026-07-26 (Claude Opus 5, L'Ingenieur).
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from mem4ristor.core import Mem4Network
from mem4ristor.graph_utils import make_ba, make_er

N = 200
STEPS = 3000
I_STIM = 0.0
HERETIC_RATIO = 0.15
SIGMA_V = 0.05
V_DEAD = -1.2
WIN = (200, 500)            # fenetre precoce : apres le transitoire, avant les morts
SEEDS = [42, 123, 777, 17, 256]

FIG_DIR = ROOT / "figures"
CSV_PATH = FIG_DIR / "expA_mechanism_drive_poc.csv"


def build_graph(topo, seed):
    if topo == "BA m=3":
        return make_ba(N, 3, seed)
    if topo == "BA m=5":
        return make_ba(N, 5, seed)
    if topo == "ER <k>~6":
        return make_er(N, 6.0 / (N - 1), seed)
    raise ValueError(topo)


def run(adj, norm, seed):
    net = Mem4Network(size=int(np.sqrt(adj.shape[0])), heretic_ratio=HERETIC_RATIO,
                      seed=seed, adjacency_matrix=adj.copy(), coupling_norm=norm,
                      cold_start=True)
    net.model.cfg["noise"]["sigma_v"] = SIGMA_V

    # scale_factors : reproduction exacte de topology.Mem4Network.step()
    if norm != "uniform":
        D = net.model.cfg["coupling"]["D"]
        scale = (net.node_weights * D) / (D / np.sqrt(net.N))
    else:
        scale = np.ones(N)

    # cible locale du champ moyen : moyenne des v des voisins (mecanisme du 01/07)
    deg_safe = np.maximum(adj.sum(1), 1.0)

    v_hist = np.empty((STEPS, N))
    drive_sum = np.zeros(N)
    targets = []
    n_win = 0
    for t in range(STEPS):
        v_before = net.v.copy()
        net.step(I_stimulus=I_STIM)
        v_hist[t] = net.v
        if WIN[0] <= t < WIN[1]:
            l_v = -(net.L @ v_before) * scale
            drive_sum += np.abs(l_v)
            targets.append((adj @ v_before) / deg_safe)
            n_win += 1
    # target_var_i : dispersion TEMPORELLE de la cible locale. Le mecanisme etabli
    # le 01/07 est un argument de VARIANCE (Var_i(cible) ~ V*<1/deg>), pas
    # d'amplitude : un hub moyenne beaucoup de voisins, sa cible est donc peu
    # bruitee -> attraction coherente -> consensus. Prediction : plus la cible
    # est bruitee, plus le noeud survit -> rho(target_var, t_mort) POSITIF.
    target_var = np.var(np.asarray(targets), axis=0)
    return v_hist, drive_sum / max(n_win, 1), target_var


def death_times(v_hist, v_dead=V_DEAD):
    above = v_hist > v_dead
    t_d = np.zeros(N, dtype=int)
    for i in range(N):
        idx = np.flatnonzero(above[:, i])
        t_d[i] = 0 if idx.size == 0 else int(idx[-1]) + 1
    return t_d


def partial_spearman(x, y, *controls):
    """Spearman partiel de x et y en controlant une ou plusieurs covariables
    (correlation des residus des rangs)."""
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    Z = np.c_[np.ones(len(rx))] if not controls else \
        np.c_[np.ones(len(rx)), *[stats.rankdata(c) for c in controls]]
    def resid(a):
        return a - Z @ np.linalg.lstsq(Z, a, rcond=None)[0]
    return float(stats.pearsonr(resid(rx), resid(ry))[0])


def boot_ci(v, n_boot=10_000, seed=0):
    v = np.asarray(v, float)
    rng = np.random.RandomState(seed)
    m = v[rng.randint(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = []
    print("=" * 104)
    print("EXPERIENCE A (suite) -- degre = cause ou proxy du champ recu ?")
    print(f"N={N} steps={STEPS} endogene cold_start | drive = <|l_v|> sur t in {WIN}")
    print("PREDICTION posee avant mesure : rho(drive, t_mort) < 0 dans LES DEUX "
          "normalisations, sinon l'interpretation tombe.")
    print("=" * 104)
    print(f"{'condition':<26}{'rho(deg,t)':>12}{'rho(drive,t)':>14}{'rho(tvar,t)':>13}"
          f"{'part deg|drive':>16}{'part deg|drv,tvar':>19}{'rho(deg,tvar)':>15}")
    print("-" * 115)

    for topo in ["BA m=3", "BA m=5", "ER <k>~6"]:
        for norm in ["uniform", "degree_linear"]:
            per_seed = []
            for s in SEEDS:
                adj = build_graph(topo, s)
                deg = adj.sum(1)
                v_hist, drive, tvar = run(adj, norm, s)
                t_d = death_times(v_hist)
                r = dict(topo=topo, norm=norm, seed=s,
                         rho_deg=float(stats.spearmanr(deg, t_d)[0]),
                         rho_drive=float(stats.spearmanr(drive, t_d)[0]),
                         rho_tvar=float(stats.spearmanr(tvar, t_d)[0]),
                         rho_partial=partial_spearman(drive, t_d, deg),
                         # les tests symetriques, ceux qui peuvent CASSER l'interpretation :
                         # si le degre garde un fort pouvoir predictif une fois les
                         # grandeurs de champ controlees, le champ n'est pas l'explication.
                         rho_partial_deg=partial_spearman(deg, t_d, drive),
                         rho_partial_deg2=partial_spearman(deg, t_d, drive, tvar),
                         rho_deg_drive=float(stats.spearmanr(deg, drive)[0]),
                         rho_deg_tvar=float(stats.spearmanr(deg, tvar)[0]))
                rows.append(r)
                per_seed.append(r)
            f = lambda k: np.mean([r[k] for r in per_seed])
            print(f"{topo + ' / ' + norm:<26}{f('rho_deg'):>+12.3f}"
                  f"{f('rho_drive'):>+14.3f}{f('rho_tvar'):>+13.3f}"
                  f"{f('rho_partial_deg'):>+16.3f}{f('rho_partial_deg2'):>+19.3f}"
                  f"{f('rho_deg_tvar'):>+15.3f}")

    with CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[csv] {CSV_PATH}")
    print("\nVERDICT MESURE (2026-07-26) -- l'interpretation posee en tete de fichier est "
          "REFUTEE :")
    print("  - rho(drive, t) garde bien un signe negatif sur les six lignes, MAIS")
    print("  - le partiel deg|drive reste a -0.26/-0.38 sous `uniform` : le champ recu ne")
    print("    SUBSUME PAS le degre, il ne fait que l'accompagner ;")
    print("  - la variance temporelle de la cible locale (la traduction par-noeud de")
    print("    l'argument d'echantillonnage du 01/07) ne predit RIEN : rho(tvar, t) ~ +0.1")
    print("    et rho(deg, tvar) ~ -0.1. L'argument du 01/07 est CINEMATIQUE (voisins")
    print("    supposes independants) ; dans la dynamique couplee les voisins sont")
    print("    correles, et la decroissance en 1/deg de la variance ne survit pas.")
    print("  => le PHENOMENE (cascade + fragmentation sous couplage uniforme) est etabli")
    print("     et replique ; son MECANISME au niveau du noeud reste OUVERT. Ne pas")
    print("     raconter 'le degre n'est qu'un proxy du champ' : c'est teste et faux.")
    print(f"Wall time: {time.time() - t0:.1f}s | {len(rows)} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
