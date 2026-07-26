#!/usr/bin/env python3
"""
EXPERIENCE A -- Cascade topologique d'entree en dead zone sur BA.

CONTEXTE (audit externe "Google DeepMind" joue par Gemini, 2026-07-17, point 3 ;
mis au programme par Julien, execute le 2026-07-26).
  Prediction de l'audit : sur un graphe heterogene (Barabasi-Albert), l'entree en
  dead zone ne serait PAS un basculement homogene mais une CASCADE INITIEE PAR
  LES HUBS -- le champ moyen sur un hub est colossal, il s'effondrerait en premier,
  et le sous-graphe encore vivant se FRAGMENTERAIT en perdant ses connecteurs.

  Nuance interne connue AVANT la mesure, qui rend le resultat interessant dans les
  deux sens : le mecanisme etabli le 01/07 (champ moyen par echantillonnage,
  `lambda2_foundation_20260701/`) est gouverne par k_harm = 1/<1/deg>, une
  grandeur DOMINEE PAR LA PERIPHERIE. Le regime GLOBAL suit donc les bas degres --
  ce qui ne dit rien sur l'ORDRE d'entree noeud par noeud, jamais mesure.

  Prudence exigee : les recits specifiques aux hubs ont deja casse 2 fois dans ce
  projet ([13] revise, P3 refute). D'ou le gate de replication sur graines
  DISJOINTES, obligatoire avant de croire quoi que ce soit ici.

DEFINITION -- "entree en dead zone" PAR NOEUD (non parametrique) :
  t_mort_i = 1 + (dernier t ou v_i(t) > V_DEAD),  = STEPS si jamais verrouille.
  C'est un VERROUILLAGE definitif, pas un franchissement instantane. Une
  reconnaissance prealable a montre que le critere instantane (v_i <= -1.2 a
  l'instant t) est pollue par le transitoire de cold start : a t=200 la quasi
  totalite des noeuds est sous le seuil, puis ~40 % en ressortent.
  Sensibilite au seuil testee (-1.0 / -1.2 / -1.35) pour repondre a la reserve
  A5 (H_cog = artefact de binning) : -1.2 est le bord de bin canonique, -1.35
  est SOUS le point fixe v* ~= -1.294 -- si le resultat tient aux deux, il ne
  vient pas du binning.

CE QUE LE SCRIPT CONTROLE (les trois portes de sortie d'un faux positif) :
  1. NORMALISATION du couplage. `uniform` = le noeud recoit la somme de ses k
     voisins ; `degree_linear` = divise par k. Ce choix agit DIRECTEMENT sur le
     rapport hub/peripherie -- le tester est necessaire, pas decoratif.
  2. RELABELING aleatoire des noeuds (BA seulement). Sur BA, degre et "age"
     (ordre d'ajout) sont confondus. Permuter les etiquettes preserve la
     distribution des degres et casse le lien indice->degre : si l'effet vient
     d'un artefact d'indice, il meurt ici.
  3. PERCOLATION ALEATOIRE comme null de fragmentation. BA est notoirement
     ROBUSTE a la suppression aleatoire et fragile a la suppression ciblee sur
     les hubs : comparer la fragmentation observee a celle d'un retrait aleatoire
     du MEME nombre de noeuds est le seul test qui distingue les deux.
  + Generalite : ER a <k> comparable (l'effet est-il specifique au scale-free ?).

STATISTIQUE -- attention, point important :
  Le Spearman intra-run porte sur des noeuds COUPLES, donc non independants : son
  p-value est optimiste et n'est PAS rapporte comme preuve. L'inference se fait au
  niveau des RUNS (1 rho par graine), par IC bootstrap sur les graines + comptage
  de signes. Le gate de replication rejoue tout sur 10 graines disjointes.

Regime : endogene (I_stim=0), cold_start, sigma_v=0.05, N=200.
  Note : a I_stim=0 les heretiques sont INACTIFS (I_eff[mask] *= -1 applique a 0
  est un no-op -- audit interne du 22/04, scope note du preprint precisee le
  26/07). Aucun confondant "heretique" dans ce regime.

SORTIES :
  figures/expA_ba_cascade_poc.csv   (1 ligne par run)
  figures/expA_ba_cascade_poc.png   (3 panneaux)

Cree : 2026-07-26 (Claude Opus 5, L'Ingenieur) -- Experience A du backlog DeepMind.
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
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from mem4ristor.core import Mem4Network
from mem4ristor.graph_utils import make_ba, make_er

# --- parametres ------------------------------------------------------------
N = 200
STEPS = 3000
I_STIM = 0.0
HERETIC_RATIO = 0.15
SIGMA_V = 0.05

V_DEAD_MAIN = -1.2          # bord de bin canonique (definition du regime dead)
V_DEAD_CTRL = -1.35         # sous le point fixe v* ~= -1.294 (controle binning)
V_DEAD_DIAG = -1.0          # diagnostic : capte la fin du transitoire, pas la mort

SEEDS_PRIMARY = [42, 123, 777, 17, 256, 8, 99, 314, 2718, 1618]
SEEDS_REPLICATION = [5001, 5002, 5003, 5004, 5005, 5006, 5007, 5008, 5009, 5010]

N_NULL_DRAWS = 200          # tirages de percolation aleatoire par run
HUB_FRAC = 0.10             # top 10 % des degres = "hubs"
PERI_FRAC = 0.50            # bottom 50 % des degres = "peripherie"

FIG_DIR = ROOT / "figures"
CSV_PATH = FIG_DIR / "expA_ba_cascade_poc.csv"
PNG_PATH = FIG_DIR / "expA_ba_cascade_poc.png"


# --- topologies ------------------------------------------------------------
def build_graph(topo: str, seed: int, shuffle: bool):
    if topo == "BA m=3":
        adj = make_ba(N, 3, seed)
    elif topo == "BA m=5":
        adj = make_ba(N, 5, seed)
    elif topo == "ER <k>~6":
        adj = make_er(N, 6.0 / (N - 1), seed)
    else:
        raise ValueError(topo)
    if shuffle:
        perm = np.random.RandomState(seed + 10_000).permutation(N)
        adj = adj[perm][:, perm]
    return adj


# --- simulation ------------------------------------------------------------
def simulate(adj: np.ndarray, norm: str, seed: int) -> np.ndarray:
    net = Mem4Network(size=int(np.sqrt(adj.shape[0])), heretic_ratio=HERETIC_RATIO,
                      seed=seed, adjacency_matrix=adj.copy(), coupling_norm=norm,
                      cold_start=True)
    net.model.cfg["noise"]["sigma_v"] = SIGMA_V
    v_hist = np.empty((STEPS, N), dtype=float)
    for t in range(STEPS):
        net.step(I_stimulus=I_STIM)
        v_hist[t] = net.v
    return v_hist


def death_times(v_hist: np.ndarray, v_dead: float) -> np.ndarray:
    """1 + dernier instant AU-DESSUS du seuil (verrouillage definitif).
    STEPS = censure (le noeud est encore ressorti au dernier pas)."""
    above = v_hist > v_dead
    t_d = np.zeros(N, dtype=int)
    for i in range(N):
        idx = np.flatnonzero(above[:, i])
        t_d[i] = 0 if idx.size == 0 else int(idx[-1]) + 1
    return t_d


# --- fragmentation ---------------------------------------------------------
def subgraph_stats(A_csr, alive_mask):
    k = int(alive_mask.sum())
    if k == 0:
        return 0, 0.0
    sub = A_csr[alive_mask][:, alive_mask]
    ncomp, labels = connected_components(sub, directed=False)
    big = int(np.bincount(labels).max())
    return int(ncomp), big / k


def fragmentation_at(adj, t_d, frac_dead_target, rng):
    """Fragmentation du sous-graphe vivant quand `frac_dead_target` des noeuds
    sont morts, comparee au retrait ALEATOIRE du meme nombre de noeuds."""
    A = csr_matrix(adj)
    n_dead_target = int(round(frac_dead_target * N))
    order = np.argsort(t_d, kind="stable")          # morts les plus precoces d'abord
    if n_dead_target == 0 or n_dead_target >= N:
        return dict(n_comp=np.nan, frac_big=np.nan, n_comp_null=np.nan,
                    frac_big_null=np.nan, pct_ncomp=np.nan)
    dead_idx = order[:n_dead_target]
    alive = np.ones(N, dtype=bool)
    alive[dead_idx] = False
    ncomp_obs, big_obs = subgraph_stats(A, alive)

    ncomps, bigs = [], []
    for _ in range(N_NULL_DRAWS):
        alive_n = np.ones(N, dtype=bool)
        alive_n[rng.choice(N, size=n_dead_target, replace=False)] = False
        c, b = subgraph_stats(A, alive_n)
        ncomps.append(c)
        bigs.append(b)
    ncomps = np.asarray(ncomps)
    return dict(n_comp=ncomp_obs, frac_big=big_obs,
                n_comp_null=float(np.median(ncomps)),
                frac_big_null=float(np.median(bigs)),
                pct_ncomp=float((ncomps < ncomp_obs).mean()))


# --- un run ----------------------------------------------------------------
def run_one(topo, norm, shuffle, seed) -> dict:
    adj = build_graph(topo, seed, shuffle)
    deg = adj.sum(1)
    v_hist = simulate(adj, norm, seed)
    rng = np.random.RandomState(seed + 777)

    row = dict(topo=topo, norm=norm, shuffle=int(shuffle), seed=seed,
               deg_min=int(deg.min()), deg_max=int(deg.max()),
               k_mean=float(deg.mean()),
               k_harm=float(1.0 / np.mean(1.0 / np.maximum(deg, 1.0))))

    for tag, v_dead in [("main", V_DEAD_MAIN), ("ctrl", V_DEAD_CTRL), ("diag", V_DEAD_DIAG)]:
        t_d = death_times(v_hist, v_dead)
        rho, p = stats.spearmanr(deg, t_d)
        row[f"rho_{tag}"] = float(rho)
        row[f"p_intrarun_{tag}"] = float(p)          # optimiste : noeuds couples
        row[f"censored_{tag}"] = int((t_d >= STEPS).sum())
        row[f"tmort_med_{tag}"] = float(np.median(t_d))

    # contraste hubs / peripherie au seuil principal
    t_d = death_times(v_hist, V_DEAD_MAIN)
    order_deg = np.argsort(deg)
    hubs = order_deg[-max(1, int(HUB_FRAC * N)):]
    peri = order_deg[:int(PERI_FRAC * N)]
    row["tmort_hubs"] = float(np.median(t_d[hubs]))
    row["tmort_peri"] = float(np.median(t_d[peri]))
    row["delta_hub_peri"] = row["tmort_hubs"] - row["tmort_peri"]

    # fragmentation a 50 % et 75 % de morts, vs percolation aleatoire
    for fd in (0.50, 0.75):
        st = fragmentation_at(adj, t_d, fd, rng)
        key = f"{int(fd * 100)}"
        row[f"ncomp{key}"] = st["n_comp"]
        row[f"ncomp{key}_null"] = st["n_comp_null"]
        row[f"big{key}"] = st["frac_big"]
        row[f"big{key}_null"] = st["frac_big_null"]
        row[f"pct_ncomp{key}"] = st["pct_ncomp"]
    return row


# --- agregation ------------------------------------------------------------
def boot_ci(vals, n_boot=10_000, seed=0):
    v = np.asarray(vals, dtype=float)
    rng = np.random.RandomState(seed)
    means = v[rng.randint(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(rows, key="rho_main"):
    vals = [r[key] for r in rows]
    lo, hi = boot_ci(vals)
    return dict(mean=float(np.mean(vals)), lo=lo, hi=hi,
                n_neg=int(sum(v < 0 for v in vals)), n=len(vals))


CONDITIONS = [
    ("BA m=3", "uniform", False),
    ("BA m=3", "degree_linear", False),
    ("BA m=3", "uniform", True),
    ("BA m=5", "uniform", False),
    ("BA m=5", "degree_linear", False),
    ("BA m=5", "uniform", True),
    ("ER <k>~6", "uniform", False),
    ("ER <k>~6", "degree_linear", False),
]


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = []

    print("=" * 108)
    print("EXPERIENCE A -- cascade topologique d'entree en dead zone (reponse DeepMind pt 3)")
    print(f"N={N} steps={STEPS} endogene(I_stim=0) cold_start sigma_v={SIGMA_V} "
          f"| seuil principal v<={V_DEAD_MAIN}")
    print(f"{len(CONDITIONS)} conditions x ({len(SEEDS_PRIMARY)} graines primaires "
          f"+ {len(SEEDS_REPLICATION)} graines DISJOINTES)")
    print("=" * 108)

    for topo, norm, shuffle in CONDITIONS:
        for group, seeds in [("primary", SEEDS_PRIMARY), ("replication", SEEDS_REPLICATION)]:
            for s in seeds:
                r = run_one(topo, norm, shuffle, s)
                r["group"] = group
                rows.append(r)
        prim = [r for r in rows if r["topo"] == topo and r["norm"] == norm
                and r["shuffle"] == int(shuffle) and r["group"] == "primary"]
        repl = [r for r in rows if r["topo"] == topo and r["norm"] == norm
                and r["shuffle"] == int(shuffle) and r["group"] == "replication"]
        sp, sr = summarize(prim), summarize(repl)
        lbl = f"{topo} / {norm}" + (" / shuffled" if shuffle else "")
        print(f"{lbl:<34} rho_primary={sp['mean']:+.3f} [{sp['lo']:+.3f},{sp['hi']:+.3f}] "
              f"({sp['n_neg']}/{sp['n']} neg)   "
              f"rho_replic={sr['mean']:+.3f} [{sr['lo']:+.3f},{sr['hi']:+.3f}] "
              f"({sr['n_neg']}/{sr['n']} neg)")

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[csv] {CSV_PATH}")

    # ---- verdicts -------------------------------------------------------
    print("\n" + "=" * 108)
    print("GATE DE REPLICATION (graines disjointes) -- un effet n'est retenu que si le "
          "signe tient ET les deux IC excluent 0")
    print("-" * 108)
    print(f"{'condition':<34}{'primary':>26}{'replication':>26}{'verdict':>18}")
    verdicts = {}
    for topo, norm, shuffle in CONDITIONS:
        sel = lambda g: [r for r in rows if r["topo"] == topo and r["norm"] == norm
                         and r["shuffle"] == int(shuffle) and r["group"] == g]
        sp, sr = summarize(sel("primary")), summarize(sel("replication"))
        held = (sp["hi"] < 0 and sr["hi"] < 0) or (sp["lo"] > 0 and sr["lo"] > 0)
        same_sign = np.sign(sp["mean"]) == np.sign(sr["mean"])
        verdict = "REPLIQUE" if (held and same_sign) else "non replique"
        lbl = f"{topo} / {norm}" + (" / shuffled" if shuffle else "")
        verdicts[lbl] = (sp, sr, verdict)
        print(f"{lbl:<34}{sp['mean']:>+9.3f} [{sp['lo']:+.2f},{sp['hi']:+.2f}]"
              f"{sr['mean']:>+13.3f} [{sr['lo']:+.2f},{sr['hi']:+.2f}]{verdict:>18}")

    print("\n" + "-" * 108)
    print("SENSIBILITE AU SEUIL (le resultat vient-il du binning ?) -- moyennes sur les "
          "20 graines des deux groupes")
    print(f"{'condition':<34}{'rho @-1.2':>12}{'rho @-1.35':>12}{'rho @-1.0 (diag)':>20}"
          f"{'censures @-1.2':>16}")
    for topo, norm, shuffle in CONDITIONS:
        sel = [r for r in rows if r["topo"] == topo and r["norm"] == norm
               and r["shuffle"] == int(shuffle)]
        lbl = f"{topo} / {norm}" + (" / shuffled" if shuffle else "")
        print(f"{lbl:<34}{np.mean([r['rho_main'] for r in sel]):>+12.3f}"
              f"{np.mean([r['rho_ctrl'] for r in sel]):>+12.3f}"
              f"{np.mean([r['rho_diag'] for r in sel]):>+20.3f}"
              f"{np.mean([r['censored_main'] for r in sel]):>16.1f}")

    print("\n" + "-" * 108)
    print("FRAGMENTATION du sous-graphe vivant -- observee vs PERCOLATION ALEATOIRE "
          "(meme nombre de noeuds retires)")
    print(f"{'condition':<34}{'a 50% morts':>30}{'a 75% morts':>30}")
    print(f"{'':<34}{'ncomp obs / null   big obs/null':>30}{'ncomp obs / null   big obs/null':>30}")
    for topo, norm, shuffle in CONDITIONS:
        sel = [r for r in rows if r["topo"] == topo and r["norm"] == norm
               and r["shuffle"] == int(shuffle)]
        lbl = f"{topo} / {norm}" + (" / shuffled" if shuffle else "")
        cells = []
        for key in ("50", "75"):
            cells.append(f"{np.mean([r[f'ncomp{key}'] for r in sel]):>6.1f} /"
                         f"{np.mean([r[f'ncomp{key}_null'] for r in sel]):>5.1f}   "
                         f"{np.mean([r[f'big{key}'] for r in sel]):>4.2f}/"
                         f"{np.mean([r[f'big{key}_null'] for r in sel]):>4.2f}")
        print(f"{lbl:<34}{cells[0]:>30}{cells[1]:>30}")

    print("\n" + "-" * 108)
    print("CONTRASTE HUBS (top 10% degre) vs PERIPHERIE (bottom 50%) -- t_mort median, "
          "negatif = les hubs meurent AVANT")
    for topo, norm, shuffle in CONDITIONS:
        sel = [r for r in rows if r["topo"] == topo and r["norm"] == norm
               and r["shuffle"] == int(shuffle)]
        lbl = f"{topo} / {norm}" + (" / shuffled" if shuffle else "")
        d = [r["delta_hub_peri"] for r in sel]
        lo, hi = boot_ci(d)
        print(f"  {lbl:<34} delta = {np.mean(d):>+8.1f} pas  IC95 [{lo:+.1f}, {hi:+.1f}]")

    make_figure(rows)
    print(f"\n[png] {PNG_PATH}")
    print(f"Wall time: {time.time() - t0:.1f}s | {len(rows)} runs")
    return 0


# --- figure ----------------------------------------------------------------
def make_figure(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))

    # Panneau 1 : t_mort vs degre, un run par normalisation (BA m=3, seed 42)
    ax = axes[0]
    for norm, color in [("uniform", "#d62728"), ("degree_linear", "#1f77b4")]:
        adj = build_graph("BA m=3", 42, False)
        deg = adj.sum(1)
        t_d = death_times(simulate(adj, norm, 42), V_DEAD_MAIN)
        ax.scatter(deg, t_d, s=26, alpha=0.7, color=color, edgecolors="k",
                   linewidths=0.3, label=f"{norm}")
    ax.set_xscale("log")
    ax.set_xlabel("node degree (log)")
    ax.set_ylabel(f"lock-in time into dead zone (steps, v<={V_DEAD_MAIN})")
    ax.set_title("BA m=3, seed 42: hubs die first only\nunder uniform coupling", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panneau 2 : rho par condition, primaires vs replication
    ax = axes[1]
    labels, xs_p, xs_r = [], [], []
    for topo, norm, shuffle in CONDITIONS:
        lbl = f"{topo}\n{norm}" + ("\nshuffled" if shuffle else "")
        labels.append(lbl)
        sel = lambda g: [r["rho_main"] for r in rows if r["topo"] == topo
                         and r["norm"] == norm and r["shuffle"] == int(shuffle)
                         and r["group"] == g]
        xs_p.append(sel("primary"))
        xs_r.append(sel("replication"))
    pos = np.arange(len(labels))
    for i, (p, r) in enumerate(zip(xs_p, xs_r)):
        ax.scatter(np.full(len(p), i - 0.13), p, s=18, color="#1f77b4", alpha=0.8,
                   label="primary seeds" if i == 0 else None)
        ax.scatter(np.full(len(r), i + 0.13), r, s=18, color="#ff7f0e", alpha=0.8,
                   marker="^", label="disjoint seeds (replication)" if i == 0 else None)
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel("Spearman rho(degree, lock-in time)")
    ax.set_title("Replication gate: one rho per seed", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    # Panneau 3 : fragmentation observee vs percolation aleatoire
    ax = axes[2]
    for topo, norm, color, ls in [("BA m=3", "uniform", "#d62728", "-"),
                                  ("BA m=3", "degree_linear", "#1f77b4", "-")]:
        adj = build_graph(topo, 42, False)
        t_d = death_times(simulate(adj, norm, 42), V_DEAD_MAIN)
        rng = np.random.RandomState(1)
        fr = np.arange(0.1, 0.95, 0.05)
        obs, null = [], []
        for f in fr:
            st = fragmentation_at(adj, t_d, f, rng)
            obs.append(st["n_comp"])
            null.append(st["n_comp_null"])
        ax.plot(fr, obs, ls, color=color, marker="o", ms=3, label=f"{norm} (observed)")
        ax.plot(fr, null, "--", color=color, alpha=0.55, label=f"{norm} (random removal)")
    ax.set_xlabel("fraction of nodes locked into dead zone")
    ax.set_ylabel("connected components of the surviving subgraph")
    ax.set_title("Fragmentation vs random percolation\n(BA m=3, seed 42)", fontsize=10)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)

    fig.suptitle("Experiment A -- topological cascade into the dead zone on Barabasi-Albert "
                 f"(N={N}, endogenous, {len(SEEDS_PRIMARY)}+{len(SEEDS_REPLICATION)} seeds)",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=140)


if __name__ == "__main__":
    raise SystemExit(main())
