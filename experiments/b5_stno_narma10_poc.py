#!/usr/bin/env python3
"""
B5 (fond) -- NARMA10 sur substrat STNO : la comparaison de performance manquante.
==================================================================================
Cree : 2026-07-11 (Claude Fable 5, L'Ingenieur) -- pending explicite du SYNAPSE du
09/07 : "B5 : comparaison de PERFORMANCE (type NARMA10) sur substrat STNO toujours
non faite (seul le mecanisme, pas la tache, a ete teste)". Ce script la fait.

CONTEXTE. Le 08/07, b5_esn_comparison.py a positionne le reservoir Mem4ristor-FHN
contre l'Echo State Network sur NARMA10 (ESN ~5.5x meilleur : M4R n'est pas une
memoire). Le 09/07, b2_stno_amplitude_phase_poc.py a porte le MECANISME du doute
sur un substrat STNO (Slavin-Tiberkevich, amplitude+phase, non-isochronicite) --
mais seulement le mecanisme (reduction de synchronisation), jamais une TACHE.
Ici on branche le meme harness NARMA10 (meme tache, memes splits, meme readout
ridge, meme taille N=100, memes seeds) sur le substrat STNO.

QUESTIONS (dans l'ordre d'importance) :
  1. Ou se place un reservoir de 100 STNO couples sur NARMA10, face aux references
     DEJA MESUREES sur exactement la meme tache/seeds (figures/b5_esn_comparison.csv:
     ESN ~0.35, M4R-FHN FULL ~1.94) ?
  2. Le doute (u_filter identique a dynamics.py, gain capteur 10 = calibre, seul
     regime ou u franchit 0.5 -- POC du 09/07) aide-t-il, nuit-il, ou est-il neutre
     sur la performance de reservoir ?
  3. Le couplage lui-meme paie-t-il ? (STNO_DECOUPLE = K=0, l'analogue du D=0 FHN
     qui GAGNAIT le 07/07 -- replication de cette question sur un autre substrat.)

PROTOCOLE (loyaute) :
  - Tache, splits, readout, metrique : STRICTEMENT ceux de reservoir_narma10_poc
    (make_narma10, T_WASH=300/T_TRAIN=1500/T_TEST=800, ridge 1e-6, NRMSE).
  - Topologie : lattice 10x10 periodique (N=100) -- la meme que le reservoir
    M4R-FHN de reference. (Le protocole Torrejon 2017 est un UNIQUE STNO
    time-multiplexe ; ici on garde le reseau spatial a masque d'entree spatial,
    pour comparer a armes egales avec M4R-FHN et l'ESN. Assume.)
  - Entree : modulation du GAIN par le courant STT (la voie physique standard :
    l'entree u_in(t), tenue K_SUB pas d'integration par symbole, module
    Gamma_plus par noeud via le masque w_in). Pres du seuil d'oscillation, c'est
    la non-linearite maximale du dispositif (Torrejon et al. 2017).
  - Lecture : puissance p=|a|^2 MOYENNE par symbole et par noeud (ce que mesure
    une diode de detection -- lecture physiquement accessible), readout ridge.
  - Fairness : chaque condition choisit son meilleur (input_scale, K_SUB,
    N_nonlin) par seed -- le substrat choisit son echelle de temps et sa
    non-linearite, comme l'ESN avait choisi (rho, iscale, leak).
  - Doute : constantes IDENTIQUES a dynamics.py (epsilon_u=0.02, tau_u=10,
    alpha_surprise=2.0, cap 5.0, leakage 0.01), gain capteur 10 (calibre).
  - Integration : Euler explicite dt=0.005 (le pre-vol du 11/07 a confirme que
    dt=0.01 diverge au pire cas gain=10 x N_nonlin=10, exactement comme la
    calibration du POC du 09/07 -> on reprend la valeur validee dt=0.005 sur
    toute la plage). Garde anti-divergence conservee : tout run divergent est
    marque NaN, exclu, compte et rapporte.

Statut : exploratoire, hors preprint, aucun claim modifie. Coeur non touche.
Sorties : figures/b5_stno_narma10_poc.csv / _agg.csv / .png
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
import reservoir_narma10_poc as rc  # noqa: E402  (make_narma10, ridge_nrmse, splits)
from mem4ristor.graph_utils import make_lattice_adj  # noqa: E402

CSV_PATH = ROOT / "figures" / "b5_stno_narma10_poc.csv"
AGG_PATH = ROOT / "figures" / "b5_stno_narma10_poc_agg.csv"
PNG_PATH = ROOT / "figures" / "b5_stno_narma10_poc.png"
REF_CSV = ROOT / "figures" / "b5_esn_comparison.csv"

SEEDS = [0, 1, 2, 3, 4]          # sous-ensemble des seeds 0..7 de b5_esn_comparison
N = 100                           # lattice 10x10, comme la reference M4R-FHN
DT = 0.005                        # dt=0.01 diverge au pire cas (verifie en pre-vol le
                                  # 11/07, coherent avec la calibration du 09/07) ->
                                  # dt=0.005, la valeur validee sur toute la plage N_nonlin
K_SUBS = [50, 100]                # pas d'integration par symbole (0.25 et 0.5 unite de
                                  # temps physique). Choix REVISE apres 1er lancement
                                  # (couts) ET meilleur pour la memoire fading : tau_p
                                  # ~2.5 unites / symbole court = ~10 et ~5 symboles de
                                  # memoire d'amplitude, l'ordre exige par NARMA10.
ISCALES = [0.1, 0.3, 0.6]         # modulation du gain (relative, via masque w_in)
N_NONLINS = [0.0, 10.0]           # non-isochronicite (0 = isochrone, 10 = STNO marque)
CONDITIONS = ["STNO_FULL", "STNO_FROZEN_U", "STNO_DECOUPLE"]

# --- Oscillateur Slavin-Tiberkevich : constantes du POC du 09/07 (identiques) ---
GAMMA_MINUS = 1.0
GAMMA_PLUS = 1.2                  # p* isole = 0.2
Q = 1.0
OMEGA0 = 1.0
SIGMA_OMEGA = 0.15
K_COUPLING = 0.3
SIGMA_NOISE = 0.02

# --- Doute : IDENTIQUE a dynamics.py (aucun reglage propre) ---
EPSILON_U = 0.02
K_U = 1.0
SIGMA_BASELINE = 0.05
TAU_U = 10.0
ALPHA_SURPRISE = 2.0
SURPRISE_CAP = 5.0
SOCIAL_LEAKAGE = 0.01
GAIN_U = 10.0                     # capteur calibre (seul regime ou u franchit 0.5, POC 09/07)


def run_stno_reservoir(u_in: np.ndarray, w_in: np.ndarray, condition: str,
                       seed: int, k_sub: int, n_nonlin: float,
                       adj: np.ndarray, deg: np.ndarray) -> np.ndarray | None:
    """Etats (moyenne de p par symbole) d'un reservoir de N STNO couples.
    Retourne None si l'integration diverge (compte par l'appelant)."""
    rng = np.random.default_rng(seed)                  # PCG64 (vitesse) -- pas de gate ici
    n = adj.shape[0]
    omega = OMEGA0 + rng.normal(0, SIGMA_OMEGA, n)
    a = 0.05 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    u = np.full(n, SIGMA_BASELINE)
    k_eff = 0.0 if condition == "STNO_DECOUPLE" else K_COUPLING
    frozen = condition == "STNO_FROZEN_U"
    inv_sqrt_dt = 1.0 / np.sqrt(DT)
    zeros_n = np.zeros(n)

    states = np.zeros((len(u_in), n))
    for t, ui in enumerate(u_in):
        gamma_plus_eff = GAMMA_PLUS + w_in * ui        # courant STT module le gain
        # bruit du symbole entier en un seul tirage (physique identique : i.i.d. par pas)
        noise = rng.normal(0.0, SIGMA_NOISE, size=(k_sub, 2, n))
        p_accum = np.zeros(n)
        for k in range(k_sub):
            # couplage : S_i = mean_j voisins (a_j - a_i)  (forme optimisee, identique)
            S = (adj @ a) / deg - a
            sigma_social_for_u = zeros_n if frozen else np.abs(S) * GAIN_U
            u_filter = np.tanh(np.pi * (0.5 - u)) + SOCIAL_LEAKAGE

            p = np.abs(a) ** 2
            growth = gamma_plus_eff - GAMMA_MINUS * (1.0 + Q * p)
            domega = omega + n_nonlin * p
            eta = (noise[k, 0] + 1j * noise[k, 1]) * inv_sqrt_dt
            da = (growth + 1j * domega) * a + k_eff * u_filter * S + eta

            sigma_safe = np.clip(sigma_social_for_u, 0.0, 100.0)
            eps_adapt = EPSILON_U * np.clip(1.0 + ALPHA_SURPRISE * sigma_safe, 1.0, SURPRISE_CAP)
            du = eps_adapt * (K_U * sigma_social_for_u + SIGMA_BASELINE - u) / TAU_U

            a = a + da * DT
            u = np.clip(u + du * DT, 0.0, 1.0)
            p_accum += np.abs(a) ** 2

        if not np.all(np.isfinite(a)):
            return None
        states[t] = p_accum / k_sub                    # lecture diode : p moyen / symbole
    return states


def load_reference(seeds: list[int]) -> dict[str, list[float]]:
    """Meilleur NRMSE par seed depuis figures/b5_esn_comparison.csv (meme tache,
    memes seeds 0..7 -> sous-ensemble). ESN = min sur sa grille ; M4R = ligne unique."""
    ref: dict[str, dict[int, float]] = {"ESN": {}, "M4R_FULL": {}, "M4R_DECOUPLE": {}}
    with REF_CSV.open() as f:
        for row in csv.DictReader(f):
            s = int(row["seed"])
            if s not in seeds:
                continue
            model = row["model"]
            nrmse = float(row["nrmse"])
            if model == "ESN":
                ref["ESN"][s] = min(ref["ESN"].get(s, np.inf), nrmse)
            elif model in ref:
                ref[model][s] = nrmse
    return {m: [d[s] for s in seeds] for m, d in ref.items() if len(d) == len(seeds)}


def boot_ci_paired(a, b, n_boot=10000, seed=20260711):
    rng = np.random.RandomState(seed)
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    m = np.empty(n_boot)
    for k in range(n_boot):
        m[k] = d[rng.randint(0, n, n)].mean()
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> int:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_steps = rc.T_WASH + rc.T_TRAIN + rc.T_TEST

    adj = make_lattice_adj(10, periodic=True).astype(float)
    deg = adj.sum(axis=1)
    assert np.all(deg > 0)

    # ---------- pre-vol : stabilite Euler sur le pire cas de la grille ----------
    print(f"[pre-vol] stabilite dt={DT} sur le pire cas (FULL, iscale=0.6, "
          f"K_SUB={max(K_SUBS)}, N_nonlin=10, seed=0, 300 symboles)...")
    u_pre, _ = rc.make_narma10(300, seed=0)
    w_pre = np.random.default_rng(1000).uniform(-1.0, 1.0, N) * 0.6
    pre = run_stno_reservoir(u_pre, w_pre, "STNO_FULL", 0, max(K_SUBS), 10.0, adj, deg)
    if pre is None:
        print("[pre-vol] DIVERGENCE au pire cas -> ce script exigerait dt plus fin.")
        print("          Campagne annulee proprement (aucun resultat partiel ecrit).")
        return 1
    print(f"[pre-vol] OK (p final moyen = {pre[-1].mean():.3f})")

    # ---------- campagne ----------
    rows = []
    diverged = 0
    best = {c: {"nrmse": [], "eff_rank": [], "hp": []} for c in CONDITIONS}
    total = len(SEEDS) * len(CONDITIONS) * len(ISCALES) * len(K_SUBS) * len(N_NONLINS)
    done = 0

    print(f"\n{'cond':<14}{'seed':>5}{'best hp (iscale,k_sub,N_nl)':>30}{'NRMSE':>10}{'eff_rank':>10}")
    print("-" * 72)
    for seed in SEEDS:
        u_in, y = rc.make_narma10(n_steps, seed=seed)
        rng_mask = np.random.default_rng(1000 + seed)
        w_in_base = rng_mask.uniform(-1.0, 1.0, N)     # meme masque que la reference
        for cond in CONDITIONS:
            b_nrmse, b_rank, b_hp = np.inf, np.nan, None
            for iscale in ISCALES:
                for k_sub in K_SUBS:
                    for n_nonlin in N_NONLINS:
                        states = run_stno_reservoir(u_in, w_in_base * iscale, cond,
                                                    seed, k_sub, n_nonlin, adj, deg)
                        done += 1
                        if states is None:
                            diverged += 1
                            rows.append((cond, seed, iscale, k_sub, n_nonlin,
                                         np.nan, np.nan))
                            continue
                        nrmse, eff_rank = rc.ridge_nrmse(states, y)
                        rows.append((cond, seed, iscale, k_sub, n_nonlin,
                                     nrmse, eff_rank))
                        if nrmse < b_nrmse:
                            b_nrmse, b_rank, b_hp = nrmse, eff_rank, (iscale, k_sub, n_nonlin)
            best[cond]["nrmse"].append(b_nrmse)
            best[cond]["eff_rank"].append(b_rank)
            best[cond]["hp"].append(b_hp)
            print(f"{cond:<14}{seed:>5}{str(b_hp):>30}{b_nrmse:>10.4f}{b_rank:>10.2f}"
                  f"   [{done}/{total}, {time.time()-t0:.0f}s]")

    if diverged:
        print(f"\n[garde] {diverged} run(s) divergents exclus (marques NaN dans le CSV).")

    # ---------- references (memes tache/seeds, mesurees le 08/07) ----------
    ref = load_reference(SEEDS)

    print("\n=== RESUME NRMSE NARMA10 (plus bas = mieux ; meilleur hyperparam par modele) ===")
    summary = {}
    for cond in CONDITIONS:
        arr = np.array(best[cond]["nrmse"])
        summary[cond] = arr
        print(f"  {cond:<16}: {arr.mean():.4f} +/- {arr.std():.4f}")
    for name, vals in ref.items():
        arr = np.array(vals)
        print(f"  {name:<16}: {arr.mean():.4f} +/- {arr.std():.4f}   "
              f"(reference 08/07, memes tache/seeds)")

    # ---------- verdict ----------
    print("\n=== VERDICT B5-STNO (honnete) ===")
    stno = summary["STNO_FULL"]
    stno_f = summary["STNO_FROZEN_U"]
    stno_d = summary["STNO_DECOUPLE"]

    d, lo, hi = boot_ci_paired(stno, stno_f)
    verdict_doubt = ("le doute AIDE" if hi < 0 else
                     ("le doute NUIT" if lo > 0 else "doute NEUTRE (IC couvre 0)"))
    print(f"  1. STNO_FULL - STNO_FROZEN_U = {d:+.4f} CI[{lo:+.4f},{hi:+.4f}] -> {verdict_doubt}")

    d2, lo2, hi2 = boot_ci_paired(stno, stno_d)
    verdict_coupling = ("le couplage(+doute) PAIE" if hi2 < 0 else
                        ("le DECOUPLE gagne (comme sur FHN le 07/07)" if lo2 > 0
                         else "parite statistique"))
    print(f"  2. STNO_FULL - STNO_DECOUPLE = {d2:+.4f} CI[{lo2:+.4f},{hi2:+.4f}] -> {verdict_coupling}")

    if "M4R_FULL" in ref:
        d3, lo3, hi3 = boot_ci_paired(stno, np.array(ref["M4R_FULL"]))
        pos = ("le substrat STNO BAT le FHN-M4R" if hi3 < 0 else
               ("le FHN-M4R bat le STNO" if lo3 > 0 else "parite avec FHN-M4R"))
        print(f"  3. STNO_FULL - M4R_FHN_FULL = {d3:+.4f} CI[{lo3:+.4f},{hi3:+.4f}] -> {pos}")
    if "ESN" in ref:
        d4, lo4, hi4 = boot_ci_paired(stno, np.array(ref["ESN"]))
        pos = ("le STNO bat l'ESN (!)" if hi4 < 0 else
               ("l'ESN reste devant" if lo4 > 0 else "parite avec l'ESN"))
        print(f"  4. STNO_FULL - ESN          = {d4:+.4f} CI[{lo4:+.4f},{hi4:+.4f}] -> {pos}")
    if stno.mean() > 1.0:
        print("  [absolu] NRMSE STNO > 1.0 : pire que predire la moyenne -- pas un "
              "reservoir utile sur NARMA10 dans ce protocole spatial.")
    elif stno.mean() < 1.0:
        print("  [absolu] NRMSE STNO < 1.0 : reservoir UTILE dans l'absolu sur NARMA10 "
              "(contrairement au FHN-M4R de reference, ~1.9).")

    # ---------- CSV ----------
    with CSV_PATH.open("w", encoding="utf-8") as f:
        f.write("condition,seed,input_scale,k_sub,n_nonlin,nrmse,eff_rank\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    with AGG_PATH.open("w", encoding="utf-8") as f:
        f.write("model,nrmse_mean,nrmse_std,n_seeds,source\n")
        for cond in CONDITIONS:
            arr = summary[cond]
            f.write(f"{cond},{arr.mean():.6f},{arr.std():.6f},{len(arr)},this_script\n")
        for name, vals in ref.items():
            arr = np.array(vals)
            f.write(f"{name},{arr.mean():.6f},{arr.std():.6f},{len(arr)},b5_esn_comparison_20260708\n")
    print(f"\n[csv] {CSV_PATH}\n[csv] {AGG_PATH}")

    # ---------- figure ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
        names, means, stds, colors = [], [], [], []
        palette = {"ESN": "#1f77b4", "STNO_FULL": "#9467bd", "STNO_FROZEN_U": "#c5b0d5",
                   "STNO_DECOUPLE": "#7f7f7f", "M4R_FULL": "#2ca02c", "M4R_DECOUPLE": "#98df8a"}
        order = ["ESN", "STNO_FULL", "STNO_FROZEN_U", "STNO_DECOUPLE", "M4R_FULL", "M4R_DECOUPLE"]
        data_all = {**{c: summary[c] for c in CONDITIONS},
                    **{k: np.array(v) for k, v in ref.items()}}
        for nm in order:
            if nm in data_all:
                names.append(nm.replace("_", "\n", 1))
                means.append(data_all[nm].mean())
                stds.append(data_all[nm].std())
                colors.append(palette.get(nm, "#333333"))
        ax1.bar(names, means, yerr=stds, color=colors, edgecolor="k", capsize=4)
        ax1.axhline(1.0, ls=":", c="red", label="NRMSE=1 (predire la moyenne)")
        ax1.set_ylabel("NARMA10 NRMSE (plus bas = mieux)")
        ax1.set_title(f"Reservoir STNO vs references (N={N}, {len(SEEDS)} seeds)")
        ax1.tick_params(axis="x", labelsize=7)
        ax1.legend(fontsize=8)
        ax1.grid(axis="y", alpha=0.3)
        x = np.arange(len(SEEDS))
        for nm in ["ESN", "STNO_FULL", "M4R_FULL"]:
            if nm in data_all:
                ax2.plot(x, data_all[nm], "o-", color=palette[nm], label=nm)
        ax2.axhline(1.0, ls=":", c="red")
        ax2.set_xlabel("seed")
        ax2.set_ylabel("NRMSE")
        ax2.set_title("Apparie par seed")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)
        fig.suptitle("B5 fond : le substrat STNO sur NARMA10 (harness et references du 08/07)",
                     fontsize=11)
        plt.tight_layout()
        plt.savefig(PNG_PATH, dpi=140)
        print(f"[png] {PNG_PATH}")
    except Exception as e:
        print(f"[png] skipped: {e}")

    print(f"\nWall time: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
