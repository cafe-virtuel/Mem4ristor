#!/usr/bin/env python3
"""
POC PISTE 2 -- LE DOUTE COMME ALLOCATEUR DE COMPUTE sur un FLUX de problemes.

Vision (Julien) : "explorer tant que le doute persiste". Ici on renverse le watchdog interne
(piste 1, un seul probleme) : un FLUX de K problemes de difficulte heterogene, et le DOUTE
decide combien de calcul chacun merite (adaptive computation time pilote par le doute).

Substrat = tache de DECISION / CONSENSUS. Le reseau doit trancher un SIGNE global d'apres
l'evidence nette injectee. "Resolu" = consensus atteint = doute (sigma_social = |L v|) retombe.
On mesure a BUDGET TOTAL DE COMPUTE EGAL combien de problemes sont bien tranches.

Familles de difficulte (les 3 en parallele, melangees dans le flux) :
  EVIDENCE      : drive faible (peu de capteurs / E petit) -> consensus lent.
  CONTRADICTION : capteurs + et - quasi a egalite -> deliberation longue et incertaine.
  TOPOLOGIE     : evidence claire mais connectivite qui ralentit l'integration (ring/BA sparse).

3 conditions d'ALLOCATION, a budget total B_total identique :
  DOUTE (ACT)   : on quitte un probleme quand sigma_social retombe sous 30% de son pic initial
                  (chute relative du desaccord = "le doute est retombe"). Signal doute-driven.
  UNIFORME      : chaque probleme recoit exactement B_total/K pas (allocation aveugle).
  CONVERGENCE   : CONTROLE HONNETE -- on quitte quand v se stabilise (||dv|| faible sur fenetre),
                  critere trivial qui n'utilise PAS le doute. Le doute doit le battre pour valoir.

Metrique decisive : # problemes correctement tranches / K, a B_total egal.
Diagnostic : compute alloue par le doute vs difficulte "oracle" (temps de consensus en budget large).

Efficacite : chaque probleme est simule UNE fois jusqu'au budget max (trajectoire deterministe),
puis chaque condition ne fait que CHOISIR son instant d'arret dans cette trajectoire.

Sortie : figures/doubt_compute_allocation_poc.csv + .png + verdict.
Cree : 2026-07-07 (Claude Opus 4.8) -- piste 2 (design flux d'entrees).
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
from mem4ristor.graph_utils import make_lattice_adj, make_ba  # noqa: E402

CSV = ROOT / "figures" / "doubt_compute_allocation_poc.csv"
PNG = ROOT / "figures" / "doubt_compute_allocation_poc.png"

SIDE, N = 10, 100
MAX_BUDGET = 2000            # pas max simules par probleme (trajectoire de reference)
WARMUP = 30                  # pas min avant d'autoriser un arret (laisser la transitoire s'etablir)
DOUBT_DROP = 0.30            # DOUTE : arret quand sigma < DOUBT_DROP * pic initial
CONV_W = 50                  # CONVERGENCE : fenetre de comparaison de la variable de decision
CONV_THR = 0.02             # CONVERGENCE : la variable de decision a cesse de bouger (|d(t)-d(t-W)|)
SEEDS = [0, 1, 2, 3, 4, 5]
N_PER_FAMILY = 4             # K = 3 familles x N_PER_FAMILY = 12 problemes

def lattice_adj():
    return make_lattice_adj(SIDE, periodic=True)

def ring_adj(k=2):
    """Anneau 1D a 2k voisins (topologie sparse -> integration lente)."""
    A = np.zeros((N, N))
    for i in range(N):
        for d in range(1, k + 1):
            A[i, (i + d) % N] = 1.0
            A[i, (i - d) % N] = 1.0
    return A

def make_problem(family, rng, seed):
    """Retourne (adj, stim, D_star, label). D_star = signe correct de la decision."""
    if family == "EVIDENCE":
        adj = lattice_adj()
        # drive faible : 1 a 3 capteurs de meme signe, E petit
        n_sens = rng.randint(1, 4)
        E = rng.uniform(0.25, 0.5)
        sign = rng.choice([-1, 1])
        nodes = rng.choice(N, size=n_sens, replace=False)
        stim = np.zeros(N); stim[nodes] = sign * E
        return adj, stim, sign, f"EVID(n={n_sens},E={E:.2f})"
    if family == "CONTRADICTION":
        adj = lattice_adj()
        # capteurs + et - quasi a egalite ; la majorite (leger desequilibre) definit D_star
        n_plus = rng.randint(3, 6)
        n_minus = n_plus - rng.randint(1, 2)  # 1 ou 2 de moins -> D_star = +1
        E = 1.0
        nodes = rng.choice(N, size=n_plus + n_minus, replace=False)
        stim = np.zeros(N)
        stim[nodes[:n_plus]] = +E
        stim[nodes[n_plus:]] = -E
        return adj, stim, +1, f"CONTRA(+{n_plus}/-{n_minus})"
    if family == "TOPOLOGIE":
        # evidence CLAIRE mais topologie lente (anneau sparse) ou dense (BA) selon tirage
        if rng.random() < 0.5:
            adj = ring_adj(k=rng.choice([1, 2]))
            top = "ring"
        else:
            adj = make_ba(N, m=rng.choice([1, 2]), seed=seed)
            top = "ba"
        n_sens = 5
        sign = rng.choice([-1, 1])
        nodes = rng.choice(N, size=n_sens, replace=False)
        stim = np.zeros(N); stim[nodes] = sign * 1.0
        return adj, stim, sign, f"TOPO({top})"
    raise ValueError(family)

def simulate(adj, stim, seed):
    """Simule le probleme jusqu'a MAX_BUDGET, en soustrayant un run de reference stim=0 au
    MEME seed (meme sequence de bruit ET meme point fixe negatif du FHN -> les deux s'annulent).
    Variable de decision d(t) = mean(v_stim) - mean(v_ref) : positif -> le reseau a bascule +.
    Retourne sigma_social (du run stimule), dec (signe de d), et d_var (d continu)."""
    net = Mem4Network(size=SIDE, heretic_ratio=0.0, seed=seed, adjacency_matrix=adj)
    ref = Mem4Network(size=SIDE, heretic_ratio=0.0, seed=seed, adjacency_matrix=adj)
    L = net.L
    zero = np.zeros(N)
    sig = np.empty(MAX_BUDGET)
    d_var = np.empty(MAX_BUDGET)               # variable de decision differentielle
    dec = np.empty(MAX_BUDGET, dtype=int)
    for t in range(MAX_BUDGET):
        net.step(I_stimulus=stim)
        ref.step(I_stimulus=zero)
        v = net.model.v
        sig[t] = float(np.mean(np.abs(L @ v)))
        d = float(np.mean(v) - np.mean(ref.model.v))
        d_var[t] = d
        dec[t] = 1 if d >= 0 else -1
    return sig, dec, d_var

def stop_doubt(sig):
    """Instant d'arret DOUTE : sigma retombe sous DOUBT_DROP * pic initial (apres WARMUP)."""
    peak = float(np.max(sig[:WARMUP + 20]))  # pic de la transitoire initiale
    thr = DOUBT_DROP * peak
    for t in range(WARMUP, len(sig)):
        if sig[t] < thr:
            return t + 1
    return len(sig)

def stop_conv(d_var):
    """CONVERGENCE (controle) : la variable de DECISION a cesse de bouger sur une fenetre W.
    Meme observable que le doute, mais on regarde la stabilite de d, pas la chute du desaccord."""
    for t in range(WARMUP + CONV_W, len(d_var)):
        if abs(d_var[t] - d_var[t - CONV_W]) < CONV_THR:
            return t + 1
    return len(d_var)

def run_seed(seed):
    rng = np.random.RandomState(2000 + seed)
    families = (["EVIDENCE"] * N_PER_FAMILY + ["CONTRADICTION"] * N_PER_FAMILY +
                ["TOPOLOGIE"] * N_PER_FAMILY)
    rng.shuffle(families)
    K = len(families)
    probs = []
    for idx, fam in enumerate(families):
        adj, stim, dstar, label = make_problem(fam, rng, seed * 100 + idx)
        sig, dec, d_var = simulate(adj, stim, seed * 100 + idx)
        c_doubt = stop_doubt(sig)
        c_conv = stop_conv(d_var)
        # difficulte "oracle" : temps pour atteindre ET tenir la bonne decision (budget large)
        correct = (dec == dstar)
        # premier t apres lequel la decision reste correcte jusqu'a la fin
        oracle = MAX_BUDGET
        for t in range(len(dec)):
            if np.all(correct[t:]):
                oracle = t + 1
                break
        probs.append(dict(fam=fam, label=label, dstar=dstar, dec=dec,
                          c_doubt=c_doubt, c_conv=c_conv, oracle=oracle))
    return K, probs

def decision_at(prob, budget_steps):
    """Decision rendue si on alloue 'budget_steps' pas a ce probleme (borne a MAX_BUDGET)."""
    if budget_steps <= 0:
        return 0            # pas de calcul -> indecis (compte comme faux)
    t = min(int(budget_steps), MAX_BUDGET) - 1
    return int(prob["dec"][t])

def allocate_and_score(probs, B_total, mode):
    """Alloue B_total pas selon 'mode' et compte les decisions correctes."""
    K = len(probs)
    solved = 0
    alloc = []
    if mode == "UNIFORME":
        per = B_total / K
        for p in probs:
            a = per
            alloc.append(a)
            solved += int(decision_at(p, a) == p["dstar"])
    else:  # DOUTE ou CONVERGENCE : glouton sequentiel, cout auto-termine, borne par le budget
        key = "c_doubt" if mode == "DOUTE" else "c_conv"
        remaining = B_total
        for p in probs:
            want = p[key]
            a = min(want, remaining)
            alloc.append(a)
            remaining -= a
            # resolu seulement si le probleme a recu tout son cout auto-termine
            got_full = a >= want
            solved += int(got_full and decision_at(p, a) == p["dstar"])
    return solved, alloc

def main():
    CSV.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    all_rows = []
    # 1) simuler tous les flux, recolter les couts auto-termines
    seed_data = []
    tot_doubt = tot_conv = tot_oracle = 0.0
    nprob = 0
    for seed in SEEDS:
        K, probs = run_seed(seed)
        seed_data.append((K, probs))
        for p in probs:
            tot_doubt += p["c_doubt"]; tot_conv += p["c_conv"]; tot_oracle += p["oracle"]; nprob += 1
    mean_doubt = tot_doubt / nprob
    mean_conv = tot_conv / nprob
    mean_oracle = tot_oracle / nprob
    print(f"Couts auto-termines moyens (pas/probleme) : "
          f"DOUTE={mean_doubt:.0f}  CONV={mean_conv:.0f}  ORACLE={mean_oracle:.0f}")

    # Diagnostic par famille : le doute SATURE-t-il (sigma jamais retombe -> cout=MAX) sur
    # certaines familles ? (famine de budget = pourquoi le doute sur-coute sans mieux resoudre)
    print(f"\n{'famille':<14}{'c_doubt moy':>12}{'% satures':>11}{'c_conv moy':>12}")
    print("-" * 49)
    for fam in ["EVIDENCE", "CONTRADICTION", "TOPOLOGIE"]:
        cds = [p["c_doubt"] for _, probs in seed_data for p in probs if p["fam"] == fam]
        ccs = [p["c_conv"] for _, probs in seed_data for p in probs if p["fam"] == fam]
        sat = 100.0 * np.mean([c >= MAX_BUDGET for c in cds])
        print(f"{fam:<14}{np.mean(cds):>12.0f}{sat:>10.0f}%{np.mean(ccs):>12.0f}")

    # 2) budget total = SCARCITE : moins que ce que le doute voudrait (force l'arbitrage)
    #    on balaie plusieurs niveaux de budget pour voir la courbe budget->reussite.
    K = seed_data[0][0]
    budget_levels = [0.5, 0.75, 1.0, 1.25]   # x (K * mean_oracle)
    print(f"\n{'budget(xKoracle)':>16}{'DOUTE':>9}{'UNIFORME':>10}{'CONV':>8}  (frac. resolue, moy. seeds)")
    print("-" * 55)
    curve = {m: [] for m in ["DOUTE", "UNIFORME", "CONVERGENCE"]}
    for lvl in budget_levels:
        B_total = lvl * K * mean_oracle
        accs = {m: [] for m in curve}
        for (Kx, probs) in seed_data:
            for m in curve:
                s, alloc = allocate_and_score(probs, B_total, m)
                accs[m].append(s / Kx)
                for p, a in zip(probs, alloc):
                    all_rows.append((lvl, m, p["fam"], p["label"], p["oracle"],
                                     p["c_doubt"], p["c_conv"], a, int(decision_at(p, a) == p["dstar"])))
        for m in curve:
            curve[m].append(float(np.mean(accs[m])))
        print(f"{lvl:>16.2f}{np.mean(accs['DOUTE']):>9.2f}"
              f"{np.mean(accs['UNIFORME']):>10.2f}{np.mean(accs['CONVERGENCE']):>8.2f}")

    # 3) diagnostic d'allocation : le doute donne-t-il PLUS aux durs (oracle eleve) ?
    ora = np.array([p["oracle"] for _, probs in seed_data for p in probs], dtype=float)
    cdb = np.array([p["c_doubt"] for _, probs in seed_data for p in probs], dtype=float)
    ccv = np.array([p["c_conv"] for _, probs in seed_data for p in probs], dtype=float)
    r_doubt = float(np.corrcoef(ora, cdb)[0, 1])
    r_conv = float(np.corrcoef(ora, ccv)[0, 1])
    print("\n=== VERDICT piste 2 (honnete) ===")
    print(f"Allocation vs difficulte oracle : corr(DOUTE,oracle)={r_doubt:+.2f}  "
          f"corr(CONV,oracle)={r_conv:+.2f}")
    # comparaison a budget serre (lvl=0.75)
    j = budget_levels.index(0.75)
    d075, u075, c075 = curve["DOUTE"][j], curve["UNIFORME"][j], curve["CONVERGENCE"][j]
    print(f"A budget serre (0.75x) : DOUTE={d075:.2f}  UNIFORME={u075:.2f}  CONVERGENCE={c075:.2f}")
    if d075 > u075 * 1.1 and d075 >= c075 - 0.02:
        print("  -> UTILE : le doute alloue mieux que l'uniforme ET tient tete au controle convergence.")
        print("     'Explorer tant que le doute persiste' = donner le compute ou le desaccord dure.")
    elif d075 > u075 * 1.1:
        print("  -> PARTIEL : le doute bat l'uniforme mais pas le controle convergence")
        print(f"     (DOUTE={d075:.2f} vs CONV={c075:.2f}) : l'allocation par doute ~ arret sur v stable.")
    else:
        print("  -> NON CONCLUANT : le doute n'ameliore pas l'allocation face a l'uniforme.")

    with CSV.open("w", encoding="utf-8") as f:
        f.write("budget_lvl,mode,family,label,oracle,c_doubt,c_conv,alloc,correct\n")
        for r in all_rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print(f"\n[csv] {CSV}")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
        colors = {"DOUTE": "#d62728", "UNIFORME": "#7f7f7f", "CONVERGENCE": "#1f77b4"}
        for m in curve:
            axes[0].plot(budget_levels, curve[m], marker="o", color=colors[m], label=m)
        axes[0].set_xlabel("Budget total (x K*oracle)"); axes[0].set_ylabel("Fraction resolue")
        axes[0].set_title("Reussite vs budget total"); axes[0].grid(alpha=0.3); axes[0].legend()
        axes[1].scatter(ora, cdb, s=18, c="#d62728", label=f"DOUTE (r={r_doubt:+.2f})", alpha=0.7)
        axes[1].scatter(ora, ccv, s=18, c="#1f77b4", label=f"CONV (r={r_conv:+.2f})", alpha=0.5, marker="x")
        lim = max(ora.max(), cdb.max(), ccv.max())
        axes[1].plot([0, lim], [0, lim], ls=":", c="k", alpha=0.4)
        axes[1].set_xlabel("Difficulte oracle (pas)"); axes[1].set_ylabel("Compute alloue (pas)")
        axes[1].set_title("Le doute donne-t-il plus aux durs ?"); axes[1].grid(alpha=0.3); axes[1].legend()
        fig.suptitle("Piste 2 : le doute comme allocateur de compute sur un flux (6 seeds, K=12)", fontsize=11)
        plt.tight_layout(); plt.savefig(PNG, dpi=140)
        print(f"[png] {PNG}")
    except Exception as e:
        print(f"[png] skipped: {e}")
    print(f"\nWall time: {time.time()-t0:.1f}s")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
