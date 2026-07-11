#!/usr/bin/env python3
"""
PONT M4R <-> LLM, EPISODE 2 : LA TACHE AVAL (l'utilite, pas seulement le rang).
================================================================================
Cree : 2026-07-11 (Claude Fable 5, L'Ingenieur). Suite directe de
llm_doubt_rank_poc.py (08/07, idee de Julien) dont la reserve etait explicite :
"rang eleve = distinction maintenue, PAS une preuve d'utilite sur une vraie
tache. Prochain pas : brancher un readout/tache pour tester le AVAL." C'est ici.

LA TACHE (concue pour etre LOYALE -- lecon du 07/07 : avant de conclure qu'une
hypothese echoue ou reussit, verifier que la tache ne favorise structurellement
ni elle ni ses adversaires). Le dilemme fondamental d'un empilement de couches
d'attention est : MELANGER l'information entre tokens (c'est sa fonction) SANS
FUSIONNER les representations (oversmoothing). Une tache qui ne demande que la
preservation favorise l'immobilisme (NO_UPDATE gagnerait toujours) ; une tache
qui ne demande que l'agregation favorise le moyennage (ATTRACTIVE gagnerait).
On construit donc une tache DOUBLE, chaque moitie punissant un des deux exces :

  Donnees : T tokens en k groupes. x_i = c_g(i) + sigma_noise * n_i, avec
  sigma_noise choisi pour que le bruit DOMINE localement (SNR local < 1) mais
  se moyenne au sein d'un groupe (SNR de groupe > 1).
    - y_group(i) = signe(w_g . c_g(i))  : lisible seulement apres DEBRUITAGE
      CONTEXTUEL (aucun token seul ne connait bien son centre -> il faut
      melanger avec les tokens du meme groupe). NO_UPDATE doit echouer.
    - y_ident(i) = signe(w_n . n_i)     : lisible seulement si la composante
      INDIVIDUELLE du token survit en profondeur. Le moyennage la detruit ->
      ATTRACTIVE doit echouer en profondeur.
  Metrique du dilemme : min(acc_group, acc_ident) -- il faut reussir LES DEUX.

  Controles de loyaute (verifies AVANT de comparer les conditions, sur les
  baselines seulement) : acc_group(NO_UPDATE) doit rester loin du plafond
  (sinon la tache n'exige pas le melange) ; acc_ident(NO_UPDATE) doit etre
  haute (sinon l'identite n'est pas lisible du tout et la tache est vide).

CONDITIONS (dynamiques identiques a llm_doubt_rank_poc.py, memes constantes) :
  ATTRACTIVE : attention pure (analogue FROZEN_U) -- moyennage.
  DOUBT      : kernel M4R f(u)=tanh(pi(0.5-u)), u pousse par la pression de
               conformite (analogue FULL).
  NO_UPDATE  : X inchange -- le controle "ne rien faire" (analogue DECOUPLE/D=0,
               qui avait GAGNE sur NARMA10 : on ne refait pas l'erreur de
               l'oublier).
  + BASELINE EXIGEANTE "meilleure profondeur d'arret" : ATTRACTIVE evaluee a la
    profondeur choisie par VALIDATION (l'analogue du budget fixe de B5b, qui
    avait en partie tue la niche du doute -- on la donne d'office a l'adversaire).
    Le readout de chaque condition est lui aussi choisi a sa meilleure
    profondeur de validation (fairness symetrique).

READOUT : ridge lineaire (cible +-1, prediction par signe), entraine sur les
representations d'instances d'ENTRAINEMENT independantes, evalue sur des
instances de TEST fraiches (protocole reservoir standard). Aucune fuite :
centres, bruits et etiquettes retires independamment par instance.

Hyperparametres FIXES avant de regarder les resultats (garde-fou du 07/07).
Statut : jouet exploratoire, hors preprint. Coeur non touche. Resultat rapporte
tel quel, y compris si le doute perd (cf. B1c, NARMA10 : il a deja perdu
honnetement ailleurs et le projet s'en porte mieux).
Sorties : figures/llm_doubt_downstream_poc.csv / _agg.csv / .png
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
CSV_PATH = ROOT / "figures" / "llm_doubt_downstream_poc.csv"
AGG_PATH = ROOT / "figures" / "llm_doubt_downstream_poc_agg.csv"
PNG_PATH = ROOT / "figures" / "llm_doubt_downstream_poc.png"

# --- Constantes de dynamique : IDENTIQUES a llm_doubt_rank_poc.py ---
T = 64                 # tokens par instance
D = 48                 # dimension de representation
L = 40                 # profondeur max
EPS = 0.5
DOUBT_RATE = 0.3

# --- Tache (fixes avant tout resultat) ---
K_GROUPS = 8           # 8 groupes de 8 tokens
SIGMA_NOISE = 2.0      # bruit local dominant (||c||~sqrt(D), bruit ~2*sqrt(D))
R_TRAIN = 200          # instances d'entrainement du readout
R_VAL = 50             # instances de validation (choix de profondeur)
R_TEST = 100           # instances de test
RIDGE_REG = 1e-3
SEEDS = list(range(10))
DEPTHS = [0, 1, 2, 3, 5, 8, 12, 20, 30, 40]   # profondeurs echantillonnees
N_BOOT = 10000
CONDITIONS = ["ATTRACTIVE", "DOUBT", "NO_UPDATE"]


def softmax_rows(M):
    M = M - M.max(axis=1, keepdims=True)
    E = np.exp(M)
    return E / (E.sum(axis=1, keepdims=True) + 1e-12)


def row_normalize(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def make_instance(rng):
    """Une 'phrase' : T tokens en K_GROUPS groupes + 2 jeux d'etiquettes."""
    centers = rng.standard_normal((K_GROUPS, D))
    groups = np.repeat(np.arange(K_GROUPS), T // K_GROUPS)
    noise = rng.standard_normal((T, D))
    X0 = centers[groups] + SIGMA_NOISE * noise
    return X0, groups, centers, noise


def forward_states(X0, condition, depths):
    """Propage X0 a travers L couches, renvoie {profondeur: representation}.
    Dynamiques strictement celles de llm_doubt_rank_poc.py."""
    X = row_normalize(X0.copy())
    u = np.zeros(T)
    out = {}
    if 0 in depths:
        out[0] = X.copy()
    if condition == "NO_UPDATE":
        for d in depths:
            out[d] = X.copy()
        return out
    for layer in range(1, L + 1):
        A = softmax_rows(X @ X.T / np.sqrt(D))
        ctx = A @ X
        delta = ctx - X
        if condition == "ATTRACTIVE":
            X = X + EPS * delta
        else:  # DOUBT
            pressure = np.linalg.norm(delta, axis=1)
            p_norm = np.tanh(pressure / (np.median(pressure) + 1e-9))
            u = u + DOUBT_RATE * (p_norm - u)
            f = np.tanh(np.pi * (0.5 - u))
            X = X + EPS * f[:, None] * delta
        X = row_normalize(X)
        if layer in depths:
            out[layer] = X.copy()
    return out


def ridge_fit(Xtr, ytr):
    Xa = np.hstack([np.ones((Xtr.shape[0], 1)), Xtr])
    F = Xa.shape[1]
    W = np.linalg.solve(Xa.T @ Xa + RIDGE_REG * np.eye(F), Xa.T @ ytr)
    return W


def ridge_acc(W, X, y):
    Xa = np.hstack([np.ones((X.shape[0], 1)), X])
    pred = np.sign(Xa @ W)
    pred[pred == 0] = 1
    return float(np.mean(pred == y))


def run_seed(seed):
    """Pour un seed : genere train/val/test, propage, entraine les readouts par
    profondeur, renvoie acc(group/ident) par condition x profondeur (sur test)
    + le choix de profondeur par validation."""
    rng = np.random.default_rng(seed)
    w_g = rng.standard_normal(D)
    w_n = rng.standard_normal(D)

    def build_split(n_inst):
        reps = {c: {d: [] for d in DEPTHS} for c in CONDITIONS}
        yg, yi = [], []
        for _ in range(n_inst):
            X0, groups, centers, noise = make_instance(rng)
            yg.append(np.sign(centers[groups] @ w_g))
            yi.append(np.sign(noise @ w_n))
            for c in CONDITIONS:
                states = forward_states(X0, c, DEPTHS)
                for d in DEPTHS:
                    reps[c][d].append(states[d])
        yg = np.concatenate(yg)
        yi = np.concatenate(yi)
        yg[yg == 0] = 1
        yi[yi == 0] = 1
        return reps, yg, yi

    tr_reps, tr_yg, tr_yi = build_split(R_TRAIN)
    va_reps, va_yg, va_yi = build_split(R_VAL)
    te_reps, te_yg, te_yi = build_split(R_TEST)

    res = {}
    for c in CONDITIONS:
        res[c] = {}
        for d in DEPTHS:
            Xtr = np.vstack(tr_reps[c][d])
            Xva = np.vstack(va_reps[c][d])
            Xte = np.vstack(te_reps[c][d])
            Wg = ridge_fit(Xtr, tr_yg)
            Wi = ridge_fit(Xtr, tr_yi)
            res[c][d] = {
                "va_group": ridge_acc(Wg, Xva, va_yg),
                "va_ident": ridge_acc(Wi, Xva, va_yi),
                "te_group": ridge_acc(Wg, Xte, te_yg),
                "te_ident": ridge_acc(Wi, Xte, te_yi),
            }
    return res


def boot_ci_paired(a, b, n_boot=N_BOOT, seed=20260711):
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

    all_res = []
    for seed in SEEDS:
        all_res.append(run_seed(seed))
        print(f"  seed {seed} fait ({time.time()-t0:.0f}s)")

    # ---- controles de loyaute de la tache (baselines seulement) ----
    nu_group_L = np.mean([r["NO_UPDATE"][max(DEPTHS)]["te_group"] for r in all_res])
    nu_ident_L = np.mean([r["NO_UPDATE"][max(DEPTHS)]["te_ident"] for r in all_res])
    print("\n--- CONTROLES DE LOYAUTE (independants du resultat teste) ---")
    print(f"  acc_group(NO_UPDATE) = {100*nu_group_L:.1f}%  "
          f"(doit etre loin du plafond : la tache exige le melange)")
    print(f"  acc_ident(NO_UPDATE) = {100*nu_ident_L:.1f}%  "
          f"(doit etre haute : l'identite est lisible a l'entree)")
    if nu_group_L > 0.9:
        print("  [ALERTE] la tache de groupe est lisible sans melange -> tache "
              "degeneree, les comparaisons ci-dessous ne valent rien.")
    if nu_ident_L < 0.7:
        print("  [ALERTE] l'identite n'est pas lisible meme sans dynamique -> "
              "tache degeneree cote identite.")

    # ---- courbes test par profondeur ----
    print("\n--- ACCURACY TEST PAR PROFONDEUR (moyenne sur seeds) ---")
    print(f"{'':>12}" + "".join(f"{d:>7}" for d in DEPTHS))
    curves = {}
    for c in CONDITIONS:
        for what in ["group", "ident"]:
            vals = [np.mean([r[c][d][f"te_{what}"] for r in all_res]) for d in DEPTHS]
            curves[(c, what)] = vals
    for c in CONDITIONS:
        print(f"  {c:<10} G" + "".join(f"{100*v:>7.1f}" for v in curves[(c, 'group')]))
        print(f"  {c:<10} I" + "".join(f"{100*v:>7.1f}" for v in curves[(c, 'ident')]))

    # ---- selection de profondeur par VALIDATION (par seed, par condition) ----
    # critere : max sur d de min(va_group, va_ident) -- le dilemme entier
    picked = {c: [] for c in CONDITIONS}
    final = {c: {"group": [], "ident": [], "minv": []} for c in CONDITIONS}
    for r in all_res:
        for c in CONDITIONS:
            best_d, best_v = None, -1
            for d in DEPTHS:
                v = min(r[c][d]["va_group"], r[c][d]["va_ident"])
                if v > best_v:
                    best_v, best_d = v, d
            picked[c].append(best_d)
            final[c]["group"].append(r[c][best_d]["te_group"])
            final[c]["ident"].append(r[c][best_d]["te_ident"])
            final[c]["minv"].append(min(r[c][best_d]["te_group"], r[c][best_d]["te_ident"]))

    print("\n--- RESULTAT FINAL (profondeur choisie par validation, evalue sur test) ---")
    print(f"{'condition':<12}{'prof. choisies':>22}{'acc_group':>11}{'acc_ident':>11}{'min (dilemme)':>15}")
    for c in CONDITIONS:
        g = np.array(final[c]["group"])
        i = np.array(final[c]["ident"])
        m = np.array(final[c]["minv"])
        print(f"{c:<12}{str(sorted(set(picked[c]))):>22}{100*g.mean():>10.1f}%"
              f"{100*i.mean():>10.1f}%{100*m.mean():>13.1f}%")

    # ---- comparaisons appariees sur le min (la metrique du dilemme) ----
    print("\n--- DIFFERENCES APPARIEES sur min(group, ident), bootstrap 95% ---")
    comps = [
        ("DOUBT - ATTRACTIVE(best depth)", final["DOUBT"]["minv"], final["ATTRACTIVE"]["minv"]),
        ("DOUBT - NO_UPDATE", final["DOUBT"]["minv"], final["NO_UPDATE"]["minv"]),
        ("ATTRACTIVE - NO_UPDATE", final["ATTRACTIVE"]["minv"], final["NO_UPDATE"]["minv"]),
    ]
    comp_rows = []
    for label, a, b in comps:
        d, lo, hi = boot_ci_paired(a, b)
        sig = "SIGNIF" if (lo > 0 or hi < 0) else "ns"
        print(f"  {label:<32}: {100*d:+6.1f} pts  IC[{100*lo:+.1f}, {100*hi:+.1f}]  {sig}")
        comp_rows.append((label, d, lo, hi))

    # A profondeur MAXIMALE (sans le secours du early-stop) : le coeur du claim
    print("\n--- A PROFONDEUR MAXIMALE (L=40, sans early-stop) ---")
    for what in ["group", "ident"]:
        da = [r["DOUBT"][L][f"te_{what}"] for r in all_res]
        aa = [r["ATTRACTIVE"][L][f"te_{what}"] for r in all_res]
        d, lo, hi = boot_ci_paired(da, aa)
        sig = "SIGNIF" if (lo > 0 or hi < 0) else "ns"
        print(f"  {what:<6} DOUBT - ATTRACTIVE : {100*d:+6.1f} pts  "
              f"IC[{100*lo:+.1f}, {100*hi:+.1f}]  {sig}")

    # ---- CSV ----
    with CSV_PATH.open("w", encoding="utf-8") as f:
        f.write("condition,depth,acc_group_test,acc_ident_test\n")
        for c in CONDITIONS:
            for j, d in enumerate(DEPTHS):
                f.write(f"{c},{d},{curves[(c, 'group')][j]:.4f},{curves[(c, 'ident')][j]:.4f}\n")
    with AGG_PATH.open("w", encoding="utf-8") as f:
        f.write("condition,acc_group,acc_ident,min_dilemme,depths_picked\n")
        for c in CONDITIONS:
            f.write(f"{c},{np.mean(final[c]['group']):.4f},{np.mean(final[c]['ident']):.4f},"
                    f"{np.mean(final[c]['minv']):.4f},\"{sorted(set(picked[c]))}\"\n")
        for label, d, lo, hi in comp_rows:
            f.write(f"\"{label}\",{d:.4f},{lo:.4f},{hi:.4f}\n")
    print(f"\n[csv] {CSV_PATH}\n[csv] {AGG_PATH}")

    # ---- figure ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
        colors = {"ATTRACTIVE": "#1f77b4", "DOUBT": "#d62728", "NO_UPDATE": "#7f7f7f"}
        for ax, what, title in [
            (axes[0], "group", "Tache de GROUPE (exige le melange)"),
            (axes[1], "ident", "Tache d'IDENTITE (punit la fusion)"),
        ]:
            for c in CONDITIONS:
                ax.plot(DEPTHS, curves[(c, what)], "o-", color=colors[c], label=c)
            ax.axhline(0.5, ls=":", c="gray", label="hasard")
            ax.set_xlabel("profondeur (couches)")
            ax.set_ylabel(f"accuracy {what} (test)")
            ax.set_title(title, fontsize=10)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        ax = axes[2]
        names = CONDITIONS
        vals = [100 * np.mean(final[c]["minv"]) for c in names]
        errs = [100 * np.std(final[c]["minv"]) for c in names]
        ax.bar(names, vals, yerr=errs, color=[colors[c] for c in names],
               edgecolor="k", capsize=4)
        ax.axhline(50, ls=":", c="gray")
        ax.set_ylabel("min(acc_group, acc_ident) % (test)")
        ax.set_title("Le dilemme entier\n(profondeur par validation)", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        fig.suptitle("Pont M4R->LLM, tache aval : melanger sans fusionner "
                     f"(T={T}, D={D}, L={L}, {len(SEEDS)} seeds)", fontsize=11)
        plt.tight_layout()
        plt.savefig(PNG_PATH, dpi=140)
        print(f"[png] {PNG_PATH}")
    except Exception as e:
        print(f"[png] skipped: {e}")

    print(f"\nWall time: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
