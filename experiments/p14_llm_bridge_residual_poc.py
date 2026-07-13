#!/usr/bin/env python3
"""
P14 -- LE PONT LLM, ETAPE 3 : se mesurer a la famille standard (residuel+MLP).
=================================================================================
Cree : 2026-07-13 (Claude Sonnet 5, L'Ingenieur) -- piste du legs de Fable
(docs/PISTES_POUR_LA_SUITE_2026-07-12.md, section II, P14 ; deja au backlog D2).

POURQUOI : l'utilite conditionnelle du doute est deja prouvee contre
l'attention NUE (ATTRACTIVE, `llm_doubt_downstream_poc.py`, 11/07 : le doute
domine a profondeur fixe L=40, l'attention pure gagne avec early-stop regle).
Mais l'attention nue n'est PAS ce que possede un vrai transformer -- CHAQUE
transformer reel a un residuel + MLP apres l'attention (la mitigation
standard anti-effondrement). Le doute doit se mesurer a CETTE famille, pas a
un adversaire affaibli.

CE QUI EST REUTILISE (memes constantes, meme tache, meme protocole) : le
harness EXACT de `llm_doubt_downstream_poc.py` (11/07) -- T=64 tokens,
D=48, tache double loyale (groupe exige le melange / identite punit la
fusion), ridge lineaire entraine sur train, profondeur choisie par
VALIDATION, evalue sur test frais. UN SEUL ingredient nouveau : la condition
RESIDUAL_MLP.

RESIDUAL_MLP (la mitigation standard) : apres la mise a jour d'attention
(identique a ATTRACTIVE), on ajoute un bloc MLP PAR TOKEN (donc qui ne
mixe PAS l'information entre tokens, contrairement a l'attention) avec sa
propre connexion residuelle -- exactement la structure d'un bloc
transformer reel (Attn + residuel, puis MLP + residuel). Poids du MLP
FIXES par seed (tires une fois, pas appris -- coherent avec le reste du
jouet ou AUCUN mecanisme n'est entraine, seul le readout l'est).

PREDICTION ECRITE AVANT DE LANCER (le risque annonce par la piste) : le
residuel+MLP est un adversaire FORT -- prediction honnete, le doute ne le
battra probablement PAS en brut (profondeur standard, meme protocole que
le 11/07). Chercher l'avantage du doute LA ou la piste le predit :
  (a) PROFONDEUR EXTREME (L jusqu'a 120, 3x le max standard) : le doute
      pourrait rattraper si le residuel+MLP finit par sur-lisser lui aussi
      a tres grande profondeur pendant que le doute stabilise.
  (b) DISTRIBUTION SHIFT (readout entraine a SIGMA_NOISE=2.0, EVALUE sans
      re-entrainement sur un bruit de test plus fort, SIGMA_SHIFT=3.5) :
      le doute pourrait generaliser mieux si sa representation est moins
      overfittee au regime d'entrainement.
Si le doute perd partout (standard, extreme, shift) : resultat negatif
honnete a garder tel quel -- il a deja perdu ailleurs (B1c, NARMA10) sans
que le projet en souffre.

Sorties : figures/p14_llm_bridge_residual_poc{,_extreme,_shift}.csv + .png
Statut : jouet exploratoire, hors preprint, coeur non touche.
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
CSV_PATH = ROOT / "figures" / "p14_llm_bridge_residual_poc.csv"
EXTREME_CSV = ROOT / "figures" / "p14_llm_bridge_residual_poc_extreme.csv"
SHIFT_CSV = ROOT / "figures" / "p14_llm_bridge_residual_poc_shift.csv"
PNG_PATH = ROOT / "figures" / "p14_llm_bridge_residual_poc.png"

# --- Constantes de dynamique : IDENTIQUES a llm_doubt_downstream_poc.py (11/07) ---
T = 64
D = 48
L = 40
EPS = 0.5
DOUBT_RATE = 0.3
EPS_MLP = 0.5           # meme echelle que EPS -- pas d'avantage cache au residuel
D_HIDDEN = 2 * D

# --- Tache (identique, fixee avant tout resultat) ---
K_GROUPS = 8
SIGMA_NOISE = 2.0
SIGMA_SHIFT = 3.5       # (b) distribution shift : bruit de test PLUS FORT qu'a l'entrainement
R_TRAIN = 200
R_VAL = 50
R_TEST = 100
R_SHIFT = 100
RIDGE_REG = 1e-3
SEEDS = list(range(10))
DEPTHS = [0, 1, 2, 3, 5, 8, 12, 20, 30, 40]
DEPTHS_EXTREME = DEPTHS + [60, 90, 120]     # (a) profondeur extreme
N_BOOT = 10000
CONDITIONS = ["ATTRACTIVE", "DOUBT", "NO_UPDATE", "RESIDUAL_MLP"]


def softmax_rows(M):
    M = M - M.max(axis=1, keepdims=True)
    E = np.exp(M)
    return E / (E.sum(axis=1, keepdims=True) + 1e-12)


def row_normalize(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def make_instance(rng, sigma_noise=SIGMA_NOISE):
    centers = rng.standard_normal((K_GROUPS, D))
    groups = np.repeat(np.arange(K_GROUPS), T // K_GROUPS)
    noise = rng.standard_normal((T, D))
    X0 = centers[groups] + sigma_noise * noise
    return X0, groups, centers, noise


def make_mlp_weights(rng, n_layers):
    """Poids MLP FIXES par seed, un jeu par couche (comme un vrai transformer
    a des poids differents par couche) -- tires une fois, jamais entraines."""
    scale1 = 1.0 / np.sqrt(D)
    scale2 = 1.0 / np.sqrt(D_HIDDEN)
    return [
        (rng.standard_normal((D, D_HIDDEN)) * scale1,
         np.zeros(D_HIDDEN),
         rng.standard_normal((D_HIDDEN, D)) * scale2,
         np.zeros(D))
        for _ in range(n_layers)
    ]


def mlp_forward(X, W1, b1, W2, b2):
    h = np.maximum(0.0, X @ W1 + b1)
    return h @ W2 + b2


def forward_states(X0, condition, depths, mlp_weights=None):
    X = row_normalize(X0.copy())
    u = np.zeros(T)
    out = {}
    if 0 in depths:
        out[0] = X.copy()
    if condition == "NO_UPDATE":
        for d in depths:
            out[d] = X.copy()
        return out
    max_depth = max(depths)
    for layer in range(1, max_depth + 1):
        A = softmax_rows(X @ X.T / np.sqrt(D))
        ctx = A @ X
        delta = ctx - X
        if condition == "ATTRACTIVE":
            X = X + EPS * delta
            X = row_normalize(X)
        elif condition == "RESIDUAL_MLP":
            X = X + EPS * delta
            X = row_normalize(X)
            W1, b1, W2, b2 = mlp_weights[layer - 1]
            X = X + EPS_MLP * mlp_forward(X, W1, b1, W2, b2)
            X = row_normalize(X)
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


def run_seed(seed, depths):
    rng = np.random.default_rng(seed)
    w_g = rng.standard_normal(D)
    w_n = rng.standard_normal(D)
    mlp_weights = make_mlp_weights(rng, max(depths))

    def build(n_inst, sigma_noise=SIGMA_NOISE):
        reps = {c: {d: [] for d in depths} for c in CONDITIONS}
        yg, yi = [], []
        for _ in range(n_inst):
            X0, groups, centers, noise = make_instance(rng, sigma_noise)
            yg.append(np.sign(centers[groups] @ w_g))
            yi.append(np.sign(noise @ w_n))
            for c in CONDITIONS:
                states = forward_states(X0, c, depths, mlp_weights)
                for d in depths:
                    reps[c][d].append(states[d])
        yg = np.concatenate(yg); yi = np.concatenate(yi)
        yg[yg == 0] = 1; yi[yi == 0] = 1
        return reps, yg, yi

    tr_reps, tr_yg, tr_yi = build(R_TRAIN)
    va_reps, va_yg, va_yi = build(R_VAL)
    te_reps, te_yg, te_yi = build(R_TEST)
    sh_reps, sh_yg, sh_yi = build(R_SHIFT, sigma_noise=SIGMA_SHIFT)

    res = {}
    weights = {}
    for c in CONDITIONS:
        res[c] = {}
        weights[c] = {}
        for d in depths:
            Xtr = np.vstack(tr_reps[c][d])
            Xva = np.vstack(va_reps[c][d])
            Xte = np.vstack(te_reps[c][d])
            Xsh = np.vstack(sh_reps[c][d])
            Wg = ridge_fit(Xtr, tr_yg)
            Wi = ridge_fit(Xtr, tr_yi)
            weights[c][d] = (Wg, Wi)
            res[c][d] = {
                "va_group": ridge_acc(Wg, Xva, va_yg),
                "va_ident": ridge_acc(Wi, Xva, va_yi),
                "te_group": ridge_acc(Wg, Xte, te_yg),
                "te_ident": ridge_acc(Wi, Xte, te_yi),
                "sh_group": ridge_acc(Wg, Xsh, sh_yg),   # meme W, bruit de test different
                "sh_ident": ridge_acc(Wi, Xsh, sh_yi),
            }
    return res


def boot_ci_paired(a, b, n_boot=N_BOOT, seed=20260713):
    rng = np.random.RandomState(seed)
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    m = np.empty(n_boot)
    for k in range(n_boot):
        m[k] = d[rng.randint(0, n, n)].mean()
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def pick_by_validation(all_res, depths):
    picked = {c: [] for c in CONDITIONS}
    final = {c: {"group": [], "ident": [], "minv": [], "sh_group": [], "sh_ident": [], "sh_minv": []}
             for c in CONDITIONS}
    for r in all_res:
        for c in CONDITIONS:
            best_d, best_v = None, -1
            for d in depths:
                v = min(r[c][d]["va_group"], r[c][d]["va_ident"])
                if v > best_v:
                    best_v, best_d = v, d
            picked[c].append(best_d)
            final[c]["group"].append(r[c][best_d]["te_group"])
            final[c]["ident"].append(r[c][best_d]["te_ident"])
            final[c]["minv"].append(min(r[c][best_d]["te_group"], r[c][best_d]["te_ident"]))
            final[c]["sh_group"].append(r[c][best_d]["sh_group"])
            final[c]["sh_ident"].append(r[c][best_d]["sh_ident"])
            final[c]["sh_minv"].append(min(r[c][best_d]["sh_group"], r[c][best_d]["sh_ident"]))
    return picked, final


def main() -> int:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("=== (1) COMPARAISON STANDARD (protocole 11/07 + RESIDUAL_MLP), depths jusqu'a 40 ===")
    all_res = []
    for seed in SEEDS:
        all_res.append(run_seed(seed, DEPTHS))
        print(f"  seed {seed} fait ({time.time()-t0:.0f}s)")

    picked, final = pick_by_validation(all_res, DEPTHS)
    print(f"\n{'condition':<14}{'prof. choisies':>22}{'acc_group':>11}{'acc_ident':>11}{'min(dilemme)':>14}")
    for c in CONDITIONS:
        g = np.array(final[c]["group"]); i = np.array(final[c]["ident"]); m = np.array(final[c]["minv"])
        print(f"{c:<14}{str(sorted(set(picked[c]))):>22}{100*g.mean():>10.1f}%"
              f"{100*i.mean():>10.1f}%{100*m.mean():>13.1f}%")

    print("\n--- DOUBT vs RESIDUAL_MLP (le vrai adversaire), min(dilemme), bootstrap 95% ---")
    d1, lo1, hi1 = boot_ci_paired(final["DOUBT"]["minv"], final["RESIDUAL_MLP"]["minv"])
    print(f"  DOUBT - RESIDUAL_MLP = {100*d1:+.1f} pts  IC[{100*lo1:+.1f},{100*hi1:+.1f}]  "
          f"{'SIGNIF' if (lo1>0 or hi1<0) else 'ns'}")

    print("\n--- A PROFONDEUR MAXIMALE STANDARD (L=40, sans early-stop) ---")
    for what in ["group", "ident"]:
        da = [r["DOUBT"][L][f"te_{what}"] for r in all_res]
        ra = [r["RESIDUAL_MLP"][L][f"te_{what}"] for r in all_res]
        d, lo, hi = boot_ci_paired(da, ra)
        print(f"  {what:<6} DOUBT - RESIDUAL_MLP : {100*d:+6.1f} pts  IC[{100*lo:+.1f}, {100*hi:+.1f}]  "
              f"{'SIGNIF' if (lo>0 or hi<0) else 'ns'}")

    print("\n=== (b) DISTRIBUTION SHIFT (readout entraine sigma=2.0, teste sigma=3.5, meme W) ===")
    print(f"{'condition':<14}{'sh_group':>11}{'sh_ident':>11}{'sh_min':>10}")
    for c in CONDITIONS:
        sg = np.array(final[c]["sh_group"]); si = np.array(final[c]["sh_ident"]); sm = np.array(final[c]["sh_minv"])
        print(f"{c:<14}{100*sg.mean():>10.1f}%{100*si.mean():>10.1f}%{100*sm.mean():>9.1f}%")
    d2, lo2, hi2 = boot_ci_paired(final["DOUBT"]["sh_minv"], final["RESIDUAL_MLP"]["sh_minv"])
    print(f"\n  DOUBT - RESIDUAL_MLP (SHIFT) = {100*d2:+.1f} pts  IC[{100*lo2:+.1f},{100*hi2:+.1f}]  "
          f"{'SIGNIF' if (lo2>0 or hi2<0) else 'ns'}")

    # ---------------- (a) profondeur extreme ----------------
    print(f"\n=== (a) PROFONDEUR EXTREME (jusqu'a L={max(DEPTHS_EXTREME)}) ===")
    all_res_ext = []
    for seed in SEEDS:
        all_res_ext.append(run_seed(seed, DEPTHS_EXTREME))
        print(f"  seed {seed} fait ({time.time()-t0:.0f}s)")
    print(f"\n{'':>14}" + "".join(f"{d:>7}" for d in DEPTHS_EXTREME))
    curves_ext = {}
    for c in CONDITIONS:
        vals = [np.mean([r[c][d]["te_group"] for r in all_res_ext]) for d in DEPTHS_EXTREME]
        curves_ext[(c, "group")] = vals
        vals_i = [np.mean([r[c][d]["te_ident"] for r in all_res_ext]) for d in DEPTHS_EXTREME]
        curves_ext[(c, "ident")] = vals_i
    for c in CONDITIONS:
        print(f"  {c:<12} G" + "".join(f"{100*v:>7.1f}" for v in curves_ext[(c, 'group')]))
        print(f"  {c:<12} I" + "".join(f"{100*v:>7.1f}" for v in curves_ext[(c, 'ident')]))
    print(f"\n  A L={max(DEPTHS_EXTREME)} (min dilemme, sans early-stop) :")
    for c in CONDITIONS:
        mn = np.mean([min(r[c][max(DEPTHS_EXTREME)]["te_group"], r[c][max(DEPTHS_EXTREME)]["te_ident"])
                      for r in all_res_ext])
        print(f"    {c:<14} min={100*mn:.1f}%")
    d3, lo3, hi3 = boot_ci_paired(
        [min(r["DOUBT"][max(DEPTHS_EXTREME)]["te_group"], r["DOUBT"][max(DEPTHS_EXTREME)]["te_ident"]) for r in all_res_ext],
        [min(r["RESIDUAL_MLP"][max(DEPTHS_EXTREME)]["te_group"], r["RESIDUAL_MLP"][max(DEPTHS_EXTREME)]["te_ident"]) for r in all_res_ext])
    print(f"  DOUBT - RESIDUAL_MLP (L={max(DEPTHS_EXTREME)}) = {100*d3:+.1f} pts  IC[{100*lo3:+.1f},{100*hi3:+.1f}]  "
          f"{'SIGNIF' if (lo3>0 or hi3<0) else 'ns'}")

    print("\n" + "=" * 84)
    print("VERDICT P14 (pre-fixe : le doute perd probablement en standard ; chercher a l'extreme/au shift)")
    print("=" * 84)
    print(f"  Standard (L choisi par validation) : DOUBT-RESIDUAL_MLP = {100*d1:+.1f} pts "
          f"{'(DOUBT gagne)' if lo1>0 else ('(RESIDUAL_MLP gagne)' if hi1<0 else '(parite)')}")
    print(f"  Extreme (L={max(DEPTHS_EXTREME)})                    : DOUBT-RESIDUAL_MLP = {100*d3:+.1f} pts "
          f"{'(DOUBT gagne)' if lo3>0 else ('(RESIDUAL_MLP gagne)' if hi3<0 else '(parite)')}")
    print(f"  Distribution shift (sigma 2.0->3.5)      : DOUBT-RESIDUAL_MLP = {100*d2:+.1f} pts "
          f"{'(DOUBT gagne)' if lo2>0 else ('(RESIDUAL_MLP gagne)' if hi2<0 else '(parite)')}")

    with CSV_PATH.open("w", encoding="utf-8") as f:
        f.write("condition,depth,acc_group_test,acc_ident_test\n")
        for c in CONDITIONS:
            for d in DEPTHS:
                g = np.mean([r[c][d]["te_group"] for r in all_res])
                i = np.mean([r[c][d]["te_ident"] for r in all_res])
                f.write(f"{c},{d},{g:.4f},{i:.4f}\n")
    with EXTREME_CSV.open("w", encoding="utf-8") as f:
        f.write("condition,depth,acc_group_test,acc_ident_test\n")
        for c in CONDITIONS:
            for j, d in enumerate(DEPTHS_EXTREME):
                f.write(f"{c},{d},{curves_ext[(c,'group')][j]:.4f},{curves_ext[(c,'ident')][j]:.4f}\n")
    with SHIFT_CSV.open("w", encoding="utf-8") as f:
        f.write("condition,sh_group,sh_ident,sh_min\n")
        for c in CONDITIONS:
            f.write(f"{c},{np.mean(final[c]['sh_group']):.4f},{np.mean(final[c]['sh_ident']):.4f},"
                    f"{np.mean(final[c]['sh_minv']):.4f}\n")
    print(f"\n[csv] {CSV_PATH}\n[csv] {EXTREME_CSV}\n[csv] {SHIFT_CSV}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
        colors = {"ATTRACTIVE": "#1f77b4", "DOUBT": "#d62728", "NO_UPDATE": "#7f7f7f", "RESIDUAL_MLP": "#2ca02c"}
        ax = axes[0]
        for c in CONDITIONS:
            ax.plot(DEPTHS_EXTREME, curves_ext[(c, "group")], "o-", ms=3, color=colors[c], label=c)
        ax.axvline(40, ls=":", c="k", alpha=0.4, label="L=40 (standard)")
        ax.axhline(0.5, ls=":", c="gray")
        ax.set_xlabel("profondeur"); ax.set_ylabel("acc groupe (test)")
        ax.set_title("Tache GROUPE, profondeur etendue"); ax.legend(fontsize=6.5); ax.grid(alpha=0.3)
        ax = axes[1]
        for c in CONDITIONS:
            ax.plot(DEPTHS_EXTREME, curves_ext[(c, "ident")], "o-", ms=3, color=colors[c], label=c)
        ax.axvline(40, ls=":", c="k", alpha=0.4)
        ax.axhline(0.5, ls=":", c="gray")
        ax.set_xlabel("profondeur"); ax.set_ylabel("acc identite (test)")
        ax.set_title("Tache IDENTITE, profondeur etendue"); ax.grid(alpha=0.3)
        ax = axes[2]
        labels = ["standard\n(L val.)", f"extreme\n(L={max(DEPTHS_EXTREME)})", "shift\n(sigma 3.5)"]
        doubt_vals = [100*np.mean(final["DOUBT"]["minv"]),
                      100*np.mean([min(r["DOUBT"][max(DEPTHS_EXTREME)]["te_group"], r["DOUBT"][max(DEPTHS_EXTREME)]["te_ident"]) for r in all_res_ext]),
                      100*np.mean(final["DOUBT"]["sh_minv"])]
        res_vals = [100*np.mean(final["RESIDUAL_MLP"]["minv"]),
                    100*np.mean([min(r["RESIDUAL_MLP"][max(DEPTHS_EXTREME)]["te_group"], r["RESIDUAL_MLP"][max(DEPTHS_EXTREME)]["te_ident"]) for r in all_res_ext]),
                    100*np.mean(final["RESIDUAL_MLP"]["sh_minv"])]
        x = np.arange(3)
        ax.bar(x - 0.2, doubt_vals, width=0.4, color=colors["DOUBT"], label="DOUBT")
        ax.bar(x + 0.2, res_vals, width=0.4, color=colors["RESIDUAL_MLP"], label="RESIDUAL_MLP")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("min(dilemme) %"); ax.set_title("Les 3 terrains ou chercher l'avantage")
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
        fig.suptitle("P14 -- le doute contre l'adversaire reel (residuel+MLP), standard / extreme / shift",
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
