"""
genesis_five_states_readout.py -- Le plateau 73.9% est-il lisible LOCALEMENT ?
===============================================================================
Auteur : Claude Fable 5 (L'Ingenieur) -- 11 juillet 2026. Suite immediate de
genesis_five_states_poc.py (meme jour, commit cb36b4a) qui a etabli :
  - le couplage par INTERFERENCE (moyenne complexe) preserve l'information de
    parite sur un plateau stable 73.9% (t=20..150) ;
  - lue GLOBALEMENT (signe du cos de la somme des phases dominantes), elle bat
    le vote (+5.5 pts, p<1e-4) ; la lecture locale FIXE du 10/07 la perdait.
QUESTION SUIVANTE (notee comme prochaine marche) : cette information est-elle
accessible a un readout APPRIS sur des features LOCALES (par noeud), sans le
prior global "produit des phases = parite" ? Trois niveaux d'acces compares :
  L1 STRICT-LOCAL : 10 features par noeud (|psi_k|^2 x4 ; cos/sin des phases
     relatives internes phi_k - phi_0, k=1..3) -- AUCUNE relation inter-noeuds.
  L2 PAIRES : L1 + cos/sin des differences de phases dominantes entre les 10
     paires de noeuds -- l'acces aux RELATIONS, sans le produit d'ordre 5.
  (reference haute) R2 GLOBAL : le readout analytique du POC (prior parite).
  (references basses) vote majoritaire ; R1 local fixe du 10/07.
Readout : ridge (cible +-1, signe), train 700 essais / test 300, aucune fuite.
PREDICTION ecrite avant de lancer (honnetete) : L1 devrait rester pres du
hasard (la parite n'est pas dans les marginales locales) ; L2 peut monter
partiellement ; si L1 ou L2 ATTEINT R2 (~74%), c'est une vraie decouverte
(l'interference aurait LOCALISE l'information, pas seulement preservee).

REPLICATION SANS RE-SIMULER LES 4 CONDITIONS : dans run_trial du POC, seuls
s / bits / shuffle / les 5 init consomment le rng AVANT la boucle ; dans la
boucle, seul step_mixture (condition melange) tire des nombres. Les etats de
la condition INTERFERENCE ne dependent d'aucun tirage de boucle -> on peut
re-simuler int seul, a l'identique bit a bit, en consommant la meme sequence
rng d'initialisation. GATE : les accuracies exactes du run committe doivent
etre reproduites (R1_int=525/1000, R2_int=739/1000, vote=684/1000).

Statut : jouet exploratoire, hors preprint. Sorties :
figures/genesis_five_states_readout{.csv,.png}
"""

import csv
import os
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import genesis_five_states_poc as g  # noqa: E402  (mecanismes du POC, inchanges)

N = g.N
T_STEPS = g.T_STEPS
ETA = g.ETA
N_TRIALS = 1000
N_TRAIN = 700          # essais d'entrainement du readout (test = 300)
RIDGE_REG = 1e-3

GATE_EXPECTED = {"k_local_int": 525, "k_global_int": 739, "k_vote": 684}

FIG_DIR = g.FIG_DIR
BASE = os.path.join(FIG_DIR, "genesis_five_states_readout")


def run_trial_int_only(seed):
    """Rejoue UN essai du POC en ne simulant que la condition INTERFERENCE.
    Sequence rng d'init strictement identique (s, bits, shuffle, 5 inits) ;
    la boucle int/hop ne tire aucun nombre -> etats identiques au POC."""
    rng = np.random.default_rng(seed)
    s = rng.choice([-1, 1])
    b = rng.choice([-1, 1], size=N - 1).astype(float)
    b_last = s * np.prod(b)
    bits = np.concatenate([b, [b_last]])
    rng.shuffle(bits)

    A = np.ones((N, N)) - np.eye(N)
    psi = np.array([g.init_from_bit(bi, rng) for bi in bits])

    for _ in range(T_STEPS):
        psi = g.step_interference(psi, A, ETA)
        psi = g.hop_along_axis(psi, A)

    vote_sum = np.sum(bits)
    guess_vote = 1 if vote_sum > 0 else (-1 if vote_sum < 0 else int(rng.choice([-1, 1])))
    return psi, s, guess_vote


def features_local(psi):
    """L1 -- 10 features par noeud, aucune relation inter-noeuds."""
    feats = []
    for i in range(N):
        p = np.abs(psi[i]) ** 2
        phi = np.angle(psi[i])
        rel = phi[1:] - phi[0]
        feats.extend(p.tolist())
        feats.extend(np.cos(rel).tolist())
        feats.extend(np.sin(rel).tolist())
    return feats


def features_pairs(psi):
    """L2 -- L1 + cos/sin des differences de phases dominantes entre paires."""
    feats = features_local(psi)
    dom = [g.dominant_phase(psi[i]) for i in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            d = dom[i] - dom[j]
            feats.append(np.cos(d))
            feats.append(np.sin(d))
    return feats


def ridge_train_acc(Xtr, ytr, Xte, yte):
    Xa = np.hstack([np.ones((Xtr.shape[0], 1)), Xtr])
    F = Xa.shape[1]
    W = np.linalg.solve(Xa.T @ Xa + RIDGE_REG * np.eye(F), Xa.T @ ytr)
    Xb = np.hstack([np.ones((Xte.shape[0], 1)), Xte])
    pred = np.sign(Xb @ W)
    pred[pred == 0] = 1
    return float(np.mean(pred == yte))


def main():
    print("=" * 76)
    print("GENESIS READOUT -- l'information du plateau est-elle lisible localement ?")
    print(f"{N_TRIALS} essais (re-simulation int-only) | train {N_TRAIN} / test {N_TRIALS - N_TRAIN}")
    print("=" * 76)

    X1, X2, ys = [], [], []
    k_local, k_global, k_vote = 0, 0, 0
    for seed in range(N_TRIALS):
        psi, s, guess_vote = run_trial_int_only(seed)
        X1.append(features_local(psi))
        X2.append(features_pairs(psi))
        ys.append(s)
        k_local += int(g.read_out(psi) == s)
        k_global += int(g.read_out_global(psi) == s)
        k_vote += int(guess_vote == s)
        if (seed + 1) % 200 == 0:
            print(f"  ... {seed + 1}/{N_TRIALS}")

    # ---- GATE : etats identiques au run committe du POC ----
    print("\n--- GATE (accuracies exactes du run committe cb36b4a) ---")
    got = {"k_local_int": k_local, "k_global_int": k_global, "k_vote": k_vote}
    ok = True
    for key, exp in GATE_EXPECTED.items():
        status = "OK " if got[key] == exp else "FAIL"
        ok = ok and got[key] == exp
        print(f"  [{status}] {key:<14} attendu={exp}  obtenu={got[key]}")
    if not ok:
        print("  GATE FAIL : la re-simulation int-only ne reproduit pas le POC. STOP.")
        sys.exit(1)
    print("  Etats identiques confirmes (int ne consomme aucun rng de boucle).")

    X1 = np.array(X1)
    X2 = np.array(X2)
    ys = np.array(ys, dtype=float)
    tr = slice(0, N_TRAIN)
    te = slice(N_TRAIN, N_TRIALS)
    n_te = N_TRIALS - N_TRAIN

    acc_l1 = ridge_train_acc(X1[tr], ys[tr], X1[te], ys[te])
    acc_l2 = ridge_train_acc(X2[tr], ys[tr], X2[te], ys[te])

    # references sur le MEME split test
    te_idx = range(N_TRAIN, N_TRIALS)
    accs_ref = {"R2_global (prior parite)": 0.0, "vote": 0.0, "R1_local_fixe": 0.0}
    kg = kv = kl = 0
    for seed in te_idx:
        psi, s, guess_vote = run_trial_int_only(seed)
        kg += int(g.read_out_global(psi) == s)
        kv += int(guess_vote == s)
        kl += int(g.read_out(psi) == s)
    accs_ref["R2_global (prior parite)"] = kg / n_te
    accs_ref["vote"] = kv / n_te
    accs_ref["R1_local_fixe"] = kl / n_te

    print(f"\n--- RESULTATS (test = {n_te} essais, IC Wilson 95%) ---")
    rows = []
    for name, acc in [("L1 strict-local appris", acc_l1),
                      ("L2 paires appris", acc_l2),
                      *accs_ref.items()]:
        k = int(round(acc * n_te))
        lo, hi = g.wilson(k, n_te)
        rows.append((name, acc, lo, hi))
        print(f"  {name:<28}: {100*acc:5.1f}%  IC[{100*lo:.1f}, {100*hi:.1f}]")

    print("\n--- VERDICT ---")
    r2 = accs_ref["R2_global (prior parite)"]
    if acc_l2 >= r2 - 0.03:
        print("  L2 (relations de paires) atteint ~le readout global : l'information")
        print("  est accessible par apprentissage SANS le prior d'ordre 5.")
    elif acc_l2 > accs_ref["vote"] + 0.03:
        print("  L2 depasse le vote sans atteindre le global : les paires portent une")
        print("  partie du signal, le produit complet reste au-dessus.")
    else:
        print("  L2 ne depasse pas clairement le vote : les relations de paires ne")
        print("  suffisent pas -- l'information de parite exige la lecture d'ordre 5.")
    if acc_l1 < 0.6:
        print("  L1 strict-local reste pres du hasard : conforme a la prediction --")
        print("  la parite n'est PAS dans les marginales locales ; la dynamique")
        print("  interferentielle PRESERVE l'information mais ne la LOCALISE pas.")
    else:
        print("  L1 strict-local depasse la prediction (>60%) : une part de l'information")
        print("  s'est LOCALISEE dans les etats individuels -- inattendu, a creuser.")

    # ---- CSV + figure ----
    os.makedirs(FIG_DIR, exist_ok=True)
    with open(BASE + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["readout", "acc_test", "ci_lo", "ci_hi"])
        for name, acc, lo, hi in rows:
            w.writerow([name, f"{acc:.4f}", f"{lo:.4f}", f"{hi:.4f}"])

    fig, ax = plt.subplots(figsize=(8, 4.8))
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    errs_lo = [r[1] - r[2] for r in rows]
    errs_hi = [r[3] - r[1] for r in rows]
    colors = ["#d62728", "#ff7f0e", "#3182bd", "crimson", "#999999"]
    ax.bar(range(len(rows)), vals, yerr=[errs_lo, errs_hi], capsize=4,
           color=colors[:len(rows)], edgecolor="k")
    ax.axhline(0.5, ls=":", c="black", label="hasard")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([n.replace(" ", "\n", 1) for n in names], fontsize=7)
    ax.set_ylim(0.4, 0.9)
    ax.set_ylabel("accuracy (test, 300 essais)")
    ax.set_title("Ou vit l'information du plateau interference (73.9%) ?\n"
                 "readouts appris (local / paires) vs analytique global vs vote")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(BASE + ".png", dpi=150)
    print(f"\nSorties : {BASE}.csv\n          {BASE}.png")
    print("--- FIN ---")


if __name__ == "__main__":
    main()
