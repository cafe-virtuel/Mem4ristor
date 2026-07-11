"""
genesis_five_states_poc.py -- Les cinq etats retrouves, a froid (Front 1)
=========================================================================
Auteur  : Claude Fable 5 (L'Ingenieur) -- 11 juillet 2026
Contexte: Mem4ristor est ne le 19 aout 2025 (Session 1 du Cafe Virtuel) comme
          une unite cognitive a 5 etats (Certitude, Probable, Incertain,
          Intuition, Oracle). Neuf mois de rigueur l'ont reduit a un scalaire
          u dans [0,1]. Le 10 juillet 2026, un premier jouet (Labo de l'Absurde,
          experience_011_les_cinq_etats_retrouves.py) a teste un axe 4 etats
          (vecteur psi dans C^4 par noeud) + Oracle (coherence de phase entre
          les extremes) sur une parite cachee a 5 bits. Tendance monotone
          encourageante (melange 51.7% < interference 61.7% < multiplicatif
          66.7%) mais AUCUN ne bat le vote (78.3%), sur 60 essais SANS
          intervalle de confiance. Decision du 10/07 : reprendre A FROID,
          d'abord la puissance statistique sur le MEME jouet, avant toute
          nouvelle complexite. C'est ce script.

Design (4 questions, une par phase) :
  A. PUISSANCE  -- les memes mecanismes, repris a l'identique (sequence de
     tirages aleatoires comprise ; gate de replication exacte sur les seeds
     0..59 du 10/07), montes a N_TRIALS=1000 essais. IC Wilson 95% par
     condition + differences APPARIEES par bootstrap (memes graines pour
     toutes les conditions d'un essai). La tendance monotone est-elle reelle ?
  B. MECANISME  -- decomposition du probleme en trois questions distinctes :
     (i)  l'information de parite est-elle PRESENTE a t=0 ? (readout global
          R2 a t=0 : somme des phases dominantes = definition meme de la
          parite, modulo bruit d'init -- controle attendu tres haut) ;
     (ii) la dynamique la PRESERVE-t-elle ? (courbe R2(t) sous chacune des
          4 dynamiques -- le melange, qui detruit les phases par construction,
          est le controle negatif attendu au hasard) ;
     (iii) le readout LOCAL (R1, celui du 10/07 : Certitude-vs-Incertain par
          noeud) la LIT-il ? L'ecart R2(T)-R1(T) mesure ce que la lecture
          locale perd de ce que le reseau a garde.
  C. HOP MULTIPLICATIF -- la question laissee ouverte le 10/07 : dans le jouet,
     la migration le long de l'axe Incertain->Certitude (hop_along_axis) est
     pilotee par l'accord LINEAIRE avec la moyenne des voisins (le consensus),
     meme dans la condition multiplicative. 4e condition ajoutee : couplage
     multiplicatif + hop pilote par l'accord avec le PRODUIT des phases des
     voisins (la Certitude se gagne par coherence avec l'information calculee,
     pas avec le consensus). Aucun tirage aleatoire ajoute : la sequence rng
     des conditions historiques reste strictement identique.
  D. ORACLE -- les episodes Oracle (les deux poles opposes portes a egale
     amplitude ET en phase) sont-ils correles a la reussite ? Difference de
     taux de reussite entre essais avec/sans episode Oracle, IC bootstrap.

Reference externe inchangee : le VOTE majoritaire. A N=5 il n'est PAS a 50%
(correlation majorite-parite ~68.75% theorique a cette taille, fait
mathematique, pas un bug) -- c'est la reference honnete a battre.

Statut : jouet exploratoire, AUCUN claim du preprint n'en depend. Resultat
rapporte tel quel, y compris s'il est decevant. Rien ici ne touche dynamics.py.
Sorties : figures/genesis_five_states_poc_raw.csv, _agg.csv, _traj.csv, .png
"""

import csv
import os
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Parametres (identiques au jouet du 10/07 pour eta/T/N ; TRIALS monte a 1000)
# ----------------------------------------------------------------------------
STATES = ["Incertain", "Probable", "Intuition", "Certitude"]
N = 5            # noeuds, graphe complet
ETA = 0.3        # taux de couplage
T_STEPS = 150    # pas de simulation
N_TRIALS = 1000  # essais (le 10/07 : 60)
N_BOOT = 10000   # reechantillons bootstrap
SAMPLE_TS = [0, 5, 10, 20, 40, 80, 150]  # echantillonnage de R2(t)

# Gate de replication : valeurs EXACTES mesurees le 10/07 sur seeds 0..59
GATE_EXPECTED = {
    "k_mix": 31, "k_int": 37, "k_mult": 40, "k_vote": 47,
    "oracle_mix": 0, "oracle_int": 536, "oracle_mult": 900,
}

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
BASE = os.path.join(FIG_DIR, "genesis_five_states_poc")


# ----------------------------------------------------------------------------
# Mecanismes du jouet -- repris A L'IDENTIQUE d'experience_011 (10/07/2026).
# Toute divergence casserait le gate de replication ci-dessous.
# ----------------------------------------------------------------------------
def normalize(psi):
    norm = np.sqrt(np.sum(np.abs(psi) ** 2, axis=-1, keepdims=True))
    return psi / (norm + 1e-12)


def init_from_bit(b, rng, noise=0.15):
    """b in {-1,+1} -> etat concentre sur Incertain (index 0), la phase encode
    b : 0 si b=+1, pi si b=-1 (+ bruit angulaire)."""
    psi = np.zeros(4, dtype=complex)
    phase = 0.0 if b > 0 else np.pi
    phase += rng.normal(0, noise)
    psi[0] = np.exp(1j * phase)
    return psi


def step_mixture(psi, A, eta, rng):
    """(a) MELANGE : les probabilites se couplent, la phase est rejetee
    (re-echantillonnee) a chaque pas -- ignorance classique. Seul consommateur
    de rng dans la boucle de simulation."""
    p = np.abs(psi) ** 2
    p_new = p.copy()
    for i in range(N):
        neighbors = [j for j in range(N) if A[i, j] > 0]
        if neighbors:
            p_avg = np.mean(p[neighbors], axis=0)
            p_new[i] = (1 - eta) * p[i] + eta * p_avg
    phases = rng.uniform(0, 2 * np.pi, size=(N, 4))
    psi_new = np.sqrt(np.clip(p_new, 0, None)) * np.exp(1j * phases)
    return normalize(psi_new)


def step_interference(psi, A, eta):
    """(b) INTERFERENCE : les amplitudes complexes se couplent (moyenne),
    la phase relative est preservee et peut s'annuler ou se renforcer."""
    psi_new = psi.copy()
    for i in range(N):
        neighbors = [j for j in range(N) if A[i, j] > 0]
        if neighbors:
            psi_avg = np.mean(psi[neighbors], axis=0)
            psi_new[i] = (1 - eta) * psi[i] + eta * psi_avg
    return normalize(psi_new)


def step_multiplicative(psi, A, eta):
    """(c) MULTIPLICATIF : la magnitude reste couplee en douceur, mais la
    phase est PRODUITE (les angles des voisins s'additionnent) -- l'analogue
    continu exact du produit de signes qui definit la parite."""
    psi_new = psi.copy()
    for i in range(N):
        neighbors = [j for j in range(N) if A[i, j] > 0]
        if neighbors:
            prod_phase = np.ones(4, dtype=complex)
            for j in neighbors:
                prod_phase = prod_phase * np.exp(1j * np.angle(psi[j]))
            mag = np.abs(psi[i])
            psi_prod = mag * prod_phase
            psi_new[i] = (1 - eta) * psi[i] + eta * psi_prod
    return normalize(psi_new)


def detect_oracle(psi, amp_thresh=0.30, diff_thresh=0.15, phase_thresh=0.3):
    """Oracle = les deux poles opposes (Incertain=0, Certitude=3) portes a
    egale amplitude significative ET en phase (interference constructive)."""
    amp_i, amp_c = np.abs(psi[0]), np.abs(psi[3])
    if amp_i < amp_thresh or amp_c < amp_thresh:
        return False
    if abs(amp_i - amp_c) > diff_thresh:
        return False
    dphi = abs(np.angle(psi[3]) - np.angle(psi[0]))
    dphi = min(dphi, 2 * np.pi - dphi)
    return dphi < phase_thresh


def local_agreement(psi, A, i):
    """Coherence du noeud i avec la MOYENNE de ses voisins (lineaire)."""
    neighbors = [j for j in range(N) if A[i, j] > 0]
    if not neighbors:
        return 0.0
    psi_avg = np.mean(psi[neighbors], axis=0)
    inner = np.vdot(psi[i], psi_avg)
    return float(np.real(inner))


def hop_along_axis(psi, A, hop_rate=0.15):
    """Migration le long de l'axe Incertain(0)..Certitude(3) : accord fort
    avec les voisins pousse vers Certitude, desaccord vers Incertain.
    Pilotage LINEAIRE (accord avec la moyenne) -- version du 10/07."""
    psi_new = psi.copy()
    for i in range(N):
        agree = local_agreement(psi, A, i)
        target = 3 if agree > 0 else 0
        strength = min(abs(agree), 1.0) * hop_rate
        phase_i = np.angle(psi[i, np.argmax(np.abs(psi[i]))])
        target_vec = np.zeros(4, dtype=complex)
        target_vec[target] = np.exp(1j * phase_i)
        psi_new[i] = (1 - strength) * psi[i] + strength * target_vec
    return normalize(psi_new)


def read_out(psi_all):
    """R1 -- lecture LOCALE du 10/07 : signe de la coherence
    Certitude-vs-Incertain, sommee sur les noeuds."""
    score = 0.0
    for psi in psi_all:
        score += (np.abs(psi[3]) ** 2 - np.abs(psi[0]) ** 2) * np.cos(
            np.angle(psi[3]) - np.angle(psi[0])
        )
    if score > 0:
        return 1
    if score < 0:
        return -1
    return 0


# ----------------------------------------------------------------------------
# Ajouts instrumentaux (aucun tirage aleatoire -> sequence rng inchangee)
# ----------------------------------------------------------------------------
def dominant_phase(psi_node):
    """Phase de la composante dominante d'un noeud."""
    return float(np.angle(psi_node[np.argmax(np.abs(psi_node))]))


def read_out_global(psi_all):
    """R2 -- lecture GLOBALE multiplicative : signe du cosinus de la SOMME des
    phases dominantes (= produit des phaseurs). A t=0 c'est la definition meme
    de la parite (modulo bruit d'init) : controle 'information presente'.
    A t=T : mesure la PRESERVATION de l'information par la dynamique,
    independamment de la lecture locale."""
    total = sum(dominant_phase(p) for p in psi_all)
    c = np.cos(total)
    if c > 0:
        return 1
    if c < 0:
        return -1
    return 0


def hop_multiplicative(psi, A, hop_rate=0.15):
    """Question laissee le 10/07 : meme migration Incertain->Certitude, mais
    pilotee par l'accord avec le PRODUIT des phases dominantes des voisins
    (l'information collective multiplicative, celle qui porte la parite) au
    lieu de l'accord lineaire avec leur moyenne (le consensus). La Certitude
    se gagne par coherence avec le calcul, pas avec la majorite."""
    psi_new = psi.copy()
    for i in range(N):
        neighbors = [j for j in range(N) if A[i, j] > 0]
        if not neighbors:
            continue
        prod_angle = sum(dominant_phase(psi[j]) for j in neighbors)
        agree = float(np.cos(dominant_phase(psi[i]) - prod_angle))
        target = 3 if agree > 0 else 0
        strength = min(abs(agree), 1.0) * hop_rate
        phase_i = dominant_phase(psi[i])
        target_vec = np.zeros(4, dtype=complex)
        target_vec[target] = np.exp(1j * phase_i)
        psi_new[i] = (1 - strength) * psi[i] + strength * target_vec
    return normalize(psi_new)


# ----------------------------------------------------------------------------
# Un essai complet (sequence rng strictement identique au jouet du 10/07 :
# s, bits, shuffle, 5 inits, puis UN rng.uniform par pas via step_mixture ;
# la 4e condition et tous les readouts n'ajoutent aucun tirage)
# ----------------------------------------------------------------------------
CONDS = ["mix", "int", "mult", "multhop"]


def run_trial(seed, eta=ETA, T=T_STEPS):
    rng = np.random.default_rng(seed)
    s = rng.choice([-1, 1])
    b = rng.choice([-1, 1], size=N - 1).astype(float)
    b_last = s * np.prod(b)
    bits = np.concatenate([b, [b_last]])
    rng.shuffle(bits)

    A = np.ones((N, N)) - np.eye(N)

    psi = {c: None for c in CONDS}
    psi["mix"] = np.array([init_from_bit(bi, rng) for bi in bits])
    psi["int"] = psi["mix"].copy()
    psi["mult"] = psi["mix"].copy()
    psi["multhop"] = psi["mix"].copy()

    oracle = {c: 0 for c in CONDS}
    # R2(t) : correct global au fil du temps, par condition
    traj = {c: {} for c in CONDS}
    for c in CONDS:
        traj[c][0] = int(read_out_global(psi[c]) == s)

    for t in range(T):
        psi["mix"] = step_mixture(psi["mix"], A, eta, rng)  # seul appel rng
        psi["mix"] = hop_along_axis(psi["mix"], A)
        psi["int"] = step_interference(psi["int"], A, eta)
        psi["int"] = hop_along_axis(psi["int"], A)
        psi["mult"] = step_multiplicative(psi["mult"], A, eta)
        psi["mult"] = hop_along_axis(psi["mult"], A)
        psi["multhop"] = step_multiplicative(psi["multhop"], A, eta)
        psi["multhop"] = hop_multiplicative(psi["multhop"], A)
        for c in CONDS:
            oracle[c] += sum(detect_oracle(p) for p in psi[c])
        if (t + 1) in SAMPLE_TS:
            for c in CONDS:
                traj[c][t + 1] = int(read_out_global(psi[c]) == s)

    vote_sum = np.sum(bits)
    guess_vote = 1 if vote_sum > 0 else (-1 if vote_sum < 0 else int(rng.choice([-1, 1])))

    out = {"seed": seed, "s": s, "correct_vote": int(guess_vote == s)}
    for c in CONDS:
        out[f"correct_local_{c}"] = int(read_out(psi[c]) == s)
        out[f"correct_global_{c}"] = int(read_out_global(psi[c]) == s)
        out[f"oracle_{c}"] = oracle[c]
        for ts in SAMPLE_TS:
            out[f"g_{c}_t{ts}"] = traj[c][ts]
    return out


# ----------------------------------------------------------------------------
# Statistiques
# ----------------------------------------------------------------------------
def wilson(k, n, z=1.96):
    """IC 95% de Wilson pour une proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((center - half) / denom, (center + half) / denom)


def paired_bootstrap_diff(a, b, n_boot=N_BOOT, seed=12345):
    """IC 95% bootstrap de mean(a-b) sur essais apparies + p bilaterale."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diffs = a - b
    rng = np.random.default_rng(seed)
    n = len(diffs)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p_le = np.mean(boots <= 0)
    p_ge = np.mean(boots >= 0)
    p_two = 2 * min(p_le, p_ge)
    return float(diffs.mean()), float(lo), float(hi), float(min(1.0, p_two))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    print("=" * 76)
    print("GENESIS FIVE STATES POC -- reprise a froid du jouet du 10/07/2026")
    print(f"N={N} noeuds (complet) | {N_TRIALS} essais | T={T_STEPS} | eta={ETA}")
    print("=" * 76)

    results = []
    for seed in range(N_TRIALS):
        results.append(run_trial(seed))
        if (seed + 1) % 100 == 0:
            print(f"  ... {seed + 1}/{N_TRIALS} essais")

    # ---------------- GATE DE REPLICATION (seeds 0..59 du 10/07) -----------
    print("\n--- GATE DE REPLICATION (seeds 0..59, valeurs exactes du 10/07) ---")
    first60 = results[:60]
    got = {
        "k_mix": sum(r["correct_local_mix"] for r in first60),
        "k_int": sum(r["correct_local_int"] for r in first60),
        "k_mult": sum(r["correct_local_mult"] for r in first60),
        "k_vote": sum(r["correct_vote"] for r in first60),
        "oracle_mix": sum(r["oracle_mix"] for r in first60),
        "oracle_int": sum(r["oracle_int"] for r in first60),
        "oracle_mult": sum(r["oracle_mult"] for r in first60),
    }
    gate_ok = True
    for key, exp in GATE_EXPECTED.items():
        status = "OK " if got[key] == exp else "FAIL"
        if got[key] != exp:
            gate_ok = False
        print(f"  [{status}] {key:<12} attendu={exp:<5} obtenu={got[key]}")
    if not gate_ok:
        print("\n  GATE FAIL : la replication du jouet n'est pas exacte.")
        print("  Les resultats ci-dessous ne sont PAS comparables au 10/07 -- STOP.")
        sys.exit(1)
    print("  Replication exacte confirmee : meme code, meme sequence aleatoire.")

    # ---------------- PHASE A : accuracies + IC ----------------------------
    n = len(results)
    print(f"\n--- PHASE A : ACCURACY ({n} essais, IC Wilson 95%) ---")
    acc = {}
    labels = [
        ("vote", "correct_vote", "VOTE majoritaire (reference)"),
        ("mix", "correct_local_mix", "(a) MELANGE   R1 local"),
        ("int", "correct_local_int", "(b) INTERFER. R1 local"),
        ("mult", "correct_local_mult", "(c) MULTIPLIC R1 local"),
        ("multhop", "correct_local_multhop", "(e) MULT+HOPM R1 local"),
    ]
    for name, key, label in labels:
        k = sum(r[key] for r in results)
        lo, hi = wilson(k, n)
        acc[name] = (k / n, lo, hi)
        print(f"  {label:<32}: {100*k/n:5.1f}%  IC[{100*lo:.1f}, {100*hi:.1f}]")

    print("\n  Differences appariees (bootstrap 95%, p bilaterale) :")
    pairs = [
        ("int - mix", "correct_local_int", "correct_local_mix"),
        ("mult - int", "correct_local_mult", "correct_local_int"),
        ("mult - mix", "correct_local_mult", "correct_local_mix"),
        ("mult - vote", "correct_local_mult", "correct_vote"),
        ("multhop - mult", "correct_local_multhop", "correct_local_mult"),
        ("multhop - vote", "correct_local_multhop", "correct_vote"),
        ("g_int - vote", "correct_global_int", "correct_vote"),
        ("g_int - l_int", "correct_global_int", "correct_local_int"),
        ("g_int - g_mult", "correct_global_int", "correct_global_mult"),
    ]
    diff_rows = []
    for label, ka, kb in pairs:
        a = [r[ka] for r in results]
        b = [r[kb] for r in results]
        d, lo, hi, p = paired_bootstrap_diff(a, b)
        sig = "SIGNIF" if (lo > 0 or hi < 0) else "ns    "
        print(f"    {label:<16}: {100*d:+6.1f} pts  IC[{100*lo:+.1f}, {100*hi:+.1f}]  p~{p:.4f}  {sig}")
        diff_rows.append((label, d, lo, hi, p))

    # ---------------- PHASE B : mecanisme (R0 / R2(t) / R1 vs R2) ----------
    print("\n--- PHASE B : OU L'INFORMATION DE PARITE VIT ET MEURT ---")
    k0 = sum(r["g_mult_t0"] for r in results)  # t=0 identique pour toutes cond.
    lo0, hi0 = wilson(k0, n)
    print(f"  (i)  Presence a t=0 (R2 global, trivial par construction) : "
          f"{100*k0/n:5.1f}%  IC[{100*lo0:.1f}, {100*hi0:.1f}]")
    print("  (ii) Preservation R2(t) par dynamique :")
    header = "       t     : " + "  ".join(f"{ts:>5d}" for ts in SAMPLE_TS)
    print(header)
    for c in CONDS:
        vals = []
        for ts in SAMPLE_TS:
            kk = sum(r[f"g_{c}_t{ts}"] for r in results)
            vals.append(f"{100*kk/n:5.1f}")
        print(f"       {c:<8}: " + "  ".join(vals))
    print("  (iii) Localisation a t=T : R1 (lecture locale) vs R2 (info globale)")
    for c in CONDS:
        k1 = sum(r[f"correct_local_{c}"] for r in results)
        k2 = sum(r[f"correct_global_{c}"] for r in results)
        print(f"       {c:<8}: R1={100*k1/n:5.1f}%  R2={100*k2/n:5.1f}%  "
              f"(perte de lecture R2-R1 = {100*(k2-k1)/n:+.1f} pts)")

    # Accord des predictions (meme s par essai -> predictions egales ssi
    # correct egaux) : R2_int est-il un simple relabel du vote, ou porte-t-il
    # un signal au-dela de la majorite ?
    agree = np.mean([r["correct_global_int"] == r["correct_vote"] for r in results])
    both_keys = [("correct_global_int", "correct_vote")]
    g_only = np.mean([r["correct_global_int"] and not r["correct_vote"] for r in results])
    v_only = np.mean([r["correct_vote"] and not r["correct_global_int"] for r in results])
    print(f"\n  Accord de predictions R2_interference vs vote : {100*agree:.1f}% "
          f"(R2 seul juste : {100*g_only:.1f}% ; vote seul juste : {100*v_only:.1f}%)")

    # ---------------- PHASE D : Oracle et reussite --------------------------
    print("\n--- PHASE D : ORACLE ET REUSSITE ---")
    for c in ["int", "mult", "multhop"]:
        ora = np.array([r[f"oracle_{c}"] for r in results])
        cor = np.array([r[f"correct_local_{c}"] for r in results])
        with_o = cor[ora > 0]
        without_o = cor[ora == 0]
        tot = int(ora.sum())
        if len(with_o) > 4 and len(without_o) > 4:
            # bootstrap non apparie de la difference de proportions
            rng = np.random.default_rng(777)
            bw = rng.integers(0, len(with_o), size=(N_BOOT, len(with_o)))
            bo = rng.integers(0, len(without_o), size=(N_BOOT, len(without_o)))
            dd = with_o[bw].mean(axis=1) - without_o[bo].mean(axis=1)
            lo, hi = np.percentile(dd, [2.5, 97.5])
            print(f"  {c:<8}: episodes tot={tot:6d} | essais avec Oracle "
                  f"{len(with_o):4d} (acc {100*with_o.mean():5.1f}%) vs sans "
                  f"{len(without_o):4d} (acc {100*without_o.mean():5.1f}%) | "
                  f"diff {100*(with_o.mean()-without_o.mean()):+5.1f} pts "
                  f"IC[{100*lo:+.1f}, {100*hi:+.1f}]")
        else:
            base = 100 * cor.mean()
            print(f"  {c:<8}: episodes tot={tot:6d} | repartition trop desequilibree "
                  f"pour comparer (avec={len(with_o)}, sans={len(without_o)}, "
                  f"acc globale {base:.1f}%)")

    # ---------------- CSV ----------------------------------------------------
    os.makedirs(FIG_DIR, exist_ok=True)
    raw_path = BASE + "_raw.csv"
    with open(raw_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    agg_path = BASE + "_agg.csv"
    with open(agg_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value", "ci_lo", "ci_hi"])
        for name, (p, lo, hi) in acc.items():
            w.writerow([f"acc_{name}", f"{p:.4f}", f"{lo:.4f}", f"{hi:.4f}"])
        for label, d, lo, hi, pv in diff_rows:
            w.writerow([f"diff_{label.replace(' ', '')}", f"{d:.4f}", f"{lo:.4f}", f"{hi:.4f}"])

    traj_path = BASE + "_traj.csv"
    with open(traj_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cond", "t", "acc", "ci_lo", "ci_hi"])
        for c in CONDS:
            for ts in SAMPLE_TS:
                kk = sum(r[f"g_{c}_t{ts}"] for r in results)
                lo, hi = wilson(kk, n)
                w.writerow([c, ts, f"{kk/n:.4f}", f"{lo:.4f}", f"{hi:.4f}"])

    # ---------------- Figure -------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    names = ["mix", "int", "mult", "multhop"]
    disp = ["melange\n(sans phase)", "interference\n(moyenne cplx)",
            "multiplicatif\n(hop lineaire)", "multiplicatif\n(hop produit)"]
    xs = np.arange(len(names))
    vals = [acc[nm][0] for nm in names]
    errs_lo = [acc[nm][0] - acc[nm][1] for nm in names]
    errs_hi = [acc[nm][2] - acc[nm][0] for nm in names]
    ax.bar(xs, vals, color=["#999999", "#6baed6", "#3182bd", "#08519c"],
           yerr=[errs_lo, errs_hi], capsize=4)
    ax.axhline(acc["vote"][0], color="crimson", ls="--", lw=1.5,
               label=f"vote majoritaire ({100*acc['vote'][0]:.1f}%)")
    ax.axhspan(acc["vote"][1], acc["vote"][2], color="crimson", alpha=0.12)
    ax.axhline(0.5, color="black", ls=":", lw=1, label="hasard (50%)")
    ax.set_xticks(xs)
    ax.set_xticklabels(disp, fontsize=8)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("accuracy (lecture locale R1)")
    ax.set_title(f"Parite cachee 5 bits -- {n} essais, IC Wilson 95%")
    ax.legend(fontsize=8)

    ax = axes[1]
    colors = {"mix": "#999999", "int": "#6baed6", "mult": "#3182bd", "multhop": "#08519c"}
    for c in CONDS:
        ys, los, his = [], [], []
        for ts in SAMPLE_TS:
            kk = sum(r[f"g_{c}_t{ts}"] for r in results)
            lo, hi = wilson(kk, n)
            ys.append(kk / n)
            los.append(lo)
            his.append(hi)
        ax.plot(SAMPLE_TS, ys, "o-", color=colors[c], label=c)
        ax.fill_between(SAMPLE_TS, los, his, color=colors[c], alpha=0.15)
    ax.axhline(0.5, color="black", ls=":", lw=1)
    ax.set_xlabel("pas de simulation t")
    ax.set_ylabel("accuracy R2 (info globale de parite)")
    ax.set_title("Survie de l'information de parite sous chaque dynamique")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(BASE + ".png", dpi=150)
    print(f"\nSorties : {raw_path}\n          {agg_path}\n          {traj_path}\n          {BASE}.png")

    # ---------------- VERDICT ------------------------------------------------
    # diff_rows[i] = (label, d, lo, hi, p) ; significatif = IC excluant 0
    def verdict_of(row):
        _, d, lo, hi, _ = row
        if lo > 0:
            return "GAIN significatif"
        if hi < 0:
            return "PERTE significative"
        return "indistinguable (IC couvre 0)"

    d_mono1 = diff_rows[0]   # int - mix
    d_mono2 = diff_rows[1]   # mult - int
    d_vote = diff_rows[3]    # mult - vote
    d_hop = diff_rows[4]     # multhop - mult
    d_gvote = diff_rows[6]   # g_int - vote (lecture globale interference vs vote)
    d_gloc = diff_rows[7]    # g_int - l_int (ce que la lecture locale perd)

    print("\n" + "=" * 76)
    print("VERDICT (a lire avec les IC ci-dessus, pas seulement les moyennes)")
    print("=" * 76)
    mono = (d_mono1[2] > 0) and (d_mono2[2] > 0)
    print(f"  1. Tendance monotone du 10/07 (mix < int < mult, lecture locale) : "
          f"{'CONFIRMEE' if mono else 'NON confirmee -> fluctuation des 60 essais'}")
    print(f"  2. Multiplicatif (lecture locale) vs vote : {verdict_of(d_vote)}")
    print(f"  3. Hop multiplicatif vs hop lineaire (question du 10/07) : "
          f"{verdict_of(d_hop)}")
    print(f"  4. LECTURE GLOBALE du reseau interference vs vote : "
          f"{verdict_of(d_gvote)}  ({100*d_gvote[1]:+.1f} pts)")
    print(f"  5. Ce que la lecture locale perdait (g_int - l_int) : "
          f"{verdict_of(d_gloc)}  ({100*d_gloc[1]:+.1f} pts)")
    print("  6. Phase B situe l'information : presence a t=0 (triviale), survie")
    print("     sous chaque dynamique (courbe R2(t)), lecture (R1 vs R2).")
    print("--- FIN ---")


if __name__ == "__main__":
    main()
