#!/usr/bin/env python3
"""
P17 -- L'ablation FROZEN_U du preprint mesure-t-elle l'ADAPTATIVITE de u, ou son NIVEAU ?
(29 juillet 2026, Claude Opus 5, demande de Julien : "relire les comparaisons FULL/FROZEN_U
avec l'oeil du 28/07")

------------------------------------------------------------------------------------------
LA QUESTION, ET POURQUOI ELLE SE POSE MAINTENANT
------------------------------------------------------------------------------------------
Le filtre de couplage vaut  u_filter = tanh(pi*(0.5 - u)) + 0.01 :

    u = 0.05  -> u_filter = +0.90   couplage SYNCHRONISANT fort
    u = 0.50  -> u_filter = +0.01   couplage QUASI NUL (reseau decouple)
    u = 0.99  -> u_filter = -0.91   couplage ANTI-SYNCHRONISANT fort

Le projet contient DEUX comparateurs distincts nommes "FROZEN_U", a deux valeurs opposees :
  - colonne A (le preprint)  : u gele a sigma_baseline = 0.05  -> ablation_coordination.py:112
  - colonne B (P15, P16)     : u gele a 0.50                   -> reseau decouple
La note du 28/07 affirmait que TOUTE comparaison FULL/FROZEN_U du projet heritait de
l'ambiguite "u=0.5 = reseau sans lien". C'est VRAI de la colonne B et FAUX du preprint.

Ce qui reste en question pour le preprint est different, et reel : son ablation fait passer le
couplage de REPULSIF (FULL, u -> 0.997) a ATTRACTIF FORT (u = 0.05). Elle change donc le SIGNE
du couplage, et pas seulement "avec / sans dynamique du doute". Or le papier ecrit :
  - abstract      : "freezing the doubt dynamics collapses the network into synchrony"
  - contribution 2: "the identification of the doubt DYNAMICS as the primary
                     anti-synchronization mechanism"
  - Table 1       : "Frozen u (no doubt dynamics)"
alors que la Discussion 5.1 decrit un mecanisme de POLARITE, donc de NIVEAU :
  "when a node's doubt exceeds u = 0.5, its coupling flips from attractive to repulsive".

Le controle qui separe les deux lectures -- u FIGE HAUT, au niveau que le doute atteint
lui-meme -- n'existe nulle part dans la colonne A. C'est lui qu'on ajoute ici.

------------------------------------------------------------------------------------------
LES QUATRE BRAS
------------------------------------------------------------------------------------------
  FULL          u adaptatif                                   (reference du papier)
  FROZ_005      u = 0.05 homogene, fige                       (l'ablation DU PAPIER)
  FROZ_UBAR     u = u_bar homogene, fige                      separe NIVEAU / ADAPTATIVITE
  FROZ_PROFILE  u = profil final par noeud de FULL, fige      separe HETEROGENEITE / NIVEAU

u_bar est MESURE ici (moyenne sur les graines canoniques de mean(u_final) en FULL), jamais
emprunte a un autre regime : Table "scaling" du preprint donne u_bar sur LATTICE, on est ici
sur BA m=3.

------------------------------------------------------------------------------------------
CRITERES, ECRITS AVANT LA MESURE (29/07/2026, avant tout lancement)
------------------------------------------------------------------------------------------
G0  (FIDELITE bit-a-bit au CSV de reference) -- ECHOUE, ET NON DEPLACE.
    Constat pose AVANT la mesure principale : figures/scratch/ablation_coordination.csv date
    du 26/04/2026, soit AVANT le fix de bruit Euler-Maruyama du 01/05 (AUDIT-024, 818cf67), et
    il n'est pas versionne (figures/scratch/ est gitignore). Une comparaison bit-a-bit a ce
    fichier ne peut PAS repondre a la question "mon harnais reproduit-il celui du papier" :
    c'est un INSTRUMENT INADAPTE (reference produite par une autre version du coeur), pas un
    seuil trop exigeant. Le rejet reste affiche ; le gate est remplace, pas repeche.
    -> voir CONTRAT_METHODE_SCIENTIFIQUE.md, regle P1.

G0b (ACCORD AGREGE AU PREPRINT, le gate qui decide).
    Les moyennes sur 10 graines canoniques doivent tomber dans [mu +/- sigma] tels que
    tab:ablations les publie :
        FULL     synchrony 0.031 +/- 0.034   LZ 1.069 +/- 0.016
        FROZEN_U synchrony 0.751 +/- 0.060   LZ 1.635 +/- 0.006
    Critere volontairement genereux (sigma = dispersion inter-graines, pas un IC) et dit tel
    quel. S'il ECHOUE, la conclusion n'est pas "H1/H2" mais : LES CHIFFRES DE TABLE 1 NE SONT
    PLUS REPRODUITS PAR LE CODE ACTUEL -- et c'est alors CE resultat qui est rapporte.

G1  (CONTROLE D'INSTRUMENT : le gel gele-t-il vraiment ?)
    Dans les trois bras figes, max|u(t=fin) - u(t=0)| < 1e-12 sur toutes les graines.
    L'ART et le watchdog peuvent ecrire u ; si ce controle tombe, l'ablation ne fait pas ce
    que son nom dit, et tout verdict en dessous est suspendu.

H1  "LE NIVEAU SUFFIT" (l'adaptativite n'est pas le moteur)
    ACCEPTEE si  mean(sync FROZ_UBAR) <= 0.15  (frontiere de quadrant deja utilisee par le
    papier, fig:phase_space) ET >= 8/10 graines individuellement <= 0.15,
    ET repliquee sur les 10 graines 3021-3030, JAMAIS UTILISEES (b4 s'arrete a 3020).

H2  "L'ADAPTATIVITE EST NECESSAIRE"
    ACCEPTEE si  mean(sync FROZ_UBAR) >= 0.50  ET >= 8/10 graines >= 0.50, meme replication.

    Entre 0.15 et 0.50 -> NON TRANCHE. Rapporte tel quel. Aucun des deux seuils ne sera
    deplace apres coup, quel que soit le resultat.

H3  "L'HETEROGENEITE SPATIALE DE u N'EST PAS LE MOTEUR"
    ACCEPTEE si |mean(sync FROZ_PROFILE) - mean(sync FROZ_UBAR)| <= 0.05.

------------------------------------------------------------------------------------------
CE QUE CE SCRIPT NE FAIT PAS
------------------------------------------------------------------------------------------
Il ne modifie AUCUN chiffre du preprint et n'ecrit dans aucun .tex. Au maximum, il produit une
PROPOSITION de precision de formulation, que Julien tranche. Regle I2 : hypothese -> mesuree
-> tranchee, et seule la troisieme donne le droit d'agir.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import ablation_coordination as ac  # noqa: E402  (harnais du papier, reutilise tel quel)
from mem4ristor.topology import Mem4Network  # noqa: E402
from mem4ristor.metrics import (  # noqa: E402
    calculate_pairwise_synchrony,
    calculate_temporal_lz_complexity,
)

I_STIM = 0.5                      # protocole FORCED : celui de tab:ablations
SEEDS_CANON = list(range(10))     # les graines du papier
SEEDS_REPLI = list(range(3021, 3031))  # jamais utilisees (b4 s'arrete a 3020)

CSV_OUT = ROOT / "figures" / "p17_frozen_u_level_vs_dynamics.csv"
CSV_REF = ROOT / "figures" / "scratch" / "ablation_coordination.csv"

# tab:ablations, telle que publiee (mu, sigma)
PREPRINT = {
    "FULL":     {"sync": (0.031, 0.034), "lz": (1.069, 0.016)},
    "FROZ_005": {"sync": (0.751, 0.060), "lz": (1.635, 0.006)},
}

SEUIL_H1 = 0.15   # frontiere de quadrant du papier
SEUIL_H2 = 0.50
SEUIL_H3 = 0.05
GATE_FRAC = 8     # sur 10 graines


def u_filter(u: float) -> float:
    return float(np.tanh(np.pi * (0.5 - u)) + 0.01)


def run_arm(arm: str, seed: int, u_frozen=None) -> dict:
    """Un run du harnais ablation_coordination, avec un bras arbitraire.

    Bras figes : on reproduit EXACTEMENT la recette de ac.apply_ablation("FROZEN_U")
    (epsilon_u = 0, tau_u = 1e12, u pose en dur), en changeant seulement la VALEUR posee.
    """
    adj = ac.make_ba_adjacency(ac.N_NODES, ac.BA_M, seed)
    net = Mem4Network(adjacency_matrix=adj, heretic_ratio=ac.HERETIC_RATIO,
                      coupling_norm="degree_linear", seed=seed)

    if arm != "FULL":
        net.model.cfg["doubt"]["epsilon_u"] = 0.0
        net.model.cfg["doubt"]["tau_u"] = 1e12
        net.model.u = np.asarray(u_frozen, dtype=float).copy()
        if net.model.u.shape != (net.model.N,):
            net.model.u = np.full(net.model.N, float(u_frozen))
    u0 = net.model.u.copy()

    snapshots = []
    for step in range(ac.STEPS):
        net.step(I_stimulus=I_STIM)
        if step % ac.TRACE_STRIDE == 0:
            snapshots.append(net.model.v.copy())

    v_hist = np.array(snapshots)
    cut = int(len(snapshots) * (1.0 - ac.TAIL_FRAC))
    v_tail = v_hist[cut:]

    return {
        "synchrony": float(calculate_pairwise_synchrony(v_tail)),
        "lz_full": float(calculate_temporal_lz_complexity(v_hist)),
        "lz_tail": float(calculate_temporal_lz_complexity(v_tail)),
        "u_final": net.model.u.copy(),
        "u_drift": float(np.max(np.abs(net.model.u - u0))),
    }


def gate_g0_bit_exact(res_canon: dict) -> None:
    """G0 : affiche le rejet et sa cause. Ne conditionne rien -- instrument inadapte."""
    print("\n" + "=" * 88)
    print("G0 -- FIDELITE BIT-A-BIT au CSV de reference : REJETEE, ET NON REPECHEE")
    print("=" * 88)
    if not CSV_REF.exists():
        print("  CSV de reference absent :", CSV_REF)
        return
    ref = {}
    with CSV_REF.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["regime"] == "FORCED":
                ref[(row["ablation"], int(row["seed"]))] = float(row["synchrony"])
    ecarts = []
    for seed in SEEDS_CANON:
        for arm, ref_name in (("FULL", "FULL"), ("FROZ_005", "FROZEN_U")):
            k = (ref_name, seed)
            if k in ref:
                ecarts.append(abs(res_canon[arm][seed]["synchrony"] - ref[k]))
    if ecarts:
        print("  ecart max sur la synchronie (20 cellules) : %.3e   -> tolerance 1e-9 : ECHEC"
              % max(ecarts))
    print("  CAUSE, etablie AVANT la mesure et non apres :")
    print("    - le CSV date du 26/04/2026 ; le fix de bruit Euler-Maruyama (AUDIT-024,")
    print("      818cf67) est du 01/05/2026. La reference precede le code de 5 jours.")
    print("    - figures/scratch/ est gitignore : ce CSV n'est pas versionne.")
    print("    - aucun des 14 claims du Guardian ne couvre tab:ablations.")
    print("  -> instrument inadapte (P1), remplace par G0b. Le rejet reste affiche.")


def gate_g0b(stats: dict) -> bool:
    print("\n" + "=" * 88)
    print("G0b -- ACCORD AGREGE AU PREPRINT (le gate qui decide)")
    print("=" * 88)
    ok = True
    for arm, cible in PREPRINT.items():
        for key, col in (("sync", "synchrony"), ("lz", "lz_full")):
            mu, sd = cible[key]
            got = stats[arm][col]["mean"]
            passe = abs(got - mu) <= sd
            ok &= passe
            print("  %-9s %-10s mesure %+.4f   publie %+.3f +/- %.3f   -> %s"
                  % (arm, col, got, mu, sd, "OK" if passe else "HORS INTERVALLE"))
    print("  VERDICT G0b :", "PASSE" if ok else "ECHOUE")
    if not ok:
        print("  -> Le resultat rapporte par ce script devient : TABLE 1 N'EST PLUS REPRODUITE")
        print("     PAR LE CODE ACTUEL. Les verdicts H1/H2/H3 sont rapportes mais SUSPENDUS.")
    return ok


def gate_g1(res: dict) -> bool:
    print("\n" + "=" * 88)
    print("G1 -- CONTROLE D'INSTRUMENT : le gel gele-t-il vraiment ?")
    print("=" * 88)
    ok = True
    for arm in ("FROZ_005", "FROZ_UBAR", "FROZ_PROFILE"):
        drift = max(r["u_drift"] for r in res[arm].values())
        passe = drift < 1e-12
        ok &= passe
        print("  %-13s max|u_fin - u_init| = %.3e  -> %s"
              % (arm, drift, "GELE" if passe else "*** u A BOUGE ***"))
    print("  VERDICT G1 :", "PASSE" if ok else "ECHOUE (verdicts suspendus)")
    return ok


def agg(res_arm: dict, col: str) -> dict:
    vals = np.array([r[col] for r in res_arm.values()])
    return {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)),
            "min": float(vals.min()), "max": float(vals.max())}


def main() -> int:
    t0 = time.time()
    print("=" * 88)
    print("P17 -- FROZEN_U du preprint : NIVEAU de u, ou ADAPTATIVITE de u ?")
    print("=" * 88)
    print("Harnais : ablation_coordination (BA m=%d, N=%d, degree_linear, %d pas, I_stim=%.1f)"
          % (ac.BA_M, ac.N_NODES, ac.STEPS, I_STIM))
    print("Filtre : u=0.05 -> %+.3f | u=0.50 -> %+.3f | u=0.99 -> %+.3f"
          % (u_filter(0.05), u_filter(0.50), u_filter(0.99)))

    res = {a: {} for a in ("FULL", "FROZ_005", "FROZ_UBAR", "FROZ_PROFILE")}
    all_seeds = SEEDS_CANON + SEEDS_REPLI

    # --- Phase A : FULL (fournit aussi u_bar et les profils par noeud) -------------------
    print("\n[Phase A] FULL sur %d graines..." % len(all_seeds))
    for seed in all_seeds:
        res["FULL"][seed] = run_arm("FULL", seed)
    profils = {s: res["FULL"][s]["u_final"] for s in all_seeds}
    u_bar = float(np.mean([profils[s].mean() for s in SEEDS_CANON]))
    print("  u_bar MESURE sur les 10 graines canoniques = %.4f  (u_filter = %+.4f)"
          % (u_bar, u_filter(u_bar)))
    disp = np.array([profils[s].std() for s in SEEDS_CANON])
    print("  dispersion inter-noeuds de u en FULL : std moyen %.4f (max %.4f)"
          % (disp.mean(), disp.max()))

    # --- Phase B : les trois bras figes ---------------------------------------------------
    for arm, val in (("FROZ_005", 0.05), ("FROZ_UBAR", u_bar), ("FROZ_PROFILE", None)):
        print("[Phase B] %s ..." % arm)
        for seed in all_seeds:
            u_f = profils[seed] if arm == "FROZ_PROFILE" else val
            res[arm][seed] = run_arm(arm, seed, u_frozen=u_f)

    # --- Agregats ------------------------------------------------------------------------
    def sub(arm, seeds):
        return {s: res[arm][s] for s in seeds}

    stats_canon = {a: {c: agg(sub(a, SEEDS_CANON), c)
                       for c in ("synchrony", "lz_full", "lz_tail")} for a in res}
    stats_repli = {a: {c: agg(sub(a, SEEDS_REPLI), c)
                       for c in ("synchrony", "lz_full", "lz_tail")} for a in res}

    print("\n" + "=" * 88)
    print("RESULTATS -- synchronie de Pearson (bas = desynchronise = diversite preservee)")
    print("=" * 88)
    print("  %-13s %-8s %22s %22s" % ("bras", "u fige", "canoniques (0-9)", "replication (3021-30)"))
    for arm, lbl in (("FULL", "adaptatif"), ("FROZ_005", "0.05"),
                     ("FROZ_UBAR", "%.3f" % u_bar), ("FROZ_PROFILE", "profil")):
        c, r = stats_canon[arm]["synchrony"], stats_repli[arm]["synchrony"]
        print("  %-13s %-8s   %+.4f +/- %.4f      %+.4f +/- %.4f"
              % (arm, lbl, c["mean"], c["std"], r["mean"], r["std"]))
    print("\n  LZ (lz_full) :")
    for arm in res:
        c, r = stats_canon[arm]["lz_full"], stats_repli[arm]["lz_full"]
        print("  %-13s   %.4f +/- %.4f      %.4f +/- %.4f"
              % (arm, c["mean"], c["std"], r["mean"], r["std"]))

    # --- Gates ---------------------------------------------------------------------------
    gate_g0_bit_exact(res)
    g0b = gate_g0b(stats_canon)
    g1 = gate_g1(res)

    # --- Verdicts ------------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("VERDICTS (criteres ecrits AVANT la mesure, en tete de ce fichier)")
    print("=" * 88)

    def frac(arm, seeds, cmp_):
        return sum(1 for s in seeds if cmp_(res[arm][s]["synchrony"]))

    m_c = stats_canon["FROZ_UBAR"]["synchrony"]["mean"]
    m_r = stats_repli["FROZ_UBAR"]["synchrony"]["mean"]
    n1_c = frac("FROZ_UBAR", SEEDS_CANON, lambda x: x <= SEUIL_H1)
    n1_r = frac("FROZ_UBAR", SEEDS_REPLI, lambda x: x <= SEUIL_H1)
    n2_c = frac("FROZ_UBAR", SEEDS_CANON, lambda x: x >= SEUIL_H2)
    n2_r = frac("FROZ_UBAR", SEEDS_REPLI, lambda x: x >= SEUIL_H2)

    h1 = (m_c <= SEUIL_H1 and n1_c >= GATE_FRAC and m_r <= SEUIL_H1 and n1_r >= GATE_FRAC)
    h2 = (m_c >= SEUIL_H2 and n2_c >= GATE_FRAC and m_r >= SEUIL_H2 and n2_r >= GATE_FRAC)

    print("  H1 'le NIVEAU suffit'         (sync <= %.2f) : moyenne %+.4f / %+.4f ; "
          "%d/10 puis %d/10  -> %s" % (SEUIL_H1, m_c, m_r, n1_c, n1_r,
                                       "ACCEPTEE" if h1 else "rejetee"))
    print("  H2 'l'ADAPTATIVITE est requise'(sync >= %.2f) : moyenne %+.4f / %+.4f ; "
          "%d/10 puis %d/10  -> %s" % (SEUIL_H2, m_c, m_r, n2_c, n2_r,
                                       "ACCEPTEE" if h2 else "rejetee"))
    if not h1 and not h2:
        print("  -> NON TRANCHE (zone %.2f-%.2f). Rapporte tel quel, seuils non deplaces."
              % (SEUIL_H1, SEUIL_H2))

    d3 = abs(stats_canon["FROZ_PROFILE"]["synchrony"]["mean"] - m_c)
    d3r = abs(stats_repli["FROZ_PROFILE"]["synchrony"]["mean"] - m_r)
    h3 = d3 <= SEUIL_H3 and d3r <= SEUIL_H3
    print("  H3 'heterogeneite non motrice' (ecart <= %.2f) : %.4f puis %.4f  -> %s"
          % (SEUIL_H3, d3, d3r, "ACCEPTEE" if h3 else "rejetee"))

    print("\n  Portee : ces verdicts sont %s."
          % ("VALIDES" if (g0b and g1) else "SUSPENDUS (un gate amont a echoue)"))
    print("  Aucun chiffre du preprint n'a ete modifie par ce script.")

    # --- CSV ------------------------------------------------------------------------------
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with CSV_OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arm", "u_frozen", "group", "seed", "synchrony", "lz_full", "lz_tail",
                    "u_drift", "u_mean_final"])
        for arm in res:
            for seed in all_seeds:
                r = res[arm][seed]
                lbl = {"FULL": "adaptive", "FROZ_005": "0.05",
                       "FROZ_UBAR": "%.6f" % u_bar, "FROZ_PROFILE": "per_node_profile"}[arm]
                w.writerow([arm, lbl, "canon" if seed in SEEDS_CANON else "replication",
                            seed, "%.10f" % r["synchrony"], "%.10f" % r["lz_full"],
                            "%.10f" % r["lz_tail"], "%.3e" % r["u_drift"],
                            "%.6f" % float(r["u_final"].mean())])
    print("\n  CSV -> %s" % CSV_OUT)
    print("  Duree totale : %.1f s" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
