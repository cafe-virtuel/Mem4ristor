#!/usr/bin/env python3
"""
P21 -- LA TRAVERSEE : que fait le reseau pendant que le doute monte a travers la bande ?
(29 juillet 2026, Claude Opus 5. Fil choisi librement, avec l'accord de Julien.)

------------------------------------------------------------------------------------------
POURQUOI CETTE QUESTION, ET POURQUOI C'EST LA SEULE QUI RESTE
------------------------------------------------------------------------------------------
Tout ce qui a ete mesure aujourd'hui dit la meme chose : le doute ADAPTATIF ne fait rien qu'un
u FIGE au bon niveau ne fasse pas.
  P17 : u fige a 0.997 desynchronise autant que FULL (10/10 puis 10/10).
  P18 : le seuil est une affaire de NIVEAU de couplage, pas d'adaptativite.
  P20b: FULL est reellement decorrele, comme u=0.95 fige.

Mais u ne SAUTE pas a 0.997 : il MONTE, de 0.05 jusqu'a ~0.997. Le filtre u_filter va donc de
+0.90 (attractif fort) a -0.91 (repulsif fort) EN PASSANT PAR la bande de re-synchronisation
que P19/P20 viennent de decouvrir (filtre -0.43 a -0.73, ou le reseau forme deux camps
cohesifs). Personne n'a jamais regarde ce passage.

C'est le seul endroit ou la DYNAMIQUE de u peut faire quelque chose que le fige ne fait pas :
un bras fige n'a pas de trajectoire, donc pas de traversee.

Vitesse, lue dans le code (dynamics.py:338) et non supposee :
    du = eps_eff * (k_u*sigma_local + sigma_baseline - u) / tau_u ,  u += du*dt , dt = 0.05
    eps_eff = 0.02 * clip(1 + 2*sigma_local, 1, 5)
-> tau_u GRAND = montee LENTE. Temps caracteristique tau_u/eps_eff ~ 100 a 500 unites de
   temps, soit 2000 a 10000 pas : la traversee occupe une large part du run, ce n'est pas un
   transitoire de demarrage.
NOTE : le preprint (Discussion, "Doubt time-scale and criticality") ecrit "for tau_u < 10,
doubt dynamics are too slow", ce qui est l'INVERSE du sens de la formule ci-dessus. Ce n'est
PAS une accusation : c'est la raison d'etre du controle T4, qui mesure le sens reel avant que
qui que ce soit n'ecrive quoi que ce soit.

------------------------------------------------------------------------------------------
LE DISPOSITIF
------------------------------------------------------------------------------------------
Harnais ablation_coordination (BA m=3, N=100, degree_linear), u ADAPTATIF (FULL) partout.
I_stim = 1.0 : la bande n'existe qu'a fort stimulus (P19 : absente a I=0.5, presente a 0.75
et 1.0). Trois vitesses : tau_u in {2, 10 (defaut), 50}, soit x5 et /5 autour du defaut.

Runs de 9000 pas (3x le standard) pour que MEME le bras le plus lent converge. Les 300
premiers instantanes correspondent exactement aux 3000 pas du harnais standard, ce qui donne
le gate de fidelite GRATUITEMENT (voir G3).

Le run est decoupe en 30 blocs de 30 instantanes (300 pas chacun). Dans chaque bloc on mesure :
u moyen, le filtre correspondant, la synchronie, et la structure en camps (partition par le
signe du premier vecteur propre, methode de P20 reutilisee telle quelle).

------------------------------------------------------------------------------------------
CRITERES, ECRITS AVANT LA MESURE (29/07/2026)
------------------------------------------------------------------------------------------
G3  (FIDELITE, gratuite) Sur les 300 premiers instantanes (= les 3000 pas du harnais
    standard), le bras tau_u = 10 doit reproduire la synchronie de queue de FULL a I=1.0
    mesuree par P20b (+0.061) a +/- 0.02. Sinon le harnais n'est pas celui des autres jours.

T4  (INSTRUMENT -- le levier fait-il ce qu'on croit ?) Le nombre de pas necessaires pour que
    u depasse 0.5 doit etre ORDONNE : t(tau=2) < t(tau=10) < t(tau=50), sur >= 9/10 graines.
    Si l'ordre est inverse ou brouille, le levier ne controle pas la vitesse et TOUS les
    verdicts ci-dessous sont suspendus.

T1  "LA TRAVERSEE LAISSE UNE REMONTEE TRANSITOIRE DE SYNCHRONIE"
    On cherche une REMONTEE LOCALE, pas un maximum : le reseau part synchronise (u=0.05 est le
    regime attractif), donc la synchronie decroit naturellement. Un pic global au debut ne
    prouverait rien.
    ACCEPTEE si, a tau_u = 10, il existe un bloc j >= 2 tel que
        sync(j) >= sync(j-1) + 0.05
    sur >= 8/10 graines canoniques ET >= 8/10 en replication (graines 3071-3080, jamais
    utilisees : P17->3021-3030, P18->3031-3040, P19->3041-3050, P20->3051-3060,
    P20b->3061-3070).

T2  "LA REMONTEE COINCIDE AVEC LA TRAVERSEE DE LA BANDE"
    Evaluee seulement si T1 passe. ACCEPTEE si le bloc de remontee maximale a un u_filter dans
    [-0.80, -0.30] -- la bande mesuree par P19/P20 -- sur >= 8/10 graines, repliquee.
    Si T1 passe mais pas T2, la remontee existe mais n'est PAS la bande : a rapporter tel quel.

T3  "LA VITESSE DE TRAVERSEE CHANGE L'ETAT FINAL"
    ACCEPTEE si, entre tau_u = 2 et tau_u = 50, l'etat final (dernier bloc) differe de
    |dsync| >= 0.05 OU |dLZ_intra| >= 0.10, sur >= 8/10 graines, repliquee.
    PRESOMPTION NEGATIVE ECRITE D'AVANCE, pour qu'on ne puisse pas me crediter d'une surprise
    fabriquee : P17 a montre qu'un SAUT direct a u=0.997 donne le meme etat final que FULL.
    Si T3 echoue, c'est une confirmation de plus que seul le niveau compte ; si elle passe,
    c'est le premier resultat de la journee ou la dynamique de u fait quelque chose de propre.

Aucun .tex touche, aucun chiffre publie modifie, coeur non touche (tau_u est un parametre de
configuration, deja balaye par le preprint lui-meme).
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

import ablation_coordination as ac  # noqa: E402
import p20_resync_band_cluster_structure as p20  # noqa: E402  (analyse reutilisee telle quelle)
from mem4ristor.topology import Mem4Network  # noqa: E402
from mem4ristor.metrics import calculate_pairwise_synchrony  # noqa: E402

I_STIM = 1.0
STEPS = 9000
STRIDE = ac.TRACE_STRIDE          # 10 -> 900 instantanes
BLOC = 30                          # 30 instantanes = 300 pas -> 30 blocs
TAUS = [2.0, 10.0, 50.0]
SEEDS_CANON = list(range(10))
SEEDS_REPLI = list(range(3071, 3081))
BANDE = (-0.80, -0.30)
CSV_OUT = ROOT / "figures" / "p21_doubt_traversal.csv"


def u_filter(u: float) -> float:
    return float(np.tanh(np.pi * (0.5 - u)) + 0.01)


def run_one(seed: int, tau_u: float) -> dict:
    adj = ac.make_ba_adjacency(ac.N_NODES, ac.BA_M, seed)
    net = Mem4Network(adjacency_matrix=adj, heretic_ratio=ac.HERETIC_RATIO,
                      coupling_norm="degree_linear", seed=seed)
    net.model.cfg["doubt"]["tau_u"] = float(tau_u)

    snaps, u_tr = [], []
    t_cross = None
    for step in range(STEPS):
        net.step(I_stimulus=I_STIM)
        if t_cross is None and float(net.model.u.mean()) > 0.5:
            t_cross = step
        if step % STRIDE == 0:
            snaps.append(net.model.v.copy())
            u_tr.append(float(net.model.u.mean()))

    v_hist = np.array(snaps)
    u_tr = np.array(u_tr)

    blocs = []
    n_blocs = len(snaps) // BLOC
    for b in range(n_blocs):
        sl = slice(b * BLOC, (b + 1) * BLOC)
        u_b = float(u_tr[sl].mean())
        st = p20.analyse(v_hist[sl])
        blocs.append({"bloc": b, "u": u_b, "filtre": u_filter(u_b),
                      "sync": st["r_global"], "r_intra": st["r_intra"],
                      "r_inter": st["r_inter"], "n_min": st["n_min"]})

    # fidelite : etat de queue tel que le calculerait le harnais standard a 3000 pas
    n_std = 3000 // STRIDE                       # 300 instantanes
    cut = int(n_std * (1.0 - ac.TAIL_FRAC))
    sync_std = float(calculate_pairwise_synchrony(v_hist[cut:n_std]))

    return {"blocs": blocs, "t_cross": t_cross, "sync_std_3000": sync_std,
            "u_final": float(u_tr[-1])}


def main() -> int:
    t0 = time.time()
    print("=" * 98)
    print("P21 -- LA TRAVERSEE : le reseau pendant que le doute monte a travers la bande")
    print("=" * 98)
    print("I_stim = %.1f (la bande n'existe qu'a fort stimulus) ; %d pas ; %d blocs de %d pas"
          % (I_STIM, STEPS, len(range(0, STEPS // STRIDE // BLOC)), BLOC * STRIDE))

    res = {}
    for tau in TAUS:
        for grp, seeds in (("canon", SEEDS_CANON), ("repli", SEEDS_REPLI)):
            print("[tau_u = %5.1f  %s]" % (tau, grp), end="", flush=True)
            res[(tau, grp)] = [run_one(s, tau) for s in seeds]
            print(" ok")

    # ------------------------------------------------------------------ G3
    sync_std = np.mean([r["sync_std_3000"] for r in res[(10.0, "canon")]])
    g3 = abs(sync_std - 0.061) <= 0.02
    print("\nG3 -- fidelite : synchronie de queue a 3000 pas, tau=10 : %+.4f "
          "(P20b : +0.061, tolerance 0.02) -> %s" % (sync_std, "PASSE" if g3 else "ECHOUE"))

    # ------------------------------------------------------------------ T4
    print("\nT4 -- le levier controle-t-il la vitesse ? (pas necessaires pour u > 0.5)")
    tc = {}
    for tau in TAUS:
        v = [r["t_cross"] for r in res[(tau, "canon")]]
        tc[tau] = v
        finis = [x for x in v if x is not None]
        print("   tau_u = %5.1f : mediane %s pas   (%d/10 graines franchissent 0.5)"
              % (tau, "%.0f" % np.median(finis) if finis else "jamais", len(finis)))
    ordre = 0
    for i in range(len(SEEDS_CANON)):
        a, b, c = tc[2.0][i], tc[10.0][i], tc[50.0][i]
        big = STEPS * 10
        a = big if a is None else a
        b = big if b is None else b
        c = big if c is None else c
        ordre += (a < b < c)
    t4 = ordre >= 9
    print("   ordre t(2) < t(10) < t(50) respecte sur %d/10 graines -> %s"
          % (ordre, "PASSE" if t4 else "ECHOUE"))
    if not t4:
        print("   -> le levier ne fait pas ce qu'on croit : verdicts T1-T3 SUSPENDUS.")

    # ------------------------------------------------------------------ trajectoire
    print("\n" + "=" * 98)
    print("LA TRAVERSEE, bloc par bloc (tau_u = 10, moyenne sur les 10 graines canoniques)")
    print("=" * 98)
    print("  bloc   pas       u    filtre     sync   intra   inter  n_min   dans la bande ?")
    ref = res[(10.0, "canon")]
    n_blocs = len(ref[0]["blocs"])
    for b in range(n_blocs):
        u = np.mean([r["blocs"][b]["u"] for r in ref])
        fl = np.mean([r["blocs"][b]["filtre"] for r in ref])
        sy = np.mean([r["blocs"][b]["sync"] for r in ref])
        ri = np.nanmean([r["blocs"][b]["r_intra"] for r in ref])
        ro = np.nanmean([r["blocs"][b]["r_inter"] for r in ref])
        nm = np.mean([r["blocs"][b]["n_min"] for r in ref])
        mark = "  <-- BANDE" if BANDE[0] <= fl <= BANDE[1] else ""
        print("  %4d %6d  %6.3f  %+7.3f  %+7.3f %+7.3f %+7.3f %6.1f%s"
              % (b, b * BLOC * STRIDE, u, fl, sy, ri, ro, nm, mark))

    # ------------------------------------------------------------------ T1 / T2
    def remontees(r):
        bl = r["blocs"]
        return [(j, bl[j]["sync"] - bl[j - 1]["sync"], bl[j]["filtre"])
                for j in range(2, len(bl))]

    print("\n" + "=" * 98)
    print("VERDICTS (criteres ecrits avant la mesure)")
    print("=" * 98)
    t1_n, t2_n = {}, {}
    for grp in ("canon", "repli"):
        n1 = n2 = 0
        for r in res[(10.0, grp)]:
            rr = remontees(r)
            hits = [x for x in rr if x[1] >= 0.05]
            if hits:
                n1 += 1
                j, d, fl = max(hits, key=lambda x: x[1])
                if BANDE[0] <= fl <= BANDE[1]:
                    n2 += 1
        t1_n[grp], t2_n[grp] = n1, n2
    t1 = t1_n["canon"] >= 8 and t1_n["repli"] >= 8
    print("  T1 'remontee transitoire de synchronie' : %d/10 puis %d/10  -> %s"
          % (t1_n["canon"], t1_n["repli"], "ACCEPTEE" if t1 else "rejetee"))
    if t1:
        t2 = t2_n["canon"] >= 8 and t2_n["repli"] >= 8
        print("  T2 'la remontee EST la bande'           : %d/10 puis %d/10  -> %s"
              % (t2_n["canon"], t2_n["repli"], "ACCEPTEE" if t2 else "rejetee"))
    else:
        print("  T2 non evaluee (T1 rejetee).")

    # ------------------------------------------------------------------ T3
    def final(r, key):
        return r["blocs"][-1][key]

    print("\n  T3 -- etat final selon la vitesse de traversee :")
    for tau in TAUS:
        s = np.mean([final(r, "sync") for r in res[(tau, "canon")]])
        ri = np.nanmean([final(r, "r_intra") for r in res[(tau, "canon")]])
        uf = np.mean([r["u_final"] for r in res[(tau, "canon")]])
        print("     tau_u = %5.1f : sync %+.4f   intra %+.4f   u final %.4f"
              % (tau, s, ri, uf))
    t3_n = {}
    for grp in ("canon", "repli"):
        n = 0
        for ra, rc in zip(res[(2.0, grp)], res[(50.0, grp)]):
            d_s = abs(final(ra, "sync") - final(rc, "sync"))
            d_i = abs(np.nan_to_num(final(ra, "r_intra")) - np.nan_to_num(final(rc, "r_intra")))
            n += (d_s >= 0.05) or (d_i >= 0.10)
        t3_n[grp] = n
    t3 = t3_n["canon"] >= 8 and t3_n["repli"] >= 8
    print("  T3 'la vitesse change l'etat final'     : %d/10 puis %d/10  -> %s"
          % (t3_n["canon"], t3_n["repli"], "ACCEPTEE" if t3 else "rejetee"))
    if not t3:
        print("     -> conforme a la presomption negative ecrite d'avance : seul le NIVEAU")
        print("        atteint compte, le chemin pour y arriver n'y change rien.")

    print("\n  Portee : verdicts %s."
          % ("VALIDES" if (g3 and t4) else "SUSPENDUS (un gate amont a echoue)"))

    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with CSV_OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tau_u", "group", "seed", "bloc", "step", "u", "u_filter",
                    "synchrony", "r_intra", "r_inter", "n_min", "t_cross_u05"])
        for tau in TAUS:
            for grp, seeds in (("canon", SEEDS_CANON), ("repli", SEEDS_REPLI)):
                for s, r in zip(seeds, res[(tau, grp)]):
                    for bl in r["blocs"]:
                        w.writerow(["%.1f" % tau, grp, s, bl["bloc"],
                                    bl["bloc"] * BLOC * STRIDE, "%.6f" % bl["u"],
                                    "%.6f" % bl["filtre"], "%.8f" % bl["sync"],
                                    "%.8f" % bl["r_intra"], "%.8f" % bl["r_inter"],
                                    bl["n_min"], r["t_cross"]])
    print("  CSV -> %s" % CSV_OUT)
    print("  Duree : %.1f s" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
