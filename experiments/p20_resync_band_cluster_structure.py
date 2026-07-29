#!/usr/bin/env python3
"""
P20 -- Que se passe-t-il dans la BANDE DE RE-SYNCHRONISATION ?
(29 juillet 2026, Claude Opus 5 -- le fait non cherche de P19, repique et non explique.)

------------------------------------------------------------------------------------------
LE FAIT A EXPLIQUER (P19, replique sur graines 3041-3050)
------------------------------------------------------------------------------------------
A fort stimulus, en allant vers le couplage repulsif, la synchronie de Pearson descend,
REMONTE, puis redescend :

  u fige      0.50    0.55    0.65    0.80    0.95
  u_filter   +0.01   -0.15   -0.43   -0.73   -0.88
  canon       0.284   0.163   0.310   0.282   0.047     (I_stim = 1.0)
  replication 0.277   0.162   0.308   0.276   0.078
Presente aussi a I_stim = 0.75. Ce n'est pas du bruit ; ce n'est pas explique.

------------------------------------------------------------------------------------------
L'HYPOTHESE, ET POURQUOI ELLE EST PLAUSIBLE
------------------------------------------------------------------------------------------
La synchronie rapportee partout dans ce projet est la moyenne des correlations de Pearson sur
TOUTES les paires de noeuds. Cette moyenne ne distingue pas "tout le monde ensemble" de "deux
groupes en anti-phase". Or si le reseau se scinde en deux groupes de tailles TRES INEGALES
(disons 85 contre 15), les paires intra-groupe -- toutes correlees positivement -- dominent
numeriquement les paires inter-groupe, et la moyenne REMONTE alors meme que le reseau est
plus structure, pas moins.

  H_CLUSTERS : dans la bande, le reseau forme deux groupes en anti-phase de tailles inegales.
               Ce serait la signature "chimera-like" que le preprint mentionne.

------------------------------------------------------------------------------------------
LE DISPOSITIF
------------------------------------------------------------------------------------------
Harnais ablation_coordination (BA m=3, N=100, degree_linear, 3000 pas), u FIGE, huit cellules
choisies pour encadrer la bande et fournir leurs propres controles :

  (I=1.00, u=0.05)  attractif fort   -- tout le monde ensemble, controle "pas de groupes"
  (I=1.00, u=0.50)  filtre ~ 0
  (I=1.00, u=0.55)  LE CREUX         -- controle discriminant
  (I=1.00, u=0.65)  LA BANDE
  (I=1.00, u=0.80)  LA BANDE
  (I=1.00, u=0.95)  APRES la bande
  (I=0.75, u=0.55)  le creux, autre stimulus
  (I=0.75, u=0.65)  la bande, autre stimulus

PARTITION, definie avant de regarder : signe du PREMIER VECTEUR PROPRE de la matrice de
correlation des trajectoires (fenetre de queue, 25 % finaux). Methode non supervisee standard
pour deux groupes en anti-phase ; elle ne connait pas la reponse attendue.

On mesure alors : r_intra (moyenne des paires dans le meme groupe), r_inter (paires de groupes
differents), et le desequilibre min(n1,n2)/N.

------------------------------------------------------------------------------------------
CRITERES, ECRITS AVANT LA MESURE (29/07/2026)
------------------------------------------------------------------------------------------
G2  (CONTROLE D'IMPLEMENTATION, gratuit) La decomposition doit etre EXACTE :
    r_global = [n1(n1-1)*r_in1 + n2(n2-1)*r_in2 + 2*n1*n2*r_inter] / [N(N-1)]
    a 1e-9 pres. Si elle ne l'est pas, le code de partition est faux et tout est suspendu.
    (Cette identite est vraie par construction -- c'est precisement pourquoi elle ne peut PAS
    servir de test de l'hypothese, seulement de test du code.)

C1  "LA BANDE EST UN ETAT A DEUX GROUPES EN ANTI-PHASE"
    ACCEPTEE si, dans les deux cellules de bande a I=1.00, r_intra >= +0.30 ET r_inter <= -0.10
    sur >= 8/10 graines canoniques, ET replique >= 8/10 sur les graines 3051-3060 (jamais
    utilisees : P17 a pris 3021-3030, P18 3031-3040, P19 3041-3050).

C2  "C'EST LE DESEQUILIBRE DES TAILLES QUI FAIT REMONTER LA SYNCHRONIE"  (test CONTREFACTUEL)
    On recalcule la synchronie globale qu'on obtiendrait avec les MEMES r_intra et r_inter mais
    des groupes EQUILIBRES (n1 = n2 = N/2).
    ACCEPTEE si, dans la bande, r_global_equilibre <= +0.05 alors que r_global observe >= +0.25,
    sur >= 8/10 graines, et replique.
    -> si elle passe : la remontee est un ARTEFACT DE MOYENNE sur des groupes inegaux, pas un
       retour a la synchronisation. Si elle echoue, l'hypothese des clusters ne suffit pas.

C3  (CONTROLE DISCRIMINANT) Le creux (u = 0.55) NE DOIT PAS avoir la meme structure que la
    bande, sinon l'explication ne distingue rien.
    ACCEPTEE si, dans le creux, r_intra - r_inter < 0.20 (pas de structure de groupes) OU
    desequilibre min(n1,n2)/N >= 0.40 (groupes equilibres).

Aucun seuil ne sera deplace apres coup. Aucun .tex n'est touche.
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
from mem4ristor.topology import Mem4Network  # noqa: E402

SEEDS_CANON = list(range(10))
SEEDS_REPLI = list(range(3051, 3061))
CELLS = [
    (1.00, 0.05, "attractif fort"),
    (1.00, 0.50, "filtre ~ 0"),
    (1.00, 0.55, "LE CREUX"),
    (1.00, 0.65, "LA BANDE"),
    (1.00, 0.80, "LA BANDE"),
    (1.00, 0.95, "apres la bande"),
    (0.75, 0.55, "le creux (I=0.75)"),
    (0.75, 0.65, "la bande (I=0.75)"),
]
BANDE = [(1.00, 0.65), (1.00, 0.80)]
CREUX = [(1.00, 0.55), (0.75, 0.55)]
CSV_OUT = ROOT / "figures" / "p20_resync_band_cluster_structure.csv"


def u_filter(u: float) -> float:
    return float(np.tanh(np.pi * (0.5 - u)) + 0.01)


def analyse(v_tail: np.ndarray) -> dict:
    """Partition par le signe du premier vecteur propre de la matrice de correlation."""
    C = np.corrcoef(v_tail.T)                 # (N, N)
    C = np.nan_to_num(C, nan=0.0)
    N = C.shape[0]
    iu = np.triu_indices(N, k=1)
    r_global = float(C[iu].mean())

    w, V = np.linalg.eigh(C)
    lead = V[:, int(np.argmax(w))]
    g = lead >= 0
    n1, n2 = int(g.sum()), int((~g).sum())

    def mean_block(mask_a, mask_b, same):
        sub = C[np.ix_(mask_a, mask_b)]
        if same:
            k = sub.shape[0]
            if k < 2:
                return np.nan, 0
            iu2 = np.triu_indices(k, k=1)
            return float(sub[iu2].mean()), k * (k - 1) // 2
        if sub.size == 0:
            return np.nan, 0
        return float(sub.mean()), sub.size

    r_in1, _ = mean_block(g, g, True)
    r_in2, _ = mean_block(~g, ~g, True)
    r_out, _ = mean_block(g, ~g, False)

    # decomposition exacte (G2)
    tot = N * (N - 1)
    parts = 0.0
    if n1 >= 2:
        parts += n1 * (n1 - 1) * r_in1
    if n2 >= 2:
        parts += n2 * (n2 - 1) * r_in2
    if n1 >= 1 and n2 >= 1:
        parts += 2.0 * n1 * n2 * r_out
    r_recomp = parts / tot

    # moyenne intra ponderee (pour C1/C2, un seul chiffre)
    w1 = n1 * (n1 - 1) if n1 >= 2 else 0
    w2 = n2 * (n2 - 1) if n2 >= 2 else 0
    r_intra = ((w1 * (r_in1 if w1 else 0.0) + w2 * (r_in2 if w2 else 0.0)) / (w1 + w2)
               if (w1 + w2) else np.nan)

    # contrefactuel C2 : memes correlations, groupes EQUILIBRES
    h = N // 2
    r_equilibre = (2 * h * (h - 1) * r_intra + 2 * h * h * r_out) / tot

    return {
        "r_global": r_global, "r_recomp": r_recomp, "r_intra": r_intra, "r_inter": r_out,
        "n_min": min(n1, n2), "desequilibre": min(n1, n2) / N,
        "r_equilibre": float(r_equilibre),
    }


def run_cell(seed: int, i_stim: float, u_val: float) -> dict:
    adj = ac.make_ba_adjacency(ac.N_NODES, ac.BA_M, seed)
    net = Mem4Network(adjacency_matrix=adj, heretic_ratio=ac.HERETIC_RATIO,
                      coupling_norm="degree_linear", seed=seed)
    net.model.cfg["doubt"]["epsilon_u"] = 0.0
    net.model.cfg["doubt"]["tau_u"] = 1e12
    net.model.u = np.full(net.model.N, float(u_val))

    snaps = []
    for step in range(ac.STEPS):
        net.step(I_stimulus=i_stim)
        if step % ac.TRACE_STRIDE == 0:
            snaps.append(net.model.v.copy())
    v_hist = np.array(snaps)
    cut = int(len(snaps) * (1.0 - ac.TAIL_FRAC))
    return analyse(v_hist[cut:])


def main() -> int:
    t0 = time.time()
    print("=" * 100)
    print("P20 -- la bande de re-synchronisation : deux groupes en anti-phase, de tailles "
          "inegales ?")
    print("=" * 100)

    res: dict = {}
    for i_stim, u, label in CELLS:
        print("[I=%.2f u=%.2f %-18s]" % (i_stim, u, label), end="", flush=True)
        res[(i_stim, u, "canon")] = [run_cell(s, i_stim, u) for s in SEEDS_CANON]
        res[(i_stim, u, "repli")] = [run_cell(s, i_stim, u) for s in SEEDS_REPLI]
        print(" ok")

    # ------------------------------------------------------------------ G2
    err = max(abs(r["r_global"] - r["r_recomp"])
              for lst in res.values() for r in lst)
    g2 = err < 1e-9
    print("\nG2 -- decomposition exacte (controle du CODE, pas de l'hypothese) : "
          "ecart max %.3e -> %s" % (err, "PASSE" if g2 else "ECHOUE"))

    def col(cell, grp, key):
        return np.array([r[key] for r in res[(cell[0], cell[1], grp)]])

    print("\n" + "=" * 100)
    print("STRUCTURE MESUREE (moyennes sur 10 graines canoniques)")
    print("=" * 100)
    print("  %-22s %8s %9s %9s %9s %8s %10s"
          % ("cellule", "filtre", "r_global", "r_intra", "r_inter", "n_min", "r_equilibre"))
    for i_stim, u, label in CELLS:
        c = (i_stim, u)
        print("  I=%.2f u=%.2f %-9s %+8.2f %+9.3f %+9.3f %+9.3f %8.1f %+10.3f"
              % (i_stim, u, label[:9], u_filter(u),
                 col(c, "canon", "r_global").mean(), col(c, "canon", "r_intra").mean(),
                 col(c, "canon", "r_inter").mean(), col(c, "canon", "n_min").mean(),
                 col(c, "canon", "r_equilibre").mean()))

    # ------------------------------------------------------------------ C1
    print("\n" + "=" * 100)
    print("VERDICTS (criteres ecrits avant la mesure)")
    print("=" * 100)
    c1_ok = True
    for c in BANDE:
        for grp, nom in (("canon", "canoniques"), ("repli", "replication")):
            n = int(((col(c, grp, "r_intra") >= 0.30) & (col(c, grp, "r_inter") <= -0.10)).sum())
            c1_ok &= n >= 8
            print("  C1 [I=%.2f u=%.2f] %-11s : %d/10 graines avec r_intra>=+0.30 ET "
                  "r_inter<=-0.10" % (c[0], c[1], nom, n))
    print("  C1 'deux groupes en anti-phase' -> %s" % ("ACCEPTEE" if c1_ok else "rejetee"))

    # ------------------------------------------------------------------ C2
    c2_ok = True
    for c in BANDE:
        for grp, nom in (("canon", "canoniques"), ("repli", "replication")):
            n = int(((col(c, grp, "r_equilibre") <= 0.05)
                     & (col(c, grp, "r_global") >= 0.25)).sum())
            c2_ok &= n >= 8
            print("  C2 [I=%.2f u=%.2f] %-11s : %d/10 (observe >= +0.25 ET contrefactuel "
                  "equilibre <= +0.05)" % (c[0], c[1], nom, n))
    print("  C2 'la remontee est un artefact de moyenne sur groupes inegaux' -> %s"
          % ("ACCEPTEE" if c2_ok else "rejetee"))

    # ------------------------------------------------------------------ C3
    c3_ok = True
    for c in CREUX:
        d = float(col(c, "canon", "r_intra").mean() - col(c, "canon", "r_inter").mean())
        des = float(col(c, "canon", "desequilibre").mean())
        ok = (d < 0.20) or (des >= 0.40)
        c3_ok &= ok
        print("  C3 [I=%.2f u=%.2f] creux : r_intra - r_inter = %.3f ; desequilibre = %.3f"
              "  -> %s" % (c[0], c[1], d, des, "distinct de la bande" if ok else "IDENTIQUE"))
    print("  C3 'le creux n'a pas la meme structure' -> %s" % ("ACCEPTEE" if c3_ok else "rejetee"))

    print("\n  Portee : verdicts %s." % ("VALIDES" if g2 else "SUSPENDUS (G2 est tombe)"))

    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with CSV_OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["i_stim", "u_frozen", "u_filter", "label", "group", "seed",
                    "r_global", "r_intra", "r_inter", "n_min", "desequilibre", "r_equilibre"])
        for i_stim, u, label in CELLS:
            for grp, seeds in (("canon", SEEDS_CANON), ("repli", SEEDS_REPLI)):
                for s, r in zip(seeds, res[(i_stim, u, grp)]):
                    w.writerow(["%.2f" % i_stim, "%.3f" % u, "%.6f" % u_filter(u), label,
                                grp, s, "%.8f" % r["r_global"], "%.8f" % r["r_intra"],
                                "%.8f" % r["r_inter"], r["n_min"],
                                "%.4f" % r["desequilibre"], "%.8f" % r["r_equilibre"]])
    print("  CSV -> %s" % CSV_OUT)
    print("  Duree : %.1f s" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
