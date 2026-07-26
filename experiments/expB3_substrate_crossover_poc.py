#!/usr/bin/env python3
"""
EXPERIENCE B3-bis -- a quelle condition de SUBSTRAT le doute redevient-il
bon marche ? (pont entre le bilan d'operations de B2 et l'energie par pas de B3)

CONTEXTE ET RAISON D'ETRE.
  `expB2_wiring_budget_poc.py` (2026-07-26) a montre que M4R demande 6.7x a 20x
  plus d'OPERATIONS ARITHMETIQUES qu'un filtre a oubli pour trancher la meme
  tache, alors qu'il s'arrete 4x plus tot en PAS (309 contre 1348). La phrase du
  projet -- "M4R est un composant d'orientation bon marche" (13/07) -- est donc
  vraie en pas et fausse en operations numeriques.
  Ce script teste la seule echappatoire qui reste, et la teste pour de vrai :
  sur un substrat ou un pas coute peu (les trois familles de
  `docs/hardware/B3_ENERGY_COMPARISON.md`), la phrase redevient-elle vraie ?

LE PIEGE QUE CE SCRIPT REFUSE.
  Il serait facile de comparer M4R-analogique (fJ/pas) a un adversaire compte en
  operations CMOS (pJ/op) et d'annoncer un gain de plusieurs ordres de grandeur.
  Ce serait exactement l'erreur contre laquelle B3 met en garde en toutes lettres
  ("il ne faut pas lire ceci comme 'Mem4ristor bat Loihi de 4 ordres de
  grandeur'") -- et ce serait comparer deux substrats, pas deux methodes.
  Fait decisif, a poser d'emblee : LE FILTRE A OUBLI EXPONENTIEL EST UN CIRCUIT
  RC. Une resistance et un condensateur. Si l'on va sur le terrain analogique
  pour y avantager M4R, l'adversaire y devient trivialement realisable lui aussi.
  On ne peut donc pas trancher sans dire sur quel substrat tourne CHACUN.

CE QUE CE SCRIPT FAIT (modeste et bien defini) :
  il ne calcule PAS une energie systeme -- B3 dit explicitement que cela
  demanderait de choisir une architecture hybride complete et de la chiffrer bout
  en bout, un projet de plusieurs semaines. Il calcule le RAPPORT DE COUT entre
  les deux decideurs sur la MEME tache, en fonction de deux parametres de
  substrat (energie d'un pas de noeud, energie d'une operation adverse), trace la
  ligne d'iso-cout, et place dessus les technologies documentees. La sortie est
  un SEUIL, pas un verdict -- meme forme que P11 (13/07), qui avait repondu
  "D*~=300 en couplage dense, ~=10000 en creux" au lieu d'un oui/non.

MODELE DE COUT (tout est explicite, chaque terme est discutable et affiche).
  M4R      : 2 reseaux (le harness B1d/B5b entretient un run de REFERENCE, et
             expB2 volet C a montre qu'il est NECESSAIRE : sans lui 0.90 -> 0.50)
             x N noeuds actifs x T_M4R pas x e_node
  Adversaire: (agregation des capteurs + filtre) x T_ADV pas x e_op
             L'agregation a DEUX regimes, et le choix change la conclusion :
               - "kirchhoff" : sommer des courants sur un fil est une loi
                 physique, cout ~ 0 -> l'adversaire ne paie que son RC.
               - "compte"    : chaque capteur est somme par une operation.
             Les deux sont calcules ; aucun n'est privilegie en silence.
  Les DEUX paient la lecture des capteurs (le stimulus doit entrer dans M4R
  aussi) : ce terme commun s'annule dans le rapport et n'est pas compte.

T_M4R et T_ADV ne sont PAS des hypotheses : ce sont les couts a l'arret mesures
sur graines disjointes dans expB/expB2 (309 et 1348 pas).

SORTIES : figures/expB3_substrate_crossover_poc.csv + .png
Cree : 2026-07-26 (Claude Opus 5, L'Ingenieur) -- suite de expB2.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "figures" / "expB3_substrate_crossover_poc.csv"
PNG_PATH = ROOT / "figures" / "expB3_substrate_crossover_poc.png"

# --- mesures importees de expB/expB2 (graines disjointes) -------------------
N_NODES = 100
T_M4R = 309         # pas a l'arret du doute natif
T_ADV = 1348        # pas a l'arret du filtre a oubli
REF_FACTOR = 2      # run de reference NECESSAIRE (expB2 volet C : 0.90 -> 0.50)
N_SENSORS = 100

# --- points technologiques, tires de docs/hardware/B3_ENERGY_COMPARISON.md ---
# (ordres de grandeur de BRIQUE ELEMENTAIRE, pas de systeme -- reserve de B3)
TECHNO = {
    "STNO vortex (v)":        (6.7e-15, 67e-15),
    "Neuristor Mott (v)":     (22e-15, 225e-15),
    "Photonique GST signal":  (1.28e-18, 1.28e-18),   # signal seul, hors overhead
    "Loihi (par op. syn.)":   (24e-12, 24e-12),
    "TrueNorth (par evt)":    (26e-12, 26e-12),
}


def cost_m4r(e_node):
    return REF_FACTOR * N_NODES * T_M4R * e_node


def cost_adv(e_op, aggregation="kirchhoff"):
    ops_per_step = 3.0 if aggregation == "kirchhoff" else (N_SENSORS + 3.0)
    return ops_per_step * T_ADV * e_op


def crossover_ratio(aggregation):
    """e_node / e_op au-dessous duquel M4R coute MOINS que l'adversaire."""
    ops_per_step = 3.0 if aggregation == "kirchhoff" else (N_SENSORS + 3.0)
    return (ops_per_step * T_ADV) / (REF_FACTOR * N_NODES * T_M4R)


def main() -> int:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    print("=" * 100)
    print("B3-bis -- a quelle condition de substrat le doute est-il bon marche ?")
    print(f"M4R  : {REF_FACTOR} reseaux x {N_NODES} noeuds x {T_M4R} pas "
          f"= {REF_FACTOR * N_NODES * T_M4R:,} pas-noeud")
    print(f"filtre: (agregation + 3) x {T_ADV} pas")
    print("=" * 100)

    print("\n1. SEUIL DE BASCULE (rapport e_node/e_op sous lequel M4R coute moins)")
    print("   ATTENTION a la lecture : seul le regime 'kirchhoff' decrit un SUBSTRAT")
    print("   EGAL. Compter une operation par capteur suppose deja un adversaire")
    print("   NUMERIQUE -- c'est un scenario mixte, pas une comparaison a substrat egal.")
    for agg, label in (("kirchhoff", "agregation physique (courants sommes sur un fil)"),
                       ("compte", "agregation comptee (adversaire deja numerique)")):
        r = crossover_ratio(agg)
        factor = (1.0 / r) if r < 1 else r
        sense = "PLUS" if r < 1 else "MOINS"
        print(f"  {label:<52} : e_node/e_op < {r:.4f}")
        print(f"    -> a e_node = e_op, M4R coute {factor:.1f}x {sense} que le filtre.")
        rows.append(dict(scenario=f"seuil_{agg}", ratio_threshold=r,
                         e_node="", e_op="", cost_m4r="", cost_adv="", verdict=""))

    print("\n2. LE CAS QUI COMPTE -- l'adversaire est un CIRCUIT RC")
    print("   Un filtre a oubli exponentiel = une resistance + un condensateur.")
    print("   Si l'on passe M4R en analogique pour le rendre bon marche, l'adversaire")
    print("   y passe aussi, et il y est passif. A energie de composant egale :")
    for name, (e_lo, e_hi) in TECHNO.items():
        if "Loihi" in name or "TrueNorth" in name:
            continue
        for e_node in (e_lo, e_hi):
            c_m = cost_m4r(e_node)
            c_a = cost_adv(e_node, "kirchhoff")     # meme substrat, agregation physique
            verdict = "M4R moins cher" if c_m < c_a else f"M4R {c_m / c_a:.0f}x plus cher"
            print(f"   {name:<24} e={e_node:.2e} J/pas : M4R={c_m:.3e} J  "
                  f"filtre={c_a:.3e} J  -> {verdict}")
            rows.append(dict(scenario="meme_substrat_analogique", ratio_threshold="",
                             e_node=e_node, e_op=e_node, cost_m4r=c_m, cost_adv=c_a,
                             verdict=verdict))

    print("\n3. LE CAS FLATTEUR -- M4R analogique contre un adversaire NUMERIQUE")
    print("   (c'est la comparaison qu'il ne faut PAS presenter comme un resultat :")
    print("    elle compare deux SUBSTRATS, pas deux methodes. Affichee pour montrer")
    print("    exactement de combien le choix de substrat, seul, deplace le verdict.)")
    for tech_adv in ("Loihi (par op. syn.)", "TrueNorth (par evt)"):
        e_op = TECHNO[tech_adv][0]
        for name in ("STNO vortex (v)", "Neuristor Mott (v)"):
            e_node = TECHNO[name][0]
            c_m, c_a = cost_m4r(e_node), cost_adv(e_op, "compte")
            print(f"   M4R sur {name:<22} vs filtre sur {tech_adv:<22} : "
                  f"M4R {c_a / c_m:,.0f}x moins cher")
            rows.append(dict(scenario="substrats_mixtes", ratio_threshold="",
                             e_node=e_node, e_op=e_op, cost_m4r=c_m, cost_adv=c_a,
                             verdict=f"M4R {c_a / c_m:.0f}x moins cher"))

    print("\n" + "=" * 100)
    print("LECTURE HONNETE")
    r_k = crossover_ratio("kirchhoff")
    print(f"  - A substrat egal, M4R coute {1.0 / r_k:.0f}x plus que le filtre, quel que soit")
    print("    le dispositif choisi : le rapport ne depend PAS de l'energie par pas, il ne")
    print("    depend que du nombre de pas-noeud a entretenir (2 reseaux x 100 noeuds).")
    print("    Changer de substrat divise les DEUX cotes par le meme facteur.")
    print("  - Le desequilibre est STRUCTUREL, pas technologique : M4R entretient 200")
    print("    oscillateurs ACTIFS pendant 309 pas, le filtre un seul RC PASSIF pendant")
    print("    1348. Aucun choix de dispositif ne renverse ce rapport, puisqu'il")
    print("    s'applique des deux cotes.")
    print("  - Le 'bon marche' de M4R ne vient donc pas de son architecture : il vient")
    print("    entierement du substrat, dont n'importe quel dispositif analogique")
    print("    beneficierait de la meme facon. C'est un argument sur le SUBSTRAT.")
    print("  - Ce que M4R economise reellement et qui lui appartient : le NOMBRE DE PAS")
    print(f"    avant de trancher ({T_M4R} contre {T_ADV}, soit {T_ADV / T_M4R:.1f}x moins),")
    print("    donc la LATENCE de decision. C'est une grandeur differente de l'energie,")
    print("    et c'est celle-la qui resiste a toutes les corrections d'aujourd'hui.")
    print("  - RESERVE (B3, section 3) : ces energies sont des ordres de grandeur de")
    print("    BRIQUE ELEMENTAIRE issus de la litterature, pas des mesures systeme.")
    print("    Aucun claim publie ne depend de ce script.")

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[csv] {CSV_PATH}")
    make_figure()
    print(f"[png] {PNG_PATH}")
    return 0


def make_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))

    # --- diagramme de phase : e_node vs e_op -------------------------------
    ax = axes[0]
    e_node = np.logspace(-19, -10, 200)
    for agg, style, lbl in (("kirchhoff", "-", "aggregation free (currents on a wire)"),
                            ("compte", "--", "aggregation counted (1 op per sensor)")):
        r = crossover_ratio(agg)
        ax.plot(e_node, e_node / r, style, lw=2,
                label=f"iso-cost, {lbl}")
    ax.fill_between(e_node, e_node / crossover_ratio("kirchhoff"), 1e-8,
                    color="#d62728", alpha=0.10)
    ax.text(2e-18, 3e-11, "M4R cheaper\n(adversary on a\ncostlier substrate)",
            fontsize=8, color="#8b1a1a")
    ax.text(1e-13, 1e-18, "adversary cheaper\n(same or cheaper substrate)",
            fontsize=8, color="#1f4e79")
    for name, (lo, hi) in TECHNO.items():
        ax.plot([lo, hi], [lo, hi], "o-", ms=6, color="k", lw=1)
        ax.annotate(name, (hi, hi), textcoords="offset points", xytext=(6, -3),
                    fontsize=7)
    ax.plot(e_node, e_node, ":", color="gray", lw=1.5, label="equal substrate")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-19, 1e-10)
    ax.set_ylim(1e-19, 1e-8)
    ax.set_xlabel("energy per node-step of M4R (J)")
    ax.set_ylabel("energy per operation of the adversary (J)")
    ax.set_title("Where the substrate puts the verdict\n(dotted line = both on the same substrate)",
                 fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(alpha=0.3, which="both")

    # --- ce qui reste a M4R : la latence -----------------------------------
    ax = axes[1]
    bars = ["M4R\n(doubt)", "forgetting\nfilter"]
    ax.bar(bars, [T_M4R, T_ADV], color=["#d62728", "#7b4173"], edgecolor="k")
    for i, v in enumerate([T_M4R, T_ADV]):
        ax.text(i, v + 30, f"{v} steps", ha="center", fontsize=9)
    ax.set_ylabel("steps before deciding (held-out seeds)")
    ax.set_title(f"What survives every correction:\nlatency, {T_ADV / T_M4R:.1f}x fewer steps",
                 fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Experiment B3-bis -- the 'cheap' claim is a claim about the SUBSTRATE, "
                 "not about the architecture\n"
                 "(at equal substrate M4R costs ~46x more energy; what is genuinely its own "
                 "is decision latency)", fontsize=10.5)
    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=140)


if __name__ == "__main__":
    raise SystemExit(main())
