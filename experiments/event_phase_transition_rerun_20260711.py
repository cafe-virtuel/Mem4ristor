#!/usr/bin/env python3
"""
[13] RE-VERIFICATION au code actuel (grille complete) -- 2026-07-11
====================================================================
Cree : 2026-07-11 (Claude Fable 5, L'Ingenieur).

POURQUOI. Le claim [13] (transition de phase evenementielle : noeud peripherique
I>=0.8 pendant >=50 pas -> dH=+1.20 bits sur BA m=3, peripherique > hub) date
d'AVRIL, avant le changement de bruit du 1er mai (AUDIT-024 : scaling
Euler-Maruyama eta/sqrt(dt), x4.47 effectif -- commit 818cf67). Le 12/06,
photonic_event_poc.py a decouvert par accident que SA reference ELEC (une seule
configuration : I=1.5, T=150, peripherique, BA m=3) donne dH=-0.764 au code
actuel : SIGNE INVERSE. [13] a ete marque "A RE-VERIFIER (grille complete)"
dans CLAIMS_REGISTER (note S08) -- mais PROJECT_STATUS.md l'affiche encore
"Confirme" (incoherence a reparer avec ce re-run). 4e victime potentielle
d'AUDIT-024 apres C01/C04/C08 et le gamma sweep.

COMMENT (protocole STRICTEMENT identique, aucune reecriture). Ce wrapper
importe event_phase_transition.py et ne change QUE les chemins de sortie :
les CSV canoniques d'avril (figures/event_phase_transition*.csv) ne sont PAS
ecrases -- la lecon de C08 (11/06 : un re-run avait ecrase un CSV commite).
Ensuite il joint l'ancien et le nouveau summary sur (topologie, cible,
amplitude, duree) et rapporte les inversions de signe et l'etat du claim.

Le preprint ne cite pas [13] (verifie par grep le 12/06) : aucune consequence
de soumission, c'est une dette du registre interne.
Sorties : figures/event_phase_transition_rerun_20260711{.csv,_summary.csv,.png}
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import event_phase_transition as ept  # noqa: E402  (protocole d'avril, inchange)

# Rediriger les sorties -- ne jamais ecraser les CSV canoniques d'avril
ept.CSV_PATH = ept.FIG_DIR / "event_phase_transition_rerun_20260711.csv"
ept.SUM_PATH = ept.FIG_DIR / "event_phase_transition_rerun_20260711_summary.csv"
ept.FIG_PATH = ept.FIG_DIR / "event_phase_transition_rerun_20260711.png"

OLD_SUM = ROOT / "figures" / "event_phase_transition_summary.csv"


def load_summary(path):
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["topology"], row["target"],
                   float(row["i_event"]), int(float(row["t_event"])))
            out[key] = float(row["delta_H_mean"])
    return out


def compare():
    old = load_summary(OLD_SUM)
    new = load_summary(ept.SUM_PATH)
    print("\n" + "=" * 88)
    print("COMPARAISON AVRIL (ancien bruit, pre-818cf67) vs 2026-07-11 (code actuel)")
    print("=" * 88)
    print(f"{'config':<44}{'dH avril':>10}{'dH actuel':>11}{'signe':>10}")
    print("-" * 88)
    n_flip, n_same, n_total = 0, 0, 0
    claim_rows = []
    for key in sorted(old.keys()):
        if key not in new:
            continue
        o, n = old[key], new[key]
        n_total += 1
        flip = (o > 0.1 and n < -0.1) or (o < -0.1 and n > 0.1)
        same = (o > 0.1 and n > 0.1) or (o < -0.1 and n < -0.1) or (abs(o) <= 0.1 and abs(n) <= 0.1)
        if flip:
            n_flip += 1
        if same:
            n_same += 1
        topo, target, i_ev, t_ev = key
        label = f"{topo[:18]} {target:<10} I={i_ev:<4} T={t_ev:<4}"
        mark = "INVERSE" if flip else ("stable" if same else "ambigu")
        print(f"{label:<44}{o:>+10.3f}{n:>+11.3f}{mark:>10}")
        # les configs du claim [13] : peripherique BA m=3, I>=0.8, T>=50
        if topo.startswith("BA_m3") and target == "peripheral" and i_ev >= 0.8:
            claim_rows.append((key, o, n))

    print("-" * 88)
    print(f"Bilan : {n_total} configurations comparees, {n_flip} inversions de signe, "
          f"{n_same} stables.")

    print("\n--- LE CLAIM [13] LUI-MEME (peripherique BA m=3, I>=0.8) ---")
    pos_old = sum(1 for _, o, _ in claim_rows if o > 0.1)
    pos_new = sum(1 for _, _, n in claim_rows if n > 0.1)
    neg_new = sum(1 for _, _, n in claim_rows if n < -0.1)
    for (topo, target, i_ev, t_ev), o, n in claim_rows:
        print(f"  I={i_ev:<4} T={t_ev:<4} : avril {o:+.3f} -> actuel {n:+.3f}")
    print(f"\n  Avril : {pos_old}/{len(claim_rows)} configs a dH>+0.1 (bifurcation positive).")
    print(f"  Actuel: {pos_new}/{len(claim_rows)} positives, {neg_new}/{len(claim_rows)} negatives.")
    if pos_new == 0 and neg_new > len(claim_rows) // 2:
        print("  => [13] NE SE REPRODUIT PAS au code actuel : l'evenement DEGRADE H_cont")
        print("     (signe inverse). Le claim d'avril etait un artefact de l'ancien bruit.")
        print("     Action : PROJECT_STATUS/[13] et CLAIMS_REGISTER a mettre a jour (REVISE).")
    elif pos_new == len(claim_rows):
        print("  => [13] SE REPRODUIT au code actuel (la config unique du 12/06 etait")
        print("     l'exception). Action : lever la marque A RE-VERIFIER.")
    else:
        print("  => Resultat MIXTE : le claim d'avril (enonce general) ne tient pas tel quel ;")
        print("     certaines configs restent positives. A requalifier finement (pas binaire).")


if __name__ == "__main__":
    ret = ept.main()
    if ret == 0:
        compare()
    raise SystemExit(ret)
