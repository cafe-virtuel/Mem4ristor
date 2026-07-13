#!/usr/bin/env python3
"""
P10 -- ROTATION PAR GROUPE : multiplexage frequentiel de deux memoires
directionnelles simultanees + lecture GLOBALE de parite (facon genese).
=============================================================================
Cree : 2026-07-13 (Claude Sonnet 5, L'Ingenieur). Suite de
`p10_next_steps_poc.py` (partie 3), qui concluait que omega_u GLOBAL (une
seule vitesse pour tout le reseau) n'ameliore pas la memoire directionnelle
a un seul pulse, et que la parite multiplicative de la genese (11/07,
`genesis_five_states_poc.py`) resterait hors de portee "sans rotation PAR
GROUPE (extension du coeur, marche suivante)". Accord explicite de Julien
le 13/07 ("P10 svp") pour cette extension. Le coeur ne bouge que d'UNE ligne
(`dynamics.py::_step_complex_doubt`, `omega_u = np.asarray(...)` au lieu de
`float(...)`) : un array de taille N est desormais accepte, bit-a-bit
identique au scalaire quand toutes les valeurs sont egales (verifie par
tests/test_complex_doubt.py::test_omega_scalar_equals_uniform_array).

QUESTION (falsifiable, pre-fixee avant de lancer) :
  Deux groupes de noeuds recoivent chacun un pulse SIGNE independant (bit
  b_A, b_B in {-1,+1}) EN MEME TEMPS. Le groupe A reste sur le canal V1
  (omega_A=0, ancre). Le groupe B recoit une rotation propre omega_B.
  (1) CROSSTALK -- decoder A degrade-t-il quand B est actif SUR LE MEME
      CANAL (omega_B=0, signal partage) ? Et est-ce REPARE en separant les
      frequences (omega_B>0, "canal" distinct via l'interference sociale
      gamma_int qui melange les voisins) ?
  (2) DECODE B -- un readout DE-ROTATE (Re(u_c * exp(-i*omega_B*T)),
      T=temps physique ecoule) recupere-t-il b_B, la ou le readout brut
      (sans de-rotation) echoue une fois omega_B != 0 ?
  (3) PARITE GLOBALE (le test genese) -- pour decoder la PARITE b_A*b_B,
      un readout GLOBAL (signe de cos(theta_A + theta_B_derot), le produit
      des phases dominantes, exactement la methode qui bat le vote dans
      genesis_five_states_poc.py) bat-il le decode SEPARE-PUIS-MULTIPLIE
      (signe(cos theta_A) * signe(cos theta_B_derot)) ?

Protocole : lattice 10x10 periodique, heretic_ratio=0.0 (comme la Partie B
de p10_complex_doubt_poc.py -- tache de memoire, pas de synchronisation).
60 noeuds tires au hasard par probleme (rng par seed), scindes en 2 groupes
de 30 ; les 40 restants = reference "idle" (jamais stimules, omega=0).
DELAY fixe a 1200 (repere "milieu de grille" de p10_next_steps_poc.py).
gamma_int laisse au defaut 0.15 (pas retouche : l'interference sociale
melange les voisins independamment du groupe, c'est la dynamique reelle,
pas une simplification a corriger).

Statut : exploratoire, hors preprint, extension du coeur committee
separement. Guardian doit rester 14/14 (aucun claim touche par ce fichier).
Sorties : figures/p10_group_rotation_poc.csv + .png
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from mem4ristor.topology import Mem4Network  # noqa: E402
from mem4ristor.graph_utils import make_lattice_adj  # noqa: E402

FIG = ROOT / "figures"
SIDE, N = 10, 100
B_E = 0.8
GROUP_SIZE = 30
B_PULSE = 200
DELAY = 1200
SEEDS = list(range(12))
SIGN_COMBOS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
OMEGA_B_VALUES = [0.0, 0.02, 0.05]


def build_groups(seed):
    rng = np.random.RandomState(70000 + seed)
    both = rng.choice(N, size=2 * GROUP_SIZE, replace=False)
    mask_a = np.zeros(N, dtype=bool)
    mask_b = np.zeros(N, dtype=bool)
    mask_a[both[:GROUP_SIZE]] = True
    mask_b[both[GROUP_SIZE:]] = True
    idle = ~(mask_a | mask_b)
    return mask_a, mask_b, idle


def run_problem(seed, b_a, b_b, omega_b, delay=DELAY, active_b=True):
    """Simule un probleme (A toujours actif, B actif sauf pour la baseline
    solo). Renvoie (u_c final, mask_a, mask_b, idle, T_physique)."""
    mask_a, mask_b, idle = build_groups(seed)
    net = Mem4Network(size=SIDE, heretic_ratio=0.0, seed=seed * 10 + 1,
                       adjacency_matrix=make_lattice_adj(SIDE, periodic=True))
    m = net.model
    m.cfg['complex_doubt']['enabled'] = True
    omega_arr = np.zeros(N)
    if active_b:
        omega_arr[mask_b] = omega_b
    m.cfg['complex_doubt']['omega_u'] = omega_arr

    stim_vec = np.zeros(N)
    stim_vec[mask_a] = b_a * B_E
    if active_b:
        stim_vec[mask_b] = b_b * B_E
    zero = np.zeros(N)
    horizon = B_PULSE + delay
    for t in range(horizon):
        s = stim_vec if t < B_PULSE else zero
        net.step(I_stimulus=s)
    T_phys = horizon * m.dt
    return m.u_c.copy(), mask_a, mask_b, idle, T_phys


def decode_theta(u_c, mask, idle, omega, T_phys):
    """Readout differentiel groupe-vs-idle, de-rotate par omega*T_phys
    (identite si omega=0 : de-rotation neutre, readout = V1 original)."""
    w = u_c * np.exp(-1j * omega * T_phys)
    diff = w[idle].mean() - w[mask].mean()
    return float(np.angle(diff))


def solo_accuracy():
    """Baseline V1 : groupe A SEUL (B jamais stimule), aucune rotation --
    reproduit le protocole original de p10_complex_doubt_poc.py Partie B."""
    correct = []
    for seed in SEEDS:
        for b_a in (1, -1):
            u_c, mask_a, mask_b, idle, T = run_problem(seed, b_a, 1, 0.0, active_b=False)
            theta_a = decode_theta(u_c, mask_a, idle, 0.0, T)
            correct.append(int(np.sign(np.cos(theta_a)) == b_a))
    return float(np.mean(correct))


def sweep():
    print("=== Baseline SOLO (groupe A seul, protocole V1 original) ===")
    acc_solo = solo_accuracy()
    print(f"  accuracy A (solo, D={DELAY}) = {acc_solo:.3f}\n")

    rows = []
    for omega_b in OMEGA_B_VALUES:
        t0 = time.time()
        acc_a, acc_b, acc_joint_sep, acc_joint_global = [], [], [], []
        for seed in SEEDS:
            for b_a, b_b in SIGN_COMBOS:
                u_c, mask_a, mask_b, idle, T = run_problem(seed, b_a, b_b, omega_b)
                theta_a = decode_theta(u_c, mask_a, idle, 0.0, T)
                theta_b = decode_theta(u_c, mask_b, idle, omega_b, T)

                dec_a = int(np.sign(np.cos(theta_a)))
                dec_b = int(np.sign(np.cos(theta_b)))
                parity_true = b_a * b_b
                parity_sep = dec_a * dec_b
                parity_global = int(np.sign(np.cos(theta_a + theta_b)))

                acc_a.append(int(dec_a == b_a))
                acc_b.append(int(dec_b == b_b))
                acc_joint_sep.append(int(parity_sep == parity_true))
                acc_joint_global.append(int(parity_global == parity_true))

        row = (omega_b, np.mean(acc_a), np.mean(acc_b),
               np.mean(acc_joint_sep), np.mean(acc_joint_global))
        rows.append(row)
        print(f"omega_b={omega_b:<5} acc_A={row[1]:.3f} (solo={acc_solo:.3f}, "
              f"delta={row[1]-acc_solo:+.3f})  acc_B={row[2]:.3f}  "
              f"parite_separee={row[3]:.3f}  parite_globale={row[4]:.3f}  "
              f"[{time.time()-t0:.0f}s]")

    print("\n=== VERDICTS (pre-fixes) ===")
    row0 = next(r for r in rows if r[0] == 0.0)
    best_sep = max(rows, key=lambda r: r[1])
    print(f"(1) CROSSTALK -- A solo={acc_solo:.3f} vs A+B canal partage "
          f"(omega_b=0)={row0[1]:.3f} (delta={row0[1]-acc_solo:+.3f}) vs "
          f"meilleur canal separe={best_sep[1]:.3f} (omega_b={best_sep[0]}, "
          f"delta vs solo={best_sep[1]-acc_solo:+.3f})")
    if row0[1] < acc_solo - 0.05 and best_sep[1] > row0[1] + 0.05:
        print("    -> Crosstalk REEL sur canal partage, ATTENUE par la separation "
              "de frequence.")
    elif row0[1] < acc_solo - 0.05:
        print("    -> Crosstalk REEL, mais la separation de frequence ne le repare "
              "PAS dans la plage testee.")
    else:
        print("    -> Pas de crosstalk mesurable meme sur canal partage (l'ancrage "
              "differentiel groupe-vs-idle suffit deja) -- la separation de "
              "frequence n'a rien a reparer ici.")

    best_b = max(rows, key=lambda r: r[2])
    print(f"(2) DECODE B -- accuracy a omega_b=0 (rien a de-rotate)={row0[2]:.3f} "
          f"vs meilleur omega_b>0={best_b[2]:.3f} (omega_b={best_b[0]})")

    best_global = max(rows, key=lambda r: r[4] - r[3])
    gap = best_global[4] - best_global[3]
    print(f"(3) PARITE -- au meilleur ecart (omega_b={best_global[0]}) : "
          f"separee={best_global[3]:.3f} vs globale={best_global[4]:.3f} "
          f"(delta={gap:+.3f})")
    if gap > 0.03:
        print("    -> Le readout GLOBAL (produit des phases) BAT le decode separe "
              "-- meme effet que la genese (11/07), porte au reseau physique.")
    elif gap < -0.03:
        print("    -> Le readout separe bat le global ICI -- l'effet genese ne se "
              "transfere PAS tel quel a ce substrat/protocole.")
    else:
        print("    -> Les deux methodes sont equivalentes ici (delta sous le bruit) "
              "-- pas de gain a lire globalement sur ce protocole.")

    FIG.mkdir(parents=True, exist_ok=True)
    with (FIG / "p10_group_rotation_poc.csv").open("w", encoding="utf-8") as f:
        f.write("omega_b,acc_A,acc_B,acc_parity_separate,acc_parity_global,acc_A_solo\n")
        for r in rows:
            f.write(",".join(f"{x:.6f}" if isinstance(x, float) else str(x) for x in r)
                     + f",{acc_solo:.6f}\n")
    return rows, acc_solo


def main() -> int:
    t0 = time.time()
    rows, acc_solo = sweep()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

        ax = axes[0]
        omegas = [r[0] for r in rows]
        ax.plot(omegas, [r[1] for r in rows], "o-", color="#d62728", label="acc A (avec B actif)")
        ax.axhline(acc_solo, ls="--", c="#d62728", alpha=0.5, label="acc A solo (ref V1)")
        ax.plot(omegas, [r[2] for r in rows], "s-", color="#1f77b4", label="acc B (de-rotate)")
        ax.set_xlabel("omega_b"); ax.set_ylabel("accuracy"); ax.set_ylim(0, 1.05)
        ax.set_title("Crosstalk A / decode B"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

        ax = axes[1]
        ax.plot(omegas, [r[3] for r in rows], "o-", color="#2ca02c", label="parite separee (decode puis x)")
        ax.plot(omegas, [r[4] for r in rows], "s-", color="#9467bd", label="parite globale (produit des phases)")
        ax.set_xlabel("omega_b"); ax.set_ylabel("accuracy parite"); ax.set_ylim(0, 1.05)
        ax.set_title("Parite b_A*b_B : separee vs globale (facon genese)")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

        fig.suptitle(f"P10 -- rotation par groupe (D={DELAY}, {len(SEEDS)} seeds x 4 signes)", fontsize=11)
        plt.tight_layout()
        plt.savefig(FIG / "p10_group_rotation_poc.png", dpi=140)
        print(f"\n[png] {FIG / 'p10_group_rotation_poc.png'}")
    except Exception as e:
        print(f"[png] skipped: {e}")

    print(f"\nWall time: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
