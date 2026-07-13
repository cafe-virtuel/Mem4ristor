#!/usr/bin/env python3
"""
P10 -- GAMMA_INT, LA BATTERIE LARGE : pousser au maximum, meme sur des
criteres qui semblent hors de propos.
=============================================================================
Cree : 2026-07-13 (Claude Sonnet 5, L'Ingenieur). Apres 4 tests consecutifs
tous defavorables a gamma_int (memoire, crosstalk, diffusion, sagesse des
foules par signal faible), demande explicite de Julien : « pousse au
maximum, meme sur des criteres qui semblent totalement hors de propos ».
Ce script teste 4 angles GENUINEMENT DIFFERENTS, pas une repetition plus
profonde du meme protocole :

  PARTIE 1 -- REJET DE MENTEURS (la vraie sagesse des foules classique,
  differente du "signal faible" deja teste : ici une MINORITE de noeuds
  recoit un stimulus intentionnellement INVERSE, un "menteur", au lieu
  d'un stimulus simplement plus faible). gamma_int aide-t-il le groupe a
  ignorer les menteurs ?

  PARTIE 2 -- VITESSE DE CONSENSUS (le terrain NATIF de gamma_int : ce
  canal EST un mecanisme de synchronisation sociale. On sait deja qu'il
  synchronise plus -- teste ce matin comme un COUT pour l'anti-sync. Mais
  la synchronisation est-elle plus RAPIDE avec gamma_int, meme si elle
  n'est pas souhaitable pour P10 ? Critere de VITESSE, pas de justesse.)

  PARTIE 3 -- BRUIT AMPLIFIE (sigma_v x4, x8 : le regime ou la moyenne
  statistique a le plus de marge theorique pour aider, si elle aide un
  jour).

  PARTIE 4 -- TOPOLOGIE BA SCALE-FREE (le maillon faible connu du projet
  depuis B1/B4/B5 -- tout a ete teste sur lattice jusqu'ici. gamma_int
  se comporte-t-il differemment sur un graphe a hubs ?)

Statut : exploratoire, hors preprint, aucune modification de dynamics.py.
Guardian doit rester 14/14. Sorties : figures/p10_gamma_int_batterie_large_poc_{1,2,3,4}.csv + .png
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
from mem4ristor.graph_utils import make_lattice_adj, make_ba  # noqa: E402

FIG = ROOT / "figures"
SIDE, N = 10, 100
B_E = 0.8
GROUP_SIZE = 30
B_PULSE = 200
DELAY = 1200
GAMMA_VALUES = [0.0, 0.05, 0.15, 0.3, 0.5]


def decode_vote(u_c, sub_mask, idle):
    idle_ref = float(np.real(u_c[idle]).mean())
    votes = np.sign(idle_ref - np.real(u_c[sub_mask]))
    votes[votes == 0] = 1
    return 1 if votes.sum() >= 0 else -1


def decode_interference(u_c, sub_mask, idle):
    diff = u_c[idle].mean() - u_c[sub_mask].mean()
    return 1 if float(np.real(diff)) >= 0 else -1


# ----------------------------------------------------------------------------
# PARTIE 1 -- rejet de menteurs
# ----------------------------------------------------------------------------
N_LIARS = 5
SEEDS_1 = list(range(15))


def build_group_liars(seed):
    rng = np.random.RandomState(81000 + seed)
    mask_nodes = rng.choice(N, size=GROUP_SIZE, replace=False)
    mask = np.zeros(N, dtype=bool)
    mask[mask_nodes] = True
    idle = ~mask
    liar_sub = rng.choice(mask_nodes, size=N_LIARS, replace=False)
    liars = np.zeros(N, dtype=bool)
    liars[liar_sub] = True
    honest = mask & (~liars)
    return mask, idle, honest, liars


def run_liars(seed, b_true, gamma_int):
    mask, idle, honest, liars = build_group_liars(seed)
    net = Mem4Network(size=SIDE, heretic_ratio=0.0, seed=seed * 10 + 1,
                       adjacency_matrix=make_lattice_adj(SIDE, periodic=True))
    m = net.model
    m.cfg['complex_doubt']['enabled'] = True
    m.cfg['complex_doubt']['gamma_int'] = gamma_int
    stim_vec = np.zeros(N)
    stim_vec[honest] = b_true * B_E
    stim_vec[liars] = -b_true * B_E  # le menteur recoit le signal INVERSE
    zero = np.zeros(N)
    for t in range(B_PULSE + DELAY):
        net.step(I_stimulus=stim_vec if t < B_PULSE else zero)
    return m.u_c.copy(), mask, idle, honest, liars


def part1_liars():
    print("=== PARTIE 1 : REJET DE MENTEURS (5/30 noeuds inverses) ===")
    rows = []
    for g in GAMMA_VALUES:
        t0 = time.time()
        acc_group_vote, acc_group_int = [], []
        for seed in SEEDS_1:
            for b_true in (1, -1):
                u_c, mask, idle, honest, liars = run_liars(seed, b_true, g)
                acc_group_vote.append(int(decode_vote(u_c, mask, idle) == b_true))
                acc_group_int.append(int(decode_interference(u_c, mask, idle) == b_true))
        row = (g, float(np.mean(acc_group_vote)), float(np.mean(acc_group_int)))
        rows.append(row)
        print(f"  gamma_int={g:<5} groupe_avec_menteurs(vote/int)={row[1]:.3f}/{row[2]:.3f}  [{time.time()-t0:.0f}s]")
    best = max(rows, key=lambda r: r[2])
    ref = next(r for r in rows if r[0] == 0.0)
    print(f"  -> meilleur point (interference) : gamma_int={best[0]} = {best[2]:.3f} "
          f"vs gamma_int=0 = {ref[2]:.3f} (gain {best[2]-ref[2]:+.3f})")
    return rows


# ----------------------------------------------------------------------------
# PARTIE 2 -- vitesse de consensus (signal PARTAGE, pas de conflit)
# ----------------------------------------------------------------------------
SEEDS_2 = list(range(10))
CONSENSUS_THRESH = 0.05  # ecart-type de Re(u_c) intra-groupe sous ce seuil = "consensus atteint"


def run_consensus_trace(seed, gamma_int, max_steps=1500):
    """BUG initial corrige : tous les noeuds demarrent a u=0.05 IDENTIQUE
    (verifie -- std=1.4e-17 a t=0), donc "consensus" etait trivialement
    deja atteint avant toute dynamique. Fix (1) : etat initial u_c RANDOMISE
    (spread reel), PUIS stimulus PARTAGE. Fix (2), decouvert en verifiant le
    fix (1) : le spread ne DECROIT jamais vers un seuil bas dans le temps
    imparti (il CROIT, 0.29 -> 0.4-0.75, cf. journal de session) -- le
    groupe ne converge PAS vers un consensus a ce seuil, quel que soit
    gamma_int. La metrique "temps jusqu'au seuil" est donc mal posee (elle
    plafonne systematiquement a max_steps). Remplacee par le spread FINAL
    a un horizon fixe -- mesure ce qui est reellement mesurable."""
    rng = np.random.RandomState(82000 + seed)
    mask_nodes = rng.choice(N, size=GROUP_SIZE, replace=False)
    mask = np.zeros(N, dtype=bool)
    mask[mask_nodes] = True
    net = Mem4Network(size=SIDE, heretic_ratio=0.0, seed=seed * 10 + 1,
                       adjacency_matrix=make_lattice_adj(SIDE, periodic=True))
    m = net.model
    m.cfg['complex_doubt']['enabled'] = True
    m.cfg['complex_doubt']['gamma_int'] = gamma_int
    m.u = np.clip(rng.uniform(0.0, 1.0, size=N), *m.cfg['doubt']['u_clamp'])
    m.u_c = m.u.astype(complex)
    stim_vec = np.zeros(N)
    stim_vec[mask] = B_E
    for t in range(max_steps):
        net.step(I_stimulus=stim_vec)
    return float(np.std(np.real(m.u_c[mask])))


def part2_consensus_speed():
    print("\n=== PARTIE 2 : SPREAD FINAL (signal PARTAGE, terrain natif de gamma_int) ===")
    print("  [NOTE] Le seuil de convergence initial n'etait JAMAIS atteint (spread "
          "CROIT, ne decroit pas, depuis un etat initial randomise) -- metrique "
          "remplacee par le spread a horizon fixe (1500 pas).")
    rows = []
    for g in GAMMA_VALUES:
        t0 = time.time()
        spreads = [run_consensus_trace(seed, g) for seed in SEEDS_2]
        row = (g, float(np.mean(spreads)), float(np.std(spreads, ddof=1) / np.sqrt(len(spreads))))
        rows.append(row)
        print(f"  gamma_int={g:<5} spread final moyen = {row[1]:.3f} +/- {row[2]:.3f}  [{time.time()-t0:.0f}s]")
    ref = next(r for r in rows if r[0] == 0.0)
    best = min(rows, key=lambda r: r[1])
    print(f"  -> le plus homogene : gamma_int={best[0]} = {best[1]:.3f} vs "
          f"gamma_int=0 = {ref[1]:.3f}")
    if best[0] > 0.0 and best[1] < ref[1] - 0.05:
        print("     -> gamma_int>0 REDUIT reellement le spread final -- son terrain "
              "natif (homogeneiser) fonctionne, meme si ce n'est utile pour aucune "
              "tache de decodage testee aujourd'hui.")
    else:
        print("     -> Meme sur son propre terrain (reduire le spread sous un "
              "stimulus partage), gamma_int n'apporte rien de net ici.")
    return rows


# ----------------------------------------------------------------------------
# PARTIE 3 -- bruit amplifie
# ----------------------------------------------------------------------------
SEEDS_3 = list(range(15))
SIGMA_V_VALUES = [0.05, 0.20, 0.40]


def run_solo_noisy(seed, b_a, gamma_int, sigma_v):
    rng = np.random.RandomState(83000 + seed)
    mask_nodes = rng.choice(N, size=GROUP_SIZE, replace=False)
    mask = np.zeros(N, dtype=bool)
    mask[mask_nodes] = True
    idle = ~mask
    net = Mem4Network(size=SIDE, heretic_ratio=0.0, seed=seed * 10 + 1,
                       adjacency_matrix=make_lattice_adj(SIDE, periodic=True))
    m = net.model
    m.cfg['complex_doubt']['enabled'] = True
    m.cfg['complex_doubt']['gamma_int'] = gamma_int
    m.cfg['noise']['sigma_v'] = sigma_v
    stim_vec = np.zeros(N)
    stim_vec[mask] = b_a * B_E
    zero = np.zeros(N)
    for t in range(B_PULSE + DELAY):
        net.step(I_stimulus=stim_vec if t < B_PULSE else zero)
    return m.u_c.copy(), mask, idle


def part3_noise():
    print("\n=== PARTIE 3 : BRUIT AMPLIFIE (sigma_v x4, x8) ===")
    rows = []
    for sigma_v in SIGMA_V_VALUES:
        for g in GAMMA_VALUES:
            t0 = time.time()
            acc = []
            for seed in SEEDS_3:
                for b_a in (1, -1):
                    u_c, mask, idle = run_solo_noisy(seed, b_a, g, sigma_v)
                    acc.append(int(decode_interference(u_c, mask, idle) == b_a))
            row = (sigma_v, g, float(np.mean(acc)))
            rows.append(row)
            print(f"  sigma_v={sigma_v:<5} gamma_int={g:<5} accuracy={row[2]:.3f}  [{time.time()-t0:.0f}s]")
    print("  -> Pour chaque sigma_v, meilleur gamma_int :")
    for sigma_v in SIGMA_V_VALUES:
        sub = [r for r in rows if r[0] == sigma_v]
        best = max(sub, key=lambda r: r[2])
        ref = next(r for r in sub if r[1] == 0.0)
        print(f"     sigma_v={sigma_v}: gamma_int={best[1]} = {best[2]:.3f} vs gamma_int=0 = {ref[2]:.3f} (gain {best[2]-ref[2]:+.3f})")
    return rows


# ----------------------------------------------------------------------------
# PARTIE 4 -- topologie BA scale-free
# ----------------------------------------------------------------------------
SEEDS_4 = list(range(12))


def run_solo_ba(seed, b_a, gamma_int, adj):
    rng = np.random.RandomState(84000 + seed)
    mask_nodes = rng.choice(N, size=GROUP_SIZE, replace=False)
    mask = np.zeros(N, dtype=bool)
    mask[mask_nodes] = True
    idle = ~mask
    net = Mem4Network(size=SIDE, heretic_ratio=0.0, seed=seed * 10 + 1,
                       adjacency_matrix=adj.copy())
    m = net.model
    m.cfg['complex_doubt']['enabled'] = True
    m.cfg['complex_doubt']['gamma_int'] = gamma_int
    stim_vec = np.zeros(N)
    stim_vec[mask] = b_a * B_E
    zero = np.zeros(N)
    for t in range(B_PULSE + DELAY):
        net.step(I_stimulus=stim_vec if t < B_PULSE else zero)
    return m.u_c.copy(), mask, idle


def part4_ba_topology():
    print("\n=== PARTIE 4 : TOPOLOGIE BA SCALE-FREE (m=3) au lieu du lattice ===")
    adj = make_ba(N, m=3, seed=1)
    rows = []
    for g in GAMMA_VALUES:
        t0 = time.time()
        acc = []
        for seed in SEEDS_4:
            for b_a in (1, -1):
                u_c, mask, idle = run_solo_ba(seed, b_a, g, adj)
                acc.append(int(decode_interference(u_c, mask, idle) == b_a))
        row = (g, float(np.mean(acc)))
        rows.append(row)
        print(f"  gamma_int={g:<5} accuracy (BA m=3) = {row[1]:.3f}  [{time.time()-t0:.0f}s]")
    best = max(rows, key=lambda r: r[1])
    ref = next(r for r in rows if r[0] == 0.0)
    print(f"  -> meilleur point : gamma_int={best[0]} = {best[1]:.3f} vs gamma_int=0 = {ref[1]:.3f} (gain {best[1]-ref[1]:+.3f})")
    return rows


def main() -> int:
    t0 = time.time()
    FIG.mkdir(parents=True, exist_ok=True)

    r1 = part1_liars()
    with (FIG / "p10_gamma_int_batterie_large_poc_1_liars.csv").open("w", encoding="utf-8") as f:
        f.write("gamma_int,acc_vote,acc_interference\n")
        for r in r1:
            f.write(",".join(f"{x:.6f}" if isinstance(x, float) else str(x) for x in r) + "\n")

    r2 = part2_consensus_speed()
    with (FIG / "p10_gamma_int_batterie_large_poc_2_consensus_speed.csv").open("w", encoding="utf-8") as f:
        f.write("gamma_int,mean_final_spread,se_final_spread\n")
        for r in r2:
            f.write(",".join(f"{x:.6f}" if isinstance(x, float) else str(x) for x in r) + "\n")

    r3 = part3_noise()
    with (FIG / "p10_gamma_int_batterie_large_poc_3_noise.csv").open("w", encoding="utf-8") as f:
        f.write("sigma_v,gamma_int,accuracy\n")
        for r in r3:
            f.write(",".join(f"{x:.6f}" if isinstance(x, float) else str(x) for x in r) + "\n")

    r4 = part4_ba_topology()
    with (FIG / "p10_gamma_int_batterie_large_poc_4_ba.csv").open("w", encoding="utf-8") as f:
        f.write("gamma_int,accuracy\n")
        for r in r4:
            f.write(",".join(f"{x:.6f}" if isinstance(x, float) else str(x) for x in r) + "\n")

    print("\n=== BILAN GLOBAL ===")
    print(f"(1) Menteurs      : {'GAIN' if max(x[2] for x in r1) > r1[0][2] + 0.05 else 'pas de gain net'}")
    print(f"(2) Homogeneiser  : {'gamma_int reduit le spread' if min(x[1] for x in r2) < r2[0][1] - 0.05 else 'pas de reduction nette'}")
    print(f"(3) Bruit ampl.   : {'GAIN a bruit eleve' if any(max(rr[2] for rr in r3 if rr[0]==sv) > next(rr for rr in r3 if rr[0]==sv and rr[1]==0.0)[2] + 0.05 for sv in SIGMA_V_VALUES) else 'pas de gain net'}")
    print(f"(4) Topologie BA  : {'GAIN' if max(x[1] for x in r4) > r4[0][1] + 0.05 else 'pas de gain net'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(11, 9))

        ax = axes[0, 0]
        ax.plot([x[0] for x in r1], [x[1] for x in r1], "o-", label="vote")
        ax.plot([x[0] for x in r1], [x[2] for x in r1], "s-", label="interference")
        ax.set_xlabel("gamma_int"); ax.set_ylabel("accuracy"); ax.set_ylim(0, 1.05)
        ax.set_title("1. Rejet de menteurs (5/30 inverses)"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

        ax = axes[0, 1]
        ax.errorbar([x[0] for x in r2], [x[1] for x in r2], yerr=[x[2] for x in r2], fmt="o-", color="#9467bd")
        ax.set_xlabel("gamma_int"); ax.set_ylabel("spread final (std Re(u_c), t=1500)")
        ax.set_title("2. Spread final sous stimulus partage (le terrain natif)"); ax.grid(alpha=0.3)

        ax = axes[1, 0]
        for sv in SIGMA_V_VALUES:
            sub = [r for r in r3 if r[0] == sv]
            ax.plot([r[1] for r in sub], [r[2] for r in sub], "o-", label=f"sigma_v={sv}")
        ax.set_xlabel("gamma_int"); ax.set_ylabel("accuracy"); ax.set_ylim(0, 1.05)
        ax.set_title("3. Bruit amplifie"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

        ax = axes[1, 1]
        ax.plot([x[0] for x in r4], [x[1] for x in r4], "o-", color="#d62728")
        ax.set_xlabel("gamma_int"); ax.set_ylabel("accuracy"); ax.set_ylim(0, 1.05)
        ax.set_title("4. Topologie BA scale-free (m=3)"); ax.grid(alpha=0.3)

        fig.suptitle("P10 -- gamma_int, batterie large (4 criteres disparates)", fontsize=12)
        plt.tight_layout()
        plt.savefig(FIG / "p10_gamma_int_batterie_large_poc.png", dpi=140)
        print(f"\n[png] {FIG / 'p10_gamma_int_batterie_large_poc.png'}")
    except Exception as e:
        print(f"[png] skipped: {e}")

    print(f"\nWall time total: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
