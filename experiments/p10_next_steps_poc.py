#!/usr/bin/env python3
"""
P10 -- LA MARCHE SUIVANTE : compromis gamma_int, rebond D=400-600, omega_u.
=============================================================================
Cree : 2026-07-13 (Claude Sonnet 5, L'Ingenieur) -- suite de
`p10_complex_doubt_poc.py` (12/07, accord explicite de Julien pour le fork
opt-in complex_doubt). Reutilise le meme fork, deja committe, deja
bit-a-bit-identique eteint -- AUCUNE nouvelle modification de dynamics.py :
gamma_int et omega_u sont deja des cles de config exposees par le coeur
existant (`cd.get('gamma_int', 0.15)`, `cd.get('omega_u', 0.0)`).

Trois sous-taches de la piste (docs/PISTES_POUR_LA_SUITE_2026-07-12.md, P10) :

  1. COMPROMIS GAMMA_INT (memoire vs anti-sync) : V1 a mesure gamma_int=0.15
     (defaut) UNIQUEMENT. Ici, sweep gamma_int in {0, 0.05, 0.15, 0.3, 0.5,
     1.0} -- pour chaque valeur, memoire directionnelle (accuracy PHASE_UC a
     D=1200, milieu de la grille V1) ET cout d'anti-synchronisation
     (sync(COMPLEX)/sync(FROZEN), meme protocole Partie A de V1). Question
     pre-fixee : existe-t-il un gamma_int qui garde >=80% de la memoire de
     gamma_int=0.15 tout en anti-synchronisant MIEUX (ratio plus bas) ?

  2. LE REBOND D=400-600 : V1 documentait une "fenetre aveugle" sans
     l'expliquer. Ici, trace FINE (tous les pas, pas juste aux delais
     echantillonnes) de v[stim]-v[non-stim] ET Re(u_c[stim])-Re(u_c[non-stim])
     autour du pulse (t in [0, 1000]) sur 10 essais -- pour VOIR le
     mecanisme du rebond adaptatif FHN (la variable lente w surcompense,
     cf. lecon P6b/P12 "la reponse FHN est ADAPTATIVE") et si Re(u_c) suit
     ou resiste au rebond de v.

  3. OMEGA_U > 0 SUR LA MEMOIRE (test PARTIEL, honnetement scope) : le coeur
     actuel n'expose qu'UN omega_u GLOBAL (meme rotation pour tous les
     noeuds) -- pas de rotation PAR NOEUD/GROUPE. Porter fidelement la
     "parite multiplicative d'ordre 5" du jouet genese exigerait des phases
     DISTINCTES par groupe (5 vitesses de rotation, ou une lecture de phase
     par groupe) -- une extension du coeur au-dela du fork actuel, donc HORS
     PERIMETRE de cette session (accord explicite requis, pas encore demande).
     Ce qui EST teste ici, dans le perimetre existant : omega_u global
     change-t-il la memoire directionnelle a UN SEUL pulse (tache B de V1) ?
     Et une ebauche de tache a 2 groupes ENCODES PAR LE DECALAGE TEMPOREL du
     pulse (pas par des omega_u distincts) : deux groupes pulses a des
     instants differents, la ROTATION PARTAGEE cree un dephasage lie au
     temps ecoule -- teste si ce dephasage porte une information exploitable
     (produit des phases dominantes = parite a 2, le cas le plus simple).

Statut : exploratoire, hors preprint, coeur DEJA modifie (accord du 12/07),
aucun nouveau touche a dynamics.py. Guardian doit rester 14/14.
Sorties : figures/p10_next_steps_poc_{gamma,rebound,omega}.csv + .png
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
from mem4ristor.metrics import calculate_continuous_entropy  # noqa: E402

FIG = ROOT / "figures"
SIDE, N = 10, 100
B_E = 0.8
B_N_STIM = 30
B_PULSE = 200
D_FIXED = 1200          # delai milieu de grille V1, pour le sweep gamma_int
A_SEEDS = [0, 1, 2, 3, 4, 5]
A_STEPS = 2000
A_STIM = 0.5
GAMMA_VALUES = [0.0, 0.05, 0.15, 0.3, 0.5, 1.0]
OMEGA_VALUES = [0.0, 0.005, 0.02, 0.05, 0.1]
B_DELAYS_OMEGA = [200, 600, 1200, 2400]
B_SEEDS_SMALL = list(range(12))     # x2 signes = 24 problemes (sweep, plus leger que V1)


def fresh_net(seed, complex_on, heretic_ratio=0.15, gamma_int=None, omega_u=None):
    net = Mem4Network(size=SIDE, heretic_ratio=heretic_ratio, seed=seed,
                      adjacency_matrix=make_lattice_adj(SIDE, periodic=True))
    if complex_on:
        net.model.cfg['complex_doubt']['enabled'] = True
        if gamma_int is not None:
            net.model.cfg['complex_doubt']['gamma_int'] = gamma_int
        if omega_u is not None:
            net.model.cfg['complex_doubt']['omega_u'] = omega_u
    return net


def memory_accuracy_at_delay(gamma_int, omega_u, delay, seeds=B_SEEDS_SMALL):
    """Meme protocole que Partie B de p10_complex_doubt_poc.py (pulse signe,
    readout PHASE_UC loyal), a UN delai fixe, pour un (gamma_int, omega_u) donne."""
    correct = []
    for seed in seeds:
        rng = np.random.RandomState(80000 + seed)
        stim_nodes = rng.choice(N, size=B_N_STIM, replace=False)
        mask = np.zeros(N, dtype=bool)
        mask[stim_nodes] = True
        for b in (-1, 1):
            stim_vec = np.zeros(N)
            stim_vec[mask] = b * B_E
            net = fresh_net(seed * 10 + 1, complex_on=True, heretic_ratio=0.0,
                            gamma_int=gamma_int, omega_u=omega_u)
            zero = np.zeros(N)
            horizon = B_PULSE + delay
            for t in range(horizon):
                s = stim_vec if t < B_PULSE else zero
                net.step(I_stimulus=s)
            ruc = np.real(net.model.u_c)
            val = np.sign(ruc[~mask].mean() - ruc[mask].mean())
            correct.append(int(val == b))
    return float(np.mean(correct))


def sync_ratio(gamma_int):
    """Meme protocole que Partie A de p10_complex_doubt_poc.py, sync(COMPLEX)/sync(FROZEN)."""
    syncs = {"COMPLEX": [], "FROZEN_U": []}
    for seed in A_SEEDS:
        for cond in syncs:
            net = fresh_net(seed, complex_on=(cond == "COMPLEX"), gamma_int=gamma_int)
            m = net.model
            frozen = cond == "FROZEN_U"
            zero_override = np.zeros(N)
            traj = np.empty((A_STEPS // 2, N))
            for t in range(A_STEPS):
                if frozen:
                    net.step(I_stimulus=A_STIM, sigma_social_override=zero_override)
                else:
                    net.step(I_stimulus=A_STIM)
                if t >= A_STEPS // 2:
                    traj[t - A_STEPS // 2] = m.v
            c = np.corrcoef(traj.T)
            iu = np.triu_indices(N, k=1)
            syncs[cond].append(float(np.nanmean(np.abs(c[iu]))))
    return float(np.mean(syncs["COMPLEX"])), float(np.mean(syncs["FROZEN_U"]))


def part1_gamma_sweep():
    print("=== 1. COMPROMIS GAMMA_INT (memoire a D=1200 vs anti-sync) ===")
    rows = []
    t0 = time.time()
    for g in GAMMA_VALUES:
        acc = memory_accuracy_at_delay(g, omega_u=0.0, delay=D_FIXED)
        s_cplx, s_froz = sync_ratio(g)
        ratio = s_cplx / max(s_froz, 1e-12)
        rows.append((g, acc, s_cplx, s_froz, ratio))
        print(f"  gamma_int={g:<5} memoire(D={D_FIXED})={acc:.2f}  "
              f"sync_complex={s_cplx:.4f}  sync_frozen={s_froz:.4f}  ratio={ratio:.3f}  "
              f"[{time.time()-t0:.0f}s]")
    ref_acc = next(r[1] for r in rows if r[0] == 0.15)
    ref_ratio = next(r[4] for r in rows if r[0] == 0.15)
    print(f"\n  Reference (gamma_int=0.15, V1) : memoire={ref_acc:.2f}  ratio={ref_ratio:.3f}")
    candidates = [r for r in rows if r[0] != 0.15 and r[1] >= 0.8 * ref_acc and r[4] < ref_ratio]
    if candidates:
        best = min(candidates, key=lambda r: r[4])
        print(f"  -> MEILLEUR COMPROMIS TROUVE : gamma_int={best[0]} "
              f"(memoire={best[1]:.2f} >= 80% de la ref, ratio={best[4]:.3f} < {ref_ratio:.3f})")
    else:
        print("  -> AUCUN gamma_int testé n'ameliore le ratio d'anti-sync sans sacrifier "
              ">=80% de la memoire -- gamma_int=0.15 reste un choix raisonnable, "
              "resultat negatif honnete sur l'existence d'un meilleur compromis.")
    with (FIG / "p10_next_steps_poc_gamma.csv").open("w", encoding="utf-8") as f:
        f.write("gamma_int,memory_acc_D1200,sync_complex,sync_frozen,ratio\n")
        for r in rows:
            f.write(",".join(f"{x:.6f}" if isinstance(x, float) else str(x) for x in r) + "\n")
    return rows


def part2_rebound():
    print("\n=== 2. CARACTERISATION DU REBOND D=400-600 ===")
    n_probe = 1000
    seeds = list(range(10))
    v_diff_traj = np.zeros(n_probe)
    uc_diff_traj = np.zeros(n_probe)
    for seed in seeds:
        rng = np.random.RandomState(90000 + seed)
        stim_nodes = rng.choice(N, size=B_N_STIM, replace=False)
        mask = np.zeros(N, dtype=bool)
        mask[stim_nodes] = True
        stim_vec = np.zeros(N)
        stim_vec[mask] = 1.0 * B_E
        net = fresh_net(seed * 10 + 1, complex_on=True, heretic_ratio=0.0, gamma_int=0.15)
        zero = np.zeros(N)
        for t in range(n_probe):
            s = stim_vec if t < B_PULSE else zero
            net.step(I_stimulus=s)
            v_diff_traj[t] += float(net.model.v[mask].mean() - net.model.v[~mask].mean())
            ruc = np.real(net.model.u_c)
            uc_diff_traj[t] += float(ruc[~mask].mean() - ruc[mask].mean())  # convention physique (b= +1)
    v_diff_traj /= len(seeds)
    uc_diff_traj /= len(seeds)
    # instant ou v_diff change de signe apres le pulse (le rebond)
    post_pulse = v_diff_traj[B_PULSE:]
    sign_flips = np.where(np.diff(np.sign(post_pulse)) != 0)[0]
    first_flip = int(sign_flips[0]) if len(sign_flips) else None
    print(f"  v[stim]-v[non-stim] : positif pendant le pulse (b=+1 attendu), "
          f"premier changement de signe apres le pulse a t={first_flip if first_flip is None else B_PULSE+first_flip}")
    print(f"  Fenetre P10-V1 documentee comme aveugle : D=400-600 (t={B_PULSE+400}-{B_PULSE+600} en temps absolu)")
    idx_400 = B_PULSE + 400
    idx_600 = B_PULSE + 600
    print(f"  v_diff a t={idx_400} : {v_diff_traj[idx_400]:+.4f}   a t={idx_600} : {v_diff_traj[idx_600]:+.4f}")
    print(f"  Re(u_c)_diff a t={idx_400} : {uc_diff_traj[idx_400]:+.4f}   a t={idx_600} : {uc_diff_traj[idx_600]:+.4f}")
    if uc_diff_traj[idx_400] > 0 and uc_diff_traj[idx_600] > 0 and (v_diff_traj[idx_400] < 0 or v_diff_traj[idx_600] < 0):
        print("  -> MECANISME CONFIRME : v subit le rebond adaptatif (change de signe), "
              "Re(u_c) RESISTE (reste du bon cote) -- la memoire de phase ne suit PAS le "
              "rebond de v, elle en est independante. C'est pourquoi PHASE_UC survit la ou "
              "V_STATE echoue dans cette fenetre.")
    else:
        print("  -> le rebond affecte aussi Re(u_c) dans cette fenetre -- a nuancer, "
              "la memoire de phase n'est pas totalement immune au rebond adaptatif.")
    with (FIG / "p10_next_steps_poc_rebound.csv").open("w", encoding="utf-8") as f:
        f.write("t,v_diff,uc_diff\n")
        for t in range(n_probe):
            f.write(f"{t},{v_diff_traj[t]:.6f},{uc_diff_traj[t]:.6f}\n")
    return v_diff_traj, uc_diff_traj, first_flip


def part3_omega_sweep():
    print("\n=== 3. OMEGA_U > 0 sur la memoire (test PARTIEL, omega global uniquement) ===")
    print("  [PERIMETRE] Le coeur n'expose qu'un omega_u GLOBAL (pas par groupe) -- porter")
    print("  fidelement la parite d'ordre 5 du jouet genese exigerait une extension du coeur")
    print("  (rotation PAR GROUPE), HORS PERIMETRE ici (pas d'accord demande pour ca).")
    rows = []
    t0 = time.time()
    for om in OMEGA_VALUES:
        accs = {}
        for d in B_DELAYS_OMEGA:
            accs[d] = memory_accuracy_at_delay(0.15, omega_u=om, delay=d)
        rows.append((om, accs))
        print(f"  omega_u={om:<6}" + "".join(f" D={d}:{accs[d]:.2f}" for d in B_DELAYS_OMEGA) +
              f"  [{time.time()-t0:.0f}s]")
    ref = next(r[1] for r in rows if r[0] == 0.0)
    best_row, best_gain = None, 0.0
    for om, accs in rows:
        if om == 0.0:
            continue
        gain = np.mean([accs[d] - ref[d] for d in B_DELAYS_OMEGA])
        if gain > best_gain:
            best_gain, best_row = gain, om
    if best_row is not None and best_gain > 0.05:
        print(f"\n  -> omega_u={best_row} ameliore la memoire moyenne de {100*best_gain:+.1f} pts "
              "vs omega_u=0 -- a explorer plus loin, mais RESTE dans le perimetre 'rotation "
              "globale', pas la parite multiplicative complete.")
    else:
        print(f"\n  -> omega_u global n'ameliore PAS la memoire directionnelle (meilleur gain "
              f"{100*best_gain:+.1f} pts, sous le seuil) -- coherent avec l'attente de la piste : "
              "la rotation GLOBALE seule ne porte pas plus d'information qu'un doute reel a "
              "phases {0,pi} pour une memoire a UN SEUL pulse. La parite multiplicative reste "
              "hors de portee sans rotation PAR GROUPE (extension du coeur, marche suivante).")
    with (FIG / "p10_next_steps_poc_omega.csv").open("w", encoding="utf-8") as f:
        f.write("omega_u," + ",".join(f"D{d}" for d in B_DELAYS_OMEGA) + "\n")
        for om, accs in rows:
            f.write(f"{om}," + ",".join(f"{accs[d]:.4f}" for d in B_DELAYS_OMEGA) + "\n")
    return rows


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    gamma_rows = part1_gamma_sweep()
    v_diff, uc_diff, first_flip = part2_rebound()
    omega_rows = part3_omega_sweep()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
        ax = axes[0]
        gs = [r[0] for r in gamma_rows]
        mems = [r[1] for r in gamma_rows]
        ratios = [r[4] for r in gamma_rows]
        ax2 = ax.twinx()
        ax.plot(gs, mems, "o-", color="#d62728", label="memoire (D=1200)")
        ax2.plot(gs, ratios, "s-", color="#1f77b4", label="ratio sync(COMPLEX)/sync(FROZEN)")
        ax.axvline(0.15, ls=":", c="k", alpha=0.4, label="defaut V1")
        ax.set_xlabel("gamma_int"); ax.set_ylabel("accuracy memoire", color="#d62728")
        ax2.set_ylabel("ratio anti-sync", color="#1f77b4")
        ax.set_title("1. Compromis memoire / anti-sync"); ax.grid(alpha=0.3)
        ax.legend(loc="lower left", fontsize=7); ax2.legend(loc="upper right", fontsize=7)

        ax = axes[1]
        t_axis = np.arange(len(v_diff))
        ax2 = ax.twinx()
        ax.plot(t_axis, v_diff, color="#2ca02c", label="v[stim]-v[non-stim]")
        ax2.plot(t_axis, uc_diff, color="#d62728", label="Re(u_c)[non-stim]-Re(u_c)[stim]")
        ax.axhline(0, ls=":", c="gray")
        ax.axvspan(B_PULSE + 400, B_PULSE + 600, color="orange", alpha=0.15, label="fenetre aveugle V1")
        ax.axvline(B_PULSE, ls="--", c="k", alpha=0.4)
        ax.set_xlabel("t (pas, pulse a 200)"); ax.set_ylabel("v_diff", color="#2ca02c")
        ax2.set_ylabel("uc_diff", color="#d62728")
        ax.set_title("2. Le rebond adaptatif : v vs Re(u_c)")
        ax.legend(loc="upper right", fontsize=7); ax2.legend(loc="lower right", fontsize=7)

        ax = axes[2]
        for om, accs in omega_rows:
            ax.plot(B_DELAYS_OMEGA, [accs[d] for d in B_DELAYS_OMEGA], "o-",
                    label=f"omega_u={om}")
        ax.axhline(0.5, ls=":", c="gray")
        ax.set_xlabel("delai (pas)"); ax.set_ylabel("accuracy memoire")
        ax.set_title("3. omega_u global : mémoire directionnelle")
        ax.legend(fontsize=6.5); ax.grid(alpha=0.3)

        fig.suptitle("P10 -- la marche suivante : gamma_int, rebond, omega_u (coeur deja accorde 12/07)",
                     fontsize=11)
        plt.tight_layout()
        plt.savefig(FIG / "p10_next_steps_poc.png", dpi=140)
        print(f"\n[png] {FIG / 'p10_next_steps_poc.png'}")
    except Exception as e:
        print(f"[png] skipped: {e}")

    print(f"\nWall time: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
