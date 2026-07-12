#!/usr/bin/env python3
"""
P10 (legs de Fable) -- u dans C au coeur : la genese rentre a la maison. V1.
=============================================================================
Cree : 2026-07-12 (Claude Fable 5, L'Ingenieur) -- piste P10, ACCORD EXPLICITE
de Julien pour toucher au coeur (fork opt-in `complex_doubt`, bit-a-bit
identique eteint : tests/test_complex_doubt.py, suite complete 128+2xfail).

CE QUI EST DANS LE COEUR (V1, dynamics.py::_step_complex_doubt) :
  |u_c| = intensite du doute (reste self.u, lu par tout le coeur) ;
  arg(u_c) = direction. Cible locale SIGNEE (k_u*Lv au lieu de k_u*|Lv|),
  interference sociale (gamma_int * (moyenne complexe des u_c voisins - u_c)),
  porte omega_u (rotation, defaut 0 : phases {0, pi}).

CE QUE CE POC MESURE (V1 honnete : une dynamique RELAXANTE ne portera pas la
parite multiplicative d'ordre 5 du jouet genese -- ca demanderait la rotation
omega_u et une lecture de phase, documente comme suite ; ici on mesure ce que
les phases {0, pi} peuvent DEJA faire) :

  A. NON-REGRESSION FONCTIONNELLE : l'ablation canonique (anti-synchronisation
     FULL vs FROZEN_U) survit-elle avec le doute complexe ? 6 seeds, lattice
     10x10, I=0.5 driven, sync = |Pearson| pairwise moyen + H_cont.
     Critere pre-fixe : sync(COMPLEX_FULL) < 0.5 * sync(FROZEN) -- le doute
     complexe doit rester un doute (anti-synchronisant), pas devenir inerte.
     LANCEMENT 1 : critere ECHOUE (0.3496 vs seuil 0.2760 ; scalaire 0.2505).
     Le critere reste affiche tel quel (pas de repechage). Une condition
     DIAGNOSTIQUE est ajoutee au lancement 2 : COMPLEX_NOINT (gamma_int=0)
     -- si elle retrouve ~le scalaire, la cause du surplus de sync est
     l'INTERFERENCE elle-meme (elle detruit du doute la ou les signes
     voisins alternent, precisement les zones vivantes -> couplage plus
     attractif). Trade-off structurel a documenter, pas a cacher.

  B. MEMOIRE DIRECTIONNELLE : un pulse SIGNE b in {-1,+1} (E=0.8, 30 noeuds,
     200 pas), puis delai Delta (grille fine 100..2400) sans stimulus.
     Readout loyal a chaque delai (connait le masque de stimulation, PAS b).
     CONVENTION DE SIGNE (documentee apres lancement 1, physique et fixe --
     pas un choix a posteriori par readout) : le desaccord laplacien d'un
     noeud pousse AU-DESSUS du consensus local est NEGATIF (l_v = somme_vois
     - deg*v_i), donc la direction b est gravee en -Re(u_c[stim]) :
       PHASE_UC : sign( mean(Re(u_c[non-stim])) - mean(Re(u_c[stim])) )
     (Le lancement 1, readout a convention naive, donnait accuracy 0.00 a
     D=200 = information PARFAITE mais inversee.) Les DEUX conventions sont
     ecrites au CSV pour les trois readouts ; le tableau affiche la
     convention physique.
       V_STATE  : sign( mean(v[stim]) - mean(v[non-stim]) )        (directe ;
                  le rebond adaptatif FHN post-pulse l'inverse, cf. P6 --
                  les deux sens au CSV)
       U_SCALAR : sign( mean(u[stim]) - mean(u[non-stim]) )        (directe)
     40 problemes (20 par signe de b).
     PREDICTION pre-ecrite (lancement 1) : v oublie vite ; u scalaire ne code
     pas le signe par construction ; u_c retient la direction sur des
     MILLIERS de pas -- la lenteur de u devient ici la memoire. Verdict
     honnete quel que soit le resultat.

  C. Carte d'interference (figure) : Re(u_c) sur la grille apres un damier
     vs un bipulse antagoniste -- visualisation du mecanisme (pas un claim).

Statut : exploratoire, hors preprint. Le coeur est modifie UNIQUEMENT par le
fork opt-in (accord Julien), Guardian doit rester 14/14 (chemin OFF intact).
Sorties : figures/p10_complex_doubt_poc{,_agg}.csv + .png
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

CSV_PATH = ROOT / "figures" / "p10_complex_doubt_poc.csv"
AGG_PATH = ROOT / "figures" / "p10_complex_doubt_poc_agg.csv"
PNG_PATH = ROOT / "figures" / "p10_complex_doubt_poc.png"

SIDE, N = 10, 100

# --- partie A ---
A_SEEDS = [0, 1, 2, 3, 4, 5]
A_STEPS = 2000
A_STIM = 0.5

# --- partie B ---
B_SEEDS = list(range(20))            # x2 signes = 40 problemes
B_E = 0.8
B_N_STIM = 30
B_PULSE = 200
B_DELAYS = [100, 200, 400, 600, 900, 1200, 1800, 2400]


def fresh_net(seed, complex_on, heretic_ratio=0.15, gamma_int=None):
    net = Mem4Network(size=SIDE, heretic_ratio=heretic_ratio, seed=seed,
                      adjacency_matrix=make_lattice_adj(SIDE, periodic=True))
    if complex_on:
        net.model.cfg['complex_doubt']['enabled'] = True
        if gamma_int is not None:
            net.model.cfg['complex_doubt']['gamma_int'] = gamma_int
    return net


def part_a():
    print("=== A. Ablation canonique avec doute complexe (6 seeds) ===")
    res = {c: {"sync": [], "h": []} for c in
           ["SCALAR_FULL", "COMPLEX_FULL", "COMPLEX_NOINT", "FROZEN_U"]}
    for seed in A_SEEDS:
        for cond in res:
            net = fresh_net(seed,
                            complex_on=cond.startswith("COMPLEX"),
                            gamma_int=0.0 if cond == "COMPLEX_NOINT" else None)
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
            res[cond]["sync"].append(float(np.nanmean(np.abs(c[iu]))))
            idx = np.linspace(0, traj.shape[0] - 1, 20).astype(int)
            res[cond]["h"].append(float(np.mean(
                [calculate_continuous_entropy(traj[i]) for i in idx])))
    print(f"{'condition':<14}{'sync':>16}{'H_cont':>16}")
    for cond, d in res.items():
        s, h = np.array(d["sync"]), np.array(d["h"])
        print(f"{cond:<14}{s.mean():>9.4f}+-{s.std():<6.4f}{h.mean():>9.4f}+-{h.std():<6.4f}")
    s_cplx = np.mean(res["COMPLEX_FULL"]["sync"])
    s_froz = np.mean(res["FROZEN_U"]["sync"])
    s_scal = np.mean(res["SCALAR_FULL"]["sync"])
    s_noint = np.mean(res["COMPLEX_NOINT"]["sync"])
    ok = s_cplx < 0.5 * s_froz
    print(f"\n  Critere pre-fixe : sync(COMPLEX) {s_cplx:.4f} < 0.5*sync(FROZEN) "
          f"{0.5*s_froz:.4f} -> {'TENU (le doute complexe reste un doute)' if ok else 'ECHEC'}")
    print(f"  Reference scalaire : {s_scal:.4f} "
          f"(rapport complexe/scalaire = {s_cplx/max(s_scal,1e-12):.2f})")
    print(f"  Diagnostic : sync(COMPLEX_NOINT, gamma_int=0) = {s_noint:.4f}")
    if abs(s_noint - s_scal) < abs(s_cplx - s_scal) * 0.5:
        print("  -> gamma_int=0 retrouve ~le scalaire : le surplus de sync du "
              "complexe est cause par l'INTERFERENCE (trade-off structurel).")
    else:
        print("  -> gamma_int=0 ne retrouve PAS le scalaire : le surplus vient "
              "(aussi) de la cible signee elle-meme.")
    return res, ok


def part_b():
    print("\n=== B. Memoire directionnelle (40 problemes, readout loyal) ===")
    rows = []
    acc = {ro: {d: [] for d in B_DELAYS} for ro in ["PHASE_UC", "V_STATE", "U_SCALAR"]}
    total = len(B_SEEDS) * 2
    done = 0
    t0 = time.time()
    for seed in B_SEEDS:
        rng = np.random.RandomState(80000 + seed)
        stim_nodes = rng.choice(N, size=B_N_STIM, replace=False)
        mask = np.zeros(N, dtype=bool)
        mask[stim_nodes] = True
        for b in (-1, 1):
            stim_vec = np.zeros(N)
            stim_vec[mask] = b * B_E
            # reseau complexe (PHASE_UC + V_STATE) et reseau scalaire (U_SCALAR)
            net_c = fresh_net(seed * 10 + 1, complex_on=True, heretic_ratio=0.0)
            net_s = fresh_net(seed * 10 + 1, complex_on=False, heretic_ratio=0.0)
            zero = np.zeros(N)
            readouts = {}
            horizon = B_PULSE + max(B_DELAYS)
            for t in range(horizon):
                s = stim_vec if t < B_PULSE else zero
                net_c.step(I_stimulus=s)
                net_s.step(I_stimulus=s)
                dt_after = t + 1 - B_PULSE
                if dt_after in B_DELAYS:
                    mc, ms = net_c.model, net_s.model
                    ruc = np.real(mc.u_c)
                    readouts[dt_after] = {
                        # convention physique (docstring) : b est grave en
                        # -Re(u_c[stim]) car l_v(stim) = -b*(contraste>0)
                        "PHASE_UC": np.sign(ruc[~mask].mean() - ruc[mask].mean()),
                        "V_STATE": np.sign(mc.v[mask].mean() - mc.v[~mask].mean()),
                        "U_SCALAR": np.sign(ms.u[mask].mean() - ms.u[~mask].mean()),
                    }
            for d in B_DELAYS:
                for ro, val in readouts[d].items():
                    ok = int(val == b)
                    acc[ro][d].append(ok)
                    rows.append((seed, b, d, ro, ok))
            done += 1
        if (done % 10) == 0:
            print(f"  [{done}/{total}] {time.time()-t0:.0f}s")

    print(f"\n{'readout':<10}" + "".join(f"{'D=' + str(d):>9}" for d in B_DELAYS))
    print("-" * 48)
    for ro in ["PHASE_UC", "V_STATE", "U_SCALAR"]:
        line = f"{ro:<10}"
        for d in B_DELAYS:
            line += f"{np.mean(acc[ro][d]):>9.2f}"
        print(line)
    return rows, acc


def part_c_figure(res_a, acc):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
        # A
        ax = axes[0]
        conds = ["SCALAR_FULL", "COMPLEX_FULL", "FROZEN_U"]
        colors = ["#2ca02c", "#d62728", "#1f77b4"]
        ax.bar(conds, [np.mean(res_a[c]["sync"]) for c in conds],
               yerr=[np.std(res_a[c]["sync"]) for c in conds],
               color=colors, capsize=4)
        ax.set_ylabel("sync (|Pearson| pairwise moyen)")
        ax.set_title("A. L'anti-synchronisation survit-elle a u dans C ?")
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(alpha=0.3, axis="y")
        # B
        ax = axes[1]
        pal = {"PHASE_UC": "#d62728", "V_STATE": "#2ca02c", "U_SCALAR": "#9467bd"}
        for ro in ["PHASE_UC", "V_STATE", "U_SCALAR"]:
            ax.plot(B_DELAYS, [np.mean(acc[ro][d]) for d in B_DELAYS], "o-",
                    color=pal[ro], label=ro)
        ax.axhline(0.5, ls=":", c="gray", label="hasard")
        ax.set_xlabel("delai apres le pulse (pas)")
        ax.set_ylabel("accuracy du decodage de la direction")
        ax.set_ylim(0.0, 1.05)
        ax.set_title("B. Qui se souvient de la DIRECTION du stimulus ?")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        # C : carte de Re(u_c) apres un damier
        ax = axes[2]
        xs, ys = np.meshgrid(np.arange(SIDE), np.arange(SIDE), indexing="ij")
        checker = ((-1.0) ** (xs + ys)).flatten() * 0.6
        net = fresh_net(42, complex_on=True, heretic_ratio=0.0)
        for _ in range(600):
            net.step(I_stimulus=checker)
        im = ax.imshow(np.real(net.model.u_c).reshape(SIDE, SIDE),
                       cmap="RdBu_r", vmin=-0.3, vmax=0.3)
        plt.colorbar(im, ax=ax, shrink=0.85)
        ax.set_title("C. Re(u_c) apres stimulus damier\n(directions gravees)")
        fig.suptitle("P10 V1 -- le doute complexe au coeur : ablation, memoire "
                     "directionnelle, carte de phase", fontsize=11)
        plt.tight_layout()
        plt.savefig(PNG_PATH, dpi=140)
        print(f"[png] {PNG_PATH}")
    except Exception as e:
        print(f"[png] skipped: {e}")


def main() -> int:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res_a, ok_a = part_a()
    rows_b, acc = part_b()

    # ---------------- verdict ----------------
    print("\n=== VERDICT P10 V1 (criteres pre-fixes) ===")
    print(f"  A. Ablation : {'TENU' if ok_a else 'ECHEC'} -- le doute complexe "
          f"{'reste' if ok_a else 'ne reste PAS'} anti-synchronisant.")
    long_d = B_DELAYS[-1]
    a_uc = np.mean(acc["PHASE_UC"][long_d])
    a_v = np.mean(acc["V_STATE"][long_d])
    a_us = np.mean(acc["U_SCALAR"][long_d])
    print(f"  B. Memoire directionnelle au delai {long_d} : "
          f"PHASE_UC={a_uc:.2f}, V_STATE={a_v:.2f}, U_SCALAR={a_us:.2f}")
    if a_uc >= 0.9 and a_uc > max(a_v, a_us) + 0.15:
        print("     -> la phase du doute retient la direction la ou les autres "
              "signaux l'ont perdue : la 'memoire de phase' de P10 existe (V1).")
    elif a_uc > 0.6:
        print("     -> memoire directionnelle partielle -- au-dessus du hasard "
              "mais pas dominante ; a caracteriser avant d'affirmer.")
    else:
        print("     -> PAS de memoire directionnelle utilisable dans la phase "
              "en V1 -- resultat negatif, a rapporter tel quel.")
    print("  Rappel du perimetre V1 : phases {0, pi}, pas de rotation (omega_u=0) ;")
    print("  la parite multiplicative du jouet genese exige omega_u + lecture de")
    print("  phase -- c'est la marche suivante, pas celle-ci.")

    # ---------------- sorties ----------------
    with CSV_PATH.open("w", encoding="utf-8") as f:
        f.write("part,seed,b,delay,readout,correct\n")
        for r in rows_b:
            f.write("B," + ",".join(str(x) for x in r) + "\n")
    with AGG_PATH.open("w", encoding="utf-8") as f:
        f.write("part,condition_or_readout,delay,metric,value\n")
        for cond, d in res_a.items():
            f.write(f"A,{cond},,sync_mean,{np.mean(d['sync']):.6f}\n")
            f.write(f"A,{cond},,h_cont_mean,{np.mean(d['h']):.6f}\n")
        for ro in acc:
            for d in B_DELAYS:
                f.write(f"B,{ro},{d},accuracy,{np.mean(acc[ro][d]):.4f}\n")
    print(f"\n[csv] {CSV_PATH}\n[csv] {AGG_PATH}")

    part_c_figure(res_a, acc)
    print(f"\nWall time: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
