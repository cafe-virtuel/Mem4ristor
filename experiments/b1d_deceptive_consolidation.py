#!/usr/bin/env python3
"""
CONSOLIDATION B1d -- robustesse du "le doute gagne sur tache trompeuse" (seeds x topologies).

Contexte. Le POC deceptive_task_poc.py (2026-07-07) a montre, sur LATTICE / 12 seeds, que sur une
tache TROMPEUSE (converger tot = se tromper) le DOUTE bat un critere de CONVERGENCE trivial de
+0.58. Reserve gravee : "12 echantillons (0.83=10/12) -> direction robuste, valeurs a N modeste ;
seed / lattice unique". Ce script consolide ce resultat -- le plus defendable du projet sur le
doute -- sur :
  - 30 seeds (au lieu de 12),
  - 3 topologies : LATTICE (regulier <k>=4), BA m=3 (scale-free), ER (aleatoire, <k>~6),
  - intervalle de confiance BOOTSTRAP (apparie par seed) sur l'ecart acc_DOUTE - acc_CONV.

Le message a verrouiller n'est PAS un chiffre exact mais une DIRECTION robuste : dans le regime
trompeur (leurre assez long), acc_DOUTE - acc_CONV > 0 avec un IC qui exclut 0, quelle que soit la
topologie. On garde aussi T_pulse=150 (piege trop court) comme controle negatif : la ou le leurre
ne dure pas, le doute ne doit PAS gagner (les deux echouent).

Substrat, readout differentiel, criteres d'arret DOUTE/CONVERGENCE : IDENTIQUES au POC (importes,
pas reimplementes) pour que la consolidation mesure exactement le meme mecanisme.

Sortie : figures/b1d_deceptive_consolidation.csv + .png + verdict.
Cree : 2026-07-08 (Claude Opus 4.8) -- consolidation Volet B1 (docs/FUTURE_WORK.md).
"""
from __future__ import annotations
import sys, time
from pathlib import Path
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from mem4ristor.graph_utils import make_lattice_adj, make_ba, make_er  # noqa: E402
# On reutilise TELLE QUELLE la logique du POC (arret doute/conv, simulate, flip, dec_at).
import deceptive_task_poc as poc  # noqa: E402

# Budget par probleme reduit de 3000 -> 2000 : le pulse le plus long teste est 700, la verite
# reprend juste apres, largement dans les 1300 pas restants (arrets doute~380 / conv~205 << 2000).
# Accelere le balayage 30 seeds x 3 topos sans changer la mesure. simulate/dec_at lisent ce global.
poc.MAX_BUDGET = 2000

CSV = ROOT / "figures" / "b1d_deceptive_consolidation.csv"
PNG = ROOT / "figures" / "b1d_deceptive_consolidation.png"

N = poc.N                              # 100
SEEDS = list(range(30))                # 30 seeds (POC : 12)
T_PULSE_LEVELS = [150, 350, 700]       # 150 = controle court (piege inoperant) ; 350/700 = trompeur
CONTROL_PULSE = 150                    # piege trop court : le doute ne doit PAS gagner
DECEPTIVE_MIN = 350                    # a partir d'ici le leurre dure assez -> regime trompeur
N_BOOT = 10000                         # tirages bootstrap
RNG_BOOT = np.random.RandomState(20260708)

# Topologies testees. <k> rapporte pour la transparence (les topos ne sont pas iso-degre :
# le but est la VARIETE structurelle, pas l'egalite de densite).
def make_topologies(seed):
    lat = make_lattice_adj(10, periodic=True)          # <k> = 4 exact
    ba = make_ba(N, m=3, seed=seed)                    # scale-free, <k> ~ 5.8
    er = make_er(N, p=0.06, seed=seed)                 # aleatoire, <k> ~ 6
    return {"LATTICE": lat, "BA_m3": ba, "ER_p06": er}

def kmean(adj):
    return float(adj.sum(axis=1).mean())

def make_deceptive_on(adj, rng):
    """Meme construction du piege que le POC, mais sur une topologie 'adj' fournie.
    (Le POC cable make_lattice_adj en dur ; ici on parametre la topologie.)"""
    dstar = rng.choice([-1, 1])
    nodes = rng.choice(N, size=poc.N_DISTRACT + poc.N_TRUE, replace=False)
    d_nodes, t_nodes = nodes[:poc.N_DISTRACT], nodes[poc.N_DISTRACT:]
    stim_on = np.zeros(N)
    stim_on[d_nodes] = -dstar * poc.E_DISTRACT
    stim_on[t_nodes] = +dstar * poc.E_TRUE
    stim_off = np.zeros(N)
    stim_off[t_nodes] = +dstar * poc.E_TRUE
    return adj, stim_on, stim_off, dstar

def boot_ci(diffs):
    """IC bootstrap 95% de la moyenne d'un vecteur d'ecarts apparies (par seed)."""
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    means = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = RNG_BOOT.randint(0, n, n)
        means[b] = diffs[idx].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

def main():
    CSV.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = []
    # summary[(topo, t_pulse)] = dict(acc_d, acc_c, af, diff, ci_lo, ci_hi, kbar)
    summary = {}
    topo_names = ["LATTICE", "BA_m3", "ER_p06"]

    print(f"{'topo':<9}{'T_pulse':>8}{'<k>':>6}{'acc_DOUTE':>11}{'acc_CONV':>10}"
          f"{'gain':>8}{'CI95':>16}{'acc_FIN':>9}")
    print("-" * 78)

    for topo in topo_names:
        for t_pulse in T_PULSE_LEVELS:
            acc_d, acc_c, acc_f, kbars = [], [], [], []
            diffs = []
            for seed in SEEDS:
                rng = np.random.RandomState(3000 + seed)
                adjs = make_topologies(seed)
                adj = adjs[topo]
                kbars.append(kmean(adj))
                _, stim_on, stim_off, dstar = make_deceptive_on(adj, rng)
                sig, dec, d_var = poc.simulate(adj, stim_on, stim_off, seed * 10 + 1, t_pulse)
                cd = poc.stop_doubt(sig); cc = poc.stop_conv(d_var)
                a_d = int(poc.dec_at(dec, cd) == dstar)
                a_c = int(poc.dec_at(dec, cc) == dstar)
                a_f = int(dec[-1] == dstar)
                acc_d.append(a_d); acc_c.append(a_c); acc_f.append(a_f)
                diffs.append(a_d - a_c)
                rows.append((topo, t_pulse, seed, dstar, cd, cc, a_d, a_c, a_f))
            ad, ac, af = np.mean(acc_d), np.mean(acc_c), np.mean(acc_f)
            diff = ad - ac
            lo, hi = boot_ci(diffs)
            kbar = float(np.mean(kbars))
            summary[(topo, t_pulse)] = dict(ad=ad, ac=ac, af=af, diff=diff, lo=lo, hi=hi, kbar=kbar)
            print(f"{topo:<9}{t_pulse:>8}{kbar:>6.1f}{ad:>11.2f}{ac:>10.2f}"
                  f"{diff:>+8.2f}  [{lo:+.2f},{hi:+.2f}]{af:>9.2f}")

    print("\n=== VERDICT consolidation B1d (honnete) ===")
    print("Regime TROMPEUR = leurre assez long (T_pulse>=350) pour que la convergence s'y engage.")
    print("Direction robuste attendue : gain > 0 avec IC excluant 0, sur les 3 topologies.\n")
    deceptive_levels = [tp for tp in T_PULSE_LEVELS if tp >= DECEPTIVE_MIN]
    robust_ok = True
    for topo in topo_names:
        for t_pulse in deceptive_levels:
            s = summary[(topo, t_pulse)]
            wins = s["lo"] > 0
            robust_ok = robust_ok and wins
            tag = "DOUTE GAGNE (IC>0)" if wins else "non concluant (IC couvre 0)"
            print(f"  {topo:<9} T_pulse={t_pulse}: gain {s['diff']:+.2f} "
                  f"CI[{s['lo']:+.2f},{s['hi']:+.2f}] -> {tag}")
    # Controle negatif : a CONTROL_PULSE le doute ne doit pas gagner (piege trop court).
    if CONTROL_PULSE in T_PULSE_LEVELS:
        print(f"\n  Controle negatif (T_pulse={CONTROL_PULSE}, piege trop court) :")
        for topo in topo_names:
            s = summary[(topo, CONTROL_PULSE)]
            print(f"    {topo:<9} gain {s['diff']:+.2f} CI[{s['lo']:+.2f},{s['hi']:+.2f}] "
                  f"(acc_FIN={s['af']:.2f})")

    print("\n  --> " + (
        "CONSOLIDE : la victoire du doute sur tache trompeuse tient sur 30 seeds ET 3 topologies."
        if robust_ok else
        "PARTIEL : la direction n'est pas confirmee (IC>0) sur toutes les topologies -- voir tableau."))

    with CSV.open("w", encoding="utf-8") as f:
        f.write("topo,t_pulse,seed,dstar,c_doubt,c_conv,acc_doubt,acc_conv,acc_final\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print(f"\n[csv] {CSV}")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
        # (1) gain doute-conv +/- IC, par topo, aux T_pulse trompeurs
        xs = np.arange(len(topo_names))
        dec_levels = [tp for tp in T_PULSE_LEVELS if tp >= DECEPTIVE_MIN] or T_PULSE_LEVELS
        width = 0.8 / max(len(dec_levels), 1)
        for k, t_pulse in enumerate(dec_levels):
            gains = [summary[(t, t_pulse)]["diff"] for t in topo_names]
            los = [summary[(t, t_pulse)]["diff"] - summary[(t, t_pulse)]["lo"] for t in topo_names]
            his = [summary[(t, t_pulse)]["hi"] - summary[(t, t_pulse)]["diff"] for t in topo_names]
            offset = (k - (len(dec_levels) - 1) / 2) * width
            axes[0].bar(xs + offset, gains, width,
                        yerr=[los, his], capsize=4, label=f"T_pulse={t_pulse}")
        axes[0].axhline(0, color="k", lw=0.8)
        axes[0].set_xticks(xs); axes[0].set_xticklabels(topo_names)
        axes[0].set_ylabel("gain doute = acc_DOUTE - acc_CONV")
        axes[0].set_title("Le doute gagne-t-il ? (IC95 bootstrap, 30 seeds)")
        axes[0].legend(); axes[0].grid(axis="y", alpha=0.3)
        # (2) acc DOUTE vs CONV vs T_pulse (moyenne sur topos)
        levels = T_PULSE_LEVELS
        for lab, key, c, mk in [("DOUTE", "ad", "#d62728", "o"),
                                ("CONVERGENCE", "ac", "#1f77b4", "s"),
                                ("budget illimite", "af", "#2ca02c", "^")]:
            ys = [np.mean([summary[(t, tp)][key] for t in topo_names]) for tp in levels]
            axes[1].plot(levels, ys, marker=mk, c=c, label=lab, ls="--" if key == "af" else "-")
        axes[1].set_xlabel("Duree du leurre T_pulse"); axes[1].set_ylabel("Taux de bonne reponse")
        axes[1].set_title("Justesse a l'arret (moy. 3 topos)"); axes[1].set_ylim(-0.05, 1.05)
        axes[1].legend(); axes[1].grid(alpha=0.3)
        fig.suptitle("Consolidation B1d : le doute bat la convergence sur tache trompeuse "
                     "(30 seeds x 3 topologies)", fontsize=11)
        plt.tight_layout(); plt.savefig(PNG, dpi=140)
        print(f"[png] {PNG}")
    except Exception as e:
        print(f"[png] skipped: {e}")
    print(f"\nWall time: {time.time()-t0:.1f}s")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
