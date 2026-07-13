#!/usr/bin/env python3
"""
P13 -- LA HIERARCHIE DE DOUTE : qui doute des douteurs ?
=========================================================================
Cree : 2026-07-13 (Claude Sonnet 5, L'Ingenieur) -- piste du legs de Fable
(docs/PISTES_POUR_LA_SUITE_2026-07-12.md, section II, P13).

POURQUOI : toute la science M4R est a UN niveau. L'ecosysteme de Julien
(Cafe Virtuel : tables d'agents, un Barman qui arbitre) est une architecture
a DEUX niveaux. Version mesurable : k sous-reseaux M4R + un signal de
desaccord ENTRE leurs consensus, qui peut retro-agir sur le doute LOCAL de
chaque sous-reseau (une table qui remarque que les autres tables ne sont
pas d'accord avec elle redouble de prudence).

ARCHITECTURE FIGEE AVANT DE REGARDER LES RESULTATS (garde-fou explicite de
la piste, meme discipline que le 07/07) :
  - 3 sous-reseaux BA(N=30, m=3), memes constantes du coeur PAR DEFAUT.
  - Chaque sous-reseau recoit sa PROPRE tache trompeuse B1d-style (leurre
    puis verite qui resiste), meme dstar GLOBAL pour les 3, mais duree de
    leurre (T_pulse) HETEROGENE par sous-reseau et par essai -- la
    tromperie locale n'est PAS parfaitement correlee entre sous-reseaux
    (certains recuperent vite, d'autres restent trompes longtemps),
    exactement la situation ou un niveau meta peut aider UN sous-reseau
    encore trompe en remarquant que ses pairs ont deja bascule.
  - Meta-couplage (HIER_META uniquement) : a chaque pas, desaccord_meta =
    ecart-type des 3 signaux de decision differentiels courants des
    sous-reseaux. Ce scalaire est ajoute (poids gamma_meta) au
    sigma_social_override de CHAQUE sous-reseau -- le hook d'ablation
    DEJA EXISTANT dans dynamics.py::step() (sigma_social_override remplace
    |laplacian_v| dans l'equation de u UNIQUEMENT ; aucune modification du
    coeur, meme mecanisme que p2_sigma_social_ablation.py). Le sous-reseau
    dont les pairs divergent voit donc son PROPRE doute monter -- "qui
    doute des douteurs" implemente sans toucher dynamics.py.
  - Decision globale (chaque condition) : moyenne des 3 signaux
    differentiels des sous-reseaux au budget B, signe de cette moyenne.

3 CONDITIONS A COUT EGAL (3*B*30 = B*90 noeuds-pas, plus l'overhead meta
negligeable pour HIER_META) :
  FLAT       : UN SEUL reseau BA(N=90, m=3) recevant les 3 stimuli
               regionaux (chaque tiers de noeuds = un "sous-reseau" fictif),
               decision = signe de la moyenne differentielle GLOBALE.
  HIER_VOTE  : 3 sous-reseaux INDEPENDANTS (aucun couplage), decision =
               signe de la MOYENNE de leurs 3 signaux differentiels
               (equivalent a un vote pondere par l'intensite).
  HIER_META  : memes 3 sous-reseaux, PLUS la retroaction meta-doute
               (gamma_meta > 0) via sigma_social_override.

CRITERE PRE-FIXE (ecrit avant de lancer) : HIER_META est UTILE si son
accuracy sur la decision globale BAT A LA FOIS FLAT ET HIER_VOTE (IC
bootstrap apparie > 0 sur les deux comparaisons). Si HIER_META ne bat que
HIER_VOTE mais pas FLAT (ou l'inverse) : nuance a rapporter, pas un succes
plein. Si aucune des trois ne domine : resultat negatif honnete --
beaucoup de degres de liberte (piege annonce par la piste elle-meme), a
rapporter sans forcer une lecture positive.

Sorties : figures/p13_doubt_hierarchy_poc.csv + .png
Statut : exploratoire, hors preprint, coeur non touche (sigma_social_override
est un hook deja existant, cf. p2_sigma_social_ablation.py).
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
from mem4ristor.graph_utils import make_ba  # noqa: E402

CSV_PATH = ROOT / "figures" / "p13_doubt_hierarchy_poc.csv"
PNG_PATH = ROOT / "figures" / "p13_doubt_hierarchy_poc.png"

N_SUB = 30
K_SUB = 3
N_FLAT = N_SUB * K_SUB          # 90, cout egal
BUDGET = 800                     # zone de signal la plus forte (P6a/P6b)
T_SIM = 1600
N_DISTRACT, N_TRUE = 8, 4        # meme ratio que b1d (26/14 sur 100 -> 8/4 sur 30)
E_TRUE, E_DISTRACT = 0.6, 1.0
T_PULSE_CHOICES = [150, 350, 700, 1200]   # heterogeneite du leurre local
GAMMA_META = 0.5
TRIALS = 20
HERETIC = 0.0


def make_sub_stimulus(n, dstar, t_pulse_choice_rng):
    """Une tache trompeuse B1d-style a l'echelle d'un sous-reseau (n=30)."""
    nodes = t_pulse_choice_rng.choice(n, size=N_DISTRACT + N_TRUE, replace=False)
    d_nodes, t_nodes = nodes[:N_DISTRACT], nodes[N_DISTRACT:]
    stim_on = np.zeros(n)
    stim_on[d_nodes] = -dstar * E_DISTRACT
    stim_on[t_nodes] = dstar * E_TRUE
    stim_off = np.zeros(n)
    stim_off[t_nodes] = dstar * E_TRUE
    return stim_on, stim_off


def run_flat(trial_seed, dstar, sub_configs):
    """FLAT : un seul reseau N_FLAT, chaque tiers recoit le stimulus de son
    sous-reseau fictif correspondant. Decision = signe de la moyenne
    differentielle GLOBALE au budget B."""
    adj = make_ba(N_FLAT, m=3, seed=trial_seed + 500)
    net = Mem4Network(adjacency_matrix=adj, heretic_ratio=HERETIC, seed=trial_seed)
    ref = Mem4Network(adjacency_matrix=adj, heretic_ratio=HERETIC, seed=trial_seed)
    zero = np.zeros(N_FLAT)
    d_hist = np.empty(BUDGET)
    for t in range(BUDGET):
        stim = np.zeros(N_FLAT)
        for k, (stim_on, stim_off, t_pulse) in enumerate(sub_configs):
            lo, hi = k * N_SUB, (k + 1) * N_SUB
            stim[lo:hi] = stim_on if t < t_pulse else stim_off
        net.step(I_stimulus=stim)
        ref.step(I_stimulus=zero)
        d_hist[t] = float(np.mean(net.model.v) - np.mean(ref.model.v))
    d_final = float(np.mean(d_hist[-50:]))     # lissage terminal (leçon P6b)
    return int(d_final >= 0), d_final


def run_hierarchical(trial_seed, dstar, sub_configs, use_meta):
    """HIER_VOTE (use_meta=False) ou HIER_META (use_meta=True) : 3 sous-reseaux
    independants, couples uniquement via sigma_social_override si use_meta."""
    nets, refs, adjs = [], [], []
    for k in range(K_SUB):
        adj = make_ba(N_SUB, m=3, seed=trial_seed * 10 + k)
        adjs.append(adj)
        nets.append(Mem4Network(adjacency_matrix=adj, heretic_ratio=HERETIC, seed=trial_seed * 10 + k))
        refs.append(Mem4Network(adjacency_matrix=adj, heretic_ratio=HERETIC, seed=trial_seed * 10 + k))
    d_hist = np.zeros((K_SUB, BUDGET))
    zero = np.zeros(N_SUB)
    d_running = np.zeros(K_SUB)     # signal differentiel courant (pour le desaccord meta)
    for t in range(BUDGET):
        meta_override = None
        if use_meta:
            meta_disagreement = float(np.std(d_running))
            meta_override = meta_disagreement   # scalaire, broadcast a tous les noeuds du sous-reseau
        for k in range(K_SUB):
            stim_on, stim_off, t_pulse = sub_configs[k]
            stim = stim_on if t < t_pulse else stim_off
            sso = None
            if use_meta:
                lap_v = nets[k].L @ nets[k].v * -1.0
                local_sigma = np.abs(lap_v)
                sso = local_sigma + GAMMA_META * meta_override
            nets[k].step(I_stimulus=stim, sigma_social_override=sso)
            refs[k].step(I_stimulus=zero)
            d_running[k] = float(np.mean(nets[k].v) - np.mean(refs[k].model.v))
            d_hist[k, t] = d_running[k]
    d_final_per_sub = np.mean(d_hist[:, -50:], axis=1)
    d_final = float(np.mean(d_final_per_sub))
    return int(d_final >= 0), d_final, d_final_per_sub


def main() -> int:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rng_master = np.random.RandomState(20260713)

    rows = []
    results = {"FLAT": [], "HIER_VOTE": [], "HIER_META": []}
    d_finals = {"FLAT": [], "HIER_VOTE": [], "HIER_META": []}

    for trial in range(TRIALS):
        rng = np.random.RandomState(9000 + trial)
        dstar = rng.choice([-1, 1])
        sub_configs = []
        for k in range(K_SUB):
            t_pulse = rng.choice(T_PULSE_CHOICES)
            stim_on, stim_off = make_sub_stimulus(N_SUB, dstar, rng)
            sub_configs.append((stim_on, stim_off, t_pulse))

        ok_flat, d_flat = run_flat(trial, dstar, sub_configs)
        ok_vote, d_vote, d_persub_vote = run_hierarchical(trial, dstar, sub_configs, use_meta=False)
        ok_meta, d_meta, d_persub_meta = run_hierarchical(trial, dstar, sub_configs, use_meta=True)

        results["FLAT"].append(int(ok_flat == (dstar > 0)))
        results["HIER_VOTE"].append(int(ok_vote == (dstar > 0)))
        results["HIER_META"].append(int(ok_meta == (dstar > 0)))
        d_finals["FLAT"].append(d_flat)
        d_finals["HIER_VOTE"].append(d_vote)
        d_finals["HIER_META"].append(d_meta)

        t_pulses_str = ";".join(str(c[2]) for c in sub_configs)
        rows.append((trial, dstar, t_pulses_str,
                     int(ok_flat == (dstar > 0)), int(ok_vote == (dstar > 0)), int(ok_meta == (dstar > 0)),
                     d_flat, d_vote, d_meta))
        print(f"  trial {trial:>2} dstar={dstar:+d} t_pulses={t_pulses_str:<15} "
              f"FLAT={'OK' if ok_flat==(dstar>0) else 'X'} "
              f"HIER_VOTE={'OK' if ok_vote==(dstar>0) else 'X'} "
              f"HIER_META={'OK' if ok_meta==(dstar>0) else 'X'}  [{time.time()-t0:.0f}s]")

    def acc(name):
        return float(np.mean(results[name]))

    print(f"\n{'condition':<12}{'accuracy':>10}")
    print("-" * 24)
    for name in ["FLAT", "HIER_VOTE", "HIER_META"]:
        print(f"{name:<12}{acc(name):>10.2f}")

    def boot_ci_paired(a, b, n_boot=10000, seed=20260713):
        rng = np.random.RandomState(seed)
        d = np.asarray(a, float) - np.asarray(b, float)
        n = len(d)
        m = np.empty(n_boot)
        for kk in range(n_boot):
            m[kk] = d[rng.randint(0, n, n)].mean()
        return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

    print("\n=== VERDICT P13 (pre-fixe : HIER_META bat FLAT ET HIER_VOTE, IC bootstrap > 0) ===")
    d1, lo1, hi1 = boot_ci_paired(results["HIER_META"], results["FLAT"])
    print(f"  HIER_META - FLAT      = {d1:+.3f} CI[{lo1:+.3f},{hi1:+.3f}]")
    d2, lo2, hi2 = boot_ci_paired(results["HIER_META"], results["HIER_VOTE"])
    print(f"  HIER_META - HIER_VOTE = {d2:+.3f} CI[{lo2:+.3f},{hi2:+.3f}]")
    d3, lo3, hi3 = boot_ci_paired(results["HIER_VOTE"], results["FLAT"])
    print(f"  HIER_VOTE - FLAT      = {d3:+.3f} CI[{lo3:+.3f},{hi3:+.3f}]  (la hierarchie SEULE, sans meta, aide-t-elle deja ?)")

    beats_both = lo1 > 0 and lo2 > 0
    beats_one = (lo1 > 0) != (lo2 > 0)
    if beats_both:
        print("\n  -> UTILE : la hierarchie de doute (meta-retroaction) bat les deux baselines.")
    elif beats_one:
        which = "FLAT" if lo1 > 0 else "HIER_VOTE"
        print(f"\n  -> PARTIEL : HIER_META bat seulement {which} -- nuance, pas un succes plein.")
    elif hi1 < 0 or hi2 < 0:
        print("\n  -> LA META-RETROACTION NUIT : au moins une des deux baselines reste devant HIER_META.")
    else:
        print("\n  -> RESULTAT NEGATIF HONNETE : aucune condition ne domine mesurablement -- "
              "beaucoup de degres de liberte (risque annonce par la piste), a ne pas forcer.")

    with CSV_PATH.open("w", encoding="utf-8") as f:
        f.write("trial,dstar,t_pulses,ok_flat,ok_hier_vote,ok_hier_meta,d_flat,d_hier_vote,d_hier_meta\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print(f"\n[csv] {CSV_PATH}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
        ax = axes[0]
        names = ["FLAT", "HIER_VOTE", "HIER_META"]
        accs = [acc(n) for n in names]
        colors = ["#7f7f7f", "#1f77b4", "#d62728"]
        ax.bar(names, accs, color=colors)
        ax.set_ylabel("accuracy (decision globale)")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"P13 -- {TRIALS} essais, cout egal ({N_FLAT} noeuds-pas x {BUDGET})")
        ax.grid(alpha=0.3, axis="y")
        ax = axes[1]
        for name, col in zip(names, colors):
            ax.hist(d_finals[name], bins=12, alpha=0.5, color=col, label=name)
        ax.axvline(0, ls=":", c="k")
        ax.set_xlabel("signal differentiel final (signe = decision)")
        ax.set_ylabel("essais"); ax.legend(fontsize=8)
        ax.set_title("Distribution du signal de decision")
        fig.suptitle("P13 -- la hierarchie de doute : qui doute des douteurs ?", fontsize=11)
        plt.tight_layout()
        plt.savefig(PNG_PATH, dpi=140)
        print(f"[png] {PNG_PATH}")
    except Exception as e:
        print(f"[png] skipped: {e}")

    print(f"\nWall time: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
