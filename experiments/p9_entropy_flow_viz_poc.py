#!/usr/bin/env python3
"""
P9 -- VISUALISATION DES FLUX D'ENTROPIE : voir la decision se prendre.
=========================================================================
Cree : 2026-07-13 (Claude Sonnet 5, L'Ingenieur) -- piste du legs de Fable
(docs/PISTES_POUR_LA_SUITE_2026-07-12.md, section I, P9). Le TODO le plus
ancien du projet (Roadmap V5 point 6, jamais fait) + reco Manus 3.

VALEUR : comprehension et communication (labo, video), PAS un claim
scientifique -- statut explicitement pose comme tel par la piste elle-meme.

CE QUI EST REUTILISE (pas invente) : `calculate_transfer_entropy` existe deja
dans `mem4ristor/metrics.py` et est deja une defense etablie du projet
(`experiments/reviewer2_transfer_entropy.py`, "TE causalite de u"). P9 est la
PREMIERE fois qu'elle est utilisee pour visualiser un FLUX SPATIAL au cours
du temps plutot qu'un chiffre unique de causalite globale heretique<->normal.

PROTOCOLE : lattice 10x10 (N=100) decoupe en 4 QUADRANTS geometriques
(25 noeuds chacun). Un stimulus constant est injecte SEULEMENT dans le
quadrant Q1 (coin) a partir de t=0 -- la "decision" qui doit se propager
physiquement a travers le reseau. On enregistre v(t) (pour la kymographe
spatiale) et les etats cognitifs (pour la TE, methode canonique du projet).
Fenetre glissante (W=200, stride=50) : TE(Qi -> Qj, t) pour les 12 paires
dirigees.

PREDICTION MINIMALE (pas un claim, une verification de coherence physique) :
la topologie lattice impose une adjacence Q1-Q2, Q1-Q3 (voisins directs) et
Q1-Q4 (diagonale, plus loin) -- le flux TE(Q1->voisin direct) devrait
apparaitre AVANT et plus fort que TE(Q1->diagonale) dans les fenetres
precoces, puis s'estomper une fois le regime permanent atteint (l'info a
fini de se propager, il ne reste que le regime stationnaire).

Sorties : figures/p9_entropy_flow_viz_poc.png (grille de kymographes +
courbes de flux TE + graphe de flux a 3 instants cles)
Statut : exploratoire, hors preprint, coeur non touche, communication pure.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

import contextlib
import io

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from mem4ristor.topology import Mem4Network  # noqa: E402
from mem4ristor.graph_utils import make_lattice_adj  # noqa: E402
from mem4ristor.metrics import calculate_transfer_entropy  # noqa: E402


def te_silent(*args, **kwargs):
    """calculate_transfer_entropy() a un print() de debug inconditionnel dans
    metrics.py (code preexistant, non touche ici) -- on l'etouffe localement
    plutot que de modifier un module partage et teste ailleurs."""
    with contextlib.redirect_stdout(io.StringIO()):
        return calculate_transfer_entropy(*args, **kwargs)

PNG_PATH = ROOT / "figures" / "p9_entropy_flow_viz_poc.png"
CSV_PATH = ROOT / "figures" / "p9_entropy_flow_viz_poc.csv"

SIDE, N = 10, 100
T_SIM = 2000
I_STIM_Q1 = 0.6
SEED = 42
W = 200        # fenetre glissante TE
STRIDE = 50
TE_BINS = 4


def quadrant_masks():
    """Q1=coin haut-gauche (stimule), Q2=haut-droit, Q3=bas-gauche, Q4=bas-droit
    (diagonalement oppose a Q1 -- le plus loin en distance de graphe lattice)."""
    idx = np.arange(N).reshape(SIDE, SIDE)
    half = SIDE // 2
    masks = {
        "Q1": idx[:half, :half].flatten(),
        "Q2": idx[:half, half:].flatten(),
        "Q3": idx[half:, :half].flatten(),
        "Q4": idx[half:, half:].flatten(),
    }
    return masks


def main() -> int:
    t0 = time.time()
    adj = make_lattice_adj(SIDE, periodic=True)
    net = Mem4Network(size=SIDE, heretic_ratio=0.0, seed=SEED, adjacency_matrix=adj)
    masks = quadrant_masks()

    stim = np.zeros(N)
    stim[masks["Q1"]] = I_STIM_Q1

    v_hist = np.empty((T_SIM, N))
    u_hist = np.empty((T_SIM, N))
    print(f"Simulation : stimulus constant dans Q1 (coin), {T_SIM} pas...")
    for t in range(T_SIM):
        net.step(I_stimulus=stim)
        v_hist[t] = net.model.v
        u_hist[t] = net.model.u

    # moyenne CONTINUE de v par quadrant, discretisee UNE SEULE FOIS en TE_BINS
    # classes (eviter la double discretisation via get_cognitive_states, qui
    # ecrase trop de signal une fois moyennee puis re-quantifiee).
    quad_names = ["Q1", "Q2", "Q3", "Q4"]
    quad_v_mean = {q: v_hist[:, masks[q]].mean(axis=1) for q in quad_names}
    quad_cog_binned = {}
    for q in quad_names:
        edges = np.quantile(quad_v_mean[q], np.linspace(0, 1, TE_BINS + 1))
        edges[0] -= 1e-9
        quad_cog_binned[q] = np.clip(np.digitize(quad_v_mean[q], edges[1:-1]), 0, TE_BINS - 1)

    # ---------------- TE glissante pour toutes les paires dirigees ----------------
    pairs = [(a, b) for a in quad_names for b in quad_names if a != b]
    centers = list(range(W, T_SIM, STRIDE))
    te_series = {p: [] for p in pairs}
    for c in centers:
        lo = c - W
        for (a, b) in pairs:
            te = te_silent(
                quad_cog_binned[a][lo:c], quad_cog_binned[b][lo:c], bins=TE_BINS)
            te_series[(a, b)].append(te)
    print(f"TE calculee sur {len(centers)} fenetres glissantes (W={W}, stride={STRIDE}) "
          f"pour {len(pairs)} paires dirigees. [{time.time()-t0:.0f}s]")

    # ---------------- verification de coherence physique (pas un claim) ----------------
    early_idx = slice(0, max(1, len(centers) // 4))   # premier quart des fenetres
    te_Q1_Q2 = np.mean(np.array(te_series[("Q1", "Q2")])[early_idx])   # voisin direct
    te_Q1_Q3 = np.mean(np.array(te_series[("Q1", "Q3")])[early_idx])   # voisin direct
    te_Q1_Q4 = np.mean(np.array(te_series[("Q1", "Q4")])[early_idx])   # diagonale (plus loin)
    print("\n=== Coherence physique (verification, pas un claim) ===")
    print(f"  TE(Q1->Q2, voisin direct) tot fenetres precoces = {te_Q1_Q2:.4f} bits")
    print(f"  TE(Q1->Q3, voisin direct) tot fenetres precoces = {te_Q1_Q3:.4f} bits")
    print(f"  TE(Q1->Q4, diagonale)     tot fenetres precoces = {te_Q1_Q4:.4f} bits")
    neighbors_first = (te_Q1_Q2 + te_Q1_Q3) / 2 > te_Q1_Q4
    print(f"  -> {'COHERENT' if neighbors_first else 'INATTENDU'} : les voisins directs "
          f"{'recoivent plus de flux tot que' if neighbors_first else 'ne dominent PAS'} "
          f"la diagonale dans les fenetres precoces.")

    # ---------------- sorties CSV ----------------
    with CSV_PATH.open("w", encoding="utf-8") as f:
        f.write("window_center,source,target,te_bits\n")
        for (a, b) in pairs:
            for c, te in zip(centers, te_series[(a, b)]):
                f.write(f"{c},{a},{b},{te:.6f}\n")
    print(f"[csv] {CSV_PATH}")

    # ---------------- figure : kymographes + flux TE + graphe de flux ----------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 4, height_ratios=[1.1, 1, 1])

        # (a) kymographes spatiaux de v a 4 instants
        snap_times = [50, 300, 800, 1900]
        for i, st in enumerate(snap_times):
            ax = fig.add_subplot(gs[0, i])
            grid = v_hist[st].reshape(SIDE, SIDE)
            im = ax.imshow(grid, cmap="RdBu_r", vmin=-2, vmax=2)
            ax.set_title(f"v(x,y) a t={st}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_ylabel("Q1 (stimule) = coin haut-gauche", fontsize=8)
        cbar_ax = fig.add_axes([0.92, 0.68, 0.012, 0.2])
        fig.colorbar(im, cax=cbar_ax, label="v")

        # (b) courbes de flux TE(Q1->*) au cours du temps
        ax_flow = fig.add_subplot(gs[1, :2])
        colors = {"Q2": "#1f77b4", "Q3": "#2ca02c", "Q4": "#d62728"}
        for target in ["Q2", "Q3", "Q4"]:
            label = f"Q1->{target}" + (" (diagonale)" if target == "Q4" else " (voisin)")
            ax_flow.plot(centers, te_series[("Q1", target)], marker="o", ms=3,
                         color=colors[target], label=label)
        ax_flow.set_xlabel("centre de la fenetre (pas)")
        ax_flow.set_ylabel("Transfer Entropy (bits)")
        ax_flow.set_title("Flux d'information depuis Q1 (source du stimulus)")
        ax_flow.legend(fontsize=8); ax_flow.grid(alpha=0.3)

        # (c) H_cont global + u_mean au cours du temps (contexte)
        ax_h = fig.add_subplot(gs[1, 2:])
        from mem4ristor.metrics import calculate_continuous_entropy
        h_series = [calculate_continuous_entropy(v_hist[t]) for t in range(0, T_SIM, 20)]
        u_series = [np.mean(u_hist[t]) for t in range(0, T_SIM, 20)]
        t_axis = list(range(0, T_SIM, 20))
        ax_h2 = ax_h.twinx()
        ax_h.plot(t_axis, h_series, color="#9467bd", label="H_cont global")
        ax_h2.plot(t_axis, u_series, color="#ff7f0e", label="u_mean global", alpha=0.7)
        ax_h.set_xlabel("pas"); ax_h.set_ylabel("H_cont (bits)", color="#9467bd")
        ax_h2.set_ylabel("u_mean", color="#ff7f0e")
        ax_h.set_title("Contexte : entropie et doute globaux")
        ax_h.grid(alpha=0.3)

        # (d) graphe de flux a 3 instants (arrows scaled by TE)
        centers_pos = {"Q1": (0, 1), "Q2": (1, 1), "Q3": (0, 0), "Q4": (1, 0)}
        snap_windows = [0, len(centers) // 3, len(centers) - 1]
        snap_labels = ["precoce", "intermediaire", "tardif (regime permanent)"]
        for i, (wi, lbl) in enumerate(zip(snap_windows, snap_labels)):
            ax = fig.add_subplot(gs[2, i])
            for q, (x, y) in centers_pos.items():
                ax.scatter([x], [y], s=400, c="#dddddd", edgecolors="k", zorder=3)
                ax.annotate(q, (x, y), ha="center", va="center", fontsize=10, zorder=4)
            max_te = max(max(te_series[p]) for p in pairs) or 1.0
            for (a, b) in pairs:
                if a != "Q1":
                    continue  # ne montrer que le flux DEPUIS la source pour lisibilite
                te = te_series[(a, b)][wi]
                xa, ya = centers_pos[a]; xb, yb = centers_pos[b]
                width = 0.5 + 4.0 * (te / max_te)
                ax.annotate("", xy=(xb, yb), xytext=(xa, ya),
                            arrowprops=dict(arrowstyle="-|>", lw=width, color="#d62728", alpha=0.7,
                                            shrinkA=20, shrinkB=20))
            ax.set_xlim(-0.5, 1.5); ax.set_ylim(-0.5, 1.5)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"Flux Q1->* ({lbl}, pas~{centers[wi]})", fontsize=9)
        ax = fig.add_subplot(gs[2, 3])
        ax.axis("off")
        ax.text(0.05, 0.7, "Epaisseur de fleche\n= Transfer Entropy\n(Q1 -> region)",
                fontsize=9, va="top")
        ax.text(0.05, 0.3, f"Coherence physique :\nvoisins > diagonale\nen precoce = "
                f"{'OUI' if neighbors_first else 'NON'}", fontsize=9, va="top",
                color="#2ca02c" if neighbors_first else "#d62728")

        fig.suptitle("P9 -- Flux d'entropie/information : voir la decision se propager "
                     "(stimulus injecte dans Q1, lattice 10x10)", fontsize=12)
        plt.tight_layout(rect=[0, 0, 0.90, 0.96])
        plt.savefig(PNG_PATH, dpi=140)
        print(f"[png] {PNG_PATH}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[png] skipped: {e}")

    print(f"\nWall time: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
