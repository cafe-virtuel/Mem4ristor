#!/usr/bin/env python3
"""
[13] Event-Driven Phase Transition — Peripheral Node High-Amplitude Forcing
============================================================================
Idée originale : Julien Chauvin (30 avril 2026, 09h28)
Analogie source : demande en mariage dans un restaurant.

Question scientifique
---------------------
Un nœud périphérique (faible degré, "table dans un coin") générant un signal
d'amplitude extraordinaire peut-il :
  1. Provoquer une synchronisation temporaire globale du réseau ?
  2. Conduire à un basculement irréversible d'attracteur après la fin du forcing ?
  3. Produire un effet différent selon qu'il s'agit d'un nœud périphérique ou d'un hub ?

Ce scénario est distinct de la dead zone (synchronisation structurelle, forcée par
la topologie). Ici la synchronisation est déclenchée par un événement singulier,
de la même façon qu'une demande en mariage dans un restaurant silencieux tout le monde
— même si personne ne se connaît.

Analogie formelle (Julien → physique)
--------------------------------------
- Restaurant au repos     = réseau FULL en régime fonctionnel (diversité maintenue)
- Demande en mariage      = un nœud périphérique avec I_stim >> 0 pendant T_event steps
- Silence qui se fait     = synchronisation temporaire globale (surge sync)
- Réponse de la femme     = paramètre de bifurcation (amplitude ou durée)
  * "Non" (faible)  → réseau revient à l'état initial (ou état dégradé)
  * "Oui" (fort)    → réseau bascule vers un nouvel attracteur (irréversible)
- "Plus personne n'est pareil après" = H_cog ou LZ post-event ≠ pré-event

Protocole
---------
- Topologies : BA m=3 (fonctionnel), BA m=5 (dead zone)
- Config : FULL (u actif, heretic_ratio=0.15)
- Nœuds cibles : le plus périphérique (degré min) + le plus hub (degré max) — comparés
- Amplitudes : I_event ∈ {0.3, 0.8, 1.5, 3.0}
- Durées : T_event ∈ {50, 150, 300} steps
- Observation : 1000 steps pré-event + T_event + 2000 steps post-event
- Seeds : 5
- Métriques :
  * sync(t) sur 3 fenêtres (pré / pendant / post) → détection synchronisation temporaire
  * H_cont pré vs post → détection basculement d'attracteur
  * delta_LZ = LZ_post - LZ_pre → détection changement de complexité trajectorielle

Outputs
-------
  figures/event_phase_transition.csv         -- résultats bruts par run
  figures/event_phase_transition_summary.csv -- synthèse par (amplitude, durée, nœud)
  figures/event_phase_transition.png         -- grille sync_peak vs delta_H_cont

Created: 2026-04-30 (Claude Sonnet 4.6 via Antigravity, item [13])
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mem4ristor.topology import Mem4Network          # noqa: E402
from mem4ristor.metrics import (                     # noqa: E402
    calculate_continuous_entropy,
    calculate_temporal_lz_complexity,
)
from mem4ristor.graph_utils import make_ba            # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

N_NODES    = 100
N_SEEDS    = 5

STEPS_PRE  = 1000   # stabilisation avant l'événement
STEPS_POST = 2000   # observation après la fin du forcing

I_EVENTS   = [0.3, 0.8, 1.5, 3.0]   # amplitudes du forcing
T_EVENTS   = [50, 150, 300]          # durées du forcing (steps)

TOPO_PARAMS = [
    ("BA_m3_fonctionnel", 3),
    ("BA_m5_dead_zone",   5),
]

SYNC_WINDOW = 100   # fenêtre (steps) pour calculer la sync moyenne

FIG_DIR  = ROOT / "figures"
CSV_PATH = FIG_DIR / "event_phase_transition.csv"
SUM_PATH = FIG_DIR / "event_phase_transition_summary.csv"
FIG_PATH = FIG_DIR / "event_phase_transition.png"

# ─────────────────────────────────────────────────────────────────────────────
# Métriques
# ─────────────────────────────────────────────────────────────────────────────

def compute_sync(snapshots: list[np.ndarray]) -> float:
    """Synchronisation : corrélation moyenne sur la fenêtre."""
    if len(snapshots) < 2:
        return 0.0
    mat = np.array(snapshots)   # (T, N)
    # Variance temporelle par nœud
    var = mat.var(axis=0)
    # Variance globale vs variance locale → synchrony = 1 - mean(local_var)/global_var
    global_var = mat.var()
    if global_var < 1e-10:
        return 1.0
    return 1.0 - var.mean() / global_var


def compute_h_cont(snapshots: list[np.ndarray]) -> float:
    if not snapshots:
        return 0.0
    v = np.array(snapshots).flatten()
    return float(calculate_continuous_entropy(v))


def compute_lz_mean(snapshots: list[np.ndarray]) -> float:
    if not snapshots:
        return 0.0
    mat = np.array(snapshots)   # (T, N)
    # calculate_temporal_lz_complexity expects (T, N) and returns per-node array
    lz_per_node = calculate_temporal_lz_complexity(mat)
    return float(np.mean(lz_per_node))


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_one(adj: np.ndarray, target_idx: int, i_event: float,
            t_event: int, seed: int) -> dict:
    """Un run complet : pré-event, event, post-event."""
    net = Mem4Network(adjacency_matrix=adj, heretic_ratio=0.15,
                      coupling_norm='degree_linear', seed=seed)

    N = N_NODES
    snap_pre   : list[np.ndarray] = []
    snap_event : list[np.ndarray] = []
    snap_post  : list[np.ndarray] = []

    # ── Phase pré-event ──────────────────────────────────────────────────────
    for step in range(STEPS_PRE):
        net.step(I_stimulus=0.0)
        if step >= STEPS_PRE - SYNC_WINDOW:
            snap_pre.append(net.model.v.copy())

    # ── Phase event (forcing sur un seul nœud) ───────────────────────────────
    I_vec = np.zeros(N)
    I_vec[target_idx] = i_event
    for _ in range(t_event):
        net.step(I_stimulus=I_vec)
        snap_event.append(net.model.v.copy())

    # ── Phase post-event ──────────────────────────────────────────────────────
    for step in range(STEPS_POST):
        net.step(I_stimulus=0.0)
        if step >= STEPS_POST - SYNC_WINDOW:
            snap_post.append(net.model.v.copy())

    # ── Métriques ─────────────────────────────────────────────────────────────
    sync_pre   = compute_sync(snap_pre)
    sync_event = compute_sync(snap_event)
    sync_post  = compute_sync(snap_post)

    h_pre  = compute_h_cont(snap_pre)
    h_post = compute_h_cont(snap_post)

    lz_pre  = compute_lz_mean(snap_pre)
    lz_post = compute_lz_mean(snap_post)

    return {
        "sync_pre":    sync_pre,
        "sync_event":  sync_event,
        "sync_post":   sync_post,
        "sync_surge":  sync_event - sync_pre,   # hausse pendant l'event
        "h_pre":       h_pre,
        "h_post":      h_post,
        "delta_H":     h_post - h_pre,           # basculement d'attracteur
        "lz_pre":      lz_pre,
        "lz_post":     lz_post,
        "delta_LZ":    lz_post - lz_pre,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    total_runs = (len(TOPO_PARAMS) * 2 *   # 2 nœuds cibles
                  len(I_EVENTS) * len(T_EVENTS) * N_SEEDS)
    print(f"[13] Event-Driven Phase Transition")
    print(f"     {len(TOPO_PARAMS)} topos x 2 cibles x "
          f"{len(I_EVENTS)} amplitudes x {len(T_EVENTS)} durées x "
          f"{N_SEEDS} seeds = {total_runs} runs")
    print(f"     PRE={STEPS_PRE} + EVENT<={max(T_EVENTS)} + POST={STEPS_POST} steps\n")

    rows: list[dict] = []
    run_idx = 0

    for topo_name, ba_m in TOPO_PARAMS:
        for seed in range(N_SEEDS):
            adj = make_ba(N_NODES, ba_m, seed)
            degrees = adj.sum(axis=1).astype(int)
            peripheral_idx = int(np.argmin(degrees))
            hub_idx        = int(np.argmax(degrees))
            targets = [
                ("peripheral", peripheral_idx, int(degrees[peripheral_idx])),
                ("hub",        hub_idx,        int(degrees[hub_idx])),
            ]

            for target_name, target_idx, target_deg in targets:
                for i_event in I_EVENTS:
                    for t_event in T_EVENTS:
                        run_idx += 1
                        res = run_one(adj, target_idx, i_event, t_event, seed)

                        elapsed = time.time() - t0
                        pct     = run_idx / total_runs
                        eta     = (elapsed / pct) * (1 - pct) if pct > 0 else 0

                        print(f"[{run_idx:3d}/{total_runs}] {topo_name} "
                              f"seed={seed} {target_name}(deg={target_deg}) "
                              f"I={i_event:.1f} T={t_event} "
                              f"| surge={res['sync_surge']:+.3f} "
                              f"dH={res['delta_H']:+.3f} "
                              f"| ETA {eta:.0f}s", flush=True)

                        rows.append({
                            "topology":    topo_name,
                            "ba_m":        ba_m,
                            "seed":        seed,
                            "target":      target_name,
                            "target_deg":  target_deg,
                            "i_event":     i_event,
                            "t_event":     t_event,
                            **res,
                        })

    # ── CSV brut ─────────────────────────────────────────────────────────────
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[csv] {CSV_PATH}  ({len(rows)} lignes)")

    # ── Synthèse ─────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("SYNTHESE — sync_surge et delta_H (mean sur seeds × topos)")
    print("="*80)
    print(f"{'target':12s}  {'I_event':7s}  {'T_event':7s}  "
          f"{'sync_surge':10s}  {'delta_H':8s}  {'delta_LZ':8s}  bifurcation?")
    print("-"*80)

    summary_rows = []
    from itertools import product
    for (topo_name, ba_m), target_name, i_event, t_event in product(
            TOPO_PARAMS, ["peripheral", "hub"], I_EVENTS, T_EVENTS):
        subset = [r for r in rows
                  if r["topology"] == topo_name
                  and r["target"] == target_name
                  and r["i_event"] == i_event
                  and r["t_event"] == t_event]
        if not subset:
            continue
        surge = np.mean([r["sync_surge"] for r in subset])
        dH    = np.mean([r["delta_H"]    for r in subset])
        dLZ   = np.mean([r["delta_LZ"]   for r in subset])
        bifurc = "YES" if abs(dH) > 0.3 else ("~" if abs(dH) > 0.1 else "no")
        print(f"{topo_name[:6]}+{target_name[:4]:4s}  "
              f"I={i_event:.1f}    T={t_event:3d}    "
              f"surge={surge:+.3f}    dH={dH:+.3f}    dLZ={dLZ:+.3f}    {bifurc}")
        summary_rows.append({
            "topology": topo_name, "target": target_name,
            "i_event": i_event, "t_event": t_event,
            "sync_surge_mean": surge, "delta_H_mean": dH, "delta_LZ_mean": dLZ,
            "bifurcation": bifurc,
        })

    with SUM_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\n[csv] {SUM_PATH}")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        "[13] Event-Driven Phase Transition\n"
        "Idée : Julien Chauvin — Analogie de la demande en mariage dans un restaurant\n"
        "Un nœud périphérique peut-il synchroniser tout le réseau et basculer l'attracteur ?",
        fontsize=10,
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # Pour chaque topo : scatter sync_surge vs delta_H, coloré par I_event
    for plot_idx, (topo_name, ba_m) in enumerate(TOPO_PARAMS):
        for target_idx2, target_name in enumerate(["peripheral", "hub"]):
            ax = fig.add_subplot(gs[plot_idx, target_idx2])
            subset = [r for r in rows
                      if r["topology"] == topo_name
                      and r["target"] == target_name]
            if not subset:
                continue
            x = [r["sync_surge"] for r in subset]
            y = [r["delta_H"]    for r in subset]
            c = [r["i_event"]    for r in subset]
            sc = ax.scatter(x, y, c=c, cmap='RdYlGn', s=40, alpha=0.7,
                            vmin=min(I_EVENTS), vmax=max(I_EVENTS))
            ax.axhline(0, ls='--', color='gray', alpha=0.5)
            ax.axvline(0, ls='--', color='gray', alpha=0.5)
            ax.axhline(0.3,  ls=':', color='green', alpha=0.4, label='bifurcation seuil')
            ax.axhline(-0.3, ls=':', color='red',   alpha=0.4)
            ax.set_xlabel("sync_surge (pendant - avant)")
            ax.set_ylabel("delta_H_cont (après - avant)")
            ax.set_title(f"{topo_name}\nnœud : {target_name}")
            ax.grid(alpha=0.2)
            plt.colorbar(sc, ax=ax, label="I_event")

    plt.savefig(FIG_PATH, dpi=140, bbox_inches='tight')
    print(f"\n[png] {FIG_PATH}")
    print(f"\nTotal wall time: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
