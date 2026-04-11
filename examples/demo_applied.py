#!/usr/bin/env python3
"""
Mem4ristor Applied Demonstration (Point 11)
============================================
Full pipeline showcasing all V3-V5 features:

  1. Sensory Pipeline   : Image -> SensoryFrontend -> Mem4Network -> dashboard
  2. V5 Hysteresis      : Latching behaviour comparison (ON vs OFF)
  3. Scale-Free Sparse  : BA(N=900, m=3) with degree-normalized coupling
  4. Phase Diversity     : Heretic vs Normal unit dynamics

Outputs 4 PNG files into the same directory.

Usage:
    python examples/demo_applied.py

Requirements:
    numpy, scipy, matplotlib
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import time

# Mem4ristor imports
from mem4ristor.core import Mem4ristorV3, Mem4Network
from mem4ristor.sensory import SensoryFrontend
from mem4ristor.viz import (
    SimHistory, plot_entropy_trace, plot_doubt_map,
    plot_phase_portrait, plot_state_distribution,
    plot_v_heatmap, dashboard
)

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for file output
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────
#  DEMO 1: Full Sensory Pipeline
# ─────────────────────────────────────────────────────────────
def demo_sensory_pipeline():
    """
    Image -> SensoryFrontend -> Mem4Network -> Dashboard.

    Demonstrates the complete perception-to-cognition pathway:
    a 64x64 geometric pattern is processed by the retinal frontend,
    projected into the neural field, and drives FHN dynamics.
    """
    print("=" * 60)
    print("DEMO 1: Full Sensory Pipeline")
    print("=" * 60)

    # 1. Create network (10x10 = 100 units)
    net = Mem4Network(size=10, heretic_ratio=0.15, seed=42)

    # 2. Create sensory frontend matching the network
    eye = SensoryFrontend(output_dim=net.N, input_shape=(64, 64), seed=42)

    # 3. Generate test patterns
    patterns = {
        'circle': eye.generate_test_pattern('circle'),
        'square': eye.generate_test_pattern('square'),
        'noise':  eye.generate_test_pattern('noise'),
    }

    # 4. Run simulation: alternate stimuli every 500 steps
    STEPS = 3000
    history = SimHistory()

    pattern_schedule = ['circle'] * 1000 + ['square'] * 1000 + ['noise'] * 1000
    stimulus_cache = {name: eye.perceive(img) for name, img in patterns.items()}

    print(f"  Running {STEPS} steps with pattern schedule: circle->square->noise")
    t0 = time.perf_counter()

    for step in range(STEPS):
        pattern_name = pattern_schedule[step]
        I_stim = stimulus_cache[pattern_name]
        net.step(I_stimulus=I_stim)
        if step % 10 == 0:
            history.record(net)

    elapsed = time.perf_counter() - t0
    H_final = net.calculate_entropy()
    print(f"  Completed in {elapsed:.2f}s — H_final = {H_final:.4f}")

    # 5. Generate dashboard
    fig, axes = dashboard(
        history, net,
        suptitle="Mem4ristor Sensory Pipeline: Circle -> Square -> Noise"
    )

    # Add vertical lines at pattern transitions
    ax_entropy = axes[0]
    ax_entropy.axvline(100, color='gray', linestyle=':', alpha=0.5)
    ax_entropy.axvline(200, color='gray', linestyle=':', alpha=0.5)
    ax_entropy.text(50, 0.1, 'Circle', ha='center', fontsize=7, color='gray')
    ax_entropy.text(150, 0.1, 'Square', ha='center', fontsize=7, color='gray')
    ax_entropy.text(250, 0.1, 'Noise', ha='center', fontsize=7, color='gray')

    outpath = os.path.join(OUTPUT_DIR, 'demo1_sensory_pipeline.png')
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> Saved: {outpath}")

    return history


# ─────────────────────────────────────────────────────────────
#  DEMO 2: V5 Hysteresis Comparison
# ─────────────────────────────────────────────────────────────
def demo_hysteresis_comparison():
    """
    Compare entropy dynamics with/without V5 hysteresis latching.

    The dead-zone [0.35, 0.65] creates Schmitt-trigger-like behaviour
    that stabilises mode transitions and prevents noise-driven chattering.
    """
    print("\n" + "=" * 60)
    print("DEMO 2: V5 Hysteresis Comparison")
    print("=" * 60)

    STEPS = 3000
    configs = {
        'V5 Hysteresis ON (default)': {},
        'V5 Hysteresis OFF (hard threshold)': {'hysteresis': {'enabled': False}},
        'V5 + Fatigue (rate=0.01)': {'hysteresis': {
            'enabled': True, 'theta_low': 0.35, 'theta_high': 0.65,
            'fatigue_rate': 0.01, 'base_hysteresis': 0.15
        }},
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
    fig.suptitle("V5 Hysteresis: Impact on Entropy Dynamics", fontsize=12, fontweight='bold')

    colors = ['#2E86C1', '#E74C3C', '#27AE60']

    for idx, (label, cfg_override) in enumerate(configs.items()):
        entropies = []
        for seed in [42, 123, 777]:
            net = Mem4Network(size=10, heretic_ratio=0.15, seed=seed)
            if cfg_override:
                for section, vals in cfg_override.items():
                    if section in net.model.cfg:
                        net.model.cfg[section].update(vals)
                    else:
                        net.model.cfg[section] = vals

            H_trace = []
            for step in range(STEPS):
                net.step(I_stimulus=0.0)
                if step % 5 == 0:
                    H_trace.append(net.calculate_entropy())
            entropies.append(H_trace)

        # Plot mean ± std
        arr = np.array(entropies)
        mean_H = np.mean(arr, axis=0)
        std_H = np.std(arr, axis=0)
        x = np.arange(len(mean_H)) * 5

        ax = axes[idx]
        ax.fill_between(x, mean_H - std_H, mean_H + std_H, alpha=0.2, color=colors[idx])
        ax.plot(x, mean_H, color=colors[idx], linewidth=1.0)
        ax.axhline(np.log2(5), color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.set_xlabel("Step")
        ax.set_title(label, fontsize=9)
        ax.grid(True, alpha=0.2)

        H_stable = np.mean(mean_H[-100:])
        print(f"  {label}: H_stable = {H_stable:.4f}")

    axes[0].set_ylabel("H (bits)")

    outpath = os.path.join(OUTPUT_DIR, 'demo2_hysteresis_comparison.png')
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> Saved: {outpath}")


# ─────────────────────────────────────────────────────────────
#  DEMO 3: Scale-Free Network with Sparse Backend
# ─────────────────────────────────────────────────────────────
def demo_scale_free_sparse():
    """
    BA(N=900, m=3) with degree-normalized coupling (LIMIT-02 fix).

    Compares:
      - uniform coupling (hub strangulation)
      - degree_linear coupling (LIMIT-02 fix)

    Uses auto-sparse backend for memory efficiency.
    """
    print("\n" + "=" * 60)
    print("DEMO 3: Scale-Free BA(900) with Sparse Backend")
    print("=" * 60)

    N = 900
    m = 3

    # Build BA graph
    print(f"  Building BA({N}, m={m})...", end=" ", flush=True)
    rng = np.random.RandomState(42)
    adj = np.zeros((N, N))
    for i in range(m + 1):
        for j in range(i + 1, m + 1):
            adj[i, j] = adj[j, i] = 1.0
    degrees = np.sum(adj, axis=1)
    for new_node in range(m + 1, N):
        total = np.sum(degrees[:new_node])
        if total == 0:
            probs = np.ones(new_node) / new_node
        else:
            probs = degrees[:new_node] / total
        targets = rng.choice(new_node, size=m, replace=False, p=probs)
        for t in targets:
            adj[new_node, t] = adj[t, new_node] = 1.0
            degrees[new_node] += 1
            degrees[t] += 1
    print("done")

    deg_final = np.sum(adj, axis=1)
    print(f"  Graph stats: N={N}, edges={int(np.sum(adj)/2)}, "
          f"deg_mean={np.mean(deg_final):.1f}, deg_max={np.max(deg_final):.0f}")

    STEPS = 500
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Scale-Free BA({N}, m={m}) — Sparse Backend", fontsize=12, fontweight='bold')

    for idx, (norm_mode, color, label) in enumerate([
        ('uniform', '#E74C3C', 'Uniform coupling (hub strangulation)'),
        ('degree_linear', '#2E86C1', 'Degree-linear coupling (LIMIT-02 fix)')
    ]):
        print(f"  Running {label}...", end=" ", flush=True)
        t0 = time.perf_counter()

        net = Mem4Network(
            size=1,
            adjacency_matrix=adj.copy(),
            heretic_ratio=0.15,
            seed=42,
            coupling_norm=norm_mode,
            auto_sparse_threshold=500  # Force sparse for BA
        )

        H_trace = []
        for step in range(STEPS):
            net.step(I_stimulus=0.0)
            if step % 10 == 0:
                H_trace.append(net.calculate_entropy())

        elapsed = time.perf_counter() - t0
        H_final = H_trace[-1]
        print(f"done in {elapsed:.1f}s — H_final={H_final:.4f}")

        # Memory usage
        if net._is_sparse:
            mem_mb = (net.L.data.nbytes + net.L.indices.nbytes + net.L.indptr.nbytes) / 1024 / 1024
            mem_label = f"Sparse: {mem_mb:.2f} MB"
        else:
            mem_mb = net.L.nbytes / 1024 / 1024
            mem_label = f"Dense: {mem_mb:.1f} MB"

        ax = axes[idx]
        x = np.arange(len(H_trace)) * 10
        ax.plot(x, H_trace, color=color, linewidth=0.8)
        ax.axhline(np.log2(5), color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.set_xlabel("Step")
        ax.set_ylabel("H (bits)")
        ax.set_title(f"{label}\nH_final={H_final:.4f} — {mem_label}", fontsize=9)
        ax.set_ylim(0, 2.5)
        ax.grid(True, alpha=0.2)

    outpath = os.path.join(OUTPUT_DIR, 'demo3_scale_free_sparse.png')
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> Saved: {outpath}")


# ─────────────────────────────────────────────────────────────
#  DEMO 4: Phase Diversity (Heretics in action)
# ─────────────────────────────────────────────────────────────
def demo_phase_diversity():
    """
    Shows how heretic units create structural diversity in phase space.

    Two 4-panel grids at step 500 (transient) and step 2500 (attractor):
    - v heatmap, u heatmap, phase portrait, state distribution
    """
    print("\n" + "=" * 60)
    print("DEMO 4: Phase Diversity — Heretics in Action")
    print("=" * 60)

    net = Mem4Network(size=12, heretic_ratio=0.15, seed=42)
    history = SimHistory()

    STEPS = 3000
    snapshot_steps = [500, 2500]
    snapshots = {}

    print(f"  Running 12x12 lattice ({net.N} units) for {STEPS} steps...")
    for step in range(STEPS):
        net.step(I_stimulus=0.0)
        if step % 10 == 0:
            history.record(net)
        if step in snapshot_steps:
            snapshots[step] = {
                'v': net.model.v.copy(),
                'w': net.model.w.copy(),
                'u': net.model.u.copy(),
                'states': net.model.get_states(),
            }

    H_final = net.calculate_entropy()
    print(f"  H_final = {H_final:.4f}")

    # Build 2x4 figure
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle("Phase Diversity: Transient (t=500) vs Attractor (t=2500)",
                 fontsize=13, fontweight='bold')

    for row, step in enumerate(snapshot_steps):
        snap = snapshots[step]

        # v heatmap
        plot_v_heatmap(snap['v'], grid_size=12, ax=axes[row, 0],
                       title=f"$v$ @ t={step}")

        # u heatmap
        plot_doubt_map(snap['u'], grid_size=12, ax=axes[row, 1],
                       title=f"Doubt $u$ @ t={step}")

        # Phase portrait
        plot_phase_portrait(snap['v'], snap['w'],
                          heretic_mask=net.model.heretic_mask,
                          ax=axes[row, 2],
                          title=f"Phase @ t={step}")

        # State distribution
        from mem4ristor.viz import BIN_EDGES
        counts, _ = np.histogram(snap['v'], bins=BIN_EDGES)
        plot_state_distribution(counts, ax=axes[row, 3],
                              title=f"States @ t={step}")

    outpath = os.path.join(OUTPUT_DIR, 'demo4_phase_diversity.png')
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> Saved: {outpath}")

    # Also save the entropy trace
    fig2, ax2 = plot_entropy_trace(
        history.entropy,
        title="15×15 Lattice Entropy Trace (3000 steps)"
    )
    ax2.axvline(50, color='orange', linestyle=':', alpha=0.7)
    ax2.axvline(250, color='orange', linestyle=':', alpha=0.7)
    ax2.text(50, 0.05, 't=500', fontsize=7, color='orange')
    ax2.text(250, 0.05, 't=2500', fontsize=7, color='orange')

    outpath2 = os.path.join(OUTPUT_DIR, 'demo4_entropy_trace.png')
    fig2.savefig(outpath2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"  -> Saved: {outpath2}")


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n+==========================================================+")
    print("|   MEM4RISTOR v3.1.1 — Applied Demonstration (Point 11) |")
    print("|   V3 Levitating Sigmoid + V4 Rewiring + V5 Hysteresis  |")
    print("+==========================================================+\n")

    t_total = time.perf_counter()

    demo_sensory_pipeline()
    demo_hysteresis_comparison()
    demo_scale_free_sparse()
    demo_phase_diversity()

    elapsed_total = time.perf_counter() - t_total

    print("\n" + "=" * 60)
    print(f"All 4 demos completed in {elapsed_total:.1f}s")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)
