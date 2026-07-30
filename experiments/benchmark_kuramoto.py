"""
Benchmark: Mem4ristor V3 vs Kuramoto (Standard & Frustrated)
=============================================================
Compares diversity preservation across 3 models using
topology-matched metrics for a fair comparison.

METHODOLOGY NOTES (V3.0 audit corrections):
  - All models use the SAME 2D lattice topology (10x10, k=4 nearest neighbors)
    to eliminate topology mismatch bias.
  - Metrics are topology-agnostic (entropy, variance) rather than
    phase-based (Kuramoto order parameter R is computed for Kuramoto only,
    since voltage-to-phase mapping is physically invalid for FHN dynamics).
  - No claim of superiority is made without statistical testing.

Metrics:
  1. Shannon Entropy H (5 uniform bins) - measures state diversity
  2. State Variance - measures spread of activations
  3. Order Parameter R (Kuramoto models only) - measures phase coherence

Usage:
  python experiments/benchmark_kuramoto.py
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import csv
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from mem4ristor.core import Mem4Network


def build_lattice_adjacency(size):
    """Build a 2D lattice adjacency matrix with periodic boundary conditions.
    Same topology used by Mem4Network internally."""
    N = size * size
    adj = np.zeros((N, N))
    for i in range(size):
        for j in range(size):
            idx = i * size + j
            neighbors = [
                ((i - 1) % size) * size + j,  # up
                ((i + 1) % size) * size + j,  # down
                i * size + (j - 1) % size,    # left
                i * size + (j + 1) % size,    # right
            ]
            for n in neighbors:
                adj[idx, n] = 1.0
    return adj


def compute_order_parameter_phase(theta):
    """Kuramoto order parameter: R = |<exp(i*theta)>|. R=1 = synchronized."""
    return np.abs(np.mean(np.exp(1j * theta)))


def shannon_entropy(values, n_bins=5):
    """Shannon entropy with uniform binning."""
    vmin, vmax = np.min(values), np.max(values)
    if vmax - vmin < 1e-10:
        return 0.0
    counts, _ = np.histogram(values, bins=n_bins, range=(vmin - 0.01, vmax + 0.01))
    total = np.sum(counts)
    if total == 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


# ---------------------------------------------------------------
# 1. Kuramoto Standard on LATTICE (same topology as Mem4ristor)
# ---------------------------------------------------------------
def simulate_kuramoto_standard_lattice(adj, N, K, steps, dt=0.01, seed=42):
    """Kuramoto on lattice: dtheta_i/dt = omega_i + K * sum_j A_ij * sin(theta_j - theta_i)."""
    rng = np.random.RandomState(seed)
    theta = rng.uniform(-np.pi, np.pi, N)
    omega = rng.normal(0, 1.0, N)

    # Normalize coupling by degree (k=4 for lattice)
    degree = np.sum(adj, axis=1)

    R_history, H_history, var_history = [], [], []

    for _ in range(steps):
        sin_diff = np.sin(theta[np.newaxis, :] - theta[:, np.newaxis])
        # Lattice coupling: only neighbors contribute (adj masks the rest)
        coupling = K * np.sum(adj * sin_diff, axis=1) / degree
        theta += (omega + coupling) * dt
        theta = np.mod(theta + np.pi, 2 * np.pi) - np.pi

        R_history.append(compute_order_parameter_phase(theta))
        H_history.append(shannon_entropy(theta, n_bins=5))
        var_history.append(np.var(theta))

    return R_history, H_history, var_history


# ---------------------------------------------------------------
# 2. Kuramoto Frustrated on LATTICE
# ---------------------------------------------------------------
def simulate_kuramoto_frustrated_lattice(adj, N, K, alpha, frustration_ratio, steps, dt=0.01, seed=42):
    """Frustrated Kuramoto on lattice with random phase shifts on edges."""
    rng = np.random.RandomState(seed)
    theta = rng.uniform(-np.pi, np.pi, N)
    omega = rng.normal(0, 1.0, N)

    # Frustration only on existing edges
    frustration_matrix = np.zeros((N, N))
    edge_mask = adj > 0
    n_edges = int(np.sum(edge_mask))
    n_frustrated = int(frustration_ratio * n_edges)
    edge_indices = np.argwhere(edge_mask)
    frustrated_idx = rng.choice(len(edge_indices), n_frustrated, replace=False)
    for idx in frustrated_idx:
        i, j = edge_indices[idx]
        frustration_matrix[i, j] = alpha

    degree = np.sum(adj, axis=1)

    R_history, H_history, var_history = [], [], []

    for _ in range(steps):
        phase_diff = theta[np.newaxis, :] - theta[:, np.newaxis] + frustration_matrix
        coupling = K * np.sum(adj * np.sin(phase_diff), axis=1) / degree
        theta += (omega + coupling) * dt
        theta = np.mod(theta + np.pi, 2 * np.pi) - np.pi

        R_history.append(compute_order_parameter_phase(theta))
        H_history.append(shannon_entropy(theta, n_bins=5))
        var_history.append(np.var(theta))

    return R_history, H_history, var_history


# ---------------------------------------------------------------
# 3. Mem4ristor V3
# ---------------------------------------------------------------
def simulate_mem4ristor(size, heretic_ratio, steps, seed=42):
    """Mem4ristor V3 on 2D lattice with periodic boundaries."""
    net = Mem4Network(size=size, heretic_ratio=heretic_ratio, seed=seed,
                      boundary='periodic')

    H_history, var_history = [], []

    for _ in range(steps):
        net.step(I_stimulus=0.0)
        H_history.append(shannon_entropy(net.v, n_bins=5))
        var_history.append(np.var(net.v))

    return H_history, var_history


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    print("=" * 70)
    print("  BENCHMARK: Mem4ristor V3 vs Kuramoto (Lattice-Matched Comparison)")
    print("=" * 70)

    SIZE = 10
    N = SIZE * SIZE  # 100
    STEPS = 1000
    K = 2.0
    ALPHA = np.pi / 4
    FRUST_RATIO = 0.15
    HERETIC_RATIO = 0.15

    print(f"\n  Topology: {SIZE}x{SIZE} periodic lattice (k=4 neighbors)")
    print(f"  N={N}, Steps={STEPS}, K={K}")
    print(f"  Frustrated Kuramoto: alpha={ALPHA:.2f}, ratio={FRUST_RATIO}")
    print(f"  Mem4ristor V3: heretic_ratio={HERETIC_RATIO}\n")

    # Build shared adjacency matrix
    adj = build_lattice_adjacency(SIZE)

    # Run simulations
    print("  [1/3] Kuramoto Standard (lattice)...", end=" ", flush=True)
    R_std, H_std, V_std = simulate_kuramoto_standard_lattice(adj, N, K, STEPS)
    print(f"H={H_std[-1]:.3f}, Var={V_std[-1]:.3f}")

    print("  [2/3] Kuramoto Frustrated (lattice)...", end=" ", flush=True)
    R_frust, H_frust, V_frust = simulate_kuramoto_frustrated_lattice(adj, N, K, ALPHA, FRUST_RATIO, STEPS)
    print(f"H={H_frust[-1]:.3f}, Var={V_frust[-1]:.3f}")

    print("  [3/3] Mem4ristor V3 (lattice)...", end=" ", flush=True)
    H_mem, V_mem = simulate_mem4ristor(SIZE, HERETIC_RATIO, STEPS)
    print(f"H={H_mem[-1]:.3f}, Var={V_mem[-1]:.3f}")

    # Results (steady-state: last 500 steps)
    ss = slice(500, None)
    print("\n" + "-" * 70)
    print(f"  {'Model':<30} {'H_mean':>8} {'H_std':>8} {'Var_mean':>10}")
    print("-" * 70)

    models_data = [
        ("Kuramoto Standard", H_std, V_std),
        ("Kuramoto Frustrated", H_frust, V_frust),
        ("Mem4ristor V3", H_mem, V_mem),
    ]

    for name, H, V in models_data:
        h_arr = np.array(H[ss])
        v_arr = np.array(V[ss])
        print(f"  {name:<30} {np.mean(h_arr):>8.3f} {np.std(h_arr):>8.3f} {np.mean(v_arr):>10.4f}")

    print("\n  NOTE: R (order parameter) is only valid for phase oscillators.")
    print("  Mem4ristor uses FHN dynamics (excitable medium), not phase oscillators.")
    print("  Direct R comparison would require a physically invalid voltage-to-phase mapping.")
    print("  We compare using topology-agnostic metrics (H, Variance) instead.")

    # CSV
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)

    csv_path = os.path.join(results_dir, 'benchmark_kuramoto_v3.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'R_std', 'R_frust', 'H_std', 'H_frust', 'H_mem4',
                         'Var_std', 'Var_frust', 'Var_mem4'])
        for t in range(STEPS):
            writer.writerow([t, R_std[t], R_frust[t],
                           H_std[t], H_frust[t], H_mem[t],
                           V_std[t], V_frust[t], V_mem[t]])
    print(f"\n  CSV: {os.path.abspath(csv_path)}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    t = np.arange(STEPS)

    # H(t)
    ax1.plot(t, H_std, color='#e74c3c', alpha=0.7, linewidth=1.5,
             label=f'Kuramoto Std (H={np.mean(H_std[ss]):.2f})')
    ax1.plot(t, H_frust, color='#f39c12', alpha=0.7, linewidth=1.5,
             label=f'Kuramoto Frust (H={np.mean(H_frust[ss]):.2f})')
    ax1.plot(t, H_mem, color='#2ecc71', alpha=0.9, linewidth=2.0,
             label=f'Mem4ristor V3 (H={np.mean(H_mem[ss]):.2f})')

    ax1.set_ylabel('Shannon Entropy H (bits)', fontsize=12)
    ax1.set_title('Diversity Preservation: Mem4ristor V3 vs Kuramoto\n'
                  f'(Same {SIZE}x{SIZE} lattice topology, k=4)', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.3)

    # Variance(t)
    ax2.plot(t, V_std, color='#e74c3c', alpha=0.7, linewidth=1.5,
             label=f'Kuramoto Std')
    ax2.plot(t, V_frust, color='#f39c12', alpha=0.7, linewidth=1.5,
             label=f'Kuramoto Frust')
    ax2.plot(t, V_mem, color='#2ecc71', alpha=0.9, linewidth=2.0,
             label=f'Mem4ristor V3')

    ax2.set_xlabel('Time Step', fontsize=12)
    ax2.set_ylabel('State Variance', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.set_ylim(bottom=0)
    ax2.grid(True, alpha=0.3)

    fig_path = os.path.join(results_dir, 'benchmark_kuramoto_v3.png')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  Figure: {os.path.abspath(fig_path)}")

    print("\n" + "=" * 70)
    print(f"  Benchmark completed - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == '__main__':
    main()
