"""
Test Scale-Free : Mem4ristor sur topologie Barabási-Albert
==========================================================
Le test crucial que LIMITATIONS.md identifie comme potentiel échec :
"15% threshold... Fails on Scale-Free Hubs"

On teste si le doute constitutionnel résiste quand quelques nœuds
hyper-connectés (hubs) dominent le réseau.

Usage :
  python experiments/robustness/test_scale_free.py
"""
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from mem4ristor.core import Mem4Network


def run_topology_test(name, adj_matrix, heretic_ratio=0.15, steps=1000, seed=42):
    """Run Mem4ristor on a given adjacency matrix and return entropy history."""
    net = Mem4Network(adjacency_matrix=adj_matrix, heretic_ratio=heretic_ratio, seed=seed)
    
    H_history = []
    u_history = []
    
    for _ in range(steps):
        net.step(I_stimulus=0.0)
        H_history.append(net.calculate_entropy(use_cognitive_bins=True))
        u_history.append(np.mean(net.model.u))
    
    return {
        'name': name,
        'H_history': H_history,
        'u_history': u_history,
        'H_final': H_history[-1],
        'H_mean': np.mean(H_history[100:]),  # Skip transient
        'u_mean': np.mean(u_history[100:]),
    }


def main():
    print("=" * 65)
    print("  TEST SCALE-FREE : Mem4ristor sur topologies réalistes")
    print("=" * 65)
    
    N = 100
    STEPS = 1000
    SEED = 42
    
    # ── Topologies ──
    topologies = {}
    
    # 1. Grid régulière (baseline — c'est ce qu'on teste normalement)
    print("\n  [1/5] Grille 10×10 régulière...", end=" ", flush=True)
    G_grid = nx.grid_2d_graph(10, 10)
    adj_grid = nx.to_numpy_array(G_grid)
    result_grid = run_topology_test("Grille 10×10", adj_grid, steps=STEPS, seed=SEED)
    print(f"H = {result_grid['H_final']:.3f}")
    
    # 2. Barabási-Albert m=2 (modéré — hubs moyens)
    print("  [2/5] Barabási-Albert m=2...", end=" ", flush=True)
    G_ba2 = nx.barabasi_albert_graph(N, m=2, seed=SEED)
    adj_ba2 = nx.to_numpy_array(G_ba2)
    result_ba2 = run_topology_test("BA m=2", adj_ba2, steps=STEPS, seed=SEED)
    print(f"H = {result_ba2['H_final']:.3f}")
    
    # 3. Barabási-Albert m=3 (agressif — gros hubs)
    print("  [3/5] Barabási-Albert m=3...", end=" ", flush=True)
    G_ba3 = nx.barabasi_albert_graph(N, m=3, seed=SEED)
    adj_ba3 = nx.to_numpy_array(G_ba3)
    result_ba3 = run_topology_test("BA m=3", adj_ba3, steps=STEPS, seed=SEED)
    print(f"H = {result_ba3['H_final']:.3f}")
    
    # 4. Barabási-Albert m=5 (extrême — hubs dominants)
    print("  [4/5] Barabási-Albert m=5...", end=" ", flush=True)
    G_ba5 = nx.barabasi_albert_graph(N, m=5, seed=SEED)
    adj_ba5 = nx.to_numpy_array(G_ba5)
    result_ba5 = run_topology_test("BA m=5", adj_ba5, steps=STEPS, seed=SEED)
    print(f"H = {result_ba5['H_final']:.3f}")
    
    # 5. Erdős-Rényi random (contrôle — pas de hubs)
    print("  [5/5] Erdős-Rényi p=0.06...", end=" ", flush=True)
    G_er = nx.erdos_renyi_graph(N, p=0.06, seed=SEED)
    # Ensure connected
    if not nx.is_connected(G_er):
        largest_cc = max(nx.connected_components(G_er), key=len)
        G_er = G_er.subgraph(largest_cc).copy()
        G_er = nx.convert_node_labels_to_integers(G_er)
    adj_er = nx.to_numpy_array(G_er)
    result_er = run_topology_test("Erdős-Rényi", adj_er, steps=STEPS, seed=SEED)
    print(f"H = {result_er['H_final']:.3f}")
    
    # ── Results Table ──
    results = [result_grid, result_ba2, result_ba3, result_ba5, result_er]
    
    print("\n" + "─" * 65)
    print(f"  {'Topology':<20} {'N':>5} {'H_final':>8} {'H_mean':>8} {'u_mean':>8} {'Status':>8}")
    print("─" * 65)
    
    for r in results:
        status = "✅" if r['H_mean'] > 1.0 else "⚠️" if r['H_mean'] > 0.5 else "❌"
        n_nodes = len(r['H_history'])  # Placeholder
        print(f"  {r['name']:<20} {N:>5} {r['H_final']:>8.3f} {r['H_mean']:>8.3f} {r['u_mean']:>8.3f} {status:>8}")
    
    # ── Degree Distribution Analysis ──
    print("\n  Hub analysis (top-5 highest degree nodes):")
    for name, G in [("BA m=3", G_ba3), ("BA m=5", G_ba5)]:
        degrees = sorted(dict(G.degree()).values(), reverse=True)[:5]
        print(f"    {name}: degrees = {degrees}")
    
    # ── Plot ──
    results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    colors = ['#3498db', '#e74c3c', '#e67e22', '#9b59b6', '#2ecc71']
    
    for i, r in enumerate(results):
        ax1.plot(r['H_history'], color=colors[i], alpha=0.8, linewidth=1.5,
                 label=f"{r['name']} (H={r['H_final']:.2f})")
        ax2.plot(r['u_history'], color=colors[i], alpha=0.8, linewidth=1.5,
                 label=r['name'])
    
    ax1.set_ylabel('Shannon Entropy H (bits)', fontsize=12)
    ax1.set_title('Mem4ristor: Scale-Free Topology Robustness Test', fontsize=14)
    ax1.legend(fontsize=9, loc='lower left')
    ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Threshold H=1.0')
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Time Step', fontsize=12)
    ax2.set_ylabel('Mean Doubt ū', fontsize=12)
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    fig_path = os.path.join(results_dir, 'test_scale_free.png')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"\n  📈 Figure: {os.path.abspath(fig_path)}")
    
    # ── Verdict ──
    all_pass = all(r['H_mean'] > 1.0 for r in results)
    ba_pass = all(r['H_mean'] > 1.0 for r in [result_ba2, result_ba3, result_ba5])
    
    if all_pass:
        print("\n  ✅ VICTOIRE : Mem4ristor résiste sur TOUTES les topologies !")
    elif ba_pass:
        print("\n  ✅ PARTIEL : Mem4ristor résiste sur Scale-Free (BA).")
    else:
        print("\n  ⚠️ LIMITE : Certaines topologies Scale-Free réduisent la diversité.")
        print("    → Ceci confirme la limitation documentée dans CAFE-VIRTUEL-LIMITATIONS.md")
    
    print("\n" + "=" * 65)


if __name__ == '__main__':
    main()
