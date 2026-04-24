#!/usr/bin/env python3
"""
Item 12 -- Doubt-Driven Community Detection (2026-04-24)

Hypothese : La matrice de doute u(i,t) porte une information sur les
communautes fonctionnelles du reseau. Les noeuds qui oscillent en phase
dans leur niveau de doute u(t) appartiendraient au meme attracteur cognitif.

Approche :
  1. Enregistrer u_history (T, N) apres convergence.
  2. Calculer la matrice de correlation Pearson des traces u(t) → C_u (N×N).
  3. Seuillage de C_u → graphe de "doubt affinity" (aretes si |corr| > theta).
  4. Detection de communautes Louvain sur ce graphe (NetworkX 3.5).
  5. Detection de communautes Louvain sur le graphe structural (adjacence).
  6. NMI entre les deux partitions → alignment.
  7. Visualisation : heatmap C_u + side-by-side communautes.

Topologies : Lattice 10x10, BA m=3 N=100.
Regime : I_stim=0.5 (force), coupling_norm='degree_linear'.

Script  : experiments/p2_doubt_community_detection.py
Figures : figures/p2_doubt_community_detection.png
CSV     : figures/p2_doubt_community_detection.csv

Reference : PROJECT_STATUS.md Item 12
"""
import sys, os, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from mem4ristor.core import Mem4Network

# -- Parametres ---------------------------------------------------------------
SEEDS   = [42, 123, 777]
I_STIM  = 0.5
STEPS   = 4000
WARM_UP = 1500
N_BA    = 100
M_BA    = 3
CORR_THETA = 0.3   # seuil de correlation pour le graphe doubt-affinity


def make_ba(n, m, seed):
    rng = np.random.RandomState(seed)
    adj = np.zeros((n, n), dtype=float)
    for i in range(m + 1):
        for j in range(i + 1, m + 1):
            adj[i, j] = adj[j, i] = 1.0
    degrees = adj.sum(axis=1)
    for new_node in range(m + 1, n):
        probs = degrees[:new_node] / degrees[:new_node].sum()
        targets = rng.choice(new_node, size=m, replace=False, p=probs)
        for t in targets:
            adj[new_node, t] = adj[t, new_node] = 1.0
        degrees = adj.sum(axis=1)
    return adj


def pearson_corr_matrix(X):
    """Matrice de correlation Pearson de X (T, N) → (N, N). Sans sklearn."""
    mu  = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    Z   = (X - mu) / std          # (T, N)
    C   = (Z.T @ Z) / X.shape[0]  # (N, N)
    return np.clip(C, -1.0, 1.0)


def nmi(labels_a, labels_b):
    """
    Normalized Mutual Information entre deux vecteurs d'etiquettes entiers.
    NMI = 2*MI(A,B) / (H(A) + H(B)). Retourne float in [0, 1].
    """
    a = np.asarray(labels_a)
    b = np.asarray(labels_b)
    n = len(a)
    classes_a = np.unique(a)
    classes_b = np.unique(b)

    # Joint distribution
    joint = np.zeros((len(classes_a), len(classes_b)))
    a_idx = {c: i for i, c in enumerate(classes_a)}
    b_idx = {c: i for i, c in enumerate(classes_b)}
    for ai, bi in zip(a, b):
        joint[a_idx[ai], b_idx[bi]] += 1
    joint /= n

    pa = joint.sum(axis=1)
    pb = joint.sum(axis=0)

    # H(A), H(B)
    ha = -float(np.sum(pa[pa > 0] * np.log2(pa[pa > 0])))
    hb = -float(np.sum(pb[pb > 0] * np.log2(pb[pb > 0])))

    # MI
    outer = np.outer(pa, pb)
    mask  = (joint > 0) & (outer > 0)
    mi    = float(np.sum(joint[mask] * np.log2(joint[mask] / outer[mask])))

    denom = (ha + hb) / 2.0
    return mi / denom if denom > 1e-12 else 0.0


def louvain_partition(G):
    """
    Louvain communities via NetworkX. Retourne un vecteur d'etiquettes
    de longueur N (labels[i] = communaute du noeud i).
    """
    import networkx.algorithms.community as nxc
    communities = nxc.louvain_communities(G, seed=42)
    labels = np.zeros(G.number_of_nodes(), dtype=int)
    for k, comm in enumerate(communities):
        for node in comm:
            labels[node] = k
    return labels


def run_one(topo, seed):
    import networkx as nx

    if topo == 'lattice':
        net = Mem4Network(size=10, heretic_ratio=0.15, seed=seed,
                          coupling_norm='degree_linear')
        # Build lattice adjacency
        size = net.size
        N    = net.N
        adj  = np.zeros((N, N))
        for i in range(size):
            for j in range(size):
                node = i * size + j
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni2, nj2 = (i + di) % size, (j + dj) % size
                    adj[node, ni2 * size + nj2] = 1.0
    else:
        adj = make_ba(N_BA, M_BA, seed)
        net = Mem4Network(adjacency_matrix=adj.copy(), heretic_ratio=0.15,
                          seed=seed, coupling_norm='degree_linear')

    N = net.N

    # Simulate and record u_history + v_history post warm-up
    u_history = []
    v_history = []
    for step in range(STEPS):
        net.step(I_stimulus=I_STIM)
        if step >= WARM_UP:
            u_history.append(net.model.u.copy())
            v_history.append(net.v.copy())

    u_arr = np.array(u_history)   # (T, N)
    v_arr = np.array(v_history)   # (T, N)

    # 1. Correlation matrix of u traces
    C_u = pearson_corr_matrix(u_arr)

    # 2. Doubt-affinity graph (seuillage |corr| > CORR_THETA)
    G_doubt = nx.Graph()
    G_doubt.add_nodes_from(range(N))
    pairs = np.argwhere(np.abs(C_u) > CORR_THETA)
    for i, j in pairs:
        if i < j:
            G_doubt.add_edge(i, j, weight=float(C_u[i, j]))

    # 3. Structural graph
    G_struct = nx.from_numpy_array(adj)

    # 4. Community detection on both graphs
    lbl_doubt  = louvain_partition(G_doubt)
    lbl_struct = louvain_partition(G_struct)

    n_comm_doubt  = len(np.unique(lbl_doubt))
    n_comm_struct = len(np.unique(lbl_struct))
    nmi_score     = nmi(lbl_doubt, lbl_struct)

    # 5. Mean u and v per doubt-community
    comm_stats = []
    for k in np.unique(lbl_doubt):
        mask_k = lbl_doubt == k
        comm_stats.append({
            'comm': k,
            'size': int(mask_k.sum()),
            'mean_u': float(u_arr[:, mask_k].mean()),
            'std_u':  float(u_arr[:, mask_k].std()),
            'mean_v': float(v_arr[:, mask_k].mean()),
        })

    return {
        'C_u': C_u,
        'lbl_doubt': lbl_doubt,
        'lbl_struct': lbl_struct,
        'n_comm_doubt': n_comm_doubt,
        'n_comm_struct': n_comm_struct,
        'nmi': nmi_score,
        'n_doubt_edges': G_doubt.number_of_edges(),
        'comm_stats': comm_stats,
        'adj': adj,
        'N': N,
    }


# -- Main ---------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 80)
    print("Item 12 -- Doubt-Driven Community Detection")
    print(f"I_stim={I_STIM} | steps={STEPS} | warm_up={WARM_UP}")
    print(f"Corr threshold theta={CORR_THETA} | seeds={SEEDS}")
    print("=" * 80)

    t0   = time.time()
    rows = []
    best_results = {}   # for figures: keep one seed per topo

    for topo in ['lattice', 'ba_m3']:
        print(f"\nTopologie : {topo}")
        print(f"  {'seed':>5}  {'#comm_doubt':>12}  {'#comm_struct':>13}  "
              f"{'NMI':>7}  {'doubt_edges':>12}")
        nmi_list = []
        for seed in SEEDS:
            res = run_one(topo, seed)
            nmi_list.append(res['nmi'])
            print(f"  {seed:>5}  {res['n_comm_doubt']:>12}  {res['n_comm_struct']:>13}  "
                  f"{res['nmi']:>7.4f}  {res['n_doubt_edges']:>12}")
            rows.append({
                'topo': topo, 'seed': seed,
                'n_comm_doubt': res['n_comm_doubt'],
                'n_comm_struct': res['n_comm_struct'],
                'nmi': res['nmi'],
                'n_doubt_edges': res['n_doubt_edges'],
            })
            # Garder seed=42 pour la figure
            if seed == 42:
                best_results[topo] = res

        print(f"  --> NMI mean={np.mean(nmi_list):.4f}  std={np.std(nmi_list):.4f}")

        # Detail des communautes du doute (seed=42)
        res42 = best_results[topo]
        print(f"\n  Communautes du doute (seed=42) :")
        for cs in sorted(res42['comm_stats'], key=lambda x: -x['size']):
            print(f"    comm {cs['comm']} : size={cs['size']:3d}  "
                  f"mean_u={cs['mean_u']:.3f}  std_u={cs['std_u']:.3f}  "
                  f"mean_v={cs['mean_v']:.3f}")

    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed:.1f}s")

    # -- CSV ------------------------------------------------------------------
    import csv, pathlib
    fig_dir = pathlib.Path(__file__).resolve().parents[1] / 'figures'
    fig_dir.mkdir(exist_ok=True)
    csv_path = fig_dir / 'p2_doubt_community_detection.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV : {csv_path}")

    # -- Figure ---------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        for row_idx, topo in enumerate(['lattice', 'ba_m3']):
            res = best_results[topo]
            N   = res['N']
            C_u = res['C_u']

            ax_corr, ax_doubt, ax_struct = axes[row_idx]

            # Panel 1 : heatmap correlation u
            im = ax_corr.imshow(C_u, vmin=-1, vmax=1, cmap='RdBu_r',
                                aspect='auto')
            plt.colorbar(im, ax=ax_corr, fraction=0.046)
            ax_corr.set_title(f'{topo} | u(t) correlation matrix\n'
                              f'(seed=42, T={STEPS-WARM_UP})')
            ax_corr.set_xlabel('node'); ax_corr.set_ylabel('node')

            # Panel 2 : communautes du doute sur graphe structural
            try:
                import networkx as nx
                G_struct = nx.from_numpy_array(res['adj'])
                pos = nx.spring_layout(G_struct, seed=42, k=0.3)
                lbl_d = res['lbl_doubt']
                n_cd  = res['n_comm_doubt']
                cmap  = plt.cm.get_cmap('tab10', max(n_cd, 1))
                node_colors = [cmap(lbl_d[i]) for i in range(N)]
                nx.draw_networkx(G_struct, pos=pos, ax=ax_doubt,
                                 node_color=node_colors, node_size=50,
                                 with_labels=False, edge_color='gray',
                                 alpha=0.7, width=0.4)
                ax_doubt.set_title(f'Doubt communities ({n_cd} comms)\n'
                                   f'NMI vs struct = {res["nmi"]:.3f}')
                ax_doubt.axis('off')

                # Panel 3 : communautes structurelles
                lbl_s = res['lbl_struct']
                n_cs  = res['n_comm_struct']
                cmap2 = plt.cm.get_cmap('tab10', max(n_cs, 1))
                node_colors2 = [cmap2(lbl_s[i]) for i in range(N)]
                nx.draw_networkx(G_struct, pos=pos, ax=ax_struct,
                                 node_color=node_colors2, node_size=50,
                                 with_labels=False, edge_color='gray',
                                 alpha=0.7, width=0.4)
                ax_struct.set_title(f'Structural communities ({n_cs} comms)')
                ax_struct.axis('off')
            except Exception as e:
                ax_doubt.text(0.5, 0.5, str(e), ha='center', va='center')
                ax_struct.text(0.5, 0.5, str(e), ha='center', va='center')

        fig.suptitle(
            f'Item 12 -- Doubt-Driven Community Detection\n'
            f'I_stim={I_STIM}, theta={CORR_THETA}, coupling=degree_linear',
            fontsize=11
        )
        plt.tight_layout()
        png_path = fig_dir / 'p2_doubt_community_detection.png'
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        print(f"Figure : {png_path}")
    except Exception as e:
        print(f"[matplotlib error] {e}")
