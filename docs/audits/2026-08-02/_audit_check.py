import sys, os
# Ce script vivait a la racine du depot le 02/08/2026 ; il a ete range dans
# docs/audits/2026-08-02/ le jour meme. La racine est donc trois niveaux au-dessus.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import numpy as np, networkx as nx
from mem4ristor.core import Mem4Network
from mem4ristor.metrics import calculate_cognitive_entropy

net = Mem4Network(size=10, heretic_ratio=0.15, seed=42)
for s in range(1000):
    net.step(I_stimulus=0.0)
print("README quickstart : H_default(cont100)=%.4f | H_cog=%.4f"
      % (net.calculate_entropy(), calculate_cognitive_entropy(net.v)))

G = nx.barabasi_albert_graph(100, 3, seed=42)
adj = nx.to_numpy_array(G)
n2 = Mem4Network(adjacency_matrix=adj, heretic_ratio=0.15,
                 coupling_norm='degree_linear', seed=42)
for s in range(3000):
    n2.step(I_stimulus=0.0)
print("README scale-free (attendu ~0.83) : H_default(cont100)=%.4f | H_cog=%.4f"
      % (n2.calculate_entropy(), calculate_cognitive_entropy(n2.v)))

a = Mem4Network(size=10, heretic_ratio=0.15, seed=42)
b = Mem4Network(size=10, heretic_ratio=0.0, seed=42)
for s in range(300):
    a.step(0.0); b.step(0.0)
print("heretiques 15%% vs 0%% a I_stim=0 : max|dv|=%.3e"
      % float(np.max(np.abs(a.v - b.v))))

c = Mem4Network(size=10, seed=42)
c.model.cfg['coupling']['D'] = 10.0
d = Mem4Network(size=10, seed=42)
for s in range(200):
    c.step(0.0); d.step(0.0)
print("cfg['coupling']['D']=10 apres ctor vs 0.15 : max|dv|=%.3e (0 => D ignore)"
      % float(np.max(np.abs(c.v - d.v))))

m = Mem4Network(size=10, heretic_ratio=0.15, seed=42)
ids = np.where(m.model.heretic_mask)[0]
print("heretiques N=100 uniform -> n=%d indices=%s max=%d/99"
      % (len(ids), list(ids), ids.max()))
m2 = Mem4Network(size=20, heretic_ratio=0.15, seed=42)
i2 = np.where(m2.model.heretic_mask)[0]
print("heretiques N=400 uniform -> n=%d max=%d/399 (couverture %.0f%%)"
      % (len(i2), i2.max(), 100 * i2.max() / 399))
