import sys, os
# Ce script vivait a la racine du depot le 02/08/2026 ; il a ete range dans
# docs/audits/2026-08-02/ le jour meme. La racine est donc trois niveaux au-dessus.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import numpy as np
from mem4ristor.core import Mem4Network
from mem4ristor.dynamics import Mem4ristorV3
from mem4ristor.metrics import (calculate_continuous_entropy,
                                calculate_pairwise_synchrony)

# --- 1. l'effet "heretique" a I_stim=0 est-il physique ou juste un decalage RNG ?
def run(hr, sigma_v, steps=300):
    n = Mem4Network(size=10, heretic_ratio=hr, seed=42)
    n.model.cfg['noise']['sigma_v'] = sigma_v
    for _ in range(steps):
        n.step(0.0)
    return n.v.copy()

print("[1] heretiques a I_stim=0")
print("    avec bruit  sigma_v=0.05 : max|dv| =", "%.3e" % np.max(np.abs(run(0.15, 0.05) - run(0.0, 0.05))))
print("    SANS bruit  sigma_v=0.0  : max|dv| =", "%.3e" % np.max(np.abs(run(0.15, 0.0) - run(0.0, 0.0))))

# --- 2. entropie continue : fuite hors de la fenetre [-3, 3]
n = Mem4Network(size=10, heretic_ratio=0.15, seed=42)
for _ in range(1000):
    n.step(0.0)
v = n.v
out = int(np.sum((v < -3) | (v > 3)))
print("[2] entropie continue, fenetre figee [-3,3]")
print("    v_min=%.3f v_max=%.3f | noeuds hors fenetre = %d/%d (ignores en silence)"
      % (v.min(), v.max(), out, len(v)))
big = np.concatenate([v, np.array([50.0, -50.0])])
print("    H(v)=%.4f  vs  H(v + 2 noeuds a +-50)=%.4f  -> l'explosion n'apparait pas"
      % (calculate_continuous_entropy(v), calculate_continuous_entropy(big)))

# --- 3. le sous-echantillonnage de paires de calculate_pairwise_synchrony sert-il ?
import time
for N in (200, 800):
    h = np.random.RandomState(0).randn(300, N)
    t0 = time.time(); calculate_pairwise_synchrony(h); dt = time.time() - t0
    print("[3] synchrony N=%4d : %.2f s  (cout einsum O(T*N^2) malgre MAX_PAIRS=2000)" % (N, dt))

# --- 4. solve_rk45 vs step() : meme modele ?
m = Mem4ristorV3(seed=42)
m._initialize_params(16)
m.cfg['noise']['sigma_v'] = 0.0
from mem4ristor.graph_utils import make_lattice_adj
adj = make_lattice_adj(4)
v0, w0, u0 = m.v.copy(), m.w.copy(), m.u.copy()
for _ in range(20):
    m.step(0.0, adj @ m.v - np.array(adj.sum(axis=1)) * m.v)
v_euler = m.v.copy()
m.v, m.w, m.u = v0.copy(), w0.copy(), u0.copy()
m.solve_rk45((0.0, 20 * 0.05), 0.0, adj_matrix=adj)
print("[4] solve_rk45 vs step() (sigma_v=0, meme t final) : max|dv| = %.3e"
      % np.max(np.abs(m.v - v_euler)))

# --- 5. les benchmarks peuvent-ils atteindre les memes etats que Mem4ristor ?
from mem4ristor.benchmarks.engine import KuramotoModel, VoterModel, ConsensusModel
from mem4ristor.metrics import get_cognitive_states
for name, mdl in (("Kuramoto", KuramotoModel(100)), ("Voter", VoterModel(100)),
                  ("Consensus", ConsensusModel(100))):
    for _ in range(500):
        mdl.step(0.0)
    print("[5] %-9s etats internes possibles = %s | get_cognitive_states(v) = %s"
          % (name, sorted(set(mdl.get_states().tolist())),
             sorted(set(get_cognitive_states(mdl.v).tolist()))))
print("    Mem4ristor      get_cognitive_states(v) = %s"
      % sorted(set(get_cognitive_states(v).tolist())))
