import sys, os, types
# Ce script vivait a la racine du depot le 02/08/2026 ; il a ete range dans
# docs/audits/2026-08-02/ le jour meme. La racine est donc trois niveaux au-dessus.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

# 1. __all__ liste les symboles viz meme si matplotlib est absent -> `import *` casse
sys.modules['matplotlib'] = None  # simule matplotlib absent
for m in [k for k in list(sys.modules) if k.startswith('mem4ristor')]:
    del sys.modules[m]
try:
    ns = {}
    exec("from mem4ristor import *", ns)
    print("[1] import * sans matplotlib : OK")
except Exception as e:
    print("[1] import * sans matplotlib : %s: %s" % (type(e).__name__, e))

del sys.modules['matplotlib']
for m in [k for k in list(sys.modules) if k.startswith('mem4ristor')]:
    del sys.modules[m]

# 2. Mem4Config.from_yaml perd silencieusement les sections V5/V6
from mem4ristor.config import Mem4Config
import yaml
p = os.path.join(ROOT, 'src', 'mem4ristor', 'config.yaml')
raw = yaml.safe_load(open(p))
kept = set(Mem4Config.from_yaml(p).to_dict().keys())
print("[2] config.yaml sections=%s" % sorted(raw.keys()))
print("    Mem4Config garde =%s" % sorted(kept))
print("    PERDUES en silence =%s" % sorted(set(raw) - kept))

# 3. deux sources de config divergentes selon qu'on passe un dict ou non
from mem4ristor.dynamics import Mem4ristorV3
a = Mem4ristorV3()                      # lit config.yaml
b = Mem4ristorV3(config={'dynamics': {'dt': 0.05}})   # NE lit PAS config.yaml
print("[3] Mem4ristorV3() sections            = %s" % sorted(a.cfg.keys()))
print("    Mem4ristorV3(config={...}) sections= %s" % sorted(b.cfg.keys()))
print("    perdues quand on passe un dict     = %s" % sorted(set(a.cfg) - set(b.cfg)))

# 4. l'operateur de couplage de solve_rk45 n'est pas celui de Mem4Network.step
import numpy as np
from mem4ristor.graph_utils import make_lattice_adj
adj = make_lattice_adj(4)
v = np.arange(16, dtype=float)
lap_step = adj @ v - adj.sum(axis=1) * v          # -(L @ v), ce que fait topology.step
lap_rk45 = adj @ v - v                            # ce que fait solve_rk45 (ligne 540)
print("[4] operateur de couplage, ecart max = %.3f (0 => identiques)"
      % float(np.max(np.abs(lap_step - lap_rk45))))
print("    -> solve_rk45 integre un AUTRE modele que step()")

# 5. NaN echappe au garde-fou ±100
m = Mem4ristorV3(seed=1); m._initialize_params(10)
m.v[0] = np.nan
try:
    m.step(0.0)
    print("[5] un v=NaN traverse step() sans erreur, remis a 0 en silence : v[0]=%.3f" % m.v[0])
except OverflowError as e:
    print("[5] OverflowError leve : %s" % e)
