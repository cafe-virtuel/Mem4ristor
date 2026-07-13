import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from mem4ristor.topology import Mem4Network
from mem4ristor.graph_utils import make_lattice_adj
import p11_warm_start_poc as pw
import p11_coupled_pipeline_poc as pc  # This overrides pw.T_READ to 30

def run_diagnostic():
    chain_seeds = list(range(40))
    
    agree_u_m = []
    conflict_u_m = []
    
    guesses = []
    b1s = []
    
    for seed in chain_seeds:
        mask, idle = pw.build_group(seed)
        b1 = 1 if seed % 2 == 0 else -1
        b1s.append(b1)
        
        # Tour 1
        net = Mem4Network(size=pw.SIDE, heretic_ratio=0.0, seed=seed * 10 + 1,
                          adjacency_matrix=make_lattice_adj(pw.SIDE, periodic=True))
        net.model.cfg['complex_doubt']['enabled'] = True
        
        stim1 = np.zeros(pw.N)
        stim1[mask] = b1 * pw.B_E
        for _ in range(pw.T_READ):
            net.step(I_stimulus=stim1)
            
        diff1 = float(np.real(net.model.u_c[idle].mean() - net.model.u_c[mask].mean()))
        guess1 = 1 if diff1 >= 0 else -1
        guesses.append(guess1)
        
        # Clone for two alternative Tour 2s
        import copy
        net_same = net
        net_flip = copy.deepcopy(net)
        
        # Tour 2 SAME (stimulus is b1)
        stim_same = np.zeros(pw.N)
        stim_same[mask] = b1 * pw.B_E
        for _ in range(pw.T_READ):
            net_same.step(I_stimulus=stim_same)
        u_m_same = float(np.abs(net_same.model.u_c[mask]).mean())
        
        # Tour 2 FLIP (stimulus is -b1)
        stim_flip = np.zeros(pw.N)
        stim_flip[mask] = -b1 * pw.B_E
        for _ in range(pw.T_READ):
            net_flip.step(I_stimulus=stim_flip)
        u_m_flip = float(np.abs(net_flip.model.u_c[mask]).mean())
        
        if b1 == guess1:
            agree_u_m.append(u_m_same)
        else:
            conflict_u_m.append(u_m_same)
            
        if -b1 == guess1:
            agree_u_m.append(u_m_flip)
        else:
            conflict_u_m.append(u_m_flip)
            
    agree_u_m = np.array(agree_u_m)
    conflict_u_m = np.array(conflict_u_m)
    
    print("=== DOUBT MAGNITUDE AT END OF TOUR 2 ===")
    print(f"Agreement u_m: mean={agree_u_m.mean():.4f}, std={agree_u_m.std():.4f}, min={agree_u_m.min():.4f}, max={agree_u_m.max():.4f}")
    print(f"Conflict u_m:  mean={conflict_u_m.mean():.4f}, std={conflict_u_m.std():.4f}, min={conflict_u_m.min():.4f}, max={conflict_u_m.max():.4f}")
    
    # Check threshold separation
    best_th = 0
    best_correct = 0
    for th in np.linspace(0, 0.5, 1000):
        # We expect Conflict to have HIGHER doubt magnitude than Agreement!
        agree_ok = (agree_u_m < th).sum()
        conflict_ok = (conflict_u_m >= th).sum()
        total = agree_ok + conflict_ok
        if total > best_correct:
            best_correct = total
            best_th = th
            
    print(f"Best threshold = {best_th:.4f} gives {best_correct}/80 correct classifications ({(best_correct/80)*100:.1f}%)")

if __name__ == "__main__":
    run_diagnostic()
