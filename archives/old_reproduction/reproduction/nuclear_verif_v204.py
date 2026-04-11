import numpy as np
import os
import sys
import yaml
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.append(os.path.dirname(__file__))

from reference_impl import Mem4ristorV2

def calculate_gini(probs):
    """Gini index for equality. 0 = total equality, 1 = total inequality."""
    if len(probs) == 0: return 0.0
    probs = np.sort(probs)
    n = len(probs)
    index = np.arange(1, n + 1)
    return (np.sum((2 * index - n - 1) * probs)) / (n * np.sum(probs))

def calculate_metrics(v_states):
    """Calculate H, MDS, Gini, and Max Fraction."""
    counts = np.bincount(v_states, minlength=6)[1:]
    n_total = len(v_states)
    probs = counts / n_total
    
    # Shannon Entropy
    p_nz = probs[probs > 0]
    if len(p_nz) <= 1:
        h = 0.0
    else:
        h = -np.sum(p_nz * np.log2(p_nz))
    
    # Occupied States
    n_occ = np.sum(probs > 0.01) # 1% threshold for "occupied"
    
    # MDS
    mds = h * (n_occ / 5.0)
    
    # Gini
    gini = calculate_gini(probs)
    
    # Max Fraction
    max_frac = np.max(probs)
    
    return {
        "H": h,
        "MDS": mds,
        "Gini": gini,
        "MaxFrac": max_frac,
        "N_occ": n_occ
    }

class NuclearProbe(Mem4ristorV2):
    """Extends Reference Implementation for multi-logic testing."""
    def __init__(self, logic_mode="v2.0.4", ic_data=None):
        super().__init__()
        self.logic_mode = logic_mode
        if ic_data:
            self.v = ic_data['v'].copy()
            self.w = ic_data['w'].copy()
            self.u = ic_data['u'].copy()
            self.heretic_mask = ic_data['heretic_mask'].copy()

    def step(self, I_stimulus=0):
        delta_v = self.adj @ self.v - self.v
        sigma_social = np.abs(delta_v)
        eta = np.random.normal(0, self.cfg['noise']['sigma_v'], self.N)
        
        # Logic Selection
        if self.logic_mode == "v2.0.4":
            u_filter = (1.0 - 2.0 * self.u)
            heretic_inv = True
        elif self.logic_mode == "v2.0.3":
            u_filter = (1.0 - self.u)
            heretic_inv = True
        else: # Ablated (v2.0.1 style or pure consensus)
            u_filter = (1.0 - self.u)
            heretic_inv = False
            
        I_coup = self.D_eff * u_filter * delta_v
        
        I_ext = np.full(self.N, float(I_stimulus))
        if heretic_inv:
            I_ext[self.heretic_mask] *= -1.0
        I_ext += I_coup
        
        dv = (self.v - (self.v**3)/self.cfg['dynamics']['v_cubic_divisor'] - self.w + I_ext - \
              self.cfg['dynamics']['alpha'] * np.tanh(self.v) + eta)
        dw = self.cfg['dynamics']['epsilon'] * (self.v + self.cfg['dynamics']['a'] - self.cfg['dynamics']['b'] * self.w)
        du = self.cfg['doubt']['epsilon_u'] * (self.cfg['doubt']['k_u'] * sigma_social + \
                                               self.cfg['doubt']['sigma_baseline'] - self.u)
        
        self.v += dv * self.dt
        self.w += dw * self.dt
        self.u += du * self.dt
        self.u = np.clip(self.u, 0, 1)

def run_nuclear_suite():
    print("[INFO] NUCLEAR VERIFICATION SUITE v2.2.0")

    print("-----------------------------------")
    
    # 1. Generate Master ICs (Controlled)
    L = 10
    N = L*L
    ic_master = {
        'v': np.zeros(N),
        'w': np.zeros(N),
        'u': np.full(N, 0.05),
        'heretic_mask': np.random.rand(N) < 0.15
    }
    
    # 2. Test A: IC Compression (Ablation Control)
    widths = [0.0, 0.01, 0.1, 1.0]
    results_a = []
    
    for w_val in widths:
        # Re-inject noise into master v
        ic_run = ic_master.copy()
        ic_run['v'] = np.random.uniform(-w_val, w_val, N)
        
        row = {"width": w_val}
        for mode in ["v2.0.4", "v2.0.3", "Ablated"]:
            probe = NuclearProbe(logic_mode=mode, ic_data=ic_run)
            for _ in range(5000): probe.step(1.1)
            metrics = calculate_metrics(probe.get_states())

            row[f"{mode}_H"] = metrics['H']
            row[f"{mode}_MDS"] = metrics['MDS']
        results_a.append(row)
        
    print("\nTEST A: IC COMPRESSION RESULTS (MDS)")
    print(f"{'Width':<8} | {'v204':<8} | {'v203':<8} | {'Ablat':<8}")
    for res in results_a:
        print(f"{res['width']:<8.3f} | {res['v2.0.4_MDS']:<8.3f} | {res['v2.0.3_MDS']:<8.3f} | {res['Ablated_MDS']:<8.3f}")

    # 3. Test B/C: Deep Time + Quality
    print("\nTEST B/C: DEEP TIME & QUALITY TRACE (v2.0.4)")
    model = NuclearProbe(logic_mode="v2.0.4", ic_data=ic_master)
    steps = 10000
    h_trace = []
    mds_trace = []
    gini_trace = []
    max_frac_trace = []
    
    for i in range(steps):
        # Stimulus Ramp
        I_s = 0.5 + 1.0 * (i / steps)
        model.step(I_s)
        
        if i % 100 == 0:
            m = calculate_metrics(model.get_states())
            h_trace.append(m['H'])
            mds_trace.append(m['MDS'])
            gini_trace.append(m['Gini'])
            max_frac_trace.append(m['MaxFrac'])

    avg_gini = np.mean(gini_trace[-20:])
    avg_max_frac = np.mean(max_frac_trace[-20:])
    print(f"Final Gini: {avg_gini:.4f} (Ideal < 0.8)")
    print(f"Final Max Fraction: {avg_max_frac:.4f} (Ideal < 0.6)")
    
    # 4. Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(h_trace, label='Entropy H', alpha=0.8)
    plt.plot(mds_trace, label='MDS (Quality)', color='green', linewidth=2)
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.3, label='H Threshold')
    plt.title("Mem4ristor v2.0.4: Deep Time Quality Trace")
    plt.xlabel("Sample (x100 steps)")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(alpha=0.3)
    save_path = os.path.join(os.path.dirname(__file__), '../results/plots/nuclear_trace_v2.2.0.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"\nGraph saved to results/plots/nuclear_trace_v2.2.0.png")

if __name__ == "__main__":
    run_nuclear_suite()
