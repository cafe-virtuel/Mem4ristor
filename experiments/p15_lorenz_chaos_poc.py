"""
EXPERIMENT P15 : Poursuite de Dynamic Chaotique (Lorenz Attractor) sous Bruit Non-Stationnaire.

Contexte :
Les signaux spatio-temporels chaotiques (ex: attracteur de Lorenz) possèdent une dynamique
non-linéaire complexe à deux ailes. Les filtres scalaires simples (ex: Filtre RC Passe-bas)
lissent le bruit stationnaire mais échouent ou prennent un retard considérable lors des bascules
brusques de régime (chocs non-stationnaires).

Mem4ristor utilise le doute u_i pour adapter le gain de couplage lors des ruptures :
quand une rupture d'attracteur survient, le désaccord monte, u s'élève et réinitialise
la réceptivité du réseau pour une capture quasi-instantanée du nouvel état.

Algorithmes comparés :
1. Mem4ristor V3 (FULL - doute u adaptatif)
2. Mem4ristor FROZEN_U (doute u fixe)
3. Filtre RC Exponentiel (RC Low-pass Filter)
4. Passe-Bas Moyen Mobile
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# S'assurer que src/ est dans le path Python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mem4ristor.graph_utils import make_lattice_adj
from mem4ristor.dynamics import Mem4ristorV3

def generate_lorenz_signal(steps=3000, dt=0.01, sigma=10.0, rho=28.0, beta=8.0/3.0):
    """Génère la trajectoire chaotique de Lorenz (x, y, z)."""
    xs = np.zeros(steps)
    ys = np.zeros(steps)
    zs = np.zeros(steps)
    
    x, y, z = 0.1, 0.0, 0.0
    for i in range(steps):
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        
        x += dx * dt
        y += dy * dt
        z += dz * dt
        
        xs[i] = x
        ys[i] = y
        zs[i] = z
        
    return xs, ys, zs

def solve_rc_filter(signal, alpha=0.05):
    """Filtre passe-bas RC exponentiel classique : y[t] = alpha * x[t] + (1-alpha) * y[t-1]."""
    filtered = np.zeros_like(signal)
    val = signal[0]
    for i in range(len(signal)):
        val = alpha * signal[i] + (1.0 - alpha) * val
        filtered[i] = val
    return filtered

def solve_moving_average(signal, window=20):
    """Filtre moyenne mobile."""
    filtered = np.zeros_like(signal)
    for i in range(len(signal)):
        start = max(0, i - window + 1)
        filtered[i] = np.mean(signal[start:i+1])
    return filtered

def run_lorenz_benchmark(n_seeds=10, N=100, steps=3000):
    print("=" * 70)
    print(f"BENCHMARK P15 : POURSUITE CHAOTIQUE DE LORENZ (N={N}, Seeds={n_seeds})")
    print("=" * 70)
    
    # Génération du signal de Lorenz cible
    raw_x, raw_y, raw_z = generate_lorenz_signal(steps=steps)
    # Normalisation du signal x dans [-1.5, +1.5] pour stimuler les neurones
    signal_clean = 1.5 * (raw_x / np.max(np.abs(raw_x)))
    
    results = []
    
    for seed in range(n_seeds):
        np.random.seed(seed)
        
        # Ajout de bruit non-stationnaire : bruit gaussien + chocs impulsionnels soudains à t=1000 et t=2000
        noise = np.random.normal(0, 0.2, steps)
        # Chocs non-stationnaires (sauts brusques)
        shock_mask = np.zeros(steps)
        shock_mask[1000:1050] = 2.0  # Choc 1
        shock_mask[2000:2050] = -2.0 # Choc 2
        
        signal_noisy = signal_clean + noise + shock_mask
        
        # 1. Filtre RC
        rc_pred = solve_rc_filter(signal_noisy, alpha=0.1)
        mse_rc = np.mean((rc_pred[500:] - signal_clean[500:]) ** 2)
        
        # 2. Moyenne Mobile
        ma_pred = solve_moving_average(signal_noisy, window=15)
        mse_ma = np.mean((ma_pred[500:] - signal_clean[500:]) ** 2)
        
        # 3. Mem4ristor FROZEN_U
        adj_grid = make_lattice_adj(int(np.sqrt(N)))
        model_frozen = Mem4ristorV3(seed=seed)
        model_frozen._initialize_params(N, cold_start=True)
        frozen_pred = np.zeros(steps)
        
        for t in range(steps):
            model_frozen.u[:] = 0.5 # Doute fixe
            l_v = np.dot(adj_grid, model_frozen.v) - model_frozen.v
            model_frozen.step(I_stimulus=signal_noisy[t], coupling_input=l_v)
            frozen_pred[t] = np.mean(model_frozen.v)
            
        mse_frozen = np.mean((frozen_pred[500:] - signal_clean[500:]) ** 2)
        
        # 4. Mem4ristor FULL (doute adaptatif u)
        model_full = Mem4ristorV3(seed=seed)
        model_full._initialize_params(N, cold_start=True)
        full_pred = np.zeros(steps)
        u_trace = np.zeros(steps)
        
        for t in range(steps):
            l_v = np.dot(adj_grid, model_full.v) - model_full.v
            model_full.step(I_stimulus=signal_noisy[t], coupling_input=l_v)
            full_pred[t] = np.mean(model_full.v)
            u_trace[t] = np.mean(model_full.u)
            
        mse_full = np.mean((full_pred[500:] - signal_clean[500:]) ** 2)
        
        # Mesure du temps de récupération après choc (t=1000)
        # Erreur post-choc t=1050..1200
        error_rc_shock = np.mean(np.abs(rc_pred[1050:1200] - signal_clean[1050:1200]))
        error_full_shock = np.mean(np.abs(full_pred[1050:1200] - signal_clean[1050:1200]))
        
        results.append({
            'seed': seed,
            'mse_rc': mse_rc,
            'mse_ma': mse_ma,
            'mse_frozen': mse_frozen,
            'mse_full': mse_full,
            'error_rc_shock': error_rc_shock,
            'error_full_shock': error_full_shock
        })
        
        print(f"Seed {seed+1:02d}/{n_seeds:02d} | "
              f"MSE RC: {mse_rc:.4f} | "
              f"MSE MA: {mse_ma:.4f} | "
              f"MSE Frozen: {mse_frozen:.4f} | "
              f"MSE M4R FULL: {mse_full:.4f}")
        
    df = pd.DataFrame(results)
    
    # Enregistrer le CSV
    os.makedirs("figures", exist_ok=True)
    csv_path = "figures/p15_lorenz_benchmark.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[CSV enregistré] : {csv_path}")
    
    # Synthèse
    print("\n" + "=" * 50)
    print("RÉSULTATS DE POURSUITE (MSE MOYEN) :")
    print(f"  Filtre RC Exponentiel : MSE = {df['mse_rc'].mean():.4f}")
    print(f"  Moyenne Mobile        : MSE = {df['mse_ma'].mean():.4f}")
    print(f"  Mem4ristor FROZEN     : MSE = {df['mse_frozen'].mean():.4f}")
    print(f"  Mem4ristor FULL       : MSE = {df['mse_full'].mean():.4f}")
    print(f"  Erreur Post-Choc RC   : {df['error_rc_shock'].mean():.4f}")
    print(f"  Erreur Post-Choc M4R  : {df['error_full_shock'].mean():.4f}")
    print("=" * 50)
    
    # Graphic output
    plt.figure(figsize=(12, 6))
    plt.plot(signal_clean[800:1500], 'k--', label='Signal Chaotique Lorenz (Vrai)', linewidth=2)
    plt.plot(signal_noisy[800:1500], color='gray', alpha=0.3, label='Bruit Non-Stationnaire')
    plt.plot(rc_pred[800:1500], 'r-', label='Filtre RC Exponentiel', alpha=0.8)
    plt.plot(full_pred[800:1500], 'g-', label='Mem4ristor FULL (Doute u)', linewidth=2)
    
    plt.axvline(x=200, color='orange', linestyle=':', label='Impulsion Choc (t=1000)')
    plt.ylabel('Amplitude du Signal', fontsize=12)
    plt.xlabel('Pas de Temps (Extrait t=800..1500)', fontsize=12)
    plt.title('Poursuite de Signal Chaotique de Lorenz sous Choc Non-Stationnaire', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    png_path = "figures/p15_lorenz_tracking.png"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Graphique enregistré] : {png_path}")

if __name__ == "__main__":
    run_lorenz_benchmark(n_seeds=10, N=100, steps=3000)
