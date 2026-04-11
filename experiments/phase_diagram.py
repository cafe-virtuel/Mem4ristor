"""
Phase Diagram : H(heretic_ratio, D)
====================================
Balayage 2D de l'espace des paramètres pour identifier les régions
de stabilité de la diversité cognitive.

Produit une heatmap publication-ready :
  - Axe X : heretic_ratio ∈ [0, 0.5]
  - Axe Y : D (couplage) ∈ [0.01, 0.5]
  - Couleur : Shannon Entropy H (bits)

Usage :
  python experiments/phase_diagram.py
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from mem4ristor.core import Mem4ristorV2, Mem4Network


def run_single_point(size, heretic_ratio, D, steps=500, seed=42):
    """Run a single (heretic_ratio, D) configuration, return mean H."""
    net = Mem4Network(size=size, heretic_ratio=heretic_ratio, seed=seed,
                      boundary='periodic')
    # Override coupling strength D
    net.model.cfg['coupling']['D'] = D
    net.model.D_eff = D / np.sqrt(net.N)
    
    H_values = []
    for t in range(steps):
        net.step(I_stimulus=0.0)
        if t >= 100:  # Skip transient
            H_values.append(net.calculate_entropy(use_cognitive_bins=True))
    
    return np.mean(H_values) if H_values else 0.0


def main():
    print("=" * 65)
    print("  PHASE DIAGRAM : H(heretic_ratio, D)")
    print("=" * 65)
    
    SIZE = 8  # 8x8 = 64 units (faster than 10x10 for sweep)
    STEPS = 500
    N_HR = 20   # Resolution on heretic_ratio axis
    N_D = 20    # Resolution on D axis
    
    hr_values = np.linspace(0.0, 0.50, N_HR)
    D_values = np.linspace(0.01, 0.50, N_D)
    
    total = N_HR * N_D
    print(f"\n  Grid: {N_HR}×{N_D} = {total} simulations")
    print(f"  Network: {SIZE}×{SIZE} = {SIZE**2} units")
    print(f"  Steps: {STEPS} (skip first 100)\n")
    
    H_grid = np.zeros((N_D, N_HR))
    
    done = 0
    for i, D in enumerate(D_values):
        for j, hr in enumerate(hr_values):
            H_grid[i, j] = run_single_point(SIZE, hr, D, steps=STEPS, seed=42)
            done += 1
            if done % 20 == 0:
                pct = done / total * 100
                print(f"  [{done}/{total}] ({pct:.0f}%) D={D:.2f}, hr={hr:.2f} → H={H_grid[i,j]:.3f}")
    
    print(f"\n  ✅ Balayage terminé. {total} simulations.")
    
    # ── Statistics ──
    print(f"\n  H_max = {np.max(H_grid):.3f}")
    print(f"  H_min = {np.min(H_grid):.3f}")
    print(f"  H_mean = {np.mean(H_grid):.3f}")
    
    # Region where H > 1.0 (good diversity)
    good_region = np.sum(H_grid > 1.0) / total * 100
    print(f"  Region H > 1.0 : {good_region:.1f}% of parameter space")
    
    # ── Plot ──
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    im = ax.imshow(H_grid, origin='lower', aspect='auto', cmap='RdYlGn',
                   extent=[hr_values[0], hr_values[-1], D_values[0], D_values[-1]],
                   interpolation='bilinear')
    
    cbar = plt.colorbar(im, ax=ax, label='Shannon Entropy H (bits)')
    
    # Contour lines
    contour = ax.contour(hr_values, D_values, H_grid, levels=[0.5, 1.0, 1.5],
                         colors='black', linewidths=1.0, linestyles=['--', '-', '--'])
    ax.clabel(contour, inline=True, fontsize=9, fmt='H=%.1f')
    
    # Mark the default parameters
    ax.plot(0.15, 0.12, marker='*', color='white', markersize=15, markeredgecolor='black',
            markeredgewidth=1.5, label='Default (hr=0.15, D=0.12)')
    
    ax.set_xlabel('Heretic Ratio', fontsize=13)
    ax.set_ylabel('Coupling Strength D', fontsize=13)
    ax.set_title('Mem4ristor Phase Diagram: Cognitive Diversity H(hr, D)', fontsize=14)
    ax.legend(fontsize=11, loc='upper right')
    
    fig_path = os.path.join(results_dir, 'phase_diagram.png')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"\n  📈 Figure: {os.path.abspath(fig_path)}")
    
    # ── Save data ──
    np.savez(os.path.join(results_dir, 'phase_diagram_data.npz'),
             H_grid=H_grid, hr_values=hr_values, D_values=D_values)
    print(f"  💾 Data: phase_diagram_data.npz")
    
    print("\n" + "=" * 65)
    print(f"  Phase diagram terminé — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)


if __name__ == '__main__':
    main()
