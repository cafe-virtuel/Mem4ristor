import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os

LAMBDA2_CRIT = 2.31

def main():
    print("=== PLOTTING DATA COLLAPSE ===")
    u4_file = '../figures/v6_binder_cumulant_U4.csv'
    if not os.path.exists(u4_file):
        print("Erreur: Fichier U4 introuvable.")
        return
        
    df = pd.read_csv(u4_file)
    
    # Les tailles N à afficher
    sizes = sorted(df['N'].unique())
    colors = {100: 'blue', 200: 'orange', 400: 'green', 800: 'red', 1600: 'purple'}
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- Panneau Gauche: Binder Cumulant ---
    ax = axes[0]
    for N in sizes:
        sub = df[df['N'] == N].sort_values('lambda2_mean')
        if len(sub) > 0:
            ax.plot(sub['lambda2_mean'], sub['U4'], marker='o', label=f'N={N}', color=colors.get(N, 'black'), alpha=0.8)
            
    ax.axvline(LAMBDA2_CRIT, color='black', linestyle='--', label=f'$\lambda_{{2,crit}}$ ({LAMBDA2_CRIT})')
    ax.set_xlabel('Algebraic Connectivity $\lambda_2$')
    ax.set_ylabel('Binder Cumulant $U_4$')
    ax.set_title('Finite-Size Scaling of the Order Parameter')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # --- Panneau Droit: Data Collapse de H_stable ---
    ax2 = axes[1]
    for N in sizes:
        sub = df[df['N'] == N].sort_values('lambda2_mean')
        if len(sub) > 0:
            ax2.plot(sub['lambda2_mean'], sub['H_mean'], marker='o', label=f'N={N}', color=colors.get(N, 'black'), alpha=0.8)
            
    ax2.axvline(LAMBDA2_CRIT, color='black', linestyle='--')
    ax2.set_xlabel('Algebraic Connectivity $\lambda_2$')
    ax2.set_ylabel('Stable Entropy $\langle H_{\mathrm{stable}} \\rangle$')
    ax2.set_title('Order Parameter Convergence')
    ax2.grid(True, alpha=0.3)
    
    # --- Inset: Dérivée (Susceptibilité) ---
    axins = inset_axes(ax2, width="35%", height="35%", loc='upper right', borderpad=2)
    
    # On calcule la dérivée pour la plus grande taille N disponible (idéalement 1600 ou 800)
    N_max = max(sizes)
    sub_max = df[df['N'] == N_max].sort_values('lambda2_mean')
    if len(sub_max) > 1:
        x = sub_max['lambda2_mean'].values
        y = sub_max['H_mean'].values
        
        # Lissage simple avant dérivée pour éviter le bruit
        y_smooth = np.convolve(y, np.ones(3)/3, mode='valid')
        x_smooth = np.convolve(x, np.ones(3)/3, mode='valid')
        
        dy_dx = np.abs(np.gradient(y_smooth, x_smooth))
        
        axins.plot(x_smooth, dy_dx, color=colors.get(N_max, 'black'), marker='.')
        axins.axvline(LAMBDA2_CRIT, color='red', linestyle=':', alpha=0.5)
        axins.set_title('$|dH/d\lambda_2|$', fontsize=10)
        axins.tick_params(axis='both', which='major', labelsize=8)
        axins.grid(True, alpha=0.2)
        
    plt.tight_layout()
    plt.savefig('../figures/v6_binder_cumulant.png', dpi=150)
    print("Figure sauvegardée avec Data Collapse et Inset : ../figures/v6_binder_cumulant.png")

if __name__ == "__main__":
    main()
