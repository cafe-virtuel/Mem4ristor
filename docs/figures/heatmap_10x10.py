# heatmap_10x10.py
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('figures', exist_ok=True)

# Données
np.random.seed(42)
states = np.random.choice([-2, -1, 0, 1, 2], size=(10, 10), 
                         p=[0.0, 0.15, 0.73, 0.10, 0.02])

# Figure
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(states, cmap='viridis', vmin=-2, vmax=2)

# Grille
ax.grid(which='both', color='gray', linestyle='-', linewidth=0.3, alpha=0.3)

# Labels
ax.set_xlabel('Column Index', fontsize=11)
ax.set_ylabel('Row Index', fontsize=11)
ax.set_title('Mem4ristor v2.0: Cognitive State Distribution (10×10 Network)', 
             fontsize=13, pad=15)

# Barre de couleur
cbar = fig.colorbar(im, ax=ax, ticks=[-2, -1, 0, 1, 2])
cbar.ax.set_yticklabels(['Oracle', 'Intuition', 'Uncertain', 'Probable', 'Certitude'])

# Sauvegarde
plt.tight_layout()
plt.savefig('figures/heatmap_10x10.pdf', dpi=300, bbox_inches='tight')
print("✅ Heatmap sauvegardée")
plt.show()



