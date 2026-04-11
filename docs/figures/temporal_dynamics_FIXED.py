import numpy as np
import matplotlib.pyplot as plt
import os

# Créer le dossier figures
os.makedirs('figures', exist_ok=True)

# ========== DONNÉES SIMULÉES ==========
np.random.seed(42)
time_steps = 1000
t = np.linspace(0, 100, time_steps)

# Dynamiques pour 4 unités (plus de variabilité)
v_heretic = 0.5 * np.sin(0.35*t) + 0.2 * np.sin(0.1*t)  # Hérétique
v_uncertain = 0.3 * np.tanh(0.08*t - 3) + 0.4 * np.sin(0.6*t) * np.exp(-0.008*t)
v_probable = -1.0 + 1.5 * (1 - np.exp(-0.025*t)) * (0.6 + 0.2*np.sin(0.25*t))
v_intuition = 0.5 * np.sin(0.2*t + 2) * (1 - 0.3*np.exp(-0.01*t))

# Bruit réaliste
noise_level = 0.07
v_heretic += noise_level * np.random.randn(time_steps)
v_uncertain += noise_level * np.random.randn(time_steps)
v_probable += noise_level * 0.5 * np.random.randn(time_steps)
v_intuition += noise_level * 0.8 * np.random.randn(time_steps)

# Clip
v_heretic = np.clip(v_heretic, -2, 2)
v_uncertain = np.clip(v_uncertain, -2, 2)
v_probable = np.clip(v_probable, -2, 2)
v_intuition = np.clip(v_intuition, -2, 2)

# Doute constitutionnel (plus dynamique)
u = 0.05 + 0.12 * np.exp(-0.006*t) * (1 + 0.4*np.sin(0.2*t))

# ========== COULEURS SOLIDES (pas de transparence excessive) ==========
colors = {
    'heretic': '#1A936F',      # Vert émeraude
    'uncertain': '#F18F01',    # Orange
    'probable': '#C73E1D',     # Rouge brique
    'intuition': '#A23B72',    # Violet
    'doubt': '#2E86AB',        # Bleu
    'zone_intuition': '#E8D4E8',  # Violet très clair
    'zone_uncertain': '#FFF0D6',  # Orange très clair
    'zone_probable': '#F8D7DA',   # Rouge très clair
}

# ========== FIGURE PRINCIPALE ==========
fig = plt.figure(figsize=(14, 10))

# ---------- A) DYNAMIQUE TEMPORELLE ----------
ax1 = plt.subplot(2, 2, 1)

# Zones de fond SOLIDES (pas alpha)
ax1.fill_between(t, -1.5, -0.8, color=colors['zone_intuition'], label='_Intuition Zone')
ax1.fill_between(t, -0.8, 0.8, color=colors['zone_uncertain'], label='_Uncertain Zone')
ax1.fill_between(t, 0.8, 1.5, color=colors['zone_probable'], label='_Probable Zone')

# Lignes de seuil ÉPAISSES
for threshold, style in [(-1.5, '-'), (-0.8, '--'), (0.8, '--'), (1.5, '-')]:
    ax1.axhline(y=threshold, color='gray', linestyle=style, alpha=0.7, linewidth=1.2)

# Trajectoires (épaissies)
ax1.plot(t, v_heretic, color=colors['heretic'], linewidth=3.0, label='Heretic Unit (15%)')
ax1.plot(t, v_uncertain, color=colors['uncertain'], linewidth=2.5, label='Uncertain Unit')
ax1.plot(t, v_probable, color=colors['probable'], linewidth=2.5, label='Probable Unit')
ax1.plot(t, v_intuition, color=colors['intuition'], linewidth=2.5, label='Intuition Unit')

# Doute sur axe secondaire
ax1u = ax1.twinx()
ax1u.plot(t, u, color=colors['doubt'], linewidth=2.5, linestyle='-', label='Constitutional Doubt $u$')
ax1u.fill_between(t, 0, u, color=colors['doubt'], alpha=0.15)

# Configuration axes
ax1.set_xlim(0, 100)
ax1.set_ylim(-2.0, 2.0)
ax1.set_ylabel('Cognitive Potential $v$', fontsize=12, fontweight='bold')
ax1.set_xlabel('Time (arbitrary units)', fontsize=11)
ax1u.set_ylim(0, 0.25)
ax1u.set_ylabel('Doubt $u$', fontsize=11, rotation=270, labelpad=20)

# Légende
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1u.get_legend_handles_labels()
# Prendre seulement la première légende de zone
filtered_lines = [lines1[0], lines1[1], lines1[2], lines1[3], lines2[0]]
filtered_labels = ['Heretic Unit (15%)', 'Uncertain Unit', 'Probable Unit', 'Intuition Unit', 'Constitutional Doubt $u$']
ax1.legend(filtered_lines, filtered_labels, loc='upper right', fontsize=9, framealpha=0.95)

ax1.set_title('A) Temporal Cognitive Dynamics', fontsize=13, fontweight='bold', pad=12)
ax1.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)

# ---------- B) DISTRIBUTION TEMPORELLE ----------
ax2 = plt.subplot(2, 2, 2)

# Données de distribution
time_points = ['t=0', 't=20', 't=40', 't=60', 't=80', 't=100']
x = np.arange(len(time_points))
width = 0.18

# Barres SOLIDES
ax2.bar(x - 1.5*width, [0.30, 0.25, 0.18, 0.15, 0.15, 0.15], width, 
        color=colors['intuition'], edgecolor='black', linewidth=0.5, label='Intuition')
ax2.bar(x - 0.5*width, [0.60, 0.65, 0.70, 0.73, 0.73, 0.73], width, 
        color=colors['uncertain'], edgecolor='black', linewidth=0.5, label='Uncertain')
ax2.bar(x + 0.5*width, [0.10, 0.08, 0.10, 0.10, 0.10, 0.10], width, 
        color=colors['probable'], edgecolor='black', linewidth=0.5, label='Probable')
ax2.bar(x + 1.5*width, [0.00, 0.02, 0.02, 0.02, 0.02, 0.02], width, 
        color='#6B2737', edgecolor='black', linewidth=0.5, label='Certitude')

ax2.set_xticks(x)
ax2.set_xticklabels(time_points, fontsize=10, rotation=0)
ax2.set_ylabel('Fraction of Units', fontsize=11)
ax2.set_ylim(0, 1.0)
ax2.set_title('B) Evolution of State Distribution', fontsize=13, fontweight='bold', pad=12)
ax2.legend(fontsize=9, loc='upper right', framealpha=0.95)
ax2.grid(True, alpha=0.2, axis='y', linestyle='-', linewidth=0.5)

# ---------- C) MÉTRIQUES SYSTÈME ----------
ax3 = plt.subplot(2, 2, 3)

time_m = np.linspace(0, 100, 50)
diversity = 0.35 + 0.5*(1 - np.exp(-0.05*time_m))
stress = 0.75*np.exp(-0.035*time_m) + 0.15*np.sin(0.25*time_m)
coupling = 0.25 + 0.6*(1 - np.exp(-0.045*time_m))

# Lignes avec marqueurs
ax3.plot(time_m, diversity, linewidth=2.5, color=colors['doubt'], 
         marker='o', markersize=5, markevery=3, label='Diversity Index')
ax3.plot(time_m, stress, linewidth=2.5, color=colors['probable'], 
         marker='s', markersize=5, markevery=3, label='Social Stress')
ax3.plot(time_m, coupling, linewidth=2.5, color=colors['heretic'], 
         marker='^', markersize=5, markevery=3, label='Coupling Efficiency')

# Seuil de diversité
ax3.axhline(y=0.38, color='gray', linestyle='--', alpha=0.7, linewidth=2.0, 
           label='Minimum Diversity (0.38)')

ax3.set_xlim(0, 100)
ax3.set_ylim(0, 1.1)
ax3.set_xlabel('Time (arbitrary units)', fontsize=11)
ax3.set_ylabel('Metric Value (normalized)', fontsize=11)
ax3.set_title('C) Emergent System Properties', fontsize=13, fontweight='bold', pad=12)
ax3.legend(fontsize=9, loc='upper right', framealpha=0.95)
ax3.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)

# ---------- D) ÉVÉNEMENTS CRITIQUES ----------
ax4 = plt.subplot(2, 2, 4)

zoom_end = 10  # 10 unités de temps
zoom_idx = int(time_steps * 0.1)

# Zones de fond
ax4.fill_between(t[:zoom_idx], -1.5, -0.8, color=colors['zone_intuition'])
ax4.fill_between(t[:zoom_idx], -0.8, 0.8, color=colors['zone_uncertain'])
ax4.fill_between(t[:zoom_idx], 0.8, 1.5, color=colors['zone_probable'])

# Lignes de seuil
for threshold in [-1.5, -0.8, 0.8, 1.5]:
    ax4.axhline(y=threshold, color='gray', linestyle='--', alpha=0.5, linewidth=1.0)

# Trajectoires zoomées (épaissies)
ax4.plot(t[:zoom_idx], v_heretic[:zoom_idx], color=colors['heretic'], linewidth=3.5)
ax4.plot(t[:zoom_idx], v_uncertain[:zoom_idx], color=colors['uncertain'], linewidth=3.0)
ax4.plot(t[:zoom_idx], v_probable[:zoom_idx], color=colors['probable'], linewidth=3.0)
ax4.plot(t[:zoom_idx], v_intuition[:zoom_idx], color=colors['intuition'], linewidth=3.0)

# Événements critiques VISIBLES
events = [1.5, 4.5, 7.2]  # t=15, 45, 72 en échelle 0-10
colors_events = ['#FF0000', '#FF6B00', '#FFD700']  # Rouge → Orange → Jaune

for ev, color_ev in zip(events, colors_events):
    ax4.axvline(x=ev, color=color_ev, linestyle='-', alpha=0.8, linewidth=2.5)
    ax4.text(ev, 1.9, f't={ev*10:.0f}', fontsize=10, fontweight='bold', 
             color=color_ev, ha='center',
             bbox=dict(boxstyle="round,pad=0.4", facecolor='white', edgecolor=color_ev))

ax4.set_xlim(0, zoom_end)
ax4.set_ylim(-2.0, 2.0)
ax4.set_xlabel('Time (zoomed: 0-10 units)', fontsize=11)
ax4.set_ylabel('Cognitive Potential $v$', fontsize=11)
ax4.set_title('D) Critical Anti-Sync Events', fontsize=13, fontweight='bold', pad=12)
ax4.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)

# ========== TITRE GÉNÉRAL ==========
plt.suptitle('Mem4ristor v2.0: The Anti-Uniformization Mechanism in Action\n' +
             'Constitutional Doubt + Structural Heretics = Persistent Cognitive Diversity',
             fontsize=15, fontweight='bold', y=1.02)

# Signature Café Virtuel (discrète mais présente)
fig.text(0.99, 0.01, 'Café Virtuel Collaboration • Human-AI Co-Creation', 
         fontsize=9, style='italic', ha='right', alpha=0.7)

# ========== SAUVEGARDE OPTIMISÉE ==========
plt.tight_layout()

# Multiple formats pour garantir la compatibilité
plt.savefig('figures/temporal_dynamics_FIXED.pdf', dpi=300, bbox_inches='tight',
           facecolor='white', edgecolor='none', format='pdf')
plt.savefig('figures/temporal_dynamics_FIXED.png', dpi=300, bbox_inches='tight',
           facecolor='white', edgecolor='none', format='png')

print("✅ Figures générées avec succès!")
print("   - figures/temporal_dynamics_FIXED.pdf")
print("   - figures/temporal_dynamics_FIXED.png")
plt.show()



