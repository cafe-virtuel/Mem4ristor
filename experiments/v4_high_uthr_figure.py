"""
V4 High u_thr — Figure de synthèse
Génère figures/v4_high_uthr_sweep.png depuis figures/v4_high_uthr_sweep.csv
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'figures', 'v4_high_uthr_sweep.csv')
OUT_PATH  = os.path.join(os.path.dirname(__file__), '..', 'figures', 'v4_high_uthr_sweep.png')

# Charger CSV
rows = []
with open(DATA_PATH, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append({k: float(v) for k, v in row.items()})

U_THRESHOLDS   = sorted(set(r['u_threshold'] for r in rows))
STEPS_REQUIRED = sorted(set(r['steps_required'] for r in rows))
COLORS = {10: '#2196F3', 50: '#FF9800', 200: '#9C27B0'}
V3_BASELINE_H = 3.41

def get(u_thr, steps_req, key):
    return next(r[key] for r in rows if r['u_threshold'] == u_thr and r['steps_required'] == steps_req)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("V4 Dynamic Heretics — Régime haute zone (u_thr > 0.9)", fontsize=14, fontweight='bold')

# Panel 1 — Cascade fraction vs u_thr
ax = axes[0]
for s in STEPS_REQUIRED:
    pcts = [get(u, s, 'heretic_pct_final_mean') for u in U_THRESHOLDS]
    stds = [get(u, s, 'heretic_pct_final_std') for u in U_THRESHOLDS]
    ax.plot(U_THRESHOLDS, pcts, 'o-', color=COLORS[s], label=f'steps={s}', lw=2, ms=5)
    ax.fill_between(U_THRESHOLDS,
                    [p - e for p, e in zip(pcts, stds)],
                    [p + e for p, e in zip(pcts, stds)],
                    color=COLORS[s], alpha=0.15)
ax.axhline(0.15, color='gray', ls=':', lw=1.5, label='Baseline (15% statiques)')
ax.axvspan(1.00, 1.05, color='red', alpha=0.12, label='Zone critique (1.00-1.05)')
ax.axvline(1.0, color='red', ls='--', lw=1.5, alpha=0.7)
ax.set_xlabel('u_threshold', fontsize=11)
ax.set_ylabel('Fraction herétique finale', fontsize=11)
ax.set_title('Cascade vs seuil', fontsize=12)
ax.legend(fontsize=9)
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)

# Panel 2 — H_final vs u_thr
ax = axes[1]
for s in STEPS_REQUIRED:
    hs = [get(u, s, 'entropy_final_mean') for u in U_THRESHOLDS]
    ax.plot(U_THRESHOLDS, hs, 'o-', color=COLORS[s], label=f'steps={s}', lw=2, ms=5)
ax.axhline(V3_BASELINE_H, color='black', ls='--', lw=1.5, label=f'Baseline V3 (~{V3_BASELINE_H})')
ax.axvspan(1.00, 1.05, color='red', alpha=0.12, label='Zone critique')
ax.axvline(1.0, color='red', ls='--', lw=1.5, alpha=0.7)
ax.set_xlabel('u_threshold', fontsize=11)
ax.set_ylabel('H cognitif final (bits)', fontsize=11)
ax.set_title('Entropie vs seuil', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 3 — Diagnostic u_max (steps=50) + annotation critique
ax = axes[2]
u_maxes = [get(u, 50, 'u_max_observed_mean') for u in U_THRESHOLDS]
pcts_50 = [get(u, 50, 'heretic_pct_final_mean') for u in U_THRESHOLDS]

ax2 = ax.twinx()
ax.plot(U_THRESHOLDS, pcts_50, 'o-', color='#FF9800', lw=2, ms=5, label='heretic% (steps=50)')
ax2.plot(U_THRESHOLDS, u_maxes, 's--', color='#607D8B', lw=1.5, ms=5, label='u_max observé')
ax2.axhline(1.0, color='#607D8B', ls=':', lw=1, alpha=0.7)
ax2.set_ylim(0.8, 1.15)
ax2.set_ylabel('u_max observé', fontsize=11, color='#607D8B')
ax2.tick_params(axis='y', labelcolor='#607D8B')

ax.axvline(1.0, color='red', ls='--', lw=1.5, alpha=0.7)
ax.axvspan(1.00, 1.05, color='red', alpha=0.12)
ax.annotate('Plafond\nu = 1.0', xy=(1.0, 0.5), xytext=(1.08, 0.6),
            fontsize=9, color='red', arrowprops=dict(arrowstyle='->', color='red'))

ax.set_xlabel('u_threshold', fontsize=11)
ax.set_ylabel('Fraction herétique finale', fontsize=11)
ax.set_title('Diagnostic : plafond u = 1.0', fontsize=12)
ax.grid(True, alpha=0.3)

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
print(f"Figure -> {OUT_PATH}")
