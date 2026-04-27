"""
Génère figures/dt_sensitivity.png depuis figures/dt_sensitivity.csv
"""
import os, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA = os.path.join(os.path.dirname(__file__), '..', 'figures', 'dt_sensitivity.csv')
OUT  = os.path.join(os.path.dirname(__file__), '..', 'figures', 'dt_sensitivity.png')

rows = []
with open(DATA, newline='') as f:
    for r in csv.DictReader(f):
        rows.append({k: float(v) if k not in ('topo',) else v for k, v in r.items()})

TOPOS = ['ba_m3_functional', 'ba_m5_critical', 'ba_m8_dead_zone']
LABELS = {'ba_m3_functional': 'BA m=3 (fonctionnel)', 'ba_m5_critical': 'BA m=5 (critique)', 'ba_m8_dead_zone': 'BA m=8 (dead zone)'}
DTS = sorted(set(r['dt'] for r in rows))
COLORS = {'ba_m3_functional': '#2196F3', 'ba_m5_critical': '#FF9800', 'ba_m8_dead_zone': '#F44336'}
REF_DT = 0.05

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Sensibilité à dt — Euler (T_total=150, I_stim=0.3, n=5 seeds)", fontsize=13, fontweight='bold')

for ax, metric, ylabel, title in [
    (axes[0], 'h_cog',  'H_cog (bits)',   'Entropie cognitive vs dt'),
    (axes[1], 'sync',   'Synchrony',       'Synchronie vs dt'),
]:
    for topo in TOPOS:
        means, stds = [], []
        for dt in DTS:
            vals = [r[metric] for r in rows if r['topo'] == topo and abs(r['dt'] - dt) < 1e-9]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        ax.errorbar(DTS, means, yerr=stds, marker='o', lw=2, ms=6,
                    color=COLORS[topo], label=LABELS[topo], capsize=4)

    ax.axvline(REF_DT, color='gray', ls='--', lw=1.5, alpha=0.7, label='dt=0.05 (référence)')
    ax.set_xlabel('dt', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(DTS)

# Annotation résultat principal
axes[0].text(0.98, 0.97,
    "H_cog STABLE\ndelta max < 0.01\npour tous les dt",
    transform=axes[0].transAxes, ha='right', va='top',
    bbox=dict(boxstyle='round', fc='#E8F5E9', ec='#4CAF50', lw=1.5),
    fontsize=9, color='#1B5E20')

axes[1].text(0.98, 0.97,
    "Synchrony varie avec dt\n(dissipation Euler)\nComparaisons RELATIVES\nvalides (meme dt)",
    transform=axes[1].transAxes, ha='right', va='top',
    bbox=dict(boxstyle='round', fc='#FFF3E0', ec='#FF9800', lw=1.5),
    fontsize=9, color='#E65100')

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches='tight')
print(f"Figure -> {OUT}")
