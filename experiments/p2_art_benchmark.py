"""
V5 — Autorégulation Topologique (ART) : Benchmark recovery_time (choc soutenu)

Compare 5 configurations face a un choc soutenu :
  - Impulse a t=SHOCK_STEP  : u[:] = 0.0 (tous les noeuds rigidifient)
  - Fenetre de choc         : I_stim = 0.0 pendant SHOCK_DURATION steps
    (sans driving, u decroit naturellement — ART est le seul mecanisme actif)
  - Apres SHOCK_END         : I_stim = 0.5 reprend, on mesure la recovery

  1. V4 pur          : pas de protection
  2. Plancher seul   : u_min=0.05, pas de retroaction topologique
  3. ART douce       : mode soft (Gemini +15%), sans plancher
  4. ART agressive   : mode hard (Grok +25%), sans plancher
  5. ART douce + plancher : soft + u_min=0.05

Metrique cle : recovery_time = nb de steps (depuis SHOCK_END) pour que
               H revienne a >= H_prechoc * 0.9
Metriques secondaires : H_min_shock (creux pendant la fenetre), H_final

Insight Cafe Virtuel #1 (GLM, Boucle 5) :
  ART = loi de Kirchhoff sur crossbar passif — retroaction emerge du cablage.
  Le plancher seul ne suffit pas sous choc synchronise (DeepSeek) :
  aucun noeud actif ne peut generer la pression de retour.
  Avec choc soutenu (I_stim=0), ART est le seul mecanisme capable de maintenir
  u > 0 : les noeuds rigides se "taxent" mutuellement via la topologie.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import numpy as np
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mem4ristor.core import Mem4Network
from mem4ristor.metrics import calculate_continuous_entropy
from mem4ristor.graph_utils import make_ba

STEPS          = 6000
SHOCK_STEP     = 2000
SHOCK_DURATION = 300                       # steps avec I_stim=0 (choc soutenu)
SHOCK_END      = SHOCK_STEP + SHOCK_DURATION  # = 2300
WARMUP         = int(SHOCK_STEP * 0.75)    # 1500 steps pour mesurer H_prechoc
SEEDS          = [42, 123, 777, 1234, 5678, 9999, 314, 2718, 8765, 4321]
N              = 100
M_BA           = 3
I_STIM         = 0.5
RECOVERY_THRESHOLD = 0.90  # H_post >= H_prechoc * 0.90

ART_CFG_BASE = {
    'rigid_threshold': 0.7,
    'alpha_art_soft':  0.15,
    'alpha_art_hard':  0.25,
}

CONDITIONS = [
    {
        "label":  "V4 pur",
        "art_enabled": False,
        "mode":    "soft",
        "u_min":   0.0,
    },
    {
        "label":  "Plancher seul (u_min=0.05)",
        "art_enabled": False,
        "mode":    "soft",
        "u_min":   0.05,
    },
    {
        "label":  "ART douce / Gemini (soft)",
        "art_enabled": True,
        "mode":    "soft",
        "u_min":   0.0,
    },
    {
        "label":  "ART agressive / Grok (hard)",
        "art_enabled": True,
        "mode":    "hard",
        "u_min":   0.0,
    },
    {
        "label":  "ART douce + plancher",
        "art_enabled": True,
        "mode":    "soft",
        "u_min":   0.05,
    },
]


def run_condition(cond, seed):
    adj = make_ba(n=N, m=M_BA, seed=seed)
    net = Mem4Network(adjacency_matrix=adj, heretic_ratio=0.15, seed=seed)

    # Configurer le plancher dans u_clamp si plancher seul (art disabled)
    if not cond['art_enabled'] and cond['u_min'] > 0:
        net.model.cfg['doubt']['u_clamp'] = [cond['u_min'], 1.0]

    # Configurer ART
    net.model.cfg['topological_regulation'] = {
        'enabled':         cond['art_enabled'],
        'u_min':           cond['u_min'] if cond['art_enabled'] else 0.0,
        'mode':            cond['mode'],
        **ART_CFG_BASE,
    }

    H_trace    = []
    H_prechoc  = None
    recovery_t = None

    for t in range(STEPS):
        # Impulse initiale : tous les noeuds rigidifient
        if t == SHOCK_STEP:
            net.model.u[:] = 0.0

        # Choc soutenu : I_stim=0 pendant SHOCK_DURATION steps
        # Sans driving externe, u decroit naturellement (faible v^2)
        # ART est le seul mecanisme actif capable de maintenir u > 0
        current_stim = 0.0 if SHOCK_STEP <= t < SHOCK_END else I_STIM

        net.step(I_stimulus=current_stim)
        H = calculate_continuous_entropy(net.model.v)
        H_trace.append(H)

        # Mesurer H_prechoc (moyenne des 25% derniers avant le choc)
        if t == SHOCK_STEP - 1:
            H_prechoc = float(np.mean(H_trace[WARMUP:SHOCK_STEP]))

        # Recovery detectable seulement depuis la fin du choc soutenu
        if H_prechoc is not None and t >= SHOCK_END and recovery_t is None:
            if H >= H_prechoc * RECOVERY_THRESHOLD:
                recovery_t = t - SHOCK_END

    H_shock = H_trace[SHOCK_STEP:SHOCK_END]   # pendant la fenetre de choc
    H_post  = H_trace[SHOCK_END:]             # apres relachement
    return {
        "H_prechoc":      H_prechoc,
        "H_min_shock":    float(np.min(H_shock)) if H_shock else float('nan'),
        "H_min_post":     float(np.min(H_post))  if H_post  else float('nan'),
        "H_final":        float(H_trace[-1]),
        "recovery_time":  recovery_t,
        "H_trace":        H_trace,
    }


print("=" * 80)
print("  V5 — ART BENCHMARK : choc soutenu (impulse u=0 + I_stim=0 x SHOCK_DURATION)")
print(f"  BA m={M_BA} N={N} | {STEPS} steps | {len(SEEDS)} seeds | I_stim={I_STIM}")
print(f"  SHOCK_STEP={SHOCK_STEP}  SHOCK_DURATION={SHOCK_DURATION}  SHOCK_END={SHOCK_END}")
print(f"  Seuil recovery (depuis SHOCK_END) : H >= H_prechoc x {RECOVERY_THRESHOLD}")
print("=" * 80)
print(f"\n  {'Condition':<40}  {'H_pre':>6}  {'H_min_s':>7}  {'H_min_p':>7}  {'H_fin':>6}  {'Rec.':>6}")
print(f"  {'-'*76}")

results = []
all_traces = {}

for cond in CONDITIONS:
    runs = [run_condition(cond, seed) for seed in SEEDS]

    H_pre_vals   = [r['H_prechoc']     for r in runs]
    H_mshk_vals  = [r['H_min_shock']   for r in runs]
    H_mpos_vals  = [r['H_min_post']    for r in runs]
    H_fin_vals   = [r['H_final']       for r in runs]
    rec_vals     = [r['recovery_time'] for r in runs]

    n_recovered = sum(1 for r in rec_vals if r is not None)
    rec_finite  = [r for r in rec_vals if r is not None]
    rec_mean    = float(np.mean(rec_finite)) if rec_finite else float('inf')
    rec_str     = f"{rec_mean:.0f}" if rec_finite else "NONE"

    r = {
        "label":          cond["label"],
        "H_prechoc":      float(np.mean(H_pre_vals)),
        "H_min_shock":    float(np.mean(H_mshk_vals)),
        "H_min_post":     float(np.mean(H_mpos_vals)),
        "H_final":        float(np.mean(H_fin_vals)),
        "recovery_time":  rec_mean,
        "n_recovered":    n_recovered,
        "traces":         [run['H_trace'] for run in runs],
    }
    results.append(r)
    all_traces[cond["label"]] = r["traces"]

    print(f"  {r['label']:<40}  {r['H_prechoc']:>6.2f}  {r['H_min_shock']:>7.2f}  "
          f"{r['H_min_post']:>7.2f}  {r['H_final']:>6.2f}  {rec_str:>6}  "
          f"({n_recovered}/{len(SEEDS)} seeds)")

# Verdict
print(f"\n{'=' * 76}")
print("  VERDICT")
print(f"{'=' * 76}\n")

v4_rec = results[0]['recovery_time']
v4_rec_str = 'NONE' if np.isinf(v4_rec) else f"{v4_rec:.0f}"
print(f"  V4 pur    : recovery_time = {v4_rec_str} steps (depuis SHOCK_END={SHOCK_END})")
for r in results[1:]:
    rec_str  = "NONE" if np.isinf(r['recovery_time']) else f"{r['recovery_time']:.0f}"
    delta_H  = r['H_final']     - results[0]['H_final']
    delta_ms = r['H_min_shock'] - results[0]['H_min_shock']
    print(f"  {r['label']:<40} : recovery={rec_str:>5} steps | "
          f"H_min_shock {delta_ms:+.2f} | H_final {delta_H:+.2f}")

# Figure : H vs t en moyenne sur les seeds (sous-plot par condition)
fig, axes = plt.subplots(len(CONDITIONS), 1, figsize=(12, 2.5 * len(CONDITIONS)),
                         sharex=True, sharey=True)
colors = ['#999999', '#4444cc', '#22aa44', '#cc4422', '#aa22cc']

for ax, cond_result, color in zip(axes, results, colors):
    traces = np.array(cond_result['traces'])          # (n_seeds, STEPS)
    mean_H = traces.mean(axis=0)
    std_H  = traces.std(axis=0)
    t      = np.arange(STEPS)

    ax.fill_between(t, mean_H - std_H, mean_H + std_H, alpha=0.2, color=color)
    ax.plot(t, mean_H, color=color, lw=1.5, label=cond_result['label'])

    # Fenetre de choc en rouge pale
    ax.axvspan(SHOCK_STEP, SHOCK_END, color='red', alpha=0.08, label=f'choc soutenu ({SHOCK_DURATION} steps)')
    ax.axvline(SHOCK_STEP, color='red',    lw=1.2, ls='--', alpha=0.6)
    ax.axvline(SHOCK_END,  color='orange', lw=1.0, ls='--', alpha=0.7, label='fin choc')

    rec = cond_result['recovery_time']
    if not np.isinf(rec):
        ax.axvline(SHOCK_END + rec, color='green', lw=1.0, ls=':', alpha=0.8,
                   label=f'recovery +{rec:.0f} steps')

    ax.set_ylabel('H (bits)', fontsize=9)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('Step', fontsize=10)
fig.suptitle(
    f'ART Benchmark — Choc soutenu : impulse u=0 a t={SHOCK_STEP} + I_stim=0 x {SHOCK_DURATION} steps',
    fontsize=10
)
plt.tight_layout()

out_dir = os.path.join(os.path.dirname(__file__), '../figures')
os.makedirs(out_dir, exist_ok=True)
fig_path = os.path.join(out_dir, 'p2_art_benchmark.png')
plt.savefig(fig_path, dpi=100, bbox_inches='tight')
print(f"\n  Figure -> {fig_path}")

# CSV
csv_path = os.path.join(out_dir, 'p2_art_benchmark.csv')
with open(csv_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['label', 'H_prechoc', 'H_min_shock', 'H_min_post', 'H_final',
                'recovery_time', 'n_recovered', 'shock_duration'])
    for r in results:
        rec = r['recovery_time']
        w.writerow([r['label'],
                    round(r['H_prechoc'],    4),
                    round(r['H_min_shock'],  4),
                    round(r['H_min_post'],   4),
                    round(r['H_final'],      4),
                    '' if np.isinf(rec) else round(rec, 1),
                    r['n_recovered'],
                    SHOCK_DURATION])
print(f"  CSV    -> {csv_path}")

print(f"\n{'=' * 80}")
print("  Pour reproduire : python experiments/p2_art_benchmark.py")
print(f"  Parametres : SHOCK_STEP={SHOCK_STEP}  SHOCK_DURATION={SHOCK_DURATION}  STEPS={STEPS}")
print(f"{'=' * 80}\n")
