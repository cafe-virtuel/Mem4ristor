"""
Verification numerique du theoreme de Poincare-Bendixson pour le noeud FHN isole.

Montre cote a cote les deux regimes :
  - alpha = 0.15 (default) : spirale stable -> point fixe (regime excitable, sous-Hopf)
  - alpha = 0.35 (> alpha_crit ~= 0.296) : cycle limite -> P-B s'applique

Noeud isole = D=0, sigma_v=0, I_stimulus=0, u quasi-statique (u=0.5 fixe).
3000 pas, dt=0.05.

Resultat attendu :
  alpha=0.15 : std(v) < 1e-3 apres convergence (spirale stable)
  alpha=0.35 : std(v) > 0.5  apres convergence (oscillation soutenue)
"""

import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

STEPS   = 3000
WARMUP  = 2000   # on mesure sur les 1000 derniers pas
DT      = 0.05
A       = 0.7
B       = 0.8
EPS     = 0.08
V_DIV   = 5.0

def simulate_isolated(alpha, v0=0.5, w0=0.0, steps=STEPS, dt=DT):
    """Euler sur le noeud FHN isole sans couplage ni bruit."""
    v, w = v0, w0
    v_hist = np.zeros(steps)
    w_hist = np.zeros(steps)
    for t in range(steps):
        dv = v - v**3 / V_DIV - w - alpha * np.tanh(v)
        dw = EPS * (v + A - B * w)
        v += dt * dv
        w += dt * dw
        v_hist[t] = v
        w_hist[t] = w
    return v_hist, w_hist

def report(alpha, v_hist, w_hist):
    window_v = v_hist[WARMUP:]
    std_v    = np.std(window_v)
    mean_v   = np.mean(window_v)
    regime   = "CYCLE LIMITE (P-B valide)" if std_v > 0.5 else "SPIRALE STABLE (regime excitable)"
    print(f"  alpha={alpha:.2f} | mean(v)={mean_v:+.4f} | std(v)={std_v:.4f} | -> {regime}")
    return std_v

print("=" * 60)
print("  VERIFICATION Poincare-Bendixson — noeud FHN isole")
print(f"  {STEPS} pas, dt={DT}, D=0, sigma=0, I=0")
print("=" * 60)

alphas = [0.15, 0.35]
results = {}
for alpha in alphas:
    v_hist, w_hist = simulate_isolated(alpha)
    results[alpha] = (v_hist, w_hist)
    std_v = report(alpha, v_hist, w_hist)

print()
print("Valeurs attendues :")
print("  alpha=0.15 : std(v) << 0.1  (spirale stable, pas de cycle limite)")
print("  alpha=0.35 : std(v)  > 0.5  (cycle limite, P-B s'applique)")
print("=" * 60)

# --- Figure ---
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle("Verification Poincare-Bendixson — Noeud FHN isole ($D=0$, $\\sigma=0$)", fontsize=13)

labels = {0.15: r"$\alpha=0.15$ (sous-Hopf, regime excitable)", 0.35: r"$\alpha=0.35$ (> $\alpha_{crit}$, cycle limite)"}
colors = {0.15: "#2E86C1", 0.35: "#C0392B"}

for row, alpha in enumerate(alphas):
    v_hist, w_hist = results[alpha]
    ax_t  = axes[row, 0]
    ax_ph = axes[row, 1]

    t_axis = np.arange(STEPS) * DT
    ax_t.plot(t_axis, v_hist, color=colors[alpha], linewidth=0.8)
    ax_t.set_title(labels[alpha])
    ax_t.set_xlabel("Temps")
    ax_t.set_ylabel("$v(t)$")
    ax_t.axvline(WARMUP * DT, color="gray", linestyle="--", linewidth=0.7, label="debut fenetre mesure")
    ax_t.legend(fontsize=7)
    ax_t.grid(True, alpha=0.3)

    ax_ph.plot(v_hist[WARMUP:], w_hist[WARMUP:], color=colors[alpha], linewidth=0.6, alpha=0.8)
    ax_ph.set_xlabel("$v$")
    ax_ph.set_ylabel("$w$")
    ax_ph.set_title("Espace des phases (fenetre post-convergence)")
    ax_ph.grid(True, alpha=0.3)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "verify_pb_isolated_node.png")
plt.savefig(out, dpi=150)
print(f"\nFigure sauvegardee : {out}")
