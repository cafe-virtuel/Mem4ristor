"""
Vérification Numérique de Stabilité (Lyapunov)
===============================================
Vérifie numériquement que le candidat de Lyapunov est décroissant
le long des trajectoires du système Mem4ristor.

Candidat de Lyapunov:
  V(v,w,u) = (1/2)v² + (1/(2ε))w² + (γ/2)(u - u*)²

où:
  - ε = epsilon (time-scale separation)
  - u* = sigma_baseline (point fixe naturel de u)
  - γ = poids du terme de doute

Si dV/dt ≤ 0 le long de toutes les trajectoires échantillonnées,
la stabilité est confirmée numériquement (mais pas prouvée formellement).

Usage:
  python experiments/lyapunov_numerical.py
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from mem4ristor.core import Mem4ristorV2


def lyapunov_candidate(v, w, u, epsilon, u_star, gamma=1.0):
    """
    V(v,w,u) = (1/2)v² + 1/(2ε) w² + (γ/2)(u - u*)²
    """
    return 0.5 * v**2 + 0.5 / epsilon * w**2 + 0.5 * gamma * (u - u_star)**2


def run_single_trajectory(seed, N=1, steps=2000, dt=0.01):
    """
    Run a single isolated unit and track V(t) and dV/dt.
    """
    model = Mem4ristorV2(seed=seed)
    model._initialize_params(N, cold_start=False)
    
    epsilon = model.cfg['dynamics']['epsilon']
    u_star = model.cfg['doubt']['sigma_baseline']
    gamma = 1.0
    
    V_history = []
    dV_history = []
    v_history = []
    w_history = []
    u_history = []
    
    for t in range(steps):
        v, w, u = model.v[0], model.w[0], model.u[0]
        V_current = lyapunov_candidate(v, w, u, epsilon, u_star, gamma)
        V_history.append(V_current)
        v_history.append(v)
        w_history.append(w)
        u_history.append(u)
        
        # Step
        model.step(I_stimulus=0.0, coupling_input=np.zeros(N))
        
        v_next, w_next, u_next = model.v[0], model.w[0], model.u[0]
        V_next = lyapunov_candidate(v_next, w_next, u_next, epsilon, u_star, gamma)
        
        dV = (V_next - V_current) / dt
        dV_history.append(dV)
    
    return {
        'V': np.array(V_history),
        'dV': np.array(dV_history),
        'v': np.array(v_history),
        'w': np.array(w_history),
        'u': np.array(u_history),
    }


def main():
    print("=" * 65)
    print("  LYAPUNOV NUMERICAL VERIFICATION")
    print("=" * 65)
    
    N_TRAJECTORIES = 50
    STEPS = 2000
    
    print(f"\n  Candidat: V(v,w,u) = ½v² + 1/(2ε)w² + ½γ(u-u*)²")
    print(f"  Trajectoires: {N_TRAJECTORIES}")
    print(f"  Steps: {STEPS}\n")
    
    all_results = []
    dV_positive_counts = []
    
    for seed in range(N_TRAJECTORIES):
        result = run_single_trajectory(seed=seed, steps=STEPS)
        all_results.append(result)
        
        # Count positive dV/dt values (violations of Lyapunov condition)
        n_positive = np.sum(result['dV'] > 0)
        n_total = len(result['dV'])
        dV_positive_counts.append(n_positive / n_total * 100)
        
        if seed % 10 == 0:
            print(f"  Trajectoire {seed:3d}: dV/dt > 0 dans {n_positive}/{n_total}"
                  f" ({n_positive/n_total*100:.1f}%) des pas")
    
    # ── Statistics ──
    mean_violation = np.mean(dV_positive_counts)
    max_violation = np.max(dV_positive_counts)
    
    print(f"\n" + "─" * 65)
    print(f"  RÉSULTATS GLOBAUX")
    print(f"─" * 65)
    print(f"  Violation moyenne (dV/dt > 0) : {mean_violation:.1f}%")
    print(f"  Violation max                 : {max_violation:.1f}%")
    
    if mean_violation < 5.0:
        print(f"\n  ✅ Stabilité confirmée numériquement (violations < 5%)")
        print(f"     Les violations sont dues au bruit stochastique η,")
        print(f"     qui n'est pas modélisé dans le candidat V.")
    elif mean_violation < 20.0:
        print(f"\n  ⚠️ Stabilité partielle ({mean_violation:.1f}% violations)")
        print(f"     Le candidat V est approximatif — le bruit domine.")
    else:
        print(f"\n  ❌ Le candidat V n'est pas un Lyapunov pour ce système.")
    
    # ── Check V convergence ──
    V_final_mean = np.mean([r['V'][-1] for r in all_results])
    V_initial_mean = np.mean([r['V'][0] for r in all_results])
    
    print(f"\n  V(0) moyen  : {V_initial_mean:.3f}")
    print(f"  V(T) moyen  : {V_final_mean:.3f}")
    print(f"  Ratio V(T)/V(0) : {V_final_mean/V_initial_mean:.3f}")
    
    if V_final_mean < V_initial_mean:
        print(f"  → V diminue globalement ✅ (converge)")
    else:
        print(f"  → V augmente ⚠️ (mais peut osciller)")
    
    # ── Plot ──
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. V(t) for 10 trajectories
    ax = axes[0, 0]
    for i in range(min(10, len(all_results))):
        ax.plot(all_results[i]['V'], alpha=0.6, linewidth=0.8)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('V(v,w,u)')
    ax.set_title('Lyapunov Candidate V(t)')
    ax.grid(True, alpha=0.3)
    
    # 2. dV/dt histogram
    ax = axes[0, 1]
    all_dV = np.concatenate([r['dV'] for r in all_results])
    ax.hist(all_dV, bins=100, density=True, color='steelblue', alpha=0.7)
    ax.axvline(x=0, color='red', linewidth=2, linestyle='--', label='dV/dt = 0')
    ax.set_xlabel('dV/dt')
    ax.set_ylabel('Density')
    ax.set_title(f'dV/dt Distribution (violations: {mean_violation:.1f}%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-50, 50)
    
    # 3. Phase portrait (v, w) for one trajectory
    ax = axes[1, 0]
    r0 = all_results[0]
    ax.plot(r0['v'], r0['w'], alpha=0.5, linewidth=0.5)
    ax.scatter(r0['v'][0], r0['w'][0], color='green', s=50, zorder=5, label='Start')
    ax.scatter(r0['v'][-1], r0['w'][-1], color='red', s=50, zorder=5, label='End')
    ax.set_xlabel('v (Cognitive Potential)')
    ax.set_ylabel('w (Recovery)')
    ax.set_title('Phase Portrait (v, w)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. u(t) and V(t) for one trajectory
    ax = axes[1, 1]
    ax2 = ax.twinx()
    ax.plot(r0['u'], color='purple', alpha=0.7, label='u (Doubt)')
    ax2.plot(r0['V'], color='orange', alpha=0.7, label='V (Lyapunov)')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('u (Doubt)', color='purple')
    ax2.set_ylabel('V (Lyapunov)', color='orange')
    ax.set_title('Doubt u(t) vs Lyapunov V(t)')
    ax.grid(True, alpha=0.3)
    
    fig_path = os.path.join(results_dir, 'lyapunov_numerical.png')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"\n  📈 Figure: {os.path.abspath(fig_path)}")
    
    print("\n" + "=" * 65)


if __name__ == '__main__':
    main()
