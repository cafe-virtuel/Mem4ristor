#!/usr/bin/env python3
"""
Faille A (Audit Manus 2026-04-25) -- Calibration eta_SPICE <-> sigma_Python

Question : eta=0.5 SPICE correspond a quel sigma Python ?
Si les amplitudes sont equivalentes, le claim "bruit thermique qualitativement
distinct" est fort. Si sigma_equiv >> 1.2 (au-dessus du sweep Item 10), le
claim est fragile (Item 10 etait juste sous-dose).

Methode :
  1. Simuler une cellule RC unitaire SPICE avec trnoise(eta) pendant T_ref.
     Mesurer la variance du bruit injecte sur V(v0).
  2. Calculer sigma_equiv = sqrt(variance / dt) -- bruit discret Python equivalent.
  3. Relancer la simulation dead-zone BA m=5 (Item 10 protocole) avec
     sigma_v = sigma_equiv et sigma_v = 2*sigma_equiv.
  4. Comparer H_cog a ces amplitudes vs H_cog SPICE a eta=0.5.

Script  : experiments/spice_noise_calibration.py
CSV     : figures/spice_noise_calibration.csv
Figures : figures/spice_noise_calibration.png

Reference : PROJECT_STATUS.md P2-AUDIT-2 Faille A + §3trigies
"""
import sys, os, time, subprocess
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
NGSPICE = Path("D:/ANTIGRAVITY/ngspice-46_64/Spice64/bin/ngspice_con.exe")
RESULTS = ROOT / 'experiments' / 'spice' / 'results'
RESULTS.mkdir(parents=True, exist_ok=True)

from mem4ristor.core import Mem4Network
from mem4ristor.graph_utils import make_ba
from mem4ristor.metrics import calculate_cognitive_entropy, calculate_continuous_entropy

# -- Parametres ---------------------------------------------------------------
ETAS_SPICE  = [0.1, 0.3, 0.5, 0.8]   # amplitudes testees en SPICE
T_REF       = 100.0                    # duree simulation de calibration (s)
DT          = 0.05                     # pas Python
N_BA        = 100
M_BA        = 5
SEEDS       = [42, 123, 777]
STEPS       = 3000
WARM_UP     = 750
I_STIM      = 0.5


# -- Helpers ------------------------------------------------------------------



def run_ngspice(netlist_path: Path) -> bool:
    if not NGSPICE.exists():
        print(f"[SKIP] ngspice not found at {NGSPICE}")
        return False
    res = subprocess.run(
        [str(NGSPICE), '-b', str(netlist_path)],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    if res.returncode != 0:
        print("STDOUT:", res.stdout[-500:])
        print("STDERR:", res.stderr[-300:])
        return False
    return True


def parse_wrdata(path: Path) -> np.ndarray:
    """Load wrdata output, return voltage column(s) only."""
    if not path.exists():
        return np.array([])
    raw = np.loadtxt(path)
    if raw.ndim == 1:
        raw = raw[None, :]
    # wrdata format: [time, v(node0), time, v(node0), ...] interleaved
    # For a single node: columns [time, v]
    return raw[:, 1]   # voltage trace


# -- PARTIE 1 : Calibration SPICE ---------------------------------------------
# Cellule RC simple avec trnoise : mesure la variance effective du bruit injecte

def make_rc_noise_netlist(eta: float, t_end: float, dt_save: float, tag: str) -> Path:
    """Netlist RC (tau=1s) avec trnoise(eta) sur la source de courant.
    V(cap) integre le bruit -> variance(V) = sigma^2 du bruit discret equivalent.
    """
    path = RESULTS / f'rc_noise_{tag}.cir'
    out  = RESULTS / f'rc_noise_{tag}.dat'
    nl   = f"""* RC noise calibration -- eta={eta}
.title RC noise eta={eta}
.param eta={eta}
C_cap cap 0 1 IC=0.0
* trnoise : white noise current source (A/sqrt(Hz)), spectral density = eta^2
I_noise 0 cap trnoise({eta} {dt_save} 0 0)
.save v(cap)
.options method=trap reltol=1e-4
.tran {dt_save} {t_end} 0 {dt_save} uic
.control
run
wrdata {out.as_posix()} v(cap)
quit
.endc
.end
"""
    path.write_text(nl, encoding='utf-8')
    return path


def calibrate_spice_noise():
    """Pour chaque eta, simuler et mesurer sigma_equiv = std(diff(V)) / sqrt(dt)."""
    print("\n" + "=" * 70)
    print("PARTIE 1 : Calibration bruit SPICE trnoise(eta) -> sigma_Python_equiv")
    print("=" * 70)

    calibration = {}  # eta -> sigma_equiv

    if not NGSPICE.exists():
        print(f"[SKIP] ngspice non trouve. Calibration theorique uniquement.")
        # Estimation theorique : trnoise(eta) injecte eta A/sqrt(Hz)
        # Sur C=1F avec dt=0.05s : sigma_V = eta * sqrt(1/(2*C)) * sqrt(1/dt)
        # (approximation bruit blanc discret)
        for eta in ETAS_SPICE:
            sigma_equiv = eta * np.sqrt(1.0 / (2.0 * DT))
            calibration[eta] = sigma_equiv
            print(f"  eta={eta:.2f}  sigma_equiv (theorique) = {sigma_equiv:.4f}")
        return calibration

    for eta in ETAS_SPICE:
        tag     = f'eta{int(eta*100):03d}'
        netlist = make_rc_noise_netlist(eta, T_REF, DT, tag)
        print(f"  eta={eta:.2f} -> ngspice... ", end='', flush=True)
        ok = run_ngspice(netlist)
        if not ok:
            print("FAILED")
            continue

        dat_path = RESULTS / f'rc_noise_{tag}.dat'
        v_trace  = parse_wrdata(dat_path)
        if len(v_trace) < 10:
            print("no data")
            continue

        # sigma_equiv : ecart-type des increments (= bruit par pas de temps)
        dv = np.diff(v_trace)
        sigma_equiv = float(np.std(dv))
        calibration[eta] = sigma_equiv
        print(f"sigma_equiv = {sigma_equiv:.4f}  (from {len(dv)} increments)")

    return calibration


# -- PARTIE 2 : Python dead-zone a sigma_equiv --------------------------------

def python_deadzone_test(sigma_equiv_dict: dict):
    """Relancer Item 10 sur BA m=5 aux amplitudes sigma_equiv et 2*sigma_equiv."""
    print("\n" + "=" * 70)
    print("PARTIE 2 : Python BA m=5 aux amplitudes calibrees")
    print(f"  I_stim={I_STIM}, steps={STEPS}, warm_up={WARM_UP}, seeds={SEEDS}")
    print("=" * 70)

    rows = []

    for eta, sigma_equiv in sorted(sigma_equiv_dict.items()):
        sigmas_to_test = [sigma_equiv, 2.0 * sigma_equiv]
        for sigma in sigmas_to_test:
            hcog_l, hcont_l = [], []
            for seed in SEEDS:
                adj = make_ba(N_BA, M_BA, seed)
                net = Mem4Network(adjacency_matrix=adj.copy(), heretic_ratio=0.15,
                                  seed=seed, coupling_norm='degree_linear')
                sigma_vec = np.full(net.N, sigma) if sigma > 1e-10 else None
                v_snaps = []
                for step in range(STEPS):
                    net.step(I_stimulus=I_STIM, sigma_v_vec=sigma_vec)
                    if step >= WARM_UP:
                        v_snaps.append(net.v.copy())
                v_s = np.array(v_snaps)
                hcog_l.append(float(np.mean([calculate_cognitive_entropy(v)  for v in v_s[::10]])))
                hcont_l.append(float(np.mean([calculate_continuous_entropy(v) for v in v_s[::10]])))

            hcog_m  = np.mean(hcog_l)
            hcont_m = np.mean(hcont_l)
            label   = f"1x_sigma_equiv" if sigma == sigma_equiv else "2x_sigma_equiv"
            print(f"  eta_spice={eta:.2f}  {label:18s}  sigma={sigma:.4f}  "
                  f"H_cog={hcog_m:.4f}  H_cont={hcont_m:.4f}")
            rows.append({
                'eta_spice':    eta,
                'sigma_python': sigma,
                'label':        label,
                'h_cog_mean':   hcog_m,  'h_cog_std':  np.std(hcog_l),
                'h_cont_mean':  hcont_m, 'h_cont_std': np.std(hcont_l),
            })

    return rows


# -- Main ---------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 70)
    print("Faille A -- Calibration eta_SPICE <-> sigma_Python")
    print("=" * 70)

    t0 = time.time()

    # Partie 1 : calibration
    calibration = calibrate_spice_noise()

    # Partie 2 : Python dead-zone test aux amplitudes calibrees
    rows = python_deadzone_test(calibration)

    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed:.1f}s")

    # -- CSV ------------------------------------------------------------------
    import csv, pathlib
    fig_dir = pathlib.Path(ROOT) / 'figures'
    fig_dir.mkdir(exist_ok=True)
    csv_path = fig_dir / 'spice_noise_calibration.csv'
    if rows:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV : {csv_path}")

    # -- Interpretation -------------------------------------------------------
    print("\n--- Interpretation ---")
    print("Calibration eta -> sigma_equiv :")
    for eta, s in sorted(calibration.items()):
        print(f"  eta={eta:.2f}  =>  sigma_equiv={s:.4f}")

    if rows:
        print("\nResultat Python a sigma_equiv :")
        for r in rows:
            rescued = "RESCUE" if r['h_cog_mean'] > 0.05 else "dead zone"
            print(f"  eta={r['eta_spice']:.2f}  {r['label']:18s}  "
                  f"sigma={r['sigma_python']:.4f}  H_cog={r['h_cog_mean']:.4f}  -> {rescued}")

    print("\nConclusion attendue :")
    print("  - Si H_cog reste ~0 a sigma_equiv : bruit thermique SPICE est")
    print("    QUALITATIVEMENT DIFFERENT du bruit Gaussien Python (renforce Paper B).")
    print("  - Si H_cog > 0 a sigma_equiv : Item 10 etait simplement sous-dose;")
    print("    la distinction qualitative est fragile.")

    # -- Figure ---------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        if not rows or not calibration:
            raise ValueError("no data to plot")

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Panel 1 : sigma_equiv vs eta
        ax = axes[0]
        etas = sorted(calibration.keys())
        sigmas = [calibration[e] for e in etas]
        ax.plot(etas, sigmas, 'o-', color='steelblue', linewidth=2, markersize=8)
        ax.axhline(1.2, color='crimson', linestyle='--', label='sigma_max Item 10 = 1.2')
        ax.set_xlabel('eta (SPICE trnoise amplitude)')
        ax.set_ylabel('sigma_equiv Python')
        ax.set_title('Calibration : eta SPICE -> sigma Python')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Panel 2 : H_cog vs sigma_python pour chaque eta
        ax2 = axes[1]
        colors = plt.cm.viridis(np.linspace(0, 1, len(etas)))
        for i, eta in enumerate(etas):
            eta_rows = [r for r in rows if r['eta_spice'] == eta]
            xs = [r['sigma_python'] for r in eta_rows]
            ys = [r['h_cog_mean']   for r in eta_rows]
            ax2.scatter(xs, ys, color=colors[i], s=100, label=f'eta={eta:.1f}',
                        zorder=3, edgecolors='k', linewidths=0.5)
        ax2.axvline(1.2, color='crimson', linestyle='--', label='sigma_max Item 10')
        ax2.axhline(0.05, color='gray', linestyle=':', alpha=0.6, label='rescue threshold')
        ax2.set_xlabel('sigma_python')
        ax2.set_ylabel('H_cog (BA m=5)')
        ax2.set_title('Python H_cog a sigma_equiv — dead zone rescuable ?')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        fig.suptitle(
            'Faille A — Calibration eta_SPICE <-> sigma_Python\n'
            'BA m=5 N=100, degree_linear, I_stim=0.5',
            fontsize=11
        )
        plt.tight_layout()
        png_path = fig_dir / 'spice_noise_calibration.png'
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        print(f"Figure : {png_path}")
    except Exception as e:
        print(f"[matplotlib error] {e}")
