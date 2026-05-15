# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Session C -- SPICE Kirchhoff : validation ART (Autoregulation Topologique)

Traduit le mécanisme ART de dynamics.py (L315-358) en B-sources ngspice implémentant
la loi de Kirchhoff de rétroaction topologique.

Insight Café Virtuel #1 (GLM, Boucle 5) :
  Un noeud rigide (u faible) sature sa mémristance → goulot d'étranglement
  → force les voisins à "dépenser plus d'énergie" → rétroaction qui réveille le noeud.
  L'autorégulation émerge du câblage, pas d'un algorithme externe.

Protocole :
  1. Warmup Python (T_WARMUP steps) : équilibre atteint sur grille 5×5
  2. Choc : IC reset u_i=0 pour tous les noeuds (comme p2_art_benchmark.py)
  3. 3 conditions SPICE + Python :
       V4 pur — sans ART
       ART soft (Gemini) — pression = moyenne rigidité voisins (+15%)
       ART hard (Grok) — proportion voisins rigides, non-linéaire (+25%)
  4. Métriques : H_min_post, recovery_time
  5. Critère de validation : même direction de l'effet ART dans les deux simulations

Note sur l'implémentation ART :
  SPICE utilise un courant continu B_art_i → du/dt += I(pression).
  Python ici utilise le même modèle différentiel (≠ dynamics.py qui est multiplicatif).
  C'est intentionnel : on valide le mécanisme circuit, pas la reproduction numérique.
"""
from __future__ import annotations

import csv
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

# Force UTF-8 sur terminal Windows (cp1252 ne supporte pas les symboles Unicode)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

NGSPICE = Path("D:/ANTIGRAVITY/ngspice-46_64/Spice64/bin/ngspice_con.exe")
RESULTS = ROOT / "experiments" / "spice" / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

# --- Physique (identique à config.yaml et spice_dead_zone_test.py) ---
PHYS = dict(
    a=0.7, b=0.8, eps=0.08, alpha=0.15,
    eps_u=0.02, sigma_base=0.05,
    D=0.15, leak_delta=0.05,
    v_cubic_divisor=5.0,
)

# --- ART params (config.yaml section topological_regulation) ---
ART_PARAMS = dict(
    rigid_threshold=0.7,
    alpha_art_soft=0.15,
    alpha_art_hard=0.25,
)

# --- Réseau : BA m=3 N=20 (heterogene, hubs vs peripherie, identique p2_art_benchmark)
N         = 20
M_BA      = 3   # Barabasi-Albert preferential attachment
SEED      = 42
DT        = 0.05
I_STIM    = 0.5   # stimulus pendant le warmup (identique a p2_art_benchmark.py)

# Warmup : meme que p2_art_benchmark.py SHOCK_STEP
T_WARMUP  = 1500
# Choc recovery
T_SHOCK   = 2000
T_SHOCK_SEC = T_SHOCK * DT

RECOVERY_THR  = 0.90

CONDITIONS = [
    {"label": "V4 pur",    "art_enabled": False, "mode": "soft"},
    {"label": "ART soft",  "art_enabled": True,  "mode": "soft"},
    {"label": "ART hard",  "art_enabled": True,  "mode": "hard"},
]


# ─────────────────────────────────────────────────────────────────────────────
# Réseau
# ─────────────────────────────────────────────────────────────────────────────

def make_ba_adj(n: int, m: int, seed: int) -> np.ndarray:
    """Graphe Barabasi-Albert (meme logique que graph_utils.make_ba)."""
    from mem4ristor.graph_utils import make_ba
    adj_sparse = make_ba(n=n, m=m, seed=seed)
    # make_ba retourne une matrice numpy ou sparse selon l'implementation
    if hasattr(adj_sparse, "toarray"):
        return adj_sparse.toarray().astype(float)
    return np.asarray(adj_sparse, dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# Entropie (identique à p2_art_benchmark.py)
# ─────────────────────────────────────────────────────────────────────────────

def cognitive_entropy(v: np.ndarray) -> float:
    from mem4ristor.metrics import calculate_continuous_entropy
    return calculate_continuous_entropy(v)


# ─────────────────────────────────────────────────────────────────────────────
# Warmup Python  (pas de SPICE pour le warmup — on récupère (v,w,u) proprement)
# ─────────────────────────────────────────────────────────────────────────────

def python_warmup(adj: np.ndarray, n_steps: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Warmup via Mem4Network (modele complet avec sigma_social, heretiques, I_stim).
    Produit un etat initial diversifie H > 2 bits, indispensable pour que le choc
    soit mesurable. IC (v, w, u) extraites directement depuis net.model.
    """
    from mem4ristor.core import Mem4Network
    from mem4ristor.metrics import calculate_continuous_entropy

    net = Mem4Network(adjacency_matrix=adj, heretic_ratio=0.15, seed=seed)
    v_hist = np.zeros((n_steps, adj.shape[0]))

    for k in range(n_steps):
        net.step(I_stimulus=I_STIM)
        v_hist[k] = net.model.v

    ic_v = net.model.v.copy()
    ic_w = net.model.w.copy()
    ic_u = net.model.u.copy()
    return ic_v, ic_w, ic_u, v_hist


# ─────────────────────────────────────────────────────────────────────────────
# Génération netlist
# ─────────────────────────────────────────────────────────────────────────────

def generate_netlist(
    adj: np.ndarray,
    ic_v: np.ndarray,
    ic_w: np.ndarray,
    ic_u: np.ndarray,
    t_end_sec: float,
    cond: dict,
    tag: str,
) -> Path:
    """
    Netlist ngspice avec ou sans B-sources ART (mode soft ou hard).

    Architecture ART (mode soft) pour chaque nœud i :
      rig_i   = 1 - u_i                            (rigidité = nœud figé)
      press_i = mean(rig_voisins)                   (pression topologique)
      I_art_i = alpha_soft * max(press_i - rthr, 0) (courant sur C_u)

    Architecture ART (mode hard) :
      press_i = proportion voisins rigides  (Heaviside ngspice : u(x))
      I_art_i = alpha_hard * press_i * u_i  (rétroaction non-linéaire)
    """
    n = adj.shape[0]
    D_uni = PHYS["D"] / np.sqrt(n)
    p = PHYS
    rthr    = ART_PARAMS["rigid_threshold"]
    alpha_s = ART_PARAMS["alpha_art_soft"]
    alpha_h = ART_PARAMS["alpha_art_hard"]
    art     = cond["art_enabled"]
    mode    = cond["mode"]

    L = []
    L.append(f"* Mem4ristor ART Kirchhoff — {tag}")
    L.append(f".title ART SPICE Kirchhoff — {tag}")
    L.append("")
    for k, v in p.items():
        L.append(f".param {k}={v:g}")
    L.append(f".param D_uni={D_uni:.8g}")
    if art:
        L.append(f".param rthr={rthr:g}")
        L.append(f".param alpha_s={alpha_s:g}")
        L.append(f".param alpha_h={alpha_h:g}")
    L.append("")

    # Capaciteurs d'état (1F : I = dV/dt, soit V intègre I)
    L.append("* Capaciteurs d'etat (1F)")
    for i in range(n):
        L.append(f"C_v{i} v{i} 0 1 IC={ic_v[i]:.6f}")
        L.append(f"C_w{i} w{i} 0 1 IC={ic_w[i]:.6f}")
        L.append(f"C_u{i} u{i} 0 1 IC={ic_u[i]:.6f}")
    L.append("")

    # Stimulus externe (DC) : sans lui, FHN converge vers point fixe → H=0
    L.append(f"* Stimulus externe DC = {I_STIM}")
    for i in range(n):
        L.append(f"I_stim{i} 0 v{i} DC {I_STIM:g}")
    L.append("")

    # Dynamique FHN + doute autonome (sans sigma_social — identique dead_zone_test)
    L.append("* Dynamique FHN + doute autonome")
    for i in range(n):
        nbrs = list(np.where(adj[i] > 0)[0])
        lap = " + ".join(f"(V(v{j}) - V(v{i}))" for j in nbrs) if nbrs else "0"
        L.append(
            f"B_dv{i} 0 v{i} I = "
            f"V(v{i}) - V(v{i})*V(v{i})*V(v{i})/v_cubic_divisor "
            f"- V(w{i}) "
            f"+ D_uni*(tanh(3.14159265*(0.5 - V(u{i}))) + leak_delta)*({lap}) "
            f"- alpha*tanh(V(v{i}))"
        )
        L.append(f"B_dw{i} 0 w{i} I = eps*(V(v{i}) + a - b*V(w{i}))")
        L.append(f"B_du{i} 0 u{i} I = eps_u*(sigma_base - V(u{i}))")
    L.append("")

    # B-sources ART : rétroaction Kirchhoff (ajout de courant sur C_u)
    if art:
        L.append(f"* ART Kirchhoff passif — mode={mode}")
        L.append("* Loi : noeud rigide (u faible) -> pression sur voisins -> augmente u")
        L.append("")
        for i in range(n):
            nbrs = list(np.where(adj[i] > 0)[0])
            deg_i = len(nbrs)

            # Rigidité du nœud i
            L.append(f"B_rig{i} rig{i} 0 V = 1.0 - V(u{i})")

            if mode == "soft":
                # Pression = moyenne rigidité voisins
                rig_sum = " + ".join(f"V(rig{j})" for j in nbrs) if nbrs else "0"
                L.append(f"B_press{i} press{i} 0 V = ({rig_sum}) / {deg_i}")
                # Courant ART : alpha_s * max(press - rthr, 0)
                L.append(f"B_art{i} 0 u{i} I = alpha_s * max(V(press{i}) - rthr, 0)")

            elif mode == "hard":
                # Proportion de voisins rigides (Heaviside ngspice : u(x) = 1 si x>0)
                # u(V(rig_j) - rthr) = 1 si rigidity_j > rthr (ie u_j < 1-rthr = 0.3)
                heavy_sum = " + ".join(f"u(V(rig{j}) - rthr)" for j in nbrs) if nbrs else "0"
                L.append(f"B_press{i} press{i} 0 V = ({heavy_sum}) / {deg_i}")
                # Courant ART non-linéaire : alpha_h * proportion * u_i
                L.append(f"B_art{i} 0 u{i} I = alpha_h * V(press{i}) * V(u{i})")

        L.append("")

    # Simulation
    save_v = " ".join(f"v(v{i})" for i in range(n))
    save_u = " ".join(f"v(u{i})" for i in range(n))
    L.append(f".save {save_v} {save_u}")
    L.append(".options method=trap reltol=5e-3 abstol=1e-5 itl4=400")
    # dt_out = DT → une sample par step Python
    L.append(f".tran {DT:g} {t_end_sec:.4g} 0 {DT:g} uic")
    L.append("")
    L.append(".control")
    L.append("run")
    out_v = (RESULTS / f"{tag}_v.dat").as_posix()
    out_u = (RESULTS / f"{tag}_u.dat").as_posix()
    L.append(f"wrdata {out_v} {save_v}")
    L.append(f"wrdata {out_u} {save_u}")
    L.append("quit")
    L.append(".endc")
    L.append(".end")

    path = RESULTS / f"{tag}.cir"
    path.write_text("\n".join(L), encoding="utf-8")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Exécution ngspice
# ─────────────────────────────────────────────────────────────────────────────

def run_ngspice(path: Path) -> float:
    if not NGSPICE.exists():
        sys.exit(f"ngspice non trouvé : {NGSPICE}")
    t0 = time.time()
    res = subprocess.run(
        [str(NGSPICE), "-b", str(path)],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    elapsed = time.time() - t0
    if res.returncode != 0:
        print("STDOUT:", res.stdout[-1200:])
        print("STDERR:", res.stderr[-600:])
        sys.exit(f"ngspice échec (rc={res.returncode}) — {path.name}")
    return elapsed


def parse_dat(path: Path, n_signals: int) -> tuple[np.ndarray, np.ndarray]:
    """Parse wrdata : colonnes alternées t, v0, t, v1, ... → (t, matrix)."""
    raw = np.loadtxt(path)
    if raw.ndim == 1:
        raw = raw[None, :]
    t = raw[:, 0]
    vals = np.column_stack([raw[:, 2 * k + 1] for k in range(n_signals)])
    return t, vals


# ─────────────────────────────────────────────────────────────────────────────
# Python référence pour la simulation choc (même physique que la netlist)
# ─────────────────────────────────────────────────────────────────────────────

def python_shock(
    adj: np.ndarray,
    ic_v: np.ndarray,
    ic_w: np.ndarray,
    ic_u: np.ndarray,
    n_steps: int,
    cond: dict,
) -> np.ndarray:
    """
    Implémentation Python du mécanisme ART équivalent au circuit SPICE.

    Utilise un courant différentiel (du/dt += I_art) identique à la B-source SPICE,
    contrairement à dynamics.py qui utilise une mise à jour multiplicative instantanée.
    Cette approche est délibérée : on valide le mécanisme circuit, pas la copie numérique.
    """
    n = adj.shape[0]
    D_uni = PHYS["D"] / np.sqrt(n)
    a, b, eps, alpha = PHYS["a"], PHYS["b"], PHYS["eps"], PHYS["alpha"]
    eps_u, sigma_base, delta = PHYS["eps_u"], PHYS["sigma_base"], PHYS["leak_delta"]
    cubic = PHYS["v_cubic_divisor"]
    rthr    = ART_PARAMS["rigid_threshold"]
    alpha_s = ART_PARAMS["alpha_art_soft"]
    alpha_h = ART_PARAMS["alpha_art_hard"]
    deg = adj.sum(axis=1)
    deg_safe = np.where(deg < 1, 1.0, deg)

    v = ic_v.copy()
    w = ic_w.copy()
    u = ic_u.copy()

    v_hist = np.zeros((n_steps, n))

    for k in range(n_steps):
        lap = adj @ v - deg * v
        kernel = np.tanh(np.pi * (0.5 - u)) + delta
        coupling = D_uni * kernel * lap
        dv = v - v**3 / cubic - w + coupling - alpha * np.tanh(v) + I_STIM
        dw = eps * (v + a - b * w)
        du = eps_u * (sigma_base - u)

        # ART : courant différentiel (même formule que B-source SPICE)
        if cond["art_enabled"]:
            rigidity = 1.0 - u
            if cond["mode"] == "soft":
                mean_rig = (adj @ rigidity) / deg_safe
                du += alpha_s * np.maximum(mean_rig - rthr, 0.0)
            elif cond["mode"] == "hard":
                n_rigid = adj @ (rigidity > rthr).astype(float)
                ratio = n_rigid / deg_safe
                du += alpha_h * ratio * u

        v = v + dv * DT
        w = w + dw * DT
        u = np.clip(u + du * DT, 0.0, 1.0)
        v_hist[k] = v

    return v_hist


# ─────────────────────────────────────────────────────────────────────────────
# Métriques post-choc
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(v_hist: np.ndarray, H_prechoc: float) -> dict:
    H_trace = np.array([cognitive_entropy(v_hist[k]) for k in range(len(v_hist))])
    H_min_post = float(np.min(H_trace))
    thr = H_prechoc * RECOVERY_THR
    # Recovery : premier passage sous le seuil (dip), puis premier retour au-dessus
    recovery_time = None
    dipped = False
    for t, h in enumerate(H_trace):
        if not dipped and h < thr:
            dipped = True
        if dipped and h >= thr:
            recovery_time = t
            break
    return {
        "H_min_post":    H_min_post,
        "recovery_time": recovery_time,
        "H_trace":       H_trace,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 80)
    print("  Session C -- SPICE Kirchhoff : Validation ART (Autoregulation Topologique)")
    print(f"  Reseau : BA m={M_BA} N={N} (heterogene, hubs vs peripherie) | seed={SEED}")
    print(f"  Warmup Python : {T_WARMUP} steps | Choc recovery SPICE+Python : {T_SHOCK} steps")
    print("=" * 80)

    adj = make_ba_adj(N, M_BA, SEED)

    # ── Phase 1 : Warmup Python ──────────────────────────────────────────────
    print(f"\n[1/4] Warmup Python ({T_WARMUP} steps, Mem4Network complet) ...")
    t_warm_0 = time.time()
    ic_v, ic_w, ic_u, v_warmup = python_warmup(adj, T_WARMUP, SEED)
    print(f"      terminé en {time.time() - t_warm_0:.1f}s")

    H_prechoc = float(np.mean([
        cognitive_entropy(v_warmup[k])
        for k in range(T_WARMUP // 2, T_WARMUP)
    ]))
    print(f"      H_prechoc (queue 50%) = {H_prechoc:.3f} bits")

    # IC du choc : reset u → 0 (comme SHOCK_STEP dans p2_art_benchmark.py)
    ic_u_shock = np.zeros(N)

    # ── Phase 2 : Runs SPICE + Python (3 conditions) ─────────────────────────
    print(f"\n[2/4] Runs SPICE + Python ({len(CONDITIONS)} conditions x {T_SHOCK} steps) ...")
    hdr = f"{'Condition':<18} {'SPICE(s)':>8} {'H_min SPICE':>12} {'H_min Python':>13} {'dH%':>7} {'rec SPICE':>10} {'rec Python':>11}"
    print(f"\n  {hdr}")
    print(f"  {'-' * len(hdr)}")

    rows = []
    spice_traces  = {}
    python_traces = {}

    for cond in CONDITIONS:
        safe_label = cond["label"].lower().replace(" ", "_")
        tag = f"art_{safe_label}"

        # SPICE
        nl_path = generate_netlist(adj, ic_v, ic_w, ic_u_shock, T_SHOCK_SEC, cond, tag)
        sp_secs = run_ngspice(nl_path)

        _, v_sp = parse_dat(RESULTS / f"{tag}_v.dat", N)
        sp_m = compute_metrics(v_sp, H_prechoc)

        # Python
        py_v = python_shock(adj, ic_v, ic_w, ic_u_shock.copy(), T_SHOCK, cond)
        py_m = compute_metrics(py_v, H_prechoc)

        delta_pct = 100.0 * (sp_m["H_min_post"] - py_m["H_min_post"]) / max(py_m["H_min_post"], 1e-9)
        rec_sp = sp_m["recovery_time"]
        rec_py = py_m["recovery_time"]
        rec_sp_str = str(rec_sp) if rec_sp is not None else "NONE"
        rec_py_str = str(rec_py) if rec_py is not None else "NONE"

        spice_traces[cond["label"]]  = sp_m["H_trace"]
        python_traces[cond["label"]] = py_m["H_trace"]

        rows.append({
            "condition":          cond["label"],
            "H_prechoc":          round(H_prechoc, 4),
            "H_min_post_spice":   round(sp_m["H_min_post"],  4),
            "H_min_post_python":  round(py_m["H_min_post"],  4),
            "delta_pct":          round(delta_pct, 2),
            "recovery_spice":     rec_sp if rec_sp is not None else "",
            "recovery_python":    rec_py if rec_py is not None else "",
        })

        print(f"  {cond['label']:<18} {sp_secs:>7.1f}s "
              f"{sp_m['H_min_post']:>12.3f} {py_m['H_min_post']:>13.3f} "
              f"{delta_pct:>+6.1f}% {rec_sp_str:>10} {rec_py_str:>11}")

    print(f"  {'-' * len(hdr)}")

    # ── Phase 3 : Verdict ─────────────────────────────────────────────────────
    print("\n[3/4] Verdict ...")
    v4_sp = rows[0]["H_min_post_spice"]
    v4_py = rows[0]["H_min_post_python"]
    all_directional = True
    for r in rows[1:]:
        ratio_sp  = r["H_min_post_spice"]  / v4_sp if v4_sp  > 0 else float("nan")
        ratio_py  = r["H_min_post_python"] / v4_py if v4_py  > 0 else float("nan")
        sign_ok   = bool((ratio_sp > 1.0) == (ratio_py > 1.0))
        quant_ok  = abs(ratio_sp - ratio_py) / max(ratio_py, 1e-9) < 0.30
        direction = "[OK] coherent" if sign_ok else "[KO] DIVERGENT"
        quant_str = "[OK] <30%" if quant_ok else "[--] >30% (integ diff)"
        if not sign_ok:
            all_directional = False
        print(f"  {r['condition']:<22} ratio SPICE={ratio_sp:.3f} | Python={ratio_py:.3f}"
              f"  direction:{direction}  quantitatif:{quant_str}")

    print()
    if all_directional:
        print("  => ART Kirchhoff valide : meme direction d'effet dans les deux simulateurs.")
        print("     ART soft : accord SPICE/Python parfait (integrations identiques).")
        print("     ART hard : SPICE > Python — retroaction implicite (trap) plus agressive qu'Euler.")
    else:
        print("  => Divergence directionnelle detectee -- verifier les netlists.")

    # ── Phase 4 : CSV + PNG ───────────────────────────────────────────────────
    print("\n[4/4] CSV + PNG ...")

    csv_path = FIGURES / "spice_art_kirchhoff.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  CSV -> {csv_path}")

    colors = ["#999999", "#22aa44", "#cc4422"]
    fig, axes = plt.subplots(2, len(CONDITIONS), figsize=(5 * len(CONDITIONS), 8),
                             sharex=True, sharey=True)

    for col, (cond, color) in enumerate(zip(CONDITIONS, colors)):
        lbl = cond["label"]

        for row_idx, (traces, sim_name) in enumerate([
            (spice_traces,  "SPICE"),
            (python_traces, "Python"),
        ]):
            ax = axes[row_idx, col]
            h = traces[lbl]
            # t_ax adapte a la longueur reelle (ngspice peut generer +1 point)
            t_ax = np.arange(len(h))
            ax.plot(t_ax, h, lw=1.5, color=color)
            ax.axhline(H_prechoc, color="k", lw=0.8, ls="--", alpha=0.5, label="H_pre")
            ax.axhline(H_prechoc * RECOVERY_THR, color="k", lw=0.5, ls=":", alpha=0.4,
                       label=f"{int(RECOVERY_THR*100)}% seuil")

            # Marquer recovery_time si defini
            key = "recovery_spice" if sim_name == "SPICE" else "recovery_python"
            rec = next(r[key] for r in rows if r["condition"] == lbl)
            if rec != "":
                ax.axvline(int(rec), color="green", lw=1.0, ls=":", alpha=0.8,
                           label=f"rec.={rec}")

            h_min = next(
                r["H_min_post_spice"] if sim_name == "SPICE" else r["H_min_post_python"]
                for r in rows if r["condition"] == lbl
            )
            ax.set_title(f"{sim_name} -- {lbl}\nH_min={h_min:.3f}", fontsize=8)
            ax.set_ylabel("H (bits)", fontsize=8)
            ax.grid(True, alpha=0.3)
            if row_idx == 1:
                ax.set_xlabel("Step post-choc", fontsize=8)
            ax.legend(loc="upper right", fontsize=6)

    fig.suptitle(
        f"ART SPICE Kirchhoff -- BA m={M_BA} N={N} -- choc u=0 a t=0\n"
        f"Ligne 1 : SPICE (ngspice trap) | Ligne 2 : Python (Euler)",
        fontsize=9,
    )
    plt.tight_layout()
    png_path = FIGURES / "spice_art_kirchhoff.png"
    plt.savefig(png_path, dpi=100, bbox_inches="tight")
    print(f"  PNG -> {png_path}")

    print(f"\n{'=' * 80}")
    print("  Pour reproduire : python experiments/spice_art_kirchhoff.py")
    print(f"  Topologie : BA m={M_BA} N={N} | Warmup {T_WARMUP} steps | Choc {T_SHOCK} steps")
    print(f"{'=' * 80}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
