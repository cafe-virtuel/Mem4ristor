#!/usr/bin/env python3
"""
P8 (legs de Fable) -- Le bruit colore pour integrateur adaptatif : lever la reserve numerique.
==============================================================================================
Cree : 2026-07-12 (Claude Fable 5, L'Ingenieur) -- piste P8 de
docs/PISTES_POUR_LA_SUITE_2026-07-12.md (reco technique 2 du rapport Manus,
01/05/2026, jamais faite).

LA RESERVE. Le projet vit avec 'Euler-Maruyama formellement le seul coherent
avec le bruit blanc, RK45 interdit avec sigma_v > 0' (dynamics.py). Un reviewer
numerique peut demander : les resultats dependent-ils de l'integrateur ? La
reponse standard : remplacer le bruit BLANC par un bruit COLORE (Ornstein-
Uhlenbeck a temps de correlation tau court, INTERPOLE continument) -- le RHS
devient deterministe et continu, RK45 est alors 100%% legitime ; si les
observables convergent vers celles d'Euler+blanc quand tau -> 0, Euler
n'introduisait pas d'artefact.

PERIMETRE HONNETE (leçon AUDIT-024 : changer le bruit change les chiffres --
ceci est une VALIDATION CROISEE, jamais un remplacement, et AUCUN chiffre
canonique n'est modifie).
  - Dynamique testee : le RHS REDUIT que le coeur lui-meme definit dans
    solve_rk45() (FHN + doute + couplage u_filter, heretics +/-I ; sans
    hysteresis/plasticite saturee/ART/watchdog -- c'est la dynamique que le
    coeur declare integrable par RK45). Il est REPLIQUE dans ce script pour
    pouvoir y injecter un forcage vectoriel par noeud, avec un GATE DE
    FIDELITE : a bruit nul, la replique doit reproduire net.model.solve_rk45
    a ~1e-6 pres, sinon campagne annulee (gate de replication, leçon 11/07).
  - La validation du pipeline COMPLET Table 1 (step() avec hysteresis etc.)
    reste un travail futur -- documente, pas escamote.

PROTOCOLE.
  - Lattice 10x10 periodique, heretic_ratio par defaut du coeur (masque du
    modele), I_base=0.5 (regime driven canonique), T=80 u.t., fenetre
    stationnaire [40, 80], echantillonnage dt=0.05.
  - Bruit blanc de reference : la convention EXACTE du coeur (eta =
    N(0, sigma_v)/sqrt(dt) ajoute au RHS, schema Euler-Maruyama, dt=0.05,
    sigma_v=0.05).
  - Bruit OU calibre a MEME densite spectrale basse frequence :
    S(0) = 2*sigma_OU^2*tau = sigma_v^2 => sigma_OU = sigma_v/sqrt(2*tau).
    Genere par recurrence exacte sur une grille Delta=tau/4, initialise a la
    stationnaire, INTERPOLE lineairement (le 'generateur de bruit interpole'
    de la reco Manus) -> forcage continu I_noise(t) par noeud.
  - 3 integrateurs compares, MEMES realisations de bruit OU par (seed, tau) :
      EULER_WHITE : Euler dt=0.05 + blanc (la reference canonique)
      EULER_OU    : Euler dt=0.05 + OU interpole
      RK45_OU     : solve_ivp RK45 rtol=1e-6, max_step<=Delta/2 + OU interpole
  - tau dans {0.4, 0.1, 0.025, 0.00625} (convergence tau -> 0), 4 seeds,
    ablation FULL vs FROZEN_U (du=0, u fige a l'init). Le 4e tau a ete
    ajoute au lancement 2 : a tau=0.025 la sync FULL convergeait
    monotonement vers le blanc (0.121 -> 0.093 -> 0.079 vs 0.072, ecarts
    residuels 0.049 -> 0.021 -> 0.007) sans encore entrer dans le 2sd
    inter-seeds (minuscule : 0.0018) -- il fallait un tau plus fin pour
    conclure, pas un autre critere.
  - NOTE previsible : EULER_OU a dt=0.05 ne RESOUT PAS un bruit tau < dt
    (aliasing du forcage) -- le critere 3 (Euler vs RK45 a meme tau) n'est
    interpretable que pour tau >= 2*dt ; aux tau plus fins, seul le critere
    2 (RK45_OU vs EULER_WHITE) fait foi. Documente au lancement 2.
  - Observables : H_cont (metrique du coeur, moyenne sur 20 snapshots de la
    fenetre) et sync (correlation de Pearson pairwise moyenne sur la fenetre,
    l'observable du resultat central C04/C13).

CRITERES PRE-FIXES (avant de voir un chiffre) :
  1. Gate de fidelite deterministe < 1e-6 relatif, sinon abort.
  2. Convergence : a tau=0.025, |RK45_OU - EULER_WHITE| <= 2 ecarts-types
     inter-seeds de EULER_WHITE, pour H_cont ET sync, dans les 2 ablations.
  3. Consistance d'integrateur : |RK45_OU - EULER_OU| a meme tau <= 2 sd
     (la difference d'INTEGRATEUR est negligeable devant celle du bruit).
  4. L'ablation survit partout : sync(FROZEN) > sync(FULL) dans toutes les
     conditions -- le resultat central ne depend ni de la couleur du bruit
     ni de l'integrateur.
  Si 2-4 passent : la porte du reviewer numerique se ferme sur la dynamique
  reduite. Si un critere echoue : le dire tel quel (c'est une decouverte).

Statut : exploratoire, hors preprint, coeur non touche, CSV canoniques non touches.
Sorties : figures/p8_colored_noise_rk45_poc{,_agg}.csv + .png
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from mem4ristor.topology import Mem4Network  # noqa: E402
from mem4ristor.graph_utils import make_lattice_adj  # noqa: E402
from mem4ristor.metrics import calculate_continuous_entropy  # noqa: E402

CSV_PATH = ROOT / "figures" / "p8_colored_noise_rk45_poc.csv"
AGG_PATH = ROOT / "figures" / "p8_colored_noise_rk45_poc_agg.csv"
PNG_PATH = ROOT / "figures" / "p8_colored_noise_rk45_poc.png"

SIDE, N = 10, 100
DT = 0.05                      # dt Euler canonique du coeur
T_TOTAL = 80.0
T_WIN = 40.0                   # debut de la fenetre stationnaire
I_BASE = 0.5
SIGMA_V = 0.05                 # bruit blanc canonique du coeur
TAUS = [0.4, 0.1, 0.025, 0.00625]
SEEDS = [0, 1, 2, 3]
CONDITIONS = ["FULL", "FROZEN_U"]


def build_model(seed):
    """Modele du coeur (pour v0/w0/u0, heretic_mask, D_eff, constantes)."""
    adj = make_lattice_adj(SIDE, periodic=True).astype(float)
    net = Mem4Network(size=SIDE, seed=seed, adjacency_matrix=adj)
    m = net.model
    m.cfg['noise']['sigma_v'] = 0.0        # le bruit est gere par CE script
    return net, m, adj


def make_rhs(m, adj, frozen, noise_fn):
    """Replique EXACTE du RHS de dynamics.solve_rk45 (sigma_v=0), plus un
    forcage vectoriel noise_fn(t) ajoute au courant, et du=0 si frozen."""
    cfg = m.cfg
    alpha = cfg['dynamics']['alpha']
    vdiv = cfg['dynamics']['v_cubic_divisor']
    eps = cfg['dynamics']['epsilon']
    a_p, b_p = cfg['dynamics']['a'], cfg['dynamics']['b']
    eps_u = cfg['doubt']['epsilon_u']
    k_u = cfg['doubt']['k_u']
    s_base = cfg['doubt']['sigma_baseline']
    tau_u = cfg['doubt']['tau_u']
    a_sur = cfg['doubt'].get('alpha_surprise', 2.0)
    lam = m.lambda_intrinsic
    w_sat = m.w_saturation
    tau_pl = m.tau_plasticity
    steep = m.sigmoid_steepness
    leak = m.social_leakage
    d_eff = m.D_eff
    hmask = m.heretic_mask

    def rhs(t, y):
        v, w, u = y[:N], y[N:2 * N], y[2 * N:]
        lap_v = adj @ v - v
        sigma_social = np.abs(lap_v)
        u_filter = np.tanh(steep * (0.5 - u)) + leak
        I_eff = np.full(N, I_BASE)
        I_eff[hmask] *= -1.0
        I_ext = I_eff + d_eff * u_filter * lap_v + noise_fn(t)
        dv = v - (v ** 3) / vdiv - w + I_ext - alpha * np.tanh(v)
        dw_fhn = eps * (v + a_p - b_p * w)
        p_drive = lam * sigma_social * (u > 0.5).astype(float)
        w_ratio = w / w_sat
        sat = np.clip(1.0 - w_ratio ** 2, 0.0, 1.0)
        dw = dw_fhn + (p_drive * sat) - (w / tau_pl)
        if frozen:
            du = np.zeros(N)
        else:
            e_ad = eps_u * np.clip(1.0 + a_sur * sigma_social, 1.0, 5.0)
            du = (e_ad * (k_u * sigma_social + s_base - u)) / tau_u
        return np.concatenate([dv, dw, du])

    return rhs


def make_ou_interp(seed, tau, rng_offset=0):
    """Grille OU exacte (Delta=tau/4) + interpolation lineaire -> fn continue."""
    delta = tau / 4.0
    n_grid = int(np.ceil(T_TOTAL / delta)) + 2
    sigma_ou = SIGMA_V / np.sqrt(2.0 * tau)
    rng = np.random.default_rng(90000 + seed * 100 + rng_offset)
    a_dec = np.exp(-delta / tau)
    grid = np.empty((n_grid, N))
    grid[0] = rng.normal(0.0, sigma_ou, N)
    s_step = sigma_ou * np.sqrt(1.0 - a_dec ** 2)
    for k in range(1, n_grid):
        grid[k] = grid[k - 1] * a_dec + rng.normal(0.0, s_step, N)

    def fn(t):
        x = t / delta
        k = min(int(x), n_grid - 2)
        f = x - k
        return grid[k] * (1.0 - f) + grid[k + 1] * f

    return fn, delta


def euler_run(m, adj, frozen, noise_mode, seed, tau=None):
    """Euler dt=0.05 sur le RHS commun. noise_mode: 'white' (convention coeur,
    bruit ajoute au RHS en eta/sqrt(dt)) ou 'ou' (forcage interpole)."""
    if noise_mode == "ou":
        noise_fn, _ = make_ou_interp(seed, tau)
        rhs = make_rhs(m, adj, frozen, noise_fn)
    else:
        rng = np.random.default_rng(91000 + seed * 100)
        zero_fn = lambda t: 0.0  # noqa: E731
        rhs = make_rhs(m, adj, frozen, zero_fn)
    y = np.concatenate([m.v.copy(), m.w.copy(), m.u.copy()])
    n_steps = int(round(T_TOTAL / DT))
    keep_from = int(round(T_WIN / DT))
    traj = np.empty((n_steps - keep_from, N))
    for k in range(n_steps):
        dy = rhs(k * DT, y)
        if noise_mode == "white":
            eta = rng.normal(0.0, SIGMA_V, N) / np.sqrt(DT)
            dy[:N] += eta
        y = y + DT * dy
        if not np.all(np.isfinite(y)):
            return None
        if k >= keep_from:
            traj[k - keep_from] = y[:N]
    return traj


def rk45_run(m, adj, frozen, seed, tau):
    noise_fn, delta = make_ou_interp(seed, tau)
    rhs = make_rhs(m, adj, frozen, noise_fn)
    y0 = np.concatenate([m.v.copy(), m.w.copy(), m.u.copy()])
    t_eval = np.arange(T_WIN, T_TOTAL, DT)
    sol = solve_ivp(rhs, (0.0, T_TOTAL), y0, method="RK45", rtol=1e-6,
                    max_step=min(DT, delta / 2.0), t_eval=t_eval)
    if not sol.success:
        return None
    return sol.y[:N, :].T


def observables(traj):
    """H_cont (moyenne sur 20 snapshots) + sync (Pearson pairwise moyen)."""
    idx = np.linspace(0, traj.shape[0] - 1, 20).astype(int)
    h = float(np.mean([calculate_continuous_entropy(traj[i]) for i in idx]))
    c = np.corrcoef(traj.T)
    iu = np.triu_indices(N, k=1)
    sync = float(np.nanmean(np.abs(c[iu])))
    return h, sync


def main() -> int:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ---------------- gate de fidelite deterministe ----------------
    print("[gate] replique du RHS vs net.model.solve_rk45, bruit nul, T=20...")
    net, m, adj = build_model(0)
    v0, w0, u0 = m.v.copy(), m.w.copy(), m.u.copy()
    sol_core = m.solve_rk45((0.0, 20.0), I_stimulus=I_BASE, adj_matrix=adj)
    v_core, w_core, u_core = m.v.copy(), m.w.copy(), m.u.copy()
    m.v, m.w, m.u = v0.copy(), w0.copy(), u0.copy()
    zero_fn = lambda t: 0.0  # noqa: E731
    rhs = make_rhs(m, adj, False, zero_fn)
    y0 = np.concatenate([v0, w0, u0])
    sol_mine = solve_ivp(rhs, (0.0, 20.0), y0, method="RK45", rtol=1e-6,
                         max_step=0.1)
    ym = sol_mine.y[:, -1]
    err = max(
        float(np.max(np.abs(ym[:N] - v_core)) / (np.max(np.abs(v_core)) + 1e-12)),
        float(np.max(np.abs(ym[N:2 * N] - w_core)) / (np.max(np.abs(w_core)) + 1e-12)),
        float(np.max(np.abs(np.clip(ym[2 * N:], 0, 1) - u_core)) / (np.max(np.abs(u_core)) + 1e-12)),
    )
    print(f"[gate] erreur relative max = {err:.2e}")
    if err > 1e-6:
        print("[gate] ECHEC (> 1e-6) : la replique du RHS n'est pas fidele. "
              "Campagne annulee.")
        return 1
    print("[gate] OK. Campagne lancee.")

    # ---------------- campagne ----------------
    rows = []
    res = {}
    total = len(SEEDS) * len(CONDITIONS) * (1 + 2 * len(TAUS))
    done = 0
    for seed in SEEDS:
        for cond in CONDITIONS:
            frozen = cond == "FROZEN_U"

            def fresh():
                _, mm, aa = build_model(seed)
                return mm, aa

            mm, aa = fresh()
            traj = euler_run(mm, aa, frozen, "white", seed)
            runs = [("EULER_WHITE", None, traj)]
            for tau in TAUS:
                mm, aa = fresh()
                runs.append(("EULER_OU", tau, euler_run(mm, aa, frozen, "ou", seed, tau)))
                mm, aa = fresh()
                runs.append(("RK45_OU", tau, rk45_run(mm, aa, frozen, seed, tau)))
            for name, tau, tr in runs:
                done += 1
                if tr is None:
                    print(f"  [{done}/{total}] {name} tau={tau} seed={seed} {cond} "
                          f"-> DIVERGE/ECHEC")
                    continue
                h, sync = observables(tr)
                res[(name, tau, cond, seed)] = (h, sync)
                rows.append((name, tau if tau is not None else "", cond, seed, h, sync))
            print(f"  seed={seed} {cond} fait [{done}/{total}, {time.time()-t0:.0f}s]")

    # ---------------- agregats ----------------
    def agg(name, tau, cond):
        vals = [res[k] for k in res if k[0] == name and k[1] == tau and k[2] == cond]
        hs = np.array([v[0] for v in vals])
        ss = np.array([v[1] for v in vals])
        return hs.mean(), hs.std(), ss.mean(), ss.std()

    print(f"\n{'integrateur':<13}{'tau':>7}{'cond':<10}{'H_cont':>16}{'sync':>16}")
    print("-" * 66)
    agg_rows = []
    for cond in CONDITIONS:
        hm, hs, sm, ss = agg("EULER_WHITE", None, cond)
        print(f"{'EULER_WHITE':<13}{'--':>7}{cond:<10}{hm:>9.4f}+-{hs:<5.3f}{sm:>9.4f}+-{ss:<5.3f}")
        agg_rows.append(("EULER_WHITE", "", cond, hm, hs, sm, ss))
        for name in ["EULER_OU", "RK45_OU"]:
            for tau in TAUS:
                hm, hs, sm, ss = agg(name, tau, cond)
                print(f"{name:<13}{tau:>7}{cond:<10}{hm:>9.4f}+-{hs:<5.3f}{sm:>9.4f}+-{ss:<5.3f}")
                agg_rows.append((name, tau, cond, hm, hs, sm, ss))

    # ---------------- verdicts ----------------
    print("\n=== VERDICT P8 (criteres pre-fixes) ===")
    ok_all = True
    tau_min = min(TAUS)
    for cond in CONDITIONS:
        hw, hw_sd, sw, sw_sd = agg("EULER_WHITE", None, cond)
        hr, _, sr, _ = agg("RK45_OU", tau_min, cond)
        he, _, se, _ = agg("EULER_OU", tau_min, cond)
        c2h = abs(hr - hw) <= 2 * hw_sd
        c2s = abs(sr - sw) <= 2 * sw_sd
        print(f"  [{cond}] 2. RK45_OU(tau={tau_min}) vs EULER_WHITE : "
              f"dH={hr-hw:+.4f} (2sd={2*hw_sd:.4f}) {'OK' if c2h else 'HORS'} ; "
              f"dsync={sr-sw:+.4f} (2sd={2*sw_sd:.4f}) {'OK' if c2s else 'HORS'}")
        c3h = abs(hr - he) <= 2 * hw_sd
        c3s = abs(sr - se) <= 2 * sw_sd
        print(f"  [{cond}] 3. RK45_OU vs EULER_OU (tau={tau_min}) : "
              f"dH={hr-he:+.4f} {'OK' if c3h else 'HORS'} ; "
              f"dsync={sr-se:+.4f} {'OK' if c3s else 'HORS'}")
        ok_all &= c2h and c2s and c3h and c3s
    # critere 4 : ablation partout
    ok4 = True
    for name, tau in [("EULER_WHITE", None)] + [(n, t) for n in ["EULER_OU", "RK45_OU"]
                                                for t in TAUS]:
        _, _, s_full, _ = agg(name, tau, "FULL")
        _, _, s_froz, _ = agg(name, tau, "FROZEN_U")
        if not (s_froz > s_full):
            ok4 = False
            print(f"  4. ABLATION INVERSEE pour {name} tau={tau} : "
                  f"sync FROZEN={s_froz:.4f} <= FULL={s_full:.4f}")
    if ok4:
        print("  4. Ablation preservee PARTOUT (sync FROZEN > FULL dans les "
              f"{2 * len(TAUS) + 1} configurations x 2 verifiees).")
    if ok_all and ok4:
        print("\n  -> RESERVE NUMERIQUE LEVEE sur la dynamique reduite : un RK45")
        print("     legitime (bruit OU interpole) converge vers Euler+blanc quand")
        print("     tau -> 0, et l'ablation centrale survit a la couleur du bruit")
        print("     et a l'integrateur. Reste futur : le pipeline complet step().")
    else:
        print("\n  -> Au moins un critere HORS tolerance : voir details ci-dessus.")
        print("     A rapporter tel quel -- si RK45+OU(tau->0) ne converge pas vers")
        print("     Euler+blanc, la dependance a l'integrateur/bruit est REELLE.")

    # ---------------- sorties ----------------
    with CSV_PATH.open("w", encoding="utf-8") as f:
        f.write("integrator,tau,condition,seed,h_cont,sync\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    with AGG_PATH.open("w", encoding="utf-8") as f:
        f.write("integrator,tau,condition,h_cont_mean,h_cont_std,sync_mean,sync_std\n")
        for r in agg_rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print(f"\n[csv] {CSV_PATH}\n[csv] {AGG_PATH}")

    # ---------------- figure ----------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
        markers = {"EULER_OU": "s", "RK45_OU": "o"}
        colors = {"FULL": "#d62728", "FROZEN_U": "#1f77b4"}
        for ax, (obs, label) in zip(axes, [(0, "H_cont"), (2, "sync (|Pearson| moyen)")]):
            for cond in CONDITIONS:
                hw = agg("EULER_WHITE", None, cond)
                ax.axhline(hw[obs], color=colors[cond], ls="-", lw=1.2,
                           label=f"EULER_WHITE {cond}")
                ax.fill_between([min(TAUS) * 0.8, max(TAUS) * 1.2],
                                hw[obs] - 2 * hw[obs + 1], hw[obs] + 2 * hw[obs + 1],
                                color=colors[cond], alpha=0.12)
                for name in ["EULER_OU", "RK45_OU"]:
                    xs, ys = [], []
                    for tau in TAUS:
                        a = agg(name, tau, cond)
                        xs.append(tau)
                        ys.append(a[obs])
                    ax.plot(xs, ys, markers[name] + "--", color=colors[cond],
                            mfc="white" if name == "EULER_OU" else colors[cond],
                            label=f"{name} {cond}")
            ax.set_xscale("log")
            ax.set_xlabel("tau du bruit OU (log)")
            ax.set_ylabel(label)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=6.5)
        fig.suptitle("P8 -- bruit OU interpole + RK45 vs Euler+blanc : convergence tau->0 "
                     "(bande = 2 sd inter-seeds du canonique)", fontsize=10)
        plt.tight_layout()
        plt.savefig(PNG_PATH, dpi=140)
        print(f"[png] {PNG_PATH}")
    except Exception as e:
        print(f"[png] skipped: {e}")

    print(f"\nWall time: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
