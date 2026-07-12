#!/usr/bin/env python3
"""
P12 (legs de Fable) -- La tache trompeuse B1d sur substrat STNO : la niche sur un corps.
=========================================================================================
Cree : 2026-07-12 (Claude Fable 5, L'Ingenieur) -- piste P12 de
docs/PISTES_POUR_LA_SUITE_2026-07-12.md, choisie par Julien (les paris de Fable,
dans l'ordre).

CONTEXTE. Le 11/07, b5_stno_narma10_poc.py a mesure le substrat STNO sur NARMA10
(le terrain de la MEMOIRE : doute neutre, decouple gagne, substrat utile en absolu).
Le terrain du DOUTE, c'est la decision trompeuse (B1d, 07/07 : gain +0.58 sur FHN ;
B5b, 08/07 : l'horloge de deliberation |Lv| ecrase les arrets naifs mais egale le
meilleur budget fixe). Si le doute gagne LA sur le substrat physique, la proposition
falsifiable B6 devient complete : "un reseau de STNO a couplage module par le
desaccord prend de meilleures decisions sous tromperie" -- testable par un labo.

PROTOCOLE (loyaute, tout pre-fixe avant lancement) :
  - Piege B1d PULSE porte tel quel : leurre NOMBREUX (26 noeuds) + FORT (E=1.0),
    retire apres T_pulse ; verite PERSISTANTE (14 noeuds), plus faible (E=0.6),
    seule active apres le pulse -> la decision globale bascule FAUX -> JUSTE
    tardivement. Memes N_DISTRACT/N_TRUE/E_* que deceptive_task_poc.py.
  - Substrat : oscillateurs Slavin-Tiberkevich, constantes STRICTEMENT celles de
    b5_stno_narma10_poc.py (GAMMA_PLUS=1.2, Q=1, K=0.3, dt=0.005 valide 11/07,
    doute aux constantes de dynamics.py, GAIN_U=10 calibre).
  - Entree physique : modulation du gain par courant STT (gamma_plus_eff =
    GAMMA_PLUS + ISCALE * stim par noeud) -- la voie standard (Torrejon 2017).
  - Lecture physique : puissance p=|a|^2 (diode), readout par PAIRE
    DIFFERENTIELLE (revision 2, voir plus bas) : deux copies jumelles du
    reseau, l'une recoit +stim, l'autre -stim, meme seed et MEME bruit.
  - Decision : d(t) = mean(p_plus) - mean(p_moins), lissee W_READ ; dec = signe.
  - 2 substrats : STNO_FULL (couplage module par u_filter, doute actif) et
    STNO_FROZEN_U (u fige a la baseline -> couplage quasi plein constant).
    Les deux copies de la paire partagent la dynamique du substrat.

REGLES D'ARRET comparees (chaque regle choisit UN hyperparametre GLOBAL sur tout
le melange de T_pulse -- l'horizon est inconnu, personne ne se regle par probleme) :
  - DOUBT_PAIR : (revision 3, l'horloge adaptee au substrat) arret quand le
                 DESACCORD D'EVIDENCE entre les deux bras de la paire,
                 mean_i |lisse_W(p+_i - p-_i)|, retombe sous frac *
                 pic-roulant-causal. Nul avant le pulse (copies identiques),
                 gros pendant le conflit (leurres + verites contrastes),
                 retombe vers la seule composante verite quand le leurre
                 s'efface -> c'est l'analogue du |Lv| de B5b transpose a la
                 lecture differentielle, purement mesurable (2 diodes/noeud),
                 aucune connaissance des masques. frac balaye {0.2,0.3,0.5}.
  - DOUBT_P    : arret quand mean|L p| (desaccord de PUISSANCE local) retombe
                 sous frac * pic-roulant-causal. CONTROLE DOCUMENTE (lancement
                 3) : ne retombe que de ~12%% -- le fond de desaccord permanent
                 (bruit, heterogeneite) domine le conflit sur ce substrat.
  - DOUBT_S    : meme regle sur mean|S| (desaccord de couplage complexe brut).
                 CONTROLE DOCUMENTE : AVEUGLE (domine par le desaccord de PHASE
                 permanent, omega heterogenes ; lancement 1 : 0.1319->0.1318).
  - DOUBT_U    : meme regle sur (u_mean - baseline) -- l'etat interne du doute,
                 pilote par |S| donc aveugle lui aussi (documente).
  - CONV       : la variable de decision LISSEE a cesse de bouger :
                 |d(t)-d(t-W)| < thr, W=200 pas, thr balaye {0.0002,0.0005,0.001}.
  - FIXED      : budget fixe B, balaye {600..9000 par 600}, MEILLEUR GLOBAL --
                 l'adversaire le plus dur (lecon B5b).
  - acc_fin    : reponse au budget max (plafond de reference).

REVISIONS (12/07, calibration de STRUCTURE, pas de p-hacking : on repare la
loyaute de la tache AVANT de re-comparer les regles ; le critere de loyaute est
acc_fin, pas l'ecart entre regles) :
  Lancement 1 (readout net vs ref-sans-stim, lecture instantanee) :
  - lecture instantanee noyee par la decorrelation net/ref (|d|~0.007 vs signal
    ~0.013) -> lecture LISSEE W_READ=400 (l'integration d'une diode reelle).
  - ISCALE 0.15 -> 0.25 (le couplage diffusif attenue le signal ~x0.6) ;
    25%% de modulation de courant reste raisonnable (Torrejon 2017).
  - MAX_BUDGET 4000 -> 6000 (30 u.t.).
  Lancement 2 (pre-vol durci 0/4) + DIAGNOSTIC (decomposition par groupe) :
  - FAIT PHYSIQUE 1 : le reseau de reference n'atteint JAMAIS l'equilibre en
    6000 pas (p_ref 0.005->0.09, en route vers p_eff << p*=0.2) -- le couplage
    entre oscillateurs desaccordes (sigma_omega=0.15, phases non verrouillees a
    K=0.3) agit comme une dissipation ~K*u_filter~0.27 comparable au gain net
    0.2 : le reseau COUPLE vit sous le seuil effectif, dynamique molle et lente.
  - FAIT PHYSIQUE 2 (cicatrice de doute) : en FULL, le conflit fait monter u_net
    durablement au-dessus de u_ref (0.42 vs 0.26 a t=6000) -> u_filter plus bas
    -> couplage (dissipation) reduit -> p_net durablement PLUS HAUT que p_ref,
    independamment du signe de l'evidence. Une lecture differentielle contre un
    ref sans stimulus confond 'doute monte' avec 'evidence positive'.
  - FAIT PHYSIQUE 3 : en K=0 (decouple), la tache marche parfaitement (leurre
    efface, verite tenue) -- la contamination vient du couplage, pas du piege.
  -> REVISION 2 : readout par PAIRE DIFFERENTIELLE (standard en electronique) :
    copie + recoit stim, copie - recoit -stim, meme seed/bruit. Le transitoire
    commun, la cicatrice u (identique au 1er ordre par symetrie |stim|) et la
    rectification de seuil s'annulent dans d = mean(p+) - mean(p-). Aucune
    connaissance des masques n'est requise (lecture loyale). En prime :
    initialisation aux amplitudes de fonctionnement (|a|=sqrt(p*), phases
    aleatoires) + warmup commun de 1000 pas AVANT la stimulation (etat copie
    aux deux copies) -- un oscillateur eteint n'amplifie rien.
  - Pre-vol durci : 4 problemes (2 seeds x 2 T_pulse), exige acc_fin >= 3/4,
    sinon campagne annulee (aucun resultat partiel ecrit).
  Lancement 3 (paire differentielle, budget 6000) -- 3 faits :
  - la tache est LOYALE sur FROZEN (100%% bascule partout, acc_fin=1.00) ;
  - DECOUVERTE : le doute-dans-la-dynamique RALENTIT le flip (FULL 4533 vs
    FROZEN 3218 pas) et le tue a T_pulse=4000 (0%% vs 100%%) -- la cicatrice u
    (couplage coupe par le conflit) VERROUILLE la trace du leurre ; mais a
    budget 6000 le flip FULL est censure (6001 = jamais) -> budget 9000 pour
    mesurer le vrai retard au lieu d'une censure ;
  - aucune horloge interne ne retombe (sigP -12%% seulement) : DOUBT_P/S/U ne
    battaient CONV qu'en ne s'arretant JAMAIS (= budget max deguise, pattern
    B5b NON replique) -> ajout de DOUBT_PAIR (desaccord d'evidence entre bras).

CRITERES PRE-FIXES (avant de voir un seul chiffre de comparaison entre regles) :
  - Niche confirmee si DOUBT_P bat CONV en accuracy globale, IC bootstrap
    apparie ne couvrant pas 0, dans le regime trompeur (flips tardifs).
  - Prediction honnete (pattern B1d/B5b, 3 replications) : DOUBT_P > CONV ;
    DOUBT_P ~ FIXED-global (egalite attendue, la niche = horizon inconnu) ;
    FULL ~ FROZEN sur l'accuracy (la modulation du couplage joue peu sur une
    lecture differentielle moyenne). Si FULL > FROZEN : element NOUVEAU pour B6.
  - Un resultat negatif (l'horloge du doute ne marche pas sur un substrat
    oscillant) est une information, pas un echec du script.

Statut : exploratoire, hors preprint, aucun claim modifie. Coeur non touche.
Sorties : figures/b1d_stno_deceptive_poc.csv / _agg.csv / .png
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from mem4ristor.graph_utils import make_lattice_adj  # noqa: E402

CSV_PATH = ROOT / "figures" / "b1d_stno_deceptive_poc.csv"
AGG_PATH = ROOT / "figures" / "b1d_stno_deceptive_poc_agg.csv"
PNG_PATH = ROOT / "figures" / "b1d_stno_deceptive_poc.png"

# ---------------- piege B1d (identique a deceptive_task_poc.py) ----------------
SIDE, N = 10, 100
N_DISTRACT = 26
N_TRUE = 14
E_TRUE = 0.6
E_DISTRACT = 1.0
SEEDS = list(range(12))

# ---------------- echelles de temps STNO (dt=0.005, tau_p ~ 2.5 u.t. = 500 pas) --
DT = 0.005
MAX_BUDGET = 9000                        # 45 unites de temps (revise lancement 3 :
                                         # a 6000 le flip FULL etait censure)
T_PULSE_LEVELS = [500, 1500, 3000, 4500]  # 2.5 / 7.5 / 15 / 22.5 unites de temps
ISCALE = 0.25                            # modulation max de gamma_plus (E=1.0),
                                         # revise 0.15 -> 0.25 (attenuation diffusive)
W_READ = 400                             # fenetre de lecture lissee (diode, 2 u.t.)
W_SIG_LONG = 2000                        # fenetre longue pour DOUBT_PAIRL (10 u.t.) :
                                         # la decorrelation chaotique entre bras a un
                                         # tau ~ 500 pas, la fenetre courte ne la
                                         # moyenne pas (lancement 4 : fond ~3x verite)

# ---------------- substrat Slavin-Tiberkevich (constantes du 09-11/07) -----------
GAMMA_MINUS = 1.0
GAMMA_PLUS = 1.2                         # p* isole = 0.2
Q = 1.0
OMEGA0 = 1.0
SIGMA_OMEGA = 0.15
K_COUPLING = 0.3
SIGMA_NOISE = 0.02

# ---------------- doute : IDENTIQUE a dynamics.py ------------------------------
EPSILON_U = 0.02
K_U = 1.0
SIGMA_BASELINE = 0.05
TAU_U = 10.0
ALPHA_SURPRISE = 2.0
SURPRISE_CAP = 5.0
SOCIAL_LEAKAGE = 0.01
GAIN_U = 10.0                            # capteur calibre (POC 09/07)

# ---------------- regles d'arret (hyperparametres balayes, choix GLOBAL) --------
WARMUP = 200                             # 1 unite de temps
DOUBT_FRACS = [0.2, 0.3, 0.5]
CONV_W = 200
CONV_THRS = [0.0002, 0.0005, 0.001]      # echelonnes au bruit de la lecture lissee
FIXED_BUDGETS = list(range(600, MAX_BUDGET + 1, 600))
CONDITIONS = ["STNO_FULL", "STNO_FROZEN_U"]


def make_deceptive(rng: np.random.RandomState):
    """Masques de stimulation B1d : leurre pulse (nombreux, fort, signe -dstar),
    verite persistante (moins nombreuse, plus faible, signe +dstar)."""
    dstar = int(rng.choice([-1, 1]))
    nodes = rng.choice(N, size=N_DISTRACT + N_TRUE, replace=False)
    d_nodes, t_nodes = nodes[:N_DISTRACT], nodes[N_DISTRACT:]
    stim_on = np.zeros(N)
    stim_on[d_nodes] = -dstar * E_DISTRACT
    stim_on[t_nodes] = +dstar * E_TRUE
    stim_off = np.zeros(N)
    stim_off[t_nodes] = +dstar * E_TRUE
    return stim_on, stim_off, dstar


def _step_one(a, u, gp, omega, adj, deg, eta, frozen):
    """Un pas d'Euler pour une copie. Retourne (a', u', |S|, p)."""
    S = (adj @ a) / deg - a
    abs_s = np.abs(S)
    sigma_for_u = np.zeros_like(u) if frozen else abs_s * GAIN_U
    u_filter = np.tanh(np.pi * (0.5 - u)) + SOCIAL_LEAKAGE
    p = np.abs(a) ** 2
    growth = gp - GAMMA_MINUS * (1.0 + Q * p)
    da = (growth + 1j * omega) * a + K_COUPLING * u_filter * S + eta
    sigma_safe = np.clip(sigma_for_u, 0.0, 100.0)
    eps_adapt = EPSILON_U * np.clip(1.0 + ALPHA_SURPRISE * sigma_safe, 1.0, SURPRISE_CAP)
    du = eps_adapt * (K_U * sigma_for_u + SIGMA_BASELINE - u) / TAU_U
    a2 = a + da * DT
    u2 = np.clip(u + du * DT, 0.0, 1.0)
    return a2, u2, abs_s, p


WARMUP_STEPS = 1000                      # equilibrage commun avant stimulation


def simulate(adj, deg, stim_on, stim_off, seed: int, t_pulse: int, condition: str):
    """PAIRE DIFFERENTIELLE : warmup commun (sans stim), puis copie + (recoit
    +stim) et copie - (recoit -stim), meme bruit partage. Retourne sig_p
    (mean|L p|, desaccord de puissance, moyenne des 2 copies), sig_s (mean|S|,
    controle), u_mean, d_var (mean(p+)-mean(p-) LISSEE W_READ), dec (signe),
    ou None si divergence."""
    rng = np.random.default_rng(seed)
    frozen = condition == "STNO_FROZEN_U"
    omega = OMEGA0 + rng.normal(0, SIGMA_OMEGA, N)
    phases = rng.uniform(0.0, 2.0 * np.pi, N)
    p_star = (GAMMA_PLUS - GAMMA_MINUS) / (GAMMA_MINUS * Q)
    a = np.sqrt(p_star) * np.exp(1j * phases)      # amplitudes de fonctionnement
    u = np.full(N, SIGMA_BASELINE)
    inv_sqrt_dt = 1.0 / np.sqrt(DT)

    # ---- warmup commun : le reseau couple relaxe vers son etat de marche ----
    for _ in range(WARMUP_STEPS):
        noise = rng.normal(0.0, SIGMA_NOISE, size=(2, N))
        eta = (noise[0] + 1j * noise[1]) * inv_sqrt_dt
        a, u, _, _ = _step_one(a, u, GAMMA_PLUS, omega, adj, deg, eta, frozen)
    if not np.all(np.isfinite(a)):
        return None

    a_pos, a_neg = a.copy(), a.copy()
    u_pos, u_neg = u.copy(), u.copy()

    sig_p = np.empty(MAX_BUDGET)
    sig_s = np.empty(MAX_BUDGET)
    u_tr = np.empty(MAX_BUDGET)
    dmat = np.empty((MAX_BUDGET, N))     # p+_i - p-_i par noeud (pour DOUBT_PAIR)

    for t in range(MAX_BUDGET):
        stim = stim_on if t < t_pulse else stim_off
        noise = rng.normal(0.0, SIGMA_NOISE, size=(2, N))
        eta = (noise[0] + 1j * noise[1]) * inv_sqrt_dt

        gp_pos = GAMMA_PLUS + ISCALE * stim
        gp_neg = GAMMA_PLUS - ISCALE * stim
        a_pos, u_pos, s_pos, p_pos = _step_one(a_pos, u_pos, gp_pos, omega,
                                               adj, deg, eta, frozen)
        a_neg, u_neg, s_neg, p_neg = _step_one(a_neg, u_neg, gp_neg, omega,
                                               adj, deg, eta, frozen)

        lp_pos = np.abs((adj @ p_pos) / deg - p_pos)
        lp_neg = np.abs((adj @ p_neg) / deg - p_neg)
        sig_p[t] = 0.5 * float(lp_pos.mean() + lp_neg.mean())
        sig_s[t] = 0.5 * float(s_pos.mean() + s_neg.mean())
        u_tr[t] = 0.5 * float(u_pos.mean() + u_neg.mean())

        if not (np.all(np.isfinite(a_pos)) and np.all(np.isfinite(a_neg))):
            return None
        dmat[t] = np.abs(a_pos) ** 2 - np.abs(a_neg) ** 2

    # lissage rolling par noeud (vectorise), CAUSAL, deux fenetres
    csum2 = np.cumsum(dmat, axis=0)

    def _roll(w):
        out = np.empty_like(dmat)
        for t in range(MAX_BUDGET):
            lo = max(0, t - w + 1)
            tot = csum2[t] - (csum2[lo - 1] if lo > 0 else 0.0)
            out[t] = tot / (t - lo + 1)
        return out

    dsm = _roll(W_READ)
    d_var = dsm.mean(axis=1)             # lecture globale lissee (decision)
    sig_pair = np.abs(dsm).mean(axis=1)  # desaccord d'evidence entre les bras
    sig_pair_l = np.abs(_roll(W_SIG_LONG)).mean(axis=1)   # variante fenetre longue
    dec = np.where(d_var >= 0, 1, -1).astype(int)
    return sig_p, sig_s, sig_pair, sig_pair_l, u_tr, d_var, dec


def stop_doubt_rolling(signal: np.ndarray, frac: float) -> int:
    """Arret quand le signal retombe sous frac * pic-roulant-causal (depuis WARMUP)."""
    peak = float(np.max(signal[:WARMUP])) if WARMUP > 0 else 0.0
    for t in range(WARMUP, len(signal)):
        peak = max(peak, float(signal[t]))
        if peak > 0 and signal[t] < frac * peak:
            return t + 1
    return len(signal)


def stop_conv(d_var: np.ndarray, thr: float) -> int:
    for t in range(WARMUP + CONV_W, len(d_var)):
        if abs(d_var[t] - d_var[t - CONV_W]) < thr:
            return t + 1
    return len(d_var)


def flip_time(dec: np.ndarray, dstar: int) -> int:
    correct = dec == dstar
    for t in range(len(dec)):
        if np.all(correct[t:]):
            return t + 1
    return MAX_BUDGET + 1


def dec_at(dec: np.ndarray, t: int) -> int:
    return int(dec[min(int(t), MAX_BUDGET) - 1])


def boot_ci_paired(a, b, n_boot=10000, seed=20260712):
    rng = np.random.RandomState(seed)
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    m = np.empty(n_boot)
    for k in range(n_boot):
        m[k] = d[rng.randint(0, n, n)].mean()
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> int:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    adj = make_lattice_adj(SIDE, periodic=True).astype(float)
    deg = adj.sum(axis=1)
    assert np.all(deg > 0)

    # ------- pre-vol durci : stabilite + LOYAUTE de la tache (acc_fin >= 3/4) -------
    print(f"[pre-vol] 4 problemes (seeds 0,1 x T_pulse 1500,2500), STNO_FULL : "
          f"stabilite dt={DT} + la verite doit gagner au budget max (>= 3/4)...")
    ok_fin = 0
    for s_pre in (0, 1):
        for tp_pre in (1500, 2500):
            rng_pre = np.random.RandomState(3000 + s_pre)
            stim_on, stim_off, dstar = make_deceptive(rng_pre)
            out = simulate(adj, deg, stim_on, stim_off, s_pre * 10 + 1, tp_pre, "STNO_FULL")
            if out is None:
                print("[pre-vol] DIVERGENCE -> dt trop grand. Campagne annulee.")
                return 1
            sg_p, sg_s, sg_pair, sg_pair_l, u_p, d_p, dec_p = out
            ft = flip_time(dec_p, dstar)
            fin_ok = int(dec_p[-1] == dstar)
            ok_fin += fin_ok
            print(f"  seed={s_pre} T_pulse={tp_pre}: dstar={dstar:+d} flip={ft} "
                  f"fin_ok={fin_ok} | sigPAIR pic={sg_pair.max():.5f} "
                  f"fin={sg_pair[-1]:.5f} | sigPAIRL pic={sg_pair_l.max():.5f} "
                  f"fin={sg_pair_l[-1]:.5f} | |d|fin={abs(d_p[-1]):.5f}")
    if ok_fin < 3:
        print(f"[pre-vol] TACHE NON LOYALE ({ok_fin}/4 verites gagnantes au budget max) "
              f"-> campagne annulee, recalibrer la structure (pas les regles).")
        return 1
    print(f"[pre-vol] OK ({ok_fin}/4). Campagne lancee.")

    # ---------------- campagne ----------------
    rows = []
    # store[cond][(t_pulse, seed)] = dict(flip, acc_fin, stops par regle/param)
    store: dict[str, dict] = {c: {} for c in CONDITIONS}
    diverged = 0
    total = len(CONDITIONS) * len(T_PULSE_LEVELS) * len(SEEDS)
    done = 0
    for cond in CONDITIONS:
        for t_pulse in T_PULSE_LEVELS:
            for seed in SEEDS:
                rng = np.random.RandomState(3000 + seed)
                stim_on, stim_off, dstar = make_deceptive(rng)
                out = simulate(adj, deg, stim_on, stim_off, seed * 10 + 1, t_pulse, cond)
                done += 1
                if out is None:
                    diverged += 1
                    continue
                sig_p, sig_s, sig_pair, sig_pair_l, u_tr, d_var, dec = out
                ft = flip_time(dec, dstar)
                acc_fin = int(dec[-1] == dstar)
                entry = {"dstar": dstar, "flip": ft, "acc_fin": acc_fin, "rules": {}}
                for frac in DOUBT_FRACS:
                    st_pr = stop_doubt_rolling(sig_pair, frac)
                    entry["rules"][("DOUBT_PAIR", frac)] = (st_pr, int(dec_at(dec, st_pr) == dstar))
                    st_pl = stop_doubt_rolling(sig_pair_l, frac)
                    entry["rules"][("DOUBT_PAIRL", frac)] = (st_pl, int(dec_at(dec, st_pl) == dstar))
                    st = stop_doubt_rolling(sig_p, frac)
                    entry["rules"][("DOUBT_P", frac)] = (st, int(dec_at(dec, st) == dstar))
                    st_s = stop_doubt_rolling(sig_s, frac)
                    entry["rules"][("DOUBT_S", frac)] = (st_s, int(dec_at(dec, st_s) == dstar))
                    st_u = stop_doubt_rolling(u_tr - SIGMA_BASELINE, frac)
                    entry["rules"][("DOUBT_U", frac)] = (st_u, int(dec_at(dec, st_u) == dstar))
                for thr in CONV_THRS:
                    st = stop_conv(d_var, thr)
                    entry["rules"][("CONV", thr)] = (st, int(dec_at(dec, st) == dstar))
                for b in FIXED_BUDGETS:
                    entry["rules"][("FIXED", b)] = (b, int(dec_at(dec, b) == dstar))
                store[cond][(t_pulse, seed)] = entry
                for (rule, param), (st, acc) in entry["rules"].items():
                    rows.append((cond, t_pulse, seed, dstar, ft, acc_fin,
                                 rule, param, st, acc))
                if done % 12 == 0:
                    print(f"  [{done}/{total}] {cond} T_pulse={t_pulse} "
                          f"({time.time()-t0:.0f}s)")

    if diverged:
        print(f"\n[garde] {diverged} run(s) divergents exclus.")

    # ---------------- choix GLOBAL des hyperparametres par regle ----------------
    print("\n=== CHOIX GLOBAL (accuracy moyenne sur tout le melange T_pulse) ===")
    best_param: dict[tuple, object] = {}
    for cond in CONDITIONS:
        problems = sorted(store[cond].keys())
        for rule, params in [("DOUBT_PAIR", DOUBT_FRACS), ("DOUBT_PAIRL", DOUBT_FRACS),
                             ("DOUBT_P", DOUBT_FRACS),
                             ("DOUBT_S", DOUBT_FRACS), ("DOUBT_U", DOUBT_FRACS),
                             ("CONV", CONV_THRS), ("FIXED", FIXED_BUDGETS)]:
            scored = []
            for p in params:
                accs = [store[cond][k]["rules"][(rule, p)][1] for k in problems]
                costs = [store[cond][k]["rules"][(rule, p)][0] for k in problems]
                scored.append((np.mean(accs), -np.mean(costs), p))
            scored.sort(reverse=True)
            best_param[(cond, rule)] = scored[0][2]
            print(f"  {cond:<14} {rule:<10} -> param={scored[0][2]} "
                  f"(acc={scored[0][0]:.3f}, cout moyen={-scored[0][1]:.0f})")

    # ---------------- tableau par T_pulse ----------------
    RULES_ALL = ["DOUBT_PAIR", "DOUBT_PAIRL", "DOUBT_P", "DOUBT_S", "DOUBT_U",
                 "CONV", "FIXED"]
    print(f"\n{'cond':<14}{'T_pulse':>8}{'%basc':>7}{'flip':>6} | "
          f"{'DOUBT_PAIR':>13}{'DOUBT_PAIRL':>12}{'CONV':>11}{'FIXED':>7}{'FIN':>6}")
    print("-" * 94)
    agg_rows = []
    for cond in CONDITIONS:
        for t_pulse in T_PULSE_LEVELS:
            keys = [k for k in store[cond] if k[0] == t_pulse]
            flips = [store[cond][k]["flip"] for k in keys]
            pct_flip = 100.0 * np.mean([f <= MAX_BUDGET for f in flips])
            cells = {}
            for rule in RULES_ALL:
                p = best_param[(cond, rule)]
                accs = [store[cond][k]["rules"][(rule, p)][1] for k in keys]
                costs = [store[cond][k]["rules"][(rule, p)][0] for k in keys]
                cells[rule] = (np.mean(accs), np.mean(costs))
            fin = np.mean([store[cond][k]["acc_fin"] for k in keys])
            print(f"{cond:<14}{t_pulse:>8}{pct_flip:>6.0f}%{np.mean(flips):>6.0f} | "
                  f"{cells['DOUBT_PAIR'][0]:>7.2f}({cells['DOUBT_PAIR'][1]:>4.0f})"
                  f"{cells['DOUBT_PAIRL'][0]:>6.2f}({cells['DOUBT_PAIRL'][1]:>4.0f})"
                  f"{cells['CONV'][0]:>5.2f}({cells['CONV'][1]:>4.0f})"
                  f"{cells['FIXED'][0]:>7.2f}"
                  f"{fin:>6.2f}")
            for rule in RULES_ALL:
                agg_rows.append((cond, t_pulse, rule, best_param[(cond, rule)],
                                 cells[rule][0], cells[rule][1], fin, pct_flip))

    # ---------------- verdicts (IC bootstrap apparies, tous problemes) ----------
    print("\n=== VERDICT P12 (honnete, criteres pre-fixes) ===")
    verdicts = []
    for cond in CONDITIONS:
        problems = sorted(store[cond].keys())

        def acc_vec(rule):
            p = best_param[(cond, rule)]
            return [store[cond][k]["rules"][(rule, p)][1] for k in problems]

        def cost_vec(rule):
            p = best_param[(cond, rule)]
            return [store[cond][k]["rules"][(rule, p)][0] for k in problems]

        a_draw, a_conv, a_fix = acc_vec("DOUBT_PAIR"), acc_vec("CONV"), acc_vec("FIXED")
        d1, lo1, hi1 = boot_ci_paired(a_draw, a_conv)
        v1 = ("DOUTE bat CONV" if lo1 > 0 else
              ("CONV bat DOUTE" if hi1 < 0 else "parite (IC couvre 0)"))
        print(f"  [{cond}] 1. DOUBT_PAIR - CONV  = {d1:+.3f} CI[{lo1:+.3f},{hi1:+.3f}] -> {v1}")
        d2, lo2, hi2 = boot_ci_paired(a_draw, a_fix)
        v2 = ("DOUTE bat le meilleur budget fixe global" if lo2 > 0 else
              ("le budget fixe bat le DOUTE" if hi2 < 0 else "parite (IC couvre 0)"))
        print(f"  [{cond}] 2. DOUBT_PAIR - FIXED = {d2:+.3f} CI[{lo2:+.3f},{hi2:+.3f}] -> {v2}")
        d2b, lo2b, hi2b = boot_ci_paired(acc_vec("DOUBT_PAIRL"), a_fix)
        v2b = ("l'horloge a fenetre longue bat le meilleur budget fixe" if lo2b > 0 else
               ("le budget fixe bat aussi la fenetre longue" if hi2b < 0
                else "parite (IC couvre 0)"))
        print(f"  [{cond}] 2b. DOUBT_PAIRL - FIXED = {d2b:+.3f} CI[{lo2b:+.3f},{hi2b:+.3f}] -> {v2b}")
        print(f"  [{cond}]    couts moyens : DOUBT_PAIR={np.mean(cost_vec('DOUBT_PAIR')):.0f}, "
              f"DOUBT_PAIRL={np.mean(cost_vec('DOUBT_PAIRL')):.0f}, "
              f"DOUBT_P={np.mean(cost_vec('DOUBT_P')):.0f}, CONV={np.mean(cost_vec('CONV')):.0f}, "
              f"FIXED={np.mean(cost_vec('FIXED')):.0f} pas (budget max {MAX_BUDGET})")
        verdicts.append((cond, d1, lo1, hi1, d2, lo2, hi2))

    # B6 : FULL vs FROZEN a l'arret doute (problemes apparies)
    pf = sorted(set(store["STNO_FULL"].keys()) & set(store["STNO_FROZEN_U"].keys()))
    afull = [store["STNO_FULL"][k]["rules"][("DOUBT_PAIR",
             best_param[("STNO_FULL", "DOUBT_PAIR")])][1] for k in pf]
    afroz = [store["STNO_FROZEN_U"][k]["rules"][("DOUBT_PAIR",
             best_param[("STNO_FROZEN_U", "DOUBT_PAIR")])][1] for k in pf]
    d3, lo3, hi3 = boot_ci_paired(afull, afroz)
    v3 = ("le couplage module par le doute AIDE la decision (element nouveau B6)"
          if lo3 > 0 else
          ("le couplage fige fait MIEUX" if hi3 < 0 else
           "FULL ~ FROZEN (prediction tenue : la niche est dans l'HORLOGE, pas le couplage)"))
    print(f"  [B6] 3. FULL - FROZEN (arret doute) = {d3:+.3f} CI[{lo3:+.3f},{hi3:+.3f}]")
    print(f"       -> {v3}")
    ffull = np.mean([store["STNO_FULL"][k]["flip"] for k in pf])
    ffroz = np.mean([store["STNO_FROZEN_U"][k]["flip"] for k in pf])
    print(f"       flip moyen : FULL={ffull:.0f}, FROZEN={ffroz:.0f} pas")

    # ---------------- CSV ----------------
    with CSV_PATH.open("w", encoding="utf-8") as f:
        f.write("condition,t_pulse,seed,dstar,flip_time,acc_final,rule,param,stop,acc\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    with AGG_PATH.open("w", encoding="utf-8") as f:
        f.write("condition,t_pulse,rule,best_param_global,acc,cost_mean,acc_final,pct_flip\n")
        for r in agg_rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print(f"\n[csv] {CSV_PATH}\n[csv] {AGG_PATH}")

    # ---------------- figure ----------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
        colors = {"DOUBT_PAIR": "#8c564b", "DOUBT_PAIRL": "#d62728", "DOUBT_P": "#ff7f0e",
                  "DOUBT_S": "#e377c2", "DOUBT_U": "#ff9896",
                  "CONV": "#1f77b4", "FIXED": "#7f7f7f"}
        for ax, cond in zip(axes, CONDITIONS):
            for rule in RULES_ALL:
                p = best_param[(cond, rule)]
                ys = []
                for t_pulse in T_PULSE_LEVELS:
                    keys = [k for k in store[cond] if k[0] == t_pulse]
                    ys.append(np.mean([store[cond][k]["rules"][(rule, p)][1]
                                       for k in keys]))
                ax.plot(T_PULSE_LEVELS, ys, "o-", color=colors[rule],
                        label=f"{rule} ({p})")
            fins = [np.mean([store[cond][k]["acc_fin"] for k in store[cond]
                             if k[0] == tp]) for tp in T_PULSE_LEVELS]
            ax.plot(T_PULSE_LEVELS, fins, "^--", c="#2ca02c", label="budget max")
            ax.set_xlabel("Duree du leurre T_pulse (pas)")
            ax.set_ylabel("Taux de bonne reponse")
            ax.set_title(cond)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        fig.suptitle("P12 : la tache trompeuse B1d sur substrat STNO "
                     "(hyperparametres globaux, horizon inconnu)", fontsize=11)
        plt.tight_layout()
        plt.savefig(PNG_PATH, dpi=140)
        print(f"[png] {PNG_PATH}")
    except Exception as e:
        print(f"[png] skipped: {e}")

    print(f"\nWall time: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
