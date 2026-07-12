#!/usr/bin/env python3
"""
P3 (legs de Fable) -- Le detecteur d'anomalies gratuit : u comme alarme d'entree corrompue.
============================================================================================
Cree : 2026-07-12 (Claude Fable 5, L'Ingenieur) -- piste P3 de
docs/PISTES_POUR_LA_SUITE_2026-07-12.md.

CONTEXTE. Section 'Resilience' de MEM4_MOE_CONCEPT.md (02/02/2026, ancien
dossier) : 'si le Sage est dupe (adversarial), il ne doute pas ; le Fou
sentira que quelque chose cloche'. Jamais teste. La promesse : u (le capteur
de desaccord, deja present dans la dynamique) detecte une entree CORROMPUE
sans entrainement supplementaire -- detection 'gratuite' si le reseau tourne.

PROTOCOLE.
  - Entrees NORMALES : motifs spatialement lisses (blob gaussien 2D, position
    aleatoire, amplitude 0.5) sur la grille 10x10, injectes en courant.
  - Entrees CORROMPUES : le MEME blob, mais signe inverse sur un damier
    aleatoire de ses pixels actifs -- les amplitudes MARGINALES par noeud sont
    inchangees, seule la COHERENCE SPATIALE est cassee (motifs contradictoires
    entre voisins : la definition meme de ce que le desaccord doit sentir).
  - Controle NOUVEAU-MAIS-PROPRE (exige par la piste : u peut reagir au simple
    volume de nouveaute) : blob jamais vu (sigma spatial 3.0 au lieu de 1.8,
    amplitude 0.7 au lieu de 0.5, position aleatoire) mais LISSE. Un bon
    detecteur d'anomalies structurelles ne doit PAS s'alarmer.
  - Episodes independants (reseau frais, seeds distincts), T=800 pas chacun,
    40 normaux + 40 corrompus + 20 nouveaux-propres (+ 40 normaux
    d'entrainement pour la baseline z-score, jamais reutilises au test).
  - SCORES d'anomalie compares (AUC normal-vs-corrompu, Mann-Whitney) :
      u_final    = mean(u) en fin d'episode        (l'etat de doute du coeur)
      sig_mean   = moyenne temporelle de mean|Lv|  (le capteur brut, lecon P6 :
                                                    plus rapide que u)
      z_score    = mean_i |x_i - mu_i|/sd_i        (baseline naive, entrainee
                                                    sur les 40 normaux du train)
      lap_input  = mean|L @ x|                     (BASELINE FORTE : le filtre
                    laplacien STATIQUE de l'entree -- si elle gagne, le reseau
                    n'ajoute rien qu'un filtre fixe n'ait deja ; construire
                    l'adversaire qui peut tuer le resultat, lecon du 08/07)
  - Et l'epreuve de NOUVEAUTE : AUC nouveau-propre-vs-corrompu (un detecteur
    structurel doit encore separer ; un detecteur de nouveaute confond).

CRITERES PRE-FIXES (avant de voir un chiffre) :
  - 'Detection gratuite' validee pour un signal reseau si AUC(normal vs
    corrompu) > 0.8 ET AUC(nouveau-propre vs corrompu) > 0.7 (il ne confond
    pas nouveaute et corruption).
  - Comparaison honnete : si AUC(lap_input) >= AUC(signaux reseau), le dire
    tel quel -- la valeur restante du reseau n'est alors que 'gratuit si le
    reseau tourne deja' (variable d'etat physique lisible), pas une capacite.
  - Prediction ecrite avant : lap_input sera fort (la corruption est haute
    frequence spatiale par construction) ; z_score sera aveugle a la
    corruption (amplitudes marginales inchangees) et s'alarmera a tort sur le
    nouveau-propre (amplitude 0.7 > train). La question ouverte est de savoir
    si u/sig atteignent le niveau de lap_input.

Statut : exploratoire, hors preprint, coeur non touche.
Sorties : figures/p3_anomaly_detector_poc{,_agg}.csv + .png
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
from mem4ristor.topology import Mem4Network  # noqa: E402
from mem4ristor.graph_utils import make_lattice_adj  # noqa: E402

CSV_PATH = ROOT / "figures" / "p3_anomaly_detector_poc.csv"
AGG_PATH = ROOT / "figures" / "p3_anomaly_detector_poc_agg.csv"
PNG_PATH = ROOT / "figures" / "p3_anomaly_detector_poc.png"

SIDE, N = 10, 100
T_EP = 800
N_TRAIN = 40
N_NORMAL = 40
N_CORRUPT = 40
N_NOVEL = 20
AMP_NORMAL, SIG_NORMAL = 0.5, 1.8
AMP_NOVEL, SIG_NOVEL = 0.7, 3.0
ACTIVE_FRAC = 0.1              # seuil de pixel actif (fraction du max du blob)


def gaussian_blob(rng, amp, sigma):
    cx, cy = rng.uniform(0, SIDE, 2)
    xs, ys = np.meshgrid(np.arange(SIDE), np.arange(SIDE), indexing="ij")
    # distance torique (grille periodique)
    dx = np.minimum(np.abs(xs - cx), SIDE - np.abs(xs - cx))
    dy = np.minimum(np.abs(ys - cy), SIDE - np.abs(ys - cy))
    blob = amp * np.exp(-(dx ** 2 + dy ** 2) / (2 * sigma ** 2))
    return blob.flatten()


def make_input(kind, rng):
    if kind == "novel_clean":
        return gaussian_blob(rng, AMP_NOVEL, SIG_NOVEL)
    x = gaussian_blob(rng, AMP_NORMAL, SIG_NORMAL)
    if kind == "corrupt":
        active = np.abs(x) > ACTIVE_FRAC * np.abs(x).max()
        flips = rng.rand(N) < 0.5
        signs = np.where(active & flips, -1.0, 1.0)
        x = x * signs
    return x


def run_episode(x, seed, adj):
    net = Mem4Network(size=SIDE, heretic_ratio=0.0, seed=seed, adjacency_matrix=adj)
    L = net.L
    sig_acc = 0.0
    for _ in range(T_EP):
        net.step(I_stimulus=x)
        sig_acc += float(np.mean(np.abs(L @ net.model.v)))
    return float(np.mean(net.model.u)), sig_acc / T_EP


def auc_mw(pos, neg):
    """AUC = P(score_pos > score_neg) (Mann-Whitney, ties=0.5)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    wins = 0.0
    for p in pos:
        wins += float(np.sum(p > neg)) + 0.5 * float(np.sum(p == neg))
    return wins / (len(pos) * len(neg))


def main() -> int:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    adj = make_lattice_adj(SIDE, periodic=True).astype(float)
    deg = adj.sum(axis=1)
    L_in = np.diag(deg) - adj

    # ---------------- jeu d'entrainement (baseline z-score seulement) --------
    rng_train = np.random.RandomState(52000)
    train = np.array([make_input("normal", rng_train) for _ in range(N_TRAIN)])
    mu, sd = train.mean(axis=0), train.std(axis=0)
    sd[sd < 1e-9] = 1e-9

    # ---------------- episodes de test ----------------
    episodes = ([("normal", k) for k in range(N_NORMAL)]
                + [("corrupt", k) for k in range(N_CORRUPT)]
                + [("novel_clean", k) for k in range(N_NOVEL)])
    rows = []
    print(f"P3 -- {len(episodes)} episodes (40 normaux / 40 corrompus / "
          f"20 nouveaux-propres), T={T_EP} pas chacun")
    for i, (kind, k) in enumerate(episodes):
        rng = np.random.RandomState(53000 + i)
        x = make_input(kind, rng)
        u_fin, sig_mean = run_episode(x, seed=61000 + i, adj=adj)
        z = float(np.mean(np.abs(x - mu) / sd))
        lap = float(np.mean(np.abs(L_in @ x)))
        rows.append({"kind": kind, "idx": k, "u_final": u_fin,
                     "sig_mean": sig_mean, "z_score": z, "lap_input": lap})
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(episodes)}] {time.time()-t0:.0f}s")

    # ---------------- analyses ----------------
    def scores(kind, name):
        return [r[name] for r in rows if r["kind"] == kind]

    SIGNALS = ["u_final", "sig_mean", "z_score", "lap_input"]
    print(f"\n{'signal':<11}{'AUC n-vs-c':>12}{'AUC novel-vs-c':>16}"
          f"{'moy normal':>12}{'moy corrompu':>14}{'moy nouveau':>13}")
    print("-" * 80)
    agg = []
    for name in SIGNALS:
        a_nc = auc_mw(scores("corrupt", name), scores("normal", name))
        a_vc = auc_mw(scores("corrupt", name), scores("novel_clean", name))
        m_n = np.mean(scores("normal", name))
        m_c = np.mean(scores("corrupt", name))
        m_v = np.mean(scores("novel_clean", name))
        agg.append((name, a_nc, a_vc, m_n, m_c, m_v))
        print(f"{name:<11}{a_nc:>12.3f}{a_vc:>16.3f}{m_n:>12.4f}{m_c:>14.4f}"
              f"{m_v:>13.4f}")

    print("\n=== VERDICT P3 (criteres pre-fixes : AUC n-vs-c > 0.8 ET "
          "AUC novel-vs-c > 0.7) ===")
    d = {name: (a_nc, a_vc) for name, a_nc, a_vc, *_ in agg}
    for name in ["u_final", "sig_mean"]:
        a_nc, a_vc = d[name]
        ok = a_nc > 0.8 and a_vc > 0.7
        print(f"  {name:<9}: AUC={a_nc:.3f} / novelty={a_vc:.3f} -> "
              f"{'DETECTION GRATUITE VALIDEE' if ok else 'ne passe pas le critere'}")
    a_lap = d["lap_input"][0]
    best_net = max(d["u_final"][0], d["sig_mean"][0])
    if a_lap >= best_net:
        print(f"  lap_input: AUC={a_lap:.3f} >= meilleur signal reseau ({best_net:.3f})")
        print("  -> HONNETEMENT : un filtre laplacien STATIQUE de l'entree fait aussi")
        print("     bien ou mieux. La valeur du reseau n'est pas la CAPACITE de")
        print("     detection mais sa gratuite (variable d'etat deja presente quand le")
        print("     reseau tourne pour autre chose) et son integration temporelle.")
    else:
        print(f"  lap_input: AUC={a_lap:.3f} < meilleur signal reseau ({best_net:.3f})")
        print("  -> le reseau ajoute une capacite au-dela du filtre statique.")
    a_z = d["z_score"]
    print(f"  z_score  : AUC={a_z[0]:.3f} / novelty={a_z[1]:.3f} "
          f"(prediction : aveugle a la corruption, alarme sur la nouveaute)")

    # ---------------- CSV ----------------
    with CSV_PATH.open("w", encoding="utf-8") as f:
        f.write("kind,idx,u_final,sig_mean,z_score,lap_input\n")
        for r in rows:
            f.write(f"{r['kind']},{r['idx']},{r['u_final']:.6f},{r['sig_mean']:.6f},"
                    f"{r['z_score']:.6f},{r['lap_input']:.6f}\n")
    with AGG_PATH.open("w", encoding="utf-8") as f:
        f.write("signal,auc_normal_vs_corrupt,auc_novel_vs_corrupt,"
                "mean_normal,mean_corrupt,mean_novel\n")
        for name, a_nc, a_vc, m_n, m_c, m_v in agg:
            f.write(f"{name},{a_nc:.4f},{a_vc:.4f},{m_n:.6f},{m_c:.6f},{m_v:.6f}\n")
    print(f"\n[csv] {CSV_PATH}\n[csv] {AGG_PATH}")

    # ---------------- figure ----------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
        colors = {"normal": "#2ca02c", "corrupt": "#d62728", "novel_clean": "#1f77b4"}
        for ax, name in zip(axes, SIGNALS):
            data = [scores(k, name) for k in ["normal", "corrupt", "novel_clean"]]
            bp = ax.boxplot(data, labels=["normal", "corrompu", "nouveau\npropre"],
                            patch_artist=True)
            for patch, k in zip(bp["boxes"], ["normal", "corrupt", "novel_clean"]):
                patch.set_facecolor(colors[k])
                patch.set_alpha(0.6)
            a_nc = d[name][0] if name in d else auc_mw(scores("corrupt", name),
                                                       scores("normal", name))
            ax.set_title(f"{name}  (AUC={a_nc:.2f})")
            ax.grid(alpha=0.3, axis="y")
        fig.suptitle("P3 -- u comme detecteur d'anomalies structurelles : scores par "
                     "type d'entree (100 episodes)", fontsize=11)
        plt.tight_layout()
        plt.savefig(PNG_PATH, dpi=140)
        print(f"[png] {PNG_PATH}")
    except Exception as e:
        print(f"[png] skipped: {e}")

    print(f"\nWall time: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
