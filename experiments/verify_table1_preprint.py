"""
Verification ET generation du Tableau 1 de preprint.tex (tab:scaling).
Reproduit : H_stable (H_cont) et mean_doubt pour lattices 4x4, 10x10, 25x25.

Parametres (legende Tableau 1) :
  - I_stimulus = 0.5, eta = 0.15, 3000 steps, derniers 25%, 10 seeds, dt=0.05
  - COLD START : v = w = 0 (cold_start=True) -- conforme au "Cold Start Protocol"
    revendique dans le texte (preprint.tex, §4.1 et legende de tab:scaling).

Historique :
  - n=3 -> n=10 seeds le 2026-05-05 (robustesse de l'ecart-type).
  - 2026-07-08 (backlog A4) : AJOUT de cold_start=True. Avant cette date le
    script tournait en init ALEATOIRE (v in [-1.5,1.5]) alors que le texte
    revendiquait v=w=0 -- incoherence relevee par l'audit externe neuromorphique (06/07).
    Le script ECRIT desormais figure/p2_table1_lattice.csv (colonne h_cont_mean
    verifiee par le Guardian, claims C02/C03), pour que la table publiee soit
    reproductible d'un seul run (aucune valeur ecrite a la main).
  - 2026-07-31 (decision de Julien) : le script calcule ET ECRIT AUSSI LA
    SYNCHRONIE (figures/p2_table1_sync.csv, claim C07b), sur la MEME trajectoire
    et dans le MEME run que l'entropie.
    POURQUOI : jusqu'ici la synchronie publiee venait de .brain/gen_c07_csv.py,
    un script du depot PILOTE qui visait un repertoire aujourd'hui MORT et qui
    tournait SANS cold_start. Le CSV etait exact et reproductible au bit pres
    (verifie le 31/07), mais il n'etait pas produit par le protocole que le
    papier revendique. Le texte affirmait pourtant, en trois endroits, que les
    deux colonnes venaient d'un seul protocole (L182, L195, L515) : c'etait
    faux, et mesure -- sans cold start 0.0197, avec 0.0359, soit plus d'un
    ecart-type publie. Les deux colonnes viennent desormais du meme run, ce qui
    rend la phrase publiee vraie au lieu de la corriger.
    NB : la synchronie utilise np.corrcoef (Pearson exact, ce que le texte
    annonce). gen_c07_csv.py divisait une covariance normalisee par (T-1) par
    des ecarts-types normalises par T : un facteur T/(T-1) = +0.13 % sur 750
    pas. Ecart mesure et negligeable, mais la formule annoncee est desormais
    celle qui tourne.

On calcule aussi I_stimulus=0.0 pour diagnostic (non ecrit au CSV : Table 1 = 0.5).
"""

import csv
import sys
import os
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # console Windows cp1252
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from mem4ristor.core import Mem4Network
from mem4ristor.metrics import calculate_continuous_entropy

STEPS    = 3000
WARMUP   = int(STEPS * 0.75)   # derniers 25%
SEEDS    = [42, 123, 777, 17, 256, 1337, 99, 314, 2024, 888]  # n=10
SIZES    = [4, 10, 25]          # 4x4=16, 10x10=100, 25x25=625
ETA      = 0.15
I_VALUES = [0.0, 0.5]           # Table 1 utilise 0.5 ; 0.0 = diagnostic
CSV_I    = 0.5                  # valeur ecrite au CSV canonique

HERE = os.path.dirname(__file__)
CSV_PATH = os.path.abspath(os.path.join(HERE, "..", "figures", "p2_table1_lattice.csv"))
SYNC_CSV_PATH = os.path.abspath(os.path.join(HERE, "..", "figures", "p2_table1_sync.csv"))


def pairwise_sync(v_history):
    """Corrélation de Pearson moyenne entre paires de noeuds, sur la fenetre
    stable -- la definition annoncee par le preprint (tab:benchmarks : « mean
    pairwise Pearson correlation of v(t) trajectories »).

    Calculee sur la MEME trajectoire que l'entropie, dans le meme run : c'est
    tout l'interet du deplacement du 31/07. Un noeud a variance nulle (jamais
    observe ici, mais possible sur un reseau totalement fige) donnerait des NaN
    dans corrcoef : on les ecarte explicitement plutot que de les propager en
    silence."""
    V = np.asarray(v_history)          # (T, N)
    if V.shape[1] < 2:
        return 0.0
    C = np.corrcoef(V, rowvar=False)   # (N, N)
    iu = np.triu_indices(V.shape[1], k=1)
    pairs = C[iu]
    pairs = pairs[np.isfinite(pairs)]
    return float(np.mean(pairs)) if pairs.size else 0.0

print("=" * 65)
print(f"  VERIFICATION TABLEAU 1 (COLD START) -- {STEPS} steps, derniers 25%, n={len(SEEDS)} seeds")
print("=" * 65)

results = {}  # (I_stim, size) -> (H_mean, H_std, u_mean, u_std, N)
sync_results = {}  # (I_stim, size) -> (sync_mean, sync_std) -- dict SEPARE, pour ne pas
                   # toucher au deballage existant de `results` (gate de fidelite : le CSV
                   # d'entropie doit rester identique au bit pres apres cet ajout).

for I_stim in I_VALUES:
    print(f"\n>>> I_stimulus = {I_stim}  (cold_start=True, v=w=0)")
    print(f"  {'Size':>6}  {'N':>5}  {'H_stable':>12}  {'std':>7}  {'doubt_mean':>11}")
    print(f"  {'-'*55}")
    for size in SIZES:
        H_runs, u_runs, sync_runs = [], [], []
        for seed in SEEDS:
            net = Mem4Network(size=size, heretic_ratio=ETA, seed=seed, cold_start=True)
            v_history, u_history = [], []
            for step in range(STEPS):
                net.step(I_stimulus=I_stim)
                if step >= WARMUP:
                    v_history.append(net.model.v.copy())
                    u_history.append(net.model.u.mean())
            H_vals = [calculate_continuous_entropy(v) for v in v_history]
            H_runs.append(np.mean(H_vals))
            u_runs.append(np.mean(u_history))
            sync_runs.append(pairwise_sync(v_history))  # meme trajectoire, meme run
        H_mean, H_std = float(np.mean(H_runs)), float(np.std(H_runs))
        u_mean, u_std = float(np.mean(u_runs)), float(np.std(u_runs))
        N = size * size
        results[(I_stim, size)] = (H_mean, H_std, u_mean, u_std, N)
        sync_results[(I_stim, size)] = (float(np.mean(sync_runs)), float(np.std(sync_runs)))
        print(f"  {size:>2}x{size:<2}   {N:>5}  {H_mean:>8.2f} bits  +/-{H_std:.2f}  "
              f"u={u_mean:.3f}  sync={sync_results[(I_stim, size)][0]:.4f}")

# --- Ecriture du CSV canonique (I_stimulus = 0.5) --------------------------
with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["topology", "N", "h_cont_mean", "h_cont_std", "doubt_mean", "doubt_std"])
    for size in SIZES:
        H_mean, H_std, u_mean, u_std, N = results[(CSV_I, size)]
        w.writerow([f"lattice_{size}x{size}", N, H_mean, H_std, u_mean, u_std])
print(f"\n[csv] {CSV_PATH}  (I_stimulus={CSV_I}, cold start)")

# --- CSV de synchronie (claim C07b, publie en tab:benchmarks) ---------------
# Memes runs que ci-dessus : c'est la raison d'etre de ce bloc (cf. en-tete, 31/07).
with open(SYNC_CSV_PATH, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["topology", "N", "sync_mean", "sync_std"])
    for size in SIZES:
        s_mean, s_std = sync_results[(CSV_I, size)]
        w.writerow([f"lattice_{size}x{size}", size * size, s_mean, s_std])
print(f"[csv] {SYNC_CSV_PATH}  (I_stimulus={CSV_I}, cold start, Pearson)")

print("\n" + "=" * 65)
print("Valeurs pour le Tableau 1 (tab:scaling, I_stim=0.5, cold start, n=10) :")
for size in SIZES:
    H_mean, H_std, u_mean, u_std, N = results[(0.5, size)]
    s_mean, s_std = sync_results[(0.5, size)]
    print(f"  {size}x{size:<2} (N={N:>3}) : H = {H_mean:.2f} +/- {H_std:.2f},  "
          f"u = {u_mean:.3f},  sync = {s_mean:.4f} +/- {s_std:.4f}")
print("=" * 65)
print("Rappel (31/07) : la synchronie ci-dessus vient des MEMES runs que H, en cold")
print("start. Le chiffre publie avant cette date (0.0197) venait d'un run SANS cold")
print("start -- exact chez lui, mais pas le protocole revendique par le papier.")
print("=" * 65)
