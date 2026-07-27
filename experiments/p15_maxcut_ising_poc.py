"""
EXPERIMENT P15 : Max-Cut & Verre de Spins d'Ising (Version Auditée & Rectifiée).

Contexte :
Le problème du Max-Cut consiste à trouver la configuration de spins s_i in {-1, +1}
qui maximise la coupe W_cut = 0.25 * sum_{i,j} J_ij (1 - s_i s_j).

Audit Scientifique (27/07/2026) :
Julien (Le Barman) a exigé un audit de vérification des paramètres.
La version initiale utilisait une convention de signe pour le Recuit Simulé qui minimisait
l'énergie d'Ising ferromagnétique (alignement s_i s_j = +1), ce qui minimisait la coupe.
Pendant ce temps, le doute de Mem4ristor (u ~ 1.0 -> u_filter ~ -0.99) inversait automatiquement
le couplage, poussant Mem4ristor vers la coupe.

Après alignement rigoureux des conventions de signe (Metropolis sur W_cut) :
- Recuit Simulé (SA) : Coupe Moyenne = 125.50 (Explore les configurations discrètes de spins)
- Recherche Gloutonne  : Coupe Moyenne = 115.80
- Mem4ristor FULL      : Coupe Moyenne = 80.90 (Relaxation continue FHN, u_filter inversé)

Conclusion d'Honnêteté Scientifique :
Mem4ristor bascule naturellement en mode anti-synchronisation grâce au doute (u_filter < 0),
ce qui le pousse vers la coupe sans algorithme explicite. Néanmoins, pour la recherche combinatoire
discrète pure (Max-Cut), le Recuit Simulé discret optimisé reste supérieur à la relaxation continue FHN.

--------------------------------------------------------------------------------
AUDIT (Claude Opus 5, 27/07/2026) -- LA PHRASE "GRACE AU DOUTE" EST CONTREDITE
PAR L'ABLATION CONTENUE DANS CE SCRIPT
--------------------------------------------------------------------------------
Le classement ci-dessus (SA > Glouton >> M4R) est exact et honnete : un glouton trivial
fait +43% de coupe. Le point d'audit porte sur l'ATTRIBUTION, pas sur le classement.

Ce script embarque l'ablation qui teste la phrase "grace au doute" -- et elle dit non :

  Mem4ristor FROZEN_U (u fige a 0.5) : coupe = 82.10
  Mem4ristor FULL (doute adaptatif)  : coupe = 80.90
  FULL - FROZEN = -1.20, IC95 [-2.07, -0.33]  (l'intervalle EXCLUT zero)

La comparaison est loyale : a u=0.5 le filtre vaut +0.01 (couplage quasi nul), contre
~ -0.99 en FULL (anti-synchronisation forte). Elle teste donc bien l'effet revendique.
Le doute adaptatif ne pousse PAS vers la coupe -- il la degrade legerement.

Ce qui est etabli : le doute n'est pas la cause de la performance de M4R ici.
Ce qui ne l'est PAS : que le doute "nuit". L'effet vaut 1.5%, mesure sur UN SEUL
groupe de 10 graines, sans gate de replication. Ne pas citer ce -1.20 au chiffre.

QUESTION OUVERTE (non expliquee) : 4 graines sur 10 donnent des resultats STRICTEMENT
identiques entre FULL et FROZEN_U (coupe ET energie). Sur ces runs, u ne change
litteralement rien au resultat -- soit best_cut est atteint avant que u ne diverge,
soit la relaxation converge au meme point. A trancher avant toute reprise de ce POC.
[Instruite le 27/07 au soir : voir AUDIT 2 ci-dessous. Les deux pistes citees dans
 cette phrase sont l'une et l'autre FAUSSES, mesure a l'appui.]

--------------------------------------------------------------------------------
AUDIT 2 (Claude Opus 5, 27/07/2026 au SOIR) -- LA QUESTION OUVERTE CI-DESSUS N'EST
PAS TRANCHEE, MAIS SON INSTRUCTION A LIVRE UN FAIT PLUS DUR QU'ELLE
--------------------------------------------------------------------------------
Diagnostic : `experiments/p15b_maxcut_identity_diagnosis.py`. Criteres d'acceptation
ecrits AVANT chaque execution, gate de fidelite sur le CSV de ce script (reproduction
au bit pres des 20 couples coupe/energie), replication sur les graines 10-19 jamais
touchees par le projet.

CE QUI EST ETABLI ET REPLIQUE (10/10 puis 10/10 graines neuves ; critere >= 8/10) :
a budget d'echantillons EGAL -- M4R lit sign(v) 300 fois (tous les 10 pas sur 3000),
on tire 300 configurations de spins uniformes -- le meilleur de 300 tirages ALEATOIRES
bat Mem4ristor :

    best-of-300 aleatoire : 91.50 (graines 0-9)  |  89.10 (graines 10-19)
    Mem4ristor FROZEN_U   : 82.10                |  80.50
    Mem4ristor FULL       : 80.90                |  79.80

La phrase de l'en-tete d'origine -- "Mem4ristor bascule naturellement en mode
anti-synchronisation grace au doute, ce qui le pousse vers la coupe sans algorithme
explicite" -- est donc contredite une seconde fois, et plus durement que par
l'ablation du matin : sur cette tache la relaxation ne pousse vers rien, elle fait
moins bien que tirer a pile ou face le meme nombre de fois. Le classement
SA > glouton >> M4R reste exact ; c'est son interpretation qui tombe.

CE QUI N'EST PAS TRANCHE : le mecanisme de l'identite FULL/FROZEN_U. Cinq
explications posees avec leur critere ecrit avant mesure, cinq rejetees :
  H_A best fige au premier echantillon         -> non (n_improve vaut 1, 2 ou 3)
  H_B best atteint avant divergence des signes -> non (graine 4 : t_div_s=10, t_best=40)
  H_C les deux relaxations convergent au meme point -> non (les signes divergent)
  R2  la relaxation degrade la coupe           -> non (stationnaire : 73.06 -> 73.28)
  R5  etats visites indiscernables d'un tirage uniforme -> 7/10, critere pose a 8/10
Sur R5, le fait brut sans repechage du critere : agregee, la coupe moyenne visitee
par M4R (73.17 FULL / 72.94 FROZEN_U) est a moins de 1 pct de l'esperance analytique
d'un tirage uniforme, 0.25*(sum|J| - tr J) = 72.75 ; mais graine par graine l'ecart
median vaut ~4 pct et le critere avait ete ecrit a 5 pct. Rejetee, donc.

CE QUE LA QUESTION A APPRIS MALGRE TOUT :
  - L'identite est GENERIQUE et non une curiosite des 10 premieres graines :
    4/10 ici, 6/10 sur les graines 10-19, soit 10 sur 20.
  - Le meilleur echantillon tombe TRES TOT : mediane de t_best = 30 pas sur 3000,
    et 16 runs sur 20 fixent leur best dans les 300 premiers pas.
  - "u n'a pas eu le temps de diverger de 0.5" est exclu par la LECTURE DU CODE,
    sans mesure : sigma_baseline = 0.05, donc en FULL u demarre a 0.05 et non a 0.5.
    Les deux conditions ont un u_filter different des le premier pas, et la mesure
    le confirme (t_div_v = 1 sur les 20 graines).

Consequence pour toute reprise de ce POC : le -1.20 FULL - FROZEN_U reste a ne pas
citer, et pour une raison de plus qu'en fin de matinee -- c'est l'ecart entre deux
maximums d'echantillons bruites dont aucun ne bat le hasard.

Statut : exploration (colonne B), resultat NEGATIF, conserve et documente.
Aucun chiffre du preprint, aucun claim du Guardian, aucun CSV canonique concerne.
Reproductibilite verifiee le 27/07 : re-execution complete, colonnes scientifiques
identiques au bit pres (seuls les chronometres varient).
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mem4ristor.graph_utils import make_ba
from mem4ristor.dynamics import Mem4ristorV3

def compute_cut_and_energy(s, J):
    energy = -0.5 * np.dot(s, np.dot(J, s))
    W_cut = 0.25 * np.sum(np.abs(J) - J * np.outer(s, s))
    return energy, W_cut

def solve_greedy_maxcut(J, max_steps=1000):
    N = J.shape[0]
    s = np.random.choice([-1, 1], size=N)
    
    for _ in range(max_steps):
        flipped = False
        nodes = np.random.permutation(N)
        for i in nodes:
            delta_W = s[i] * np.dot(J[i], s)
            if delta_W > 0:
                s[i] = -s[i]
                flipped = True
        if not flipped:
            break
            
    energy, cut = compute_cut_and_energy(s, J)
    return s, energy, cut

def solve_sa_maxcut(J, steps=5000, T_init=10.0, T_min=0.001):
    N = J.shape[0]
    s = np.random.choice([-1, 1], size=N)
    best_s = s.copy()
    _, best_cut = compute_cut_and_energy(s, J)
    current_cut = best_cut
    
    alpha = (T_min / T_init) ** (1.0 / steps)
    T = T_init
    
    for step in range(steps):
        i = np.random.randint(N)
        delta_W = s[i] * np.dot(J[i], s)
        
        if delta_W > 0 or np.random.rand() < np.exp(delta_W / T):
            s[i] = -s[i]
            current_cut += delta_W
            if current_cut > best_cut:
                best_cut = current_cut
                best_s = s.copy()
                
        T *= alpha
        
    energy, final_cut = compute_cut_and_energy(best_s, J)
    return best_s, energy, final_cut

def best_of_random_cut(J, n_draws, seed):
    """Contre-epreuve a budget d'echantillons EGAL (audit du 27/07 au soir).

    M4R lit sign(v) une fois tous les 10 pas ; on tire ici le meme nombre de
    configurations de spins uniformes et on garde la meilleure coupe.
    RNG dedie : ne consomme ni le RNG global ni celui des modeles, donc n'altere
    aucun run existant de ce script (verifie : colonnes scientifiques inchangees).
    """
    N = J.shape[0]
    rng = np.random.RandomState(10_000 + seed)
    best = 0.0
    for _ in range(n_draws):
        s = rng.choice([-1.0, 1.0], size=N)
        _, cut = compute_cut_and_energy(s, J)
        if cut > best:
            best = cut
    return best

def solve_mem4ristor_maxcut(J, steps=3000, frozen_u=False, seed=42):
    N = J.shape[0]
    model = Mem4ristorV3(seed=seed)
    model.cfg['coupling']['heretic_ratio'] = 0.15
    model._initialize_params(N, cold_start=True)
    
    best_cut = 0
    best_s = None
    best_E = 0
    
    for step in range(steps):
        if frozen_u:
            model.u[:] = 0.5
            
        l_v = np.dot(J, model.v)
        model.step(I_stimulus=0.0, coupling_input=l_v)
        
        s = np.sign(model.v)
        s[s == 0] = 1
        
        if step % 10 == 0:
            E, cut = compute_cut_and_energy(s, J)
            if cut > best_cut:
                best_cut = cut
                best_E = E
                best_s = s.copy()
                
    return best_s, best_E, best_cut

def run_maxcut_benchmark(n_seeds=10, N=100, steps=3000):
    print("=" * 70)
    print(f"BENCHMARK P15 AUDITÉ : MAX-CUT & VERRE DE SPINS D'ISING (N={N}, Seeds={n_seeds})")
    print("=" * 70)
    
    results = []
    
    for seed in range(n_seeds):
        np.random.seed(seed)
        
        adj_base = make_ba(N, m=3, seed=seed)
        weights = np.random.choice([-1.0, 1.0], size=adj_base.shape)
        J = adj_base * weights
        J = (J + J.T) / 2.0
        
        t0 = time.time()
        _, E_greedy, cut_greedy = solve_greedy_maxcut(J)
        t_greedy = time.time() - t0
        
        t0 = time.time()
        _, E_sa, cut_sa = solve_sa_maxcut(J, steps=5000)
        t_sa = time.time() - t0
        
        t0 = time.time()
        _, E_frozen, cut_frozen = solve_mem4ristor_maxcut(J, steps=steps, frozen_u=True, seed=seed)
        t_frozen = time.time() - t0
        
        t0 = time.time()
        _, E_m4r, cut_m4r = solve_mem4ristor_maxcut(J, steps=steps, frozen_u=False, seed=seed)
        t_m4r = time.time() - t0
        
        # Audit 27/07 soir : baseline a budget d'echantillons egal. Calculee APRES
        # les quatre solveurs, avec un RNG dedie, donc sans effet sur eux.
        cut_random = best_of_random_cut(J, steps // 10, seed)

        results.append({
            'seed': seed,
            'E_greedy': E_greedy, 'cut_greedy': cut_greedy, 't_greedy': t_greedy,
            'E_sa': E_sa, 'cut_sa': cut_sa, 't_sa': t_sa,
            'E_frozen': E_frozen, 'cut_frozen': cut_frozen, 't_frozen': t_frozen,
            'E_m4r': E_m4r, 'cut_m4r': cut_m4r, 't_m4r': t_m4r,
            'cut_random_budget': cut_random
        })

        print(f"Seed {seed+1:02d}/{n_seeds:02d} | "
              f"Greedy: {cut_greedy:.0f} | "
              f"SA: {cut_sa:.0f} | "
              f"M4R Frozen: {cut_frozen:.0f} | "
              f"M4R FULL: {cut_m4r:.0f} | "
              f"Aleatoire (budget egal): {cut_random:.0f}")
        
    df = pd.DataFrame(results)
    
    os.makedirs("figures", exist_ok=True)
    csv_path = "figures/p15_maxcut_benchmark.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[CSV enregistré] : {csv_path}")
    
    mean_cut_greedy = df['cut_greedy'].mean()
    mean_cut_sa = df['cut_sa'].mean()
    mean_cut_frozen = df['cut_frozen'].mean()
    mean_cut_m4r = df['cut_m4r'].mean()
    
    print("\n" + "=" * 50)
    print("RÉSULTATS CORRECTIFS AUDITÉS DE LA COUPE (MAX-CUT) :")
    print(f"  Recherche Gloutonne  : Coupe Moyenne = {mean_cut_greedy:.2f}")
    print(f"  Recuit Simulé (SA)   : Coupe Moyenne = {mean_cut_sa:.2f}")
    print(f"  Mem4ristor FROZEN    : Coupe Moyenne = {mean_cut_frozen:.2f}")
    print(f"  Mem4ristor FULL      : Coupe Moyenne = {mean_cut_m4r:.2f}")
    print("=" * 50)

    # Verdict d'attribution (audit du 27/07) : le classement etait honnete, mais la
    # phrase "grace au doute" de l'en-tete est contredite par cette ablation meme.
    # ASCII pur dans les print (piege cp1252).
    d = df['cut_m4r'] - df['cut_frozen']
    sem = d.std(ddof=1) / np.sqrt(len(d))
    n_ident = int(((df.cut_m4r == df.cut_frozen) & (df.E_m4r == df.E_frozen)).sum())
    print("")
    print("ATTRIBUTION : le doute adaptatif n'est PAS la cause de la coupe obtenue.")
    print("  FULL - FROZEN_U = %+.2f, IC95 [%+.2f, %+.2f] -- l'intervalle exclut zero."
          % (d.mean(), d.mean() - 1.96 * sem, d.mean() + 1.96 * sem))
    print("  Geler u a 0.5 fait marginalement MIEUX que le laisser s'adapter.")
    print("  Effet de 1.5 pct, un seul groupe de graines, sans gate de replication :")
    print("  ne pas citer cet ecart au chiffre, et ne pas conclure que le doute nuit.")
    print("  Question ouverte : %d graines sur %d sont STRICTEMENT identiques FULL/FROZEN."
          % (n_ident, len(df)))
    print("    -> instruite le 27/07 au soir : 5 explications posees, 5 rejetees a leur")
    print("       propre critere. Non tranchee. Voir p15b_maxcut_identity_diagnosis.py")
    print("=" * 50)

    # Audit 27/07 soir : le verdict le plus dur du script, recalcule a chaque
    # execution pour qu'aucun chiffre en dur ne puisse deriver de ses donnees.
    n_rnd_win = int((df.cut_random_budget >= df.cut_m4r).sum())
    mean_rnd = df.cut_random_budget.mean()
    print("")
    print("BUDGET D'ECHANTILLONS EGAL : M4R lit sign(v) %d fois, on tire %d spins au hasard."
          % (steps // 10, steps // 10))
    print("  Le meilleur de %d tirages ALEATOIRES bat M4R FULL sur %d graines sur %d."
          % (steps // 10, n_rnd_win, len(df)))
    print("  Aleatoire %.2f | FROZEN_U %.2f | FULL %.2f" % (mean_rnd, mean_cut_frozen, mean_cut_m4r))
    if n_rnd_win >= 0.8 * len(df):
        print("  >> Sur cette tache la relaxation M4R ne pousse vers rien : elle fait moins")
        print("     bien que tirer a pile ou face le meme nombre de fois. Replique sur les")
        print("     graines 10-19 jamais touchees (10/10) -- voir p15b_maxcut_identity_diagnosis.py.")
    print("=" * 50)
    
    # La barre "Aleatoire (budget egal)" est ajoutee le 27/07 au soir : sans elle,
    # la figure laissait croire que M4R etait dans la course. Il est derriere le hasard.
    plt.figure(figsize=(11, 6))
    methods = ['Glouton', 'Recuit Simulé', 'Aléatoire\n(budget égal)',
               'Mem4ristor (FROZEN)', 'Mem4ristor (FULL)']
    cuts = [mean_cut_greedy, mean_cut_sa, mean_rnd, mean_cut_frozen, mean_cut_m4r]
    stds = [df['cut_greedy'].std(), df['cut_sa'].std(), df['cut_random_budget'].std(),
            df['cut_frozen'].std(), df['cut_m4r'].std()]
    colors = ['#888888', '#e74c3c', '#34495e', '#f39c12', '#2ecc71']
    
    bars = plt.bar(methods, cuts, yerr=stds, capsize=5, color=colors, alpha=0.85, edgecolor='black')
    plt.ylabel('Valeur de la Coupe (Max-Cut)', fontsize=12)
    plt.title(f'Benchmark Max-Cut Audité sur Verre de Spins (N={N}, {n_seeds} seeds)', fontsize=14, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar, cut in zip(bars, cuts):
        plt.text(bar.get_x() + bar.get_width()/2.0, cut / 2.0, f'{cut:.1f}',
                 ha='center', va='bottom', color='white', fontweight='bold', fontsize=12)
                 
    png_path = "figures/p15_maxcut_faceoff.png"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Graphique enregistré] : {png_path}")

if __name__ == "__main__":
    run_maxcut_benchmark(n_seeds=10, N=100, steps=3000)
