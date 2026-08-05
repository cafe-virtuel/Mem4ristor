import numpy as np
import time
from mem4ristor.core import Mem4Network

def run_byzantine_attack():
    """
    Simulation of a 'Byzantine Fault' (Forced Consensus).
    Des acteurs malveillants tentent de forcer tout le réseau vers l'état +1.5 (Certitude Positive).
    
    Hypothèse :
    - Réseau Standard : Doit tomber dans le consensus forcé (Fail).
    - Réseau Mem4ristor (avec Hérétiques/Doute) : Doit résister et garder de le diversité (Pass).
    """
    SIZE = 50
    N = SIZE * SIZE
    ATTACKER_RATIO = 0.05 # 5% de noeuds malveillants suffisent souvent a faire basculer un reseau classique
    
    print(f"=== TEST DE RESILIENCE (Byzantine Attack) N={N} ===")
    print(f"Scenario : {int(ATTACKER_RATIO*100)}% d'attaquants forcent l'etat +1.5")

    # --- CAS A : Réseau Classique (Pas d'hérétiques, Doute faible) ---
    print("\n[CAS A] Réseau Naif (0% Hérétiques)")
    net_A = Mem4Network(size=SIZE, heretic_ratio=0.0, seed=42)
    net_A.model.D_eff = 0.5 # Couplage fort pour propager l'attaque
    
    # Selection des attaquants
    n_attackers = int(N * ATTACKER_RATIO)
    attacker_indices = np.random.choice(N, n_attackers, replace=False)
    
    start = time.time()
    for t in range(500):
        # 1. Force Attackers
        net_A.model.v[attacker_indices] = 2.0
        net_A.model.w[attacker_indices] = 0.0
        
        # 2. Step with Global Bias (Propaganda)
        # Propaganda pushes everyone towards the attackers' view (+1.0)
        net_A.step(I_stimulus=0.8) 
    
    dist_A = net_A.get_state_distribution()
    entropy_A = net_A.calculate_entropy()
    victim_count_A = dist_A.get('bin_4', 0)
    victim_ratio_A = victim_count_A / N
    
    print(f"Resultat A : Entropie={entropy_A:.4f}, Victimes={victim_ratio_A*100:.1f}%")

    # --- CAS B : Réseau Mem4ristor (15% Hérétiques + Doute Actif) ---
    print("\n[CAS B] Réseau Mem4ristor (15% Hérétiques)")
    net_B = Mem4Network(size=SIZE, heretic_ratio=0.15, seed=42)
    net_B.model.D_eff = 0.5
    
    # Selection des attaquants (memes indices pour comparaison equitable)
    # Attention: l'heretic mask est interne, on risque de transformer un heretique en attaquant, ce qui est ok (traire)
    
    start = time.time()
    for t in range(500):
        # 1. Force Attackers
        net_B.model.v[attacker_indices] = 2.0
        net_B.model.w[attacker_indices] = 0.0
        
        # 2. Step with Global Bias
        net_B.step(I_stimulus=0.8)
        
    dist_B = net_B.get_state_distribution()
    entropy_B = net_B.calculate_entropy()
    victim_count_B = dist_B.get('bin_4', 0)
    victim_ratio_B = victim_count_B / N
    
    print(f"Resultat B : Entropie={entropy_B:.4f}, Victimes={victim_ratio_B*100:.1f}%")
    
    # --- VERDICT ---
    print("\n=== VERDICT ===")
    entropy_gain = entropy_B - entropy_A
    print(f"Gain d'Entropie (Diversité) : +{entropy_gain:.4f}")
    
    # Critere de succes : Le reseau Mem4ristor doit maintenir une diversite (Entropie > 1.0)
    # la ou le reseau temoin s'effondre dans le consensus (Entropie < 1.0).
    if entropy_B > 1.5 and entropy_A < 1.0:
        print(">> VICTOIRE : Le système a maintenu la diversité cognitive face à la propagande.")
    elif entropy_B > entropy_A:
        print(">> PARTIEL : Plus de diversité, mais pas décisif.")
    else:
        print(">> ECHEC : Pas de gain de diversité.")

if __name__ == "__main__":
    try:
        # Import fix
        import sys, os
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
        run_byzantine_attack()
    except Exception as e:
        # Fallback si lancé depuis la racine
        print(f"Erreur lancement direct: {e}")
        import sys, os
        sys.path.append(os.path.abspath('src'))
        from mem4ristor.core import Mem4Network
        run_byzantine_attack()
