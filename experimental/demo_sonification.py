import os
import sys
import numpy as np

# Ajouter le chemin src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from mem4ristor.topology import Mem4Network
from mem4ristor.sonification import sonify_u_history

def run_demo():
    print("Initialisation du réseau (N=20)...")
    N = 20
    # On crée une matrice d'adjacence simple (ex: anneau)
    adjacency = np.zeros((N, N))
    for i in range(N):
        adjacency[i, (i+1)%N] = 1
        adjacency[(i+1)%N, i] = 1
        
    # Mem4Network gère les hérétiques via le ratio
    network = Mem4Network(
        adjacency_matrix=adjacency,
        heretic_ratio=0.1
    )
    
    # Configuration des paramètres via cfg
    network.model.cfg['noise']['eta'] = 0.5
    network.model.cfg['dynamics']['alpha'] = 0.1
    network.model.cfg['doubt']['tau_u'] = 100.0
    
    print("Simulation en cours (2000 étapes)...")
    steps = 2000
    u_history = np.zeros((steps, N))
    
    for t in range(steps):
        network.step()
        u_history[t] = network.model.u.copy()
        
    # On veut entendre la séparation des hérétiques
    # Les hérétiques devraient avoir un u qui s'éloigne de la majorité
    
    output_wav = os.path.join(os.path.dirname(__file__), 'sonification_output.wav')
    print(f"Génération du son vers {output_wav}...")
    
    # f_base = 110 (La) pour la masse, f_range = 440 pour entendre clairement les hérétiques
    sonify_u_history(
        u_history, 
        filename=output_wav, 
        duration_sec=8.0, 
        f_base=110.0, 
        f_range=440.0
    )
    
    print("Terminé ! Ouvrez le fichier .wav pour écouter la transition de phase.")

if __name__ == "__main__":
    run_demo()
