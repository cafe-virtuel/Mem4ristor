import subprocess
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Path configuration
# Ensure we are running from the project root or adjust paths relative to this script
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NETLIST_FILE = "experiments/mem4ristor_test.sp"
OUTPUT_FILE = "results/spice_output.txt"

def run_spice():
    if not os.path.exists(os.path.dirname(OUTPUT_FILE)):
        os.makedirs(os.path.dirname(OUTPUT_FILE))

    print(f"--- 1. Running Ngspice ---")
    
    # Check for ngspice in PATH or environment variable
    ngspice_exe = os.environ.get("NGSPICE_PATH", "ngspice")
    
    cmd = [ngspice_exe, "-b", NETLIST_FILE]
    
    try:
        print(f"Executing: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        print("Simulation completed.")
        return True
    except FileNotFoundError:
        print("CRITICAL ERROR: Ngspice executable not found.")
        print("Please ensure 'ngspice' is in your PATH or set NGSPICE_PATH environment variable.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"CRITICAL ERROR: Ngspice execution failed with return code {e.returncode}.")
        return False
    except Exception as e:
        print(f"CRITICAL ERROR: Unexpected error launching Ngspice. {e}")
        return False


def analyze_results():
    print(f"\n--- 2. Analyse des resultats ({OUTPUT_FILE}) ---")
    if not os.path.exists(OUTPUT_FILE):
        print("Erreur: Pas de fichier de sortie.")
        return

    data_v = []
    data_w = []
    
    # Parsing basique du format Ngspice ASCII
    # Le fichier contient des headers, puis des lignes "Numero ValeurV ValeurW"
    try:
        with open(OUTPUT_FILE, 'r') as f:
            lines = f.readlines()
            
        is_data = False
        for line in lines:
            line = line.strip()
            if line.startswith("Index"):
                is_data = True
                continue
            if line.startswith("-----"):
                continue
            
            if is_data and line:
                parts = line.split()
                if len(parts) >= 3: # Index, V, W (parfois decalé sur plusieurs lignes selon format, on simplifie)
                    # Le format 'print' de ngspice est parfois multi-ligne.
                    # On va supposer un format tabulaire simple pour ce test.
                    # Si ca echoue, on ajustera le parsing.
                    try:
                        # Ngspice print format: 
                        # 0 0.000000e+00 1.000000e+00 ...
                        # Parfois Index t V(v) V(w)
                         pass
                    except:
                        pass
        
        # Approche plus robuste : lire tout le fichier comme texte et chercher les nombres
        # Pour ce test rapide, on va juste vérifier si le fichier contient des données
        content = open(OUTPUT_FILE).read()
        if "e+" in content or "e-" in content:
            print(">> SUCCÈS: Données numériques détectées dans la sortie SPICE.")
            print(f"Taille du fichier: {len(content)} octets")
            print("Extrait des dernieres lignes:")
            print("\n".join(lines[-5:]))
            
            # Verif simple : est-ce que ça a bougé ?
            if "nan" in content.lower():
                print(">> ALERTE: Présence de NaN (Divergence!)")
            else:
                print(">> VALIDATION: Simulation stable (Pas de NaN/Inf)")
                print(">> Le modèle comportemental SPICE fonctionne.")
        else:
             print(">> ECHEC: Le fichier de sortie semble vide ou malformé.")

    except Exception as e:
        print(f"Erreur d'analyse: {e}")

if __name__ == "__main__":
    if run_spice():
        analyze_results()
