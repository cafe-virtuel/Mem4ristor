import numpy as np
from scipy.io import wavfile

def sonify_u_history(u_history, filename="output.wav", duration_sec=5.0, sample_rate=44100, f_base=220.0, f_range=880.0):
    """
    Sonifie l'historique du Doute (u) du réseau Mem4ristor.
    u_history: array de forme (T, N) où T est le nombre d'étapes de temps, N le nombre de nœuds.
    Chaque nœud est un oscillateur dont la fréquence dépend de son niveau de doute u.
    """
    T, N = u_history.shape
    L = int(duration_sec * sample_rate)
    
    # Interpolation de u_history pour correspondre au nombre d'échantillons audio
    time_sim = np.linspace(0, 1, T)
    time_audio = np.linspace(0, 1, L)
    
    audio_signal = np.zeros(L)
    
    for i in range(N):
        # Interpoler l'évolution de u pour ce nœud
        u_node = np.interp(time_audio, time_sim, u_history[:, i])
        
        # Calcul de la fréquence instantanée
        f_instant = f_base + u_node * f_range
        
        # Intégration pour obtenir la phase
        phase = np.cumsum(2 * np.pi * f_instant / sample_rate)
        
        # Ajout au signal global
        audio_signal += np.sin(phase)
        
    # Normalisation
    audio_signal = audio_signal / N
    
    # Enveloppe pour éviter les clics (fade in/out)
    fade_len = int(sample_rate * 0.1)
    fade_in = np.linspace(0, 1, fade_len)
    fade_out = np.linspace(1, 0, fade_len)
    
    if L > 2 * fade_len:
        audio_signal[:fade_len] *= fade_in
        audio_signal[-fade_len:] *= fade_out
        
    # Conversion en entier 16 bits
    audio_16bit = np.int16(audio_signal * 32767 * 0.9) # 0.9 pour éviter l'écrêtage
    
    # Écriture du fichier WAV
    wavfile.write(filename, sample_rate, audio_16bit)
    print(f"Fichier audio généré avec succès : {filename} ({duration_sec}s)")
    
    return filename
