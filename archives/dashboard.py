import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys

# Integration avec le dossier src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from mem4ristor.core import Mem4Network

st.set_page_config(page_title="Mem4ristor v2 Dashboard", layout="wide")

st.title("🧠 Mem4ristor v2.0.4 - Dashboard de Justice Cognitive")
st.markdown("""
*Visualisation temps réel de la dynamique neuromorphique avec **Doute Constitutionnel**.*
""")

# Barre latérale pour les contrôles
st.sidebar.header("🕹️ Contrôles du Réseau")
size = st.sidebar.slider("Taille de la Grille (NxN)", 5, 50, 20)
heretic_ratio = st.sidebar.slider("Ratio d'Hérétiques", 0.0, 0.5, 0.15)
d_coupling = st.sidebar.slider("Force de Couplage (D)", 0.0, 1.0, 0.15)

st.sidebar.markdown("---")
st.sidebar.header("⚡ Stimulus & Dynamique")
i_stim = st.sidebar.slider("Stimulus Externe (I_stim)", 0.0, 2.0, 1.1)
dt = st.sidebar.number_input("Pas de temps (dt)", 0.01, 0.5, 0.1)

# Initialisation du modèle dans le state Streamlit
if 'model' not in st.session_state or st.sidebar.button("Réinitialiser le Réseau"):
    st.session_state.model = Mem4Network(size=size, heretic_ratio=heretic_ratio)
    st.session_state.history = []
    st.session_state.step_count = 0

# Mise à jour des paramètres dynamiques (via le dictionnaire cfg interne)
st.session_state.model.model.D_eff = d_coupling / np.sqrt(size * size)
st.session_state.model.model.dt = dt

# Layout Principal
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🌐 Grille de Potentiel Cognitif (V)")
    heatmap_placeholder = st.empty()

with col2:
    st.subheader("📊 Métriques")
    entropy_placeholder = st.empty()
    dist_placeholder = st.empty()

# Boucle de simulation
run = st.checkbox("▶️ Lancer la Simulation", value=True)

while run:
    # Calculer N steps pour fluidité Streamlit
    for _ in range(5):
        st.session_state.model.step(I_stimulus=i_stim)
    
    st.session_state.step_count += 5
    
    # Visualisation Heatmap
    v_grid = st.session_state.model.v.reshape((size, size))
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(v_grid, cmap='RdBu_r', vmin=-2.0, vmax=2.0)
    ax.axis('off')
    plt.colorbar(im, ax=ax)
    heatmap_placeholder.pyplot(fig)
    plt.close(fig)
    
    # Métriques
    entropy = st.session_state.model.calculate_entropy()
    st.session_state.history.append(entropy)
    if len(st.session_state.history) > 100:
        st.session_state.history.pop(0)
        
    entropy_placeholder.metric("Entropie Shannon (Diversité)", f"{entropy:.4f}")
    
    # Distribution des états
    dist = st.session_state.model.get_state_distribution()
    dist_placeholder.bar_chart(dist)
    
    time.sleep(0.01)
    
    if not run:
        break
