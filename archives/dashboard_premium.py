import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import os
import sys

# Integration avec le dossier src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from mem4ristor.core import Mem4Network

def calculate_gini(v):
    """Calcul du coefficient de Gini appliqué à la certitude (abs(v))."""
    x = np.abs(v)
    if np.sum(x) == 0: return 0
    n = len(x)
    x = np.sort(x)
    index = np.arange(1, n + 1)
    return (np.sum((2 * index - n - 1) * x)) / (n * np.sum(x))

# Config Page
st.set_page_config(page_title="Mem4ristor Premium Dashboard", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for "Cafe Virtuel" Aesthetic
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stMetric { background-color: #1a1c23; border-radius: 10px; padding: 10px; border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

st.title("🌌 Mem4RISTOR v2.2 - Justice Cognitive & Percolation")
st.markdown("---")

# Sidebar
st.sidebar.image("https://raw.githubusercontent.com/Jusyl236/Cafe-Virtuel/main/logo.png", width=100) # Placeholder logo
st.sidebar.title("Configuration")

with st.sidebar.expander("🛠️ Paramètres du Réseau", expanded=True):
    size = st.slider("Taille du Réseau (NxN)", 10, 200, 50, step=10)
    topo_type = st.selectbox("Topologie", ["Grille 2D", "Small-World (Complex)"])
    heretic_ratio = st.slider("Ratio d'Hérétiques", 0.0, 0.4, 0.15)
    d_coupling = st.slider("Force de Couplage (D)", 0.0, 1.0, 0.2)

with st.sidebar.expander("🧠 Psychologie & Bruit", expanded=True):
    i_stim = st.slider("Intensité de Pression (I_stim)", 0.0, 2.5, 1.1)
    tau_u = st.slider("Inertie du Doute (tau_u)", 0.1, 10.0, 1.0)
    rtn_noise = st.checkbox("Activer Bruit RTN (Hardware Allied)", value=False)

def init_model(size, heretic_ratio, topo_type):
    adj = None
    if topo_type == "Small-World (Complex)":
        import networkx as nx
        G = nx.barabasi_albert_graph(size*size, 3)
        adj = nx.to_numpy_array(G)
    return Mem4Network(size=size, heretic_ratio=heretic_ratio, adjacency_matrix=adj)

if st.sidebar.button("♻️ Réinitialiser la Grille", use_container_width=True):
    st.session_state.model = init_model(size, heretic_ratio, topo_type)
    st.session_state.h_history = []
    st.session_state.gini_history = []
    st.rerun()

# Logic
if 'model' not in st.session_state:
    st.session_state.model = init_model(size, heretic_ratio, topo_type)
    st.session_state.h_history = []
    st.session_state.gini_history = []

# Main Layout
col_main, col_metrics = st.columns([3, 1])

with col_main:
    placeholder_map = st.empty()
    placeholder_chart = st.empty()

with col_metrics:
    st.subheader("⚖️ Radar de Justice")
    placeholder_radar = st.empty()
    st.markdown("---")
    m1 = st.empty()
    m2 = st.empty()
    m3 = st.empty()
    st.markdown("---")
    placeholder_dist = st.empty()

with st.sidebar.expander("🚀 Performance", expanded=False):
    steps_per_frame = st.slider("Étapes par image", 5, 50, 20)
    render_delay = st.slider("Pause (s)", 0.0, 0.1, 0.0, step=0.01)

# Simulation Loop
run = st.toggle("▶️ Activer le Flux Cognitif", value=False)

while run:
    # Appliquer les paramètres dynamiques
    st.session_state.model.model.cfg['doubt']['tau_u'] = tau_u
    if rtn_noise:
        st.session_state.model.model.cfg['noise']['sigma_v'] = 0.15 
    
    # Simulation massive avant affichage
    for _ in range(steps_per_frame):
        st.session_state.model.step(I_stimulus=i_stim)
    
    v = st.session_state.model.v
    entropy = st.session_state.model.calculate_entropy()
    gini = calculate_gini(v)
    
    st.session_state.h_history.append(entropy)
    st.session_state.gini_history.append(gini)
    if len(st.session_state.h_history) > 100: 
        st.session_state.h_history.pop(0)
        st.session_state.gini_history.pop(0)

    # 1. Map (Grid or Graph)
    if st.session_state.model.use_stencil:
        current_model_size = st.session_state.model.size
        try:
            v_grid = v.reshape((current_model_size, current_model_size))
            fig_map = px.imshow(v_grid, color_continuous_scale="RdBu_r", zmin=-2, zmax=2, 
                                 template="plotly_dark")
        except ValueError:
            st.error("⚠️ Erreur de taille. Réinitialise la grille.")
            run = False
            break
    else:
        fig_map = px.scatter(x=np.cos(np.linspace(0, 2*np.pi, len(v))), 
                             y=np.sin(np.linspace(0, 2*np.pi, len(v))), 
                             color=v, color_continuous_scale="RdBu_r")
    
    fig_map.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=450)
    placeholder_map.plotly_chart(fig_map, use_container_width=True)

    # 2. Radar de Justice (Plotly)
    categories = ['Entropie', 'Gini (Inv)', 'Doute (u)']
    values = [entropy/2.32, 1-gini, np.mean(st.session_state.model.model.u)]
    fig_radar = go.Figure(data=go.Scatterpolar(r=values, theta=categories, fill='toself'))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), 
                            showlegend=False, template="plotly_dark", height=280, margin=dict(l=40, r=40, t=20, b=20))
    placeholder_radar.plotly_chart(fig_radar, use_container_width=True)

    # 3. Metrics
    m1.metric("Entropie (Diversité)", f"{entropy:.4f}")
    m2.metric("Inégalité (Gini)", f"{gini:.4f}")
    m3.metric("Doute Moyen (u)", f"{np.mean(st.session_state.model.model.u):.4f}")
    
    # 4. Evolution (Streamlit Native Chart - MUCH FASTER)
    hist_data = np.array([st.session_state.h_history, st.session_state.gini_history]).T
    placeholder_chart.line_chart(hist_data, height=200)

    if not run: break
    if render_delay > 0:
        time.sleep(render_delay)
    else:
        # Petit sleep forcé pour laisser souffler le navigateur
        time.sleep(0.001)

st.info("💡 Astuce : Augmente le stimulus pour voir si les 'Bulles de Doute' s'effondrent ou résistent.")
