"""
Mem4ristor v4.0.0 — Neuromorphic-inspired diversity-preserving cognitive architecture.

Modules:
    core        — Mem4ristorV3 engine & Mem4Network (lattice / adjacency)
    config      — Typed dataclass configuration
    metrics     — Entropy, LZ complexity, pairwise synchrony
    graph_utils — Canonical topology generators (make_ba, make_er, make_lattice_adj)
    sensory     — SensoryFrontend (image → stimulus)
    cortex      — LearnableCortex (learnable decision layer)
    symbiosis   — CreativeProjector (creative divergence mapping)
    inception   — DreamVisualizer (internal state visualization)
    viz         — Plotting utilities (entropy, doubt maps, phase portraits)
"""

# Core engine
from .core import Mem4ristorV3, Mem4ristorV2, Mem4Network

# Configuration
from .config import Mem4Config, DynamicsConfig, CouplingConfig, DoubtConfig, NoiseConfig

# Graph utilities
from .graph_utils import make_ba, make_er, make_lattice_adj

# Metrics
from .metrics import (
    calculate_continuous_entropy,
    calculate_cognitive_entropy,
    get_cognitive_states,
    calculate_temporal_lz_complexity,
    calculate_pairwise_synchrony,
)

# Extension modules
from .sensory import SensoryFrontend
from .cortex import LearnableCortex
from .symbiosis import CreativeProjector
from .inception import DreamVisualizer

# Visualization (soft dependency on matplotlib)
try:
    from .viz import (
        SimHistory,
        plot_entropy_trace,
        plot_doubt_map,
        plot_phase_portrait,
        plot_state_distribution,
        plot_v_heatmap,
        dashboard,
    )
except ImportError:
    pass  # matplotlib not installed — viz functions unavailable

__version__ = "4.0.0"

__all__ = [
    # Graph utilities
    "make_ba",
    "make_er",
    "make_lattice_adj",
    # Core
    "Mem4ristorV3",
    "Mem4ristorV2",
    "Mem4Network",
    # Metrics
    "calculate_continuous_entropy",
    "calculate_cognitive_entropy",
    "get_cognitive_states",
    "calculate_temporal_lz_complexity",
    "calculate_pairwise_synchrony",
    # Config
    "Mem4Config",
    "DynamicsConfig",
    "CouplingConfig",
    "DoubtConfig",
    "NoiseConfig",
    # Extensions
    "SensoryFrontend",
    "LearnableCortex",
    "CreativeProjector",
    "DreamVisualizer",
    # Viz
    "SimHistory",
    "plot_entropy_trace",
    "plot_doubt_map",
    "plot_phase_portrait",
    "plot_state_distribution",
    "plot_v_heatmap",
    "dashboard",
]
