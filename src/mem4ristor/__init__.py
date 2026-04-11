"""
Mem4ristor v3.1.1 — Neuromorphic-inspired diversity-preserving cognitive architecture.

Modules:
    core        — Mem4ristorV3 engine & Mem4Network (lattice / adjacency)
    config      — Typed dataclass configuration
    sensory     — SensoryFrontend (image → stimulus)
    cortex      — LearnableCortex (learnable decision layer)
    hierarchy   — HierarchicalChimera (multi-scale chimera states)
    symbiosis   — CreativeProjector (creative divergence mapping)
    arena       — GladiatorMem4ristor (competitive evaluation)
    inception   — DreamVisualizer (internal state visualization)
    viz         — Plotting utilities (entropy, doubt maps, phase portraits)
"""

# Core engine
from .core import Mem4ristorV3, Mem4ristorV2, Mem4Network

# Configuration
from .config import Mem4Config, DynamicsConfig, CouplingConfig, DoubtConfig, NoiseConfig

# Extension modules
from .sensory import SensoryFrontend
from .cortex import LearnableCortex
from .hierarchy import HierarchicalChimera
from .symbiosis import CreativeProjector
from .arena import GladiatorMem4ristor
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

__version__ = "3.1.1"

__all__ = [
    # Core
    "Mem4ristorV3",
    "Mem4ristorV2",
    "Mem4Network",
    # Config
    "Mem4Config",
    "DynamicsConfig",
    "CouplingConfig",
    "DoubtConfig",
    "NoiseConfig",
    # Extensions
    "SensoryFrontend",
    "LearnableCortex",
    "HierarchicalChimera",
    "CreativeProjector",
    "GladiatorMem4ristor",
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
