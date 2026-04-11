"""
Mem4ristor Visualization Module.

Provides ready-to-use plotting functions for:
  - Entropy time series
  - Doubt (u) heatmaps on 2D lattices
  - Phase portraits (v vs w)
  - State distribution bar charts
  - Network-level dashboard (all of the above in one figure)

Usage:
    from mem4ristor.viz import plot_entropy_trace, plot_doubt_map, dashboard
    from mem4ristor.core import Mem4Network

    net = Mem4Network(size=10)
    history = run_simulation(net, steps=2000)
    dashboard(history, net)

All functions return (fig, ax) tuples so callers can further customise.
Requires matplotlib (soft dependency — ImportError gives a clear message).
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.gridspec import GridSpec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes


def _require_mpl():
    if not HAS_MPL:
        raise ImportError(
            "matplotlib is required for mem4ristor.viz. "
            "Install it with: pip install matplotlib"
        )


# ── Colour palette ────────────────────────────────────────────────
STATE_COLORS = {
    "Oracle":    "#6C3483",   # deep purple
    "Intuition": "#2E86C1",   # blue
    "Uncertain": "#27AE60",   # green
    "Probable":  "#F39C12",   # amber
    "Certitude": "#E74C3C",   # red
}
STATE_NAMES = list(STATE_COLORS.keys())
BIN_EDGES = [-np.inf, -1.5, -0.8, 0.8, 1.5, np.inf]


# ── Helper: record history during simulation ──────────────────────
class SimHistory:
    """Lightweight recorder — call .record(net) each step."""

    def __init__(self):
        self.entropy: List[float] = []
        self.v_snapshots: List[np.ndarray] = []
        self.w_snapshots: List[np.ndarray] = []
        self.u_snapshots: List[np.ndarray] = []
        self.state_counts: List[np.ndarray] = []

    def record(self, net) -> None:
        """Record current state from a Mem4Network instance."""
        model = net.model
        self.entropy.append(model.calculate_entropy())
        self.v_snapshots.append(model.v.copy())
        self.w_snapshots.append(model.w.copy())
        self.u_snapshots.append(model.u.copy())

        counts, _ = np.histogram(model.v, bins=BIN_EDGES)
        self.state_counts.append(counts)

    @property
    def steps(self) -> int:
        return len(self.entropy)


# ── Plot functions ────────────────────────────────────────────────
def plot_entropy_trace(
    entropy: List[float],
    ax: Optional["Axes"] = None,
    title: str = "Shannon Entropy H(t)",
    show_hmax: bool = True,
) -> Tuple["Figure", "Axes"]:
    """Plot entropy over time with optional H_max reference line."""
    _require_mpl()
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))
    else:
        fig = ax.figure

    steps = np.arange(len(entropy))
    ax.plot(steps, entropy, color="#2E86C1", linewidth=0.8, alpha=0.9)

    if show_hmax:
        h_max = np.log2(5)
        ax.axhline(h_max, color="#E74C3C", linestyle="--", linewidth=0.7,
                    label=f"$H_{{max}} = \\log_2 5 \\approx {h_max:.2f}$")
        ax.legend(fontsize=8)

    ax.set_xlabel("Step")
    ax.set_ylabel("H (bits)")
    ax.set_title(title, fontsize=10)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_doubt_map(
    u: np.ndarray,
    grid_size: Optional[int] = None,
    ax: Optional["Axes"] = None,
    title: str = "Constitutional Doubt $u$",
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> Tuple["Figure", "Axes"]:
    """Heatmap of doubt values on a 2D grid."""
    _require_mpl()
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure

    if grid_size is None:
        grid_size = int(np.sqrt(len(u)))

    u_grid = u[:grid_size**2].reshape(grid_size, grid_size)

    im = ax.imshow(u_grid, cmap="magma", vmin=vmin, vmax=vmax,
                   interpolation="nearest", origin="lower")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    return fig, ax


def plot_phase_portrait(
    v: np.ndarray,
    w: np.ndarray,
    heretic_mask: Optional[np.ndarray] = None,
    ax: Optional["Axes"] = None,
    title: str = "Phase Portrait ($v$ vs $w$)",
) -> Tuple["Figure", "Axes"]:
    """Scatter plot of v vs w, optionally colouring heretics."""
    _require_mpl()
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure

    if heretic_mask is not None:
        normal = ~heretic_mask
        ax.scatter(v[normal], w[normal], s=8, alpha=0.6,
                   color="#2E86C1", label="Normal", zorder=2)
        ax.scatter(v[heretic_mask], w[heretic_mask], s=14, alpha=0.8,
                   color="#E74C3C", marker="^", label="Heretic", zorder=3)
        ax.legend(fontsize=8)
    else:
        ax.scatter(v, w, s=8, alpha=0.6, color="#2E86C1", zorder=2)

    ax.set_xlabel("$v$ (cognitive potential)")
    ax.set_ylabel("$w$ (recovery)")
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axvline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)

    return fig, ax


def plot_state_distribution(
    counts: np.ndarray,
    ax: Optional["Axes"] = None,
    title: str = "Cognitive State Distribution",
) -> Tuple["Figure", "Axes"]:
    """Bar chart of the 5 cognitive states."""
    _require_mpl()
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3))
    else:
        fig = ax.figure

    colors = list(STATE_COLORS.values())
    total = np.sum(counts) or 1
    pct = counts / total * 100

    bars = ax.bar(STATE_NAMES, pct, color=colors, edgecolor="white", linewidth=0.5)

    for bar, p in zip(bars, pct):
        if p > 3:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{p:.0f}%", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("% of units")
    ax.set_title(title, fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    return fig, ax


def plot_v_heatmap(
    v: np.ndarray,
    grid_size: Optional[int] = None,
    ax: Optional["Axes"] = None,
    title: str = "Cognitive Potential $v$",
) -> Tuple["Figure", "Axes"]:
    """Heatmap of v values on a 2D grid."""
    _require_mpl()
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure

    if grid_size is None:
        grid_size = int(np.sqrt(len(v)))

    v_grid = v[:grid_size**2].reshape(grid_size, grid_size)

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "cognitive",
        ["#6C3483", "#2E86C1", "#27AE60", "#F39C12", "#E74C3C"],
    )
    im = ax.imshow(v_grid, cmap=cmap, vmin=-2.5, vmax=2.5,
                   interpolation="nearest", origin="lower")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    return fig, ax


# ── Dashboard ─────────────────────────────────────────────────────
def dashboard(
    history: SimHistory,
    net=None,
    step: int = -1,
    figsize: Tuple[float, float] = (14, 8),
    suptitle: Optional[str] = None,
) -> Tuple["Figure", list]:
    """
    Full 4-panel dashboard:
      [entropy trace] [state distribution]
      [doubt map    ] [phase portrait    ]

    Args:
        history: SimHistory with recorded data.
        net: Mem4Network (for heretic_mask; optional).
        step: Which snapshot to show (-1 = last).
        figsize: Figure size.
        suptitle: Optional super-title.

    Returns:
        (fig, [ax1, ax2, ax3, ax4])
    """
    _require_mpl()
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # 1. Entropy trace
    plot_entropy_trace(history.entropy, ax=ax1)

    # 2. State distribution at given step
    counts = history.state_counts[step]
    plot_state_distribution(counts, ax=ax2,
                            title=f"States @ step {step if step >= 0 else history.steps - 1}")

    # 3. Doubt map
    u = history.u_snapshots[step]
    grid_size = net.size if net is not None else None
    plot_doubt_map(u, grid_size=grid_size, ax=ax3)

    # 4. Phase portrait
    v = history.v_snapshots[step]
    w = history.w_snapshots[step]
    heretic_mask = net.model.heretic_mask if net is not None else None
    plot_phase_portrait(v, w, heretic_mask=heretic_mask, ax=ax4)

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=1.01)

    return fig, [ax1, ax2, ax3, ax4]
