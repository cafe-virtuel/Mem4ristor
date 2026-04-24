"""
Smoke tests for trajectory-based coordination metrics (P1.5bis).

Verifies mathematical properties without running full simulations:
  - LZ complexity of a constant sequence = 0 (maximally structured)
  - LZ complexity of a random sequence approaches 1 (maximally random)
  - Synchrony of identical traces = 1
  - Synchrony of independent traces ≈ 0
  - Both functions handle edge cases gracefully
"""
import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mem4ristor.metrics import (
    calculate_temporal_lz_complexity,
    calculate_pairwise_synchrony,
    _lz76_phrases,
)


# ── LZ76 helper ──────────────────────────────────────────────────────────────

def test_lz76_constant_sequence():
    """A constant sequence has far fewer phrases than a fully novel sequence.
    LZ76 on 'aaa...a' grows as ~log2(n), not linearly.
    """
    n = 32
    c_const = _lz76_phrases("1" * n)
    c_novel = _lz76_phrases("".join(str(i % 10) for i in range(n)))
    assert c_const < c_novel, (
        f"Constant ({c_const}) should have fewer phrases than novel ({c_novel})"
    )


def test_lz76_increasing_sequence():
    """Each new character is novel → many phrases."""
    c = _lz76_phrases("123456789")
    assert c > 3


def test_lz76_empty():
    assert _lz76_phrases("") == 0


def test_lz76_single_char():
    assert _lz76_phrases("7") == 1


# ── calculate_temporal_lz_complexity ─────────────────────────────────────────

def test_lz_constant_history():
    """All nodes frozen at same value → maximally structured, LZ ≈ 0."""
    v = np.ones((300, 50))
    lz = calculate_temporal_lz_complexity(v)
    assert lz < 0.1, f"Expected near 0, got {lz}"


def test_lz_random_history_higher_than_structured():
    """Random history should have higher LZ than a periodic history."""
    rng = np.random.RandomState(0)
    T, N = 300, 50
    v_random = rng.standard_normal((T, N))
    t = np.linspace(0, 4 * np.pi, T)
    v_periodic = np.outer(np.sin(t), np.ones(N))

    lz_random = calculate_temporal_lz_complexity(v_random)
    lz_periodic = calculate_temporal_lz_complexity(v_periodic)
    assert lz_random > lz_periodic, (
        f"Random LZ ({lz_random:.3f}) should exceed periodic LZ ({lz_periodic:.3f})"
    )


def test_lz_output_bounded():
    """Normalized LZ must be positive for non-trivial sequences."""
    rng = np.random.RandomState(42)
    v = rng.standard_normal((200, 20))
    lz = calculate_temporal_lz_complexity(v)
    assert lz > 0
    assert np.isfinite(lz)


def test_lz_short_history():
    """Single timestep should not crash and returns 0."""
    v = np.zeros((1, 10))
    assert calculate_temporal_lz_complexity(v) == 0.0


# ── calculate_pairwise_synchrony ─────────────────────────────────────────────

def test_synchrony_identical_traces():
    """Identical traces → synchrony = 1."""
    T, N = 300, 10
    rng = np.random.RandomState(7)
    base = rng.standard_normal(T)
    v = np.outer(base, np.ones(N))
    sync = calculate_pairwise_synchrony(v)
    assert abs(sync - 1.0) < 1e-6, f"Expected 1.0, got {sync}"


def test_synchrony_independent_traces_near_zero():
    """Independent random traces → synchrony ≈ 0 (converges as T grows)."""
    rng = np.random.RandomState(99)
    v = rng.standard_normal((2000, 50))
    sync = calculate_pairwise_synchrony(v)
    assert abs(sync) < 0.1, f"Expected near 0, got {sync}"


def test_synchrony_anti_correlated():
    """Two anti-correlated groups → synchrony near -1 when N=2."""
    T = 300
    rng = np.random.RandomState(3)
    base = rng.standard_normal(T)
    v = np.column_stack([base, -base])
    sync = calculate_pairwise_synchrony(v)
    assert abs(sync - (-1.0)) < 1e-6, f"Expected -1.0, got {sync}"


def test_synchrony_output_bounded():
    rng = np.random.RandomState(1)
    v = rng.standard_normal((300, 100))
    sync = calculate_pairwise_synchrony(v)
    assert -1.0 <= sync <= 1.0
    assert np.isfinite(sync)


def test_synchrony_short_history():
    v = np.zeros((1, 10))
    assert calculate_pairwise_synchrony(v) == 0.0


def test_synchrony_single_node():
    v = np.random.standard_normal((100, 1))
    assert calculate_pairwise_synchrony(v) == 0.0
