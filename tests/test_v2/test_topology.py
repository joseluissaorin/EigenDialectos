"""Tests for persistent homology."""

from __future__ import annotations

import numpy as np
import pytest

from eigendialectos.topology.persistent_homology import PersistentHomologyAnalysis


class TestPersistentHomology:
    def test_single_point(self):
        ph = PersistentHomologyAnalysis()
        data = np.array([[1.0, 2.0, 3.0]])
        result = ph.compute(data)
        assert result.betti_numbers[0] == 1

    def test_two_distant_points(self):
        ph = PersistentHomologyAnalysis(max_dimension=1)
        data = np.array([[0, 0, 0], [100, 100, 100]], dtype=np.float64)
        result = ph.compute(data)
        # H₀: eventually they connect, so 1 persistent component
        assert result.betti_numbers[0] >= 1

    def test_persistence_entropy_nonnegative(self):
        rng = np.random.default_rng(42)
        ph = PersistentHomologyAnalysis()
        data = rng.standard_normal((8, 5))
        result = ph.compute(data)
        assert result.persistence_entropy >= 0.0

    def test_well_separated_clusters(self):
        rng = np.random.default_rng(42)
        ph = PersistentHomologyAnalysis()
        # 3 clusters far apart
        data = np.vstack([
            rng.standard_normal((5, 3)) + [100, 0, 0],
            rng.standard_normal((5, 3)) + [0, 100, 0],
            rng.standard_normal((5, 3)) + [0, 0, 100],
        ])
        result = ph.compute(data)
        # Should detect multiple components
        h0_dgm = result.diagrams[0]
        assert h0_dgm.shape[0] >= 3  # At least 3 features (some die late)

    def test_interpret(self):
        rng = np.random.default_rng(42)
        ph = PersistentHomologyAnalysis()
        data = rng.standard_normal((8, 5))
        result = ph.compute(data)
        interp = ph.interpret(result)
        assert "n_dialect_families" in interp
        assert "circular_contacts" in interp
        assert "summary" in interp
