"""Topological Data Analysis of dialectal eigenspectral space.

Applies persistent homology to the point cloud of dialect eigenspectra
to discover:
- H₀ (connected components): How many truly distinct dialect families?
- H₁ (loops): Circular contact relationships impossible in tree models
- H₂ (voids): "Impossible dialects" — linguistically incoherent regions

Uses ripser if available, otherwise falls back to a pure-numpy
implementation using Union-Find for H₀ and cycle detection for H₁.
"""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import pdist, squareform

from eigendialectos.types import PersistenceResult

logger = logging.getLogger(__name__)

try:
    import ripser as _ripser

    _HAS_RIPSER = True
except ImportError:
    _HAS_RIPSER = False
    logger.debug("ripser not available, using fallback implementation")


class _UnionFind:
    """Union-Find (Disjoint Set Union) data structure for H₀ computation."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n
        self.n_components = n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> bool:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        self.n_components -= 1
        return True


class PersistentHomologyAnalysis:
    """TDA of dialectal eigenspectral point clouds.

    Each dialect is a point in R^k where k = number of eigenvalues.
    Persistent homology reveals the topological structure of this space.
    """

    def __init__(self, max_dimension: int = 2) -> None:
        self.max_dim = max_dimension

    def compute(
        self,
        eigenspectra: npt.NDArray[np.float64],
        labels: list[str] | None = None,
    ) -> PersistenceResult:
        """Compute persistent homology of eigenspectral point cloud.

        Parameters
        ----------
        eigenspectra : ndarray, shape (n_dialects, n_eigenvalues)
            Each row is a dialect's eigenvalue spectrum (uses magnitudes).
        labels : list[str] or None
            Dialect names for interpretation.

        Returns
        -------
        PersistenceResult
        """
        data = np.abs(eigenspectra.astype(np.float64))
        n = data.shape[0]

        if n < 2:
            return PersistenceResult(
                diagrams={0: np.array([[0.0, np.inf]])},
                betti_numbers={0: 1},
                persistence_entropy=0.0,
            )

        if _HAS_RIPSER:
            diagrams = self._compute_ripser(data)
        else:
            diagrams = self._compute_fallback(data)

        # Compute Betti numbers (number of features that persist to infinity)
        betti_numbers: dict[int, int] = {}
        for dim, dgm in diagrams.items():
            if dgm.shape[0] == 0:
                betti_numbers[dim] = 0
                continue
            # Features with death == inf are persistent
            betti_numbers[dim] = int(np.sum(np.isinf(dgm[:, 1])))

        # Persistence entropy
        entropy = self._persistence_entropy(diagrams)

        logger.info(
            "Persistent homology: %d points, Betti numbers = %s, entropy = %.4f",
            n,
            betti_numbers,
            entropy,
        )

        return PersistenceResult(
            diagrams=diagrams,
            betti_numbers=betti_numbers,
            persistence_entropy=entropy,
        )

    def _compute_ripser(
        self,
        data: npt.NDArray[np.float64],
    ) -> dict[int, npt.NDArray[np.float64]]:
        """Compute using ripser library."""
        result = _ripser.ripser(data, maxdim=self.max_dim)
        diagrams: dict[int, npt.NDArray[np.float64]] = {}
        for dim, dgm in enumerate(result["dgms"]):
            diagrams[dim] = np.array(dgm, dtype=np.float64)
        return diagrams

    def _compute_fallback(
        self,
        data: npt.NDArray[np.float64],
    ) -> dict[int, npt.NDArray[np.float64]]:
        """Fallback implementation using Union-Find for H₀ and edge-based H₁.

        H₀: Track connected components as edges are added in order of length.
        H₁: Detect cycles by finding edges that connect already-connected components.
        """
        n = data.shape[0]
        dist_matrix = squareform(pdist(data))

        # Get all edges sorted by distance
        edges: list[tuple[float, int, int]] = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((dist_matrix[i, j], i, j))
        edges.sort(key=lambda x: x[0])

        # === H₀: Connected components via Union-Find ===
        uf = _UnionFind(n)
        h0_pairs: list[list[float]] = []

        # Each point starts as its own component (born at distance 0)
        # Components die when merged
        component_birth: dict[int, float] = {i: 0.0 for i in range(n)}

        for dist, u, v in edges:
            ru, rv = uf.find(u), uf.find(v)
            if ru != rv:
                # One component dies (the younger one)
                birth_u = component_birth.get(ru, 0.0)
                birth_v = component_birth.get(rv, 0.0)
                # Kill the younger (higher birth) component
                dying = max(birth_u, birth_v)
                surviving = min(birth_u, birth_v)
                h0_pairs.append([dying, dist])
                uf.union(u, v)
                # The new root after union
                new_root = uf.find(u)
                # Remove old roots, set new root's birth
                component_birth.pop(ru, None)
                component_birth.pop(rv, None)
                component_birth[new_root] = surviving

        # Remaining components persist to infinity
        for root, birth in component_birth.items():
            h0_pairs.append([birth, np.inf])

        h0 = np.array(h0_pairs, dtype=np.float64) if h0_pairs else np.zeros((0, 2))

        # === H₁: Cycle detection ===
        # Reset union-find and look for edges that create cycles
        uf2 = _UnionFind(n)
        h1_pairs: list[list[float]] = []

        for dist, u, v in edges:
            ru, rv = uf2.find(u), uf2.find(v)
            if ru == rv:
                # This edge creates a cycle — H₁ feature born at this distance
                # Death: estimate as next scale where cycle becomes trivial
                # For simple approximation, use 2x birth distance
                h1_pairs.append([dist, dist * 2.0])
            else:
                uf2.union(u, v)

        h1 = np.array(h1_pairs, dtype=np.float64) if h1_pairs else np.zeros((0, 2))

        # === H₂: placeholder (requires more sophisticated computation) ===
        h2 = np.zeros((0, 2), dtype=np.float64)

        diagrams: dict[int, npt.NDArray[np.float64]] = {0: h0, 1: h1}
        if self.max_dim >= 2:
            diagrams[2] = h2

        return diagrams

    def _persistence_entropy(
        self,
        diagrams: dict[int, npt.NDArray[np.float64]],
    ) -> float:
        """Compute persistence entropy across all dimensions.

        H = -Σ p_i log(p_i) where p_i = persistence_i / total_persistence
        """
        all_lifetimes: list[float] = []

        for dim, dgm in diagrams.items():
            if dgm.shape[0] == 0:
                continue
            for birth, death in dgm:
                if np.isinf(death):
                    continue  # Skip infinite-persistence features
                lifetime = death - birth
                if lifetime > 0:
                    all_lifetimes.append(lifetime)

        if not all_lifetimes:
            return 0.0

        lifetimes = np.array(all_lifetimes, dtype=np.float64)
        total = lifetimes.sum()
        if total < 1e-15:
            return 0.0

        probs = lifetimes / total
        # Filter out zeros for log
        probs = probs[probs > 0]
        entropy = -float(np.sum(probs * np.log(probs)))

        return entropy

    def interpret(
        self,
        result: PersistenceResult,
        labels: list[str] | None = None,
    ) -> dict[str, object]:
        """Linguistic interpretation of topological features.

        Parameters
        ----------
        result : PersistenceResult
        labels : optional dialect names

        Returns
        -------
        dict with interpretation keys:
        - 'n_dialect_families': from H₀ persistent components
        - 'circular_contacts': from H₁ loops
        - 'impossible_regions': from H₂ voids
        - 'summary': text summary
        """
        h0_persistent = result.betti_numbers.get(0, 0)
        h1_persistent = result.betti_numbers.get(1, 0)
        h2_persistent = result.betti_numbers.get(2, 0)

        # Count significant (non-trivial) H₀ features
        h0_dgm = result.diagrams.get(0, np.zeros((0, 2)))
        significant_h0 = 0
        if h0_dgm.shape[0] > 0:
            lifetimes = h0_dgm[:, 1] - h0_dgm[:, 0]
            # Replace inf with a large finite value for thresholding
            lifetimes = np.where(np.isinf(lifetimes), np.nanmax(lifetimes[~np.isinf(lifetimes)]) * 2 if np.any(~np.isinf(lifetimes)) else 1.0, lifetimes)
            if len(lifetimes) > 0:
                median_lifetime = float(np.median(lifetimes))
                significant_h0 = int(np.sum(lifetimes > median_lifetime))

        # Count significant H₁ features
        h1_dgm = result.diagrams.get(1, np.zeros((0, 2)))
        significant_h1 = 0
        if h1_dgm.shape[0] > 0:
            lifetimes = h1_dgm[:, 1] - h1_dgm[:, 0]
            lifetimes = lifetimes[~np.isinf(lifetimes)]
            if len(lifetimes) > 0:
                median_lt = float(np.median(lifetimes))
                significant_h1 = int(np.sum(lifetimes > median_lt))

        summary_parts = [
            f"H₀: {h0_persistent} persistent component(s) → {max(h0_persistent, significant_h0)} distinct dialect families",
            f"H₁: {significant_h1} significant loop(s) → circular contact relationships",
        ]
        if self.max_dim >= 2:
            summary_parts.append(
                f"H₂: {h2_persistent} void(s) → impossible dialect regions"
            )
        summary_parts.append(f"Persistence entropy: {result.persistence_entropy:.4f}")

        return {
            "n_dialect_families": max(h0_persistent, significant_h0),
            "circular_contacts": significant_h1,
            "impossible_regions": h2_persistent,
            "persistence_entropy": result.persistence_entropy,
            "summary": "\n".join(summary_parts),
        }
