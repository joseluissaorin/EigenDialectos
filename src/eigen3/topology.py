"""Persistent homology (pure numpy, no external TDA libraries).

Computes H0 persistence via Kruskal-style filtration on a distance
matrix, plus Betti numbers, persistence entropy, and interpretation.
"""

from __future__ import annotations

import logging

import numpy as np

from eigen3.types import PersistenceDiagram

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Union-Find (disjoint set) for H0 computation
# ---------------------------------------------------------------------------

class _UnionFind:
    """Standard disjoint-set with union-by-rank and path compression."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n
        self.n_components = n

    def find(self, x: int) -> int:
        """Find root with path compression."""
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        # Path compression
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, x: int, y: int) -> bool:
        """Merge sets containing x and y. Returns True if a merge happened."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        # Union by rank
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        self.n_components -= 1
        return True

    def connected(self, x: int, y: int) -> bool:
        return self.find(x) == self.find(y)


# ---------------------------------------------------------------------------
# Persistent homology
# ---------------------------------------------------------------------------

def persistent_homology(
    distance_matrix: np.ndarray,
    max_dim: int = 1,
) -> list[PersistenceDiagram]:
    """Compute persistent homology from a distance matrix.

    Parameters
    ----------
    distance_matrix : np.ndarray
        (n, n) symmetric pairwise distance matrix.
    max_dim : int
        Maximum homological dimension. H0 is always computed.
        H1 (and above) returns an empty diagram (exact Vietoris-Rips
        H1 requires heavy computation; this is a lightweight stub).

    Returns
    -------
    list[PersistenceDiagram]
        One diagram per dimension 0..max_dim.
    """
    n = distance_matrix.shape[0]
    D = distance_matrix.copy()

    # ---- H0: connected components via Kruskal's algorithm ----
    # Extract upper-triangular edges, sort by weight
    edges: list[tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((float(D[i, j]), i, j))
    edges.sort(key=lambda e: e[0])

    uf = _UnionFind(n)
    # Every vertex is born at t=0
    birth_time = {i: 0.0 for i in range(n)}
    # Track which component root maps to birth time
    component_birth: dict[int, float] = {i: 0.0 for i in range(n)}
    h0_pairs: list[tuple[float, float]] = []

    for weight, u, v in edges:
        ru, rv = uf.find(u), uf.find(v)
        if ru == rv:
            continue
        # The younger component dies (born later or arbitrary tie-break)
        birth_u = component_birth[ru]
        birth_v = component_birth[rv]
        # Kill the younger one; if same birth, kill arbitrary
        if birth_u <= birth_v:
            # rv's component dies
            h0_pairs.append((birth_v, weight))
            uf.union(u, v)
            new_root = uf.find(u)
            component_birth[new_root] = birth_u
        else:
            h0_pairs.append((birth_u, weight))
            uf.union(u, v)
            new_root = uf.find(u)
            component_birth[new_root] = birth_v

    # The final surviving component lives forever (birth=0, death=inf)
    h0_pairs.append((0.0, float("inf")))

    h0_bd = np.array(h0_pairs, dtype=np.float64).reshape(-1, 2)
    diagrams = [PersistenceDiagram(dimension=0, birth_death=h0_bd)]

    # ---- H1+ : stub (empty diagrams) ----
    for d in range(1, max_dim + 1):
        empty = np.empty((0, 2), dtype=np.float64)
        diagrams.append(PersistenceDiagram(dimension=d, birth_death=empty))

    return diagrams


# ---------------------------------------------------------------------------
# Betti numbers
# ---------------------------------------------------------------------------

def betti_numbers(
    diagrams: list[PersistenceDiagram],
    threshold: float,
) -> list[int]:
    """Count features alive at a given filtration threshold.

    A feature (b, d) is alive at threshold t if b <= t < d.

    Parameters
    ----------
    diagrams : list[PersistenceDiagram]
        Persistence diagrams (one per dimension).
    threshold : float
        Filtration value.

    Returns
    -------
    list[int]
        Betti number for each dimension.
    """
    result: list[int] = []
    for diag in diagrams:
        bd = diag.birth_death
        if bd.shape[0] == 0:
            result.append(0)
            continue
        alive = (bd[:, 0] <= threshold) & (bd[:, 1] > threshold)
        result.append(int(alive.sum()))
    return result


# ---------------------------------------------------------------------------
# Persistence entropy
# ---------------------------------------------------------------------------

def persistence_entropy(diagram: PersistenceDiagram) -> float:
    """Shannon entropy on feature lifetimes.

    Lifetime l_i = death_i - birth_i (excluding infinite features).
    Entropy = -sum( p_i * log(p_i) ) where p_i = l_i / sum(l).

    Parameters
    ----------
    diagram : PersistenceDiagram
        A single persistence diagram.

    Returns
    -------
    float
        Persistence entropy (0 if no finite features).
    """
    bd = diagram.birth_death
    if bd.shape[0] == 0:
        return 0.0

    # Filter out infinite-death features
    finite_mask = np.isfinite(bd[:, 1])
    bd_finite = bd[finite_mask]
    if bd_finite.shape[0] == 0:
        return 0.0

    lifetimes = bd_finite[:, 1] - bd_finite[:, 0]
    lifetimes = lifetimes[lifetimes > 0]
    if len(lifetimes) == 0:
        return 0.0

    total = lifetimes.sum()
    p = lifetimes / total
    return float(-np.sum(p * np.log(p)))


# ---------------------------------------------------------------------------
# Human-readable interpretation
# ---------------------------------------------------------------------------

def interpret(diagrams: list[PersistenceDiagram]) -> dict:
    """Produce a human-readable summary of persistence diagrams.

    Parameters
    ----------
    diagrams : list[PersistenceDiagram]
        Persistence diagrams (one per dimension).

    Returns
    -------
    dict
        Keys: "n_components", "n_loops", "betti_at_median",
        "entropy_h0", "longest_h0_features", "summary".
    """
    result: dict = {}

    # H0 analysis
    if len(diagrams) > 0:
        h0 = diagrams[0]
        bd = h0.birth_death
        n_total = bd.shape[0]
        n_inf = int(np.isinf(bd[:, 1]).sum()) if n_total > 0 else 0

        result["n_components"] = n_inf  # components surviving to infinity
        result["entropy_h0"] = persistence_entropy(h0)

        # Longest finite features
        finite_mask = np.isfinite(bd[:, 1])
        if finite_mask.any():
            bd_finite = bd[finite_mask]
            lifetimes = bd_finite[:, 1] - bd_finite[:, 0]
            top_k = min(5, len(lifetimes))
            top_idx = np.argsort(-lifetimes)[:top_k]
            result["longest_h0_features"] = [
                {"birth": float(bd_finite[i, 0]), "death": float(bd_finite[i, 1]),
                 "lifetime": float(lifetimes[i])}
                for i in top_idx
            ]

            # Betti at median filtration value
            median_thresh = float(np.median(bd_finite[:, 1]))
            result["betti_at_median"] = betti_numbers(diagrams, median_thresh)
        else:
            result["longest_h0_features"] = []
            result["betti_at_median"] = [n_inf] + [0] * (len(diagrams) - 1)

    # H1 analysis
    if len(diagrams) > 1:
        h1 = diagrams[1]
        result["n_loops"] = h1.birth_death.shape[0]
    else:
        result["n_loops"] = 0

    # Summary text
    summary_parts = [
        f"{result.get('n_components', '?')} connected component(s) survive to infinity.",
        f"H0 persistence entropy: {result.get('entropy_h0', 0.0):.4f}.",
    ]
    if result.get("n_loops", 0) > 0:
        summary_parts.append(f"{result['n_loops']} loop feature(s) detected in H1.")
    else:
        summary_parts.append("No H1 loop features (H1 not computed or trivial).")
    result["summary"] = " ".join(summary_parts)

    return result
