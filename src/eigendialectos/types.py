"""Core data structures for EigenDialectos."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from eigendialectos.constants import DialectCode


@dataclass
class DialectSample:
    """A single text sample annotated with dialect metadata."""

    text: str
    dialect_code: DialectCode
    source_id: str
    confidence: float
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class CorpusSlice:
    """A collection of dialect samples for a single variety."""

    samples: list[DialectSample]
    dialect_code: DialectCode

    @property
    def stats(self) -> dict[str, object]:
        """Compute basic corpus statistics."""
        count = len(self.samples)
        if count == 0:
            return {"count": 0, "avg_length": 0.0, "min_length": 0, "max_length": 0}
        lengths = [len(s.text) for s in self.samples]
        return {
            "count": count,
            "avg_length": sum(lengths) / count,
            "min_length": min(lengths),
            "max_length": max(lengths),
        }


@dataclass
class EmbeddingMatrix:
    """Dense embedding matrix for a dialect vocabulary."""

    data: npt.NDArray[np.float64]
    vocab: list[str]
    dialect_code: DialectCode

    @property
    def dim(self) -> int:
        """Embedding dimensionality (number of columns)."""
        return int(self.data.shape[1])


@dataclass
class TransformationMatrix:
    """Linear mapping between two dialect embedding spaces."""

    data: npt.NDArray[np.float64]
    source_dialect: DialectCode
    target_dialect: DialectCode
    regularization: float

    @property
    def shape(self) -> tuple[int, ...]:
        """Matrix shape."""
        return tuple(self.data.shape)


@dataclass
class EigenDecomposition:
    """Eigendecomposition of a dialect transformation matrix."""

    eigenvalues: npt.NDArray[np.complex128]
    eigenvectors: npt.NDArray[np.complex128]
    eigenvectors_inv: npt.NDArray[np.complex128]
    dialect_code: DialectCode

    @property
    def rank(self) -> int:
        """Effective rank (number of non-negligible eigenvalues)."""
        return int(np.sum(np.abs(self.eigenvalues) > 1e-10))


@dataclass
class DialectalSpectrum:
    """Spectral profile summarising dialectal variation."""

    eigenvalues_sorted: npt.NDArray[np.float64]
    entropy: float
    dialect_code: DialectCode

    @property
    def cumulative_energy(self) -> npt.NDArray[np.float64]:
        """Cumulative proportion of total spectral energy."""
        total = np.sum(self.eigenvalues_sorted)
        if total == 0:
            return np.zeros_like(self.eigenvalues_sorted)
        return np.cumsum(self.eigenvalues_sorted) / total


@dataclass
class TensorDialectal:
    """Multi-dialect tensor representation."""

    data: npt.NDArray[np.float64]
    dialect_codes: list[DialectCode]

    @property
    def shape(self) -> tuple[int, ...]:
        """Tensor shape."""
        return tuple(self.data.shape)


@dataclass
class ExperimentResult:
    """Captured output of a single experiment run."""

    experiment_id: str
    metrics: dict[str, object]
    artifact_paths: list[str]
    timestamp: str
    config: dict[str, object]


# -----------------------------------------------------------------------
# V2 types: multi-level parsing, spectral stack, geometry, compiler
# -----------------------------------------------------------------------


@dataclass
class ParsedText:
    """Multi-level parsed representation of a text."""

    original: str
    morphemes: list[list[str]]         # L1: morpheme segmentation per token
    words: list[str]                   # L2: token list
    phrases: list[list[str]]           # L3: phrase chunks (lists of tokens)
    sentences: list[str]               # L4: sentence strings
    discourse: dict[str, Any]          # L5: discourse features


@dataclass
class LevelEmbedding:
    """Embedding for a single linguistic level."""

    level: int
    vectors: npt.NDArray[np.float64]    # (n_units, dim)
    labels: list[str]                   # unit labels
    vocabulary: dict[str, int]          # label -> index


@dataclass
class SpectralStackResult:
    """Result of spectral stack transform across all levels."""

    level_results: dict[int, EigenDecomposition]
    level_transforms: dict[int, npt.NDArray[np.float64]]
    alphas: dict[int, float]


@dataclass
class LieAlgebraResult:
    """Result of Lie algebra analysis of dialectal transformations."""

    generators: dict[str, npt.NDArray[np.complex128]]
    commutators: dict[tuple[str, str], npt.NDArray[np.complex128]]
    commutator_norms: dict[tuple[str, str], float]


@dataclass
class RiemannianResult:
    """Result of Riemannian geometry analysis on dialect manifold."""

    metric_tensors: dict[str, npt.NDArray[np.float64]]
    geodesic_distances: npt.NDArray[np.float64]
    ricci_curvatures: dict[str, float]
    dialect_labels: list[str] = field(default_factory=list)


@dataclass
class PersistenceResult:
    """Result of topological data analysis (persistent homology)."""

    diagrams: dict[int, npt.NDArray[np.float64]]
    betti_numbers: dict[int, int]
    persistence_entropy: float


@dataclass
class EigenFieldResult:
    """Result of eigenvalue field estimation via Gaussian Processes."""

    coordinates: npt.NDArray[np.float64]        # (n_dialects, 2)
    eigenvalue_surfaces: npt.NDArray[np.float64] # (n_eigenvalues, n_grid, n_grid)
    gp_lengthscales: npt.NDArray[np.float64]     # per eigenvalue
    uncertainties: npt.NDArray[np.float64]        # (n_eigenvalues, n_grid, n_grid)
    grid_lat: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    grid_lon: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([]))


@dataclass
class FisherResult:
    """Result of Fisher Information analysis."""

    fim: npt.NDArray[np.float64]
    fim_eigenvalues: npt.NDArray[np.float64]
    fim_eigenvectors: npt.NDArray[np.float64]
    most_diagnostic: list[tuple[str, float]]


@dataclass
class SDCResult:
    """Result of Spectral Dialectal Compilation."""

    input_text: str
    output_text: str
    source_variety: str
    target_variety: str
    alphas: dict[int, float]
    change_log: list[dict[str, Any]]
