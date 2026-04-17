"""Core dataclasses for eigen3."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class DialectEmbeddings:
    """Per-variety word embedding matrices + shared vocabulary."""

    embeddings: dict[str, np.ndarray]   # variety -> (vocab_size, dim)
    vocab: list[str]
    dim: int

    def __post_init__(self):
        for v, emb in self.embeddings.items():
            assert emb.shape == (len(self.vocab), self.dim), (
                f"{v}: expected ({len(self.vocab)}, {self.dim}), got {emb.shape}"
            )


@dataclass
class TransformationMatrix:
    """W matrix mapping source → target embedding space."""

    W: np.ndarray
    source: str
    target: str
    condition: float

    @property
    def shape(self) -> tuple[int, int]:
        return self.W.shape


@dataclass
class EigenDecomp:
    """Eigendecomposition W = P @ diag(eigenvalues) @ P_inv."""

    P: np.ndarray               # (n, n) eigenvector matrix
    eigenvalues: np.ndarray     # (n,) complex eigenvalues
    P_inv: np.ndarray           # (n, n) inverse of P
    W_original: np.ndarray      # (n, n) original W matrix
    variety: str = ""

    @property
    def n_modes(self) -> int:
        return len(self.eigenvalues)

    @property
    def magnitudes(self) -> np.ndarray:
        return np.abs(self.eigenvalues)


@dataclass
class EigenSpectrum:
    """Sorted eigenvalue spectrum with entropy and effective rank."""

    magnitudes: np.ndarray      # sorted descending by magnitude
    entropy: float              # Shannon entropy on normalized magnitudes
    effective_rank: int         # exp(entropy) rounded


@dataclass
class AlphaVector:
    """Per-mode intensity control vector."""

    values: np.ndarray  # (n_modes,)

    @classmethod
    def from_dict(cls, d: dict[int, float], n_modes: int, default: float = 0.0) -> AlphaVector:
        v = np.full(n_modes, default, dtype=np.float64)
        for idx, val in d.items():
            v[idx] = val
        return cls(values=v)

    @classmethod
    def uniform(cls, n_modes: int, value: float = 1.0) -> AlphaVector:
        return cls(values=np.full(n_modes, value, dtype=np.float64))

    @classmethod
    def zeros(cls, n_modes: int) -> AlphaVector:
        return cls(values=np.zeros(n_modes, dtype=np.float64))

    @classmethod
    def ones(cls, n_modes: int) -> AlphaVector:
        return cls(values=np.ones(n_modes, dtype=np.float64))

    def __len__(self) -> int:
        return len(self.values)


@dataclass
class ChangeEntry:
    """Single word replacement in a compiled transformation."""

    position: int
    original: str
    replacement: str
    confidence: float
    mode_idx: int
    eigenvalue: float


@dataclass
class ScoreResult:
    """Dialect scoring output."""

    probabilities: dict[str, float]
    mode_activations: np.ndarray
    top_dialect: str


@dataclass
class AnalysisResult:
    """Eigenmode analysis output."""

    mode_names: dict[int, str]
    mode_strengths: np.ndarray
    per_word_modes: dict[str, np.ndarray]


@dataclass
class TransformResult:
    """SDC compilation output."""

    text: str
    changes: list[ChangeEntry]
    alpha: Optional[AlphaVector]
    source: str
    target: str


@dataclass
class ComposeResult:
    """Dialect algebra composition output."""

    spectrum: np.ndarray
    W: np.ndarray
    condition: float


@dataclass
class PersistenceDiagram:
    """Topological persistence diagram."""

    dimension: int
    birth_death: np.ndarray     # (n_features, 2)


@dataclass
class NullModelResult:
    """Null model significance testing output."""

    p_values: np.ndarray        # per-mode p-values
    significant_modes: list[int]
    n_permutations: int
