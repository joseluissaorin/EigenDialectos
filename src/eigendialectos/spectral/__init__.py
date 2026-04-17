"""Spectral analysis module for EigenDialectos.

Provides eigendecomposition, spectral entropy, eigenvector analysis,
and distance metrics for dialect transformation matrices.
"""

from eigendialectos.spectral.distance import (
    combined_distance,
    compute_distance_matrix,
    entropy_distance,
    frobenius_distance,
    spectral_distance,
    subspace_distance,
)
from eigendialectos.spectral.eigendecomposition import (
    decompose,
    eigendecompose,
    svd_decompose,
)
from eigendialectos.spectral.eigenspectrum import (
    compare_spectra,
    compute_eigenspectrum,
    rank_k_approximation,
)
from eigendialectos.spectral.eigenvector_analysis import (
    compare_eigenvectors,
    find_shared_axes,
    find_unique_axes,
    interpret_eigenvector,
)
from eigendialectos.spectral.entropy import (
    compare_entropies,
    compute_dialectal_entropy,
)
from eigendialectos.spectral.transformation import (
    compute_all_transforms,
    compute_transformation_matrix,
)
from eigendialectos.spectral.utils import (
    check_condition_number,
    handle_complex_eigenvalues,
    is_orthogonal,
    is_positive_definite,
    regularize_matrix,
    safe_inverse,
    stable_log,
)

__all__ = [
    # utils
    "check_condition_number",
    "regularize_matrix",
    "handle_complex_eigenvalues",
    "stable_log",
    "is_orthogonal",
    "is_positive_definite",
    "safe_inverse",
    # transformation
    "compute_transformation_matrix",
    "compute_all_transforms",
    # eigendecomposition
    "eigendecompose",
    "svd_decompose",
    "decompose",
    # eigenspectrum
    "compute_eigenspectrum",
    "compare_spectra",
    "rank_k_approximation",
    # entropy
    "compute_dialectal_entropy",
    "compare_entropies",
    # eigenvector_analysis
    "interpret_eigenvector",
    "compare_eigenvectors",
    "find_shared_axes",
    "find_unique_axes",
    # distance
    "frobenius_distance",
    "spectral_distance",
    "subspace_distance",
    "entropy_distance",
    "combined_distance",
    "compute_distance_matrix",
]
