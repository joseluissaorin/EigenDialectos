"""Visualization module for EigenDialectos.

Re-exports all public plot functions and the shared colour palette.
"""

from eigendialectos.visualization._colors import (
    DIALECT_COLORS,
    DIALECT_MARKERS,
    dialect_label,
)
from eigendialectos.visualization.dialect_maps import (
    plot_dialect_dendrogram,
    plot_dialect_distance_matrix,
    plot_dialect_mds,
)
from eigendialectos.visualization.embedding_plots import (
    plot_alignment_quality,
    plot_embeddings_tsne,
    plot_embeddings_umap,
    plot_pca_variance,
)
from eigendialectos.visualization.gradient_plots import (
    plot_alpha_gradient,
    plot_feature_activation_heatmap,
    plot_threshold_annotations,
)
from eigendialectos.visualization.interactive import (
    create_embedding_explorer,
    create_gradient_slider,
    create_spectral_dashboard,
)
from eigendialectos.visualization.spectral_plots import (
    plot_cumulative_energy,
    plot_eigenspectrum_heatmap,
    plot_eigenvalue_bars,
    plot_eigenvalue_decay,
    plot_entropy_comparison,
)
from eigendialectos.visualization.tensor_plots import (
    plot_cp_components,
    plot_factor_loadings_heatmap,
    plot_reconstruction_scree,
)

__all__ = [
    # Colors
    "DIALECT_COLORS",
    "DIALECT_MARKERS",
    "dialect_label",
    # Spectral
    "plot_eigenvalue_bars",
    "plot_eigenvalue_decay",
    "plot_cumulative_energy",
    "plot_entropy_comparison",
    "plot_eigenspectrum_heatmap",
    # Embeddings
    "plot_embeddings_tsne",
    "plot_embeddings_umap",
    "plot_alignment_quality",
    "plot_pca_variance",
    # Dialect maps
    "plot_dialect_distance_matrix",
    "plot_dialect_dendrogram",
    "plot_dialect_mds",
    # Gradient
    "plot_alpha_gradient",
    "plot_feature_activation_heatmap",
    "plot_threshold_annotations",
    # Tensor
    "plot_factor_loadings_heatmap",
    "plot_cp_components",
    "plot_reconstruction_scree",
    # Interactive
    "create_spectral_dashboard",
    "create_embedding_explorer",
    "create_gradient_slider",
]
