# System Architecture

## Overview

EigenDialectos is a spectral analysis framework for Spanish dialect variation. It models the space of 8 major Spanish dialect varieties through the eigendecomposition of inter-dialect embedding transformation matrices. The project is organised as 11 phases (P0--P10 plus P11 documentation), each implemented as a self-contained Python package under `src/eigendialectos/`.

## Module Dependency Graph

```
P0 Foundation
 |  types.py, constants.py, utils/
 |
 +---> P1 Corpus -------> P2 Embeddings -------> P3 Spectral Analysis
        corpus/             embeddings/            spectral/
        base, synthetic,    subword, word,         transformation,
        preprocessing,      sentence,              eigendecomposition,
        registry            contrastive,           eigenspectrum,
                            alignment              entropy, distance
                                                     |
                                     +---------------+---------------+
                                     |               |               |
                                     v               v               v
                               P4 Generative    P5 Tensor       P6 Algebra
                               generative/      tensor/         algebra/
                               dial, mixing,    construction,   model, lexical,
                               intensity,       tucker, cp,     morphosyntactic,
                               generator,       analysis        pragmatic,
                               constraints,                     phonological
                               lora_integration
                                     |               |               |
                                     +-------+-------+-------+-------+
                                             |
                                             v
                                       P7 Visualization
                                       visualization/
                                       spectral_plots, embedding_plots,
                                       dialect_maps, gradient_plots,
                                       tensor_plots, interactive
                                             |
                                   +---------+---------+
                                   v                   v
                             P8 Experiments      P9 Validation
                             experiments/        validation/
                             exp1..exp7,         metrics, holdout,
                             base, runner        perplexity, survey
                                   |
                                   v
                             P10 CLI
                             cli/
                             main, commands,
                             corpus_commands
```

## Data Flow Diagram

The core pipeline follows a six-stage linear flow:

```
1. Corpus (raw text per dialect)
       |  CorpusSource.download() -> preprocess -> DialectSample -> CorpusSlice
       v
2. Embeddings (E_i for each dialect i)
       |  EmbeddingModel.train(CorpusSlice) -> EmbeddingMatrix
       v
3. Alignment (map all E_i into a shared reference space)
       |  CrossVarietyAligner.align_all() using Procrustes / VecMap / MUSE
       v
4. Transformation matrices
       |  W_i = E_i @ E_ref^T @ (E_ref @ E_ref^T + lambda*I)^{-1}
       v
5. Eigendecomposition
       |  W_i = P_i @ Lambda_i @ P_i^{-1}
       |  -> DialectalSpectrum (sorted |lambda|, entropy H_i)
       v
6. DIAL generation
       |  W_i(alpha) = P_i @ Lambda_i^alpha @ P_i^{-1}
       |  encode -> DIAL transform -> nearest-neighbour decode
       v
   Output: dialectal text at intensity alpha
```

## Key Abstractions

### Core Data Types (P0: `types.py`)

| Type | Shape / Fields | Purpose |
|------|---------------|---------|
| `DialectSample` | text, dialect_code, source_id, confidence | Single annotated text sample |
| `CorpusSlice` | samples[], dialect_code | Collection of samples for one variety |
| `EmbeddingMatrix` | data (d x V), vocab, dialect_code | Dense embedding matrix with vocabulary |
| `TransformationMatrix` | data (d x d), source/target dialect, lambda | Linear map between two dialect spaces |
| `EigenDecomposition` | eigenvalues, P, P^{-1}, dialect_code | Eigenvalues and eigenvector matrices |
| `DialectalSpectrum` | eigenvalues_sorted, entropy, dialect_code | Spectral profile of a dialect |
| `TensorDialectal` | data (d x d x m), dialect_codes | Multi-dialect tensor representation |
| `ExperimentResult` | experiment_id, metrics, artifact_paths, config | Captured experiment output |

### CorpusSource (P1)

Abstract base class. Subclasses implement `download()`, `load()`, `dialect_codes()`, and `citation()`. The `SyntheticGenerator` provides rule-based synthetic dialect samples for prototyping.

### EmbeddingModel (P2)

Abstract base class with a uniform interface (`train`, `encode`, `encode_words`, `save`, `load`) for all backends: FastText, BPE (subword), Word2Vec, GloVe (word), BETO, MarIA, SpanBERT (sentence). A registry pattern (`register_model`, `get_model`) enables pluggable backends.

### Experiment (P8)

Abstract base class enforcing a four-stage lifecycle: `setup(config)` -> `run()` -> `evaluate(result)` -> `visualize(result)`. The `ExperimentRunner` handles discovery, dependency verification, and sequential execution of all 7 experiments.

## Configuration

Hydra YAML configurations under `configs/` are organised by module:

```
configs/
  config.yaml              # Top-level defaults
  corpus/                  # demo.yaml, default.yaml, full.yaml
  embeddings/              # fasttext.yaml, word2vec.yaml, beto.yaml, contrastive.yaml
  spectral/                # default.yaml
  generative/              # dial.yaml
  tensor/                  # default.yaml
  validation/              # default.yaml
  experiments/             # exp1_spectral_map.yaml .. exp7_zeroshot_transfer.yaml
```

## Visualisation Architecture (P7)

Six specialised modules share a consistent colour palette (`_colors.py`):

- `spectral_plots` -- eigenvalue bars, decay curves, cumulative energy, entropy, heatmaps
- `embedding_plots` -- t-SNE, UMAP, alignment quality, PCA variance
- `dialect_maps` -- distance matrix heatmap, dendrogram, MDS projection
- `gradient_plots` -- alpha gradient curves, feature activation heatmaps
- `tensor_plots` -- factor loadings, CP components, reconstruction scree
- `interactive` -- Plotly dashboards (spectral explorer, embedding explorer, gradient slider)

## Testing Strategy

Tests mirror the source tree under `tests/`. Target: >= 80% line coverage. Shared fixtures in `conftest.py` provide synthetic data and small embeddings for fast test execution.

```bash
pytest --cov=eigendialectos --cov-report=html
```
