# API Reference

Module-by-module quick reference for the EigenDialectos public API.

---

## eigendialectos.constants

| Export | Type | Description |
|--------|------|-------------|
| `DialectCode` | `str, Enum` | 8 dialect codes: ES_PEN, ES_AND, ES_CAN, ES_RIO, ES_MEX, ES_CAR, ES_CHI, ES_AND_BO |
| `FeatureCategory` | `str, Enum` | LEXICAL, MORPHOSYNTACTIC, PRAGMATIC, PHONOLOGICAL, TEMPORAL |
| `DIALECT_NAMES` | `dict` | Code -> human-readable name |
| `DIALECT_REGIONS` | `dict` | Code -> geographic region |
| `EMBEDDING_DIMS` | `dict` | subword=300, word=300, sentence=768 |
| `ALPHA_RANGE` | `tuple` | (0.0, 1.5, 0.1) |
| `DEFAULT_SEED` | `int` | 42 |
| `MIN_CORPUS_SIZE` | `int` | 1000 |

---

## eigendialectos.types

| Class | Key Fields | Properties |
|-------|-----------|------------|
| `DialectSample` | text, dialect_code, source_id, confidence, metadata | -- |
| `CorpusSlice` | samples, dialect_code | `.stats` -> count, avg/min/max length |
| `EmbeddingMatrix` | data (d x V), vocab, dialect_code | `.dim` -> int |
| `TransformationMatrix` | data (d x d), source_dialect, target_dialect, regularization | `.shape` -> tuple |
| `EigenDecomposition` | eigenvalues, eigenvectors (P), eigenvectors_inv (P^-1), dialect_code | `.rank` -> int |
| `DialectalSpectrum` | eigenvalues_sorted, entropy, dialect_code | `.cumulative_energy` -> array |
| `TensorDialectal` | data (d x d x m), dialect_codes | `.shape` -> tuple |
| `ExperimentResult` | experiment_id, metrics, artifact_paths, timestamp, config | -- |

---

## eigendialectos.corpus

**`CorpusSource`** (ABC) -- `download(dir)`, `load(path)`, `dialect_codes()`, `citation()`, `name()`

**`SyntheticGenerator`** -- `generate(n, dialect)`, `generate_all(n_per_dialect)`, `add_base_sentences(sentences)`

**`DialectTemplate`** -- `apply_lexical(text)`, `apply_morphological(text)`, `apply_phonological(text)`, `apply_all(text)`

---

## eigendialectos.embeddings

**`EmbeddingModel`** (ABC) -- `train(corpus)`, `encode(texts)`, `encode_words(words)`, `save(path)`, `load(path)`

**`CrossVarietyAligner`** -- `align_all(embeddings, reference, method, anchors)` -> aligned dict

**Registry:** `register_model(name, cls)`, `get_model(name)`, `list_available()`

**Backends:** FastText, BPE (subword), Word2Vec, GloVe (word), BETO, MarIA, SpanBERT (sentence)

**Alignment:** `ProcrustesAligner`, `VecMapAligner`, `MUSEAligner`

---

## eigendialectos.spectral

### Transformation

```python
compute_transformation_matrix(source, target, method="lstsq", regularization=0.01)
compute_all_transforms(embeddings, reference, method, regularization)
```

### Eigendecomposition

```python
eigendecompose(W) -> EigenDecomposition
svd_decompose(W) -> (U, Sigma, Vt)
decompose(W, method="both") -> dict
```

### Eigenspectrum

```python
compute_eigenspectrum(eigen) -> DialectalSpectrum
compare_spectra(spectra) -> dict
rank_k_approximation(eigen, k) -> TransformationMatrix
```

### Entropy

```python
compute_dialectal_entropy(spectrum_or_eigenvalues, epsilon=1e-10, base="natural") -> float
compare_entropies(entropies) -> dict  # rankings, mean, std, min, max, range, interpretation
```

### Distance

```python
frobenius_distance(W_a, W_b) -> float
spectral_distance(spec_a, spec_b) -> float          # Wasserstein-1 EMD
subspace_distance(P_a, P_b, k=10) -> float
entropy_distance(H_a, H_b) -> float
combined_distance(W_a, W_b, spec_a, spec_b, H_a, H_b, weights=None) -> float
compute_distance_matrix(transforms, spectra, entropies, method="combined") -> ndarray
```

### Eigenvector Analysis

```python
interpret_eigenvector(eigenvector, vocab, top_k=10) -> list[(word, loading)]
compare_eigenvectors(P_a, P_b, k=5) -> dict
find_shared_axes(eigendecomps, threshold=0.7) -> dict
find_unique_axes(eigendecomps, threshold=0.3) -> dict
```

### Utilities

```python
regularize_matrix(M, lambda_reg, method="ridge")
check_condition_number(M)
handle_complex_eigenvalues(eigenvalues, method="magnitude")
safe_inverse(M) -> ndarray
stable_log(x) -> float
is_orthogonal(M) -> bool
is_positive_definite(M) -> bool
```

---

## eigendialectos.generative

### DIAL

```python
apply_dial(eigen, alpha) -> TransformationMatrix
dial_transform_embedding(embedding, eigen, alpha) -> ndarray
compute_dial_series(eigen, alpha_range=ALPHA_RANGE) -> list[TransformationMatrix]
```

### Mixing

```python
mix_dialects(transforms: list[(TransformationMatrix, float)]) -> TransformationMatrix
log_euclidean_mix(transforms: list[(TransformationMatrix, float)]) -> TransformationMatrix
mix_eigendecompositions(eigens: list[(EigenDecomposition, float)]) -> EigenDecomposition
```

### Intensity

```python
IntensityController(tolerance=0.01, max_iterations=50)
  .generate_at_intensity(embedding, eigen, alpha) -> ndarray
  .sweep_intensities(embedding, eigen, start, stop, step) -> list[(alpha, ndarray)]
  .find_recognition_threshold(embedding, eigen, classifier) -> float
  .find_naturalness_threshold(embedding, eigen, quality_fn, quality_floor) -> float
```

### Generator

```python
DialectGenerator(transforms, eigendecomps, vocab, embeddings)
  .generate(text, target_dialect, alpha=1.0) -> str
  .generate_mixed(text, dialect_weights, alpha=1.0) -> str
  .generate_gradient(text, target_dialect, n_steps=16) -> list[(alpha, str)]
```

### Constraints

```python
validate_transform(W, max_cond=1000, max_eigenval=100, min_eigenval=0.001) -> (bool, list[str])
clip_eigenvalues(eigen, max_val=100, min_val=0.001) -> EigenDecomposition
check_feasibility(alpha, eigen) -> (bool, str)
```

### LoRA Integration

```python
LoRADialectAdapter(output_dir)  # Bridges DIAL with PEFT/LoRA fine-tuning
```

---

## eigendialectos.tensor

```python
build_dialect_tensor(transforms) -> TensorDialectal      # Stack W_k into R^{d x d x m}
extract_slice(tensor, dialect) -> TransformationMatrix    # Extract single W_k
analyze_factors(factors) -> dict
find_shared_factors(factors, threshold) -> dict
find_variety_specific_factors(factors, threshold) -> dict
```

Tucker: `tucker_decompose(tensor, ranks)`, `tucker_reconstruct(core, factors)`, `explained_variance()`

CP: `cp_decompose(tensor, rank)`, `core_consistency(tensor, rank)`

---

## eigendialectos.algebra

```python
DialectAlgebra(transforms, eigendecomps)
  .dim -> int
  .dialects -> list[DialectCode]
  .compose(d1, d2) -> TransformationMatrix            # W_d1 @ W_d2
  .invert(d) -> TransformationMatrix                  # W_d^{-1}
  .interpolate(d, alpha) -> TransformationMatrix       # expm(alpha logm(W))
  .project_onto_subspace(d, V) -> TransformationMatrix # P_V W P_V
  .is_approximate_group(tol=1e-6) -> dict              # closure, assoc, identity, inverse

LexicalOperator, MorphosyntacticOperator, PragmaticOperator, PhonologicalOperator
decompose_regionalism(), multiplicative_decomposition()
```

---

## eigendialectos.visualization

**Spectral:** `plot_eigenvalue_bars`, `plot_eigenvalue_decay`, `plot_cumulative_energy`, `plot_entropy_comparison`, `plot_eigenspectrum_heatmap`

**Embedding:** `plot_embeddings_tsne`, `plot_embeddings_umap`, `plot_alignment_quality`, `plot_pca_variance`

**Dialect maps:** `plot_dialect_distance_matrix`, `plot_dialect_dendrogram`, `plot_dialect_mds`

**Gradient:** `plot_alpha_gradient`, `plot_feature_activation_heatmap`, `plot_threshold_annotations`

**Tensor:** `plot_factor_loadings_heatmap`, `plot_cp_components`, `plot_reconstruction_scree`

**Interactive:** `create_spectral_dashboard`, `create_embedding_explorer`, `create_gradient_slider`

**Palette:** `DIALECT_COLORS`, `DIALECT_MARKERS`, `dialect_label(code)`

---

## eigendialectos.validation

```python
compute_bleu(reference, hypothesis, max_n=4) -> float
compute_chrf(reference, hypothesis, n=6, beta=2.0) -> float
compute_dialectal_perplexity_ratio(text, target_probs, baseline_probs) -> float
compute_classification_accuracy(predictions, ground_truth) -> float
compute_confusion_matrix(predictions, ground_truth, labels) -> ndarray
compute_frobenius_error(W_true, W_predicted) -> float
compute_eigenspectrum_divergence(spec_a, spec_b) -> float   # KL divergence
compute_krippendorff_alpha(ratings) -> float
```

---

## eigendialectos.experiments

**Base:** `Experiment` (ABC) -- lifecycle: `setup(config)` -> `run()` -> `evaluate(result)` -> `visualize(result)` -> `report(result)`

**Runner:** `ExperimentRunner(config, data_dir, output_dir)` -- `.run_experiment(id)`, `.run_all()`

| ID | Class |
|----|-------|
| exp1_spectral_map | SpectralMapExperiment |
| exp2_full_generation | FullGenerationExperiment |
| exp3_dialectal_gradient | DialectalGradientExperiment |
| exp4_impossible_dialects | ImpossibleDialectsExperiment |
| exp5_archaeology | DialectalArchaeologyExperiment |
| exp6_evolution | EvolutionExperiment |
| exp7_zeroshot | ZeroshotTransferExperiment |

---

## eigendialectos.cli

Click-based CLI. Entry point: `eigendialectos = eigendialectos.cli.main:cli`

```bash
eigendialectos run --all
eigendialectos run --experiment exp1_spectral_map
eigendialectos run --experiment exp3_dialectal_gradient --dim 100 --seed 123
```

---

## eigendialectos.utils

- `utils.io` -- caching, file I/O, serialisation
- `utils.logging` -- structured logging, metric tracking
- `utils.reproducibility` -- seed management, deterministic settings
- `utils.gpu` -- device selection, memory management
- `utils.checkpointing` -- intermediate result saving and resumption
