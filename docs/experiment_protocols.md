# Experiment Protocols

Detailed protocols for all 7 experiments in EigenDialectos.

---

## Experiment 1: Spectral Map of Spanish Dialect Varieties

**ID:** `exp1_spectral_map` | **Class:** `SpectralMapExperiment`

### Objective

Compute the eigenspectrum of every dialect transformation matrix W_i (relative to Peninsular Standard), build a pairwise spectral distance matrix, and compare it with known dialectological classifications.

### Protocol

1. **Setup:** Generate or load embedding matrices E_i for all 8 dialects (default: synthetic, dim=50, vocab=200).
2. **Compute W_i:** Ridge regression with lambda=0.01 for each dialect relative to E_ref (ES_PEN).
3. **Eigendecompose:** W_i = P_i Lambda_i P_i^{-1}.
4. **Compute spectra:** Sort eigenvalue magnitudes; compute entropy H_i.
5. **Distance matrix:** Pairwise L2 distance between sorted eigenvalue vectors.
6. **Evaluate:** Verify that known close pairs (PEN-AND, PEN-CAN, AND-CAN, CAR-AND) have smaller distance than known distant pairs (PEN-RIO, PEN-CHI, AND-MEX).

### Success Criterion

Mean close-pair distance < mean distant-pair distance.

### Outputs

- Eigenspectrum bar chart (entropy per dialect)
- Pairwise distance heatmap
- Ward-linkage dendrogram
- Metrics: entropies, distance_matrix, mean_distance, max_distance, ordering_correct

### Config

| Parameter | Default | Description |
|-----------|---------|-------------|
| seed | 42 | Random seed |
| dim | 50 | Embedding dimensionality |
| vocab_size | 200 | Vocabulary size |
| regularization | 0.01 | Ridge lambda |

---

## Experiment 2: Full Dialect Generation (alpha=1.0)

**ID:** `exp2_full_generation` | **Class:** `FullGenerationExperiment`

### Objective

Generate text in all 8 varieties at full intensity (alpha=1.0) and evaluate with automatic metrics.

### Protocol

1. **Setup:** Build synthetic embeddings and eigendecompositions. Prepare n_sentences neutral input embeddings.
2. **Generate:** Apply `dial_transform_embedding(neutral, eigen, alpha=1.0)` per dialect.
3. **Measure displacement:** Mean L2 norm of (transformed - neutral).
4. **Evaluate with proxy metrics:**
   - BLEU proxy: average cosine similarity (generated vs reference)
   - chrF proxy: correlation between flattened vectors
   - Perplexity proxy: mean displacement norm
5. **Compare:** Identify best/worst dialects by BLEU.

### Outputs

- Grouped bar chart: BLEU, chrF, perplexity proxy across dialects
- Per-dialect metric table
- Metrics: per_dialect, mean_bleu, mean_chrf, mean_perplexity_proxy

---

## Experiment 3: Dialectal Gradient (Alpha Sweep)

**ID:** `exp3_dialectal_gradient` | **Class:** `DialectalGradientExperiment`

### Objective

Sweep alpha from 0.0 to 1.5 and identify the recognition threshold (where a classifier first detects the dialect) and the naturalness threshold (where quality degrades).

### Protocol

1. **Setup:** Build embeddings with controlled perturbation directions. 20 neutral inputs. Eigendecompose.
2. **Sweep:** For each alpha in [0.0, 1.5, step=0.1]:
   - Transform: `dial_transform_embedding(neutral, eigen, alpha)`
   - Confidence proxy: `sigmoid(10 * (relative_dist - 0.3))`
   - Record embedding norm
3. **Recognition threshold:** First alpha where confidence > 0.5
4. **Naturalness threshold:** First alpha where norm diverges >50% from baseline

### Outputs

- Alpha vs confidence curves (all 8 dialects)
- Per-dialect thresholds (recognition, naturalness)
- Mean thresholds

---

## Experiment 4: Impossible Dialects

**ID:** `exp4_impossible_dialects` | **Class:** `ImpossibleDialectsExperiment`

### Objective

Create dialect mixtures with contradictory linguistic features and analyse whether the algebraic framework detects the incoherence.

### Feature Conflicts Tested

| Conflict | Dialects | Category | Reason |
|----------|----------|----------|--------|
| Voseo + vosotros | ES_RIO + ES_PEN | MORPHOSYNTACTIC | Same person slot, incompatible forms |
| Seseo + ceceo | ES_MEX + ES_AND | PHONOLOGICAL | Opposite /theta/-/s/ mappings |
| Aspiration + full /s/ | ES_CAR + ES_AND_BO | PHONOLOGICAL | Mutually exclusive /s/ treatments |

### Protocol

1. **Mix:** W_mix = 0.5*W_a + 0.5*W_b via `mix_dialects()`
2. **Coherence check:** Rank, condition number, coherence score
3. **Conflict score:** KL divergence on eigenvalue distributions, modulated by eigenvector misalignment
4. **Full matrix:** Pairwise conflict scores for all dialect pairs

### Outputs

- Feature conflict heatmap (all pairs)
- Per-combination coherence vs conflict chart
- Evaluation: n_samples=30, coherence_check + native_speaker_eval

---

## Experiment 5: Dialectal Archaeology

**ID:** `exp5_archaeology` | **Class:** `DialectalArchaeologyExperiment`

### Objective

Apply inverse transforms to historical texts (Golden Age Spanish, ~15th-17th century) to approximate proto-dialectal forms.

### Historical Sources

- Golden Age texts (16th-17th century): Don Quijote, Lazarillo, La Celestina
- Colonial era texts (18th-19th century)

### Protocol

1. **Setup:** Build embeddings and eigendecompositions. Generate pseudo-embeddings for historical text.
2. **Apply inverse:** DIAL at alpha=-1.0 (full inverse) and alpha=-0.5 (half inverse) for each text-dialect pair.
3. **Measure displacement:** L2 norm of (inverse - original).
4. **Identify closest dialect:** Dialect with smallest displacement indicates the historical text is closest to that dialect's origin.
5. **Feature alignment:** Evaluate consistency with known philological data.

### Outputs

- Displacement charts per historical text
- Closest dialect identification per text period
- Comparison with philological ground truth (n_samples=20)

---

## Experiment 6: Dialectal Evolution (Phylogenetic Analysis)

**ID:** `exp6_evolution` | **Class:** `EvolutionExperiment`

### Objective

Build a phylogenetic tree from eigenvector similarities and validate against known historical dialect groupings.

### Known Historical Clusters

- Iberian: PEN, AND, CAN
- Southern Cone: RIO, CHI
- Seseo link: CAN, CAR

### Protocol

1. **Setup:** Create embeddings with structure reflecting known clusters. Eigendecompose.
2. **Pairwise similarity:** Compare top-k eigenvectors (default k=20) via cosine similarity.
3. **Shared axes:** Pairs with cosine > 0.7.
4. **Unique axes:** Per-dialect axes with low similarity to all others.
5. **Phylogenetic tree:** Convert similarity to distance (1 - sim), apply UPGMA (average-linkage).
6. **Validate:** Intra-cluster similarity > inter-cluster similarity.

### Outputs

- Eigenvector similarity heatmap
- UPGMA phylogenetic dendrogram
- Shared/unique axis counts
- Historical correlation assessment

---

## Experiment 7: Zero-Shot Dialect Transfer

**ID:** `exp7_zeroshot` | **Class:** `ZeroshotTransferExperiment`

### Objective

Hold out 2 dialect varieties, decompose the remaining tensor, reconstruct the held-out matrices, and measure reconstruction error.

### Holdout Pairs

| Fold | Held Out |
|------|----------|
| 1 | ES_CAN, ES_AND |
| 2 | ES_RIO, ES_CHI |
| 3 | ES_MEX, ES_CAR |
| 4 | ES_AND_BO, ES_PEN |

### Protocol

1. **Setup:** Build T of shape (d, d, 8). Default d=10.
2. **Leave-2-out:** For each pair, remove 2 slices from T -> T' of shape (d, d, 6).
3. **Reconstruct:**
   - SVD-based: mode-3 unfold T', truncated SVD, project mean onto right singular vectors
   - Tucker-based (alternative): decompose T', extrapolate held-out via mode-3 factor mean
4. **Error:** Absolute and relative Frobenius error per held-out dialect.
5. **Evaluate:** Mean relative error < 0.5 for success.

### Config

| Parameter | Default | Description |
|-----------|---------|-------------|
| seed | 42 | Random seed |
| dim | 10 | Embedding dim (small for speed) |
| vocab_size | 50 | Vocabulary size |
| max_holdout_pairs | 10 | Max leave-2-out pairs |
| decomposition_method | "svd" | "svd" or "tucker" |

### Outputs

- Per-holdout error bars
- Per-dialect average reconstruction error
- Generalisation pass/fail assessment

---

## Running Experiments

### CLI

```bash
eigendialectos run --experiment exp1_spectral_map
eigendialectos run --all
eigendialectos run --experiment exp3_dialectal_gradient --dim 100 --seed 123
```

### Programmatic

```python
from pathlib import Path
from eigendialectos.experiments.runner import ExperimentRunner

runner = ExperimentRunner(
    config={"seed": 42, "dim": 50},
    data_dir=Path("data/"),
    output_dir=Path("outputs/"),
)
result = runner.run_experiment("exp1_spectral_map")
all_results = runner.run_all()
```

### Output Structure

```
outputs/<experiment_id>/
    result.json     # Serialised ExperimentResult
    report.md       # Human-readable report
    *.png           # Visualisation figures
```
