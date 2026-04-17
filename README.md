# EigenDialectos

**Spectral dialect classifier for 8 Spanish varieties.**

EigenDialectos classifies Spanish text by dialect using a BETO transformer fine-tuned with LoRA adapters and supervised contrastive learning (SupCon + MoCo + DCL). It combines two complementary classification methods: direct ArcFace angular-margin classification and spectral eigenmode fingerprinting via inter-dialect transformation matrices.

## How It Works

```
Input text: "Che boludo qué decís"
    |
    v
BETO (frozen, 112M params) + LoRA adapters (2.6M params)
    |
    v
Dialect-aware attention pooling (learned weighted avg over all tokens)
    |
    v
Projection head (768 -> 384, MLP + BatchNorm)
    |
    +--> ArcFace classifier (384 -> 8 varieties)  --> "ES_RIO" (direct)
    |
    +--> L2-normalized embedding (384-dim)
         |
         +--> Nearest centroid in projection space  --> "ES_RIO" (79.1%)
         |
         +--> Word-level extraction -> Procrustes alignment
              -> W transformation matrices -> Eigendecomposition
              -> Spectral dialect fingerprinting              --> "ES_RIO" (eigenmode)
```

## 8 Spanish Dialect Varieties

| Code | Variety | Region |
|------|---------|--------|
| `ES_PEN` | Peninsular | Spain (centre-north) |
| `ES_AND` | Andalusian | Andalusia |
| `ES_CAN` | Canarian | Canary Islands |
| `ES_RIO` | Rioplatense | Argentina / Uruguay |
| `ES_MEX` | Mexican | Mexico |
| `ES_CAR` | Caribbean | Cuba / PR / Dominican Rep. |
| `ES_CHI` | Chilean | Chile |
| `ES_AND_BO` | Andean | Peru / Bolivia / Ecuador |

## Quick Start

```bash
# Install
git clone https://github.com/joseluissaorin/EigenDialectos.git
cd EigenDialectos
pip install -e ".[dev]"

# Classify text (projection-space method)
python scripts/classify_projection.py "Che boludo qué decís"
# -> ES_RIO (Rioplatense) 79.1%

# Interactive mode
python scripts/classify_projection.py --interactive

# Build centroids from corpus (first time only)
python scripts/classify_projection.py --build-centroids
```

## Training

```bash
# Full training with MoCo contrastive learning
python scripts/train_eigen3.py \
    --corpus-dir data/processed_v4 \
    --output-dir outputs/eigen3 \
    --epochs 10 --lr 2e-4 \
    --queue-size 4096 --moco-momentum 0.999 \
    --moco-start-epoch 4 --queue-ramp-steps 5000 \
    --supcon-temperature 0.07 \
    --samples-per-variety 4

# Evaluate a checkpoint
python scripts/eval_checkpoint.py
```

### Training Architecture

The training pipeline uses a multi-task objective:

**L = w_mlm * L_mlm + w_cls * L_cls + w_con * L_supcon + w_center * L_center**

- **MLM** (masked language modeling): Standard BERT MLM on Spanish text
- **CLS** (ArcFace classification): Angular-margin cross-entropy on dialect labels
- **SupCon** (supervised contrastive): DCL variant with MoCo momentum queue
- **Center** (warmup): Pulls embeddings toward class centroids during early training

**Two-phase training:**
1. Epochs 1-2: MLM + CLS only (pretrain phase)
2. Epochs 3-10: Full multi-task with contrastive + curriculum weight shift

**MoCo (Momentum Contrast):**
- Momentum encoder (EMA, m=0.999) provides consistent queue entries
- Queue of 4096 negative examples ramps in gradually from epoch 4
- Solves the staleness problem of cross-batch memory approaches

## Architecture Details

### Model (`src/eigen3/model.py`)
- **Base**: BETO (`dccuchile/bert-base-spanish-wwm-cased`), 112M params frozen
- **LoRA**: r=16, alpha=32, on attention Q/K/V + output + FFN (~2.6M trainable)
- **Variety tokens**: 8 special tokens `[VAR_ES_PEN]`...`[VAR_ES_AND_BO]`
- **Attention pooling**: Learned weighted average over all token positions
- **Projection**: Linear(768,384) -> BN -> ReLU -> Linear(384,384) -> L2-norm
- **ArcFace**: Angular-margin classifier on projected features (s=30, m=0.3)

### Spectral Pipeline

After training, word-level embeddings are extracted and processed:

1. **Contextual extraction**: Each vocabulary word is embedded in corpus contexts per variety
2. **Procrustes alignment**: All varieties mapped to a shared reference space (ES_PEN)
3. **Transformation matrices**: W_i = E_i @ E_ref^+ (regularized pseudoinverse)
4. **Eigendecomposition**: W_i = P_i * Lambda_i * P_i^{-1}
5. **DIAL transform**: W_i(alpha) = P_i * Lambda_i^alpha * P_i^{-1} for intensity control

The eigenspectrum of each W_i serves as a spectral fingerprint of each dialect.

## Project Structure

```
EigenDialectos/
  src/eigen3/              # Core library
    model.py               # BETO + LoRA + ArcFace + projection
    trainer.py             # TransformerTrainer with MoCo
    loss.py                # Multi-task loss (MLM + ArcFace + SupCon + center)
    moco.py                # MomentumEncoder + MoCoQueue
    dataset.py             # DialectMLMDataset + BalancedVarietySampler
    corpus.py              # Corpus loading, balancing, synthetic augmentation
    composer.py            # Contextual word embedding extraction
    alignment.py           # Procrustes alignment
    transformation.py      # W matrix computation
    decomposition.py       # Eigendecomposition
    distance.py            # Spectral distance metrics
    scorer.py              # DialectScorer (eigenmode classification)
    emb_pipeline.py        # End-to-end pipeline orchestration
    ...
  scripts/
    train_eigen3.py        # Training entry point
    classify_projection.py # Projection-space classifier
    classify_text.py       # ArcFace classifier
    eval_checkpoint.py     # Checkpoint evaluation
    ...
  tests/test_eigen3/       # 758 tests
  paper/                   # LaTeX paper
  docs/                    # Documentation
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- transformers >= 4.30
- peft >= 0.4
- numpy, scipy

Full dependencies in `pyproject.toml`.

## License

MIT License. See [LICENSE](LICENSE).

## Citation

If you use EigenDialectos in your research:

```bibtex
@software{saorin2026eigendialectos,
  author = {Saorin, Jose Luis},
  title = {EigenDialectos: Spectral Dialect Classification for Spanish Varieties},
  year = {2026},
  url = {https://github.com/joseluissaorin/EigenDialectos}
}
```
