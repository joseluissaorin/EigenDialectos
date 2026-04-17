"""Dialect-Contrastive Loss (DCL) embedding training pipeline.

Trains per-variety embedding tables with a three-term loss:
  1. Same-variety skip-gram attraction
  2. Cross-variety context repulsion
  3. Anchor regularisation for shared (non-regionalism) words

Public API
----------
- :class:`DCLEmbeddingModel` -- per-variety embedding tables
- :class:`DCLTrainer` -- end-to-end word-level training orchestrator
- :class:`SubwordDCLTrainer` -- subword-level (BPE) training orchestrator
- :class:`DCLDataset` -- word-level skip-gram + cross-variety dataset
- :class:`SubwordDCLDataset` -- subword-level skip-gram dataset
- :class:`DialectContrastiveLoss` -- the DCL loss function
- :data:`REGIONALISMS`, :data:`ALL_REGIONALISMS` -- merged regionalism sets
"""

from eigendialectos.embeddings.dcl.dataset import DCLDataset
from eigendialectos.embeddings.dcl.loss import DialectContrastiveLoss
from eigendialectos.embeddings.dcl.model import DCLEmbeddingModel
from eigendialectos.embeddings.dcl.regionalisms import ALL_REGIONALISMS, REGIONALISMS
from eigendialectos.embeddings.dcl.subword_dataset import SubwordDCLDataset
from eigendialectos.embeddings.dcl.trainer import DCLTrainer, SubwordDCLTrainer

__all__ = [
    "DCLDataset",
    "DCLEmbeddingModel",
    "DCLTrainer",
    "SubwordDCLTrainer",
    "SubwordDCLDataset",
    "DialectContrastiveLoss",
    "ALL_REGIONALISMS",
    "REGIONALISMS",
]
