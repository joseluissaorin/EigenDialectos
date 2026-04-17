"""BETO sentence embedding model.

Fine-tunes ``dccuchile/bert-base-spanish-wwm-cased`` and extracts
sentence embeddings via [CLS] pooling or mean pooling.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np

from eigendialectos.constants import EMBEDDING_DIMS, DialectCode
from eigendialectos.embeddings.base import EmbeddingModel
from eigendialectos.types import CorpusSlice, EmbeddingMatrix

logger = logging.getLogger(__name__)

try:
    import torch
    from transformers import (
        AutoModel,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False


def _require_transformers() -> None:
    if not _HAS_TRANSFORMERS:
        raise ImportError(
            "transformers and torch are required for BETOModel. "
            "Install with:  pip install transformers torch  "
            "or:  pip install eigendialectos[sentence]"
        )


_DEFAULT_MODEL = "dccuchile/bert-base-spanish-wwm-cased"


class BETOModel(EmbeddingModel):
    """BETO-based sentence embedding model.

    Parameters
    ----------
    dialect_code:
        Target dialect variety.
    model_name:
        HuggingFace model identifier.
    pooling:
        ``'cls'`` for [CLS] token, ``'mean'`` for mean-pooling.
    max_length:
        Maximum token sequence length.
    batch_size:
        Inference batch size.
    device:
        PyTorch device string (``'cpu'``, ``'cuda'``, ``'mps'``).
        ``None`` for automatic selection.
    """

    def __init__(
        self,
        dialect_code: DialectCode | None = None,
        model_name: str = _DEFAULT_MODEL,
        pooling: Literal["cls", "mean"] = "mean",
        max_length: int = 128,
        batch_size: int = 32,
        device: str | None = None,
        **kwargs: Any,
    ) -> None:
        _require_transformers()
        super().__init__(dialect_code=dialect_code, **kwargs)
        self._model_name = model_name
        self._pooling = pooling
        self._max_length = max_length
        self._batch_size = batch_size
        self._device = device or self._auto_device()
        self._tokenizer: Any = None
        self._model: Any = None

    @staticmethod
    def _auto_device() -> str:
        if not _HAS_TRANSFORMERS:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    # ------------------------------------------------------------------
    # EmbeddingModel interface
    # ------------------------------------------------------------------

    def train(self, corpus: CorpusSlice, config: dict[str, Any] | None = None) -> None:
        """Fine-tune BETO on the dialect corpus using MLM.

        For lightweight usage (no fine-tuning), simply call
        ``_load_pretrained()`` directly.  This method applies a short
        masked-language-modelling pass to adapt the model to the
        dialect variety.
        """
        _require_transformers()
        cfg = config or {}
        self._dialect_code = corpus.dialect_code
        self._load_pretrained()

        texts = [s.text for s in corpus.samples]
        epochs = cfg.get("epochs", 1)
        lr = cfg.get("learning_rate", 2e-5)
        output_dir = cfg.get("output_dir", "/tmp/beto_finetune")

        logger.info(
            "Fine-tuning BETO (%s) for %d epochs on %d texts (%s)",
            self._model_name, epochs, len(texts), corpus.dialect_code.value,
        )

        # Build a simple MLM dataset
        encodings = self._tokenizer(
            texts,
            truncation=True,
            max_length=self._max_length,
            padding=True,
            return_tensors="pt",
        )

        class _MLMDataset(torch.utils.data.Dataset):
            def __init__(self, enc):
                self.enc = enc

            def __len__(self):
                return len(self.enc["input_ids"])

            def __getitem__(self, idx):
                item = {k: v[idx] for k, v in self.enc.items()}
                item["labels"] = item["input_ids"].clone()
                return item

        dataset = _MLMDataset(encodings)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=min(self._batch_size, len(texts)),
            learning_rate=lr,
            logging_steps=max(1, len(texts) // self._batch_size),
            save_strategy="no",
            report_to="none",
            use_cpu=(self._device == "cpu"),
        )

        trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=dataset,
            data_collator=DataCollatorWithPadding(self._tokenizer),
        )
        trainer.train()
        self._model.eval()
        self._is_trained = True

    def encode(self, texts: list[str]) -> np.ndarray:
        self._ensure_loaded()
        all_vectors: list[np.ndarray] = []

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            inputs = self._tokenizer(
                batch,
                truncation=True,
                max_length=self._max_length,
                padding=True,
                return_tensors="pt",
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model(**inputs)

            hidden = outputs.last_hidden_state  # (B, seq_len, dim)

            if self._pooling == "cls":
                vecs = hidden[:, 0, :]
            else:
                # Mean pooling with attention mask
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                summed = (hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)
                vecs = summed / counts

            all_vectors.append(vecs.cpu().numpy())

        return np.vstack(all_vectors).astype(np.float64)

    def encode_words(self, words: list[str]) -> EmbeddingMatrix:
        vecs = self.encode(words)
        return EmbeddingMatrix(
            data=vecs,
            vocab=list(words),
            dialect_code=self._dialect_code or DialectCode.ES_PEN,
        )

    def save(self, path: Path) -> None:
        self._ensure_loaded()
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(str(p))
        self._tokenizer.save_pretrained(str(p))
        self._save_meta(p)
        logger.info("BETO model saved to %s", p)

    def load(self, path: Path) -> None:
        _require_transformers()
        p = Path(path)
        self._tokenizer = AutoTokenizer.from_pretrained(str(p))
        self._model = AutoModel.from_pretrained(str(p)).to(self._device)
        self._model.eval()
        meta = self._load_meta(p)
        if meta.get("dialect_code"):
            self._dialect_code = DialectCode(meta["dialect_code"])
        self._is_trained = True
        logger.info("BETO model loaded from %s", p)

    def vocab_size(self) -> int:
        self._ensure_loaded()
        return self._tokenizer.vocab_size

    def embedding_dim(self) -> int:
        if self._model is not None:
            return int(self._model.config.hidden_size)
        return EMBEDDING_DIMS["sentence"]

    def level(self) -> str:
        return "sentence"

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_pretrained(self) -> None:
        """Load the pretrained model and tokenizer if not already loaded."""
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        if self._model is None:
            self._model = AutoModel.from_pretrained(self._model_name).to(self._device)
            self._model.eval()
        self._is_trained = True

    def _ensure_loaded(self) -> None:
        if self._model is None or self._tokenizer is None:
            self._load_pretrained()
