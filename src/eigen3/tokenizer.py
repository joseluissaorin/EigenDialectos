"""SentencePiece BPE tokenizer training and loading."""

from __future__ import annotations

import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# Morpheme-aware pre-tokenization: split on common Spanish suffixes
_MORPHEME_RE = re.compile(
    r"(mente|ción|sión|idad|ismo|ista|ando|endo|iendo|ado|ido|"
    r"amos|emos|imos|aron|ieron|aban|ían|aría|ería|iría|"
    r"ando|iendo|mente)$"
)


def train_tokenizer(
    texts: list[str],
    model_prefix: str = "bpe",
    vocab_size: int = 8000,
    model_type: str = "bpe",
    output_dir: Optional[str | Path] = None,
) -> str:
    """Train a SentencePiece BPE model on texts. Returns path to .model file."""
    try:
        import sentencepiece as spm
    except ImportError:
        raise ImportError("sentencepiece required: pip install sentencepiece")

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="eigen3_bpe_")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write training data
    train_file = output_dir / "train_text.txt"
    with open(train_file, "w", encoding="utf-8") as f:
        for text in texts:
            f.write(text.strip() + "\n")

    model_path = str(output_dir / model_prefix)
    spm.SentencePieceTrainer.train(
        input=str(train_file),
        model_prefix=model_path,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=1.0,
        pad_id=3,
        unk_id=0,
        bos_id=1,
        eos_id=2,
    )

    result_path = model_path + ".model"
    logger.info("Trained BPE tokenizer: vocab_size=%d, path=%s", vocab_size, result_path)
    return result_path


class Tokenizer:
    """Wrapper around a trained SentencePiece model."""

    def __init__(self, model_path: str | Path):
        try:
            import sentencepiece as spm
        except ImportError:
            raise ImportError("sentencepiece required: pip install sentencepiece")

        self._sp = spm.SentencePieceProcessor()
        self._sp.load(str(model_path))
        self._model_path = str(model_path)

    @property
    def vocab_size(self) -> int:
        return self._sp.get_piece_size()

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        return self._sp.encode(text)

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs to text."""
        return self._sp.decode(ids)

    def piece_to_id(self, piece: str) -> int:
        """Get ID for a specific piece/token."""
        return self._sp.piece_to_id(piece)

    def id_to_piece(self, idx: int) -> str:
        """Get piece/token for a specific ID."""
        return self._sp.id_to_piece(idx)

    def encode_as_pieces(self, text: str) -> list[str]:
        """Encode text to pieces (subword strings)."""
        return self._sp.encode_as_pieces(text)
