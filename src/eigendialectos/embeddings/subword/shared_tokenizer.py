"""Shared SentencePiece BPE tokenizer with optional morpheme-aware pre-tokenization.

Trains a single BPE vocabulary on the combined corpus of all dialect
varieties.  When ``morpheme_aware=True``, applies the rule-based morpheme
parser to insert zero-width space hints at morpheme boundaries before BPE
training, biasing BPE to split at linguistically meaningful points.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Optional

import sentencepiece as spm

logger = logging.getLogger(__name__)

# Zero-width space used as morpheme boundary hint.
_MORPH_BOUNDARY = "\u200B"


class SharedSubwordTokenizer:
    """Wrapper around a trained SentencePiece model.

    Parameters
    ----------
    model_path:
        Path to the ``.model`` file produced by SentencePiece training.
    """

    def __init__(self, model_path: Path | str) -> None:
        self.model_path = Path(model_path)
        self._sp = spm.SentencePieceProcessor()
        self._sp.Load(str(self.model_path))

    # ------------------------------------------------------------------
    # Public tokenisation API
    # ------------------------------------------------------------------

    def tokenize(self, text: str) -> list[int]:
        """Encode text as a list of subword token IDs."""
        return self._sp.EncodeAsIds(text)

    def tokenize_to_pieces(self, text: str) -> list[str]:
        """Encode text as a list of subword string pieces."""
        return self._sp.EncodeAsPieces(text)

    def tokenize_word(self, word: str) -> list[int]:
        """Encode a single word (no whitespace splitting)."""
        return self._sp.EncodeAsIds(word)

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to text."""
        return self._sp.DecodeIds(ids)

    @property
    def vocab_size(self) -> int:
        return self._sp.GetPieceSize()

    def id_to_piece(self, token_id: int) -> str:
        return self._sp.IdToPiece(token_id)

    def piece_to_id(self, piece: str) -> int:
        return self._sp.PieceToId(piece)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @classmethod
    def train(
        cls,
        corpus_texts: dict[str, list[str]],
        output_path: Path | str,
        vocab_size: int = 8000,
        morpheme_aware: bool = True,
        character_coverage: float = 0.9999,
        max_sentence_length: int = 4096,
        input_sentence_size: int = 500_000,
        seed: int = 42,
    ) -> "SharedSubwordTokenizer":
        """Train a shared BPE tokenizer on combined corpus.

        Parameters
        ----------
        corpus_texts:
            Mapping from variety name to list of text documents.
        output_path:
            Directory where the trained model will be saved.
        vocab_size:
            BPE vocabulary size.
        morpheme_aware:
            If True, apply morpheme parser as pre-tokeniser to bias
            BPE towards linguistically meaningful splits.
        character_coverage:
            Fraction of characters covered by the vocabulary.
            0.9999 ensures all Spanish accented characters are included.
        max_sentence_length:
            Maximum sentence length in bytes.
        input_sentence_size:
            Number of sentences sampled for BPE training.
        seed:
            Random seed for reproducibility.

        Returns
        -------
        SharedSubwordTokenizer
            Loaded tokenizer ready for use.
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        model_prefix = str(output_path / "shared_bpe")

        # Optionally load morpheme parser for pre-tokenization
        pre_tokenize = None
        if morpheme_aware:
            try:
                from eigendialectos.corpus.parsing.morpheme_parser import (
                    parse_morphemes,
                )
                pre_tokenize = parse_morphemes
                logger.info("Morpheme-aware pre-tokenization enabled.")
            except ImportError:
                logger.warning(
                    "morpheme_parser not available; training without "
                    "morpheme boundary hints."
                )

        # Write combined corpus to a temp file for SentencePiece
        logger.info(
            "Preparing BPE training corpus (vocab_size=%d, morpheme_aware=%s) ...",
            vocab_size,
            morpheme_aware,
        )
        total_lines = 0
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            tmp_path = f.name
            for variety, docs in sorted(corpus_texts.items()):
                for doc in docs:
                    line = doc.strip()
                    if not line:
                        continue
                    if pre_tokenize is not None:
                        line = _apply_morpheme_hints(line, pre_tokenize)
                    f.write(line + "\n")
                    total_lines += 1

        logger.info("Wrote %d lines to temp corpus file.", total_lines)

        # Build user_defined_symbols list
        user_symbols = []
        if morpheme_aware:
            user_symbols.append(_MORPH_BOUNDARY)

        # Train SentencePiece BPE (kwargs API handles paths with spaces)
        logger.info("Training SentencePiece BPE ...")
        train_kwargs = dict(
            input=tmp_path,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type="bpe",
            character_coverage=character_coverage,
            max_sentence_length=max_sentence_length,
            input_sentence_size=input_sentence_size,
            shuffle_input_sentence=True,
            split_by_whitespace=True,
            split_by_unicode_script=True,
            seed_sentencepiece_size=1000000,
            num_threads=4,
        )
        if user_symbols:
            train_kwargs["user_defined_symbols"] = user_symbols

        spm.SentencePieceTrainer.Train(**train_kwargs)

        model_file = Path(f"{model_prefix}.model")
        logger.info("BPE model saved to %s (vocab_size=%d)", model_file, vocab_size)

        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)

        # Save metadata
        meta = {
            "vocab_size": vocab_size,
            "morpheme_aware": morpheme_aware,
            "character_coverage": character_coverage,
            "total_training_lines": total_lines,
            "n_varieties": len(corpus_texts),
            "varieties": sorted(corpus_texts.keys()),
        }
        with open(output_path / "tokenizer_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        return cls(model_file)

    @classmethod
    def load(cls, model_dir: Path | str) -> "SharedSubwordTokenizer":
        """Load a previously trained tokenizer from directory."""
        model_dir = Path(model_dir)
        model_file = model_dir / "shared_bpe.model"
        if not model_file.exists():
            raise FileNotFoundError(f"No BPE model at {model_file}")
        return cls(model_file)


def _apply_morpheme_hints(text: str, parse_fn) -> str:
    """Insert zero-width space at morpheme boundaries within words.

    Example: "hablábamos" → "habl\u200Bábamos" (if the parser splits it
    as ["habl", "ábamos"]).

    The zero-width space acts as a BPE-merge hint: SentencePiece is more
    likely to break at these positions because they appear as separate
    tokens in the training data.
    """
    words = text.split()
    result_tokens: list[str] = []

    for word in words:
        # Only apply to non-trivial words
        if len(word) <= 3:
            result_tokens.append(word)
            continue

        try:
            morphemes = parse_fn([word])  # returns list of list[str]
            if morphemes and len(morphemes[0]) > 1:
                # Join morphemes with zero-width space hint
                result_tokens.append(_MORPH_BOUNDARY.join(morphemes[0]))
            else:
                result_tokens.append(word)
        except Exception:
            # If parsing fails for any word, keep original
            result_tokens.append(word)

    return " ".join(result_tokens)
