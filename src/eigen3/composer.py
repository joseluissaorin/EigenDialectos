"""Word embedding extraction from trained models.

TransformerWordComposer — runs corpus through transformer, averages contextual
                          embeddings per (word, variety) to produce static vectors.
SubwordToWordComposer   — legacy mean-pooling from BPE subword embeddings.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
import torch

logger = logging.getLogger(__name__)


class TransformerWordComposer:
    """Extract static word embeddings from a trained dialect transformer.

    For each (word, variety): collect all contextual embeddings of that word
    across all corpus occurrences, then mean-pool to a single static vector.
    Words not seen in corpus get a fallback isolated-word embedding.
    """

    def __init__(
        self,
        model,
        device: torch.device | None = None,
        batch_size: int = 64,
        max_length: int = 256,
    ) -> None:
        self.model = model
        self.tokenizer = model.tokenizer
        self.device = device or next(model.parameters()).device
        self.batch_size = batch_size
        self.max_length = max_length
        self.proj_dim = model.proj_dim

    def compose_vocabulary(
        self,
        vocab: list[str],
        corpus_by_variety: dict[str, list[str]],
    ) -> dict[str, np.ndarray]:
        """Extract per-variety word embeddings from corpus via the transformer.

        Returns dict[variety, ndarray(vocab_size, proj_dim)] — same interface
        as the legacy pipeline, fully compatible with downstream W computation.
        """
        self.model.eval()

        # Build word -> vocab index lookup
        word_to_idx = {w: i for i, w in enumerate(vocab)}
        vocab_size = len(vocab)

        result: dict[str, np.ndarray] = {}

        for variety, docs in corpus_by_variety.items():
            logger.info("Extracting embeddings for %s (%d docs)", variety, len(docs))

            # Accumulators: sum of embeddings and count per word
            emb_sum = np.zeros((vocab_size, self.proj_dim), dtype=np.float64)
            emb_count = np.zeros(vocab_size, dtype=np.int64)

            # Process in batches
            texts = [f"[VAR_{variety}] {doc}" for doc in docs if doc.strip()]

            for batch_start in range(0, len(texts), self.batch_size):
                batch_texts = texts[batch_start : batch_start + self.batch_size]
                batch_raw_docs = [doc for doc in docs[batch_start : batch_start + self.batch_size] if doc.strip()]

                # Tokenize
                encoding = self.tokenizer(
                    batch_texts,
                    max_length=self.max_length,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    return_offsets_mapping=True,
                )

                input_ids = encoding["input_ids"].to(self.device)
                attention_mask = encoding["attention_mask"].to(self.device)

                # Get projected embeddings
                with torch.no_grad():
                    projected = self.model.get_projected_embeddings(
                        input_ids, attention_mask,
                    )  # (batch, seq_len, proj_dim)

                projected_np = projected.cpu().numpy()

                # Map tokens back to original words
                for i, raw_doc in enumerate(batch_raw_docs):
                    words = raw_doc.lower().split()
                    # Use tokenizer to get word_ids mapping
                    enc_single = self.tokenizer(
                        batch_texts[i],
                        max_length=self.max_length,
                        padding=False,
                        truncation=True,
                    )

                    # Decode each token to approximate word mapping
                    token_ids = enc_single["input_ids"]
                    tokens = self.tokenizer.convert_ids_to_tokens(token_ids)

                    # Build token->word mapping via character offsets
                    current_word_idx = -1  # -1 for [VAR_*] prefix
                    word_positions: dict[int, list[int]] = defaultdict(list)

                    for tok_idx, token in enumerate(tokens):
                        if tok_idx >= projected_np.shape[1]:
                            break
                        # Skip special tokens
                        if token in ("[CLS]", "[SEP]", "[PAD]") or token.startswith("[VAR_"):
                            continue
                        # WordPiece: tokens starting with ## are continuations
                        if token.startswith("##"):
                            if current_word_idx >= 0:
                                word_positions[current_word_idx].append(tok_idx)
                        else:
                            current_word_idx += 1
                            if current_word_idx < len(words):
                                word_positions[current_word_idx].append(tok_idx)

                    # Accumulate embeddings per word
                    for w_idx, tok_positions in word_positions.items():
                        if w_idx >= len(words):
                            break
                        word = words[w_idx]
                        vocab_idx = word_to_idx.get(word)
                        if vocab_idx is None:
                            continue

                        # Mean of token embeddings for this word
                        word_emb = projected_np[i, tok_positions, :].mean(axis=0)
                        emb_sum[vocab_idx] += word_emb
                        emb_count[vocab_idx] += 1

            # Average accumulated embeddings
            mask = emb_count > 0
            embeddings = np.zeros((vocab_size, self.proj_dim), dtype=np.float64)
            embeddings[mask] = emb_sum[mask] / emb_count[mask, np.newaxis]

            # Fallback for unseen words: encode in isolation
            unseen = np.where(~mask)[0]
            if len(unseen) > 0:
                logger.info("%s: %d/%d words unseen, using fallback", variety, len(unseen), vocab_size)
                for idx in unseen:
                    embeddings[idx] = self._fallback_embed(vocab[idx], variety)

            result[variety] = embeddings
            logger.info("%s: composed %d word vectors (%.1f%% from context)",
                        variety, vocab_size, 100 * mask.sum() / vocab_size)

        return result

    def _fallback_embed(self, word: str, variety: str) -> np.ndarray:
        """Embed a single word in isolation with variety token."""
        text = f"[VAR_{variety}] {word}"
        encoding = self.tokenizer(
            text,
            max_length=32,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            projected = self.model.get_projected_embeddings(input_ids, attention_mask)

        # Average non-special tokens (skip [CLS], [VAR_*], [SEP], [PAD])
        mask = attention_mask.squeeze(0).cpu().numpy().astype(bool)
        emb = projected.squeeze(0).cpu().numpy()
        # Skip first 2 tokens ([CLS] and [VAR_*])
        if mask.sum() > 2:
            return emb[2:mask.sum()].mean(axis=0)
        return emb[mask].mean(axis=0) if mask.any() else np.zeros(self.proj_dim)


# ---------------------------------------------------------------------------
# Legacy composer (kept for backward compatibility)
# ---------------------------------------------------------------------------

class SubwordToWordComposer:
    """Compose word-level vectors from BPE subword embeddings by mean-pooling."""

    def __init__(self, tokenizer, subword_embeddings: np.ndarray):
        self._tokenizer = tokenizer
        self._emb = subword_embeddings

    def compose_word(self, word: str) -> np.ndarray:
        """Compose a single word's embedding from its subword pieces."""
        ids = self._tokenizer.encode(word)
        if not ids:
            return np.zeros(self._emb.shape[1], dtype=np.float64)
        vectors = []
        for idx in ids:
            if 0 <= idx < len(self._emb):
                vectors.append(self._emb[idx])
        if not vectors:
            return np.zeros(self._emb.shape[1], dtype=np.float64)
        return np.mean(vectors, axis=0).astype(np.float64)

    def compose_vocabulary(self, words: list[str]) -> np.ndarray:
        """Compose word-level embedding matrix for a vocabulary.

        Returns (n_words, dim) word-level embedding matrix.
        """
        dim = self._emb.shape[1]
        result = np.zeros((len(words), dim), dtype=np.float64)
        for i, word in enumerate(words):
            result[i] = self.compose_word(word)
        return result
