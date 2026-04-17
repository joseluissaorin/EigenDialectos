"""Per-variety embedding tables for DCL training.

Uses a single flat ``nn.Embedding`` of size ``(n_varieties * vocab_size, dim)``
so that per-batch variety dispatch is a simple index arithmetic operation
(``variety * vocab_size + token``), eliminating the expensive ``torch.stack``
that the original ``nn.ModuleList`` approach required on every forward pass.

This is critical for MPS (Apple Silicon GPU) throughput where kernel dispatch
overhead and memory allocation dominate for small models.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DCLEmbeddingModel(nn.Module):
    """Per-variety word and context embedding tables (flat layout).

    Stores all varieties in a single ``nn.Embedding`` table of size
    ``(n_varieties * vocab_size, embedding_dim)``.  Variety-specific
    lookup is: ``emb(variety_idx * vocab_size + token_idx)``.

    Parameters
    ----------
    vocab_size:
        Number of tokens in the shared vocabulary.
    embedding_dim:
        Dimensionality of each embedding vector.
    n_varieties:
        Number of dialect varieties (default 8 for EigenDialectos).
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        n_varieties: int = 8,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.n_varieties = n_varieties

        total = n_varieties * vocab_size
        self.word_emb = nn.Embedding(total, embedding_dim)
        self.ctx_emb = nn.Embedding(total, embedding_dim)

        nn.init.xavier_uniform_(self.word_emb.weight)
        nn.init.xavier_uniform_(self.ctx_emb.weight)

    def _lookup(
        self,
        emb: nn.Embedding,
        token_indices: torch.Tensor,
        variety_indices: torch.Tensor,
    ) -> torch.Tensor:
        flat_idx = variety_indices * self.vocab_size + token_indices
        return emb(flat_idx)

    def forward(
        self,
        word_indices: torch.Tensor,
        ctx_indices_a: torch.Tensor,
        ctx_indices_b: torch.Tensor,
        variety_a_idx: torch.Tensor,
        variety_b_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Look up embeddings for a mini-batch.

        All index tensors have shape ``(batch,)``.

        Returns
        -------
        tuple of four ``(batch, dim)`` tensors:
            ``(word_emb_a, ctx_emb_a, ctx_emb_b, word_emb_b)``
        """
        word_emb_a = self._lookup(self.word_emb, word_indices, variety_a_idx)
        ctx_emb_a = self._lookup(self.ctx_emb, ctx_indices_a, variety_a_idx)
        ctx_emb_b = self._lookup(self.ctx_emb, ctx_indices_b, variety_b_idx)
        word_emb_b = self._lookup(self.word_emb, word_indices, variety_b_idx)
        return word_emb_a, ctx_emb_a, ctx_emb_b, word_emb_b

    def get_word_embeddings(self, variety_idx: int) -> torch.Tensor:
        """Return the full word embedding matrix for a given variety.

        Returns
        -------
        torch.Tensor
            ``(vocab_size, embedding_dim)`` weight matrix (detached).
        """
        start = variety_idx * self.vocab_size
        end = start + self.vocab_size
        return self.word_emb.weight[start:end].detach()
