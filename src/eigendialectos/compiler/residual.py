"""Residual correction network for the SDC compiler.

A small transformer-based network (2-4 layers, NOT an LLM) that learns
to fix systematic errors from pure algebraic transformation. Trained on
(SDC_output, ground_truth) pairs from the corpus.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


if _HAS_TORCH:

    class ResidualCorrectionNetwork(nn.Module):
        """Small transformer encoder for correcting SDC output.

        NOT an LLM — just 2-4 transformer encoder layers trained to
        fix systematic errors from algebraic transformation.

        Parameters
        ----------
        vocab_size : int
            Size of the shared vocabulary.
        d_model : int
            Model dimension (default 256).
        nhead : int
            Number of attention heads (default 4).
        num_layers : int
            Number of transformer encoder layers (default 2).
        dim_feedforward : int
            Feedforward dimension (default 512).
        max_seq_len : int
            Maximum sequence length (default 512).
        dropout : float
            Dropout rate (default 0.1).
        """

        def __init__(
            self,
            vocab_size: int,
            d_model: int = 256,
            nhead: int = 4,
            num_layers: int = 2,
            dim_feedforward: int = 512,
            max_seq_len: int = 512,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.vocab_size = vocab_size
            self.d_model = d_model

            self.embedding = nn.Embedding(vocab_size, d_model)
            self.pos_encoding = nn.Parameter(
                self._sinusoidal_encoding(max_seq_len, d_model)
            )
            self.dropout = nn.Dropout(dropout)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
            )
            self.output_proj = nn.Linear(d_model, vocab_size)

            # Initialize weights
            nn.init.xavier_uniform_(self.embedding.weight)
            nn.init.xavier_uniform_(self.output_proj.weight)
            nn.init.zeros_(self.output_proj.bias)

        @staticmethod
        def _sinusoidal_encoding(max_len: int, d_model: int) -> torch.Tensor:
            """Create sinusoidal positional encoding."""
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            return pe.unsqueeze(0)  # (1, max_len, d_model)

        def forward(
            self,
            token_ids: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """Forward pass: token_ids → correction logits.

            Parameters
            ----------
            token_ids : Tensor, shape (batch, seq_len)
            mask : optional padding mask, shape (batch, seq_len), True = pad

            Returns
            -------
            Tensor, shape (batch, seq_len, vocab_size)
            """
            seq_len = token_ids.size(1)
            x = self.embedding(token_ids) * math.sqrt(self.d_model)
            x = x + self.pos_encoding[:, :seq_len, :]
            x = self.dropout(x)
            x = self.transformer(x, src_key_padding_mask=mask)
            logits = self.output_proj(x)
            return logits

        @property
        def num_parameters(self) -> int:
            """Total number of trainable parameters."""
            return sum(p.numel() for p in self.parameters() if p.requires_grad)


    class ResidualTrainer:
        """Train the residual correction network.

        Training data: (SDC_output, ground_truth) pairs where ground_truth
        is actual text in the target variety from the corpus.

        Parameters
        ----------
        vocab_size : int
        d_model : int
        num_layers : int
        lr : float
        epochs : int
        batch_size : int
        """

        def __init__(
            self,
            vocab_size: int,
            d_model: int = 256,
            num_layers: int = 2,
            lr: float = 1e-4,
            epochs: int = 10,
            batch_size: int = 32,
        ) -> None:
            self.vocab_size = vocab_size
            self.d_model = d_model
            self.num_layers = num_layers
            self.lr = lr
            self.epochs = epochs
            self.batch_size = batch_size
            self.model: Optional[ResidualCorrectionNetwork] = None
            self.vocab: dict[str, int] = {}
            self.idx_to_word: dict[int, str] = {}

        def build_vocab(self, texts: list[str]) -> dict[str, int]:
            """Build vocabulary from training texts."""
            word_set: set[str] = {"<pad>", "<unk>"}
            for text in texts:
                for word in text.lower().split():
                    word_set.add(word)

            self.vocab = {w: i for i, w in enumerate(sorted(word_set))}
            self.idx_to_word = {i: w for w, i in self.vocab.items()}
            return self.vocab

        def _tokenize(self, text: str, max_len: int = 128) -> list[int]:
            """Convert text to token indices."""
            pad_idx = self.vocab.get("<pad>", 0)
            unk_idx = self.vocab.get("<unk>", 1)
            tokens = text.lower().split()[:max_len]
            indices = [self.vocab.get(t, unk_idx) for t in tokens]
            # Pad
            while len(indices) < max_len:
                indices.append(pad_idx)
            return indices

        def train(
            self,
            sdc_outputs: list[str],
            ground_truths: list[str],
        ) -> ResidualCorrectionNetwork:
            """Train residual correction network.

            Parameters
            ----------
            sdc_outputs : list of SDC output texts
            ground_truths : list of corresponding actual target texts

            Returns
            -------
            Trained ResidualCorrectionNetwork
            """
            assert len(sdc_outputs) == len(ground_truths)

            # Build vocabulary from both sources
            all_texts = sdc_outputs + ground_truths
            self.build_vocab(all_texts)
            actual_vocab_size = len(self.vocab)

            # Create model
            self.model = ResidualCorrectionNetwork(
                vocab_size=actual_vocab_size,
                d_model=self.d_model,
                num_layers=self.num_layers,
            )
            logger.info(
                "Residual network: %d parameters, vocab size %d",
                self.model.num_parameters,
                actual_vocab_size,
            )

            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            loss_fn = nn.CrossEntropyLoss(
                ignore_index=self.vocab.get("<pad>", 0)
            )

            # Prepare data
            max_len = 128
            src_ids = [self._tokenize(t, max_len) for t in sdc_outputs]
            tgt_ids = [self._tokenize(t, max_len) for t in ground_truths]

            src_tensor = torch.tensor(src_ids, dtype=torch.long)
            tgt_tensor = torch.tensor(tgt_ids, dtype=torch.long)

            n = len(src_ids)
            self.model.train()

            for epoch in range(self.epochs):
                # Shuffle
                perm = torch.randperm(n)
                epoch_loss = 0.0
                n_batches = 0

                for start in range(0, n, self.batch_size):
                    end = min(start + self.batch_size, n)
                    idx = perm[start:end]

                    src_batch = src_tensor[idx]
                    tgt_batch = tgt_tensor[idx]

                    # Padding mask
                    pad_mask = src_batch == self.vocab.get("<pad>", 0)

                    # Forward
                    logits = self.model(src_batch, mask=pad_mask)

                    # Loss: predict target tokens from source tokens
                    loss = loss_fn(
                        logits.reshape(-1, actual_vocab_size),
                        tgt_batch.reshape(-1),
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()

                    epoch_loss += loss.item()
                    n_batches += 1

                avg_loss = epoch_loss / max(n_batches, 1)
                logger.info("Epoch %d/%d: loss = %.4f", epoch + 1, self.epochs, avg_loss)

            self.model.eval()
            return self.model

        def save(self, output_dir: Path) -> None:
            """Save model and vocabulary."""
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            if self.model is not None:
                torch.save(self.model.state_dict(), output_dir / "residual_model.pt")
            with open(output_dir / "residual_vocab.json", "w") as f:
                json.dump(self.vocab, f, ensure_ascii=False, indent=2)

        def load(self, output_dir: Path) -> ResidualCorrectionNetwork:
            """Load model and vocabulary."""
            output_dir = Path(output_dir)
            with open(output_dir / "residual_vocab.json") as f:
                self.vocab = json.load(f)
            self.idx_to_word = {i: w for w, i in self.vocab.items()}

            self.model = ResidualCorrectionNetwork(
                vocab_size=len(self.vocab),
                d_model=self.d_model,
                num_layers=self.num_layers,
            )
            self.model.load_state_dict(
                torch.load(output_dir / "residual_model.pt", weights_only=True)
            )
            self.model.eval()
            return self.model


    class ResidualCorrector:
        """Inference wrapper for the residual correction network."""

        def __init__(
            self,
            model: ResidualCorrectionNetwork,
            vocab: dict[str, int],
        ) -> None:
            self.model = model
            self.vocab = vocab
            self.idx_to_word = {i: w for w, i in vocab.items()}
            self.model.eval()

        def correct(self, sdc_output: str, max_len: int = 128) -> str:
            """Apply residual correction to SDC output.

            For each position, if the model's top prediction differs from
            the input and has high confidence, replace the token.
            """
            pad_idx = self.vocab.get("<pad>", 0)
            unk_idx = self.vocab.get("<unk>", 1)

            tokens = sdc_output.lower().split()[:max_len]
            indices = [self.vocab.get(t, unk_idx) for t in tokens]
            original_len = len(indices)

            while len(indices) < max_len:
                indices.append(pad_idx)

            input_tensor = torch.tensor([indices], dtype=torch.long)
            pad_mask = input_tensor == pad_idx

            with torch.no_grad():
                logits = self.model(input_tensor, mask=pad_mask)
                probs = F.softmax(logits[0], dim=-1)

            corrected_tokens: list[str] = []
            for i in range(original_len):
                top_prob, top_idx = probs[i].max(dim=0)
                top_word = self.idx_to_word.get(top_idx.item(), tokens[i])

                # Only correct if model is confident and suggests something different
                if top_prob.item() > 0.6 and top_word != "<pad>" and top_word != "<unk>":
                    corrected_tokens.append(top_word)
                else:
                    corrected_tokens.append(tokens[i])

            return " ".join(corrected_tokens)

else:
    # Stub when torch is not available
    class ResidualCorrectionNetwork:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for ResidualCorrectionNetwork")

    class ResidualTrainer:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for ResidualTrainer")

    class ResidualCorrector:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for ResidualCorrector")
