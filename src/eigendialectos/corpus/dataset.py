"""Unified dialect dataset with train/val/test splitting.

Wraps a collection of :class:`CorpusSlice` instances (one per dialect)
into a single :class:`DialectDataset` that supports random access,
stratified splitting, and basic statistics.
"""

from __future__ import annotations

import math
import random
from typing import Optional, Sequence

from eigendialectos.constants import DialectCode
from eigendialectos.types import CorpusSlice, DialectSample


class DialectDataset:
    """Unified dialect dataset backed by per-dialect corpus slices.

    Parameters
    ----------
    slices:
        Mapping from :class:`DialectCode` to :class:`CorpusSlice`.
        Each slice contains the samples for one dialect.
    """

    def __init__(self, slices: dict[DialectCode, CorpusSlice]) -> None:
        self._slices = dict(slices)
        # Build a flat index for random access
        self._flat: list[DialectSample] = []
        for code in sorted(self._slices, key=lambda c: c.value):
            self._flat.extend(self._slices[code].samples)

    # ------------------------------------------------------------------
    # Core access
    # ------------------------------------------------------------------

    def get_slice(self, code: DialectCode) -> CorpusSlice:
        """Return the :class:`CorpusSlice` for *code*.

        Raises
        ------
        KeyError
            If *code* is not in the dataset.
        """
        if code not in self._slices:
            available = [c.value for c in self._slices]
            raise KeyError(
                f"Dialect {code.value} not in dataset. "
                f"Available: {available}"
            )
        return self._slices[code]

    def all_samples(self) -> list[DialectSample]:
        """Return a flat list of all samples across all dialects."""
        return list(self._flat)

    def sample(
        self,
        n: int,
        dialect_code: Optional[DialectCode] = None,
        seed: Optional[int] = None,
    ) -> list[DialectSample]:
        """Return *n* random samples.

        Parameters
        ----------
        n:
            Number of samples to draw (with replacement if *n* exceeds
            the available pool).
        dialect_code:
            If given, sample only from that dialect.  Otherwise sample
            from all dialects.
        seed:
            Random seed for reproducibility.
        """
        rng = random.Random(seed)
        if dialect_code is not None:
            pool = self.get_slice(dialect_code).samples
        else:
            pool = self._flat

        if not pool:
            return []
        if n >= len(pool):
            return list(pool)
        return rng.sample(pool, n)

    @property
    def dialect_codes(self) -> list[DialectCode]:
        """Sorted list of dialect codes present in the dataset."""
        return sorted(self._slices, key=lambda c: c.value)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, object]:
        """Return summary statistics for the entire dataset.

        Returns a dict with:
        - ``total_samples``: total number of samples.
        - ``dialect_counts``: per-dialect sample counts.
        - ``dialect_stats``: per-dialect detailed statistics
          (from :attr:`CorpusSlice.stats`).
        - ``avg_confidence``: mean confidence across all samples.
        """
        dialect_counts: dict[str, int] = {}
        dialect_stats: dict[str, dict[str, object]] = {}
        total = 0
        confidence_sum = 0.0

        for code in self.dialect_codes:
            sl = self._slices[code]
            count = len(sl.samples)
            total += count
            dialect_counts[code.value] = count
            dialect_stats[code.value] = sl.stats
            confidence_sum += sum(s.confidence for s in sl.samples)

        return {
            "total_samples": total,
            "dialect_counts": dialect_counts,
            "dialect_stats": dialect_stats,
            "avg_confidence": round(confidence_sum / total, 4) if total > 0 else 0.0,
        }

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------

    def split(
        self,
        train: float = 0.7,
        val: float = 0.15,
        test: float = 0.15,
        seed: int = 42,
    ) -> tuple[DialectDataset, DialectDataset, DialectDataset]:
        """Split into train, validation, and test datasets.

        The split is **stratified by dialect**: each dialect's samples are
        independently shuffled and divided according to the requested
        proportions, ensuring balanced representation across all splits.

        Parameters
        ----------
        train:
            Fraction of samples for the training set.
        val:
            Fraction for the validation set.
        test:
            Fraction for the test set.
        seed:
            Random seed for reproducibility.

        Returns
        -------
        tuple[DialectDataset, DialectDataset, DialectDataset]
            ``(train_ds, val_ds, test_ds)``

        Raises
        ------
        ValueError
            If the proportions do not sum to approximately 1.0.
        """
        total = train + val + test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Split proportions must sum to 1.0, got {total:.6f} "
                f"(train={train}, val={val}, test={test})"
            )

        rng = random.Random(seed)

        train_slices: dict[DialectCode, CorpusSlice] = {}
        val_slices: dict[DialectCode, CorpusSlice] = {}
        test_slices: dict[DialectCode, CorpusSlice] = {}

        for code in self.dialect_codes:
            samples = list(self._slices[code].samples)
            rng.shuffle(samples)
            n = len(samples)

            n_train = math.floor(n * train)
            n_val = math.floor(n * val)
            # test gets the remainder to avoid off-by-one losses
            n_test = n - n_train - n_val

            train_samples = samples[:n_train]
            val_samples = samples[n_train : n_train + n_val]
            test_samples = samples[n_train + n_val :]

            train_slices[code] = CorpusSlice(
                samples=train_samples, dialect_code=code,
            )
            val_slices[code] = CorpusSlice(
                samples=val_samples, dialect_code=code,
            )
            test_slices[code] = CorpusSlice(
                samples=test_samples, dialect_code=code,
            )

        return (
            DialectDataset(train_slices),
            DialectDataset(val_slices),
            DialectDataset(test_slices),
        )

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._flat)

    def __getitem__(self, index: int) -> DialectSample:
        return self._flat[index]

    def __repr__(self) -> str:
        codes = ", ".join(c.value for c in self.dialect_codes)
        return f"<DialectDataset n={len(self)} dialects=[{codes}]>"

    def __contains__(self, code: DialectCode) -> bool:
        return code in self._slices

    def __iter__(self):
        return iter(self._flat)
