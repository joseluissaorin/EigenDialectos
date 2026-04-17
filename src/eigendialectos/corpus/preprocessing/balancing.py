"""Temperature-scaled corpus balancing across dialect varieties.

Uses the XLM-R approach: n_target = n_max * (n_i / n_max)^T where T
controls how aggressively to upsample.  T=1 means no change, T=0 means
all varieties get equal count, T=0.7 is a moderate middle ground.
"""

from __future__ import annotations

import logging
import random
from typing import Optional

logger = logging.getLogger(__name__)


def balance_corpus(
    corpus_by_variety: dict[str, list[str]],
    temperature: float = 0.7,
    max_ratio: float = 3.0,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Balance corpus sizes across varieties via temperature-scaled upsampling.

    Parameters
    ----------
    corpus_by_variety:
        Mapping from variety name to list of text documents.
    temperature:
        Controls upsampling aggressiveness.  T=1 means no change,
        T=0 means all varieties reach the same count.
    max_ratio:
        Maximum upsampling ratio for any single variety (prevents
        excessive duplication that leads to memorisation).
    seed:
        Random seed for reproducible sampling.

    Returns
    -------
    dict with same keys, values are balanced document lists.
    """
    rng = random.Random(seed)
    sizes = {v: len(docs) for v, docs in corpus_by_variety.items()}
    n_max = max(sizes.values())

    balanced: dict[str, list[str]] = {}
    for variety, docs in corpus_by_variety.items():
        n_orig = len(docs)
        if n_orig == 0:
            balanced[variety] = []
            continue

        ratio = n_orig / n_max
        n_target = int(n_max * (ratio ** temperature))
        n_target = min(n_target, int(n_orig * max_ratio))
        n_target = max(n_target, n_orig)  # never downsample

        if n_target > n_orig:
            repeats = n_target // n_orig
            remainder = n_target % n_orig
            balanced_docs = docs * repeats + rng.sample(docs, remainder)
        else:
            balanced_docs = list(docs)

        balanced[variety] = balanced_docs
        if n_target != n_orig:
            logger.info(
                "%s: %d → %d documents (%.1fx)",
                variety, n_orig, n_target, n_target / n_orig,
            )

    return balanced
