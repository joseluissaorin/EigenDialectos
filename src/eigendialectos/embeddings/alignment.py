"""Cross-variety alignment orchestrator.

Provides a high-level interface that aligns embedding matrices from
all dialect varieties into a shared reference space, delegating to
one of the concrete alignment algorithms (Procrustes, VecMap, MUSE).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from eigendialectos.constants import DialectCode
from eigendialectos.embeddings.contrastive.muse import MUSEAligner
from eigendialectos.embeddings.contrastive.procrustes import ProcrustesAligner
from eigendialectos.embeddings.contrastive.vecmap import VecMapAligner
from eigendialectos.types import EmbeddingMatrix

logger = logging.getLogger(__name__)

_ALIGNER_MAP = {
    "procrustes": ProcrustesAligner,
    "vecmap": VecMapAligner,
    "muse": MUSEAligner,
}


class CrossVarietyAligner:
    """Orchestrates alignment of multiple dialect embedding spaces.

    Given a dictionary mapping ``DialectCode`` to ``EmbeddingMatrix``,
    aligns every variety into the space of a chosen reference dialect.

    Parameters
    ----------
    method:
        Alignment algorithm: ``'procrustes'``, ``'vecmap'``, or ``'muse'``.
    reference:
        Reference dialect whose space all others are mapped into.
    aligner_kwargs:
        Extra keyword arguments forwarded to the aligner constructor.
    """

    def __init__(
        self,
        method: str = "procrustes",
        reference: DialectCode = DialectCode.ES_PEN,
        **aligner_kwargs: Any,
    ) -> None:
        if method not in _ALIGNER_MAP:
            raise ValueError(
                f"Unknown alignment method '{method}'. "
                f"Choose from: {', '.join(sorted(_ALIGNER_MAP))}"
            )
        self._method = method
        self._reference = reference
        self._aligner_kwargs = aligner_kwargs
        self._alignment_matrices: dict[DialectCode, np.ndarray] = {}

    def align_all(
        self,
        embeddings: dict[DialectCode, EmbeddingMatrix],
        reference: DialectCode | None = None,
        method: str | None = None,
        anchors: dict[DialectCode, list[str]] | None = None,
    ) -> dict[DialectCode, EmbeddingMatrix]:
        """Align all dialect varieties to the reference space.

        Parameters
        ----------
        embeddings:
            Maps each ``DialectCode`` to its trained ``EmbeddingMatrix``.
        reference:
            Override the default reference dialect.
        method:
            Override the default alignment method.
        anchors:
            Per-dialect anchor word lists.  If provided, ``anchors[dc]``
            is the list of anchor words for aligning dialect ``dc`` to
            the reference.  If ``None``, vocabulary intersections are used.

        Returns
        -------
        dict[DialectCode, EmbeddingMatrix]
            Aligned embedding matrices.  The reference dialect is
            included unchanged.
        """
        ref = reference or self._reference
        meth = method or self._method

        if ref not in embeddings:
            raise ValueError(
                f"Reference dialect {ref.value} not found in embeddings. "
                f"Available: {[dc.value for dc in embeddings]}"
            )

        target = embeddings[ref]
        result: dict[DialectCode, EmbeddingMatrix] = {ref: target}
        self._alignment_matrices = {}

        for dc, source in embeddings.items():
            if dc == ref:
                continue

            logger.info(
                "Aligning %s -> %s using %s",
                dc.value, ref.value, meth,
            )

            # Create a fresh aligner for each pair
            aligner_cls = _ALIGNER_MAP[meth]
            aligner = aligner_cls(**self._aligner_kwargs)

            dc_anchors = None
            if anchors is not None and dc in anchors:
                dc_anchors = anchors[dc]

            W = aligner.align(source, target, anchors=dc_anchors)
            self._alignment_matrices[dc] = W

            aligned = aligner.transform(source, W)
            result[dc] = aligned

        logger.info(
            "Alignment complete: %d varieties aligned to %s",
            len(result) - 1, ref.value,
        )
        return result

    @property
    def alignment_matrices(self) -> dict[DialectCode, np.ndarray]:
        """Return the computed alignment matrices keyed by source dialect.

        The reference dialect is not included (identity).
        """
        return dict(self._alignment_matrices)

    @property
    def reference(self) -> DialectCode:
        return self._reference

    @property
    def method(self) -> str:
        return self._method
