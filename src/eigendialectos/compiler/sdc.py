"""Spectral Dialectal Compiler: the full SDC pipeline.

parse → per-level embed → spectral transform → kNN decode → reconstruct

Every change produces a traceable CHANGE LOG with:
level, eigenvector, eigenvalue, α applied, confidence score.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from eigendialectos.compiler.parser import CompilerParser
from eigendialectos.compiler.reconstructor import TextReconstructor
from eigendialectos.compiler.transformer import SpectralTransformer
from eigendialectos.constants import LinguisticLevel
from eigendialectos.spectral.stack import SpectralStack
from eigendialectos.types import LevelEmbedding, ParsedText, SDCResult

logger = logging.getLogger(__name__)


class SpectralDialectalCompiler:
    """The full SDC pipeline: parse → transform → reconstruct → (optional) correct.

    Usage::

        compiler = SpectralDialectalCompiler(
            spectral_stack=stack,
            source_embeddings=src_embs,
            target_embeddings=tgt_embs,
        )
        result = compiler.compile(
            "El autobús llega a las tres",
            target="canario",
            alphas={1: 0.9, 2: 0.9, 3: 0.9, 4: 0.5, 5: 0.3},
        )
        print(result.output_text)
        print(result.change_log)
    """

    def __init__(
        self,
        spectral_stack: SpectralStack,
        source_embeddings: dict[int, LevelEmbedding],
        target_embeddings: dict[int, LevelEmbedding],
        residual: Optional[Any] = None,
        source_variety: str = "neutral",
        target_variety: str = "unknown",
    ) -> None:
        self.parser = CompilerParser()
        self.transformer = SpectralTransformer(
            spectral_stack=spectral_stack,
            source_embeddings=source_embeddings,
            target_embeddings=target_embeddings,
        )
        self.reconstructor = TextReconstructor()
        self.residual = residual
        self.source_variety = source_variety
        self.target_variety = target_variety

    def compile(
        self,
        text: str,
        target: Optional[str] = None,
        alphas: Optional[dict[int, float]] = None,
        source: Optional[str] = None,
    ) -> SDCResult:
        """Full compilation pipeline.

        Parameters
        ----------
        text : str
            Input text in source variety.
        target : str or None
            Target variety name (for metadata; transform uses pre-loaded stack).
        alphas : dict or None
            Per-level α intensities. Default: all levels at 1.0.
        source : str or None
            Source variety name (for metadata).

        Returns
        -------
        SDCResult with input_text, output_text, change_log.
        """
        target_name = target or self.target_variety
        source_name = source or self.source_variety

        # Default alphas: all levels at 1.0
        if alphas is None:
            alphas = {lv.value: 1.0 for lv in LinguisticLevel}

        # Step 1: Parse
        parsed = self.parser.parse(text)
        logger.debug("Parsed %d words, %d phrases, %d sentences",
                      len(parsed.words), len(parsed.phrases), len(parsed.sentences))

        # Step 2: Transform at each level
        level_replacements: dict[int, list[tuple[str, dict[str, Any]]]] = {}
        change_log: list[dict[str, Any]] = []

        # Level 2 (Word) — primary transformation level
        if 2 in alphas:
            word_results = self.transformer.transform_level(
                level=2,
                units=parsed.words,
                alpha=alphas.get(2, 1.0),
                context=parsed.words,
            )
            level_replacements[2] = word_results

            for replacement, meta in word_results:
                if meta.get("changed", False):
                    change_log.append({
                        "change": f"{meta.get('original', '?')} → {replacement}",
                        "level": 2,
                        "level_name": "Word/Lemma",
                        "eigenvector_idx": meta.get("eigenvector_idx", -1),
                        "eigenvalue": meta.get("eigenvalue", 0),
                        "alpha": meta.get("alpha", 0),
                        "confidence": meta.get("confidence", meta.get("score", 0)),
                        "used_spectral": meta.get("used_spectral", False),
                    })

        # Level 1 (Morpheme) — secondary, fills gaps
        if 1 in alphas:
            # Flatten morphemes for transformation
            flat_morphemes = [m for morphs in parsed.morphemes for m in morphs]
            if flat_morphemes:
                morph_results = self.transformer.transform_level(
                    level=1,
                    units=flat_morphemes,
                    alpha=alphas.get(1, 1.0),
                )
                level_replacements[1] = morph_results

                for replacement, meta in morph_results:
                    if meta.get("changed", False):
                        change_log.append({
                            "change": f"{meta.get('original', '?')} → {replacement}",
                            "level": 1,
                            "level_name": "Morpheme",
                            "eigenvector_idx": meta.get("eigenvector_idx", -1),
                            "eigenvalue": meta.get("eigenvalue", 0),
                            "alpha": meta.get("alpha", 0),
                            "confidence": meta.get("score", 0),
                        })

        # Level 3 (Phrase) — phrasal replacements
        if 3 in alphas:
            phrase_strings = [" ".join(p) for p in parsed.phrases]
            if phrase_strings:
                phrase_results = self.transformer.transform_level(
                    level=3,
                    units=phrase_strings,
                    alpha=alphas.get(3, 1.0),
                    context=parsed.words,
                )
                level_replacements[3] = phrase_results

                for replacement, meta in phrase_results:
                    if meta.get("changed", False):
                        change_log.append({
                            "change": f"{meta.get('original', '?')} → {replacement}",
                            "level": 3,
                            "level_name": "Phrase/Collocation",
                            "eigenvector_idx": meta.get("eigenvector_idx", -1),
                            "eigenvalue": meta.get("eigenvalue", 0),
                            "alpha": meta.get("alpha", 0),
                            "confidence": meta.get("score", 0),
                        })

        # Step 3: Reconstruct
        output_text = self.reconstructor.reconstruct(parsed, level_replacements)

        # Step 4: Optional residual correction
        if self.residual is not None:
            try:
                output_text = self.residual.correct(output_text)
            except Exception as e:
                logger.warning("Residual correction failed: %s", e)

        logger.info(
            "SDC: '%s' → '%s' (%d changes, target=%s)",
            text[:50],
            output_text[:50],
            len(change_log),
            target_name,
        )

        return SDCResult(
            input_text=text,
            output_text=output_text,
            source_variety=source_name,
            target_variety=target_name,
            alphas=alphas,
            change_log=change_log,
        )
