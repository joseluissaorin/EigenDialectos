"""Synthetic dialect sample generator.

Transforms a bank of neutral Spanish sentences through dialect-specific
templates to produce labelled :class:`DialectSample` instances.
"""

from __future__ import annotations

import random
from typing import Optional

from eigendialectos.constants import DialectCode
from eigendialectos.types import CorpusSlice, DialectSample

from eigendialectos.corpus.synthetic.templates import (
    DIALECT_TEMPLATES,
    DialectTemplate,
)

# ======================================================================
# Neutral base sentences (Peninsular-neutral register)
# ======================================================================

BASE_SENTENCES: list[str] = [
    "Vamos a coger el autobús para ir al centro de la ciudad.",
    "¿Has visto la película que echan esta noche en la televisión?",
    "Me he comprado un ordenador nuevo y estoy muy contento.",
    "No me apetece nada tener que madrugar mañana por la mañana.",
    "Quedamos a las ocho en la plaza del pueblo.",
    "El piso que hemos visto en el centro está muy bien.",
    "¿Vosotros ya habéis cenado o queréis pedir algo?",
    "Hoy he ido al médico y me ha dicho que estoy bien.",
    "¿Tienes gafas de sol? Las he olvidado en casa.",
    "Hace mucho frío, vamos a meternos en el bar.",
    "He aparcado el coche junto a la fuente de la plaza.",
    "Vosotros sabéis que aquí siempre llueve en noviembre.",
    "Le he llamado por el móvil pero no me ha contestado.",
    "Este fin de semana nos vamos al pueblo de mis abuelos.",
    "Me gusta mucho este sitio para tomar algo por la tarde.",
    "¿Sabes dónde para el autobús cerca de aquí?",
    "He estado trabajando todo el día y estoy muy cansado.",
    "¿Quieres que pidamos algo para cenar esta noche?",
    "El chico de la tienda me ha dado un precio muy bueno.",
    "Vamos a comprar un poco de pan y fruta en el mercado.",
    "La chica que trabaja en la oficina es muy simpática.",
    "He perdido el trabajo y ahora estoy buscando algo nuevo.",
    "¿Vosotros habéis estado alguna vez en esa ciudad?",
    "Hace mucho calor, vamos a buscar un sitio con sombra.",
    "Mi amigo me ha prestado dinero para pagar el alquiler.",
    "¿Puedes venir un momento? Necesito hablar contigo.",
    "El profesor nos ha puesto un examen para la semana que viene.",
    "Hemos quedado con los amigos para ir a cenar fuera.",
    "No entiendo por qué siempre llegas tarde a todos los sitios.",
    "¿Has probado la comida de ese restaurante nuevo del centro?",
    "Me he comprado un coche nuevo y gasta muy poco.",
    "La verdad es que no me gusta nada este tiempo tan malo.",
    "¿Vosotros queréis ir al cine o preferís hacer otra cosa?",
    "He visto a tu hermana en la calle y me ha saludado.",
    "Necesito comprarme unas gafas nuevas porque estas están rotas.",
    "¿Sabes si hay algún autobús que vaya directamente al aeropuerto?",
    "Mi madre me ha llamado para decirme que viene a visitarme.",
    "El médico me ha recetado unas pastillas para el dolor de cabeza.",
    "¿Tienes el móvil a mano? Necesito hacer una llamada urgente.",
    "Vamos a salir a dar un paseo antes de que anochezca.",
    "He encontrado un piso muy bonito y además es bastante barato.",
    "¿Vosotros sabéis cómo se llega al museo desde aquí?",
    "Mi hermano trabaja en una fábrica y gana bastante dinero.",
    "La película que hemos visto esta tarde era muy aburrida.",
    "¿Puedes ayudarme a mover estos muebles al otro lado?",
    "He conocido a una chica muy interesante en la fiesta de anoche.",
    "Creo que vamos a necesitar más dinero para el viaje.",
    "¿Sabes si el autobús pasa por esta calle o por la otra?",
    "Me parece que hoy hace más frío que ayer por la mañana.",
    "Nos hemos mudado a un piso nuevo cerca del parque.",
    "¿Vosotros habéis reservado mesa para la cena de esta noche?",
    "El niño no quiere comer nada y su madre está preocupada.",
    "He terminado el trabajo y ahora puedo descansar un poco.",
    "¿Quieres ir a tomar algo después del trabajo esta tarde?",
    "Mi padre siempre dice que hay que madrugar para ser productivo.",
]


class SyntheticGenerator:
    """Generates synthetic dialect samples by applying transformation templates
    to neutral base sentences.

    Parameters
    ----------
    seed:
        Random seed for reproducible generation.  ``None`` disables seeding.
    """

    def __init__(self, seed: Optional[int] = 42) -> None:
        self._rng = random.Random(seed)
        self._base_sentences = list(BASE_SENTENCES)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        n_samples: int,
        dialect_code: DialectCode,
    ) -> list[DialectSample]:
        """Generate *n_samples* synthetic samples for the given dialect.

        Sentences are drawn (with replacement if *n_samples* exceeds the
        base bank) from the neutral sentence bank, then transformed through
        the dialect's template.  A pragmatic marker may be prepended or
        appended to add naturalness.

        Parameters
        ----------
        n_samples:
            Number of samples to generate.
        dialect_code:
            Target dialect for transformation.

        Returns
        -------
        list[DialectSample]
            Generated samples with ``source_id="synthetic_generator"`` and
            ``confidence`` reflecting the transformation quality (0.7-0.9).
        """
        template = DIALECT_TEMPLATES.get(dialect_code)
        if template is None:
            label = dialect_code.value if hasattr(dialect_code, "value") else dialect_code
            raise ValueError(
                f"No transformation template for dialect {label}"
            )

        chosen = self._rng.choices(self._base_sentences, k=n_samples)
        samples: list[DialectSample] = []

        for i, base in enumerate(chosen):
            transformed = template.apply_all(base)
            transformed = self._maybe_add_marker(transformed, template)
            confidence = round(self._rng.uniform(0.7, 0.9), 3)

            samples.append(
                DialectSample(
                    text=transformed,
                    dialect_code=dialect_code,
                    source_id="synthetic_generator",
                    confidence=confidence,
                    metadata={
                        "base_sentence": base,
                        "generation_index": i,
                    },
                )
            )

        return samples

    def generate_all(
        self,
        n_per_dialect: int,
    ) -> dict[DialectCode, CorpusSlice]:
        """Generate samples for every dialect.

        Parameters
        ----------
        n_per_dialect:
            Number of samples per dialect.

        Returns
        -------
        dict[DialectCode, CorpusSlice]
        """
        result: dict[DialectCode, CorpusSlice] = {}
        for code in DialectCode:
            samples = self.generate(n_per_dialect, code)
            result[code] = CorpusSlice(samples=samples, dialect_code=code)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_add_marker(
        self,
        text: str,
        template: DialectTemplate,
    ) -> str:
        """Randomly prepend or append a pragmatic marker (~30 % chance)."""
        if not template.pragmatic_markers:
            return text
        if self._rng.random() < 0.3:
            marker = self._rng.choice(template.pragmatic_markers)
            # Markers that are tag questions go at the end
            if marker.startswith("¿"):
                text = text.rstrip(".") + ", " + marker
            else:
                # Capitalise and prepend
                text = marker.capitalize() + ", " + text[0].lower() + text[1:]
        return text

    @property
    def base_sentence_count(self) -> int:
        """Number of neutral base sentences in the bank."""
        return len(self._base_sentences)

    def add_base_sentences(self, sentences: list[str]) -> None:
        """Add additional base sentences to the bank."""
        self._base_sentences.extend(sentences)
