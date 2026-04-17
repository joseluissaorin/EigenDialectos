"""Fetch song lyrics as dialect corpus data via public APIs and web scraping.

Song lyrics are excellent sources of dialectal features because regional music
genres use authentic local vocabulary, slang, and speech patterns.  Each dialect
variety is associated with a curated registry of (artist, song, genre) tuples
representing genres that are rich in dialectal markers.

Lyrics are fetched in order of preference:

1. **lyrics.ovh** free API (``https://api.lyrics.ovh/v1/{artist}/{title}``).
2. **Fallback web scraping** from a lyrics aggregator site using BeautifulSoup.

Fetched lyrics are cached as plain-text files under
``data/raw/lyrics/{dialect}/{artist}_{title}.txt`` so that subsequent runs skip
already-downloaded songs.  Each stanza (separated by double newlines) becomes
one :class:`~eigendialectos.types.DialectSample`.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, NamedTuple

import requests

from eigendialectos.constants import DialectCode
from eigendialectos.types import DialectSample

try:
    from bs4 import BeautifulSoup  # type: ignore[import-untyped]

    _HAS_BS4 = True
except ImportError:
    _HAS_BS4 = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LYRICS_OVH_URL = "https://api.lyrics.ovh/v1/{artist}/{title}"
_USER_AGENT = "EigenDialectos/0.1 (research; dialect-corpus-builder)"
_REQUEST_DELAY_SECONDS = 1.0
_MIN_STANZA_LENGTH = 20

# Confidence tiers
_CONFIDENCE_GENRE_SPECIFIC = 0.80  # tango, flamenco, ranchera, etc.
_CONFIDENCE_GENERAL = 0.65  # pop, rock, etc.

# Genres considered strongly dialectal
_HIGH_CONFIDENCE_GENRES = frozenset({
    "tango",
    "flamenco",
    "sevillanas",
    "copla",
    "ranchera",
    "corrido",
    "son cubano",
    "salsa",
    "cumbia villera",
    "huayno",
    "chicha peruana",
    "música canaria",
    "timple",
    "cumbia andina",
    "folclore andino",
    "nueva canción",
    "reggaeton",
    "merengue",
    "bachata",
    "cueca",
    "regional mexicano",
})


# ---------------------------------------------------------------------------
# Song registry
# ---------------------------------------------------------------------------


class SongEntry(NamedTuple):
    """A single song in the curated registry."""

    artist: str
    title: str
    genre: str


_SONG_REGISTRY: dict[DialectCode, list[SongEntry]] = {
    # ------------------------------------------------------------------
    # ES_PEN: Peninsular Standard
    # ------------------------------------------------------------------
    DialectCode.ES_PEN: [
        # Pop español
        SongEntry("Mecano", "Me cuesta tanto olvidarte", "pop español"),
        SongEntry("Mecano", "Hijo de la luna", "pop español"),
        SongEntry("Mecano", "Mujer contra mujer", "pop español"),
        SongEntry("Alejandro Sanz", "Corazón partío", "pop español"),
        SongEntry("Alejandro Sanz", "Amiga mía", "pop español"),
        SongEntry("Alejandro Sanz", "Y si fuera ella", "pop español"),
        SongEntry("Rosalía", "Malamente", "pop español"),
        SongEntry("C. Tangana", "Tú me dejaste de querer", "pop español"),
        SongEntry("C. Tangana", "Ingobernable", "pop español"),
        # Rock español
        SongEntry("Extremoduro", "So payaso", "rock español"),
        SongEntry("Extremoduro", "La vereda de la puerta de atrás", "rock español"),
        SongEntry("Los Planetas", "Un buen día", "rock español"),
        SongEntry("Vetusta Morla", "Copenhague", "rock español"),
        SongEntry("Vetusta Morla", "Los días raros", "rock español"),
        SongEntry("Fito y Fitipaldis", "Soldadito marinero", "rock español"),
        SongEntry("Fito y Fitipaldis", "Por la boca vive el pez", "rock español"),
        SongEntry("Héroes del Silencio", "Entre dos tierras", "rock español"),
        SongEntry("Héroes del Silencio", "La sirena varada", "rock español"),
        SongEntry("El Canto del Loco", "Besos", "rock español"),
        # Indie español
        SongEntry("Triángulo de Amor Bizarro", "De la monarquía a la criptocracia", "indie español"),
        SongEntry("La Casa Azul", "La revolución sexual", "indie español"),
        SongEntry("Amaral", "Sin ti no soy nada", "pop español"),
        SongEntry("La Oreja de Van Gogh", "Rosas", "pop español"),
        SongEntry("Joaquín Sabina", "Y nos dieron las diez", "cantautor"),
        SongEntry("Joaquín Sabina", "19 días y 500 noches", "cantautor"),
        SongEntry("Joaquín Sabina", "Contigo", "cantautor"),
        SongEntry("Leiva", "Terriblemente cruel", "rock español"),
        SongEntry("Izal", "Copacabana", "indie español"),
        SongEntry("Pereza", "Estrella Polar", "rock español"),
        SongEntry("Los Secretos", "Déjame", "pop español"),
    ],
    # ------------------------------------------------------------------
    # ES_AND: Andalusian
    # ------------------------------------------------------------------
    DialectCode.ES_AND: [
        # Flamenco
        SongEntry("Camarón de la Isla", "Como el agua", "flamenco"),
        SongEntry("Camarón de la Isla", "La leyenda del tiempo", "flamenco"),
        SongEntry("Camarón de la Isla", "Volando voy", "flamenco"),
        SongEntry("Camarón de la Isla", "Soy gitano", "flamenco"),
        SongEntry("Pata Negra", "Yo me quedo en Sevilla", "flamenco"),
        SongEntry("Pata Negra", "Camarón", "flamenco"),
        SongEntry("Niña Pastori", "Cai", "flamenco"),
        SongEntry("Niña Pastori", "Cañailla", "flamenco"),
        SongEntry("El Barrio", "Mal de amores", "flamenco"),
        SongEntry("El Barrio", "Yo sueno flamenco", "flamenco"),
        SongEntry("Demarco Flamenco", "Pa eso estoy yo", "flamenco"),
        SongEntry("Demarco Flamenco", "La isla del amor", "flamenco"),
        SongEntry("Rosalía", "De aquí no sales", "flamenco"),
        SongEntry("Rosalía", "Catalina", "flamenco"),
        SongEntry("Ketama", "No estamos lokos", "flamenco"),
        # Sevillanas
        SongEntry("Los del Río", "Macarena", "sevillanas"),
        SongEntry("Los del Río", "Sevilla tiene un color especial", "sevillanas"),
        SongEntry("Ecos del Rocío", "Amigo", "sevillanas"),
        SongEntry("Ecos del Rocío", "Una barca muy lejana", "sevillanas"),
        # Copla
        SongEntry("Lola Flores", "Pena penita pena", "copla"),
        SongEntry("Lola Flores", "A tu vera", "copla"),
        SongEntry("Rocío Jurado", "Como una ola", "copla"),
        SongEntry("Rocío Jurado", "Se nos rompió el amor", "copla"),
        SongEntry("Isabel Pantoja", "Marinero de luces", "copla"),
        SongEntry("Isabel Pantoja", "Así fue", "copla"),
        SongEntry("Estrella Morente", "Volver", "flamenco"),
        SongEntry("Manolo García", "Pájaros de barro", "pop andaluz"),
        SongEntry("Chambao", "Papeles mojados", "flamenco chill"),
        SongEntry("Chambao", "Pokito a poko", "flamenco chill"),
        SongEntry("Raimundo Amador", "Noche de flamenco y blues", "flamenco"),
    ],
    # ------------------------------------------------------------------
    # ES_CAN: Canarian
    # ------------------------------------------------------------------
    DialectCode.ES_CAN: [
        # Música canaria
        SongEntry("Los Sabandeños", "Ay Teror", "música canaria"),
        SongEntry("Los Sabandeños", "Canarias", "música canaria"),
        SongEntry("Los Sabandeños", "Polka canaria", "música canaria"),
        SongEntry("Los Sabandeños", "Guadalajara en un llano", "música canaria"),
        SongEntry("Los Sabandeños", "Romance del Conde Olinos", "música canaria"),
        SongEntry("Los Sabandeños", "Arrorró", "música canaria"),
        SongEntry("Los Sabandeños", "El reloj", "música canaria"),
        SongEntry("Mestisay", "Isa de la gallinita", "música canaria"),
        SongEntry("Mestisay", "Folías del Hierro", "música canaria"),
        SongEntry("Mestisay", "Canarias es mi tierra", "música canaria"),
        SongEntry("Braulio", "Esa mujer", "balada canaria"),
        SongEntry("Braulio", "Vivir así es morir de amor", "balada canaria"),
        SongEntry("Pedro Guerra", "Contamíname", "cantautor canario"),
        SongEntry("Pedro Guerra", "Ganas", "cantautor canario"),
        SongEntry("Pedro Guerra", "Debajo del puente", "cantautor canario"),
        # Timple / folk
        SongEntry("José Antonio Ramos", "Malagueña canaria", "timple"),
        SongEntry("Arístides Moreno", "Timple de mi tierra", "timple"),
        SongEntry("Taburiente", "Retama", "folk canario"),
        SongEntry("Taburiente", "Sobre la marcha", "folk canario"),
        SongEntry("Los Gofiones", "Sombra del nublo", "música canaria"),
        SongEntry("Los Gofiones", "La farola de La Isleta", "música canaria"),
        SongEntry("Mary Sánchez", "Isla de Gran Canaria", "música canaria"),
        SongEntry("Mary Sánchez", "Folías canarias", "música canaria"),
        SongEntry("Olga Cerpa", "Arrorró mi niño", "música canaria"),
        SongEntry("Olga Cerpa", "La sombra del nublo", "música canaria"),
        SongEntry("Benito Cabrera", "Timple", "timple"),
        SongEntry("Benito Cabrera", "Sirinoque", "timple"),
        SongEntry("Los Huaracheros", "Islas Canarias", "música canaria"),
        SongEntry("Los Huaracheros", "Las mañanitas", "música canaria"),
        SongEntry("Caco Senante", "Valentina", "cantautor canario"),
        SongEntry("Caco Senante", "Señora del Mar", "cantautor canario"),
    ],
    # ------------------------------------------------------------------
    # ES_RIO: Rioplatense
    # ------------------------------------------------------------------
    DialectCode.ES_RIO: [
        # Tango
        SongEntry("Carlos Gardel", "Cambalache", "tango"),
        SongEntry("Carlos Gardel", "Yira yira", "tango"),
        SongEntry("Carlos Gardel", "El día que me quieras", "tango"),
        SongEntry("Carlos Gardel", "Mi Buenos Aires querido", "tango"),
        SongEntry("Carlos Gardel", "Volver", "tango"),
        SongEntry("Carlos Gardel", "Por una cabeza", "tango"),
        SongEntry("Carlos Gardel", "Cuesta abajo", "tango"),
        SongEntry("Carlos Gardel", "Mano a mano", "tango"),
        SongEntry("Carlos Gardel", "Melodía de arrabal", "tango"),
        SongEntry("Carlos Gardel", "Sus ojos se cerraron", "tango"),
        # Rock argentino
        SongEntry("Charly García", "Inconsciente colectivo", "rock argentino"),
        SongEntry("Charly García", "Rasguña las piedras", "rock argentino"),
        SongEntry("Charly García", "Demoliendo hoteles", "rock argentino"),
        SongEntry("Fito Páez", "Mariposa technicolor", "rock argentino"),
        SongEntry("Fito Páez", "11 y 6", "rock argentino"),
        SongEntry("Soda Stereo", "De música ligera", "rock argentino"),
        SongEntry("Soda Stereo", "Persiana americana", "rock argentino"),
        SongEntry("Los Redonditos de Ricota", "Jijiji", "rock argentino"),
        SongEntry("Los Redonditos de Ricota", "Vencedores vencidos", "rock argentino"),
        SongEntry("La Renga", "El final es en donde partí", "rock argentino"),
        SongEntry("Andrés Calamaro", "Flaca", "rock argentino"),
        SongEntry("Andrés Calamaro", "Estadio Azteca", "rock argentino"),
        # Cumbia villera
        SongEntry("Damas Gratis", "Me vas a extrañar", "cumbia villera"),
        SongEntry("Damas Gratis", "Se te ve la tanga", "cumbia villera"),
        SongEntry("Pibes Chorros", "Llegamos los pibes chorros", "cumbia villera"),
        SongEntry("Pibes Chorros", "El punga", "cumbia villera"),
        # Trap argentino
        SongEntry("Duki", "She don't give a FO", "trap argentino"),
        SongEntry("Duki", "Goteo", "trap argentino"),
        SongEntry("Trueno", "Dance Crip", "trap argentino"),
        SongEntry("Wos", "Canguro", "trap argentino"),
        SongEntry("Wos", "Arrancarmelo", "trap argentino"),
    ],
    # ------------------------------------------------------------------
    # ES_MEX: Mexican
    # ------------------------------------------------------------------
    DialectCode.ES_MEX: [
        # Ranchera
        SongEntry("Vicente Fernández", "El rey", "ranchera"),
        SongEntry("Vicente Fernández", "Volver volver", "ranchera"),
        SongEntry("Vicente Fernández", "Mujeres divinas", "ranchera"),
        SongEntry("Vicente Fernández", "Por tu maldito amor", "ranchera"),
        SongEntry("José Alfredo Jiménez", "El rey", "ranchera"),
        SongEntry("José Alfredo Jiménez", "Si nos dejan", "ranchera"),
        SongEntry("José Alfredo Jiménez", "Ella", "ranchera"),
        SongEntry("José Alfredo Jiménez", "Caminos de Guanajuato", "ranchera"),
        SongEntry("Pedro Infante", "Cielito lindo", "ranchera"),
        SongEntry("Pedro Infante", "Amorcito corazón", "ranchera"),
        SongEntry("Pedro Infante", "Cien años", "ranchera"),
        SongEntry("Jorge Negrete", "México lindo y querido", "ranchera"),
        SongEntry("Jorge Negrete", "Ay Jalisco no te rajes", "ranchera"),
        # Corrido
        SongEntry("Los Tigres del Norte", "La jaula de oro", "corrido"),
        SongEntry("Los Tigres del Norte", "Jefe de jefes", "corrido"),
        SongEntry("Los Tigres del Norte", "La puerta negra", "corrido"),
        SongEntry("Chalino Sánchez", "Nieves de enero", "corrido"),
        SongEntry("Chalino Sánchez", "Alma enamorada", "corrido"),
        # Cumbia mexicana
        SongEntry("Los Ángeles Azules", "Cómo te voy a olvidar", "cumbia mexicana"),
        SongEntry("Los Ángeles Azules", "Mis sentimientos", "cumbia mexicana"),
        # Rock mexicano
        SongEntry("Café Tacvba", "La ingrata", "rock mexicano"),
        SongEntry("Café Tacvba", "Eres", "rock mexicano"),
        SongEntry("Molotov", "Gimme tha Power", "rock mexicano"),
        SongEntry("Molotov", "Frijolero", "rock mexicano"),
        SongEntry("Maná", "Rayando el sol", "rock mexicano"),
        SongEntry("Maná", "En el muelle de San Blas", "rock mexicano"),
        SongEntry("Maná", "Labios compartidos", "rock mexicano"),
        # Trap / urbano mexicano
        SongEntry("Natanael Cano", "Amor tumbado", "regional mexicano"),
        SongEntry("Peso Pluma", "Ella baila sola", "regional mexicano"),
        SongEntry("Peso Pluma", "AMG", "regional mexicano"),
        SongEntry("Santa Fe Klan", "Debo entender", "rap mexicano"),
    ],
    # ------------------------------------------------------------------
    # ES_CAR: Caribbean
    # ------------------------------------------------------------------
    DialectCode.ES_CAR: [
        # Salsa
        SongEntry("Celia Cruz", "La vida es un carnaval", "salsa"),
        SongEntry("Celia Cruz", "Quimbara", "salsa"),
        SongEntry("Celia Cruz", "La negra tiene tumbao", "salsa"),
        SongEntry("Rubén Blades", "Pedro Navaja", "salsa"),
        SongEntry("Rubén Blades", "Decisiones", "salsa"),
        SongEntry("Rubén Blades", "Plástico", "salsa"),
        SongEntry("Héctor Lavoe", "El cantante", "salsa"),
        SongEntry("Héctor Lavoe", "Periódico de ayer", "salsa"),
        SongEntry("Héctor Lavoe", "Mi gente", "salsa"),
        SongEntry("Willie Colón", "Idilio", "salsa"),
        SongEntry("Willie Colón", "Calle luna calle sol", "salsa"),
        # Reggaeton
        SongEntry("Daddy Yankee", "Gasolina", "reggaeton"),
        SongEntry("Daddy Yankee", "Lo que pasó pasó", "reggaeton"),
        SongEntry("Bad Bunny", "Yo perreo sola", "reggaeton"),
        SongEntry("Bad Bunny", "Dakiti", "reggaeton"),
        SongEntry("Bad Bunny", "Titi me preguntó", "reggaeton"),
        SongEntry("Don Omar", "Dile", "reggaeton"),
        SongEntry("Don Omar", "Dale Don Dale", "reggaeton"),
        SongEntry("Tego Calderón", "Pa' que retozen", "reggaeton"),
        SongEntry("Ivy Queen", "Quiero bailar", "reggaeton"),
        # Son cubano
        SongEntry("Buena Vista Social Club", "Chan Chan", "son cubano"),
        SongEntry("Buena Vista Social Club", "Candela", "son cubano"),
        SongEntry("Compay Segundo", "Guantanamera", "son cubano"),
        SongEntry("Compay Segundo", "Chan Chan", "son cubano"),
        # Merengue
        SongEntry("Juan Luis Guerra", "La bilirrubina", "merengue"),
        SongEntry("Juan Luis Guerra", "Ojalá que llueva café", "merengue"),
        SongEntry("Juan Luis Guerra", "Burbujas de amor", "merengue"),
        # Bachata
        SongEntry("Romeo Santos", "Propuesta indecente", "bachata"),
        SongEntry("Romeo Santos", "Eres mía", "bachata"),
        SongEntry("Aventura", "Obsesión", "bachata"),
        SongEntry("Aventura", "Un beso", "bachata"),
    ],
    # ------------------------------------------------------------------
    # ES_CHI: Chilean
    # ------------------------------------------------------------------
    DialectCode.ES_CHI: [
        # Nueva canción
        SongEntry("Violeta Parra", "Gracias a la vida", "nueva canción"),
        SongEntry("Violeta Parra", "Volver a los 17", "nueva canción"),
        SongEntry("Violeta Parra", "Run Run se fue pa'l norte", "nueva canción"),
        SongEntry("Violeta Parra", "La jardinera", "nueva canción"),
        SongEntry("Víctor Jara", "Te recuerdo Amanda", "nueva canción"),
        SongEntry("Víctor Jara", "El derecho de vivir en paz", "nueva canción"),
        SongEntry("Víctor Jara", "Manifiesto", "nueva canción"),
        SongEntry("Víctor Jara", "Luchín", "nueva canción"),
        SongEntry("Inti-Illimani", "El pueblo unido jamás será vencido", "nueva canción"),
        SongEntry("Inti-Illimani", "Venceremos", "nueva canción"),
        # Rock chileno
        SongEntry("Los Prisioneros", "Tren al sur", "rock chileno"),
        SongEntry("Los Prisioneros", "El baile de los que sobran", "rock chileno"),
        SongEntry("Los Prisioneros", "La voz de los '80", "rock chileno"),
        SongEntry("Los Prisioneros", "¿Por qué no se van?", "rock chileno"),
        SongEntry("Los Tres", "He barrido el sol", "rock chileno"),
        SongEntry("Los Tres", "Déjate caer", "rock chileno"),
        SongEntry("Lucybell", "Cuando respiro en tu boca", "rock chileno"),
        SongEntry("Lucybell", "Mil caminos", "rock chileno"),
        SongEntry("Los Bunkers", "Llueve sobre la ciudad", "rock chileno"),
        SongEntry("Los Bunkers", "Bailando solo", "rock chileno"),
        SongEntry("La Ley", "El duelo", "rock chileno"),
        SongEntry("La Ley", "Día cero", "rock chileno"),
        # Trap chileno
        SongEntry("Pablo Chill-E", "Dime tú", "trap chileno"),
        SongEntry("Young Cister", "Oh Dios mío", "trap chileno"),
        SongEntry("Paloma Mami", "Not Steady", "trap chileno"),
        SongEntry("Paloma Mami", "Fingías", "trap chileno"),
        SongEntry("Kidd Voodoo", "Amorfoda", "trap chileno"),
        # Cumbia chilena
        SongEntry("La Sonora de Tommy Rey", "El galeón español", "cumbia chilena"),
        SongEntry("Américo", "Que levante la mano", "cumbia chilena"),
        SongEntry("Noche de Brujas", "Ella es", "cumbia chilena"),
        SongEntry("Los Viking 5", "Tu cariñito", "cumbia chilena"),
    ],
    # ------------------------------------------------------------------
    # ES_AND_BO: Andean
    # ------------------------------------------------------------------
    DialectCode.ES_AND_BO: [
        # Cumbia andina / folclore
        SongEntry("Los Kjarkas", "Llorando se fue", "cumbia andina"),
        SongEntry("Los Kjarkas", "Bolivia", "cumbia andina"),
        SongEntry("Los Kjarkas", "Wayayay", "cumbia andina"),
        SongEntry("Los Kjarkas", "El picaflor", "cumbia andina"),
        SongEntry("Savia Andina", "El minero", "folclore andino"),
        SongEntry("Savia Andina", "La cochabambina", "folclore andino"),
        SongEntry("Savia Andina", "Ojos azules", "folclore andino"),
        SongEntry("Savia Andina", "Lamento del inca", "folclore andino"),
        # Huayno / chicha
        SongEntry("Los Shapis", "El aguajal", "chicha peruana"),
        SongEntry("Los Shapis", "No me quieras tanto", "chicha peruana"),
        SongEntry("Los Shapis", "Chofercito", "chicha peruana"),
        SongEntry("Chacalón", "Soy provinciano", "chicha peruana"),
        SongEntry("Chacalón", "Viento", "chicha peruana"),
        SongEntry("Dina Páucar", "El alcoholismo", "huayno"),
        SongEntry("Dina Páucar", "No le digas", "huayno"),
        SongEntry("William Luna", "Niñachay", "huayno"),
        SongEntry("William Luna", "Negra del alma", "huayno"),
        # Música boliviana
        SongEntry("Kalamarca", "Cuando florezca el chuño", "folclore andino"),
        SongEntry("Kalamarca", "Ama sua ama llulla ama qella", "folclore andino"),
        SongEntry("Luzmila Carpio", "Arawi", "folclore andino"),
        SongEntry("Luzmila Carpio", "Phaxsi", "folclore andino"),
        SongEntry("Enriqueta Ulloa", "Valicha", "huayno"),
        SongEntry("Los Destellos", "Elsa", "cumbia andina"),
        SongEntry("Los Destellos", "Cariñito", "cumbia andina"),
        SongEntry("Los Mirlos", "La danza de los mirlos", "cumbia andina"),
        SongEntry("Los Mirlos", "Sonido amazónico", "cumbia andina"),
        SongEntry("Grupo Femenino Bolivia", "Ingrata", "cumbia andina"),
        SongEntry("Proyección", "Illimani", "folclore andino"),
        SongEntry("Proyección", "Caporales", "folclore andino"),
        SongEntry("Rumillajta", "Atipiri", "folclore andino"),
        SongEntry("Ernesto Cavour", "Vuelo de cóndores", "folclore andino"),
    ],
}


# ---------------------------------------------------------------------------
# LyricsFetcher
# ---------------------------------------------------------------------------


class LyricsFetcher:
    """Fetch song lyrics and convert them to dialect-annotated samples.

    Parameters
    ----------
    cache_dir:
        Root directory for cached lyrics files.  Defaults to
        ``data/raw/lyrics/`` relative to the project root.
    request_delay:
        Seconds to wait between consecutive HTTP requests.
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        request_delay: float = _REQUEST_DELAY_SECONDS,
    ) -> None:
        if cache_dir is None:
            project_root = Path(__file__).resolve().parents[4]
            self._cache_dir = project_root / "data" / "raw" / "lyrics"
        else:
            self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._request_delay = request_delay
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": _USER_AGENT})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_all(self) -> dict[DialectCode, list[DialectSample]]:
        """Fetch lyrics for all dialect varieties.

        Returns
        -------
        dict[DialectCode, list[DialectSample]]
            Mapping from dialect code to stanza-level samples.
        """
        results: dict[DialectCode, list[DialectSample]] = {}
        total_songs = sum(len(songs) for songs in _SONG_REGISTRY.values())
        fetched_count = 0

        for dialect in DialectCode:
            logger.info("Fetching lyrics for %s ...", dialect.value)
            samples = self.fetch_dialect(dialect)
            results[dialect] = samples
            songs = _SONG_REGISTRY.get(dialect, [])
            fetched_count += len(songs)
            logger.info(
                "  %s: %d stanza samples from %d songs",
                dialect.value,
                len(samples),
                len(songs),
            )

        total_samples = sum(len(s) for s in results.values())
        logger.info(
            "Lyrics fetch complete: %d total stanza samples from %d songs across %d dialects",
            total_samples,
            total_songs,
            len(results),
        )
        return results

    def fetch_dialect(self, dialect: DialectCode) -> list[DialectSample]:
        """Fetch lyrics for a single dialect variety.

        Parameters
        ----------
        dialect:
            The dialect code to fetch lyrics for.

        Returns
        -------
        list[DialectSample]
            Stanza-level samples extracted from fetched lyrics.
        """
        songs = _SONG_REGISTRY.get(dialect, [])
        if not songs:
            logger.warning("No songs configured for dialect %s", dialect.value)
            return []

        samples: list[DialectSample] = []
        for idx, entry in enumerate(songs, start=1):
            logger.info(
                "  [%d/%d] %s — %s (%s)",
                idx,
                len(songs),
                entry.artist,
                entry.title,
                entry.genre,
            )
            lyrics = self._get_lyrics(entry.artist, entry.title, dialect)
            if not lyrics:
                logger.warning("    Lyrics not found, skipping.")
                continue

            confidence = (
                _CONFIDENCE_GENRE_SPECIFIC
                if entry.genre.lower() in _HIGH_CONFIDENCE_GENRES
                else _CONFIDENCE_GENERAL
            )
            stanza_samples = self._split_into_stanzas(
                lyrics, entry, dialect, confidence
            )
            samples.extend(stanza_samples)
            logger.info(
                "    Got %d stanzas (%d chars total)",
                len(stanza_samples),
                sum(len(s.text) for s in stanza_samples),
            )

        return samples

    # ------------------------------------------------------------------
    # Lyrics retrieval with caching
    # ------------------------------------------------------------------

    def _get_lyrics(
        self, artist: str, title: str, dialect: DialectCode
    ) -> str | None:
        """Retrieve lyrics, checking cache first then fetching from sources.

        Parameters
        ----------
        artist:
            Song artist name.
        title:
            Song title.
        dialect:
            Dialect code (used for cache subdirectory).

        Returns
        -------
        str or None
            The lyrics text, or ``None`` if not found.
        """
        cache_path = self._cache_path(dialect, artist, title)

        # Return cached version if available
        if cache_path.exists():
            cached_text = cache_path.read_text(encoding="utf-8").strip()
            if cached_text:
                logger.debug("    Cache hit for '%s - %s'", artist, title)
                return cached_text

        # Try fetching from sources
        lyrics = self._search_lyrics(artist, title)

        if lyrics:
            # Cache the result
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(lyrics, encoding="utf-8")
            logger.debug(
                "    Cached '%s - %s' (%d chars) -> %s",
                artist,
                title,
                len(lyrics),
                cache_path,
            )

        return lyrics

    def _search_lyrics(self, artist: str, title: str) -> str | None:
        """Search for lyrics across multiple sources.

        Tries sources in order of preference:
        1. lyrics.ovh free API
        2. Fallback web scraping

        Parameters
        ----------
        artist:
            Song artist name.
        title:
            Song title.

        Returns
        -------
        str or None
            Cleaned lyrics text, or ``None`` if not found anywhere.
        """
        # Source 1: lyrics.ovh API
        lyrics = self._fetch_lyrics_ovh(artist, title)
        if lyrics:
            return lyrics

        # Source 2: Fallback web scraping
        lyrics = self._fetch_lyrics_scrape(artist, title)
        if lyrics:
            return lyrics

        return None

    def _fetch_lyrics_ovh(self, artist: str, title: str) -> str | None:
        """Fetch lyrics from the lyrics.ovh free API.

        Parameters
        ----------
        artist:
            Song artist name.
        title:
            Song title.

        Returns
        -------
        str or None
            Lyrics text if found, otherwise ``None``.
        """
        url = _LYRICS_OVH_URL.format(artist=artist, title=title)
        try:
            response = self._session.get(url, timeout=15)
            time.sleep(self._request_delay)

            if response.status_code == 200:
                data = response.json()
                lyrics = data.get("lyrics", "")
                if lyrics and len(lyrics.strip()) > _MIN_STANZA_LENGTH:
                    logger.debug("    Found on lyrics.ovh")
                    return _clean_lyrics(lyrics)
            elif response.status_code == 404:
                logger.debug("    Not found on lyrics.ovh (404)")
            else:
                logger.debug(
                    "    lyrics.ovh returned status %d", response.status_code
                )

        except requests.RequestException as exc:
            logger.debug("    lyrics.ovh request failed: %s", exc)
        except (ValueError, KeyError) as exc:
            logger.debug("    lyrics.ovh parse error: %s", exc)

        return None

    def _fetch_lyrics_scrape(self, artist: str, title: str) -> str | None:
        """Fallback: scrape lyrics from a lyrics aggregator site.

        Uses letras.com (a major Spanish-language lyrics site) as the
        scraping target.

        Parameters
        ----------
        artist:
            Song artist name.
        title:
            Song title.

        Returns
        -------
        str or None
            Lyrics text if found, otherwise ``None``.
        """
        if not _HAS_BS4:
            logger.debug("    bs4 not available, skipping web scraping fallback")
            return None

        # Build a search-friendly URL slug
        artist_slug = _slugify(artist)
        title_slug = _slugify(title)
        url = f"https://www.letras.com/{artist_slug}/{title_slug}/"

        try:
            response = self._session.get(url, timeout=15)
            time.sleep(self._request_delay)

            if response.status_code != 200:
                logger.debug(
                    "    letras.com returned status %d for %s",
                    response.status_code,
                    url,
                )
                return None

            soup = BeautifulSoup(response.text, "html.parser")

            # letras.com stores lyrics in a div with class "lyric-original"
            # or within article elements
            lyrics_div = soup.find("div", class_="lyric-original")
            if not lyrics_div:
                lyrics_div = soup.find("div", class_="cnt-letra")
            if not lyrics_div:
                # Try broader selectors
                lyrics_div = soup.find("div", {"class": re.compile(r"lyric|letra")})

            if lyrics_div:
                # Extract text, preserving paragraph breaks
                paragraphs = lyrics_div.find_all("p")
                if paragraphs:
                    text = "\n\n".join(p.get_text(separator="\n") for p in paragraphs)
                else:
                    text = lyrics_div.get_text(separator="\n")

                if text and len(text.strip()) > _MIN_STANZA_LENGTH:
                    logger.debug("    Found on letras.com")
                    return _clean_lyrics(text)

        except requests.RequestException as exc:
            logger.debug("    letras.com request failed: %s", exc)
        except Exception as exc:
            logger.debug("    letras.com parse error: %s", exc)

        return None

    # ------------------------------------------------------------------
    # Stanza splitting
    # ------------------------------------------------------------------

    def _split_into_stanzas(
        self,
        lyrics: str,
        entry: SongEntry,
        dialect: DialectCode,
        confidence: float,
    ) -> list[DialectSample]:
        """Split lyrics into stanza-level DialectSample instances.

        Parameters
        ----------
        lyrics:
            Full lyrics text.
        entry:
            The song registry entry.
        dialect:
            Dialect code to assign.
        confidence:
            Confidence score for this genre.

        Returns
        -------
        list[DialectSample]
            One sample per qualifying stanza (>= 20 chars).
        """
        # Split on double newlines (stanza separators)
        stanzas = re.split(r"\n{2,}", lyrics)

        samples: list[DialectSample] = []
        for stanza in stanzas:
            cleaned = stanza.strip()

            if len(cleaned) < _MIN_STANZA_LENGTH:
                continue

            # Skip obvious non-lyric content
            if _is_non_lyric(cleaned):
                continue

            sample = DialectSample(
                text=cleaned,
                dialect_code=dialect,
                source_id=f"lyrics:{entry.artist}",
                confidence=confidence,
                metadata={
                    "artist": entry.artist,
                    "song": entry.title,
                    "genre": entry.genre,
                },
            )
            samples.append(sample)

        return samples

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_path(
        self, dialect: DialectCode, artist: str, title: str
    ) -> Path:
        """Return the local cache file path for a given song."""
        safe_artist = _sanitize_filename(artist)
        safe_title = _sanitize_filename(title)
        return self._cache_dir / dialect.value / f"{safe_artist}_{safe_title}.txt"

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save_samples(
        self,
        results: dict[DialectCode, list[DialectSample]],
        output_dir: str | Path | None = None,
    ) -> Path:
        """Save fetched samples to a JSON file for downstream processing.

        Parameters
        ----------
        results:
            The output of :meth:`fetch_all` or a subset thereof.
        output_dir:
            Directory to write the output file.  Defaults to
            ``data/processed/``.

        Returns
        -------
        Path
            Path to the written JSON file.
        """
        if output_dir is None:
            project_root = Path(__file__).resolve().parents[4]
            output_dir = project_root / "data" / "processed"
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        out_file = output_path / "lyrics_corpus.json"
        serializable: dict[str, list[dict[str, Any]]] = {}
        for code, samples in results.items():
            serializable[code.value] = [asdict(s) for s in samples]

        out_file.write_text(
            json.dumps(serializable, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Saved lyrics corpus to %s", out_file)
        return out_file


# ---------------------------------------------------------------------------
# Module-level utilities
# ---------------------------------------------------------------------------


def _clean_lyrics(text: str) -> str:
    """Normalize and clean raw lyrics text.

    Removes instrumental markers, excessive whitespace, and common
    non-lyric annotations while preserving stanza structure.
    """
    # Remove common annotations / headers added by lyrics sites
    text = re.sub(
        r"(?i)\[?(intro|outro|verse|chorus|bridge|instrumental|estribillo|"
        r"coro|solo|repeat|x\d+)\]?",
        "",
        text,
    )
    # Remove content in square brackets (e.g., [Verse 1])
    text = re.sub(r"\[.*?\]", "", text)
    # Collapse runs of 3+ newlines into exactly 2 (stanza boundary)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip trailing/leading whitespace from each line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text


def _sanitize_filename(name: str) -> str:
    """Convert an artist/title string into a safe filename component."""
    safe = name.lower()
    # Replace problematic characters
    safe = re.sub(r"[/\\<>:\"|?*']", "", safe)
    # Replace spaces and special chars with underscores
    safe = re.sub(r"[\s.]+", "_", safe)
    # Remove accents for filename safety (keep original in metadata)
    import unicodedata

    nfkd = unicodedata.normalize("NFKD", safe)
    safe = "".join(c for c in nfkd if not unicodedata.combining(c))
    # Collapse multiple underscores
    safe = re.sub(r"_+", "_", safe).strip("_")
    # Limit length
    if len(safe) > 100:
        safe = safe[:100]
    return safe


def _slugify(text: str) -> str:
    """Create a URL-friendly slug from text (for lyrics site URLs)."""
    import unicodedata

    slug = text.lower().strip()
    # Normalize unicode
    nfkd = unicodedata.normalize("NFKD", slug)
    slug = "".join(c for c in nfkd if not unicodedata.combining(c))
    # Replace non-alphanumeric with hyphens
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = slug.strip("-")
    return slug


def _is_non_lyric(text: str) -> bool:
    """Heuristic check for non-lyric content in a stanza."""
    lower = text.lower().strip()
    # Skip translation credits, writer credits, etc.
    skip_patterns = [
        "traducción",
        "translation",
        "written by",
        "compositor",
        "copyright",
        "all rights reserved",
        "lyrics licensed",
        "paroles de",
    ]
    return any(lower.startswith(pat) for pat in skip_patterns)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the lyrics fetcher as a standalone script."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Fetch song lyrics for the EigenDialectos dialect corpus.",
    )
    parser.add_argument(
        "--dialect",
        type=str,
        default=None,
        help=(
            "Fetch only one dialect (e.g. ES_PEN, ES_RIO).  "
            "If omitted, fetches all dialects."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Root directory for cached lyrics files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write the processed corpus JSON.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=_REQUEST_DELAY_SECONDS,
        help=f"Seconds between HTTP requests (default: {_REQUEST_DELAY_SECONDS}).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG-level) logging.",
    )
    args = parser.parse_args()

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    fetcher = LyricsFetcher(
        cache_dir=args.cache_dir,
        request_delay=args.delay,
    )

    if args.dialect:
        try:
            dialect = DialectCode(args.dialect.upper())
        except ValueError:
            valid = ", ".join(c.value for c in DialectCode)
            logger.error("Unknown dialect '%s'. Valid: %s", args.dialect, valid)
            sys.exit(1)

        logger.info("Fetching lyrics for %s ...", dialect.value)
        samples = fetcher.fetch_dialect(dialect)
        results = {dialect: samples}
    else:
        logger.info("Fetching lyrics for all dialects ...")
        results = fetcher.fetch_all()

    # Print summary
    total = 0
    for code in sorted(results, key=lambda c: c.value):
        n = len(results[code])
        total += n
        print(f"  {code.value}: {n} stanza samples")
    print(f"  TOTAL: {total} stanza samples")

    # Save
    out_path = fetcher.save_samples(results, output_dir=args.output_dir)
    print(f"\nLyrics corpus saved to: {out_path}")


if __name__ == "__main__":
    main()
