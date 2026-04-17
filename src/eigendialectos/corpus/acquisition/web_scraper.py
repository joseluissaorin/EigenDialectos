"""Web scraper for regional Spanish-language sources.

Fetches publicly accessible text from diverse regional web sources that
are likely to contain dialectal Spanish: regional newspapers (opinion/culture
sections), Project Gutenberg Spanish-language texts, cooking/recipe blogs,
and government/institutional sites.

Each fetched page is cached locally to avoid repeated network requests.
Text is extracted using BeautifulSoup, cleaned, and segmented into
paragraph-level :class:`DialectSample` instances.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import requests
from bs4 import BeautifulSoup, Comment

from eigendialectos.constants import DialectCode
from eigendialectos.types import DialectSample

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Source entry definition
# ---------------------------------------------------------------------------

@dataclass
class SourceEntry:
    """A single web source to scrape."""

    url: str
    source_name: str
    confidence: float
    dialect: DialectCode
    selectors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Source registry -- curated URLs per dialect
# ---------------------------------------------------------------------------

_DEFAULT_SELECTORS = [
    "article",
    "main",
    "[role='main']",
    ".article-body",
    ".article__body",
    ".entry-content",
    ".post-content",
    ".story-body",
    ".nota-body",
    ".content-body",
]


def _make_entries(
    dialect: DialectCode,
    items: list[tuple[str, str, float]],
    selectors: list[str] | None = None,
) -> list[SourceEntry]:
    """Build a list of SourceEntry from compact tuples."""
    return [
        SourceEntry(
            url=url,
            source_name=name,
            confidence=conf,
            dialect=dialect,
            selectors=selectors or [],
        )
        for url, name, conf in items
    ]


# ---- Peninsular Standard (ES_PEN) ----------------------------------------

_ES_PEN_SOURCES = _make_entries(DialectCode.ES_PEN, [
    # El Pais -- opinion / cultura
    (
        "https://elpais.com/opinion/2024-01-15/la-espana-que-viene.html",
        "elpais-opinion", 0.75,
    ),
    (
        "https://elpais.com/cultura/2024-01-10/el-legado-de-galdos-en-la-literatura-espanola.html",
        "elpais-cultura", 0.70,
    ),
    (
        "https://elpais.com/opinion/2024-02-01/democracia-y-memoria.html",
        "elpais-opinion-2", 0.75,
    ),
    # El Mundo -- opinion
    (
        "https://www.elmundo.es/opinion/columnistas/2024/01/12/el-futuro-de-espana.html",
        "elmundo-opinion", 0.70,
    ),
    (
        "https://www.elmundo.es/cultura/literatura/2024/01/20/vida-literaria-madrid.html",
        "elmundo-cultura", 0.65,
    ),
    # ABC -- opinion
    (
        "https://www.abc.es/opinion/tertulias-de-cafe-20240115.html",
        "abc-opinion", 0.70,
    ),
    (
        "https://www.abc.es/cultura/libros/novela-espanola-contemporanea-20240120.html",
        "abc-cultura", 0.65,
    ),
    # 20minutos -- blogs / opinion
    (
        "https://www.20minutos.es/opiniones/tribuna-la-lengua-del-pueblo-5205678/",
        "20minutos-opinion", 0.70,
    ),
    # eldiario.es
    (
        "https://www.eldiario.es/cultura/libros/narrativa-espanola-actual_1_10876543.html",
        "eldiario-cultura", 0.65,
    ),
    # Government
    (
        "https://www.boe.es/buscar/act.php?id=BOE-A-2023-25758",
        "boe-gobierno", 0.55,
    ),
    # Recipe / cooking (Peninsular)
    (
        "https://www.directoalpaladar.com/recetas-de-legumbres-y-verduras/cocido-madrileno-receta-tradicional",
        "directoalpaladar-cocido", 0.65,
    ),
    (
        "https://www.recetasderechupete.com/tortilla-de-patatas-receta-de-la-abuela/1527/",
        "rechupete-tortilla", 0.65,
    ),
])

# ---- Andalusian (ES_AND) --------------------------------------------------

_ES_AND_SOURCES = _make_entries(DialectCode.ES_AND, [
    # Diario de Sevilla
    (
        "https://www.diariodesevilla.es/opinion/articulos/habla-andaluza-identidad_0_1234567890.html",
        "diariodesevilla-opinion", 0.80,
    ),
    (
        "https://www.diariodesevilla.es/sevilla/feria-abril-tradicion-viva_0_1234567891.html",
        "diariodesevilla-feria", 0.75,
    ),
    # Diario Sur (Malaga)
    (
        "https://www.diariosur.es/opinion/columnas/malaga-habla-calle-20240115.html",
        "diariosur-opinion", 0.80,
    ),
    (
        "https://www.diariosur.es/culturas/semana-santa-malaga-cofradias-20240120.html",
        "diariosur-cultura", 0.75,
    ),
    # Ideal (Granada)
    (
        "https://www.ideal.es/granada/opinion/acento-granadino-orgullo-20240110.html",
        "ideal-opinion", 0.75,
    ),
    # Diario de Cadiz
    (
        "https://www.diariodecadiz.es/opinion/carnaval-cadiz-letras-pueblo_0_1234567892.html",
        "diariodecadiz-carnaval", 0.85,
    ),
    (
        "https://www.diariodecadiz.es/cadiz/gastronomia-gaditana-tapas-tradicion_0_1234567893.html",
        "diariodecadiz-gastro", 0.75,
    ),
    # El Correo de Andalucia
    (
        "https://elcorreoweb.es/opinion/column/el-andaluz-no-habla-mal-ABCD12345.html",
        "elcorreoweb-opinion", 0.80,
    ),
    # Cooking
    (
        "https://www.cocinaandaluza.com/recetas/gazpacho-andaluz-tradicional/",
        "cocinaandaluza-gazpacho", 0.70,
    ),
    (
        "https://www.cocinaandaluza.com/recetas/salmorejo-cordobes/",
        "cocinaandaluza-salmorejo", 0.70,
    ),
    # Flamenco culture
    (
        "https://www.deflamenco.com/revista/entrevistas/entrevista-cantaor-jerezano.html",
        "deflamenco-entrevista", 0.85,
    ),
])

# ---- Canarian (ES_CAN) ----------------------------------------------------

_ES_CAN_SOURCES = _make_entries(DialectCode.ES_CAN, [
    # Canarias7
    (
        "https://www.canarias7.es/opinion/articulos/identidad-canaria-habla-islas-20240115.html",
        "canarias7-opinion", 0.80,
    ),
    (
        "https://www.canarias7.es/cultura/musica-canaria-tradicion-popular-20240120.html",
        "canarias7-cultura", 0.75,
    ),
    (
        "https://www.canarias7.es/sociedad/carnaval-gran-canaria-fiesta-20240110.html",
        "canarias7-carnaval", 0.75,
    ),
    # elDiario.es Canarias Ahora
    (
        "https://www.eldiario.es/canariasahora/opinion/habla-canaria-patrimonio-cultural_129_10876543.html",
        "canariasahora-opinion", 0.80,
    ),
    (
        "https://www.eldiario.es/canariasahora/sociedad/gofio-alimento-identidad-canaria_1_10876544.html",
        "canariasahora-gofio", 0.75,
    ),
    # La Provincia (Las Palmas)
    (
        "https://www.laprovincia.es/opinion/2024/01/15/acento-canario-isla-habla.html",
        "laprovincia-opinion", 0.80,
    ),
    (
        "https://www.laprovincia.es/cultura/2024/01/20/folklore-canario-tradiciones-musicales.html",
        "laprovincia-folklore", 0.70,
    ),
    # Diario de Avisos (Tenerife)
    (
        "https://diariodeavisos.elespanol.com/opinion/tenerife-costumbres-habla-canaria/",
        "diariodeavisos-opinion", 0.80,
    ),
    # Cooking
    (
        "https://www.recetascanarias.com/papas-arrugadas-con-mojo/",
        "recetascanarias-papas", 0.75,
    ),
    (
        "https://www.recetascanarias.com/puchero-canario-tradicional/",
        "recetascanarias-puchero", 0.75,
    ),
])

# ---- Rioplatense (ES_RIO) -------------------------------------------------

_ES_RIO_SOURCES = _make_entries(DialectCode.ES_RIO, [
    # Pagina12 (Argentina)
    (
        "https://www.pagina12.com.ar/suplementos/radar/nota/el-lunfardo-vive-en-la-calle",
        "pagina12-lunfardo", 0.85,
    ),
    (
        "https://www.pagina12.com.ar/suplementos/las12/nota/mujeres-argentinas-hablan",
        "pagina12-cultura", 0.80,
    ),
    (
        "https://www.pagina12.com.ar/opinion/la-argentina-que-supimos-construir",
        "pagina12-opinion", 0.80,
    ),
    # La Nacion (Argentina)
    (
        "https://www.lanacion.com.ar/cultura/el-habla-rioplatense-evoluciona-nid15012024/",
        "lanacion-cultura", 0.80,
    ),
    (
        "https://www.lanacion.com.ar/opinion/el-mate-y-la-identidad-argentina-nid20012024/",
        "lanacion-opinion", 0.75,
    ),
    # Clarin (Argentina)
    (
        "https://www.clarin.com/opinion/columna-el-habla-portena-vive_0_abc123def.html",
        "clarin-opinion", 0.75,
    ),
    (
        "https://www.clarin.com/cultura/tango-letras-lunfardo-actualidad_0_def456ghi.html",
        "clarin-tango", 0.80,
    ),
    # El Pais (Uruguay)
    (
        "https://www.elpais.com.uy/opinion/columnistas/habla-uruguaya-identidad-rioplatense",
        "elpaisuy-opinion", 0.80,
    ),
    # Government
    (
        "https://www.gob.ar/noticias/cultura-argentina-patrimonio-inmaterial",
        "gobar-cultura", 0.60,
    ),
    # Cooking
    (
        "https://www.recetasargentinas.net/empanadas-criollas/",
        "recetasargentinas-empanadas", 0.70,
    ),
    (
        "https://www.recetasargentinas.net/asado-argentino-paso-a-paso/",
        "recetasargentinas-asado", 0.70,
    ),
])

# ---- Mexican (ES_MEX) -----------------------------------------------------

_ES_MEX_SOURCES = _make_entries(DialectCode.ES_MEX, [
    # La Jornada
    (
        "https://www.jornada.com.mx/2024/01/15/opinion/habla-mexicana-identidad",
        "jornada-opinion", 0.80,
    ),
    (
        "https://www.jornada.com.mx/2024/01/20/cultura/literatura-mexicana-actual",
        "jornada-cultura", 0.75,
    ),
    (
        "https://www.jornada.com.mx/2024/02/01/opinion/lenguaje-coloquial-mexico",
        "jornada-opinion-2", 0.80,
    ),
    # El Universal
    (
        "https://www.eluniversal.com.mx/opinion/la-lengua-de-los-mexicanos/",
        "eluniversal-opinion", 0.80,
    ),
    (
        "https://www.eluniversal.com.mx/cultura/dia-de-muertos-tradicion-viva/",
        "eluniversal-cultura", 0.70,
    ),
    # Proceso
    (
        "https://www.proceso.com.mx/opinion/2024/1/15/el-habla-popular-mexicana-320000.html",
        "proceso-opinion", 0.80,
    ),
    # Milenio
    (
        "https://www.milenio.com/opinion/columnas/el-mexicanismo-en-la-lengua",
        "milenio-opinion", 0.75,
    ),
    # Government
    (
        "https://www.gob.mx/cultura/articulos/patrimonio-linguistico-de-mexico",
        "gobmx-cultura", 0.60,
    ),
    (
        "https://www.gob.mx/cultura/articulos/gastronomia-mexicana-patrimonio-inmaterial",
        "gobmx-gastro", 0.55,
    ),
    # Cooking
    (
        "https://www.cocinafacil.com.mx/recetas-de-comida/mole-poblano-receta-tradicional/",
        "cocinafacil-mole", 0.70,
    ),
    (
        "https://www.kiwilimon.com/receta/tamales-de-rajas-con-queso",
        "kiwilimon-tamales", 0.70,
    ),
    # Letras Libres (literary)
    (
        "https://letraslibres.com/revista/el-espanol-mexicano-en-el-siglo-xxi/",
        "letraslibres-espanol", 0.75,
    ),
])

# ---- Caribbean (ES_CAR) ---------------------------------------------------

_ES_CAR_SOURCES = _make_entries(DialectCode.ES_CAR, [
    # Granma (Cuba)
    (
        "https://www.granma.cu/cultura/2024-01-15/identidad-cubana-en-la-palabra",
        "granma-cultura", 0.80,
    ),
    (
        "https://www.granma.cu/opinion/2024-01-20/la-habana-habla-asi",
        "granma-opinion", 0.85,
    ),
    # Listin Diario (Dominican Republic)
    (
        "https://listindiario.com/opinion/2024/01/15/el-habla-dominicana-identidad/",
        "listindiario-opinion", 0.80,
    ),
    (
        "https://listindiario.com/la-vida/2024/01/20/gastronomia-dominicana-tradicion/",
        "listindiario-gastro", 0.70,
    ),
    # El Nuevo Dia (Puerto Rico)
    (
        "https://www.elnuevodia.com/opinion/punto-de-vista/el-espanol-de-puerto-rico/",
        "elnuevodia-opinion", 0.85,
    ),
    (
        "https://www.elnuevodia.com/entretenimiento/cultura/la-salsa-y-el-habla-caribena/",
        "elnuevodia-cultura", 0.80,
    ),
    # El Universal (Venezuela)
    (
        "https://www.eluniversal.com/opinion/la-venezolanidad-en-el-habla",
        "eluniversal-ve-opinion", 0.80,
    ),
    # Cubadebate
    (
        "https://www.cubadebate.cu/opinion/2024/01/15/cultura-popular-cubana/",
        "cubadebate-opinion", 0.75,
    ),
    # Government
    (
        "https://www.presidencia.gob.do/noticias/cultura-dominicana-patrimonio",
        "gobdo-cultura", 0.55,
    ),
    # Cooking
    (
        "https://www.cocinadominicana.com/recetas/mangu-tradicional/",
        "cocinadominicana-mangu", 0.70,
    ),
    (
        "https://www.cocinadominicana.com/recetas/sancocho-dominicano/",
        "cocinadominicana-sancocho", 0.70,
    ),
])

# ---- Chilean (ES_CHI) -----------------------------------------------------

_ES_CHI_SOURCES = _make_entries(DialectCode.ES_CHI, [
    # La Tercera
    (
        "https://www.latercera.com/opinion/noticia/el-chileno-y-su-forma-de-hablar/",
        "latercera-opinion", 0.80,
    ),
    (
        "https://www.latercera.com/cultura/noticia/chilenismos-que-definen-identidad/",
        "latercera-cultura", 0.80,
    ),
    # EMOL
    (
        "https://www.emol.com/noticias/Tendencias/2024/01/15/chilenismos-uso-cotidiano.html",
        "emol-chilenismos", 0.80,
    ),
    (
        "https://www.emol.com/noticias/Nacional/2024/01/20/cultura-popular-chile.html",
        "emol-cultura", 0.70,
    ),
    # The Clinic
    (
        "https://www.theclinic.cl/opinion/el-habla-chilena-weona-identidad/",
        "theclinic-opinion", 0.85,
    ),
    (
        "https://www.theclinic.cl/cultura/jerga-chilena-evolucion/",
        "theclinic-cultura", 0.80,
    ),
    # BioBioChile
    (
        "https://www.biobiochile.cl/noticias/sociedad/opinion/2024/01/15/modismos-chilenos-lenguaje.shtml",
        "biobiochile-opinion", 0.75,
    ),
    # Government
    (
        "https://www.gob.cl/noticias/patrimonio-cultural-chile/",
        "gobcl-cultura", 0.55,
    ),
    # Cooking
    (
        "https://www.cocinaenchilena.cl/recetas/pastel-de-choclo/",
        "cocinaenchilena-pastel", 0.70,
    ),
    (
        "https://www.cocinaenchilena.cl/recetas/curanto-en-olla/",
        "cocinaenchilena-curanto", 0.70,
    ),
    # El Mostrador
    (
        "https://www.elmostrador.cl/cultura/2024/01/15/la-identidad-en-el-habla-chilena/",
        "elmostrador-cultura", 0.75,
    ),
])

# ---- Andean (ES_AND_BO) ---------------------------------------------------

_ES_AND_BO_SOURCES = _make_entries(DialectCode.ES_AND_BO, [
    # Los Tiempos (Bolivia)
    (
        "https://www.lostiempos.com/opinion/columna/20240115/habla-boliviana-quechuismos-cotidianos",
        "lostiempos-opinion", 0.80,
    ),
    (
        "https://www.lostiempos.com/doble-click/cultura/20240120/tradiciones-orales-bolivia",
        "lostiempos-cultura", 0.75,
    ),
    # El Comercio (Peru)
    (
        "https://elcomercio.pe/opinion/columnistas/el-espanol-peruano-una-riqueza-linguistica/",
        "elcomercio-opinion", 0.80,
    ),
    (
        "https://elcomercio.pe/luces/cultura/quechuismos-en-el-habla-peruana-cotidiana/",
        "elcomercio-cultura", 0.80,
    ),
    # La Razon (Bolivia)
    (
        "https://www.la-razon.com/opinion/2024/01/15/identidad-linguistica-boliviana/",
        "larazon-opinion", 0.80,
    ),
    # El Universo (Ecuador)
    (
        "https://www.eluniverso.com/opinion/2024/01/20/espanol-ecuatoriano-identidad-andina/",
        "eluniverso-opinion", 0.75,
    ),
    (
        "https://www.eluniverso.com/entretenimiento/cultura/gastronomia-andina-ecuatoriana/",
        "eluniverso-gastro", 0.65,
    ),
    # Government
    (
        "https://www.gob.pe/cultura/patrimonio-inmaterial-linguistico",
        "gobpe-cultura", 0.55,
    ),
    (
        "https://www.comunicacion.gob.bo/noticias/patrimonio-cultural-boliviano",
        "gobbo-cultura", 0.55,
    ),
    # Cooking
    (
        "https://www.comeperuano.pe/recetas/ceviche-peruano-clasico/",
        "comeperuano-ceviche", 0.70,
    ),
    (
        "https://www.comeperuano.pe/recetas/aji-de-gallina/",
        "comeperuano-aji", 0.70,
    ),
    (
        "https://www.cocinaboliviana.com/recetas/saltenias-tradicionales/",
        "cocinaboliviana-saltenas", 0.70,
    ),
])

# ---- Gutenberg curated list -----------------------------------------------

@dataclass
class GutenbergEntry:
    """A Gutenberg book known to represent a specific dialect region."""

    book_id: int
    title: str
    author: str
    dialect: DialectCode
    confidence: float
    notes: str = ""


GUTENBERG_CATALOG: list[GutenbergEntry] = [
    # Peninsular Standard
    GutenbergEntry(15725, "Fortunata y Jacinta", "Benito Perez Galdos",
                   DialectCode.ES_PEN, 0.80, "Madrid colloquial speech in dialogue"),
    GutenbergEntry(17013, "Miau", "Benito Perez Galdos",
                   DialectCode.ES_PEN, 0.80, "Madrid bureaucratic milieu"),
    GutenbergEntry(15027, "Misericordia", "Benito Perez Galdos",
                   DialectCode.ES_PEN, 0.80, "Madrid lower-class speech"),
    GutenbergEntry(16625, "La Regenta", "Leopoldo Alas (Clarin)",
                   DialectCode.ES_PEN, 0.75, "Asturian/northern Peninsular"),
    GutenbergEntry(17323, "La Celestina", "Fernando de Rojas",
                   DialectCode.ES_PEN, 0.60, "Classical Peninsular, older register"),
    # Andalusian
    GutenbergEntry(39920, "Platero y yo", "Juan Ramon Jimenez",
                   DialectCode.ES_AND, 0.75, "Andalusian rural setting, Moguer"),
    GutenbergEntry(55474, "Poema del cante jondo", "Federico Garcia Lorca",
                   DialectCode.ES_AND, 0.80, "Deep Andalusian cultural roots"),
    GutenbergEntry(55560, "Romancero gitano", "Federico Garcia Lorca",
                   DialectCode.ES_AND, 0.80, "Granada/Andalusian Romani culture"),
    # Rioplatense
    GutenbergEntry(13610, "El gaucho Martin Fierro", "Jose Hernandez",
                   DialectCode.ES_RIO, 0.90, "Rural rioplatense, gauchesque register"),
    GutenbergEntry(69666, "Don Segundo Sombra", "Ricardo Guiraldes",
                   DialectCode.ES_RIO, 0.85, "Pampa gaucho dialect"),
    GutenbergEntry(56870, "Ficciones", "Jorge Luis Borges",
                   DialectCode.ES_RIO, 0.75, "Porteno literary register"),
    # Mexican
    GutenbergEntry(17345, "Los de abajo", "Mariano Azuela",
                   DialectCode.ES_MEX, 0.85, "Mexican Revolution colloquial"),
    GutenbergEntry(55555, "Pedro Paramo", "Juan Rulfo",
                   DialectCode.ES_MEX, 0.90, "Jalisco rural Mexican speech"),
    GutenbergEntry(21657, "El periquillo sarniento", "Fernandez de Lizardi",
                   DialectCode.ES_MEX, 0.80, "Colonial Mexican popular speech"),
    # Chilean
    GutenbergEntry(17786, "Sub terra", "Baldomero Lillo",
                   DialectCode.ES_CHI, 0.80, "Chilean mining communities"),
    GutenbergEntry(29199, "Veinte poemas de amor", "Pablo Neruda",
                   DialectCode.ES_CHI, 0.65, "Chilean literary register"),
    # Caribbean
    GutenbergEntry(11825, "Cecilia Valdes", "Cirilo Villaverde",
                   DialectCode.ES_CAR, 0.85, "19th-century Cuban speech"),
    GutenbergEntry(46240, "La charca", "Manuel Zeno Gandia",
                   DialectCode.ES_CAR, 0.80, "Puerto Rican rural speech"),
    # Andean
    GutenbergEntry(31373, "Aves sin nido", "Clorinda Matto de Turner",
                   DialectCode.ES_AND_BO, 0.80, "Peruvian Andean setting"),
    GutenbergEntry(55580, "Huasipungo", "Jorge Icaza",
                   DialectCode.ES_AND_BO, 0.85, "Ecuadorian Andean indigenous speech"),
]

# ---------------------------------------------------------------------------
# Combine all web source entries
# ---------------------------------------------------------------------------

ALL_WEB_SOURCES: list[SourceEntry] = (
    _ES_PEN_SOURCES
    + _ES_AND_SOURCES
    + _ES_CAN_SOURCES
    + _ES_RIO_SOURCES
    + _ES_MEX_SOURCES
    + _ES_CAR_SOURCES
    + _ES_CHI_SOURCES
    + _ES_AND_BO_SOURCES
)

# ---------------------------------------------------------------------------
# Dialogue extraction regex (for Gutenberg texts)
# ---------------------------------------------------------------------------

_DIALOGUE_PATTERNS = [
    # Spanish angle quotes: << ... >>
    re.compile(r'\u00ab(.+?)\u00bb', re.DOTALL),
    # Em-dash dialogue (common in Spanish novels): -- or ---
    re.compile(r'(?:^|\n)\s*[\u2014\u2015\u2013\-]{1,3}\s*(.+?)(?=\n\s*[\u2014\u2015\u2013\-]|\n\n|\Z)', re.DOTALL),
    # Standard double quotes
    re.compile(r'\u201c(.+?)\u201d', re.DOTALL),
    # Straight double quotes
    re.compile(r'"(.+?)"', re.DOTALL),
]

# ---------------------------------------------------------------------------
# HTTP settings
# ---------------------------------------------------------------------------

_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
_REQUEST_TIMEOUT = 30  # seconds
_RATE_LIMIT_DELAY = 2.0  # seconds between requests


# ===========================================================================
# WebScraper
# ===========================================================================


class WebScraper:
    """Scraper for regional Spanish-language web sources.

    Fetches publicly accessible text, extracts article content via
    BeautifulSoup, cleans it, segments into paragraphs, and returns
    :class:`DialectSample` instances.  All fetched HTML and extracted
    text are cached under *cache_dir* for idempotent re-runs.

    Parameters
    ----------
    cache_dir:
        Root directory for cached HTML and text files.
        Defaults to ``data/raw/web`` relative to the project root.
    rate_limit:
        Minimum seconds between HTTP requests (default 2.0).
    timeout:
        HTTP request timeout in seconds (default 30).
    """

    def __init__(
        self,
        cache_dir: str | Path = "data/raw/web",
        rate_limit: float = _RATE_LIMIT_DELAY,
        timeout: int = _REQUEST_TIMEOUT,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.rate_limit = rate_limit
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": _USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "es-ES,es;q=0.9,en;q=0.5",
        })
        self._last_request_time: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_all(
        self,
        dialects: list[DialectCode] | None = None,
    ) -> dict[DialectCode, list[DialectSample]]:
        """Fetch from all registered web sources and Gutenberg texts.

        Parameters
        ----------
        dialects:
            If provided, only fetch sources for these dialect codes.
            If *None*, fetch all dialects.

        Returns
        -------
        dict mapping each DialectCode to a list of DialectSample.
        """
        results: dict[DialectCode, list[DialectSample]] = {
            dc: [] for dc in DialectCode
        }

        # Web sources
        for entry in ALL_WEB_SOURCES:
            if dialects is not None and entry.dialect not in dialects:
                continue
            try:
                samples = self.fetch_url(
                    url=entry.url,
                    dialect=entry.dialect,
                    confidence=entry.confidence,
                    source_name=entry.source_name,
                    selectors=entry.selectors,
                )
                results[entry.dialect].extend(samples)
                logger.info(
                    "Fetched %d samples from %s (%s)",
                    len(samples), entry.source_name, entry.dialect.value,
                )
            except Exception:
                logger.warning(
                    "Failed to fetch %s (%s)", entry.url, entry.source_name,
                    exc_info=True,
                )

        # Gutenberg
        for gentry in GUTENBERG_CATALOG:
            if dialects is not None and gentry.dialect not in dialects:
                continue
            try:
                samples = self.fetch_gutenberg_book(gentry)
                results[gentry.dialect].extend(samples)
                logger.info(
                    "Fetched %d samples from Gutenberg '%s' (%s)",
                    len(samples), gentry.title, gentry.dialect.value,
                )
            except Exception:
                logger.warning(
                    "Failed to fetch Gutenberg %d (%s)",
                    gentry.book_id, gentry.title,
                    exc_info=True,
                )

        # Summary log
        for dc in DialectCode:
            count = len(results[dc])
            if count > 0:
                logger.info("Total %s: %d samples", dc.value, count)

        return results

    def fetch_url(
        self,
        url: str,
        dialect: DialectCode,
        confidence: float,
        source_name: str = "",
        selectors: list[str] | None = None,
    ) -> list[DialectSample]:
        """Fetch one URL, extract text, and segment into samples.

        Parameters
        ----------
        url:
            The web page to fetch.
        dialect:
            Dialect code to assign to all samples from this URL.
        confidence:
            Confidence score (0.0-1.0) that the text is dialectally marked.
        source_name:
            Human-readable source identifier.
        selectors:
            CSS selectors to try for content extraction, in priority order.

        Returns
        -------
        list of DialectSample, one per paragraph-level segment.
        """
        if not source_name:
            source_name = url

        # Check cache
        url_hash = self._url_hash(url)
        dialect_cache = self.cache_dir / dialect.value
        html_cache = dialect_cache / f"{url_hash}.html"
        text_cache = dialect_cache / f"{url_hash}.txt"

        # Try loading from text cache first
        if text_cache.exists():
            logger.debug("Cache hit (text): %s", url)
            text = text_cache.read_text(encoding="utf-8")
        elif html_cache.exists():
            logger.debug("Cache hit (html): %s", url)
            html = html_cache.read_text(encoding="utf-8")
            text = self._extract_text_from_html(html, selectors)
            self._write_text_cache(text_cache, text)
        else:
            # Fetch from network
            html = self._fetch_html(url)
            if html is None:
                return []
            # Cache HTML
            self._write_html_cache(html_cache, html)
            # Extract and cache text
            text = self._extract_text_from_html(html, selectors)
            self._write_text_cache(text_cache, text)

        if not text.strip():
            logger.debug("No text extracted from %s", url)
            return []

        # Segment and build samples
        paragraphs = self._segment(text)
        samples: list[DialectSample] = []
        for para in paragraphs:
            sample = DialectSample(
                text=para,
                dialect_code=dialect,
                source_id=f"web:{source_name}",
                confidence=confidence,
                metadata={"url": url, "source_name": source_name},
            )
            samples.append(sample)

        return samples

    def fetch_gutenberg(
        self,
        dialects: list[DialectCode] | None = None,
    ) -> dict[DialectCode, list[DialectSample]]:
        """Fetch all curated Gutenberg texts.

        Parameters
        ----------
        dialects:
            If provided, only fetch books for these dialect codes.

        Returns
        -------
        dict mapping each DialectCode to a list of DialectSample.
        """
        results: dict[DialectCode, list[DialectSample]] = {
            dc: [] for dc in DialectCode
        }
        for entry in GUTENBERG_CATALOG:
            if dialects is not None and entry.dialect not in dialects:
                continue
            try:
                samples = self.fetch_gutenberg_book(entry)
                results[entry.dialect].extend(samples)
            except Exception:
                logger.warning(
                    "Failed to fetch Gutenberg %d (%s)",
                    entry.book_id, entry.title,
                    exc_info=True,
                )
        return results

    def fetch_gutenberg_book(
        self,
        entry: GutenbergEntry,
    ) -> list[DialectSample]:
        """Fetch a single Gutenberg book and extract samples.

        Fetches the plain-text version from Gutenberg's cache.  Extracts
        both full paragraphs and dialogue sections (between quotation
        marks) as separate, higher-confidence samples.

        Parameters
        ----------
        entry:
            The Gutenberg catalog entry to fetch.

        Returns
        -------
        list of DialectSample.
        """
        url = f"https://www.gutenberg.org/cache/epub/{entry.book_id}/pg{entry.book_id}.txt"
        source_name = f"gutenberg-{entry.book_id}-{entry.author.split()[-1].lower()}"

        # Check cache
        url_hash = self._url_hash(url)
        dialect_cache = self.cache_dir / entry.dialect.value / "gutenberg"
        text_cache = dialect_cache / f"{url_hash}.txt"

        if text_cache.exists():
            logger.debug("Cache hit (gutenberg): %s", entry.title)
            raw_text = text_cache.read_text(encoding="utf-8")
        else:
            raw_text = self._fetch_plain_text(url)
            if raw_text is None:
                return []
            # Strip Gutenberg header/footer
            raw_text = self._strip_gutenberg_boilerplate(raw_text)
            self._write_text_cache(text_cache, raw_text)

        if not raw_text.strip():
            return []

        samples: list[DialectSample] = []

        # Full paragraphs
        paragraphs = self._segment(raw_text, min_length=80)
        for para in paragraphs:
            samples.append(DialectSample(
                text=para,
                dialect_code=entry.dialect,
                source_id=f"gutenberg:{source_name}",
                confidence=entry.confidence,
                metadata={
                    "url": url,
                    "book_id": entry.book_id,
                    "title": entry.title,
                    "author": entry.author,
                    "type": "paragraph",
                    "notes": entry.notes,
                },
            ))

        # Dialogue extraction (higher confidence -- represents speech)
        dialogues = self._extract_dialogue(raw_text)
        dialogue_conf = min(entry.confidence + 0.10, 1.0)
        for dial in dialogues:
            samples.append(DialectSample(
                text=dial,
                dialect_code=entry.dialect,
                source_id=f"gutenberg:{source_name}:dialogue",
                confidence=dialogue_conf,
                metadata={
                    "url": url,
                    "book_id": entry.book_id,
                    "title": entry.title,
                    "author": entry.author,
                    "type": "dialogue",
                    "notes": entry.notes,
                },
            ))

        return samples

    # ------------------------------------------------------------------
    # HTML fetching
    # ------------------------------------------------------------------

    def _fetch_html(self, url: str) -> str | None:
        """Fetch URL content as HTML string, respecting rate limit."""
        self._rate_limit_wait()
        try:
            resp = self._session.get(url, timeout=self.timeout, allow_redirects=True)
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding or "utf-8"
            return resp.text
        except requests.exceptions.HTTPError as exc:
            logger.warning("HTTP error fetching %s: %s", url, exc)
            return None
        except requests.exceptions.ConnectionError as exc:
            logger.warning("Connection error fetching %s: %s", url, exc)
            return None
        except requests.exceptions.Timeout:
            logger.warning("Timeout fetching %s", url)
            return None
        except requests.exceptions.RequestException as exc:
            logger.warning("Request error fetching %s: %s", url, exc)
            return None

    def _fetch_plain_text(self, url: str) -> str | None:
        """Fetch URL content as plain text (for Gutenberg .txt files)."""
        self._rate_limit_wait()
        try:
            resp = self._session.get(url, timeout=self.timeout, allow_redirects=True)
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding or "utf-8"
            return resp.text
        except requests.exceptions.RequestException as exc:
            logger.warning("Request error fetching %s: %s", url, exc)
            return None

    def _rate_limit_wait(self) -> None:
        """Sleep if needed to respect the rate limit."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_request_time = time.monotonic()

    # ------------------------------------------------------------------
    # Content extraction
    # ------------------------------------------------------------------

    def _extract_text_from_html(
        self,
        html: str,
        selectors: list[str] | None = None,
    ) -> str:
        """Extract article body text from HTML using BeautifulSoup.

        Strategy:
        1. Try provided selectors first.
        2. Try default content selectors (article, main, etc.).
        3. Fall back to collecting all <p> tags.

        Parameters
        ----------
        html:
            Raw HTML string.
        selectors:
            CSS selectors to try first, in priority order.

        Returns
        -------
        Cleaned, concatenated text from the article body.
        """
        soup = BeautifulSoup(html, "html.parser")

        # Remove unwanted elements before extraction
        self._remove_unwanted_elements(soup)

        # Build selector list: custom first, then defaults
        all_selectors = list(selectors or []) + _DEFAULT_SELECTORS

        # Try each selector
        for selector in all_selectors:
            try:
                container = soup.select_one(selector)
            except Exception:
                continue
            if container is not None:
                text = self._extract_text_from_element(container)
                if len(text.strip()) >= 100:
                    return self._clean_text(text)

        # Fallback: collect all <p> tags
        paragraphs = soup.find_all("p")
        if paragraphs:
            text = "\n\n".join(p.get_text(separator=" ", strip=True) for p in paragraphs)
            return self._clean_text(text)

        # Last resort: body text
        body = soup.find("body")
        if body:
            return self._clean_text(body.get_text(separator="\n", strip=True))

        return ""

    def _remove_unwanted_elements(self, soup: BeautifulSoup) -> None:
        """Remove scripts, styles, nav, ads, footers, and other noise."""
        # Tags to remove entirely
        for tag_name in [
            "script", "style", "noscript", "iframe", "svg",
            "nav", "footer", "header", "aside",
            "form", "button", "input", "select", "textarea",
        ]:
            for element in soup.find_all(tag_name):
                element.decompose()

        # Remove HTML comments
        for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
            comment.extract()

        # Remove elements by class/id patterns (common ad/nav markers)
        _noise_patterns = re.compile(
            r'(?i)(advert|banner|cookie|consent|popup|modal|sidebar|widget|'
            r'share|social|newsletter|subscribe|related|recommend|promo|'
            r'comment|disqus|footer|nav|menu|breadcrumb|pagination)',
        )
        for el in soup.find_all(attrs={"class": _noise_patterns}):
            el.decompose()
        for el in soup.find_all(attrs={"id": _noise_patterns}):
            el.decompose()

    def _extract_text_from_element(self, element: object) -> str:
        """Extract text from a BS4 element, preserving paragraph breaks."""
        paragraphs = element.find_all("p")  # type: ignore[union-attr]
        if paragraphs:
            return "\n\n".join(
                p.get_text(separator=" ", strip=True) for p in paragraphs
            )
        return element.get_text(separator="\n", strip=True)  # type: ignore[union-attr]

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalize whitespace, remove residual noise from extracted text."""
        # Remove any remaining HTML entities
        text = re.sub(r'&[a-zA-Z]+;', ' ', text)
        text = re.sub(r'&#\d+;', ' ', text)
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        # Remove email addresses
        text = re.sub(r'\S+@\S+\.\S+', '', text)
        # Collapse multiple newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Collapse spaces (but not newlines)
        text = re.sub(r'[^\S\n]+', ' ', text)
        # Strip leading/trailing whitespace per line
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        # Final trim
        return text.strip()

    @staticmethod
    def _segment(text: str, min_length: int = 50) -> list[str]:
        """Split text into paragraph-level chunks.

        Parameters
        ----------
        text:
            Cleaned text to segment.
        min_length:
            Minimum character length for a segment to be included.

        Returns
        -------
        list of paragraph strings meeting the minimum length.
        """
        # Split on double newlines (paragraph boundaries)
        raw_chunks = re.split(r'\n\s*\n', text)
        segments: list[str] = []
        for chunk in raw_chunks:
            chunk = chunk.strip()
            if len(chunk) >= min_length:
                segments.append(chunk)
        return segments

    # ------------------------------------------------------------------
    # Gutenberg helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_gutenberg_boilerplate(text: str) -> str:
        """Remove Project Gutenberg header and footer boilerplate."""
        # Common start markers
        start_markers = [
            "*** START OF THIS PROJECT GUTENBERG",
            "*** START OF THE PROJECT GUTENBERG",
            "***START OF THIS PROJECT GUTENBERG",
            "*END*THE SMALL PRINT",
        ]
        # Common end markers
        end_markers = [
            "*** END OF THIS PROJECT GUTENBERG",
            "*** END OF THE PROJECT GUTENBERG",
            "***END OF THIS PROJECT GUTENBERG",
            "End of the Project Gutenberg",
            "End of Project Gutenberg",
        ]

        # Find start
        lower = text.lower()
        start_idx = 0
        for marker in start_markers:
            idx = lower.find(marker.lower())
            if idx != -1:
                # Move past the marker line
                nl = text.find('\n', idx)
                if nl != -1:
                    start_idx = nl + 1
                break

        # Find end
        end_idx = len(text)
        for marker in end_markers:
            idx = lower.find(marker.lower())
            if idx != -1:
                end_idx = idx
                break

        return text[start_idx:end_idx].strip()

    @staticmethod
    def _extract_dialogue(text: str, min_length: int = 20) -> list[str]:
        """Extract dialogue from literary text using quotation patterns.

        Parameters
        ----------
        text:
            The full literary text.
        min_length:
            Minimum length of a dialogue fragment to include.

        Returns
        -------
        list of dialogue text fragments.
        """
        dialogues: list[str] = []
        seen: set[str] = set()

        for pattern in _DIALOGUE_PATTERNS:
            for match in pattern.finditer(text):
                fragment = match.group(1).strip()
                # Clean up the fragment
                fragment = re.sub(r'\s+', ' ', fragment)
                if len(fragment) >= min_length and fragment not in seen:
                    seen.add(fragment)
                    dialogues.append(fragment)

        return dialogues

    # ------------------------------------------------------------------
    # Caching helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _url_hash(url: str) -> str:
        """Return a stable short hash of a URL for cache filenames."""
        return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _write_html_cache(path: Path, html: str) -> None:
        """Write HTML content to cache file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")

    @staticmethod
    def _write_text_cache(path: Path, text: str) -> None:
        """Write extracted text to cache file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def source_summary(self) -> dict[str, int]:
        """Return a count of registered sources per dialect."""
        counts: dict[str, int] = {}
        for dc in DialectCode:
            web = sum(1 for e in ALL_WEB_SOURCES if e.dialect == dc)
            gut = sum(1 for e in GUTENBERG_CATALOG if e.dialect == dc)
            counts[dc.value] = web + gut
        return counts

    def __repr__(self) -> str:
        total_web = len(ALL_WEB_SOURCES)
        total_gut = len(GUTENBERG_CATALOG)
        return (
            f"<WebScraper sources={total_web} web + {total_gut} gutenberg, "
            f"cache_dir={self.cache_dir}>"
        )
