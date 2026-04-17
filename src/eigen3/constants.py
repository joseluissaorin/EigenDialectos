"""Project-wide constants for eigen3."""

from __future__ import annotations

from enum import Enum


class DialectCode(str, Enum):
    """ISO-style codes for 8 major Spanish dialect varieties."""

    ES_PEN = "ES_PEN"      # Peninsular Standard (Madrid)
    ES_AND = "ES_AND"      # Andalusian (Sevilla)
    ES_CAN = "ES_CAN"      # Canarian (Las Palmas)
    ES_RIO = "ES_RIO"      # Rioplatense (Buenos Aires)
    ES_MEX = "ES_MEX"      # Mexican (CDMX)
    ES_CAR = "ES_CAR"      # Caribbean (La Habana)
    ES_CHI = "ES_CHI"      # Chilean (Santiago)
    ES_AND_BO = "ES_AND_BO"  # Andean (La Paz)


ALL_VARIETIES: list[str] = [d.value for d in DialectCode]
N_VARIETIES: int = len(ALL_VARIETIES)
REFERENCE_VARIETY: str = DialectCode.ES_PEN.value

DIALECT_NAMES: dict[str, str] = {
    "ES_PEN": "Castellano peninsular estándar",
    "ES_AND": "Andaluz",
    "ES_CAN": "Canario",
    "ES_RIO": "Rioplatense",
    "ES_MEX": "Mexicano",
    "ES_CAR": "Caribeño",
    "ES_CHI": "Chileno",
    "ES_AND_BO": "Andino",
}

DIALECT_FAMILIES: dict[str, list[str]] = {
    "peninsular": ["ES_PEN", "ES_AND", "ES_CAN"],
    "caribbean": ["ES_CAR"],
    "southern_cone": ["ES_RIO", "ES_CHI"],
    "mesoamerican": ["ES_MEX"],
    "andean": ["ES_AND_BO"],
}

DIALECT_COORDINATES: dict[str, tuple[float, float]] = {
    "ES_PEN": (40.4, -3.7),        # Madrid
    "ES_AND": (37.4, -6.0),        # Sevilla
    "ES_CAN": (28.1, -15.4),       # Las Palmas
    "ES_RIO": (-34.6, -58.4),      # Buenos Aires
    "ES_MEX": (19.4, -99.1),       # CDMX
    "ES_CAR": (23.1, -82.4),       # La Habana
    "ES_CHI": (-33.4, -70.6),      # Santiago
    "ES_AND_BO": (-16.5, -68.1),   # La Paz
}

# Affinity between variety pairs — controls negative sampling.
# High affinity = rarely sample negatives from each other → cluster together.
VARIETY_AFFINITIES: dict[tuple[str, str], float] = {
    ("ES_CAN", "ES_CAR"): 0.92,
    ("ES_AND", "ES_AND_BO"): 0.90,
    ("ES_CHI", "ES_RIO"): 0.70,
}
AFFINITY_BASE: float = 0.10

# Corpus blending pairs: (variety_a, variety_b, blend_fraction)
BLEND_PAIRS: list[tuple[str, str, float]] = [
    ("ES_CAN", "ES_CAR", 0.20),
    ("ES_AND", "ES_AND_BO", 0.15),
]

# Curated regionalisms per variety
REGIONALISMS: dict[str, set[str]] = {
    "ES_PEN": {
        "tío", "tía", "majo", "maja", "currar", "mogollón", "mola",
        "flipar", "guay", "vosotros", "chaval", "piso", "coche",
        "zumo", "patata", "ordenador", "gilipollas", "quedada",
    },
    "ES_AND": {
        "quillo", "picha", "churumbel", "chiquillo", "pisha", "arsa",
        "illo", "compae", "bulla", "pringao", "gazpacho", "salmorejo",
        "malaje", "chipén", "chirigota", "mijilla",
    },
    "ES_CAN": {
        "guagua", "papa", "mojo", "gofio", "pelete", "baifo",
        "guanche", "machango", "beletén", "tunera", "magua",
        "enyesque", "perenquén", "bubango", "sancocho", "frangollo",
    },
    "ES_RIO": {
        "che", "pibe", "piba", "mina", "laburo", "bondi", "birra",
        "guita", "fiaca", "quilombo", "morfar", "trucho", "groso",
        "chabón", "pucho", "colectivo", "subte", "boludo", "mate",
        "remera", "campera", "vereda",
    },
    "ES_MEX": {
        "güey", "wey", "chido", "neta", "chamba", "chamaco", "mole",
        "órale", "naco", "fresa", "chafa", "cuate", "pinche", "morro",
        "chela", "camión", "antro", "chavo", "lana", "alberca",
        "banqueta", "cajuela", "chamarra", "popote",
    },
    "ES_CAR": {
        "chévere", "vaina", "chamo", "pana", "jeva", "guagua",
        "bachata", "jíbaro", "bochinche", "tripear", "habichuela",
        "mangú", "mofongo", "asere", "guarapo", "bemba",
    },
    "ES_CHI": {
        "pololo", "polola", "fome", "cachai", "bacán", "luca",
        "pololear", "carrete", "cuático", "huevón", "weón", "gallo",
        "pega", "micro", "flaite", "copete", "polera", "guata",
        "altiro", "palta", "once", "completo",
    },
    "ES_AND_BO": {
        "cholo", "chompa", "pollera", "chuño", "wawa", "soroche",
        "chacra", "cancha", "calato", "cuy", "ñaño", "pata",
        "jato", "causa", "chaufa", "chicha", "charqui", "quinua",
        "yapa", "chifa", "huayno", "puna", "anticucho", "choclo",
    },
}

ALL_REGIONALISMS: frozenset[str] = frozenset().union(*REGIONALISMS.values())

# Default hyperparameters
DEFAULT_SEED: int = 42
EMBEDDING_DIM: int = 100
BPE_VOCAB_SIZE: int = 8000

# ---------------------------------------------------------------------------
# Scraper configuration
# ---------------------------------------------------------------------------

SCRAPER_SUBREDDITS: dict[str, list[str]] = {
    "ES_PEN": ["spain", "SpainPolitics", "preguntaReddit", "Barcelona", "Madrid"],
    "ES_AND": ["Andalucia", "Sevilla", "Granada", "Malaga", "Cordoba",
               "Cadiz", "Huelva", "Jerez", "almeria"],
    "ES_CAN": ["canarias", "LasPalmas", "Tenerife", "SantaCruzdeTenerife",
               "GranCanaria", "Lanzarote", "Fuerteventura"],
    "ES_RIO": ["argentina", "Republica_Argentina", "uruguay", "BuenosAires"],
    "ES_MEX": ["mexico", "Mujico", "monterrey", "Guadalajara"],
    "ES_CAR": ["cuba", "vzla", "Colombia", "PuertoRico", "Dominican"],
    "ES_CHI": ["chile", "RepublicadeChile", "Santiago"],
    "ES_AND_BO": ["Bolivia", "PERU", "Ecuador"],
}

SCRAPER_WIKI_ARTICLES: dict[str, list[str]] = {
    "ES_PEN": [
        "Español peninsular", "Dialecto madrileño", "Ceceo y seseo",
        "Leísmo", "Gramática del español", "Madrid", "Cultura de España",
        "Gastronomía de España", "Historia de España", "Lenguas de España",
        "Comunidad de Madrid", "Castilla", "Literatura española",
        "Real Academia Española", "Fiestas de España", "Deporte en España",
    ],
    "ES_AND": [
        "Dialecto andaluz", "Sevilla", "Andalucía", "Flamenco",
        "Gastronomía de Andalucía", "Semana Santa en Sevilla",
        "Feria de Abril", "Historia de Andalucía", "Córdoba (España)",
        "Granada", "Málaga", "Cádiz", "Al-Ándalus",
    ],
    "ES_CAN": [
        "Español canario", "Canarias", "Las Palmas de Gran Canaria",
        "Guanche", "Tenerife", "Gastronomía de Canarias",
        "Gofio", "Mojo picón", "Carnaval de Santa Cruz de Tenerife",
        "Silbo gomero", "Gran Canaria", "Historia de Canarias",
    ],
    "ES_RIO": [
        "Español rioplatense", "Lunfardo", "Voseo", "Buenos Aires",
        "Tango", "Mate (infusión)", "Gastronomía de Argentina",
        "Cultura de Argentina", "Historia de Argentina", "Montevideo",
        "Uruguay", "Río de la Plata", "Fútbol en Argentina",
    ],
    "ES_MEX": [
        "Español mexicano", "Nahuatlismos", "Ciudad de México",
        "Gastronomía de México", "Cultura de México", "Mariachi",
        "Día de Muertos", "Historia de México", "Azteca",
        "Guadalajara (México)", "Monterrey", "Oaxaca",
    ],
    "ES_CAR": [
        "Español caribeño", "La Habana", "Cultura de Cuba",
        "República Dominicana", "Salsa (música)", "Reguetón",
        "Béisbol en el Caribe", "Venezuela", "Gastronomía de Cuba",
        "Historia de Cuba", "San Juan (Puerto Rico)", "Merengue (música)",
    ],
    "ES_CHI": [
        "Español chileno", "Chilenismos", "Santiago de Chile",
        "Cultura de Chile", "Mapuche", "Gastronomía de Chile",
        "Historia de Chile", "Patagonia", "Viña del Mar",
        "Valparaíso", "Desierto de Atacama", "Vino chileno",
    ],
    "ES_AND_BO": [
        "Español andino", "Quechua", "La Paz", "Bolivia",
        "Cultura de Bolivia", "Perú", "Lima", "Cusco",
        "Gastronomía de Bolivia", "Gastronomía del Perú",
        "Aymara", "Altiplano", "Quito", "Ecuador",
    ],
}

SCRAPER_FILMS: dict[str, list[str]] = {
    "ES_PEN": ["Volver", "Todo sobre mi madre", "Mar adentro", "El laberinto del fauno",
               "Abre los ojos", "Tesis", "El día de la bestia", "Torrente"],
    "ES_AND": ["Solas", "Seis puntos sobre Emma", "Grupo 7", "La isla mínima",
               "7 Vírgenes", "El camino de los ingleses",
               "Vivir es fácil con los ojos cerrados", "La peste", "Caníbal"],
    "ES_CAN": ["Mararía", "Hierro", "El jardín de las delicias",
               "Una hora más en Canarias", "Guarapo", "La piel del volcán",
               "Mana", "La isla del viento"],
    "ES_RIO": ["El secreto de sus ojos", "Nueve reinas", "Relatos salvajes",
               "El ángel", "Pizza birra faso", "Un cuento chino", "Esperando la carroza"],
    "ES_MEX": ["Amores perros", "Y tu mamá también", "Roma", "El infierno",
               "Nosotros los nobles", "La ley de Herodes", "Como agua para chocolate"],
    "ES_CAR": ["Fresa y chocolate", "Conducta", "La estrategia del caracol",
               "Colmena", "Pelo malo", "Secuestro Express"],
    "ES_CHI": ["Una mujer fantástica", "No", "Gloria", "El club",
               "Machuca", "Tony Manero", "Violeta se fue a los cielos"],
    "ES_AND_BO": ["La nación clandestina", "Quien mató a la llamita blanca",
                   "La teta asustada", "Magallanes", "Zona sur"],
}

SCRAPER_SONGS: dict[str, list[tuple[str, str]]] = {
    "ES_PEN": [
        ("Rosalía", "Malamente"), ("Mecano", "Hijo de la luna"),
        ("Alejandro Sanz", "Corazón partío"), ("Joaquín Sabina", "19 días y 500 noches"),
        ("Estopa", "Tu calorro"), ("Vetusta Morla", "Copenhague"),
    ],
    "ES_AND": [
        ("Camarón de la Isla", "Como el agua"), ("El Barrio", "Ángel malherido"),
        ("Niña Pastori", "Cai"), ("Paco de Lucía", "Entre dos aguas"),
        ("Chambao", "Papeles mojados"), ("Dellafuente", "Bienvenido a la fiesta"),
        ("SFDK", "Los veteranos"), ("Haze", "Buscándome la vida"),
        ("Mala Rodríguez", "Quien manda"), ("Raimundo Amador", "Noche de flamenco y blues"),
    ],
    "ES_CAN": [
        ("Pedro Guerra", "Contamíname"), ("Los Sabandeños", "Islas Canarias"),
        ("Mestisay", "Folías canarias"), ("Braulio", "Esa canaria"),
        ("Taburiente", "Tierra de libertad"), ("Arístides Moreno", "Ay Lanzarote"),
        ("Pedro Guerra", "Volver a los 17"), ("Los Gofiones", "Sombra del nublo"),
    ],
    "ES_RIO": [
        ("Carlos Gardel", "Por una cabeza"), ("Mercedes Sosa", "Gracias a la vida"),
        ("Gustavo Cerati", "Crimen"), ("Charly García", "Cerca de la revolución"),
        ("Jorge Drexler", "Todo se transforma"), ("La Renga", "Balada del diablo y la muerte"),
    ],
    "ES_MEX": [
        ("José Alfredo Jiménez", "El rey"), ("Juan Gabriel", "Amor eterno"),
        ("Maná", "Rayando el sol"), ("Café Tacvba", "La ingrata"),
        ("Molotov", "Gimme tha Power"), ("Natalia Lafourcade", "Hasta la raíz"),
    ],
    "ES_CAR": [
        ("Celia Cruz", "La vida es un carnaval"), ("Rubén Blades", "Pedro Navaja"),
        ("Juan Luis Guerra", "Ojalá que llueva café"), ("Buena Vista Social Club", "Chan Chan"),
        ("Oscar D'León", "Llorarás"), ("Bad Bunny", "Dakiti"),
    ],
    "ES_CHI": [
        ("Violeta Parra", "Gracias a la vida"), ("Víctor Jara", "Te recuerdo Amanda"),
        ("Los Prisioneros", "El baile de los que sobran"), ("Los Bunkers", "Bailando solo"),
        ("Javiera Mena", "Espada"), ("Mon Laferte", "Tu falta de querer"),
    ],
    "ES_AND_BO": [
        ("Los Kjarkas", "Llorando se fue"), ("Savia Andina", "El minero"),
        ("Luzmila Carpio", "Ch'uwa yacu"), ("Eva Ayllón", "Mal paso"),
        ("Susana Baca", "María Landó"), ("Gian Marco", "Se me olvidó"),
    ],
}

OPENSUB_API_KEY: str = "Bqbl4xBUGPOQzfIkBeShLfREYLNZ3US4"
OPENSUB_BASE_URL: str = "https://api.opensubtitles.com/api/v1"

# Country TLD → dialect mapping for web corpus labeling (mC4/OSCAR)
TLD_TO_DIALECT: dict[str, str] = {
    ".es": "ES_PEN",
    ".ar": "ES_RIO", ".uy": "ES_RIO",
    ".mx": "ES_MEX",
    ".cu": "ES_CAR", ".ve": "ES_CAR", ".co": "ES_CAR",
    ".pr": "ES_CAR", ".do": "ES_CAR",
    ".cl": "ES_CHI",
    ".pe": "ES_AND_BO", ".bo": "ES_AND_BO", ".ec": "ES_AND_BO",
}
# mC4 country codes (used in allenai/c4 "es" split URL filtering)
COUNTRY_TO_DIALECT: dict[str, str] = {
    "spain": "ES_PEN", "es": "ES_PEN",
    "argentina": "ES_RIO", "ar": "ES_RIO",
    "uruguay": "ES_RIO", "uy": "ES_RIO",
    "mexico": "ES_MEX", "mx": "ES_MEX",
    "cuba": "ES_CAR", "cu": "ES_CAR",
    "venezuela": "ES_CAR", "ve": "ES_CAR",
    "colombia": "ES_CAR", "co": "ES_CAR",
    "puerto_rico": "ES_CAR", "pr": "ES_CAR",
    "dominican_republic": "ES_CAR", "do": "ES_CAR",
    "chile": "ES_CHI", "cl": "ES_CHI",
    "peru": "ES_AND_BO", "pe": "ES_AND_BO",
    "bolivia": "ES_AND_BO", "bo": "ES_AND_BO",
    "ecuador": "ES_AND_BO", "ec": "ES_AND_BO",
}

# Spanish word markers for language detection fallback
_SPANISH_MARKERS: frozenset[str] = frozenset({
    "de", "la", "el", "en", "que", "los", "las", "del", "por", "con",
    "una", "para", "como", "pero", "más", "este", "esta", "todo", "también",
    "fue", "ser", "tiene", "era", "hay", "muy", "puede", "ese", "eso",
    "ya", "así", "sin", "sobre", "entre", "cuando", "donde", "desde",
    "otro", "otra", "sus", "nos", "cada", "hasta", "bien", "hacer",
    "porque", "entonces", "después", "antes", "solo", "mismo", "ahora",
})
_ENGLISH_MARKERS: frozenset[str] = frozenset({
    "the", "is", "are", "was", "were", "have", "has", "had", "been",
    "will", "would", "could", "should", "may", "might", "can", "this",
    "that", "with", "from", "they", "their", "which", "about", "into",
    "than", "been", "its", "after", "also", "who", "did", "just",
})

# ---------------------------------------------------------------------------
# Dialect keyword markers — for rescuing CAN/AND docs from .es mC4 pool.
# High-precision words that almost exclusively indicate a specific dialect
# when found in .es-domain text.
# ---------------------------------------------------------------------------

DIALECT_MARKERS: dict[str, frozenset[str]] = {
    "ES_CAN": frozenset({
        # Guanche substrate + Canarian-only terms
        "guagua", "gofio", "mojo", "papas arrugadas", "pelete", "baifo",
        "guanche", "machango", "beletén", "tunera", "magua", "enyesque",
        "perenquén", "bubango", "sancocho", "frangollo", "jareas",
        "guarapo", "timple", "tajinaste", "drago", "sabina",
        # Canarian place markers (in context with dialect words)
        "tenerife", "gran canaria", "lanzarote", "fuerteventura",
        "gomera", "hierro", "palm", "teide", "guanches",
        # Canarian expressions
        "ño", "ñol", "cambado", "perenquén", "irse de roscas",
        "mi niño", "mi niña", "irse de cachimba",
    }),
    "ES_AND": frozenset({
        # High-precision Andalusian terms
        "quillo", "picha", "churumbel", "pisha", "arsa", "illo",
        "compae", "bulla", "pringao", "malaje", "chipén", "chirigota",
        "mijilla", "miarma", "ozú", "chiquillo", "chaval",
        # Flamenco/cultural markers
        "cante jondo", "tablao", "compás", "bulería", "soleá",
        "seguiriya", "fandango", "duende",
        # Andalusian food/culture
        "gazpacho", "salmorejo", "pescaíto", "rebujito",
        "feria", "semana santa", "romería", "procesión",
        # Place-in-context markers
        "sevilla", "málaga", "cádiz", "córdoba", "granada",
        "jerez", "huelva", "almería", "jaén",
    }),
}

# ---------------------------------------------------------------------------
# Regional news sites — RSS feeds and article URLs for CAN and AND scraping
# ---------------------------------------------------------------------------

SCRAPER_NEWS_SITES: dict[str, list[dict[str, str]]] = {
    "ES_CAN": [
        {"name": "Canarias7", "rss": "https://www.canarias7.es/rss/section/18001"},
        {"name": "ElDía", "rss": "https://www.eldia.es/rss/section/18007"},
        {"name": "LaProvincia", "rss": "https://www.laprovincia.es/rss/section/18000"},
        {"name": "DiarioDeAvisos", "rss": "https://www.diariodeavisos.com/feed/"},
        {"name": "CanariasAhora", "rss": "https://www.eldiario.es/canariasahora/rss/"},
    ],
    "ES_AND": [
        {"name": "DiarioDeSevilla", "rss": "https://www.diariodesevilla.es/rss/section/18005"},
        {"name": "Ideal", "rss": "https://www.ideal.es/rss/2.0/?section=granada"},
        {"name": "MalagaHoy", "rss": "https://www.malagahoy.es/rss/section/18006"},
        {"name": "DiarioSur", "rss": "https://www.diariosur.es/rss/2.0/"},
        {"name": "DiarioDeCadiz", "rss": "https://www.diariodecadiz.es/rss/section/18004"},
        {"name": "CórdobaHoy", "rss": "https://www.eldiadecordoba.es/rss/section/18003"},
        {"name": "AndalucíaInfo", "rss": "https://www.eldiario.es/andalucia/rss/"},
    ],
}

# ---------------------------------------------------------------------------
# Phonological rules for synthetic dialect augmentation (PEN → CAN/AND).
# Each rule: (pattern, replacement, description)
# Applied sequentially; order matters.
# ---------------------------------------------------------------------------

PHONOLOGICAL_RULES: dict[str, list[tuple[str, str, str]]] = {
    "ES_CAN": [
        # /s/ aspiration before consonants: "estos" → "ehtoh", "mismo" → "mihmo"
        (r"s(?=[bcdfghjklmnñpqrstvwxyz])", "h", "s-aspiration before consonant"),
        # Final /s/ deletion/aspiration: "los" → "loh", "más" → "máh"
        (r"s\b", "h", "final s-aspiration"),
        # /d/ intervocalic weakening: "cansado" → "cansao", "helado" → "helao"
        (r"(?<=[aeiouáéíóú])d(?=[ao]\b)", "", "intervocalic d-deletion"),
        # Seseo: "z" before e/i → "s": "cerveza" → "servesa"
        (r"c(?=[ei])", "s", "seseo c→s"),
        (r"z", "s", "seseo z→s"),
    ],
    "ES_AND": [
        # /s/ aspiration (stronger than CAN): "estos" → "ehtoh"
        (r"s(?=[bcdfghjklmnñpqrstvwxyz])", "h", "s-aspiration before consonant"),
        # Final /s/ aspiration: "los" → "loh"
        (r"s\b", "h", "final s-aspiration"),
        # /d/ intervocalic deletion (very frequent): "cansado"→"cansao"
        (r"(?<=[aeiouáéíóú])d(?=[aoe]\b)", "", "intervocalic d-deletion"),
        # /d/ deletion at end of word: "verdad"→"verdá", "ciudad"→"ciudá"
        (r"d\b", "", "final d-deletion"),
        # Seseo/ceceo: "z"→"s" (seseo variant, most common)
        (r"c(?=[ei])", "s", "seseo c→s"),
        (r"z", "s", "seseo z→s"),
        # Ustedes for vosotros: morphological, handled separately
    ],
}

# Vosotros → ustedes substitutions for ES_AND (morphological, not regex)
USTEDES_MAP: dict[str, str] = {
    "vosotros": "ustedes", "vosotras": "ustedes",
    "os": "se", "vuestro": "su", "vuestra": "su",
    "vuestros": "sus", "vuestras": "sus",
    # Verb forms: 2pl → 3pl
    "tenéis": "tienen", "estáis": "están", "sois": "son",
    "habéis": "han", "podéis": "pueden", "sabéis": "saben",
    "queréis": "quieren", "venís": "vienen", "decís": "dicen",
    "hacéis": "hacen", "vais": "van", "coméis": "comen",
    "vivís": "viven", "habláis": "hablan", "pensáis": "piensan",
    "creéis": "creen", "sentís": "sienten",
}

# Canarian lexical substitutions (common PEN → CAN word swaps)
CANARIAN_LEXICON: dict[str, str] = {
    "autobús": "guagua", "patata": "papa", "zumo": "jugo",
    "conducir": "guiar", "enfadarse": "emberrenchinarse",
    "niño": "chacho", "trabajar": "bregar", "coche": "carro",
    "piso": "apartamento", "acera": "andén", "frigorífico": "nevera",
}

# ---------------------------------------------------------------------------
# Bulk corpus downloader configuration (v4)
# ---------------------------------------------------------------------------

# HuggingFace dataset identifiers
HF_CULTURAX_DATASET: str = "uonlp/CulturaX"
HF_ARCTIC_DATASET: str = "open-index/arctic"
HF_TWEETS_DATASET: str = "pysentimiento/spanish-tweets"
HF_LEIPZIG_DATASET: str = "imvladikon/leipzig_corpora_collection"

# Twitter / tweet country code → dialect mapping
TWITTER_COUNTRY_TO_DIALECT: dict[str, str] = {
    "ES": "ES_PEN", "AR": "ES_RIO", "UY": "ES_RIO",
    "MX": "ES_MEX", "CL": "ES_CHI",
    "CU": "ES_CAR", "VE": "ES_CAR", "CO": "ES_CAR",
    "PR": "ES_CAR", "DO": "ES_CAR", "PA": "ES_CAR",
    "BO": "ES_AND_BO", "PE": "ES_AND_BO", "EC": "ES_AND_BO",
}

# Geo bounding boxes for intra-Spain classification (lat_min, lat_max, lon_min, lon_max)
GEO_BOUNDS_CAN: tuple[float, float, float, float] = (27.5, 29.5, -18.5, -13.0)
GEO_BOUNDS_AND: tuple[float, float, float, float] = (36.0, 38.7, -7.5, -1.6)

# Leipzig Corpora direct download URLs — {corpus_id: (download_url, dialect_code)}
# URL base: https://downloads.wortschatz-leipzig.de/corpora/
# Spain uses "spa_" (no country code); others use "spa-{cc}"
# Chile has NO Leipzig corpus; Bolivia only 10K
LEIPZIG_CORPORA: dict[str, tuple[str, str]] = {
    # Spain (generic "spa_")
    "spa_news_2024": (
        "https://downloads.wortschatz-leipzig.de/corpora/spa_news_2024_1M.tar.gz", "ES_PEN"),
    "spa_newscrawl_2018": (
        "https://downloads.wortschatz-leipzig.de/corpora/spa_newscrawl_2018_1M.tar.gz", "ES_PEN"),
    # Argentina + Uruguay → ES_RIO
    "spa-ar_web_2016": (
        "https://downloads.wortschatz-leipzig.de/corpora/spa-ar_web_2016_1M.tar.gz", "ES_RIO"),
    "spa-uy_web_2016": (
        "https://downloads.wortschatz-leipzig.de/corpora/spa-uy_web_2016_1M.tar.gz", "ES_RIO"),
    # Mexico → ES_MEX
    "spa-mx_web_2015": (
        "https://downloads.wortschatz-leipzig.de/corpora/spa-mx_web_2015_1M.tar.gz", "ES_MEX"),
    # Caribbean: Cuba + Venezuela + Colombia + Dominican Republic → ES_CAR
    "spa-cu_web_2015": (
        "https://downloads.wortschatz-leipzig.de/corpora/spa-cu_web_2015_1M.tar.gz", "ES_CAR"),
    "spa-ve_web_2016": (
        "https://downloads.wortschatz-leipzig.de/corpora/spa-ve_web_2016_1M.tar.gz", "ES_CAR"),
    "spa-co_web_2015": (
        "https://downloads.wortschatz-leipzig.de/corpora/spa-co_web_2015_1M.tar.gz", "ES_CAR"),
    "spa-do_web_2015": (
        "https://downloads.wortschatz-leipzig.de/corpora/spa-do_web_2015_1M.tar.gz", "ES_CAR"),
    # Andean: Peru + Ecuador + Bolivia → ES_AND_BO
    "spa-pe_web_2016": (
        "https://downloads.wortschatz-leipzig.de/corpora/spa-pe_web_2016_1M.tar.gz", "ES_AND_BO"),
    "spa-ec_web_2015": (
        "https://downloads.wortschatz-leipzig.de/corpora/spa-ec_web_2015_1M.tar.gz", "ES_AND_BO"),
    "spa-bo_web_2015": (
        "https://downloads.wortschatz-leipzig.de/corpora/spa-bo_web_2015_10K.tar.gz", "ES_AND_BO"),
    # Chile: NO Leipzig corpus available — will rely on CulturaX/tweets/Reddit
}

# OPUS OpenSubtitles bulk download (monolingual Spanish)
OPUS_OPENSUB_URL: str = "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2024/mono/es.txt.gz"

# Arctic Shift: subreddits to download — reuses SCRAPER_SUBREDDITS mapping above

# Literary works catalog — {dialect: [{title, author, source, id}]}
# source: "gutenberg" (use gutenberg ID), "manual" (user places in data/literary/{dialect}/)
LITERARY_WORKS: dict[str, list[dict[str, str]]] = {
    "ES_CAN": [
        {"title": "Panza de burro", "author": "Andrea Abreu", "source": "manual"},
        {"title": "Terramores", "author": "Acerina Cruz", "source": "manual"},
    ],
    "ES_AND": [
        {"title": "Bodas de sangre", "author": "Federico García Lorca", "source": "manual"},
        {"title": "La casa de Bernarda Alba", "author": "Federico García Lorca",
         "source": "manual"},
        {"title": "Yerma", "author": "Federico García Lorca", "source": "manual"},
    ],
    "ES_RIO": [
        {"title": "Martín Fierro", "author": "José Hernández", "source": "gutenberg",
         "id": "14765"},
        {"title": "El Aleph", "author": "Jorge Luis Borges", "source": "manual"},
    ],
    "ES_MEX": [
        {"title": "Pedro Páramo", "author": "Juan Rulfo", "source": "manual"},
        {"title": "El llano en llamas", "author": "Juan Rulfo", "source": "manual"},
    ],
    "ES_CAR": [
        {"title": "Poesía completa", "author": "Nicolás Guillén", "source": "manual"},
    ],
    "ES_CHI": [
        {"title": "Canto general", "author": "Pablo Neruda", "source": "manual"},
        {"title": "Los detectives salvajes", "author": "Roberto Bolaño", "source": "manual"},
    ],
    "ES_AND_BO": [
        {"title": "Raza de bronce", "author": "Alcides Arguedas", "source": "manual"},
        {"title": "Los ríos profundos", "author": "José María Arguedas", "source": "manual"},
    ],
}

# Enhanced dialect markers for better .es reclassification (extends DIALECT_MARKERS)
DIALECT_MARKERS_EXTENDED: dict[str, frozenset[str]] = {
    "ES_CAN": DIALECT_MARKERS["ES_CAN"] | frozenset({
        "isleño", "canario", "canaria", "calima", "alisio",
        "mojo picón", "plátano canario", "silbo", "gomero",
    }),
    "ES_AND": DIALECT_MARKERS["ES_AND"] | frozenset({
        "andaluz", "andaluza", "andalucía", "mare mía",
        "ustede", "ehtoy", "mihmo", "sevillano", "sevillana",
        "malagueño", "gaditano", "cordobés", "jiennense",
    }),
}

# Base confidence scores per data source
SOURCE_CONFIDENCE: dict[str, float] = {
    "twitter_geo": 0.90,
    "literary": 0.85,
    "culturax_tld": 0.75,
    "leipzig": 0.75,
    "twitter_content": 0.60,
    "reddit_sub": 0.70,
    "opensubtitles": 0.60,
    "culturax_reclassified": 0.55,
    "synthetic": 0.40,
    # v4.5 new sources
    "preseea": 0.95,          # Academic sociolinguistic gold standard
    "coser": 0.95,            # Rural academic corpus, province-tagged
    "tweet_hisp_city": 0.90,  # Tweets with explicit city+country
    "lanzarote_cabildo": 0.88,  # Pure Canarian government transcripts
    "cv17_accent": 0.85,      # Common Voice with accent metadata
    "parcan": 0.92,           # Parlamento de Canarias official transcripts
}

# ======================================================================
# v4.5 Additional sources (high-quality, region-tagged)
# ======================================================================

# HuggingFace dataset IDs for v4.5 new sources
HF_TWEET_HISP_DATASET: str = "johnatanebonilla/tweet_hisp"
HF_PRESEEA_DATASET: str = "marianbasti/preseea"
HF_COSER_DATASET: str = "cladsu/COSER-2024"
HF_LANZAROTE_DATASET: str = "StephannyPulido/corpus_registro_canarias"
HF_CV17_ES_DATASET: str = "projecte-aina/cv17_es_other_automatically_verified"

# Parlamento de Canarias base URL pattern
PARCAN_BASE_URL: str = "https://www.parcan.es/files/pub/diarios"
PARCAN_MAX_LEGISLATURE: int = 11
PARCAN_MAX_SESSIONS_PER_LEG: int = 200

# Spanish province → dialect (for COSER and tweet_hisp city field)
# Covers all 50 provinces + 2 autonomous cities
SPAIN_PROVINCE_TO_DIALECT: dict[str, str] = {
    # Andalusian (8 provinces)
    "almería": "ES_AND", "almeria": "ES_AND",
    "cádiz": "ES_AND", "cadiz": "ES_AND",
    "córdoba": "ES_AND", "cordoba": "ES_AND",
    "granada": "ES_AND",
    "huelva": "ES_AND",
    "jaén": "ES_AND", "jaen": "ES_AND",
    "málaga": "ES_AND", "malaga": "ES_AND",
    "sevilla": "ES_AND", "seville": "ES_AND",
    # Canarian (2 official provinces + individual islands as used by COSER)
    "las palmas": "ES_CAN",
    "santa cruz de tenerife": "ES_CAN", "tenerife": "ES_CAN",
    "santa cruz tenerife": "ES_CAN",
    # Individual islands (COSER uses these directly as "provincia")
    "gran canaria": "ES_CAN",
    "fuerteventura": "ES_CAN",
    "lanzarote": "ES_CAN",
    "la palma": "ES_CAN",
    "la gomera": "ES_CAN",
    "el hierro": "ES_CAN",
    "la graciosa": "ES_CAN",
    # Balearic Islands → Peninsular
    "menorca": "ES_PEN",
    "ibiza": "ES_PEN", "eivissa": "ES_PEN",
    "formentera": "ES_PEN",
    # Rest of Spain → Peninsular
    "a coruña": "ES_PEN", "a coruna": "ES_PEN", "la coruña": "ES_PEN",
    "álava": "ES_PEN", "alava": "ES_PEN",
    "albacete": "ES_PEN",
    "alicante": "ES_PEN",
    "asturias": "ES_PEN", "oviedo": "ES_PEN",
    "ávila": "ES_PEN", "avila": "ES_PEN",
    "badajoz": "ES_PEN",
    "barcelona": "ES_PEN",
    "bizkaia": "ES_PEN", "vizcaya": "ES_PEN", "bilbao": "ES_PEN",
    "burgos": "ES_PEN",
    "cáceres": "ES_PEN", "caceres": "ES_PEN",
    "cantabria": "ES_PEN", "santander": "ES_PEN",
    "castellón": "ES_PEN", "castellon": "ES_PEN",
    "ciudad real": "ES_PEN",
    "cuenca": "ES_PEN",
    "girona": "ES_PEN", "gerona": "ES_PEN",
    "gipuzkoa": "ES_PEN", "guipúzcoa": "ES_PEN", "guipuzcoa": "ES_PEN",
    "guadalajara": "ES_PEN",
    "huesca": "ES_PEN",
    "illes balears": "ES_PEN", "baleares": "ES_PEN", "mallorca": "ES_PEN",
    "la rioja": "ES_PEN", "logroño": "ES_PEN",
    "león": "ES_PEN", "leon": "ES_PEN",
    "lleida": "ES_PEN", "lérida": "ES_PEN", "lerida": "ES_PEN",
    "lugo": "ES_PEN",
    "madrid": "ES_PEN",
    "murcia": "ES_PEN",
    "navarra": "ES_PEN", "pamplona": "ES_PEN",
    "ourense": "ES_PEN", "orense": "ES_PEN",
    "palencia": "ES_PEN",
    "pontevedra": "ES_PEN",
    "salamanca": "ES_PEN",
    "segovia": "ES_PEN",
    "soria": "ES_PEN",
    "tarragona": "ES_PEN",
    "teruel": "ES_PEN",
    "toledo": "ES_PEN",
    "valencia": "ES_PEN", "valència": "ES_PEN",
    "valladolid": "ES_PEN",
    "zamora": "ES_PEN",
    "zaragoza": "ES_PEN",
    "ceuta": "ES_AND",  # Culturally/linguistically Andalusian-adjacent
    "melilla": "ES_AND",
}

# Canarian cities/islands (for tweet_hisp city matching)
CANARIAN_CITIES: frozenset[str] = frozenset({
    "las palmas", "las palmas de gran canaria", "lpa", "lpgc",
    "santa cruz de tenerife", "santa cruz", "tenerife", "gran canaria",
    "fuerteventura", "lanzarote", "la palma", "la gomera", "el hierro",
    "puerto del rosario", "arrecife", "la laguna", "san cristóbal de la laguna",
    "adeje", "arona", "telde", "mogán", "maspalomas", "ingenio",
    "candelaria", "icod de los vinos", "los realejos", "santa lucía de tirajana",
    "agüimes", "arucas", "gáldar", "tacoronte", "granadilla de abona",
    "puerto de la cruz", "los llanos de aridane", "breña alta",
    "san bartolomé de tirajana", "puerto de las nieves",
})

# Andalusian cities (for tweet_hisp city matching)
ANDALUSIAN_CITIES: frozenset[str] = frozenset({
    "sevilla", "seville", "málaga", "malaga", "granada", "córdoba", "cordoba",
    "jaén", "jaen", "huelva", "cádiz", "cadiz", "almería", "almeria",
    "jerez de la frontera", "jerez", "marbella", "dos hermanas", "algeciras",
    "roquetas de mar", "roquetas", "fuengirola", "motril", "linares",
    "antequera", "el ejido", "chiclana de la frontera", "chiclana",
    "san fernando", "sanlúcar de barrameda", "sanlúcar", "úbeda", "ubeda",
    "ronda", "mijas", "estepona", "puerto de santa maría", "puerto santa maria",
    "vélez-málaga", "velez-malaga", "lucena", "ecija", "écija", "lepe",
    "utrera", "alcalá de guadaíra", "alcala de guadaira", "tomares",
    "montilla", "baena", "priego de córdoba", "puente genil",
    "benalmádena", "benalmadena", "torremolinos", "rincón de la victoria",
    "cartaya", "isla cristina", "aracena", "alcalá la real", "baza",
    "loja", "motril", "andújar", "andujar", "martos", "alcaudete",
    "dos hermanas", "morón de la frontera", "moron", "marchena", "carmona",
})

# tweet_hisp country → dialect (same shape as TWITTER_COUNTRY_TO_DIALECT but matches tweet_hisp schema)
TWEET_HISP_COUNTRY_TO_DIALECT: dict[str, str] = {
    # country field values may be full names or ISO codes
    "españa": "ES_PEN", "spain": "ES_PEN", "es": "ES_PEN",
    "argentina": "ES_RIO", "ar": "ES_RIO",
    "uruguay": "ES_RIO", "uy": "ES_RIO",
    "paraguay": "ES_RIO", "py": "ES_RIO",
    "méxico": "ES_MEX", "mexico": "ES_MEX", "mx": "ES_MEX",
    "chile": "ES_CHI", "cl": "ES_CHI",
    "cuba": "ES_CAR", "cu": "ES_CAR",
    "venezuela": "ES_CAR", "ve": "ES_CAR",
    "colombia": "ES_CAR", "co": "ES_CAR",
    "república dominicana": "ES_CAR", "republica dominicana": "ES_CAR",
    "dominican republic": "ES_CAR", "do": "ES_CAR",
    "puerto rico": "ES_CAR", "pr": "ES_CAR",
    "panamá": "ES_CAR", "panama": "ES_CAR", "pa": "ES_CAR",
    "bolivia": "ES_AND_BO", "bo": "ES_AND_BO",
    "perú": "ES_AND_BO", "peru": "ES_AND_BO", "pe": "ES_AND_BO",
    "ecuador": "ES_AND_BO", "ec": "ES_AND_BO",
    "guatemala": "ES_MEX", "gt": "ES_MEX",
    "honduras": "ES_MEX", "hn": "ES_MEX",
    "nicaragua": "ES_MEX", "ni": "ES_MEX",
    "el salvador": "ES_MEX", "sv": "ES_MEX",
    "costa rica": "ES_MEX", "cr": "ES_MEX",
}

# PRESEEA city codes/names → dialect (city found in metadata.csv or XML)
# PRESEEA uses 3-letter city codes + full city names in metadata
PRESEEA_CITY_TO_DIALECT: dict[str, str] = {
    # Canary Islands
    "lpa": "ES_CAN", "lpgc": "ES_CAN", "las palmas": "ES_CAN",
    "las palmas de gran canaria": "ES_CAN",
    # Andalusia
    "sev": "ES_AND", "sevilla": "ES_AND",
    "mál": "ES_AND", "mala": "ES_AND", "málaga": "ES_AND", "malaga": "ES_AND",
    "gra": "ES_AND", "granada": "ES_AND",
    "alm": "ES_AND", "almería": "ES_AND", "almeria": "ES_AND",
    "cór": "ES_AND", "córdoba": "ES_AND", "cordoba": "ES_AND",
    # Peninsular Spain
    "alc": "ES_PEN", "alcalá": "ES_PEN", "alcalá de henares": "ES_PEN",
    "mad": "ES_PEN", "madrid": "ES_PEN",
    "val": "ES_PEN", "valencia": "ES_PEN", "valència": "ES_PEN",
    # Mexico
    "mex": "ES_MEX", "méxico": "ES_MEX", "mexico": "ES_MEX",
    "monterrey": "ES_MEX", "mty": "ES_MEX",
    "puebla": "ES_MEX", "pue": "ES_MEX",
    "guadalajara": "ES_MEX", "gdl": "ES_MEX",
    # Chile
    "sgo": "ES_CHI", "santiago": "ES_CHI", "santiago de chile": "ES_CHI",
    "stgo": "ES_CHI",
    # Argentina / Uruguay (Rioplatense)
    "bue": "ES_RIO", "baires": "ES_RIO", "buenos aires": "ES_RIO",
    "mvd": "ES_RIO", "montevideo": "ES_RIO",
    "rosario": "ES_RIO",
    # Caribbean
    "hab": "ES_CAR", "habana": "ES_CAR", "la habana": "ES_CAR", "havana": "ES_CAR",
    "ccs": "ES_CAR", "caracas": "ES_CAR",
    "bogotá": "ES_CAR", "bogota": "ES_CAR", "bog": "ES_CAR",
    "medellín": "ES_CAR", "medellin": "ES_CAR", "mde": "ES_CAR",
    "cali": "ES_CAR", "barranquilla": "ES_CAR",
    "san juan": "ES_CAR", "sj": "ES_CAR",
    "santo domingo": "ES_CAR", "sd": "ES_CAR",
    # Andean-Bolivian
    "lpz": "ES_AND_BO", "la paz": "ES_AND_BO",
    "lim": "ES_AND_BO", "lima": "ES_AND_BO",
    "quito": "ES_AND_BO", "uio": "ES_AND_BO",
    "guayaquil": "ES_AND_BO", "gye": "ES_AND_BO",
    "cochabamba": "ES_AND_BO", "cbb": "ES_AND_BO",
    "sucre": "ES_AND_BO",
}

# Common Voice 17 accent labels → dialect
# The accent field has verbose descriptions like
#   "Andino-Pacífico: Colombia, Perú, Ecuador, oeste de Bolivia y Venezuela andina"
# Map by substring matching.
CV_ACCENT_SUBSTRING_TO_DIALECT: list[tuple[str, str]] = [
    # Most specific first — order matters for substring matching.
    # CAN before AND, AND before MEX, etc.
    ("canari", "ES_CAN"),  # matches "canarias", "canario", "canary"
    ("canary", "ES_CAN"),
    ("islas canarias", "ES_CAN"),
    # Andalusian (Sur peninsular bundles AND+Extremadura+Murcia in CV17)
    ("sur peninsular", "ES_AND"),
    ("andaluz", "ES_AND"),
    ("andalusi", "ES_AND"),
    ("andalucia", "ES_AND"),
    ("andalucía", "ES_AND"),
    ("extremadura", "ES_AND"),
    ("murcia", "ES_AND"),
    # Rioplatense
    ("rioplatense", "ES_RIO"),
    ("rio de la plata", "ES_RIO"),
    ("río de la plata", "ES_RIO"),
    ("argentino", "ES_RIO"),
    ("uruguayo", "ES_RIO"),
    # Chilean
    ("chileno", "ES_CHI"),
    ("chilean", "ES_CHI"),
    # Mexican — bare word "méxico"/"mexico" first to catch CV17's "México" label
    ("ciudad de méxico", "ES_MEX"),
    ("ciudad de mexico", "ES_MEX"),
    ("méxico", "ES_MEX"),
    ("mexico", "ES_MEX"),
    ("mexicano", "ES_MEX"),
    ("mexican", "ES_MEX"),
    ("centroamerican", "ES_MEX"),
    ("central american", "ES_MEX"),
    ("américa central", "ES_MEX"),
    ("america central", "ES_MEX"),
    # Caribbean
    ("caribe", "ES_CAR"),
    ("caribbean", "ES_CAR"),
    ("cubano", "ES_CAR"),
    ("venezolano", "ES_CAR"),
    # Andean
    ("andino-pacífico", "ES_AND_BO"),
    ("andino pacifico", "ES_AND_BO"),
    ("andean", "ES_AND_BO"),
    ("peruano", "ES_AND_BO"),
    ("boliviano", "ES_AND_BO"),
    ("ecuatoriano", "ES_AND_BO"),
    # Spain variants (peninsular last so it doesn't swallow andaluz/canari)
    ("norte peninsular", "ES_PEN"),
    ("centro-sur peninsular", "ES_PEN"),
    ("centro peninsular", "ES_PEN"),
    ("este peninsular", "ES_PEN"),
    ("comunidad valenciana", "ES_PEN"),
    ("castellano", "ES_PEN"),
    ("madrid", "ES_PEN"),
    ("galicia", "ES_PEN"),
    ("españa", "ES_PEN"),
    ("spain", "ES_PEN"),
]
