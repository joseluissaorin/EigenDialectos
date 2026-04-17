"""Hand-crafted dialect fixture sentences for testing and baseline evaluation.

Each dialect has ~15-20 sentences that showcase its distinctive features.
The DIALECT_FEATURES dictionary documents the key linguistic features
with concrete examples for each variety.
"""

from __future__ import annotations

from eigendialectos.constants import DialectCode, FeatureCategory
from eigendialectos.types import DialectSample

# ======================================================================
# Key dialect features with examples
# ======================================================================

DIALECT_FEATURES: dict[DialectCode, dict[FeatureCategory, list[str]]] = {
    DialectCode.ES_PEN: {
        FeatureCategory.LEXICAL: [
            "ordenador (computadora)", "coche (carro)", "autobús (camión/colectivo)",
            "móvil (celular)", "gafas (lentes)", "piso (departamento)",
            "mola (gusta)", "tío/tía (coloquial para persona)",
        ],
        FeatureCategory.MORPHOSYNTACTIC: [
            "vosotros/as: vosotros tenéis, vosotras sabéis",
            "leísmo aceptado: le vi ayer (a él)",
            "pretérito perfecto compuesto para pasado reciente: hoy he comido",
        ],
        FeatureCategory.PRAGMATIC: [
            "vale (marcador de acuerdo)", "tío/tía (vocativo)",
            "¿no? (tag question)", "mola (aprobación)",
        ],
        FeatureCategory.PHONOLOGICAL: [
            "distinción /θ/ vs /s/: caza ≠ casa",
            "conservación de -s final",
            "conservación de -d- intervocálica: comido, cansado",
        ],
    },
    DialectCode.ES_AND: {
        FeatureCategory.LEXICAL: [
            "quillo/quilla (chico/chica)", "picha (vocativo coloquial)",
            "bulla (prisa)", "pechá (gran cantidad)",
        ],
        FeatureCategory.MORPHOSYNTACTIC: [
            "ustedes por vosotros: ustedes sabéis (mezcla)",
            "pérdida consonantes finales: comío, cantao",
        ],
        FeatureCategory.PRAGMATIC: [
            "quillo (vocativo)", "arsa (interjección)",
            "¿vale? (confirmación)", "bah (desacuerdo)",
        ],
        FeatureCategory.PHONOLOGICAL: [
            "seseo/ceceo: casa=caza, sena=cena",
            "aspiración de /s/ implosiva: ehto por esto",
            "caída de -d- intervocálica: comío, cansao",
            "rotacismo: cuerpo→cuelpo",
            "aspiración de /x/: hijo→hiho",
        ],
    },
    DialectCode.ES_CAN: {
        FeatureCategory.LEXICAL: [
            "guagua (autobús)", "papa (patata)",
            "gofio (harina de millo)", "chacho/chacha (vocativo)",
            "machango (tonto)", "perenquén (salamanquesa)",
        ],
        FeatureCategory.MORPHOSYNTACTIC: [
            "ustedes en lugar de vosotros: ustedes saben",
            "uso extendido de pretérito indefinido",
        ],
        FeatureCategory.PRAGMATIC: [
            "chacho/chacha (vocativo)", "¿verdad? (muletilla)",
            "mijo/mija (vocativo afectuoso)",
        ],
        FeatureCategory.PHONOLOGICAL: [
            "seseo generalizado",
            "aspiración de /s/ implosiva",
            "sonorización de /tʃ/: mushasho",
        ],
    },
    DialectCode.ES_RIO: {
        FeatureCategory.LEXICAL: [
            "colectivo (autobús)", "bondi (colectivo, lunfardo)",
            "pibe/piba (chico/chica)", "laburar (trabajar)",
            "afanar (robar/trabajar)", "morfar (comer)",
            "birra (cerveza)", "guita (dinero)",
            "laburo (trabajo)", "mina (mujer coloquial)",
        ],
        FeatureCategory.MORPHOSYNTACTIC: [
            "voseo: vos tenés, vos sabés, vos querés",
            "imperativo voseante: mirá, vení, decí",
            "pretérito indefinido dominante: hoy comí (no he comido)",
        ],
        FeatureCategory.PRAGMATIC: [
            "che (vocativo/interjección)", "dale (acuerdo/despedida)",
            "boludo/a (vocativo coloquial)", "mirá (marcador discursivo)",
            "¿viste? (muletilla)", "¿entendés? (tag question)",
        ],
        FeatureCategory.PHONOLOGICAL: [
            "yeísmo rehilado: /ʃ/ o /ʒ/ para ll/y: sho por yo",
            "seseo generalizado",
            "aspiración de /s/ en posición implosiva",
        ],
    },
    DialectCode.ES_MEX: {
        FeatureCategory.LEXICAL: [
            "camión (autobús urbano)", "chamba (trabajo)",
            "chamaco (niño)", "güey/wey (vocativo)",
            "neta (verdad)", "chido/padre/padrísimo (bueno)",
            "chela (cerveza)", "feria (dinero/cambio)",
            "lana (dinero)", "cuate (amigo)",
        ],
        FeatureCategory.MORPHOSYNTACTIC: [
            "tuteo: tú tienes, tú sabes",
            "uso extendido de diminutivos: ahorita, cerquita, todito",
            "le intensivo: ándale, córrele, híjole",
        ],
        FeatureCategory.PRAGMATIC: [
            "güey/wey (vocativo)", "¿va? (confirmación)",
            "órale (sorpresa/acuerdo)", "ándale (acuerdo)",
            "¿mande? (¿cómo?)", "no manches (sorpresa)",
        ],
        FeatureCategory.PHONOLOGICAL: [
            "seseo generalizado",
            "conservación de /s/ final",
            "debilitamiento vocálico en sílaba átona: psicología→[pskolo'xia]",
        ],
    },
    DialectCode.ES_CAR: {
        FeatureCategory.LEXICAL: [
            "guagua (autobús, Cuba/PR)", "carro (coche)",
            "chamo/chama (chico/chica, Venezuela)",
            "vaina (cosa)", "chevere (genial)",
            "pana (amigo)", "real (dinero, Venezuela)",
        ],
        FeatureCategory.MORPHOSYNTACTIC: [
            "tuteo dominante",
            "sujeto pronominal expreso: yo quiero, tú sabes",
            "no inversión en interrogativas: ¿qué tú quieres?",
        ],
        FeatureCategory.PRAGMATIC: [
            "mijo/mija (vocativo)", "¿oíste? (confirmación)",
            "chévere (aprobación)", "asere (vocativo, Cuba)",
            "¿tú sabes? (muletilla)",
        ],
        FeatureCategory.PHONOLOGICAL: [
            "seseo generalizado",
            "aspiración o pérdida de -s: ehte, lo' niño'",
            "lambdacismo: r→l en posición implosiva: puelta por puerta",
            "velarización de /n/ final: pan→[paŋ]",
        ],
    },
    DialectCode.ES_CHI: {
        FeatureCategory.LEXICAL: [
            "micro (autobús urbano)", "polola/pololo (novia/novio)",
            "pega (trabajo)", "al tiro (inmediatamente)",
            "fome (aburrido)", "bacán (genial)",
            "la raja (excelente)", "luca (mil pesos)",
            "cachar (entender)", "carrete (fiesta)",
        ],
        FeatureCategory.MORPHOSYNTACTIC: [
            "voseo mixto verbal: tú sabís, tú querís, tú tenís",
            "uso coloquial de segunda persona: ¿cachai?",
            "dequeísmo frecuente: pienso de que...",
        ],
        FeatureCategory.PRAGMATIC: [
            "¿cachai? (¿entiendes?)", "hueón/weón (vocativo)",
            "po (pues enfático): sí po, no po, ya po",
            "la raja (aprobación)", "¿cachái o no cachái?",
        ],
        FeatureCategory.PHONOLOGICAL: [
            "seseo generalizado",
            "aspiración de /s/ implosiva",
            "elisión frecuente de -s final",
            "palatalización de /tʃ/",
        ],
    },
    DialectCode.ES_AND_BO: {
        FeatureCategory.LEXICAL: [
            "bus (autobús)", "carro (coche)", "cholo/chola (mestizo)",
            "plata (dinero)", "caserito (vendedor habitual)",
            "chacra (terreno agrícola)", "combo (menú del día)",
        ],
        FeatureCategory.MORPHOSYNTACTIC: [
            "tuteo/voseo variable según zona",
            "conservación del pretérito perfecto compuesto",
            "uso de nomás: pase nomás, sírvase nomás",
            "doble posesivo: su casa de ella",
        ],
        FeatureCategory.PRAGMATIC: [
            "¿ya? (confirmación)", "pues (muletilla frecuente)",
            "nomás (atenuación)", "oye/oiga (vocativo)",
            "pe (pues apocopado, Perú): ya pe, sí pe",
        ],
        FeatureCategory.PHONOLOGICAL: [
            "seseo generalizado",
            "conservación de /s/ final en muchas zonas",
            "distinción /ʎ/ vs /ʝ/ en zona andina (lleísmo residual)",
            "entonación con tendencia ascendente",
        ],
    },
}


# ======================================================================
# Hand-crafted fixture sentences
# ======================================================================

_FIXTURE_TEXTS: dict[DialectCode, list[str]] = {
    # ------------------------------------------------------------------
    # ES_PEN  --  Peninsular Standard Spanish
    # ------------------------------------------------------------------
    DialectCode.ES_PEN: [
        "Vamos a coger el autobús para ir al centro de la ciudad.",
        "¿Habéis visto la película que echan esta noche en la tele?",
        "Me he comprado un ordenador nuevo y me mola mogollón.",
        "Tío, no me apetece nada tener que madrugar mañana.",
        "Quedamos a las ocho en la plaza, ¿vale?",
        "El piso que hemos visto en Malasaña está genial.",
        "¿Vosotros ya habéis cenado o queréis pedir unas cañas?",
        "Hoy he ido al médico y me ha dicho que estoy bien.",
        "Oye, ¿tenéis gafas de sol? Las he olvidado en casa.",
        "Joder, qué frío hace, vamos a meternos en el bar.",
        "He aparcado el coche junto a la fuente de la plaza.",
        "Vosotras sabéis que aquí siempre llueve en noviembre.",
        "Le he llamado por el móvil pero no me ha cogido.",
        "Este fin de semana nos vamos al pueblo de mis abuelos.",
        "Me flipa cómo mola este sitio para tomar el vermú.",
        "¿Cogemos el metro o preferís ir andando hasta la Gran Vía?",
    ],
    # ------------------------------------------------------------------
    # ES_AND  --  Andalusian
    # ------------------------------------------------------------------
    DialectCode.ES_AND: [
        "Vamoh a coger er autobú pa ir ar sentro.",
        "¿Uhtedeh han vihto la pelíscula que echan ehta noche?",
        "Me he comprao un ordenadó nuevo y me mola musho.",
        "Quillo, no me apetese bah tené que madrugá mañana.",
        "Quedamoh a lah ocho en la plasa, ¿vale?",
        "Er piho que hemoh vihto en er barrio ehtá genial.",
        "¿Uhtedeh ya habéih senao o queréih pedí unah cañah?",
        "Hoy he ío ar médico y me ha disho que ehtoy bien.",
        "Oye, ¿tenéih gafah de sol? Lah he olvidao en casa.",
        "Arsa, qué frío hase, vamoh a meternoh en er bá.",
        "He aparcao er coshe junto a la fuente de la plasa.",
        "Uhtedeh sabéih que aquí siempre llueve en noviembre.",
        "Le he llamao por er móvil pero no me ha cogío.",
        "Ehte fin de semana noh vamoh ar pueblo de mih abueloh.",
        "Me flipa cómo mola ehte sitio pa tomá er vermú.",
        "Quilla, ven pa acá que te cuento lo que me ha pasao.",
        "Noh hemoh comío una pechá de pescaíto frito en la playa.",
    ],
    # ------------------------------------------------------------------
    # ES_CAN  --  Canarian
    # ------------------------------------------------------------------
    DialectCode.ES_CAN: [
        "Vamos a coger la guagua para ir al centro.",
        "¿Ustedes han visto la película que echan esta noche?",
        "Me he comprado un ordenador nuevo, es brutal.",
        "Chacho, no me gusta nada tener que madrugar mañana.",
        "Quedamos a las ocho en la plaza, ¿verdad?",
        "El piso que hemos visto en Las Palmas está genial.",
        "¿Ustedes ya han cenado o quieren pedir unas papas?",
        "Hoy he ido al médico y me ha dicho que estoy bien.",
        "Chacha, ¿tienes gafas de sol? Las he olvidado en casa.",
        "Mijo, qué frío hace, vamos a meternos en el bar.",
        "He aparcado el carro junto a la fuente de la plaza.",
        "Ustedes saben que aquí siempre llueve en noviembre.",
        "Le he llamado por el móvil pero no me ha cogido.",
        "Este fin de semana nos vamos al pueblo a comer gofio.",
        "Me gusta mucho este sitio para tomar un barraquito.",
        "Aquí las papas arrugadas con mojo están pa morirse.",
        "Chacho, ¿tú sabes dónde queda la parada de la guagua?",
    ],
    # ------------------------------------------------------------------
    # ES_RIO  --  Rioplatense (Argentina / Uruguay)
    # ------------------------------------------------------------------
    DialectCode.ES_RIO: [
        "Vamos a tomar el colectivo para ir al centro.",
        "¿Viste la película que dan esta noche en la tele?",
        "Me compré una computadora nueva, está re genial.",
        "Che, no me copa nada tener que madrugar mañana.",
        "Nos vemos a las ocho en la plaza, ¿dale?",
        "El departamento que vimos en Palermo está bárbaro.",
        "¿Vos ya cenaste o querés pedir unas birras?",
        "Hoy fui al médico y me dijo que estoy bien.",
        "Mirá, ¿tenés anteojos de sol? Me los olvidé en casa.",
        "Boludo, qué frío que hace, vamos a meternos en el bar.",
        "Estacioné el auto al lado de la fuente de la plaza.",
        "Vos sabés que acá siempre llueve en noviembre.",
        "Lo llamé por el celular pero no me atendió.",
        "Este finde nos vamos al campo de mis abuelos.",
        "Me re copa este lugar para tomar un fernet.",
        "Che, ¿vos sabés dónde para el bondi acá?",
        "Estuve laburando todo el día y estoy re cansado.",
        "¿Querés que pidamos unas empanadas y un fernet con coca?",
    ],
    # ------------------------------------------------------------------
    # ES_MEX  --  Mexican
    # ------------------------------------------------------------------
    DialectCode.ES_MEX: [
        "Vamos a tomar el camión para ir al centro.",
        "¿Ya vieron la película que pasan esta noche en la tele?",
        "Me compré una computadora nueva, está bien padrísima.",
        "Güey, no me late nada tener que madrugar mañana.",
        "Nos vemos a las ocho en la plaza, ¿va?",
        "El departamento que vimos en la Condesa está muy chido.",
        "¿Tú ya cenaste o quieres pedir unas chelas?",
        "Hoy fui al doctor y me dijo que estoy bien.",
        "Oye, ¿tienes lentes de sol? Se me olvidaron en la casa.",
        "No manches, qué frío hace, vamos a meternos a la cantina.",
        "Estacioné el carro junto a la fuente de la plaza.",
        "Tú sabes que aquí siempre llueve en noviembre.",
        "Le marqué al celular pero no me contestó.",
        "Este fin de semana nos vamos al rancho de mis abuelos.",
        "Me late mucho este lugar para echarnos unos taquitos.",
        "Güey, ¿tú sabes dónde para el camión aquí?",
        "Estuve chambeando todo el día y ando bien cansado.",
        "¿Quieres que pidamos unos tacos y unas micheladas?",
        "Órale, qué padre que viniste a la fiesta ahorita.",
    ],
    # ------------------------------------------------------------------
    # ES_CAR  --  Caribbean
    # ------------------------------------------------------------------
    DialectCode.ES_CAR: [
        "Vamos a coger la guagua pa' ir al centro.",
        "¿Ustedes vieron la película que dan esta noche?",
        "Me compré una computadora nueva, está chévere.",
        "Mijo, no me gusta na' tener que madrugar mañana.",
        "Nos vemos a las ocho en la plaza, ¿oíste?",
        "El apartamento que vimos en el Vedado está brutal.",
        "¿Tú ya cenaste o quieres pedir unas cervezas?",
        "Hoy fui al médico y me dijo que estoy bien.",
        "Oye, ¿tú tienes gafas de sol? Se me olvidaron en casa.",
        "Asere, qué frío hace, vamos a meternos en el bar.",
        "Estacioné el carro junto a la fuente de la plaza.",
        "Tú sabes que aquí siempre llueve en noviembre.",
        "Lo llamé por el celular pero no me contestó.",
        "Este fin de semana nos vamos pa' la playa.",
        "Me encanta este sitio pa' comer una vaina chévere.",
        "Mira, pana, ¿tú sabes dónde para la guagua aquí?",
        "Estuve trabajando to' el día y estoy agotao.",
        "¿Qué tú quieres que pidamos pa' cenar esta noche?",
    ],
    # ------------------------------------------------------------------
    # ES_CHI  --  Chilean
    # ------------------------------------------------------------------
    DialectCode.ES_CHI: [
        "Vamos a tomar la micro para ir al centro.",
        "¿Cacharon la película que dan esta noche en la tele?",
        "Me compré un computador nuevo, está la raja.",
        "Hueón, no me tinca nada tener que madrugar mañana.",
        "Nos juntamos a las ocho en la plaza, ¿cachai?",
        "El depa que vimos en Providencia está bacán.",
        "¿Tú ya cenaste o querís pedir unas chelas?",
        "Hoy fui al médico y me dijo que estoy bien, po.",
        "Oye, ¿tenís lentes de sol? Se me quedaron en la casa.",
        "Hueón, qué frío que hace, vamos a meternos al bar.",
        "Estacioné el auto al lado de la fuente de la plaza.",
        "Tú sabís que acá siempre llueve en noviembre.",
        "Lo llamé al celu pero no me contestó.",
        "Este finde nos vamos al campo, po.",
        "Me tinca caleta este lugar pa' tomarnos unos copetes.",
        "Oye, ¿sabís dónde para la micro acá?",
        "Estuve en la pega todo el día y estoy cagao de cansancio.",
        "¿Querís que pidamos unas empanadas y unos copetes?",
        "Ya po, dale no más, vamos al carrete altiro.",
    ],
    # ------------------------------------------------------------------
    # ES_AND_BO  --  Andean (Peru / Bolivia / Ecuador highlands)
    # ------------------------------------------------------------------
    DialectCode.ES_AND_BO: [
        "Vamos a tomar el bus para ir al centro de la ciudad.",
        "¿Ya han visto la película que pasan esta noche?",
        "Me he comprado una computadora nueva, es bien bonita.",
        "Oye, no me gusta nada tener que madrugar mañana.",
        "Nos vemos a las ocho en la plaza, ¿ya?",
        "El departamento que hemos visto en Miraflores está lindo.",
        "¿Ya cenaron o quieren pedir algo de comer, pues?",
        "Hoy he ido al médico y me ha dicho que estoy bien.",
        "Oiga, ¿tiene lentes de sol? Me he olvidado los míos.",
        "Ay, qué frío hace, vamos a meternos al restaurante.",
        "He estacionado el carro junto a la fuente de la plaza.",
        "Usted sabe que aquí siempre llueve en noviembre.",
        "Le he llamado al celular pero no me ha contestado.",
        "Este fin de semana nos vamos al pueblo, pues.",
        "Me gusta mucho este sitio para comer un ceviche, pe.",
        "Oye, ¿sabes dónde para el bus aquí, pues?",
        "He estado trabajando todo el día y estoy bien cansado.",
        "Pase nomás, caserito, sírvase nomás una sopita caliente.",
        "Su casa de mi mamá queda cerquita del mercado, pues.",
    ],
}


def get_fixtures() -> dict[DialectCode, list[DialectSample]]:
    """Return hand-crafted fixture sentences as :class:`DialectSample` lists.

    Each sample has ``source_id="synthetic_fixture"`` and ``confidence=1.0``.
    """
    result: dict[DialectCode, list[DialectSample]] = {}
    for code, texts in _FIXTURE_TEXTS.items():
        result[code] = [
            DialectSample(
                text=t,
                dialect_code=code,
                source_id="synthetic_fixture",
                confidence=1.0,
                metadata={"fixture_index": i},
            )
            for i, t in enumerate(texts)
        ]
    return result


def get_dialect_features() -> dict[DialectCode, dict[FeatureCategory, list[str]]]:
    """Return a copy of the dialect features dictionary."""
    return {k: dict(v) for k, v in DIALECT_FEATURES.items()}
