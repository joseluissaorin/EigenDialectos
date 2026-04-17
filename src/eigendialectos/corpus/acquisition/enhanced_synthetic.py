"""Enhanced synthetic dialect sample generator.

Massively expanded version of the base :mod:`eigendialectos.corpus.synthetic.generator`
with 300+ base sentences across 14 thematic domains, 80+ lexical entries per major
dialect, expanded morphological/phonological rules, and combinatorial generation
that produces 2000-4000 unique samples per dialect variety.

The generator reuses :class:`DialectTemplate` and :class:`TransformationRule` from
``eigendialectos.corpus.synthetic.templates`` and creates *enhanced* template
overlays that are merged with the originals.
"""

from __future__ import annotations

import random
import re
from copy import deepcopy
from typing import Optional

from eigendialectos.constants import DialectCode
from eigendialectos.types import DialectSample

from eigendialectos.corpus.synthetic.templates import (
    DIALECT_TEMPLATES,
    DialectTemplate,
    TransformationRule,
)


# ======================================================================
# ENHANCED BASE SENTENCES — 300+ neutral Peninsular Spanish
# ======================================================================

ENHANCED_BASE_SENTENCES: list[str] = [
    # ==================================================================
    # DAILY LIFE / HOUSEHOLD (30)
    # ==================================================================
    "Hoy he cocinado una tortilla de patatas para toda la familia.",
    "Tengo que fregar los platos antes de que llegue mi madre.",
    "¿Has puesto la lavadora? La ropa lleva tres días sin lavar.",
    "Mi abuela siempre dice que hay que barrer antes de fregar.",
    "Vamos a hacer la compra que no queda nada en la nevera.",
    "He perdido las llaves de casa y no puedo entrar.",
    "¿Quién ha dejado la luz del baño encendida toda la noche?",
    "Mi hermana se ha mudado con su novio a un piso más grande.",
    "Necesitamos cambiar el colchón porque este está muy viejo.",
    "Se ha roto la calefacción y hace un frío tremendo en casa.",
    "¿Puedes ir a comprar leche y pan al supermercado?",
    "El fontanero viene mañana a arreglar el grifo de la cocina.",
    "He recogido a los niños del colegio y los he llevado al parque.",
    "Pon la mesa que ya está lista la cena.",
    "Mi padre se ha quedado dormido viendo la televisión otra vez.",
    "¿Has sacado la basura? Mañana pasan a recogerla temprano.",
    "Se nos ha estropeado la lavadora y no tenemos dinero para otra.",
    "Hay que llamar al electricista porque se va la luz cada dos por tres.",
    "Mi madre nos ha preparado un guiso de lentejas que está buenísimo.",
    "¿Le has dado de comer al perro o se te ha olvidado?",
    "Hemos pintado el salón de blanco y ha quedado muy bonito.",
    "Tengo que planchar las camisas para toda la semana.",
    "Mi vecina me ha pedido que le riegue las plantas mientras está fuera.",
    "¿Dónde has metido el mando de la televisión? No lo encuentro.",
    "He puesto una lavadora de ropa oscura y otra de ropa blanca.",
    "Mis padres van a venir a cenar el domingo con nosotros.",
    "Se ha inundado el cuarto de baño por una tubería rota.",
    "Voy a preparar café, ¿quieres una taza?",
    "Los niños han dejado todo tirado por el suelo del salón.",
    "¿Has regado las plantas de la terraza? Están secas.",

    # ==================================================================
    # WORK / PROFESSIONAL (30)
    # ==================================================================
    "He llegado tarde al trabajo porque había mucho tráfico.",
    "Mi jefe me ha dicho que tengo que quedarme a hacer horas extra.",
    "¿Has terminado el informe que hay que entregar mañana?",
    "Tenemos una reunión a las diez con el director del departamento.",
    "Me han subido el sueldo después de tres años sin aumento.",
    "Estoy buscando trabajo porque me despidieron la semana pasada.",
    "Mi compañera de trabajo me cae muy bien, siempre me ayuda.",
    "¿Puedes enviarme ese documento por correo electrónico?",
    "He tenido una entrevista de trabajo y creo que me ha ido bien.",
    "El jefe está de mal humor hoy, mejor no le digas nada.",
    "Necesito pedir unos días libres para resolver un asunto personal.",
    "¿Sabes si van a contratar a alguien nuevo para el puesto?",
    "Me paso el día entero sentado delante del ordenador.",
    "He cogido el metro para llegar antes a la oficina.",
    "Mi contrato se acaba el mes que viene y no sé si me renuevan.",
    "¿Tú trabajas los fines de semana o solo de lunes a viernes?",
    "La empresa va a cerrar una de las oficinas del centro.",
    "He quedado con un cliente para comer y hablar del proyecto.",
    "Estoy agotado, llevo tres semanas sin un día libre.",
    "¿Te han pagado ya la nómina de este mes?",
    "Mi hermano ha montado un negocio y le está yendo muy bien.",
    "Tenemos que presentar el presupuesto antes del viernes.",
    "El ordenador se ha colgado otra vez en mitad de la presentación.",
    "He pedido un traslado a la oficina de Barcelona.",
    "¿Quieres que te lleve en coche al trabajo mañana?",
    "Me han ofrecido un puesto en otra ciudad pero no sé si aceptar.",
    "La reunión de esta tarde se ha cancelado, podemos irnos antes.",
    "Tengo mucho trabajo acumulado y no doy abasto.",
    "Mi jefa siempre nos invita a desayunar los viernes.",
    "¿Has fichado al entrar o se te ha olvidado otra vez?",

    # ==================================================================
    # SOCIAL / FRIENDSHIP (30)
    # ==================================================================
    "Vamos a salir esta noche a tomar unas cervezas con los amigos.",
    "¿Has hablado con María? Hace meses que no sé nada de ella.",
    "Mi mejor amigo se casa el mes que viene y yo soy el padrino.",
    "Hemos organizado una fiesta sorpresa para el cumpleaños de Luis.",
    "¿Quieres venir con nosotros al concierto del sábado?",
    "Me he peleado con mi amigo y llevamos dos semanas sin hablarnos.",
    "Quedamos en el bar de siempre a las nueve, ¿vale?",
    "Mi amiga me ha contado un chisme increíble sobre el vecino.",
    "¿Conoces a ese chico? Me han dicho que es muy simpático.",
    "Vamos a hacer una barbacoa en casa de Pedro este fin de semana.",
    "He invitado a todos mis amigos a cenar en casa.",
    "¿Sabes que Laura y Miguel se han separado después de diez años?",
    "No me apetece salir hoy, prefiero quedarme en casa viendo una serie.",
    "Mi compañero de piso es muy desordenado y siempre estamos discutiendo.",
    "¿Te apetece que hagamos un plan tranquilo este domingo?",
    "He quedado con una chica que conocí en una aplicación del móvil.",
    "Mis amigos me han gastado una broma pesada en el trabajo.",
    "¿Vosotros vais a ir a la boda de Ana y Carlos?",
    "Llámame cuando salgas del trabajo y hacemos algo juntos.",
    "Mi grupo de amigos del instituto se reúne una vez al año.",
    "No puedo ir a tu fiesta, tengo otro compromiso esa noche.",
    "¿Por qué no invitamos también a los nuevos compañeros?",
    "Hemos estado toda la noche hablando y riéndonos en el bar.",
    "Mi prima me ha presentado a un amigo suyo que es muy majo.",
    "¿Te acuerdas de cuando íbamos juntos al campamento de verano?",
    "Voy a pasar la tarde con mi abuela que está sola en casa.",
    "Nos hemos hecho un grupo de chat para organizar la cena.",
    "¿Sabes dónde se ha metido Juan? No contesta al teléfono.",
    "He traído unas cervezas y unas patatas para ver el partido.",
    "¿Vamos a tomar un café antes de que cierre la cafetería?",

    # ==================================================================
    # FOOD / RESTAURANTS (30)
    # ==================================================================
    "¿Habéis probado el restaurante nuevo que han abierto en la esquina?",
    "Vamos a pedir una paella para compartir entre todos.",
    "Me encanta la comida de mi abuela, nadie cocina como ella.",
    "¿Tienes la receta del gazpacho que hiciste el otro día?",
    "He ido al mercado y he comprado fruta fresca y verduras.",
    "Este restaurante es carísimo pero la comida merece la pena.",
    "¿Quieres que pida también un postre o ya estamos llenos?",
    "Mi madre hace unas croquetas que están para chuparse los dedos.",
    "Vamos a comer unas tapas en el bar de la plaza.",
    "He aprendido a hacer pan casero durante el confinamiento.",
    "¿Puedes pasarme la sal? Esta sopa está un poco sosa.",
    "El camarero nos ha recomendado el plato del día y estaba muy bueno.",
    "Hoy no me apetece cocinar, vamos a pedir comida a domicilio.",
    "¿Tú comes carne o eres vegetariano? Para saber qué preparo.",
    "He reservado mesa en un restaurante italiano para esta noche.",
    "Mi hermano es cocinero y trabaja en un restaurante del centro.",
    "¿Has desayunado ya o quieres que te prepare algo?",
    "En este bar ponen unas raciones enormes y muy baratas.",
    "Necesitamos comprar aceite de oliva, se nos ha acabado.",
    "¿Vosotros preferís comer aquí o nos llevamos la comida?",
    "La tarta que ha traído Pedro estaba buenísima.",
    "He preparado una ensalada y un filete para cenar rápido.",
    "¿Sabes cocinar arroz con pollo? Quiero aprender la receta.",
    "El café de esta cafetería es el mejor del barrio.",
    "Nos hemos tomado unas cañas con unas aceitunas de aperitivo.",
    "¿Quieres probar un trozo de esta empanada? La he hecho yo.",
    "Mi vecina me ha regalado unos tomates de su huerto.",
    "Vamos a comprar pescado fresco al puerto esta mañana.",
    "He comido demasiado en la comida y ahora me duele el estómago.",
    "¿Tienes algo de picar? Tengo hambre y aún falta para la cena.",

    # ==================================================================
    # TRAVEL / TRANSPORT (25)
    # ==================================================================
    "¿Sabes a qué hora sale el próximo autobús para Madrid?",
    "He reservado un hotel en el centro para tres noches.",
    "Vamos a coger un taxi porque ya no hay metro a estas horas.",
    "El avión lleva dos horas de retraso y no nos dicen nada.",
    "¿Tienes el billete de tren o hay que sacarlo en la estación?",
    "Me he perdido en esta ciudad porque las calles son un laberinto.",
    "¿Puedes llevarme al aeropuerto mañana a primera hora?",
    "Hemos alquilado un coche para recorrer la costa durante el verano.",
    "El tráfico está horrible a esta hora, mejor vamos en metro.",
    "¿Vosotros habéis viajado alguna vez al extranjero?",
    "He perdido la maleta en el aeropuerto y nadie me da una solución.",
    "Necesito renovar el pasaporte antes de irme de viaje.",
    "¿Cuánto cuesta un billete de ida y vuelta a Barcelona?",
    "Mi hermana vive en otro país y solo la veo una vez al año.",
    "Vamos a ir de vacaciones a la playa si encontramos algo barato.",
    "He cogido un avión por primera vez y estoy un poco nervioso.",
    "¿Sabes si hay aparcamiento cerca del centro comercial?",
    "El tren de las seis siempre va lleno y no encuentras asiento.",
    "Hemos recorrido toda la isla en tres días y ha sido increíble.",
    "¿Dónde puedo coger un taxi por aquí? No veo ninguno.",
    "Mi padre me ha enseñado a conducir este verano.",
    "He tenido un pinchazo en la carretera y no tengo rueda de repuesto.",
    "¿Vienes a recogerme a la estación cuando llegue mi tren?",
    "Nos hemos equivocado de salida en la autopista y hemos dado una vuelta enorme.",
    "El autobús pasa cada quince minutos por esta parada.",

    # ==================================================================
    # HEALTH / BODY (20)
    # ==================================================================
    "Me duele mucho la cabeza y creo que me está subiendo la fiebre.",
    "He ido al dentista y me ha dicho que tengo dos caries.",
    "¿Tienes alguna pastilla para el dolor de estómago?",
    "Mi abuelo se ha caído en la calle y le han llevado al hospital.",
    "Necesito empezar a hacer ejercicio porque estoy muy flojo.",
    "He dormido fatal esta noche y hoy no puedo ni con mi alma.",
    "¿Has pedido cita con el médico? Llevas una semana con esa tos.",
    "Me he torcido el tobillo jugando al fútbol y me duele mucho.",
    "El médico me ha mandado una analítica de sangre completa.",
    "Tengo alergia al polen y en primavera lo paso muy mal.",
    "¿Puedes acompañarme al hospital? No me encuentro bien.",
    "Mi madre está a dieta y no quiere comer dulces.",
    "He empezado a correr por las mañanas para ponerme en forma.",
    "¿Sabes si la farmacia de guardia está abierta a estas horas?",
    "Me he quemado con el aceite mientras estaba cocinando.",
    "El niño tiene fiebre y no quiere comer nada desde ayer.",
    "He dejado de fumar hace un mes y me siento mucho mejor.",
    "¿Tú tomas alguna medicina todos los días?",
    "Mi hermano se ha roto el brazo esquiando y lleva un yeso enorme.",
    "Necesito ir al oculista porque veo borroso con el ojo derecho.",

    # ==================================================================
    # TECHNOLOGY (20)
    # ==================================================================
    "Se me ha roto la pantalla del móvil y arreglarlo cuesta un dineral.",
    "¿Puedes enviarme las fotos por el grupo de chat?",
    "He comprado un ordenador portátil nuevo para trabajar desde casa.",
    "No me funciona el internet y llevo una hora esperando al técnico.",
    "¿Sabes cómo se descarga esa aplicación en el teléfono?",
    "Mi hijo se pasa el día entero pegado al móvil y no estudia.",
    "He perdido todas las fotos del viaje porque no hice copia de seguridad.",
    "¿Te has creado una cuenta en la nueva red social que ha salido?",
    "El ordenador va muy lento, creo que tiene un virus.",
    "Necesito cargar el móvil, ¿tienes un cargador por ahí?",
    "He subido las fotos de la cena a las redes sociales.",
    "¿Puedes llamarme por videollamada cuando llegues a casa?",
    "Se me ha olvidado la contraseña del correo electrónico.",
    "Mi abuelo no sabe usar el móvil y me pide ayuda todo el rato.",
    "He cambiado de compañía de teléfono porque la otra era muy cara.",
    "¿Has visto el vídeo que se ha hecho viral en internet?",
    "Necesito imprimir unos documentos y la impresora no funciona.",
    "Me han hackeado la cuenta y he tenido que cambiar todas las contraseñas.",
    "¿Cuánto pagas al mes por la conexión a internet de tu casa?",
    "He comprado unos auriculares inalámbricos y suenan genial.",

    # ==================================================================
    # SPORTS / ENTERTAINMENT (20)
    # ==================================================================
    "¿Has visto el partido de anoche? Fue increíble el gol del final.",
    "Vamos al cine esta tarde, echan una película que pinta muy bien.",
    "Mi equipo ha ganado la liga y estamos todos contentísimos.",
    "¿Quieres que veamos juntos la serie nueva que han estrenado?",
    "He empezado a ir al gimnasio tres veces por semana.",
    "El concierto de ayer fue lo mejor que he visto en mi vida.",
    "¿Juegas al fútbol o prefieres otro deporte?",
    "He sacado entradas para el teatro y nos han costado un ojo de la cara.",
    "Mi hermana es fanática de un grupo de música y tiene todos sus discos.",
    "¿Habéis visto el documental ese sobre la naturaleza? Es muy bueno.",
    "Vamos a jugar un partido de baloncesto en el polideportivo.",
    "He apostado con mi amigo que mi equipo va a ganar el clásico.",
    "¿Tienes la televisión puesta? Pon el canal de deportes.",
    "Mi hijo quiere apuntarse a clases de natación este verano.",
    "El árbitro ha pitado un penalti que no era y estoy enfadado.",
    "Hemos ido a ver un musical al teatro y ha sido espectacular.",
    "¿Sabes a qué hora empieza el programa de esta noche?",
    "Me he comprado una entrada para el festival de música del verano.",
    "Los jugadores del equipo están fatal este año, no ganan ni un partido.",
    "¿Quieres que vayamos a dar un paseo en bicicleta por el parque?",

    # ==================================================================
    # SHOPPING / MONEY (20)
    # ==================================================================
    "He ido de compras y me he gastado todo el dinero en ropa.",
    "¿Cuánto cuesta esta camiseta? Me parece que está muy cara.",
    "Vamos al centro comercial a ver si encontramos algo de oferta.",
    "Me he comprado unas zapatillas nuevas que estaban rebajadas.",
    "¿Tienes cambio de veinte? Necesito pagar el aparcamiento.",
    "Este mes no llego a fin de mes, he gastado demasiado.",
    "He comparado precios en varias tiendas y esta es la más barata.",
    "¿Puedes prestarme algo de dinero hasta que me paguen?",
    "Mi mujer se ha comprado un bolso que le ha costado una fortuna.",
    "Vamos a la tienda de la esquina a comprar unas cosas que faltan.",
    "He devuelto el jersey porque la talla no era la correcta.",
    "¿Has mirado cuánto cuesta alquilar un coche para el fin de semana?",
    "Los precios del supermercado han subido muchísimo este año.",
    "Necesito sacar dinero del cajero antes de ir al mercado.",
    "¿Aceptan tarjeta en esa tienda o solo pagan en efectivo?",
    "He pagado a plazos el sofá nuevo porque no tenía para pagarlo entero.",
    "Mi padre nunca compra nada sin antes mirar el precio tres veces.",
    "¿Cuánto te ha costado el móvil nuevo? Yo quiero uno igual.",
    "He encontrado una ganga en internet: un abrigo a mitad de precio.",
    "Vamos a ver los precios de los pisos en el barrio, por curiosidad.",

    # ==================================================================
    # WEATHER / NATURE (15)
    # ==================================================================
    "Hoy hace un calor insoportable, no se puede ni salir a la calle.",
    "Lleva lloviendo toda la semana y estoy harto de llevar paraguas.",
    "¿Has visto qué bonito está el campo con las flores de primavera?",
    "Este invierno ha nevado mucho más de lo normal en las montañas.",
    "El cielo está nublado, creo que va a llover esta tarde.",
    "Me encanta pasear por la playa cuando hace bueno.",
    "¿Tú crees que va a hacer buen tiempo para el fin de semana?",
    "El río ha crecido mucho con las lluvias y se ha inundado el pueblo.",
    "En verano aquí se pasa mucho calor, sobre todo por las tardes.",
    "He ido a la montaña y las vistas desde arriba eran espectaculares.",
    "¿Sabes si mañana va a hacer frío? No sé qué ropa ponerme.",
    "Las cosechas de este año han sido muy buenas por las lluvias.",
    "Me gusta mucho el otoño porque los árboles cambian de color.",
    "El viento de anoche ha tirado un árbol en la carretera.",
    "Este pueblo está rodeado de olivos y viñedos, es precioso.",

    # ==================================================================
    # EDUCATION (15)
    # ==================================================================
    "Mi hijo ha aprobado todas las asignaturas y estamos orgullosos.",
    "¿Has estudiado para el examen de mañana o todavía no?",
    "El profesor nos ha mandado un montón de deberes para el lunes.",
    "Voy a la universidad todos los días en autobús.",
    "He sacado una beca para estudiar un máster el año que viene.",
    "¿Sabes si hay clase mañana o es festivo?",
    "Mi hija quiere estudiar medicina cuando termine el instituto.",
    "El examen de matemáticas ha sido muy difícil y creo que lo he suspendido.",
    "He aprobado la oposición después de tres años preparándola.",
    "¿Puedes explicarme este problema? No lo entiendo.",
    "Mis compañeros de clase y yo nos juntamos a estudiar en la biblioteca.",
    "El rector de la universidad ha anunciado cambios en el plan de estudios.",
    "Tengo que entregar el trabajo de fin de carrera antes de junio.",
    "¿Vosotros habéis elegido ya las optativas del próximo curso?",
    "Mi profesora dice que tengo que mejorar la ortografía.",

    # ==================================================================
    # EMOTIONS / OPINIONS (15)
    # ==================================================================
    "Estoy muy contento porque me han dado la noticia que esperaba.",
    "No puedo creer lo que me has dicho, estoy en shock total.",
    "Me parece fatal que hayan subido otra vez los precios.",
    "Estoy muy enfadado con mi hermano por lo que me ha hecho.",
    "¿No te da rabia que siempre nos traten como si fuéramos tontos?",
    "Me pone muy nervioso esperar tanto tiempo en la cola.",
    "Creo que la situación va a mejorar pronto, hay que tener paciencia.",
    "Estoy harto de que nadie me escuche cuando hablo.",
    "Me emociona mucho ver a toda la familia junta en Navidad.",
    "¿Tú qué opinas de lo que ha pasado con el tema del ayuntamiento?",
    "Es una vergüenza que no haya agua caliente en el hospital.",
    "Me da mucha pena ver cómo ha cambiado el barrio para peor.",
    "Estoy ilusionado con el viaje que vamos a hacer en verano.",
    "No me parece justo que unos trabajen más que otros por el mismo sueldo.",
    "Me alegro mucho de que todo te haya salido bien al final.",

    # ==================================================================
    # NEWS / POLITICS (15)
    # ==================================================================
    "¿Has visto las noticias? Han convocado elecciones anticipadas.",
    "La economía del país está fatal y la gente no llega a fin de mes.",
    "Ha habido una manifestación enorme en el centro de la ciudad.",
    "El alcalde ha prometido arreglar las calles del barrio.",
    "¿Tú crees que van a bajar los impuestos después de las elecciones?",
    "Han aprobado una ley nueva que afecta a los trabajadores autónomos.",
    "Mi padre discute con todo el mundo cuando se habla de política.",
    "Ha subido otra vez el precio de la gasolina y es insostenible.",
    "¿Sabes quién ha ganado las elecciones municipales?",
    "La oposición ha presentado una moción de censura contra el gobierno.",
    "Los sindicatos han convocado una huelga general para el jueves.",
    "¿Habéis oído lo que ha dicho el presidente en la rueda de prensa?",
    "La corrupción es un problema que afecta a todos los partidos.",
    "Han cerrado el hospital público del pueblo y ahora hay que ir a la ciudad.",
    "Me parece increíble que todavía no hayamos resuelto el problema del desempleo.",

    # ==================================================================
    # HOUSING / NEIGHBORHOOD (15)
    # ==================================================================
    "Los vecinos de arriba hacen ruido todas las noches y no puedo dormir.",
    "¿Has visto cuánto piden por un piso de alquiler en el centro?",
    "Mi barrio ha cambiado mucho en los últimos diez años.",
    "Nos hemos comprado una casa en las afueras con un jardín grande.",
    "El alquiler me come la mitad del sueldo cada mes.",
    "¿Conoces al vecino nuevo que se ha mudado al tercero?",
    "Hay mucho ruido en esta calle por las obras que están haciendo.",
    "He puesto una denuncia en el ayuntamiento por la suciedad del barrio.",
    "¿Vosotros pagáis mucho de comunidad en vuestro edificio?",
    "Mi casero no quiere arreglar la gotera del techo y estoy harto.",
    "El barrio de al lado tiene mucha más vida nocturna que este.",
    "Han abierto un parque nuevo en la calle de atrás y está genial.",
    "Necesitamos un piso más grande porque viene otro niño en camino.",
    "Los pisos de esta zona están por las nubes, imposible comprar.",
    "Mi vecina me trae siempre comida casera y es un encanto de persona.",
]

# For reference: total sentence count
assert len(ENHANCED_BASE_SENTENCES) == 300, (
    f"Expected 300 base sentences, got {len(ENHANCED_BASE_SENTENCES)}"
)


# ======================================================================
# ENHANCED DIALECT TEMPLATES
# Overlays that extend the base DIALECT_TEMPLATES with extra entries
# ======================================================================

ENHANCED_TEMPLATES: dict[DialectCode, DialectTemplate] = {

    # ==================================================================
    # ES_PEN — Peninsular (baseline/identity)
    # ==================================================================
    DialectCode.ES_PEN: DialectTemplate(
        lexical={},
        morphological=[],
        pragmatic_markers=[
            "vale", "tío", "tía", "mola", "¿no?", "joder", "oye",
            "mira", "venga", "¿sabes?", "hombre", "en plan",
            "flipas", "tronco", "colega", "macho",
        ],
        phonological=[],
    ),

    # ==================================================================
    # ES_RIO — Rioplatense (80+ lexical entries)
    # ==================================================================
    DialectCode.ES_RIO: DialectTemplate(
        lexical={
            # Original entries
            "autobús": "colectivo",
            "ordenador": "computadora",
            "coche": "auto",
            "carro": "auto",
            "gafas": "anteojos",
            "piso": "departamento",
            "apartamento": "departamento",
            "móvil": "celular",
            "teléfono móvil": "celular",
            "chico": "pibe",
            "chica": "piba",
            "trabajo": "laburo",
            "trabajar": "laburar",
            "dinero": "guita",
            "cerveza": "birra",
            "genial": "bárbaro",
            # Expanded lunfardo / rioplatense
            "robar": "afanar",
            "colectivo urbano": "bondi",
            "discoteca": "boliche",
            "policía": "cana",
            "discurso": "chamuyo",
            "pereza": "fiaca",
            "pagar": "garpar",
            "excelente": "groso",
            "peso": "mango",
            "comer": "morfar",
            "ropa": "pilcha",
            "desastre": "quilombo",
            "falso": "trucho",
            "exagerar": "zarpar",
            "suerte": "culo",
            "cara": "jeta",
            "tonto": "boludo",
            "cabeza": "balero",
            "borracho": "mamado",
            "mujer": "mina",
            "hombre": "chabón",
            "niño": "pibito",
            "problema": "bardo",
            "amigo": "cumpa",
            "miedo": "cagazo",
            "enojado": "caliente",
            "loco": "chapita",
            "mentira": "chamuyo",
            "pelea": "quilombo",
            "comida": "morfi",
            "apurado": "a los pedos",
            "asado": "asado",
            "mate": "mate",
            "fiesta": "joda",
            "hermoso": "re lindo",
            "comprar": "rajar",
            "hambre": "garrón",
            "calor": "calor de la san puta",
            "cansado": "hecho bolsa",
            "zapatos": "zapas",
            "bonito": "re lindo",
            "feo": "horrible",
            "llorar": "moquear",
            "patatas": "papas",
            "tortilla": "tortilla",
            "aceite": "aceite",
            "nevera": "heladera",
            "supermercado": "súper",
            "alquiler": "alquiler",
            "barrio": "barrio",
            "esquina": "esquina",
            "acera": "vereda",
            "manzana": "cuadra",
            "piscina": "pileta",
            "frigorífico": "heladera",
            "jersey": "buzo",
            "zapatillas": "zapatillas",
            "camiseta": "remera",
            "chaqueta": "campera",
            "pantalones": "pantalones",
            "falda": "pollera",
            "calcetines": "medias",
            "ascensor": "ascensor",
            "conducir": "manejar",
            "aparcar": "estacionar",
            "enfadado": "caliente",
            "camarero": "mozo",
            "zumo": "jugo",
            "melocotón": "durazno",
            "plátano": "banana",
            "fresa": "frutilla",
            "maíz": "choclo",
            "judías": "porotos",
            "mantequilla": "manteca",
            "grifo": "canilla",
        },
        morphological=[
            # Pronoun
            TransformationRule(
                r'\btú\b', 'vos', is_regex=True,
                description="pronombre vos",
            ),
            # ----- Complete voseo indicative present -----
            TransformationRule(
                r'\btienes\b', 'tenés', is_regex=True,
                description="voseo: tienes -> tenés",
            ),
            TransformationRule(
                r'\bquieres\b', 'querés', is_regex=True,
                description="voseo: quieres -> querés",
            ),
            TransformationRule(
                r'\bsabes\b', 'sabés', is_regex=True,
                description="voseo: sabes -> sabés",
            ),
            TransformationRule(
                r'\bpuedes\b', 'podés', is_regex=True,
                description="voseo: puedes -> podés",
            ),
            TransformationRule(
                r'\bvienes\b', 'venís', is_regex=True,
                description="voseo: vienes -> venís",
            ),
            TransformationRule(
                r'\bpiensas\b', 'pensás', is_regex=True,
                description="voseo: piensas -> pensás",
            ),
            TransformationRule(
                r'\bsientes\b', 'sentís', is_regex=True,
                description="voseo: sientes -> sentís",
            ),
            TransformationRule(
                r'\beres\b', 'sos', is_regex=True,
                description="voseo: eres -> sos",
            ),
            TransformationRule(
                r'\bvas\b', 'vas', is_regex=True,
                description="voseo: vas unchanged",
            ),
            TransformationRule(
                r'\bhaces\b', 'hacés', is_regex=True,
                description="voseo: haces -> hacés",
            ),
            TransformationRule(
                r'\bdices\b', 'decís', is_regex=True,
                description="voseo: dices -> decís",
            ),
            TransformationRule(
                r'\bestás\b', 'estás', is_regex=True,
                description="voseo: estás stays same",
            ),
            TransformationRule(
                r'\bcrees\b', 'creés', is_regex=True,
                description="voseo: crees -> creés",
            ),
            TransformationRule(
                r'\bconoces\b', 'conocés', is_regex=True,
                description="voseo: conoces -> conocés",
            ),
            TransformationRule(
                r'\bprefieres\b', 'preferís', is_regex=True,
                description="voseo: prefieres -> preferís",
            ),
            TransformationRule(
                r'\bentiendes\b', 'entendés', is_regex=True,
                description="voseo: entiendes -> entendés",
            ),
            TransformationRule(
                r'\bnecesitas\b', 'necesitás', is_regex=True,
                description="voseo: necesitas -> necesitás",
            ),
            TransformationRule(
                r'\bpagas\b', 'pagás', is_regex=True,
                description="voseo: pagas -> pagás",
            ),
            TransformationRule(
                r'\bmiras\b', 'mirás', is_regex=True,
                description="voseo: miras -> mirás",
            ),
            TransformationRule(
                r'\bjuegas\b', 'jugás', is_regex=True,
                description="voseo: juegas -> jugás",
            ),
            TransformationRule(
                r'\bencuentras\b', 'encontrás', is_regex=True,
                description="voseo: encuentras -> encontrás",
            ),
            TransformationRule(
                r'\brecuerdas\b', 'te acordás', is_regex=True,
                description="voseo: recuerdas -> te acordás",
            ),
            TransformationRule(
                r'\bempiezas\b', 'empezás', is_regex=True,
                description="voseo: empiezas -> empezás",
            ),
            TransformationRule(
                r'\bvuelves\b', 'volvés', is_regex=True,
                description="voseo: vuelves -> volvés",
            ),
            TransformationRule(
                r'\bcomes\b', 'comés', is_regex=True,
                description="voseo: comes -> comés",
            ),
            TransformationRule(
                r'\btomas\b', 'tomás', is_regex=True,
                description="voseo: tomas -> tomás",
            ),
            TransformationRule(
                r'\bpones\b', 'ponés', is_regex=True,
                description="voseo: pones -> ponés",
            ),
            # ----- Voseo imperatives -----
            TransformationRule(
                r'\bmira\b', 'mirá', is_regex=True,
                description="imperativo voseante: mira -> mirá",
            ),
            TransformationRule(
                r'\bven\b', 'vení', is_regex=True,
                description="imperativo voseante: ven -> vení",
            ),
            TransformationRule(
                r'\bdi\b', 'decí', is_regex=True,
                description="imperativo voseante: di -> decí",
            ),
            TransformationRule(
                r'\bpon\b', 'poné', is_regex=True,
                description="imperativo voseante: pon -> poné",
            ),
            TransformationRule(
                r'\bsale?\b', 'salí', is_regex=True,
                description="imperativo voseante: sal -> salí",
            ),
            TransformationRule(
                r'\btoma\b', 'tomá', is_regex=True,
                description="imperativo voseante: toma -> tomá",
            ),
            TransformationRule(
                r'\bllama\b', 'llamá', is_regex=True,
                description="imperativo voseante: llama -> llamá",
            ),
            TransformationRule(
                r'\bespera\b', 'esperá', is_regex=True,
                description="imperativo voseante: espera -> esperá",
            ),
            TransformationRule(
                r'\bescucha\b', 'escuchá', is_regex=True,
                description="imperativo voseante: escucha -> escuchá",
            ),
            # ----- Pretérito simple preference -----
            TransformationRule(
                r'\bhe ido\b', 'fui', is_regex=True,
                description="perfecto simple: he ido -> fui",
            ),
            TransformationRule(
                r'\bhe comido\b', 'comí', is_regex=True,
                description="perfecto simple: he comido -> comí",
            ),
            TransformationRule(
                r'\bhe visto\b', 'vi', is_regex=True,
                description="perfecto simple: he visto -> vi",
            ),
            TransformationRule(
                r'\bhe comprado\b', 'compré', is_regex=True,
                description="perfecto simple: he comprado -> compré",
            ),
            TransformationRule(
                r'\bhe tenido\b', 'tuve', is_regex=True,
                description="perfecto simple: he tenido -> tuve",
            ),
            TransformationRule(
                r'\bhe hecho\b', 'hice', is_regex=True,
                description="perfecto simple: he hecho -> hice",
            ),
            TransformationRule(
                r'\bhe puesto\b', 'puse', is_regex=True,
                description="perfecto simple: he puesto -> puse",
            ),
            TransformationRule(
                r'\bhe dicho\b', 'dije', is_regex=True,
                description="perfecto simple: he dicho -> dije",
            ),
            TransformationRule(
                r'\bhe dormido\b', 'dormí', is_regex=True,
                description="perfecto simple: he dormido -> dormí",
            ),
            TransformationRule(
                r'\bhe perdido\b', 'perdí', is_regex=True,
                description="perfecto simple: he perdido -> perdí",
            ),
            TransformationRule(
                r'\bhe encontrado\b', 'encontré', is_regex=True,
                description="perfecto simple: he encontrado -> encontré",
            ),
            TransformationRule(
                r'\bhe sacado\b', 'saqué', is_regex=True,
                description="perfecto simple: he sacado -> saqué",
            ),
            TransformationRule(
                r'\bhe empezado\b', 'empecé', is_regex=True,
                description="perfecto simple: he empezado -> empecé",
            ),
            TransformationRule(
                r'\bhe dejado\b', 'dejé', is_regex=True,
                description="perfecto simple: he dejado -> dejé",
            ),
            TransformationRule(
                r'\bhe estado\b', 'estuve', is_regex=True,
                description="perfecto simple: he estado -> estuve",
            ),
            TransformationRule(
                r'\bhe pagado\b', 'pagué', is_regex=True,
                description="perfecto simple: he pagado -> pagué",
            ),
            TransformationRule(
                r'\bme he comprado\b', 'me compré', is_regex=True,
                description="perfecto simple: me he comprado -> me compré",
            ),
            TransformationRule(
                r'\bme he mudado\b', 'me mudé', is_regex=True,
                description="perfecto simple: me he mudado -> me mudé",
            ),
            TransformationRule(
                r'\bme he torcido\b', 'me torcí', is_regex=True,
                description="perfecto simple: me he torcido -> me torcí",
            ),
            TransformationRule(
                r'\bme he quemado\b', 'me quemé', is_regex=True,
                description="perfecto simple: me he quemado -> me quemé",
            ),
            TransformationRule(
                r'\bme he roto\b', 'me rompí', is_regex=True,
                description="perfecto simple: me he roto -> me rompí",
            ),
            TransformationRule(
                r'\bme he peleado\b', 'me peleé', is_regex=True,
                description="perfecto simple: me he peleado -> me peleé",
            ),
            TransformationRule(
                r'\bse ha roto\b', 'se rompió', is_regex=True,
                description="perfecto simple: se ha roto -> se rompió",
            ),
            TransformationRule(
                r'\bse ha estropeado\b', 'se rompió', is_regex=True,
                description="perfecto simple: se ha estropeado -> se rompió",
            ),
            TransformationRule(
                r'\bha dicho\b', 'dijo', is_regex=True,
                description="perfecto simple: ha dicho -> dijo",
            ),
            TransformationRule(
                r'\bha subido\b', 'subió', is_regex=True,
                description="perfecto simple: ha subido -> subió",
            ),
            TransformationRule(
                r'\bme ha dicho\b', 'me dijo', is_regex=True,
                description="perfecto simple: me ha dicho -> me dijo",
            ),
            TransformationRule(
                r'\bme ha dado\b', 'me dio', is_regex=True,
                description="perfecto simple: me ha dado -> me dio",
            ),
            TransformationRule(
                r'\bme ha llamado\b', 'me llamó', is_regex=True,
                description="perfecto simple: me ha llamado -> me llamó",
            ),
            TransformationRule(
                r'\bme ha contado\b', 'me contó', is_regex=True,
                description="perfecto simple: me ha contado -> me contó",
            ),
            TransformationRule(
                r'\bme ha pedido\b', 'me pidió', is_regex=True,
                description="perfecto simple: me ha pedido -> me pidió",
            ),
            TransformationRule(
                r'\bme ha costado\b', 'me costó', is_regex=True,
                description="perfecto simple: me ha costado -> me costó",
            ),
            TransformationRule(
                r'\bha sido\b', 'fue', is_regex=True,
                description="perfecto simple: ha sido -> fue",
            ),
            TransformationRule(
                r'\bha ganado\b', 'ganó', is_regex=True,
                description="perfecto simple: ha ganado -> ganó",
            ),
            TransformationRule(
                r'\bha pasado\b', 'pasó', is_regex=True,
                description="perfecto simple: ha pasado -> pasó",
            ),
            TransformationRule(
                r'\bhan subido\b', 'subieron', is_regex=True,
                description="perfecto simple: han subido -> subieron",
            ),
            TransformationRule(
                r'\bhan abierto\b', 'abrieron', is_regex=True,
                description="perfecto simple: han abierto -> abrieron",
            ),
            TransformationRule(
                r'\bhan convocado\b', 'convocaron', is_regex=True,
                description="perfecto simple: han convocado -> convocaron",
            ),
            TransformationRule(
                r'\bhemos quedado\b', 'quedamos', is_regex=True,
                description="perfecto simple: hemos quedado -> quedamos",
            ),
            TransformationRule(
                r'\bhemos visto\b', 'vimos', is_regex=True,
                description="perfecto simple: hemos visto -> vimos",
            ),
            TransformationRule(
                r'\bhemos organizado\b', 'organizamos', is_regex=True,
                description="perfecto simple: hemos organizado -> organizamos",
            ),
            TransformationRule(
                r'\bhemos estado\b', 'estuvimos', is_regex=True,
                description="perfecto simple: hemos estado -> estuvimos",
            ),
            # ----- Coger -> agarrar (taboo in Rioplatense) -----
            TransformationRule(
                r'\bcoger\b', 'agarrar', is_regex=True,
                description="coger -> agarrar (taboo in RP)",
            ),
            TransformationRule(
                r'\bcogido\b', 'agarrado', is_regex=True,
                description="cogido -> agarrado",
            ),
            TransformationRule(
                r'\bcogemos\b', 'agarramos', is_regex=True,
                description="cogemos -> agarramos",
            ),
            TransformationRule(
                r'\bcoger\b', 'tomar', is_regex=True,
                description="coger -> tomar (transport)",
            ),
        ],
        pragmatic_markers=[
            "che", "dale", "boludo", "boluda", "¿viste?", "mirá",
            "¿entendés?", "re", "bárbaro", "de una", "¿sabés qué?",
            "mala mía", "¿me seguís?", "al toque", "de taquito",
            "a full", "flasheás", "manso", "está buenísimo",
        ],
        phonological=[
            # Yeísmo rehilado
            TransformationRule(
                r'\byo\b', 'sho', is_regex=True,
                description="yeísmo rehilado: yo -> sho",
            ),
            TransformationRule(
                r'\bya\b', 'sha', is_regex=True,
                description="yeísmo rehilado: ya -> sha",
            ),
        ],
    ),

    # ==================================================================
    # ES_MEX — Mexican (80+ lexical entries)
    # ==================================================================
    DialectCode.ES_MEX: DialectTemplate(
        lexical={
            # Original
            "autobús": "camión",
            "ordenador": "computadora",
            "coche": "carro",
            "gafas": "lentes",
            "piso": "departamento",
            "apartamento": "departamento",
            "móvil": "celular",
            "genial": "padrísimo",
            "trabajo": "chamba",
            "trabajar": "chambear",
            "chico": "chamaco",
            "dinero": "lana",
            "cerveza": "chela",
            # Expanded nahuatlismos and Mexican slang
            "aguacate": "aguacate",
            "patata": "papa",
            "patatas": "papas",
            "maíz": "elote",
            "niño": "escuincle",
            "amigo": "cuate",
            "tonto": "menso",
            "verdad": "neta",
            "bonito": "chido",
            "bueno": "padre",
            "fiesta": "pachanga",
            "borracho": "pedo",
            "enojado": "encabronado",
            "mujer": "vieja",
            "hombre": "vato",
            "guapo": "cuero",
            "comida": "trago",
            "problema": "bronca",
            "policía": "chota",
            "cárcel": "bote",
            "camiseta": "playera",
            "chaqueta": "chamarra",
            "calcetines": "calcetas",
            "acera": "banqueta",
            "manzana": "cuadra",
            "piscina": "alberca",
            "frigorífico": "refri",
            "ascensor": "elevador",
            "zumo": "jugo",
            "melocotón": "durazno",
            "plátano": "plátano",
            "fresa": "fresa",
            "judías": "frijoles",
            "mantequilla": "mantequilla",
            "conducir": "manejar",
            "aparcar": "estacionar",
            "enfadado": "enojado",
            "camarero": "mesero",
            "alquiler": "renta",
            "grifo": "llave",
            "nevera": "refri",
            "jersey": "suéter",
            "zapatillas": "tenis",
            "tortilla": "tortilla",
            "autobús escolar": "camión escolar",
            "perezoso": "huevón",
            "difícil": "canijo",
            "excelente": "chingón",
            "increíble": "con madre",
            "loco": "loco",
            "miedo": "caguengue",
            "rápido": "volado",
            "apurado": "apurado",
            "hambre": "hambre",
            "grande": "grandote",
            "pequeño": "chiquito",
            "suerte": "suertudo",
            "perro": "perro",
            "gato": "gato",
            "barrio": "colonia",
            "supermercado": "súper",
            "tienda": "tiendita",
            "esquina": "esquina",
            "bocadillo": "torta",
            "sandwich": "torta",
            "tarta": "pastel",
            "galletas": "galletas",
            "caramelos": "dulces",
            "palomitas": "palomitas",
            "desayuno": "desayuno",
            "falda": "falda",
            "pantalones": "pantalones",
        },
        morphological=[
            # Diminutives
            TransformationRule(
                r'\bahora\b', 'ahorita', is_regex=True,
                description="diminutivo: ahora -> ahorita",
            ),
            TransformationRule(
                r'\bcerca\b', 'cerquita', is_regex=True,
                description="diminutivo: cerca -> cerquita",
            ),
            TransformationRule(
                r'\btodo\b', 'todito', is_regex=True,
                description="diminutivo: todo -> todito",
            ),
            TransformationRule(
                r'\btoda\b', 'todita', is_regex=True,
                description="diminutivo: toda -> todita",
            ),
            TransformationRule(
                r'\bluego\b', 'lueguito', is_regex=True,
                description="diminutivo: luego -> lueguito",
            ),
            TransformationRule(
                r'\btemprano\b', 'tempranito', is_regex=True,
                description="diminutivo: temprano -> tempranito",
            ),
            TransformationRule(
                r'\bpoco\b', 'poquito', is_regex=True,
                description="diminutivo: poco -> poquito",
            ),
            TransformationRule(
                r'\bun momento\b', 'un momentito', is_regex=True,
                description="diminutivo: un momento -> un momentito",
            ),
            TransformationRule(
                r'\brápido\b', 'rapidito', is_regex=True,
                description="diminutivo: rápido -> rapidito",
            ),
            # Le intensivo
            TransformationRule(
                r'\bánda\b', 'ándale', is_regex=True,
                description="le intensivo: ánda -> ándale",
            ),
            # Coger -> tomar (neutral in Mexico but avoided)
            TransformationRule(
                r'\bcoger\b', 'tomar', is_regex=True,
                description="coger -> tomar",
            ),
            TransformationRule(
                r'\bcogemos\b', 'tomamos', is_regex=True,
                description="cogemos -> tomamos",
            ),
            TransformationRule(
                r'\bcogido\b', 'tomado', is_regex=True,
                description="cogido -> tomado",
            ),
            # Pretérito simple preference (like RP)
            TransformationRule(
                r'\bhe ido\b', 'fui', is_regex=True,
                description="perfecto simple: he ido -> fui",
            ),
            TransformationRule(
                r'\bhe visto\b', 'vi', is_regex=True,
                description="perfecto simple: he visto -> vi",
            ),
            TransformationRule(
                r'\bhe comprado\b', 'compré', is_regex=True,
                description="perfecto simple: he comprado -> compré",
            ),
            TransformationRule(
                r'\bhe comido\b', 'comí', is_regex=True,
                description="perfecto simple: he comido -> comí",
            ),
            TransformationRule(
                r'\bhe tenido\b', 'tuve', is_regex=True,
                description="perfecto simple: he tenido -> tuve",
            ),
            TransformationRule(
                r'\bhe hecho\b', 'hice', is_regex=True,
                description="perfecto simple: he hecho -> hice",
            ),
            TransformationRule(
                r'\bhe puesto\b', 'puse', is_regex=True,
                description="perfecto simple: he puesto -> puse",
            ),
            TransformationRule(
                r'\bhe perdido\b', 'perdí', is_regex=True,
                description="perfecto simple: he perdido -> perdí",
            ),
            TransformationRule(
                r'\bhe estado\b', 'estuve', is_regex=True,
                description="perfecto simple: he estado -> estuve",
            ),
            TransformationRule(
                r'\bme he comprado\b', 'me compré', is_regex=True,
                description="perfecto simple: me he comprado -> me compré",
            ),
            TransformationRule(
                r'\bme he mudado\b', 'me mudé', is_regex=True,
                description="perfecto simple: me he mudado -> me mudé",
            ),
            TransformationRule(
                r'\bha dicho\b', 'dijo', is_regex=True,
                description="perfecto simple: ha dicho -> dijo",
            ),
            TransformationRule(
                r'\bme ha dicho\b', 'me dijo', is_regex=True,
                description="perfecto simple: me ha dicho -> me dijo",
            ),
            TransformationRule(
                r'\bha sido\b', 'fue', is_regex=True,
                description="perfecto simple: ha sido -> fue",
            ),
            TransformationRule(
                r'\bhan subido\b', 'subieron', is_regex=True,
                description="perfecto simple: han subido -> subieron",
            ),
            TransformationRule(
                r'\bhemos visto\b', 'vimos', is_regex=True,
                description="perfecto simple: hemos visto -> vimos",
            ),
            TransformationRule(
                r'\bhemos quedado\b', 'quedamos', is_regex=True,
                description="perfecto simple: hemos quedado -> quedamos",
            ),
            # Vosotros -> ustedes
            TransformationRule(
                r'\bvosotros\b', 'ustedes', is_regex=True,
                description="ustedes replaces vosotros",
            ),
            TransformationRule(
                r'\bvosotras\b', 'ustedes', is_regex=True,
                description="ustedes replaces vosotras",
            ),
            TransformationRule(
                r'\bhabéis\b', 'han', is_regex=True,
                description="ustedes conjugation",
            ),
            TransformationRule(
                r'\btenéis\b', 'tienen', is_regex=True,
                description="ustedes conjugation",
            ),
            TransformationRule(
                r'\bqueréis\b', 'quieren', is_regex=True,
                description="ustedes conjugation",
            ),
            TransformationRule(
                r'\bsabéis\b', 'saben', is_regex=True,
                description="ustedes conjugation",
            ),
            TransformationRule(
                r'\bpagáis\b', 'pagan', is_regex=True,
                description="ustedes conjugation",
            ),
            TransformationRule(
                r'\bpreferís\b', 'prefieren', is_regex=True,
                description="ustedes conjugation",
            ),
        ],
        pragmatic_markers=[
            "güey", "wey", "¿va?", "órale", "ándale", "¿mande?",
            "no manches", "chido", "padre", "neta", "simón",
            "nel", "a poco", "¿qué onda?", "¿cómo ves?",
            "no mames", "híjole", "a huevo", "¿qué pedo?",
            "chance", "fierro", "ya estuvo", "sale",
        ],
        phonological=[
            # Seseo
            TransformationRule(
                r'z([aeiou])', r's\1', is_regex=True,
                description="seseo: z -> s ante vocal",
            ),
            TransformationRule(
                r'c([ei])', r's\1', is_regex=True,
                description="seseo: ce/ci -> se/si",
            ),
        ],
    ),

    # ==================================================================
    # ES_CHI — Chilean (80+ lexical entries)
    # ==================================================================
    DialectCode.ES_CHI: DialectTemplate(
        lexical={
            # Original
            "autobús": "micro",
            "ordenador": "computador",
            "coche": "auto",
            "gafas": "lentes",
            "piso": "depa",
            "apartamento": "depa",
            "móvil": "celu",
            "genial": "bacán",
            "inmediatamente": "al tiro",
            "novia": "polola",
            "novio": "pololo",
            "trabajo": "pega",
            "aburrido": "fome",
            "fiesta": "carrete",
            "cerveza": "chela",
            # Expanded chilenismos
            "mucho": "caleta",
            "amigo": "compadre",
            "dinero": "lucas",
            "comida": "comida",
            "niño": "cabro chico",
            "chico": "cabro",
            "chica": "cabra",
            "tonto": "aweonao",
            "borracho": "curao",
            "bueno": "la raja",
            "malo": "penca",
            "bonito": "bonito",
            "feo": "horrible",
            "rápido": "al tiro",
            "problema": "cacho",
            "cosa": "weá",
            "tipo": "loco",
            "mujer": "mina",
            "hombre": "gallo",
            "guapo": "churro",
            "enojado": "choreado",
            "cansado": "reventado",
            "difícil": "complicado",
            "excelente": "la raja",
            "pequeño": "chiquitito",
            "grande": "grandote",
            "hambre": "hambre",
            "frío": "frío",
            "calor": "calor",
            "patata": "papa",
            "patatas": "papas",
            "zumo": "jugo",
            "melocotón": "durazno",
            "plátano": "plátano",
            "fresa": "frutilla",
            "judías": "porotos",
            "mantequilla": "mantequilla",
            "nevera": "refri",
            "frigorífico": "refri",
            "jersey": "chaleco",
            "zapatillas": "zapatillas",
            "camiseta": "polera",
            "chaqueta": "chaqueta",
            "calcetines": "calcetines",
            "conducir": "manejar",
            "aparcar": "estacionar",
            "acera": "vereda",
            "barrio": "barrio",
            "camarero": "garzón",
            "alquiler": "arriendo",
            "grifo": "llave",
            "ascensor": "ascensor",
            "piscina": "piscina",
            "supermercado": "súper",
            "enfadado": "choreado",
            "manzana": "cuadra",
            "bocadillo": "sándwich",
            "tarta": "torta",
            "desayuno": "desayuno",
            "falda": "falda",
            "pantalones": "pantalones",
            "policía": "paco",
            "perezoso": "flojo",
            "loco": "loco",
            "suerte": "cuea",
            "miedo": "cuco",
            "mentira": "mentira",
            "verdad": "verdad",
            "increíble": "la zorra",
        },
        morphological=[
            # ----- Voseo chileno (tú + -ís/-ái) -----
            TransformationRule(
                r'\bsabes\b', 'sabís', is_regex=True,
                description="voseo chileno: sabes -> sabís",
            ),
            TransformationRule(
                r'\bquieres\b', 'querís', is_regex=True,
                description="voseo chileno: quieres -> querís",
            ),
            TransformationRule(
                r'\btienes\b', 'tenís', is_regex=True,
                description="voseo chileno: tienes -> tenís",
            ),
            TransformationRule(
                r'\bpuedes\b', 'podís', is_regex=True,
                description="voseo chileno: puedes -> podís",
            ),
            TransformationRule(
                r'\bvienes\b', 'venís', is_regex=True,
                description="voseo chileno: vienes -> venís",
            ),
            TransformationRule(
                r'\bpiensas\b', 'pensái', is_regex=True,
                description="voseo chileno: piensas -> pensái",
            ),
            TransformationRule(
                r'\bentiendes\b', 'entendís', is_regex=True,
                description="voseo chileno: entiendes -> entendís",
            ),
            TransformationRule(
                r'\bhaces\b', 'hacís', is_regex=True,
                description="voseo chileno: haces -> hacís",
            ),
            TransformationRule(
                r'\bdices\b', 'decís', is_regex=True,
                description="voseo chileno: dices -> decís",
            ),
            TransformationRule(
                r'\bcrees\b', 'creís', is_regex=True,
                description="voseo chileno: crees -> creís",
            ),
            TransformationRule(
                r'\bconoces\b', 'conocís', is_regex=True,
                description="voseo chileno: conoces -> conocís",
            ),
            TransformationRule(
                r'\bprefieres\b', 'preferís', is_regex=True,
                description="voseo chileno: prefieres -> preferís",
            ),
            TransformationRule(
                r'\bnecesitas\b', 'necesitái', is_regex=True,
                description="voseo chileno: necesitas -> necesitái",
            ),
            TransformationRule(
                r'\bpagas\b', 'pagai', is_regex=True,
                description="voseo chileno: pagas -> pagai",
            ),
            TransformationRule(
                r'\bjuegas\b', 'jugái', is_regex=True,
                description="voseo chileno: juegas -> jugái",
            ),
            TransformationRule(
                r'\bmiras\b', 'mirái', is_regex=True,
                description="voseo chileno: miras -> mirái",
            ),
            TransformationRule(
                r'\btomas\b', 'tomai', is_regex=True,
                description="voseo chileno: tomas -> tomai",
            ),
            TransformationRule(
                r'\bcomes\b', 'comís', is_regex=True,
                description="voseo chileno: comes -> comís",
            ),
            TransformationRule(
                r'\bpones\b', 'ponís', is_regex=True,
                description="voseo chileno: pones -> ponís",
            ),
            TransformationRule(
                r'\beres\b', 'soi', is_regex=True,
                description="voseo chileno: eres -> soi",
            ),
            TransformationRule(
                r'\bvuelves\b', 'volvís', is_regex=True,
                description="voseo chileno: vuelves -> volvís",
            ),
            # Coger -> tomar
            TransformationRule(
                r'\bcoger\b', 'tomar', is_regex=True,
                description="coger -> tomar",
            ),
            TransformationRule(
                r'\bcogemos\b', 'tomamos', is_regex=True,
                description="cogemos -> tomamos",
            ),
            # Vosotros -> ustedes
            TransformationRule(
                r'\bvosotros\b', 'ustedes', is_regex=True,
                description="ustedes replaces vosotros",
            ),
            TransformationRule(
                r'\bvosotras\b', 'ustedes', is_regex=True,
                description="ustedes replaces vosotras",
            ),
            TransformationRule(
                r'\bhabéis\b', 'han', is_regex=True,
                description="ustedes conjugation",
            ),
            TransformationRule(
                r'\btenéis\b', 'tienen', is_regex=True,
                description="ustedes conjugation",
            ),
            TransformationRule(
                r'\bqueréis\b', 'quieren', is_regex=True,
                description="ustedes conjugation",
            ),
            TransformationRule(
                r'\bsabéis\b', 'saben', is_regex=True,
                description="ustedes conjugation",
            ),
            TransformationRule(
                r'\bpagáis\b', 'pagan', is_regex=True,
                description="ustedes conjugation",
            ),
            TransformationRule(
                r'\bpreferís\b', 'prefieren', is_regex=True,
                description="ustedes conjugation preferís",
            ),
            # Pretérito simple
            TransformationRule(
                r'\bhe ido\b', 'fui', is_regex=True,
                description="perfecto simple",
            ),
            TransformationRule(
                r'\bhe visto\b', 'vi', is_regex=True,
                description="perfecto simple",
            ),
            TransformationRule(
                r'\bhe comprado\b', 'compré', is_regex=True,
                description="perfecto simple",
            ),
            TransformationRule(
                r'\bhe comido\b', 'comí', is_regex=True,
                description="perfecto simple",
            ),
            TransformationRule(
                r'\bhe tenido\b', 'tuve', is_regex=True,
                description="perfecto simple",
            ),
            TransformationRule(
                r'\bme he comprado\b', 'me compré', is_regex=True,
                description="perfecto simple",
            ),
            TransformationRule(
                r'\bha dicho\b', 'dijo', is_regex=True,
                description="perfecto simple",
            ),
            TransformationRule(
                r'\bme ha dicho\b', 'me dijo', is_regex=True,
                description="perfecto simple",
            ),
            TransformationRule(
                r'\bha sido\b', 'fue', is_regex=True,
                description="perfecto simple",
            ),
            TransformationRule(
                r'\bhemos visto\b', 'vimos', is_regex=True,
                description="perfecto simple",
            ),
        ],
        pragmatic_markers=[
            "¿cachai?", "hueón", "weón", "po", "sí po", "no po",
            "ya po", "la raja", "al tiro", "caleta", "¿cachái o no?",
            "wena", "filo", "pucha", "brijido", "heavy",
            "la media", "¿te tinco?", "oye", "¿sabís qué?",
            "de chacota", "terrible", "su'a",
        ],
        phonological=[
            # Seseo
            TransformationRule(
                r'z([aeiou])', r's\1', is_regex=True,
                description="seseo",
            ),
            TransformationRule(
                r'c([ei])', r's\1', is_regex=True,
                description="seseo",
            ),
            # para -> pa'
            TransformationRule(
                r'\bpara\b', "pa'", is_regex=True,
                description="apócope: para -> pa'",
            ),
        ],
    ),

    # ==================================================================
    # ES_CAR — Caribbean (60+ lexical entries)
    # ==================================================================
    DialectCode.ES_CAR: DialectTemplate(
        lexical={
            # Original
            "autobús": "guagua",
            "ordenador": "computadora",
            "coche": "carro",
            "gafas": "gafas",
            "piso": "apartamento",
            "móvil": "celular",
            "genial": "chévere",
            "bueno": "chévere",
            "amigo": "pana",
            "cosa": "vaina",
            # Expanded Caribbean
            "chico": "chamo",
            "chica": "chama",
            "niño": "chamito",
            "hombre": "tipo",
            "mujer": "jeva",
            "novia": "jeva",
            "novio": "jevo",
            "dinero": "cuartos",
            "trabajo": "brega",
            "trabajar": "bregar",
            "fiesta": "rumba",
            "cerveza": "fría",
            "bonito": "bacano",
            "excelente": "brutal",
            "tonto": "pendejo",
            "borracho": "jarto",
            "enojado": "arrecho",
            "problema": "vaina",
            "comida": "jama",
            "comer": "jamar",
            "rápido": "ligero",
            "mucho": "un poco e'",
            "patata": "papa",
            "patatas": "papas",
            "zumo": "jugo",
            "melocotón": "durazno",
            "plátano": "plátano",
            "fresa": "fresa",
            "judías": "caraotas",
            "nevera": "nevera",
            "jersey": "suéter",
            "zapatillas": "tenis",
            "camiseta": "franela",
            "chaqueta": "chaqueta",
            "conducir": "manejar",
            "aparcar": "estacionar",
            "acera": "acera",
            "camarero": "mesonero",
            "alquiler": "alquiler",
            "grifo": "pluma",
            "ascensor": "ascensor",
            "piscina": "piscina",
            "supermercado": "supermercado",
            "enfadado": "arrecho",
            "manzana": "cuadra",
            "bocadillo": "sándwich",
            "tarta": "torta",
            "naranja": "china",
            "guapo": "papacito",
            "grande": "grande",
            "pequeño": "chiquitico",
            "hambre": "hambre",
            "calor": "calor",
            "frío": "frío",
            "loco": "loco",
            "policía": "policía",
            "barrio": "barrio",
            "suerte": "suerte",
            "mantequilla": "mantequilla",
            "falda": "falda",
        },
        morphological=[
            # Non-inverted questions (expanded)
            TransformationRule(
                r'¿qué quieres', '¿qué tú quieres', is_regex=False,
                description="sujeto pronominal: ¿qué quieres -> ¿qué tú quieres",
            ),
            TransformationRule(
                r'¿dónde vas', '¿dónde tú vas', is_regex=False,
                description="sujeto pronominal: ¿dónde vas -> ¿dónde tú vas",
            ),
            TransformationRule(
                r'¿cómo estás', '¿cómo tú estás', is_regex=False,
                description="sujeto pronominal: ¿cómo estás -> ¿cómo tú estás",
            ),
            TransformationRule(
                r'¿qué haces', '¿qué tú haces', is_regex=False,
                description="sujeto pronominal: ¿qué haces -> ¿qué tú haces",
            ),
            TransformationRule(
                r'¿qué piensas', '¿qué tú piensas', is_regex=False,
                description="sujeto pronominal: ¿qué piensas -> ¿qué tú piensas",
            ),
            TransformationRule(
                r'¿dónde vives', '¿dónde tú vives', is_regex=False,
                description="sujeto pronominal: ¿dónde vives -> ¿dónde tú vives",
            ),
            TransformationRule(
                r'¿cuánto cuesta', '¿cuánto tú crees que cuesta', is_regex=False,
                description="sujeto pronominal expansion",
            ),
            # Coger is fine in Caribbean but replace less common peninsular terms
            # Vosotros -> ustedes
            TransformationRule(
                r'\bvosotros\b', 'ustedes', is_regex=True,
                description="ustedes replaces vosotros",
            ),
            TransformationRule(
                r'\bvosotras\b', 'ustedes', is_regex=True,
                description="ustedes replaces vosotras",
            ),
            TransformationRule(
                r'\bhabéis\b', 'han', is_regex=True,
                description="ustedes conjugation",
            ),
            TransformationRule(
                r'\btenéis\b', 'tienen', is_regex=True,
                description="ustedes conjugation",
            ),
            TransformationRule(
                r'\bqueréis\b', 'quieren', is_regex=True,
                description="ustedes conjugation",
            ),
            TransformationRule(
                r'\bsabéis\b', 'saben', is_regex=True,
                description="ustedes conjugation",
            ),
            TransformationRule(
                r'\bpagáis\b', 'pagan', is_regex=True,
                description="ustedes conjugation",
            ),
            TransformationRule(
                r'\bpreferís\b', 'prefieren', is_regex=True,
                description="ustedes conjugation",
            ),
            # Pretérito simple
            TransformationRule(
                r'\bhe ido\b', 'fui', is_regex=True,
                description="perfecto simple",
            ),
            TransformationRule(
                r'\bhe visto\b', 'vi', is_regex=True,
                description="perfecto simple",
            ),
            TransformationRule(
                r'\bhe comprado\b', 'compré', is_regex=True,
                description="perfecto simple",
            ),
            TransformationRule(
                r'\bhe comido\b', 'comí', is_regex=True,
                description="perfecto simple",
            ),
            TransformationRule(
                r'\bme he comprado\b', 'me compré', is_regex=True,
                description="perfecto simple",
            ),
            TransformationRule(
                r'\bha dicho\b', 'dijo', is_regex=True,
                description="perfecto simple",
            ),
            TransformationRule(
                r'\bme ha dicho\b', 'me dijo', is_regex=True,
                description="perfecto simple",
            ),
            TransformationRule(
                r'\bha sido\b', 'fue', is_regex=True,
                description="perfecto simple",
            ),
            TransformationRule(
                r'\bhemos visto\b', 'vimos', is_regex=True,
                description="perfecto simple",
            ),
        ],
        pragmatic_markers=[
            "mijo", "mija", "¿oíste?", "chévere", "asere", "pana",
            "¿tú sabes?", "pa'", "vale", "mira", "dime",
            "¿verdad?", "epa", "coño", "¿entiendes?",
            "bacano", "compai", "mi hermano", "mi pana",
            "chamo", "nota e'", "qué ladilla", "fino",
        ],
        phonological=[
            # Seseo
            TransformationRule(
                r'z([aeiou])', r's\1', is_regex=True,
                description="seseo",
            ),
            TransformationRule(
                r'c([ei])', r's\1', is_regex=True,
                description="seseo",
            ),
            # Aspiration / loss of -s
            TransformationRule(
                r's\b', "'", is_regex=True,
                description="aspiración/elisión de -s final",
            ),
            # para -> pa'
            TransformationRule(
                r'\bpara\b', "pa'", is_regex=True,
                description="apócope: para -> pa'",
            ),
            # -ado -> -ao
            TransformationRule(
                r'\b(\w+)ado\b', r"\1ao", is_regex=True,
                description="elisión de -d- intervocálica en -ado",
            ),
            # -ido -> -ío (less systematic than Andalusian)
            TransformationRule(
                r'\b(\w+)ido\b', r"\1ío", is_regex=True,
                description="elisión de -d- intervocálica en -ido",
            ),
        ],
    ),

    # ==================================================================
    # ES_AND — Andalusian (50+ lexical entries)
    # ==================================================================
    DialectCode.ES_AND: DialectTemplate(
        lexical={
            # Original
            "chico": "quillo",
            "chica": "quilla",
            "prisa": "bulla",
            "mucho": "musho",
            # Expanded andalucismos
            "trabajar": "currar",
            "trabajo": "curro",
            "fiesta": "feria",
            "amigo": "primo",
            "novio": "churri",
            "novia": "churri",
            "dinero": "parné",
            "tonto": "boquino",
            "guapo": "pibón",
            "borracho": "moñas",
            "comida": "comía",
            "niño": "zagal",
            "niña": "zagala",
            "genial": "de puta madre",
            "mal": "malamente",
            "cansado": "reventao",
            "enfadado": "encendío",
            "problema": "marrón",
            "calor": "caló",
            "grande": "grande",
            "pequeño": "chiquitito",
            "mujer": "parienta",
            "hombre": "pisha",
            "cerveza": "cervesita",
            "loco": "pirao",
            "rápido": "volando",
            "hambre": "gusa",
            "cosa": "cosita",
            "excelente": "tela",
            "bonito": "bonito",
            "feo": "feucho",
            "policía": "madera",
            "alquiler": "alquilé",
            "barrio": "barrio",
            "supermercado": "súper",
            "tienda": "tienda",
            "patata": "patata",
            "patatas": "patatas",
            "zumo": "zumo",
            "ascensor": "ascensó",
            "jersey": "rebeca",
            "zapatillas": "zapatillas",
            "camiseta": "camiseta",
            "chaqueta": "chaqueta",
            "conducir": "conducí",
            "grifo": "grifo",
            "nevera": "nevera",
            "camarero": "camarero",
            "bocadillo": "bocata",
            "tarta": "tarta",
            "falda": "farda",
        },
        morphological=[
            # Ustedes replaces vosotros
            TransformationRule(
                r'\bvosotros\b', 'ustedes', is_regex=True,
                description="ustedes replaces vosotros",
            ),
            TransformationRule(
                r'\bvosotras\b', 'ustedes', is_regex=True,
                description="ustedes replaces vosotras",
            ),
            # Ustedes + vosotros verb ending mix (hallmark of western Andalusian)
            # Some speakers mix: "ustedes sabéis", "ustedes tenéis"
            # We leave habéis, tenéis, queréis etc. unchanged when combined
            # with ustedes -- this creates the characteristic mixing.
            # -ado -> -ao (participio)
            TransformationRule(
                r'\b(\w+)ado\b', r'\1ao', is_regex=True,
                description="caída de -d- intervocálica en -ado",
            ),
            # -ido -> -ío
            TransformationRule(
                r'\b(\w+)ido\b', r'\1ío', is_regex=True,
                description="caída de -d- intervocálica en -ido",
            ),
            # -ada -> -á
            TransformationRule(
                r'\b(\w+)ada\b', r"\1á", is_regex=True,
                description="caída de -d- intervocálica en -ada",
            ),
            # nada -> ná
            TransformationRule(
                r'\bnada\b', 'bah', is_regex=True,
                description="nada -> ná colloquial (replaced with ná in phono)",
            ),
            # todo -> to
            TransformationRule(
                r'\btodo\b', "to'", is_regex=True,
                description="todo -> to'",
            ),
            TransformationRule(
                r'\btoda\b', "to'a", is_regex=True,
                description="toda -> to'a",
            ),
        ],
        pragmatic_markers=[
            "quillo", "quilla", "arsa", "vale", "bah", "picha",
            "primo", "mi arma", "illo", "isha", "vamos",
            "venga ya", "anda ya", "compae", "¿sabes?",
            "de puta madre", "tela", "miarma", "pisha",
        ],
        phonological=[
            # Aspiración de -s ante consonante
            TransformationRule(
                r's\b', 'h', is_regex=True,
                description="aspiración de -s implosiva a final de sílaba",
            ),
            # el -> er (rotacismo)
            TransformationRule(
                r'\bel\b', 'er', is_regex=True,
                description="rotacismo: el -> er",
            ),
            # para -> pa
            TransformationRule(
                r'\bpara\b', 'pa', is_regex=True,
                description="apócope: para -> pa",
            ),
            # nada -> ná (after morphological step)
            TransformationRule(
                r'\bnada\b', "ná", is_regex=True,
                description="nada -> ná",
            ),
        ],
    ),

    # ==================================================================
    # ES_CAN — Canarian (40+ lexical entries)
    # ==================================================================
    DialectCode.ES_CAN: DialectTemplate(
        lexical={
            # Original
            "autobús": "guagua",
            "patata": "papa",
            "patatas": "papas",
            # Expanded canarismos
            "niño": "niño",
            "chico": "muchacho",
            "chica": "muchacha",
            "amigo": "compadre",
            "bonito": "lindo",
            "tonto": "machango",
            "fiesta": "fiesta",
            "dinero": "chavos",
            "comida": "comida",
            "maíz": "millo",
            "calabaza": "bubango",
            "lagarto": "perenquén",
            "cactus": "tunera",
            "harina de maíz": "gofio",
            "camiseta": "camiseta",
            "conducir": "manejar",
            "borracho": "jumao",
            "genial": "brutal",
            "cansado": "jarto",
            "loco": "ido",
            "calor": "calor",
            "rápido": "ligero",
            "hambre": "hambre",
            "problema": "lío",
            "cerveza": "cerveza",
            "jersey": "jersey",
            "zapatillas": "zapatillas",
            "chaqueta": "chaqueta",
            "zumo": "jugo",
            "piscina": "piscina",
            "supermercado": "supermercado",
            "alquiler": "alquiler",
            "barrio": "barrio",
            "enfadado": "cabreado",
            "grifo": "grifo",
            "nevera": "nevera",
            "camarero": "camarero",
            "ascensor": "ascensor",
            "acera": "acera",
            "bocadillo": "bocadillo",
            "tarta": "tarta",
        },
        morphological=[
            # Ustedes replaces vosotros
            TransformationRule(
                r'\bvosotros\b', 'ustedes', is_regex=True,
                description="ustedes replaces vosotros",
            ),
            TransformationRule(
                r'\bvosotras\b', 'ustedes', is_regex=True,
                description="ustedes replaces vosotras",
            ),
            TransformationRule(
                r'\bhabéis\b', 'han', is_regex=True,
                description="ustedes conjugation",
            ),
            TransformationRule(
                r'\btenéis\b', 'tienen', is_regex=True,
                description="ustedes conjugation",
            ),
            TransformationRule(
                r'\bqueréis\b', 'quieren', is_regex=True,
                description="ustedes conjugation",
            ),
            TransformationRule(
                r'\bsabéis\b', 'saben', is_regex=True,
                description="ustedes conjugation",
            ),
            TransformationRule(
                r'\bpagáis\b', 'pagan', is_regex=True,
                description="ustedes conjugation",
            ),
            TransformationRule(
                r'\bpreferís\b', 'prefieren', is_regex=True,
                description="ustedes conjugation",
            ),
            # Pretérito simple preference (moderate)
            TransformationRule(
                r'\bhe ido\b', 'fui', is_regex=True,
                description="perfecto simple",
            ),
            TransformationRule(
                r'\bhe visto\b', 'vi', is_regex=True,
                description="perfecto simple",
            ),
            TransformationRule(
                r'\bhe comprado\b', 'compré', is_regex=True,
                description="perfecto simple",
            ),
            TransformationRule(
                r'\bha dicho\b', 'dijo', is_regex=True,
                description="perfecto simple",
            ),
            TransformationRule(
                r'\bme ha dicho\b', 'me dijo', is_regex=True,
                description="perfecto simple",
            ),
        ],
        pragmatic_markers=[
            "chacho", "chacha", "mijo", "mija", "¿verdad?",
            "no me digas", "oye", "hombre", "mira",
            "¿sabes?", "ni boba", "bendito", "ay mi madre",
            "¿ves?", "fíjate",
        ],
        phonological=[
            # Seseo
            TransformationRule(
                r'z([aeiou])', r's\1', is_regex=True,
                description="seseo: z -> s ante vocal",
            ),
            TransformationRule(
                r'c([ei])', r's\1', is_regex=True,
                description="seseo: ce/ci -> se/si",
            ),
            # Aspiration of /s/ before consonant (moderate)
            # We don't apply final -s aspiration as systematically as Andalusian
        ],
    ),

    # ==================================================================
    # ES_AND_BO — Andean (40+ lexical entries)
    # ==================================================================
    DialectCode.ES_AND_BO: DialectTemplate(
        lexical={
            # Original
            "autobús": "bus",
            "coche": "carro",
            "ordenador": "computadora",
            "gafas": "lentes",
            "piso": "departamento",
            "apartamento": "departamento",
            "móvil": "celular",
            # Expanded andinismos
            "amigo": "causa",
            "casa": "jato",
            "hermano": "ñaño",
            "dinero": "plata",
            "gratis": "de gana",
            "niño": "guagua",
            "chico": "cholo",
            "chica": "chola",
            "trabajo": "chamba",
            "trabajar": "chambear",
            "fiesta": "fiesta",
            "cerveza": "chela",
            "borracho": "borracho",
            "tonto": "cojudo",
            "genial": "chevere",
            "bueno": "bacán",
            "bonito": "lindo",
            "feo": "feo",
            "comida": "comida",
            "mercado": "mercado",
            "campo": "chacra",
            "extra": "yapa",
            "mareo de altura": "soroche",
            "cansado": "cansado",
            "enfadado": "molesto",
            "problema": "problema",
            "calor": "calor",
            "frío": "frío",
            "rápido": "rápido",
            "patata": "papa",
            "patatas": "papas",
            "zumo": "jugo",
            "melocotón": "durazno",
            "plátano": "plátano",
            "fresa": "fresa",
            "judías": "frejoles",
            "mantequilla": "mantequilla",
            "nevera": "refrigeradora",
            "jersey": "chompa",
            "zapatillas": "zapatillas",
            "camiseta": "polo",
            "chaqueta": "casaca",
            "conducir": "manejar",
            "aparcar": "estacionar",
            "camarero": "mozo",
            "alquiler": "alquiler",
            "grifo": "caño",
            "ascensor": "ascensor",
            "piscina": "piscina",
            "supermercado": "mercado",
            "acera": "vereda",
            "barrio": "barrio",
            "bocadillo": "sánguche",
            "tarta": "torta",
            "maíz": "choclo",
        },
        morphological=[
            # Nomás attenuation (expanded)
            TransformationRule(
                r'\bpase\b', 'pase nomás', is_regex=True,
                description="atenuación andina: pase -> pase nomás",
            ),
            TransformationRule(
                r'\bsírvase\b', 'sírvase nomás', is_regex=True,
                description="atenuación andina: sírvase -> sírvase nomás",
            ),
            TransformationRule(
                r'\bvenga\b', 'venga nomás', is_regex=True,
                description="atenuación andina: venga -> venga nomás",
            ),
            TransformationRule(
                r'\bsiente\b', 'siéntese nomás', is_regex=True,
                description="atenuación andina",
            ),
            # Doble posesivo
            TransformationRule(
                r'la casa de mi (\w+)', r'su casa de mi \1', is_regex=True,
                description="doble posesivo andino",
            ),
            TransformationRule(
                r'el hermano de mi (\w+)', r'su hermano de mi \1', is_regex=True,
                description="doble posesivo andino",
            ),
            TransformationRule(
                r'el hijo de mi (\w+)', r'su hijo de mi \1', is_regex=True,
                description="doble posesivo andino",
            ),
            # Loísmo/leísmo patterns: lo/la -> le (for animate objects)
            TransformationRule(
                r'\blo llamé\b', 'le llamé', is_regex=True,
                description="leísmo andino: lo -> le para animado",
            ),
            TransformationRule(
                r'\blo he llamado\b', 'le he llamado', is_regex=True,
                description="leísmo andino",
            ),
            # Vosotros -> ustedes
            TransformationRule(
                r'\bvosotros\b', 'ustedes', is_regex=True,
                description="ustedes replaces vosotros",
            ),
            TransformationRule(
                r'\bvosotras\b', 'ustedes', is_regex=True,
                description="ustedes replaces vosotras",
            ),
            TransformationRule(
                r'\bhabéis\b', 'han', is_regex=True,
                description="ustedes conjugation",
            ),
            TransformationRule(
                r'\btenéis\b', 'tienen', is_regex=True,
                description="ustedes conjugation",
            ),
            TransformationRule(
                r'\bqueréis\b', 'quieren', is_regex=True,
                description="ustedes conjugation",
            ),
            TransformationRule(
                r'\bsabéis\b', 'saben', is_regex=True,
                description="ustedes conjugation",
            ),
            TransformationRule(
                r'\bpagáis\b', 'pagan', is_regex=True,
                description="ustedes conjugation",
            ),
            TransformationRule(
                r'\bpreferís\b', 'prefieren', is_regex=True,
                description="ustedes conjugation",
            ),
        ],
        pragmatic_markers=[
            "¿ya?", "pues", "pe", "nomás", "oye", "oiga",
            "sí pues", "ya pues", "¿no cierto?", "pues sí",
            "oe", "causa", "¿manyas?", "al toque",
            "de una", "caserito", "¿ya pe?", "hijito",
        ],
        phonological=[
            # Seseo
            TransformationRule(
                r'z([aeiou])', r's\1', is_regex=True,
                description="seseo",
            ),
            TransformationRule(
                r'c([ei])', r's\1', is_regex=True,
                description="seseo",
            ),
        ],
    ),
}


# ======================================================================
# Dialect-specific sentence connectors for compound sentence generation
# ======================================================================

_DIALECT_CONNECTORS: dict[DialectCode, list[str]] = {
    DialectCode.ES_PEN: ["y además", "es que", "pero vamos", "o sea", "en plan"],
    DialectCode.ES_RIO: ["y aparte", "es que", "pero mirá", "tipo", "bah"],
    DialectCode.ES_MEX: ["y aparte", "es que", "pero pues", "o sea", "la neta"],
    DialectCode.ES_CHI: ["y aparte", "es que", "pero oye", "o sea", "la weá es que"],
    DialectCode.ES_CAR: ["y aparte", "es que", "pero mira", "o sea", "la vaina es que"],
    DialectCode.ES_AND: ["y encima", "es que", "pero bueno", "o sea", "vamos"],
    DialectCode.ES_CAN: ["y encima", "es que", "pero oye", "o sea", "mira"],
    DialectCode.ES_AND_BO: ["y aparte", "es que", "pero pues", "o sea", "la cosa es que"],
}


# ======================================================================
# Generator class
# ======================================================================

class EnhancedSyntheticGenerator:
    """Generate thousands of dialectally-transformed samples using 300+ base
    sentences, expanded lexical dictionaries, morphological rules, and
    combinatorial variant generation.

    The generator produces 4-8 variants per base sentence per dialect by:
    - Randomly inserting pragmatic markers (~30 % of the time)
    - Randomly combining short sentences into compound sentences
    - Applying phonological rules at varying intensities (~50 % per rule)
    - Varying pragmatic-marker placement (prepend vs. append)

    Parameters
    ----------
    seed:
        Random seed for reproducibility.  ``None`` disables seeding.
    """

    def __init__(self, seed: Optional[int] = 42) -> None:
        self._rng = random.Random(seed)
        self._base_sentences = list(ENHANCED_BASE_SENTENCES)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_all(self) -> dict[DialectCode, list[DialectSample]]:
        """Generate thousands of samples for every dialect.

        Returns roughly 2000-4000 unique samples per dialect depending on
        the combinatorial expansion.

        Returns
        -------
        dict[DialectCode, list[DialectSample]]
        """
        result: dict[DialectCode, list[DialectSample]] = {}
        for code in DialectCode:
            result[code] = self.generate_dialect(code)
        return result

    def generate_dialect(
        self,
        dialect: DialectCode,
        n: int = 500,
    ) -> list[DialectSample]:
        """Generate *n* base transformations for a single dialect, then expand
        each through combinatorial variation to produce ~4-8x as many total
        unique samples.

        Parameters
        ----------
        dialect:
            Target dialect code.
        n:
            Number of base sentences to draw (with replacement if *n* exceeds
            the bank).  The final sample count will be *much larger* due to
            variant expansion.

        Returns
        -------
        list[DialectSample]
            Deduplicated list of dialect samples.
        """
        template = self._get_merged_template(dialect)
        connectors = _DIALECT_CONNECTORS.get(dialect, ["y", "pero", "es que"])

        chosen = self._rng.choices(self._base_sentences, k=n)
        samples: list[DialectSample] = []
        seen_texts: set[str] = set()

        for base_idx, base in enumerate(chosen):
            variants = self._generate_variants(
                base, template, connectors, dialect, chosen, base_idx,
            )
            for var_idx, text in enumerate(variants):
                normalised = text.strip()
                if normalised in seen_texts:
                    continue
                seen_texts.add(normalised)
                confidence = round(self._rng.uniform(0.70, 0.92), 3)
                samples.append(
                    DialectSample(
                        text=normalised,
                        dialect_code=dialect,
                        source_id="enhanced_synthetic_generator",
                        confidence=confidence,
                        metadata={
                            "base_sentence": base,
                            "generation_index": base_idx,
                            "variant_index": var_idx,
                        },
                    )
                )

        return samples

    # ------------------------------------------------------------------
    # Template merging
    # ------------------------------------------------------------------

    def _get_merged_template(self, dialect: DialectCode) -> DialectTemplate:
        """Merge the base template (from ``DIALECT_TEMPLATES``) with the
        enhanced overlay.  Enhanced entries take precedence for lexical
        conflicts; rule lists are concatenated (deduped on description).
        """
        base = DIALECT_TEMPLATES.get(dialect)
        enhanced = ENHANCED_TEMPLATES.get(dialect)

        if enhanced is None and base is None:
            raise ValueError(f"No template for dialect {dialect.value}")
        if enhanced is None:
            return deepcopy(base)  # type: ignore[arg-type]
        if base is None:
            return deepcopy(enhanced)

        # Merge lexical: enhanced wins on conflicts
        merged_lexical = dict(base.lexical)
        merged_lexical.update(enhanced.lexical)

        # Merge morphological: collect unique by description
        merged_morph = list(base.morphological)
        existing_desc = {r.description for r in merged_morph}
        for rule in enhanced.morphological:
            if rule.description not in existing_desc:
                merged_morph.append(rule)
                existing_desc.add(rule.description)

        # Merge pragmatic markers: union (preserve order)
        seen_markers: set[str] = set()
        merged_pragmatic: list[str] = []
        for m in list(base.pragmatic_markers) + list(enhanced.pragmatic_markers):
            if m not in seen_markers:
                merged_pragmatic.append(m)
                seen_markers.add(m)

        # Merge phonological
        merged_phono = list(base.phonological)
        existing_phono_desc = {r.description for r in merged_phono}
        for rule in enhanced.phonological:
            if rule.description not in existing_phono_desc:
                merged_phono.append(rule)
                existing_phono_desc.add(rule.description)

        return DialectTemplate(
            lexical=merged_lexical,
            morphological=merged_morph,
            pragmatic_markers=merged_pragmatic,
            phonological=merged_phono,
        )

    # ------------------------------------------------------------------
    # Variant generation (core combinatorial logic)
    # ------------------------------------------------------------------

    def _generate_variants(
        self,
        base: str,
        template: DialectTemplate,
        connectors: list[str],
        dialect: DialectCode,
        all_chosen: list[str],
        base_idx: int,
    ) -> list[str]:
        """Produce 4-8 variants from a single base sentence.

        Variant types:
        1. Full transformation (lexical + morphological + phonological)
        2. Full transformation + pragmatic marker (prepended)
        3. Full transformation + pragmatic marker (appended as tag)
        4. Partial phonological application (50% chance per rule)
        5. Compound sentence (combine with another random base)
        6-8. Additional marker/phono combos based on RNG
        """
        variants: list[str] = []

        # --- Variant 1: full transform ---
        v1 = template.apply_all(base)
        variants.append(v1)

        # --- Variant 2: full transform + prepend marker ---
        v2 = self._prepend_marker(v1, template)
        if v2 != v1:
            variants.append(v2)

        # --- Variant 3: full transform + append tag marker ---
        v3 = self._append_marker(v1, template)
        if v3 != v1:
            variants.append(v3)

        # --- Variant 4: partial phonological ---
        v4 = self._apply_partial_phonology(base, template)
        if v4 != v1:
            variants.append(v4)

        # --- Variant 5: compound sentence ---
        if len(all_chosen) > 1:
            partner_idx = self._rng.randint(0, len(all_chosen) - 1)
            if partner_idx != base_idx:
                partner = all_chosen[partner_idx]
                compound = self._make_compound(
                    base, partner, template, connectors,
                )
                variants.append(compound)

        # --- Variant 6-8: extra marker + partial phono combos ---
        if self._rng.random() < 0.6:
            v6 = self._prepend_marker(v4, template)
            if v6 not in variants:
                variants.append(v6)

        if self._rng.random() < 0.5:
            v7 = self._append_marker(
                self._apply_partial_phonology(base, template), template,
            )
            if v7 not in variants:
                variants.append(v7)

        if self._rng.random() < 0.4:
            # Double marker: prepend one, append another
            v8 = self._prepend_marker(
                self._append_marker(v1, template), template,
            )
            if v8 not in variants:
                variants.append(v8)

        return variants

    # ------------------------------------------------------------------
    # Marker insertion helpers
    # ------------------------------------------------------------------

    def _prepend_marker(self, text: str, template: DialectTemplate) -> str:
        """Prepend a random pragmatic marker to *text*."""
        if not template.pragmatic_markers:
            return text
        # Filter to markers that work as sentence-initial (skip tag questions)
        initial = [m for m in template.pragmatic_markers if not m.startswith("¿")]
        if not initial:
            return text
        marker = self._rng.choice(initial)
        return marker.capitalize() + ", " + text[0].lower() + text[1:]

    def _append_marker(self, text: str, template: DialectTemplate) -> str:
        """Append a random pragmatic marker as a tag question / filler."""
        if not template.pragmatic_markers:
            return text
        # Prefer tag-question style markers
        tags = [m for m in template.pragmatic_markers if m.startswith("¿")]
        fillers = [m for m in template.pragmatic_markers if not m.startswith("¿")]
        if tags and self._rng.random() < 0.6:
            marker = self._rng.choice(tags)
            return text.rstrip(".!") + ", " + marker
        elif fillers:
            marker = self._rng.choice(fillers)
            return text.rstrip(".!") + ", " + marker + "."
        return text

    # ------------------------------------------------------------------
    # Partial phonological application
    # ------------------------------------------------------------------

    def _apply_partial_phonology(
        self, base: str, template: DialectTemplate,
    ) -> str:
        """Apply lexical + morphological fully, but only ~50% of
        phonological rules (randomly chosen)."""
        text = template.apply_lexical(base)
        text = template.apply_morphological(text)

        # Apply each phono rule with 50% probability
        for rule in template.phonological:
            if self._rng.random() < 0.5:
                if rule.is_regex:
                    text = re.sub(rule.pattern, rule.replacement, text)
                else:
                    text = text.replace(rule.pattern, rule.replacement)

        return text

    # ------------------------------------------------------------------
    # Compound sentence generation
    # ------------------------------------------------------------------

    def _make_compound(
        self,
        sent_a: str,
        sent_b: str,
        template: DialectTemplate,
        connectors: list[str],
    ) -> str:
        """Transform two base sentences and join them with a dialect-specific
        connector into a compound sentence."""
        a = template.apply_all(sent_a)
        b = template.apply_all(sent_b)

        # Remove trailing period from first sentence, lowercase second
        a = a.rstrip(".")
        if b:
            b = b[0].lower() + b[1:]

        connector = self._rng.choice(connectors)
        compound = f"{a}, {connector} {b}"

        # Maybe add a pragmatic marker to the compound
        if self._rng.random() < 0.3:
            compound = self._prepend_marker(compound, template)

        return compound

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def base_sentence_count(self) -> int:
        """Number of base sentences in the enhanced bank."""
        return len(self._base_sentences)

    def add_base_sentences(self, sentences: list[str]) -> None:
        """Extend the base sentence bank with additional entries."""
        self._base_sentences.extend(sentences)
