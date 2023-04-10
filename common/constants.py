import os

from dotenv import load_dotenv

load_dotenv()

C_REPORT_ID = 'report_id'
C_COMPLAINT = 'complaint'
C_KEYWORDS = 'keywords'
C_ADDRESS = 'address'
C_NEIGHBORHOOD = 'neighborhood'
C_DATE = 'date'
C_SECTOR = 'sector'
C_LOCATION = 'location'
C_SCORE = 'score'
C_LATITUDE = 'latitude'
C_LONGITUDE = 'longitude'

GOOGLE_MAPS_API_URL = 'https://maps.googleapis.com/maps/api/geocode/json'
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_API_ENV = os.getenv('PINECONE_API_ENV')
PINECONE_INDEX_COMPLAINTS = 'citizen-complaints-2023'
PINECONE_INDEX_NAME = 'ai-urban-planning'
DOCS_INDEX_DIR = 'data/db/urban-planning-documents'
KEEP_COLUMNS = [C_REPORT_ID, C_COMPLAINT, C_DATE, C_SECTOR, C_NEIGHBORHOOD, C_KEYWORDS]
MONGODB_CONNECTION = os.getenv('MONGODB_CONNECTION')
MONGODB_PROJECT = 'comunia'

FILE_CITIZEN_REPORTS = 'data/jsons/citizen_reports_2023.yaml'
FILE_NEIGHBORHOOD_REPORTS = 'data/jsons/comunity_reports_2023.yaml'
FILE_URBAN_DOCUMENTS = 'data/jsons/urban_planning_documents.yaml'
DIR_PDF = 'data/pdfs'

GPT_EMBEDDING_MODEL = 'text-embedding-ada-002'
GPT_CHAT_MODEL = 'gpt-3.5-turbo'
GPT_SETTINGS_SUMMARIZE = dict(
    max_tokens=600,
    temperature=0,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
)
GPT_SETTINGS_SUMMARIZE_KNOWLEDGE = dict(
    max_tokens=250,
    temperature=0,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
)
GPT_SETTINGS_EXTRACT = dict(
    max_tokens=500,
    temperature=0.7,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
)

MESSAGES_EXTRACT_DATA = [
    """
    Dado un texto que contiene la queja de un ciudadano sobre un problema de movilidad en una región determinada, genera un YAML que incluya la siguiente información en formato clave-valor:
    ---
    # Problematica(s) que experimenta el ciudadano, incluyendo la naturaleza del problema y peligros presentes
    problemas_ciudadano: []
    # Propuesta(s) del ciudadano ante el problema
    propuestas_ciudadano: []
    # Tipo(s) de transporte usado o afectado por el problema
    tipo_transportes: []
    # Tipo de personas morales y fisicas afectados por el problema
    afectados: []
    # Detalles precisos sobre la ubicación del problema, como la dirección, colonia, rutas, etc
    ubicaciones: []
    # Cuándo se presenta el problema
    horarios: []
    # Cada cuanto se presenta el problema
    frecuencia: ""

    Completa cada valor con la información proporcionada por en la queja ciudadana, asegurando que se extraiga toda la informacion mencionada por el ciudadano.
    IMPORTANTE: SOLO INCLUIR INFORMACION PRESENTE DIRECTAMENTE EN LA QUEJA.
    """,
    """
    Un oficial de tránsito en el cruce de Alfonso Reyes y Av. Las Sendas en horarios de entrada y salidas del colegio 7:00-8:00 am 1:00-3:00 pm 4:00-4:30 pm y 5:45-6:15 pm. Es imposible dar vuelta de Alfonso Reyes a Av. Las Sendas de oriente a poniente. Hay demasiado tráfico vehicular y ahora hasta de camiones urbanos que se circulan muy rápido y que ahora dan vuelta a Av. Corregidora (donde detienen el tráfico esperando su luz verde) urge alguna adaptación o algún oficial de tránsito porque el tráfico de la construcción del centro comercial en la esquina ha ido en aumento. Es un verdadero problema en las horas antes mencionadas. Urge atención. Gracias.
    """,
    """
    problemas_ciudadano: ["Trafico en horarios de entrada y salida del colegio en el cruce de Alfonso Reyes y Av. Las Sendas, ya que camiones dan vuelta en Av Corregidora, generando trafico y dificulta dar vuelta de Alfonso Reyes a Av. Las Sendas de oriente a poniente.", "La construcción del Centro Comercial contribuye al aumento de tráfico"]
    propuestas_ciudadano: ["adaptar la zona", "agregar oficial de transito"]
    tipo_transporte: ["Vehiculo", "Autobús"]
    afectados: ["Colegio", "Construccion centro comercial"]
    ubicaciones: ["CRUCE DE ALFONSO REYES Y AV. LAS SENDAS"]
    horario: ["7:00-8:00 AM", "1:00-3:00 PM", "4:00-4:30 PM", "5:45-6:15 PM"]
    frecuencia: ""
    """,
    """{complaint}"""
]
MESSAGES_COMMUNITY_REPORT = [
    """
    Eres un asistente que transforma quejas ciudadanas en reportes comunitarios.
    Haz un resumen de un parrafo que SOLAMENTE use las quejas ciudadanas, sintetizando puntos clave y manteniendo como horarios, especificos.
    """,
    """
    QUEJAS CIUDADANAS:
    1.- SE METEN EN CONTRA, Y AHÍ SALEN NIÑOS, NO ES POSIBLE QUE ME ENVÍEN UNA NOTIFICACIÓN DONDE EL REPORTE YA SE LE DIO ATENCIÓN Y LO DEN DE BAJA. SE METEN MUCHOS CARROS EN CONTRA. EXIJO POR EL BIEN DE LOS NIÑOS QUE UN OFICIAL, PUEDE SER SIN UNIDAD, EN UN HORARIO SOLO DE 7:30 A 8:30 Y DE 3:30 A 4:30 POR SU SEGURIDAD, EL OTRO DÍA VI COMO POR MUY POCO CASI ATROPELLAN A UN PERSONA Y A SU BEBÉ. SUPER MAL
    2.- TODOS LOS DIAS LUNES SE INSTALA UN MERCADO RODANTE EN LA CALLE FRANCISCO SILLER Y MANUEL VARGAS AYALA, POR ESTE MOTIVO LA CALLE DIEGO SALDIVAR LATERAL AL MERCADO QUEDA COMO ENTRADA Y SALIDA A LA COLONIA POR DONDE CIRCULAN TODO TIPO DE VEHICULOS COMO CAMIONES URBANOS, DE VOLTEO , MENSAJERIA ETC. Y LOS MISMO VECINOS DEL SECTOR. HAGO MENCION DE ESTO YA QUE SE HA HECHO FRECUENTE QUE AL NO CONTAR CON LA DISPONIBILIDAD DEL ESPACIO DEL MERCADO PARA QUE QUIENES ACUDEN AL INSTITUTO NUEVO AMANECER, GIMNASIO O AL MISMO MERCADO , ESTACIONEN SUS VEHICULOS EN POR LA CALLE DIEGO SALDIVAR DESDE LAZARO GARZA AYALA A MANUEL VARGAS AYALA EN AMBOS SENTIDOS INCLUSIVE SOBRE LAS BANQUETAS , DEJANDO SOLO EL AREA CENTRAL DE LA CALLE DIEGO SALDIVAR PARA QUE CIRCULEN VEHICULOS EN LOS DOS SENTIDOS. POR LO QUE SOLICITO APOYO CON PERSONAL DE VIALIDAD. PARA CORREGIR ESTA PROBLEMATICA DE TODOS LOS LUNES DESDE LAS 7 AM A LAS 3 PM
    3.- CIUDADANA SOLICITA MAS RONDINES DE VIGILANCIA YA QUE COMENTA QUE EL DÍA LUNES QUE SE PONE EL MERCADO LOS CARROS SE ESTACIONAN MAL Y SE SUBEN A LA BANQUETA Y HAY MUCHO DESORDEN VIAL.
    """,
    """
    Se reporta la falta de espacio disponible para estacionamiento en la calle Francisco Siller y Manuel Vargas Ayala los Lunes debido al establecimiento de un mercado rodante.
    Debido a esto, vehiculos que acuden al Instituto Nuevo Amanecer, al gimnasio o al mercado se estacionan sobre la banqueta en ambos sentidos en la calle Diego Saldivar (lateral al mercado). Esta calle queda como entrada y salida a la colonia, por donde camiones urbanos, de volteo, de mensajería y vehiculos circulan en ambos sentidos, yendo en sentido contrario. Esto ocasiona situaciones de peligro a peatones, especialmente a ninos.
    Se solicita personal de vialidad y rondines de vigilancia los Lunes de 7:30am a 8:30am y de 3:30pm a 4:30pm.
    """,
    """
    QUEJAS CIUDADANAS:
    {complaints}
    """
]
MESSAGES_KNOWLEDGE = [
    """
    Extrae de los segmentos de documentos de planeacion urbana puntos clave relevantes a las siguientes problematicas.
    PROBLEMATICAS:
    {problems}
    """,
    """
    SEGMENTOS DOCUMENTOS PLANEACION URBANA:
    {listed_knowledge}
    """
]
MESSAGES_QUESTION = [
    """
    Eres un asistente que ayuda a planeadores urbanos a analizar el congestionamiento y la movilidad en la zona.
    El planeador urbano quiere entender a mayor detalle las quejas de los ciudadanos.
    Tu labor es usar las quejas ciudadanas para responder dudas usando el fragmento del documento de planeacion urbana para complementar la relacion que tiene con las quejas ciudadanas.

    FRAGMENTO DOCUMENTO PLANEACION URBANA:
    {knowledge}

    QUEJAS CIUDADANAS:
    {complaints}
    """,
    """
    {question}
    """
]
TEMPLATE_QUESTION = """
Eres una IA de planeacion urbana que ayuda a planeadores urbanos a analizar la movilidad en la zona.
El planeador urbano quiere entender las problematicas que se presentan en las quejas ciudadanas.
Tu labor es:
 - Usar las quejas ciudadanas para responder dudas.
 - Usar el segmento de planeacion urbana para dar contexto.

SEGMENTO DE PLANEACION URBANA:
{knowledge}

QUEJAS CIUDADANAS:
{complaints}
"""
TEMPLATE_KNOWLEDGE = """
UBICACIONES: {locations}
REPORTE COMUNITARIO:
{complaints}
"""

PREFIX = """
Regina es un modelo de lenguaje grande entrenado para ser una asistente en el analisis de planificacion urbana.
Regina está diseñada para ayudar a planeadores urbanos resolver cualquier duda respecto a problemas de movilidad y planeacion urbana en la zona de San Pedro, Mexico.
Como modelo de lenguaje, Regina es capaz de realizar una gran variedad de analisis detallados a varios tipos de problematicas, proporcionar respuestas coherentes y relevantes a las dudas del planeador urbano.

Regina está constantemente entendiendo las problematicas, y sus capacidades para proporcionar soluciones son siempre basadas en el conocimiento que adquiere de las herramientas a su disposicion.
Es capaz de comprender y analizar grandes cantidades de texto, y puede utilizar este conocimiento para proporcionar respuestas detalladas e informativas a una amplia gama de preguntas.
Además, Regina es capaz de generar su propio texto basado en la entrada que recibe, lo que le permite participar en discusiones y proporcionar explicaciones y descripciones sobre una amplia variedad de temas.

En general, Regina es una herramienta poderosa que puede ayudar con una amplia variedad de tareas y proporcionar información valiosa y perspectivas sobre una amplia variedad de temas en la planeacion urbana. Ya sea que necesite ayuda con una pregunta específica o quiera entender a grandes razgos sobre un tema, Regina está aquí para ayudar.

HERRAMIENTAS:
------

Regina tiene acceso a las siguientes herramientas:"""
FORMAT_INSTRUCTIONS = """Para usar una herramienta, por favor use el siguiente formato:

```
Thought: ¿Necesito usar una herramienta? Si
Action: Atención ciudadana
Action Input: [la entrada para la acción]
Observation: [el resultado de la acción]
```

Cuando tengas una respuesta que decir al planeador urbano, o si no necesitas usar una herramienta, DEBES usar el siguiente formato:

```
Thought: ¿Necesito usar una herramienta? No
{ai_prefix}: [su respuesta aquí]
```"""

SUFFIX = """Comienza!

Historial de conversación previa:
{chat_history}

New input: {input}
{agent_scratchpad}"""
