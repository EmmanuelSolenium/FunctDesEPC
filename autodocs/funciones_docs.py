import pandas as pd
import re
import io

# ==============================
# TAMAÑO POR DEFECTO DE IMÁGENES
# ==============================
DEFAULT_IMAGE_WIDTH_PT  = 300
DEFAULT_IMAGE_HEIGHT_PT = 200

# ==============================
# MAPEO DE TIPOS DESDE EL EXCEL
# ==============================
TIPO_MAP = {
    "texto":          "text",
    "número":         "text",
    "numero":         "text",
    "condicional":    "text",
    "imagen":         "image",
    "tabla":          "table",
    "imagen / tabla": "image",  # se trata como imagen por ahora
}


# ==============================
# CARGA DEL DICCIONARIO DESDE EXCEL
# ==============================
def cargar_diccionario(archivo, project_filter=None):
    """
    Lee el Excel del diccionario y lo convierte al esquema unificado:

    {
        "alias": {
            "placeholder": "{{ Nombre del dato }}",
            "type": "text" | "image" | "table",
            "value": <valor según tipo>
        }
    }

    Para type="text"  → value es string o None
    Para type="image" → value es {"file_id": ..., "width_pt": ..., "height_pt": ...}
    Para type="table" → value es None (Fase 3)

    Columnas opcionales en el Excel:
        "width_pt" y "height_pt" para imágenes — si no existen, se usan los defaults.
    """

    if isinstance(archivo, str):
        df = pd.read_csv(archivo) if archivo.endswith(".csv") else pd.read_excel(archivo)
    else:
        df = pd.read_excel(archivo)

    # Primera fila es el header real
    df.columns = df.iloc[0]
    df = df[1:].reset_index(drop=True)

    # Normalizar nombres de columnas
    df.columns = [str(col).strip().lower() for col in df.columns]

    df = df.rename(columns={
        "nombre del dato": "placeholder",
        "valor":           "value",
        "tipo":            "type",
        "alias":           "alias",
    })

    tiene_width  = "width_pt"  in df.columns
    tiene_height = "height_pt" in df.columns

    data = {}

    for _, row in df.iterrows():
        placeholder = str(row.get("placeholder", "")).strip()
        raw_value   = row.get("value")
        raw_type    = str(row.get("type", "")).strip().lower()
        alias       = str(row.get("alias", "")).strip()

        if not placeholder and not alias:
            continue

        key_limpia = limpiar_placeholder(placeholder)
        final_key  = alias if alias else key_limpia

        tipo_normalizado = TIPO_MAP.get(raw_type, "text")

        if tipo_normalizado == "text":
            value = _convertir_a_texto(raw_value)

        elif tipo_normalizado == "image":
            file_id = str(raw_value).strip() if not _es_nulo(raw_value) else None
            width_pt  = _leer_dim(row, "width_pt")  if tiene_width  else DEFAULT_IMAGE_WIDTH_PT
            height_pt = _leer_dim(row, "height_pt") if tiene_height else DEFAULT_IMAGE_HEIGHT_PT
            value = {
                "file_id":   file_id,
                "width_pt":  width_pt,
                "height_pt": height_pt,
            }

        elif tipo_normalizado == "table":
            value = None  # Fase 3

        else:
            value = _convertir_a_texto(raw_value)

        data[final_key] = {
            "placeholder": placeholder,
            "type":        tipo_normalizado,
            "value":       value,
        }

    return data


# ==============================
# HELPERS INTERNOS
# ==============================

def limpiar_placeholder(texto):
    """Convierte '{{ Voltaje }}' → 'Voltaje'"""
    return re.sub(r"[{}]", "", texto).strip()


def _es_nulo(valor):
    """Devuelve True si el valor es NaN, None o string vacío."""
    try:
        return pd.isna(valor)
    except Exception:
        return valor is None or str(valor).strip() == ""


def _convertir_a_texto(valor):
    """Convierte el valor de celda a string, respetando None."""
    if _es_nulo(valor):
        return None
    if isinstance(valor, float) and valor == int(valor):
        return str(int(valor))
    return str(valor)


def _leer_dim(row, col_name):
    """Lee una dimensión numérica de una fila; devuelve el default si está vacía."""
    val = row.get(col_name)
    if _es_nulo(val):
        return DEFAULT_IMAGE_WIDTH_PT if col_name == "width_pt" else DEFAULT_IMAGE_HEIGHT_PT
    try:
        return int(float(val))
    except Exception:
        return DEFAULT_IMAGE_WIDTH_PT if col_name == "width_pt" else DEFAULT_IMAGE_HEIGHT_PT


# ==============================
# REEMPLAZO DE TEXTO EN GOOGLE DOCS
# ==============================

def reemplazar_textos(doc_id, diccionario, docs_service):
    """
    Reemplaza todos los placeholders de tipo 'text' en un Google Doc,
    preservando el formato existente del documento.

    Usa batchUpdate con replaceAllText, que respeta el formato del párrafo
    donde vive el placeholder (fuente, tamaño, color, negrita, etc.).

    Args:
        doc_id       (str):  ID del documento de Google Docs.
        diccionario  (dict): Diccionario unificado generado por cargar_diccionario().
        docs_service:        Servicio autenticado de Google Docs API.

    Returns:
        dict: {"reemplazados": [...], "omitidos": [...]}
    """
    reemplazados = []
    omitidos     = []
    requests     = []

    for alias, entrada in diccionario.items():

        if entrada.get("type") != "text":
            continue

        placeholder = entrada.get("placeholder", "").strip()
        value       = entrada.get("value")

        if not placeholder:
            omitidos.append({"alias": alias, "razon": "placeholder vacío"})
            continue

        texto_reemplazo = "" if value is None else str(value)

        requests.append({
            "replaceAllText": {
                "containsText": {
                    "text":      placeholder,
                    "matchCase": True
                },
                "replaceText": texto_reemplazo
            }
        })

        reemplazados.append({
            "alias":       alias,
            "placeholder": placeholder,
            "value":       texto_reemplazo
        })

    if not requests:
        print("⚠️  No se encontraron entradas de tipo 'text' para reemplazar.")
        return {"reemplazados": [], "omitidos": omitidos}

    resultado = docs_service.documents().batchUpdate(
        documentId=doc_id,
        body={"requests": requests}
    ).execute()

    # Asociar ocurrencias reales (las replies vienen en el mismo orden que los requests)
    for i, respuesta in enumerate(resultado.get("replies", [])):
        ocurrencias = respuesta.get("replaceAllText", {}).get("occurrencesChanged", 0)
        reemplazados[i]["ocurrencias"] = ocurrencias
        if ocurrencias == 0:
            omitidos.append({
                "alias": reemplazados[i]["alias"],
                "razon": "placeholder no encontrado en el documento"
            })

    reemplazados_encontrados = [r for r in reemplazados if r.get("ocurrencias", 0) > 0]

    print(f"✅ Texto reemplazado: {len(reemplazados_encontrados)} placeholders")
    for r in reemplazados_encontrados:
        print(f"   {r['placeholder']} → '{r['value']}' ({r['ocurrencias']} ocurrencia/s)")

    if omitidos:
        print(f"⚠️  Omitidos: {len(omitidos)}")
        for o in omitidos:
            print(f"   {o['alias']}: {o['razon']}")

    return {
        "reemplazados": reemplazados_encontrados,
        "omitidos":     omitidos
    }


# ==============================
# CONDICIONALES EN GOOGLE DOCS
# ==============================

def procesar_condicionales(doc_id, diccionario, docs_service):
    """
    Procesa bloques condicionales {% if variable %}...{% endif %} en un Google Doc.

    - Si la variable es True  → elimina solo las etiquetas, conserva el contenido
    - Si la variable es False → elimina el bloque completo incluyendo etiquetas
    - Soporta {% else %} opcional
    - Opera de atrás hacia adelante para no desplazar índices

    Args:
        doc_id       (str):  ID del documento de Google Docs.
        diccionario  (dict): Diccionario unificado generado por cargar_diccionario().
        docs_service:        Servicio autenticado de Google Docs API.

    Returns:
        dict: {"procesados": [...], "omitidos": [...]}
    """
    import re

    # -------------------------------------------------------
    # 1. Obtener el documento y reconstruir texto plano
    #    con mapa de índices reales de la Docs API
    # -------------------------------------------------------
    documento = docs_service.documents().get(documentId=doc_id).execute()
    contenido = documento.get("body", {}).get("content", [])

    texto_plano, mapa_indices = _extraer_texto_con_indices(contenido)

    # -------------------------------------------------------
    # 2. Encontrar todos los bloques {% if %}...{% endif %}
    # -------------------------------------------------------
    patron = re.compile(
        r"\{%-?\s*if\s+(\w+)\s*-?%\}"  # {% if variable %}
        r"(.*?)"                         # contenido interno
        r"\{%-?\s*endif\s*-?%\}",       # {% endif %}
        re.DOTALL
    )

    bloques = list(patron.finditer(texto_plano))

    if not bloques:
        print("ℹ️  No se encontraron bloques condicionales en el documento.")
        return {"procesados": [], "omitidos": []}

    procesados        = []
    omitidos          = []
    rangos_a_eliminar = []  # lista de (start_doc, end_doc) en índices de Docs API

    for bloque in bloques:
        nombre_variable  = bloque.group(1).strip()
        contenido_bloque = bloque.group(2)

        # Buscar la variable en el diccionario
        entrada = diccionario.get(nombre_variable)
        if entrada is None:
            omitidos.append({
                "variable": nombre_variable,
                "razon":    "variable no encontrada en el diccionario"
            })
            continue

        valor        = entrada.get("value")
        es_verdadero = _evaluar_booleano(valor)

        # Posiciones en el texto plano
        inicio_bloque = bloque.start()    # inicio de {% if %}
        fin_bloque    = bloque.end()      # fin de {% endif %}
        fin_if        = bloque.start(2)   # fin de la etiqueta {% if variable %}
        inicio_endif  = bloque.end(2)     # inicio de {% endif %}

        # Detectar si hay {% else %}
        patron_else = re.compile(r"\{%-?\s*else\s*-?%\}")
        match_else  = patron_else.search(contenido_bloque)

        if match_else:
            inicio_else_abs = bloque.start(2) + match_else.start()
            fin_else_abs    = bloque.start(2) + match_else.end()

            if es_verdadero:
                # Conservar rama if → eliminar etiqueta {% if %} y desde {% else %} hasta fin
                rangos_a_eliminar.append((_idx(inicio_bloque, mapa_indices), _idx(fin_if, mapa_indices)))
                rangos_a_eliminar.append((_idx(inicio_else_abs, mapa_indices), _idx(fin_bloque, mapa_indices)))
            else:
                # Conservar rama else → eliminar desde {% if %} hasta fin de {% else %}, y {% endif %}
                rangos_a_eliminar.append((_idx(inicio_bloque, mapa_indices), _idx(fin_else_abs, mapa_indices)))
                rangos_a_eliminar.append((_idx(inicio_endif, mapa_indices), _idx(fin_bloque, mapa_indices)))

        else:
            if es_verdadero:
                # Conservar contenido → eliminar solo las dos etiquetas
                rangos_a_eliminar.append((_idx(inicio_bloque, mapa_indices), _idx(fin_if, mapa_indices)))
                rangos_a_eliminar.append((_idx(inicio_endif, mapa_indices), _idx(fin_bloque, mapa_indices)))
            else:
                # Eliminar bloque completo
                rangos_a_eliminar.append((_idx(inicio_bloque, mapa_indices), _idx(fin_bloque, mapa_indices)))

        procesados.append({
            "variable":     nombre_variable,
            "valor":        valor,
            "es_verdadero": es_verdadero,
            "tiene_else":   match_else is not None
        })

    # -------------------------------------------------------
    # 3. Ejecutar eliminaciones de atrás hacia adelante
    # -------------------------------------------------------
    if rangos_a_eliminar:
        rangos_validos = sorted(
            [(s, e) for s, e in rangos_a_eliminar if s is not None and e is not None and s < e],
            key=lambda r: r[0],
            reverse=True
        )

        requests = [
            {"deleteContentRange": {"range": {"startIndex": s, "endIndex": e}}}
            for s, e in rangos_validos
        ]

        if requests:
            docs_service.documents().batchUpdate(
                documentId=doc_id,
                body={"requests": requests}
            ).execute()

    # -------------------------------------------------------
    # 4. Log de resultados
    # -------------------------------------------------------
    print(f"✅ Condicionales procesadas: {len(procesados)}")
    for p in procesados:
        icono    = "✔" if p["es_verdadero"] else "✘"
        else_str = " (con else)" if p["tiene_else"] else ""
        print(f"   {icono} {{% if {p['variable']} %}} → {p['valor']}{else_str}")

    if omitidos:
        print(f"⚠️  Omitidos: {len(omitidos)}")
        for o in omitidos:
            print(f"   {o['variable']}: {o['razon']}")

    return {"procesados": procesados, "omitidos": omitidos}


# ==============================
# HELPERS INTERNOS
# ==============================

def _extraer_texto_con_indices(contenido):
    """
    Recorre el body del documento y construye:
    - texto_plano  (str):  todo el texto concatenado
    - mapa_indices (list): posición N en texto_plano → índice real en Docs API
    """
    texto_plano  = ""
    mapa_indices = []

    def recorrer(elementos):
        nonlocal texto_plano, mapa_indices
        for elemento in elementos:
            if "paragraph" in elemento:
                for run in elemento["paragraph"].get("elements", []):
                    contenido_run = run.get("textRun", {}).get("content", "")
                    start_index   = run.get("startIndex", 0)
                    for i, char in enumerate(contenido_run):
                        mapa_indices.append(start_index + i)
                        texto_plano += char
            elif "table" in elemento:
                for fila in elemento["table"].get("tableRows", []):
                    for celda in fila.get("tableCells", []):
                        recorrer(celda.get("content", []))

    recorrer(contenido)
    return texto_plano, mapa_indices


def _idx(pos, mapa_indices):
    """Convierte posición en texto plano a índice real de Docs API."""
    if pos < 0 or pos >= len(mapa_indices):
        return None
    return mapa_indices[pos]


def _evaluar_booleano(valor):
    """Convierte el valor de la variable condicional a booleano."""
    if valor is None:
        return False
    if isinstance(valor, bool):
        return valor
    return str(valor).strip().lower() in ("true", "1", "si", "yes", "sí")


# ==============================
# IMÁGENES EN GOOGLE DOCS
# ==============================

def reemplazar_imagenes(doc_id, diccionario, docs_service, drive_service):
    """
    Reemplaza placeholders de tipo 'image' en un Google Doc.

    Flujo por cada imagen:
        1. Hace el archivo de Drive temporalmente público
        2. Localiza el placeholder en el documento y obtiene su índice
        3. Borra el texto del placeholder
        4. Inserta la imagen en esa posición usando la URL pública
        5. Revoca el acceso público inmediatamente

    Opera de atrás hacia adelante para no desplazar índices.

    Args:
        doc_id        (str):  ID del documento de Google Docs.
        diccionario   (dict): Diccionario unificado generado por cargar_diccionario().
        docs_service:         Servicio autenticado de Google Docs API.
        drive_service:        Servicio autenticado de Google Drive API.

    Returns:
        dict: {"reemplazados": [...], "omitidos": [...]}
    """
    reemplazados = []
    omitidos     = []

    # Filtrar solo entradas de tipo image
    entradas_imagen = [
        (alias, entrada)
        for alias, entrada in diccionario.items()
        if entrada.get("type") == "image"
    ]

    if not entradas_imagen:
        print("ℹ️  No se encontraron entradas de tipo 'image' para reemplazar.")
        return {"reemplazados": [], "omitidos": omitidos}

    for alias, entrada in entradas_imagen:
        placeholder = entrada.get("placeholder", "").strip()
        valor       = entrada["value"]
        raw_file_id = valor.get("file_id")
        if not raw_file_id:
            omitidos.append({"alias": alias, "razon": "file_id vacío o no definido"})
            continue
        file_id = extraer_id_gdoc(raw_file_id)
        width_pt    = valor.get("width_pt",  DEFAULT_IMAGE_WIDTH_PT)
        height_pt   = valor.get("height_pt", DEFAULT_IMAGE_HEIGHT_PT)

        if not placeholder:
            omitidos.append({"alias": alias, "razon": "placeholder vacío"})
            continue

        permission_id = None

        try:
            # -------------------------------------------------
            # 1. Hacer el archivo público temporalmente
            # -------------------------------------------------
            permiso = drive_service.permissions().create(
                fileId=file_id,
                body={"type": "anyone", "role": "reader"},
                fields="id"
            ).execute()
            permission_id = permiso.get("id")

            url_publica = f"https://drive.google.com/uc?export=download&id={file_id}"

            # -------------------------------------------------
            # 2. Leer el documento y localizar el placeholder
            # -------------------------------------------------
            documento = docs_service.documents().get(documentId=doc_id).execute()
            contenido = documento.get("body", {}).get("content", [])
            texto_plano, mapa_indices = _extraer_texto_con_indices(contenido)

            pos = texto_plano.find(placeholder)
            if pos == -1:
                omitidos.append({"alias": alias, "razon": "placeholder no encontrado en el documento"})
                continue

            # Índices reales en la Docs API
            start_idx = _idx(pos, mapa_indices)
            end_idx   = _idx(pos + len(placeholder), mapa_indices)

            if start_idx is None or end_idx is None:
                omitidos.append({"alias": alias, "razon": "no se pudo mapear el índice del placeholder"})
                continue

            # -------------------------------------------------
            # 3. Borrar el placeholder e insertar la imagen
            #    en un solo batchUpdate
            # -------------------------------------------------
            docs_service.documents().batchUpdate(
                documentId=doc_id,
                body={"requests": [
                    # Primero borrar el texto del placeholder
                    {
                        "deleteContentRange": {
                            "range": {
                                "startIndex": start_idx,
                                "endIndex":   end_idx
                            }
                        }
                    },
                    # Luego insertar la imagen en la misma posición
                    {
                        "insertInlineImage": {
                            "location":  {"index": start_idx},
                            "uri":       url_publica,
                            "objectSize": {
                                "width":  {"magnitude": width_pt,  "unit": "PT"},
                                "height": {"magnitude": height_pt, "unit": "PT"}
                            }
                        }
                    }
                ]}
            ).execute()

            reemplazados.append({
                "alias":       alias,
                "placeholder": placeholder,
                "file_id":     file_id,
                "width_pt":    width_pt,
                "height_pt":   height_pt
            })

        except Exception as e:
            omitidos.append({
                "alias": alias,
                "razon": f"error al procesar: {str(e)}"
            })

        finally:
            # -------------------------------------------------
            # 4. Revocar permiso público siempre, incluso si hubo error
            # -------------------------------------------------
            if permission_id:
                try:
                    drive_service.permissions().delete(
                        fileId=file_id,
                        permissionId=permission_id
                    ).execute()
                except Exception:
                    pass  # Si falla la revocación no interrumpimos el flujo

    # Log de resultados
    print(f"✅ Imágenes reemplazadas: {len(reemplazados)}")
    for r in reemplazados:
        print(f"   {r['placeholder']} → {r['file_id']} ({r['width_pt']}x{r['height_pt']} pt)")

    if omitidos:
        print(f"⚠️  Omitidos: {len(omitidos)}")
        for o in omitidos:
            print(f"   {o['alias']}: {o['razon']}")

    return {"reemplazados": reemplazados, "omitidos": omitidos}


# ==============================
# GOOGLE AUTH
# ==============================
import os
from google.oauth2 import service_account, credentials as oauth2_credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

SCOPES = [
    'https://www.googleapis.com/auth/documents',
    'https://www.googleapis.com/auth/drive',
    'https://www.googleapis.com/auth/spreadsheets'
]


def autenticar_oauth(ruta_client_secret=None, ruta_token=None):
    """
    Autenticación OAuth2 con cuenta personal de Google.

    Las rutas se pueden pasar como argumento o desde variables de entorno:
        GOOGLE_OAUTH_CLIENT → ruta al client_secret.json
        GOOGLE_OAUTH_TOKEN  → ruta donde guardar/leer el token (opcional)

    El token se renueva automáticamente si expiró.
    Si no existe, abre el navegador para autorizar por primera vez.
    """
    if ruta_client_secret is None:
        ruta_client_secret = os.getenv("GOOGLE_OAUTH_CLIENT")
        if not ruta_client_secret:
            raise Exception("Define la variable de entorno GOOGLE_OAUTH_CLIENT con la ruta al client_secret.json")

    if not os.path.exists(ruta_client_secret):
        raise FileNotFoundError(f"No existe el archivo: {ruta_client_secret}")

    if ruta_token is None:
        ruta_token = os.getenv(
            "GOOGLE_OAUTH_TOKEN",
            os.path.join(os.path.dirname(ruta_client_secret), "token.json")
        )

    creds = None

    if os.path.exists(ruta_token):
        creds = oauth2_credentials.Credentials.from_authorized_user_file(ruta_token, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(ruta_client_secret, SCOPES)
            creds = flow.run_local_server(port=0)

        with open(ruta_token, "w") as f:
            f.write(creds.to_json())

    docs_service   = build('docs',   'v1', credentials=creds)
    drive_service  = build('drive',  'v3', credentials=creds)
    sheets_service = build('sheets', 'v4', credentials=creds)

    return docs_service, drive_service, sheets_service


def autenticar_servicio(ruta_credenciales):
    """
    Autenticación con service account (para cuando se configure la Unidad Compartida).
    """
    creds = service_account.Credentials.from_service_account_file(
        ruta_credenciales,
        scopes=SCOPES
    )
    docs_service   = build('docs',   'v1', credentials=creds)
    drive_service  = build('drive',  'v3', credentials=creds)
    sheets_service = build('sheets', 'v4', credentials=creds)

    return docs_service, drive_service, sheets_service


# ==============================
# OPERACIONES DE DRIVE
# ==============================

def descargar_excel_drive(file_id, drive_service):
    """
    Descarga un archivo de Google Drive por file_id y lo retorna como BytesIO.
    """
    request = drive_service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)

    done = False
    while not done:
        status, done = downloader.next_chunk()

    fh.seek(0)
    return fh


def extraer_id_gdoc(url_o_id):
    """
    Acepta una URL de Google Docs/Drive o directamente un ID y devuelve el file ID.

    Ejemplos aceptados:
        https://docs.google.com/document/d/FILE_ID/edit
        https://drive.google.com/drive/folders/FILE_ID
        FILE_ID (directamente)
    """
    match = re.search(r"/(?:d|folders)/([a-zA-Z0-9_-]+)", url_o_id)
    if match:
        return match.group(1)
    return url_o_id.strip()


def copiar_documento(doc_id, nombre_nuevo, drive_service, carpeta_destino_id=None):
    """
    Crea una copia del documento en Drive y devuelve el ID del nuevo archivo.

    Args:
        doc_id             (str): ID del documento original.
        nombre_nuevo       (str): Nombre para la copia.
        drive_service:           Servicio autenticado de Google Drive API.
        carpeta_destino_id (str): ID de la carpeta destino. Si se omite,
                                  la copia queda en la raíz del Drive del usuario.
    Returns:
        str: ID del documento copiado.
    """
    body = {"name": nombre_nuevo}
    if carpeta_destino_id:
        body["parents"] = [carpeta_destino_id]

    copia = drive_service.files().copy(
        fileId=doc_id,
        body=body,
        supportsAllDrives=True
    ).execute()

    return copia["id"]


def listar_archivos_drive(drive_service, page_size=10):
    """Lista archivos visibles por la cuenta autenticada."""
    results = drive_service.files().list(
        pageSize=page_size,
        fields="files(id, name)"
    ).execute()
    return results.get('files', [])


# ==============================
# UTILIDADES
# ==============================

def obtener_ruta_credenciales():
    """Obtiene la ruta del JSON de service account desde variable de entorno GOOGLE_CREDS."""
    ruta = os.getenv("GOOGLE_CREDS")
    if not ruta:
        raise Exception("Debes definir la variable de entorno GOOGLE_CREDS")
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No existe el archivo: {ruta}")
    return ruta
