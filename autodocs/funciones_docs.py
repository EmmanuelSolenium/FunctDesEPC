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
# VERBOSE / LOGGING
# ==============================
VERBOSE = False

def _log(*args, **kwargs):
    """Imprime solo si VERBOSE está activado."""
    if VERBOSE:
        print(*args, **kwargs)


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
        if archivo.endswith(".csv"):
            df = pd.read_csv(archivo)
        else:
            df = pd.read_excel(archivo, header=None)
    else:
        df = pd.read_excel(archivo, header=None)

    # Detectar fila de encabezado: buscar la primera fila con 'Nombre del dato'
    header_idx = 0
    for i, row in df.iterrows():
        if any(str(v).strip().lower() == "nombre del dato" for v in row.values):
            header_idx = i
            break

    df.columns = df.iloc[header_idx]
    df = df[header_idx + 1:].reset_index(drop=True)

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
    tiene_sheet      = "sheet"      in df.columns
    tiene_header_row = "header_row" in df.columns

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
            raw_url = str(raw_value).strip() if not _es_nulo(raw_value) else None
            sheet      = str(row.get("sheet", "")).strip() if tiene_sheet else None
            sheet      = sheet if sheet and sheet.lower() != "nan" else None
            header_row = None
            if tiene_header_row:
                hr = row.get("header_row")
                if not _es_nulo(hr):
                    try:
                        header_row = int(float(hr))
                    except Exception:
                        header_row = None
            value = {"file_id": raw_url, "sheet": sheet, "header_row": header_row} if raw_url else None

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
        _log("⚠️  No se encontraron entradas de tipo 'text' para reemplazar.")
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

    _log(f"✅ Texto reemplazado: {len(reemplazados_encontrados)} placeholders")
    for r in reemplazados_encontrados:
        _log(f"   {r['placeholder']} → '{r['value']}' ({r['ocurrencias']} ocurrencia/s)")

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
        _log("ℹ️  No se encontraron bloques condicionales en el documento.")
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
    _log(f"✅ Condicionales procesadas: {len(procesados)}")
    for p in procesados:
        icono    = "✔" if p["es_verdadero"] else "✘"
        else_str = " (con else)" if p["tiene_else"] else ""
        _log(f"   {icono} {{% if {p['variable']} %}} → {p['valor']}{else_str}")

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
        _log("ℹ️  No se encontraron entradas de tipo 'image' para reemplazar.")
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
    _log(f"✅ Imágenes reemplazadas: {len(reemplazados)}")
    for r in reemplazados:
        _log(f"   {r['placeholder']} → {r['file_id']} ({r['width_pt']}x{r['height_pt']} pt)")

    omitidos_reales = [o for o in omitidos if o.get("razon") != "placeholder no encontrado en el documento"]
    if omitidos_reales:
        _log(f"⚠️  Imágenes omitidas: {len(omitidos_reales)}")
        for o in omitidos_reales:
            _log(f"   {o['alias']}: {o['razon']}")

    return {"reemplazados": reemplazados, "omitidos": omitidos}


# ==============================
# TABLAS EN GOOGLE DOCS
# ==============================







# ==============================
# TABLAS EN GOOGLE DOCS
# ==============================

def reemplazar_tablas(doc_id, diccionario, docs_service, drive_service):
    """
    Reemplaza placeholders de tipo 'table' en un Google Doc.

    Enfoque: sin copiar formato. Lee los datos del Excel/Sheets y los pega
    en una tabla nueva con formato predeterminado de Google Docs.

    Sintaxis en el documento (sin tabla plantilla):
        {% table alias %}
        {{ alias }}
        {% end_table alias %}

    Flujo por cada tabla:
        1. Encuentra el bloque {% table %}...{% end_table %}
        2. Descarga los datos desde Excel o Google Sheets (URL en el diccionario)
        3. Elimina todo el contenido del bloque (tags + placeholder)
        4. Inserta una tabla nueva en esa posición
        5. Llena las celdas con los datos (encabezados en fila 1, datos en el resto)

    Args:
        doc_id        (str):  ID del documento de Google Docs.
        diccionario   (dict): Diccionario unificado generado por cargar_diccionario().
        docs_service:         Servicio autenticado de Google Docs API.
        drive_service:        Servicio autenticado de Google Drive API.

    Returns:
        dict: {"reemplazados": [...], "omitidos": [...]}
    """
    import re

    reemplazados = []
    omitidos     = []

    entradas_tabla = [
        (alias, entrada)
        for alias, entrada in diccionario.items()
        if entrada.get("type") == "table"
        and entrada.get("value") is not None
    ]

    if not entradas_tabla:
        _log("ℹ️  No se encontraron entradas de tipo 'table' para reemplazar.")
        return {"reemplazados": [], "omitidos": omitidos}

    for alias, entrada in entradas_tabla:
        placeholder = entrada.get("placeholder", "").strip()
        valor       = entrada["value"]
        url_tabla   = valor.get("file_id")
        sheet_name  = valor.get("sheet")
        header_row  = valor.get("header_row")

        if not url_tabla:
            omitidos.append({"alias": alias, "razon": "URL de tabla vacía"})
            continue

        if not placeholder:
            omitidos.append({"alias": alias, "razon": "placeholder vacío"})
            continue

        try:
            # -------------------------------------------------
            # 1. Descargar datos y formato desde Excel o Google Sheets
            # -------------------------------------------------
            df, cell_formats, merges = _cargar_datos_tabla(url_tabla, sheet_name, drive_service, header_row)
            if df is None or df.empty:
                omitidos.append({"alias": alias, "razon": "no se pudieron cargar los datos"})
                continue

            n_filas = len(df) + 1   # +1 por fila de encabezados
            n_cols  = len(df.columns)

            # -------------------------------------------------
            # 2. Leer documento y localizar el bloque {% table %}
            # -------------------------------------------------
            documento = docs_service.documents().get(documentId=doc_id).execute()
            contenido = documento.get("body", {}).get("content", [])
            texto_plano, mapa_indices = _extraer_texto_con_indices(contenido)

            patron_bloque = re.compile(
                r"\{%-?\s*table\s+" + re.escape(alias) + r"\s*-?%\}"
                r"(.*?)"
                r"\{%-?\s*end_table\s+" + re.escape(alias) + r"\s*-?%\}",
                re.DOTALL
            )
            match_bloque = patron_bloque.search(texto_plano)
            if not match_bloque:
                omitidos.append({"alias": alias, "razon": f"bloque {{% table {alias} %}} no encontrado"})
                continue

            inicio_bloque = _idx(match_bloque.start(), mapa_indices)
            fin_bloque    = _idx(match_bloque.end() - 1, mapa_indices)
            if fin_bloque is None:
                fin_bloque = mapa_indices[-1] if mapa_indices else 0
            fin_bloque += 1  # endIndex es exclusivo en la API

            # -------------------------------------------------
            # 3. Eliminar todo el bloque e insertar tabla vacía
            #    en un solo batchUpdate
            # -------------------------------------------------
            docs_service.documents().batchUpdate(
                documentId=doc_id,
                body={"requests": [
                    {"deleteContentRange": {
                        "range": {
                            "startIndex": inicio_bloque,
                            "endIndex":   fin_bloque
                        }
                    }},
                    {"insertTable": {
                        "rows":     n_filas,
                        "columns":  n_cols,
                        "location": {"index": inicio_bloque}
                    }}
                ]}
            ).execute()

            # -------------------------------------------------
            # 3b. Aplicar merges del Excel a la tabla recién insertada
            # -------------------------------------------------
            # Releer para obtener el startIndex real de la tabla recién insertada
            doc_tmp   = docs_service.documents().get(documentId=doc_id).execute()
            tabla_tmp = _encontrar_tabla_cerca(doc_tmp.get("body", {}).get("content", []), inicio_bloque)
            if not tabla_tmp:
                omitidos.append({"alias": alias, "razon": "no se pudo localizar la tabla recién insertada"})
                continue
            tabla_start_real = tabla_tmp["startIndex"]

            if merges:
                # Aplicar merges de derecha a izquierda y de abajo hacia arriba.
                # Cada merge se aplica con verificación post-merge: se relee la
                # tabla y se confirma que el columnSpan esperado quedó aplicado;
                # si no, se reintenta hasta MAX_REINTENTOS veces.
                MAX_REINTENTOS = 3
                merges_ordenados = sorted(merges, key=lambda m: (-m["min_row"], -m["min_col"]))

                tabla_start_actual = tabla_start_real

                for merge in merges_ordenados:
                    aplicado = False
                    for intento in range(1, MAX_REINTENTOS + 1):
                        # Releer y localizar la tabla actual
                        doc_m = docs_service.documents().get(documentId=doc_id).execute()
                        tabla_m = _encontrar_tabla_cerca(doc_m.get("body", {}).get("content", []), inicio_bloque)
                        if not tabla_m:
                            print(f"[MERGE] No se encontró la tabla. Abortando merges restantes.")
                            break
                        tabla_start_actual = tabla_m["startIndex"]

                        # Aplicar el merge
                        try:
                            docs_service.documents().batchUpdate(
                                documentId=doc_id,
                                body={"requests": [{
                                    "mergeTableCells": {
                                        "tableRange": {
                                            "tableCellLocation": {
                                                "tableStartLocation": {"index": tabla_start_actual},
                                                "rowIndex":    merge["min_row"],
                                                "columnIndex": merge["min_col"]
                                            },
                                            "rowSpan":    merge["row_span"],
                                            "columnSpan": merge["col_span"]
                                        }
                                    }
                                }]}
                            ).execute()
                        except Exception as e:
                            print(f"[MERGE] Error al aplicar merge "
                                  f"(row={merge['min_row']}, col={merge['min_col']}, "
                                  f"rowSpan={merge['row_span']}, colSpan={merge['col_span']}): {e}")

                        # Verificar que el merge se aplicó releyendo la tabla
                        doc_v = docs_service.documents().get(documentId=doc_id).execute()
                        tabla_v = _encontrar_tabla_cerca(doc_v.get("body", {}).get("content", []), inicio_bloque)
                        if not tabla_v:
                            break

                        filas_v = tabla_v.get("tableRows", [])
                        if merge["min_row"] >= len(filas_v):
                            break

                        # Recorrer celdas físicas hasta encontrar la columna lógica esperada
                        celdas_v = filas_v[merge["min_row"]].get("tableCells", [])
                        col_off = 0
                        span_real = None
                        for c in celdas_v:
                            sp = c.get("tableCellStyle", {}).get("columnSpan", 1)
                            if col_off == merge["min_col"]:
                                span_real = sp
                                break
                            col_off += sp

                        if span_real == merge["col_span"]:
                            aplicado = True
                            if intento > 1:
                                print(f"[MERGE] OK tras {intento} intentos "
                                      f"(row={merge['min_row']}, col={merge['min_col']}, "
                                      f"colSpan={merge['col_span']})")
                            break
                        else:
                            print(f"[MERGE] Reintento {intento}/{MAX_REINTENTOS}: "
                                  f"se esperaba colSpan={merge['col_span']} en "
                                  f"row={merge['min_row']}, col={merge['min_col']}; "
                                  f"se obtuvo span_real={span_real}")

                    if not aplicado:
                        print(f"[MERGE] FALLO definitivo tras {MAX_REINTENTOS} intentos: "
                              f"row={merge['min_row']}, col={merge['min_col']}, "
                              f"colSpan={merge['col_span']}, rowSpan={merge['row_span']}")


            # Releer DESPUÉS de aplicar los merges para obtener la estructura real post-merge
            documento2 = docs_service.documents().get(documentId=doc_id).execute()
            contenido2 = documento2.get("body", {}).get("content", [])
            # Usar el último tabla_start_actual conocido para localizar la tabla correcta.
            # Si no hubo merges, caer a tabla_start_real como referencia.
            ref = tabla_start_actual if merges else tabla_start_real
            tabla_nueva = _encontrar_tabla_cerca(contenido2, ref)
            if not tabla_nueva:
                omitidos.append({"alias": alias, "razon": "no se pudo localizar la tabla recién insertada"})
                continue
            print(f"[DEBUG2] tabla_nueva startIndex={tabla_nueva['startIndex']}, "
                  f"ref={ref}, "
                  f"filas={len(tabla_nueva.get('tableRows',[]))}, "
                  f"cols_fila0={len(tabla_nueva.get('tableRows',[[]])[0].get('tableCells',[]))}")

            # -------------------------------------------------
            # 7. Llenar celdas con datos (de atrás hacia adelante).
            #    Las filas con merges tienen menos celdas físicas;
            #    se usa col_offset para mapear celda física → col Excel.
            # -------------------------------------------------
            todas_las_filas = [list(df.columns)] + df.values.tolist()
            requests_texto  = []

            for fi in range(len(todas_las_filas) - 1, -1, -1):
                fila_doc   = tabla_nueva.get("tableRows", [])[fi]
                fila_datos = todas_las_filas[fi]
                celdas_doc = fila_doc.get("tableCells", [])

                # Construir lista (celda_doc, col_excel) de atrás hacia adelante
                pares = []
                col_offset = 0
                for celda in celdas_doc:
                    span = celda.get("tableCellStyle", {}).get("columnSpan", 1)
                    if celda.get("content"):  # omitir celdas absorbidas (content=[])
                        pares.append((celda, col_offset))
                    col_offset += span

                for celda, col_excel in reversed(pares):
                    texto = str(fila_datos[col_excel]) if col_excel < len(fila_datos) else ""
                    if not texto or texto in ("nan", "None"):
                        continue

                    parrafos  = celda.get("content", [])
                    elementos = parrafos[0].get("paragraph", {}).get("elements", []) if parrafos else []
                    if not elementos:
                        continue

                    idx_insercion = elementos[0].get("startIndex", 0)
                    requests_texto.append({
                        "insertText": {
                            "location": {"index": idx_insercion},
                            "text":     texto
                        }
                    })

            # ── DEBUG TEMPORAL ─────────────────────────────────────────
            print(f"\n[DEBUG] tabla_nueva tiene {len(tabla_nueva.get('tableRows',[]))} filas")
            fila0 = tabla_nueva.get("tableRows", [])[0]
            celdas0 = fila0.get("tableCells", [])
            print(f"[DEBUG] Fila 0: {len(celdas0)} celdas físicas")
            col_off = 0
            for i, c in enumerate(celdas0):
                span = c.get("tableCellStyle", {}).get("columnSpan", 1)
                paras = c.get("content", [])
                idx = paras[0].get("paragraph",{}).get("elements",[{}])[0].get("startIndex","?") if paras else "?"
                txt_mapped = str(todas_las_filas[0][col_off]) if col_off < len(todas_las_filas[0]) else "OOB"
                print(f"  celda[{i}]: span={span}, col_excel={col_off}, startIndex={idx}, texto_mapeado={repr(txt_mapped)}")
                col_off += span
            print(f"[DEBUG] requests_texto row0 entries:")
            for r in requests_texto:
                loc = r.get("insertText",{}).get("location",{}).get("index")
                txt = r.get("insertText",{}).get("text","")
                if txt in ("TENDIDOS","EQUIPOS","COORDENADAS","POSTE","CIMENTACIÓN","ARMADOS"):
                    print(f"  insertText idx={loc} text={repr(txt)}")
            # ── FIN DEBUG ───────────────────────────────────────────────

            if requests_texto:
                docs_service.documents().batchUpdate(
                    documentId=doc_id,
                    body={"requests": requests_texto}
                ).execute()

            # -------------------------------------------------
            # 6. Releer documento y aplicar formato del Excel
            #    (negrita, color de fondo, color de fuente)
            # -------------------------------------------------
            if cell_formats:
                documento3 = docs_service.documents().get(documentId=doc_id).execute()
                contenido3 = documento3.get("body", {}).get("content", [])
                tabla_fmt  = _encontrar_tabla_cerca(contenido3, ref)
                if tabla_fmt:
                    fmt_requests = _generar_requests_formato_excel(tabla_fmt, cell_formats)
                    if fmt_requests:
                        docs_service.documents().batchUpdate(
                            documentId=doc_id,
                            body={"requests": fmt_requests}
                        ).execute()

            reemplazados.append({
                "alias":   alias,
                "n_filas": n_filas,
                "n_cols":  n_cols
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            omitidos.append({"alias": alias, "razon": f"error: {str(e)}"})

    _log(f"✅ Tablas reemplazadas: {len(reemplazados)}")
    for r in reemplazados:
        _log(f"   {r['alias']} → {r['n_filas']} filas × {r['n_cols']} columnas")

    if omitidos:
        _log(f"⚠️  Tablas omitidas: {len(omitidos)}")
        for o in omitidos:
            _log(f"   {o['alias']}: {o['razon']}")

    return {"reemplazados": reemplazados, "omitidos": omitidos}


# ==============================
# HELPERS DE TABLAS
# ==============================

def _cargar_datos_tabla(url, sheet_name, drive_service, header_row=None):
    """
    Descarga datos, formato y merges desde un .xlsx en Drive o Google Sheets nativo.
    Devuelve (df, cell_formats, merges):
        df           : DataFrame con los datos (sin columnas fantasma)
        cell_formats : { (fila_doc, col_excel): {bold, bg_color, font_color, font_size} }
        merges       : [ {min_row, min_col, max_row, max_col} ] en coordenadas 0-based
                       relativas al header_idx (fila_doc 0 = primera fila de la tabla)
    """
    import io
    import pandas as pd
    import openpyxl
    from googleapiclient.http import MediaIoBaseDownload
    from googleapiclient.errors import HttpError

    file_id = extraer_id_gdoc(url)

    def _descargar(request):
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        fh.seek(0)
        return fh

    try:
        fh = _descargar(drive_service.files().get_media(fileId=file_id))
    except HttpError as e:
        if e.resp.status in (400, 403):
            fh = _descargar(drive_service.files().export_media(
                fileId=file_id,
                mimeType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            ))
        else:
            raise

    fh.seek(0)
    wb = openpyxl.load_workbook(fh, data_only=True)
    ws = wb[sheet_name] if sheet_name and sheet_name in wb.sheetnames else wb.active

    header_idx = (header_row - 1) if header_row and header_row > 1 else 0

    # ── Detectar última columna real (ignorar columnas fantasma vacías) ───────
    max_col_real = 0
    for fila in ws.iter_rows():
        for celda in fila:
            val = celda.value
            if val is not None and str(val).strip() not in ("", "'"):
                max_col_real = max(max_col_real, celda.column)
    if max_col_real == 0:
        max_col_real = ws.max_column

    # ── Extraer datos (solo hasta max_col_real) ───────────────────────────────
    filas_excel = list(ws.iter_rows(min_row=header_idx + 1, max_col=max_col_real, values_only=False))
    encabezados = [str(c.value) if c.value is not None else "" for c in filas_excel[0]]
    datos_filas = []
    for fila in filas_excel[1:]:
        datos_filas.append([str(c.value) if c.value is not None else "" for c in fila])

    df = pd.DataFrame(datos_filas, columns=encabezados)
    df = df.fillna("").astype(str)

    # ── Extraer formato (solo hasta max_col_real) ─────────────────────────────
    cell_formats = {}
    for doc_ri, fila in enumerate(filas_excel):
        for ci, celda in enumerate(fila):
            if ci >= max_col_real:
                break
            fmt = {}

            # Color de fondo
            fill = celda.fill
            if fill and fill.fill_type not in (None, "none"):
                color = fill.fgColor
                if color and color.type == "rgb" and color.rgb not in ("00000000", "FFFFFFFF", "FF000000"):
                    fmt["bg_color"] = _hex_argb_a_rgb_float(color.rgb)

            # Negrita
            if celda.font and celda.font.bold:
                fmt["bold"] = True

            # Tamaño de fuente (solo si está definido explícitamente)
            if celda.font and celda.font.size:
                fmt["font_size"] = float(celda.font.size)

            # Color de fuente (solo si no es negro por defecto)
            if celda.font and celda.font.color:
                fc = celda.font.color
                if fc.type == "rgb" and fc.rgb not in ("00000000", "FF000000"):
                    fmt["font_color"] = _hex_argb_a_rgb_float(fc.rgb)

            if fmt:
                cell_formats[(doc_ri, ci)] = fmt

    # ── Extraer merges (ajustados a header_idx y recortados a max_col_real) ───
    merges = []
    for rng in ws.merged_cells.ranges:
        # Convertir a 0-based y relativos a header_idx
        doc_min_row = rng.min_row - 1 - header_idx
        doc_max_row = rng.max_row - 1 - header_idx
        min_col     = rng.min_col - 1
        max_col     = min(rng.max_col - 1, max_col_real - 1)
        # Solo incluir merges dentro del rango visible
        if doc_min_row >= 0 and min_col < max_col_real:
            merges.append({
                "min_row":  doc_min_row,
                "max_row":  doc_max_row,
                "min_col":  min_col,
                "max_col":  max_col,
                "row_span": doc_max_row - doc_min_row + 1,
                "col_span": max_col - min_col + 1,
            })

    return df, cell_formats, merges


def _hex_argb_a_rgb_float(hex_argb):
    """Convierte 'FF00FF00' → {'red': 0.0, 'green': 1.0, 'blue': 0.0}"""
    hex_rgb = hex_argb[-6:]
    r = int(hex_rgb[0:2], 16) / 255
    g = int(hex_rgb[2:4], 16) / 255
    b = int(hex_rgb[4:6], 16) / 255
    return {"red": round(r, 4), "green": round(g, 4), "blue": round(b, 4)}


def _generar_requests_formato_excel(tabla_nueva, cell_formats):
    """
    Genera requests de updateTableCellStyle y updateTextStyle.
    Mapea (fila_doc, col_doc_ajustado) → celda del Doc, teniendo en cuenta
    que las filas con merges tienen menos celdas físicas.
    """
    requests  = []
    tabla_id  = tabla_nueva.get("startIndex", 0)
    filas_doc = tabla_nueva.get("tableRows", [])

    for (fi, ci_excel), fmt in cell_formats.items():
        if fi >= len(filas_doc):
            continue

        # Resolver el índice de celda física en el Doc teniendo en cuenta merges
        # Cada celda puede tener colSpan > 1, lo que desplaza el ci real
        celdas_doc = filas_doc[fi].get("tableCells", [])
        celda_doc  = None
        ci_doc     = None
        col_offset = 0
        for idx, c in enumerate(celdas_doc):
            span = c.get("tableCellStyle", {}).get("columnSpan", 1)
            if col_offset == ci_excel:
                if c.get("content"):  # omitir celdas absorbidas (content=[])
                    celda_doc = c
                    ci_doc    = idx
                break
            col_offset += span

        if celda_doc is None:
            continue

        # ── Color de fondo ───────────────────────────────────────────────────
        if "bg_color" in fmt:
            col_span = celda_doc.get("tableCellStyle", {}).get("columnSpan", 1)
            requests.append({
                "updateTableCellStyle": {
                    "tableRange": {
                        "tableCellLocation": {
                            "tableStartLocation": {"index": tabla_id},
                            "rowIndex":    fi,
                            "columnIndex": ci_excel
                        },
                        "rowSpan":    1,
                        "columnSpan": col_span
                    },
                    "tableCellStyle": {
                        "backgroundColor": {
                            "color": {"rgbColor": fmt["bg_color"]}
                        }
                    },
                    "fields": "backgroundColor"
                }
            })

        # ── Texto: negrita, tamaño, color ────────────────────────────────────
        text_style  = {}
        text_fields = []

        if fmt.get("bold"):
            text_style["bold"] = True
            text_fields.append("bold")

        if "font_size" in fmt:
            text_style["fontSize"] = {"magnitude": fmt["font_size"], "unit": "PT"}
            text_fields.append("fontSize")

        if "font_color" in fmt:
            text_style["foregroundColor"] = {
                "color": {"rgbColor": fmt["font_color"]}
            }
            text_fields.append("foregroundColor")

        if text_style:
            parrafos  = celda_doc.get("content", [])
            elementos = parrafos[0].get("paragraph", {}).get("elements", []) if parrafos else []
            if elementos:
                ts_start = elementos[0].get("startIndex", 0)
                ts_end   = celda_doc.get("endIndex", 0) - 1
                if ts_start < ts_end:
                    requests.append({
                        "updateTextStyle": {
                            "range": {
                                "startIndex": ts_start,
                                "endIndex":   ts_end
                            },
                            "textStyle": text_style,
                            "fields":    ",".join(text_fields)
                        }
                    })

    return requests


def _encontrar_tabla_cerca(contenido, indice_referencia):
    """
    Encuentra la tabla cuyo startIndex esté más cercano (y no antes) al índice dado.
    """
    mejor     = None
    menor_dist = float("inf")
    for elemento in contenido:
        if "table" not in elemento:
            continue
        start = elemento.get("startIndex", 0)
        dist  = abs(start - indice_referencia)
        if dist < menor_dist:
            menor_dist = dist
            tabla      = elemento["table"]
            tabla["startIndex"] = start
            tabla["endIndex"]   = elemento.get("endIndex", 0)
            mejor = tabla
    return mejor


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