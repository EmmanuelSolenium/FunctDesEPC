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

    return {"reemplazados": reemplazados, "omitidos": omitidos}


# ==============================
# TABLAS EN GOOGLE DOCS
# ==============================


def reemplazar_tablas(doc_id, diccionario, docs_service, drive_service):
    """
    Reemplaza placeholders de tipo 'table' en un Google Doc.

    Sintaxis en el documento:
        {% table alias %}
        [tabla plantilla con formato]
        {{ alias }}
        {% end_table alias %}

    Flujo por cada tabla:
        1. Encuentra el bloque {% table %}...{% end_table %}
        2. Extrae el formato de la tabla plantilla (merges, colores, bordes, fuentes)
        3. Descarga los datos desde Excel o Google Sheets
        4. Inserta tabla nueva en la posición del placeholder
        5. Aplica formato copiado de la plantilla
        6. Aplica merges según las reglas definidas
        7. Elimina plantilla, placeholder y etiquetas

    Reglas de merge:
        - Merge de fila completa → se replica abarcando todas las columnas nuevas
        - Merge parcial → se replica en las mismas posiciones, columnas extra independientes

    Reglas de formato cuando datos tienen más columnas que la plantilla:
        - Columnas extra heredan formato de la última columna de la plantilla

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
        print("ℹ️  No se encontraron entradas de tipo 'table' para reemplazar.")
        return {"reemplazados": [], "omitidos": omitidos}

    for alias, entrada in entradas_tabla:
        placeholder = entrada.get("placeholder", "").strip()
        valor       = entrada["value"]
        url_tabla   = valor.get("file_id")
        sheet_name  = valor.get("sheet")

        print(f"\nDEBUG tabla [{alias}]: url={str(url_tabla)[:70]} | sheet={sheet_name} | placeholder={placeholder}")

        if not url_tabla:
            print(f"  → OMITIDA: URL de tabla vacía")
            omitidos.append({"alias": alias, "razon": "URL de tabla vacía"})
            continue

        if not placeholder:
            print(f"  → OMITIDA: placeholder vacío")
            omitidos.append({"alias": alias, "razon": "placeholder vacío"})
            continue

        try:
            # -------------------------------------------------
            # 1. Leer el documento fresco
            # -------------------------------------------------
            print(f"  [1] Leyendo documento...")
            documento = docs_service.documents().get(documentId=doc_id).execute()
            contenido = documento.get("body", {}).get("content", [])
            texto_plano, mapa_indices = _extraer_texto_con_indices(contenido)

            # -------------------------------------------------
            # 2. Encontrar el bloque {% table alias %}...{% end_table alias %}
            # -------------------------------------------------
            print(f"  [2] Buscando bloque {{% table {alias} %}}...")
            patron_bloque = re.compile(
                r"\{%-?\s*table\s+" + re.escape(alias) + r"\s*-?%\}"
                r"(.*?)"
                r"\{%-?\s*end_table\s+" + re.escape(alias) + r"\s*-?%\}",
                re.DOTALL
            )
            match_bloque = patron_bloque.search(texto_plano)
            if not match_bloque:
                print(f"  → OMITIDA: bloque no encontrado en el documento")
                omitidos.append({"alias": alias, "razon": f"bloque {{% table {alias} %}} no encontrado"})
                continue

            inicio_bloque = match_bloque.start()
            fin_bloque    = match_bloque.end()
            print(f"  [2] Bloque encontrado en posición {inicio_bloque}-{fin_bloque}")

            # -------------------------------------------------
            # 3. Localizar la tabla plantilla dentro del bloque
            # -------------------------------------------------
            print(f"  [3] Buscando tabla plantilla en el bloque...")
            start_bloque_doc = _idx(inicio_bloque, mapa_indices)
            end_bloque_doc   = _idx(fin_bloque - 1, mapa_indices)
            if end_bloque_doc is None:
                end_bloque_doc = mapa_indices[-1] if mapa_indices else 0

            tabla_plantilla = _encontrar_tabla_en_rango(
                contenido, start_bloque_doc, end_bloque_doc
            )
            if not tabla_plantilla:
                print(f"  → OMITIDA: tabla plantilla no encontrada en el bloque")
                omitidos.append({"alias": alias, "razon": "tabla plantilla no encontrada en el bloque"})
                continue
            print(f"  [3] Tabla plantilla encontrada")

            # -------------------------------------------------
            # 4. Extraer formato de la plantilla
            # -------------------------------------------------
            print(f"  [4] Extrayendo formato de plantilla...")
            formato_plantilla = _extraer_formato_plantilla(tabla_plantilla)
            n_cols_plantilla  = formato_plantilla["n_cols"]
            print(f"  [4] Formato extraído: {formato_plantilla['n_rows']} filas x {n_cols_plantilla} cols")

            # -------------------------------------------------
            # 5. Descargar datos desde Excel o Google Sheets
            # -------------------------------------------------
            print(f"  [5] Descargando datos de la tabla...")
            header_row = valor.get("header_row")
            df = _cargar_datos_tabla(url_tabla, sheet_name, drive_service, header_row)
            if df is None or df.empty:
                print(f"  → OMITIDA: no se pudieron cargar los datos")
                omitidos.append({"alias": alias, "razon": "no se pudieron cargar los datos de la tabla"})
                continue

            n_filas = len(df) + 1   # +1 por encabezados
            n_cols  = len(df.columns)
            print(f"  [5] Datos cargados: {n_filas} filas x {n_cols} cols")

            # -------------------------------------------------
            # 6. Localizar el placeholder {{ alias }} en el bloque
            # -------------------------------------------------
            print(f"  [6] Buscando placeholder '{placeholder}' en el bloque...")
            pos_placeholder = texto_plano.find(placeholder, inicio_bloque, fin_bloque)
            if pos_placeholder == -1:
                print(f"  → OMITIDA: placeholder no encontrado en el bloque")
                omitidos.append({"alias": alias, "razon": f"placeholder {placeholder} no encontrado en el bloque"})
                continue

            idx_placeholder_start = _idx(pos_placeholder, mapa_indices)
            idx_placeholder_end   = _idx(pos_placeholder + len(placeholder), mapa_indices)
            print(f"  [6] Placeholder en índices {idx_placeholder_start}-{idx_placeholder_end}")

            # -------------------------------------------------
            # 7. Insertar tabla vacía en la posición del placeholder
            # -------------------------------------------------
            print(f"  [7] Insertando tabla nueva ({n_filas}x{n_cols})...")
            docs_service.documents().batchUpdate(
                documentId=doc_id,
                body={"requests": [
                    {"deleteContentRange": {
                        "range": {
                            "startIndex": idx_placeholder_start,
                            "endIndex":   idx_placeholder_end
                        }
                    }},
                    {"insertTable": {
                        "rows":     n_filas,
                        "columns":  n_cols,
                        "location": {"index": idx_placeholder_start}
                    }}
                ]}
            ).execute()
            print(f"  [7] Tabla insertada")

            # -------------------------------------------------
            # 8. Leer el documento de nuevo para obtener índice real
            # -------------------------------------------------
            print(f"  [8] Localizando tabla recién insertada...")
            documento2  = docs_service.documents().get(documentId=doc_id).execute()
            contenido2  = documento2.get("body", {}).get("content", [])
            texto2, mi2 = _extraer_texto_con_indices(contenido2)

            tabla_nueva = _encontrar_tabla_en_rango(
                contenido2,
                idx_placeholder_start - 1,
                idx_placeholder_start + (n_filas * n_cols * 10)
            )
            if not tabla_nueva:
                print(f"  → OMITIDA: no se pudo localizar la tabla recién insertada")
                omitidos.append({"alias": alias, "razon": "no se pudo localizar la tabla recién insertada"})
                continue
            print(f"  [8] Tabla nueva localizada en índice {tabla_nueva.get('startIndex')}")

            # -------------------------------------------------
            # 9. Llenar la tabla con datos y aplicar formato
            # -------------------------------------------------
            print(f"  [9] Aplicando contenido y formato...")
            requests_formato = []
            requests_formato += _generar_requests_contenido(tabla_nueva, df)
            requests_formato += _generar_requests_formato(
                tabla_nueva, formato_plantilla, n_filas, n_cols
            )

            if requests_formato:
                docs_service.documents().batchUpdate(
                    documentId=doc_id,
                    body={"requests": requests_formato}
                ).execute()
            print(f"  [9] Formato aplicado ({len(requests_formato)} requests)")

            # -------------------------------------------------
            # 10. Aplicar merges
            # -------------------------------------------------
            print(f"  [10] Aplicando merges...")
            requests_merges = _generar_requests_merges(
                tabla_nueva, formato_plantilla, n_filas, n_cols, n_cols_plantilla
            )
            for merge_request in requests_merges:
                docs_service.documents().batchUpdate(
                    documentId=doc_id,
                    body={"requests": [merge_request]}
                ).execute()
            print(f"  [10] Merges aplicados ({len(requests_merges)})")

            # -------------------------------------------------
            # 11. Eliminar plantilla y etiquetas
            # -------------------------------------------------
            print(f"  [11] Limpiando plantilla y etiquetas...")
            documento3  = docs_service.documents().get(documentId=doc_id).execute()
            contenido3  = documento3.get("body", {}).get("content", [])
            texto3, mi3 = _extraer_texto_con_indices(contenido3)

            match3 = patron_bloque.search(texto3)
            if match3:
                start3 = _idx(match3.start(), mi3)
                end3   = _idx(match3.end() - 1, mi3)
                tabla_plantilla3 = _encontrar_tabla_en_rango(contenido3, start3, end3)

                requests_limpieza = []

                pos_end_tag = texto3.find(
                    "{%", match3.start(1) + len(match3.group(1)) - 1
                )
                if pos_end_tag != -1:
                    requests_limpieza.append({
                        "deleteContentRange": {
                            "range": {
                                "startIndex": _idx(pos_end_tag, mi3),
                                "endIndex":   _idx(match3.end(), mi3)
                            }
                        }
                    })

                if tabla_plantilla3:
                    requests_limpieza.append({
                        "deleteContentRange": {
                            "range": {
                                "startIndex": tabla_plantilla3["startIndex"],
                                "endIndex":   tabla_plantilla3["endIndex"]
                            }
                        }
                    })

                pos_start_tag_end = texto3.find("%}", match3.start()) + 2
                requests_limpieza.append({
                    "deleteContentRange": {
                        "range": {
                            "startIndex": _idx(match3.start(), mi3),
                            "endIndex":   _idx(pos_start_tag_end, mi3)
                        }
                    }
                })

                requests_limpieza.sort(
                    key=lambda r: r["deleteContentRange"]["range"]["startIndex"],
                    reverse=True
                )
                if requests_limpieza:
                    docs_service.documents().batchUpdate(
                        documentId=doc_id,
                        body={"requests": requests_limpieza}
                    ).execute()
            print(f"  [11] Limpieza completada")

            reemplazados.append({
                "alias":   alias,
                "n_filas": n_filas,
                "n_cols":  n_cols
            })
            print(f"  ✅ Tabla [{alias}] reemplazada exitosamente")

        except Exception as e:
            import traceback
            print(f"  ❌ ERROR en tabla [{alias}]: {str(e)}")
            traceback.print_exc()
            omitidos.append({"alias": alias, "razon": f"error: {str(e)}"})

    print(f"\n✅ Tablas reemplazadas: {len(reemplazados)}")
    for r in reemplazados:
        print(f"   {r['alias']} → {r['n_filas']} filas x {r['n_cols']} columnas")

    if omitidos:
        print(f"⚠️  Tablas omitidas: {len(omitidos)}")
        for o in omitidos:
            print(f"   {o['alias']} → {o['razon']}")

    return {"reemplazados": reemplazados, "omitidos": omitidos}


# ==============================
# HELPERS DE TABLAS
# ==============================

def _cargar_datos_tabla(url, sheet_name, drive_service, header_row=None):
    """
    Descarga datos desde Excel en Drive o Google Sheets.

    Estrategia de descarga:
        1. Intenta get_media (para .xlsx subidos a Drive)
        2. Si falla con 403/fileNotExportable, intenta export_media (para Sheets nativos)

    Esto resuelve el caso en que la URL tiene 'docs.google.com/spreadsheets'
    pero el archivo es un .xlsx subido (no un Google Sheets nativo).

    Args:
        url         (str): URL del archivo en Drive o Google Sheets.
        sheet_name  (str): Nombre de la hoja. Si es None usa la primera.
        drive_service:     Servicio autenticado de Google Drive API.
        header_row  (int): Fila donde está el encabezado (1-based).
                           Si es None asume fila 1.
    """
    import io
    import pandas as pd
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

    # Intentar primero get_media (funciona con .xlsx subidos a Drive)
    try:
        fh = _descargar(drive_service.files().get_media(fileId=file_id))
    except HttpError as e:
        if e.resp.status in (403, 400):
            # El archivo es un Google Sheets nativo → usar export_media
            fh = _descargar(drive_service.files().export_media(
                fileId=file_id,
                mimeType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            ))
        else:
            raise

    # header es 0-based en pandas, header_row es 1-based
    pandas_header = (header_row - 1) if header_row and header_row > 1 else 0

    if sheet_name:
        df = pd.read_excel(fh, sheet_name=sheet_name, header=pandas_header)
    else:
        df = pd.read_excel(fh, header=pandas_header)

    # Convertir todo a string para inserción en Docs
    df = df.fillna("").astype(str)
    return df



def _encontrar_tabla_en_rango(contenido, start_idx, end_idx):
    """
    Busca la primera tabla en el contenido del documento
    cuyo startIndex esté dentro del rango dado.
    Retorna el objeto tabla con startIndex y endIndex agregados.

    Si start_idx o end_idx son None, usa un rango amplio para
    no descartar tablas válidas.
    """
    # Si los índices son inválidos, usar rango máximo posible
    if start_idx is None:
        start_idx = 0
    if end_idx is None:
        end_idx = 9999999

    for elemento in contenido:
        if "table" in elemento:
            elem_start = elemento.get("startIndex", 0)
            elem_end   = elemento.get("endIndex", 0)
            if start_idx <= elem_start <= end_idx:
                tabla = elemento["table"]
                tabla["startIndex"] = elem_start
                tabla["endIndex"]   = elem_end
                return tabla
    return None


def _extraer_formato_plantilla(tabla):
    """
    Extrae el formato completo de la tabla plantilla celda por celda.

    Retorna:
        {
            "n_cols": int,
            "n_rows": int,
            "celdas": {
                (fila, col): {
                    "background":  color dict o None,
                    "borders":     dict con top/bottom/left/right,
                    "text_style":  dict con bold, italic, fontSize, etc.,
                    "alignment":   str,
                    "row_span":    int,
                    "col_span":    int,
                }
            }
        }
    """
    filas  = tabla.get("tableRows", [])
    n_rows = len(filas)
    n_cols = tabla.get("columns", 0)
    celdas = {}

    for fi, fila in enumerate(filas):
        for ci, celda in enumerate(fila.get("tableCells", [])):
            estilo_celda = celda.get("tableCellStyle", {})

            # Color de fondo
            bg = estilo_celda.get("backgroundColor", {}).get("color", {}).get("rgbColor")

            # Bordes
            bordes = {}
            for lado in ["borderTop", "borderBottom", "borderLeft", "borderRight"]:
                borde = estilo_celda.get(lado, {})
                bordes[lado] = {
                    "color": borde.get("color", {}).get("color", {}).get("rgbColor"),
                    "width": borde.get("width", {}).get("magnitude", 1),
                    "unit":  borde.get("width", {}).get("unit", "PT"),
                    "dashStyle": borde.get("dashStyle", "SOLID")
                }

            # Formato de texto (del primer run del primer párrafo)
            text_style = {}
            parrafos   = celda.get("content", [])
            alineacion = "START"
            if parrafos:
                primer_parrafo = parrafos[0].get("paragraph", {})
                alineacion     = primer_parrafo.get("paragraphStyle", {}).get("alignment", "START")
                elementos      = primer_parrafo.get("elements", [])
                if elementos:
                    ts = elementos[0].get("textRun", {}).get("textStyle", {})
                    text_style = {
                        "bold":          ts.get("bold", False),
                        "italic":        ts.get("italic", False),
                        "underline":     ts.get("underline", False),
                        "fontSize":      ts.get("fontSize", {}).get("magnitude", 11),
                        "fontSizeUnit":  ts.get("fontSize", {}).get("unit", "PT"),
                        "foregroundColor": ts.get("foregroundColor", {})
                                           .get("color", {}).get("rgbColor"),
                        "weightedFontFamily": ts.get("weightedFontFamily", {})
                                               .get("fontFamily", "Arial")
                    }

            # Merges
            row_span = celda.get("tableCellStyle", {}).get("rowSpan", 1)
            col_span = celda.get("tableCellStyle", {}).get("columnSpan", 1)

            celdas[(fi, ci)] = {
                "background": bg,
                "borders":    bordes,
                "text_style": text_style,
                "alignment":  alineacion,
                "row_span":   row_span,
                "col_span":   col_span,
            }

    return {"n_rows": n_rows, "n_cols": n_cols, "celdas": celdas}


def _get_formato_celda(formato_plantilla, fila, col):
    """
    Retorna el formato para una celda dada.
    Si está fuera del rango de la plantilla, usa la última celda disponible.
    """
    celdas       = formato_plantilla["celdas"]
    n_rows_plant = formato_plantilla["n_rows"]
    n_cols_plant = formato_plantilla["n_cols"]

    fi_ref = min(fila, n_rows_plant - 1)
    ci_ref = min(col,  n_cols_plant - 1)

    return celdas.get((fi_ref, ci_ref), {})


def _generar_requests_contenido(tabla_nueva, df):
    """
    Genera requests para llenar la tabla con los datos del DataFrame.
    Primera fila = encabezados, resto = datos.
    """
    requests = []
    filas    = tabla_nueva.get("tableRows", [])

    todas_las_filas = [list(df.columns)] + df.values.tolist()

    for fi, (fila_doc, fila_datos) in enumerate(zip(filas, todas_las_filas)):
        celdas_doc = fila_doc.get("tableCells", [])
        for ci, (celda_doc, valor) in enumerate(zip(celdas_doc, fila_datos)):
            texto = str(valor)
            if not texto:  # saltar celdas vacías — la API rechaza insertText con texto=""
                continue

            parrafos = celda_doc.get("content", [])
            if not parrafos:
                continue
            primer_parrafo = parrafos[0].get("paragraph", {})
            elementos      = primer_parrafo.get("elements", [])
            if not elementos:
                continue
            idx_insercion = elementos[0].get("startIndex", 0)

            requests.append({
                "insertText": {
                    "location": {"index": idx_insercion},
                    "text":     texto
                }
            })

    return requests

def _generar_requests_formato(tabla_nueva, formato_plantilla, n_filas, n_cols):
    """
    Genera requests para aplicar formato a cada celda de la tabla nueva.
    """
    requests = []
    filas    = tabla_nueva.get("tableRows", [])
    tabla_id = tabla_nueva.get("startIndex", 0)

    for fi, fila_doc in enumerate(filas):
        for ci, celda_doc in enumerate(fila_doc.get("tableCells", [])):
            fmt = _get_formato_celda(formato_plantilla, fi, ci)
            if not fmt:
                continue

            start_celda = celda_doc.get("startIndex", 0)
            end_celda   = celda_doc.get("endIndex", 0)

            # Formato de celda (fondo y bordes)
            cell_style     = {}
            cell_style_fields = []

            if fmt.get("background"):
                cell_style["backgroundColor"] = {"color": {"rgbColor": fmt["background"]}}
                cell_style_fields.append("backgroundColor")

            bordes = fmt.get("borders", {})
            for lado, key in [
                ("borderTop",    "borderTop"),
                ("borderBottom", "borderBottom"),
                ("borderLeft",   "borderLeft"),
                ("borderRight",  "borderRight")
            ]:
                b = bordes.get(lado, {})
                if b:
                    cell_style[key] = {
                        "color":     {"color": {"rgbColor": b.get("color") or {"red": 0, "green": 0, "blue": 0}}},
                        "width":     {"magnitude": b.get("width", 1), "unit": b.get("unit", "PT")},
                        "dashStyle": b.get("dashStyle", "SOLID")
                    }
                    cell_style_fields.append(key)

            if cell_style:
                requests.append({
                    "updateTableCellStyle": {
                        "tableRange": {
                            "tableCellLocation": {
                                "tableStartLocation": {"index": tabla_id},
                                "rowIndex":    fi,
                                "columnIndex": ci
                            },
                            "rowSpan":    1,
                            "columnSpan": 1
                        },
                        "tableCellStyle": cell_style,
                        "fields": ",".join(cell_style_fields)
                    }
                })

            # Formato de texto
            ts = fmt.get("text_style", {})
            if ts:
                parrafos  = celda_doc.get("content", [])
                elementos = parrafos[0].get("paragraph", {}).get("elements", []) if parrafos else []
                if elementos:
                    ts_start = elementos[0].get("startIndex", 0)
                    ts_end   = celda_doc.get("endIndex", 0) - 1

                    text_style     = {}
                    text_fields    = []

                    if ts.get("bold") is not None:
                        text_style["bold"] = ts["bold"]
                        text_fields.append("bold")
                    if ts.get("italic") is not None:
                        text_style["italic"] = ts["italic"]
                        text_fields.append("italic")
                    if ts.get("underline") is not None:
                        text_style["underline"] = ts["underline"]
                        text_fields.append("underline")
                    if ts.get("fontSize"):
                        text_style["fontSize"] = {
                            "magnitude": ts["fontSize"],
                            "unit":      ts.get("fontSizeUnit", "PT")
                        }
                        text_fields.append("fontSize")
                    if ts.get("foregroundColor"):
                        text_style["foregroundColor"] = {
                            "color": {"rgbColor": ts["foregroundColor"]}
                        }
                        text_fields.append("foregroundColor")
                    if ts.get("weightedFontFamily"):
                        text_style["weightedFontFamily"] = {
                            "fontFamily": ts["weightedFontFamily"]
                        }
                        text_fields.append("weightedFontFamily")

                    if text_style and ts_start < ts_end:
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

            # Alineación de párrafo
            alineacion = fmt.get("alignment")
            if alineacion and parrafos:
                p_start = parrafos[0].get("paragraph", {}).get("elements", [{}])[0].get("startIndex", 0)
                p_end   = celda_doc.get("endIndex", 0) - 1
                if p_start < p_end:
                    requests.append({
                        "updateParagraphStyle": {
                            "range": {
                                "startIndex": p_start,
                                "endIndex":   p_end
                            },
                            "paragraphStyle": {"alignment": alineacion},
                            "fields": "alignment"
                        }
                    })

    return requests


def _generar_requests_merges(tabla_nueva, formato_plantilla, n_filas, n_cols, n_cols_plantilla):
    """
    Genera requests de merge aplicando las reglas:
    - Merge de fila completa → abarca todas las columnas de la tabla nueva
    - Merge parcial → replica en las mismas posiciones, columnas extra independientes
    """
    requests  = []
    celdas    = formato_plantilla["celdas"]
    tabla_id  = tabla_nueva.get("startIndex", 0)

    celdas_procesadas = set()

    for (fi, ci), fmt in celdas.items():
        col_span = fmt.get("col_span", 1)
        row_span = fmt.get("row_span", 1)

        if col_span <= 1 and row_span <= 1:
            continue
        if (fi, ci) in celdas_procesadas:
            continue

        # Determinar si es merge de fila completa
        es_fila_completa = (col_span >= n_cols_plantilla)

        col_span_final = n_cols if es_fila_completa else col_span
        row_span_final = min(row_span, n_filas - fi)

        # Validar que el merge cabe en la tabla nueva
        if ci + col_span_final > n_cols:
            col_span_final = n_cols - ci
        if fi + row_span_final > n_filas:
            row_span_final = n_filas - fi

        if col_span_final <= 1 and row_span_final <= 1:
            continue

        requests.append({
            "mergeTableCells": {
                "tableRange": {
                    "tableCellLocation": {
                        "tableStartLocation": {"index": tabla_id},
                        "rowIndex":    fi,
                        "columnIndex": ci
                    },
                    "rowSpan":    row_span_final,
                    "columnSpan": col_span_final
                }
            }
        })

        # Marcar celdas cubiertas por este merge como procesadas
        for r in range(fi, fi + row_span_final):
            for c in range(ci, ci + col_span_final):
                celdas_procesadas.add((r, c))

    return requests


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
