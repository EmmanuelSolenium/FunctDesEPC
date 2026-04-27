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
# TABLAS EN GOOGLE DOCS
# ==============================

def reemplazar_tablas(doc_id, tablas_data, docs_service):
    """
    Rellena tablas en un Google Doc con datos de DataFrames.

    Detecta automáticamente cuáles filas son encabezado (fondo no blanco o
    con formato especial) y cuáles son datos (fondo neutro, estructura uniforme).
    Las filas de encabezado se preservan; las de datos se eliminan y se
    regeneran a partir del DataFrame.

    Convención de uso en la plantilla
    ----------------------------------
    Coloca el placeholder como único texto en la PRIMERA CELDA de la tabla:

        ┌─────────────────────┬──────────┬──────────┐
        │ {{ TABLA_DATOS_MT }}│  Col B   │  Col C   │  ← fila de encabezado
        ├─────────────────────┼──────────┼──────────┤
        │  (fila de datos)    │          │          │  ← se detecta y reemplaza
        └─────────────────────┴──────────┴──────────┘

    Args:
        doc_id      (str):  ID del documento Google Docs.
        tablas_data (dict): { placeholder: pd.DataFrame }
                            ej: {"{{ TABLA_DATOS_MT }}": datos_iniciales_red_mt}
        docs_service:       Servicio autenticado de Google Docs API.

    Returns:
        dict: {"reemplazados": [...], "omitidos": [...]}

    Orden de ejecución recomendado en doc_filler.py
    ------------------------------------------------
        procesar_condicionales(...)   # 1
        reemplazar_tablas(...)        # 2  ← antes que los demás
        reemplazar_textos(...)        # 3
        reemplazar_imagenes(...)      # 4
    """
    import pandas as pd

    reemplazados = []
    omitidos     = []

    if not tablas_data:
        print("ℹ️  No se proporcionaron tablas para reemplazar.")
        return {"reemplazados": [], "omitidos": []}

    for placeholder, df in tablas_data.items():

        try:
            # ------------------------------------------------------------------
            # 1. Obtener documento fresco y localizar la tabla por placeholder
            # ------------------------------------------------------------------
            doc      = docs_service.documents().get(documentId=doc_id).execute()
            contenido = doc.get("body", {}).get("content", [])

            tabla_info = _encontrar_tabla_por_placeholder(contenido, placeholder)

            if tabla_info is None:
                omitidos.append({
                    "placeholder": placeholder,
                    "razon": "placeholder no encontrado en ninguna celda de la tabla"
                })
                continue

            table_el, table_start_idx = tabla_info
            filas   = table_el.get("tableRows", [])
            n_filas = len(filas)
            n_cols  = max(len(f.get("tableCells", [])) for f in filas) if filas else 0

            # ------------------------------------------------------------------
            # 2. Detectar cuántas filas son encabezado
            # ------------------------------------------------------------------
            n_header = _detectar_encabezados_gdoc(filas)

            if n_header >= n_filas:
                omitidos.append({
                    "placeholder": placeholder,
                    "razon": (
                        f"todas las {n_filas} filas parecen encabezado. "
                        "Agrega al menos una fila de datos de ejemplo en la plantilla."
                    )
                })
                continue

            # ------------------------------------------------------------------
            # 3. Validar que las columnas del DataFrame coincidan con la tabla
            # ------------------------------------------------------------------
            if df.shape[1] != n_cols:
                omitidos.append({
                    "placeholder": placeholder,
                    "razon": (
                        f"el DataFrame tiene {df.shape[1]} columnas "
                        f"pero la tabla tiene {n_cols}. Deben coincidir."
                    )
                })
                continue

            # ------------------------------------------------------------------
            # 4. Eliminar filas de datos existentes (de abajo hacia arriba
            #    para no desplazar los índices de filas superiores)
            # ------------------------------------------------------------------
            n_data_rows_originales = n_filas - n_header
            if n_data_rows_originales > 0:
                delete_requests = [
                    {
                        "deleteTableRow": {
                            "tableCellLocation": {
                                "tableStartLocation": {"index": table_start_idx},
                                "rowIndex":    n_filas - 1 - i,
                                "columnIndex": 0
                            }
                        }
                    }
                    for i in range(n_data_rows_originales)
                ]
                docs_service.documents().batchUpdate(
                    documentId=doc_id,
                    body={"requests": delete_requests}
                ).execute()

            # ------------------------------------------------------------------
            # 5. Insertar filas vacías (una por fila del DataFrame)
            #    Todas se insertan debajo de la última fila de encabezado.
            #    Se acumulan en un solo batchUpdate para eficiencia.
            # ------------------------------------------------------------------
            if len(df) > 0:
                insert_requests = [
                    {
                        "insertTableRow": {
                            "tableCellLocation": {
                                "tableStartLocation": {"index": table_start_idx},
                                "rowIndex":    n_header - 1 + i,
                                "columnIndex": 0
                            },
                            "insertBelow": True
                        }
                    }
                    for i in range(len(df))
                ]
                docs_service.documents().batchUpdate(
                    documentId=doc_id,
                    body={"requests": insert_requests}
                ).execute()

            # ------------------------------------------------------------------
            # 6. Re-obtener el documento con los índices actualizados
            # ------------------------------------------------------------------
            doc      = docs_service.documents().get(documentId=doc_id).execute()
            contenido = doc.get("body", {}).get("content", [])
            tabla_info = _encontrar_tabla_por_placeholder(contenido, placeholder)
            if tabla_info is None:
                # Fallback: buscar por posición aproximada
                tabla_info = _encontrar_tabla_por_indice(contenido, table_start_idx)
            if tabla_info is None:
                omitidos.append({
                    "placeholder": placeholder,
                    "razon": "no se pudo relocalizar la tabla tras insertar filas"
                })
                continue

            table_el, _ = tabla_info
            filas_nuevas = table_el.get("tableRows", [])[n_header:]

            # ------------------------------------------------------------------
            # 7. Escribir valores en las celdas de las nuevas filas
            #    Orden: de atrás hacia adelante (fila y columna) para no
            #    desplazar los índices de celdas que aún no se han escrito.
            # ------------------------------------------------------------------
            text_requests = []

            for row_idx in range(len(filas_nuevas) - 1, -1, -1):
                df_row = df.iloc[row_idx]
                celdas = filas_nuevas[row_idx].get("tableCells", [])

                for col_idx in range(len(celdas) - 1, -1, -1):
                    celda  = celdas[col_idx]
                    valor  = df_row.iloc[col_idx]

                    # El primer párrafo de la celda es el punto de inserción
                    parrafos = celda.get("content", [])
                    if not parrafos:
                        continue
                    primer_parrafo = parrafos[0].get("paragraph", {})
                    elementos      = primer_parrafo.get("elements", [])
                    if not elementos:
                        continue

                    insert_idx = elementos[0].get("startIndex")
                    if insert_idx is None:
                        continue

                    # Convertir valor a string (respetar None / NaN → vacío)
                    es_nulo = False
                    try:
                        es_nulo = pd.isna(valor)
                    except (TypeError, ValueError):
                        pass

                    texto = "" if es_nulo else str(valor).strip()
                    if texto.lower() in ("nan", "none"):
                        texto = ""

                    if texto:
                        text_requests.append({
                            "insertText": {
                                "location": {"index": insert_idx},
                                "text":     texto
                            }
                        })

            if text_requests:
                docs_service.documents().batchUpdate(
                    documentId=doc_id,
                    body={"requests": text_requests}
                ).execute()

            # ------------------------------------------------------------------
            # 8. Limpiar el placeholder de la primera celda del encabezado
            #    (replaceAllText lo reemplazará por "" en el paso de textos,
            #    pero lo limpiamos aquí para consistencia)
            # ------------------------------------------------------------------
            docs_service.documents().batchUpdate(
                documentId=doc_id,
                body={"requests": [{
                    "replaceAllText": {
                        "containsText": {"text": placeholder, "matchCase": True},
                        "replaceText":  ""
                    }
                }]}
            ).execute()

            reemplazados.append({
                "placeholder":  placeholder,
                "filas_datos":  len(df),
                "columnas":     df.shape[1],
                "n_encabezados": n_header,
            })

        except Exception as e:
            omitidos.append({
                "placeholder": placeholder,
                "razon":       f"error inesperado: {str(e)}"
            })

    # Log de resultados
    print(f"✅ Tablas reemplazadas: {len(reemplazados)}")
    for r in reemplazados:
        print(
            f"   {r['placeholder']} → "
            f"{r['filas_datos']} filas × {r['columnas']} cols  "
            f"({r['n_encabezados']} fila(s) de encabezado preservada(s))"
        )

    if omitidos:
        print(f"⚠️  Omitidos: {len(omitidos)}")
        for o in omitidos:
            print(f"   {o['placeholder']}: {o['razon']}")

    return {"reemplazados": reemplazados, "omitidos": omitidos}


# ── Helpers internos de tablas ────────────────────────────────────────────────

def _encontrar_tabla_por_placeholder(contenido, placeholder):
    """
    Busca en todos los elementos del body la tabla que tenga el placeholder
    en alguna de sus celdas. Devuelve (table_element, table_start_index) o None.
    """
    for elemento in contenido:
        if "table" not in elemento:
            continue
        table     = elemento["table"]
        start_idx = elemento.get("startIndex", 0)
        for fila in table.get("tableRows", []):
            for celda in fila.get("tableCells", []):
                texto_celda = _texto_celda_gdoc(celda)
                if placeholder in texto_celda:
                    return table, start_idx
    return None


def _encontrar_tabla_por_indice(contenido, table_start_idx):
    """
    Localiza una tabla por su startIndex aproximado (útil tras insertar filas,
    cuando el startIndex puede haber cambiado ligeramente).
    Devuelve (table_element, table_start_index) o None.
    """
    mejor      = None
    menor_dist = float("inf")
    for elemento in contenido:
        if "table" not in elemento:
            continue
        idx  = elemento.get("startIndex", 0)
        dist = abs(idx - table_start_idx)
        if dist < menor_dist:
            menor_dist = dist
            mejor      = (elemento["table"], idx)
    return mejor


def _texto_celda_gdoc(celda):
    """Extrae el texto plano de todos los párrafos de una celda."""
    partes = []
    for bloque in celda.get("content", []):
        for elem in bloque.get("paragraph", {}).get("elements", []):
            partes.append(elem.get("textRun", {}).get("content", ""))
    return "".join(partes)


def _color_es_neutro(color):
    """
    Devuelve True si el color de fondo es neutro (blanco o sin definir).
    La API devuelve colores como {red: float, green: float, blue: float} con 1.0 = blanco.
    """
    if not color:
        return True
    r = color.get("red",   1.0)
    g = color.get("green", 1.0)
    b = color.get("blue",  1.0)
    # Considerar neutro si es blanco puro o muy cercano a él
    return r >= 0.95 and g >= 0.95 and b >= 0.95


def _detectar_encabezados_gdoc(filas):
    """
    Detecta cuántas filas al inicio de la tabla son encabezado.

    Heurística (en orden de prioridad):
      1. Si la celda tiene fondo no neutro  → fila de encabezado.
      2. Si todas las celdas tienen texto en negrita → fila de encabezado.
      3. En caso de empate, al menos 1 fila es siempre encabezado.

    Devuelve el número de filas de encabezado (int ≥ 1).
    """
    n_header = 0

    for fila in filas:
        celdas = fila.get("tableCells", [])
        es_encabezado = False

        for celda in celdas:
            # Criterio 1: color de fondo
            bg = (
                celda.get("tableCellStyle", {})
                     .get("backgroundColor", {})
                     .get("color", {})
                     .get("rgbColor", {})
            )
            if not _color_es_neutro(bg):
                es_encabezado = True
                break

            # Criterio 2: texto en negrita
            for bloque in celda.get("content", []):
                for elem in bloque.get("paragraph", {}).get("elements", []):
                    estilo = elem.get("textRun", {}).get("textStyle", {})
                    if estilo.get("bold"):
                        es_encabezado = True
                        break
                if es_encabezado:
                    break
            if es_encabezado:
                break

        if es_encabezado:
            n_header += 1
        else:
            break  # las filas de datos son contiguas desde aquí

    return max(n_header, 1)  # mínimo 1 fila de encabezado siempre


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
