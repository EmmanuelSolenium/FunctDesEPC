"""
exportar_tablas.py
==================
Funciones de exportación a Excel con formato y previsualización en Colab
para las TABLAS DE FLECHADO del proyecto FunctDesEPC.

Uso típico (en crear_tablas_mec.ipynb):
----------------------------------------
    from exportar_tablas import exportar_tablas_flechado, previsualizar_tabla_flechado

    # Exportar todas las tablas a Excel (n hojas normales + m hojas secundarias)
    exportar_tablas_flechado(
        tab_fle=tab_fle,          # lista de DataFrames con header de cada cantón normal
        tablas_p=tablas_p,        # lista de DataFrames con datos de flechado cantón normal
        tab_fle_s=tab_fle_s,      # lista de DataFrames con header de cada cantón secundario
        tablas_s=tablas_s,        # lista de DataFrames con datos de flechado cantón secundario
        filepath="Tablas_Flechado.xlsx"
    )

    # Previsualizar una tabla individual en Colab
    previsualizar_tabla_flechado(tab_fle[0], tablas_p[0], titulo="Cantón 1")
    previsualizar_tabla_flechado(tab_fle_s[0], tablas_s[0], titulo="Cantón 1S")
"""

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import (
    Font, Alignment, Border, Side, PatternFill, numbers
)
from openpyxl.utils import get_column_letter
from IPython.display import display, HTML


# ---------------------------------------------------------------------------
# Estilos reutilizables
# ---------------------------------------------------------------------------

def _thin():
    return Side(style="thin", color="FF000000")

def _medium():
    return Side(style="medium", color="FF000000")

def _border_thin_all():
    t = _thin()
    return Border(left=t, right=t, top=t, bottom=t)

def _border_medium_all():
    m = _medium()
    return Border(left=m, right=m, top=m, bottom=m)

def _border_medium_outer_thin_inner(left=True, right=True, top=True, bottom=True):
    """Borde medium en los lados indicados, thin en el resto."""
    def s(is_medium):
        return _medium() if is_medium else _thin()
    return Border(
        left=s(left), right=s(right), top=s(top), bottom=s(bottom)
    )

FILL_HEADER = PatternFill("solid", fgColor="D9E1F2")   # azul claro para etiquetas
FILL_DATA   = PatternFill("solid", fgColor="FFFFFF")    # blanco para datos
FONT_BOLD   = Font(bold=True, size=9)
FONT_NORMAL = Font(bold=False, size=9)
ALIGN_CENTER = Alignment(horizontal="center", vertical="center", wrap_text=True)
ALIGN_LEFT   = Alignment(horizontal="left",   vertical="center", wrap_text=True)
NUM_FMT_2DEC = '0.00'
NUM_FMT_4DEC = '0.0000'


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _write_cell(ws, row, col, value, font=None, alignment=None,
                border=None, fill=None, number_format=None):
    """Escribe un valor en una celda con formato opcional."""
    cell = ws.cell(row=row, column=col, value=value)
    if font is not None:
        cell.font = font
    if alignment is not None:
        cell.alignment = alignment
    if border is not None:
        cell.border = border
    if fill is not None:
        cell.fill = fill
    if number_format is not None:
        cell.number_format = number_format
    return cell


def _merge_and_write(ws, row, col_start, col_end, value,
                     font=None, alignment=None, border=None, fill=None):
    """Fusiona celdas y escribe el valor en la primera."""
    if col_start < col_end:
        ws.merge_cells(
            start_row=row, start_column=col_start,
            end_row=row,   end_column=col_end
        )
    _write_cell(ws, row, col_start, value, font=font,
                alignment=alignment, border=border, fill=fill)


def _auto_col_width(ws, min_width=8, max_width=30):
    """Ajusta el ancho de columnas automáticamente según contenido."""
    for col in ws.columns:
        max_len = 0
        col_letter = get_column_letter(col[0].column)
        for cell in col:
            try:
                if cell.value is not None:
                    max_len = max(max_len, len(str(cell.value)))
            except Exception:
                pass
        ws.column_dimensions[col_letter].width = max(min_width, min(max_len + 2, max_width))


# ---------------------------------------------------------------------------
# Escritura del bloque HEADER de una tabla de flechado
# ---------------------------------------------------------------------------

def _escribir_header_tabla(ws, tab_fle_df, start_row=1, start_col=1):
    """
    Escribe la parte superior de la tabla (header) a partir de tab_fle_df.

    tab_fle_df se asume un DataFrame cuyas filas son:
        0: Conductor
        1: Cantón, Poste Inicial
        2: Vano de Regulación, Poste Final
        3: Vano (números de vano)
        4: Longitud (m)
        5: Poste inicial
        6: Poste final
        7: Desnivel

    La primera columna contiene las etiquetas (ej. "Conductor", "Cantón", etc.)
    y las demás columnas contienen los valores.
    Las celdas con valor NaN se dejan completamente sin formato ni contenido.

    Retorna la siguiente fila disponible (start_row + número de filas escritas).
    """
    b_med = _border_medium_all()
    b_thin = _border_thin_all()
    label_fill = FILL_HEADER

    n_rows, n_cols = tab_fle_df.shape
    label_col = start_col
    data_col_start = start_col + 1

    for rel_row, (idx, row_data) in enumerate(tab_fle_df.iterrows()):
        abs_row = start_row + rel_row
        row_vals = row_data.tolist()

        label = row_vals[0] if len(row_vals) > 0 else ""
        data_vals = row_vals[1:]

        is_vano_row = str(label).strip().lower() in {
            "vano", "longitud (m)", "poste inicial", "poste final", "desnivel"
        }
        border = b_med if is_vano_row else b_thin

        # Etiqueta — solo escribir si no es NaN
        if pd.notna(label) and str(label).strip() != "":
            _merge_and_write(
                ws, abs_row, label_col, label_col + 1,
                value=label,
                font=FONT_BOLD,
                alignment=ALIGN_LEFT,
                border=border,
                fill=label_fill
            )

        # Valores de datos — omitir celdas NaN completamente (sin valor ni formato)
        for rel_col, val in enumerate(data_vals):
            c = data_col_start + 1 + rel_col
            try:
                es_nan = pd.isna(val)
            except (TypeError, ValueError):
                es_nan = False
            if es_nan:
                continue
            num_fmt = None
            if isinstance(val, float):
                num_fmt = NUM_FMT_4DEC if abs(val) < 10 else NUM_FMT_2DEC
            _write_cell(
                ws, abs_row, c, val,
                font=FONT_NORMAL,
                alignment=ALIGN_CENTER,
                border=border,
                fill=FILL_DATA,
                number_format=num_fmt
            )

    return start_row + n_rows


# ---------------------------------------------------------------------------
# Escritura del bloque DATOS (tablas_p / tablas_s)
# ---------------------------------------------------------------------------

def _escribir_datos_tabla(ws, tablas_df, start_row=1, start_col=1):
    """
    Escribe la parte inferior de la tabla (datos de temperatura/tense/flechas).

    tablas_df se asume un DataFrame con columnas:
        Temperatura, Tense, f_vano1, f_vano2, ...

    Escribe una fila de encabezado con los nombres de columna y luego
    los datos fila a fila. Las celdas con NaN se omiten (sin formato).
    No se escribe fila de unidades (eliminada por ser redundante).

    Retorna la siguiente fila disponible.
    """
    b_med = _border_medium_all()
    b_thin = _border_thin_all()

    # Encabezado de columnas
    cols = list(tablas_df.columns)
    for rel_col, col_name in enumerate(cols):
        c = start_col + rel_col
        _write_cell(
            ws, start_row, c, col_name,
            font=FONT_BOLD,
            alignment=ALIGN_CENTER,
            border=b_med,
            fill=FILL_HEADER
        )

    # Datos — directamente en la fila siguiente (sin fila de unidades)
    data_start_row = start_row + 1
    for rel_row, (_, row_data) in enumerate(tablas_df.iterrows()):
        abs_row = data_start_row + rel_row
        for rel_col, val in enumerate(row_data):
            c = start_col + rel_col
            try:
                es_nan = pd.isna(val)
            except (TypeError, ValueError):
                es_nan = False
            if es_nan:
                continue  # celda NaN: sin valor ni formato
            num_fmt = None
            if isinstance(val, float):
                num_fmt = NUM_FMT_4DEC
            _write_cell(
                ws, abs_row, c, val,
                font=FONT_NORMAL,
                alignment=ALIGN_CENTER,
                border=b_thin,
                fill=FILL_DATA,
                number_format=num_fmt
            )

    return data_start_row + len(tablas_df)


# ---------------------------------------------------------------------------
# Función principal de exportación
# ---------------------------------------------------------------------------

def exportar_tablas_flechado(
    tab_fle,
    tablas_p,
    tab_fle_s,
    tablas_s,
    filepath="Tablas_Flechado.xlsx"
):
    """
    Exporta las tablas de flechado a un archivo Excel con formato.

    Genera n + m hojas:
      - n hojas para cantones normales  → nombres: "Canton_1", "Canton_2", ...
      - m hojas para cantones secundarios → nombres: "Canton_1S", "Canton_2S", ...

    Cada hoja contiene:
      - Parte superior: header del cantón (tab_fle[i] o tab_fle_s[i])
      - Parte inferior: datos de temperatura/flechas (tablas_p[i] o tablas_s[i])

    Parámetros
    ----------
    tab_fle : list[pd.DataFrame]
        Lista de DataFrames con el header de cada cantón normal.
    tablas_p : list[pd.DataFrame]
        Lista de DataFrames con los datos de flechado de cada cantón normal.
        Debe tener la misma longitud que tab_fle.
    tab_fle_s : list[pd.DataFrame]
        Lista de DataFrames con el header de cada cantón secundario.
    tablas_s : list[pd.DataFrame]
        Lista de DataFrames con los datos de flechado de cada cantón secundario.
        Debe tener la misma longitud que tab_fle_s.
    filepath : str
        Ruta del archivo Excel de salida.

    Retorna
    -------
    str
        Ruta del archivo generado.
    """
    if len(tab_fle) != len(tablas_p):
        raise ValueError(
            f"tab_fle tiene {len(tab_fle)} elementos pero tablas_p tiene {len(tablas_p)}. "
            "Deben tener la misma longitud."
        )
    if len(tab_fle_s) != len(tablas_s):
        raise ValueError(
            f"tab_fle_s tiene {len(tab_fle_s)} elementos pero tablas_s tiene {len(tablas_s)}. "
            "Deben tener la misma longitud."
        )

    wb = Workbook()
    # Eliminar hoja vacía por defecto
    wb.remove(wb.active)

    # -- Cantones normales --
    for i, (header_df, datos_df) in enumerate(zip(tab_fle, tablas_p)):
        sheet_name = f"Canton_{i + 1}"
        ws = wb.create_sheet(title=sheet_name)
        next_row = _escribir_header_tabla(ws, header_df, start_row=1, start_col=1)
        # Sin fila en blanco: datos pegados directamente al header
        _escribir_datos_tabla(ws, datos_df, start_row=next_row, start_col=1)
        _auto_col_width(ws)

    # -- Cantones secundarios --
    for i, (header_df, datos_df) in enumerate(zip(tab_fle_s, tablas_s)):
        sheet_name = f"Canton_{i + 1}S"
        ws = wb.create_sheet(title=sheet_name)
        next_row = _escribir_header_tabla(ws, header_df, start_row=1, start_col=1)
        _escribir_datos_tabla(ws, datos_df, start_row=next_row, start_col=1)
        _auto_col_width(ws)

    wb.save(filepath)
    print(f"✅ Archivo guardado: {filepath}  ({len(tab_fle)} cantones + {len(tab_fle_s)} secundarios)")
    return filepath


# ---------------------------------------------------------------------------
# Previsualización en Colab con HTML estilizado
# ---------------------------------------------------------------------------

def _df_to_html_styled(df, caption=""):
    """Convierte un DataFrame a HTML con estilos inline para Colab."""
    th_style = (
        "background-color:#D9E1F2; font-weight:bold; border:1px solid #888; "
        "padding:4px 8px; text-align:center; font-size:11px;"
    )
    td_style = (
        "border:1px solid #ccc; padding:3px 6px; text-align:center; font-size:11px;"
    )
    td_label_style = (
        "border:1px solid #ccc; padding:3px 6px; text-align:left; font-size:11px; "
        "background-color:#EEF2FB; font-weight:bold;"
    )

    html = f'<table style="border-collapse:collapse; margin:4px 0;">'
    if caption:
        html += f'<caption style="font-weight:bold; font-size:13px; margin-bottom:4px;">{caption}</caption>'

    # Encabezado de columnas
    html += "<thead><tr>"
    for col in df.columns:
        html += f'<th style="{th_style}">{col}</th>'
    html += "</tr></thead><tbody>"

    # Filas
    for _, row in df.iterrows():
        html += "<tr>"
        for j, val in enumerate(row):
            # Primera columna = etiqueta
            style = td_label_style if j == 0 else td_style
            # Formateo numérico
            if isinstance(val, float):
                display_val = f"{val:.4f}" if abs(val) < 100 else f"{val:.2f}"
            else:
                display_val = "" if pd.isna(val) else str(val)
            html += f'<td style="{style}">{display_val}</td>'
        html += "</tr>"

    html += "</tbody></table>"
    return html


def previsualizar_tabla_flechado(tab_fle_i, tablas_i, titulo="Tabla de Flechado"):
    """
    Previsualiza una tabla de flechado completa en Jupyter/Colab.

    Muestra primero el header (tab_fle_i) y luego los datos (tablas_i)
    con formato HTML estilizado.

    Parámetros
    ----------
    tab_fle_i : pd.DataFrame
        DataFrame con el header del cantón (una de las listas tab_fle o tab_fle_s).
    tablas_i : pd.DataFrame
        DataFrame con los datos de temperatura/flechas (una de las listas tablas_p o tablas_s).
    titulo : str
        Título mostrado encima de la tabla (ej. "Cantón 1" o "Cantón 1S").
    """
    wrapper_style = (
        "border:2px solid #4472C4; border-radius:6px; padding:12px; "
        "margin:8px 0; display:inline-block; max-width:100%; overflow-x:auto;"
    )
    title_style = (
        "color:#2F5496; font-size:15px; font-weight:bold; "
        "font-family:Arial,sans-serif; margin-bottom:8px;"
    )
    section_style = "color:#555; font-size:11px; font-weight:bold; margin:6px 0 2px 0;"

    html_parts = [f'<div style="{wrapper_style}">']
    html_parts.append(f'<div style="{title_style}">📊 {titulo}</div>')

    # Header
    html_parts.append(f'<div style="{section_style}">Datos del Cantón</div>')
    html_parts.append(_df_to_html_styled(tab_fle_i))

    # Separador
    html_parts.append('<hr style="border:none;border-top:1px solid #ccc;margin:8px 0;">')

    # Datos
    html_parts.append(f'<div style="{section_style}">Tablas de Flechado</div>')
    html_parts.append(_df_to_html_styled(tablas_i))

    html_parts.append("</div>")

    display(HTML("".join(html_parts)))


def previsualizar_todas(tab_fle, tablas_p, tab_fle_s=None, tablas_s=None):
    """
    Previsualiza todas las tablas de flechado en Colab.

    Muestra primero los cantones normales y luego los secundarios.

    Parámetros
    ----------
    tab_fle : list[pd.DataFrame]
    tablas_p : list[pd.DataFrame]
    tab_fle_s : list[pd.DataFrame] | None
    tablas_s : list[pd.DataFrame] | None
    """
    for i, (header, datos) in enumerate(zip(tab_fle, tablas_p)):
        previsualizar_tabla_flechado(header, datos, titulo=f"Cantón {i + 1}")

    if tab_fle_s and tablas_s:
        for i, (header, datos) in enumerate(zip(tab_fle_s, tablas_s)):
            previsualizar_tabla_flechado(header, datos, titulo=f"Cantón {i + 1}S")
