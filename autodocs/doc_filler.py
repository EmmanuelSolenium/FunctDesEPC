# ==============================
# AJUSTE DE PATH (para imports)
# ==============================
import sys
import os

CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# ==============================
# IMPORTAR FUNCIONES
# ==============================
from autodocs import funciones_docs


# ==============================
# CONFIGURACIÓN
# ==============================

# ID del Excel con el diccionario de datos (en Drive)
EXCEL_FILE_ID = "1t9utg6qjm4KG9tQec6DzwoR0SYuUy53o"

# URL o ID del documento plantilla en Drive
URL_PLANTILLA = "https://docs.google.com/document/d/15gUhRpMmIXnj8mgUlOBMwZKqjH7kfZ1jC2Q5sElpg4o/edit"

# URL o ID de la carpeta donde se guardará la copia generada
# Debe ser una carpeta en tu Drive personal a la que tengas acceso
URL_CARPETA_DESTINO = "https://drive.google.com/drive/folders/1zUPZXuCLZyA63EK0HkzirBXY0Vw6CQkX"

# Nombre del documento generado
NOMBRE_DOCUMENTO = "Declaración de Cumplimiento - Generado"


# ==============================
# TABLAS: importar DataFrames
# ==============================
# Importa aquí los DataFrames que quieres insertar en las tablas del documento.
# La clave es el placeholder que pusiste en la primera celda de cada tabla
# en la plantilla de Google Docs.
#
# Ejemplo:
#   from mi_modulo_calculos import datos_iniciales_red_mt, informacion_del_apoyo
#
# Si los DataFrames se generan en otro script, impórtalos desde allí.

def _construir_tablas_data():
    """
    Devuelve el diccionario {placeholder: DataFrame} con las tablas a rellenar.
    Ajusta este bloque según las tablas de tu plantilla y tus fuentes de datos.
    """
    # ── Ejemplo: descomentar y adaptar ───────────────────────────────────────
    # from mi_modulo import datos_iniciales_red_mt, informacion_del_apoyo
    # return {
    #     "{{ TABLA_DATOS_INICIALES_MT }}": datos_iniciales_red_mt,
    #     "{{ TABLA_INFO_APOYO }}":         informacion_del_apoyo,
    # }
    # ─────────────────────────────────────────────────────────────────────────

    # Por ahora devuelve vacío (sin tablas que reemplazar)
    return {}


# ==============================
# MAIN
# ==============================
def main():
    try:
        # 1. Autenticación OAuth2 con tu cuenta de Google
        #    Lee GOOGLE_OAUTH_CLIENT desde variables de entorno
        docs_service, drive_service, sheets_service = funciones_docs.autenticar_oauth()
        print("✅ Autenticación exitosa")

        # 2. Descargar el Excel desde Drive
        archivo_excel = funciones_docs.descargar_excel_drive(EXCEL_FILE_ID, drive_service)
        print("✅ Archivo Excel descargado desde Drive")

        # 3. Construir el diccionario unificado
        diccionario = funciones_docs.cargar_diccionario(archivo_excel)
        print("✅ Diccionario cargado correctamente")

        # 4. Extraer IDs desde las URLs
        doc_id_plantilla   = funciones_docs.extraer_id_gdoc(URL_PLANTILLA)
        carpeta_destino_id = funciones_docs.extraer_id_gdoc(URL_CARPETA_DESTINO)

        # 5. Crear una copia del documento plantilla en tu Drive
        doc_id_nuevo = funciones_docs.copiar_documento(
            doc_id             = doc_id_plantilla,
            nombre_nuevo       = NOMBRE_DOCUMENTO,
            drive_service      = drive_service,
            carpeta_destino_id = carpeta_destino_id
        )
        print(f"✅ Copia creada con ID: {doc_id_nuevo}")

        # 6. Procesar condicionales ({% if %}...{% endif %})
        #    Debe ejecutarse ANTES del reemplazo de texto y tablas
        funciones_docs.procesar_condicionales(doc_id_nuevo, diccionario, docs_service)

        # 7. Rellenar tablas con DataFrames
        #    Debe ejecutarse ANTES de reemplazar_textos para que el placeholder
        #    de la tabla aún exista cuando se busca la tabla
        tablas_data = _construir_tablas_data()
        resultado_tabla = funciones_docs.reemplazar_tablas(doc_id_nuevo, tablas_data, docs_service)

        # 8. Reemplazar textos en la copia
        resultado_texto = funciones_docs.reemplazar_textos(doc_id_nuevo, diccionario, docs_service)

        # 9. Reemplazar imágenes en la copia
        resultado_imagen = funciones_docs.reemplazar_imagenes(doc_id_nuevo, diccionario, docs_service, drive_service)

        print("\n✅ Proceso completado")
        print(f"   Tablas reemplazadas:   {len(resultado_tabla['reemplazados'])}")
        print(f"   Textos reemplazados:   {len(resultado_texto['reemplazados'])}")
        print(f"   Imágenes reemplazadas: {len(resultado_imagen['reemplazados'])}")

    except Exception as e:
        print("❌ ERROR:")
        print(e)


# ==============================
# ENTRY POINT
# ==============================
if __name__ == "__main__":
    main()
