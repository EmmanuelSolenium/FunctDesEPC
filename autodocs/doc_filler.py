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
# VERBOSE
# Pon True para ver el detalle de cada paso en la consola
# ==============================
VERBOSE = True 


# ==============================
# CONFIGURACIÓN
# Reemplaza cada URL con el enlace de compartir del archivo en Drive
# (clic derecho → Compartir → Copiar enlace)
#
# NOTA: No hay URL_TABLAS global. Cada parámetro de tipo 'tabla' debe
# tener su propia URL en la columna 'Valor' del diccionario, lo que
# permite que cada tabla apunte a un Excel distinto.
# ==============================

# Excel con el diccionario de datos
URL_DICCIONARIO = "https://docs.google.com/spreadsheets/d/14VDGM9Yg0YyHByRdFT3526CmMbB-Ur4d/edit?usp=drive_link&ouid=109812277537374132162&rtpof=true&sd=true"

# Documento plantilla de Google Docs
URL_PLANTILLA = "https://docs.google.com/document/d/1s2k-y_WqI0HZhA10gaSgSj6RN_5sN5RTVyQ58INydv4/edit?usp=drive_link"

# Carpeta donde se guardará el documento generado
URL_CARPETA_DESTINO = "https://drive.google.com/drive/folders/1zUPZXuCLZyA63EK0HkzirBXY0Vw6CQkX?usp=drive_link"

# Nombre del documento generado
NOMBRE_DOCUMENTO = "Plantilla proyecto de redes"


# ==============================
# MAIN
# ==============================
def main():
    try:
        # 1. Autenticación OAuth2 con tu cuenta de Google
        docs_service, drive_service, sheets_service = funciones_docs.autenticar_oauth()
        funciones_docs.VERBOSE = VERBOSE

        # 2. Descargar el Excel del diccionario desde Drive
        file_id_diccionario = funciones_docs.extraer_id_gdoc(URL_DICCIONARIO)
        archivo_excel = funciones_docs.descargar_excel_drive(file_id_diccionario, drive_service)

        # 3. Construir el diccionario unificado
        #    Cada entrada de tipo 'tabla' debe tener su URL en la columna 'Valor'
        diccionario = funciones_docs.cargar_diccionario(archivo_excel)

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

        # 6. Procesar condicionales ({% if %}...{% endif %})
        #    Debe ejecutarse ANTES del reemplazo de texto
        funciones_docs.procesar_condicionales(doc_id_nuevo, diccionario, docs_service)

        # 7. Reemplazar textos en la copia
        resultado_texto = funciones_docs.reemplazar_textos(doc_id_nuevo, diccionario, docs_service)

        # 8. Reemplazar imágenes en la copia
        resultado_imagen = funciones_docs.reemplazar_imagenes(doc_id_nuevo, diccionario, docs_service, drive_service)

        # 9. Reemplazar tablas en la copia
        resultado_tabla = funciones_docs.reemplazar_tablas(doc_id_nuevo, diccionario, docs_service, drive_service)

        print("\nProceso completado")
        print(f"   Textos reemplazados:   {len(resultado_texto['reemplazados'])}")
        print(f"   Imagenes reemplazadas: {len(resultado_imagen['reemplazados'])}")
        print(f"   Tablas reemplazadas:   {len(resultado_tabla['reemplazados'])}")

    except Exception as e:
        print("ERROR:")
        print(e)


# ==============================
# ENTRY POINT
# ==============================
if __name__ == "__main__":
    main()