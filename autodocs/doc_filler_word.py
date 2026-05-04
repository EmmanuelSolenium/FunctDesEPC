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
# CONFIGURACION
# Reemplaza cada URL con el enlace de compartir del archivo en Drive
# (clic derecho -> Compartir -> Copiar enlace)
#
# NOTA: No hay URL_TABLAS global. Cada parametro de tipo 'tabla' debe
# tener su propia URL en la columna 'Valor' del diccionario, lo que
# permite que cada tabla apunte a un Excel distinto.
# ==============================

# Excel con el diccionario de datos
URL_DICCIONARIO = "https://docs.google.com/spreadsheets/d/14VDGM9Yg0YyHByRdFT3526CmMbB-Ur4d/edit?usp=drive_link&ouid=109812277537374132162&rtpof=true&sd=true"

# Plantilla en formato .docx en Google Drive  <- coloca aqui tu enlace
URL_PLANTILLA = "https://drive.google.com/file/d/REEMPLAZA_CON_ID_DE_TU_DOCX/view?usp=sharing"

# Carpeta donde se subira el documento generado
URL_CARPETA_DESTINO = "https://drive.google.com/drive/folders/1zUPZXuCLZyA63EK0HkzirBXY0Vw6CQkX?usp=drive_link"

# Nombre del documento generado (sin extension)
NOMBRE_DOCUMENTO = "Plantilla proyecto de redes"


# ==============================
# MAIN
# ==============================
def main():
    try:
        # 1. Autenticacion OAuth2 con tu cuenta de Google
        docs_service, drive_service, sheets_service = funciones_docs.autenticar_oauth()
        funciones_docs.VERBOSE = VERBOSE

        # 2. Descargar el Excel del diccionario desde Drive
        file_id_diccionario = funciones_docs.extraer_id_gdoc(URL_DICCIONARIO)
        archivo_excel = funciones_docs.descargar_excel_drive(file_id_diccionario, drive_service)

        # 3. Construir el diccionario unificado
        #    Cada entrada de tipo 'tabla' / 'imagen' / 'loop' debe tener su URL en la columna 'Valor'
        diccionario = funciones_docs.cargar_diccionario(archivo_excel)

        # 4. Extraer IDs desde las URLs
        doc_id_plantilla   = funciones_docs.extraer_id_gdoc(URL_PLANTILLA)
        carpeta_destino_id = funciones_docs.extraer_id_gdoc(URL_CARPETA_DESTINO)

        # 5. Descargar la plantilla .docx a un temporal local
        #    (sustituye la antigua copia en Drive)
        ruta_docx = funciones_docs.copiar_documento(
            doc_id             = doc_id_plantilla,
            nombre_nuevo       = NOMBRE_DOCUMENTO,
            drive_service      = drive_service,
            carpeta_destino_id = carpeta_destino_id,
        )

        # 6. Procesar condicionales ({% if %}...{% endif %})
        #    Debe ejecutarse ANTES del reemplazo de texto
        funciones_docs.procesar_condicionales(ruta_docx, diccionario)

        # 7. Reemplazar textos en el .docx local
        resultado_texto = funciones_docs.reemplazar_textos(ruta_docx, diccionario)

        # 8. Reemplazar imagenes en el .docx local
        resultado_imagen = funciones_docs.reemplazar_imagenes(ruta_docx, diccionario, None, drive_service)

        # 9. Reemplazar tablas estaticas en el .docx local
        resultado_tabla = funciones_docs.reemplazar_tablas(ruta_docx, diccionario, None, drive_service)

        # 10. Reemplazar loops (tablas dinamicas multi-hoja) en el .docx local
        resultado_loop = funciones_docs.reemplazar_loops(ruta_docx, diccionario, None, drive_service)

        # 11. Subir el .docx resultante a la carpeta destino en Drive
        nombre_archivo = NOMBRE_DOCUMENTO + ".docx"
        file_id_nuevo  = funciones_docs.subir_docx_drive(
            ruta_local    = ruta_docx,
            nombre        = nombre_archivo,
            carpeta_id    = carpeta_destino_id,
            drive_service = drive_service,
        )

        print("\nProceso completado")
        print(f"   Textos reemplazados:   {len(resultado_texto['reemplazados'])}")
        print(f"   Imagenes reemplazadas: {len(resultado_imagen['reemplazados'])}")
        print(f"   Tablas reemplazadas:   {len(resultado_tabla['reemplazados'])}")
        print(f"   Loops reemplazados:    {len(resultado_loop['reemplazados'])}")
        print(f"\n   Documento en Drive (ID): {file_id_nuevo}")
        print(f"   https://drive.google.com/file/d/{file_id_nuevo}/view")

    except Exception as e:
        import traceback
        print("ERROR:")
        traceback.print_exc()


# ==============================
# ENTRY POINT
# ==============================
if __name__ == "__main__":
    main()
