# ==============================
# AJUSTE DE PATH (para imports)
# ==============================
import sys
import os
import io

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# ==============================
# IMPORTAR FUNCIONES
# ==============================
from autodocs import funciones_docs


# ==============================
# MAIN
# ==============================
def main():
    try:
        # 1. Obtener ruta de credenciales desde variable de entorno
        ruta_credenciales = funciones_docs.obtener_ruta_credenciales()
        print("Usando credenciales en:", ruta_credenciales)

        # 2. Autenticación
        docs_service, drive_service, sheets_service = funciones_docs.autenticar_servicio(
            ruta_credenciales
        )

        print("✅ Autenticación exitosa")

        # 3. ID del archivo en Drive (Diccionario Template.xlsx)
        file_id = "1t9utg6qjm4KG9tQec6DzwoR0SYuUy53o"

        # 4. Descargar archivo desde Drive
        archivo_excel = funciones_docs.descargar_excel_drive(file_id, drive_service)
        print("✅ Archivo descargado desde Drive")

        # 5. Convertir a diccionario
        diccionario = funciones_docs.cargar_diccionario(archivo_excel)
        print("✅ Diccionario cargado correctamente")
        print(diccionario)



        # 6. Mostrar resultado
        #print("\n--- DICCIONARIO ---")
        #for k, v in diccionario.items():
            #print(f"{k}: {v}")

    except Exception as e:
        print("❌ ERROR:")
        print(e)


# ==============================
# ENTRY POINT
# ==============================
if __name__ == "__main__":
    main()


