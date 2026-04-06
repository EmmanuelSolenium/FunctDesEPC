import pandas as pd
import re

def cargar_diccionario(archivo, project_filter=None):
    """
    archivo puede ser:
    - ruta local (str)
    - archivo en memoria (BytesIO desde Drive)
    """

    # Detectar tipo de entrada
    if isinstance(archivo, str):
        if archivo.endswith(".csv"):
            df = pd.read_csv(archivo)
        else:
            df = pd.read_excel(archivo)
    else:
        # archivo en memoria (Drive)
        df = pd.read_excel(archivo)

    # Usar la primera fila como header real
    df.columns = df.iloc[0]
    df = df[1:].reset_index(drop=True)

    # Normalizar nombres de columnas
    df.columns = [str(col).strip().lower() for col in df.columns]

    # Renombrar por seguridad
    df = df.rename(columns={
        "nombre del dato": "placeholder",
        "valor": "value",
        "tipo": "type",
        "alias": "alias"
    })

    data = {}

    for _, row in df.iterrows():
        placeholder = str(row.get("placeholder", "")).strip()
        value = row.get("value")
        tipo = str(row.get("type", "text")).lower()
        alias = str(row.get("alias", "")).strip()

        key = limpiar_placeholder(placeholder)
        value = convertir_tipo(value, tipo)

        final_key = alias if alias else key
        data[final_key] = value

    return data

def limpiar_placeholder(texto):
    """
    Convierte '{{ Voltaje }}' → 'Voltaje'
    """
    return re.sub(r"[{}]", "", texto).strip()


def convertir_tipo(valor, tipo):
    if pd.isna(valor):
        return None

    if tipo == "bool":
        return str(valor).lower() in ["true", "1", "si", "yes"]

    elif tipo == "int":
        return int(valor)

    elif tipo == "float":
        return float(valor)

    elif tipo == "text":
        return str(valor)

    return valor



def descargar_excel_drive(file_id, drive_service):
    """
    Descarga un archivo de Google Drive usando su file_id
    y lo carga en memoria como BytesIO.
    """
    request = drive_service.files().get_media(fileId=file_id)

    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)

    done = False
    while not done:
        status, done = downloader.next_chunk()

    fh.seek(0)
    return fh