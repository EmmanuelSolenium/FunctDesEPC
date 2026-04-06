import pandas as pd
import re

def cargar_diccionario(ruta_archivo, project_filter=None):
    # Leer archivo (soporta csv o excel)
    if ruta_archivo.endswith(".csv"):
        df = pd.read_csv(ruta_archivo)
    else:
        df = pd.read_excel(ruta_archivo)

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

        # limpiar {{ }}
        key = limpiar_placeholder(placeholder)

        # conversión de tipos
        value = convertir_tipo(value, tipo)

        # prioridad: alias > placeholder
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