"""
Datos propios de cada usuario/PC, guardados FUERA de la carpeta del programa
(en %APPDATA%\\UPME AutoCarga) para que las actualizaciones no los borren y
para que nunca viajen dentro del ZIP ni del repositorio:

  - .env          credenciales de Bizagi (importadas o guardadas desde la GUI)
  - config.json   preferencias (maestro, etc.)
"""
import json
import os
import sys

APP_DATA_DIR = os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~"), "UPME AutoCarga")
ENV_PATH = os.path.join(APP_DATA_DIR, ".env")
CONFIG_PATH = os.path.join(APP_DATA_DIR, "config.json")

CLAVE_USUARIO = "BIZAGI_USER_NEW"
CLAVE_PASSWORD = "BIZAGI_PASSWORD_NEW"


def carpeta_programa():
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def _asegurar_carpeta():
    os.makedirs(APP_DATA_DIR, exist_ok=True)


# ---------------- Preferencias ---------------- #
def leer_config():
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def guardar_config(**cambios):
    cfg = leer_config()
    cfg.update(cambios)
    _asegurar_carpeta()
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


# ---------------- Credenciales ---------------- #
def leer_credenciales_de(path):
    """Devuelve el dict de variables de un archivo .env (vacío si no existe)."""
    from dotenv import dotenv_values
    if not os.path.isfile(path):
        return {}
    return {k: v for k, v in dotenv_values(path).items() if v is not None}


def _escapar(valor):
    return "'" + valor.replace("\\", "\\\\").replace("'", "\\'") + "'"


def guardar_credenciales(usuario, password):
    """Guarda usuario/contraseña en el .env del usuario, conservando las demás claves."""
    datos = leer_credenciales_de(ENV_PATH)
    datos[CLAVE_USUARIO] = usuario
    datos[CLAVE_PASSWORD] = password
    _asegurar_carpeta()
    with open(ENV_PATH, "w", encoding="utf-8") as f:
        for k, v in datos.items():
            f.write(f"{k}={_escapar(v)}\n")


def importar_credenciales(origen):
    """Copia un .env compartido al .env del usuario. Devuelve (usuario, password)."""
    datos = leer_credenciales_de(origen)
    if not datos.get(CLAVE_USUARIO):
        raise ValueError(f"El archivo no contiene la clave {CLAVE_USUARIO}.")
    actuales = leer_credenciales_de(ENV_PATH)
    actuales.update(datos)
    _asegurar_carpeta()
    with open(ENV_PATH, "w", encoding="utf-8") as f:
        for k, v in actuales.items():
            f.write(f"{k}={_escapar(v)}\n")
    return datos.get(CLAVE_USUARIO, ""), datos.get(CLAVE_PASSWORD, "")


def borrar_credenciales():
    if os.path.isfile(ENV_PATH):
        os.remove(ENV_PATH)
