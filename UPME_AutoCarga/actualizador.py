"""
Actualizaciones automáticas desde GitHub.

Compara el version.json local (carpeta del programa) con el publicado en
  github.com/EmmanuelSolenium/FunctDesEPC  ->  UPME_AutoCarga/version.json
Si hay una versión mayor:
  - Normal: descarga solo los archivos listados, verifica su SHA-256 y los
    reemplaza; luego reinicia el programa.
  - Si la nueva versión exige un paquete más nuevo que el instalado
    (_runtime/paquete.txt < "paquete_minimo"), abre la carpeta de Drive del
    ZIP nuevo ("url_paquete"). El ZIP no se distribuye por GitHub.
Cualquier fallo de red se ignora: el programa abre con la versión actual.
Para desactivarlo (desarrollo): variable de entorno UPME_SIN_ACTUALIZAR=1.
"""
import hashlib
import json
import os
import shutil
import ssl
import subprocess
import sys
import urllib.parse
import urllib.request

REPO = "EmmanuelSolenium/FunctDesEPC"
RAMA = "main"
CARPETA_REPO = "UPME_AutoCarga"
TIMEOUT = 8

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VERSION_LOCAL = os.path.join(BASE_DIR, "version.json")
PAQUETE_LOCAL = os.path.join(BASE_DIR, "_runtime", "paquete.txt")
TMP_DIR = os.path.join(BASE_DIR, "_actualizacion_tmp")


def _version_tuple(v):
    try:
        return tuple(int(p) for p in str(v).strip().split("."))
    except ValueError:
        return (0,)


def leer_version_local():
    try:
        with open(VERSION_LOCAL, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"version": "0.0.0", "archivos": {}}


def version_paquete_local():
    try:
        with open(PAQUETE_LOCAL, encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return "0.0.0"


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "UPME-AutoCarga"})
    with urllib.request.urlopen(req, timeout=TIMEOUT, context=ssl.create_default_context()) as r:
        return r.read()


def _commit_actual():
    """SHA del último commit de la carpeta, para descargar todo de una misma
    versión (raw.githubusercontent cachea 'main' unos minutos)."""
    try:
        url = f"https://api.github.com/repos/{REPO}/commits?sha={RAMA}&path={CARPETA_REPO}&per_page=1"
        return json.loads(_get(url))[0]["sha"]
    except Exception:
        return RAMA


def _url_raw(ref, nombre):
    return f"https://raw.githubusercontent.com/{REPO}/{ref}/{CARPETA_REPO}/{urllib.parse.quote(nombre)}"


def exe_real():
    """Ruta del ejecutable real ("UPME AutoCarga.exe"); sys.executable dice pythonw.exe."""
    try:
        import ctypes
        buf = ctypes.create_unicode_buffer(32768)
        ctypes.windll.kernel32.GetModuleFileNameW(None, buf, len(buf))
        return buf.value
    except Exception:
        return sys.executable


def reiniciar():
    subprocess.Popen([exe_real(), os.path.join(BASE_DIR, "iniciar.py")], cwd=BASE_DIR)
    os._exit(0)


def _aplicar(ref, remoto):
    """Descarga, verifica y reemplaza. Lanza excepción si algo no cuadra."""
    archivos = remoto.get("archivos", {})
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    os.makedirs(TMP_DIR)
    try:
        locales = leer_version_local().get("archivos", {})
        pendientes = [n for n, h in archivos.items()
                      if locales.get(n) != h or not os.path.isfile(os.path.join(BASE_DIR, n))]
        for nombre in pendientes:
            datos = _get(_url_raw(ref, nombre))
            if hashlib.sha256(datos).hexdigest() != archivos[nombre]:
                raise RuntimeError(f"El archivo descargado {nombre} no coincide con su huella SHA-256.")
            with open(os.path.join(TMP_DIR, nombre), "wb") as f:
                f.write(datos)
        for nombre in pendientes:
            os.replace(os.path.join(TMP_DIR, nombre), os.path.join(BASE_DIR, nombre))
        # version.json al final: si algo falla antes, se reintenta la próxima vez.
        with open(VERSION_LOCAL, "w", encoding="utf-8") as f:
            json.dump(remoto, f, ensure_ascii=False, indent=2)
    finally:
        shutil.rmtree(TMP_DIR, ignore_errors=True)


def verificar(parent=None):
    """Busca y ofrece actualizaciones. Si se actualiza, reinicia el programa
    (no retorna). En cualquier otro caso retorna y el programa sigue normal."""
    if os.environ.get("UPME_SIN_ACTUALIZAR") == "1":
        return
    from tkinter import messagebox

    local = leer_version_local()
    try:
        ref = _commit_actual()
        remoto = json.loads(_get(_url_raw(ref, "version.json")))
    except Exception:
        return  # sin internet / GitHub caído: seguir con la versión actual
    if _version_tuple(remoto.get("version")) <= _version_tuple(local.get("version")):
        return

    v_nueva = remoto.get("version")
    notas = remoto.get("notas", "").strip()
    detalle = f"\n\nCambios:\n{notas}" if notas else ""

    if _version_tuple(remoto.get("paquete_minimo", "0")) > _version_tuple(version_paquete_local()):
        url = remoto.get("url_paquete")  # carpeta de Drive con el ZIP
        if not url:
            messagebox.showinfo(
                "Actualización disponible",
                f"Hay una nueva versión {v_nueva} (tienes {local.get('version')}).{detalle}\n\n"
                "Esta versión requiere instalar el paquete completo (ZIP) de nuevo.\n"
                "Pídale el ZIP nuevo al responsable del programa.",
                parent=parent,
            )
            return
        if messagebox.askyesno(
            "Actualización disponible",
            f"Hay una nueva versión {v_nueva} (tienes {local.get('version')}).{detalle}\n\n"
            "Esta versión requiere instalar el paquete completo (ZIP) de nuevo.\n"
            "¿Abrir la carpeta de Drive para descargarlo?",
            parent=parent,
        ):
            import webbrowser
            webbrowser.open(url)
        return

    if not messagebox.askyesno(
        "Actualización disponible",
        f"Hay una nueva versión {v_nueva} (tienes {local.get('version')}).{detalle}\n\n¿Actualizar ahora?",
        parent=parent,
    ):
        return
    try:
        _aplicar(ref, remoto)
    except Exception as e:
        messagebox.showerror("Actualización", f"No se pudo actualizar:\n{e}\n\nSe abrirá la versión actual.",
                             parent=parent)
        return
    messagebox.showinfo("Actualización", f"Actualizado a la versión {v_nueva}. El programa se reiniciará.",
                        parent=parent)
    reiniciar()
