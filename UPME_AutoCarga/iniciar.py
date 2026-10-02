"""
Punto de entrada de "UPME AutoCarga.exe" (lo lanza _runtime/sitecustomize.py).

1. Busca actualizaciones en GitHub antes de importar el resto del programa,
   así una actualización puede corregir incluso errores de los otros módulos.
2. La primera vez en cada instalación precarga los binarios de _runtime para
   que Smart App Control termine de evaluarlos (ver _precargar_binarios).
3. La primera vez ofrece crear un acceso directo en el escritorio.
4. Abre la interfaz. Como pythonw no tiene consola, cualquier error de
   arranque se muestra en una ventana.
"""
import os
import subprocess
import time
import tkinter as tk
import traceback
from tkinter import messagebox

import actualizador
import config_usuario

ERROR_BLOQUEO_SAC = 4551  # ERROR_SYSTEM_INTEGRITY_POLICY_VIOLATION


def _es_paquete():
    return os.path.basename(actualizador.exe_real()).lower().startswith("upme")


def _intentar_cargar(rutas):
    """Carga cada .pyd/.dll y devuelve las que Windows bloqueó por política."""
    import ctypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.LoadLibraryExW.restype = ctypes.c_void_p
    k32.FreeLibrary.argtypes = [ctypes.c_void_p]
    # SEM_FAILCRITICALERRORS: sin esto, cada bloqueo abre la ventana de Windows
    # "...no está diseñado para ejecutarse en Windows... 0xc0e90002" y el
    # usuario tiene que cerrarla a mano; así el reintento ocurre en silencio.
    anterior = ctypes.c_uint()
    k32.SetThreadErrorMode(0x0001, ctypes.byref(anterior))
    bloqueadas = []
    try:
        for ruta in rutas:
            # 0x1100 = SEARCH_DLL_LOAD_DIR | SEARCH_DEFAULT_DIRS. Si falta una
            # dependencia (error 126) no importa: la verificación de firma ya ocurrió.
            h = k32.LoadLibraryExW(ruta, None, 0x1100)
            if h:
                k32.FreeLibrary(h)
            elif ctypes.get_last_error() == ERROR_BLOQUEO_SAC:
                bloqueadas.append(ruta)
    finally:
        k32.SetThreadErrorMode(anterior.value, None)
    return bloqueadas


def _precargar_binarios(root):
    """Smart App Control evalúa la reputación de cada binario sin firma la
    primera vez que se carga y, si la consulta en la nube no ha terminado, lo
    bloquea; al reintentar suele permitirlo y Windows recuerda la decisión.
    Se hace una vez por instalación para que el fallo no aparezca a mitad de
    un import (p. ej. pandas)."""
    marca = f"{actualizador.version_paquete_local()}|{actualizador.BASE_DIR}"
    if not _es_paquete() or config_usuario.leer_config().get("binarios_ok") == marca:
        return True
    rutas = [os.path.join(d, f)
             for d, _, fs in os.walk(os.path.join(actualizador.BASE_DIR, "_runtime"))
             for f in fs if f.lower().endswith((".pyd", ".dll"))]
    bloqueadas = _intentar_cargar(rutas)
    aviso = None
    for _ in range(10):
        if not bloqueadas:
            break
        if aviso is None:
            aviso = tk.Toplevel(root)
            aviso.title("UPME AutoCarga")
            tk.Label(aviso, text="Preparando el programa por primera vez...\n"
                                 "Windows está verificando sus componentes.", padx=30, pady=20).pack()
            aviso.update()
        time.sleep(3)
        bloqueadas = _intentar_cargar(bloqueadas)
    if aviso is not None:
        aviso.destroy()
    if bloqueadas:
        nombres = "\n".join(os.path.relpath(r, actualizador.BASE_DIR) for r in bloqueadas[:8])
        messagebox.showerror(
            "UPME AutoCarga",
            "Windows (Control inteligente de aplicaciones) bloqueó estos componentes:\n\n"
            f"{nombres}\n\nCierre y vuelva a abrir el programa en unos minutos. "
            "Si el problema continúa, avise al responsable del programa.",
            parent=root,
        )
        return False
    config_usuario.guardar_config(binarios_ok=marca)
    return True


def _crear_acceso_directo():
    exe = actualizador.exe_real().replace("'", "''")
    ps = (
        "$d=[Environment]::GetFolderPath('Desktop');"
        "$s=(New-Object -ComObject WScript.Shell).CreateShortcut((Join-Path $d 'UPME AutoCarga.lnk'));"
        f"$s.TargetPath='{exe}';$s.WorkingDirectory='{os.path.dirname(exe)}';"
        f"$s.IconLocation='{exe},0';$s.Save()"
    )
    subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                   creationflags=subprocess.CREATE_NO_WINDOW, timeout=30, check=True)


def _ofrecer_acceso_directo(root):
    if config_usuario.leer_config().get("acceso_directo_ofrecido"):
        return
    config_usuario.guardar_config(acceso_directo_ofrecido=True)
    if not _es_paquete():
        return  # ejecutado con python.exe (desarrollo)
    if messagebox.askyesno("UPME AutoCarga", "¿Crear un acceso directo en el escritorio?", parent=root):
        try:
            _crear_acceso_directo()
        except Exception as e:
            messagebox.showwarning("UPME AutoCarga", f"No se pudo crear el acceso directo:\n{e}", parent=root)


def main():
    root = tk.Tk()
    root.withdraw()
    try:
        # Primero la actualización (solo usa la librería estándar, firmada):
        # así una corrección llega aunque Windows esté bloqueando binarios.
        actualizador.verificar(parent=root)
        if not _precargar_binarios(root):
            return
        _ofrecer_acceso_directo(root)
        import upme_autocarga_gui
    except Exception:
        messagebox.showerror("UPME AutoCarga", f"No se pudo iniciar el programa:\n\n{traceback.format_exc()[-1500:]}",
                             parent=root)
        return
    finally:
        root.destroy()
    upme_autocarga_gui.main()


if __name__ == "__main__":
    main()
