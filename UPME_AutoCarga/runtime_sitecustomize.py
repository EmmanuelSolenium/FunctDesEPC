# ==========================================================================
#  Lanzador de "UPME AutoCarga.exe" (copia renombrada de pythonw.exe, firmada
#  por la Python Software Foundation, que Smart App Control si deja ejecutar).
#  Python importa este archivo al arrancar. Si el ejecutable se llama
#  "UPME AutoCarga.exe" y se abrio sin argumentos (doble clic), vuelve a
#  lanzarse a si mismo con iniciar.py y termina.
# ==========================================================================
import os
import sys


def _exe_real():
    # sys.executable siempre dice "pythonw.exe"; se pide a Windows el nombre real.
    import ctypes
    buf = ctypes.create_unicode_buffer(32768)
    ctypes.windll.kernel32.GetModuleFileNameW(None, buf, len(buf))
    return buf.value


def _lanzar():
    exe = _exe_real()
    if not os.path.basename(exe).lower().startswith("upme"):
        return
    base = os.path.dirname(exe)
    rt = os.path.join(base, "_runtime")
    os.environ.setdefault("TCL_LIBRARY", os.path.join(rt, "tcl", "tcl8.6"))
    os.environ.setdefault("TK_LIBRARY", os.path.join(rt, "tcl", "tk8.6"))
    if os.environ.get("UPME_HIJO") == "1":
        return
    os.environ["UPME_HIJO"] = "1"
    # iniciar.py busca actualizaciones antes de abrir la GUI (paquetes viejos
    # sin iniciar.py abren la GUI directamente).
    entrada = os.path.join(base, "iniciar.py")
    if not os.path.isfile(entrada):
        entrada = os.path.join(base, "upme_autocarga_gui.py")
    try:
        import subprocess
        subprocess.Popen([exe, entrada], cwd=base)
    except Exception as e:
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, f"No se pudo iniciar UPME AutoCarga:\n{e}",
                                         "UPME AutoCarga", 0x10)
    os._exit(0)


_lanzar()
