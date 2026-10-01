# UPME AutoCarga

Herramienta de escritorio (Windows) para cargar soportes técnicos a Bizagi (UPME)
y procesar el Formato 3 (PEPC + BOM).

- **Usuarios:** el ZIP de instalación se reparte por Google Drive (no por GitHub);
  sigan [LEEME.txt](LEEME.txt). Esta carpeta del repo solo sirve las actualizaciones.
- Las credenciales **no** están en este repositorio; cada usuario las importa desde
  un `.env` compartido de forma privada (formato en [.env.ejemplo](.env.ejemplo)).

## Cómo funciona la distribución

| Pieza | Dónde vive |
|---|---|
| Código (`*.py`) y `El_Papá.xlsx` | Esta carpeta del repo |
| Python embebido (`_runtime`, DLLs, `UPME AutoCarga.exe`) | Solo en el ZIP, en Google Drive |
| Credenciales y preferencias | `%APPDATA%\UPME AutoCarga\` de cada PC |
| Fichas técnicas (PDF) | Carpeta compartida, elegida con "Examinar..." |

Al abrir, `iniciar.py` llama a `actualizador.py`, que compara el `version.json` local
con el de esta carpeta en `main`. Si hay una versión mayor descarga los archivos
listados, verifica su SHA-256, los reemplaza y reinicia. Si `paquete_minimo` es mayor
que el `_runtime\paquete.txt` instalado, abre `url_paquete` (la carpeta de Drive)
para que el usuario descargue el ZIP nuevo.

## Publicar una versión (desde la carpeta de desarrollo)

```powershell
# Solo cambios de código / El_Papá.xlsx
.\publicar.ps1 -Version 1.0.1 -Notas "Qué cambió"

# Cambios en _runtime (nuevas librerías): genera también el ZIP y espera a que
# lo suba a Drive antes de publicar en GitHub
.\publicar.ps1 -Version 1.1.0 -Notas "Qué cambió" -PaqueteCompleto
```

`-UrlPaquete "<enlace de la carpeta de Drive>"` guarda el enlace en `version.json`
(se recuerda en las versiones siguientes).

`runtime_sitecustomize.py` es una copia de referencia de `_runtime\sitecustomize.py`
(el lanzador del `.exe`), que solo se distribuye dentro del ZIP.
