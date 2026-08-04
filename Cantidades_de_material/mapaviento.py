"""
zona_viento.py
================

Determina a que zona de viento (segun el mapa de velocidades de viento de
la region Caribe de Colombia) pertenece un punto dado por sus coordenadas
geograficas (latitud, longitud).

COMO FUNCIONA
-------------
El mapa original (una imagen PNG) NO esta georreferenciado: es un dibujo
esquematico donde cada zona se distingue por un patron/color de relleno
(rayado morado, rayado gris, achurado verde). Para poder "consultar" el
mapa con coordenadas reales se hizo lo siguiente:

1. Se localizaron en la imagen los puntos (en pixeles) de varias ciudades
   cuya latitud/longitud real se conoce (Riohacha, Santa Marta,
   Barranquilla, Cartagena, Valledupar, Sincelejo y Monteria).
2. Con esos "puntos de control" se ajusta una transformacion afin que
   convierte (lat, lon) -> (pixel_x, pixel_y) por minimos cuadrados.
3. Dado un punto cualquiera, se calcula su pixel correspondiente y se
   examina una pequena ventana de pixeles alrededor de ese punto. Cada
   pixel de la ventana se clasifica por su color (morado/gris/verde) y se
   toma la clase mas frecuente como la zona del punto. Usar una ventana
   (en vez de un solo pixel) es necesario porque las zonas estan rellenas
   con patrones de rayas/achurado que dejan huecos blancos entre lineas.

LIMITACIONES IMPORTANTES (leelas antes de confiar en el resultado)
--------------------------------------------------------------------
* El mapa es un dibujo esquematico, no una proyeccion cartografica exacta.
  La transformacion afin es una buena aproximacion cerca de las ciudades
  usadas como puntos de control, pero puede tener errores mayores en los
  bordes del mapa (p. ej. la punta de la peninsula de La Guajira, o el
  extremo sur de Bolivar/Cesar), donde no hay puntos de control cercanos.
* El mapa solo cubre los departamentos de la costa Caribe (Atlantico,
  Magdalena, Guajira, Cesar, Bolivar, Sucre, Cordoba). Coordenadas fuera
  de esa region (el resto de Colombia) no tienen zona definida.
* Si necesitas mas precision, la forma mas facil de mejorarlo es agregar
  mas puntos de control (ciudades o accidentes geograficos identificables
  en la imagen) a la lista CONTROL_POINTS mas abajo.

USO
---
    from zona_viento import obtener_zona_viento

    resultado = obtener_zona_viento(10.9639, -74.7964)  # Barranquilla
    print(resultado)
    # {'zona': 'A', 'velocidad_kmh': '130-80', 'confianza': 1.0, ...}

Tambien se puede ejecutar directamente desde la terminal:

    python3 zona_viento.py 10.9639 -74.7964
"""

import sys
import os
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# 1. Puntos de control: (nombre, pixel_x, pixel_y, lat, lon)
#    Los pixeles se ubicaron inspeccionando la imagen "mapa_viento.png"
#    (606x438 px). Si cambias de imagen o de resolucion, hay que volver a
#    ubicar estos puntos.
# ---------------------------------------------------------------------------
CONTROL_POINTS = [
    ("Riohacha",     443,  63, 11.5444, -72.9072),
    ("Santa Marta",  303, 101, 11.2408, -74.1990),
    ("Barranquilla", 244, 121, 10.9639, -74.7964),
    ("Cartagena",    191, 167, 10.3910, -75.4794),
    ("Valledupar",   376, 165, 10.4631, -73.2532),
    ("Sincelejo",    196, 263,  9.3047, -75.3978),
    ("Monteria",     151, 297,  8.7479, -75.8814),
]

# Informacion de cada zona segun la leyenda ("Convenciones") del mapa.
ZONAS = {
    "A": {"velocidad_kmh": "130-80", "descripcion": "Zona A (rayado morado/lila vertical)"},
    "B": {"velocidad_kmh": "100-60", "descripcion": "Zona B (rayado gris horizontal)"},
    "C": {"velocidad_kmh": "80-50",  "descripcion": "Zona C (achurado verde)"},
}

RUTA_IMAGEN_DEFECTO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mapa_viento.png")


def _ajustar_transformacion(control_points):
    """Ajusta por minimos cuadrados: pixel = a*lon + b*lat + c (una ecuacion para x, otra para y)."""
    A = np.array([[lon, lat, 1.0] for (_, px, py, lat, lon) in control_points])
    bx = np.array([px for (_, px, py, lat, lon) in control_points], dtype=float)
    by = np.array([py for (_, px, py, lat, lon) in control_points], dtype=float)
    coef_x, *_ = np.linalg.lstsq(A, bx, rcond=None)
    coef_y, *_ = np.linalg.lstsq(A, by, rcond=None)
    return coef_x, coef_y


_COEF_X, _COEF_Y = _ajustar_transformacion(CONTROL_POINTS)


def latlon_a_pixel(lat, lon):
    """Convierte una coordenada (lat, lon) al pixel correspondiente en la imagen del mapa."""
    px = _COEF_X[0] * lon + _COEF_X[1] * lat + _COEF_X[2]
    py = _COEF_Y[0] * lon + _COEF_Y[1] * lat + _COEF_Y[2]
    return px, py


def _clasificar_color(r, g, b):
    """Clasifica un pixel en zona 'A' (morado), 'B' (gris), 'C' (verde) o None (blanco/otro)."""
    r, g, b = int(r), int(g), int(b)
    if g - r > 35 and g - b > 35 and g > 140:
        return "C"
    if b - r > 12 and b - g > 12 and r > 120 and g > 120:
        return "A"
    if abs(r - g) < 10 and abs(g - b) < 10 and 170 <= r <= 236:
        return "B"
    return None


_IMG_CACHE = {}


def _cargar_imagen(ruta_imagen):
    if ruta_imagen not in _IMG_CACHE:
        img = Image.open(ruta_imagen).convert("RGB")
        _IMG_CACHE[ruta_imagen] = np.array(img)
    return _IMG_CACHE[ruta_imagen]


def obtener_zona_viento(lat, lon, ruta_imagen=RUTA_IMAGEN_DEFECTO, ventana=15):
    """
    Devuelve la zona de viento correspondiente a una coordenada (lat, lon).

    Parametros
    ----------
    lat, lon : float
        Coordenadas geograficas del punto a consultar.
    ruta_imagen : str
        Ruta al archivo de imagen del mapa (por defecto "mapa_viento.png"
        en la misma carpeta que este script).
    ventana : int
        Tamano (en pixeles) del lado de la ventana cuadrada usada para
        muestrear el color alrededor del pixel objetivo. Ventanas mas
        grandes son mas robustas al patron de rayado pero menos precisas
        cerca de los limites entre zonas.

    Retorna
    -------
    dict con las llaves:
        zona            : 'A', 'B', 'C' o None si no se pudo determinar
        velocidad_kmh   : rango de velocidad de la zona (o None)
        descripcion     : descripcion de la zona (o None)
        pixel           : (x, y) pixel estimado en la imagen
        confianza       : proporcion de pixeles de la ventana que
                           coincidieron con la zona elegida (0 a 1)
        mensaje         : explicacion cuando zona es None
    """
    arr = _cargar_imagen(ruta_imagen)
    h, w, _ = arr.shape

    px, py = latlon_a_pixel(lat, lon)
    x, y = int(round(px)), int(round(py))

    resultado_base = {"zona": None, "velocidad_kmh": None, "descripcion": None,
                       "pixel": (round(px, 1), round(py, 1)), "confianza": 0.0, "mensaje": None}

    if x < 0 or y < 0 or x >= w or y >= h:
        resultado_base["mensaje"] = (
            "El punto cae fuera del area cubierta por el mapa (fuera de la region "
            "Caribe de Colombia representada en la imagen)."
        )
        return resultado_base

    half = ventana // 2
    conteos = {"A": 0, "B": 0, "C": 0}
    total_pixeles_ventana = 0
    for yy in range(max(0, y - half), min(h, y + half + 1)):
        for xx in range(max(0, x - half), min(w, x + half + 1)):
            total_pixeles_ventana += 1
            clase = _clasificar_color(*arr[yy, xx])
            if clase:
                conteos[clase] += 1

    total_clasificados = sum(conteos.values())
    if total_clasificados == 0:
        resultado_base["mensaje"] = (
            "No se encontro ningun color de zona en esa ubicacion (probablemente cae en "
            "el mar, fuera del pais, o en un area blanca del mapa)."
        )
        return resultado_base

    zona = max(conteos, key=conteos.get)
    confianza = conteos[zona] / total_pixeles_ventana

    resultado_base["zona"] = zona
    resultado_base["velocidad_kmh"] = ZONAS[zona]["velocidad_kmh"]
    resultado_base["descripcion"] = ZONAS[zona]["descripcion"]
    resultado_base["confianza"] = round(confianza, 2)
    return resultado_base


if __name__ == "__main__":
    if len(sys.argv) == 3:
        lat, lon = float(sys.argv[1]), float(sys.argv[2])
        print(obtener_zona_viento(lat, lon))
    else:
        print("Uso: python3 zona_viento.py <lat> <lon>\n")
        print("Ejemplos de prueba (ciudades usadas como puntos de control):\n")
        for nombre, px, py, lat, lon in CONTROL_POINTS:
            r = obtener_zona_viento(lat, lon)
            print(f"  {nombre:15s} lat={lat:8.4f} lon={lon:9.4f} -> zona {r['zona']} "
                  f"({r['velocidad_kmh']} km/h), confianza={r['confianza']}")