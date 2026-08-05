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
   Barranquilla, Cartagena, Valledupar, Sincelejo y Monteria), y ademas dos
   puntos sobre el perimetro/costa del mapa: la punta mas al norte y la
   punta mas al este de la peninsula de La Guajira (coordenadas reales
   proporcionadas por el usuario).
2. Con esos 9 puntos de control se construye una triangulacion (Delaunay)
   y, para cualquier coordenada, se interpola su posicion en pixeles usando
   el triangulo de puntos de control mas cercano (coordenadas
   baricentricas). Esto hace que la conversion sea EXACTA en los 9 puntos
   de control y razonablemente precisa en las zonas intermedias, evitando
   el error de extrapolacion que tendria una unica transformacion afin
   global (que es lo que se usaba en una version anterior de este script).
3. Dado un punto cualquiera, se calcula su pixel correspondiente y se
   examina una pequena ventana de pixeles alrededor de ese punto. Cada
   pixel de la ventana se clasifica por su color (morado/gris/verde) y se
   toma la clase mas frecuente como la zona del punto. Usar una ventana
   (en vez de un solo pixel) es necesario porque las zonas estan rellenas
   con patrones de rayas/achurado que dejan huecos blancos entre lineas.

LIMITACIONES IMPORTANTES (leelas antes de confiar en el resultado)
--------------------------------------------------------------------
* El mapa es un dibujo esquematico, no una proyeccion cartografica exacta.
  La interpolacion es exacta en los 9 puntos de control, pero para puntos
  MUY alejados de todos ellos (fuera del area que "cubren" en conjunto)
  el resultado se extrapola desde el triangulo mas cercano y puede tener
  mas error.
* El mapa solo cubre los departamentos de la costa Caribe (Atlantico,
  Magdalena, Guajira, Cesar, Bolivar, Sucre, Cordoba). Coordenadas fuera
  de esa region (el resto de Colombia) no tienen zona definida.
* Si necesitas mas precision, la forma mas facil de mejorarlo es agregar
  mas puntos de control (ciudades o accidentes geograficos identificables
  en la imagen, idealmente sobre el perimetro/costa) a la lista
  CONTROL_POINTS mas abajo.

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
from scipy.spatial import Delaunay

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
    # Puntos sobre el perimetro/costa de la peninsula de La Guajira
    # (coordenadas reales proporcionadas por el usuario, no son ciudades).
    ("Punta norte Guajira", 512,  9, 12.458389735254798, -71.66858282234507),
    ("Punta este Guajira",  556, 35, 12.049824621847678, -71.11335219163357),
]

# Informacion de cada zona segun la leyenda ("Convenciones") del mapa.
ZONAS = {
    "A": {"velocidad_kmh": "130-80", "descripcion": "Zona A (rayado morado/lila vertical)"},
    "B": {"velocidad_kmh": "100-60", "descripcion": "Zona B (rayado gris horizontal)"},
    "C": {"velocidad_kmh": "80-50",  "descripcion": "Zona C (achurado verde)"},
}

RUTA_IMAGEN_DEFECTO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mapa_viento.png")


def _construir_triangulacion(control_points):
    lonlat = np.array([[lon, lat] for (_, px, py, lat, lon) in control_points])
    pix = np.array([[px, py] for (_, px, py, lat, lon) in control_points], dtype=float)
    tri = Delaunay(lonlat)
    return tri, lonlat, pix


_TRI, _LONLAT, _PIX = _construir_triangulacion(CONTROL_POINTS)


def latlon_a_pixel(lat, lon):
    """Convierte una coordenada (lat, lon) al pixel correspondiente, interpolando
    entre los puntos de control mas cercanos (triangulacion de Delaunay). El
    resultado es exacto en los puntos de control y se interpola/extrapola
    linealmente en el resto."""
    punto = np.array([lon, lat])
    simplex = int(_TRI.find_simplex(np.array([[lon, lat]]))[0])
    if simplex == -1:
        # Fuera del area cubierta por los puntos de control: se extrapola
        # usando el triangulo asociado al punto de control mas cercano.
        d = np.sum((_LONLAT - punto) ** 2, axis=1)
        idx = int(np.argmin(d))
        simplex = 0
        for si, s in enumerate(_TRI.simplices):
            if idx in s:
                simplex = si
                break
    verts = _TRI.simplices[simplex]
    T = _TRI.transform[simplex, :2]
    r0 = _TRI.transform[simplex, 2]
    bary = T.dot(punto - r0)
    bary = np.append(bary, 1 - bary.sum())
    p = bary[0] * _PIX[verts[0]] + bary[1] * _PIX[verts[1]] + bary[2] * _PIX[verts[2]]
    return float(p[0]), float(p[1])


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
