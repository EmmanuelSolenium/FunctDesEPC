"""
zona_contaminacion.py
======================

Determina el nivel de contaminacion ("Altamente contaminado" o
"Contaminacion normal") de un punto, segun el mapa "Figura 2 - Mapa de
Niveles de Contaminacion" de la region Caribe de Colombia.

COMO FUNCIONA (y por que este mapa es MAS incierto que el de zonas de viento)
------------------------------------------------------------------------------
Se uso la misma idea que en "zona_viento.py": ubicar puntos de control (en
pixeles) de ciudades con coordenadas reales conocidas, y ajustar una
transformacion para poder convertir cualquier (lat, lon) a un pixel de la
imagen.

La diferencia importante es que ESTE mapa no tiene las ciudades marcadas
con nombre (solo aparecen puntos negros sin etiqueta, y los nombres que sí
aparecen son de los departamentos, no de las ciudades). Para ubicar los 7
puntos de control se identificaron los puntos negros que, por su posicion
relativa dentro de cada departamento y su cercania a la costa o accidentes
geograficos reconocibles (la curva de la costa en Santa Marta, la bahia de
Cartagena, etc.), corresponden con mayor probabilidad a Riohacha, Santa
Marta, Barranquilla, Cartagena, Valledupar, Sincelejo y Monteria.

Esto quiere decir que la calibracion de ESTE mapa es menos confiable que la
del mapa de viento (donde las ciudades SI estaban rotuladas y marcadas con
un circulo de color). Se valido cada punto verificando que las ciudades
costeras (Riohacha, Santa Marta, Barranquilla, Cartagena, Sincelejo) caen
dentro de la franja gris ("Altamente contaminado") y que las ciudades del
interior (Valledupar, Monteria) caen en zona blanca ("Contaminacion
normal"), que es justamente lo que se observa en la imagen. Aun asi, cerca
de los bordes de la franja gris el resultado puede equivocarse.

Se uso ademas una interpolacion por triangulacion (en vez de una unica
transformacion afin global) para que la posicion de las 7 ciudades de
referencia sea exacta y el error se reparta solo entre ellas, en vez de
acumularse en todo el mapa. Adicionalmente se agregaron 2 puntos sobre el
perimetro/costa del mapa (la punta mas al norte y la punta mas al este de
la peninsula de La Guajira, con coordenadas reales indicadas por el
usuario) para mejorar la precision en esa zona, donde antes no habia
ningun punto de control cercano.

LIMITACIONES
------------
* Igual que con el mapa de viento, esto solo cubre la región Caribe
  (Atlántico, Magdalena, Guajira, Cesar, Bolívar, Sucre, Córdoba).
* La franja "Altamente contaminado" es angosta en varios tramos; cerca de
  sus bordes la clasificacion puede tener error de varios kilometros.
* Si tienes forma de confirmar la ubicacion real de alguna de las 7
  ciudades de referencia (o de agregar mas puntos de control), edita la
  lista CONTROL_POINTS para mejorar la precision.

USO
---
    from zona_contaminacion import obtener_nivel_contaminacion

    resultado = obtener_nivel_contaminacion(10.9639, -74.7964)  # Barranquilla
    print(resultado)

Tambien se puede ejecutar desde la terminal:

    python3 zona_contaminacion.py 10.9639 -74.7964
"""

import sys
import os
import numpy as np
from PIL import Image
from scipy.spatial import Delaunay

# ---------------------------------------------------------------------------
# Puntos de control: (nombre, pixel_x, pixel_y, lat, lon)
# Ubicados sobre "mapa_contaminacion.png" (789x867 px).
# ---------------------------------------------------------------------------
CONTROL_POINTS = [
    ("Riohacha",      712, 143, 11.5444, -72.9072),
    ("Santa Marta",   406, 291, 11.2408, -74.1990),
    ("Barranquilla",  304, 328, 10.9639, -74.7964),
    ("Cartagena",     215, 447, 10.3910, -75.4794),
    ("Valledupar",    621, 277, 10.4631, -73.2532),
    ("Sincelejo",     266, 568,  9.3047, -75.3978),
    ("Monteria",      318, 610,  8.7479, -75.8814),
    # Puntos sobre el perimetro/costa de la peninsula de La Guajira
    # (coordenadas reales proporcionadas por el usuario, no son ciudades).
    ("Punta norte Guajira", 713, 141, 12.458389735254798, -71.66858282234507),
    ("Punta este Guajira",  787, 197, 12.049824621847678, -71.11335219163357),
]

RUTA_IMAGEN_DEFECTO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mapa_contaminacion.png")

# Umbral: si la fraccion de pixeles "gris/achurado" en la ventana es mayor o
# igual a esto, se considera "Altamente contaminado".
UMBRAL_CONFIANZA = 0.15


def _construir_triangulacion(control_points):
    lonlat = np.array([[lon, lat] for (_, px, py, lat, lon) in control_points])
    pix = np.array([[px, py] for (_, px, py, lat, lon) in control_points], dtype=float)
    tri = Delaunay(lonlat)
    return tri, lonlat, pix


_TRI, _LONLAT, _PIX = _construir_triangulacion(CONTROL_POINTS)


def latlon_a_pixel(lat, lon):
    """Convierte (lat, lon) al pixel correspondiente, interpolando entre los
    puntos de control mas cercanos (triangulacion de Delaunay)."""
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


def _es_gris_achurado(r, g, b):
    r, g, b = int(r), int(g), int(b)
    return abs(r - g) < 8 and abs(g - b) < 8 and 140 <= r <= 245


_IMG_CACHE = {}


def _cargar_imagen(ruta_imagen):
    if ruta_imagen not in _IMG_CACHE:
        img = Image.open(ruta_imagen).convert("RGB")
        _IMG_CACHE[ruta_imagen] = np.array(img)
    return _IMG_CACHE[ruta_imagen]


def obtener_nivel_contaminacion(lat, lon, ruta_imagen=RUTA_IMAGEN_DEFECTO, ventana=15):
    """
    Devuelve el nivel de contaminacion correspondiente a una coordenada.

    Retorna un dict con:
        nivel      : 'Alto', 'Normal' o None si no se pudo determinar
        pixel      : (x, y) pixel estimado en la imagen
        confianza  : proporcion de pixeles "gris/achurado" en la ventana
        mensaje    : explicacion cuando nivel es None
    """
    arr = _cargar_imagen(ruta_imagen)
    h, w, _ = arr.shape

    px, py = latlon_a_pixel(lat, lon)
    x, y = int(round(px)), int(round(py))

    resultado = {"nivel": None, "pixel": (round(px, 1), round(py, 1)),
                 "confianza": 0.0, "mensaje": None}

    if x < 0 or y < 0 or x >= w or y >= h:
        resultado["mensaje"] = (
            "El punto cae fuera del area cubierta por el mapa (fuera de la region "
            "Caribe de Colombia representada en la imagen)."
        )
        return resultado

    half = ventana // 2
    grises = 0
    total = 0
    for yy in range(max(0, y - half), min(h, y + half + 1)):
        for xx in range(max(0, x - half), min(w, x + half + 1)):
            total += 1
            if _es_gris_achurado(*arr[yy, xx]):
                grises += 1

    confianza = grises / total if total else 0.0
    resultado["confianza"] = round(confianza, 2)
    resultado["nivel"] = "Alto" if confianza >= UMBRAL_CONFIANZA else "Normal"
    return resultado


if __name__ == "__main__":
    if len(sys.argv) == 3:
        lat, lon = float(sys.argv[1]), float(sys.argv[2])
        print(obtener_nivel_contaminacion(lat, lon))
    else:
        print("Uso: python3 zona_contaminacion.py <lat> <lon>\n")
        print("Ejemplos de prueba (puntos de control):\n")
        for nombre, px, py, lat, lon in CONTROL_POINTS:
            r = obtener_nivel_contaminacion(lat, lon)
            print(f"  {nombre:15s} lat={lat:8.4f} lon={lon:9.4f} -> nivel {r['nivel']:6s} "
                  f"confianza={r['confianza']}")
