"""
zona_viento.py
================

Determina la zona de viento (A, B o C) de un punto de la Costa Caribe a partir de
sus coordenadas, según los mapas por departamento del Anexo A1 del "Proyecto Tipo
Líneas Eléctricas Aéreas MT sin Neutro" (Electricaribe v5, 2020), que usan AFINIA
y AIR-E:

    Zona A  130/80 km/h  (rural/urbana)
    Zona B  100/60 km/h
    Zona C   80/50 km/h

COMO FUNCIONA
-------------
Lee "zona_viento_caribe.json", una rejilla lon/lat (celda de 0,0025° ≈ 275 m) con
la zona, el departamento y el municipio de cada celda. La rejilla la genera
GeoElec (estudios-de-conexion-desktop, network_design_engine/tools/
generar_zona_viento_caribe.py): cada uno de los 7 mapas departamentales del Anexo
A1 se georreferenció contra los límites municipales del DANE, usando las
cabeceras municipales (DIVIPOLA) como puntos de control y una corrección local
thin-plate. El departamento de cada punto sale de los límites DANE, así que cada
punto toma la zona del mapa de su propio departamento.

PRECISION
---------
Validación cruzada (dejando cada cabecera fuera del ajuste): 98,9 % de las
cabeceras quedan en la zona que muestra el mapa; error de posición mediano
~0,7 km. Los fallos están a 1-4 km de un límite de zona, por eso el resultado
informa las otras zonas que hay a menos de `radio_km` (zonas_cercanas): si la
lista no está vacía, conviene confirmar en el mapa.

La versión anterior de este script (mapa regional de 606x438 px, 9 puntos de
control con triangulación) acertaba 77 % de las cabeceras; la de transformación
afín (mapaviento.py), 91 %.

USO
---
    from zona_viento import obtener_zona_viento

    obtener_zona_viento(10.9639, -74.7964)   # Barranquilla
    # {'zona': 'A', 'velocidad_kmh': '130-80', 'departamento': 'Atlántico',
    #  'municipio': 'Barranquilla', 'zonas_cercanas': [], ...}

    python zona_viento.py 10.9639 -74.7964
"""

import json
import math
import os
import sys

RUTA_DATOS_DEFECTO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zona_viento_caribe.json")

ZONAS = {
    "A": {"velocidad_kmh": "130-80", "descripcion": "Zona A (130/80 km/h rural/urbana)"},
    "B": {"velocidad_kmh": "100-60", "descripcion": "Zona B (100/60 km/h rural/urbana)"},
    "C": {"velocidad_kmh": "80-50", "descripcion": "Zona C (80/50 km/h rural/urbana)"},
}
_LETRAS = "ABC"

_CACHE = {}


def _cargar(ruta):
    if ruta not in _CACHE:
        with open(ruta, encoding="utf8") as f:
            d = json.load(f)
        d["_zona"] = {}       # filas decodificadas bajo demanda
        d["_mun"] = {}
        _CACHE[ruta] = d
    return _CACHE[ruta]


def _fila_zona(d, i):
    if i not in d["_zona"]:
        fila, k, txt = bytearray(d["columnas"]), 0, d["filas"][i]
        j = 0
        while j < len(txt):
            codigo = ord(txt[j]) - 65
            j += 1
            ini = j
            while j < len(txt) and not ("A" <= txt[j] <= "Z"):
                j += 1
            n = int(txt[ini:j], 36)
            fila[k:k + n] = bytes([codigo]) * n
            k += n
        d["_zona"][i] = fila
    return d["_zona"][i]


def _fila_municipio(d, i):
    if i not in d["_mun"]:
        fila = []
        for tramo in d["filas_municipio"][i].split(";"):
            v, n = tramo.split(":")
            fila.extend([int(v)] * int(n, 36))
        d["_mun"][i] = fila
    return d["_mun"][i]


def _codigo(d, i, j):
    if i < 0 or i >= len(d["filas"]) or j < 0 or j >= d["columnas"]:
        return 0
    return _fila_zona(d, i)[j]


def obtener_zona_viento(lat, lon, ruta_datos=RUTA_DATOS_DEFECTO, radio_km=1.5):
    """
    Zona de viento en (lat, lon).

    Retorna dict:
        zona                : 'A', 'B', 'C' o None fuera de los 7 departamentos
        velocidad_kmh       : rango de velocidad de la zona (o None)
        descripcion         : descripción de la zona (o None)
        departamento        : departamento DANE del punto
        municipio           : municipio DANE del punto
        zonas_cercanas      : otras zonas a menos de radio_km (vacía = lejos de un límite)
        distancia_limite_km : distancia a la celda más cercana de otra zona (o None)
        confianza           : 1.0 lejos de un límite; distancia/radio si hay otra zona cerca
        mensaje             : explicación cuando zona es None
    """
    d = _cargar(ruta_datos)
    i = int(math.floor((d["lat_norte"] - lat) / d["paso"]))
    j = int(math.floor((lon - d["lon_oeste"]) / d["paso"]))
    base = {"zona": None, "velocidad_kmh": None, "descripcion": None, "departamento": None,
            "municipio": None, "zonas_cercanas": [], "distancia_limite_km": None,
            "confianza": 0.0, "mensaje": None}
    codigo = _codigo(d, i, j)
    if not codigo:
        base["mensaje"] = ("El punto cae fuera de los departamentos del mapa de vientos de la "
                           "Costa Caribe (Atlántico, Bolívar, Magdalena, La Guajira, Córdoba, "
                           "Sucre, Cesar) o en el mar.")
        return base
    zona = _LETRAS[(codigo - 1) % 3]

    km_lat = 111.32
    km_lon = 111.32 * math.cos(math.radians(lat))
    ri = math.ceil(radio_km / (d["paso"] * km_lat))
    rj = math.ceil(radio_km / (d["paso"] * km_lon))
    vecinas, dist_min = {}, None
    for di in range(-ri, ri + 1):
        for dj in range(-rj, rj + 1):
            c = _codigo(d, i + di, j + dj)
            if not c:
                continue
            z = _LETRAS[(c - 1) % 3]
            if z == zona:
                continue
            dist = math.hypot(di * d["paso"] * km_lat, dj * d["paso"] * km_lon)
            if dist > radio_km:
                continue
            vecinas[z] = min(vecinas.get(z, dist), dist)
            dist_min = dist if dist_min is None else min(dist_min, dist)

    base.update(ZONAS[zona])
    base.update({
        "zona": zona,
        "departamento": d["departamentos"][(codigo - 1) // 3],
        "municipio": d["municipios"][_fila_municipio(d, i)[j]] or None,
        "zonas_cercanas": sorted(vecinas),
        "distancia_limite_km": round(dist_min, 2) if dist_min is not None else None,
        "confianza": 1.0 if dist_min is None else round(dist_min / radio_km, 2),
    })
    return base


if __name__ == "__main__":
    if len(sys.argv) == 3:
        print(obtener_zona_viento(float(sys.argv[1]), float(sys.argv[2])))
    else:
        print("Uso: python zona_viento.py <lat> <lon>\n")
        for nombre, lat, lon in [("Barranquilla", 10.9639, -74.7964), ("Valledupar", 10.4631, -73.2532),
                                 ("Montería", 8.7479, -75.8814), ("Mompox", 9.2442, -74.4282)]:
            r = obtener_zona_viento(lat, lon)
            print(f"  {nombre:12s} -> zona {r['zona']} ({r['velocidad_kmh']} km/h) "
                  f"{r['municipio']}, {r['departamento']}  cercanas={r['zonas_cercanas']}")
