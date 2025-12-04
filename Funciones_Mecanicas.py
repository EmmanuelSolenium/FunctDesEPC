import numpy as np
import pandas as pd

def kgf_a_daN(F_kgf, g=9.8066500):
    """
    Convierte kilogramo-fuerza (kgf) a decanewton (daN).

    Conversión:
        1 kgf = g N
        1 daN = 10 N
        daN = (kgf * g) / 10
    """
    return F_kgf * g / 10.0

def N_a_daN(F_N, g=9.8066500):
    """
    Convierte N a daN.
    La constante g se mantiene solo para estandarización,
    aunque no interviene en la conversión directa.

    Fórmula:
        1 daN = 10 N
        daN = N / 10
    """
    return F_N / 10.0

def suma_vectores(magnitudes, angulos_relativos_deg):
    """
    Calcula la magnitud de la suma de N vectores dados por:
    - magnitudes[i]: magnitud del vector i
    - angulos_relativos_deg[i]: ángulo del vector i respecto al primer vector (en grados)

    El primer vector se asume con ángulo absoluto 0°.
    El resto se posiciona sumándole el ángulo relativo.

    Parámetros:
        magnitudes (list or array): magnitudes de los vectores [m1, m2, ..., mn]
        angulos_relativos_deg (list or array): ángulos relativos respecto al primer vector (en grados)
                                               Debe tener longitud n, donde el primer ángulo debe ser 0.

    Retorna:
        tuple: (magnitud_total, suma_x, suma_y)
    """

    n = len(magnitudes)

    if len(angulos_relativos_deg) != n:
        raise ValueError("La lista de ángulos debe tener la misma longitud que la de magnitudes.")

    # Convertir grados a radianes
    ang_rad = np.radians(angulos_relativos_deg)

    # Componentes X y Y
    comp_x = np.sum(magnitudes * np.cos(ang_rad))
    comp_y = np.sum(magnitudes * np.sin(ang_rad))

    # Magnitud total del vector resultante
    magnitud_total = np.sqrt(comp_x**2 + comp_y**2)

    return magnitud_total

""" print(suma_vectores([1,1,1],[0,90,-90])) """