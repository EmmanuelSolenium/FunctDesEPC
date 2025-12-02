import pandas as pd
import numpy as np

def Reg_V(P_kw, V, fases, R_ohm_km, X_ohm_km, longitud_m, fp=1.0):
    """
    Calcula la regulación de tensión según criterios compatibles con NTC 2050 y RETIE.
    Pensado para instalaciones FV donde FP = 1 es el caso típico.

    P_kw        : Potencia activa en kW
    V           : Voltaje nominal en V
    fases       : Número de fases (1 o 3)
    R_ohm_km    : Resistencia del conductor (ohm/km)
    X_ohm_km    : Reactancia del conductor (ohm/km)
    longitud_m  : Longitud del circuito en metros
    fp          : Factor de potencia (por defecto = 1.0)
    """

    # Convertir unidades
    P = P_kw * 1000                # kW → W
    L = longitud_m / 1000          # m  → km
    
    # Ángulo asociado al FP
    phi = np.arccos(fp)

    # Corriente
    if fases == 1:
        I = P / (V * fp)
        dV = I * (R_ohm_km * L * fp + X_ohm_km * L * np.sin(phi))

    elif fases == 3:
        I = P / (np.sqrt(3) * V * fp)
        dV = np.sqrt(3) * I * (R_ohm_km * L * fp + X_ohm_km * L * np.sin(phi))

    else:
        raise ValueError("El número de fases debe ser 1 o 3.")

    # Regulación %
    VR = (dV / V) * 100

    return (I,dV,VR)
A = Reg_V(1000,800,3,0.0021,0.003,150)

print(A[1]+A[2])


def llenado_tuberia_por_grupos(diametro_interno_mm, diametros_mm, cantidades):
    """
    Calcula el llenado de una tubería según NTC 2050 y RETIE usando:
    - lista de diámetros externos de conductores
    - cantidad de conductores por cada diámetro

    Salida (lista):
        [
            area_tuberia_mm2,
            areas_por_grupo_mm2,
            area_total_conductores_mm2,
            porcentaje_llenado,
            total_conductores,
            limite_permitido,
            cumple_norma
        ]
    """

    # Validación de parámetros
    if len(diametros_mm) != len(cantidades):
        raise ValueError("Las listas de diámetros y cantidades deben tener la misma longitud.")

    # Área interna de la tubería
    area_tuberia = np.pi * (diametro_interno_mm / 2) ** 2

    # Cálculo de áreas individuales por grupo
    areas_conductores = [
        (np.pi * (d / 2) ** 2) * n
        for d, n in zip(diametros_mm, cantidades)
    ]

    # Área total
    area_total_conductores = np.sum(areas_conductores)

    # Total de conductores
    total_conductores = np.sum(cantidades)

    # Porcentaje de llenado
    porcentaje_llenado = (area_total_conductores / area_tuberia) * 100

    # Límite según NTC 2050 / RETIE
    if total_conductores == 1:
        limite = 53
    elif total_conductores == 2:
        limite = 31
    else:
        limite = 40

    # Cumplimiento normativo
    cumple = porcentaje_llenado <= limite

    # Salida como lista
    return [
        area_tuberia,
        areas_conductores,
        area_total_conductores,
        porcentaje_llenado,
        total_conductores,
        limite,
        cumple
    ]
