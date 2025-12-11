import pandas as pd
import numpy as np
import math


def Reg_V(P_kw, V, fases, R_ohm_km, X_ohm_km, longitud_m, fp=1.0, detallado=False):
    """
    Calcula la regulación de tensión en AC.
    Por defecto retorna solo la regulación (%).
    Si detallado=True retorna todos los valores internos.
    """

    P = P_kw * 1000          # kW → W
    L = longitud_m / 1000    # m → km
    phi = np.arccos(fp)

    if fases == 1:
        I = P / (V * fp)
        dV = I * (R_ohm_km * L * fp + X_ohm_km * L * np.sin(phi))

    elif fases == 3:
        I = P / (np.sqrt(3) * V * fp)
        dV = np.sqrt(3) * I * (R_ohm_km * L * fp + X_ohm_km * L * np.sin(phi))

    else:
        raise ValueError("El número de fases debe ser 1 o 3.")

    VR = (dV / V) * 100

    if not detallado:
        return VR

    return {
        "corriente_A": I,
        "caida_V": dV,
        "regulacion_pct": VR
    }


# ----------------- PRUEBA -----------------
A = Reg_V(300, 800, 3, 0.141, 0.128, 150)
print(A)





def Llenado_Tub(diametro_interno_pulg, diametros_mm, cantidades, detallado=False):
    """
    Calcula porcentaje de llenado de tubería según NTC 2050 / RETIE.
    Por defecto retorna solo el porcentaje.
    Si detallado=True retorna valores internos.
    """

    if len(diametros_mm) != len(cantidades):
        raise ValueError("Las listas de diámetros y cantidades deben tener la misma longitud.")

    # pulgadas → mm
    diametro_interno_mm = diametro_interno_pulg * 25.4

    # área de tubería
    area_tuberia = np.pi * (diametro_interno_mm / 2)**2

    # áreas por grupo
    areas_por_grupo = [
        (np.pi * (d / 2)**2) * n
        for d, n in zip(diametros_mm, cantidades)
    ]

    # total de área en conductores
    area_total = np.sum(areas_por_grupo)
    total_conductores = np.sum(cantidades)

    porcentaje_llenado = (area_total / area_tuberia) * 100

    if total_conductores == 1:
        limite = 53
    elif total_conductores == 2:
        limite = 31
    else:
        limite = 40

    cumple = porcentaje_llenado <= limite

    if not detallado:
        return porcentaje_llenado

    return {
        "area_total_mm2": area_total,
        "areas_por_grupo_mm2": areas_por_grupo,
        "area_tuberia_mm2": area_tuberia,
        "porcentaje_llenado": porcentaje_llenado,
        "limite_pct": limite,
        "cumple_norma": cumple
    }


# ----------------- PRUEBA -----------------
b = Llenado_Tub(4, [22.2504, 9.97], [3, 1])
print(b)



def Reg_Vdc(V_mppt, I_mppt, num_paneles, R_ohm_km, longitud_m, detallado=False):
    """
    Regulación de tensión DC con parámetros MPPT del panel.
    Por defecto retorna solo la regulación (%).
    """

    V_string = V_mppt * num_paneles
    R_total = R_ohm_km * (longitud_m * 2 / 1000)  # ida + vuelta
    delta_V = I_mppt * R_total
    porcentaje_reg = (delta_V / V_string) * 100

    if not detallado:
        return porcentaje_reg

    return {
        "V_string": V_string,
        "R_total_ohm": R_total,
        "delta_V_volt": delta_V,
        "porcentaje_regulacion": porcentaje_reg
    }


# ----------------- PRUEBA -----------------
c = Reg_Vdc(43.22, 13.42, 28, 5.09, 150)
print(c)




def Perdidas_AC(P_kw, V, fases, R_ohm_km, longitud_m, fp=1.0, detallado=False):
    """
    Cálculo de pérdidas resistivas AC.
    Por defecto retorna solo pérdidas totales (W).
    """

    # corriente
    if fases == 1:
        I = 1000 * P_kw / (V * fp)
    elif fases == 3:
        I = 1000 * P_kw / (np.sqrt(3) * V * fp)
    else:
        raise ValueError("El número de fases debe ser 1 o 3.")

    R_fase = R_ohm_km * (longitud_m / 1000)

    if fases == 1:
        perdidas_fase = I**2 * (2 * R_fase)
        perdidas_tot = perdidas_fase
    else:
        perdidas_fase = I**2 * R_fase
        perdidas_tot = 3 * perdidas_fase

    perdidas_pct = (perdidas_tot / (P_kw * 1000)) * 100

    if not detallado:
        return perdidas_tot

    return {
        "corriente_A": I,
        "perdidas_fase_W": perdidas_fase,
        "perdidas_totales_W": perdidas_tot,
        "perdidas_pct": perdidas_pct
    }


# ----------------- PRUEBA -----------------
d = Perdidas_AC(300, 800, 3, 0.141, 150)
print(d)


def Bandeja_Calc(
    ancho_util_mm,
    diametros_mm,
    cantidades,
    pesos_kg_km,
    capacidad_carga_kg_m,
    detallado=False
):
    """
    Cálculo simplificado interno para bandejas:
    - Área ocupada por los conductores (mm²)
    - Número de capas (geométrico)
    - Capacidad de carga requerida (kg/m)
    - Validación de capacidad de carga mecánica

    NOTA: La validación de ocupación (% llenado) SE DEJA A LOS INGENIEROS.
    """

    # Validaciones básicas
    if len(diametros_mm) != len(cantidades) or len(diametros_mm) != len(pesos_kg_km):
        raise ValueError("Listas de diámetros, cantidades y pesos deben tener la misma longitud.")

    # -------------------------------------
    # 1. Área ocupada total (mm²)
    # -------------------------------------
    area_total_mm2 = np.sum([
        n * (np.pi * (d / 2)**2)
        for d, n in zip(diametros_mm, cantidades)
    ])

    # -------------------------------------
    # 2. Número de capas (sin validar ocupación)
    # -------------------------------------
    ancho_ocupado_mm = np.sum([d * n for d, n in zip(diametros_mm, cantidades)])
    num_capas = math.ceil(ancho_ocupado_mm / ancho_util_mm)

    # -------------------------------------
    # 3. Peso total requerido por metro (kg/m)
    # -------------------------------------
    peso_total_kg_m = np.sum([
        n * (p / 1000)      # kg/km → kg/m
        for p, n in zip(pesos_kg_km, cantidades)
    ])

    # -------------------------------------
    # 4. Validación mecánica
    # -------------------------------------
    cumple_carga = peso_total_kg_m <= capacidad_carga_kg_m
    # -------------------------------------
    # 5. Area permisibile por bandeja
    # -------------------------------------
    Area_permisible = ancho_util_mm*28
    validar_area = Area_permisible>area_total_mm2
    # -------------------------------------
    # Salida simple (por defecto)
    # -------------------------------------
    if not detallado:
        return area_total_mm2

    # -------------------------------------
    # Salida detallada (diccionario)
    # -------------------------------------
    return {
        "area_ocupada_mm2": area_total_mm2,
        "num_capas": num_capas,
        "peso_total_kg_m": peso_total_kg_m,
        "capacidad_carga_kg_m": capacidad_carga_kg_m,
        "cumple_carga": cumple_carga,
        "Area Permisible": Area_permisible,
        "cumple_area": validar_area
    }

# ----------------- PRUEBA -----------------

e = Bandeja_Calc(
    ancho_util_mm=300,
    diametros_mm=[26, 9.97],
    cantidades=[9, 3],
    pesos_kg_km=[895, 150],
    capacidad_carga_kg_m=74,
    detallado=True
)
print(e)




