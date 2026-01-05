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

def vano_regulacion(vanos_m, desniveles_m, usar_k_truxa=True):
    """
    Calcula el vano ideal de regulación de un cantón según el método de Truxá.

    Parámetros
    ----------
    vanos_m : array-like
        Longitud horizontal de cada vano (a_i), en metros.
    desniveles_m : array-like o None
        Desnivel de cada vano (b_i), en metros.
        Si es None, se supone bi = 0 para todos los vanos (cantón nivelado).
    usar_k_truxa : bool, opcional (default = True)
        Si True, aplica el factor de Truxá k.
        Si False, asume k = 1 (equivalente a ignorar el desnivel en el vano ideal).

    Returns
    -------
    ar : float
        Longitud del vano ideal de regulación (m).
    k : float
        Factor de Truxá utilizado (k = 1 para vanos nivelados).
    """

    # Convertir a arrays
    a = np.asarray(vanos_m, dtype=float)

    if a.ndim != 1:
        raise ValueError("vanos_m debe ser un vector 1D (lista o array de vanos).")

    if desniveles_m is None:
        b = np.zeros_like(a)
    else:
        b = np.asarray(desniveles_m, dtype=float)
        if b.shape != a.shape:
            raise ValueError("vanos_m y desniveles_m deben tener la misma longitud.")

    # Longitud real de cada vano (á_i)
    a_real = np.sqrt(a**2 + b**2)

    # Vano equivalente base (caso nivelado)
    suma_a3 = np.sum(a**3)
    suma_a = np.sum(a)

    if suma_a <= 0:
        raise ValueError("La suma de los vanos debe ser mayor que cero.")

    ar_base = np.sqrt(suma_a3 / suma_a)

    if usar_k_truxa:
        # Factor de Truxá k (forma reconstruida a partir de formularios típicos)
        # Propiedades:
        #  - adimensional
        #  - si a_real == a  => k = 1
        num = np.sum(a_real**3) * np.sum(a**2)
        den = np.sum(a**3) * np.sum(a * a_real)

        if den <= 0:
            raise ValueError("Datos de vanos/desniveles inválidos (denominador de k <= 0).")

        k = np.sqrt(num / den)
    else:
        k = 1.0

    # Vano ideal de regulación
    ar = k * ar_base
    return ar

""" print(vano_regulacion([21,22,24,45,23],[1,0,2,3,0.5])) """

def identificar_poste(codigo: str, detallado: bool = False):
    """
    Identifica el tipo de poste según el código de armado de AFINIA.

    Si detallado=False → retorna solo las siglas del tipo de poste: FL, AL, ANG, ANC.
    Si detallado=True  → retorna un diccionario con información completa.
    """

    # --- Validación básica ---
    if "-" not in codigo:
        raise ValueError("El código debe tener el formato 'CCC###-#'.")

    parte_armado, parte_tension = codigo.split("-")

    # Letras iniciales (2 o 3)
    letras = ''.join([c for c in parte_armado if c.isalpha()])
    numeros = ''.join([c for c in parte_armado if c.isdigit()])

    if len(numeros) != 3:
        raise ValueError("El código debe contener tres dígitos consecutivos para el armado.")

    # --- Interpretación de letras ---
    nivel_tension = letras[:2]
    if nivel_tension == "BT":
        nivel = "Baja Tensión"
    elif nivel_tension == "MT":
        nivel = "Media Tensión"
    else:
        nivel = "Desconocido"

    # Tipo de cable
    tipo_cable = "Forrado" if (len(letras) == 3 and letras[2] == "F") else "Desnudo"

    # --- Interpretación de dígitos ---
    d1 = int(numeros[0])
    d2 = int(numeros[1])
    d3 = int(numeros[2])

    # Armado general
    if d1 == 6:
        armado_general = "Autosoportado (1 circuito)"
    elif d1 == 7:
        armado_general = "Autosoportado (2 circuitos)"
    else:
        armado_general = f"Armado general tipo {d1}"

    # Fases
    fases = "Trifásico" if d2 == 3 else ("Bifásico" if d2 == 2 else f"{d2} fases")

    # Tipo de poste → SIGLAS
    if d3 == 1:
        sigla_poste = "FL"
        tipo_poste = "Fin de Línea"
    elif d3 == 2:
        sigla_poste = "AL"
        tipo_poste = "Alineación"
    elif d3 == 3:
        sigla_poste = "ANG"
        tipo_poste = "Ángulo"
    elif d3 in (4, 5):
        sigla_poste = "ANC"
        tipo_poste = "Anclaje"
    else:
        sigla_poste = f"({d3})"
        tipo_poste = "Desconocido"

    # Tensión del circuito
    if parte_tension == "1":
        tension = "13.2 kV"
    elif parte_tension == "2":
        tension = "34.5 kV"
    else:
        tension = f"Tensión desconocida ({parte_tension})"

    # --- Salida ---
    if not detallado:
        return sigla_poste  # <-- 🔥 SOLO SIGLAS (FL, AL, ANG, ANC)

    # Salida completa
    return {
        "Código": codigo,
        "Sigla": sigla_poste,
        "Tipo de Poste": tipo_poste,
        "Nivel de Tensión": nivel,
        "Tipo de Cable": tipo_cable,
        "Armado General": armado_general,
        "Fases": fases,
        "Tensión del Circuito": tension
    }

""" print(identificar_poste("MTF331-2",True)) """


def calcular_cantones(armados, rutas, postes, vanos_adelante, detallado=False):
    """
    Calcula la longitud de los cantones de una línea de MT a partir de:
    - armados: lista de códigos de armado
    - rutas: lista que indica la ruta/derivación a la que pertenece cada poste
    - postes: nombres/identificadores de cada poste
    - vanos_adelante: distancia al siguiente poste de la misma ruta
    """

    # Agrupar índices de postes por ruta
    rutas_dict = {}
    for i, ruta in enumerate(rutas):
        rutas_dict.setdefault(ruta, []).append(i)

    # Asegurar que cada ruta queda en el orden en que aparecen
    for r in rutas_dict:
        rutas_dict[r].sort()

    cantones = []
    num_canton = 0

    for nombre_ruta, indices in rutas_dict.items():
        if len(indices) < 2:
            continue  # una ruta con 1 poste no genera canton

        # inicio del primer cantón
        inicio = indices[0]
        longitud = 0.0

        for j in range(len(indices) - 1):
            actual = indices[j]
            siguiente = indices[j + 1]

            # sumar el vano desde actual → siguiente
            longitud += vanos_adelante[actual]

            # identificar tipo del siguiente poste
            tipo_sig = identificar_poste(armados[siguiente])

            # condiciones para cerrar el cantón
            fin_por_tipo = tipo_sig in ("FL", "ANC")
            fin_por_ruta = (j + 1 == len(indices) - 1)

            if fin_por_tipo or fin_por_ruta:
                num_canton += 1

                if not detallado:
                    cantones.append(longitud)
                else:
                    cantones.append({
                        "canton": num_canton,
                        "ruta": nombre_ruta,
                        "poste_inicio": postes[inicio],
                        "poste_fin": postes[siguiente],
                        "longitud": longitud
                    })

                # reiniciar para el siguiente cantón
                inicio = siguiente
                longitud = 0.0

    return cantones

""" armados = ["MTF331-2", "MTF332-1","MTF334-1", "MTF332-1","MTF334-1", "MTF331-1"]
rutas   = ["ruta1",     "ruta1",   "ruta1",  "ruta1",     "ruta2",     "ruta2"]
postes  = ["EPP01",     "EPP02",  "EPP03",      "EPP04",     "EPP05", "EPP06"]
vanos_adelante = [25, 30,45, 0,24,0]

print(calcular_cantones(armados, rutas, postes, vanos_adelante, detallado=True)) """

import re

def extraer_datos_poste(cadena):
    """
    Extrae la altura del poste y la carga de rotura (convertida a daN)
    desde un string con formato: "PH ##/#### kg-f".
    
    Parámetros
    ----------
    cadena : str
        Texto con el formato del poste: "PH ##/#### kg-f"
    kgf_to_daN : function
        Función previamente creada que convierte kgf a daN.
    
    Retorna
    -------
    (altura, carga_daN)
        altura : int     → en metros
        carga_daN : float → capacidad en daN
    """

    # Buscar el patrón "PH XX/YYYY"
    patron = r"PH\s*(\d{2})/(\d{3,4})"
    match = re.search(patron, cadena.upper())

    if not match:
        raise ValueError(f"Formato no válido: {cadena}")

    altura = int(match.group(1))
    carga_kgf = int(match.group(2))

    # Convertir a daN usando la función proporcionada
    carga_daN = kgf_a_daN(carga_kgf)

    return altura, carga_daN

""" print(extraer_datos_poste("PH 12/1050 kg-f")) """

def construir_c2t1(tabla1, tabla2, c1t1, c2t1, c1t2, c2t2):
    """
    Construye o actualiza una columna existente en tabla1 a partir de la relación
    entre tabla1 y tabla2.

    Para cada valor de la columna c1t1 en tabla1, se buscan las filas
    correspondientes en tabla2 donde c1t2 coincide. A partir de la columna c2t2
    se determina un valor único válido, ignorando NaN y ceros.

    Reglas:
    - Si todos los valores válidos de c2t2 son iguales, se asigna ese valor.
    - Si hay valores NaN, - o 0 mezclados con un único valor válido, se asigna
      dicho valor.
    - Si existen dos o más valores válidos distintos, se lanza un error.

    La función modifica directamente tabla1 sobrescribiendo la columna c2t1
    y retorna el DataFrame resultante.
    """

    resultados = []

    for valor in tabla1[c1t1]:
        valores_c2t2 = tabla2.loc[tabla2[c1t2] == valor, c2t2]

        valores_validos = (
            valores_c2t2
            .replace(0, pd.NA)
            .replace("-", pd.NA)
            .dropna()
            .unique()
        )

        if len(valores_validos) == 0:
            resultados.append(pd.NA)

        elif len(valores_validos) == 1:
            resultados.append(valores_validos[0])

        else:
            raise ValueError(
                f"Conflicto para '{valor}': "
                f"valores distintos en '{c2t2}': {list(valores_validos)}"
            )

    tabla1[c2t1] = resultados
    return tabla1

