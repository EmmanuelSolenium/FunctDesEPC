import numpy as np
import pandas as pd
import re 
import math 
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
    Extrae la altura del poste y la carga de rotura
    desde un string con formato: "PH ##/#### kg-f".
    
    Parámetros
    ----------
    cadena : str
        Texto con el formato del poste: "PH ##/#### kg-f"

    
    Retorna
    -------
    (altura, carga, altura libre, altura del esfuerzo)
        altura : int     → en metros
        carga_daN : float → capacidad en daN
    """

    # Buscar el patrón "PH XX/YYYY"
    patron = r"PH\s*(\d{2})/(\d{3,4})"
    match = re.search(patron, cadena.upper())

    if not match:
        raise ValueError(f"Formato no válido: {cadena}")

    altura = int(match.group(1))
    altura_libre = altura-2
    altura_esfuerzo = altura_libre-0.2
    carga = int(match.group(2))

    

    return altura, carga, altura_libre, altura_esfuerzo

""" print(extraer_datos_poste("PH 12/1050 kg-f")) """


def construir_c2t1(
    tabla1: pd.DataFrame,
    c1t1: pd.Series,
    c2t1: str,
    c1t2: pd.Series,
    c2t2: pd.Series
):
    """
    Construye o actualiza la columna `c2t1` en tabla1 a partir de Series externas.

    Reglas:
    - Para cada valor en c1t1 se buscan coincidencias en c1t2.
    - Se toman los valores correspondientes de c2t2.
    - Se ignoran NaN, "-", cadenas vacías y 0.
    - Si queda un único valor válido, se asigna.
    - Si no hay valores válidos, se asigna NaN.
    - Si hay más de un valor válido distinto, se lanza error.

    La función modifica tabla1 in-place y retorna el DataFrame.
    """

    if len(c1t2) != len(c2t2):
        raise ValueError("c1t2 y c2t2 deben tener la misma longitud")

    ref = pd.DataFrame({
        "key": c1t2,
        "value": c2t2
    })

    resultados = []

    for valor in c1t1:
        valores_validos = (
            ref.loc[ref["key"] == valor, "value"]
            .replace([0, "-", ""], pd.NA)
            .dropna()
            .unique()
        )

        if len(valores_validos) == 0:
            resultados.append(pd.NA)
        elif len(valores_validos) == 1:
            resultados.append(valores_validos[0])
        else:
            raise ValueError(
                f"Conflicto para '{valor}': valores distintos {list(valores_validos)}"
            )

    tabla1[c2t1] = resultados
    return tabla1

def construir_c2t1_vano(
    tabla1: pd.DataFrame,
    c1t1: pd.Series,
    c2t1: str,
    c1t2: pd.Series,
    c2t2: pd.Series
):
    """
    Construye o actualiza la columna `c2t1` en tabla1 tomando el valor máximo válido.

    Reglas:
    - Se ignoran NaN, '-', cadenas vacías y 0.
    - Si no hay valores válidos → NaN.
    - Si hay uno o más valores válidos → se asigna el VALOR MÁXIMO.

    La función modifica tabla1 in-place y retorna el DataFrame.
    """

    if len(c1t2) != len(c2t2):
        raise ValueError("c1t2 y c2t2 deben tener la misma longitud")

    ref = pd.DataFrame({
        "key": c1t2,
        "value": c2t2
    })

    resultados = []

    for valor in c1t1:
        valores_validos = (
            ref.loc[ref["key"] == valor, "value"]
            .replace([0, "-", ""], pd.NA)
            .dropna()
        )

        resultados.append(
            pd.NA if valores_validos.empty else valores_validos.max()
        )

    tabla1[c2t1] = resultados
    return tabla1


def convertir_texto_kgf_a_daN(texto: str) -> str:
    """
    Convierte expresiones del tipo 'PH ##/#### kg-f' a 'PH ##/XXX daN'.

    - Extrae el valor numérico después del slash (/)
    - Convierte de kgf a daN
    - Redondea hacia arriba a la unidad más cercana
    - Reemplaza 'kg-f' por 'daN'
    """

    patron = r"(.*?/)(\d+)(\s*kg-f)"

    match = re.search(patron, texto)
    if not match:
        raise ValueError(f"Formato no reconocido: {texto}")

    prefijo = match.group(1)        # 'PH 12/'
    valor_kgf = float(match.group(2))
    
    valor_daN = kgf_a_daN(valor_kgf)
    valor_daN_red = round(valor_daN)

    return f"{prefijo}{valor_daN_red} daN"

""" texto = "PH 12/1350 kg-f"
resultado = convertir_texto_kgf_a_daN(texto)

print(resultado) """

def limpiar_saltos_linea_columnas(df):
    """
    Reemplaza saltos de línea '\\n' por espacios simples en los nombres
    de columnas, incluyendo MultiIndex.
    """

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = pd.MultiIndex.from_tuples(
            tuple(
                str(level).replace("\n", " ").strip()
                for level in col
            )
            for col in df.columns
        )
    else:
        df.columns = (
            df.columns
            .astype(str)
            .str.replace("\n", " ", regex=False)
            .str.strip()
        )

    return df




def extraer_series_por_indice(
    df: pd.DataFrame,
    nombre: str,
    nivel: int = 1
) -> list[pd.Series]:
    """
    Extrae todas las Series de un DataFrame con columnas MultiIndex
    cuyo nombre en un nivel dado coincide con `nombre`, excluyendo
    aquellas columnas que estén completamente vacías o inválidas.

    Se consideran valores inválidos:
    - NaN
    - 0
    - "-"
    - cadenas vacías

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con columnas MultiIndex.
    nombre : str
        Nombre a buscar en el nivel especificado.
    nivel : int, default 1
        Nivel del MultiIndex donde se realizará la búsqueda.

    Retorna
    -------
    list[pd.Series]
        Lista de Series válidas encontradas.
    """

    if not isinstance(df.columns, pd.MultiIndex):
        raise TypeError("El DataFrame no tiene columnas MultiIndex")

    series_validas = []

    for col in df.columns:
        if col[nivel] != nombre:
            continue

        serie = df[col]

        # Normalización de valores inválidos
        serie_limpia = (
            serie
            .replace([0, "-", ""], pd.NA)
            .dropna()
        )

        # Si después de limpiar no queda nada, se ignora la columna
        if serie_limpia.empty:
            continue

        series_validas.append(serie)

    return series_validas

def sumar_lista_series(lista):
    """
    Suma fila a fila una lista de pd.Series.
    Retorna una pd.Series o None si la lista está vacía.
    """
    if not lista:
        return None
    df = pd.concat(lista, axis=1)
    return df.replace("-", np.nan).astype(float).sum(axis=1)


import numpy as np

def calcular_ftvc(
    tabla,
    o_postes,
    l_postes,
    angulo_b,          # ángulo de DEFLEXIÓN (δ)
    f_viento_at,
    f_viento_ad,
    tiro_at,
    tiro_ad,
    nombre_columna="FTVC"
):
    """
    Calcula el esfuerzo transversal por viento sobre conductores en postes.

    Notas geométricas:
    - El ángulo de entrada angulo_b es el ángulo de deflexión δ.
    - En el CASO 1 se trabaja sobre la bisectriz → se usa δ directamente.
    - En el CASO 2 (derivaciones) se usa el ángulo entre vanos:
          θ = π − δ
      porque el eje de referencia es el del vano principal.
    """

    # --- Preparar fuerzas equivalentes ---
    fv_at = sumar_lista_series(f_viento_at)
    fv_ad = sumar_lista_series(f_viento_ad)
    ta = sumar_lista_series(tiro_at)
    td = sumar_lista_series(tiro_ad)

    # Inicializar columna resultado
    tabla[nombre_columna] = np.nan

    # Iterar por poste ordenado (resultado final)
    for poste in o_postes:

        mask = l_postes == poste
        n_rep = mask.sum()

        if n_rep == 0:
            continue

        # δ: ángulo de deflexión (rad)
        delta = np.deg2rad(angulo_b[mask].astype(float))

        fv_at_p = fv_at[mask] if fv_at is not None else 0
        fv_ad_p = fv_ad[mask] if fv_ad is not None else 0
        ta_p = ta[mask] if ta is not None else 0
        td_p = td[mask] if td is not None else 0

        # =========================
        # CASO 1: SIN DERIVACIONES
        # =========================
        if n_rep == 1:

            # Bisectriz del ángulo de deflexión
            sen_delta_2 = np.sin(delta.iloc[0] / 2)

            ftvc = (
                fv_at_p.iloc[0]
                + fv_ad_p.iloc[0]
                + (ta_p.iloc[0] + td_p.iloc[0]) * sen_delta_2
            )

            tabla.loc[tabla[o_postes.name] == poste, nombre_columna] = ftvc
            continue

        # =========================
        # CASO 2: CON DERIVACIONES
        # =========================

        # θ = ángulo entre vanos
        theta = np.pi - delta

        # Normalización de viento (exportado respecto a bisectriz)
        cos_delta_2 = np.cos(delta / 2)
        fv_at_c = fv_at_p / cos_delta_2
        fv_ad_c = fv_ad_p / cos_delta_2

        # Detectar poste de referencia p1
        both = ((ta_p > 0) & (td_p > 0)) | ((fv_at_p > 0) & (fv_ad_p > 0))

        if both.sum() > 1:
            raise ValueError(
                f"Error en poste {poste}: más de una derivación tiene esfuerzos adelante y atrás."
            )

        if both.sum() == 1:
            idx_p1 = both.idxmax()
        else:
            idx_p1 = mask[mask].index[0]

        # θ de referencia
        theta_ref = theta.loc[idx_p1]

        # --- Componentes transversales ---

        # Tiros (referidos al eje del vano p1)
        tiros_transv = ta_p * np.sin(theta)
        tiros_transv.loc[idx_p1] = 0  # tiro atrás de p1 no aporta

        # Viento
        # - viento atrás de p1: 90° → aporte completo
        # - viento adelante: proyección cos(θ)
        viento_transv = (
            fv_at_c
            + fv_ad_c * np.cos(theta)
        )

        # Suma total
        ftvc = tiros_transv.sum() + viento_transv.sum()

        tabla.loc[tabla[o_postes.name] == poste, nombre_columna] = ftvc

    return tabla



def deflexion_a_angulo(delta, grados=True):
    """
    Convierte un ángulo de deflexión en el ángulo real entre dos vanos.

    Parámetros
    ----------
    delta : float, array-like o pandas Series
        Ángulo de deflexión (entre la prolongación de un vano y el siguiente).
    grados : bool, default=True
        True si delta está en grados.
        False si delta está en radianes.

    Retorna
    -------
    float, array-like o pandas Series
        Ángulo entre los dos vanos.
    """
    if grados:
        return 180.0 - delta
    else:
        return np.pi - delta
    


def calcular_flmc(
    tabla,
    o_postes,
    l_postes,
    angulo_b,          # ángulo de DEFLEXIÓN (δ)
    f_viento_at,
    f_viento_ad,
    tiro_at,
    tiro_ad,
    nombre_columna="FLMC"
):
    """
    Calcula el esfuerzo longitudinal mecánico combinado (FLMC)
    sobre conductores en postes.

    Es el complemento longitudinal de FTVC:
    - sen <-> cos
    - fuerzas anuladas pasan a ser completas y viceversa
    """

    # --- Preparar fuerzas equivalentes ---
    fv_at = sumar_lista_series(f_viento_at)
    fv_ad = sumar_lista_series(f_viento_ad)
    ta = sumar_lista_series(tiro_at)
    td = sumar_lista_series(tiro_ad)

    # Inicializar columna resultado
    tabla[nombre_columna] = np.nan

    # Iterar por poste ordenado
    for poste in o_postes:

        mask = l_postes == poste
        n_rep = mask.sum()

        if n_rep == 0:
            continue

        # δ: ángulo de deflexión
        delta = np.deg2rad(angulo_b[mask].astype(float))

        fv_at_p = fv_at[mask] if fv_at is not None else 0
        fv_ad_p = fv_ad[mask] if fv_ad is not None else 0
        ta_p = ta[mask] if ta is not None else 0
        td_p = td[mask] if td is not None else 0

        # =========================
        # CASO 1: SIN DERIVACIONES
        # =========================
        if n_rep == 1:

            cos_delta_2 = np.cos(delta.iloc[0] / 2)

            flmc = (
                ta_p.iloc[0] + (td_p.iloc[0]) 
            )

            tabla.loc[tabla[o_postes.name] == poste, nombre_columna] = flmc
            continue

        # =========================
        # CASO 2: CON DERIVACIONES
        # =========================

        # θ = ángulo entre vanos
        theta = np.pi - delta

        # Normalización de viento (exportado respecto a bisectriz)
        cos_delta_2 = np.cos(delta / 2)
        fv_at_c = fv_at_p / cos_delta_2
        fv_ad_c = fv_ad_p / cos_delta_2

        # Detectar p1
        both = ((ta_p > 0) & (td_p > 0)) | ((fv_at_p > 0) & (fv_ad_p > 0))

        if both.sum() > 1:
            raise ValueError(
                f"Error en poste {poste}: más de una derivación tiene esfuerzos adelante y atrás."
            )

        if both.sum() == 1:
            idx_p1 = both.idxmax()
        else:
            idx_p1 = mask[mask].index[0]

        # --- Componentes longitudinales ---

        # Tiros
        tiros_long = ta_p * np.cos(theta)
        tiros_long.loc[idx_p1] = ta_p.loc[idx_p1]  # tiro atrás p1 completo

        # Viento
        # - viento atrás p1: no aporta
        # - viento adelante: sin(θ)
        viento_long = fv_ad_c * np.sin(theta)
        viento_long.loc[idx_p1] = 0

        # Suma total
        flmc = tiros_long.sum() + viento_long.sum()

        tabla.loc[tabla[o_postes.name] == poste, nombre_columna] = flmc

    return tabla

########################################################################


tabla = pd.DataFrame({
    "Numero de apoyo": ["P01", "P02", "P03", "P04"]
})
o_postes = tabla["Numero de apoyo"]
l_postes = pd.Series([
    "P01",
    "P02",
    "P02",  # derivación
    "P03",
    "P04"
])
angulo_b = pd.Series([
    0,   # P01
    20,   # P02 (vano principal)
    30,   # P02 (derivación)
    0,   # P03
    0     # P04 (alineado)
])
f_viento_at = [
    pd.Series([
        20,  # P01
        30,  # P02 principal
        10,  # P02 derivación
        20,  # P03
        0    # P04
    ])
]
f_viento_ad = [
    pd.Series([
        20,  # P01
        30,  # P02 principal
        0,  # P02 derivación
        30,  # P03
        0    # P04
    ])
]
tiro_at = [
    pd.Series([
        40,  # P01
        50,  # P02 principal
        10,   # P02 derivación
        30,  # P03
        20   # P04
    ])
]
tiro_ad = [
    pd.Series([
        40,  # P01
        50,  # P02 principal
        0,   # P02 derivación
        30,  # P03
        20   # P04
    ])
]

tabla = calcular_flmc(tabla,o_postes,l_postes,angulo_b,f_viento_at,f_viento_ad,tiro_at,tiro_ad)
tabla = calcular_ftvc(tabla,o_postes,l_postes,angulo_b,f_viento_at,f_viento_ad,tiro_at,tiro_ad)
print(tabla)