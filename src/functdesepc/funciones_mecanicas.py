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

""" print(identificar_poste("MTF331-2")) """


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
    

import numpy as np
import pandas as pd


def calcular_ftvc_flmc(
    tabla,
    o_postes,
    l_postes,
    angulo_b,          # ángulo de DEFLEXIÓN δ (grados)
    f_viento_at,
    f_viento_ad,
    tiro_at,
    tiro_ad,
    col_ftvc="FTVC",
    col_flmc="FLMC"
):
    """
    Calcula simultáneamente:
    - FTVC: esfuerzo transversal por viento
    - FLMC: esfuerzo longitudinal mecánico combinado

    Geometría:
    - Caso 1 (sin derivaciones): eje = bisectriz del ángulo de deflexión
    - Caso 2 (con derivaciones):
        * Se calcula el vector resultante de tensiones
        * Se calcula la fuerza sobre conductores por el viento tomando  
            el viento como perpendicular al vano atrás del poste que tiene 
            fuerzas adelante  y atrás o del poste de referencia (angulo 0) 
        *se define el eje transversal con la misma orientación que el vector de fuerzas del viento
        *se proyecta el vector de tensiones sobre cada eje y se suma con el vector de viento para obtener ambas fuerzas
    """

    fv_at = sumar_lista_series(f_viento_at)
    fv_ad = sumar_lista_series(f_viento_ad)
    ta = sumar_lista_series(tiro_at)
    td = sumar_lista_series(tiro_ad)

    tabla[col_ftvc] = np.nan
    tabla[col_flmc] = np.nan

    for poste in o_postes:

        mask = l_postes == poste
        n_rep = mask.sum()
        if n_rep == 0:
            continue
        
        delta = np.deg2rad(angulo_b[mask].astype(float))

        fv_at_p = fv_at[mask] if fv_at is not None else pd.Series(0, index=delta.index)
        fv_ad_p = fv_ad[mask] if fv_ad is not None else pd.Series(0, index=delta.index)
        ta_p = ta[mask] if ta is not None else pd.Series(0, index=delta.index)
        td_p = td[mask] if td is not None else pd.Series(0, index=delta.index)

        # ============================================================
        # CASO 1: SIN DERIVACIONES
        # ============================================================
        if n_rep == 1:

            d = delta.iloc[0]
            sen_d2 = np.sin(d / 2)
            cos_d2 = np.cos(d / 2)

            # ---------------- FTVC ----------------
            # El viento YA está en la bisectriz → NO se normaliza 
            # (en redlin cuando se tiene un poste con vanos adelante 
            # y atrás el programa calcula la fuerza del viento proyectado
            # sobre transversal al eje de la bisectriz, sin embargo como aquí 
            # se calculan fuerzas longitudinales y transversales se normaliza  
            # para poder obtener la proyección sobre el eje longitudinal )

            ftvc = (
                fv_at_p.iloc[0]
                + fv_ad_p.iloc[0]
                + (ta_p.iloc[0] + td_p.iloc[0]) * sen_d2  #si el poste está en angulo las tensiones tienen una componente transversal en la fuerza
            )

            # ---------------- FLMC ----------------
            # Normalizar SOLO si hay viento adelante y atrás
            if fv_at_p.iloc[0] > 0 and fv_ad_p.iloc[0] > 0:
                fv_at_c = fv_at_p.iloc[0] / cos_d2  
                fv_ad_c = fv_ad_p.iloc[0] / cos_d2
            else:
                fv_at_c = fv_at_p.iloc[0]
                fv_ad_c = fv_ad_p.iloc[0]

            flmc = (
                (td_p.iloc[0] - ta_p.iloc[0]) * cos_d2
                + (fv_at_c - fv_ad_c) * sen_d2  #si el poste está en angulo las fuerzas del viento tienen una componente longitudinal en la fuerza
            )

            tabla.loc[tabla[o_postes.name] == poste, col_ftvc] = ftvc
            tabla.loc[tabla[o_postes.name] == poste, col_flmc] = flmc
            continue

        # ============================================================
        # CASO 2: CON DERIVACIONES
        # ============================================================

        theta = np.pi - delta  # ángulo real entre vanos

        # Normalización del viento SOLO si hay adelante y atrás
        fv_at_c = fv_at_p.copy()
        fv_ad_c = fv_ad_p.copy()

        for idx in delta.index:
            if fv_at_p.loc[idx] > 0 and fv_ad_p.loc[idx] > 0:
                fv_at_c.loc[idx] /= np.cos(delta.loc[idx] / 2)
                fv_ad_c.loc[idx] /= np.cos(delta.loc[idx] / 2)

        # ------------------------------------------------------------
        # 1) Vector resultante de tensiones
        # ------------------------------------------------------------
        T_res = np.array([0.0, 0.0])

        for idx in delta.index:
            th = theta.loc[idx]
            dt = delta.loc[idx]

            if ta_p.loc[idx] > 0 and td_p.loc[idx] > 0:
                # eje alineado con tiro atrás

                tx  = ta_p.loc[idx]*np.cos(0) + td_p.loc[idx]*np.cos(th)
                ty = ta_p.loc[idx]*np.sin(0) + td_p.loc[idx]*np.sin(th)
                
            else:
                T = ta_p.loc[idx] if ta_p.loc[idx] > 0 else td_p.loc[idx]
                tx = T*np.cos(th) if dt != 0 else T*np.cos(dt) #se agrega un condicional para reconocer el poste de referencia, en caso contrario como se convierte el angulo de deflexión a angulo real entonces quedaría con la direccción contraria
                ty = T*np.sin(th) if dt != 0 else T*np.sin(dt)
                

            T_res += np.array([
                tx,
                ty
            ])



        # ------------------------------------------------------------
        # 2) Proyección de fuerzas
        # ------------------------------------------------------------
        V_vec = np.array([0.0, 0.0])

        for idx in delta.index:

            th = theta.loc[idx]

            # ---------- VIENTO ----------
            if fv_at_c.loc[idx] > 0 and fv_ad_c.loc[idx] > 0:
                
                
                # se define la dirección del viento a + 90° del vano atrás
                v1a = np.array([
                    fv_at_c.loc[idx] * np.cos( np.pi / 2),
                    fv_at_c.loc[idx] * np.sin( np.pi / 2)
                ])

                v1d = np.array([
                    fv_ad_c.loc[idx] * np.cos(th + np.pi / 2),
                    fv_ad_c.loc[idx] * np.sin(th + np.pi / 2)
                ])            
                v2d = np.array([
                    fv_ad_c.loc[idx] * np.cos(th - np.pi / 2),
                    fv_ad_c.loc[idx] * np.sin(th - np.pi / 2)
                ])
                
                Vvecd = v1d if np.dot(v1d, v1a) >= 0 else v2d
                V_vect = v1a + Vvecd
                

            
            else:
                Fv = fv_at_c.loc[idx] if fv_at_c.loc[idx] > 0 else fv_ad_c.loc[idx]
                v1 = np.array([
                    Fv * np.cos(th + np.pi / 2),
                    Fv * np.sin(th + np.pi / 2)
                ])
                v2 = np.array([
                    Fv * np.cos(th - np.pi / 2),
                    Fv * np.sin(th - np.pi / 2)
                ])
                # Dirección del viento tomando el vano atrás como base
                V_vect = v1 if np.dot(v1,[0,1]) >= 0 else v2   
            V_vec += V_vect
                    
        if np.linalg.norm(V_vec) == 0:
            e_L = np.array([1.0, 0.0])
        else:
            e_T = V_vec / np.linalg.norm(V_vec)

        e_L = np.array([-e_T[1], e_T[0]])


        flmc = np.dot(V_vec, e_L) +  np.dot(T_res, e_L)
        ftvc = np.dot(V_vec, e_T) +  np.dot(T_res, e_T)
        tabla.loc[tabla[o_postes.name] == poste, col_flmc] = flmc
        
        tabla.loc[tabla[o_postes.name] == poste, col_ftvc] = ftvc

    return tabla

########### Prueba función ####################

""" tabla = pd.DataFrame({
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
    0,   # P02 (vano principal)
    11,   # P02 (derivación)
    45,   # P03
    0     # P04 (alineado)
])
f_viento_at = [
    pd.Series([
        20,  # P01
        1,  # P02 principal
        1,  # P02 derivación
        20,  # P03
        10    # P04
    ])
]
f_viento_ad = [
    pd.Series([
        20,  # P01
        1,  # P02 principal
        0,  # P02 derivación
        30,  # P03
        0    # P04
    ])
]
tiro_at = [
    pd.Series([
        40,  # P01
        50,  # P02 principal
        50,   # P02 derivación
        30,  # P03
        20   # P04
    ])
]
tiro_ad = [
    pd.Series([
        40,  # P01
        0,  # P02 principal
        0,   # P02 derivación
        30,  # P03
        0   # P04
    ])
] 


tabla = calcular_ftvc_flmc(tabla,o_postes,l_postes,angulo_b,f_viento_at,f_viento_ad,tiro_at,tiro_ad)

tabla["F_check"] = np.sqrt(tabla["FTVC"]**2 + tabla["FLMC"]**2)

print(tabla)  """

def calcular_ftve(
    mec,
    zona_viento,              # "A", "B" o "C"
    area,                     # "Rural" o "Urbana"
    tabla_B2_4,               # DataFrame
    Sxe,                      # área frontal del reconectador (m²)
    postes_reco,              # pd.Series con nombres de postes
    altura_reconectador=None, # pd.Series opcional (m)
    col_poste="Numero de apoyo",
    col_salida="FTVE"
):
    """
    Calcula la Fuerza Transversal por Viento sobre Equipo (FTVE)
    y agrega la columna correspondiente al dataframe mec.

    FTVE = q0 * Cxe * Gt * Sxe
    """

    # ------------------------------------------------------------
    # Constante
    # ------------------------------------------------------------
    Cxe = 2.0

    # ------------------------------------------------------------
    # Inicializar columna
    # ------------------------------------------------------------
    mec[col_salida] = 0.0

    # ------------------------------------------------------------
    # Obtener q0 desde Tabla B2.4
    # ------------------------------------------------------------
    fila_q0 = tabla_B2_4[
        (tabla_B2_4["Area"] == area) &
        (tabla_B2_4["Zona"] == zona_viento)
    ]

    if fila_q0.empty:
        raise ValueError(
            f"No se encontró q0 para Zona={zona_viento}, Area={area}"
        )

    q0 = fila_q0.iloc[0]["q0 (daN / (m ^ 2))"]


    # ------------------------------------------------------------
    # Altura del reconectador
    # ------------------------------------------------------------
    if altura_reconectador is None:
        he = pd.Series(
            5.0,
            index=postes_reco.index
        )
    else:
        he = altura_reconectador.astype(float)

    # ------------------------------------------------------------
    # Calcular Gt según zona y área
    # ------------------------------------------------------------
    def calcular_gt(h):
        if area == "Rural" and zona_viento == "A":
            return -0.0002 * h**2 + 0.0232 * h + 1.4661

        if area == "Rural" and zona_viento in ["B", "C"]:
            return -0.0002 * h**2 + 0.0274 * h + 1.6820

        if area == "Urbana":
            return -0.0002 * h**2 + 0.0384 * h + 2.9284

        raise ValueError("Combinación Zona/Área no válida")

    Gt = he.apply(calcular_gt)

    # ------------------------------------------------------------
    # Calcular FTVE solo en postes con reconectador
    # ------------------------------------------------------------
    for idx, poste in postes_reco.items():

        mask = mec[col_poste] == poste

        if not mask.any():
            continue

        ftve = q0 * Cxe * Gt.loc[idx] * Sxe.loc[idx]
        mec.loc[mask, col_salida] = ftve
        

    return mec


def calcular_flee(
    mec,
    postes_reco,
    LE=None,   # distancia centro reconectador – poste (m)
    HE=None,   # altura del reconectador (m)
    PE=None,   # peso del equipo (daN)
    col_poste="Numero de apoyo",
    col_flee="FLEE"
):
    """
    Calcula la Fuerza Longitudinal Equivalente por Excentricidad del peso
    del equipo (FLEE).

    Fórmula:
        FLEE = HE * LE * PE

    Condiciones:
    - Solo se calcula para postes con reconectador
    - Postes sin reconectador → FLEE = 0
    - Valores por defecto (solo para postes con reconectador):
        * LE = 0.75 m
        * HE = 5.0 m
        * PE = 600 kg
    """

    # Inicializar columna
    mec[col_flee] = 0.0

    # Valores por defecto
    LE_def = 0.75
    HE_def = 5.0
    PE_def = 600.0  # daN (solo postes con reconectador)

    for idx, row in mec.iterrows():

        poste = row[col_poste]

        if poste not in postes_reco.values:
            continue

        # LE
        if LE is not None and poste in LE.index:
            le = LE.loc[poste]
        else:
            le = LE_def

        # HE
        if HE is not None and poste in HE.index:
            he = HE.loc[poste]
        else:
            he = HE_def

        # PE
        if PE is not None and poste in PE.index:
            pe = PE.loc[poste]
        else:
            pe = PE_def

        # Cálculo FLEE
        mec.at[idx, col_flee] =  le * pe / he

    return mec



def calcular_fve(
    mec,
    postes_reco,
    PE=None,                 # peso del equipo por poste (daN)
    col_poste="Numero de apoyo",
    col_fve="Fve"
):
    """
    Calcula la Fuerza Vertical por Equipos (FVE).

    Definición:
    - FVE = PE

    Condiciones:
    - Solo se aplica a postes con reconectador
    - Postes sin reconectador → FVE = 0
    - Valor por defecto (solo postes con reconectador):
        * PE = 600 daN
    """

    # Inicializar columna
    mec[col_fve] = 0.0

    PE_def = 600.0  # daN

    for idx, row in mec.iterrows():

        poste = row[col_poste]

        if poste not in postes_reco.values:
            continue

        # Peso del equipo
        if PE is not None and poste in PE.index:
            pe = PE.loc[poste]
        else:
            pe = PE_def

        mec.at[idx, col_fve] = pe

    return mec

def agregar_columna_suma(
    df,
    postes_df,      # Serie del dataframe SIN repetición
    postes_rep,     # Serie CON repetición
    valores,        # Serie de valores a sumar
    nombre_columna
):
    """
    Crea una nueva columna en df con la suma de valores por poste.
    Respeta el orden del dataframe original.
    """

    # 1) Agrupar suma por poste (serie con índice = nombre del poste)
    suma_por_poste = valores.groupby(postes_rep).sum()

    # 2) Mapear al dataframe SIN perder el orden
    df[nombre_columna] = postes_df.map(suma_por_poste).fillna(0)

    return df

import numpy as np
import pandas as pd
import re

def calcular_FTEC(
    mec,
    armados,            # Serie SIN repetición
    Fvc,                # Serie SIN repetición
    altura_postes,      # Serie SIN repetición
    d_cruceta=1.12,     # float o Serie
    nombre_columna="FTEC"
):
    """
    Calcula la Fuerza por Excentricidad del peso de los conductores (FTEC).

    Solo aplica para postes con armado tipo bandera.
    """

    # ---------------------------------------------------------
    # Preparar d_cruceta
    # ---------------------------------------------------------
    if isinstance(d_cruceta, pd.Series):
        d = d_cruceta.reindex(mec.index).astype(float)
    else:
        d = pd.Series(d_cruceta, index=mec.index, dtype=float)

    # ---------------------------------------------------------
    # Identificación de armados tipo bandera
    # patrón: 1##-#
    # ---------------------------------------------------------
    def es_bandera(codigo):
        if pd.isna(codigo):
            return False
        m = re.search(r'(\d{3}-\d)$', str(codigo))
        if not m:
            return False
        return m.group(1)[0] == "1"

    bandera = armados.apply(es_bandera)

    # ---------------------------------------------------------
    # Inicializar columna
    # ---------------------------------------------------------
    mec[nombre_columna] = 0.0

    # ---------------------------------------------------------
    # Cálculo FTEC solo en postes bandera
    # ---------------------------------------------------------
    mask = bandera.values

    mec.loc[mask, nombre_columna] = (
        Fvc.loc[mask].astype(float)
        * d.loc[mask]
        / altura_postes.loc[mask].astype(float)
    )

    return mec



def calcular_EVU(
    mec,
    postes,
    altura_postes,
    carga_rotura_poste,
    tabla_capacidad,
    Hn=None,
    nombre_columna="E.V.U."
):
    """
    Calcula el Esfuerzo Vertical Último (E.V.U) por poste,
    tomando de la tabla el valor MENOR más cercano de altura
    y carga de rotura.
    """

    mec[nombre_columna] = 0.0

    # Columnas de referencia Hn
    hn_offsets = {
        "hN": 0.0,
        "hN_0_4m": 0.4,
        "hN_0_8m": 0.8,
        "hN_3_3m": 3.3
    }

    for idx in postes.index:

        poste = postes.loc[idx]
        h_poste = float(altura_postes.loc[idx])
        carga = float(carga_rotura_poste.loc[idx])

        # ---------------- Hn efectivo ----------------
        if Hn is None:
            hn_val = h_poste - 2.0
        else:
            hn_val = float(Hn.loc[idx])

        # ------------------------------------------------
        # 1) Selección de altura menor más cercana
        # ------------------------------------------------
        alturas_validas = tabla_capacidad[
            tabla_capacidad["altura_m"] <= h_poste
        ]

        if alturas_validas.empty:
            mec.loc[mec[postes.name] == poste, nombre_columna] = np.nan
            continue

        altura_sel = alturas_validas["altura_m"].max()

        tabla_altura = alturas_validas[
            alturas_validas["altura_m"] == altura_sel
        ]

        # ------------------------------------------------
        # 2) Selección de carga menor más cercana
        # ------------------------------------------------
        cargas_validas = tabla_altura[
            tabla_altura["carga_flexion_daN"] <= carga
        ]

        if cargas_validas.empty:
            mec.loc[mec[postes.name] == poste, nombre_columna] = np.nan
            continue

        carga_sel = cargas_validas["carga_flexion_daN"].max()

        fila = cargas_validas[
            cargas_validas["carga_flexion_daN"] == carga_sel
        ].iloc[0]

        # ------------------------------------------------
        # 3) Selección de columna Hn más cercana
        # ------------------------------------------------
        diferencias = {
            col: abs(hn_val - (altura_sel - off))
            for col, off in hn_offsets.items()
        }

        col_sel = min(diferencias, key=diferencias.get)

        EVU = fila[col_sel]

        mec.loc[mec[postes.name] == poste, nombre_columna] = EVU

    return mec



def calcular_FI(
    mec,
    postes_orden,
    postes_export,
    tipo_poste_orden,
    df_tiros,
    nombre_tiro_ad,
    nombre_tiro_at,
    nombre_columna="Fl"
):
    """
    Calcula el Esfuerzo Horizontal Longitudinal por 50% de desequilibrio de tracciones.

    - Aplica solo a postes tipo ANC o FL
    - Busca el máximo absoluto entre una cantidad indefinida de columnas
      identificadas por nombres en un MultiIndex
    - Considera todas las repeticiones del poste
    """

    # Inicialización conservadora
    mec[nombre_columna] = 0.0

    tipos_validos = {"ANC", "FL"}

    # Identificación de columnas relevantes en el MultiIndex
    cols_ad = [c for c in df_tiros.columns if c[1] == nombre_tiro_ad]
    cols_at = [c for c in df_tiros.columns if c[1] == nombre_tiro_at]

    cols_tiros = cols_ad + cols_at

    if len(cols_tiros) == 0:
        return mec

    # Iteración por poste final (sin repeticiones)
    for poste in postes_orden:

        # Tipo de poste (alineado con postes_orden)
        tipo = tipo_poste_orden.loc[
            tipo_poste_orden.index[postes_orden == poste][0]
        ]

        if tipo not in tipos_validos:
            continue

        # Filas exportadas correspondientes a este poste
        mask = postes_export == poste

        if not mask.any():
            continue

        # Subconjunto de tiros:
        # filas → repeticiones
        # columnas → todos los tiros adelante y atrás
        tiros_poste = df_tiros.loc[mask, cols_tiros]

        # Máximo absoluto global
        t_max = tiros_poste.abs().to_numpy().max()

        if pd.isna(t_max):
            continue

        FI = 0.5 * t_max

        # Escritura final por poste
        mec.loc[mec[postes_orden.name] == poste, nombre_columna] = FI

    return mec

def calcular_mr(
    mec,
    postes_orden,
    postes_export,
    tipo_poste_orden,
    df_tiros,
    nombre_tiro_ad,
    nombre_tiro_at,
    brazo,
    nombre_columna="Mr"
):
    """
    Calcula el momento torsor mr a partir del máximo desequilibrio de tensiones.

    - Aplica solo a postes tipo ANC o FL
    - El brazo puede ser:
        * una Series numérica (brazo por poste)
        * una Series de armados (string), de la cual se deduce el brazo
    """

    # Inicialización conservadora
    mec[nombre_columna] = 0.0

    tipos_validos = {"ANC", "FL"}

    # Identificación de columnas de tensiones en el MultiIndex
    cols_ad = [c for c in df_tiros.columns if c[1] == nombre_tiro_ad]
    cols_at = [c for c in df_tiros.columns if c[1] == nombre_tiro_at]
    cols_tiros = cols_ad + cols_at

    if len(cols_tiros) == 0:
        return mec

    # Identificar si el brazo es numérico por poste
    brazo_es_numerico = brazo.map(
        lambda x: isinstance(x, (int, float)) and not pd.isna(x)
    )

    # Iteración por poste final
    for poste in postes_orden:

        # Tipo de poste
        tipo = tipo_poste_orden.loc[
            tipo_poste_orden.index[postes_orden == poste][0]
        ]

        if tipo not in tipos_validos:
            continue

        # Filas exportadas correspondientes a este poste
        mask = postes_export == poste

        if not mask.any():
            continue

        # Subconjunto de tensiones del poste
        tiros_poste = df_tiros.loc[mask, cols_tiros]

        # Máximo absoluto global
        t_max = tiros_poste.abs().to_numpy().max()

        if pd.isna(t_max):
            continue

        # Índice del poste en la serie ordenada
        idx_poste = postes_orden.index[postes_orden == poste][0]

        # Obtención del brazo
        if brazo_es_numerico.loc[idx_poste]:
            brazo_p = brazo.loc[idx_poste]
        else:
            armado = str(brazo.loc[idx_poste]).strip()

            primer_digito = None
            for ch in armado:
                if ch.isdigit():
                    primer_digito = ch
                    break

            if primer_digito in {"6", "7"}:
                brazo_p = 0.52
            else:
                brazo_p = 1.12

        # Momento torsor
        mr = t_max * brazo_p

        # Escritura final por poste
        mec.loc[mec[postes_orden.name] == poste, nombre_columna] = mr

    return mec

def calcular_Mut(
    mec,
    postes,
    altura_postes,
    carga_rotura_poste,
    tabla_capacidad,
    nombre_columna="Mut"
):
    """
    Calcula el Momento Último de Torsión (Mut) por poste.

    Criterio:
    1) Se selecciona la mayor altura de tabla <= altura del poste.
    2) Para esa altura:
       - Si existe alguna carga con diferencia relativa <= 5% respecto
         a la carga del poste, se toma la más cercana (aunque sea mayor).
       - Si no, se toma la mayor carga menor inmediata.
    3) Si la combinación no existe, se itera hacia cargas menores
       hasta encontrar una fila válida.
    """

    mec[nombre_columna] = 0.0

    for idx in postes.index:

        poste = postes.loc[idx]
        h_poste = float(altura_postes.loc[idx])
        carga_poste = float(carga_rotura_poste.loc[idx])

        # ------------------------------------------------
        # 1) Altura menor o igual más cercana
        # ------------------------------------------------
        alturas_validas = tabla_capacidad[
            tabla_capacidad["altura_m"] <= h_poste
        ]

        if alturas_validas.empty:
            mec.loc[mec[postes.name] == poste, nombre_columna] = np.nan
            continue

        altura_sel = alturas_validas["altura_m"].max()

        tabla_altura = alturas_validas[
            alturas_validas["altura_m"] == altura_sel
        ]

        # ------------------------------------------------
        # 2) Selección de carga con criterio 5%
        # ------------------------------------------------
        cargas_tabla = tabla_altura["carga_flexion_daN"].unique()

        if len(cargas_tabla) == 0:
            mec.loc[mec[postes.name] == poste, nombre_columna] = np.nan
            continue

        # Diferencias relativas
        diffs = {
            c: abs(c - carga_poste) / carga_poste
            for c in cargas_tabla
        }

        # Cargas dentro del 5%
        cargas_5 = [c for c, d in diffs.items() if d <= 0.05]

        if cargas_5:
            # Más cercana, aunque sea mayor
            carga_sel = min(cargas_5, key=lambda c: abs(c - carga_poste))
        else:
            # Piso inmediato
            cargas_menores = [c for c in cargas_tabla if c <= carga_poste]
            if not cargas_menores:
                mec.loc[mec[postes.name] == poste, nombre_columna] = np.nan
                continue
            carga_sel = max(cargas_menores)

        # ------------------------------------------------
        # 3) Fallback si la combinación no existe
        # ------------------------------------------------
        cargas_ordenadas = sorted(cargas_tabla, reverse=True)

        fila = None
        for c in cargas_ordenadas:
            if c > carga_sel:
                continue
            candidata = tabla_altura[
                tabla_altura["carga_flexion_daN"] == c
            ]
            if not candidata.empty:
                fila = candidata.iloc[0]
                break

        if fila is None:
            mec.loc[mec[postes.name] == poste, nombre_columna] = np.nan
            continue

        # ------------------------------------------------
        # 4) Momento último de torsión
        # ------------------------------------------------
        Mut = fila["momento_torsion_daN_m"]

        mec.loc[mec[postes.name] == poste, nombre_columna] = Mut

    return mec


def crear_fase_mensajero(
    carac_postes,
    postes_orden,
    postes_export,
    numero_estructura_export,
    texto_export,
    col_fase="Fase",
    col_mensajero="Mensajero"
):
    """
    Crea las columnas 'Fase' y 'Mensajero' a partir de un texto exportado
    con formato: 'FASE / MENSAJERO'.

    Reglas:
    - Datos provenientes de exportación (desordenados y con repeticiones)
    - Para postes repetidos se toma el PRIMER valor válido
    - Valor válido: no NaN, no "", no "-", no "0"
    - NUEVO:
      Si el poste es FIN DE LÍNEA y el mensajero es inválido,
      se hereda el valor inmediatamente anterior en la exportación
    """

    # ---------------------------------------------------------
    # Inicialización
    # ---------------------------------------------------------
    carac_postes[col_fase] = np.nan
    carac_postes[col_mensajero] = np.nan

    # ---------------------------------------------------------
    # Validación: más de un separador " / "
    # ---------------------------------------------------------
    n_sep = texto_export.str.count(" / ")

    if (n_sep > 1).any():
        filas_err = texto_export[n_sep > 1]
        raise ValueError(
            f"Error: se encontró más de un ' / ' en los siguientes registros:\n{filas_err}"
        )

    # ---------------------------------------------------------
    # Separación segura
    # ---------------------------------------------------------
    partes = texto_export.str.split(" / ", expand=True)
    fase = partes.iloc[:, 0]
    mensajero = partes.iloc[:, 1]

    # ---------------------------------------------------------
    # Funciones auxiliares
    # ---------------------------------------------------------
    def es_valido(v):
        if pd.isna(v):
            return False
        v_str = str(v).strip()
        return v_str not in {"", "-", "0"}

    def primer_valor_valido(serie):
        for v in serie:
            if es_valido(v):
                return str(v).strip()
        return np.nan

    # ---------------------------------------------------------
    # Identificación de postes finales
    # ---------------------------------------------------------
    num_est = numero_estructura_export.values
    es_final = np.zeros(len(num_est), dtype=bool)

    for i in range(len(num_est)):
        if i == len(num_est) - 1:
            es_final[i] = True
        elif num_est[i] != 0 and num_est[i + 1] == 0:
            es_final[i] = True

    # ---------------------------------------------------------
    # Iteración por poste ordenado
    # ---------------------------------------------------------
    for poste in postes_orden:

        idxs = np.where(postes_export.values == poste)[0]

        if len(idxs) == 0:
            continue

        # -------------------------
        # FASE
        # -------------------------
        fase_sel = primer_valor_valido(fase.iloc[idxs])

        # -------------------------
        # MENSAJERO (con herencia en fin de línea)
        # -------------------------
        mensajero_sel = np.nan

        for i in idxs:
            val = mensajero.iloc[i]

            if es_valido(val):
                mensajero_sel = str(val).strip()
                break

            # NUEVA REGLA: fin de línea → heredar anterior
            if es_final[i] and i > 0:
                val_ant = mensajero.iloc[i - 1]
                if es_valido(val_ant):
                    mensajero_sel = str(val_ant).strip()
                    break

        # -------------------------
        # Asignación
        # -------------------------
        carac_postes.loc[
            carac_postes[postes_orden.name] == poste, col_fase
        ] = fase_sel

        carac_postes.loc[
            carac_postes[postes_orden.name] == poste, col_mensajero
        ] = mensajero_sel

    return carac_postes

def determinar_tense(
    carac_postes,
    postes_orden,
    postes_export,
    armado_export,
    tiro_adelante_export,
    tiro_atras_export,
    tabla_tiro_rotura,
    cable_export,
    nombre_columna="Tense"
):
    """
    Determina el tipo de tense por poste.

    Reglas:
    - Si el primer dígito numérico del armado es 6 o 7 → "Normal"
    - En otro caso:
        * Se obtiene el tiro máximo (adelante / atrás) considerando todas
          las repeticiones del poste
        * Se obtiene la carga de rotura máxima de los cables asociados al poste
        * Si tiro_max > 0.08 * carga_rotura → "Normal"
          else → "Reducido"
    """

    # ------------------------------------------------------------
    # Inicialización
    # ------------------------------------------------------------
    carac_postes[nombre_columna] = np.nan

    # ------------------------------------------------------------
    # Consolidación de tiros (listas de Series → Series única)
    # ------------------------------------------------------------
    if isinstance(tiro_adelante_export, list):
        tiro_adelante = sumar_lista_series(tiro_adelante_export)
    else:
        tiro_adelante = tiro_adelante_export

    if isinstance(tiro_atras_export, list):
        tiro_atras = sumar_lista_series(tiro_atras_export)
    else:
        tiro_atras = tiro_atras_export

    postes_exp = postes_export.values

    # ------------------------------------------------------------
    # Función auxiliar: primer dígito numérico del armado
    # ------------------------------------------------------------
    def primer_digito_numerico(txt):
        if not isinstance(txt, str):
            return None
        for ch in txt:
            if ch.isdigit():
                return int(ch)
        return None

    # ------------------------------------------------------------
    # Función auxiliar: carga de rotura del cable
    # ------------------------------------------------------------
    def carga_rotura_cable(nombre_cable):
        if not isinstance(nombre_cable, str):
            return None

        for _, fila in tabla_tiro_rotura.iterrows():
            conductor = fila["Conductor"]
            if isinstance(conductor, str) and nombre_cable in conductor:
                return fila["Carga de Rotura (daN)"]

        return None

    # ------------------------------------------------------------
    # Iteración por poste final (ordenado)
    # ------------------------------------------------------------
    for idx in postes_orden.index:

        poste = postes_orden.loc[idx]

        # Repeticiones del poste en exportación
        idxs = np.where(postes_exp == poste)[0]

        if len(idxs) == 0:
            continue

        # --------------------------------------------------------
        # 1) Evaluación directa por armado
        # --------------------------------------------------------
        armado = armado_export.loc[idx]
        dig = primer_digito_numerico(armado)

        if dig in (6, 7):
            carac_postes.loc[
                carac_postes[postes_orden.name] == poste,
                nombre_columna
            ] = "Normal"
            continue

        # --------------------------------------------------------
        # 2) Tiro máximo (adelante / atrás) del poste
        # --------------------------------------------------------
        tiros = []

        for i in idxs:
            if not pd.isna(tiro_adelante.iloc[i]):
                tiros.append(abs(tiro_adelante.iloc[i]))
            if not pd.isna(tiro_atras.iloc[i]):
                tiros.append(abs(tiro_atras.iloc[i]))

        if not tiros:
            continue

        tiro_max = max(tiros)

        # --------------------------------------------------------
        # 3) Carga de rotura máxima de los cables del poste
        # --------------------------------------------------------
        cargas = []

        for i in idxs:
            carga = carga_rotura_cable(cable_export.iloc[i])
            if carga is not None:
                cargas.append(carga)

        if not cargas:
            continue

        carga_rotura_max = max(cargas)

        # --------------------------------------------------------
        # 4) Comparación 8%
        # --------------------------------------------------------
        if tiro_max > 0.08 * carga_rotura_max:
            tense = "Normal"
        else:
            tense = "Reducido"

        # Escritura final por poste
        carac_postes.loc[
            carac_postes[postes_orden.name] == poste,
            nombre_columna
        ] = tense

    return carac_postes



def calcular_vanos_adelante_atras(
    carac_postes,
    postes_orden,
    postes_export,
    numero_estructura_export,
    vano_adelante_export,
    col_vano_post="Vano posterior",
    col_vano_ant="Vano anterior"
):
    """
    Determina el vano anterior y vano posterior por poste, considerando
    rutas, posiciones (inicial / intermedia / final) y repeticiones.
    """

    carac_postes[col_vano_post] = 0.0
    carac_postes[col_vano_ant] = 0.0

    ne = numero_estructura_export.values
    vanos = vano_adelante_export.values
    postes_exp = postes_export.values

    n = len(ne)

    # ------------------------------------------------------------
    # 1) Clasificar cada fila de la exportación
    # ------------------------------------------------------------
    tipo_posicion = []

    for i in range(n):
        if ne[i] == 0:
            tipo_posicion.append("inicio")
        else:
            # final si el siguiente es 0 o si es el último registro
            if i == n - 1 or ne[i + 1] == 0:
                tipo_posicion.append("final")
            else:
                tipo_posicion.append("intermedio")

    tipo_posicion = np.array(tipo_posicion)

    # ------------------------------------------------------------
    # 2) Funciones auxiliares
    # ------------------------------------------------------------
    def primera_posicion(indices, tipo):
        for i in indices:
            if tipo_posicion[i] == tipo:
                return i
        return None

    # ------------------------------------------------------------
    # 3) Iteración por poste final ordenado
    # ------------------------------------------------------------
    for poste in postes_orden:

        idxs = np.where(postes_exp == poste)[0]

        if len(idxs) == 0:
            continue

        # Clasificar repeticiones
        idx_inicio = [i for i in idxs if tipo_posicion[i] == "inicio"]
        idx_inter = [i for i in idxs if tipo_posicion[i] == "intermedio"]
        idx_final  = [i for i in idxs if tipo_posicion[i] == "final"]

        vano_ant = 0.0
        vano_post = 0.0

        # --------------------------------------------------------
        # CASO 1: hay al menos una aparición intermedia
        # --------------------------------------------------------
        if idx_inter:
            i = idx_inter[0]
            vano_post = vanos[i]
            vano_ant = vanos[i - 1] if i > 0 else 0.0

        # --------------------------------------------------------
        # CASO 2: poste inicial
        # --------------------------------------------------------
        elif idx_inicio:
            # 2.1 hay repetición final
            if idx_final:
                i_ini = idx_inicio[0]
                i_fin = idx_final[0]
                vano_post = vanos[i_ini]
                vano_ant = vanos[i_fin - 1] if i_fin > 0 else 0.0

            # 2.2 solo repeticiones iniciales
            elif len(idx_inicio) >= 2:
                vano_post = vanos[idx_inicio[0]]
                vano_ant = vanos[idx_inicio[1]]

            # 2.3 único inicial
            else:
                i = idx_inicio[0]
                vano_post = vanos[i]
                vano_ant = 0.0

        # --------------------------------------------------------
        # CASO 3: poste final
        # --------------------------------------------------------
        elif idx_final:
            # 3.1 solo finales
            if len(idx_final) >= 2:
                i1 = idx_final[0]
                i2 = idx_final[1]
                vano_ant = vanos[i1 - 1] if i1 > 0 else 0.0
                vano_post = vanos[i2 - 1] if i2 > 0 else 0.0

            # 3.2 único final
            else:
                i = idx_final[0]
                vano_ant = vanos[i - 1] if i > 0 else 0.0
                vano_post = 0.0

        # --------------------------------------------------------
        # Escritura en carac_postes
        # --------------------------------------------------------
        carac_postes.loc[
            carac_postes[postes_orden.name] == poste, col_vano_ant
        ] = vano_ant

        carac_postes.loc[
            carac_postes[postes_orden.name] == poste, col_vano_post
        ] = vano_post

    return carac_postes

def identificar_retenida(
    carac_postes,
    postes_orden,
    postes_export,
    retenidas_export,
    armado_orden,
    col_bisectora="Bisectora",
    col_90="Conjunto a 90º"
):
    """
    Determina si un poste tiene retenida bisectora o conjunto a 90°.

    - Si todas las retenidas (incluyendo repeticiones) son 0, "-", "", o NaN:
      → ambas columnas quedan NaN.
    - Si existe al menos una retenida válida:
      → se evalúa el armado del poste.
        * Si los dos últimos dígitos antes del último guión y el dígito final son 35
          → Conjunto a 90º = "X"
        * En caso contrario
          → Bisectora = "X"
    """

    # Inicialización
    carac_postes[col_bisectora] = np.nan
    carac_postes[col_90] = np.nan

    postes_exp = postes_export.values
    retenidas = retenidas_export.values

    # ------------------------------------------------------------
    # Función auxiliar: valor válido de retenida
    # ------------------------------------------------------------
    def es_retenida_valida(val):
        if pd.isna(val):
            return False
        if isinstance(val, str) and val.strip() == "":
            return False
        if val == "-" or val == 0:
            return False
        return True

    # ------------------------------------------------------------
    # Iteración por poste final (ordenado)
    # ------------------------------------------------------------
    for idx in postes_orden.index:

        poste = postes_orden.loc[idx]

        # Repeticiones en exportación
        idxs = np.where(postes_exp == poste)[0]

        if len(idxs) == 0:
            continue

        # ¿Existe al menos una retenida válida?
        hay_retenida = False
        for i in idxs:
            if es_retenida_valida(retenidas[i]):
                hay_retenida = True
                break

        if not hay_retenida:
            continue

        # --------------------------------------------------------
        # Evaluación del armado
        # --------------------------------------------------------
        armado = armado_orden.loc[idx]

        if pd.isna(armado) or not isinstance(armado, str):
            continue

        # Se espera formato "$$$$-###-#"
        partes = armado.split("-")

        if len(partes) < 3:
            # Formato no reconocible → conservador: bisectora
            carac_postes.loc[
                carac_postes[postes_orden.name] == poste, col_bisectora
            ] = "X"
            continue

        try:
            ult_dos = partes[-2][-2:]   # dos últimos dígitos antes del último guión
            ult_uno = partes[-1]        # dígito final
            codigo = ult_dos + ult_uno
        except Exception:
            codigo = ""

        # --------------------------------------------------------
        # Asignación final
        # --------------------------------------------------------
        if codigo == "35":
            carac_postes.loc[
                carac_postes[postes_orden.name] == poste, col_90
            ] = "X"
        else:
            carac_postes.loc[
                carac_postes[postes_orden.name] == poste, col_bisectora
            ] = "X"

    return carac_postes


def calcular_tiro_maximo(
    carac_postes,
    postes_orden,
    postes_export,
    tiro_adelante_export,
    tiro_atras_export,
    nombre_columna="Tiro_max"
):
    """
    Calcula por cada poste el tiro máximo entre tiro adelante y tiro atrás,
    considerando todas las repeticiones provenientes de la exportación.

    - Se toma el máximo absoluto.
    - Si el poste se repite, se evalúan todas sus repeticiones.
    - Escritura final ordenada por postes_orden.
    """

    # Inicialización conservadora
    carac_postes[nombre_columna] = np.nan

    postes_exp = postes_export.values
    ta = tiro_adelante_export.values
    td = tiro_atras_export.values

    # ------------------------------------------------------------
    # Iteración por poste final (ordenado)
    # ------------------------------------------------------------
    for idx in postes_orden.index:

        poste = postes_orden.loc[idx]

        # Repeticiones en exportación
        idxs = np.where(postes_exp == poste)[0]

        if len(idxs) == 0:
            continue

        # Tiros de todas las repeticiones
        tiros = []

        for i in idxs:
            if not pd.isna(ta[i]):
                tiros.append(abs(ta[i]))
            if not pd.isna(td[i]):
                tiros.append(abs(td[i]))

        if not tiros:
            continue

        tiro_max = max(tiros)

        # Escritura final por poste
        carac_postes.loc[
            carac_postes[postes_orden.name] == poste, nombre_columna
        ] = tiro_max

    return carac_postes
