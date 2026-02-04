import pandas as pd

TABLA_AFINIA_VIENTO = {
    (11, 500):  {"A": {"rural": 61.00,   "urbana": 52.10},
                 "B": {"rural": 47.30,   "urbana": 29.30},
                 "C": {"rural": 30.27,   "urbana": 20.35}},
    (11, 735):  {"A": {"rural": 61.00,   "urbana": 52.10},
                 "B": {"rural": 47.30,   "urbana": 29.30},
                 "C": {"rural": 30.27,   "urbana": 20.35}},
    (11, 1030): {"A": {"rural": 55.20,   "urbana": 66.14},
                 "B": {"rural": 57.13,   "urbana": 37.20},
                 "C": {"rural": 38.44,   "urbana": 25.84}},
    (11, 1324): {"A": {"rural": 57.59,   "urbana": 69.03},
                 "B": {"rural": 57.41,   "urbana": 38.83},
                 "C": {"rural": 40.13,   "urbana": 26.97}},

    (12, 500):  {"A": {"rural": 65.74,   "urbana": 58.72},
                 "B": {"rural": 53.36,   "urbana": 33.03},
                 "C": {"rural": 34.15,   "urbana": 22.94}},
    (12, 735):  {"A": {"rural": 65.51,   "urbana": 64.90},
                 "B": {"rural": 59.00,   "urbana": 36.51},
                 "C": {"rural": 37.76,   "urbana": 25.35}},
    (12, 1030): {"A": {"rural": 61.92,   "urbana": 74.15},
                 "B": {"rural": 61.66,   "urbana": 41.71},
                 "C": {"rural": 43.13,   "urbana": 28.96}},
    (12, 1324): {"A": {"rural": 64.52,   "urbana": 77.29},
                 "B": {"rural": 61.85,   "urbana": 43.47},
                 "C": {"rural": 44.97,   "urbana": 30.19}},
    (12, 1600): {"A": {"rural": 73.52,   "urbana": 88.07},
                 "B": {"rural": 61.52,   "urbana": 49.54},
                 "C": {"rural": 49.94,   "urbana": 34.40}},
    (12, 2500): {"A": {"rural": 82.53,   "urbana": 94.71},
                 "B": {"rural": 56.17,   "urbana": 55.59},
                 "C": {"rural": 50.33,   "urbana": 38.61}},
    (12, 4000): {"A": {"rural": 100.80,  "urbana": 94.65},
                 "B": {"rural": 68.59,   "urbana": 67.88},
                 "C": {"rural": 43.90,   "urbana": 47.14}},

    (14, 735):  {"A": {"rural": 66.92,   "urbana": 80.07},
                 "B": {"rural": 69.52,   "urbana": 45.04},
                 "C": {"rural": 46.64,   "urbana": 31.28}},
    (14, 1030): {"A": {"rural": 76.18,   "urbana": 91.09},
                 "B": {"rural": 70.25,   "urbana": 51.24},
                 "C": {"rural": 53.09,   "urbana": 35.58}},
    (14, 1324): {"A": {"rural": 79.19,   "urbana": 94.73},
                 "B": {"rural": 70.22,   "urbana": 53.28},
                 "C": {"rural": 55.22,   "urbana": 37.00}},
    (14, 1600): {"A": {"rural": 90.02,   "urbana": 105.52},
                 "B": {"rural": 61.25,   "urbana": 60.55},
                 "C": {"rural": 57.26,   "urbana": 42.05}},
    (14, 2500): {"A": {"rural": 100.69,  "urbana": 106.36},
                 "B": {"rural": 68.55,   "urbana": 67.73},
                 "C": {"rural": 57.35,   "urbana": 47.03}},
    (14, 4000): {"A": {"rural": 122.10,  "urbana": 91.25},
                 "B": {"rural": 83.16,   "urbana": 80.59},
                 "C": {"rural": 53.22,   "urbana": 57.03}},
}
def df_tabla_afinia_viento(tabla_dict):
    registros = []
    for (altura, carga), zonas in tabla_dict.items():
        for zona, valores in zonas.items():
            registros.append({
                "altura_m": altura,
                "carga_daN": carga,
                "zona": zona,
                "rural_daN": valores["rural"],
                "urbana_daN": valores["urbana"]
            })
    return pd.DataFrame(registros)

TABLA_AFINIA_VIENTO_DF = df_tabla_afinia_viento(TABLA_AFINIA_VIENTO)

""" print(TABLA_AFINIA_VIENTO_DF)
 """


# ============================================================
#  TABLA 36 - CAPACIDAD MECÁNICA DE POSTES  (AFINIA)
# ============================================================

TABLA_AFINIA_CAPACIDAD_MECANICA = {
    (11, 500):  {
        "esfuerzo_flexion_daN": 500,
        "momento_torsion_daN_m": 351,
        "hN": 709,
        "hN_0_4": 936,
        "hN_0_8": 1074,
        "hN_3_3": 6888
    },
    (11, 735):  {
        "esfuerzo_flexion_daN": 735,
        "momento_torsion_daN_m": 380,
        "hN": 709,
        "hN_0_4": 936,
        "hN_0_8": 1074,
        "hN_3_3": 6888
    },
    (11, 1030): {
        "esfuerzo_flexion_daN": 1030,
        "momento_torsion_daN_m": 891,
        "hN": 2539,
        "hN_0_4": 3199,
        "hN_0_8": 3591,
        "hN_3_3": 18228
    },
    (11, 1324): {
        "esfuerzo_flexion_daN": 1324,
        "momento_torsion_daN_m": 1079,
        "hN": 3240,
        "hN_0_4": 4067,
        "hN_0_8": 4558,
        "hN_3_3": 22720
    },

    (12, 500):  {
        "esfuerzo_flexion_daN": 500,
        "momento_torsion_daN_m": 351,
        "hN": 568,
        "hN_0_4": 742,
        "hN_0_8": 847,
        "hN_3_3": 4919
    },
    (12, 735):  {
        "esfuerzo_flexion_daN": 735,
        "momento_torsion_daN_m": 380,
        "hN": 568,
        "hN_0_4": 742,
        "hN_0_8": 847,
        "hN_3_3": 4919
    },
    (12, 1030): {
        "esfuerzo_flexion_daN": 1030,
        "momento_torsion_daN_m": 891,
        "hN": 2034,
        "hN_0_4": 2538,
        "hN_0_8": 2835,
        "hN_3_3": 13060
    },
    (12, 1324): {
        "esfuerzo_flexion_daN": 1324,
        "momento_torsion_daN_m": 1079,
        "hN": 2608,
        "hN_0_4": 3243,
        "hN_0_8": 3616,
        "hN_3_3": 16365
    },
    (12, 1600): {
        "esfuerzo_flexion_daN": 1600,
        "momento_torsion_daN_m": 5096,
        "hN": 6212,
        "hN_0_4": 7581,
        "hN_0_8": 0,        # NO SE PROPORCIONA EN TABLA ORIGINAL
        "hN_3_3": 27487
    },
    (12, 2500): {
        "esfuerzo_flexion_daN": 2500,
        "momento_torsion_daN_m": 8831,
        "hN": 10604,
        "hN_0_4": 12755,
        "hN_0_8": 0,        # NO SE PROPORCIONA EN TABLA ORIGINAL
        "hN_3_3": 43322
    },
    (12, 4000): {
        "esfuerzo_flexion_daN": 4000,
        "momento_torsion_daN_m": 22194,
        "hN": 26131,
        "hN_0_4": 30842,
        "hN_0_8": 0,        # NO SE PROPORCIONA EN TABLA ORIGINAL
        "hN_3_3": 94253
    },

    (14, 735):  {
        "esfuerzo_flexion_daN": 735,
        "momento_torsion_daN_m": 568,
        "hN": 652,
        "hN_0_4": 821,
        "hN_0_8": 920,
        "hN_3_3": 4139
    },
    (14, 1030): {
        "esfuerzo_flexion_daN": 1030,
        "momento_torsion_daN_m": 891,
        "hN": 1372,
        "hN_0_4": 1687,
        "hN_0_8": 1869,
        "hN_3_3": 7474
    },
    (14, 1324): {
        "esfuerzo_flexion_daN": 1324,
        "momento_torsion_daN_m": 1079,
        "hN": 1779,
        "hN_0_4": 2180,
        "hN_0_8": 2412,
        "hN_3_3": 9478
    },
    (14, 1600): {
        "esfuerzo_flexion_daN": 1600,
        "momento_torsion_daN_m": 3466,
        "hN": 4164,
        "hN_0_4": 5003,
        "hN_0_8": 0,        # NO SE PROPORCIONA
        "hN_3_3": 16112
    },
    (14, 2500): {
        "esfuerzo_flexion_daN": 2500,
        "momento_torsion_daN_m": 6046,
        "hN": 7159,
        "hN_0_4": 8482,
        "hN_0_8": 0,
        "hN_3_3": 25346
    },
    (14, 4000): {
        "esfuerzo_flexion_daN": 4000,
        "momento_torsion_daN_m": 15225,
        "hN": 17680,
        "hN_0_4": 20561,
        "hN_0_8": 0,
        "hN_3_3": 55400
    },
}

# ============================================================
# Convertir a DataFrame
# ============================================================

def df_tabla_afinia_capacidad(tabla_dict):
    registros = []
    for (altura, carga), valores in tabla_dict.items():
        fila = {
            "altura_m": altura,
            "carga_flexion_daN": carga,
            "momento_torsion_daN_m": valores["momento_torsion_daN_m"],
            "hN": valores["hN"],
            "hN_0_4m": valores["hN_0_4"],
            "hN_0_8m": valores["hN_0_8"],
            "hN_3_3m": valores["hN_3_3"],
        }
        registros.append(fila)
    return pd.DataFrame(registros)

TABLA_AFINIA_CAPACIDAD_MECANICA_DF = df_tabla_afinia_capacidad(TABLA_AFINIA_CAPACIDAD_MECANICA)



# ============================================================
#  TABLA 46 - CAPACIDAD VERTICAL DE POSTES CON RETENIDAS (AFINIA)
# ============================================================

TABLA_AFINIA_CAP_VERTICAL_RETENIDAS = {
    (11, 500): {
        "hN": 2127,
        "hN_0_4": 2808,
        "hN_0_8": 3222,
        "hN_3_3": 20664
    },
    (11, 735): {
        "hN": 2127,
        "hN_0_4": 2808,
        "hN_0_8": 3222,
        "hN_3_3": 20664
    },
    (11, 1030): {
        "hN": 7617,
        "hN_0_4": 9597,
        "hN_0_8": 10773,
        "hN_3_3": 54684
    },
    (11, 1324): {
        "hN": 9720,
        "hN_0_4": 12201,
        "hN_0_8": 13674,
        "hN_3_3": 68160
    },

    (12, 500): {
        "hN": 1704,
        "hN_0_4": 2226,
        "hN_0_8": 2541,
        "hN_3_3": 14757
    },
    (12, 735): {
        "hN": 1704,
        "hN_0_4": 2226,
        "hN_0_8": 2541,
        "hN_3_3": 14757
    },
    (12, 1030): {
        "hN": 6102,
        "hN_0_4": 7614,
        "hN_0_8": 8505,
        "hN_3_3": 39180
    },
    (12, 1324): {
        "hN": 7824,
        "hN_0_4": 9729,
        "hN_0_8": 10848,
        "hN_3_3": 49095
    },

    (14, 735): {
        "hN": 1956,
        "hN_0_4": 2463,
        "hN_0_8": 2760,
        "hN_3_3": 12417
    },
    (14, 1030): {
        "hN": 4116,
        "hN_0_4": 5061,
        "hN_0_8": 5607,
        "hN_3_3": 22422
    },
    (14, 1324): {
        "hN": 5337,
        "hN_0_4": 6540,
        "hN_0_8": 7236,
        "hN_3_3": 28434
    },

    (12, 1600): {
        "hN": 15288,
        "hN_0_4": 18637,
        "hN_0_8": 22743,
        "hN_3_3": 83540
    },
    (12, 2500): {
        "hN": 26493,
        "hN_0_4": 31812,
        "hN_0_8": 38265,
        "hN_3_3": 129968
    },
    (12, 4000): {
        "hN": 66582,
        "hN_0_4": 78394,
        "hN_0_8": 92528,
        "hN_3_3": 282761
    },

    (14, 1600): {
        "hN": 10398,
        "hN_0_4": 12493,
        "hN_0_8": 15009,
        "hN_3_3": 48338
    },
    (14, 2500): {
        "hN": 18140,
        "hN_0_4": 21478,
        "hN_0_8": 25448,
        "hN_3_3": 76038
    },
    (14, 4000): {
        "hN": 45675,
        "hN_0_4": 53041,
        "hN_0_8": 61683,
        "hN_3_3": 166202
    },
}

# ============================================================
#  CONVERTIR A DATAFRAME
# ============================================================

def df_tabla_afinia_vertical(tabla_dict):
    registros = []
    for (altura, carga), valores in tabla_dict.items():
        fila = {
            "altura_m": altura,
            "carga_flexion_daN": carga,
            "hN": valores["hN"],
            "hN_0_4m": valores["hN_0_4"],
            "hN_0_8m": valores["hN_0_8"],
            "hN_3_3m": valores["hN_3_3"],
        }
        registros.append(fila)
    return pd.DataFrame(registros)

TABLA_AFINIA_CAP_VERTICAL_DF = df_tabla_afinia_vertical(TABLA_AFINIA_CAP_VERTICAL_RETENIDAS)

""" print(TABLA_AFINIA_CAP_VERTICAL_DF) """

df_cables_acero_galvanizado = pd.DataFrame({
    "Denominación": ["3/8''", "1/2''"],
    "Sección Transversal Total (mm²)": [51.1, 96.5],
    "Nº Alambres": [7, 7],
    "Diámetro nominal del alambre (mm)": [3.05, 4.19],
    "Diámetro nominal del cable (mm)": [9.52, 12.7],
    "Peso (daN/m)": [0.399, 0.755],
    "Carga de Rotura (daN)": [6840, 11960],
    "Módulo de Elasticidad (daN/mm²)": [18130, 18130],
    "Coeficiente de dilatación lineal (°C⁻¹)": [11.5e-6, 11.5e-6]
})

""" print(df_cables_acero_galvanizado["Carga de Rotura (daN)"])  """



tabla_B2_4 = pd.DataFrame({
    "Zona": ["A", "A", "B", "B", "C", "C"],
    "Area": ["Rural", "Urbana", "Rural", "Urbana", "Rural", "Urbana"],
    "Categoría Terreno": [
        "Tipo I",
        "Tipo IV",
        "Tipo II",
        "Tipo IV",
        "Tipo II",
        "Tipo IV"
    ],
    "q0 (daN / (m ^ 2))": [
        42.55,
        16.12,
        25.18,
        9.06,
        16.12,
        6.30
    ]
})

""" print(tabla_B2_4) """


tabla_cap_conductores = pd.DataFrame({
    "Conductor": [
        "477 kcmil",
        "336,4 kcmil",
        "266,8 kcmil",
        "4/0 AWG",
        "1/0 AWG"
    ],
    "Carga de Rotura (daN)": [
        8677,
        6270,
        5028,
        3716,
        1949
    ]
})



columnas = pd.MultiIndex.from_product(
    [
        ["≤1.2 m", "1.2 < L ≤ 2.2 m", ">3.0 m"],
        ["A", "B", "C"]
    ],
    names=["Altura retenida", "Grupo"]
)

data = [
    [0.11, 1.54, 1.82, 0.16, 1.46, 1.69, 0.19, 1.40, 1.62],
    [0.21, 1.37, 1.65, 0.24, 1.32, 1.44, 0.33, 1.16, 1.56]
]

index = pd.Index(
    ["500–735", "1030–1324"],
    name="Carga (daN)"
)

c_retenida_3_8 = pd.DataFrame(data, index=index, columns=columnas)



def expandir_rangos_carga(df):
    filas = []

    for idx, row in df.iterrows():
        # Separar rango "500–735"
        limites = idx.replace("–", "-").split("-")
        limites = [int(l.strip()) for l in limites]

        for carga in limites:
            nueva_fila = row.copy()
            nueva_fila.name = carga
            filas.append(nueva_fila)

    df_expandido = pd.DataFrame(filas)
    df_expandido.index.name = "Carga (daN)"

    return df_expandido.sort_index()

c_retenida_3_8 = expandir_rangos_carga(c_retenida_3_8)


# Columnas MultiIndex
columnas = pd.MultiIndex.from_product(
    [
        ["≤1.2 m", "1.2 < L ≤ 2.2 m", ">3.0 m"],
        ["A", "B", "C"]
    ],
    names=["Longitud poste", "Grupo"]
)

# Datos
data = [
    [0.06, 1.63, 1.88, 0.09, 1.58, 1.82, 0.11, 1.54, 1.78],
    [0.12, 1.51, 1.75, 0.14, 1.48, 1.71, 0.21, 1.38, 1.59]
]

# Índice con rangos
index = pd.Index(
    ["500–735", "1030–1324"],
    name="Carga (daN)"
)

c_retenida_1_2 = pd.DataFrame(data, index=index, columns=columnas)
c_retenida_1_2 =expandir_rangos_carga(c_retenida_1_2) 
""" print(c_retenida_1_2) """



# Columnas MultiIndex
columnas = pd.MultiIndex.from_product(
    [
        ["≤1.2 m", "1.2 < L ≤ 2.2 m", ">3.0 m"],
        ["A", "B", "CP"]
    ],
    names=["Longitud poste", "Grupo"]
)

# Datos organizados por beta
bloques = {
    20: [
        ["500–735",  0.11, 1.54, 1.73, 0.16, 1.46, 1.64, 0.19, 1.40, 1.57],
        ["1030–1324",0.21, 1.37, 1.53, 0.24, 1.32, 1.47, 0.33, 1.16, 1.30],
    ],
    30: [
        ["500–735",  0.11, 1.54, 1.66, 0.16, 1.46, 1.58, 0.19, 1.40, 1.51],
        ["1030–1324",0.21, 1.37, 1.47, 0.24, 1.32, 1.42, 0.33, 1.16, 1.25],
    ],
    40: [
        ["500–735",  0.11, 1.54, 1.57, 0.16, 1.46, 1.49, 0.19, 1.40, 1.43],
        ["1030–1324",0.21, 1.37, 1.39, 0.24, 1.32, 1.34, 0.33, 1.16, 1.19],
    ],
    50: [
        ["500–735",  0.11, 1.54, 1.46, 0.16, 1.46, 1.39, 0.19, 1.40, 1.33],
        ["1030–1324",0.21, 1.37, 1.30, 0.24, 1.32, 1.25, 0.33, 1.16, 1.10],
    ],
    60: [
        ["500–735",  0.11, 1.54, 1.34, 0.16, 1.46, 1.27, 0.19, 1.40, 1.21],
        ["1030–1324",0.21, 1.37, 1.18, 0.24, 1.32, 1.14, 0.33, 1.16, 1.01],
    ],
    70: [
        ["500–735",  0.11, 1.54, 1.19, 0.16, 1.46, 1.13, 0.19, 1.40, 1.08],
        ["1030–1324",0.21, 1.37, 1.06, 0.24, 1.32, 1.02, 0.33, 1.16, 0.90],
    ],
    80: [
        ["500–735",  0.11, 1.54, 1.04, 0.16, 1.46, 0.99, 0.19, 1.40, 0.95],
        ["1030–1324",0.21, 1.37, 0.93, 0.24, 1.32, 0.89, 0.33, 1.16, 0.79],
    ],
    90: [
        ["500–735",  0.11, 1.54, 0.89, 0.16, 1.46, 0.84, 0.19, 1.40, 0.81],
        ["1030–1324",0.21, 1.37, 0.79, 0.24, 1.32, 0.76, 0.33, 1.16, 0.67],
    ],
}

frames = []

for beta, filas in bloques.items():
    df_beta = pd.DataFrame(
        [f[1:] for f in filas],
        index=pd.Index([f[0] for f in filas], name="Carga (daN)"),
        columns=columnas
    )
    df_beta["β"] = beta
    frames.append(df_beta)

c_ret_3_8_90 = pd.concat(frames).set_index("β", append=True).reorder_levels(["β", "Carga (daN)"])

def expandir_rangos_carga_multiindex(df):
    filas = []

    for (beta, carga_rango), row in df.iterrows():
        limites = carga_rango.replace("–", "-").split("-")
        limites = [int(l.strip()) for l in limites]

        for carga in limites:
            nueva = row.copy()
            nueva.name = (beta, carga)
            filas.append(nueva)

    df_out = pd.DataFrame(filas)
    df_out.index = pd.MultiIndex.from_tuples(
        df_out.index, names=["β (°)", "Carga (daN)"]
    )

    return df_out.sort_index()

c_ret_3_8_90 = expandir_rangos_carga_multiindex(c_ret_3_8_90)

""" print(c_ret_3_8_90) """


# Columnas MultiIndex
columnas = pd.MultiIndex.from_product(
    [
        ["≤1.2 m", "1.2 < L ≤ 2.2 m", ">3.0 m"],
        ["A", "B", "CP"]
    ],
    names=["Longitud poste", "Grupo"]
)

bloques = {
    20: [
        ["500–735",  0.06, 1.63, 1.82, 0.09, 1.58, 1.77, 0.11, 1.54, 1.72],
        ["1030–1324",0.12, 1.52, 1.70, 0.14, 1.48, 1.66, 0.21, 1.38, 1.54],
    ],
    30: [
        ["500–735",  0.06, 1.63, 1.75, 0.09, 1.58, 1.70, 0.11, 1.54, 1.66],
        ["1030–1324",0.12, 1.52, 1.64, 0.14, 1.48, 1.60, 0.21, 1.38, 1.48],
    ],
    40: [
        ["500–735",  0.06, 1.63, 1.66, 0.09, 1.58, 1.61, 0.11, 1.54, 1.57],
        ["1030–1324",0.12, 1.52, 1.55, 0.14, 1.48, 1.51, 0.21, 1.38, 1.40],
    ],
    50: [
        ["500–735",  0.06, 1.63, 1.54, 0.09, 1.58, 1.50, 0.11, 1.54, 1.46],
        ["1030–1324",0.12, 1.52, 1.44, 0.14, 1.48, 1.41, 0.21, 1.38, 1.30],
    ],
    60: [
        ["500–735",  0.06, 1.63, 1.41, 0.09, 1.58, 1.37, 0.11, 1.54, 1.33],
        ["1030–1324",0.12, 1.52, 1.31, 0.14, 1.48, 1.29, 0.21, 1.38, 1.19],
    ],
    70: [
        ["500–735",  0.06, 1.63, 1.26, 0.09, 1.58, 1.22, 0.11, 1.54, 1.19],
        ["1030–1324",0.12, 1.52, 1.18, 0.14, 1.48, 1.15, 0.21, 1.38, 1.07],
    ],
    80: [
        ["500–735",  0.06, 1.63, 1.10, 0.09, 1.58, 1.07, 0.11, 1.54, 1.04],
        ["1030–1324",0.12, 1.52, 1.03, 0.14, 1.48, 1.01, 0.21, 1.38, 0.93],
    ],
    90: [
        ["500–735",  0.06, 1.63, 0.94, 0.09, 1.58, 0.91, 0.11, 1.54, 0.89],
        ["1030–1324",0.12, 1.52, 0.88, 0.14, 1.48, 0.86, 0.21, 1.38, 0.79],
    ],
}

frames = []

for beta, filas in bloques.items():
    df_beta = pd.DataFrame(
        [f[1:] for f in filas],
        index=pd.Index([f[0] for f in filas], name="Carga (daN)"),
        columns=columnas
    )
    df_beta["β"] = beta
    frames.append(df_beta)

c_ret_1_2_90 = (
    pd.concat(frames)
      .set_index("β", append=True)
      .reorder_levels(["β", "Carga (daN)"])
)

c_ret_1_2_90 = expandir_rangos_carga_multiindex(c_ret_1_2_90)


 import numpy as np
import pandas as pd


def calcular_fuerza_residual_retenidas(
    carac_postes,
    postes_orden,
    postes_export,
    retenidas_export,
    flmc,
    altura_postes,
    capacidad_poste,
    tablas_coeficientes,
    tipo_retenida,
    tipo_poste,
    tabla_cables_acero,
    calibre_retenida="3/8",
    altura_retenidas=None,
    col_fres="Fuerza Residual Fres (daN)",
    col_fvert="Fuerza vertical por retenida Fvert (daN)",
    col_trac="Tracción total cable ret.(daN)",
    col_pret="Pretensionado de la Retenida (daN)",
    col_rot="Carga rotura cable ret. (daN)",
):
    """
    Calcula la fuerza residual por retenidas y agrega la columna
    'Fuerza Residual Fres (daN)' al dataframe carac_postes.
    """

    # Inicialización
    for c in [col_fres, col_fvert, col_trac, col_pret, col_rot]:
        carac_postes[c] = np.nan
        
    if altura_retenidas is None:
        altura_retenidas = altura_postes

    # Selección de tablas según calibre
    if calibre_retenida == "3/8":
        tablas_calibre = [tablas_coeficientes[0], tablas_coeficientes[2]]
    else:
        tablas_calibre = [tablas_coeficientes[1], tablas_coeficientes[3]]
    # --------------------------------------------------
    # Funciones auxiliares
    # --------------------------------------------------        
    def obtener_ru(calibre):
        fila = tabla_cables_acero[
            tabla_cables_acero["Denominación"].astype(str).str.contains(str(calibre))
        ]
        if fila.empty:
            return np.nan
        return float(fila.iloc[0]["Carga de Rotura (daN)"])
    
    def seleccionar_longitud(delta_h):
        if delta_h <= 1.2:
            return "≤1.2 m"
        elif delta_h <= 2.2:
            return "1.2 < L ≤ 2.2 m"
        else:
            return ">3.0 m"

    def valor_cercano(valores, objetivo):
        valores = np.array(sorted(valores))
        dif_rel = np.abs(valores - objetivo) / objetivo

        mask_5 = dif_rel <= 0.05
        if mask_5.any():
            return valores[mask_5][
                np.argmin(np.abs(valores[mask_5] - objetivo))
            ]

        menores = valores[valores <= objetivo]
        if len(menores) > 0:
            return menores[-1]

        return valores[0]

    # Iteración por poste (orden final)
    for i, poste in enumerate(postes_orden):

        mask = postes_export == poste

        if not mask.any():
            continue

        # 1. Verificar retenidas
        ret_vals = retenidas.loc[mask]
        ret_vals = ret_vals[ret_vals > 0]

        if ret_vals.empty:
            continue

        # 2. Alturas
        Hn = altura_postes.iloc[i]
        Hj = altura_retenidas.iloc[i]
        delta_h = Hn - Hj

        # 3. Tipo de retenida
        tipo_fila = tipo_retenida.iloc[i]

        if tipo_fila["Bisectora"] == "X":
            es_90 = False
        elif tipo_fila["Conjunto a 90º"] == "X":
            es_90 = True
        else:
            continue

        # 4. Selección de tabla
        if es_90:
            tabla = tablas_calibre[1]  # con beta
            betas = tabla.index.get_level_values("β (°)").unique()
            beta_sel = valor_cercano(betas, 90)
        else:
            tabla = tablas_calibre[0]  # sin beta

        # Carga de rotura
        carga_poste = capacidad_poste.iloc[i]

        if es_90:
            cargas_tabla = tabla.index.get_level_values("Carga (daN)").unique()
        else:
            cargas_tabla = tabla.index

        carga_sel = valor_cercano(cargas_tabla, carga_poste)

        # Longitud
        col_long = seleccionar_longitud(delta_h)

        # Coeficientes A,B,C
        if es_90:
            A = tabla.loc[(beta_sel, carga_sel), (col_long, "A")]
            B = tabla.loc[(beta_sel, carga_sel), (col_long, "B")]
            C = tabla.loc[(beta_sel, carga_sel), (col_long, "C")]
        else:
            A = tabla.loc[carga_sel, (col_long, "A")]
            B = tabla.loc[carga_sel, (col_long, "B")]
            C = tabla.loc[carga_sel, (col_long, "C")]

        # 5. Fuerza residual
        fres = flmc.iloc[i] * A * Hj / Hn
        fver = flmc.iloc[i]*B
        trac = flmc.iloc[i]*C*1.5
        Ru = obtener_ru(calibre_retenida)
        tp = tipo_poste.iloc[i]
        if tp == "ANC":
            pretr = max(2 * abs(A * flmc.iloc[i] - carga_poste / 1.5), 0.05 * Ru)
        else:
            pretr = max(2 * abs(A * flmc.iloc[i] - carga_poste / 2.5), 0.05 * Ru)

        carac_postes.loc[
            carac_postes[postes_orden.name] == poste, col_fres
        ] = fres
        carac_postes.loc[i, col_fvert] = fver
        carac_postes.loc[i, col_trac] = trac
        carac_postes.loc[i, col_pret] = pretr
        carac_postes.loc[i, col_rot] = Ru        
        

    return carac_postes
