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

""" print(TABLA_AFINIA_CAPACIDAD_MECANICA_DF) """

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

print(TABLA_AFINIA_CAP_VERTICAL_DF)

