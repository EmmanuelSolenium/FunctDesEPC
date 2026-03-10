import pandas as pd
import os

# Ruta base: carpeta "data" en el mismo repositorio donde se ejecuta el codigo
base_dir = os.path.dirname(os.path.abspath(__file__))  # Directorio del script actual
data_dir = os.path.join(base_dir, "data")

# Rutas de los archivos CSV
path_3phase = os.path.join(data_dir, "3_phase_reenergization.csv")
path_reener = os.path.join(data_dir, "reenergization.csv")

# Nombres de columnas
column_names = ["Ue2", "Up2/Ue2"]

# Importar los CSV como DataFrames (separador ; , decimal , , sin encabezado)
df_3phase = pd.read_csv(path_3phase, sep=";", decimal=",", header=None, names=column_names)
df_reener = pd.read_csv(path_reener, sep=";", decimal=",", header=None, names=column_names)

up2_3phase = df_3phase
up2_reener = df_reener 

""" # Verificar que se cargaron correctamente
print("=== 3_phase_reenergization.csv ===")
print(f"Shape: {df_3phase.shape}")
print(df_3phase.head())

print("\n=== reenergization.csv ===")
print(f"Shape: {df_reener.shape}")
print(df_reener.head())

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

# Graficar cada DataFrame
plt.plot(df_3phase["Ue2"], df_3phase["Up2/Ue2"], label="3-Phase Reenergization")
plt.plot(df_reener["Ue2"], df_reener["Up2/Ue2"], label="Reenergization")

# Etiquetas y formato
plt.xlabel("Ue2")
plt.ylabel("Up2/Ue2")
plt.title("Comparación de curvas de reenergización")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show() """
