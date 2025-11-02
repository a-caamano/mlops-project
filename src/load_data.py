import pandas as pd
import numpy as np


class InsuranceDataProcessor:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.data = None
        self.cleaned_data = None

    def load_data(self):
        """Carga el dataset desde la ruta especificada."""
        self.data = pd.read_csv(self.input_path, header=None)
        print("Datos cargados correctamente.")
        print(self.data.head(20))

    def clean_data(self):
        """Convierte columnas object a numéricas, maneja valores nulos y elimina la última columna."""
        print("Forma original:", self.data.shape)
        for col in self.data.columns:
            if self.data[col].dtype == 'object':
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
                self.data[col] = self.data[col].round().astype('Int64')

        self.data.fillna(value=np.nan, inplace=True)
        nulos_pre = self.data.isnull().mean() * 100
        print("Porcentaje de nulos por columna:")
        print(nulos_pre.sort_values(ascending=False))

        last_col = self.data.columns[-1]
        self.data.drop(columns=[last_col], inplace=True)
        print(f"Última columna eliminada: {last_col}")

    def validate_target_variable(self):
        """Valida la variable de salida (última columna) y elimina registros inválidos."""
        target_col = self.data.columns[-1]
        null_count = self.data[target_col].isnull().sum()
        print(f"Nulos en variable objetivo: {null_count}")
        self.data.dropna(subset=[target_col], inplace=True)
        self.data = self.data[self.data[target_col].isin([0, 1])]
        print("Validación completada: solo valores 0 y 1 en la variable objetivo.")

    def export_data(self):
        """Exporta el dataset limpio a la ruta especificada."""
        self.data.to_csv(self.output_path, index=False)
        print(f"Datos exportados correctamente a {self.output_path}.")

    def load_clean_data(self):
        """Guarda el dataset limpio en una variable."""
        self.cleaned_data = self.data.copy()
        print("Dataset limpio guardado en 'cleaned_data'.")
