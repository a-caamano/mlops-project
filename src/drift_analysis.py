# src/drift_analysis.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score

from alibi_detect.cd import KSDrift
from sklearn.impute import SimpleImputer
from preprocess_data import InsurancePipeline


# ===========================
#        CONFIGURACIÓN
# ===========================

PROCESSED_DATA = Path("data/processed")
MODEL_PATH = Path("models/insurance_model.pkl")

COLUMN = "42"
TARGET_COL = "85"

# Umbrales de monitoreo
PVAL_THRESHOLD = 0.05
MAX_F1_DROP = 0.03


# ===========================
#       FUNCIONES UTILES
# ===========================
def impute_data(df):
    """Imputa NaN usando medianas para columnas numéricas."""
    num_cols = df.select_dtypes(include=[np.number]).columns

    imputer = SimpleImputer(strategy="median")
    df[num_cols] = imputer.fit_transform(df[num_cols])

    return df


def clean_data(df):
    """
    Convierte columnas object a numéricas,
    sin redondear ni convertir a Int64.
    Mantiene distribución intacta.
    """
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.fillna(np.nan)
    return df


def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def evaluate(X, y, model):
    """Evalúa el modelo ya preprocesado"""
    y_pred = model.predict(X)
    return {
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1": f1_score(y, y_pred),
    }


def run_ks_drift(X_ref, X_test, p_val=0.05):
    """Corre KSDrift para una sola columna."""
    cd = KSDrift(X_ref, p_val=p_val)
    preds = cd.predict(X_test)
    return preds


def save_metrics(baseline, drifted, deltas, path="reports/drift_metrics.csv"):
    """Guarda métricas y diferencias."""
    df_out = pd.DataFrame([
        {"set": "baseline", **baseline},
        {"set": "drifted", **drifted},
        {"set": "delta", **deltas}
    ])

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(path, index=False)


def plot_kde(original, drifted, column, output_path=None):
    """Grafica KDE ajustando límites del eje X para evitar compresión."""
    plt.figure(figsize=(10, 5))

    sns.kdeplot(original[column], label="Original", fill=True, alpha=0.5)
    sns.kdeplot(drifted[column], label="Drifted", fill=True, alpha=0.5)

    # Rango visible
    low = min(original[column].quantile(0.01), drifted[column].quantile(0.01))
    high = max(original[column].quantile(0.99), drifted[column].quantile(0.99))
    plt.xlim(low, high)

    plt.title(f"Distribución KDE – Original vs Drift ({column})")
    plt.xlabel(column)
    plt.ylabel("Densidad")
    plt.legend()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)

    plt.show()


# ===========================
#       PROCESO PRINCIPAL
# ===========================

def main():

    print("\n=== CARGANDO Y LIMPIANDO DATASETS ===")
    df_ref = pd.read_csv(PROCESSED_DATA / "insurance_clean.csv")
    df_drift = pd.read_csv(PROCESSED_DATA / "insurance_clean_drift.csv")

    df_ref = clean_data(df_ref)
    df_drift = clean_data(df_drift)
    df_ref = impute_data(df_ref)
    df_drift = impute_data(df_drift)

    # Para KSDrift necesitamos (N,1)
    X_ref = df_ref[[COLUMN]].values
    X_test = df_drift[[COLUMN]].values

    print("\n=== CORRIENDO KSDrift ===")
    preds = run_ks_drift(X_ref, X_test, p_val=PVAL_THRESHOLD)

    ks_is_drift = preds["data"]["is_drift"]
    ks_p_val = preds["data"]["p_val"][0]

    print(f"¿Drift detectado?: {ks_is_drift == 1}")
    print(f"p-value: {ks_p_val}")

    pipeline_ref = InsurancePipeline(df_ref.copy())
    X_ref_final, _, y_ref, _ = pipeline_ref.preprocess()

    pipeline_drift = InsurancePipeline(df_drift.copy())
    X_drift_final, _, y_drift, _ = pipeline_drift.preprocess()

    print("\n=== EVALUANDO MODELO ===")
    model = load_model()

    baseline = evaluate(X_ref_final, y_ref, model)
    drifted = evaluate(X_drift_final, y_drift, model)

    deltas = {metric: drifted[metric] - baseline[metric]
              for metric in baseline}

    print("Métricas baseline:", baseline)
    print("Métricas drifted:", drifted)
    print("Deltas:", deltas)

    save_metrics(baseline, drifted, deltas)

    print("\n=== GENERANDO GRAFICO KDE ===")
    plot_kde(df_ref, df_drift, COLUMN, "reports/kde_plot.png")

    print("\n=== VERIFICANDO UMBRALES ===")
    alerts = []

    if ks_is_drift == 1 and ks_p_val < PVAL_THRESHOLD:
        alerts.append(f"KSDrift detectó drift (p-value={ks_p_val:.4f})")

    if -deltas["f1"] > MAX_F1_DROP:
        alerts.append(f"Caída en F1 mayor a {MAX_F1_DROP}: {deltas['f1']:.4f}")

    if alerts:
        print("\nALERTA DE DRIFT / PERFORMANCE ")
        for a in alerts:
            print("- " + a)

        print("\nAcción propuesta:")
        print("Revisar el pipeline de features y considerar reentrenar el modelo con los datos recientes.")
    else:
        print("\nNo hay alertas. Drift y desempeño dentro de tolerancia.")


if __name__ == "__main__":
    main()
