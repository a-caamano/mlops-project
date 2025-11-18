# src/data/simulate_drift.py

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


def simulate_drift(
    df: pd.DataFrame,
    drift_strength: float = 0.5,
    column: str = "42",
    n_components: int = 3,
    random_state: int = 42
):
    df_drifted = df.copy()

    # Imputar NaN
    df_drifted[column] = df[column].fillna(df[column].median())

    # ======================
    # 1) ESCALAR LA COLUMNA
    # ======================
    scaler = StandardScaler()
    X = scaler.fit_transform(df_drifted[[column]])

    # ======================
    # 2) AJUSTAR GMM EN ESCALA NORMAL
    # ======================
    gmm = GaussianMixture(
        n_components=n_components,
        random_state=random_state,
        covariance_type="full"
    )
    gmm.fit(X)

    # ======================
    # 3) DRIFT ADITIVO EN ESCALA NORMAL
    # ======================
    shift = drift_strength  # drift en std units
    gmm.means_ = gmm.means_ + shift

    from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky
    gmm.precisions_cholesky_ = _compute_precision_cholesky(
        gmm.covariances_, gmm.covariance_type
    )

    # Sample en espacio escalado
    X_drifted_scaled, _ = gmm.sample(len(df_drifted))

    X_drifted = scaler.inverse_transform(X_drifted_scaled)

    df_drifted[column] = X_drifted.reshape(-1)

    return df_drifted, [column]


def plot_drift(original_df, drifted_df, column="42"):
    plt.figure(figsize=(10, 5))

    sns.kdeplot(original_df[column], label="Original", fill=True,)
    sns.kdeplot(drifted_df[column], label="Drifted", fill=True,)

    # Ajustar el rango del eje X
    low = min(original_df[column].quantile(0.01),
              drifted_df[column].quantile(0.01))
    high = max(original_df[column].quantile(0.99),
               drifted_df[column].quantile(0.99))

    plt.xlim(low, high)

    plt.title(f"KDE Plot - Distribución original vs drift ({column})")
    plt.xlabel(column)
    plt.ylabel("Densidad")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
