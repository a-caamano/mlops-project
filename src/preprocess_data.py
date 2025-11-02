import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt


class MedianOutlierHandler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.medians_ = X.median()
        self.bounds_ = {}
        for col in X.columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            self.bounds_[col] = (lower, upper)
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col in X_copy.columns:
            median = self.medians_[col]
            lower, upper = self.bounds_[col]
            X_copy[col] = X_copy[col].fillna(median)
            X_copy[col] = X_copy[col].apply(
                lambda x: median if x < lower or x > upper else x)
        return X_copy


class CorrelationRemover(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        corr_matrix = X.corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.to_drop_ = [column for column in upper_triangle.columns if any(
            upper_triangle[column] > 0.75)]
        return self

    def transform(self, X):
        return X.drop(columns=self.to_drop_, errors='ignore')


class InsurancePipeline:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.pipeline = None
        self.n_components_90 = None

    def preprocess(self):
        """Limpieza, filtrado de nulos, y PCA."""
        self.df = self.df.drop_duplicates()
        porcentaje_nulos = self.df.isnull().mean(axis=1) * 100
        self.df = self.df[porcentaje_nulos <= 5]

        X = self.df.iloc[:, :-1]
        y = self.df.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        self.pipeline = Pipeline([
            ('outlier_handler', MedianOutlierHandler()),
            ('corr_remover', CorrelationRemover()),
            ('scaler', StandardScaler()),
            ('pca', PCA())
        ])

        X_train_transformed = self.pipeline.fit_transform(X_train)
        cumulative_variance = np.cumsum(
            self.pipeline.named_steps['pca'].explained_variance_ratio_)
        self.n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1
        print(f"Componentes PCA ≥90% varianza: {self.n_components_90}")

        plt.plot(cumulative_variance * 100)
        plt.axhline(y=90, color='r', linestyle='--')
        plt.title('PCA Analysis')
        plt.show()

        self.pipeline.named_steps['pca'].n_components = self.n_components_90
        X_train_final = self.pipeline.fit_transform(X_train)
        X_test_final = self.pipeline.transform(X_test)

        return X_train_final, X_test_final, y_train, y_test
