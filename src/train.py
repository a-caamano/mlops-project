from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import pandas as pd
from evaluate import Evaluator


class ModelTrainerEvaluator:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.results = {}

    def show_class_distribution(self):
        dist = self.y_train.value_counts(normalize=True).mul(
            100).round(1).astype(str) + '%'
        print(f"Distribución de clases:\n{dist}")

    def correlation_with_target(self):
        df = pd.concat([pd.DataFrame(self.X_train), pd.Series(
            self.y_train, name='target')], axis=1)
        corr = abs(df.corr()['target'].sort_values(ascending=False))
        print("Correlación con la variable dependiente:\n", corr)

    def apply_smote(self):
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(
            self.X_train, self.y_train)
        return X_resampled, y_resampled

    def train_and_evaluate(self, model_name, model):
        """Entrena el modelo, genera predicciones y usa Evaluator para mostrar resultados."""
        X_resampled, y_resampled = self.apply_smote()
        model.fit(X_resampled, y_resampled)
        y_pred = model.predict(self.X_test)

        # Evaluación visual y textual
        Evaluator.show_results(self.y_test, y_pred, model_name)

        # Guardar resultados en memoria
        report = classification_report(
            self.y_test, y_pred, zero_division=0, output_dict=True)
        self.results[model_name] = report

    def run_all_models(self):
        """Ejecuta todos los modelos definidos."""
        models = [
            LogisticRegression(max_iter=1000, class_weight='balanced'),
            DecisionTreeClassifier(
                max_depth=10, random_state=42, class_weight='balanced'),
            XGBClassifier(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.1,
                colsample_bytree=1,
                subsample=0.8,
                random_state=42,
                eval_metric='aucpr'
            )
        ]

        for model in models:
            self.train_and_evaluate(model.__class__.__name__, model)
