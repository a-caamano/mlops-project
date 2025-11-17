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
        self.models = {}

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
        self.models[model_name] = model

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


if __name__ == "__main__":

    import pickle
    from load_data import InsuranceDataProcessor
    from preprocess_data import InsurancePipeline
    from train import ModelTrainerEvaluator

    # === 1. Cargar y limpiar datos ===
    processor = InsuranceDataProcessor(
        input_path='data/interim/insurance_company_modified.csv',
        output_path='data/processed/insurance_clean.csv'
    )
    processor.load_data()
    processor.clean_data()
    processor.validate_target_variable()
    processor.export_data()
    processor.load_clean_data()

    # === 1.1 Opcional: Simular Drift ===
    from simulate_drift import simulate_drift

    simulate = False   # Cambia a False para correr sin drift

    if simulate:
        print("\n Generando dataset con DRIFT...")
        print(processor.cleaned_data)
        processor.cleaned_data.columns = processor.cleaned_data.columns.astype(
            str)
        df_drift, drift_cols = simulate_drift(
            processor.cleaned_data, drift_strength=0.9)

        print("Columnas afectadas por drift:", drift_cols)

        df_drift.to_csv(
            "data/processed/insurance_clean_drift.csv", index=False)
        df_to_use = df_drift
    else:
        df_to_use = processor.cleaned_data

    # === 2. Preprocesamiento y reducción de dimensionalidad ===
    pipeline = InsurancePipeline(df_to_use)
    X_train_final, X_test_final, y_train, y_test = pipeline.preprocess()

    # === 3. Entrenar y evaluar modelos ===
    trainer = ModelTrainerEvaluator(
        X_train_final, X_test_final, y_train, y_test)
    trainer.show_class_distribution()
    trainer.correlation_with_target()
    trainer.run_all_models()

    # === 4. Mostrar resultados finales ===
    print("\n=== Resumen Final de Resultados ===")
    for model_name, metrics in trainer.results.items():
        print(f"\nModelo: {model_name}")
        print(f"Precisión: {metrics['weighted avg']['precision']:.3f}")
        print(f"Recall: {metrics['weighted avg']['recall']:.3f}")
        print(f"F1-Score: {metrics['weighted avg']['f1-score']:.3f}")

    # Save the model to a file
   # Guardar el modelo entrenado
    model = next(iter(trainer.models.values()))

    if simulate:
        # Guardar como modelo con drift
        with open("models/insurance_model_drift.pkl", "wb") as f:
            pickle.dump(model, f)
        print("Model saved as 'insurance_model_drift.pkl'")
    else:
        # Guardar como modelo original
        with open("models/insurance_model.pkl", "wb") as f:
            pickle.dump(model, f)
        print("Model saved as 'insurance_model.pkl'")
