from load_data import InsuranceDataProcessor
from preprocess_data import InsurancePipeline
from train import ModelTrainerEvaluator


def main():
    # === 1. Cargar y limpiar datos ===
    processor = InsuranceDataProcessor(
        input_path='data/insurance_company_modified.csv',
        output_path='data/insurance_clean.csv'
    )
    processor.load_data()
    processor.clean_data()
    processor.validate_target_variable()
    processor.export_data()
    processor.load_clean_data()

    # === 2. Preprocesamiento y reducción de dimensionalidad ===
    pipeline = InsurancePipeline(processor.cleaned_data)
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


if __name__ == "__main__":
    main()
