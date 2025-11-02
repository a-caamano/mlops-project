from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


class Evaluator:
    @staticmethod
    def show_results(y_test, y_pred, model_name):
        print(f"\n=== Reporte de {model_name} ===")
        report = classification_report(y_test, y_pred, zero_division=0)
        print(report)

        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, index=['Real 0', 'Real 1'], columns=[
                             'Pred 0', 'Pred 1'])

        plt.figure(figsize=(6, 4))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Matriz de Confusión - {model_name}')
        plt.xlabel("Predicción")
        plt.ylabel("Valor Real")
        plt.tight_layout()
        plt.show()
