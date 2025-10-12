import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from src.data.load_coil2000 import load_data
from src.features.preprocess import preprocess

mlflow.set_experiment("coil2000_caravan_prediction")

def train():
    X, y = load_data()
    X = preprocess(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)

    with mlflow.start_run(run_name="gbm_classifier"):
        model = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05)
        model.fit(X_train, y_train)

        preds = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, preds)

        mlflow.log_param("n_estimators", 300)
        mlflow.log_param("learning_rate", 0.05)
        mlflow.log_metric("roc_auc", roc_auc)

        mlflow.sklearn.log_model(model, "model")
        print(f"ROC-AUC: {roc_auc:.3f}")

if __name__ == "__main__":
    train()