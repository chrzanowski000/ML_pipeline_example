import mlflow
import torch
import joblib
from src.ingestion import load_data
from src.validation import validate_schema
from src.preprocessing import preprocess
from src.model import TextClassifier
from src.train import train_model
from src.evaluate import validate_model

RAW_DATA_PATH = "data/raw/reports.csv"
ACCEPTANCE_THRESHOLD = 0.75

def run_pipeline():
    mlflow.set_experiment("PV_Text_Classifier_GxP")

    with mlflow.start_run():
        mlflow.log_param("data_path", RAW_DATA_PATH)
        mlflow.log_param("model_type", "PyTorch Linear")

        df = load_data(RAW_DATA_PATH)
        df = validate_schema(df)

        X_train, X_test, y_train, y_test, vectorizer = preprocess(df)

        model = TextClassifier(input_dim=X_train.shape[1])
        model = train_model(X_train, y_train, model)

        score = validate_model(model, X_test, y_test, ACCEPTANCE_THRESHOLD)

        # Freeze model
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        # Log to MLflow registry
        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            registered_model_name="pv_text_classifier"
        )

        # Log preprocessing artifact
        joblib.dump(vectorizer, "vectorizer.pkl")
        mlflow.log_artifact("vectorizer.pkl")

        print(f"Pipeline completed successfully. F1={score}")

if __name__ == "__main__":
    run_pipeline()
