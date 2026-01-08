import mlflow
import torch
import joblib
from src.model import TextClassifier

MODEL_NAME = "pv_text_classifier"
OUTPUT_DIR = "artifacts/release"

def export_production_model():
    # 1. Load APPROVED model from MLflow registry
    model = mlflow.pytorch.load_model(
        f"models:/{MODEL_NAME}/Production"
    )

    # 2. Freeze model (safety check)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # 3. Export frozen weights
    torch.save(
        model.state_dict(),
        f"{OUTPUT_DIR}/model_state.pt"
    )

    # 4. Export preprocessing artifact
    vectorizer = joblib.load("vectorizer.pkl")
    joblib.dump(
        vectorizer,
        f"{OUTPUT_DIR}/vectorizer.pkl"
    )

    print("Production model exported as frozen artifacts.")

if __name__ == "__main__":
    export_production_model()
