import torch
import mlflow
from sklearn.metrics import f1_score

def validate_model(model, X_test, y_test, threshold=0.75):
    model.eval()

    X = torch.tensor(X_test.toarray(), dtype=torch.float32)
    preds = (model(X).detach().numpy() > 0.5).astype(int)

    score = f1_score(y_test, preds)
    mlflow.log_metric("f1_score", score)

    if score < threshold:
        raise RuntimeError("Validation failed: acceptance criteria not met")

    return score
