from fastapi import FastAPI
import torch
import joblib
from src.model import TextClassifier

app = FastAPI()

vectorizer = joblib.load("artifacts/release/vectorizer.pkl")

model = TextClassifier(input_dim=len(vectorizer.vocabulary_))
model.load_state_dict(torch.load("artifacts/release/model_state.pt"))
model.eval()

@app.post("/predict")
def predict(text: str):
    X = vectorizer.transform([text])
    X = torch.tensor(X.toarray(), dtype=torch.float32)
    pred = model(X).item()
    return {"prediction": int(pred > 0.5)}
