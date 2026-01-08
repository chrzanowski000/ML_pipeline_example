import torch
import torch.nn as nn
import torch.optim as optim
import mlflow

def train_model(X_train, y_train, model):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()

    X = torch.tensor(X_train.toarray(), dtype=torch.float32)
    y = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)

    for epoch in range(10):
        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        mlflow.log_metric("train_loss", loss.item(), step=epoch)

    return model
