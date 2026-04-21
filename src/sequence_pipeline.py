import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class SequenceDrawDataset(Dataset):
    def __init__(self, draws: np.ndarray, window: int = 10):
        self.window = window
        self.inputs = []
        self.targets = []

        for i in range(window, len(draws)):
            self.inputs.append(draws[i - window : i])
            self.targets.append(draws[i])

        self.inputs = np.stack(self.inputs).astype(np.float32)
        self.targets = np.stack(self.targets).astype(np.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class TransformerDrawPredictor(nn.Module):
    def __init__(self, window: int = 10, d_model: int = 128, nhead: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.window = window
        self.input_projection = nn.Linear(60, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(window, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.output_layer = nn.Linear(d_model, 60)

    def forward(self, x):
        x = self.input_projection(x)
        x = x + self.pos_embedding.unsqueeze(0)
        x = self.encoder(x)
        x = x.mean(dim=1)
        x = self.norm(x)
        return self.output_layer(x)


def one_hot_draws(df: pd.DataFrame) -> np.ndarray:
    dezena_cols = [f"dezena_{i}" for i in range(1, 7)]
    draws = df[dezena_cols].values.astype(int)
    one_hot = np.zeros((len(draws), 60), dtype=np.float32)
    for i, row in enumerate(draws):
        one_hot[i, row - 1] = 1.0
    return one_hot


def train_sequence_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    window: int = 10,
    epochs: int = 15,
    lr: float = 1e-3,
    device: Optional[str] = None,
) -> tuple[nn.Module, dict]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerDrawPredictor(window=window).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_batch.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * x_batch.size(0)

        val_loss /= len(val_loader.dataset)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

    return model, history


def build_sequence_datasets(
    df: pd.DataFrame,
    window: int = 10,
    cutoff_date: str = "2021-01-01",
    test_batch_size: int = 32,
    train_batch_size: int = 32,
) -> tuple[DataLoader, DataLoader, pd.DataFrame]:
    df = df.copy()
    df["data_parsed"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
    df = df.sort_values("concurso").reset_index(drop=True)

    cutoff = pd.to_datetime(cutoff_date)
    train_df = df[df["data_parsed"] < cutoff].reset_index(drop=True)
    test_df = df[df["data_parsed"] >= cutoff].reset_index(drop=True)

    train_draws = one_hot_draws(train_df)
    test_draws = one_hot_draws(pd.concat([train_df.tail(window), test_df], ignore_index=True))

    train_dataset = SequenceDrawDataset(train_draws, window=window)
    test_dataset = SequenceDrawDataset(test_draws, window=window)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    return train_loader, test_loader, test_df


def get_top_predictions(logits: np.ndarray, top_k: int = 6) -> list[int]:
    probs = 1 / (1 + np.exp(-logits))
    top_indices = np.argsort(probs)[::-1][:top_k]
    return (top_indices + 1).tolist()


def evaluate_sequence_predictions(model: nn.Module, loader: DataLoader, device: Optional[str] = None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    metrics = {"top6_accuracy": 0.0, "samples": 0}
    hits_total = 0
    total_numbers = 0

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            targets = y_batch.cpu().numpy()

            for pred_prob, true_vec in zip(probs, targets):
                pred_top6 = set(get_top_predictions(pred_prob, top_k=6))
                true_top6 = set(np.nonzero(true_vec)[0] + 1)
                hits_total += len(pred_top6 & true_top6)
                total_numbers += 6
                metrics["samples"] += 1

    metrics["hit_rate"] = hits_total / total_numbers if total_numbers > 0 else 0.0
    metrics["avg_hits_per_sample"] = hits_total / metrics["samples"] if metrics["samples"] > 0 else 0.0
    return metrics


def predict_sequence_draws(
    model: nn.Module,
    loader: DataLoader,
    device: Optional[str] = None,
) -> list[dict]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    results = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            truths = y_batch.cpu().numpy()

            for prob_vec, truth_vec in zip(probs, truths):
                top_predicted = get_top_predictions(prob_vec, top_k=6)
                results.append({
                    "probabilities": prob_vec.tolist(),
                    "predicted": top_predicted,
                    "truth": (np.nonzero(truth_vec)[0] + 1).tolist(),
                })

    return results


def run_sequence_analysis(
    df: pd.DataFrame,
    window: int = 10,
    epochs: int = 15,
    batch_size: int = 32,
    cutoff_date: str = "2021-01-01",
) -> dict:
    train_loader, test_loader, test_df = build_sequence_datasets(
        df,
        window=window,
        cutoff_date=cutoff_date,
        train_batch_size=batch_size,
        test_batch_size=batch_size,
    )

    model, history = train_sequence_model(train_loader, test_loader, window=window, epochs=epochs)
    metrics = evaluate_sequence_predictions(model, test_loader)
    predictions = predict_sequence_draws(model, test_loader)

    return {
        "model": model,
        "history": history,
        "metrics": metrics,
        "predictions": predictions,
        "test_dataframe": test_df,
    }
