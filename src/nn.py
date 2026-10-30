import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

from src.data import build_preprocessor

# On macOS torch ships its own libomp, which conflicts with the Homebrew libomp
# used by LightGBM/XGBoost and crashes the process in multithreaded torch ops.
torch.set_num_threads(1)


class TitanicMLP(nn.Module):
    """Fully connected binary classifier that outputs one logit per sample."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (32, 16),
        dropout: float = 0.2,
    ):
        super().__init__()

        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze(1)


def make_loader(
    X: np.ndarray,
    y: pd.Series | np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Wrap preprocessed features and binary target into a DataLoader."""
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(np.asarray(y), dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
) -> float:
    """Run one training epoch and return the mean loss over samples."""
    model.train()
    total_loss = 0.0

    for xb, yb in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(xb)

    return total_loss / len(loader.dataset)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
) -> tuple[float, float]:
    """Return the mean loss and accuracy of the model on the loader."""
    model.eval()
    total_loss, correct = 0.0, 0

    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            total_loss += loss_fn(logits, yb).item() * len(xb)
            correct += ((logits > 0).float() == yb).sum().item()

    n_samples = len(loader.dataset)
    return total_loss / n_samples, correct / n_samples


def fit_mlp(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_val: np.ndarray,
    y_val: pd.Series,
    hidden_dims: tuple[int, ...] = (32, 16),
    dropout: float = 0.2,
    lr: float = 1e-3,
    batch_size: int = 32,
    n_epochs: int = 50,
    random_state: int = 42,
) -> tuple[TitanicMLP, dict[str, list[float]]]:
    """Train TitanicMLP on preprocessed data and return it with per-epoch metrics."""
    torch.manual_seed(random_state)

    train_loader = make_loader(X_train, y_train, batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, batch_size, shuffle=False)

    model = TitanicMLP(X_train.shape[1], hidden_dims, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    for _ in range(n_epochs):
        history["train_loss"].append(train_epoch(model, train_loader, optimizer, loss_fn))
        val_loss, val_acc = evaluate(model, val_loader, loss_fn)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

    return model, history


def cross_validate_mlp(
    X: pd.DataFrame,
    y: pd.Series,
    num_features: list[str],
    cat_features: list[str],
    cv: StratifiedKFold,
    **fit_params,
) -> np.ndarray:
    """Return final-epoch validation accuracy of TitanicMLP on each CV fold.

    The preprocessor is fitted on the training part of every fold to avoid leakage.
    """
    scores = []

    for train_idx, val_idx in cv.split(X, y):
        preprocessor = build_preprocessor(num_features, cat_features)
        X_train = preprocessor.fit_transform(X.iloc[train_idx])
        X_val = preprocessor.transform(X.iloc[val_idx])

        _, history = fit_mlp(
            X_train, y.iloc[train_idx],
            X_val, y.iloc[val_idx],
            **fit_params,
        )
        scores.append(history["val_acc"][-1])

    return np.array(scores)
