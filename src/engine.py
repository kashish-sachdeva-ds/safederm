"""Training and evaluation loops for SafeDerm models.

Kept model-agnostic -- both 06_baseline_model.ipynb and
07_champion_model.ipynb import the same train_one_epoch/evaluate
functions, so results are comparable: any difference in metrics reflects
the model, not a difference in how it was trained or measured.
"""

import pandas as pd
import torch

from src.dataset import CLASS_TO_IDX
from src.labels import ALL_CLASSES


def compute_class_weights(split_csv_path) -> torch.Tensor:
    """Inverse-frequency class weights, ordered to match CLASS_TO_IDX.

    Counters the ~67% nv imbalance confirmed in 03_eda.ipynb -- without
    this, a model can score high accuracy by just always predicting nv.
    """
    df = pd.read_csv(split_csv_path)
    counts = df["dx"].value_counts()

    weights = torch.zeros(len(ALL_CLASSES))
    total = len(df)
    for cls, idx in CLASS_TO_IDX.items():
        class_count = counts.get(cls, 0)
        weights[idx] = total / (len(ALL_CLASSES) * class_count) if class_count > 0 else 0.0

    return weights


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    """One pass over the training data. Returns average loss."""
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Runs the model over a loader with no gradient tracking.

    Returns (avg_loss, accuracy, all_predictions, all_labels). Predictions
    and labels are returned raw so the caller can build a confusion matrix
    or compute a custom metric (like malignant recall) afterward -- this
    function stays reusable rather than baking in one specific report.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = running_loss / len(loader.dataset)
    accuracy = correct / len(loader.dataset)

    return avg_loss, accuracy, all_preds, all_labels
