"""Training and evaluation loops for SafeDerm models.

Kept model-agnostic -- both 06_baseline_model.ipynb and
07_champion_model.ipynb import the same functions, so a difference in
results reflects the architecture, not a difference in how each was
trained or measured (same principle src/model.py already states).

Changes vs. the original version (see
docs/decisions/ADR-006-training-recipe-and-champion-model.md):

- set_seed(): training runs are now reproducible, not just the data split.
- get_param_groups() / freeze_backbone() / unfreeze_all(): support
  discriminative-LR fine-tuning instead of updating every parameter at one
  LR, which is what produced the overfitting visible in 06's original
  training curve (val_loss bottoms out at epoch 5 and never improves again
  through epoch 10, while train_loss keeps falling).
- EarlyStopper: a run stops itself once it plateaus, instead of always
  spending a fixed epoch budget.
- malignant_recall(): pulled out of 06's notebook (cell 16) into a shared,
  vectorized function, so 07 and 08 call this instead of re-copying the
  calculation and risking it drifting.
- fit(): orchestrates all of the above so the train/validate/checkpoint
  loop itself isn't duplicated between notebooks.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.dataset import CLASS_TO_IDX
from src.labels import ALL_CLASSES, MALIGNANT_CLASSES

logger = logging.getLogger(__name__)

MALIGNANT_IDXS = {CLASS_TO_IDX[cls] for cls in MALIGNANT_CLASSES}


def set_seed(seed: int = 42) -> None:
    """Seeds python/numpy/torch (CPU + CUDA) so a training run is reproducible.

    ADR-003's random_state=42 only makes the *data split* reproducible.
    Nothing previously seeded the *training run* itself -- weight init,
    dropout, augmentation, and DataLoader shuffling were all left to
    whatever RNG state happened to exist.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_class_weights(split_csv_path) -> torch.Tensor:
    """Inverse-frequency class weights, ordered to match CLASS_TO_IDX.

    Counters the ~67%-nv imbalance documented in 03_eda.ipynb -- without
    this, a model can reach high accuracy by mostly predicting nv.

    Raises instead of silently zero-weighting a class with no samples,
    matching the hard-assertion style already used in 01/02 rather than
    letting a class's loss contribution silently vanish to 0.0.
    """
    df = pd.read_csv(split_csv_path)
    counts = df["dx"].value_counts()
    total = len(df)

    weights = torch.zeros(len(ALL_CLASSES))
    for cls, idx in CLASS_TO_IDX.items():
        class_count = int(counts.get(cls, 0))
        if class_count == 0:
            raise ValueError(
                f"Class '{cls}' has zero samples in {split_csv_path} -- "
                "cannot compute an inverse-frequency weight for it."
            )
        weights[idx] = total / (len(ALL_CLASSES) * class_count)

    return weights


def get_param_groups(
    model: nn.Module,
    head_lr: float,
    backbone_lr: float,
    head_attr: str = "fc",
) -> list[dict]:
    """Splits model parameters into (backbone, head) groups with different LRs.

    Fine-tuning every parameter at one LR is what overfit 06's baseline.
    A lower LR on the pretrained backbone (already close to a good
    solution) and a higher LR on the freshly-initialized head is the
    standard fix: the backbone adapts gently, the head learns fast.

    `head_attr` names the submodule to treat as "the fresh part" -- "fc"
    for build_baseline_model(), "transformer_head" for build_champion_model().
    """
    head = getattr(model, head_attr)
    head_param_ids = {id(p) for p in head.parameters()}
    backbone_params = [p for p in model.parameters() if id(p) not in head_param_ids]
    head_params = list(head.parameters())

    return [
        {"params": backbone_params, "lr": backbone_lr, "name": "backbone"},
        {"params": head_params, "lr": head_lr, "name": "head"},
    ]


def freeze_backbone(model: nn.Module, head_attr: str = "fc") -> None:
    """Freezes every parameter except `head_attr`.

    Used for a short warmup phase so the randomly-initialized head
    doesn't push large, noisy gradients back through the pretrained
    backbone before the head has learned anything sensible.
    """
    head = getattr(model, head_attr)
    head_param_ids = {id(p) for p in head.parameters()}
    for p in model.parameters():
        p.requires_grad = id(p) in head_param_ids


def unfreeze_all(model: nn.Module) -> None:
    """Re-enables gradients on every parameter. Call after the warmup phase."""
    for p in model.parameters():
        p.requires_grad = True


class EarlyStopper:
    """Stops training when `metric` hasn't improved for `patience` epochs.

    mode="min" for a loss, mode="max" for recall/accuracy. Exists so a run
    stops itself once it plateaus, instead of always running a fixed
    NUM_EPOCHS and eyeballing the printed table afterward -- which is how
    06 originally spent epochs 6-10 with val_loss already worse than
    epoch 5's.
    """

    def __init__(self, patience: int = 4, mode: str = "min", min_delta: float = 0.0):
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best = float("inf") if mode == "min" else float("-inf")
        self.best_epoch = 0
        self.num_bad_epochs = 0

    def step(self, value: float, epoch: int) -> bool:
        """Records `value` for `epoch`. Returns True if training should stop."""
        improved = (
            value < self.best - self.min_delta
            if self.mode == "min"
            else value > self.best + self.min_delta
        )
        if improved:
            self.best = value
            self.best_epoch = epoch
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        return self.num_bad_epochs >= self.patience


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    """One pass over the training data. Returns the average training loss.

    Unchanged from the original -- still the right shape for a single
    epoch step; the fixes live in what calls it (see fit() below).
    """
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
    """Runs the model over `loader` with no gradient tracking.

    Returns (avg_loss, accuracy, all_predictions, all_labels). Predictions
    and labels are returned raw so the caller can build a confusion matrix
    or a custom metric (e.g. malignant_recall) without re-running inference.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    all_preds: list[int] = []
    all_labels: list[int] = []

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


def malignant_recall(preds, labels, malignant_idxs: Optional[set] = None) -> dict:
    """Recall on the malignant risk group -- the project's actual safety
    metric (ADR-004 / ADR-005), as opposed to overall accuracy.

    Vectorized with numpy instead of the python zip/sum loops that were
    duplicated inline in 06_baseline_model.ipynb, cell 16. Pulled out here
    so 07 and 08 call this instead of re-copying it -- the same class of
    bug as 03_eda.ipynb hardcoding MALIGNANT_CLASSES instead of importing
    it from src.labels.
    """
    if malignant_idxs is None:
        malignant_idxs = MALIGNANT_IDXS

    labels_arr = np.asarray(labels)
    preds_arr = np.asarray(preds)
    malignant_list = list(malignant_idxs)

    is_malignant = np.isin(labels_arr, malignant_list)
    predicted_malignant = np.isin(preds_arr, malignant_list)

    tp = int(np.sum(is_malignant & predicted_malignant))
    fn = int(np.sum(is_malignant & ~predicted_malignant))
    total_malignant = tp + fn

    return {
        "malignant_total": total_malignant,
        "malignant_recall": (tp / total_malignant) if total_malignant > 0 else 0.0,
        "malignant_missed": fn,
    }


def fit(
    model: nn.Module,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device,
    checkpoint_path: Path,
    num_epochs: int = 30,
    scheduler=None,
    early_stopper: Optional[EarlyStopper] = None,
    selection_metric: str = "val_loss",
) -> dict:
    """Full train/validate/checkpoint/early-stop loop, shared by 06 and 07
    so both models are trained under an identical, fair procedure -- any
    difference in results reflects the architecture, not the recipe.

    selection_metric: "val_loss" (weighted CE, ADR-005's original choice)
    or "malignant_recall" (the project's real safety target). Both
    checkpoint selection and early stopping use whichever is passed.
    """
    if early_stopper is None:
        mode = "min" if selection_metric == "val_loss" else "max"
        early_stopper = EarlyStopper(patience=4, mode=mode)

    history = {
        "train_loss": [], "val_loss": [], "val_accuracy": [],
        "malignant_recall": [], "lr": [],
    }
    best_value = float("inf") if early_stopper.mode == "min" else float("-inf")
    final_epoch = 0

    for epoch in range(1, num_epochs + 1):
        final_epoch = epoch
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
        recall_stats = malignant_recall(val_preds, val_labels)

        current_lr = optimizer.param_groups[-1]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["malignant_recall"].append(recall_stats["malignant_recall"])
        history["lr"].append(current_lr)

        tracked = val_loss if selection_metric == "val_loss" else recall_stats["malignant_recall"]

        logger.info(
            "epoch %d/%d - train_loss=%.4f val_loss=%.4f val_acc=%.4f "
            "malignant_recall=%.4f lr=%.2e",
            epoch, num_epochs, train_loss, val_loss, val_acc,
            recall_stats["malignant_recall"], current_lr,
        )

        is_best = tracked < best_value if early_stopper.mode == "min" else tracked > best_value
        if is_best:
            best_value = tracked
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(
                "  new best (%s=%.4f) - checkpoint saved to %s",
                selection_metric, tracked, checkpoint_path,
            )

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(tracked)
            else:
                scheduler.step()

        if early_stopper.step(tracked, epoch):
            logger.info(
                "  early stopping at epoch %d - no improvement in %d epochs "
                "(best %s=%.4f at epoch %d)",
                epoch, early_stopper.patience, selection_metric,
                early_stopper.best, early_stopper.best_epoch,
            )
            break

    history["stopped_epoch"] = final_epoch
    history["best_epoch"] = early_stopper.best_epoch
    return history
