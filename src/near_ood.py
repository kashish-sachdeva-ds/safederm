"""SafeDerm — Gateway 2: Near-OOD Detection (k-NN on model embeddings)

Complements src/gateway.py (BiomedCLIP, far-OOD). Where Gateway 1 asks
"is this skin at all", Gateway 2 asks "does this look like the kind of
skin image our own model was actually trained on" — catches things
that fool a generic vision-language model (wood grain, glitter/confetti
textures, wrong lighting/zoom) but clearly aren't in-distribution for
a dermoscopy classifier.

No new model is trained. We reuse the already-trained clinical model
(baseline now, champion later — swap the checkpoint, nothing else
changes) and read out its penultimate-layer embedding via a forward
hook on `model.avgpool`. That assumes a standard torchvision ResNet
(avgpool -> flatten -> fc) — if build_baseline_model() wraps things
differently, adjust `_get_embedding` accordingly.

Method: k-NN distance in embedding space (Sun et al., "Out-of-Distribution
Detection with Deep Nearest Neighbors", ICML 2022) — non-parametric, no
Gaussian assumption like Mahalanobis, and consistently benchmarks better.
For HAM10000-scale data (~10K train images, 2048-dim ResNet-50 features)
a plain in-memory tensor + torch.cdist is enough — no FAISS needed.
"""

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

FEATURE_BANK_FILENAME = "feature_bank.pt"
DEFAULT_K = 5
# Starting point only — tune against real in-distribution vs known-OOD
# samples (wood grain, glitter, etc.) the same way src/gateway.py's
# threshold gets tuned in notebooks/gateway_threshold_tuning.ipynb.
DEFAULT_DISTANCE_THRESHOLD = 0.90  # fill in after running the notebook


@dataclass
class FeatureBank:
    embeddings: torch.Tensor  # (N, D), L2-normalized
    device: str


def _get_embedding(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Pull the penultimate-layer (pre-fc) embedding via a forward hook.
    Assumes a torchvision-style ResNet with a `model.avgpool` module."""
    captured = {}

    def hook(_module, _input, output):
        captured["emb"] = output.flatten(1)

    handle = model.avgpool.register_forward_hook(hook)
    with torch.no_grad():
        model(x)
    handle.remove()

    emb = captured["emb"]
    return emb / emb.norm(dim=-1, keepdim=True)  # L2-normalize


@torch.no_grad()
def build_feature_bank(
    model: nn.Module,
    train_loader: DataLoader,
    device: str,
) -> FeatureBank:
    """Run every training image through the model once and collect its
    embedding. This is the 'index' that new images get compared against
    at inference time — not training, just a forward pass over data
    we already have."""
    model.eval().to(device)
    all_embeddings = []

    for images, _labels in train_loader:
        images = images.to(device)
        emb = _get_embedding(model, images)
        all_embeddings.append(emb.cpu())

    embeddings = torch.cat(all_embeddings, dim=0).to(device)
    return FeatureBank(embeddings=embeddings, device=device)


def save_feature_bank(bank: FeatureBank, models_dir: Path) -> Path:
    path = models_dir / FEATURE_BANK_FILENAME
    torch.save({"embeddings": bank.embeddings}, path)
    return path


def load_feature_bank(models_dir: Path, device: str) -> FeatureBank:
    path = models_dir / FEATURE_BANK_FILENAME
    data = torch.load(path, map_location=device)
    return FeatureBank(embeddings=data["embeddings"].to(device), device=device)


@torch.no_grad()
def knn_distance(
    image_tensor: torch.Tensor,
    model: nn.Module,
    bank: FeatureBank,
    k: int = DEFAULT_K,
) -> float:
    """Distance from one preprocessed image (already batched, shape (1,C,H,W))
    to its k-th nearest neighbor in the training feature bank. Higher = more
    likely near-OOD."""
    model.eval().to(bank.device)
    image_tensor = image_tensor.to(bank.device)

    query_emb = _get_embedding(model, image_tensor)  # (1, D)
    dists = torch.cdist(query_emb, bank.embeddings.to(query_emb.device))  # (1, N)
    kth_dist = dists.topk(k, largest=False).values[0, -1]
    return kth_dist.item()


def verify_in_distribution(
    image_tensor: torch.Tensor,
    model: nn.Module,
    bank: FeatureBank,
    threshold: float,
    k: int = DEFAULT_K,
) -> tuple[bool, float]:
    """Returns (is_in_distribution, distance). Mirrors
    src/gateway.py's verify_image_is_skin() return shape so the API
    route can call both gates the same way."""
    distance = knn_distance(image_tensor, model, bank, k=k)
    return distance <= threshold, distance
