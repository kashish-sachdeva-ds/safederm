"""SafeDerm — Gateway 2: Near-OOD Detection (k-NN on model embeddings)

Complements src/gateway.py (BiomedCLIP, far-OOD). Where Gateway 1 asks
"is this skin at all", Gateway 2 asks "does this look like the kind of
skin image our own model was actually trained on" — catches things
that fool a generic vision-language model (wood grain, glitter/confetti
textures, wrong lighting/zoom) but clearly aren't in-distribution for
a dermoscopy classifier.

No new model is trained. We reuse whatever clinical model is currently
deployed and read out an embedding from it — but HOW to read that
embedding is architecture-specific and does NOT survive a checkpoint
swap for free. The baseline (plain torchvision ResNet-50) exposes a
clean `avgpool` module. The champion model feeds the raw feature map
straight into a transformer head with no pooling step, so `avgpool`
does not exist on it — calling this with an unadapted extractor raises
AttributeError on the champion model. Confirmed directly (see
SAFEDERM_MODEL_VARIANT=champion test, Aug 2026); an earlier version of
this docstring claimed "swap the checkpoint, nothing else changes",
which was wrong and has been removed.

Fix: `_get_embedding` no longer hardcodes `avgpool`. Every caller must
pass an `embedding_fn(model, x) -> Tensor` appropriate to that model's
architecture. `resnet_avgpool_embedding_fn` below covers the baseline.
For the champion model, someone who knows its transformer head (mean-
pool over the sequence output? CLS token? a dedicated projection head?)
needs to write the equivalent extractor and pass it in explicitly — do
not assume one architecture's extractor works on another's model.

Method: k-NN distance in embedding space (Sun et al., "Out-of-Distribution
Detection with Deep Nearest Neighbors", ICML 2022) — non-parametric, no
Gaussian assumption like Mahalanobis, and consistently benchmarks better.
For HAM10000-scale data (~10K train images, 2048-dim ResNet-50 features)
a plain in-memory tensor + torch.cdist is enough — no FAISS needed.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

FEATURE_BANK_FILENAME = "feature_bank.pt"
DEFAULT_K = 5
# Starting point only — tune against real in-distribution vs known-OOD
# samples (wood grain, glitter, etc.) the same way src/gateway.py's
# threshold gets tuned in notebooks/09_gateway_threshold_tuning.ipynb.
# NOTE: a threshold tuned on the baseline's embedding space does not
# transfer to the champion model either — its embeddings live in a
# different space entirely. Re-tune per architecture, not just per checkpoint.
DEFAULT_DISTANCE_THRESHOLD = 1.20
DEFAULT_DISTANCE_THRESHOLD_CHAMPION = 0.50

EmbeddingFn = Callable[[nn.Module, torch.Tensor], torch.Tensor]


@dataclass
class FeatureBank:
    embeddings: torch.Tensor  # (N, D), L2-normalized
    device: str


def resnet_avgpool_embedding_fn(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Embedding extractor for the BASELINE model only — a standard
    torchvision ResNet-50 (avgpool -> flatten -> fc). Pulls the
    pre-fc feature via a forward hook on `model.avgpool`.

    Do NOT pass this for the champion model — it does not have an
    `avgpool` module and this will raise AttributeError."""
    captured = {}

    def hook(_module, _input, output):
        captured["emb"] = output.flatten(1)

    handle = getattr(model, "avgpool").register_forward_hook(hook)
    with torch.no_grad():
        model(x)
    handle.remove()

    emb = captured["emb"]
    return emb / emb.norm(dim=-1, keepdim=True)  # L2-normalize


def _get_embedding(model: nn.Module, x: torch.Tensor, embedding_fn: EmbeddingFn) -> torch.Tensor:
    """Thin dispatch to the caller-supplied, architecture-specific extractor.
    Deliberately has no default — a silent default is exactly what caused
    the champion-model crash this function is now guarding against."""
    return embedding_fn(model, x)


def champion_embedding_fn(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """NOT IMPLEMENTED — placeholder so the missing piece is visible in
    code rather than buried in a comment.

    Champion feeds the raw CNN feature map into a transformer head with
    no pooling step, so there is no single `avgpool`-equivalent to hook.
    Whoever owns ADR-002 (the champion architecture) needs to decide what
    counts as "the embedding" here — candidates: mean-pool the transformer's
    output sequence, take a CLS-token-style summary vector, or a dedicated
    pre-classification projection layer. Whichever it is, wire it up the
    same way resnet_avgpool_embedding_fn does above, then re-run
    notebooks/near_ood_feature_bank.ipynb against the champion checkpoint
    to rebuild the feature bank and re-tune the distance threshold —
    neither carries over from the baseline."""
    embeddings = []
    def hook(module, input, output):
        embeddings.append(output)
    
    head = getattr(model, "transformer_head")
    handle = getattr(head, "norm").register_forward_hook(hook)
    try:
        model(x)
    finally:
        handle.remove()
        
    return embeddings[0]


@torch.no_grad()
def build_feature_bank(
    model: nn.Module,
    train_loader: DataLoader,
    device: str,
    embedding_fn: EmbeddingFn,
) -> FeatureBank:
    """Run every training image through the model once and collect its
    embedding. This is the 'index' that new images get compared against
    at inference time — not training, just a forward pass over data
    we already have.

    embedding_fn is required, not defaulted — pass
    resnet_avgpool_embedding_fn for the baseline, or a champion-specific
    extractor once one exists. A feature bank built with the wrong
    extractor for the wrong model will not error loudly; it will just
    produce meaningless distances, which is worse than a crash."""
    model.eval().to(device)
    all_embeddings = []

    for images, _labels in train_loader:
        images = images.to(device)
        emb = _get_embedding(model, images, embedding_fn)
        all_embeddings.append(emb.cpu())

    embeddings = torch.cat(all_embeddings, dim=0)
    return FeatureBank(embeddings=embeddings, device=device)


def save_feature_bank(bank: FeatureBank, models_dir: Path) -> Path:
    path = models_dir / FEATURE_BANK_FILENAME
    torch.save({"embeddings": bank.embeddings}, path)
    return path


def load_feature_bank_if_exists(models_dir: Path, device: str, variant: str = "baseline") -> FeatureBank | None:
    path = models_dir / f"feature_bank_{variant}.pt"
    # For backward compatibility, if variant is baseline we also check the un-suffixed name
    if not path.exists() and variant == "baseline":
        path = models_dir / FEATURE_BANK_FILENAME
    
    if not path.exists():
        return None
    data = torch.load(path, map_location=device, weights_only=False)
    return FeatureBank(embeddings=data["embeddings"].to(device), device=device)


@torch.no_grad()
def knn_distance(
    image_tensor: torch.Tensor,
    model: nn.Module,
    bank: FeatureBank,
    embedding_fn: EmbeddingFn,
    k: int = DEFAULT_K,
) -> float:
    """Distance from one preprocessed image (already batched, shape (1,C,H,W))
    to its k-th nearest neighbor in the training feature bank. Higher = more
    likely near-OOD.

    embedding_fn MUST be the same one used to build `bank` — mixing
    extractors gives a distance number that looks valid but means nothing,
    since it's comparing embeddings from two different spaces."""
    model.eval().to(bank.device)
    image_tensor = image_tensor.to(bank.device)

    query_emb = _get_embedding(model, image_tensor, embedding_fn)  # (1, D)
    dists = torch.cdist(query_emb, bank.embeddings)  # (1, N)
    kth_dist = dists.topk(k, largest=False).values[0, -1]
    return kth_dist.item()


def verify_in_distribution(
    image_tensor: torch.Tensor,
    model: nn.Module,
    bank: FeatureBank,
    threshold: float,
    embedding_fn: EmbeddingFn,
    k: int = DEFAULT_K,
) -> tuple[bool, float]:
    """Returns (is_in_distribution, distance). Mirrors
    src/gateway.py's verify_image_is_skin() return shape so the API
    route can call both gates the same way."""
    distance = knn_distance(image_tensor, model, bank, embedding_fn, k=k)
    return distance <= threshold, distance
