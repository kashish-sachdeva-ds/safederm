"""Model architectures for SafeDerm.

Baseline: plain ResNet-50, ImageNet-pretrained, final layer replaced for
7-class output. Defined once here so 06_baseline_model.ipynb and any
later comparison against 07_champion_model.ipynb build the exact same
architecture -- no accidental drift between "the baseline" as described
in ADR-002 and what actually gets trained.
"""

import torch.nn as nn
from torchvision import models

from src.labels import ALL_CLASSES

NUM_CLASSES = len(ALL_CLASSES)


def build_baseline_model() -> nn.Module:
    """ResNet-50, ImageNet-pretrained, final FC layer replaced for NUM_CLASSES outputs."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model
