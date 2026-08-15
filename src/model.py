"""Model architectures for SafeDerm.

Baseline: plain ResNet-50, ImageNet-pretrained, final layer replaced for
7-class output. Defined once here so 06_baseline_model.ipynb and
07_champion_model.ipynb build the exact same baseline architecture --
no accidental drift between "the baseline" as described in ADR-002 and
what actually gets trained.

Champion: ResNet-50 backbone (through its last conv block) feeding a
small Transformer encoder over the resulting 7x7 spatial grid, instead of
average-pooling straight into a linear layer like the baseline does.
See ADR-002 for why (interpretability via attention maps) and ADR-006 for
the specific architecture and training-recipe decisions.
"""

import torch
import torch.nn as nn
from torchvision import models

from src.labels import ALL_CLASSES

NUM_CLASSES = len(ALL_CLASSES)


def build_baseline_model() -> nn.Module:
    """ResNet-50, ImageNet-pretrained, final FC layer replaced for NUM_CLASSES outputs."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


class _TransformerHead(nn.Module):
    """Projects a CNN feature map into a token sequence, prepends a CLS
    token, runs a small Transformer encoder over it, and classifies from
    the CLS token's output.

    Includes a real nn.Dropout layer deliberately: MC Dropout (ADR-002)
    needs at least one dropout layer to sample stochastically over at
    inference time, and a plain torchvision resnet50 (the baseline) has
    none. If 08_calibration_conformal.ipynb is going to run MC Dropout
    against this model, that dropout layer has to already exist here --
    it can't be bolted on afterwards without retraining.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 3,
        num_tokens: int = 49,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.projection = nn.Linear(in_channels, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        # +1 for the CLS token. num_tokens=49 matches a 7x7 grid, which is
        # what a 224x224 input produces through ResNet-50's layer4
        # (src.transforms.IMAGE_SIZE = 224, and ResNet halves spatial
        # dims 5 times: 224 / 32 = 7). If IMAGE_SIZE ever changes, this
        # has to change with it.
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens + 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feature_map.shape
        tokens = feature_map.flatten(2).transpose(1, 2)      # [B, H*W, C]
        tokens = self.projection(tokens)                      # [B, H*W, d_model]

        cls = self.cls_token.expand(b, -1, -1)                 # [B, 1, d_model]
        tokens = torch.cat([cls, tokens], dim=1)                # [B, H*W+1, d_model]
        tokens = tokens + self.pos_embed[:, : tokens.size(1), :]

        encoded = self.encoder(tokens)
        cls_out = encoded[:, 0, :]                               # [B, d_model]

        cls_out = self.dropout(self.norm(cls_out))
        return self.classifier(cls_out)

    def attention_rollout(self, feature_map: torch.Tensor) -> torch.Tensor:
        """Returns the CLS token's attention weights over the 49 spatial
        tokens from the last encoder layer -- the "where did the model
        look" map ADR-002 cites as the reason for choosing a transformer
        over a plain CNN. Reshape the returned [B, 49] tensor to [B, 7, 7]
        and upsample to overlay on the input image.
        """
        b, c, h, w = feature_map.shape
        tokens = feature_map.flatten(2).transpose(1, 2)
        tokens = self.projection(tokens)
        cls = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embed[:, : tokens.size(1), :]

        last_layer = self.encoder.layers[-1]
        # need_weights=True, average_attn_weights=True -> [B, seq, seq]
        _, attn_weights = last_layer.self_attn(
            tokens, tokens, tokens, need_weights=True, average_attn_weights=True
        )
        patch_attn = attn_weights[:, 0, 1:]  # CLS's attention to the 49 patch tokens
        # Renormalize to a clean distribution over just the patches (drops
        # the CLS-attends-to-itself component so the 49 values sum to 1 --
        # otherwise they'd sum to (1 - self-attention), which is correct
        # but confusing to visualize as a heatmap).
        return patch_attn / patch_attn.sum(dim=1, keepdim=True)


class ChampionModel(nn.Module):
    """CNN + Transformer hybrid (ADR-002): a ResNet-50 backbone up through
    its last conv block, feeding a Transformer encoder over the resulting
    7x7 spatial grid.

    `self.backbone` / `self.transformer_head` are named deliberately so
    src.engine.get_param_groups(model, head_attr="transformer_head") gives
    the pretrained CNN weights a low LR and the freshly-initialized
    transformer + classifier a higher one. See ADR-006.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        backbone_out_channels = resnet.fc.in_features  # 2048 for resnet50

        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        )
        self.transformer_head = _TransformerHead(
            in_channels=backbone_out_channels,
            num_classes=num_classes,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_map = self.backbone(x)  # [B, 2048, 7, 7] for a 224x224 input
        return self.transformer_head(feature_map)

    def attention_rollout(self, x: torch.Tensor) -> torch.Tensor:
        feature_map = self.backbone(x)
        return self.transformer_head.attention_rollout(feature_map)


def build_champion_model() -> nn.Module:
    """CNN + Transformer hybrid, ImageNet-pretrained backbone, for
    NUM_CLASSES outputs. See ChampionModel's docstring and ADR-002/ADR-006.
    """
    return ChampionModel(num_classes=NUM_CLASSES)
