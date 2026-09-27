"""Model B (advanced) for static-image FER: ImageNet-pretrained ResNet18 backbone."""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class FERResNet18(nn.Module):
    """Input: (batch, 3, H, W) RGB image. Output: (logits, embedding).

    Uses ImageNet-pretrained weights and fine-tunes the whole network by
    default (freeze=True switches to feature-extraction mode, which is the
    safer choice for small datasets like Yale -- see PLAN_EXPERIMENTAL.md 1.1).
    """

    def __init__(self, n_classes: int, embedding_dim: int = 128, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        backbone_out_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.embed = nn.Linear(backbone_out_dim, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, n_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        embedding = torch.relu(self.embed(features))
        logits = self.classifier(embedding)
        return logits, embedding
