"""Image encoders used by ACoD."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


def _build_densenet121(pretrained: bool) -> Tuple[nn.Module, int]:
    try:
        weights = tv_models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = tv_models.densenet121(weights=weights)
    except (AttributeError, TypeError):
        model = tv_models.densenet121(pretrained=pretrained)
    return model.features, 1024


def _build_resnet50(pretrained: bool) -> Tuple[nn.Module, int]:
    try:
        weights = tv_models.ResNet50_Weights.DEFAULT if pretrained else None
        model = tv_models.resnet50(weights=weights)
    except (AttributeError, TypeError):
        model = tv_models.resnet50(pretrained=pretrained)
    return nn.Sequential(*list(model.children())[:-1]), 2048


def _build_resnet18(pretrained: bool) -> Tuple[nn.Module, int]:
    try:
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        model = tv_models.resnet18(weights=weights)
    except (AttributeError, TypeError):
        model = tv_models.resnet18(pretrained=pretrained)
    return nn.Sequential(*list(model.children())[:-1]), 512


class ImageBackbone(nn.Module):
    def __init__(self, name="densenet121", pretrained=True, normalize_imagenet=True):
        super().__init__()
        name = str(name).lower()
        if name == "densenet121":
            self.net, self.out_dim = _build_densenet121(pretrained)
            self.kind = "densenet"
        elif name == "resnet50":
            self.net, self.out_dim = _build_resnet50(pretrained)
            self.kind = "resnet"
        elif name == "resnet18":
            self.net, self.out_dim = _build_resnet18(pretrained)
            self.kind = "resnet"
        else:
            raise ValueError("Unsupported backbone: {}".format(name))
        self.normalize_imagenet = bool(normalize_imagenet)
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1,3,1,1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1,3,1,1))

    def forward(self, image):
        image = image.float()
        if self.normalize_imagenet:
            image = (image - self.image_mean) / self.image_std
        x = self.net(image)
        if self.kind == "densenet":
            x = F.relu(x, inplace=False)
            x = F.adaptive_avg_pool2d(x, (1,1))
        return x.flatten(1)
