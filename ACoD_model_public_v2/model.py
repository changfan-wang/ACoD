#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Final Stage-I spatial dual-view model.

The model combines a target patch with three-ring expression-free spatial context,
then predicts complementary global-scale and within-slide spatial-variation views.

Core context path:
    center image -> local feature
    first 12 neighbours -> stable K12 mean anchor
    3 rings (1-4, 5-12, 13-32) -> cross-attention contexts
    adaptive multi-ring context learns only a bounded deviation from K12 mean
    bounded context-strength gate -> fused representation

The context cache is expression-free.  No HER2 task checkpoint, old embedding,
teacher or test expression is used by this model.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from modules import (
    FullGeneDecoder,
    ImageBackbone,
    ResidualProjection,
    RingCrossAttention,
    RobustMultiRingSpatialContext,
)


class FreshStage1V5R(nn.Module):
    def __init__(
        self,
        gene_dim=None,
        backbone_name="densenet121",
        imagenet_pretrained=True,
        normalize_imagenet=True,
        shared_dim=384,
        branch_dim=256,
        decoder_hidden=512,
        dropout=0.12,
        context_dim=1024,
        attention_heads=4,
        max_neighbors=32,
        neighbor_dropout=0.15,
        max_adaptive_mix=0.70,
        max_context_strength=0.80,
        ring_fusion_mode="adaptive",
        context_mode="full",
        coordinate_mode="dual",
    ):
        super().__init__()
        if int(max_neighbors) < 32:
            raise ValueError("FreshStage1V5R requires max_neighbors >= 32")
        if gene_dim is None:
            raise ValueError("Specify gene_dim when creating ACoD")
        self.gene_dim = int(gene_dim)
        self.shared_dim = int(shared_dim)
        self.branch_dim = int(branch_dim)
        self.max_neighbors = int(max_neighbors)
        self.context_mode = str(context_mode).lower()
        self.coordinate_mode = str(coordinate_mode).lower()
        if self.coordinate_mode not in {"dual", "global_only", "global_global"}:
            raise ValueError("Unsupported coordinate_mode: {}".format(coordinate_mode))

        self.backbone = ImageBackbone(
            backbone_name,
            pretrained=imagenet_pretrained,
            normalize_imagenet=normalize_imagenet,
        )
        self.context_encoder = RobustMultiRingSpatialContext(
            local_dim=self.backbone.out_dim,
            context_dim=int(context_dim),
            shared_dim=self.shared_dim,
            heads=attention_heads,
            dropout=dropout,
            neighbor_dropout=neighbor_dropout,
            max_adaptive_mix=max_adaptive_mix,
            max_context_strength=max_context_strength,
            ring_fusion_mode=ring_fusion_mode,
            context_mode=self.context_mode,
        )

        self.raw_projection = ResidualProjection(self.shared_dim, self.branch_dim, dropout)
        self.trend_projection = ResidualProjection(self.shared_dim, self.branch_dim, dropout)
        self.raw_decoder = FullGeneDecoder(self.branch_dim, self.gene_dim, decoder_hidden, dropout)
        self.trend_decoder = FullGeneDecoder(self.branch_dim, self.gene_dim, decoder_hidden, dropout)
        if self.coordinate_mode == "global_only":
            for module in (self.trend_projection, self.trend_decoder):
                for parameter in module.parameters():
                    parameter.requires_grad = False
        self.backbone_phase = "heads"

    def train(self, mode=True):
        """Keep frozen backbone BatchNorm statistics fixed during training.

        This explicitly corrects the original script, which froze parameters
        but continued updating all backbone BatchNorm running statistics.
        """
        super().train(mode)
        if mode:
            self.backbone.eval()
            if self.backbone_phase == "full":
                self.backbone.train(True)
            elif self.backbone_phase == "last_block":
                modules = [self.backbone.net.denseblock4, self.backbone.net.norm5] if self.backbone.kind == "densenet" else [self.backbone.net[-2]]
                for module in modules:
                    module.train(True)
            if self.coordinate_mode == "global_only":
                self.trend_projection.eval()
                self.trend_decoder.eval()
        return self

    def set_backbone_phase(self, phase):
        phase = str(phase)
        if phase not in {"heads", "last_block", "full"}:
            raise ValueError("Unknown phase: {}".format(phase))
        self.backbone_phase = phase
        for p in self.backbone.parameters():
            p.requires_grad = False
        if phase == "heads":
            return
        modules = [self.backbone.net.denseblock4, self.backbone.net.norm5] if self.backbone.kind == "densenet" else [self.backbone.net[-2]]
        for module in modules:
            for p in module.parameters():
                p.requires_grad = True
        if phase == "full":
            for p in self.backbone.parameters():
                p.requires_grad = True

    def forward(
        self,
        image,
        neighbor_feature,
        neighbor_delta_xy,
    ) -> Dict[str, torch.Tensor]:
        local_feature = self.backbone(image)
        if self.context_mode != "none" and (neighbor_feature is None or neighbor_delta_xy is None):
            raise RuntimeError("V5R requires whole-slide neighbour features and relative coordinates")
        context = self.context_encoder(
            local_feature,
            None if neighbor_feature is None else neighbor_feature[:, :self.max_neighbors],
            None if neighbor_delta_xy is None else neighbor_delta_xy[:, :self.max_neighbors],
        )
        shared = context["shared"]

        raw_embedding = self.raw_projection(shared)
        if self.coordinate_mode == "global_only":
            return {"global_z": self.raw_decoder(raw_embedding)}
        trend_embedding = self.trend_projection(shared)
        global_z = self.raw_decoder(raw_embedding)
        trend_z = self.trend_decoder(trend_embedding)
        return {"global_z": global_z, "trend_z": trend_z}


ACoD = FreshStage1V5R
