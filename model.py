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

from typing import Dict, Optional, Tuple
import math

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


class ResidualProjection(nn.Module):
    def __init__(self, input_dim, output_dim, dropout):
        super().__init__()
        self.skip = nn.Linear(input_dim, output_dim)
        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim), nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        return self.norm(self.skip(x) + self.block(x))


class FullGeneDecoder(nn.Module):
    def __init__(self, input_dim, gene_dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout), nn.Linear(hidden_dim, gene_dim),
        )

    def forward(self, x):
        return self.net(x)



class RingCrossAttention(nn.Module):
    """Cross-attention for one spatial ring.

    Each ring has a stable mean path and an adaptive attention path.  Attention
    learns a correction to the ring mean instead of being allowed to replace it
    completely.  During training, random neighbour tokens are replaced by the
    ring mean; this prevents brittle dependence on one exact neighbour.
    """
    def __init__(
        self,
        context_dim,
        shared_dim,
        heads=4,
        dropout=0.12,
        neighbor_dropout=0.15,
        attention_mix_init=0.30,
    ):
        super().__init__()
        if shared_dim % heads != 0:
            raise ValueError("shared_dim must be divisible by heads")
        self.neighbor_dropout = float(neighbor_dropout)
        self.neighbor_proj = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, shared_dim),
            nn.GELU(),
            nn.LayerNorm(shared_dim),
        )
        self.pos_proj = nn.Sequential(
            nn.Linear(4, shared_dim), nn.GELU(),
            nn.Linear(shared_dim, shared_dim),
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=shared_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        init = min(max(float(attention_mix_init), 1e-3), 1.0 - 1e-3)
        self.attention_mix_raw = nn.Parameter(torch.tensor(math.log(init / (1.0 - init)), dtype=torch.float32))
        self.norm = nn.LayerNorm(shared_dim)

    def forward(self, local, neighbor_feature, delta_xy, refinement_mode="full"):
        content = self.neighbor_proj(neighbor_feature.float())
        stable_mean = content.mean(dim=1)
        if refinement_mode == "no_attention":
            return self.norm(stable_mean), stable_mean

        dist = torch.sqrt(torch.clamp((delta_xy.float() ** 2).sum(dim=-1, keepdim=True), min=1e-8))
        pos_in = torch.cat([delta_xy.float(), dist, torch.log1p(dist)], dim=-1)
        tokens = content + self.pos_proj(pos_in)

        if self.training and self.neighbor_dropout > 0.0:
            keep = torch.rand(tokens.shape[0], tokens.shape[1], 1, device=tokens.device) >= self.neighbor_dropout
            replacement = tokens.mean(dim=1, keepdim=True)
            tokens = torch.where(keep, tokens, replacement)

        q = local.unsqueeze(1)
        attended, _ = self.attn(q, tokens, tokens, need_weights=False)
        attended = attended.squeeze(1)

        if refinement_mode == "attention_only":
            # Remove only mean-referenced ring refinement; the downstream
            # anchor, ring aggregation and gates remain unchanged.
            context = attended
        else:
            mix = torch.sigmoid(self.attention_mix_raw)
            context = stable_mean + mix * (attended - stable_mean)
        context = self.norm(context)
        return context, stable_mean


class RobustMultiRingSpatialContext(nn.Module):
    """K12-anchored, multi-ring spatial context.

    The first twelve neighbours provide a deterministic stable mean anchor.
    Three ring attentions can deviate from that anchor, but both the adaptive
    deviation and the total context contribution are bounded by learned gates.
    """
    def __init__(
        self,
        local_dim,
        context_dim,
        shared_dim,
        heads=4,
        dropout=0.12,
        ring_sizes=(4, 8, 20),
        neighbor_dropout=0.15,
        max_adaptive_mix=0.70,
        max_context_strength=0.80,
        ring_fusion_mode="adaptive",
        context_mode="full",
    ):
        super().__init__()
        self.ring_sizes = tuple(int(v) for v in ring_sizes)
        if self.ring_sizes != (4, 8, 20):
            raise ValueError("V5R currently expects ring_sizes=(4,8,20)")
        self.max_adaptive_mix = float(max_adaptive_mix)
        self.max_context_strength = float(max_context_strength)
        self.context_mode = str(context_mode).lower()
        if self.context_mode not in {"full", "none", "no_anchor", "no_attention", "attention_only", "anchor32"}:
            raise ValueError("Unsupported context_mode: {}".format(context_mode))
        self.ring_fusion_mode = str(ring_fusion_mode).lower()
        if self.ring_fusion_mode not in {"adaptive", "fixed_equal"}:
            raise ValueError(
                "ring_fusion_mode must be 'adaptive' or 'fixed_equal', got {}".format(
                    ring_fusion_mode
                )
            )

        self.local_proj = ResidualProjection(local_dim, shared_dim, dropout)
        self.rings = nn.ModuleList([
            RingCrossAttention(
                context_dim=context_dim,
                shared_dim=shared_dim,
                heads=heads,
                dropout=dropout,
                neighbor_dropout=neighbor_dropout,
                attention_mix_init=0.30,
            )
            for _ in self.ring_sizes
        ])

        self.ring_gate = nn.Sequential(
            nn.Linear(shared_dim * 4, shared_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(shared_dim, 3),
        )
        # Start by preferring the <=12-neighbour scales and keep the far ring as
        # a smaller auxiliary signal.
        nn.init.zeros_(self.ring_gate[-1].weight)
        with torch.no_grad():
            self.ring_gate[-1].bias.copy_(torch.tensor([0.30, 0.30, -0.30]))

        self.adaptive_gate = nn.Sequential(
            nn.Linear(shared_dim * 3, shared_dim), nn.GELU(),
            nn.Linear(shared_dim, 1),
        )
        nn.init.zeros_(self.adaptive_gate[-1].weight)
        nn.init.constant_(self.adaptive_gate[-1].bias, -0.85)  # sigmoid ~= 0.30

        self.context_strength_gate = nn.Sequential(
            nn.Linear(shared_dim * 2, shared_dim), nn.GELU(),
            nn.Linear(shared_dim, 1),
        )
        nn.init.zeros_(self.context_strength_gate[-1].weight)
        nn.init.constant_(self.context_strength_gate[-1].bias, -0.40)  # 0.8*sigmoid ~= 0.32

        self.context_norm = nn.LayerNorm(shared_dim)
        self.out_norm = nn.LayerNorm(shared_dim)

    def forward(self, local_feature, neighbor_feature, delta_xy):
        local = self.local_proj(local_feature)
        if self.context_mode == "none":
            return {"shared": local}
        if neighbor_feature.shape[1] < 32:
            raise RuntimeError("Robust multi-ring mode requires >=32 neighbours")

        ring_contexts = []
        ring_means = []
        start = 0
        for size, ring in zip(self.ring_sizes, self.rings):
            end = start + size
            ctx, stable = ring(
                local,
                neighbor_feature[:, start:end],
                delta_xy[:, start:end],
                refinement_mode=self.context_mode,
            )
            ring_contexts.append(ctx)
            ring_means.append(stable)
            start = end

        # Exact first-12 stable anchor in projected feature space.
        if self.context_mode == "anchor32":
            stable12 = (4.0 * ring_means[0] + 8.0 * ring_means[1] + 20.0 * ring_means[2]) / 32.0
        else:
            stable12 = (4.0 * ring_means[0] + 8.0 * ring_means[1]) / 12.0
        stable12 = self.context_norm(stable12)

        stacked = torch.stack(ring_contexts, dim=1)
        if self.ring_fusion_mode == "fixed_equal":
            # Ablation: remove sample-adaptive scale selection while preserving
            # all three rings, the first-12 stable anchor, ring attention,
            # adaptive-mix gating, and bounded context-strength gating.
            ring_weight = torch.full(
                (local.shape[0], len(self.ring_sizes)),
                1.0 / float(len(self.ring_sizes)),
                dtype=local.dtype,
                device=local.device,
            )
        else:
            logits = self.ring_gate(torch.cat([local] + ring_contexts, dim=-1))
            ring_weight = torch.softmax(logits, dim=-1)
        adaptive_context = (ring_weight.unsqueeze(-1) * stacked).sum(dim=1)
        adaptive_context = self.context_norm(adaptive_context)

        if self.context_mode == "no_anchor":
            hybrid_context = adaptive_context
        else:
            adaptive_mix = self.max_adaptive_mix * torch.sigmoid(
                self.adaptive_gate(torch.cat([local, stable12, adaptive_context], dim=-1))
            )
            hybrid_context = stable12 + adaptive_mix * (adaptive_context - stable12)
            hybrid_context = self.context_norm(hybrid_context)

        context_strength = self.max_context_strength * torch.sigmoid(
            self.context_strength_gate(torch.cat([local, hybrid_context], dim=-1))
        )
        fused = self.out_norm(local + context_strength * hybrid_context)

        return {"shared": fused}


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
