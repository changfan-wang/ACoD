"""Anchor-guided multi-ring spatial context encoding."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .projection import ResidualProjection


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
