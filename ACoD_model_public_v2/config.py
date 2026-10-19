"""Model configuration and benchmark gene dimensions."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


GENE_DIMENSIONS = {"her2": 785, "cscc": 171, "alex": 685}


@dataclass(frozen=True)
class ModelConfig:
    """Arguments accepted by :class:`model.ACoD`.

    ``context_dim`` is the dimension of the externally supplied neighboring
    image features; it need not equal the center-image backbone output size.
    ``imagenet_pretrained=True`` may download torchvision backbone weights.
    """

    gene_dim: int
    backbone_name: str = "densenet121"
    imagenet_pretrained: bool = True
    normalize_imagenet: bool = True
    shared_dim: int = 384
    branch_dim: int = 256
    decoder_hidden: int = 512
    dropout: float = 0.12
    context_dim: int = 1024
    attention_heads: int = 4
    max_neighbors: int = 32
    neighbor_dropout: float = 0.15
    max_adaptive_mix: float = 0.70
    max_context_strength: float = 0.80
    ring_fusion_mode: str = "adaptive"
    context_mode: str = "full"
    coordinate_mode: str = "dual"

    @classmethod
    def for_dataset(cls, dataset: str, **overrides: Any) -> "ModelConfig":
        """Set only the gene dimension for ``her2``, ``cscc`` or ``alex``.

        These presets do not load data, define gene ordering, or preprocess it.
        All other model options can be supplied explicitly as overrides.
        """
        name = str(dataset).lower()
        if name not in GENE_DIMENSIONS:
            raise ValueError("dataset must be her2, cscc or alex")
        values = {"gene_dim": GENE_DIMENSIONS[name]}
        values.update(overrides)
        return cls(**values)

    def to_dict(self) -> Dict[str, Any]:
        """Return exactly the constructor keyword arguments."""
        return asdict(self)

    def build(self):
        from model import ACoD

        return ACoD(**self.to_dict())
