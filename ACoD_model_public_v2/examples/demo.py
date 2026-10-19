"""CPU shape demonstration with synthetic inputs and random model weights."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from config import ModelConfig
from inference import predict_slide


def main():
    torch.manual_seed(0)
    # Small dimensions make this an offline smoke example, not an experiment.
    config = ModelConfig(
        gene_dim=8,
        backbone_name="resnet18",
        imagenet_pretrained=False,
        shared_dim=32,
        branch_dim=16,
        decoder_hidden=64,
        context_dim=32,
    )
    model = config.build().cpu()
    images = torch.rand(4, 3, 224, 224)
    neighbors = torch.randn(4, 32, config.context_dim)
    offsets = torch.randn(4, 32, 2)
    # Identity statistics are synthetic placeholders for the shape demo only.
    # For trained models, supply their actual training-derived statistics.
    predictions = predict_slide(
        model, images, neighbors, offsets,
        fold_mean=np.zeros(config.gene_dim),
        fold_std=np.ones(config.gene_dim),
        batch_size=2,
    )
    print("Synthetic inputs and random weights; no performance evaluation.")
    for name in ("global_z", "trend_z", "global", "spatial", "fusion", "primary"):
        print("{}: {}".format(name, predictions[name].shape))


if __name__ == "__main__":
    main()
