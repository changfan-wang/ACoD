"""Batched inference from prepared tensors, with fusion over a complete slide.

Neighbor features and relative coordinates must match the center patches and
the conventions used to train the supplied model.
"""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from config import ModelConfig
from fusion import prediction_views


def _check_tensor(value: torch.Tensor, name: str, ndim: int) -> None:
    if not isinstance(value, torch.Tensor) or value.ndim != ndim:
        raise ValueError("{} must be a {}-dimensional Tensor".format(name, ndim))
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ValueError("{} must contain finite floating-point values".format(name))


def predict_slide(
    model: torch.nn.Module,
    images: torch.Tensor,
    neighbor_features: Optional[torch.Tensor],
    neighbor_delta_xy: Optional[torch.Tensor],
    fold_mean: np.ndarray,
    fold_std: np.ndarray,
    batch_size: int = 32,
) -> Dict[str, np.ndarray]:
    """Predict every spot of one slide in input order.

    Inputs:
        images: Unnormalized image intensities in [0, 1], shape [N, 3, 224, 224].
        neighbor_features: Expression-free image features [N, K, context_dim].
        neighbor_delta_xy: Relative spatial offsets [N, K, 2], with neighbors in
            the same order as the features. K must be at least 32. Both neighbor
            arguments can be None only when model.context_mode is ``none``.
        fold_mean/fold_std: Training-derived gene-wise normalization statistics
            in the same gene order as the model output. No held-out expression
            is required. Standard deviations must be positive.

    The caller places the model on its desired device; input batches are moved
    there as needed. Raw heads are concatenated for the complete slide before
    prediction_views is called once, preserving slide-wise fusion statistics.
    The previous training/evaluation flag of every submodule is restored even
    if inference fails. No model parameters or device placement are changed.

    Returns:
        NumPy arrays [N, gene_dim]. Includes raw ``global_z`` (and ``trend_z``
        where available), restored ``global`` and ``primary`` predictions, and
        the additional views produced by fusion.prediction_views for the model's
        coordinate mode. ``primary`` is equal-weight fusion in dual/global_global
        mode, and the global prediction in global_only mode.
    """
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    _check_tensor(images, "images", 4)
    n = images.shape[0]
    if not n or tuple(images.shape[1:]) != (3, 224, 224):
        raise ValueError("images must have nonempty shape [N, 3, 224, 224]")
    if bool((images < 0).any()) or bool((images > 1).any()):
        raise ValueError("images must contain unnormalized intensities in [0, 1]")

    if getattr(model, "context_mode", None) != "none":
        _check_tensor(neighbor_features, "neighbor_features", 3)
        _check_tensor(neighbor_delta_xy, "neighbor_delta_xy", 3)
        if neighbor_features.shape[0] != n or neighbor_features.shape[1] < 32:
            raise ValueError("neighbor_features must have shape [N, K>=32, context_dim]")
        if tuple(neighbor_delta_xy.shape) != tuple(neighbor_features.shape[:2]) + (2,):
            raise ValueError("neighbor_delta_xy must have matching shape [N, K, 2]")

    coordinate_mode = getattr(model, "coordinate_mode", None)
    if coordinate_mode not in {"dual", "global_only", "global_global"}:
        raise ValueError("model has no supported coordinate_mode")
    # Validate normalization before running the image backbone.
    gene_dim = int(model.gene_dim)
    mean = np.asarray(fold_mean, dtype=np.float64).reshape(-1)
    std = np.asarray(fold_std, dtype=np.float64).reshape(-1)
    if mean.size != gene_dim or std.size != gene_dim:
        raise ValueError("fold_mean and fold_std must each contain gene_dim values")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("fold statistics must be finite with positive std")

    parameter = next(model.parameters(), None)
    device = parameter.device if parameter is not None else torch.device("cpu")
    training_states = [(module, module.training) for module in model.modules()]
    head_names = ("global_z",) if coordinate_mode == "global_only" else ("global_z", "trend_z")
    collected = {name: [] for name in head_names}
    try:
        model.eval()
        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                output = model(
                    images[start:end].to(device=device, dtype=torch.float32),
                    None if model.context_mode == "none" else neighbor_features[start:end].to(device=device, dtype=torch.float32),
                    None if model.context_mode == "none" else neighbor_delta_xy[start:end].to(device=device, dtype=torch.float32),
                )
                for name in head_names:
                    value = output[name]
                    if tuple(value.shape) != (end - start, gene_dim):
                        raise ValueError("{} has an unexpected output shape".format(name))
                    collected[name].append(value.detach().float().cpu().numpy())
    finally:
        # Assign saved flags directly so custom train() overrides do not change
        # a previously frozen backbone or individually configured child module.
        for module, training in training_states:
            module.training = training

    heads = {name: np.concatenate(parts, axis=0) for name, parts in collected.items()}
    views, _ = prediction_views(
        heads["global_z"], heads.get("trend_z"), mean, std, coordinate_mode
    )
    return dict(heads, **views)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True,
                        help="JSON with ModelConfig constructor fields")
    parser.add_argument("--weights", type=Path, required=True,
                        help="matching PyTorch state_dict, or dictionary with a state_dict key")
    parser.add_argument("--inputs", type=Path, required=True,
                        help="one-slide NPZ: images, neighbor_features, neighbor_delta_xy, fold_mean, fold_std")
    parser.add_argument("--output", type=Path, required=True,
                        help="output NPZ containing named prediction views")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    values = json.loads(args.config.read_text(encoding="utf-8"))
    # Checkpoint weights replace the backbone as well; avoid any weight download.
    values["imagenet_pretrained"] = False
    model = ModelConfig(**values).build()
    load_options = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_options["weights_only"] = True
    state = torch.load(str(args.weights), **load_options)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.to(torch.device(args.device))

    with np.load(str(args.inputs), allow_pickle=False) as inputs:
        images = torch.from_numpy(inputs["images"].astype(np.float32, copy=False))
        if model.context_mode == "none":
            neighbors = offsets = None
        else:
            neighbors = torch.from_numpy(inputs["neighbor_features"].astype(np.float32, copy=False))
            offsets = torch.from_numpy(inputs["neighbor_delta_xy"].astype(np.float32, copy=False))
        predictions = predict_slide(
            model, images, neighbors, offsets, inputs["fold_mean"], inputs["fold_std"],
            batch_size=args.batch_size,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as destination:
        np.savez_compressed(destination, **predictions)
    print("Saved {} spot predictions to {}".format(images.shape[0], args.output))


if __name__ == "__main__":
    main()
