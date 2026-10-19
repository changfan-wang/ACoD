# ACoD

Anchor-Guided Spatial Context and Dual-Coordinate Learning for Spatial Gene Expression Prediction.

ACoD combines center histology patches with spatially ordered neighboring image features. Ring-wise means and a short-range anchor guide spatial aggregation. Global and slide-specific prediction views are aligned and fused over the current slide.

## Installation

Install matching PyTorch and torchvision builds for your hardware, then run:

    python -m pip install -r requirements.txt

## Code structure

    model.py                     ACoD model and dual prediction heads
    fusion.py                    Gene-wise alignment and prediction fusion
    config.py                    Model configuration interface
    inference.py                 Batched inference over a complete slide
    modules/
      backbone.py                Histology image encoders
      projection.py              Residual feature projections
      decoder.py                 Gene expression decoders
      spatial_context.py         Ring attention and anchor-guided aggregation
      __init__.py                Component exports
    configs/
      her2.json                  HER2+ model configuration, 785 genes
      cscc.json                  cSCC model configuration, 171 genes
      alex.json                  Alex+10x model configuration, 685 genes
    examples/
      demo.py                    CPU demonstration with synthetic inputs

## Model interface

    from config import ModelConfig
    config = ModelConfig.for_dataset("her2", imagenet_pretrained=False)
    model = config.build()

`from model import ACoD` is also supported. The model retains the constructor and parameter names of `FreshStage1V5R`. Configurations describe model dimensions and options; gene ordering and feature preprocessing must match the supplied weights.

| Input | Shape | Convention |
| --- | --- | --- |
| `image` | `[N, 3, 224, 224]` | RGB intensities in `[0, 1]`; ImageNet normalization is applied inside the encoder when enabled |
| `neighbor_feature` | `[N, K, context_dim]` | Expression-free neighboring image features in the spatial order used by the model; `K >= 32` |
| `neighbor_delta_xy` | `[N, K, 2]` | Relative spatial offsets with matching neighbor order and coordinate scale |

The first 32 neighbors form rings of 4, 8 and 20. The first 12 define the short-range anchor. The standard model returns `global_z` and `trend_z`, each shaped `[N, gene_dim]`.

## Slide prediction

    from inference import predict_slide
    predictions = predict_slide(
        model, images, neighbor_features, neighbor_delta_xy,
        fold_mean, fold_std, batch_size=32,
    )
    expression = predictions["primary"]

The caller places the model on the desired device. The function collects every spot of the slide in input order before aligning and fusing predictions. `fold_mean` and `fold_std` are training-derived normalization statistics in the model's gene order. Fusion restores the expression scale using the predicted Global mean and standard deviation.

Apply fusion once over the complete slide. Treating each batch as a separate slide changes the normalization statistics.

## Command-line inference

Provide a matching raw PyTorch `state_dict`, or a dictionary containing a `state_dict` key, with an explicit JSON model configuration. `slide.npz` contains `images`, `neighbor_features`, `neighbor_delta_xy`, `fold_mean` and `fold_std`. Images and features use floating-point arrays with the shapes above.

    python inference.py --config configs/her2.json --weights state_dict.pt --inputs slide.npz --output predictions.npz --device cpu --batch-size 32

Use `--device cuda:0` for a compatible GPU. The output contains raw heads and named prediction views; the standard dual-coordinate model includes `global`, `spatial`, `fusion` and `primary`. Command-line inference loads backbone parameters from the supplied state dictionary and does not download ImageNet weights.

## Example

    python examples/demo.py

The example uses fixed-seed synthetic tensors, random model weights and a compact configuration. It prints output array shapes and runs on CPU without downloading weights.


