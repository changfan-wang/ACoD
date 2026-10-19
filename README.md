# ACoD

Model architecture and dual-coordinate prediction fusion for histology-based spatial gene expression prediction.

## Installation

    python -m pip install -r requirements.txt

## Model

    from model import ACoD
    model = ACoD(gene_dim=785, imagenet_pretrained=False)

Gene dimensions: HER2+ = 785, cSCC = 171, Alex+10x = 685.

The forward inputs are image patches, neighboring features, and relative spatial coordinates.

## Prediction fusion

    from fusion import prediction_views
    views, _ = prediction_views(
        global_z, trend_z, fold_mean, fold_std, 'dual'
    )
    prediction = views['primary']

Apply fusion after collecting predictions for all spots on the current slide. fold_mean and fold_std are training-derived normalization statistics.
