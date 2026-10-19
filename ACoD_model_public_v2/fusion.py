"""Dual-coordinate prediction fusion."""
from __future__ import annotations
from typing import Any, Tuple
import numpy as np

FUSION_EPS = 1e-6

def finite_matrix(value: Any, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or not value.shape[0] or not value.shape[1]:
        raise ValueError("{} must be a nonempty [spot, gene] matrix".format(name))
    if not np.isfinite(value).all():
        raise ValueError("{} contains non-finite values".format(name))
    return value

def standardize_shape(value: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    value = finite_matrix(value, "prediction view")
    mu = value.mean(axis=0)
    sd = value.std(axis=0, ddof=0)
    valid = sd > FUSION_EPS
    z = (value - mu[None, :]) / np.where(valid, sd, 1.0)[None, :]
    z[:, ~valid] = 0.0
    return z, mu, sd, valid

def shape_fusion(global_pred: np.ndarray, secondary_pred: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Weight secondary shape by alpha, restoring predicted Global moments.

    A constant secondary gene contributes zero with its fixed original weight;
    there is no available-branch weight renormalization. A near-constant Global
    gene retains its original prediction. There is no post-fusion z-score.
    """
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    global_pred = finite_matrix(global_pred, "global prediction")
    secondary_pred = finite_matrix(secondary_pred, "secondary prediction")
    if global_pred.shape != secondary_pred.shape:
        raise ValueError("prediction-view shapes differ")
    z_g, mu_g, sd_g, valid_g = standardize_shape(global_pred)
    z_s, _, _, _ = standardize_shape(secondary_pred)
    fused = mu_g[None, :] + sd_g[None, :] * ((1.0 - alpha) * z_g + alpha * z_s)
    fused[:, ~valid_g] = global_pred[:, ~valid_g]
    return fused

def prediction_views(global_z: np.ndarray, trend_z: np.ndarray, fold_mean: np.ndarray,
                     fold_std: np.ndarray, coordinate_mode: str):
    """Build every view before touching the held-out targets."""
    global_z = finite_matrix(global_z, "global_z")
    mean = np.asarray(fold_mean, dtype=np.float64).reshape(1, -1)
    std = np.asarray(fold_std, dtype=np.float64).reshape(1, -1)
    if mean.shape[1] != global_z.shape[1] or std.shape != mean.shape:
        raise ValueError("fold statistics have incorrect gene dimension")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("fold statistics must be finite with positive std")
    direct = mean + std * global_z
    views = {"global": direct}
    pcc_sources = {}
    if coordinate_mode == "global_only":
        views["primary"] = direct
        return views, pcc_sources
    trend_z = finite_matrix(trend_z, "trend_z")
    if trend_z.shape != global_z.shape:
        raise ValueError("head-output shapes differ")
    if coordinate_mode == "dual":
        secondary = trend_z
        views["spatial"] = shape_fusion(direct, secondary, alpha=1.0)
        pcc_sources["spatial"] = trend_z
    elif coordinate_mode == "global_global":
        secondary = mean + std * trend_z
        views["secondary_global"] = secondary
    else:
        raise ValueError("unsupported coordinate_mode: {}".format(coordinate_mode))
    views["fusion"] = shape_fusion(direct, secondary, alpha=0.5)
    views["alpha_0.25"] = shape_fusion(direct, secondary, alpha=0.25)
    views["alpha_0.75"] = shape_fusion(direct, secondary, alpha=0.75)
    views["primary"] = views["fusion"]
    return views, pcc_sources

