"""Métricas descritivas; não ajustam modelos nem escolhem novos candidatos."""

import numpy as np

from .experiment import require


def error_metrics(prediction, truth, baseline=None):
    require(prediction.shape == truth.shape and truth.size > 0, "Formas/amostra inválidas")
    require(np.isfinite(prediction).all() and np.isfinite(truth).all(), "Valores não finitos")
    err = prediction.astype(np.float64)-truth
    absolute = np.abs(err)
    mse = np.square(err).mean()
    bias = err.mean()
    variance = np.var(err)
    require(np.isclose(mse, bias*bias+variance, rtol=1e-12, atol=1e-12), "Decomposição do erro")
    result = {"count": int(err.size), "rmse": float(np.sqrt(mse)),
              "mae": float(absolute.mean()), "bias": float(bias),
              "error_std": float(np.sqrt(variance)), "sse": float(mse*err.size),
              "observed_mean": float(truth.mean(dtype=np.float64)),
              "predicted_mean": float(prediction.mean(dtype=np.float64))}
    for tolerance in (.5, 1, 2):
        result[f"within_{tolerance:g}_fraction"] = float((absolute <= tolerance).mean())
    if baseline is not None:
        require(baseline.shape == truth.shape and np.isfinite(baseline).all(), "Baseline inválido")
        base_abs = np.abs(baseline.astype(np.float64)-truth)
        base_rmse = float(np.sqrt(np.square(base_abs).mean()))
        result.update({"baseline_rmse": base_rmse,
                       "rmse_gain": base_rmse-result["rmse"],
                       "rmse_reduction_pct": 100*(1-result["rmse"]/base_rmse) if base_rmse else None,
                       "improved_fraction": float((absolute < base_abs).mean()),
                       "tied_fraction": float((absolute == base_abs).mean()),
                       "worsened_fraction": float((absolute > base_abs).mean())})
    return result


def spatial_metrics(prediction, truth, baseline):
    require(prediction.shape == truth.shape == baseline.shape and truth.ndim == 3,
            "Esperado tempo/latitude/longitude alinhados")
    error = prediction.astype(np.float64)-truth
    base_error = baseline.astype(np.float64)-truth
    mse, bias = np.square(error).mean(axis=0), error.mean(axis=0)
    variance = np.var(error, axis=0)
    require(np.allclose(mse, bias*bias+variance, rtol=1e-12, atol=1e-12), "Decomposição espacial")
    return {"rmse": np.sqrt(mse), "bias": bias, "error_std": np.sqrt(variance),
            "mae": np.abs(error).mean(axis=0),
            "rmse_gain": np.sqrt(np.square(base_error).mean(axis=0))-np.sqrt(mse),
            "improved_fraction": (np.abs(error) < np.abs(base_error)).mean(axis=0),
            "within_1_fraction": (np.abs(error) <= 1).mean(axis=0)}


def region_masks(lat, lon):
    """Nove setores disjuntos, incluindo oceano; não são divisões administrativas."""
    yy, xx = np.meshgrid(lat, lon, indexing="ij")
    masks = []
    for a, (south, north) in enumerate(((-90, -35), (-35, -15), (-15, 90))):
        for b, (west, east) in enumerate(((-180, -70), (-70, -50), (-50, 180))):
            mask = (yy >= south) & (yy < north) & (xx >= west) & (xx < east)
            if mask.any():
                masks.append((f"L{a+1}C{b+1}", mask))
    require(np.all(np.sum([m for _, m in masks], axis=0) == 1), "Setores não particionam a grade")
    return masks
