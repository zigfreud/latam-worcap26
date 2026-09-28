"""E09: contexto local sem usar alvos vizinhos ou estatísticas futuras."""

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter

from .experiment import ATMOSPHERIC, require
from .temporal import (
    prediction_temporal_features,
    prepare_temporal_training,
    temporal_feature_names,
)


def local_mean(values):
    """Média 5x5 dos vizinhos existentes; eixos anteriores nunca são misturados."""
    values = np.asarray(values, dtype=np.float64)
    require(values.ndim >= 2 and np.isfinite(values).all(), "Campo espacial inválido")
    size = (1,)*(values.ndim-2)+(5, 5)
    numerator = uniform_filter(values, size=size, mode="constant", cval=0)
    denominator = uniform_filter(np.ones(values.shape[-2:]), size=5, mode="constant", cval=0)
    return (numerator/denominator).astype(np.float32)


def spatial_feature_names(arm):
    require(arm in ("spatial", "pressure"), "Braço espacial inválido")
    extra = ([f"local5_anomaly_{v}" for v in ATMOSPHERIC] if arm == "spatial" else
             ["raw_surface_pressure", "local5_raw_surface_pressure", "pressure_local_contrast"])
    return temporal_feature_names(3)+extra


def extra_features(datasets, pre, dates, arm, points):
    """Mesma transformação para treino e inferência; subtração na precisão original."""
    names = ATMOSPHERIC if arm == "spatial" else ("surface_pressure",)
    sources = pd.DatetimeIndex(dates)-pd.DateOffset(months=1)
    columns = []
    for name in names:
        raw = datasets[name][name].sel(time=sources).transpose("time", "lat", "lon").values
        require(np.isfinite(raw).all(), "Campo não finito")
        if arm == "spatial":
            means = pre.feature_month_mean.sel(feature=name).values[sources.month.to_numpy()-1]
            field = raw-means
            columns.append(local_mean(field).reshape(len(dates), -1)[:, points])
        else:
            local = local_mean(raw).reshape(len(dates), -1)[:, points]
            point = raw.reshape(len(dates), -1)[:, points].astype(np.float32)
            columns.extend([point, local, point-local])
    return np.stack(columns, axis=-1).reshape(len(dates)*len(points), -1)


def prepare_spatial_training(datasets, tp, start, end, points, arm):
    base, target, pre = prepare_temporal_training(datasets, tp, start, end, points, 3)
    pre.attrs["spatial_arm"] = arm
    pre.attrs["spatial_window"] = 5
    pre.attrs["pressure_units_verified"] = "false"
    dates = pd.date_range(f"{start}-01-01", f"{end}-12-01", freq="MS")
    extra = extra_features(datasets, pre, dates, arm, points)
    return np.concatenate([base, extra], axis=1), target, pre


def prediction_spatial_features(datasets, pre, date):
    base = prediction_temporal_features(datasets, pre, date)
    extra = extra_features(datasets, pre, [date], pre.attrs["spatial_arm"], np.arange(len(base)))
    return np.concatenate([base, extra], axis=1)
