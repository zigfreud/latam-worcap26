"""Memória atmosférica com estatísticas ajustadas apenas no passado do treino."""

import numpy as np
import pandas as pd
import xarray as xr

from .boosting import FEATURES, prediction_features, prepare_training
from .experiment import ATMOSPHERIC, require
from .ridge import source_dates


def temporal_feature_names(months):
    require(months in (1, 2, 3), "Memória deve ser 1, 2 ou 3 meses")
    names = list(FEATURES)
    for lag in range(2, months+1):
        names += [f"lag{lag}_anomaly_{v}" for v in ATMOSPHERIC]
        names += [f"delta_lag1_lag{lag}_{v}" for v in ATMOSPHERIC]
    return names


def prepare_temporal_training(datasets, tp, start, end, points, months):
    names = temporal_feature_names(months)
    base, target, pre = prepare_training(datasets, tp, start, end, points)
    if months == 1:
        return base, target, pre
    dates = pd.date_range(f"{start}-01-01", f"{end}-12-01", freq="MS")
    means = np.empty((months-1, 12, tp.sizes["lat"], tp.sizes["lon"], 9), dtype=np.float32)
    columns = [base]
    for lag in range(2, months+1):
        sources = dates-pd.DateOffset(months=lag)
        anomaly = np.empty((len(dates), len(points), 9), dtype=np.float32)
        for j, name in enumerate(ATMOSPHERIC):
            da = datasets[name][name].sel(time=sources).transpose("time", "lat", "lon")
            require(np.array_equal(da.time.values, sources.values), f"Datas da memória: {name}")
            values = da.values
            require(np.isfinite(values).all(), "História atmosférica não finita")
            for month in range(1, 13):
                means[lag-2, month-1, :, :, j] = values[sources.month == month].mean(axis=0)
            sampled = values.reshape(len(dates), -1)[:, points]
            anomaly[:, :, j] = sampled-means[lag-2, :, :, :, j].reshape(12, -1)[
                sources.month.to_numpy()-1][:, points]
        older = anomaly.reshape(-1, 9)
        columns.extend([older, base[:, :9]-older])
    pre["lag_feature_month_mean"] = (("lag", "month", "lat", "lon", "feature"), means)
    pre = pre.assign_coords(lag=np.arange(2, months+1))
    pre.attrs["memory_months"] = months
    features = np.concatenate(columns, axis=1)
    require(features.shape[1] == len(names) and np.isfinite(features).all(), "Atributos inválidos")
    return features, target, pre


def prediction_temporal_features(datasets, pre, date):
    base = prediction_features(datasets, pre, date)
    months = int(pre.attrs.get("memory_months", 1))
    columns = [base]
    for lag in range(2, months+1):
        source = pd.Timestamp(date)-pd.DateOffset(months=lag)
        means = pre.lag_feature_month_mean.sel(lag=lag, month=source.month).values.reshape(-1, 9)
        older = np.empty((len(base), 9), dtype=np.float32)
        for j, name in enumerate(ATMOSPHERIC):
            values = datasets[name][name].sel(time=source).transpose("lat", "lon").values
            older[:, j] = values.ravel()-means[:, j]
        columns.extend([older, base[:, :9]-older])
    features = np.concatenate(columns, axis=1)
    require(np.isfinite(features).all(), "Memória de previsão não finita")
    return features


def bridge_test_atmosphere(training, test, months=3):
    """Disponibilidade de defasagens: eixo-fonte explícito, sem alvos de chuva."""
    temporal_feature_names(months)
    dates = pd.DatetimeIndex(test.time.values)
    require(np.array_equal(test.time_origem.values, source_dates(dates).values), "Origem do teste")
    first_source = dates[0]-pd.DateOffset(months=months)
    bridge = {}
    for name in ATMOSPHERIC:
        train = training[name][name]
        for coord in ("lat", "lon"):
            require(np.array_equal(train[coord], test[coord]), f"Grade: {name}/{coord}")
        new = test[name].reset_coords(drop=True).assign_coords(time=test.time_origem.values)
        old = train.sel(time=slice(first_source, None))
        overlap = np.intersect1d(old.time.values, new.time.values)
        if len(overlap):
            require(np.array_equal(old.sel(time=overlap).values, new.sel(time=overlap).values),
                    f"Sobreposição atmosférica diferente: {name}")
        prefix = old.sel(time=old.time < new.time.values[0])
        combined = xr.concat([prefix, new], dim="time")
        needed = pd.date_range(first_source, new.time.values[-1], freq="MS")
        require(np.array_equal(combined.time.values, needed.values), f"Buraco nas defasagens: {name}")
        bridge[name] = combined.to_dataset(name=name)
    return bridge
