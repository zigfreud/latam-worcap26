"""SST mensal CPC: temperaturas brutas, defasagem explícita, anomalias só do treino."""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .experiment import require

OCEAN_NAMES = ["sst_nino12_anomaly", "sst_nino3_anomaly", "sst_nino4_anomaly",
               "sst_nino34_anomaly"]


def read_sst(path: Path):
    raw = pd.read_csv(path, sep=r"\s+", skiprows=1, header=None)
    require(raw.shape[1] == 10, "Esperadas dez colunas no arquivo CPC")
    require(np.isfinite(raw.values).all(), "Arquivo CPC contém valores não finitos")
    require((raw.iloc[:, :2].values == raw.iloc[:, :2].values.astype(int)).all(),
            "Ano/mês devem ser inteiros")
    dates = pd.to_datetime({"year": raw.iloc[:, 0].astype(int),
                            "month": raw.iloc[:, 1].astype(int), "day": 1})
    index = pd.DatetimeIndex(dates)
    require(index.is_unique and index.is_monotonic_increasing, "Datas CPC duplicadas/desordenadas")
    require(np.array_equal(index, pd.date_range(index[0], index[-1], freq="MS")),
            "Série CPC deve ser mensal completa")
    # As colunas ANOM publicadas usam uma climatologia que pode incluir o futuro do fold.
    values = raw.iloc[:, [2, 4, 6, 8]].to_numpy(dtype=np.float32)
    require(((values > -5) & (values < 45)).all(), "SST fora da faixa; possível sentinela")
    return pd.DataFrame(values, index=index, columns=OCEAN_NAMES)


def ocean_sources(dates, lag):
    require(isinstance(lag, int) and lag >= 2, "CPC mensal exige defasagem mínima de dois meses")
    dates = pd.DatetimeIndex(dates)
    require((dates.day == 1).all(), "Usar meses-alvo no primeiro dia")
    return dates - pd.DateOffset(months=lag)


def ocean_rows(table, dates, lag, means=None):
    sources = ocean_sources(dates, lag)
    require(sources.isin(table.index).all(), "Meses oceânicos ausentes; não interpolar")
    values = table.loc[sources].to_numpy(dtype=np.float32)
    if means is None:
        require(set(sources.month) == set(range(1, 13)), "Treino oceânico deve conter todos os meses")
        means = np.stack([values[sources.month == month].mean(axis=0) for month in range(1, 13)])
    rows = values - means[sources.month.to_numpy()-1]
    require(np.isfinite(rows).all(), "Anomalias oceânicas inválidas")
    return rows, means


def append_ocean_training(features, dates, points_count, preprocessing, table, lag):
    rows, means = ocean_rows(table, dates, lag)
    expanded = np.repeat(rows, points_count, axis=0)
    require(len(features) == len(expanded), "Ordem tempo-ponto incompatível")
    preprocessing["ocean_month_mean"] = xr.DataArray(means,
        dims=("ocean_month", "ocean_feature"),
        coords={"ocean_month": np.arange(1, 13), "ocean_feature": OCEAN_NAMES})
    preprocessing.attrs["ocean_lag_months"] = lag
    preprocessing.attrs["historical_vintages_verified"] = "false"
    return np.concatenate([features, expanded], axis=1)


def append_ocean_prediction(features, date, preprocessing, table):
    require(table is not None, "Modelo oceânico exige a tabela CPC")
    rows, _ = ocean_rows(table, [date], int(preprocessing.attrs["ocean_lag_months"]),
                          preprocessing.ocean_month_mean.values)
    return np.concatenate([features, np.repeat(rows, len(features), axis=0)], axis=1)
