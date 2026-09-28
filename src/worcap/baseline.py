"""Climatologia por mês do alvo e ponto da grade, sem dados futuros."""

import numpy as np
import pandas as pd
import xarray as xr


def monthly_climatology(tp: xr.DataArray, start_year: int, end_year: int) -> xr.DataArray:
    """Use precipitação observada (treino_tp), NÃO o alvo já deslocado."""
    if start_year > end_year:
        raise ValueError("start_year deve ser menor ou igual a end_year")
    history = tp.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))
    if history.sizes.get("time", 0) == 0:
        raise ValueError("Nenhuma observação na janela de climatologia")
    expected = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-01", freq="MS")
    if not np.array_equal(history.time.values, expected.values):
        raise ValueError("A janela deve conter uma série mensal completa, ordenada e sem duplicatas")
    if not bool(np.isfinite(history).all().compute().item()):
        raise ValueError("Histórico contém valores não finitos")
    result = history.groupby("time.month").mean("time", skipna=True)
    if result.sizes["month"] != 12:
        raise ValueError("A janela precisa conter todos os 12 meses")
    if not bool(np.isfinite(result).all().compute().item()):
        raise ValueError("Climatologia contém pontos sem observações válidas")
    return result


def predict_climatology(climatology: xr.DataArray, target_time: xr.DataArray) -> xr.DataArray:
    """Selecione o mês ALVO: time no teste já aponta para ele."""
    return climatology.sel(month=target_time.dt.month).rename("tp_mm_day")


def rmse(prediction: xr.DataArray, target: xr.DataArray) -> float:
    """RMSE não ponderado; falhe em desalinhamento ou valores não finitos."""
    prediction, target = xr.align(prediction, target, join="exact")
    if set(prediction.dims) != set(target.dims):
        raise ValueError("Dimensões diferentes entre previsão e alvo")
    if not bool((np.isfinite(prediction) & np.isfinite(target)).all().compute().item()):
        raise ValueError("RMSE exige previsões e alvos finitos")
    return float(np.sqrt(((prediction - target) ** 2).mean()).compute().item())
