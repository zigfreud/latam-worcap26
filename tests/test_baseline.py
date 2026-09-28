import numpy as np
import pandas as pd
import pytest
import xarray as xr

from worcap.baseline import monthly_climatology, predict_climatology, rmse


def test_future_values_do_not_change_climatology_and_target_month_is_used():
    time = pd.date_range("2019-01-01", periods=36, freq="MS")
    values = np.array(time.month, dtype=float)
    values[24:] = 99999  # Futuro artificialmente extremo: não pode entrar no ajuste.
    tp = xr.DataArray(values[:, None, None], dims=("time", "lat", "lon"),
                      coords={"time": time, "lat": [-30.0], "lon": [-53.0]})
    climatology = monthly_climatology(tp, 2019, 2020)
    target = xr.DataArray(pd.to_datetime(["2021-02-01", "2021-12-01"]), dims="time",
                          coords={"time": pd.to_datetime(["2021-02-01", "2021-12-01"])})
    prediction = predict_climatology(climatology, target)
    np.testing.assert_allclose(prediction.values.ravel(), [2, 12])
    assert rmse(prediction, prediction) == 0


def test_rmse_rejects_different_coordinates_and_nan():
    a = xr.DataArray([1.0, 2.0], dims="lat", coords={"lat": [0, 1]})
    with pytest.raises(ValueError):
        rmse(a, a.assign_coords(lat=[1, 2]))
    with pytest.raises(ValueError):
        rmse(a, a * np.nan)


def test_incomplete_seasonal_cycle_is_rejected():
    tp = xr.DataArray([1.0], dims="time", coords={"time": pd.to_datetime(["2020-01-01"])})
    with pytest.raises(ValueError, match="mensal completa"):
        monthly_climatology(tp, 2020, 2020)
