"""Standalone pressure forecasts retain the October/November memory bridge."""

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from worcap.boosting import train_model
from worcap.delivery import infer_pressure, predict_pressure
from worcap.experiment import ATMOSPHERIC, sha256
from worcap.spatial import prediction_spatial_features, prepare_spatial_training
from worcap.temporal import bridge_test_atmosphere


@pytest.fixture
def pressure_package(tmp_path):
    dates = pd.date_range("2019-01-01", "2023-01-01", freq="MS")
    coords = {"time": dates, "lat": [-20.0, -19.75], "lon": [-50.0, -49.75]}
    rng = np.random.default_rng(413)
    values = rng.normal(size=(len(dates), 2, 2)).astype(np.float32)
    tp = xr.DataArray(np.abs(values) + 2, dims=("time", "lat", "lon"), coords=coords)
    datasets = {
        name: xr.Dataset({name: (("time", "lat", "lon"), values.copy() + j)},
                         coords=coords)
        for j, name in enumerate(ATMOSPHERIC)
    }
    x, y, pre = prepare_spatial_training(
        datasets, tp, 2020, 2022, np.arange(4), "pressure")
    config = {
        "memory_months": 3, "spatial_arm": "pressure", "seed": 413,
        "params": {"objective": "regression", "num_threads": 1, "verbosity": -1,
                   "num_leaves": 3, "min_data_in_leaf": 2},
    }
    model = train_model(x, y, config, 5)
    forecast_dates = pd.date_range("2023-01-01", "2023-02-01", freq="MS")
    origins = forecast_dates - pd.DateOffset(months=1)
    test = xr.Dataset({
        name: datasets[name][name].sel(time=origins).assign_coords(time=forecast_dates)
        for name in ATMOSPHERIC
    }).assign_coords(time_origem=("time", origins))
    # No rainfall variables exist in this test file. Historical observations are
    # deliberately never persisted: standalone inference cannot read them.
    test_path = tmp_path / "test.nc"
    test.to_netcdf(test_path)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    history = xr.Dataset({
        name: datasets[name][name].sel(time=slice("2022-10-01", "2022-11-01"))
        for name in ATMOSPHERIC
    })
    history.to_netcdf(model_dir / "atmosphere_history.nc")
    pre.to_netcdf(model_dir / "preprocessing.nc")
    model.save_model(str(model_dir / "model.txt"))
    recipe = {
        "iterations": 5, "threads": 1,
        "forecast_start": "2023-01-01", "forecast_end": "2023-02-01",
        "test_sha256": sha256(test_path),
        "artifact_sha256": {
            name: sha256(model_dir / name)
            for name in ("model.txt", "preprocessing.nc", "atmosphere_history.nc")
        },
    }
    (model_dir / "recipe.json").write_text(json.dumps(recipe), encoding="utf-8")
    return model_dir, test_path, model, pre, datasets, test, history, forecast_dates


def test_standalone_pressure_matches_direct_january_and_february(pressure_package):
    model_dir, test_path, model, pre, datasets, test, history, dates = pressure_package
    bridged = bridge_test_atmosphere({name: history[[name]] for name in ATMOSPHERIC}, test)
    np.testing.assert_array_equal(
        bridged["t2"].time.values,
        pd.date_range("2022-10-01", "2023-01-01", freq="MS").values,
    )
    for date in dates:
        # Feature equivalence is stricter than forecast equivalence: a tiny tree
        # might ignore an incorrectly reconstructed lag.
        np.testing.assert_array_equal(
            prediction_spatial_features(bridged, pre, date),
            prediction_spatial_features(datasets, pre, date),
        )
    expected = predict_pressure(model, pre, datasets, dates, 5, threads=1)
    actual = infer_pressure(model_dir, test_path)
    np.testing.assert_array_equal(actual.tp_mm_day.values, expected)
    np.testing.assert_array_equal(actual.time.values, dates.values)
    assert actual.tp_mm_day.dtype == np.float32
    assert actual.tp_mm_day.attrs["units"] == "mm/day"
    assert set(actual.data_vars) == {"tp_mm_day"}


@pytest.mark.parametrize("changed", ["model.txt", "atmosphere_history.nc", "test.nc"])
def test_standalone_pressure_rejects_changed_inputs(pressure_package, changed):
    model_dir, test_path, *_ = pressure_package
    path = test_path if changed == "test.nc" else model_dir / changed
    with path.open("ab") as stream:
        stream.write(b"changed-after-freeze")
    with pytest.raises(ValueError, match="mudou"):
        infer_pressure(model_dir, test_path)
