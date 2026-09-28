import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from worcap.boosting import (
    FEATURES,
    predict_month,
    prediction_features,
    prepare_training,
    run_boosting,
)
from worcap.experiment import ATMOSPHERIC, evaluate_fold, sha256


def test_temporal_features_and_training_are_independent_of_future():
    dates = pd.date_range("1999-01-01", "2004-12-01", freq="MS")
    coords = {"time": dates, "lat": [-20.], "lon": [-50., -49.]}
    values = np.arange(len(dates)*2, dtype=np.float32).reshape(-1, 1, 2)
    tp = xr.DataArray(2+values/100, dims=("time", "lat", "lon"), coords=coords, name="tp")
    datasets = {name: xr.Dataset({name: (("time", "lat", "lon"), values.copy())}, coords=coords)
                for name in ATMOSPHERIC}
    x, y, pre = prepare_training(datasets, tp, 2000, 2002, np.array([0, 1]))
    expected = values[dates.get_loc("2002-12-01")].ravel() - pre.feature_month_mean.sel(
        month=12).values[:, :, 0].ravel()
    pred_x = prediction_features(datasets, pre, pd.Timestamp("2003-01-01"))
    np.testing.assert_allclose(pred_x[:, 0], expected)
    np.testing.assert_allclose(pred_x[:, 11:13], [[0, 1], [0, 1]])
    # Future values may alter forecasts, but cannot change fitted preprocessing or training rows.
    tp.loc[{"time": slice("2003", None)}] = 99999
    for name, ds in datasets.items():
        ds[name].loc[{"time": slice("2003", None)}] = 99999
    x2, y2, pre2 = prepare_training(datasets, tp, 2000, 2002, np.array([0, 1]))
    np.testing.assert_array_equal(x, x2)
    np.testing.assert_array_equal(y, y2)
    xr.testing.assert_equal(pre, pre2)
    assert x.shape == (72, len(FEATURES))


@pytest.mark.parametrize("with_ocean", [False, True])
def test_complete_boosting_selection_and_reloaded_model(tmp_path, with_ocean):
    rng = np.random.default_rng(19)
    dates = pd.date_range("1999-01-01", "2010-12-01", freq="MS")
    coords = {"time": dates, "lat": [-30., -29.], "lon": [-53., -52., -51.]}
    raw = tmp_path / "raw"
    raw.mkdir()
    datasets = {}
    for name in ATMOSPHERIC:
        values = rng.normal(size=(len(dates), 2, 3)).astype(np.float32)
        datasets[name] = xr.Dataset({name: (("time", "lat", "lon"), values)}, coords=coords)
        datasets[name].to_netcdf(raw / f"treino_{name}.nc")
    rain = 2 + np.square(np.roll(datasets[ATMOSPHERIC[0]][ATMOSPHERIC[0]].values, 1, axis=0))
    tp = xr.DataArray(rain, dims=("time", "lat", "lon"), coords=coords, name="tp",
                      attrs={"units": "mm/day"})
    tp.to_netcdf(raw / "treino_tp.nc")
    fold = {"train_end": 2008, "validation_start": 2009, "validation_end": 2010}
    baseline_result, _, _ = evaluate_fold(tp, fold, 3)
    hashes = {"treino_tp.nc": sha256(raw / "treino_tp.nc")}
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"folds": [baseline_result], "input_sha256": hashes,
                                   "rmse_mm_day": baseline_result["rmse"]}))
    ridge = tmp_path / "ridge.json"
    ridge.write_text(json.dumps({"folds": [{"fold": "2009-2010", "ridge_rmse": 1.5}],
                                "ridge_rmse": 1.5, "input_sha256": hashes}))
    config = tmp_path / "lightgbm.yaml"
    config.write_text(yaml.safe_dump({"data_dir": str(raw), "baseline_results": str(baseline),
        "ridge_results": str(ridge), "window_years": 3, "inner_validation_years": 2,
        "training_points": 4, "seed": 19, "iterations": [5, 10], "clip_min": 0.,
        "params": {"objective": "regression", "num_leaves": 7, "min_data_in_leaf": 5,
                   "num_threads": 1, "verbosity": -1, "deterministic": True,
                   "force_col_wise": True}, "folds": [fold]}), encoding="utf-8")
    ocean = None
    if with_ocean:
        from worcap.ocean import read_sst
        path = tmp_path / "sstoi.indices"
        lines = ["YR MON NINO12 ANOM NINO3 ANOM NINO4 ANOM NINO34 ANOM"]
        for date in dates:
            values = 25 + rng.normal(size=4)
            lines.append(f"{date.year} {date.month} " + " ".join(f"{v} 0" for v in values))
        path.write_text("\n".join(lines)+"\n")
        ocean = read_sst(path)
        cfg = yaml.safe_load(config.read_text())
        cfg.update({"experiment": "E03", "ocean": {"path": str(path), "sha256": sha256(path),
                    "lag_months": 2}, "control_results": str(baseline)})
        # Synthetic control with a LightGBM score, distinct from the E00 schema.
        control = tmp_path / "control.json"
        control.write_text(json.dumps({"folds": [{"fold": "2009-2010", "lightgbm_rmse": 1.5}],
                                       "lightgbm_rmse": 1.5, "input_sha256": hashes}))
        cfg["control_results"] = str(control)
        config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    out = tmp_path / "run"
    report = run_boosting(config, out)
    chosen = report["folds"][0]["chosen_iterations"]
    selection = json.loads((out / "2009-2010/selection.json").read_text())
    assert chosen == min(selection["candidates"], key=lambda r: r["rmse"])["iterations"]
    assert selection["validation_end"] == "2008-12-01"
    assert report["folds"][0]["count"] == 24*6  # Entire grid, not four sampled points.
    model = lgb.Booster(model_file=str(out / "2009-2010/model.txt"))
    with xr.open_dataset(out / "2009-2010/preprocessing.nc") as pre:
        rows = prediction_features(datasets, pre, pd.Timestamp("2009-01-01"), ocean)
        pred, _ = predict_month(model, rows, chosen, 1)
    with xr.open_dataset(out / "2009-2010/predictions.nc") as ds:
        np.testing.assert_array_equal(pred, ds.tp_mm_day.values[0].ravel())
        error = ds.tp_mm_day.values.astype(float) - tp.sel(time=ds.time).values
        assert report["lightgbm_rmse"] == float(np.sqrt(np.square(error).mean()))
