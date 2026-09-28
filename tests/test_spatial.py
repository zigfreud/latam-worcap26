import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from worcap.boosting import predict_month, train_model
from worcap.experiment import ATMOSPHERIC
from worcap.spatial import local_mean, prediction_spatial_features, prepare_spatial_training
from worcap.temporal import prediction_temporal_features, prepare_temporal_training


def test_local_mean_edges_and_no_time_mixing():
    field = np.arange(84, dtype=np.float32).reshape(2, 6, 7)
    expected = np.empty_like(field)
    for t in range(2):
        for y in range(6):
            for x in range(7):
                expected[t, y, x] = field[t, max(0, y-2):y+3, max(0, x-2):x+3].mean()
    np.testing.assert_allclose(local_mean(field), expected, atol=1e-6)


@pytest.mark.parametrize("arm,width", [("spatial", 59), ("pressure", 53)])
def test_spatial_alignment_isolation_and_reload(tmp_path, arm, width):
    dates = pd.date_range("1999-01-01", "2004-12-01", freq="MS")
    coords = {"time": dates, "lat": np.arange(6), "lon": np.arange(7)}
    rng = np.random.default_rng(81)
    values = rng.normal(size=(len(dates), 6, 7)).astype(np.float32)
    tp = xr.DataArray(np.abs(values), dims=("time", "lat", "lon"), coords=coords)
    ds = {n: xr.Dataset({n: (("time", "lat", "lon"), values.copy()+j)}, coords=coords)
          for j, n in enumerate(ATMOSPHERIC)}
    points = np.array([0, 10, 41])
    base, target, _ = prepare_temporal_training(ds, tp, 2000, 2002, points, 3)
    x, y, pre = prepare_spatial_training(ds, tp, 2000, 2002, points, arm)
    assert x.shape == (108, width)
    np.testing.assert_array_equal(x[:, :50], base)
    np.testing.assert_array_equal(y, target)
    train_dates = pd.date_range("2000-01-01", "2002-12-01", freq="MS")
    for i in (0, 11, 35):
        f = prediction_spatial_features(ds, pre, train_dates[i])
        np.testing.assert_array_equal(f[points], x.reshape(36, 3, width)[i])
    date = pd.Timestamp("2003-01-01")
    f = prediction_spatial_features(ds, pre, date)
    np.testing.assert_array_equal(f[:, :50], prediction_temporal_features(ds, pre, date))
    for n in ATMOSPHERIC:
        ds[n][n].loc[{"time": slice("2003-01-01", None)}] = -100000
    np.testing.assert_array_equal(prediction_spatial_features(ds, pre, date), f)
    x2, y2, pre2 = prepare_spatial_training(ds, tp, 2000, 2002, points, arm)
    np.testing.assert_array_equal(x, x2)
    np.testing.assert_array_equal(y, y2)
    xr.testing.assert_equal(pre, pre2)
    config = {"memory_months": 3, "spatial_arm": arm, "seed": 81,
              "params": {"objective": "regression", "num_threads": 1, "verbosity": -1,
                         "num_leaves": 3, "min_data_in_leaf": 2}}
    model = train_model(x, y, config, 5)
    pred, _ = predict_month(model, f, 5, 1)
    model.save_model(str(tmp_path/"model.txt"))
    pre.to_netcdf(tmp_path/"pre.nc")
    restored = lgb.Booster(model_file=str(tmp_path/"model.txt"))
    with xr.open_dataset(tmp_path/"pre.nc") as saved:
        actual, _ = predict_month(restored, prediction_spatial_features(ds, saved, date), 5, 1)
    np.testing.assert_array_equal(actual, pred)
