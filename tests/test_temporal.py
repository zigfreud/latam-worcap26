import numpy as np
import pandas as pd
import pytest
import xarray as xr

from worcap.boosting import prepare_training
from worcap.experiment import ATMOSPHERIC
from worcap.temporal import (
    bridge_test_atmosphere,
    prediction_temporal_features,
    prepare_temporal_training,
)


def fixture():
    dates = pd.date_range("1999-01-01", "2004-12-01", freq="MS")
    coords = {"time": dates, "lat": [-20.], "lon": [-50., -49.]}
    values = np.arange(len(dates)*2, dtype=np.float32).reshape(-1, 1, 2)
    tp = xr.DataArray(values/100+2, dims=("time", "lat", "lon"), coords=coords, name="tp")
    datasets = {n: xr.Dataset({n: (("time", "lat", "lon"), values.copy())}, coords=coords) for n in ATMOSPHERIC}
    return tp, datasets


def test_lags_dates_difference_and_future_isolation():
    tp, datasets = fixture()
    points = np.array([0, 1])
    x, y, pre = prepare_temporal_training(datasets, tp, 2000, 2002, points, 3)
    base, target, _ = prepare_training(datasets, tp, 2000, 2002, points)
    np.testing.assert_array_equal(x[:, :14], base)
    np.testing.assert_array_equal(y, target)
    assert x.shape == (72, 50)
    np.testing.assert_array_equal(x[:, 23:32], x[:, :9]-x[:, 14:23])
    date = pd.Timestamp("2003-01-01")
    f = prediction_temporal_features(datasets, pre, date)
    for lag, col in [(2, 14), (3, 32)]:
        source = date-pd.DateOffset(months=lag)
        expected = datasets['t2'].t2.sel(time=source).values.ravel()-pre.lag_feature_month_mean.sel(
            lag=lag, month=source.month, feature='t2').values.ravel()
        np.testing.assert_array_equal(f[:, col], expected)
    for n in ATMOSPHERIC:
        datasets[n][n].loc[{"time": slice("2003-01-01", None)}] = -99999
    np.testing.assert_array_equal(prediction_temporal_features(datasets, pre, date), f)
    x2, y2, pre2 = prepare_temporal_training(datasets, tp, 2000, 2002, points, 3)
    np.testing.assert_array_equal(x, x2)
    np.testing.assert_array_equal(y, y2)
    xr.testing.assert_equal(pre, pre2)


def test_test_bridge_uses_source_dates_and_checks_overlap():
    _tp, datasets = fixture()
    origins = pd.date_range("2002-12-01", "2003-02-01", freq="MS")
    targets = origins+pd.DateOffset(months=1)
    test = xr.Dataset({n: datasets[n][n].sel(time=origins).assign_coords(time=targets) for n in ATMOSPHERIC})
    test = test.assign_coords(time_origem=("time", origins))
    test["tp_alvo"] = xr.full_like(test.t2, np.nan)
    training = {n: ds.sel(time=slice(None, '2002-12-01')) for n, ds in datasets.items()}
    result = bridge_test_atmosphere(training, test, 3)
    assert result['t2'].time.values[0] == pd.Timestamp('2002-10-01')
    assert set(result) == set(ATMOSPHERIC)
    test['t2'].values[0] += 1
    with pytest.raises(ValueError, match="Sobreposição"):
        bridge_test_atmosphere(training, test, 3)


@pytest.mark.parametrize("memory", [2, 3])
def test_memory_model_and_preprocessing_reload(tmp_path, memory):
    import lightgbm as lgb

    from worcap.boosting import predict_month, train_model

    tp, datasets = fixture()
    x, y, pre = prepare_temporal_training(datasets, tp, 2000, 2002, np.array([0, 1]), memory)
    config = {"memory_months": memory, "seed": 12,
              "params": {"objective": "regression", "num_threads": 1, "verbosity": -1,
                         "num_leaves": 3, "min_data_in_leaf": 2}}
    model = train_model(x, y, config, 5)
    date = pd.Timestamp('2003-01-01')
    prediction, _ = predict_month(model, prediction_temporal_features(datasets, pre, date), 5, 1)
    model.save_model(str(tmp_path/'model.txt'))
    pre.to_netcdf(tmp_path/'pre.nc')
    saved = lgb.Booster(model_file=str(tmp_path/'model.txt'))
    with xr.open_dataset(tmp_path/'pre.nc') as restored:
        actual, _ = predict_month(saved, prediction_temporal_features(datasets, restored, date), 5, 1)
    np.testing.assert_array_equal(prediction, actual)
