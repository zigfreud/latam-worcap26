import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import Polygon, box

from worcap.boosting import predict_month, train_model
from worcap.experiment import ATMOSPHERIC, sha256
from worcap.geography import (
    GEO_FEATURES,
    assign_categories,
    prediction_geographic_features,
    prepare_geographic_training,
)
from worcap.spatial import prepare_spatial_training


def test_polygon_holes_boundaries_and_outside():
    outer = Polygon([(0, 0), (3, 0), (3, 3), (0, 3)],
                    holes=[[(1, 1), (2, 1), (2, 2), (1, 2)]])
    values, overlaps = assign_categories([outer, box(3, 0, 4, 3)], [9, 2],
                                         np.array([.5, 1.5]), np.array([-1, .5, 1.5, 3, 3.5]))
    np.testing.assert_array_equal(values, [[0, 9, 9, 2, 2], [0, 9, 0, 2, 2]])
    assert overlaps == 2


def test_geographic_features_alignment_categorical_reload_and_hash(tmp_path):
    dates = pd.date_range("1999-01-01", "2004-12-01", freq="MS")
    coords = {"time": dates, "lat": [-20., -19.], "lon": [-50., -49., -48.]}
    rng = np.random.default_rng(17)
    values = rng.normal(size=(len(dates), 2, 3)).astype(np.float32)
    tp = xr.DataArray(np.abs(values)+3, dims=("time", "lat", "lon"), coords=coords)
    datasets = {n: xr.Dataset({n: (("time", "lat", "lon"), values.copy()+j)}, coords=coords)
                for j, n in enumerate(ATMOSPHERIC)}
    codes = np.array([[0, 11, 35], [11, 35, 0]], dtype=np.int32)
    geo = xr.Dataset({k: (("lat", "lon"), codes) for k in GEO_FEATURES},
                     coords={"lat": coords["lat"], "lon": coords["lon"]})
    path = tmp_path/"geo.nc"
    geo.to_netcdf(path)
    digest = sha256(path)
    points = np.arange(6)
    x, y, pre = prepare_geographic_training(datasets, tp, 2000, 2002, points, path, digest)
    base, target, _ = prepare_spatial_training(datasets, tp, 2000, 2002, points, "pressure")
    np.testing.assert_array_equal(x[:, :53], base)
    np.testing.assert_array_equal(y, target)
    expected = np.tile(np.stack([codes.ravel()]*3, axis=-1), (36, 1))
    np.testing.assert_array_equal(x[:, 53:], expected)
    first = prediction_geographic_features(datasets, pre, pd.Timestamp("2000-01-01"))
    np.testing.assert_array_equal(first, x[:6])
    cfg = {"memory_months": 3, "spatial_arm": "pressure", "geography": {"path": str(path)},
           "seed": 17, "params": {"objective": "regression", "num_threads": 1,
                                  "num_leaves": 3, "min_data_in_leaf": 2, "verbosity": -1}}
    model = train_model(x, y, cfg, 5)
    assert model.num_feature() == 56
    info = model.dump_model()["feature_infos"]
    assert all(info[k]["values"] for k in GEO_FEATURES)
    date = pd.Timestamp("2003-01-01")
    f = prediction_geographic_features(datasets, pre, date)
    for n in ATMOSPHERIC:
        datasets[n][n].loc[{"time": slice("2003-01-01", None)}] = 100000
    np.testing.assert_array_equal(prediction_geographic_features(datasets, pre, date), f)
    pred, _ = predict_month(model, f, 5, 1)
    model.save_model(str(tmp_path/"model.txt"))
    pre.to_netcdf(tmp_path/"pre.nc")
    restored = lgb.Booster(model_file=str(tmp_path/"model.txt"))
    with xr.open_dataset(tmp_path/"pre.nc") as saved:
        actual, _ = predict_month(restored, prediction_geographic_features(datasets, saved, date), 5, 1)
    np.testing.assert_array_equal(actual, pred)
    with pytest.raises(ValueError, match="Mapa geográfico mudou"):
        prepare_geographic_training(datasets, tp, 2000, 2002, points, path, "incorrect")
