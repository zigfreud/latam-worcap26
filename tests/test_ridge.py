import numpy as np
import pandas as pd
import xarray as xr
import yaml
from sklearn.linear_model import Ridge

from worcap.experiment import ATMOSPHERIC, evaluate_fold, sha256
from worcap.ridge import coefficients, fit_local, predict_local, run_ridge, source_dates


def test_month_alignment_crosses_year_boundary():
    dates = source_dates(pd.to_datetime(["2021-01-01", "2021-02-01"]))
    assert list(dates.strftime("%Y-%m")) == ["2020-12", "2021-01"]


def test_batched_ridge_matches_sklearn_and_validation_does_not_change_fit():
    rng = np.random.default_rng(42)
    months = np.tile(np.arange(1, 13), 5)
    targets = months % 12 + 1
    x = rng.normal(size=(60, 2, 3))
    y = 4 + 2*x[:,:,0] - x[:,:,1] + rng.normal(size=(60,2))*.1
    clim = np.full((12,2), 4.)
    model = fit_local(x, y, months, targets, clim)
    beta = coefficients(model, 10.)
    for p in range(2):
        z = (x[:,p] - model["means"][months-1,p]) / model["scale"][p]
        reference = Ridge(alpha=10).fit(z, y[:,p]-4)
        np.testing.assert_allclose(beta[p], reference.coef_, atol=1e-10)
        predicted, _ = predict_local(model, beta, x, months, targets)
        np.testing.assert_allclose(predicted[:,p], np.maximum(0, 4+reference.predict(z)), atol=1e-6)
    before = model["means"].copy()
    predict_local(model, beta, x*10000, months, targets)
    np.testing.assert_array_equal(before, model["means"])


def test_constant_features_produce_finite_climatology():
    months = np.tile(np.arange(1,13), 2)
    model = fit_local(np.ones((24,1,2)), np.full((24,1), 3.), months, months,
                      np.full((12,1), 3.))
    beta = coefficients(model, 100)
    pred, clipped = predict_local(model, beta, np.ones((24,1,2)), months, months)
    np.testing.assert_array_equal(pred, 3.)
    assert clipped == 0


def test_complete_experiment_and_saved_model_reproduce_predictions(tmp_path):
    import json

    rng = np.random.default_rng(7)
    dates = pd.date_range("2000-01-01", "2010-12-01", freq="MS")
    coords = {"time": dates, "lat": [-30., -29.], "lon": [-53., -52., -51.]}
    raw = tmp_path / "raw"
    raw.mkdir()
    fields = []
    for name in ATMOSPHERIC:
        values = rng.normal(size=(len(dates),2,3)).astype(np.float32)
        fields.append(values)
        xr.Dataset({name: (("time","lat","lon"), values)}, coords=coords).to_netcdf(
            raw / f"treino_{name}.nc")
    rain = np.maximum(0, 4 + np.roll(fields[0], 1, axis=0))
    tp = xr.DataArray(rain, dims=("time","lat","lon"), coords=coords, name="tp",
                      attrs={"units":"mm/day"})
    tp.to_netcdf(raw / "treino_tp.nc")
    fold = {"train_end":2008, "validation_start":2009, "validation_end":2010}
    baseline_result, _, _ = evaluate_fold(tp, fold, 3)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"folds":[baseline_result], "input_sha256":{
        "treino_tp.nc":sha256(raw / "treino_tp.nc")}}), encoding="utf-8")
    config = tmp_path / "ridge.yaml"
    config.write_text(yaml.safe_dump({"data_dir":str(raw), "baseline_results":str(baseline),
        "window_years":3, "inner_validation_years":2, "alphas":[10.,100.],
        "latitude_batch":1, "clip_min":0., "folds":[fold]}), encoding="utf-8")
    output = tmp_path / "run"
    result = run_ridge(config, output)
    with xr.open_dataset(output / "2009-2010/model.nc") as ds:
        model = {"means": ds.feature_month_mean.values.reshape(12,6,9),
                 "scale": ds.feature_scale.values.reshape(6,9),
                 "intercept": ds.intercept.values.reshape(6),
                 "climatology": ds.target_climatology.values.reshape(12,6)}
        beta = ds.beta.values.reshape(6,9)
    target = pd.date_range("2009-01-01", "2010-12-01", freq="MS")
    source = source_dates(target)
    positions = dates.get_indexer(source)
    x = np.stack([v[positions].reshape(24,6) for v in fields], axis=-1)
    reproduced, _ = predict_local(model, beta, x, source.month.to_numpy(), target.month.to_numpy())
    with xr.open_dataset(output / "2009-2010/predictions.nc") as saved:
        np.testing.assert_array_equal(reproduced.reshape(24,2,3), saved.tp_mm_day.values)
    error = reproduced.astype(float) - tp.sel(time=target).values.reshape(24,6)
    assert np.isclose(np.sqrt(np.square(error).mean()), result["ridge_rmse"])
