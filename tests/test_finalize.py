import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from worcap.experiment import ATMOSPHERIC
from worcap.finalize import export_csv, forecast_saved, run_final
from worcap.finalize import test_sources as adapt_test_sources
from worcap.ridge import source_dates


def test_test_calendar_requires_exact_source_and_ignores_rain():
    dates = pd.date_range("2023-01-01", periods=2, freq="MS")
    coords = {"time": dates, "lat": [-20.], "lon": [-50., -49.],
              "time_origem": ("time", source_dates(dates))}
    ds = xr.Dataset({n: (("time", "lat", "lon"), np.arange(4).reshape(2, 1, 2))
                     for n in ATMOSPHERIC}, coords=coords)
    ds["tp_alvo"] = xr.full_like(ds[ATMOSPHERIC[0]], np.nan, dtype=float)
    ds["tp_ultima_obs"] = ds.tp_alvo
    sources = adapt_test_sources(ds, dates, ds.lat.values, ds.lon.values)
    assert set(sources) == set(ATMOSPHERIC)
    np.testing.assert_array_equal(sources["t2"].time, source_dates(dates))
    np.testing.assert_array_equal(sources["t2"].t2.sel(time="2022-12-01"), [[0, 1]])
    with pytest.raises(ValueError, match="time_origem"):
        adapt_test_sources(ds.assign_coords(time_origem=("time", dates)), dates,
                           ds.lat.values, ds.lon.values)
    with pytest.raises(ValueError, match="lat"):
        adapt_test_sources(ds, dates, [-21.], ds.lon.values)


def test_csv_rejects_reordering_and_invalid_predictions(tmp_path):
    dates = pd.date_range("2023-01-01", periods=2, freq="MS")
    coords = {"time": dates, "lat": [-20.], "lon": [-50., -49.]}
    ds = xr.Dataset({"tp_mm_day": (("time", "lat", "lon"),
                                    np.array([0, 1/3, 2, 10], dtype=np.float32).reshape(2, 1, 2))},
                    coords=coords)
    sample = tmp_path / "sample.csv"
    ids = [f"{d:%Y_%m}_{a:.2f}_{b:.2f}" for d in dates for a in coords["lat"] for b in coords["lon"]]
    pd.DataFrame({"id": ids, "tp_mm_day": 0}).to_csv(sample, index=False)
    report = export_csv(ds, sample, tmp_path / "good.csv")
    assert report["rows"] == 4 and report["float32_roundtrip_exact"]
    pd.DataFrame({"id": ids[::-1], "tp_mm_day": 0}).to_csv(sample, index=False)
    with pytest.raises(ValueError, match="IDs/ordem"):
        export_csv(ds, sample, tmp_path / "reordered.csv")
    assert not (tmp_path / "reordered.csv").exists()
    ds.tp_mm_day.values[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="Previsões inválidas"):
        export_csv(ds, sample, tmp_path / "nan.csv")


def test_full_final_selection_refit_and_independent_inference(tmp_path):
    rng = np.random.default_rng(19)
    dates = pd.date_range("1999-01-01", "2010-12-01", freq="MS")
    forecast = pd.date_range("2011-01-01", "2012-12-01", freq="MS")
    coords = {"time": dates, "lat": [-30., -29.], "lon": [-53., -52., -51.]}
    raw = tmp_path / "raw"
    raw.mkdir()
    test = xr.Dataset(coords={**coords, "time": forecast,
                               "time_origem": ("time", source_dates(forecast))})
    rain = None
    for name in ATMOSPHERIC:
        values = rng.normal(size=(len(dates), 2, 3)).astype(np.float32)
        if rain is None:
            rain = 3 + 0.4*np.roll(values, 1, axis=0)
        xr.Dataset({name: (("time", "lat", "lon"), values)}, coords=coords).to_netcdf(
            raw / f"treino_{name}.nc")
        test[name] = (("time", "lat", "lon"), rng.normal(size=(24, 2, 3)).astype(np.float32))
        test[name].values[0] = values[-1]
    test["tp_alvo"] = xr.full_like(test.t2, np.nan)
    test["tp_ultima_obs"] = xr.full_like(test.t2, -999.)
    test.to_netcdf(raw / "teste_features.nc")
    xr.DataArray(rain, name="tp", dims=("time", "lat", "lon"), coords=coords,
                 attrs={"units": "mm/day"}).to_netcdf(raw / "treino_tp.nc")
    ids = [f"{d:%Y_%m}_{a:.2f}_{b:.2f}" for d in forecast
           for a in coords["lat"] for b in coords["lon"]]
    pd.DataFrame({"id": ids, "tp_mm_day": 0.}).to_csv(raw / "sample_submission.csv", index=False)
    cfg = {"data_dir": str(raw), "train_end": 2010, "window_years": 3,
           "inner_validation_years": 2, "forecast_start": "2011-01-01",
           "forecast_end": "2012-12-01", "alphas": [1, 10], "iterations": [5, 10],
           "lightgbm_weights": [0, 0.5, 1], "latitude_batch": 1, "training_points": 4,
           "seed": 19, "clip_min": 0., "params": {"objective": "regression",
           "num_leaves": 7, "min_data_in_leaf": 5, "num_threads": 1, "verbosity": -1,
           "deterministic": True, "force_col_wise": True}}
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    out = tmp_path / "final"
    report = run_final(config, out)
    selection = report["selection"]
    assert selection["training_target_end"] == "2008-12-01"
    assert selection["validation_end"] == "2010-12-01"
    assert report["recipe"]["training_target_start"] == "2008-01-01"
    assert selection["ridge_alpha"] == min(selection["ridge_candidates"], key=lambda c: c["rmse"])["alpha"]
    assert selection["lightgbm_iterations"] == min(selection["lightgbm_candidates"],
                                                  key=lambda c: c["rmse"])["iterations"]
    assert selection["lightgbm_weight"] == min(selection["weight_candidates"],
                                              key=lambda c: c["rmse"])["lightgbm_weight"]
    assert report["verification"]["rows"] == 144
    assert report["verification"]["all_months_models_reproduced_exactly"]
    # Remove access to historical inputs before reusing the saved final bundle.
    for path in raw.glob("treino_*.nc"):
        path.rename(path.with_suffix(".hidden"))
    rerun = tmp_path / "inference"
    forecast_saved(out, raw / "teste_features.nc", raw / "sample_submission.csv", rerun)
    assert (out / "candidate_unsubmitted.csv").read_bytes() == (rerun / "candidate_unsubmitted.csv").read_bytes()
    assert not json.loads((rerun / "verification.json").read_text())["submitted"]
    with pytest.raises(ValueError, match="já existe"):
        run_final(config, out)
