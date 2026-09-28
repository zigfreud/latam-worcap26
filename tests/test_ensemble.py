from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from worcap.boosting import run_boosting
from worcap.ensemble import blend, choose_weight, json_read, run_ensemble
from worcap.experiment import ATMOSPHERIC, evaluate_fold, sha256
from worcap.ridge import run_ridge


def test_convex_blend_cancels_complementary_errors_and_preserves_endpoints():
    r, g, y = np.array([1., 3.]), np.array([3., 1.]), np.array([2., 2.])
    weight, candidates = choose_weight(r, g, y, [0, .25, .5, .75, 1])
    assert weight == .5 and candidates[2]["rmse"] == 0
    np.testing.assert_array_equal(blend(r, g, 0), r)
    np.testing.assert_array_equal(blend(r, g, 1), g)
    assert choose_weight(r, r, y, [1, .5, 0])[0] == 0  # Explicit tie policy.
    with pytest.raises(ValueError, match="entre"):
        blend(r, g, 1.1)
    with pytest.raises(ValueError, match="desalinhados"):
        blend(r, g[:1], .5)


@pytest.mark.parametrize("variant", ["base", "memory3", "pressure"])
def test_complete_ensemble_from_inner_selection_and_saved_outer_components(tmp_path, variant):
    import json

    rng = np.random.default_rng(31)
    dates = pd.date_range("2000-01-01", "2010-12-01", freq="MS")
    coords = {"time": dates, "lat": [-30., -29.], "lon": [-53., -52., -51.]}
    raw = tmp_path / "raw"
    raw.mkdir()
    fields = []
    for name in ATMOSPHERIC:
        values = rng.normal(size=(len(dates), 2, 3)).astype(np.float32)
        fields.append(values)
        xr.Dataset({name: (("time", "lat", "lon"), values)}, coords=coords).to_netcdf(raw / f"treino_{name}.nc")
    tp = xr.DataArray(np.maximum(0, 3+np.roll(fields[0], 1, axis=0)), dims=("time", "lat", "lon"),
                      coords=coords, name="tp", attrs={"units": "mm/day"})
    tp.to_netcdf(raw / "treino_tp.nc")
    fold = {"train_end": 2008, "validation_start": 2009, "validation_end": 2010}
    ref, _, _ = evaluate_fold(tp, fold, 3)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"folds": [ref], "rmse_mm_day": ref["rmse"],
                                   "input_sha256": {"treino_tp.nc": sha256(raw / "treino_tp.nc")}}))
    rcfg = {"data_dir": str(raw), "baseline_results": str(baseline), "window_years": 3,
            "inner_validation_years": 2, "alphas": [10., 100.], "latitude_batch": 1,
            "clip_min": 0., "folds": [fold]}
    rpath, rdir = tmp_path / "ridge.yaml", tmp_path / "ridge"
    rpath.write_text(yaml.safe_dump(rcfg))
    run_ridge(rpath, rdir)
    gcfg = {**rcfg, "ridge_results": str(rdir / "results.json"), "training_points": 4,
            "seed": 31, "iterations": [5, 10], "params": {"objective": "regression",
            "num_leaves": 7, "min_data_in_leaf": 5, "num_threads": 1,
            "deterministic": True, "force_col_wise": True, "verbosity": -1}}
    gpath, gdir = tmp_path / "boosting.yaml", tmp_path / "boosting"
    if variant in ("memory3", "pressure"):
        gcfg["memory_months"] = 3
    if variant == "pressure":
        gcfg["spatial_arm"] = "pressure"
    gpath.write_text(yaml.safe_dump(gcfg))
    run_boosting(gpath, gdir)
    config = tmp_path / "ensemble.yaml"
    config.write_text(yaml.safe_dump({"ridge_dir": str(rdir), "boosting_dir": str(gdir),
                                     "latitude_batch": 1, "lightgbm_weights": [0, .5, 1]}))
    out = tmp_path / "ensemble"
    report = run_ensemble(config, out)
    folder = out / "2009-2010"
    selection = json_read(folder / "selection.json")
    assert selection["validation_end"] == "2008-12-01"
    with xr.open_dataset(folder / "inner_predictions.nc") as inner:
        chosen, _ = choose_weight(inner.ridge.values, inner.lightgbm.values,
                                   tp.sel(time=inner.time).values, [0, .5, 1])
    assert chosen == selection["lightgbm_weight"]
    with xr.open_dataset(folder / "predictions.nc") as pred:
        expected = float(np.sqrt(np.square(pred.tp_mm_day.values.astype(float)-tp.sel(time=pred.time).values).mean()))
    assert report["ensemble_rmse"] == expected
    checks = json_read(out / "verification.json")
    assert checks[0]["first_month_from_models_and_recipe_exact"]
    recipe = json_read(folder / "model_recipe.json")
    for path, digest in recipe["component_artifacts_sha256"].items():
        assert sha256(Path(path)) == digest
    if variant == "pressure":
        cached_config = yaml.safe_load(config.read_text())
        cached_config["inner_ridge_cache"] = str(out)
        cached_path = tmp_path/"cached.yaml"
        cached_path.write_text(yaml.safe_dump(cached_config))
        cached_report = run_ensemble(cached_path, tmp_path/"cached")
        assert cached_report["ensemble_rmse"] == report["ensemble_rmse"]
