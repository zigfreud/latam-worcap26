"""E04: peso global escolhido em previsões internas, sem SST ou alvos externos."""

import importlib.metadata
import json
import os
import shutil
import time
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from threadpoolctl import threadpool_limits

from .baseline import monthly_climatology
from .boosting import predict_month, prediction_features, prepare_training
from .experiment import ATMOSPHERIC, require, sha256
from .ridge import coefficients, fit_local, predict_local, read_x, source_dates


def blend(ridge, boosting, weight):
    require(np.isfinite(weight) and 0 <= weight <= 1, "Peso deve estar entre zero e um")
    require(ridge.shape == boosting.shape, "Componentes desalinhados")
    require(np.isfinite(ridge).all() and np.isfinite(boosting).all(), "Componentes não finitos")
    require((ridge >= 0).all() and (boosting >= 0).all(), "Componentes devem ser não negativos")
    return ((1-weight)*ridge.astype(np.float64)+weight*boosting.astype(np.float64)).astype(np.float32)


def choose_weight(ridge, boosting, truth, candidates):
    require(truth.shape == ridge.shape and np.isfinite(truth).all(), "Alvo interno inválido")
    weights = sorted(set(candidates))
    require(bool(weights), "Nenhum peso candidato")
    scores = []
    for weight in weights:
        pred = blend(ridge, boosting, weight)
        scores.append({"lightgbm_weight": float(weight),
                       "rmse": float(np.sqrt(np.square(pred.astype(float)-truth).mean()))})
    return min(scores, key=lambda row: row["rmse"])["lightgbm_weight"], scores


def json_read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def inner_ridge(datasets, tp, training, dates, alpha, step):
    sources, val_sources = source_dates(training), source_dates(dates)
    clim = monthly_climatology(tp, training[0].year, training[-1].year).values
    y = tp.sel(time=training).values
    pred = np.empty((len(dates), tp.sizes["lat"], tp.sizes["lon"]), dtype=np.float32)
    for lo in range(0, tp.sizes["lat"], step):
        sl = slice(lo, min(lo+step, tp.sizes["lat"]))
        x, vx = read_x(datasets, sources, sl), read_x(datasets, val_sources, sl)
        model = fit_local(x, y[:, sl].reshape(len(training), -1), sources.month.to_numpy(),
                          training.month.to_numpy(), clim[:, sl].reshape(12, -1))
        batch, _ = predict_local(model, coefficients(model, alpha), vx,
                                 val_sources.month.to_numpy(), dates.month.to_numpy())
        pred[:, sl] = batch.reshape(pred[:, sl].shape)
    return pred


def load_ridge_first_month(path, datasets, date):
    with xr.open_dataset(path) as ds:
        nlat, nlon = ds.sizes["lat"], ds.sizes["lon"]
        model = {"means": ds.feature_month_mean.values.reshape(12, -1, 9),
                 "scale": ds.feature_scale.values.reshape(-1, 9),
                 "intercept": ds.intercept.values.ravel(),
                 "climatology": ds.target_climatology.values.reshape(12, -1)}
        x = read_x(datasets, source_dates(pd.DatetimeIndex([date])), slice(0, nlat))
        pred, _ = predict_local(model, ds.beta.values.reshape(-1, 9), x,
                                 np.array([source_dates([date])[0].month]), np.array([date.month]))
        return pred.reshape(nlat, nlon)


def run_ensemble(config_path: Path, output: Path):
    require(not output.exists(), f"Saída já existe: {output}")
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    rdir, gdir = Path(cfg["ridge_dir"]), Path(cfg["boosting_dir"])
    rcfg = yaml.safe_load((rdir / "config.yaml").read_text(encoding="utf-8"))
    gcfg = yaml.safe_load((gdir / "source_snapshot/config.yaml").read_text(encoding="utf-8"))
    rref, gref = json_read(rdir / "results.json"), json_read(gdir / "results.json")
    require(not gcfg.get("ocean"), "E04 não permite SST")
    prepare, features = prepare_training, prediction_features
    if gcfg.get("spatial_arm"):
        from .spatial import prediction_spatial_features, prepare_spatial_training

        def prepare(datasets, tp, start, end, points):
            return prepare_spatial_training(datasets, tp, start, end, points, gcfg["spatial_arm"])

        features = prediction_spatial_features
    elif gcfg.get("memory_months", 1) > 1:
        from .temporal import prediction_temporal_features, prepare_temporal_training

        def prepare(datasets, tp, start, end, points):
            return prepare_temporal_training(datasets, tp, start, end, points, gcfg["memory_months"])

        features = prediction_temporal_features
    for key in ("folds", "window_years", "inner_validation_years", "data_dir", "clip_min"):
        require(rcfg[key] == gcfg[key], f"Configurações incompatíveis: {key}")
    require(cfg["latitude_batch"] > 0 and rcfg["clip_min"] == 0, "Configuração inválida")
    output.mkdir(parents=True)
    started = time.perf_counter()
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
    provenance = {}
    for path in [config_path, Path(cfg.get("protocol", "docs/E04_PROTOCOLO.md")), Path("pyproject.toml"),
                 Path("uv.lock"), *sorted(Path("src/worcap").glob("*.py"))]:
        rel = Path("config.yaml") if path == config_path else path
        dest = output / "source_snapshot" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        provenance[str(path)] = sha256(path)
    for name, root in [("ridge", rdir), ("lightgbm", gdir)]:
        shutil.copy2(root / "results.json", output / f"{name}_reference.json")
    points = np.load(gdir / "training_points.npy")
    require(sha256(gdir / "training_points.npy") == gref["training_points_sha256"], "Amostra mudou")
    raw = Path(rcfg["data_dir"])
    folds, monthly, maps, verification = [], [], [], []
    with ExitStack() as stack, threadpool_limits(limits=2):
        tp = stack.enter_context(xr.open_dataset(raw / "treino_tp.nc")).tp.load()
        require(tp.attrs.get("units") == "mm/day", "Unidade inesperada")
        datasets = {v: stack.enter_context(xr.open_dataset(raw / f"treino_{v}.nc")) for v in ATMOSPHERIC}
        inputs = {p.name: sha256(p) for p in [raw / "treino_tp.nc",
                  *[raw / f"treino_{v}.nc" for v in ATMOSPHERIC]]}
        for reference in (rref, gref):
            for filename, digest in inputs.items():
                require(reference["input_sha256"][filename] == digest, f"Entrada mudou: {filename}")
        for name, ds in datasets.items():
            for coord in ("time", "lat", "lon"):
                require(np.array_equal(ds[coord], tp[coord]), f"Grade/datas: {name}")
        nlat, nlon = tp.sizes["lat"], tp.sizes["lon"]
        for fold in rcfg["folds"]:
            label = f"{fold['validation_start']}-{fold['validation_end']}"
            folder = output / label
            folder.mkdir()
            rsel, gsel = json_read(rdir / label / "selection.json"), json_read(gdir / label / "selection.json")
            for key in ("training_target_start", "training_target_end", "validation_start", "validation_end"):
                require(rsel[key] == gsel[key], f"Seleções incompatíveis: {key}")
            training = pd.date_range(rsel["training_target_start"], rsel["training_target_end"], freq="MS")
            dates = pd.date_range(rsel["validation_start"], rsel["validation_end"], freq="MS")
            require(training[-1] < dates[0] and dates[-1].year <= fold["train_end"], "Vazamento temporal")
            require(fold["train_end"] < fold["validation_start"], "Bloco externo inválido")
            print(f"{label}: reconstruindo componentes internos {dates[0].year}-{dates[-1].year}", flush=True)
            if cfg.get("inner_ridge_cache"):
                cache = Path(cfg["inner_ridge_cache"])
                require(sha256(cache/"ridge_reference.json") == sha256(rdir/"results.json"), "Cache Ridge referência")
                cached_selection = json_read(cache/label/"selection.json")
                require(cached_selection["ridge_alpha"] == rsel["chosen_alpha"], "Cache alpha")
                path = cache/label/"inner_predictions.nc"
                with xr.open_dataset(path) as cached:
                    cached, _ = xr.align(cached, tp.sel(time=dates), join="exact")
                    require(np.array_equal(cached.time.values, dates.values), "Cache datas")
                    rp = cached.ridge.values
                provenance[str(path)] = sha256(path)
            else:
                rp = inner_ridge(datasets, tp, training, dates, rsel["chosen_alpha"], cfg["latitude_batch"])
            x, y, pre = prepare(datasets, tp, training[0].year, training[-1].year, points)
            del x, y
            internal_model = lgb.Booster(model_file=str(gdir / label / "inner_model.txt"))
            gp = np.empty_like(rp)
            for i, date in enumerate(dates):
                pred, _ = predict_month(internal_model, features(datasets, pre, date),
                                        gsel["chosen_iterations"], gcfg["params"]["num_threads"])
                gp[i] = pred.reshape(nlat, nlon)
            truth = tp.sel(time=dates).values
            for pred, selection, key, chosen in [(rp, rsel, "alpha", "chosen_alpha"),
                                                (gp, gsel, "iterations", "chosen_iterations")]:
                score = float(np.sqrt(np.square(pred.astype(float)-truth).mean()))
                expected = next(c["rmse"] for c in selection["candidates"] if c[key] == selection[chosen])
                require(np.isclose(score, expected, atol=1e-7, rtol=0), "RMSE interno não reproduzido")
            weight, candidates = choose_weight(rp, gp, truth, cfg["lightgbm_weights"])
            selection = {"lightgbm_weight": weight, "ridge_weight": 1-weight,
                         "candidates": candidates, "validation_start": str(dates[0].date()),
                         "validation_end": str(dates[-1].date()), "ridge_alpha": rsel["chosen_alpha"],
                         "lightgbm_iterations": gsel["chosen_iterations"], "internal_scores_reproduced": True}
            (folder / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
            internal = xr.Dataset({"ridge": (("time", "lat", "lon"), rp),
                                   "lightgbm": (("time", "lat", "lon"), gp)},
                                  coords={"time": dates, "lat": tp.lat, "lon": tp.lon})
            internal.to_netcdf(folder / "inner_predictions.nc", encoding={k: {"zlib": True} for k in internal})
            print(f"  peso congelado antes de avaliar bloco externo: LightGBM={weight:g}", flush=True)
            del internal, rp, gp, pre
            # Só depois de persistir a seleção são abertas as previsões e alvos externos.
            dates = pd.date_range(f"{fold['validation_start']}-01-01", f"{fold['validation_end']}-12-01", freq="MS")
            with xr.open_dataset(rdir / label / "predictions.nc") as rds, xr.open_dataset(
                    gdir / label / "predictions.nc") as gds:
                rds, gds, target = xr.align(rds, gds, tp.sel(time=dates), join="exact")
                require(np.array_equal(rds.time.values, dates.values), "Datas externas incorretas")
                rp, gp, truth = rds.tp_mm_day.values, gds.tp_mm_day.values, target.values
                prediction = blend(rp, gp, weight)
            re, ge, ee = rp.astype(float)-truth, gp.astype(float)-truth, prediction.astype(float)-truth
            scores = {"ridge_rmse": float(np.sqrt(np.square(re).mean())),
                      "lightgbm_rmse": float(np.sqrt(np.square(ge).mean())),
                      "ensemble_rmse": float(np.sqrt(np.square(ee).mean()))}
            rr = next(r for r in rref["folds"] if r["fold"] == label)
            gr = next(r for r in gref["folds"] if r["fold"] == label)
            for key, ref in [("ridge_rmse", rr), ("lightgbm_rmse", gr)]:
                require(np.isclose(scores[key], ref[key], atol=1e-12, rtol=0), "RMSE externo mudou")
            result = {"fold": label, "lightgbm_weight": weight, **scores,
                      "count": int(prediction.size), "squared_error_sum": float(np.square(ee).sum()),
                      "bias": float(ee.mean()),
                      "reduction_vs_lightgbm_pct": 100*(1-scores["ensemble_rmse"]/scores["lightgbm_rmse"]),
                      "reduction_vs_ridge_pct": 100*(1-scores["ensemble_rmse"]/scores["ridge_rmse"])}
            folds.append(result)
            coords = {"time": dates, "lat": tp.lat, "lon": tp.lon}
            saved = xr.Dataset({"tp_mm_day": (("time", "lat", "lon"), prediction)}, coords=coords)
            saved.tp_mm_day.attrs["units"] = "mm/day"
            saved.to_netcdf(folder / "predictions.nc", encoding={"tp_mm_day": {"zlib": True}})
            artifacts = [rdir / label / "model.nc", gdir / label / "model.txt",
                         gdir / label / "preprocessing.nc", gdir / label / "inner_model.txt",
                         rdir / label / "selection.json", gdir / label / "selection.json",
                         rdir / label / "predictions.nc", gdir / label / "predictions.nc"]
            recipe = {"lightgbm_weight": weight, "ridge_alpha": rsel["chosen_alpha"],
                      "lightgbm_iterations": gsel["chosen_iterations"],
                      "component_artifacts_sha256": {str(p.resolve()): sha256(p) for p in artifacts},
                      "ridge_model": str((rdir / label / "model.nc").resolve()),
                      "lightgbm_model": str((gdir / label / "model.txt").resolve()),
                      "lightgbm_preprocessing": str((gdir / label / "preprocessing.nc").resolve())}
            (folder / "model_recipe.json").write_text(json.dumps(recipe, indent=2), encoding="utf-8")
            recipe = json_read(folder / "model_recipe.json")
            rfirst = load_ridge_first_month(Path(recipe["ridge_model"]), datasets, dates[0])
            gb = lgb.Booster(model_file=recipe["lightgbm_model"])
            with xr.open_dataset(recipe["lightgbm_preprocessing"]) as pre:
                gfirst, _ = predict_month(gb, features(datasets, pre, dates[0]),
                                          recipe["lightgbm_iterations"], gcfg["params"]["num_threads"])
            require(np.array_equal(rfirst, rp[0]), "Ridge externo não reproduzido")
            require(np.array_equal(gfirst.reshape(nlat, nlon), gp[0]), "LightGBM externo não reproduzido")
            reconstructed = blend(rfirst, gfirst.reshape(nlat, nlon), recipe["lightgbm_weight"])
            require(np.array_equal(reconstructed, prediction[0]), "Receita não reproduzida")
            with xr.open_dataset(folder / "predictions.nc") as check:
                require(np.array_equal(check.tp_mm_day.values, prediction), "Previsões salvas diferem")
                require(float(np.sqrt(np.square(check.tp_mm_day.values.astype(float)-truth).mean()))
                        == scores["ensemble_rmse"], "RMSE salvo diverge")
            verification.append({"fold": label, "internal_scores_reproduced": True,
                                 "outer_component_scores_reproduced": True,
                                 "first_month_from_models_and_recipe_exact": True,
                                 "saved_predictions_and_rmse_exact": True,
                                 "predictions_sha256": sha256(folder / "predictions.nc"),
                                 "recipe_sha256": sha256(folder / "model_recipe.json")})
            for i, date in enumerate(dates):
                monthly.append({"fold": label, "target_month": str(date.date()),
                                "ensemble_rmse": float(np.sqrt(np.square(ee[i]).mean())),
                                "ridge_rmse": float(np.sqrt(np.square(re[i]).mean())),
                                "lightgbm_rmse": float(np.sqrt(np.square(ge[i]).mean()))})
            maps.append(xr.Dataset({f"{name}_rmse": (("lat", "lon"), np.sqrt(np.square(error).mean(axis=0)))
                        for name, error in [("ensemble", ee), ("ridge", re), ("lightgbm", ge)]},
                        coords={"lat": tp.lat, "lon": tp.lon}).expand_dims(fold=[label]))
            print(f"{label}: E04={scores['ensemble_rmse']:.6f}; E02={scores['lightgbm_rmse']:.6f}", flush=True)
            (output / "progress.json").write_text(json.dumps(folds, indent=2), encoding="utf-8")
    score = float(np.sqrt(sum(r["squared_error_sum"] for r in folds)/sum(r["count"] for r in folds)))
    report = {"experiment": cfg.get("experiment", "E04"), "created_utc": datetime.now(UTC).isoformat(),
              "ensemble_rmse": score, "ridge_rmse": rref["ridge_rmse"],
              "lightgbm_rmse": gref["lightgbm_rmse"], "baseline_rmse": rref["baseline_rmse"],
              "reduction_vs_lightgbm_pct": 100*(1-score/gref["lightgbm_rmse"]),
              "reduction_vs_baseline_pct": 100*(1-score/rref["baseline_rmse"]),
              "folds": folds, "input_sha256": inputs, "source_sha256": provenance,
              "component_references_sha256": {str(p.resolve()): sha256(p) for p in
                                               [rdir / "results.json", gdir / "results.json"]},
              "versions": {v: importlib.metadata.version(v) for v in ["numpy", "xarray", "lightgbm"]},
              "seconds": time.perf_counter()-started,
              "interpretation": "Historical development; inner weights; official data only."}
    (output / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    pd.DataFrame(folds).to_csv(output / "fold_metrics.csv", index=False)
    pd.DataFrame(monthly).to_csv(output / "monthly_metrics.csv", index=False)
    xr.concat(maps, dim="fold").to_netcdf(output / "rmse_maps.nc")
    return report
