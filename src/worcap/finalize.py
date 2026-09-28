"""E05: seleção histórica, reajuste e candidato local. Sem capacidade de upload."""

import importlib.metadata
import json
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
from .boosting import predict_month, prediction_features, prepare_training, train_model
from .ensemble import blend, choose_weight, json_read
from .experiment import ATMOSPHERIC, require, sha256, stats, verify_sample
from .ridge import coefficients, fit_local, predict_local, read_x, source_dates


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def months(start, end):
    return pd.date_range(f"{start}-01-01", f"{end}-12-01", freq="MS")


def test_sources(test, dates, lat, lon):
    """Adapte somente a atmosfera de teste: seu eixo original é o mês-alvo."""
    for key, expected in [("time", dates.values), ("lat", lat), ("lon", lon)]:
        require(np.array_equal(test[key].values, expected), f"Teste desalinhado: {key}")
        require(pd.Index(expected).is_unique, f"Coordenada duplicada: {key}")
    require("time_origem" in test.coords, "Teste sem time_origem")
    require(np.array_equal(test.time_origem.values, source_dates(dates).values),
            "time_origem deve ser exatamente o mês anterior ao alvo")
    sources = {}
    for name in ATMOSPHERIC:
        require(test[name].dims == ("time", "lat", "lon"), f"Dimensões: {name}")
        # Não acessar variáveis de chuva nem o lag da última chuva observada.
        da = test[name].reset_coords(drop=True).assign_coords(time=source_dates(dates))
        require(np.isfinite(da.values).all(), f"Teste não finito: {name}")
        sources[name] = da.to_dataset(name=name)
    return sources


def ridge_fit_predict(datasets, tp, start, end, dates, alphas, step,
                      prediction_datasets=None, save_path=None):
    """Uma fatoração estatística por lote; resolver cada alpha sem reler treino."""
    prediction_datasets = datasets if prediction_datasets is None else prediction_datasets
    training = months(start, end)
    sources, val_sources = source_dates(training), source_dates(dates)
    clim = monthly_climatology(tp, start, end).values
    y = tp.sel(time=training).values
    nlat, nlon = tp.sizes["lat"], tp.sizes["lon"]
    predictions = np.empty((len(alphas), len(dates), nlat, nlon), dtype=np.float32)
    params = None
    if save_path is not None:
        require(len(alphas) == 1, "Persistência exige um alpha escolhido")
        params = {"means": np.empty((12, nlat, nlon, 9)),
                  "scale": np.empty((nlat, nlon, 9)), "beta": np.empty((nlat, nlon, 9)),
                  "intercept": np.empty((nlat, nlon))}
    for lo in range(0, nlat, step):
        sl = slice(lo, min(lo+step, nlat))
        model = fit_local(read_x(datasets, sources, sl), y[:, sl].reshape(len(training), -1),
                          sources.month.to_numpy(), training.month.to_numpy(),
                          clim[:, sl].reshape(12, -1))
        vx = read_x(prediction_datasets, val_sources, sl)
        for i, alpha in enumerate(alphas):
            beta = coefficients(model, alpha)
            pred, _ = predict_local(model, beta, vx, val_sources.month.to_numpy(),
                                    dates.month.to_numpy())
            predictions[i, :, sl] = pred.reshape(len(dates), sl.stop-sl.start, nlon)
        if params is not None:
            params["means"][:, sl] = model["means"].reshape(12, sl.stop-sl.start, nlon, 9)
            for name in ("scale", "intercept"):
                params[name][sl] = model[name].reshape(params[name][sl].shape)
            params["beta"][sl] = beta.reshape(params["beta"][sl].shape)
    if params is not None:
        saved = xr.Dataset({
            "feature_month_mean": (("month", "lat", "lon", "feature"), params["means"]),
            "feature_scale": (("lat", "lon", "feature"), params["scale"]),
            "beta": (("lat", "lon", "feature"), params["beta"]),
            "intercept": (("lat", "lon"), params["intercept"]),
            "target_climatology": (("month", "lat", "lon"), clim)},
            coords={"month": np.arange(1, 13), "lat": tp.lat, "lon": tp.lon,
                    "feature": list(ATMOSPHERIC)},
            attrs={"alpha": float(alphas[0]), "training_target_start": str(training[0].date()),
                   "training_target_end": str(training[-1].date())})
        saved.to_netcdf(save_path, encoding={k: {"zlib": True} for k in saved})
    return predictions


def boosting_predict(model, pre, datasets, dates, iterations, threads):
    pred = np.empty((len(dates), pre.sizes["lat"], pre.sizes["lon"]), dtype=np.float32)
    for i, date in enumerate(dates):
        values, _ = predict_month(model, prediction_features(datasets, pre, date),
                                   iterations, threads)
        pred[i] = values.reshape(pred.shape[1:])
    return pred


def prediction_dataset(rp, gp, weight, dates, lat, lon):
    ds = xr.Dataset({"ridge": (("time", "lat", "lon"), rp),
                     "lightgbm": (("time", "lat", "lon"), gp),
                     "tp_mm_day": (("time", "lat", "lon"), blend(rp, gp, weight))},
                    coords={"time": dates, "lat": lat, "lon": lon})
    for da in ds.data_vars.values():
        da.attrs["units"] = "mm/day"
    return ds


def infer_saved(model_dir, test_path):
    """Inferência autônoma: receita e modelos locais, sem alvos/treino histórico."""
    recipe = json_read(model_dir / "recipe.json")
    for name, digest in recipe["model_sha256"].items():
        require(sha256(model_dir / name) == digest, f"Modelo mudou: {name}")
    require(sha256(test_path) == recipe["test_sha256"], "Arquivo de teste mudou")
    dates = pd.date_range(recipe["forecast_start"], recipe["forecast_end"], freq="MS")
    with xr.open_dataset(test_path) as test, xr.open_dataset(
            model_dir / "preprocessing.nc") as pre, xr.open_dataset(
            model_dir / "ridge_model.nc") as ridge, threadpool_limits(limits=2):
        for key in ("lat", "lon", "feature", "month"):
            require(np.array_equal(pre[key].values, ridge[key].values), f"Modelos: {key}")
        require(pre.feature.values.tolist() == list(ATMOSPHERIC), "Atributos fora de ordem")
        datasets = test_sources(test, dates, pre.lat.values, pre.lon.values)
        local = {"means": ridge.feature_month_mean.values.reshape(12, -1, 9),
                 "scale": ridge.feature_scale.values.reshape(-1, 9),
                 "intercept": ridge.intercept.values.ravel(),
                 "climatology": ridge.target_climatology.values.reshape(12, -1)}
        beta = ridge.beta.values.reshape(-1, 9)
        rp = np.empty((len(dates), pre.sizes["lat"], pre.sizes["lon"]), dtype=np.float32)
        for i, date in enumerate(dates):
            source = source_dates([date])
            values, _ = predict_local(local, beta,
                read_x(datasets, source, slice(0, pre.sizes["lat"])),
                source.month.to_numpy(), np.array([date.month]))
            rp[i] = values.reshape(rp.shape[1:])
        model = lgb.Booster(model_file=str(model_dir / "lightgbm_model.txt"))
        gp = boosting_predict(model, pre, datasets, dates, recipe["lightgbm_iterations"],
                              recipe["num_threads"])
        return prediction_dataset(rp, gp, recipe["lightgbm_weight"], dates,
                                  pre.lat.values, pre.lon.values)


def export_csv(predictions, sample_path, output_path):
    require(not output_path.exists(), f"CSV já existe: {output_path}")
    require(predictions.tp_mm_day.dims == ("time", "lat", "lon"), "Ordem das dimensões")
    for key in ("time", "lat", "lon"):
        require(predictions.indexes[key].is_unique, f"Coordenada duplicada: {key}")
    values = predictions.tp_mm_day.values.ravel()
    require(np.isfinite(values).all() and (values >= 0).all(), "Previsões inválidas")
    sample = verify_sample(sample_path, predictions.time.values,
                           predictions.lat.values, predictions.lon.values)
    offset = 0
    for chunk in pd.read_csv(sample_path, chunksize=100_000):
        chunk["tp_mm_day"] = values[offset:offset+len(chunk)]
        chunk.to_csv(output_path, index=False, mode="w" if offset == 0 else "a",
                     header=offset == 0, float_format="%.9g")
        offset += len(chunk)
    offset = 0
    ids = set()
    with pd.read_csv(sample_path, chunksize=100_000) as originals, pd.read_csv(
            output_path, chunksize=100_000) as saved:
        for original, chunk in zip(originals, saved, strict=True):
            require(list(chunk.columns) == ["id", "tp_mm_day"], "Colunas incorretas")
            require(chunk.id.equals(original.id), "IDs/ordem alterados na exportação")
            require(chunk.id.is_unique and ids.isdisjoint(chunk.id), "IDs duplicados")
            ids.update(chunk.id)
            require(np.array_equal(chunk.tp_mm_day.to_numpy(dtype=np.float32),
                                   values[offset:offset+len(chunk)]), "CSV perdeu precisão float32")
            offset += len(chunk)
    require(offset == len(values), "CSV incompleto")
    return {"rows": offset, "all_ids_and_order_verified": sample["all_ids_and_order_verified"],
            "unique_ids": True, "float32_roundtrip_exact": True,
            "prediction_stats": stats(values), "sha256": sha256(output_path),
            "submitted": False}


def forecast_saved(model_dir, test_path, sample_path, output):
    require(not output.exists(), f"Saída já existe: {output}")
    recipe = json_read(model_dir / "recipe.json")
    require(sha256(sample_path) == recipe["sample_sha256"], "Sample mudou")
    predictions = infer_saved(model_dir, test_path)
    output.mkdir(parents=True)
    predictions.to_netcdf(output / "predictions.nc",
                          encoding={k: {"zlib": True} for k in predictions})
    verification = export_csv(predictions, sample_path, output / "candidate_unsubmitted.csv")
    write_json(output / "verification.json", verification)
    return verification


def run_final(config_path, output):
    require(not output.exists(), f"Saída já existe: {output}")
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    require(cfg["clip_min"] == 0 and not cfg.get("ocean"), "E05 exige clip zero e sem SST")
    alphas, iterations = sorted(set(cfg["alphas"])), sorted(set(cfg["iterations"]))
    require(alphas and all(np.isfinite(a) and a > 0 for a in alphas), "Alphas inválidos")
    require(iterations and all(isinstance(n, int) and n > 0 for n in iterations),
            "Iterações inválidas")
    for key in ("window_years", "inner_validation_years", "latitude_batch", "training_points"):
        require(isinstance(cfg[key], int) and cfg[key] > 0, f"Configuração: {key}")
    end = cfg["train_end"]
    inner_end = end-cfg["inner_validation_years"]
    inner_start = inner_end-cfg["window_years"]+1
    final_start = end-cfg["window_years"]+1
    validation = months(inner_end+1, end)
    dates = pd.date_range(cfg["forecast_start"], cfg["forecast_end"], freq="MS")
    require(len(dates) > 0 and dates[0] == pd.Timestamp(f"{end+1}-01-01"),
            "Previsão deve começar após o corte de treino")
    output.mkdir(parents=True)
    started = time.perf_counter()
    provenance = {}
    for path in [config_path, Path("docs/E05_PROTOCOLO.md"), Path("pyproject.toml"),
                 Path("uv.lock"), *sorted(Path("src/worcap").glob("*.py"))]:
        dest = output / "source_snapshot" / (Path("config.yaml") if path == config_path else path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        provenance[str(path)] = sha256(path)
    raw = Path(cfg["data_dir"])
    inputs = {p.name: sha256(p) for p in [raw / "treino_tp.nc", raw / "teste_features.nc",
              raw / "sample_submission.csv", *[raw / f"treino_{v}.nc" for v in ATMOSPHERIC]]}
    if cfg.get("contract_audit"):
        audit = json_read(Path(cfg["contract_audit"]))
        require(audit["status"] == "passed", "Auditoria anterior não passou")
        for name, digest in inputs.items():
            require(audit["files"][name]["sha256"] == digest, f"Entrada mudou: {name}")
        shutil.copy2(cfg["contract_audit"], output / "contract_reference.json")
    with ExitStack() as stack, threadpool_limits(limits=2):
        tp = stack.enter_context(xr.open_dataset(raw / "treino_tp.nc")).tp.load()
        require(tp.attrs.get("units") == "mm/day", "Unidade de chuva inesperada")
        require(pd.Timestamp(tp.time.values[-1]) == pd.Timestamp(f"{end}-12-01"),
                "Corte de treino incompatível")
        datasets = {v: stack.enter_context(xr.open_dataset(raw / f"treino_{v}.nc"))
                    for v in ATMOSPHERIC}
        for name, ds in datasets.items():
            require(ds[name].dims == ("time", "lat", "lon"), f"Dimensões: {name}")
            for coord in ("time", "lat", "lon"):
                require(np.array_equal(ds[coord], tp[coord]), f"Grade/datas: {name}/{coord}")
        test = stack.enter_context(xr.open_dataset(raw / "teste_features.nc"))
        sources = test_sources(test, dates, tp.lat.values, tp.lon.values)
        verify_sample(raw / "sample_submission.csv", dates, tp.lat.values, tp.lon.values)
        npoints = tp.sizes["lat"]*tp.sizes["lon"]
        points = np.sort(np.random.default_rng(cfg["seed"]).choice(
            npoints, min(cfg["training_points"], npoints), replace=False))
        np.save(output / "training_points.npy", points)
        truth = tp.sel(time=validation).values
        print(f"E05: seleção Ridge em {validation[0].year}-{end}", flush=True)
        rp_all = ridge_fit_predict(datasets, tp, inner_start, inner_end, validation,
                                   alphas, cfg["latitude_batch"])
        rscore = [{"alpha": float(a), "rmse": float(np.sqrt(np.square(p.astype(float)-truth).mean()))}
                  for a, p in zip(alphas, rp_all)]
        alpha = min(rscore, key=lambda row: row["rmse"])["alpha"]
        rp = rp_all[alphas.index(alpha)].copy()
        del rp_all
        print(f"E05: alpha={alpha:g}; selecionando LightGBM", flush=True)
        x, y, pre = prepare_training(datasets, tp, inner_start, inner_end, points)
        model = train_model(x, y, cfg, max(iterations))
        model.save_model(str(output / "inner_lightgbm_model.txt"))
        del x, y
        gscore, gp = [], None
        for n in iterations:
            candidate = boosting_predict(model, pre, datasets, validation, n, cfg["params"]["num_threads"])
            score = float(np.sqrt(np.square(candidate.astype(float)-truth).mean()))
            if not gscore or score < min(row["rmse"] for row in gscore):
                gp = candidate
            gscore.append({"iterations": n, "rmse": score})
        chosen = min(gscore, key=lambda row: row["rmse"])["iterations"]
        weight, weights = choose_weight(rp, gp, truth, cfg["lightgbm_weights"])
        selection = {"training_target_start": f"{inner_start}-01-01",
                     "training_target_end": f"{inner_end}-12-01",
                     "validation_start": str(validation[0].date()),
                     "validation_end": str(validation[-1].date()),
                     "ridge_candidates": rscore, "lightgbm_candidates": gscore,
                     "weight_candidates": weights, "ridge_alpha": alpha,
                     "lightgbm_iterations": chosen, "lightgbm_weight": weight,
                     "scores_are_selection_only": True}
        write_json(output / "selection.json", selection)
        internal = prediction_dataset(rp, gp, weight, validation, tp.lat.values, tp.lon.values)
        internal.to_netcdf(output / "inner_predictions.nc",
                           encoding={k: {"zlib": True} for k in internal})
        del internal, rp, gp, truth, pre, model, candidate
        print(f"E05: seleção congelada: alpha={alpha:g}, iterações={chosen}, peso GB={weight:g}", flush=True)
        print(f"E05: reajuste {final_start}-{end}", flush=True)
        rp = ridge_fit_predict(datasets, tp, final_start, end, dates, [alpha],
                                cfg["latitude_batch"], sources, output / "ridge_model.nc")[0]
        x, y, pre = prepare_training(datasets, tp, final_start, end, points)
        model = train_model(x, y, cfg, chosen)
        del x, y
        model.save_model(str(output / "lightgbm_model.txt"))
        pre.to_netcdf(output / "preprocessing.nc", encoding={k: {"zlib": True} for k in pre})
        gp = boosting_predict(model, pre, sources, dates, chosen, cfg["params"]["num_threads"])
        expected = prediction_dataset(rp, gp, weight, dates, tp.lat.values, tp.lon.values)
    recipe = {"experiment": "E05", "training_target_start": f"{final_start}-01-01",
              "training_target_end": f"{end}-12-01", "ridge_alpha": alpha,
              "lightgbm_iterations": chosen, "lightgbm_weight": weight,
              "ridge_weight": 1-weight, "num_threads": cfg["params"]["num_threads"],
              "forecast_start": str(dates[0].date()), "forecast_end": str(dates[-1].date()),
              "test_sha256": inputs["teste_features.nc"],
              "sample_sha256": inputs["sample_submission.csv"],
              "model_sha256": {name: sha256(output / name) for name in (
                  "ridge_model.nc", "lightgbm_model.txt", "preprocessing.nc")},
              "submitted": False}
    write_json(output / "recipe.json", recipe)
    print("E05: conferindo recarga dos modelos nos 24 meses e exportando CSV local", flush=True)
    reloaded = infer_saved(output, raw / "teste_features.nc")
    for name in expected:
        require(np.array_equal(expected[name].values, reloaded[name].values),
                f"Recarga não reproduziu todos os meses: {name}")
    reloaded.to_netcdf(output / "predictions.nc", encoding={k: {"zlib": True} for k in reloaded})
    verification = export_csv(reloaded, raw / "sample_submission.csv",
                               output / "candidate_unsubmitted.csv")
    verification.update({"all_months_models_reproduced_exactly": True,
                         "forecast_months": len(dates), "test_rmse_available": False})
    write_json(output / "verification.json", verification)
    report = {"experiment": "E05", "status": "local_candidate_ready_unsubmitted",
              "created_utc": datetime.now(UTC).isoformat(), "selection": selection,
              "recipe": recipe, "verification": verification, "input_sha256": inputs,
              "source_sha256": provenance, "elapsed_seconds": time.perf_counter()-started,
              "versions": {p: importlib.metadata.version(p) for p in (
                  "numpy", "pandas", "xarray", "lightgbm", "scipy", "netCDF4")},
              "artifact_sha256": {p.name: sha256(p) for p in output.iterdir() if p.is_file()}}
    write_json(output / "results.json", report)
    print(f"E05 concluído localmente: {verification['rows']} linhas; nenhum envio", flush=True)
    return report
