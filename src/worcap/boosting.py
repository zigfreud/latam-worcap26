"""E02: correção LightGBM global, seleção temporal interna e avaliação na grade inteira."""

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

from .baseline import monthly_climatology
from .experiment import ATMOSPHERIC, require, sha256
from .ocean import OCEAN_NAMES, append_ocean_prediction, append_ocean_training, read_sst
from .ridge import source_dates

FEATURES = [f"anomaly_{v}" for v in ATMOSPHERIC] + [
    "latitude", "longitude", "target_month_sin", "target_month_cos", "rain_climatology"]


def context_features(dates, lat, lon, climatology, points):
    """Ordem das linhas: tempo, depois ponto; nunca inclui o ano como atributo."""
    months = pd.DatetimeIndex(dates).month.to_numpy()
    n = len(points)
    out = np.empty((len(dates), n, len(FEATURES)), dtype=np.float32)
    yy, xx = np.meshgrid(lat, lon, indexing="ij")
    out[:, :, 9] = yy.ravel()[points]
    out[:, :, 10] = xx.ravel()[points]
    out[:, :, 11] = np.sin(2*np.pi*(months-1)/12)[:, None]
    out[:, :, 12] = np.cos(2*np.pi*(months-1)/12)[:, None]
    out[:, :, 13] = climatology.reshape(12, -1)[months-1][:, points]
    return out


def prepare_training(datasets, tp, start, end, points, ocean=None, ocean_lag=2):
    dates = pd.date_range(f"{start}-01-01", f"{end}-12-01", freq="MS")
    sources = source_dates(dates)
    clim = monthly_climatology(tp, start, end).values.astype(np.float32)
    features = context_features(dates, tp.lat.values, tp.lon.values, clim, points)
    means = np.empty((12, tp.sizes["lat"], tp.sizes["lon"], 9), dtype=np.float32)
    for j, name in enumerate(ATMOSPHERIC):
        da = datasets[name][name].sel(time=sources).transpose("time", "lat", "lon")
        require(np.array_equal(da.time.values, sources.values), f"Datas incorretas: {name}")
        values = da.values
        require(np.isfinite(values).all(), f"Treino não finito: {name}")
        for month in range(1, 13):
            means[month-1, :, :, j] = values[sources.month == month].mean(axis=0)
        sampled = values.reshape(len(dates), -1)[:, points]
        features[:, :, j] = sampled - means[:, :, :, j].reshape(12, -1)[
            sources.month.to_numpy()-1][:, points]
    truth = tp.sel(time=dates).values.reshape(len(dates), -1)[:, points]
    target = truth.astype(np.float64) - features[:, :, 13]
    require(np.isfinite(features).all() and np.isfinite(target).all(), "Treino inválido")
    preprocessing = xr.Dataset({
        "feature_month_mean": (("month", "lat", "lon", "feature"), means),
        "target_climatology": (("month", "lat", "lon"), clim),
    }, coords={"month": np.arange(1, 13), "lat": tp.lat.values, "lon": tp.lon.values,
               "feature": list(ATMOSPHERIC)}, attrs={
                   "training_target_start": str(dates[0].date()),
                   "training_target_end": str(dates[-1].date())})
    rows = features.reshape(-1, len(FEATURES))
    if ocean is not None:
        rows = append_ocean_training(rows, dates, len(points), preprocessing, ocean, ocean_lag)
    return rows, target.ravel(), preprocessing


def prediction_features(datasets, preprocessing, date, ocean=None):
    dates = pd.DatetimeIndex([date])
    source = source_dates(dates)[0]
    n = preprocessing.sizes["lat"] * preprocessing.sizes["lon"]
    features = context_features(dates, preprocessing.lat.values, preprocessing.lon.values,
                                preprocessing.target_climatology.values, np.arange(n))[0]
    means = preprocessing.feature_month_mean.sel(month=source.month).values.reshape(n, 9)
    for j, name in enumerate(ATMOSPHERIC):
        values = datasets[name][name].sel(time=source).transpose("lat", "lon").values
        features[:, j] = values.ravel() - means[:, j]
    require(np.isfinite(features).all(), "Atributos de previsão não finitos")
    if "ocean_month_mean" in preprocessing:
        features = append_ocean_prediction(features, date, preprocessing, ocean)
    return features


def predict_month(model, features, iterations, threads, clip_min=0.):
    raw = features[:, 13].astype(np.float64) + model.predict(
        features, num_iteration=iterations, num_threads=threads)
    return np.maximum(raw, clip_min).astype(np.float32), int((raw < clip_min).sum())


def train_model(features, target, config, iterations):
    params = dict(config["params"], seed=config["seed"])
    names = FEATURES + (OCEAN_NAMES if features.shape[1] == len(FEATURES)+4 else [])
    if config.get("memory_months", 1) > 1:
        from .temporal import temporal_feature_names
        names = temporal_feature_names(config["memory_months"])
    if config.get("spatial_arm"):
        from .spatial import spatial_feature_names
        names = spatial_feature_names(config["spatial_arm"])
    categorical = "auto"
    if config.get("geography"):
        from .geography import GEO_FEATURES, geographic_feature_names
        names = geographic_feature_names()
        categorical = GEO_FEATURES
    require(features.shape[1] == len(names), "Número de atributos inesperado")
    data = lgb.Dataset(features, label=target, feature_name=names, categorical_feature=categorical)
    return lgb.train(params, data, num_boost_round=iterations)


def configured_points(config, key, npoints, default):
    """Load frozen grid indices with a mandatory hash and strict index checks."""
    spec = config.get(key)
    if spec is None:
        return default
    path = Path(spec["path"])
    require(sha256(path) == spec["sha256"], f"Spatial index file changed: {key}")
    points = np.load(path, allow_pickle=False)
    require(points.ndim == 1 and len(points) > 0, f"Empty/invalid indices: {key}")
    require(np.issubdtype(points.dtype, np.integer), f"Noninteger indices: {key}")
    require((points >= 0).all() and (points < npoints).all(), f"Out of grid: {key}")
    require(np.all(points[1:] > points[:-1]), f"Indices must be sorted and unique: {key}")
    return points.astype(np.int64, copy=False)


def run_boosting(config_path: Path, output: Path):
    require(not output.exists(), f"Saída já existe: {output}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment = config.get("experiment", "E02")
    ocean_config = config.get("ocean")
    memory = config.get("memory_months", 1)
    require(memory in (1, 2, 3) and not (memory > 1 and ocean_config), "Memória/SST incompatíveis")
    if config.get("geography"):
        from .geography import (
            geographic_feature_names,
            prediction_geographic_features,
            prepare_geographic_training,
        )
        require(memory == 3 and config.get("spatial_arm") == "pressure" and not ocean_config,
                "Contexto geográfico exige E09 pressure")

        def prepare(datasets, tp, start, end, points, ocean, ocean_lag):
            geo = config["geography"]
            return prepare_geographic_training(datasets, tp, start, end, points,
                                               Path(geo["path"]), geo["sha256"])

        def make_features(datasets, pre, date, ocean):
            return prediction_geographic_features(datasets, pre, date)

        feature_names = geographic_feature_names()
    elif config.get("spatial_arm"):
        from .spatial import (
            prediction_spatial_features,
            prepare_spatial_training,
            spatial_feature_names,
        )
        require(memory == 3 and not ocean_config, "E09 exige memória 3 sem SST")

        def prepare(datasets, tp, start, end, points, ocean, ocean_lag):
            return prepare_spatial_training(datasets, tp, start, end, points, config["spatial_arm"])

        def make_features(datasets, pre, date, ocean):
            return prediction_spatial_features(datasets, pre, date)

        feature_names = spatial_feature_names(config["spatial_arm"])
    elif memory > 1:
        from .temporal import (
            prediction_temporal_features,
            prepare_temporal_training,
            temporal_feature_names,
        )

        def prepare(datasets, tp, start, end, points, ocean, ocean_lag):
            return prepare_temporal_training(datasets, tp, start, end, points, memory)

        def make_features(datasets, pre, date, ocean):
            return prediction_temporal_features(datasets, pre, date)

        feature_names = temporal_feature_names(memory)
    else:
        prepare, make_features = prepare_training, prediction_features
        feature_names = FEATURES + (OCEAN_NAMES if ocean_config else [])
    ocean = read_sst(Path(ocean_config["path"])) if ocean_config else None
    candidates = sorted(set(config["iterations"]))
    require(candidates and all(isinstance(v, int) and v > 0 for v in candidates),
            "Número de árvores inválido")
    require(config["training_points"] > 0, "Amostra vazia")
    output.mkdir(parents=True)
    started = time.perf_counter()
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
    provenance = {}
    for path in [config_path, Path("pyproject.toml"), Path("uv.lock"),
                 Path(config.get("protocol", "docs/E02_PROTOCOLO.md")),
                 *sorted(Path("src/worcap").glob("*.py"))]:
        dest = output / "source_snapshot" / (Path("config.yaml") if path == config_path else path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        provenance[str(path)] = sha256(path)
    references = {}
    for name in ("baseline", "ridge"):
        path = Path(config[f"{name}_results"])
        references[name] = json.loads(path.read_text(encoding="utf-8"))
        shutil.copy2(path, output / f"{name}_reference.json")
    if config.get("control_results"):
        path = Path(config["control_results"])
        references["control"] = json.loads(path.read_text(encoding="utf-8"))
        shutil.copy2(path, output / "control_reference.json")
    ocean_provenance = None
    if ocean_config:
        path = Path(ocean_config["path"])
        require(sha256(path) == ocean_config["sha256"], "Série oceânica mudou")
        shutil.copy2(path, output / "sstoi.indices")
        ocean_provenance = {"sha256": sha256(path), "path": str(path),
                            "lag_months": ocean_config["lag_months"],
                            "historical_vintages_verified": False}
        meta = path.parent / "provenance.json"
        if meta.exists():
            shutil.copy2(meta, output / "ocean_source_provenance.json")
    raw = Path(config["data_dir"])
    results, monthly, maps, verification = [], [], [], []
    threads = config["params"]["num_threads"]
    with ExitStack() as stack:
        tp = stack.enter_context(xr.open_dataset(raw / "treino_tp.nc")).tp.load()
        require(tp.attrs.get("units") == "mm/day", "Unidade inesperada")
        datasets = {v: stack.enter_context(xr.open_dataset(raw / f"treino_{v}.nc"))
                    for v in ATMOSPHERIC}
        for name, ds in datasets.items():
            for coord in ("time", "lat", "lon"):
                require(np.array_equal(tp[coord].values, ds[coord].values), f"Grade/datas: {name}")
        input_hashes = {path.name: sha256(path) for path in
                        [raw / "treino_tp.nc", *[raw / f"treino_{v}.nc" for v in ATMOSPHERIC]]}
        for name, reference in references.items():
            for filename, digest in reference["input_sha256"].items():
                if filename in input_hashes:
                    require(input_hashes[filename] == digest, f"Entrada mudou desde {name}")
        nlat, nlon = tp.sizes["lat"], tp.sizes["lon"]
        npoints = nlat*nlon
        points = np.sort(np.random.default_rng(config["seed"]).choice(
            npoints, min(config["training_points"], npoints), replace=False))
        points = configured_points(config, "training_points_file", npoints, points)
        require(len(points) == min(config["training_points"], npoints),
                "Configured training point count differs")
        selection_points = configured_points(config, "selection_points_file", npoints,
                                             np.arange(npoints))
        np.save(output / "training_points.npy", points)
        np.save(output / "selection_points.npy", selection_points)
        for fold in config["folds"]:
            end, first, last = (fold[k] for k in
                               ("train_end", "validation_start", "validation_end"))
            require(end < first <= last, "Treino deve preceder validação")
            label = f"{first}-{last}"
            folder = output / label
            folder.mkdir()
            inner_end = end - config["inner_validation_years"]
            chosen = None
            for stage in ("selection", "evaluation"):
                cutoff = inner_end if stage == "selection" else end
                start = cutoff - config["window_years"] + 1
                val_start, val_end = (inner_end+1, end) if stage == "selection" else (first, last)
                dates = pd.date_range(f"{val_start}-01-01", f"{val_end}-12-01", freq="MS")
                print(f"{label} | {stage}: treino {start}-{cutoff}; avaliação {val_start}-{val_end}",
                      flush=True)
                x, y, pre = prepare(datasets, tp, start, cutoff, points, ocean,
                                             ocean_config["lag_months"] if ocean_config else 2)
                rounds = max(candidates) if stage == "selection" else chosen
                print(f"  treinamento: {len(y)} linhas, {rounds} árvores", flush=True)
                model = train_model(x, y, config, rounds)
                del x, y
                if stage == "selection":
                    sse = np.zeros(len(candidates), dtype=np.float64)
                    for i, date in enumerate(dates):
                        features = make_features(datasets, pre, date, ocean)
                        truth = tp.sel(time=date).values.ravel()
                        for j, candidate in enumerate(candidates):
                            pred, _ = predict_month(model, features, candidate, threads,
                                                    config["clip_min"])
                            residual = (pred.astype(np.float64)-truth)[selection_points]
                            sse[j] += np.square(residual).sum()
                        if (i+1) % 12 == 0:
                            print(f"  seleção: {i+1}/{len(dates)} meses, {len(selection_points)} pontos", flush=True)
                    chosen = int(candidates[int(np.argmin(sse))])
                    selection = {"training_target_start": f"{start}-01-01",
                                 "training_target_end": f"{cutoff}-12-01",
                                 "validation_start": str(dates[0].date()),
                                 "validation_end": str(dates[-1].date()),
                                 "selection_points": len(selection_points),
                                 "selection_points_sha256": sha256(output / "selection_points.npy"),
                                 "chosen_iterations": chosen, "candidates": [
                                     {"iterations": c, "rmse": float(np.sqrt(s/(len(dates)*len(selection_points))))}
                                     for c, s in zip(candidates, sse)]}
                    (folder / "selection.json").write_text(json.dumps(selection, indent=2),
                                                            encoding="utf-8")
                    model.save_model(str(folder / "inner_model.txt"))
                    print(f"  escolhido no passado: {chosen} árvores", flush=True)
                else:
                    model.save_model(str(folder / "model.txt"))
                    pre.to_netcdf(folder / "preprocessing.nc", encoding={
                        key: {"zlib": True} for key in pre.data_vars})
                    predictions = np.empty((len(dates), nlat, nlon), dtype=np.float32)
                    clipped = 0
                    for i, date in enumerate(dates):
                        features = make_features(datasets, pre, date, ocean)
                        pred, negatives = predict_month(model, features, chosen, threads,
                                                        config["clip_min"])
                        predictions[i] = pred.reshape(nlat, nlon)
                        clipped += negatives
                    saved = xr.Dataset({"tp_mm_day": (("time", "lat", "lon"), predictions)},
                                       coords={"time": dates, "lat": tp.lat, "lon": tp.lon})
                    saved.tp_mm_day.attrs["units"] = "mm/day"
                    saved.to_netcdf(folder / "predictions.nc", encoding={"tp_mm_day": {"zlib": True}})
            truth = tp.sel(time=dates).values
            base_pred = pre.target_climatology.values[dates.month.to_numpy()-1]
            error = predictions.astype(np.float64)-truth
            sq, base_sq = error**2, (base_pred.astype(np.float64)-truth)**2
            score, base_score = float(np.sqrt(sq.mean())), float(np.sqrt(base_sq.mean()))
            base_ref = next(r for r in references["baseline"]["folds"] if r["fold"] == label)
            ridge_ref = next(r for r in references["ridge"]["folds"] if r["fold"] == label)
            require(np.isclose(base_score, base_ref["rmse"], atol=1e-7, rtol=0), "E00 não reproduzido")
            with xr.open_dataset(folder / "predictions.nc") as ds:
                require(np.array_equal(ds.tp_mm_day.values, predictions), "Previsões salvas diferem")
                saved_score = float(np.sqrt(np.square(ds.tp_mm_day.values.astype(float)-truth).mean()))
                require(saved_score == score, "RMSE salvo difere")
            reloaded = lgb.Booster(model_file=str(folder / "model.txt"))
            with xr.open_dataset(folder / "preprocessing.nc") as ds:
                f = make_features(datasets, ds, dates[0], ocean)
                reconstructed, _ = predict_month(reloaded, f, chosen, threads, config["clip_min"])
            require(np.array_equal(reconstructed.reshape(nlat, nlon), predictions[0]),
                    "Modelo recarregado diverge")
            require(np.isfinite(predictions).all() and (predictions >= 0).all(), "Previsão inválida")
            verification.append({"fold": label, "saved_prediction_rmse": saved_score,
                                 "first_month_full_grid_reproduced": True,
                                 "finite_nonnegative": True,
                                 "sha256": {p.name: sha256(p) for p in [folder / "model.txt",
                                            folder / "preprocessing.nc", folder / "predictions.nc"]}})
            result = {"fold": label, "train_start": start, "train_end": end,
                      "chosen_iterations": chosen, "training_points": len(points),
                      "baseline_rmse": base_score, "ridge_rmse": ridge_ref["ridge_rmse"],
                      "lightgbm_rmse": score, "count": int(sq.size),
                      "squared_error_sum": float(sq.sum()), "bias": float(error.mean()),
                      "clipped_predictions": clipped,
                      "reduction_vs_baseline_pct": 100*(1-score/base_score),
                      "reduction_vs_ridge_pct": 100*(1-score/ridge_ref["ridge_rmse"])}
            if "control" in references:
                control_ref = next(r for r in references["control"]["folds"] if r["fold"] == label)
                result["control_rmse"] = control_ref["lightgbm_rmse"]
                result["reduction_vs_control_pct"] = 100*(1-score/result["control_rmse"])
            results.append(result)
            for i, date in enumerate(dates):
                monthly.append({"fold": label, "target_month": str(date.date()),
                                "lightgbm_rmse": float(np.sqrt(sq[i].mean())),
                                "baseline_rmse": float(np.sqrt(base_sq[i].mean())),
                                "lightgbm_sse": float(sq[i].sum()), "count": npoints})
            maps.append(xr.Dataset({
                "lightgbm_rmse": (("lat", "lon"), np.sqrt(sq.mean(axis=0))),
                "baseline_rmse": (("lat", "lon"), np.sqrt(base_sq.mean(axis=0)))},
                coords={"lat": tp.lat, "lon": tp.lon}).expand_dims(fold=[label]))
            (output / "progress.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"{label}: E00={base_score:.6f}; E01={ridge_ref['ridge_rmse']:.6f}; E02={score:.6f}",
                  flush=True)
    score = float(np.sqrt(sum(r["squared_error_sum"] for r in results)/sum(r["count"] for r in results)))
    base = references["baseline"]["rmse_mm_day"]
    ridge = references["ridge"]["ridge_rmse"]
    report = {"experiment": experiment, "created_utc": datetime.now(UTC).isoformat(),
              "lightgbm_rmse": score, "baseline_rmse": base, "ridge_rmse": ridge,
              "reduction_vs_baseline_pct": 100*(1-score/base),
              "reduction_vs_ridge_pct": 100*(1-score/ridge), "folds": results,
              "features": feature_names, "memory_months": memory,
              "geography": config.get("geography"),
              "ocean_provenance": ocean_provenance,
              "source_sha256": provenance, "input_sha256": input_hashes,
              "reference_sha256": {name: sha256(output / f"{name}_reference.json")
                                   for name in references},
              "training_points_sha256": sha256(output / "training_points.npy"),
              "selection_points_sha256": sha256(output / "selection_points.npy"),
              "spatial_sampling": {"training": config.get("training_points_file"),
                                   "selection": config.get("selection_points_file")},
              "versions": {v: importlib.metadata.version(v) for v in
                           ["numpy", "pandas", "xarray", "lightgbm"]},
              "seconds": time.perf_counter()-started,
              "interpretation": "Historical development comparison."}
    if "control" in references:
        report["control_rmse"] = references["control"]["lightgbm_rmse"]
        report["reduction_vs_control_pct"] = 100*(1-score/report["control_rmse"])
    if ocean_config:
        report["interpretation"] = "Retrospective revised SST; historical vintages NOT verified."
    (output / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    pd.DataFrame(results).to_csv(output / "fold_metrics.csv", index=False)
    pd.DataFrame(monthly).to_csv(output / "monthly_metrics.csv", index=False)
    xr.concat(maps, dim="fold").to_netcdf(output / "rmse_maps.nc")
    write_report(output, report, monthly, Path(config["ridge_results"]).parent)
    return report


def write_report(output, report, monthly, ridge_folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = pd.DataFrame(monthly)
    ridge_monthly_path = ridge_folder / "monthly_metrics.csv"
    fig, ax = plt.subplots(figsize=(11, 4))
    for col, label in [("baseline_rmse", "E00 — climatologia"),
                       ("lightgbm_rmse", f"{report['experiment']} — LightGBM")]:
        ax.plot(pd.to_datetime(table.target_month), table[col], label=label)
    if ridge_monthly_path.exists():
        ridge = pd.read_csv(ridge_monthly_path)
        ax.plot(pd.to_datetime(ridge.target_month), ridge.ridge_rmse, label="E01 — Ridge", alpha=.8)
    ax.set(title="Comparação mensal fora do treino", xlabel="Mês alvo", ylabel="RMSE (mm/dia)")
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=150)
    plt.close(fig)
    lines = [f"# {report['experiment']} — LightGBM global", "",
             f"RMSE agregado: **{report['lightgbm_rmse']:.6f} mm/dia**.",
             f"Redução frente ao E00: {report['reduction_vs_baseline_pct']:.2f}%.",
             f"Redução frente ao E01: {report['reduction_vs_ridge_pct']:.2f}%.", "",
             f"| Bloco | Árvores | E00 | E01 | {report['experiment']} |", "|---|---:|---:|---:|---:|"]
    for r in report["folds"]:
        lines.append(f"| {r['fold']} | {r['chosen_iterations']} | {r['baseline_rmse']:.6f} | "
                     f"{r['ridge_rmse']:.6f} | {r['lightgbm_rmse']:.6f} |")
    lines += ["", "Treino do corretor em amostra espacial; avaliação em toda a grade.",
              "Seleção de árvores somente em validação temporal interna.",
              "Comparação de desenvolvimento: não demonstra causalidade ou significância.",
              "Previsões e modelos recarregados e conferidos.",
              "", "![Comparação mensal](comparison.png)"]
    if report.get("ocean_provenance"):
        lines += ["", "SST revisada: versões disponíveis em cada data histórica não verificadas.",
                  "Defasagem de dois meses e anomalias ajustadas somente no treino.",
                  "Experimento retrospectivo; não certifica habilidade operacional em tempo real.",
                  f"Redução frente ao controle E02: {report['reduction_vs_control_pct']:.3f}%."]
    (output / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
