"""Ridge local em lotes de latitude, com seleção temporal interna de alpha."""

import importlib.metadata
import json
import os
import shutil
import time
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from threadpoolctl import threadpool_limits

from .baseline import monthly_climatology
from .experiment import ATMOSPHERIC, require, sha256


def source_dates(target_dates):
    return pd.DatetimeIndex(target_dates) - pd.DateOffset(months=1)


def fit_local(x, y, source_months, target_months, climatology):
    """x=(tempo,ponto,atributo); y=(tempo,ponto). Todos são apenas treino."""
    require(np.isfinite(x).all() and np.isfinite(y).all(), "Treino não finito")
    x = np.asarray(x, dtype=np.float64)
    means = np.stack([x[source_months == month].mean(axis=0) for month in range(1, 13)])
    anomaly = x - means[source_months - 1]
    scale = anomaly.std(axis=0)
    scale = np.where(scale > 1e-10, scale, 1.0)
    z = anomaly / scale
    residual = y.astype(np.float64) - climatology[target_months - 1]
    intercept = residual.mean(axis=0)
    residual -= intercept
    gram = np.einsum("tpf,tpg->pfg", z, z, optimize=True)
    rhs = np.einsum("tpf,tp->pf", z, residual, optimize=True)
    return {"means": means, "scale": scale, "intercept": intercept,
            "gram": gram, "rhs": rhs, "climatology": climatology}


def coefficients(model, alpha):
    require(alpha > 0, "Alpha deve ser positivo")
    gram = model["gram"] + alpha * np.eye(model["rhs"].shape[-1])[None]
    return np.linalg.solve(gram, model["rhs"][..., None])[..., 0]


def predict_local(model, beta, x, source_months, target_months, clip_min=0.0):
    require(np.isfinite(x).all(), "Entrada de previsão não finita")
    z = (x.astype(np.float64) - model["means"][source_months - 1]) / model["scale"]
    correction = np.einsum("tpf,pf->tp", z, beta, optimize=True) + model["intercept"]
    raw = model["climatology"][target_months - 1] + correction
    return np.maximum(raw, clip_min).astype(np.float32), int((raw < clip_min).sum())


def read_x(datasets, dates, latitude_slice):
    arrays = []
    for name in ATMOSPHERIC:
        da = datasets[name][name].sel(time=dates).isel(lat=latitude_slice)
        require(np.array_equal(da.time.values, dates.values), f"Datas incorretas: {name}")
        arrays.append(da.values.reshape(len(dates), -1))
    return np.stack(arrays, axis=-1)


def run_ridge(config_path: Path, output: Path):
    require(not output.exists(), f"Saída já existe: {output}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw = Path(config["data_dir"])
    alphas = config["alphas"]
    require(len(alphas) > 0 and all(a > 0 for a in alphas), "Alphas inválidos")
    output.mkdir(parents=True)
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
    started = time.perf_counter()
    provenance = {}
    for path in [config_path, Path("pyproject.toml"), Path("uv.lock"),
                 Path("docs/E01_PROTOCOLO.md"), *sorted(Path("src/worcap").glob("*.py"))]:
        snapshot_path = Path("config.yaml") if path == config_path else path
        dest = output / "source_snapshot" / snapshot_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        provenance[str(path)] = sha256(path)
    baseline = json.loads(Path(config["baseline_results"]).read_text(encoding="utf-8"))
    shutil.copy2(config_path, output / "config.yaml")
    results, monthly, maps = [], [], []
    input_hashes = {}
    with ExitStack() as stack, threadpool_limits(limits=2):
        tp_ds = stack.enter_context(xr.open_dataset(raw / "treino_tp.nc"))
        require(tp_ds.tp.attrs.get("units") == "mm/day", "Unidade inesperada")
        tp = tp_ds.tp.load()
        datasets = {v: stack.enter_context(xr.open_dataset(raw / f"treino_{v}.nc"))
                    for v in ATMOSPHERIC}
        for v, ds in datasets.items():
            for coord in ("time", "lat", "lon"):
                require(np.array_equal(ds[coord].values, tp[coord].values), f"Grade/datas: {v}")
        for path in [raw / "treino_tp.nc", *[raw / f"treino_{v}.nc" for v in ATMOSPHERIC]]:
            input_hashes[path.name] = sha256(path)
        require(input_hashes["treino_tp.nc"] == baseline["input_sha256"]["treino_tp.nc"],
                "Chuva mudou desde o E00")
        nlat, nlon = tp.sizes["lat"], tp.sizes["lon"]
        step = config["latitude_batch"]
        require(step > 0, "Lote inválido")
        for fold in config["folds"]:
            end, first, last = (fold[k] for k in
                                ["train_end", "validation_start", "validation_end"])
            require(end < first <= last, "Treino deve preceder validação")
            label = f"{first}-{last}"
            fold_dir = output / label
            fold_dir.mkdir()
            inner_end = end - config["inner_validation_years"]
            inner_sse = np.zeros(len(alphas), dtype=np.float64)
            chosen = None
            clipped = 0
            for stage in ("selection", "evaluation"):
                cutoff = inner_end if stage == "selection" else end
                vstart, vend = (inner_end + 1, end) if stage == "selection" else (first, last)
                begin = cutoff - config["window_years"] + 1
                train_dates = pd.date_range(f"{begin}-01-01", f"{cutoff}-12-01", freq="MS")
                val_dates = pd.date_range(f"{vstart}-01-01", f"{vend}-12-01", freq="MS")
                train_sources, val_sources = source_dates(train_dates), source_dates(val_dates)
                clim = monthly_climatology(tp, begin, cutoff).values
                train_y = tp.sel(time=train_dates).values
                val_y = tp.sel(time=val_dates).values
                if stage == "evaluation":
                    predictions = np.empty(val_y.shape, dtype=np.float32)
                    params = {
                        "means": np.empty((12, nlat, nlon, 9)),
                        "scale": np.empty((nlat, nlon, 9)),
                        "beta": np.empty((nlat, nlon, 9)),
                        "intercept": np.empty((nlat, nlon)),
                    }
                print(f"{label} | {stage} | alvos treino {begin}-{cutoff}; validação {vstart}-{vend}",
                      flush=True)
                for lo in range(0, nlat, step):
                    sl = slice(lo, min(lo + step, nlat))
                    x = read_x(datasets, train_sources, sl)
                    vx = read_x(datasets, val_sources, sl)
                    model = fit_local(x, train_y[:, sl].reshape(len(train_dates), -1),
                                      train_sources.month.to_numpy(), train_dates.month.to_numpy(),
                                      clim[:, sl].reshape(12, -1))
                    candidates = alphas if stage == "selection" else [chosen]
                    for idx, alpha in enumerate(candidates):
                        beta = coefficients(model, alpha)
                        pred, negative = predict_local(model, beta, vx,
                                                       val_sources.month.to_numpy(),
                                                       val_dates.month.to_numpy(), config["clip_min"])
                        if stage == "selection":
                            diff = pred.astype(np.float64) - val_y[:, sl].reshape(len(val_dates), -1)
                            inner_sse[idx] += float(np.square(diff).sum())
                        else:
                            height = sl.stop - sl.start
                            predictions[:, sl] = pred.reshape(len(val_dates), height, nlon)
                            clipped += negative
                            params["means"][:, sl] = model["means"].reshape(12, height, nlon, 9)
                            for key in ("scale", "intercept"):
                                params[key][sl] = model[key].reshape(params[key][sl].shape)
                            params["beta"][sl] = beta.reshape(height, nlon, 9)
                    if lo == 0 or sl.stop == nlat or (lo // step) % 4 == 0:
                        print(f"  {stage}: {sl.stop}/{nlat} latitudes", flush=True)
                if stage == "selection":
                    chosen = float(alphas[int(np.argmin(inner_sse))])
                    tuning = [{"alpha": a, "rmse": float(np.sqrt(s / val_y.size))}
                              for a, s in zip(alphas, inner_sse)]
                    (fold_dir / "selection.json").write_text(json.dumps({
                        "training_target_start": str(train_dates[0].date()),
                        "training_target_end": str(train_dates[-1].date()),
                        "validation_start": str(val_dates[0].date()),
                        "validation_end": str(val_dates[-1].date()),
                        "candidates": tuning, "chosen_alpha": chosen,
                    }, indent=2), encoding="utf-8")
                    print(f"  alpha escolhido apenas no passado: {chosen}", flush=True)
            base_pred = clim[val_dates.month.to_numpy() - 1]
            error = predictions.astype(np.float64) - val_y
            base_error = base_pred.astype(np.float64) - val_y
            sq, base_sq = error ** 2, base_error ** 2
            base_rmse, ridge_rmse = float(np.sqrt(base_sq.mean())), float(np.sqrt(sq.mean()))
            ref = next(r for r in baseline["folds"] if r["fold"] == label)
            require(np.isclose(base_rmse, ref["rmse"], atol=1e-7, rtol=0), "E00 não reproduzido")
            result = {"fold": label, "train_start": end - config["window_years"] + 1,
                      "train_end": end, "chosen_alpha": chosen, "baseline_rmse": base_rmse,
                      "ridge_rmse": ridge_rmse, "rmse_reduction_pct": 100 * (1-ridge_rmse/base_rmse),
                      "count": int(sq.size), "squared_error_sum": float(sq.sum()),
                      "baseline_squared_error_sum": float(base_sq.sum()),
                      "bias": float(error.mean()), "clipped_predictions": clipped}
            results.append(result)
            for i, date in enumerate(val_dates):
                monthly.append({"fold": label, "target_month": str(date.date()),
                                "count": nlat*nlon, "ridge_sse": float(sq[i].sum()),
                                "baseline_sse": float(base_sq[i].sum()),
                                "ridge_rmse": float(np.sqrt(sq[i].mean())),
                                "baseline_rmse": float(np.sqrt(base_sq[i].mean()))})
            coords = {"time": val_dates, "lat": tp.lat.values, "lon": tp.lon.values}
            saved = xr.Dataset({"tp_mm_day": (("time", "lat", "lon"), predictions)}, coords=coords)
            saved.tp_mm_day.attrs["units"] = "mm/day"
            saved.to_netcdf(fold_dir / "predictions.nc", encoding={"tp_mm_day": {"zlib": True}})
            parameters = xr.Dataset({
                "feature_month_mean": (("month", "lat", "lon", "feature"), params["means"]),
                "feature_scale": (("lat", "lon", "feature"), params["scale"]),
                "beta": (("lat", "lon", "feature"), params["beta"]),
                "intercept": (("lat", "lon"), params["intercept"]),
                "target_climatology": (("month", "lat", "lon"), clim),
            }, coords={"month": np.arange(1,13), "lat": tp.lat.values, "lon": tp.lon.values,
                       "feature": list(ATMOSPHERIC)}, attrs={"alpha": chosen})
            parameters.to_netcdf(fold_dir / "model.nc", encoding={
                key: {"zlib": True} for key in parameters.data_vars})
            maps.append(xr.Dataset({
                "ridge_rmse": (("lat", "lon"), np.sqrt(sq.mean(axis=0))),
                "baseline_rmse": (("lat", "lon"), np.sqrt(base_sq.mean(axis=0))),
            }, coords={"lat": tp.lat.values, "lon": tp.lon.values}).expand_dims(fold=[label]))
            print(f"{label}: E00={base_rmse:.6f}; E01={ridge_rmse:.6f}; "
                  f"redução={result['rmse_reduction_pct']:.2f}%", flush=True)
            (output / "progress.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    count = sum(r["count"] for r in results)
    score = float(np.sqrt(sum(r["squared_error_sum"] for r in results) / count))
    base_score = float(np.sqrt(sum(r["baseline_squared_error_sum"] for r in results) / count))
    report = {"experiment": "E01", "created_utc": datetime.now(UTC).isoformat(),
              "ridge_rmse": score, "baseline_rmse": base_score,
              "rmse_reduction_pct": 100*(1-score/base_score), "folds": results,
              "source_sha256": provenance, "input_sha256": input_hashes,
              "versions": {v: importlib.metadata.version(v) for v in ["numpy", "xarray", "pandas"]},
              "seconds": time.perf_counter() - started,
              "interpretation": "Historical nested temporal evaluation."}
    (output / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(results).to_csv(output / "fold_metrics.csv", index=False)
    pd.DataFrame(monthly).to_csv(output / "monthly_metrics.csv", index=False)
    xr.concat(maps, dim="fold").to_netcdf(output / "rmse_maps.nc")
    write_report(output, report, monthly)
    return report


def write_report(output, report, monthly):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = pd.DataFrame(monthly)
    fig, ax = plt.subplots(figsize=(11,4))
    ax.plot(pd.to_datetime(table.target_month), table.baseline_rmse, label="E00 — climatologia")
    ax.plot(pd.to_datetime(table.target_month), table.ridge_rmse, label="E01 — Ridge")
    ax.set(title="Comparação mensal fora do treino", ylabel="RMSE (mm/dia)", xlabel="Mês alvo")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=150)
    plt.close(fig)
    lines = ["# E01 — Ridge local", "",
             f"RMSE: **{report['ridge_rmse']:.6f} mm/dia**; E00: {report['baseline_rmse']:.6f}.",
             f"Redução relativa do RMSE: **{report['rmse_reduction_pct']:.2f}%**.", "",
             "| Bloco | Alpha interno | E00 | E01 | Redução RMSE |",
             "|---|---:|---:|---:|---:|"]
    for r in report["folds"]:
        lines.append(f"| {r['fold']} | {r['chosen_alpha']:g} | {r['baseline_rmse']:.6f} | "
                     f"{r['ridge_rmse']:.6f} | {r['rmse_reduction_pct']:.2f}% |")
    lines += ["", "Alpha selecionado nos quatro anos anteriores a cada bloco externo.",
              "Nove variáveis locais em M; alvo M+1; sem chuva observada como entrada.",
              "Padronização e climatologias usam apenas o treino. Previsões limitadas a zero.",
              "Avaliação histórica; não é resultado no Kaggle nem prova causal ou de significância.",
              "", "![Comparação mensal](comparison.png)"]
    (output / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
