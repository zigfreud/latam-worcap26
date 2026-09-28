"""Auditoria numérica e E00: referência climatológica em blocos futuros."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from .baseline import monthly_climatology, predict_climatology

ATMOSPHERIC = (
    "t2", "cloud_cover", "shum_850", "surface_pressure", "u_850", "v_850",
    "temperature_850", "rel_hum_850", "geopotential_850",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def stats(values: np.ndarray) -> dict:
    finite = np.isfinite(values)
    valid = values[finite]
    return {
        "count": int(values.size), "nonfinite": int((~finite).sum()),
        "negative": int((valid < 0).sum()),
        "min": float(valid.min()) if valid.size else None,
        "max": float(valid.max()) if valid.size else None,
        "mean": float(valid.mean(dtype=np.float64)) if valid.size else None,
    }


def verify_sample(path: Path, times, lat, lon) -> dict:
    """Verifique TODOS os IDs e sua ordem, sem reconstruir um arquivo de submissão."""
    total = len(times) * len(lat) * len(lon)
    offset = 0
    for chunk in pd.read_csv(path, chunksize=100_000):
        require(list(chunk.columns) == ["id", "tp_mm_day"], "Colunas do sample incorretas")
        positions = np.arange(offset, offset + len(chunk))
        require(bool(np.all(positions < total)), "Sample contém linhas a mais")
        months = pd.DatetimeIndex(times)[positions // (len(lat) * len(lon))]
        latitudes = lat[(positions // len(lon)) % len(lat)]
        longitudes = lon[positions % len(lon)]
        expected = [f"{date:%Y_%m}_{a:.2f}_{b:.2f}"
                    for date, a, b in zip(months, latitudes, longitudes)]
        require(chunk.id.tolist() == expected, f"IDs/ordem incompatíveis a partir de {offset}")
        require(bool(np.isfinite(chunk.tp_mm_day).all()), "Sample contém NaN/Inf")
        require(bool((chunk.tp_mm_day == 0).all()), "Sample não contém apenas zeros")
        offset += len(chunk)
    require(offset == total, f"Sample incompleto: {offset} de {total}")
    return {"rows": offset, "all_ids_and_order_verified": True, "all_predictions_zero": True}


def audit_contract(data_dir: Path, output: Path) -> dict:
    started = time.perf_counter()
    train_time = pd.date_range("1940-01-01", "2022-12-01", freq="MS")
    test_time = pd.date_range("2023-01-01", "2024-12-01", freq="MS")
    lat = np.arange(-60, 15.01, 0.25)
    lon = np.arange(-90, -24.99, 0.25)
    report = {"status": "running", "files": {}, "checks": {}, "warnings": []}
    output.parent.mkdir(parents=True, exist_ok=True)

    def grid(ds, dates):
        for name, expected in [("time", dates.values), ("lat", lat), ("lon", lon)]:
            require(np.array_equal(ds[name].values, expected), f"Coordenada {name} incompatível")
        for name, var in ds.data_vars.items():
            require(var.dims == ("time", "lat", "lon"), f"Dimensões inesperadas: {name}")

    try:
        with xr.open_dataset(data_dir / "treino_tp.nc") as ds:
            grid(ds, train_time)
            require(ds.tp.attrs.get("units") == "mm/day", "Unidade de tp inesperada")
            tp = ds.tp.load()
        require(bool(np.isfinite(tp.values).all()), "Chuva de treino contém NaN/Inf")
        for path in sorted(data_dir.glob("*.nc")):
            print(f"Auditando {path.name}", flush=True)
            record = {"bytes": path.stat().st_size, "sha256": sha256(path), "variables": {}}
            with xr.open_dataset(path) as ds:
                grid(ds, test_time if path.name == "teste_features.nc" else train_time)
                for name in ds.data_vars:
                    values = ds[name].values
                    record["variables"][name] = stats(values)
                    if name == "tp_alvo":
                        require(ds[name].attrs.get("units") == "mm/day", "Unidade do alvo")
                        if path.name == "teste_features.nc":
                            require(bool(np.isnan(values).all()), "Alvo de teste não está vazio")
                        else:
                            require(bool(np.isnan(values[-1]).all()), "Último alvo deve ser NaN")
                            # Comparação por posição: coordenadas de entrada e alvo diferem um mês.
                            require(np.array_equal(values[:-1], tp.values[1:]), "Alvo não é M+1")
                    else:
                        require(bool(np.isfinite(values).all()), f"NaN/Inf em {path.name}:{name}")
                    if name in ("tp", "tp_ultima_obs") and np.any(values < 0):
                        report["warnings"].append(f"Valores negativos em {path.name}:{name}")
                    del values
                if path.name == "teste_features.nc":
                    require(set(ds.data_vars) == set(ATMOSPHERIC) | {"tp_alvo", "tp_ultima_obs"},
                            "Variáveis de teste inesperadas")
                    require(np.array_equal(ds.time_origem.values,
                                           pd.date_range("2022-12-01", "2024-11-01", freq="MS")),
                            "time_origem não é mês anterior ao alvo")
                    require(np.array_equal(ds.lag_meses.values, np.arange(1, 25)), "lag_meses")
                    require(bool(np.all(ds.tp_ultima_obs.values == tp.values[-1])),
                            "tp_ultima_obs não corresponde a dezembro/2022")
            report["files"][path.name] = record
        expected_files = {f"treino_{v}.nc" for v in ATMOSPHERIC} | {
            "treino_tp.nc", "treino_tp_alvo.nc", "teste_features.nc"}
        require(set(report["files"]) == expected_files, "Conjunto de NetCDF inesperado")
        with xr.open_dataset(data_dir / "teste_features.nc") as test:
            for name in ATMOSPHERIC:
                with xr.open_dataset(data_dir / f"treino_{name}.nc") as train:
                    require(set(train.data_vars) == {name}, f"Variável errada em treino_{name}")
                    require(np.array_equal(test[name].isel(time=0), train[name].isel(time=-1)),
                            f"Entrada de janeiro/2023 não coincide com dezembro/2022: {name}")
        sample = data_dir / "sample_submission.csv"
        report["sample"] = verify_sample(sample, test_time, lat, lon)
        report["files"][sample.name] = {"bytes": sample.stat().st_size, "sha256": sha256(sample)}
        report["checks"] = {
            "grid_and_dates": True, "training_target_is_next_month": True,
            "test_target_all_nan": True, "rainfall_frozen_december_2022": True,
            "test_origin_previous_month": True, "test_first_features_match_last_train": True,
        }
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        raise
    finally:
        report["seconds"] = time.perf_counter() - started
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def evaluate_fold(tp: xr.DataArray, fold: dict, window_years: int):
    end = int(fold["train_end"])
    start = end - window_years + 1
    vstart, vend = int(fold["validation_start"]), int(fold["validation_end"])
    require(window_years > 0 and end < vstart <= vend, "Treino deve preceder validação")
    truth = tp.sel(time=slice(f"{vstart}-01-01", f"{vend}-12-31"))
    expected = pd.date_range(f"{vstart}-01-01", f"{vend}-12-01", freq="MS")
    require(np.array_equal(truth.time.values, expected.values), "Validação mensal incompleta")
    require(bool(np.isfinite(truth).all().item()), "Alvos de validação não finitos")
    clim = monthly_climatology(tp, start, end)
    pred = predict_climatology(clim, truth.time).transpose("time", "lat", "lon")
    pred, truth = xr.align(pred, truth, join="exact")
    error = pred.astype("float64") - truth.astype("float64")
    square = error ** 2
    label = f"{vstart}-{vend}"
    rows = []
    for index, date in enumerate(pd.DatetimeIndex(truth.time.values)):
        part = square.isel(time=index).values
        rows.append({"fold": label, "target_month": str(date.date()), "count": part.size,
                     "squared_error_sum": float(part.sum()), "rmse": float(np.sqrt(part.mean()))})
    result = {
        "fold": label, "train_start": start, "train_end": end,
        "validation_start": vstart, "validation_end": vend,
        "training_months": window_years * 12, "validation_months": len(expected),
        "count": int(square.size), "squared_error_sum": float(square.sum().item()),
        "rmse": float(np.sqrt(square.mean()).item()),
        "bias": float(error.mean().item()),
    }
    return result, rows, np.sqrt(square.mean("time")).rename("rmse_mm_day")


def run_backtest(config_path: Path, output: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    require(not output.exists(), f"Saída já existe; escolha outra pasta: {output}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    started = time.perf_counter()
    raw = Path(config["data_dir"])
    output.mkdir(parents=True)
    (output / "config.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    results, monthly, maps = [], [], []
    with xr.open_dataset(raw / "treino_tp.nc") as ds:
        require(ds.tp.attrs.get("units") == "mm/day", "Unidade inesperada")
        tp = ds.tp.load()
        for fold in config["backtest"]["folds"]:
            result, rows, error_map = evaluate_fold(tp, fold, config["backtest"]["window_years"])
            results.append(result)
            monthly.extend(rows)
            maps.append(error_map.expand_dims(fold=[result["fold"]]))
            print(f"{result['fold']}: RMSE={result['rmse']:.6f} mm/dia", flush=True)
    all_dates = [row["target_month"] for row in monthly]
    require(len(set(all_dates)) == len(all_dates), "Blocos de validação se sobrepõem")
    overall = float(np.sqrt(sum(r["squared_error_sum"] for r in results)
                            / sum(r["count"] for r in results)))
    table = pd.DataFrame(monthly)
    table.to_csv(output / "monthly_metrics.csv", index=False)
    pd.DataFrame(results).to_csv(output / "fold_metrics.csv", index=False)
    xr.concat(maps, dim="fold").to_netcdf(output / "rmse_maps.nc")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    source_paths = [config_path, Path("pyproject.toml"), Path("uv.lock")]
    source_paths += sorted(Path("src/worcap").glob("*.py"))
    report = {
        "experiment": "E00", "model": "30-year monthly grid climatology",
        "created_utc": datetime.now(UTC).isoformat(),
        "rmse_mm_day": overall, "aggregation": "sqrt(total squared error / total count)",
        "weighting": "equal grid cells and months; entire supplied rectangle, including ocean",
        "folds": results, "python": platform.python_version(),
        "versions": {name: importlib.metadata.version(name)
                     for name in ["numpy", "pandas", "xarray", "netCDF4", "matplotlib"]},
        "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "source_sha256": {str(p): sha256(p) for p in source_paths},
        "input_sha256": {"treino_tp.nc": sha256(raw / "treino_tp.nc")},
        "seconds": time.perf_counter() - started,
        "interpretation": "Historical baseline only; no learned model or Kaggle score.",
    }
    (output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    fig, ax = plt.subplots(figsize=(11, 4))
    for label, group in table.groupby("fold", sort=False):
        ax.plot(pd.to_datetime(group.target_month), group.rmse, marker=".", label=label)
    ax.set(title="E00 — erro mensal da climatologia (validação histórica)",
           ylabel="RMSE (mm/dia)", xlabel="Mês previsto")
    ax.legend(title="Bloco de validação")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "monthly_rmse.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, len(maps), figsize=(14, 5), squeeze=False)
    vmax = max(float(m.max()) for m in maps)
    for ax, m, r in zip(axes.flat, maps, results):
        m.squeeze("fold", drop=True).plot(x="lon", y="lat", ax=ax, vmin=0, vmax=vmax,
                                          cmap="magma", add_colorbar=False)
        ax.set_title(f"{r['fold']} | RMSE {r['rmse']:.3f}")
    fig.colorbar(axes.flat[0].collections[0], ax=list(axes.flat), label="RMSE (mm/dia)",
                 shrink=0.65)
    fig.suptitle("Erro por ponto da grade — mesma escala em todos os blocos")
    fig.savefig(output / "spatial_rmse.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    lines = ["# E00 — climatologia: primeiro backtest", "",
             f"RMSE agregado: **{overall:.6f} mm/dia**.", "",
             "| Validação | Histórico da climatologia | RMSE (mm/dia) |",
             "|---|---|---:|"]
    lines += [f"| {r['fold']} | {r['train_start']}–{r['train_end']} | {r['rmse']:.6f} |"
              for r in results]
    lines += ["", "Cada bloco usa os 30 anos anteriores, sem ajuste com seus alvos.",
              "Avaliação em todos os pontos do retângulo, incluindo oceano, sem pesos de área.",
              "O RMSE agregado usa a soma dos erros quadráticos; não é a média dos RMSEs.",
              "Este resultado é uma referência histórica, não uma pontuação no Kaggle.",
              "Nenhum modelo aprendido foi comparado ainda.", "",
              "![RMSE mensal](monthly_rmse.png)", "", "![RMSE espacial](spatial_rmse.png)"]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"RMSE agregado: {overall:.6f} mm/dia. Relatório: {output.resolve()}", flush=True)
    return report
