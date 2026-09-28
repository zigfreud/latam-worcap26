"""Comandos iniciais, executados a partir da raiz do projeto."""

import argparse
import importlib
import json
import platform
import sys
from pathlib import Path


def doctor() -> None:
    modules = ["numpy", "pandas", "xarray", "netCDF4", "h5netcdf", "h5py", "dask",
               "scipy", "sklearn", "lightgbm", "matplotlib", "jupyterlab", "ipykernel"]
    print(f"Python: {sys.version.split()[0]} | {platform.system()}")
    print(f"Executável: {sys.executable}")
    for name in modules:
        module = importlib.import_module(name)
        print(f"OK {name}: {getattr(module, '__version__', 'importado')}")


def audit(data_dir: Path, output: Path) -> None:
    import xarray as xr

    files = sorted(data_dir.glob("*.nc"))
    if not files:
        raise SystemExit(f"Nenhum NetCDF encontrado em {data_dir.resolve()}. Veja data/README.md.")
    report = {}
    for path in files:
        with xr.open_dataset(path, chunks={}) as ds:
            report[path.name] = {
                "dimensions": dict(ds.sizes),
                "variables": {
                    name: {"dimensions": list(var.dims), "dtype": str(var.dtype),
                           "units": str(var.attrs.get("units", "não informada"))}
                    for name, var in ds.data_vars.items()
                },
                "coordinates": {
                    name: {"first": str(coord.values.flat[0]),
                           "last": str(coord.values.flat[-1])}
                    for name, coord in ds.coords.items() if coord.size
                },
            }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Metadados de {len(files)} arquivos: {output.resolve()}")
    print("Esta auditoria de metadados não certifica ausência de vazamento ou valores inválidos.")


def main() -> None:
    parser = argparse.ArgumentParser(description="WorCAP — ferramentas de experimentação")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Verificar ambiente e bibliotecas")
    audit_parser = commands.add_parser("audit", help="Inspecionar metadados dos NetCDF")
    audit_parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    audit_parser.add_argument("--output", type=Path, default=Path("outputs/data_audit.json"))
    verify_parser = commands.add_parser("verify-data", help="Verificar contrato numérico e hashes")
    verify_parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    verify_parser.add_argument("--output", type=Path, default=Path("outputs/contract_audit.json"))
    backtest_parser = commands.add_parser("backtest", help="Executar E00 em blocos históricos")
    backtest_parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    backtest_parser.add_argument("--output", type=Path, required=True)
    ridge_parser = commands.add_parser("ridge", help="E01: Ridge local com seleção temporal interna")
    ridge_parser.add_argument("--config", type=Path, default=Path("configs/ridge.yaml"))
    ridge_parser.add_argument("--output", type=Path, required=True)
    boosting_parser = commands.add_parser("lightgbm", help="E02: LightGBM com seleção temporal interna")
    boosting_parser.add_argument("--config", type=Path, default=Path("configs/lightgbm.yaml"))
    boosting_parser.add_argument("--output", type=Path, required=True)
    ensemble_parser = commands.add_parser("ensemble", help="E04: peso Ridge/LightGBM escolhido no passado")
    ensemble_parser.add_argument("--config", type=Path, default=Path("configs/ensemble.yaml"))
    ensemble_parser.add_argument("--output", type=Path, required=True)
    final_parser = commands.add_parser("finalize", help="E05: treinar e gerar candidato local")
    final_parser.add_argument("--config", type=Path, default=Path("configs/final.yaml"))
    final_parser.add_argument("--output", type=Path, required=True)
    forecast_parser = commands.add_parser("forecast", help="Inferência local dos modelos salvos")
    forecast_parser.add_argument("--model-dir", type=Path, required=True)
    forecast_parser.add_argument("--test", type=Path, default=Path("data/raw/teste_features.nc"))
    forecast_parser.add_argument("--sample", type=Path, default=Path("data/raw/sample_submission.csv"))
    forecast_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "doctor":
        doctor()
    elif args.command == "audit":
        audit(args.data_dir, args.output)
    elif args.command == "verify-data":
        from .experiment import audit_contract
        report = audit_contract(args.data_dir, args.output)
        print(f"Contrato: {report['status']}. Relatório: {args.output.resolve()}")
    elif args.command == "backtest":
        from .experiment import run_backtest
        run_backtest(args.config, args.output)
    elif args.command == "ridge":
        from .ridge import run_ridge
        run_ridge(args.config, args.output)
    elif args.command == "lightgbm":
        from .boosting import run_boosting
        run_boosting(args.config, args.output)
    elif args.command == "ensemble":
        from .ensemble import run_ensemble
        run_ensemble(args.config, args.output)
    elif args.command == "finalize":
        from .finalize import run_final
        run_final(args.config, args.output)
    elif args.command == "forecast":
        from .finalize import forecast_saved
        forecast_saved(args.model_dir, args.test, args.sample, args.output)
