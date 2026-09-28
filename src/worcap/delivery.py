"""Finalização da família pressão com memória, sem upload ou alvos de teste."""
import json
import shutil
from contextlib import ExitStack
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from threadpoolctl import threadpool_limits

from .boosting import predict_month, train_model
from .experiment import ATMOSPHERIC, require, sha256
from .finalize import export_csv
from .spatial import prediction_spatial_features, prepare_spatial_training, spatial_feature_names
from .temporal import bridge_test_atmosphere


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def predict_pressure(model, pre, datasets, dates, iterations, threads=4):
    result = np.empty((len(dates), pre.sizes['lat'], pre.sizes['lon']), dtype=np.float32)
    require(model.feature_name() == spatial_feature_names('pressure'), 'Ordem dos 53 atributos')
    for i, date in enumerate(dates):
        values, _ = predict_month(model, prediction_spatial_features(datasets, pre, date), iterations, threads)
        result[i] = values.reshape(result.shape[1:])
    return result


def infer_pressure(model_dir, test_path):
    recipe = json.loads((model_dir/'recipe.json').read_text(encoding='utf-8'))
    for name, digest in recipe['artifact_sha256'].items():
        require(sha256(model_dir/name) == digest, f'Artefato mudou: {name}')
    require(sha256(test_path) == recipe['test_sha256'], 'Teste mudou')
    with xr.open_dataset(test_path) as test, xr.open_dataset(model_dir/'preprocessing.nc') as pre, xr.open_dataset(model_dir/'atmosphere_history.nc') as history:
        dates = pd.date_range(recipe['forecast_start'], recipe['forecast_end'], freq='MS')
        require(np.array_equal(test.time.values, dates.values), 'Meses do teste')
        datasets = bridge_test_atmosphere({v: history[[v]] for v in ATMOSPHERIC}, test, months=3)
        model = lgb.Booster(model_file=str(model_dir/'model.txt'))
        values = predict_pressure(model, pre, datasets, dates, recipe['iterations'], recipe['threads'])
        return xr.Dataset({'tp_mm_day': (('time','lat','lon'), values, {'units':'mm/day'})},
                          coords={'time':dates,'lat':pre.lat,'lon':pre.lon})


def run_pressure_final(output, points_path=None, selection_mask=None):
    require(not output.exists(), 'Diretório final já existe')
    output.mkdir(parents=True)
    cfg = yaml.safe_load(Path('configs/e09_pressure.yaml').read_text(encoding='utf-8'))
    points_path = Path('outputs/E09_spatial_v1/pressure/training_points.npy') if points_path is None else points_path
    points = np.load(points_path)
    shutil.copy2(points_path, output/'training_points.npy')
    for p in [Path('docs/ENTREGA_FINAL_PROTOCOLO.md'), Path('configs/e09_pressure.yaml'), Path('pyproject.toml'), Path('uv.lock'), *Path('src/worcap').glob('*.py')]:
        dst = output/'source_snapshot'/p
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
    raw = Path('data/raw')
    paths = [raw/'treino_tp.nc', raw/'teste_features.nc', raw/'sample_submission.csv', *[raw/f'treino_{v}.nc' for v in ATMOSPHERIC]]
    inputs = {p.name: sha256(p) for p in paths}
    prior = json.loads(Path('outputs/contract_audit.json').read_text(encoding='utf-8'))
    for name, digest in inputs.items():
        require(prior['files'][name]['sha256'] == digest, f'Dado mudou: {name}')
    write_json(output/'input_sha256.json', inputs)
    with ExitStack() as stack, threadpool_limits(limits=2):
        tp = stack.enter_context(xr.open_dataset(raw/'treino_tp.nc')).tp
        require(tp.attrs['units'] == 'mm/day' and pd.Timestamp(tp.time.values[-1]) == pd.Timestamp('2022-12-01'), 'Contrato de chuva')
        datasets = {v:stack.enter_context(xr.open_dataset(raw/f'treino_{v}.nc')) for v in ATMOSPHERIC}
        test = stack.enter_context(xr.open_dataset(raw/'teste_features.nc'))
        dates = pd.date_range('2023-01-01','2024-12-01',freq='MS')
        require(np.array_equal(test.time.values, dates.values), 'Alvos esperados')
        bridge = bridge_test_atmosphere(datasets, test, months=3)
        history = xr.Dataset({v:datasets[v][v].sel(time=slice('2022-10-01','2022-11-01')).load() for v in ATMOSPHERIC})
        history.to_netcdf(output/'atmosphere_history.nc', encoding={v:{'zlib':True} for v in history})
        validation = pd.date_range('2019-01-01','2022-12-01',freq='MS')
        print('Final: seleção 1989-2018 -> 2019-2022', flush=True)
        x,y,pre = prepare_spatial_training(datasets,tp,1989,2018,points,'pressure')
        model = train_model(x,y,cfg,300)
        del x,y
        model.save_model(str(output/'inner_model.txt'))
        truth = tp.sel(time=validation).values
        mask = np.ones(truth.shape[1:],dtype=bool) if selection_mask is None else selection_mask
        candidates = []
        for iterations in [100,300]:
            predicted = predict_pressure(model,pre,datasets,validation,iterations)
            score = float(np.sqrt(np.square(predicted[:,mask].astype(float)-truth[:,mask]).mean()))
            candidates.append({'iterations':iterations,'rmse_selection_only':score})
            print(f'  {iterations} árvores: seleção RMSE {score:.6f}', flush=True)
        chosen = min(candidates,key=lambda r:r['rmse_selection_only'])['iterations']
        write_json(output/'selection.json', {'training':'1989-2018','validation':'2019-2022','candidates':candidates,'chosen_iterations':chosen,'selection_only':True,'selection_region':'global' if selection_mask is None else 'Brazil'})
        del predicted,truth,pre,model
        print(f'Final: {chosen} árvores; reajuste 1993-2022', flush=True)
        x,y,pre = prepare_spatial_training(datasets,tp,1993,2022,points,'pressure')
        model = train_model(x,y,cfg,chosen)
        del x,y
        model.save_model(str(output/'model.txt'))
        pre.to_netcdf(output/'preprocessing.nc', encoding={k:{'zlib':True} for k in pre})
        expected = predict_pressure(model,pre,bridge,dates,chosen)
        recipe = {'family':'pressure_memory3','features':spatial_feature_names('pressure'), 'training':'1993-2022',
            'training_scope':'global' if selection_mask is None else 'Brazil',
            'training_points_sha256':sha256(output/'training_points.npy'),
            'iterations':chosen,'threads':4,'forecast_start':'2023-01-01','forecast_end':'2024-12-01',
            'test_sha256':inputs['teste_features.nc'],'sample_sha256':inputs['sample_submission.csv'],
            'artifact_sha256':{n:sha256(output/n) for n in ['model.txt','preprocessing.nc','atmosphere_history.nc','training_points.npy','selection.json']},
            'test_rmse_available':False,'submitted':False}
        write_json(output/'recipe.json',recipe)
    print('Final: recarga autônoma dos 24 meses',flush=True)
    reconstructed = infer_pressure(output,raw/'teste_features.nc')
    require(np.array_equal(expected,reconstructed.tp_mm_day.values),'Recarga dos 24 meses diverge')
    reconstructed.to_netcdf(output/'predictions.nc',encoding={'tp_mm_day':{'zlib':True}})
    verification = export_csv(reconstructed,raw/'sample_submission.csv',output/'candidate_unsubmitted.csv')
    verification.update({'all_24_months_reproduced_exactly':True,'test_rmse_available':False,'history_months':['2022-10','2022-11']})
    write_json(output/'verification.json',verification)
    print('Final: CSV exportado e validado',flush=True)
    return verification
