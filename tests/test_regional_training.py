import numpy as np
import pytest

from worcap.boosting import configured_points
from worcap.experiment import sha256


def test_frozen_regional_indices_and_default(tmp_path):
    path = tmp_path/'points.npy'
    points = np.array([1, 4, 7], dtype=np.int64)
    np.save(path, points)
    cfg = {'training_points_file': {'path': str(path), 'sha256': sha256(path)}}
    actual = configured_points(cfg, 'training_points_file', 8, np.arange(8))
    np.testing.assert_array_equal(actual, points)
    default = np.arange(3)
    assert configured_points({}, 'training_points_file', 8, default) is default
    np.save(path, np.array([2, 4, 7]))
    with pytest.raises(ValueError, match='changed'):
        configured_points(cfg, 'training_points_file', 8, default)


@pytest.mark.parametrize('points', [[], [[1, 2]], [1.5, 3.0], [-1, 2], [1, 8], [2, 2], [3, 1], np.array([3, 1], dtype=np.uint64)])
def test_reject_invalid_regional_indices(tmp_path, points):
    path = tmp_path/'points.npy'
    np.save(path, np.array(points))
    cfg = {'selection_points_file': {'path': str(path), 'sha256': sha256(path)}}
    with pytest.raises(ValueError):
        configured_points(cfg, 'selection_points_file', 8, np.arange(8))


def test_regional_selection_scores_only_frozen_region(tmp_path):
    import json

    import lightgbm as lgb
    import pandas as pd
    import xarray as xr
    import yaml

    from worcap.boosting import predict_month, prediction_features, prepare_training, run_boosting
    from worcap.experiment import ATMOSPHERIC, evaluate_fold

    dates = pd.date_range('2001-01-01', '2008-12-01', freq='MS')
    coords = {'time': dates, 'lat': [-20., -19.], 'lon': [-50., -49., -48.]}
    rng = np.random.default_rng(19)
    raw = tmp_path/'raw'
    raw.mkdir()
    datasets = {}
    for name in ATMOSPHERIC:
        values = rng.normal(size=(len(dates), 2, 3)).astype(np.float32)
        datasets[name] = xr.Dataset({name: (('time', 'lat', 'lon'), values)}, coords=coords)
        datasets[name].to_netcdf(raw/f'treino_{name}.nc')
    values = 2 + np.square(np.roll(datasets[ATMOSPHERIC[0]][ATMOSPHERIC[0]].values, 1, axis=0))
    # Exterior is deliberately more variable; regional score must not include it.
    values[:, :, 1] *= 15
    tp = xr.DataArray(values, dims=('time', 'lat', 'lon'), coords=coords,
                      name='tp', attrs={'units': 'mm/day'})
    tp.to_netcdf(raw/'treino_tp.nc')
    fold = {'train_end': 2006, 'validation_start': 2007, 'validation_end': 2008}
    base, _, _ = evaluate_fold(tp, fold, 3)
    baseline = tmp_path/'baseline.json'
    baseline.write_text(json.dumps({'folds': [base], 'rmse_mm_day': base['rmse'],
                                    'input_sha256': {'treino_tp.nc': sha256(raw/'treino_tp.nc')}}))
    ridge = tmp_path/'ridge.json'
    ridge.write_text(json.dumps({'folds': [{'fold': '2007-2008', 'ridge_rmse': 5.}],
                                 'ridge_rmse': 5., 'input_sha256': {}}))
    train_points, selected = np.array([0, 2]), np.array([0, 2, 4])
    cfg = {'data_dir': str(raw), 'baseline_results': str(baseline), 'ridge_results': str(ridge),
           'window_years': 3, 'inner_validation_years': 2, 'training_points': 2,
           'seed': 19, 'iterations': [2, 5], 'clip_min': 0., 'folds': [fold],
           'params': {'objective': 'regression', 'num_leaves': 3, 'min_data_in_leaf': 2,
                      'num_threads': 1, 'verbosity': -1, 'deterministic': True, 'force_col_wise': True}}
    for name, points in [('training_points_file', train_points), ('selection_points_file', selected)]:
        path = tmp_path/f'{name}.npy'
        np.save(path, points)
        cfg[name] = {'path': str(path), 'sha256': sha256(path)}
    config_path = tmp_path/'config.yaml'
    config_path.write_text(yaml.safe_dump(cfg))
    out = tmp_path/'experiment'
    run_boosting(config_path, out)
    np.testing.assert_array_equal(np.load(out/'training_points.npy'), train_points)
    selection = json.loads((out/'2007-2008/selection.json').read_text())
    assert selection['selection_points'] == 3
    _, _, pre = prepare_training(datasets, tp, 2002, 2004, train_points)
    model = lgb.Booster(model_file=str(out/'2007-2008/inner_model.txt'))
    for row in selection['candidates']:
        errors = []
        for date in pd.date_range('2005-01-01', '2006-12-01', freq='MS'):
            pred, _ = predict_month(model, prediction_features(datasets, pre, date), row['iterations'], 1)
            errors.append(pred[selected].astype(float)-tp.sel(time=date).values.ravel()[selected])
        assert row['rmse'] == pytest.approx(float(np.sqrt(np.square(errors).mean())), abs=1e-12)
    assert selection['chosen_iterations'] == min(selection['candidates'], key=lambda r: r['rmse'])['iterations']
