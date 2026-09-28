import numpy as np
import pandas as pd
import pytest
import xarray as xr

from worcap.experiment import evaluate_fold, verify_sample


def example_data():
    dates = pd.date_range("2019-01-01", "2022-12-01", freq="MS")
    values = np.asarray(dates.month, dtype=float).copy()
    values[24:36] += 1
    values[36:] += 3
    return xr.DataArray(values[:, None, None], dims=("time", "lat", "lon"),
                        coords={"time": dates, "lat": [-30.0], "lon": [-53.0]})


def test_fold_uses_past_and_pools_squared_errors():
    result, months, error_map = evaluate_fold(
        example_data(), {"train_end": 2020, "validation_start": 2021, "validation_end": 2022}, 2
    )
    assert result["rmse"] == pytest.approx(np.sqrt(5))
    assert result["bias"] == pytest.approx(-2)
    assert result["count"] == 24
    assert result["training_months"] == 24
    assert months[0]["rmse"] == pytest.approx(1)
    assert months[-1]["rmse"] == pytest.approx(3)
    assert error_map.item() == pytest.approx(np.sqrt(5))


def test_fold_rejects_overlap_and_incomplete_validation():
    with pytest.raises(ValueError, match="preceder"):
        evaluate_fold(example_data(),
                      {"train_end": 2021, "validation_start": 2021, "validation_end": 2022}, 2)
    with pytest.raises(ValueError, match="incompleta"):
        evaluate_fold(example_data().isel(time=slice(None, -1)),
                      {"train_end": 2020, "validation_start": 2021, "validation_end": 2022}, 2)


def test_sample_rejects_wrong_order_and_truncation(tmp_path):
    sample = tmp_path / "sample.csv"
    rows = pd.DataFrame({"id": ["2023_01_-30.00_-53.00", "2023_01_-30.00_-52.75"],
                         "tp_mm_day": [0, 0]})
    dates, lat, lon = pd.to_datetime(["2023-01-01"]), np.array([-30.]), np.array([-53., -52.75])
    rows.to_csv(sample, index=False)
    assert verify_sample(sample, dates, lat, lon)["rows"] == 2
    rows.iloc[::-1].to_csv(sample, index=False)
    with pytest.raises(ValueError, match="IDs/ordem"):
        verify_sample(sample, dates, lat, lon)
    rows.iloc[:1].to_csv(sample, index=False)
    with pytest.raises(ValueError, match="incompleto"):
        verify_sample(sample, dates, lat, lon)
