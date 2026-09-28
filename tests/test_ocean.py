import numpy as np
import pandas as pd
import pytest

from worcap.ocean import OCEAN_NAMES, ocean_rows, ocean_sources, read_sst


def test_raw_sst_columns_and_reject_duplicate_or_missing_months(tmp_path):
    path = tmp_path / "sst.txt"
    header = "YR MON NINO12 ANOM NINO3 ANOM NINO4 ANOM NINO34 ANOM\n"
    rows = ["2000 1 20 -999 21 -999 22 -999 23 -999\n",
            "2000 2 24 -999 25 -999 26 -999 27 -999\n"]
    path.write_text(header+"".join(rows))
    table = read_sst(path)
    np.testing.assert_array_equal(table.iloc[0], [20, 21, 22, 23])
    path.write_text(header+rows[0]+rows[0])
    with pytest.raises(ValueError, match="duplicadas"):
        read_sst(path)
    path.write_text(header+rows[0]+rows[1].replace("2000 2", "2000 3"))
    with pytest.raises(ValueError, match="completa"):
        read_sst(path)


def test_lag_boundary_and_no_future_in_normalization_or_prediction():
    dates = pd.date_range("1999-01-01", "2005-12-01", freq="MS")
    rng = np.random.default_rng(17)
    table = pd.DataFrame(25+rng.normal(size=(len(dates), 4)), index=dates, columns=OCEAN_NAMES)
    training = pd.date_range("2000-01-01", "2002-12-01", freq="MS")
    rows, means = ocean_rows(table, training, 2)
    assert ocean_sources([pd.Timestamp("2003-01-01")], 2)[0] == pd.Timestamp("2002-11-01")
    pred, _ = ocean_rows(table, [pd.Timestamp("2003-01-01")], 2, means)
    table.loc["2002-12-01":] = 1000
    rows2, means2 = ocean_rows(table, training, 2)
    pred2, _ = ocean_rows(table, [pd.Timestamp("2003-01-01")], 2, means2)
    np.testing.assert_array_equal(rows, rows2)
    np.testing.assert_array_equal(means, means2)
    np.testing.assert_array_equal(pred, pred2)
    with pytest.raises(ValueError, match="mínima"):
        ocean_sources(training, 1)
    with pytest.raises(ValueError, match="ausentes"):
        ocean_rows(table, [pd.Timestamp("1990-01-01")], 2, means)
