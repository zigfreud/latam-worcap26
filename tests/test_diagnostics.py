import numpy as np
import pytest

from worcap.diagnostics import error_metrics, region_masks, spatial_metrics


def test_error_decomposition_tolerances_and_strict_win():
    truth = np.array([0., 2., 4., 6.])
    pred = np.array([0., 3., 2., 9.])
    base = np.array([0., 4., 5., 3.])
    m = error_metrics(pred, truth, base)
    assert m["rmse"] == np.sqrt(3.5)
    assert m["bias"] == .5
    assert m["within_1_fraction"] == .5
    assert m["improved_fraction"] == .25
    assert m["tied_fraction"] == .5
    assert m["worsened_fraction"] == .25
    fields = spatial_metrics(pred.reshape(2, 1, 2), truth.reshape(2, 1, 2), base.reshape(2, 1, 2))
    assert np.mean(fields["rmse"]**2) == pytest.approx(m["rmse"]**2)
    np.testing.assert_allclose(fields["rmse"]**2, fields["bias"]**2+fields["error_std"]**2)
    with pytest.raises(ValueError, match="não finitos"):
        error_metrics(np.array([np.nan]), np.array([1.]))


def test_region_boundaries_cover_each_point_once():
    masks = region_masks(np.array([-60., -35., -15., 15.]), np.array([-90., -70., -50., -25.]))
    np.testing.assert_array_equal(np.sum([mask for _, mask in masks], axis=0), np.ones((4, 4)))
    assert dict(masks)["L2C2"][1, 1]
    assert dict(masks)["L3C3"][2, 2]
