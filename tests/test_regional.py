import numpy as np
import pytest

from worcap.regional import choose_brazil, route_brazil


def test_routing_preserves_exterior_and_inner_decision_when_outer_winner_changes():
    mask = np.array([[True, False], [False, True]])
    base = np.full((3, 2, 2), 1, dtype=np.float32)
    specialist = np.full_like(base, 5)
    truth = specialist.copy()
    truth[:, ~mask] = 1000  # exterior não pode influenciar a escolha brasileira
    selected, _ = choose_brazil(base, specialist, truth, mask)
    assert selected is True
    outer_truth = base.copy()  # no futuro, o outro componente vence
    output = route_brazil(base, specialist, mask, selected)
    assert np.array_equal(output[:, ~mask], base[:, ~mask])
    assert np.array_equal(output[:, mask], specialist[:, mask])
    assert np.square(output-outer_truth).sum() > 0
    assert np.array_equal(route_brazil(base, specialist, mask, False), base)


def test_tie_stays_control_and_invalid_geography_fails():
    mask = np.array([[True, False]])
    base = np.ones((2, 1, 2), dtype=np.float32)
    assert choose_brazil(base, base, base, mask)[0] is False
    for bad in (np.zeros_like(mask), np.ones_like(mask), mask.astype(int), mask.T):
        with pytest.raises(ValueError):
            route_brazil(base, base, bad, True)
    with pytest.raises(ValueError):
        route_brazil(base, base.astype(np.float64), mask, True)
