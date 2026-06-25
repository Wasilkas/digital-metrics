import pytest

from digital_metrics.ci import calculate_confidence_interval


def test_bounds_in_unit_interval() -> None:
    lower, upper = calculate_confidence_interval(70, 100)
    assert 0.0 <= lower <= 1.0
    assert 0.0 <= upper <= 1.0


def test_lower_le_estimate_le_upper() -> None:
    tp, total = 70.0, 100.0
    lower, upper = calculate_confidence_interval(tp, total)
    estimate = tp / total
    assert lower <= estimate
    assert estimate <= upper


def test_total_zero_returns_zero() -> None:
    assert calculate_confidence_interval(0, 0) == (0.0, 0.0)


def test_all_positive() -> None:
    lower, upper = calculate_confidence_interval(100, 100)
    assert lower > 0.9
    assert upper == pytest.approx(1.0)


def test_all_negative() -> None:
    lower, upper = calculate_confidence_interval(0, 100)
    assert lower == pytest.approx(0.0)
    assert upper < 0.05


def test_unknown_method_raises() -> None:
    with pytest.raises(ValueError, match="Unknown method"):
        calculate_confidence_interval(50, 100, method="bootstrap")


def test_symmetric_around_half() -> None:
    lower, upper = calculate_confidence_interval(50, 100)
    mid = (lower + upper) / 2
    assert mid == pytest.approx(0.5, abs=0.01)
