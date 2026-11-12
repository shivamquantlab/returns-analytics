import numpy as np
import pytest

from returns_analytics.metrics import (
    log_returns,
    max_drawdown,
    risk_summary,
    simple_returns,
)


def test_simple_and_log_returns_match_known_prices():
    prices = np.array([100.0, 110.0, 99.0])
    np.testing.assert_allclose(simple_returns(prices), [0.1, -0.1])
    np.testing.assert_allclose(log_returns(prices), np.log([1.1, 0.9]))


def test_max_drawdown_is_peak_to_trough():
    assert max_drawdown(np.array([0.1, -0.2, 0.0])) == pytest.approx(-0.2)


def test_summary_annualizes_sample_standard_deviation():
    values = np.array([-0.01, 0.01, -0.01, 0.01])
    summary = risk_summary(values, periods_per_year=4)
    assert summary["annualized_volatility"] == pytest.approx(values.std(ddof=1) * 2)


def test_nonpositive_prices_are_rejected():
    with pytest.raises(ValueError, match="positive"):
        simple_returns(np.array([100.0, 0.0]))

