"""Small return statistics with explicit input validation."""

import numpy as np


def _prices(values: np.ndarray) -> np.ndarray:
    prices = np.asarray(values, dtype=float)
    if prices.ndim != 1 or prices.size < 2:
        raise ValueError("prices must be a one-dimensional array with at least 2 values")
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("prices must be finite and strictly positive")
    return prices


def simple_returns(prices: np.ndarray) -> np.ndarray:
    """Close-to-close simple returns; the first price has no prior return."""
    values = _prices(prices)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        result = values[1:] / values[:-1] - 1.0
    if not np.isfinite(result).all():
        raise ValueError("computed simple returns exceed the finite float range")
    return result


def log_returns(prices: np.ndarray) -> np.ndarray:
    """Close-to-close continuously compounded returns."""
    values = _prices(prices)
    return np.diff(np.log(values))


def max_drawdown(returns: np.ndarray) -> float:
    """Return the most negative peak-to-trough wealth drawdown."""
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("returns must be a non-empty one-dimensional array")
    if not np.isfinite(values).all() or (values <= -1).any():
        raise ValueError("returns must be finite and greater than -100%")
    log_wealth = np.concatenate(([0.0], np.cumsum(np.log1p(values))))
    log_peaks = np.maximum.accumulate(log_wealth)
    return float(np.min(np.expm1(log_wealth - log_peaks)))


def risk_summary(returns: np.ndarray, periods_per_year: int = 252) -> dict[str, float]:
    """Descriptive arithmetic-return statistics, annualized by sqrt(periods)."""
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("returns must contain at least 2 observations")
    if not np.isfinite(values).all():
        raise ValueError("returns must be finite")
    if (
        isinstance(periods_per_year, (bool, np.bool_))
        or not isinstance(periods_per_year, (int, np.integer))
        or periods_per_year <= 0
    ):
        raise ValueError("periods_per_year must be a positive integer")
    scale = float(np.max(np.abs(values)))
    if scale == 0.0:
        mean_return = 0.0
        sample_volatility = 0.0
    else:
        scaled = values / scale
        mean_return = float(scale * scaled.mean())
        sample_volatility = float(scale * scaled.std(ddof=1))
    annualized_mean = mean_return * periods_per_year
    annualized_volatility = sample_volatility * np.sqrt(periods_per_year)
    if not np.isfinite([mean_return, sample_volatility, annualized_mean, annualized_volatility]).all():
        raise ValueError("return statistics exceed the finite float range")
    return {
        "observations": float(values.size),
        "mean_period_return": mean_return,
        "annualized_mean": float(annualized_mean),
        "annualized_volatility": float(annualized_volatility),
        "max_drawdown": max_drawdown(values),
    }

