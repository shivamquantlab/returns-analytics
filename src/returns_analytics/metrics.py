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
    return values[1:] / values[:-1] - 1.0


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
    wealth = np.concatenate(([1.0], np.cumprod(1.0 + values)))
    peaks = np.maximum.accumulate(wealth)
    return float(np.min(wealth / peaks - 1.0))


def risk_summary(returns: np.ndarray, periods_per_year: int = 252) -> dict[str, float]:
    """Descriptive arithmetic-return statistics, annualized by sqrt(periods)."""
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("returns must contain at least 2 observations")
    if not np.isfinite(values).all() or periods_per_year <= 0:
        raise ValueError("returns must be finite and periods_per_year positive")
    return {
        "observations": float(values.size),
        "mean_period_return": float(values.mean()),
        "annualized_mean": float(values.mean() * periods_per_year),
        "annualized_volatility": float(values.std(ddof=1) * np.sqrt(periods_per_year)),
        "max_drawdown": max_drawdown(values),
    }

